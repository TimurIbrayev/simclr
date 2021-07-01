#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 22:57:08 2020

@author: tibrayev

"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import torch.optim as optim
import numpy as np
import random
import sys
import time
import argparse
import copy
import json

torch.set_printoptions(linewidth = 160)
np.set_printoptions(linewidth = 160)
np.set_printoptions(precision=4)
np.set_printoptions(suppress='True')


from custom_models_cifar_vgg import customizable_VGG
from torchvision.models import resnet50
from utilities import get_data_loaders

import matplotlib.pyplot as plt
from torchvision.utils import make_grid as grid


parser = argparse.ArgumentParser(description='Bare minimim script for training linear classifier on top of frozen pre-trained feature extractor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed',           default=1,                  type=int,       help='Seed for random numbers')
parser.add_argument('--log',            default=None,               type=str,       help='Indicator of the log file. Defaults to .txt file in results directory. If set to "sys" sets to sys.stdout')
parser.add_argument('--experiment',     default='simclr',           type=str,       help='Indicator of the experiment performed to separate experiments into separate folders')
parser.add_argument('--dataset',        default='cifar10',          type=str,       help='Dataset name', choices=['mnist', 'fashionmnist', 'cifar10', 'imagenet2012'])
parser.add_argument('--valid_split',    default=0.0,                type=float,     help='Fraction of training set dedicated for validation')
parser.add_argument('--arch',           default='VGG6',             type=str,       help='Model architecture to be trained')
parser.add_argument('--parallel',       default=False,              type=bool,      help='Flag to whether parallelize model over multiple GPUs')
parser.add_argument('--checkpoint',     default=None,               type=str,       help='Path to checkpoint file')
parser.add_argument('--inference_only', default=False,              type=bool,      help='Flag to run only inference without training')

parser.add_argument('--batch_size',     default=128,                type=int,       help='Batch size for data loading')
parser.add_argument('--num_epochs',     default=100,                type=int,       help='The number of training epochs')
parser.add_argument('--optimizer',      default='adam',             type=str,       help='Type of optimizer to use for training', choices=['sgd', 'adam'])
parser.add_argument('--init_lr',        default=1.0e-3,             type=float,     help='Learning rate during training')
parser.add_argument('--weight_decay',   default=0.000001,           type=float,     help='Weight decay factor (L2 regularization)')
parser.add_argument('--momentum',       default=0.9,                type=float,     help='Momentum')
parser.add_argument('--lr_schedule',    default='1.0',              type=str,       help='Intervals at which to reduce lr, expressed as %%age of total epochs')
parser.add_argument('--lr_gamma',       default=0.1,                type=int,       help='Reduction factor for learning rate')

parser.add_argument('--version',        default='withtrainedlinearclassifier',          type=str,       help='Version ID to distinguish different runs of the same experiment')


#%% Parse script parameters.
global args
args = parser.parse_args()

SEED            = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


LOG             = args.log
EXPERIMENT      = args.experiment
DATASET         = args.dataset
VALID_SPLIT     = args.valid_split
MODEL           = args.arch.lower()
PARALLEL        = args.parallel
CHECKPOINT      = args.checkpoint
INFERENCE_ONLY  = args.inference_only

BATCH_SIZE      = args.batch_size
NUM_EPOCHS      = args.num_epochs
TRAIN = {
    "OPTIMIZER":        args.optimizer,
    "INIT_LR":          args.init_lr,    
    "WEIGHT_DECAY":     args.weight_decay,
    "MOMENTUM":         args.momentum,
    "LR_SCHEDULE":      [int(float(value)*NUM_EPOCHS) for value in args.lr_schedule.split()],
    "LR_GAMMA":         args.lr_gamma
    }


VERSION_ID      = args.version
if not os.path.exists('./results/{}/{}'.format(DATASET, EXPERIMENT)): os.makedirs('./results/{}/{}'.format(DATASET, EXPERIMENT))
if LOG is None:
    f = open('./results/{}/{}/log_model_{}.txt'.format(DATASET, EXPERIMENT, VERSION_ID), 'a', buffering=1)
elif LOG == "sys":
    f = sys.stdout
else:
    f = open(LOG, 'a', buffering=1)

SAVE_DIR        = './results/{}/{}/ckp_model_{}.pth'.format(DATASET, EXPERIMENT, VERSION_ID)


# Timestamp
f.write('\n*******************************************************************\n')
f.write('==>> Run on: '+time.strftime("%Y-%m-%d %H:%M:%S")+'\n')
f.write('==>> Seed was set to: {}\n'.format(SEED))


# Record arguments
f.write('\n\n Arguments:')
for arg in vars(args):
    if arg == 'lr_schedule':
        f.write('\n\t {:20} : {}'.format(arg, TRAIN['LR_SCHEDULE']))
    else:
        f.write('\n\t {:20} : {}'.format(arg, getattr(args, arg)))

# Device instantiation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#%% Load the dataset.
if DATASET == 'imagenet2012':
    root_dir = '/local/a/imagenet/imagenet2012'
else:
    root_dir = './dataset/{}'.format(DATASET)

train_loader, valid_loader, test_loader = get_data_loaders(dataset=DATASET, 
                                                           data_dir=root_dir, 
                                                           batch_size=BATCH_SIZE, 
                                                           augment=False, #NOTE: augmentation is off 
                                                           normalize='static0to1',
                                                           device=device,
                                                           random_seed=SEED, 
                                                           valid_size=VALID_SPLIT, 
                                                           shuffle=True, 
                                                           num_workers=8, 
                                                           pin_memory=True)

f.write('\n==>> Total training batches: {}\n'.format(len(train_loader)))
f.write('==>> Total validation batches: {}\n'.format(len(valid_loader)))
f.write('==>> Total testing batches: {}\n'.format(len(test_loader)))


#%% Model instantiation.
if 'vgg' in MODEL:
    model = customizable_VGG(dataset=DATASET, vgg_name=MODEL, downsampling='M', projection_dim=0, fc1=0, fc2=0)
# elif 'resnet50' in MODEL:
#     model = resnet50(pretrained=False, num_classes=len(test_loader.dataset.classes))



model.to(device)
if PARALLEL:
    model = nn.DataParallel(model)

grad_requirement_dict = {name: param.requires_grad for name, param in model.named_parameters()}
f.write("{}\n".format(model))


#%% Training settings.
criterion = nn.CrossEntropyLoss()
if TRAIN['OPTIMIZER'] == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=TRAIN['INIT_LR'], momentum=TRAIN['MOMENTUM'], weight_decay=TRAIN['WEIGHT_DECAY'])
elif TRAIN['OPTIMIZER'] == 'adam':
    optimizer = optim.Adam(model.parameters(), lr=TRAIN['INIT_LR'], weight_decay=TRAIN['WEIGHT_DECAY'])

if VALID_SPLIT > 0.0:
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=TRAIN['LR_GAMMA'], verbose=True)
else:
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=TRAIN['LR_SCHEDULE'], gamma = TRAIN['LR_GAMMA'])



#%% Updating model, optimizer, lr_scheduler, tracking variables, etc. if CHECKPOINT is specified...
if not CHECKPOINT is None:
    f.write("==>> Feature extractor is loaded from checkpoint: {}\n".format(CHECKPOINT))
    ckpt = torch.load(CHECKPOINT, map_location=device)
    feature_extractor_dict = {key: value for key, value in ckpt['model'].items() if 'features' in key}
    missing_keys = model.load_state_dict(feature_extractor_dict, strict=False)
    f.write("==>> Warning: {}\n".format(missing_keys))
    for name, param in model.named_parameters():
        if not 'classifier' in name:
            param.requires_grad_(False)
    
    start_epoch             = 0
    train_loss              = []
    train_acc               = []
    valid_loss              = []
    valid_acc               = []
    
else:
    raise ValueError("This script is only for training linear classifiers for GIVEN feature extractor models! Checkpoint for feature extractor needs to be provided!")

f.write("==>> Optimizer settings: {}\n".format(optimizer))
f.write("==>> LR scheduler type: {}\n".format(lr_scheduler.__class__))
f.write("==>> LR scheduler state: {}\n".format(lr_scheduler.state_dict()))
f.write("==>> Number of training epochs: {}\n".format(NUM_EPOCHS))


#%% TRAIN.
if not INFERENCE_ONLY:
    for epoch in range(start_epoch, NUM_EPOCHS):
        # Train for one epoch
        model.train()
        correct             = 0.0
        ave_loss            = 0.0
        total               = 0
        for batch_idx, (x_train, y_train) in enumerate(train_loader):
            x_train, y_train = x_train.to(device), y_train.to(device)
        
            optimizer.zero_grad()
            output = model(x_train)
            loss = criterion(output, y_train)
            
            loss.backward()
            optimizer.step()
                
            _, predictions      = torch.max(output.data, 1)
            total               += y_train.size(0)
            correct             += (predictions == y_train).sum().item()
            ave_loss            += loss.item()
        
            if (batch_idx+1) == len(train_loader):
                    f.write('==>>> TRAIN-PRUNE | train epoch: {}, loss: {:.6f}, acc: {:.4f}\n'.format(
                            epoch, ave_loss*1.0/(batch_idx + 1), correct*1.0/total))
        train_loss.append(ave_loss*1.0/(batch_idx + 1))
        train_acc.append(correct*100.0/total)
        
        # Evaluate on the clean val set
        model.eval()
        correct     = 0.0
        ave_loss    = 0.0
        total       = 0
        with torch.no_grad():
            for batch_idx, (x_val, y_val) in enumerate(valid_loader):
                x_val, y_val = x_val.to(device), y_val.to(device)
                output = model(x_val)
                loss   = criterion(output, y_val)
                
                _, predictions   = torch.max(output.data, 1)
                total           += y_val.size(0)
                correct         += (predictions == y_val).sum().item()
                ave_loss        += loss.item()
                
                if (batch_idx+1) == len(valid_loader):
                    f.write('==>>> CLEAN VALIDATE | epoch: {}, batch index: {}, val loss: {:.6f}, val acc: {:.4f}\n'.format(
                            epoch, batch_idx+1, ave_loss*1.0/(batch_idx + 1), correct*1.0/total))
            valid_loss.append(ave_loss*1.0/(batch_idx+1))
            valid_acc.append(correct*100.0/total)
    
        # Adjust learning rate
        if VALID_SPLIT > 0.0:
            lr_scheduler.step(ave_loss*1.0/(batch_idx+1))
        else:
            lr_scheduler.step()
    
        # if (correct*100.0/total) >= best_val_acc:
        #     best_val_acc = correct*100.0/total
        #     best_epoch   = copy.deepcopy(epoch)
        #     best_msdict  = copy.deepcopy(model.state_dict())
        #     best_val_loss = ave_loss*1.0/(batch_idx+1)
            
        torch.save({'SEED': SEED,
                    'model': model.state_dict(),
                    #'best_msdict': best_msdict,
                    #'best_epoch': best_epoch,
                    #'best_val_acc': best_val_acc,
                    #'best_val_loss': best_val_loss,
                    'grad_requirement_dict': grad_requirement_dict,
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'num_epochs': NUM_EPOCHS,
                    'train_loss': train_loss,
                    'valid_loss': valid_loss,
                    'train_acc': train_acc,
                    'valid_acc': valid_acc}, SAVE_DIR)
    
    #f.write("Best val accuracy during training: {:.2f}\n".format(best_val_acc))
    #model.load_state_dict(best_msdict)

#%% Test set model evaluation.
# model.eval()
# correct     = 0.0
# ave_loss    = 0.0
# total       = 0
# with torch.no_grad():
#     for batch_idx, (x_val, y_val) in enumerate(test_loader):
#         x_val, y_val = x_val.to(device), y_val.to(device)
#         x_norm = normalization_func(x_val)
#         output = model(x_norm)
#         loss   = F.cross_entropy(output, y_val)
        
#         _, predictions   = torch.max(output.data, 1)
#         total           += y_val.size(0)
#         correct         += (predictions == y_val).sum().item()
#         ave_loss        += loss.item()
        
# f.write('==>>> MODEL EVAL ON TEST SET | val loss: {:.6f}, val acc: {:.4f}\n'.format(
#                 ave_loss*1.0/(batch_idx + 1), correct*1.0/total))
