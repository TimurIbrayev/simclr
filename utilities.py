#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 00:09:35 2020

"""
import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf
import torchvision.datasets as dset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import numpy as np
import copy
from custom_normalization_functions import *


dataset_stats = {
    "imagenet2012": {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
    "cifar10":      {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
    "cifar100":     {},
    "mnist":        {},
    "fashionmnist": {}
    }


def get_data_loaders(dataset,
                     data_dir, 
                     batch_size, 
                     augment,
                     random_seed,
                     normalize,
                     device,
                     valid_size=0.0,
                     shuffle=True,
                     num_workers=1,
                     pin_memory=True):

    dataset = copy.deepcopy(dataset).lower() 
    assert dataset in ['mnist', 'fashionmnist', 'cifar10', 'imagenet2012']
    assert ((valid_size >= 0) and (valid_size <= 1)), "Error: valid_size should be in the range [0, 1]."
    assert normalize in ['static0to1', 'staticminus1to1', 'staticzeromeanunitvar', 'separate_per_image', 'separate_per_dataset'], \
        "Error: Unknown normalization request! Choices are 'static0to1', 'staticminus1to1', 'staticzeromeanunitvar', 'separate_per_image', 'separate_per_dataset'"


    # CIFAR10
    if dataset == 'cifar10':
        test_transform          = transforms.Compose(
                                        [transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_transform         = transforms.Compose(
                                        [transforms.RandomCrop(size=32, padding=4),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_set = dset.CIFAR10(root=data_dir, train=True,  transform=train_transform, download=True)
        valid_set = dset.CIFAR10(root=data_dir, train=True,  transform=test_transform,  download=True)
        test_set  = dset.CIFAR10(root=data_dir, train=False, transform=test_transform,  download=True)
    
    # Splitting into validation set if requested...
    print('\nInfo: Forming the samplers for train and validation splits with split fraction={}'.format(valid_size))
    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    
    # Shuffling if requested...
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    print('\nInfo: Preparing dataloaders...')
    if valid_size == 0.0:
        print('\nWarning: Since valid_size=0.0, providing test_loader as valid_loader!')
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
        return train_loader, test_loader, test_loader
    else:
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
        valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, sampler=valid_sampler, num_workers=num_workers, pin_memory=pin_memory)
        test_loader  = torch.utils.data.DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)    
        return train_loader, valid_loader, test_loader
    
    