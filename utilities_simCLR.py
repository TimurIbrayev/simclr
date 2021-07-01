#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 23:09:58 2021

@author: tibrayev
@reference: https://github.com/leftthomas/SimCLR/blob/master/utils.py
"""

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as tvf
import torchvision.datasets as dset
from torch.utils.data.sampler import SequentialSampler, SubsetRandomSampler
import numpy as np
import copy
from custom_normalization_functions import *
from PIL import Image

dataset_stats = {
    "imagenet2012": {'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225)},
    "cifar10":      {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
    "cifar100":     {},
    "mnist":        {},
    "fashionmnist": {}
    }

class CIFAR10Pair(dset.CIFAR10):
    """ Customized CIFAR10 dataset."""
    
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img_1 = self.transform(img)
            img_2 = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img_1, img_2, target

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
                                        [transforms.RandomResizedCrop(size=32),
                                         transforms.RandomHorizontalFlip(p=0.5),
                                         transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                         transforms.RandomGrayscale(p=0.2),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
        
        train_set = CIFAR10Pair(root=data_dir, train=True,  transform=train_transform, download=True)
        valid_set = CIFAR10Pair(root=data_dir, train=True,  transform=test_transform,  download=True)
        test_set  = CIFAR10Pair(root=data_dir, train=False, transform=test_transform,  download=True)

    
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
    



def contrastive_loss_direct(vec_1, vec_2):
    
    epsilon = 0.0#1e-12
    
    N = vec_1.shape[0]
    vec_1_normalized = vec_1/(vec_1.norm(p=2, dim=-1, keepdim=True) + epsilon)
    vec_2_normalized = vec_2/(vec_2.norm(p=2, dim=-1, keepdim=True) + epsilon)
    long_vector      = torch.cat((vec_1_normalized, vec_2_normalized), dim=0)
    
    positive_similarities   = []

    for i in range(N):
        similarity_positive = (vec_1_normalized[i]*vec_2_normalized[i]).sum()
        numerator           = similarity_positive.exp()
        similarity_all      = torch.matmul(long_vector, vec_1_normalized[i].unsqueeze(-1))
        mask                = torch.ones_like(similarity_all)
        mask[i]             = 0.0
        denominator         = ((similarity_all.exp())*mask).sum() + epsilon
        if i == 0:
            loss            = -torch.log(numerator/denominator)
        else:
            loss           += -torch.log(numerator/denominator)
        positive_similarities.append(similarity_positive.detach())
        
    for j in range(N):
        similarity_positive = (vec_2_normalized[j]*vec_1_normalized[j]).sum()
        numerator           = similarity_positive.exp()
        similarity_all      = torch.matmul(long_vector, vec_2_normalized[j].unsqueeze(-1))
        mask                = torch.ones_like(similarity_all)
        mask[N+j]           = 0.0
        denominator         = ((similarity_all.exp())*mask).sum() + epsilon
        loss               += -torch.log(numerator/denominator)        
    
    loss /= (2*N)
    
    return loss, torch.stack(positive_similarities)#, mask


def contrastive_loss_compact(vec_1, vec_2, temperature=1.0):
 
    epsilon = 0.0#1e-12
    
    N = vec_1.shape[0]
    vec_1_normalized = vec_1/(vec_1.norm(p=2, dim=-1, keepdim=True) + epsilon)
    vec_2_normalized = vec_2/(vec_2.norm(p=2, dim=-1, keepdim=True) + epsilon)
    long_vector      = torch.cat((vec_1_normalized, vec_2_normalized), dim=0)
    
    
    pos_sim         = torch.exp(torch.sum(vec_1_normalized * vec_2_normalized, dim=-1) / temperature)
    numerator       = torch.cat((pos_sim, pos_sim), dim=0)
    positive_similarities = torch.log(pos_sim).detach() * temperature
    
    sim_matrix      = torch.exp(torch.mm(long_vector, long_vector.t().contiguous()) / temperature)
    mask            = (torch.ones_like(sim_matrix) - torch.eye(2 * N, device=sim_matrix.device)).bool()
    sim_matrix      = sim_matrix.masked_select(mask).view(2 * N, -1)
    denominator     = sim_matrix.sum(dim=-1) + epsilon
    
    loss            = (-torch.log(numerator/denominator)).mean()
    return loss, positive_similarities

    