#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:55:07 2020

@modified by: tibrayev
"""

import torch
import torch.nn as nn
import copy

# 'D' - stands for downsampling, for which choices are: 'M' - max pooling, 'A' - average pooling, 'C' - strided convolution
cfgs = {
    # first, custom ones, not present in original VGG paper:
    'vgg6_narrow' : [16, 'D', 32, 'D', 32, 'D'],
    'vgg6' : [64, 'D', 128, 'D', 128, 'D'],
    'vgg9' : [64, 'D', 128, 'D', 256, 256, 'D', 512, 512, 'D'],
    
    # next, default ones under VGG umbrella term:
    'vgg11': [64, 'D', 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512],
    'vgg13': [64, 64, 'D', 128, 128, 'D', 256, 256, 'D', 512, 512, 'D', 512, 512],
    'vgg16': [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 'D', 512, 512, 512, 'D', 512, 512, 512],
    'vgg19': [64, 64, 'D', 128, 128, 'D', 256, 256, 256, 256, 'D', 512, 512, 512, 512, 'D', 512, 512, 512, 512],
}

class customizable_VGG(nn.Module):
    def __init__(self, dataset='MNIST', vgg_name='vgg6_narrow', downsampling='M', 
                 projection_dim=128, fc1=512, fc2=512, 
                 dropout=0.5, batch_norm=False, init_weights=True,):
        super(customizable_VGG, self).__init__()
        
        # Dataset configuration
        self.dataset        = None
        self.num_classes    = None
        self.in_channels    = None
        self.in_feat_dim    = None
        self._infer_dataset_stats(dataset)

        # Network configuration
        self.vgg_name       = vgg_name.lower()
        if downsampling in ['M', 'A', 'C']: 
            self.downsampling   = downsampling
        else:
            raise ValueError("Error: Unknown downsampling. Choices are 'M' - max pooling, 'A' - average pooling, 'C' - strided convolution!")
        
        self.projection_dim         = projection_dim
        # If projection_dim != 0, fc1 define the projection_head, otherwise fc1 and fc2 define classifier_layers
        self.fc1                    = fc1
        self.fc2                    = fc2
        self.dropout                = dropout
        self.batch_norm             = batch_norm
        
        # Creating layers
        self.features, feature_channels, feature_dim    = self._make_feature_layers(cfgs[self.vgg_name])
        if self.projection_dim == 0:
            self.classifier                             = self._make_classifier_layers(feature_channels, feature_dim)
        else:
            self.projection                             = self._make_projection_head(feature_channels, feature_dim)
        if init_weights: self._initialize_weights()

    def _infer_dataset_stats(self, dataset):
        if dataset.lower() == 'mnist':
            self.dataset        = 'mnist'
            self.num_classes    = 10
            self.in_channels    = 1
            self.in_feat_dim    = 28
        elif dataset.lower() == 'fashionmnist':
            self.dataset        = 'fashionmnist'
            self.num_classes    = 10
            self.in_channels    = 1
            self.in_feat_dim    = 28
        elif dataset.lower() == 'cifar10':
            self.dataset        = 'cifar10'
            self.num_classes    = 10
            self.in_channels    = 3
            self.in_feat_dim    = 32
        elif dataset.lower() == 'cifar100':
            self.dataset        = 'fashionmnist'
            self.num_classes    = 100
            self.in_channels    = 3
            self.in_feat_dim    = 32
        else:
            raise ValueError("Error: Unknown dataset. This model is written only for images of 'mnist', 'fashionmnist', 'cifar10', 'cifar100'!")

    def forward(self, x, with_latent = False):
        x = self.features(x)
        features = torch.flatten(x, 1)
        
        if self.projection_dim == 0:
            outputs  = self.classifier(features)
        else:
            outputs  = self.projection(features)
        
        if with_latent:
            return outputs, features
        else:
            return outputs

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def _make_feature_layers(self, cfg):
        layers = []
        in_channels     = copy.deepcopy(self.in_channels)
        feature_dim     = copy.deepcopy(self.in_feat_dim)
        
        for v in cfg:
            if v == 'D':
                if self.downsampling == 'M':
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                elif self.downsampling == 'A':
                    layers += [nn.AvgPool2d(kernel_size=2, stride=2)]
                elif self.downsampling == 'C':
                    layers += [nn.Conv2d(kernel_size=2, stride=2, bias=False)]
                feature_dim //=2
                
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if self.batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        
        return nn.Sequential(*layers), in_channels, feature_dim


    def _make_projection_head(self, feature_channels, feature_dim) :
        layers = []
        feature_flat_dims = feature_channels*feature_dim*feature_dim
        
        if self.fc1 == 0: # Linear projection head
            layers += [nn.Linear(feature_flat_dims, self.projection_dim)]
        elif self.fc1 != 0: # Non-linear projection head
            layers += [nn.Linear(feature_flat_dims, self.fc1, bias=False)]
            layers += [nn.BatchNorm1d(self.fc1)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Linear(self.fc1, self.projection_dim)]
        return nn.Sequential(*layers)


    def _make_classifier_layers(self, feature_channels, feature_dim) :
        layers = []
        feature_flat_dims = feature_channels*feature_dim*feature_dim
        
        if self.fc1 == 0 and self.fc2 == 0:
            layers += [nn.Linear(feature_flat_dims, self.num_classes)]
        elif self.fc1 == 0 and self.fc2 != 0:
            raise ValueError("Received ambiguous pair of classifier parameters: fc1 = 0, but fc2 = {}. ".format(self.fc2) + 
                             "If only two FC layers are needed (including last linear classifier), please specify its dims as fc1 and set fc2=0.")
        elif self.fc1 != 0 and self.fc2 == 0:
            layers += [nn.Linear(feature_flat_dims, self.fc1)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(self.fc1, self.num_classes)]
        else:
            layers += [nn.Linear(feature_flat_dims, self.fc1)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(self.fc1, self.fc2)]
            layers += [nn.ReLU(inplace=True)]
            layers += [nn.Dropout(self.dropout)]
            layers += [nn.Linear(self.fc2, self.num_classes)]
        return nn.Sequential(*layers)    


def test():
    for d in ['MNIsT', 'FashionMnISt', 'Cifar10', 'CiFAR100']:
        for a in cfgs.keys():
            print("test under {} for {}".format(d.lower(), a))
            model = customizable_VGG(dataset=d, vgg_name=a)
            print("feature_flat_dims: {}".format(model.classifier[0].in_features))
            if d == 'MNIsT' or d == 'FashionMnISt':
                x = torch.rand(2, 1, 28, 28)
                y = model(x)
                assert y.shape == (2, 10)
            elif d == 'Cifar10':
                x = torch.rand(5, 3, 32, 32)
                y = model(x)
                assert y.shape == (5, 10)
            elif d == 'CiFAR100':
                x = torch.rand(3, 3, 32, 32)
                y = model(x)
                assert y.shape == (3, 100)

if __name__ == '__main__':
    test()
            