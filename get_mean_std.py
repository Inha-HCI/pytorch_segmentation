'''Get dataset mean and std with PyTorch.'''
from __future__ import print_function

import logging
from datetime import datetime
from copy import deepcopy
import re

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from torchvision import datasets

import os
import argparse
import numpy as np
import models
import utils
import time
from dataloaders.khnp import KHNP

if __name__ == '__main__':
        
        
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.ImageFolder(root='./images',
                                transform=transforms.ToTensor())
    image_means = torch.stack([t.mean(1).mean(1) for t, c in train_dataset])
    
    print("mean: " + str(image_means.mean(0)))
    print("std: " + str(image_means.std(0)))
    print()
    print('Done!')