 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.1
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import torch
import random
import sys
import time
import cv2
import matplotlib
import os
import pickle
import platform

import nibabel as nib
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import torchvision.transforms as transforms
import statsmodels.api as sm
import matplotlib.pyplot as plt
import albumentations as A

from torch import nn
from torch.utils.data import DataLoader
from sklearn import preprocessing  # pip install scikit-learn

from Preprocessing.dirs_logs import FileDirectoryWorker


########################################################################################################################
# Show software and harware
########################################################################################################################
print(f"Python Platform: {platform.platform()}")
print(f'python version: {sys.version}')
print(f'torch version: {torch.__version__}')
print(f'numpy version: {np.__version__}')
print(f'pandas version: {pd.__version__}')


class MetaParameters:
    FREEZE_BN = True
    PRETRAIN = True
    AUGMENTATION = True
    NOISE = True
    EMPTY = True
    MULTYGAP = True
    UNET1 = True
    UNET2 = True

    AUGMENTATION = False
    FREEZE_BN = False
    PRETRAIN = False
    NOISE = False
    EMPTY = False
    MULTYGAP = False
    # UNET1 = False
    UNET2 = False

    """ Network configuration """
    KERNEL = 256
    CHANNELS = 2
    LR = 1e-3
    BT_SZ = 32      # 512 x 512 - max batch_size == 8 if features == 32
    EPOCHS = 1000
    DROPOUT = 0.1
    FEATURES = 8   # 32 - 
    WDC = 1e-4
    EARLY_STOPPING = 50
    TMAX = 50
    SHUFFLE = True
    # CLIP_RATE = [0.1, 0.9]
    CLIP_RATE = None
    
    BINARY_DICT_CLASS = {0: "Background", 1: "Hyper", 2: "Hypo", # 3: "Semi", 
                  }

    MULTY_DICT_CLASS = {
                0: "Background",        
                1: "Lesion",                
                2: "Cyst",                  
                3: "Leikoareosis",          
                4: "Meningioma",            
                5: "Multiplesclerosis",     
                6: "Poststroke",            
                7: "Cavernoma",     
                8: "AVM",
                9: "MTS",
                10: "Edema",
                11: "Acutestroke",
                12: "Subacutestroke",
                }

    BINARY_CE_WEIGHTS = torch.FloatTensor([0.2, 0.4, 0.9])
    MULTY_CE_WEIGHTS = torch.FloatTensor([0.05, 0.7, 0.3, 0.7, 0.7, 0.9, 0.9, 0.9, 0.7, 0.7, 0.7, 0.7, 0.9])
    
    if UNET2 is True:
        DICT_CLASS = MULTY_DICT_CLASS
        CE_WEIGHTS = MULTY_CE_WEIGHTS
            
    elif UNET1 is True and UNET2 is False:
        DICT_CLASS = BINARY_DICT_CLASS
        CE_WEIGHTS = BINARY_CE_WEIGHTS

    NUM_CLASS = len(DICT_CLASS)

    """ Project configuration """
    FOLD_NAME = 'full'  # [01:05] xor full
    DATASET_NAME = 'Complex_HEAD'
    
    MODEL_NAME = 'model_best'
    DATASET_DIR = './Dataset/'

    ## NEW DIRECTNAMES
    UNET1_FOLD = f'Unet1_Fold_{FOLD_NAME}/'
    UNET2_FOLD = f'Unet2_Fold_{FOLD_NAME}/'

    PROJ_NAME = f'./Results/{DATASET_NAME}'

    ORIGS_DIR = f'{DATASET_DIR}{DATASET_NAME}_images'
    MASKS_DIR = f'{DATASET_DIR}{DATASET_NAME}_masks'

    NEW_DATA_PATH = f'./Dataset/{DATASET_NAME}_images_new/'
    NEW_UNET1_MASK_PATH = f'./Dataset/{DATASET_NAME}_Unet1_mask_new/'
    NEW_UNET2_MASK_PATH = f'./Dataset/{DATASET_NAME}_Unet2_mask_new/'


class ChooseDevice:
    def __init__(self):    
        self.print_device

    @staticmethod
    def __device():
        if torch.backends.mps.is_available():
            device = torch.device('mps')
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        return device

    @property
    def device(self):
        return self.__device()

    @property
    def print_device(self):
        print(self.device)


global device


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight = None, gamma = 2,reduction = 'mean'):    #reduction='sum'
        super(FocalLoss, self).__init__(weight,reduction = reduction)
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target,reduction = self.reduction,weight = self.weight)
        pt = torch.exp( - ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
        

class ChooseKernelSize(MetaParameters):
    def __init__(self):    
        super().__init__()

    def matrix_size(self, unet_type = None):
        if unet_type is None:
            if self.UNET2 is True:
                return 'default'
            elif self.UNET1 is True and self.UNET2 is False:
                return 'default'
            else: 
                raise ValueError("Check UNET configuration. Make shure that all UNET MetaParameters before target UNET(N) is True")
        else:
            return unet_type

    def kernel_size(self, unet_type = None):
        matrix_size = self.matrix_size(unet_type)

        if matrix_size == 'default':
            return self.KERNEL
        else:
            raise ValueError("Check matrix size value. Make shure that input value is correct for chosen matrix preprocessing")


class ChooseLossFunction(MetaParameters):
    def __init__(self):
        super(MetaParameters, self).__init__()
        self.print_loss_function()

    @property
    def choose_loss_function(self):
        try:
            loss_function = nn.CrossEntropyLoss(weight = self.CE_WEIGHTS).to(device)
            return loss_function

        except:
            loss_function = nn.CrossEntropyLoss().to(device)
            return loss_function

    @property
    def loss_function(self):
        return self.choose_loss_function

    def print_loss_function(self):
        if self.CE_WEIGHTS is not None:
            print(f'{self.loss_function} With {self.CE_WEIGHTS} Was Chosen !!!')
        else:            
            print(f'{self.loss_function} without CE_WEIGHTS Was Chosen !!!')


device = ChooseDevice().device
meta = MetaParameters()
fdwr = FileDirectoryWorker()
chklsz = ChooseKernelSize()
loss_function = ChooseLossFunction().loss_function


########################################################################################################################
## Main image transforms in Dataloder
########################################################################################################################
default_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

transform_01 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomRotation((-10, 10), expand = False),
    transforms.RandomHorizontalFlip(0.7),
    transforms.RandomVerticalFlip(0.7),
    transforms.ToTensor(),
])

transform_02 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomVerticalFlip(1.0),
    transforms.ToTensor(),
])

transform_03 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(1.0),
    transforms.ToTensor(),
])

transform_04 = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomVerticalFlip(0.5),
    transforms.RandomAffine(degrees = (-2, 2), translate = (0.05, 0.25), scale = (0.75, 1.25)),
    transforms.ToTensor(),
])


