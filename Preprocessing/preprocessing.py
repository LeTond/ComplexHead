 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.2
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
import pydicom as dicom
import numpy as np
import pandas as pd
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import statsmodels.api as sm
import matplotlib.pyplot as plt

from torch import nn
from skimage.transform import resize, rescale, downscale_local_mean
from scipy.ndimage import rotate as rotate_image
from torch.utils.data import DataLoader
from sklearn import preprocessing        #pip install scikit-learn
from scipy import ndimage
from configuration import *


class ReadImages():
    def __init__(self, path_to_file):
        self.__path_to_file = path_to_file

    @property
    def path_to_file(self):
        return self.__path_to_file

    @property
    def get_nii(self):
        img = nib.load(self.path_to_file)

        return img

    def get_dcm(self):
        origin_dicom = dicom.dcmread(self.path_to_file)
        new_dicom = np.array(origin_dicom.pixel_array)
        
        if len(list(new_dicom.shape)) == 2:
            new_dicom = new_dicom[:, :, np.newaxis]
        else:
            new_dicom = new_dicom.transpose(2, 1, 0)

        return new_dicom

    def get_nii_fov(self):
        img = nib.load(self.path_to_file)
        return img.header.get_zooms()

    @property
    def view_matrix(self):
        # np.set_printoptions(threshold=sys.maxsize)
        return np.array(self.get_nii.dataobj)

    @property
    def get_file_list(self):
        files = os.listdir(self.path_to_file)
        files.sort()
        return files

    def get_file_path_list(self):
        path_list = []

        for root, subfolder, files in os.walk(self.path_to_file):
            for item in files:
                if item.endswith('.nii') or item.endswith('.dcm'):
                    filenamepath = str(os.path.join(root, item))
                    path_list.append(filenamepath)

        return path_list

    def get_dataset_list(self):
        return list(self.get_file_list)


class PreprocessData(MetaParameters):
    def __init__(self, image, mask = None, template = None, names = None, unet_type = None, mask_type = None):
        super().__init__()
        self.__image = image
        self.__mask = mask
        self.__template = template
        self.__names = names
        self.__mask_type = mask_type
        self.__unet_type = unet_type
        self.kernel_size = chklsz.kernel_size(unet_type)

    @property
    def names(self):
        return self.__names

    @property
    def mask_type(self):
        return self.__mask_type

    @property
    def image(self):
        return self.__image

    @property
    def mask(self):
        return self.__mask
   
    @property
    def template(self):
        return self.__template

    @property
    def preprocessing(self):
        image = np.array(self.image, dtype = np.float32)
        image = self.clipping(image)
        image = self.normalization(image)
        image = self.equalization_matrix(matrix = image)
        image = self.rescale_matrix(matrix = image, order = None)
        image = np.array(image.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)

        if self.mask is not None:
            mask = np.array(self.mask, dtype = np.float32)
            mask = self.equalization_matrix(matrix = mask)
            mask = self.rescale_matrix(matrix = mask, order = 0)
            mask = np.array(mask.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)
        else:
            mask = None

        if self.template is not None:
            template = np.array(self.template, dtype = np.float32)

            if self.mask_type != 'multy_mask':
                template = self.clipping(template)
                template = self.clahe_normalization(template)

            template = self.equalization_matrix(matrix = template)
            template = self.rescale_matrix(matrix = template, order = 0)
            template = np.array(template.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)
        else:
            template = None

        return image, mask, template

    def clipping(self, image):
        image_max = np.max(image)

        if self.CLIP_RATE is not None:
            image = np.clip(image, self.CLIP_RATE[0] * image_max, self.CLIP_RATE[1] * image_max)
        
        return image

    @staticmethod
    def normalization(image):
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        
        return image / np.max(image)

    @staticmethod
    def z_normalization(image):
        mean, std = np.mean(image), np.std(image)
        image = (image - mean) / std
        image += abs(np.min(image))
        
        return image / np.max(image)

    @staticmethod
    def hyst_normalization(image):
        minimum, maximum = 0, 4095
        cur_minimum, cur_maximum = np.min(image), np.max(image)
        
        normalyzed_image = image.copy()
        normalyzed_image = (maximum - minimum) / (cur_maximum - cur_minimum) * (normalyzed_image - cur_minimum) + minimum

        normalyzed_image[normalyzed_image < minimum] = minimum
        normalyzed_image[normalyzed_image > maximum] = maximum

        return normalyzed_image / np.max(normalyzed_image)

    @staticmethod
    def equalize_normalization(image):
        image = image / np.max(image) * 255
        image = image.astype("uint8")
        image = cv2.equalizeHist(image)
        image = image / 255

        return image

    @staticmethod
    def clahe_normalization(image):
        image = image / np.max(image) * 255
        image = image.astype("uint8")
        clahe = cv2.createCLAHE(clipLimit = 4, tileGridSize = (5, 1))
        image = clahe.apply(image)

        return image / np.max(image)

    @staticmethod
    def equalization_matrix(matrix):
        max_kernel = max(matrix.shape[0], matrix.shape[1])
        new_matrix = np.zeros((max_kernel, max_kernel))
        new_matrix[: matrix.shape[0], : matrix.shape[1]] = matrix
        matrix = new_matrix
        
        return matrix

    @staticmethod
    def center_cropping(matrix):
        y, x = matrix.shape
        min_kernel = min(matrix.shape[0], matrix.shape[1])
        startx = (x - min_kernel) // 4 * 3
        starty = (y - min_kernel) // 4 * 3
        
        return matrix[starty:starty + min_kernel, startx:startx + min_kernel]

    def rescale_matrix(self, matrix, order = None):
        shp = matrix.shape
        max_kernel = max(matrix.shape[0], matrix.shape[1])
        scale =  self.kernel_size / max_kernel
        
        return rescale(matrix, (scale, scale), anti_aliasing = False, order = order)

    @property
    def shuff_dataset(self):
        temp = list(zip(self.image, self.mask, self.template, self.names))
        random.shuffle(temp)
        images, masks, templates, names = zip(*temp)
        
        return list(images), list(masks), list(templates), list(names)


class MaskPreprocessing(MetaParameters):
    def __init__(self, image, mask = None, template = None, mask_type = None):    
        super(MetaParameters, self).__init__()
        self.__image = image
        self.__mask = mask
        self.__template = template
        self.__mask_type = mask_type

    @property
    def image(self):
        return self.__image

    @property
    def mask(self):
        return self.__mask
   
    @property
    def template(self):
        return self.__template
    
    @property
    def mask_type(self):
        return self.__mask_type

    @property
    def binary_mask_preprocessing(self):
        mask = self.mask.copy()                
        template = self.template.copy()        

        mask[mask==1] = 1
        mask[mask==3] = 1
        mask[mask==5] = 1
        mask[mask==9] = 1
        mask[mask==10] = 1
        mask[mask==11] = 1
        mask[mask==12] = 1

        mask[mask==2] = 2
        mask[mask==7] = 2
        mask[mask==8] = 2

        mask[mask==4] = 1
        mask[mask==6] = 1

        return self.image, mask, self.template

    @property
    def multy_mask_preprocessing(self):
        template = self.mask.copy()
        
        template[template>0] = 1

        return self.image, self.mask, template

    @property
    def choose_mask_preprocessing(self):
        if self.mask_type == 'multy_mask':
            return self.multy_mask_preprocessing

        elif self.mask_type == 'binary_mask':
            return self.binary_mask_preprocessing
        
        elif self.mask_type is None:
            return self.image, self.mask, self.template
        
        else:
            return self.lv_preprocessing

    @property
    def mask_preprocessing(self):
        return self.choose_mask_preprocessing


class Augmentation(MetaParameters):
    def __init__(self, image, mask = None, template = None, unet_type = None):
        super().__init__()
        self.__image = image
        self.__mask = mask
        self.__template = template
        self.__unet_type = unet_type
        self.kernel_size = chklsz.kernel_size(unet_type)

    @property
    def unet_type(self):
        return self.__unet_type

    @property
    def image(self):
        return self.__image

    @property
    def mask(self):
        return self.__mask
   
    @property
    def template(self):
        return self.__template

    @property
    def define_kernel_size(self):
        return self.image.shape[0]

    @property
    def angle_list(self):
        if self.AUGMENTATION:
            angle_list = list(set([random.choice([0, 90, 180, 270]) for i in range(2)]))
        
        else: 
            angle_list = [0]

        return angle_list

    @property
    def angle(self):
        return random.choice(self.angle_list)

    @property
    def rotate_2d(self):
        angle = self.angle
        image = rotate_image(self.image, angle)
        mask = rotate_image(self.mask, angle)

        if self.template is not None:
            template = rotate_image(self.template, angle)
        else:
            template = None

        return image, mask, template

    @property
    def gauss_noise(self):
        sigma, mean = 2, 0.5
        
        noise = np.random.normal(mean, sigma ** 0.5, self.image.shape)
        noisy_image = self.image + noise
        
        return noisy_image

    @property
    def rician_noise_transforms(self):
        random_v = random.choice([10, 20, 5, 25])
        random_s = random.choice([50, 25, 30, 70])
        num_samples = self.kernel_size * self.kernel_size

        try:    
            noise = np.random.normal(scale = random_s, size = (num_samples, 2)) + [[random_v, 0]]
            noise = np.linalg.norm(noise, axis = 1)
            noise = np.array(noise.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)
            
            noisy_image = self.image + noise

        except:
            print('Rician Noise Application Error')
        
        return noisy_image




