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
# import pydicom as dicom
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



class ReadImages():
    def __init__(self, path_to_file):
        self.path_to_file = path_to_file

    def get_nii(self):
        # matplotlib.use('TkAgg')
        img = nib.load(self.path_to_file)
        return img

    def get_dcm(self):
        origin_dicom = dicom.dcmread(self.path_to_file)
        new_dicom = np.array(origin_dicom.pixel_array)
        
        if len(list(new_dicom.shape)) == 2:
            new_dicom = new_dicom[:, :, np.newaxis]
        
        return new_dicom

    def get_nii_fov(self):
        # matplotlib.use('TkAgg')
        img = nib.load(self.path_to_file)
        return img.header.get_zooms()

    def view_matrix(self):
        return np.array(self.get_nii().dataobj)

    def get_file_list(self):
        files = os.listdir(self.path_to_file)
        files.sort()
        return files

    def get_file_path_list(self):
        path_list = []

        for root, subfolder, files in os.walk(self.path_to_file):
            for item in files:
                if item.endswith('.nii') or item.endswith('.dcm'):
                    filenamepath = str(os.path.join(root, item)).split('/')[-1]
                    path_list.append(filenamepath)

        return path_list

    def get_dataset_list(self):
        return list(self.get_file_list())


class NiftiSaver():
    def __init__(self, file_name):         
        self.file_name = file_name

    def save_new_mask(self):
        old_mask = ReadImages(f'./Dataset/ALMAZ_mask/{self.file_name}').view_matrix()

        old_mask[old_mask==0] = 20
        old_mask[old_mask==1] = 20
        old_mask[old_mask==2] = 0
        old_mask[old_mask==3] = 0
        old_mask[old_mask==4] = 19

        new_image = nib.Nifti1Image(old_mask, affine = np.eye(4))
        nib.save(new_image, f'./Dataset/ALMAZ_BULL/{self.file_name}')



class BullEyeContour:
    def __init__(self, path_to_files):
        self.path_to_files = path_to_files

    def get_old_mask(self):
        dataset_list = ReadImages(self.path_to_files).get_file_path_list()
        return dataset_list

    def save_new_mask(self):
        for sub in self.get_old_mask():
            NiftiSaver(sub).save_new_mask()



if __name__ == "__main__":
    beye = BullEyeContour('./Dataset/ALMAZ_mask/')
    beye.save_new_mask()





