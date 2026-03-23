 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.1
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


from torch import nn

import torch
import cv2
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects

from matplotlib import pylab as plt
from matplotlib.backends.backend_pdf import PdfPages
from skimage.transform import resize, rescale       #pip install scikit-image
from skimage.transform import resize, rescale, downscale_local_mean

from Model.unet2D import UNet_2D, UNet_2D_AttantionLayer
from Preprocessing.preprocessing import *
from Postprocessing.postprocessing import *
from configuration import *


class GetListImages(MetaParameters):
    def __init__(self, file_path, path_to_data, dataset_path, unet_type = None, mask_type = None):
        super(MetaParameters, self).__init__()
        self.file_path = file_path
        self.file_name = file_path.split('/')[-1]
        self.path_to_data = path_to_data
        self.dataset_path = dataset_path
        self.def_coord = None
        self.__unet_type = unet_type
        self.__mask_type = mask_type
        self.cropp_gap = 8
    
    @property
    def unet_type(self):
        return self.__unet_type

    @property
    def mask_type(self):
        return self.__mask_type

    def nifti_list(self, masks):
        list_images, list_templates = [], []
        images = ReadImages(f"{self.dataset_path}{self.file_name}").view_matrix
        templates = images.copy()

        orig_img_shape = images.shape

        if masks is not None:
            images, masks, templates, self.def_coord = \
            CroppPreprocessData(images, masks, templates, \
                unet_type = self.unet_type).presegmentation_tissues(None, self.cropp_gap)

        else:
            masks = np.zeros((images.shape))

        for slc in range(images.shape[2]):
            image, mask, template = \
            PreprocessData(images[:, :, slc], masks[:, :, slc], templates[:, :, slc], \
                unet_type = self.unet_type, mask_type = self.mask_type).preprocessing

            image, mask, template = \
            MaskPreprocessing(image, mask, template, \
                mask_type = self.mask_type).mask_preprocessing

            list_images.append(image)
            list_templates.append(template)

        return list_images, list_templates, orig_img_shape, self.def_coord


class PredictionMask(MetaParameters):
    def __init__(self, model, images, templates, image_shp, def_coord, unet_type):
        super().__init__()

        self.__model = model
        self.__device = device
        self.__images = images
        self.__image_shp = image_shp
        self.__templates = templates
        self.__def_coord = def_coord
        self.__unet_type = unet_type
        self.kernel_size = chklsz.kernel_size(unet_type)    

    @property
    def model(self):
        return self.__model

    @property
    def device(self):
        return self.__device

    @property
    def images(self):
        return self.__images

    @property
    def image_shp(self):
        return self.__image_shp

    @property
    def templates(self):
        return self.__templates

    @property
    def def_coord(self):
        return self.__def_coord

    def predict(self, image):
        self.model.eval()

        with torch.no_grad():
            image = np.expand_dims(image, 1)
            image = image.transpose(1, 0, 2, 3)
            image = torch.from_numpy(image).to(self.device)

            predict = torch.softmax(self.model(image), dim = 1)
            predict = torch.argmax(predict, dim = 1).cpu()

        return predict, image

    @property
    def get_predicted_mask(self):
        mask_list = []
        smooth = 1e-6

        for slc in range(0, len(self.images)):
            image = self.images[slc]
            template = self.templates[slc]
            
            image = np.array([image, template], dtype = np.float32)[:, :, :, 0]            
            predict, image = self.predict(image)
            predict = np.reshape(predict, (self.kernel_size, self.kernel_size))
            predict = np.array(predict, dtype = np.float32)

            predict = self.expand_matrix(predict, self.image_shp[0], self.image_shp[1])
            predict = resize(predict, (self.image_shp[0], self.image_shp[1]), anti_aliasing_sigma = False)
            
            mask_list.append(predict)

        mask_list = self.postprocess_matrix(mask_list)

        return mask_list

    def expand_matrix(self, mask, row_img, column_img):
        new_matrix = np.zeros((row_img, column_img))
        
        ## After prediction of the resized and rescaled image
        if self.def_coord is None:
            row_msk, column_msk = mask.shape
            max_kernel = max(row_img, column_img)
            mask = rescale(mask, (max_kernel / mask.shape[0], max_kernel / mask.shape[1]), anti_aliasing = False, order = 0)
            new_matrix = mask[: row_img, : column_img]

        ## After prediction of cropped and rescaled image
        elif self.def_coord is not None:
            X = (self.def_coord[0] - self.CROPP_KERNEL // 2)
            Y = (self.def_coord[1] - self.CROPP_KERNEL // 2)
            new_matrix[X: X + self.CROPP_KERNEL, Y: Y + self.CROPP_KERNEL] = mask

        return new_matrix

    @staticmethod
    def postprocess_matrix(mask_list):
        shp = list(mask_list[0].shape)
        zero_matrix = np.zeros((len(mask_list), shp[0], shp[1]))

        for slc in range(len(mask_list)):
            zero_matrix[slc, :shp[0], :shp[1]] = mask_list[slc]
        
        mask_list = zero_matrix.copy()
        mask_list = np.array(mask_list, dtype = np.float32)
        mask_list = mask_list.transpose(1, 2, 0)
        mask_list = np.round(mask_list)
        
        return mask_list


class NiftiSaver(MetaParameters):
    def __init__(self, masks_list, file_path, inference_directory):         
        super(MetaParameters, self).__init__()

        self.__masks_list = masks_list
        self.__inference_directory = inference_directory
        self.__file_name = file_path.split('/')[-1]

    @property
    def masks_list(self):
        return self.__masks_list

    @property
    def inference_directory(self):
        return self.__inference_directory

    @property
    def file_name(self):
        return self.__file_name

    @property
    def save_nifti(self):
        new_image = nib.Nifti1Image(self.masks_list, affine = np.eye(4))
        nib.save(new_image, f'{self.inference_directory}/{self.file_name}')













