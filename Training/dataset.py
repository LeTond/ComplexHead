 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.2
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import torch
import numpy as np
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from multiprocessing import Pool, TimeoutError, current_process

from Preprocessing.preprocessing import *
from configuration import MetaParameters
from Preprocessing.dirs_logs import *
from Inference.inference import *


class GetData(MetaParameters):
    def __init__(self, files = None, augmentation = None):
        super(MetaParameters, self).__init__()
        self.files = files
        self.augmentation = augmentation
        
    @property
    def unet_type(self):
        if self.UNET2 is True:
            return 'default'
        elif self.UNET1 is True and self.UNET2 is False:
            return 'default'
        else:
            raise ValueError 

    @property
    def mask_type(self):
        if self.UNET2 is True:
            return 'multy_mask'
        elif self.UNET1 is True and self.UNET2 is False:
            return 'binary_mask'
        else:
            return None

    @property
    def create_dict_class(self):
        dict_class_stats = {}
  
        dict_class_stats.update(
                {
                    f'{self.DICT_CLASS[key]}' : 
                        {'Subjects': 0, 'pixels': 0} for key in range(1, self.NUM_CLASS)
                }
            )

        return dict_class_stats

    def count_pathology(self, sub_names):
        diction = self.create_dict_class

        for sub_name in sub_names:
            if sub_name.endswith('.nii'):
                masks = ReadImages(f"{self.MASKS_DIR}/{sub_name}").view_matrix

                for key in range(1, self.NUM_CLASS):
                    if (masks == key).any():
                        diction[f'{self.DICT_CLASS[key]}'].update(
                            {
                                'Subjects': diction[f'{self.DICT_CLASS[key]}']['Subjects'] + 1,
                                'pixels': diction[f'{self.DICT_CLASS[key]}']['pixels'] + masks[masks == key].sum().item()
                            }
                        )

        return diction

    def check_mask(self, mask, sub_name, slc):
        if self.EMPTY is False and (mask > 0).any() is False:
            print(f"Subject {sub_name} slice {slc} was passed because EMPY is FALSE")
            return False

        elif (mask > (self.NUM_CLASS - 1)).any():
            print(f"Subject {sub_name} slice {slc} has class out of range class {self.NUM_CLASS}")
            return False
        
        else:
            return True

    def pool_worker(self, file_name):
        list_images, list_masks, list_templates, list_names = [], [], [], []

        if file_name.endswith('.nii'):            
            images = ReadImages(f"{self.ORIGS_DIR}/{file_name}").view_matrix
            masks = ReadImages(f"{self.MASKS_DIR}/{file_name}").view_matrix

            sub_name = file_name.replace('.nii', '')
            if self.mask_type == 'multy_mask':
                templates = ReadImages(f"{self.MASKS_DIR}/{file_name}").view_matrix

            else:
                templates = images.copy()

            for slc in range(images.shape[2]):
                image = images[:, :, slc]
                mask = masks[:, :, slc]
                template = templates[:, :, slc]

                try:
                    image, mask, template = \
                    Augmentation(image, mask, template, unet_type = self.unet_type).rotate_2d
                except:
                    print(f'Data Augmentation Problem with {sub_name}')
            
                try:
                    image, mask, template = \
                    PreprocessData(image, mask, template, unet_type = None, mask_type = self.mask_type).preprocessing

                except:
                    print(f'Data Preprocessing Problem with {sub_name}')
                
                try:
                    image, mask, template = \
                    MaskPreprocessing(image, mask, template, mask_type = self.mask_type).mask_preprocessing
                except:
                    print(f'Data MaskPreprocessing Problem with {sub_name}')

                if self.check_mask(mask, sub_name, slc):    
                    list_images.append(image)
                    list_masks.append(mask)
                    list_templates.append(template)
                    list_names.append(f'{sub_name} Slice {images.shape[2] - slc}')

        return list_images, list_masks, list_templates, list_names

    @property
    def generated_data_list(self):
        list_images, list_masks, list_templates, list_names = [], [], [], []

        for subject in self.files:
            try:
                images, masks, templates, sub_names = self.pool_worker(subject)
                for slc in range(len(images)): 
                    list_images.append(images[slc])
                    list_masks.append(masks[slc])
                    list_templates.append(templates[slc])
                    list_names.append(sub_names[slc])
            except:
                pass 

        if self.AUGMENTATION and self.augmentation:
            for subject in self.files:
                try:
                    images, masks, templates, sub_names = self.pool_worker(subject)
                    for slc in range(len(images)): 
                        list_images.append(images[slc])
                        list_masks.append(masks[slc])
                        list_templates.append(templates[slc])
                        list_names.append(sub_names[slc])
                except:
                    pass 

        try:
            list_images, list_masks, list_templates, list_names = \
            PreprocessData(list_images, list_masks, list_templates, list_names).shuff_dataset
        except:
            print('Shuffle was broken')
            pass

        return list_images, list_masks, list_templates, list_names


class MyDataset(Dataset, MetaParameters):
    def __init__(self, ds_images, ds_masks, ds_templates, ds_names, transform = None, images_and_labels = []):
        super().__init__()

        self.transform = transform
        self.images_and_labels = images_and_labels
        self.images = ds_images
        self.masks = ds_masks
        self.templates = ds_templates
        self.names = ds_names
        self.kernel_size = chklsz.kernel_size(None)

        for i in range(len(self.images)):
            self.images_and_labels.append((i, i, i, i))

    def preprocessing(self, image, mask, template):
        image = TF.to_pil_image(image)
        image = TF.pil_to_tensor(image)

        mask = mask / self.NUM_CLASS
        mask = TF.to_pil_image(mask)
        mask = TF.pil_to_tensor(mask)

        template = TF.to_pil_image(template)
        template = TF.pil_to_tensor(template)

        tcat = torch.cat((image, mask, template), 0)
        image, mask, template = self.transform(tcat)

        image = np.array(image.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)
        mask = np.array(mask.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)
        template = np.array(template.reshape(self.kernel_size, self.kernel_size, 1), dtype = np.float32)

        mask = np.round(mask * self.NUM_CLASS)
        
        return image, mask, template
        
    def __getitem__(self, item):
        imgs, labs, templs, sub_nms = self.images_and_labels[item]
        image = self.images[imgs][:][:]
        mask = self.masks[labs][:][:]
        sub_names = self.names[sub_nms]
        
        template = self.templates[templs][:][:]
        image, mask, template = self.preprocessing(image, mask, template)

        image = np.array([image, template], dtype = np.float32)[:, :, :, 0]

        mask = np.resize(mask, (self.kernel_size, self.kernel_size))
        mask = np.array(mask, dtype = np.int8)
        mask = np.eye(self.NUM_CLASS)[mask]
        mask = np.array(mask, dtype = np.float32)
        mask = mask.transpose(2, 0, 1)

        return image, mask, sub_names

    def __len__(self):
        return len(self.images)

