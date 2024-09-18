 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.1
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import os

import nibabel as nib
import numpy as np
 


def get_nii(path_to_file):
    img = nib.load(path_to_file)

    return img


def view_matrix(path):
    matrix = np.array(get_nii(path).dataobj)

    return matrix


def save_nifti(masks_list, name):
    new_image = nib.Nifti1Image(masks_list, affine = np.eye(4))
    nib.save(new_image, f'./Dataset/RS/copy_{name}')



if __name__ == '__main__':
    for name in os.listdir('./Dataset/RS/RS_results'):
        try:
            # bull_templs = view_matrix(f"./Dataset/BULLEYE_mask/{name}")
            masks = view_matrix(f'./Dataset/RS/RS_results/{name}')

            new_masks = []            

            for slc in range(masks.shape[2]):
                mask = masks[:, :, slc]

                # bull_templ = bull_templs[:, :, slc]
                # mask[mask!=0] = 1
                # for basal in range(1, 7):
                #     if (bull_templ==basal).any():
                #         mask[mask==1] = 1
                # for medial in range(7, 13):
                #     if (bull_templ==medial).any():
                #         mask[mask==1] = 2
                # for apical in range(13, 17):
                #     if (bull_templ==apical).any():
                #         mask[mask==1] = 3
                # for apex in range(17, 18):
                #     if (bull_templ==apex).any():
                #         mask[mask==1] = 4

                mask[mask == 1] = 5

                new_masks.append(mask)

            new_masks = np.array(new_masks, dtype = np.float32)
            new_masks = new_masks.transpose(1,2,0)

            save_nifti(new_masks, name)
        except:
            pass







