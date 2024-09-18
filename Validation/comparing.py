 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.1
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


from configuration import MetaParameters
from Preprocessing.preprocessing import ReadImages
from scipy.ndimage import _ni_support
from scipy.spatial.distance import directed_hausdorff
from medpy import metric
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
# from sklearn.metrics import confusion_matrix

import numpy as np
import nibabel as nib
import pandas as pd

import numpy



class CompareMatrix(MetaParameters):
    def __init__(self, path_to_label: str, path_to_prediction: str, layer: int, path_to_bull_templs: str = None):
        super(MetaParameters, self).__init__()

        self.path_to_label = path_to_label
        self.path_to_prediction = path_to_prediction
        self.path_to_bull_templs = path_to_bull_templs
        self.layer = layer

        self.label = self.load_matrix(self.path_to_label, layer, False, 'etalon')
        self.prediction = self.load_matrix(self.path_to_prediction, layer, True, 'predict')
        # self.bull_templs = self.load_matrix(self.path_to_bull_templs, layer, True, 'predict')
        
        self.length = self.label.shape[-1]

        self.smooth = 1e-5
        self.GT = self.label.sum()
        self.CM = self.prediction.sum()
        self.TP = (self.label * self.prediction).sum()
        self.FN = np.abs(self.GT - self.TP)
        self.FP = np.abs(self.CM - self.TP)

    def sub_name(self):
        name = self.path_to_label.split('/')[-1]
        name = name.rstrip('.nii')
        
        return name

    @staticmethod
    def load_matrix(path_to_matrix: str, layer: int, cond: bool, marker: str):
        matrix = nib.load(path_to_matrix)
        matrix = np.array(matrix.dataobj)

        # if cond is True:
        # for slc in range(matrix.shape[-1]):        
        #     if matrix[:,:,slc][matrix[:,:,slc]==3].sum().item() < 10:
        #         matrix[:,:,slc][matrix[:,:,slc]==3] = 2

        ##   Myo + Fib
        # if layer == 2:
        #     matrix[matrix==3] = 2

        matrix[matrix != layer] = 0
        matrix[matrix == layer] = 1

        return matrix                                     

    def dice_2d(self):
        list_dsc = []
        for slc in range(self.length):
            self.GT = self.label[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()

            # if self.GT == 0:
            #     pass
            # else:
            self.TP = (self.label[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)
        
            # print(f'{self.sub_name()}: FN pixels {self.FN}, FP pixels {self.FP}, TP pixels: {self.TP}')
            # print(f'{self.sub_name()} Slice: {self.length - slc} Dice = {self.dice()}')
            list_dsc.append(self.dice())

        # print(f"Sub {self.sub_name()}: {list_dsc}")

        return list_dsc

    def recall_2d(self):
        for slc in range(self.length):
            self.GT = self.label[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()
            self.TP = (self.label[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)
            
            print(f'{self.sub_name()} Slice: {self.length - slc} Recall = {self.recall()}')

    def precision_2d(self):
        for slc in range(self.length):
            self.GT = self.label[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()
            self.TP = (self.label[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)
            
            print(f'{self.sub_name()} Slice: {self.length - slc} Precision = {self.precision()}')


    @staticmethod
    def surface_distances(result, reference, voxelspacing = None, connectivity = 1):
        """
        The distances between the surface voxel of binary objects in result and their
        nearest partner surface voxel of a binary object in reference.
        """
        result = numpy.atleast_1d(result.astype(np.bool_))
        reference = numpy.atleast_1d(reference.astype(np.bool_))
        if voxelspacing is not None:
            voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
            voxelspacing = numpy.asarray(voxelspacing, dtype = numpy.float64)
            if not voxelspacing.flags.contiguous:
                voxelspacing = voxelspacing.copy()
                
        # binary structure
        footprint = generate_binary_structure(result.ndim, connectivity)
        
        # test for emptiness
        if 0 == numpy.count_nonzero(result): 
            raise RuntimeError('The first supplied array does not contain any binary object.')
        if 0 == numpy.count_nonzero(reference): 
            raise RuntimeError('The second supplied array does not contain any binary object.')    
                
        # extract only 1-pixel border line of objects
        result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
        reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)
        
        # compute average surface distance        
        # Note: scipys distance transform is calculated only inside the borders of the
        #       foreground objects, therefore the input has to be reversed
        dt = distance_transform_edt(~reference_border, sampling = voxelspacing)
        sds = dt[result_border]
        
        return sds

    def hd(self, result, reference, voxelspacing = None, connectivity = 1):
        try:
            hd1 = self.surface_distances(result, reference, voxelspacing, connectivity).max()
            hd2 = self.surface_distances(reference, result, voxelspacing, connectivity).max()
            hd = max(hd1, hd2)
        except:
            hd = 0

        return hd

    def hd95(self, result, reference, voxelspacing = None, connectivity = 1):
        hd1 = self.surface_distances(result, reference, voxelspacing, connectivity).max()
        hd2 = self.surface_distances(reference, result, voxelspacing, connectivity).max()
        
        hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
        
        return hd95

    def hausdorff_distance_2d(self): 
        for slc in range(self.length):
            self.GT = self.label[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()
            self.TP = (self.label[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)
            
            drh = self.hd(self.label[:, :, slc], self.prediction[:, :, slc], 2, 1)
            # drh = self.hd95(self.label[:,:,slc], self.prediction[:,:,slc], 2, 1)
            # drh = directed_hausdorff(self.label[:,:,slc], self.prediction[:,:,slc])
            # drh = max(directed_hausdorff(self.label[:,:,slc], self.prediction[:,:,slc], 2)[0], directed_hausdorff(self.prediction[:,:,slc], self.label[:,:,slc], 2)[0])
            print(f'{self.sub_name()} Slice: {self.length - slc} Hausdorff Distance = {drh}')


            # voxel_spacing = np.array(self.label[:,:,slc].GetSpacing())[::-1]
            # print(voxel_spacing)

    def fpr_2d(self):
        for slc in range(self.length):
            self.GT = self.label[:, :, slc].sum()
            self.CM = self.prediction[:, :, slc].sum()
            self.TP = (self.label[:, :, slc] * self.prediction[:, :, slc]).sum()
            self.FN = np.abs(self.GT - self.TP)
            self.FP = np.abs(self.CM - self.TP)        
            self.TN = np.abs(self.CROPP_KERNEL * self.CROPP_KERNEL - self.GT - self.FP)

            print(f'{self.sub_name()} Slice: {self.length - slc} FPR = {self.fpr()}')

        fpr = round(float((self.FP + self.smooth) / (self.FP + self.TN + self.smooth)), 3)

    def dice(self):
        dice = round(float((2 * self.TP + self.smooth) / (2 * self.TP + self.FP + self.FN + self.smooth)), 3)

        return dice

    def recall(self):
        recall = round(float((self.TP + self.smooth) / (self.TP + self.FN + self.smooth)), 3)    

        return recall

    def precision(self):
        precision = round(float((self.TP + self.smooth) / (self.TP + self.FP + self.smooth)), 3)

        return precision

    def fpr(self):
        # self.TN = np.abs(192 * 144 - (self.TP + self.FP + self.FN))
        # fpr = round(float((self.FP + self.smooth) / (self.FP + self.TN + self.smooth)), 3)
        
        tn = int(((self.label == 0) * (self.prediction == 0)).sum())
        fpr = 1 - (round(float((tn + self.smooth) / (self.FP + tn + self.smooth)), 3))
        
        return fpr

    def hausdorff_distance(self):  
        drh = max(directed_hausdorff(self.label, self.prediction, 2)[0], directed_hausdorff(self.prediction, self.label, 2)[0])
        # drh = self.hd(u, v, 2, 2)

        return drh

    def jaccard(self):
        jac = round(float((self.TP + self.smooth) / (self.TP + self.FP + self.FN + self.smooth)), 3)

        return jac

    def tissue_volume(self, matrix):
        fov = ReadImages(self.path_to_label).get_nii_fov()
        volume_size = fov[0] * fov[1] * fov[2]
        mask_volume = round(matrix.sum().item() / 1000 * volume_size, 2)

        return mask_volume

    def tissue_volume_2d(self):
        for slc in range(self.length):     
            print(
                f'{self.sub_name()} Slice: {self.length - slc} GT volume = {self.tissue_volume(self.label[:, :, slc])} ml'
                f'Slice: {self.length - slc} CM volume = {self.tissue_volume(self.prediction[:, :, slc])} ml'
                )

    def pixels_count(self, matrix):
        
        return matrix.sum()

    def pixel_count_2d(self):
        for slc in range(self.length):     
            print(
                f'{self.sub_name()} '
                f'Slice: {self.length - slc} '
                f'GT pixels = {self.pixels_count(self.label[:, :, slc])} '
                f'CM pixels = {self.pixels_count(self.prediction[:, :, slc])}'
                )

    def print(self):
        # print(f"Statistics was counted for {self.DICT_CLASS[self.layer]} tissue")
        print(
            f'{self.sub_name()}: '
            f' Mean Dice = {self.dice()}, '
        #     f' Mean Recall = {self.recall()}, '
        #     f' Mean Precision = {self.precision()}, '
            # f' Mean Jaccard = {self.jaccard()}, '
            # f' Mean HD = {self.hausdorff_distance()}, '
            # f' Mean FPR = {self.fpr()}, '
            )
        # print(f'{self.sub_name()}: FN pixels {self.FN}, FP pixels {self.FP}, TP pixels: {self.TP}')

        # print(
        #     f'{self.sub_name()}: '
        #     f' GT volume = {self.tissue_volume(self.label)} ml, '
        #     f' CM volume = {self.tissue_volume(self.prediction)} ml'
        #     f' Difference = {round(self.tissue_volume(self.label) - self.tissue_volume(self.prediction), 2)} ml'
        #     )
        
        # print(
        #     f'{self.sub_name()}: '
        #     f' Count of GT pixels = {self.pixels_count(self.label)} '
        #     f' Count of CM pixels = {self.pixels_count(self.prediction)}'
        #     )

        # self.dice_2d()
        # self.recall_2d()
        # self.precision_2d()
        # self.fpr_2d()
        # self.hausdorff_distance_2d()
        # self.tissue_volume_2d()
        # self.pixel_count_2d()





