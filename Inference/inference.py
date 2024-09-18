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

        if self.mask_type == 'infer_bull_level':
            templates = ReadImages(f'./Dataset/ALMAZ_Unet3_mask_new/{self.file_name}').view_matrix

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

    @staticmethod
    def old_dicom(file_path):
        old_dicom = dicom.dcmread(file_path)
        old_dicom = old_dicom.PatientName

        return old_dicom

    def dicom_array(self, def_coord = None, masks = None):
        list_images, list_templates = [], []
        folder_name = self.old_dicom(self.file_path)

        images = ReadImages(f"{self.file_path}").get_dcm()
        templates = images.copy()

        orig_img_shape = images.shape

        if self.mask_type == 'infer_bull_level':
            templates = ReadImages(f'./Dataset/ALMAZ_Unet3_mask_new/{self.file_name}').view_matrix

        if masks is not None:
            images, masks, templates, def_coord = \
            CroppPreprocessData(images, masks, templates, \
                unet_type = self.unet_type).presegmentation_tissues(def_coord, self.cropp_gap)
        
        else:
            masks = np.zeros((images.shape))

        for slc in range(images.shape[2]):
            image, mask, template = \
            PreprocessData(images[:, :, slc], masks[:, :, slc], templates[:, :, slc], \
                unet_type = self.unet_type, mask_type = self.mask_type).preprocessing
            # image, mask, template = \
            # PreprocessData(images[:, :, slc], None, templates[:, :, slc], \
            #     unet_type = self.unet_type, mask_type = self.mask_type).preprocessing
            
            image, mask, template = \
            MaskPreprocessing(image, mask, template, \
                mask_type = self.mask_type).mask_preprocessing

            list_images.append(image)
            list_templates.append(template)

        return list_images, list_templates, orig_img_shape, def_coord


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
            
            predict = self.threshhold_myo_level(predict)
            predict = self.threshhold_prediction(predict)
            predict = self.expand_matrix(predict, self.image_shp[0], self.image_shp[1])
            predict = resize(predict, (self.image_shp[0], self.image_shp[1]), anti_aliasing_sigma = False)
            
            mask_list.append(predict)

        mask_list = self.postprocess_matrix(mask_list)

        return mask_list

    def threshhold_myo_level(self, predict):
        if self.UNET4 is True and self.UNET5 is False:
            try: 
                unique, counts = np.unique(predict, return_counts = True)
                test_dict = dict(zip(unique, counts))
                myo_level = int(list(test_dict.keys())[0])

                if myo_level != 0:
                    predict[predict != 0] = myo_level
                else:
                    myo_level = int(list(test_dict.keys())[1])
                    predict[predict != 0] = myo_level

            except:
                pass

        return predict

    def threshhold_prediction(self, predict):
        try:
            if self.DICT_CLASS[2] == 'MYO' and self.DICT_CLASS[3] == 'FIB':
                pred_fib = predict[predict == 3]            
                pred_myo = predict[predict == 2]
                rel_volume = (pred_fib.sum().item() + smooth) / (pred_fib.sum().item() + pred_myo.sum().item() + smooth) * 100
                
                if rel_volume < 2 and (predict == 3).sum().item() > 0:
                    predict[predict == 3] = 2
        except:
            pass

        return predict

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


class DicomSaver(MetaParameters):
    def __init__(self, masks_list, file_path, inference_directory):         
        super(MetaParameters, self).__init__()

        self.masks_list = masks_list
        self.file_name = file_path
        self.inference_directory = inference_directory
        self.orig_dir = self.NEW_DATA_PATH

    def old_dicom(self):
        old_dicom = dicom.dcmread(self.file_name)

        return old_dicom

    def change_name(self, old_dicom):
        seq_name = old_dicom[0x0018, 0x1030]
        seq_name.value += '_Mask'
        seq_number = old_dicom[0x0020, 0x0011]
        seq_number.value = int(seq_number.value) + 1000

        return old_dicom        

    def change_grey_to_color(self, old_dicom):
        old_dicom.PhotometricInterpretation = 'RGB'
        old_dicom.SamplesPerPixel = 3
        old_dicom.BitsAllocated = 8
        old_dicom.BitsStored = 8
        old_dicom.HighBit = 7
        old_dicom.add_new(0x00280006, 'US', 0)

        return old_dicom

    def new_dicom_array(self):
        dcm2 = self.old_dicom().pixel_array

        if len(list(dcm2.shape)) == 2:
            new_dicom_array = cv2.cvtColor(dcm2, cv2.COLOR_GRAY2RGB)
            new_dicom_array = new_dicom_array / 4095 * 255
            new_dicom_array = new_dicom_array.astype(np.uint8)

            mask = self.masks_list[:,:,0].astype(np.float16)
            
            # new_dicom_array[:,:,2][mask == 1] += 100
            # new_dicom_array[:,:,2][mask == 2] -= 150
            # new_dicom_array[:,:,1][mask == 3] -= 220

            new_dicom_array[:,:,0][mask == 1] = 51
            new_dicom_array[:,:,1][mask == 1] = 51
            new_dicom_array[:,:,2][mask == 1] = 255

            new_dicom_array[:,:,0][mask == 2] = 204
            new_dicom_array[:,:,1][mask == 2] = 204
            new_dicom_array[:,:,2][mask == 2] = 0

            new_dicom_array[:,:,0][mask == 3] = 0
            new_dicom_array[:,:,1][mask == 3] = 153
            new_dicom_array[:,:,2][mask == 3] = 0

        else:
            new_dicom_array = np.zeros((dcm2.shape[0], dcm2.shape[1], 3, dcm2.shape[2]))

            for slc in range(dcm2.shape[2]):
                new_dicom_array[:,:,:,slc] = cv2.cvtColor(dcm2[:,:,slc], cv2.COLOR_GRAY2RGB)

            new_dicom_array = new_dicom_array / 4095 * 255
            new_dicom_array = new_dicom_array.astype(np.uint8)
            mask = self.masks_list[:,:,:].astype(np.float16)
            mask = mask.transpose(2, 1, 0)
            
            mask = np.expand_dims(mask, -2)

            for slc in range(mask.shape[3]):
                masks = mask[:,:,0,slc]
                # new_dicom_array[:,:,2,slc][masks == 1] = 220
                # new_dicom_array[:,:,1,slc][masks == 2] = 150
                # new_dicom_array[:,:,2,slc][masks == 3] = 100

                new_dicom_array[:,:,0,slc][masks == 1] = 51
                new_dicom_array[:,:,1,slc][masks == 1] = 51
                new_dicom_array[:,:,2,slc][masks == 1] = 255

                new_dicom_array[:,:,0,slc][masks == 2] = 204
                new_dicom_array[:,:,1,slc][masks == 2] = 204
                new_dicom_array[:,:,2,slc][masks == 2] = 0
                
                new_dicom_array[:,:,0,slc][masks == 3] = 0
                new_dicom_array[:,:,1,slc][masks == 3] = 153
                new_dicom_array[:,:,2,slc][masks == 3] = 0

            new_dicom_array = new_dicom_array.transpose(0, 1, 3, 2)

        return new_dicom_array

    def new_dicom_array_3d(self):
        dcm2 = self.old_dicom().pixel_array
        # dcm2 = dcm2.transpose(2,1,0)

        new_dicom_array = np.zeros((dcm2.shape[0], dcm2.shape[1], 3, dcm2.shape[2]))

        for slc in range(dcm2.shape[2]):
            new_dicom_array[:,:,:,slc] = cv2.cvtColor(dcm2[:,:,slc], cv2.COLOR_GRAY2RGB)

        # new_dicom_array = cv2.cvtColor(dcm2, cv2.COLOR_GRAY2RGB)
        new_dicom_array = new_dicom_array / 4095 * 255
        new_dicom_array = new_dicom_array.astype(np.uint8)
        mask = self.masks_list[:,:,:].astype(np.float16)
        mask = mask.transpose(2, 1, 0)
        
        mask = np.expand_dims(mask, -2)

        for slc in range(mask.shape[3]):
            msk = mask[:,:,0,slc]
            new_dicom_array[:,:,2,slc][msk == 1] = 220
            new_dicom_array[:,:,1,slc][msk == 2] = 150
            new_dicom_array[:,:,2,slc][msk == 3] = 100

        new_dicom_array = new_dicom_array.transpose(0, 1, 3, 2)

        return new_dicom_array


    def change_value_range_info(self, old_dicom):
        old_dicom.SmallestImagePixelValue = np.min(self.new_dicom_array())
        old_dicom.LargestImagePixelValue = np.max(self.new_dicom_array())

        return old_dicom

    def dicom_file_name(self):
        new_file_name = self.file_name.split('/')[-1]
        
        return new_file_name

    def save_dicom_mask(self):
        old_dicom = self.change_name(self.old_dicom())
        
        # dcm2 = self.old_dicom().pixel_array
        
        if len(list(old_dicom.pixel_array.shape)) == 2:
            mask = self.masks_list[:,:,0].astype(np.float16)
            old_dicom.PixelData = mask.tostring()
        else:
            mask = self.masks_list[:,:,:].astype(np.float16)
            mask = mask.transpose(2, 1, 0)
            old_dicom.PixelData = mask.tostring()

        # mask = self.masks_list[:,:,0].astype(np.float16)
        # old_dicom.PixelData = mask.tostring()
        new_dir_name = old_dicom.PatientName           
        fdwr.create_dir(project_name = f'{self.inference_directory}/{new_dir_name}')
        old_dicom.save_as(f'{self.inference_directory}/{new_dir_name}/{self.dicom_file_name()}')


    def save_dicom_mask_3d(self):
        old_dicom = self.change_name(self.old_dicom())
        mask = self.masks_list[:,:,:].astype(np.float16)
        mask = mask.transpose(2, 1, 0)
        old_dicom.PixelData = mask.tostring()
        new_dir_name = old_dicom.PatientName           
        fdwr.create_dir(project_name = f'{self.inference_directory}/{new_dir_name}')
        old_dicom.save_as(f'{self.inference_directory}/{new_dir_name}/{self.dicom_file_name()}')


    def save_dicom(self):
        old_dicom = self.change_name(self.old_dicom())
        old_dicom = self.change_grey_to_color(old_dicom)
        old_dicom = self.change_value_range_info(old_dicom)

        old_dicom.PixelData = self.new_dicom_array().tostring()

        new_dir_name = old_dicom.PatientName
        fdwr.create_dir(project_name = f'{self.inference_directory}/{new_dir_name}')

        old_dicom.save_as(f'{self.inference_directory}/{new_dir_name}/{self.dicom_file_name()}')


class PdfSaver(MetaParameters):
    def __init__(self, file_path, dataset_path, inference_directory):
        super(MetaParameters, self).__init__()

        self.dataset_path = dataset_path
        self.inference_directory = inference_directory
        self.file_name = file_path.split('/')[-1]
        self.images_list = ReadImages(f"{self.dataset_path}{self.file_name}").view_matrix
        self.masks_list = ReadImages(f"{self.inference_directory}/{self.file_name}").view_matrix

        # self.masks_list = ReadImages(f"./Dataset/ALMAZ_mask/{self.file_name}").view_matrix
        # self.fib_masks_list = ReadImages(f"/Users/aglevchuk/Documents/PycharmProjects/Unet_Cardiac/BullEyeMapUnet/Dataset/BULLEYE_Unet3_mask_new/{self.file_name}").view_matrix()
        # self.fib_masks_list = ReadImages(f"/Users/aglevchuk/Documents/PycharmProjects/Unet_Cardiac/BullEyeMapUnet/Dataset/HCM_adult_Unet3_mask_new/{self.file_name}").view_matrix()

        self.images_list = self.images_list.transpose(2, 0, 1)
        self.masks_list = self.masks_list.transpose(2, 0, 1)
        # self.fib_masks_list = self.fib_masks_list.transpose(2, 0, 1)

        self.smooth = 1e-5
        self.rows = 3

    @property
    def create_dict_volume_class(self):
        volume_dict = {}
    
        for key in range(1, self.NUM_CLASS):
            volume_dict[f'Volume_{self.DICT_CLASS[key]}'] = []
            volume_dict[f'Chunk_{self.DICT_CLASS[key]}'] = []

        return volume_dict

    @property
    def get_stats_parameters(self):
        volume_list_dict = {}
        fov = ReadImages(f"{self.dataset_path}{self.file_name}").get_nii_fov()
        volume_size = fov[0] * fov[1] * fov[2]

        for key in range(1, self.NUM_CLASS):
            volume_list_dict[f'Volume_{self.DICT_CLASS[key]}'] = []

        for mask in self.masks_list:
            for key in range(1, self.NUM_CLASS):
                mask_layer = (mask == key)
                volume_list_dict[f'Volume_{self.DICT_CLASS[key]}'].append(round((mask_layer.sum()) * volume_size, 0))

        return volume_list_dict

    @staticmethod
    def divide_chunks(l, n):
        for i in range(0, len(l), n):
            yield l[i:i + n]

    @property
    def save_pdf(self):
        volume_list_dict = self.get_stats_parameters
        volume_dict_class = self.create_dict_volume_class
        
        for key in range(1, self.NUM_CLASS):
            volume_dict_class[f'Volume_{self.DICT_CLASS[key]}'] = volume_list_dict[f'Volume_{self.DICT_CLASS[key]}']

        num_chunk = len(self.images_list) % self.rows
        chunk_list_masks = list(self.divide_chunks(self.masks_list, self.rows))
        chunk_list_images = list(self.divide_chunks(self.images_list, self.rows))
        # chunk_list_fib_masks = list(self.divide_chunks(self.fib_masks_list, self.rows))

        for key in range(1, self.NUM_CLASS): 
            volume_dict_class[f'Chunk_{self.DICT_CLASS[key]}'] = list(self.divide_chunks(volume_dict_class[f'Volume_{self.DICT_CLASS[key]}'], self.rows))

        num_chunk = len(chunk_list_images)
        pp = PdfPages(f'{self.inference_directory}/{self.file_name}_results.pdf')
        
        for chunk in range(num_chunk):
            masks = chunk_list_masks[chunk]
            images = chunk_list_images[chunk]            
            # fib_masks = chunk_list_fib_masks[chunk]

            len_chunk = len(masks)
            
            # masks[masks == 1] = 0
            # fib_masks[fib_masks==1] = 0

            if len_chunk > 1:
                figure, ax = plt.subplots(nrows = len_chunk, ncols = 2, figsize = (12, 12))
                colormap = plt.cm.get_cmap('viridis')  # 'plasma' or 'viridis'
                colormap.set_under('k', alpha = .5)

                bbox = dict(boxstyle = "round", fc = "0.8")
                arrowprops = dict(arrowstyle = "->", connectionstyle = "angle, angleA = 0, angleB = 90,rad = 10")

                for i in range(len_chunk):
                    mask_i  = np.flip(masks[i], (1))
                    image_i = np.flip(images[i], (1))
                    # fib_mask_i  = np.flip(fib_masks[i], (1))
                    
                    mask_i = np.rot90(mask_i, k = 1, axes = (0, 1))
                    image_i = np.rot90(image_i, k = 1, axes = (0, 1))
                    # fib_mask_i = np.rot90(fib_mask_i, k = 1, axes = (0, 1))

                    for clss in range(self.NUM_CLASS):
                        if mask_i[mask_i == clss].sum().item() > 3:
                            mark_mask = mask_i.copy()
                            mark_mask[mark_mask != clss] = 0
                            weight_mass_y, weight_mass_x = ndimage.measurements.center_of_mass(mark_mask)

                            ax[i, 1].annotate(f'S{clss}', 
                                xy = (weight_mass_x, weight_mass_y), 
                                fontsize = 6, xytext = (weight_mass_x + 5, weight_mass_y - 5), 
                                # arrowprops = dict(facecolor = 'red'),
                                arrowprops = arrowprops,
                                bbox = bbox, 
                                color = 'black')

                            ax[i, 1].plot([weight_mass_x], [weight_mass_y],  marker = ".", color = 'orange')

                    for key in range(1, self.NUM_CLASS): 
                        mask_i[0][key - 1] = key
                    # for fkey in range(4):
                    #     fib_mask_i[0][fkey - 1] = fkey

                    ax[i, 0].imshow(image_i, plt.get_cmap('gray'))

                    ax[i, 1].imshow(image_i, plt.get_cmap('gray'))
                    ax[i, 1].imshow(mask_i, alpha = 0.5, interpolation = None, cmap = colormap,  vmin = 0.5)
                    ax[i, 1].contour(mask_i, alpha = 0.5)

                    # ax[i, 2].imshow(image_i, plt.get_cmap('gray'))
                    # ax[i, 2].imshow(fib_mask_i, alpha = 0.7, interpolation = None, cmap = colormap,  vmin = 0.5)

                    report_title = ''
                    
                    ################################################################################
                    try:
                        if self.DICT_CLASS[2] == 'MYO' and self.DICT_CLASS[3] == 'FIB':
                            MYOv = volume_dict_class[f'Chunk_{self.DICT_CLASS[2]}'][chunk]
                            FIBv = volume_dict_class[f'Chunk_{self.DICT_CLASS[3]}'][chunk]
                            relVolume = round((FIBv[i] / (FIBv[i] + MYOv[i] + self.smooth)) * 100, 2)
                            report_title += (f'RelVol of FIB: {relVolume} % ')
                    except:
                        pass
                    ################################################################################

                    for key in range(1, self.NUM_CLASS):
                        if volume_dict_class[f"Chunk_{self.DICT_CLASS[key]}"][chunk][i] / 1000 > 1:
                            report_title += (
                                f'{self.DICT_CLASS[key]}_vol: {volume_dict_class[f"Chunk_{self.DICT_CLASS[key]}"][chunk][i] / 1000} ml, ' 
                                )

                    ax[i, 1].set_title(report_title, fontsize = 8, fontweight = 'bold', loc = 'right')

                    figure.tight_layout()
                pp.savefig(figure)
                
            elif len_chunk == 1:
                figure, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (8, 8))

                colormap = plt.cm.get_cmap('viridis')  # 'plasma' or 'viridis'
                colormap.set_under('k', alpha = .5)

                bbox = dict(boxstyle = "round", fc = "0.8")
                arrowprops = dict(arrowstyle = "->", connectionstyle = "angle,angleA=0,angleB=90,rad=10")

                mask_0  = np.flip(masks[0], (1))
                image_0 = np.flip(images[0], (1))
                # fib_mask_0  = np.flip(fib_masks[0], (1))

                mask_0 = np.rot90(mask_0, k = 1, axes = (0, 1))
                image_0 = np.rot90(image_0, k = 1, axes = (0, 1))
                # fib_mask_0 = np.rot90(fib_mask_0, k = 1, axes = (0, 1))

                for clss in range(self.NUM_CLASS):
                    if mask_0[mask_0 == clss].sum().item() > 3:
                        mark_mask = mask_0.copy()
                        mark_mask[mark_mask != clss] = 0
                        weight_mass_y, weight_mass_x = ndimage.measurements.center_of_mass(mark_mask)

                        ax[1].annotate(f'S{clss}',
                            xy = (weight_mass_x, weight_mass_y),
                            fontsize = 6, xytext = (weight_mass_x + 5, weight_mass_y - 5),
                            # arrowprops = dict(facecolor = 'red'),
                            arrowprops = arrowprops,
                            bbox = bbox,
                            color = 'black')
                        
                        ax[1].plot([weight_mass_x], [weight_mass_y], marker = ".", color = 'orange')

                for key in range(1, self.NUM_CLASS): 
                    mask_0[0][key - 1] = key
                # for fkey in range(4):
                #     fib_mask_0[0][fkey - 1] = fkey

                ax[0].imshow(image_0, plt.get_cmap('gray'))

                ax[1].imshow(image_0, plt.get_cmap('gray'))
                ax[1].imshow(mask_0, alpha = 0.5, interpolation = None, cmap = colormap,  vmin = 0.5)
                ax[1].contour(mask_0, alpha = 0.5)
                
                # ax[2].imshow(image_0, plt.get_cmap('gray'))
                # ax[2].imshow(fib_mask_0, alpha = 0.7, interpolation = None, cmap = colormap,  vmin = 0.5)

                report_title = ''

                ################################################################################
                try:
                    if self.DICT_CLASS[2] == 'MYO' and self.DICT_CLASS[3] == 'FIB':
                        relVolume = round((FIBv[0] / (FIBv[0] + MYOv[0] + self.smooth)) * 100, 2)
                        report_title += (f'RelVol of FIB: {relVolume} % ')
                except:
                    pass
                ################################################################################

                for key in range(1, self.NUM_CLASS):
                    if volume_dict_class[f"Chunk_{self.DICT_CLASS[key]}"][chunk][0] / 1000 > 1:
                        report_title += (
                            f'{self.DICT_CLASS[key]}_vol: {volume_dict_class[f"Chunk_{self.DICT_CLASS[key]}"][chunk][0] / 1000} ml, ' 
                            )

                ax[1].set_title(report_title, fontsize = 8, fontweight = 'bold', loc = 'right')

                figure.tight_layout()
                pp.savefig(figure)

        report_title = ''

        for key in range(1, self.NUM_CLASS):
            report_title += (
                f'Full {self.DICT_CLASS[key]} volume: {sum(volume_dict_class[f"Volume_{self.DICT_CLASS[key]}"]) / 1000} ml, \n' 
                )
        ################################################################################
        try:
            if self.DICT_CLASS[2] == 'MYO' and self.DICT_CLASS[3] == 'FIB':
                related_full_fib_volume = round((
                    (sum(volume_dict_class[f'Volume_{self.DICT_CLASS[3]}'])) / 
                    (sum(volume_dict_class[f'Volume_{self.DICT_CLASS[2]}']) + 
                        sum(volume_dict_class[f'Volume_{self.DICT_CLASS[3]}']) + self.smooth)) * 100, 0)
                report_title += f'Full relative volume: ≈ {related_full_fib_volume} %'
        except:
            pass
        ################################################################################

        fig = plt.figure(figsize = (8, 8))
        text = fig.text(0.2, 0.7, report_title, ha = 'left', va = 'top', size = 14)

        text.set_path_effects([path_effects.Normal()])
        pp.savefig(fig)
        
        pp.close()















