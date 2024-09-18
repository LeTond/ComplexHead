 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.1
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import pydicom as dicom



class DicomSaver(MetaParameters):
    def __init__(self, masks_list, file_name, evaluate_directory):         
        super(MetaParameters, self).__init__()

        self.masks_list = masks_list
        self.file_name = file_name
        self.evaluate_directory = evaluate_directory
        self.orig_dir = f'{self.DATASET_DIR}{self.DATASET_NAME}_origin_new/'
        # self.orig_dir = f''


    def old_dicom(self):
        old_dicom = dicom.dcmread(self.orig_dir + self.file_name)
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

        new_dicom_array = cv2.cvtColor(dcm2, cv2.COLOR_GRAY2RGB)
        new_dicom_array = new_dicom_array / 4095 * 255
        new_dicom_array = new_dicom_array.astype(np.uint8)

        mask = self.masks_list[:,:,0].astype(np.float16)

        # new_dicom_array[:,:,1][mask == 1] -= 50
        new_dicom_array[:,:,2][mask == 2] -= 50
        new_dicom_array[:,:,2][mask == 3] += 50
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
        mask = self.masks_list[:,:,0].astype(np.float16)
        old_dicom.PixelData = mask.tostring()

        new_file_name = f'{self.evaluate_directory}{self.file_name}'.split('/')[-1]
        new_dir_name = f'{self.evaluate_directory}{self.file_name}'.rstrip(new_file_name)
        create_dir(new_dir_name)

        old_dicom.save_as(f'{self.evaluate_directory}{self.file_name}')

    def save_dicom(self):
        old_dicom = self.change_name(self.old_dicom())
        old_dicom = self.change_grey_to_color(old_dicom)
        old_dicom = self.change_value_range_info(old_dicom)
        old_dicom.PixelData = self.new_dicom_array().tostring()

        new_dir_name = old_dicom.PatientName
        create_dir(f'{self.evaluate_directory}/{new_dir_name}')

        print(f'{self.evaluate_directory}/{new_dir_name}/')
        # old_dicom.save_as(f'{self.evaluate_directory}/{new_dir_name}/{self.file_name}')
        old_dicom.save_as(f'{self.evaluate_directory}/{new_dir_name}/{self.dicom_file_name()}')



def old_dicom():
    old_dicom = dicom.dcmread(path_to_dcm)
    return old_dicom



if __name__ == "__main__":
    path_to_dcm = 'path_to_dicom_file.dcm'
    meta_param = old_dicom()

    print(
        f'Size of thickness: {meta_param[0x0018, 0x0088].value} '
        f'Size of pixel: {meta_param[0x0028, 0x0030].value}'
        )

    print(meta_param)


