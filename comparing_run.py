 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.1
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


from Validation.comparing import *
from Preprocessing.split_dataset import *


jsnlst = JsonFoldList()
test_list = jsnlst.load_dataset_list('test_list')
list_ = test_list


for key in range(1, meta.NUM_CLASS):
	summ_dice = []
	summ_dsc_ps = []

	for lst in list_:	
		# path_to_label = f'./Dataset/HCM_adult_mask/{lst}'
		path_to_label = f'./Dataset/ALMAZ_mask/{lst}'
		# path_to_label = f'./Dataset/BULLEYE_mask/{lst}'
		# path_to_prediction = f'./Dataset/BULLEYE_Unet5_mask_new/{lst}'
		path_to_prediction = f'./Dataset/ALMAZ_Unet3_mask_new/{lst}'
		path_to_bull_templs = f'./Dataset/BULLEYE_masks_etalon/{lst}'

		cm = CompareMatrix(path_to_label, path_to_prediction, key, path_to_bull_templs)
		cm.print()

		summ_dice.append(cm.dice())

		for dsc_2d in cm.dice_2d():
			summ_dsc_ps.append(dsc_2d)

	print(f'DSC Class_{meta.DICT_CLASS[key]}: Average per sub = {round(np.mean(summ_dice), 3)}, '
		f'Median per sub = {round(np.median(summ_dice), 3)}')

	print(f'DSC Class_{meta.DICT_CLASS[key]}: Average per slice = {round(np.mean(summ_dsc_ps), 3)}, '
		f'Median per slice = {round(np.median(summ_dsc_ps), 3)}')



