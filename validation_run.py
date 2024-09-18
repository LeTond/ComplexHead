 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.1
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


from configuration import *
from Validation.validation import *
from Training.dataset import *
from Preprocessing.split_dataset import *


class PlotResults(MetaParameters):
    def __init__(self):         
        super(MetaParameters, self).__init__()

        if self.UNET2 is True:
            self.kernel_sz = self.CROPP_KERNEL
        elif self.UNET2 is False:
            self.kernel_sz = self.KERNEL

        self.dict_class_stats = self.create_dict_class()

    def data_loader(self, data_list, kernel_sz, augmentation = False):
        getds_origin, getds_mask, getds_template, getds_names = GetData(data_list, augmentation).generated_data_list
        data_set = MyDataset(getds_origin, getds_mask, getds_template, getds_names, default_transform)
        
        data_batch_size = len(data_set)
        data_loader = DataLoader(data_set, data_batch_size, drop_last = True, shuffle = False, pin_memory = True)

        return data_loader

    def create_dict_class(self):
        dict_class_stats = {}

        for key in range(1, self.NUM_CLASS): 
            dict_class_stats[f'Precision_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'Recall_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'Accuracy_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'Dice_{self.DICT_CLASS[key]}'] = []

            dict_class_stats[f'FN_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'FP_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'GTPix_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'CMPix_{self.DICT_CLASS[key]}'] = []

            dict_class_stats[f'GTVol_{self.DICT_CLASS[key]}'] = []
            dict_class_stats[f'CMVol_{self.DICT_CLASS[key]}'] = []

        return dict_class_stats

    def bland_altman_per_subject(self, model, test_list, meta, kernel_sz):
        for subj in test_list:
            try:
                for key in range(1, self.NUM_CLASS): 
                    data_loader = self.data_loader([subj], kernel_sz, False)
                    tm = TissueMetrics(model, data_loader)
                    dict_class_sub_stats = tm.bland_altman_metrics()
                    
                    self.dict_class_stats[f'GTVol_{self.DICT_CLASS[key]}'].append(np.sum(dict_class_sub_stats[f'GTVol_{self.DICT_CLASS[key]}']))
                    self.dict_class_stats[f'CMVol_{self.DICT_CLASS[key]}'].append(np.sum(dict_class_sub_stats[f'CMVol_{self.DICT_CLASS[key]}']))
            
            except ValueError:
                print(f'Subject {subj} has no suitable images !!!!')

        return self.dict_class_stats

    def stats_per_subject(self, model, test_list, meta, kernel_sz):
        for subj in test_list:
            try:
                data_loader = self.data_loader([subj], kernel_sz, False)
                tm = TissueMetrics(model, data_loader)
                dict_class_sub_stats = tm.image_metrics()

                for key in range(1, self.NUM_CLASS):
                    self.dict_class_stats[f'Precision_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Precision_{self.DICT_CLASS[key]}']
                    self.dict_class_stats[f'Recall_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Recall_{self.DICT_CLASS[key]}']
                    self.dict_class_stats[f'Accuracy_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Accuracy_{self.DICT_CLASS[key]}']
                    self.dict_class_stats[f'Dice_{self.DICT_CLASS[key]}'] += dict_class_sub_stats[f'Dice_{self.DICT_CLASS[key]}']

            except ValueError:
                print(f'Subject {subj} has no suitable images !!!!')

        return self.dict_class_stats

    def prepare_plot(self, sub_names, origImage, origMask, predMask, dice_layers):
        figure, ax = plt.subplots(nrows = 1, ncols = 4, figsize = (12, 12))

        origImage = np.resize(origImage.cpu(), (self.kernel_sz, self.kernel_sz))        
        predMask = np.resize(predMask.cpu(), (self.kernel_sz, self.kernel_sz))
        origMask = np.resize(origMask.cpu(), (self.kernel_sz, self.kernel_sz))

        for key in range(1, self.NUM_CLASS):
            predMask[0][key-1] = key
            origMask[0][key-1] = key

        colormap = plt.cm.get_cmap('viridis')  # 'plasma' or 'viridis'
        colormap.set_under('k', alpha=0.5)

        ax[0].imshow(origImage, plt.get_cmap('gray'))
        ax[1].imshow(origImage, plt.get_cmap('gray'))
        ax[1].imshow(origMask, alpha = 0.5, interpolation=None, cmap=colormap,  vmin=0.5)
        # ax[1].contour(origMask, alpha = 0.5)

        ax[2].imshow(origImage, plt.get_cmap('gray'))
        ax[2].imshow(predMask, alpha = 0.5, interpolation=None, cmap=colormap,  vmin=0.5)
        # ax[2].contour(predMask, alpha = 0.5)
        ax[3].imshow(predMask, alpha = 0.5)

        ax[0].set_title(f"{sub_names}", fontsize = 10, fontweight = 'bold')
        ax[1].set_title(f"Dice: {dice_layers} \nManual mask", fontsize = 10, fontweight ='bold')
        ax[2].set_title(f"Computed mask", fontsize = 10, fontweight='bold')
        ax[3].set_title(f"Computed mask", fontsize = 10, fontweight='bold')
        
        figure.set_edgecolor("green")
        figure.tight_layout()
        
        return figure

    def show_predicted(self, predicted_masks):
        for i in range(predicted_masks[0][0]):
            dice_layers = str('')

            for key in range(1, self.NUM_CLASS):
                if round(predicted_masks[5].get(f'{self.DICT_CLASS[key]}')[i], 3) > 0 and round(predicted_masks[5].get(f'{self.DICT_CLASS[key]}')[i], 3) != 1:
                    dice_layers += f' {self.DICT_CLASS[key]} = '
                    dice_layers += str(round(predicted_masks[5].get(f'{self.DICT_CLASS[key]}')[i], 3))

            # if round(predicted_masks[5].get(f'{self.DICT_CLASS[key]}')[i], 3) == 0.0:
            self.prepare_plot(predicted_masks[1][i], predicted_masks[2][i], predicted_masks[3][i], predicted_masks[4][i], dice_layers)


checkpoint = torch.load(f'{meta.PROJ_NAME}/{meta.DATASET_NAME}_model.pth')

if meta.UNET2 is False and meta.UNET3 is False:
    checkpoint = checkpoint[f'Net_{meta.DATASET_NAME}_{meta.UNET1_FOLD}']
    model = checkpoint[f'Model']
    model.load_state_dict(checkpoint['weights'])
    kernel_sz = meta.KERNEL

elif meta.UNET2 is True and meta.UNET3 is False:
    checkpoint = checkpoint[f'Net_{meta.DATASET_NAME}_{meta.UNET2_FOLD}']
    model = checkpoint[f'Model']
    model.load_state_dict(checkpoint['weights'])
    kernel_sz = meta.CROPP_KERNEL 

elif meta.UNET3 is True and meta.UNET4 is False:
    checkpoint = checkpoint[f'Net_{meta.DATASET_NAME}_{meta.UNET3_FOLD}']
    model = checkpoint[f'Model']
    model.load_state_dict(checkpoint['weights'])
    kernel_sz = meta.CROPP_KERNEL 

elif meta.UNET4 is True and meta.UNET5 is False:
    checkpoint = checkpoint[f'Net_{meta.DATASET_NAME}_{meta.UNET4_FOLD}']
    model = checkpoint[f'Model']
    model.load_state_dict(checkpoint['weights'])
    kernel_sz = meta.CROPP_KERNEL 

elif meta.UNET5 is True:
    checkpoint = checkpoint[f'Net_{meta.DATASET_NAME}_{meta.UNET5_FOLD}']
    model = checkpoint[f'Model']
    model.load_state_dict(checkpoint['weights'])
    kernel_sz = meta.CROPP_KERNEL 


pltres = PlotResults()
jsnlst = JsonFoldList()
test_list = jsnlst.load_dataset_list('test_list')
test_loader = pltres.data_loader(test_list, kernel_sz, False)

print(f'Test size: {len(test_list)}')


ds = DiceLoss()
show_predicted_masks = MaskPrediction().prediction_masks(model, test_loader)
tm = TissueMetrics(model, test_loader)

# try:
#     bland_dict_class_stats = pltres.bland_altman_per_subject(model, test_list, meta, kernel_sz)
#     dict_class_stats = pltres.stats_per_subject(model, test_list, meta, kernel_sz)

# except ValueError:
#     print(f'Subjects has no suitable images !!!!')


