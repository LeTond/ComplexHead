 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.1
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


from Preprocessing.preprocessing import ReadImages
from Postprocessing.postprocessing import *
from Preprocessing.dirs_logs import *
from Training.dataset import *

from configuration import *


path_to_origs = meta.ORIGS_DIR
path_to_masks = meta.MASKS_DIR


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(pred, target):
        smooth = 1e-5
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()
        A_sum = torch.sum(iflat * iflat)
        B_sum = torch.sum(tflat * tflat)
        return (2.0 * intersection + smooth) / (A_sum + B_sum + smooth)


def create_hist(value_list: list):
    img_np = np.array(value_list)
    plt.hist(img_np.ravel(), bins=20, density=False)
    plt.xlabel("DSC")
    plt.ylabel("Images")
    plt.title("Distribution of dice")


class MaskPrediction(MetaParameters):
    def __init__(self):
        super(MetaParameters, self).__init__()
        self.ds = DiceLoss()

    def prediction_masks(self, model, dataset_):
        model.eval()
        dice_layers = {}
        smooth = 1e-6

        for key in range(1, self.NUM_CLASS):
            dice_layers[f'{self.DICT_CLASS[key]}'] = list([])

        with torch.no_grad():
            for inputs, labels, sub_names in dataset_:
                inputs, labels, sub_names = inputs.to(device), labels.to(device), list(sub_names)   

                predict = torch.softmax(model(inputs), dim = 1)
                predict = torch.argmax(predict, dim = 1)
                labels = torch.argmax(labels, dim = 1)

                for slc in range(labels.shape[0]):
                    pred = predict[slc]
                    labl = labels[slc]
                    
                    if self.UNET4 is True and self.UNET5 is False:
                        try: 
                            unique, counts = np.unique(pred.cpu(), return_counts=True)
                            test_dict = dict(zip(unique, counts))
                            myo_level = int(list(test_dict.keys())[0])

                            if myo_level != 0:
                                pred[pred!=0] = myo_level
                            else:
                                myo_level = int(list(test_dict.keys())[1])
                                pred[pred!=0] = myo_level
                        except:
                            pass

                    # pred = np.array(pred.cpu(), dtype=np.float32)
                    # threshold_matrix = InstancesFinder(pred, kernel = 64).threshold_matrix()
                    # pred = TF.to_pil_image(threshold_matrix)
                    # pred = TF.pil_to_tensor(pred)

                    # rel_volume = ((pred==3).sum().item() + smooth) / ((pred==2).sum().item() + (pred==3).sum().item() + smooth) * 100
                    
                    # if rel_volume < 3 and 10 > (pred==3).sum().item() > 0:
                    #     predict[slc][predict[slc]==3] = 2

                    for key in range(1, self.NUM_CLASS):
                        predict_ = (pred == key)
                        labels_ = (labl == key)
                        dice_metric = float(self.ds(predict_, labels_))
                        # if dice_metric>0 and dice_metric!=0:
                        dice_layers[f'{self.DICT_CLASS[key]}'].append(dice_metric)

            shp = predict.shape

        return shp, sub_names, inputs, labels, predict, dice_layers


class TissueMetrics(MaskPrediction):

    def __init__(self, Net, dataset=None):         
        super(MaskPrediction, self).__init__()
        model = Net
        dataset_ = dataset
        mp = MaskPrediction().prediction_masks(model, dataset_)

        if meta.UNET2 is True:
            self.kernel_sz = self.CROPP_KERNEL
        elif meta.UNET2 is False:
            self.kernel_sz = self.KERNEL

        self.dict_class_stats = self.create_dict_class()

        self.shp = mp[0]
        self.sub_names = mp[1]
        self.inputs = mp[2]
        self.labels = mp[3]
        self.predict = mp[4]

        self.smooth = 1e-5

    @staticmethod
    def get_orig_slice(sub_name):
        sub_name_list = sub_name.split(' ')
        
        return sub_name_list

    def get_image_contrast(self, num_label):
        masks_matrix = np.copy(self.masks_matrix)
        masks_matrix[masks_matrix != num_label] = 0
        masks_matrix[masks_matrix == num_label] = 1
        orig_matrix = np.copy(self.images_matrix) * masks_matrix
        summ_mask_matrix = masks_matrix.sum()
        summ_orig_matrix = orig_matrix.sum()
        mean_contrast = round(float((summ_orig_matrix + self.smooth) / (summ_mask_matrix + self.smooth)), 2)   #7

        return mean_contrast

    def get_image_metrics(self, label, prediction, metric_name: str):
        GT = label.sum()
        CM = prediction.sum()
        TP = (label * prediction).sum()
        FN = np.abs(GT - TP)
        FP = np.abs(CM - TP)
        TN = np.abs(self.kernel_sz * self.kernel_sz - GT - FP)
        
        precision = round(float((TP + self.smooth) / (TP + FP + self.smooth)), 2)
        recall = round(float((TP + self.smooth) / (TP + FN + self.smooth)), 2)    
        accuracy = round(float((TP + TN + self.smooth) / (TP + TN + FP + FN + self.smooth)), 2)
        dice = round(float((2 * TP + self.smooth) / (2 * TP + FP + FN + self.smooth)), 2)

        return precision, recall, accuracy, dice 

    def get_fnfp_metrics(self, label, prediction, metric_name: str):
        GT = label.sum()
        CM = prediction.sum()
        TP = (label * prediction).sum()
        FN = np.abs(GT - TP)
        FP = np.abs(CM - TP)
        TN = np.abs(self.kernel_sz * self.kernel_sz - GT - FP)
        
        return FN, FP 

    def main_stat_parameters(self, value_list: list):
        median_value = round(float(np.median(value_list)), 2)
        mean_value = round(float(np.mean(value_list)), 2)
        std_value = round(float(np.std(value_list)), 2)
        
        return median_value, mean_value, std_value

    def image_contrast(self):
        for i in range(self.shp[0]):
            orig_slc = int(self.get_orig_slice(self.sub_names[i])[2]) 
            orig_sub = str(self.get_orig_slice(self.sub_names[i])[0])
            
            self.images_matrix = ReadImages(f"{self.ORIGS_DIR}/{orig_sub}.nii").view_matrix()[:,:,-orig_slc] 
            self.masks_matrix = ReadImages(f"{self.MASKS_DIR}/{orig_sub}.nii").view_matrix()[:,:,-orig_slc] 

            mean_contrast_lv = self.get_image_contrast(num_label = 1)
            mean_contrast_myo = self.get_image_contrast(num_label = 2)
            mean_contrast_fib = self.get_image_contrast(num_label = 3)

            print(f'{self.sub_names[i]} - mean LV {mean_contrast_lv}, mean Myo {mean_contrast_myo}, mean Fib {mean_contrast_fib}')

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
            
            # dict_class_stats[f'Dice_{self.DICT_CLASS[key]}'] += float(self.ds(predict_, labels_))

        return dict_class_stats

    def image_metrics(self):
        for i in range(self.shp[0]):
            for key in range(1, self.NUM_CLASS):
                self.labe_class = (self.labels == key).cpu()
                self.pred_class = (self.predict == key).cpu()
                
                class_metrics = self.get_image_metrics(self.labe_class[i], self.pred_class[i], f'{self.DICT_CLASS[key]}')

                self.dict_class_stats[f'Precision_{self.DICT_CLASS[key]}'].append(class_metrics[0])
                self.dict_class_stats[f'Recall_{self.DICT_CLASS[key]}'].append(class_metrics[1])
                self.dict_class_stats[f'Accuracy_{self.DICT_CLASS[key]}'].append(class_metrics[2])
                self.dict_class_stats[f'Dice_{self.DICT_CLASS[key]}'].append(class_metrics[3])

                fnfp_metrics = self.get_fnfp_metrics(self.labe_class[i], self.pred_class[i], f'{self.DICT_CLASS[key]}')
                self.dict_class_stats[f'FN_{self.DICT_CLASS[key]}'].append(fnfp_metrics[0])
                self.dict_class_stats[f'FP_{self.DICT_CLASS[key]}'].append(fnfp_metrics[1])

        # for key in range(3, self.NUM_CLASS):
        #     mean_precision = round(np.sum(dict_class_stats[f'Precision_{self.DICT_CLASS[key]}']) / self.shp[0], 3) 
        #     mean_recall = round(np.sum(dict_class_stats[f'Recall_{self.DICT_CLASS[key]}']) / self.shp[0], 3)
        #     mean_accur = round(np.sum(dict_class_stats[f'Accuracy_{self.DICT_CLASS[key]}']) / self.shp[0], 3)
        #     mean_dice = round(np.sum(dict_class_stats[f'Dice_{self.DICT_CLASS[key]}']) / self.shp[0], 3)

        #     sum_fn = np.sum(dict_class_stats[f'FN_{self.DICT_CLASS[key]}'])
        #     sum_fp = np.sum(dict_class_stats[f'FP_{self.DICT_CLASS[key]}'])

        #     print(f'{self.sub_names[0]} '
        #         f'Class_{self.DICT_CLASS[key]}, '
        #         f'Precision: {mean_precision}, '
        #         f'Recall: {mean_recall}, '
        #         f'Accuracy: {mean_accur}, '
        #         f'Dice: {mean_dice}, '
        #         f'FN: {sum_fn}, '
        #         f'FP: {sum_fp} '
        #         )

        return self.dict_class_stats

    @staticmethod
    def get_volume(label):
        lab_volume = label.sum().item()

        return lab_volume

    def get_recent_volume(self):
        Vgt = round(float((GT_fib + self.smooth) / (GT_myo + GT_fib + self.smooth) * 100), 2)
        Vcm = round(float((CM_fib + self.smooth) / (CM_myo + CM_fib + self.smooth) * 100), 2)
        
        return Vgt, Vcm

    def bland_altman_metrics(self):
        for i in range(self.shp[0]):
            for key in range(1, self.NUM_CLASS):
                self.labe_class = (self.labels == key).cpu()
                self.pred_class = (self.predict == key).cpu()

                self.dict_class_stats[f'GTVol_{self.DICT_CLASS[key]}'].append(self.get_volume(self.labe_class[i]) * 32 / 1000)
                self.dict_class_stats[f'CMVol_{self.DICT_CLASS[key]}'].append(self.get_volume(self.pred_class[i]) * 32 / 1000)

                self.dict_class_stats[f'GTPix_{self.DICT_CLASS[key]}'].append(self.labe_class[i].numpy().sum())
                self.dict_class_stats[f'CMPix_{self.DICT_CLASS[key]}'].append(self.pred_class[i].numpy().sum())

        return self.dict_class_stats

    def relative_volume(self):        
        Vgt, Vcm = [], []

        labe_class_mlb = (self.labels == 2).cpu()
        pred_class_mpd = (self.predict == 2).cpu()
        labe_class_flb = (self.labels == 3).cpu()
        pred_class_fpd = (self.predict == 3).cpu()
        
        for i in range(self.shp[0]):
            # for key in range(1, self.NUM_CLASS):

            GT_myo = self.get_volume(labe_class_mlb[i])
            CM_myo = self.get_volume(pred_class_mpd[i])
            GT_fib = self.get_volume(labe_class_flb[i])
            CM_fib = self.get_volume(pred_class_fpd[i])

            Vgt.append(round(float((GT_fib + self.smooth) / (GT_myo + GT_fib + self.smooth) * 100), 2))
            Vcm.append(round(float((CM_fib + self.smooth) / (CM_myo + CM_fib + self.smooth) * 100), 2))
        
        return Vgt, Vcm




