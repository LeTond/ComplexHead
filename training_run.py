 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.2
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


import torch

from configuration import *
from Preprocessing.split_dataset import *
from Preprocessing.dirs_logs import FileDirectoryWorker
from Model.unet2D import UNet_2D, UNet_2D_AttantionLayer, UNetResnet, SegNet
# from Model.unet3D import UNet_3D, UNet_3D_AttantionLayer
# from Model.FCT.utils.model import FCT
# from Model.resnet import ResNet, BasicBlock
# from Model.models import bounding_box_CNN
from Training.train import *
from Training.dataset import *
from Training.ranger import Ranger
from Training.optimizer import Lion



class ChooseModelConfig(MetaParameters):
    def __init__(self):    
        super(MetaParameters, self).__init__()
        self.__model = self.choose_train_model
        
        if self.PRETRAIN:
            self.print_model_key

    @property
    def model(self):
        return self.__model

    @property  
    def choose_model_key(self):
        if self.UNET2 is True and self.UNET3 is False:
            return self.UNET2_FOLD
        elif self.UNET1 is True and self.UNET2 is False:
            return self.UNET1_FOLD

    @property
    def model_key(self):
        return self.choose_model_key

    @property
    def print_model_key(self):
        print('\n' + f'Model KEY {self.model_key} Was Chosen' + '\n')

    @property
    def choose_train_model(self):
        if self.PRETRAIN:    
            try:
                checkpoint = torch.load(f'{self.PROJ_NAME}/{self.DATASET_NAME}_model.pth')
                checkpoint = checkpoint[f'Net_{self.DATASET_NAME}_{self.model_key}']

                model = checkpoint['Model']
                model.load_state_dict(checkpoint['weights'])  
                model.eval()      
                print(f'Model Loaded: {self.DATASET_NAME}/{self.MODEL_NAME}.pth !!!')

            except:
                print('\n' + 'No Trained Models !!!' + '\n')
                model = UNet_2D_AttantionLayer().to(device = device)
        else:
            model = UNet_2D_AttantionLayer().to(device = device)
            # model = UNet_2D().to(device=device)
            # model = UNetResnet().to(device=device)
            # model = SegNet().to(device=device)

        return model

    @property
    def freeze_model_bn(self):
        if self.FREEZE_BN is True:
            for name, child in self.model.named_children(): 
                if name in ['encoder4', 'bottleneck', 'decoder4', 'decoder3','decoder2', 'decoder1', 'upconv4', 'upconv3', 'upconv2', 'upconv1', 'conv', 'Att4', 'Att3', 'Att2', 'Att1']:
                    print(name + ' Has Been Unfrozen !!!') 
                    for param in child.parameters(): 
                        param.requires_grad = True 
                else: 
                    for param in child.parameters(): 
                        param.requires_grad = False

        # optimizer = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr = meta.LR, weight_decay = meta.WDC)
        # optimizer = torch.optim.Adam(model.parameters(), lr = meta.LR, weight_decay = meta.WDC)
        # optimizer = torch.optim.AdamW(model.parameters(), lr = meta.LR, weight_decay = meta.WDC)
        # optimizer = Lion(model.parameters(), lr = meta.LR, betas = (0.9, 0.99), weight_decay = meta.WDC)
        # optimizer = torch.optim.AdamW(model.parameters(), lr = learning_rate, weight_decay = wdc, amsgrad = False)
        # optimizer = torch.optim.SGD(model.parameters(), lr = meta.LR, weight_decay = meta.WDC, momentum = 0.9, nesterov = True)        
        optimizer = Ranger(self.model.parameters(), lr = self.LR, k = 6, N_sma_threshhold = 5, weight_decay = self.WDC)
        # optimizer = Ranger(filter(lambda x: x.requires_grad, self.model.parameters()),  lr = self.LR, k = 6, N_sma_threshhold = 5, weight_decay = self.WDC)

        return optimizer

    @property
    def optimizer(self):
        return self.freeze_model_bn

    @property
    def choose_scheduler_gen(self):
        scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max = self.TMAX, eta_min = 0, last_epoch = -1, verbose = True)

        # scheduler_gen = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode = 'min', factor = 0.8, patience = 5, threshold = 0.0001, threshold_mode = 'rel', 
        #     cooldown = 0, min_lr = 0, eps = 1e-08, verbose = 'deprecated')

        return scheduler_gen

    @property
    def scheduler_gen(self):
        return self.choose_scheduler_gen


if __name__ == '__main__':
    cmc = ChooseModelConfig()

    model = cmc.model
    optimizer = cmc.optimizer
    scheduler_gen = cmc.scheduler_gen

    fdwr.create_dir_log(project_name = meta.PROJ_NAME)

    ########################################################################################################################
    # Creating loaders for training and validating network
    ########################################################################################################################
    jsnlst = JsonFoldList()
    jsnlst.create_folds_list

    train_list = jsnlst.load_dataset_list('train_list')
    valid_list = jsnlst.load_dataset_list('valid_list')

    jsnlst.pprint('train_list')
    jsnlst.pprint('valid_list')

    train_ds_origin, train_ds_mask, train_ds_template, train_ds_names = GetData(train_list, meta.AUGMENTATION).generated_data_list
    valid_ds_origin, valid_ds_mask, valid_ds_template, valid_ds_names = GetData(valid_list, False).generated_data_list

    train_set = MyDataset(train_ds_origin, train_ds_mask, train_ds_template, train_ds_names, default_transform)
    for i in range(2):
        train_set += MyDataset(train_ds_origin, train_ds_mask, train_ds_template, train_ds_names, transform_04)
        train_set += MyDataset(train_ds_origin, train_ds_mask, train_ds_template, train_ds_names, transform_01)
        # train_set += MyDataset(train_ds_origin, train_ds_mask, train_ds_template, train_ds_names, transform_05)
        # train_set += MyDataset(train_ds_origin, train_ds_mask, train_ds_template, train_ds_names, transform_06)

    train_loader = DataLoader(train_set, meta.BT_SZ, drop_last = True, shuffle = True, pin_memory = False)

    valid_set = MyDataset(valid_ds_origin, valid_ds_mask, valid_ds_template, valid_ds_names, default_transform)
    valid_batch_size = len(valid_set)
    valid_loader = DataLoader(valid_set, meta.BT_SZ, drop_last = True, shuffle = True, pin_memory = False)


    print(f'Train size: {len(train_set)} | Valid size: {len(valid_set)}')
    model = TrainNetwork(model, optimizer, loss_function, scheduler_gen, train_loader, valid_loader, meta, ds).train()


    # summary(model,input_size=(1,512, 512))




