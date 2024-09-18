 # -*- coding: utf-8 -*-
"""
Name: Anatoliy Levchuk
Version: 1.1
Date: 03-09-2024
Email: feuerlag999@yandex.ru
GitHub: https://github.com/LeTond
"""


from configuration import MetaParameters
from torch import nn
from collections import OrderedDict
from Model import resnet
from torchvision import models

import torch.nn.functional as F

import torch


class UNet_2D(nn.Module, MetaParameters):

    def __init__(self):
        super(UNet_2D, self).__init__()
        super(MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        out_channels = self.NUM_CLASS
        dropout = self.DROPOUT

        self.dropout = nn.Dropout2d(dropout)
        self.encoder1 = UNet_2D.Conv2x2(in_channels, features, name = "enc1")
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder2 = UNet_2D.Conv2x2(features, features * 2, name = "enc2")
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder3 = UNet_2D.Conv2x2(features * 2, features * 4, name = "enc3")
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder4 = UNet_2D.Conv2x2(features * 4, features * 8, name = "enc4")
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.bottleneck = UNet_2D.Conv2x2(features * 8, features * 16, name = "bottleneck")
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size = 2, stride = 2)
        self.decoder4 = UNet_2D.Conv2x2((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size = 2, stride = 2)
        self.decoder3 = UNet_2D.Conv2x2((features * 4) * 2, features * 4, name = "dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size = 2, stride = 2)
        self.decoder2 = UNet_2D.Conv2x2((features * 2) * 2, features * 2, name = "dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size = 2, stride = 2)
        self.decoder1 = UNet_2D.Conv2x2(features * 2, features, name = "dec1")
        self.conv = nn.Conv2d(in_channels = features, out_channels = out_channels, kernel_size = 1)

    def forward(self, x):

        enc1 = self.encoder1(x)
        enc1 = self.dropout(enc1)

        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.dropout(enc2)

        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.dropout(enc3)

        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = self.dropout(enc4)

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim = 1)
        dec4 = self.dropout(dec4)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim = 1)
        dec3 = self.dropout(dec3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim = 1)
        dec2 = self.dropout(dec2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim = 1)
        dec1 = self.dropout(dec1)
        dec1 = self.decoder1(dec1)

        # return torch.softmax(self.conv(dec1), dim=1)
        # return torch.sigmoid(self.conv(dec1))
        return self.conv(dec1)

    @staticmethod
    def Conv2x2(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = features,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features = features, affine = True)),   #, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True
                    # (name + "relu1", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    (name + "relu1", nn.ReLU()),

                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels = features,
                            out_channels = features,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features = features, affine = True)),
                    # (name + "relu2", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    (name + "relu2", nn.ReLU()),

                ]
            )
        )


class UNet_2D_AttantionLayer(nn.Module, MetaParameters):

    def __init__(self):
        super(UNet_2D_AttantionLayer, self).__init__()
        super(MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        out_channels = self.NUM_CLASS
        dropout = self.DROPOUT
        freeze_bn = self.FREEZE_BN

        self.dropout = nn.Dropout2d(dropout)
        self.encoder1 = UNet_2D_AttantionLayer.Conv2x2(in_channels, features, name = "enc1")
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # self.pool1 = nn.AvgPool2d(kernel_size = 2, stride = 2)

        self.encoder2 = UNet_2D_AttantionLayer.Conv2x2(features, features * 2, name = "enc2")
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # self.pool2 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.encoder3 = UNet_2D_AttantionLayer.Conv2x2(features * 2, features * 4, name = "enc3")
        self.pool3 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # self.pool3 = nn.AvgPool2d(kernel_size = 2, stride = 2)
        
        self.encoder4 = UNet_2D_AttantionLayer.Conv2x2(features * 4, features * 8, name = "enc4")
        self.pool4 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        # self.pool4 = nn.AvgPool2d(kernel_size = 2, stride = 2)

        self.bottleneck = UNet_2D_AttantionLayer.Conv2x2(features * 8, features * 16, name = "bottleneck")
        
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size = 2, stride = 2)
        self.Att4 = Attention_2D(features * 8,features * 8,features * 4)
        self.decoder4 = UNet_2D_AttantionLayer.Conv2x2((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size = 2, stride = 2)
        self.Att3 = Attention_2D(features * 4,features * 4,features * 2)
        self.decoder3 = UNet_2D_AttantionLayer.Conv2x2((features * 4) * 2, features * 4, name = "dec3")
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size = 2, stride = 2)
        self.Att2 = Attention_2D(features * 2,features * 2,features * 1)
        self.decoder2 = UNet_2D_AttantionLayer.Conv2x2((features * 2) * 2, features * 2, name = "dec2")
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size = 2, stride = 2)
        self.Att1 = Attention_2D(features, features, features // 2)
        self.decoder1 = UNet_2D_AttantionLayer.Conv2x2(features * 2, features, name = "dec1")
        
        self.conv = nn.Conv2d(in_channels = features, out_channels = out_channels, kernel_size = 1)

    def forward(self, x):

        enc1 = self.encoder1(x)
        enc1 = self.dropout(enc1)

        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.dropout(enc2)

        enc3 = self.encoder3(self.pool2(enc2))
        enc3 = self.dropout(enc3)

        enc4 = self.encoder4(self.pool3(enc3))
        enc4 = self.dropout(enc4)

        bottleneck = self.bottleneck(self.pool4(enc4))
        # bottleneck = self.dropout(bottleneck)

        dec4 = self.upconv4(bottleneck)
        enc4 = self.Att4(dec4,enc4)
        dec4 = torch.cat((dec4, enc4), dim = 1)
        dec4 = self.dropout(dec4)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        enc3 = self.Att3(dec3,enc3)
        dec3 = torch.cat((dec3, enc3), dim = 1)
        dec3 = self.dropout(dec3)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        enc2 = self.Att2(dec2,enc2)
        dec2 = torch.cat((dec2, enc2), dim = 1)
        dec2 = self.dropout(dec2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        enc1 = self.Att1(dec1,enc1)
        dec1 = torch.cat((dec1, enc1), dim = 1)
        dec1 = self.dropout(dec1)
        dec1 = self.decoder1(dec1)

        # return torch.softmax(self.conv(dec1), dim=1)
        # return torch.sigmoid(self.conv(dec1))
        return self.conv(dec1)

    @staticmethod
    def Conv2x2(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = features,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features = features)),   #, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True
                    # (name + "norm1", nn.InstanceNorm2d(features, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    # (name + "relu1", nn.ReLU()),

                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels = features,
                            out_channels = features,
                            kernel_size = 3,
                            stride = 1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features = features)),
                    # (name + "norm1", nn.InstanceNorm2d(features, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = False)),
                    (name + "relu2", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    # (name + "relu1", nn.ReLU()),

                ]
            )
        )


class Attention_2D(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_2D,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size = 1, stride = 1, padding = 0, bias = False),
            nn.BatchNorm2d(1),
            nn.Softmax(dim = 1)
        )
        
        # self.relu = nn.ReLU(inplace = True)
        self.relu = nn.LeakyReLU(negative_slope = 0.1, inplace = True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels = None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace = True),
            nn.Conv2d(mid_channels, out_channels, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet_2D_mini(nn.Module, MetaParameters):

    def __init__(self):
        super(UNet_2D_mini, self).__init__()
        super(MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        out_channels = self.NUM_CLASS
        dropout = self.DROPOUT

        self.dropout = nn.Dropout2d(dropout)
        self.encoder1 = UNet_2D_mini.Conv2x2(in_channels, features, name = "enc1")
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.encoder2 = UNet_2D_mini.Conv2x2(features, features * 2, name = "enc2")
        self.pool2 = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.bottleneck = UNet_2D_mini.Conv2x2(features * 2, features * 4, name = "bottleneck")

        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size = 2, stride = 2)
        self.decoder2 = UNet_2D_mini.Conv2x2((features * 2) * 2, features * 2, name = "dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size = 2, stride = 2)
        self.decoder1 = UNet_2D_mini.Conv2x2(features * 2, features, name = "dec1")
        self.conv = nn.Conv2d(in_channels = features, out_channels = out_channels, kernel_size = 1)

    def forward(self, x):

        enc1 = self.encoder1(x)
        enc1 = self.dropout(enc1)

        enc2 = self.encoder2(self.pool1(enc1))
        enc2 = self.dropout(enc2)

        bottleneck = self.bottleneck(self.pool2(enc2))

        dec2 = self.upconv2(bottleneck)
        dec2 = torch.cat((dec2, enc2), dim = 1)
        dec2 = self.dropout(dec2)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim = 1)
        dec1 = self.dropout(dec1)
        dec1 = self.decoder1(dec1)

        return torch.softmax(self.conv(dec1), dim=1)

    @staticmethod
    def Conv2x2(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels = in_channels,
                            out_channels = features,
                            kernel_size = 3,
                            # stride=1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features = features, affine=False)),   #, eps=1e-05, momentum=0.5, affine=True, track_running_stats=True
                    # (name + "norm1", nn.InstanceNorm2d(32, eps = 1e-5, momentum = 0.1, affine = True, num_features = features)),
                    # (name + "relu1", nn.LeakyReLU(negative_slope = 0.01, inplace = True)),
                    (name + "relu1", nn.ReLU()),

                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels = features,
                            out_channels = features,
                            kernel_size = 3,
                            # stride=1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features = features, affine=False)),
                    # (name + "relu2", nn.LeakyReLU(negative_slope = 0.01, inplace = True)),
                    (name + "relu2", nn.ReLU()),

                ]
            )
        )


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear = True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class U_Net(nn.Module):

    def __init__(self, img_ch=1, num_classes=4):
        super(U_Net, self).__init__()
 
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
 
        self.Conv1 = convConv2x2(ch_in=img_ch, ch_out=64)
        self.Conv2 = convConv2x2(ch_in=64, ch_out=128)
        self.Conv3 = convConv2x2(ch_in=128, ch_out=256)
        self.Conv4 = convConv2x2(ch_in=256, ch_out=512)
        self.Conv5 = convConv2x2(ch_in=512, ch_out=1024)
 
        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_conv5 = convConv2x2(ch_in=1024, ch_out=512)
 
        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_conv4 = convConv2x2(ch_in=512, ch_out=256)
 
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv3 = convConv2x2(ch_in=256, ch_out=128)
 
        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv2 = convConv2x2(ch_in=128, ch_out=64)
 
        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)
        initialize_weights(self)
 
    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)
 
        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
 
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)
 
        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)
 
        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)
 
        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
 
        d5 = self.Up_conv5(d5)
 
        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)
 
        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)
 
        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
 
        d1 = self.Conv_1x1(d2)
 
        return d1


class CNN(nn.Module):
    
    # Constructor
    def __init__(self, out_1=1, out_2=4, out_mp2=7*7):
        super(CNN, self).__init__()
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=out_1, 
                              kernel_size=5, padding=2)
        self.maxpool1=nn.MaxPool2d(kernel_size=2)

        self.cnn2 = nn.Conv2d(in_channels=out_1, out_channels=out_2,
                              kernel_size=5, stride=1, padding=2)
        self.maxpool2=nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(out_2 * out_mp2, 10)
    
    # Prediction
    def forward(self, x):
        x = self.cnn1(x)
        x = torch.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = torch.relu(x)
        x = self.maxpool2(x)
        # Flatten the matrices
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()


class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()
        # self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self):
        raise NotImplementedError

    def summary(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info(f'Nbr of trainable parameters: {nbr_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        nbr_params = sum([np.prod(p.size()) for p in model_parameters])
        return super(BaseModel, self).__str__() + f'\nNbr of trainable parameters: {nbr_params}'
        #return summary(self, input_shape=(2, 3, 224, 224))


class UNetResnet(BaseModel, MetaParameters):
    def __init__(self, backbone='resnet50', pretrained=False, freeze_bn=False, freeze_backbone=False, **_):
        super(UNetResnet, self).__init__()
        super(MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        num_classes = self.NUM_CLASS
        dropout = self.DROPOUT
        freeze_bn = self.FREEZE_BN

        model = getattr(resnet, backbone)(pretrained, norm_layer=nn.BatchNorm2d)

        self.initial = list(model.children())[:4]
        if in_channels != 3:
            self.initial[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.initial = nn.Sequential(*self.initial)

        # encoder
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        # decoder
        self.conv1 = nn.Conv2d(2048, 192, kernel_size=3, stride=1, padding=1)
        self.upconv1 =  nn.ConvTranspose2d(192, 128, 4, 2, 1, bias=False)

        self.conv2 = nn.Conv2d(1152, 128, kernel_size=3, stride=1, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 96, 4, 2, 1, bias=False)

        self.conv3 = nn.Conv2d(608, 96, kernel_size=3, stride=1, padding=1)
        self.upconv3 = nn.ConvTranspose2d(96, 64, 4, 2, 1, bias=False)

        self.conv4 = nn.Conv2d(320, 64, kernel_size=3, stride=1, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 48, 4, 2, 1, bias=False)
        
        self.conv5 = nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1)
        self.upconv5 = nn.ConvTranspose2d(48, 32, 4, 2, 1, bias=False)

        self.conv6 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(32, num_classes, kernel_size=1, bias=False)

        initialize_weights(self)

        if freeze_bn:
            self.freeze_bn()
        if freeze_backbone: 
            set_trainable([self.initial, self.layer1, self.layer2, self.layer3, self.layer4], False)

    def forward(self, x):
        H, W = x.size(2), x.size(3)
        x1 = self.layer1(self.initial(x))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        
        x = self.upconv1(self.conv1(x4))
        x = F.interpolate(x, size=(x3.size(2), x3.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x3], dim=1)
        x = self.upconv2(self.conv2(x))

        x = F.interpolate(x, size=(x2.size(2), x2.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x2], dim=1)
        x = self.upconv3(self.conv3(x))

        x = F.interpolate(x, size=(x1.size(2), x1.size(3)), mode="bilinear", align_corners=True)
        x = torch.cat([x, x1], dim=1)

        x = self.upconv4(self.conv4(x))

        x = self.upconv5(self.conv5(x))

        # if the input is not divisible by the output stride
        if x.size(2) != H or x.size(3) != W:
            x = F.interpolate(x, size=(H, W), mode="bilinear", align_corners=True)

        x = self.conv7(self.conv6(x))
        return x

    def get_backbone_params(self):
        return chain(self.initial.parameters(), self.layer1.parameters(), self.layer2.parameters(), 
                    self.layer3.parameters(), self.layer4.parameters())

    def get_decoder_params(self):
        return chain(self.conv1.parameters(), self.upconv1.parameters(), self.conv2.parameters(), self.upconv2.parameters(),
                    self.conv3.parameters(), self.upconv3.parameters(), self.conv4.parameters(), self.upconv4.parameters(),
                    self.conv5.parameters(), self.upconv5.parameters(), self.conv6.parameters(), self.conv7.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


class SegNet(BaseModel, MetaParameters):
    def __init__(self, pretrained=False, freeze_bn=False, **_):
        super(SegNet, self).__init__()
        super(MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        num_classes = self.NUM_CLASS
        dropout = self.DROPOUT
        freeze_bn = self.FREEZE_BN

        vgg_bn = models.vgg16_bn(pretrained= pretrained)
        encoder = list(vgg_bn.features.children())

        # Adjust the input size
        if in_channels != 3:
            encoder[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)

        # Encoder, VGG without any maxpooling
        self.stage1_encoder = nn.Sequential(*encoder[:6])
        self.stage2_encoder = nn.Sequential(*encoder[7:13])
        self.stage3_encoder = nn.Sequential(*encoder[14:23])
        self.stage4_encoder = nn.Sequential(*encoder[24:33])
        self.stage5_encoder = nn.Sequential(*encoder[34:-1])
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)

        # Decoder, same as the encoder but reversed, maxpool will not be used
        decoder = encoder
        decoder = [i for i in list(reversed(decoder)) if not isinstance(i, nn.MaxPool2d)]
        # Replace the last conv layer
        decoder[-1] = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        # When reversing, we also reversed conv->batchN->relu, correct it
        decoder = [item for i in range(0, len(decoder), 3) for item in decoder[i:i+3][::-1]]
        # Replace some conv layers & batchN after them
        for i, module in enumerate(decoder):
            if isinstance(module, nn.Conv2d):
                if module.in_channels != module.out_channels:
                    decoder[i+1] = nn.BatchNorm2d(module.in_channels)
                    decoder[i] = nn.Conv2d(module.out_channels, module.in_channels, kernel_size=3, stride=1, padding=1)

        self.stage1_decoder = nn.Sequential(*decoder[0:9])
        self.stage2_decoder = nn.Sequential(*decoder[9:18])
        self.stage3_decoder = nn.Sequential(*decoder[18:27])
        self.stage4_decoder = nn.Sequential(*decoder[27:33])
        self.stage5_decoder = nn.Sequential(*decoder[33:],
                nn.Conv2d(64, num_classes, kernel_size=3, stride=1, padding=1)
        )
        
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)
        # self.unpool = nn.MaxPool2d(kernel_size=2, stride=2)


        self._initialize_weights(self.stage1_decoder, self.stage2_decoder, self.stage3_decoder,
                                    self.stage4_decoder, self.stage5_decoder)
        if freeze_bn: 
            self.freeze_bn()
        # else: 
            # set_trainable([self.stage1_encoder, self.stage2_encoder, self.stage3_encoder, self.stage4_encoder, self.stage5_encoder], False)

    def _initialize_weights(self, *stages):
        for modules in stages:
            for module in modules.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight)
                    if module.bias is not None:
                        module.bias.data.zero_()
                elif isinstance(module, nn.BatchNorm2d):
                    module.weight.data.fill_(1)
                    module.bias.data.zero_()

    def forward(self, x):
        # Encoder
        x = self.stage1_encoder(x)
        x1_size = x.size()
        x, indices1 = self.pool(x)

        x = self.stage2_encoder(x)
        x2_size = x.size()
        x, indices2 = self.pool(x)

        x = self.stage3_encoder(x)
        x3_size = x.size()
        x, indices3 = self.pool(x)

        x = self.stage4_encoder(x)
        x4_size = x.size()
        x, indices4 = self.pool(x)

        x = self.stage5_encoder(x)
        x5_size = x.size()
        x, indices5 = self.pool(x)

        # Decoder
        x = self.unpool(x, indices=indices5, output_size=x5_size)
        x = self.stage1_decoder(x)

        x = self.unpool(x, indices=indices4, output_size=x4_size)
        x = self.stage2_decoder(x)

        x = self.unpool(x, indices=indices3, output_size=x3_size)
        x = self.stage3_decoder(x)

        x = self.unpool(x, indices=indices2, output_size=x2_size)
        x = self.stage4_decoder(x)

        x = self.unpool(x, indices=indices1, output_size=x1_size)
        x = self.stage5_decoder(x)

        return x

    def get_backbone_params(self):
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()



