from parameters import MetaParameters

import torch

from torch import nn
from collections import OrderedDict
# from Model import resnet
from torchvision import models


class UNet_3D(nn.Module, MetaParameters):

    def __init__(self):
        super(UNet_3D, self).__init__()
        super(MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        out_channels = self.NUM_CLASS
        dropout = self.DROPOUT

        self.dropout = nn.Dropout3d(dropout, inplace=False)
        self.encoder1 = UNet_3D.Conv2x2(in_channels, features, name = "enc1")
        self.pool1 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder2 = UNet_3D.Conv2x2(features, features * 2, name = "enc2")
        self.pool2 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder3 = UNet_3D.Conv2x2(features * 2, features * 4, name = "enc3")
        self.pool3 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder4 = UNet_3D.Conv2x2(features * 4, features * 8, name = "enc4")
        self.pool4 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.bottleneck = UNet_3D.Conv2x2(features * 8, features * 16, name = "bottleneck")
        self.upconv4 = nn.ConvTranspose3d(features * 16, features * 8, kernel_size = 2, stride = 2)
        self.decoder4 = UNet_3D.Conv2x2((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose3d(features * 8, features * 4, kernel_size = 2, stride = 2)
        self.decoder3 = UNet_3D.Conv2x2((features * 4) * 2, features * 4, name = "dec3")
        self.upconv2 = nn.ConvTranspose3d(features * 4, features * 2, kernel_size = 2, stride = 2)
        self.decoder2 = UNet_3D.Conv2x2((features * 2) * 2, features * 2, name = "dec2")
        self.upconv1 = nn.ConvTranspose3d(features * 2, features, kernel_size = 2, stride = 2)
        self.decoder1 = UNet_3D.Conv2x2(features * 2, features, name = "dec1")
        self.conv = nn.Conv3d(in_channels = features, out_channels = out_channels, kernel_size = 1)

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
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def Conv2x2(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels = in_channels,
                            out_channels = features,
                            kernel_size = 3,
                            stride=1,
                            padding = 1,
                            bias = False,
                        ),
                    ),

                    (name + "norm1", nn.InstanceNorm3d(320, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    # (name + "relu1", nn.ReLU()),

                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels = features,
                            out_channels = features,
                            kernel_size = 3,
                            stride=1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(320, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = False)),
                    (name + "relu2", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    # (name + "relu2", nn.ReLU()),

                ]
            )
        )


class UNet_3D_AttantionLayer(nn.Module, MetaParameters):

    def __init__(self):
        super(UNet_3D_AttantionLayer, self).__init__()
        super(MetaParameters, self).__init__()

        features = self.FEATURES
        in_channels = self.CHANNELS
        out_channels = self.NUM_CLASS
        dropout = self.DROPOUT
        freeze_bn = self.FREEZE_BN

        self.dropout = nn.Dropout3d(dropout)
        self.encoder1 = UNet_3D_AttantionLayer.Conv2x2(in_channels, features, name = "enc1")
        self.pool1 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder2 = UNet_3D_AttantionLayer.Conv2x2(features, features * 2, name = "enc2")
        self.pool2 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder3 = UNet_3D_AttantionLayer.Conv2x2(features * 2, features * 4, name = "enc3")
        self.pool3 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.encoder4 = UNet_3D_AttantionLayer.Conv2x2(features * 4, features * 8, name = "enc4")
        self.pool4 = nn.MaxPool3d(kernel_size = 2, stride = 2)
        self.bottleneck = UNet_3D_AttantionLayer.Conv2x2(features * 8, features * 16, name = "bottleneck")
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size = 2, stride = 2)
        self.Att4 = Attention_3D(features * 8,features * 8,features * 4)
        self.decoder4 = UNet_3D_AttantionLayer.Conv2x2((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size = 2, stride = 2)
        self.Att3 = Attention_3D(features * 4,features * 4,features * 2)
        self.decoder3 = UNet_3D_AttantionLayer.Conv2x2((features * 4) * 2, features * 4, name = "dec3")
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size = 2, stride = 2)
        self.Att2 = Attention_3D(features * 2,features * 2,features * 1)
        self.decoder2 = UNet_3D_AttantionLayer.Conv2x2((features * 2) * 2, features * 2, name = "dec2")
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size = 2, stride = 2)
        self.Att1 = Attention_3D(features, features, features // 2)
        self.decoder1 = UNet_3D_AttantionLayer.Conv2x2(features * 2, features, name = "dec1")
        self.conv = nn.Conv3d(in_channels = features, out_channels = out_channels, kernel_size = 1)

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
        return torch.sigmoid(self.conv(dec1))

    @staticmethod
    def Conv2x2(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv3d(
                            in_channels = in_channels,
                            out_channels = features,
                            kernel_size = 3,
                            stride=1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(320, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = False)),
                    (name + "relu1", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    # (name + "relu1", nn.ReLU()),

                    (
                        name + "conv2",
                        nn.Conv3d(
                            in_channels = features,
                            out_channels = features,
                            kernel_size = 3,
                            stride=1,
                            padding = 1,
                            bias = False,
                        ),
                    ),
                    (name + "norm1", nn.InstanceNorm3d(320, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = False)),
                    (name + "relu2", nn.LeakyReLU(negative_slope = 0.1, inplace = True)),
                    # (name + "relu1", nn.ReLU()),

                ]
            )
        )


class Attention_3D(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_3D,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm3d(F_int, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = False),
            )
        
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm3d(F_int, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = False),
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size = 3, stride = 1, padding = 1, bias = False),
            nn.InstanceNorm3d(1, eps = 1e-5, momentum = 0.1, affine = True, track_running_stats = False),
            nn.Softmax(dim = 1)
        )
        
        self.relu = nn.LeakyReLU(negative_slope = 0.1, inplace = True),
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


