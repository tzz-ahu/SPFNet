#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import scipy.misc
import torch
import torch.nn as nn
from torchvision import transforms
import torchvision
import torch.nn.functional as F
from .resnet import ResNet50
from .darknet import CSPDarknet
from .darknet import TCSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv, Siu, mysigmoid,DilatedEncoder,GELU
import cv2
from .attention import CBAM, SE,ChannelAttention,SpatialAttention
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
            self,
            depth=1.0,
            width=1.0,
            in_features=("stem", "dark2", "dark3", "dark4", "dark5"),
            in_channels=[256, 512, 1024],
            depthwise=False,
            act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.tbackbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.backbonet = TCSPDarknet(depth, width, depthwise=depthwise, act="noact")
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.toPIL = transforms.ToPILImage()

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        self.convx0 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.convx1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.convx2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.convx23 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.convx3 = nn.Conv2d(128 + 64, 128, kernel_size=3, padding=1)
        self.convx4 = nn.Conv2d(256 + 64, 256, kernel_size=3, padding=1)
        self.convx5 = nn.Conv2d(512 + 64, 512, kernel_size=3, padding=1)
        self.convx32 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.convx31 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.convx30 = nn.Conv2d(64, 512, kernel_size=3, padding=1)
        self.convxa = nn.Conv2d(6, 1, kernel_size=3, padding=1)
        self.convxb = nn.Conv2d(2, 1, kernel_size=3, padding=1)

        self.crelu = Siu()
        self.gelu = GELU()
        self.pool = nn.MaxPool2d((2, 2), 2)
        self.cbam_00 = CBAM(int(1024 * width))
        self.cbam_0 = CBAM(int(in_channels[2] * width))
        self.cbam_1 = CBAM(int(in_channels[1] * width))
        self.cbam_2 = CBAM(int(in_channels[0] * width))
        self.cbam_3 = CBAM(int(64 * width))
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.CAE0 = ChannelAttention(512)
        self.CAE1 = ChannelAttention(256)
        self.CAE2 = ChannelAttention(128)
        self.sa0 = SpatialAttention(512)
        self.sa1 = SpatialAttention(256)
        self.sa2 = SpatialAttention(128)
        

    def forward(self, input,input1=torch.zeros(1, 3, 64, 64), tinput=torch.zeros(1, 3, 64, 64),tinput1=torch.zeros(1, 3, 64, 64)):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """
        #  backbone
        out_features = self.backbone(input,tinput,input1)
        features = [out_features[f] for f in self.in_features]
        [x4, x3, x2, x1, x0] = features
        out_tfea = self.tbackbone(tinput,input,input1)
        tfea = [out_tfea[f] for f in self.in_features]
        [b4, b3, b2, b1, b0] = tfea
        out_fea = self.backbone(input1,tinput,input)
        fea = [out_fea[f] for f in self.in_features]
        [a4, a3, a2, a1, a0] = fea
        x2 = self.cbam_2(x2,a2,b2,x2-a2-b2)   #TDF Block
        x1 = self.cbam_1(x1,a1,b1,x1-a1-b1)
        x0 = self.cbam_0(x0,a0,b0,x0-a0-b0)
        outputs = (x2, x1, x0)
        return outputs


# B, C, H, W = x0.size()
#
# features1 = x0
# features2 = b0
#
# plt.figure(figsize=(8, 6))
#
# for i in range(C):
#     features1_flattened = features1[:, i, :, :].reshape(1, -1)
#     features2_flattened = features2[:, i, :, :].reshape(1, -1)
#     features_combined = np.vstack([features1_flattened, features2_flattened])  # 形状变为(2, C*H*W)
#
#     pca = PCA(n_components=2)
#     pca_features = pca.fit_transform(features_combined)
#
#     plt.scatter(pca_features[0, 0], pca_features[0, 1], c='#780000')
#     plt.scatter(np.abs(pca_features[1, 0]), pca_features[1, 1], c='#669bbc')
#
# plt.xlim([-60, 60])
#
# plt.savefig('3d_pca_distance_visualization.png', dpi=400)