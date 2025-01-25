#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch.nn.functional as F
from torch import nn
import torch
import cv2
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck,Siu


class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output chanels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("stem","dark2","dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act="silu")  # PIF Block

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act="silu"),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        
        self.convfc=nn.Conv2d(64,32,kernel_size=3,padding=1)
        self.convd2=nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.convd3=nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.convd4=nn.Conv2d(512,256,kernel_size=3,padding=1)
        self.convd5=nn.Conv2d(1024,512,kernel_size=3,padding=1)
        self.crelu = Siu()

    def forward(self, x,y,z):
       
        outputs = {}
        # x=torch.cat([x,y],dim=1)
        # x=self.conv(x)
        x = self.stem(x,y,z)                            #x: 1,32,32,32
        outputs["stem"] = x
        
       

        x = self.dark2(x)                           #x: 1,64,16,16
        outputs["dark2"] = x
        


        '''with torch.no_grad():
          input_tensor = x.clone().detach()
          # 到cpu
          input_tensor = input_tensor.to(torch.device('cpu'))
          # 反归一化
          # input_tensor = unnormalize(input_tensor)
          # 去掉批次维度
          input_tensor=input_tensor[0]
          input_tensor=input_tensor[0:3,:,:]
          #print(input_tensor)
          #input_tensor = input_tensor.squeeze()
          # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2   .add_(0.5).mul_(255)
          input_tensor = input_tensor.clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
          # RGB转BRG
          input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
          cv2.imwrite('dark2.jpg', input_tensor)'''

        x = self.dark3(x)                          #x: 1,128,8,8
        outputs["dark3"] = x

        x = self.dark4(x)                          #x: 1,256,4,4
        outputs["dark4"] = x


        
        x = self.dark5(x)                          #x: 1,512,2,2
        outputs["dark5"] = x
        



        return {k: v for k, v in outputs.items() if k in self.out_features}

               
class TCSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("stem","dark2","dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act="silu")

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act="silu"),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )
        
        self.convfc=nn.Conv2d(64,32,kernel_size=3,padding=1)
        self.convd2=nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.convd3=nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.convd4=nn.Conv2d(512,256,kernel_size=3,padding=1)
        self.convd5=nn.Conv2d(1024,512,kernel_size=3,padding=1)
        
    def forward(self, x):
        outputs = {}
        # x=torch.cat([x,y],dim=1)
        # x=self.conv(x)
        x = self.stem(x)                            #x: 1,32,32,32
        outputs["stem"] = x

        x = self.dark2(x)                           #x: 1,64,16,16
        outputs["dark2"] = x
        '''with torch.no_grad():
          input_tensor = x.clone().detach()
          # 到cpu
          input_tensor = input_tensor.to(torch.device('cpu'))
          # 反归一化
          # input_tensor = unnormalize(input_tensor)
          # 去掉批次维度
          input_tensor=input_tensor[0]
          input_tensor=input_tensor[0:3,:,:]
          #print(input_tensor)
          #input_tensor = input_tensor.squeeze()
          # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2   .add_(0.5).mul_(255)
          input_tensor = input_tensor.clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
          # RGB转BRG
          input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
          cv2.imwrite('dark2.jpg', input_tensor)'''

        x = self.dark3(x)                          #x: 1,128,8,8
        outputs["dark3"] = x

        x = self.dark4(x)                          #x: 1,256,4,4
        outputs["dark4"] = x

        x = self.dark5(x)                          #x: 1,512,2,2
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
