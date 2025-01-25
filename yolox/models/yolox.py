#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import torch
import torch.nn as nn
import cv2
import numpy
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN


class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        if backbone is None:
            backbone = YOLOPAFPN()
        if head is None:
            head = YOLOXHead(7)

        self.backbone = backbone
        self.head = head
        self.conv=nn.Conv2d(6,3,kernel_size=3,padding=1)



        
    def forward(self, x, x1 = torch.zeros(1,3,64,64), y = torch.zeros(1,3,64,64),y1 = torch.zeros(1,3,64,64),targets=None):
        # fpn output content features of [dark3, dark4, dark5]
        
        ''' with torch.no_grad():
          input_tensor = x.clone().detach()
          # 到cpu
          input_tensor = input_tensor.to(torch.device('cpu'))
          # 反归一化
          # input_tensor = unnormalize(input_tensor)
          # 去掉批次维度
          input_tensor=input_tensor[0]
          #print(input_tensor)
          #input_tensor = input_tensor.squeeze()
          # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2   .add_(0.5).mul_(255)
          input_tensor = input_tensor.clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
          # RGB转BRG
          input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
          cv2.imwrite('inputx.jpg', input_tensor)
          input_tensor2 = y.clone().detach()
          # 到cpu
          input_tensor2 = input_tensor2.to(torch.device('cpu'))
          # 反归一化
          # input_tensor = unnormalize(input_tensor)
          # 去掉批次维度
          input_tensor2=input_tensor2[0]
          #print(input_tensor)
          #input_tensor = input_tensor.squeeze()
          # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2   .add_(0.5).mul_(255)
          input_tensor2 = input_tensor2.clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
          # RGB转BRG
          input_tensor2 = cv2.cvtColor(input_tensor2, cv2.COLOR_RGB2BGR)
          cv2.imwrite('inputy.jpg', input_tensor2)'''
        '''with torch.no_grad():
            img = x.clone().detach()
            imgt = y.clone().detach()
            img = img.to(torch.device('cpu'))
            imgt = imgt.to(torch.device('cpu'))
            # print(img.shape)
            img = img[0]
            # print(img.shape)
            imgt = imgt[0]
            img = img.clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            imgt = imgt.clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imgt = cv2.cvtColor(imgt, cv2.COLOR_RGB2BGR)
            cv2.imwrite('input_x.jpg', img)
            cv2.imwrite('input_y.jpg', imgt)
        with torch.no_grad():
            img = x2.clone().detach()
            imgt = y2.clone().detach()
            img = img.to(torch.device('cpu'))
            imgt = imgt.to(torch.device('cpu'))
            # print(img.shape)
            img = img[0]
            # print(img.shape)
            imgt = imgt[0]
            img = img.clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            imgt = imgt.clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            imgt = cv2.cvtColor(imgt, cv2.COLOR_RGB2BGR)
            cv2.imwrite('input_x1.jpg', img)
            cv2.imwrite('input_y1.jpg', imgt)'''
        '''x=torch.cat([x,y],dim=1)
        x=self.conv(x)'''
        #print(x.shape)
        fpn_outs = self.backbone(x,x1,y,y1)
        
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, x
            )
            outputs = {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
        else:
            outputs = self.head(fpn_outs)

        return outputs
