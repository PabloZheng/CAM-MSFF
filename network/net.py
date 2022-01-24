#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F


class VGG(nn.Module):

    def __init__(self, features,  num_classes=7, init_weights=True):
        super(VGG, self).__init__()
        self.features = features
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(5, 1024)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 5)
        self.Conv2 = nn.Conv2d(64, num_classes, 1, padding=0)
        self.Conv7 = nn.Conv2d(128, num_classes, 1, padding=0)
        self.Conv16 = nn.Conv2d(256, num_classes, 1, padding=0)
        self.Conv25 = nn.Conv2d(512, num_classes, 1, padding=0)
        self.Conv34 = nn.Conv2d(512, num_classes, 1, padding=0)
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self.ms_input = {}

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        for i in range(36):
            x = self.features[i](x)
            if i == 3:
                scale_1 = self.Conv2(x)
                self.ms_input[2] = scale_1
                w2 = self.avg_pool(scale_1).sum(dim=1).unsqueeze(dim=1)/64
            elif i == 8:
                scale_2 = self.Conv7(x)
                self.ms_input[7] = scale_2
                w7 = self.avg_pool(scale_2).sum(dim=1).unsqueeze(dim=1)/128
            elif i == 17:
                scale_3 = self.Conv16(x)
                self.ms_input[16] = scale_3
                w16 = self.avg_pool(scale_3).sum(dim=1).unsqueeze(dim=1)/256
            elif i == 26:
                scale_4 = self.Conv25(x)
                self.ms_input[25] = scale_4
                w25 = self.avg_pool(scale_4).sum(dim=1).unsqueeze(dim=1)/512
            elif i == 35:
                scale_5 = self.Conv34(x)
                self.ms_input[34] = scale_5
                w34 = self.avg_pool(scale_5).sum(dim=1).unsqueeze(dim=1)/512

        scale_w = torch.cat((w2, w7, w16, w25, w34), dim=1)
        scale_w = self.fc1(scale_w.squeeze())
        scale_w = self.fc2(self.relu1(scale_w))
        scale_w = scale_w.unsqueeze(dim=2).unsqueeze(dim=2)
        scale_w = self.sigmoid(scale_w)
        b, c, h, w = self.ms_input[2].shape[0], self.ms_input[2].shape[1], \
                     self.ms_input[2].shape[-2], self.ms_input[2].shape[-1]

        x_scale_2 = F.interpolate(self.ms_input[7], size=(h, w))
        x_scale_3 = F.interpolate(self.ms_input[16], size=(h, w))
        x_scale_4 = F.interpolate(self.ms_input[25], size=(h, w))
        x_scale_5 = F.interpolate(self.ms_input[34], size=(h, w))

        fusion_x = scale_w[:, 0, :, :].unsqueeze(1)*self.ms_input[2] + \
                   scale_w[:, 1, :, :].unsqueeze(1)*x_scale_2 + scale_w[:, 2, :, :].unsqueeze(1)*x_scale_3 + \
                   scale_w[:, 3, :, :].unsqueeze(1)*x_scale_4 + scale_w[:, 4, :, :].unsqueeze(1)*x_scale_5

        out = self.softmax(self.avg_pool(fusion_x)).squeeze(dim=-1).squeeze(dim=-1)
        outlist = self.classifier(self.ms_input)
        outlist.append(out)

        return outlist, scale_w

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(batch_norm=False):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
    
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)









