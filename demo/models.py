import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # print(y.expand_as(x))
        return x * y.expand_as(x)


class Encoder_2D_dilation_se_res_new(nn.Module):
    def __init__(self,kernel_size,dilation):
        super(Encoder_2D_dilation_se_res_new, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=11, stride=1,dilation=dilation, padding=int(dilation*(11 - 1) / 2)),
            nn.BatchNorm2d(4))
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=9, stride=1, dilation=2 * dilation,
                      padding=int(2 * dilation * (9 - 1) / 2)),
            nn.BatchNorm2d(8))
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=9, stride=1, dilation=4 * dilation,
                      padding=int(4 * dilation * (9 - 1) / 2)),
            nn.BatchNorm2d(16))
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, dilation=8 * dilation,
                      padding=int(8 * dilation * (kernel_size - 1) / 2)),
            nn.BatchNorm2d(32))
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, dilation=16 * dilation,
                      padding=int(16 * dilation * (kernel_size - 1) / 2)),
            nn.BatchNorm2d(64))
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, dilation=32 * dilation,
                      padding=int(32 * dilation * (kernel_size - 1) / 2)),
            nn.BatchNorm2d(128))

        self.res3 = nn.Conv2d(16, 32, kernel_size=1)
        self.res4 = nn.Conv2d(32, 64, kernel_size=1)
        self.res5 = nn.Conv2d(64, 128, kernel_size=1)
        # self.res6 = nn.Conv2d(128, 256, kernel_size=1)

        self.se1 = SELayer(4, 1)
        self.se2 = SELayer(8, 1)
        self.se3 = SELayer(16, 1)
        self.se41 = SELayer(32, 32)
        self.se42 = SELayer(32, 32)
        self.se51 = SELayer(64, 64)
        self.se52 = SELayer(64, 64)
        self.se61 = SELayer(128, 128)
        self.se62 = SELayer(128, 128)

        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(128,1, kernel_size=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.relu(out1)
        out1 = self.se1(out1)

        out2 = self.conv2(out1)
        out2 = self.relu(out2)
        out2 = self.se2(out2)

        out3 = self.conv3(out2)
        out3 = self.relu(out3)
        out3 = self.se3(out3)

        out33 = self.res3(out3)
        out33 = self.se41(out33)
        out4 = self.conv4(out3)
        out4 = self.se42(out4)
        out4 = out4 + out33
        out4 = self.relu(out4)


        out44 = self.res4(out4)
        out44 = self.se51(out44)
        out5 = self.conv5(out4)
        out5 = self.se52(out5)
        out5 = out5 + out44
        out5 = self.relu(out5)


        out55 = self.res5(out5)
        out55 = self.se61(out55)
        out6 = self.conv6(out5)
        out6 = self.se62(out6)
        out6 = out6 + out55
        out6 = self.relu(out6)
      

        out = self.conv1x1(out6)

        return out


class Encoder_2D_dilation_se_res_new8(nn.Module):
    def __init__(self,kernel_size,dilation):
        super(Encoder_2D_dilation_se_res_new8, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=11, stride=1,dilation=dilation, padding=int(dilation*(11 - 1) / 2)),
            nn.BatchNorm2d(4))
        self.conv2 = nn.Sequential(
            nn.Conv2d(4, 8, kernel_size=9, stride=1, dilation=2 * dilation,
                      padding=int(2 * dilation * (9 - 1) / 2)),
            nn.BatchNorm2d(8))
        self.conv3 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=9, stride=1, dilation=4 * dilation,
                      padding=int(4 * dilation * (9 - 1) / 2)),
            nn.BatchNorm2d(16))
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=9, stride=1, dilation=8 * dilation,
                      padding=int(8 * dilation * (9 - 1) / 2)),
            nn.BatchNorm2d(32))
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=9, stride=1, dilation=16 * dilation,
                      padding=int(16 * dilation * (9 - 1) / 2)),
            nn.BatchNorm2d(64))
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=kernel_size, stride=1, dilation=32 * dilation,
                      padding=int(32 * dilation * (kernel_size - 1) / 2)),
            nn.BatchNorm2d(128))
        self.conv7 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, dilation=32 * dilation,
                      padding=int(32 * dilation * (kernel_size - 1) / 2)),
            nn.BatchNorm2d(128))
        self.conv8 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=kernel_size, stride=1, dilation=32 * dilation,
                      padding=int(32 * dilation * (kernel_size - 1) / 2)),
            nn.BatchNorm2d(128))

        self.res4 = nn.Conv2d(32, 64, kernel_size=1)
        self.res5 = nn.Conv2d(64, 128, kernel_size=1)
        self.res6 = nn.Conv2d(128, 128, kernel_size=1)
        self.res7 = nn.Conv2d(128, 128, kernel_size=1)

        self.se1 = SELayer(4, 1)
        self.se2 = SELayer(8, 1)
        self.se3 = SELayer(16, 1)
        self.se4 = SELayer(32, 1)
        self.se41 = SELayer(64, 64)
        self.se42 = SELayer(64, 64)
        self.se51 = SELayer(128, 128)
        self.se52 = SELayer(128, 128)
        self.se61 = SELayer(128, 128)
        self.se62 = SELayer(128, 128)
        self.se71 = SELayer(128, 128)
        self.se72 = SELayer(128, 128)

        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(128,1, kernel_size=1)

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.relu(out1)
        out1 = self.se1(out1)

        out2 = self.conv2(out1)
        out2 = self.relu(out2)
        out2 = self.se2(out2)

        out3 = self.conv3(out2)
        out3 = self.relu(out3)
        out3 = self.se3(out3)

        out4 = self.conv4(out3)
        out4 = self.relu(out4)
        out4 = self.se4(out4)

        out44 = self.res4(out4)
        out44 = self.se41(out44)
        out5 = self.conv5(out4)
        out5 = self.se42(out5)
        out5 = out5 + out44
        out5 = self.relu(out5)

        out55 = self.res5(out5)
        out55 = self.se51(out55)
        out6 = self.conv6(out5)
        out6 = self.se52(out6)
        out6 = out6 + out55
        out6 = self.relu(out6)

        out66 = self.res6(out6)
        out66 = self.se61(out66)
        out7 = self.conv7(out6)
        out7 = self.se61(out7)
        out7 = out7 + out66
        out7 = self.relu(out7)

        out77 = self.res7(out7)
        out77 = self.se71(out77)
        out8 = self.conv8(out7)
        out8 = self.se72(out8)
        out8 = out8 + out77
        out8 = self.relu(out8)
        
        out = self.conv1x1(out8)

        return out

