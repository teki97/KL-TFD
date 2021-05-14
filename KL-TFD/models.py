import torch
from torch import nn
from torch.nn import functional as F
from bam_gn import BAM_GN


class Encoder_conv_bam_skip(nn.Module):
    def __init__(self, nlayers):
        super(Encoder_conv_bam_skip, self).__init__()

        self.nlayers = nlayers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
        ) for _ in range(self.nlayers - 1)])

        self.bam = nn.ModuleList([BAM_GN(8) for _ in range(int(self.nlayers/2))])
        self.conv1x1 = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x_temp = x
        for i in range(self.nlayers - 1):
            if ((i+1)%2==0):
                x_temp1 = x
                x = x + x_temp
                x_temp = x_temp1
                x = self.bam[int((i-1)/2)](x)
            x = self.conv2[i](x)

        x = self.conv1x1(x)

        return F.sigmoid(x)

    def get_params_prev(self, lidx):
        if lidx==0:
            for param in self.conv1.parameters():
                yield param
        else:
            for param in self.conv1.parameters():
                yield param
            for i in range(lidx):
                for param in self.conv2[i].parameters():
                    yield param
                if ((i-1)%2==0):
                    for param in self.bam[int((i-1)/2)].parameters():
                        yield param

    def get_params_latt(self, lidx, eidx):
        for i in range(lidx, eidx):
            for param in self.conv2[i].parameters():
                yield param
        for param in self.conv1x1.parameters():
            yield param
        if ((i - 1) % 2 == 0):
            for param in self.bam[int((i - 1) / 2)].parameters():
                yield param


class Encoder_conv_nobam_skip(nn.Module):
    def __init__(self, nlayers):
        super(Encoder_conv_nobam_skip, self).__init__()

        self.nlayers = nlayers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
        ) for _ in range(self.nlayers - 1)])

        self.conv1x1 = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x_temp = x
        for i in range(self.nlayers - 1):
            if ((i+1)%2==0):
                x_temp1 = x
                x = x + x_temp
                x_temp = x_temp1
                # x = self.bam[int((i-1)/2)](x)
            x = self.conv2[i](x)

        x = self.conv1x1(x)

        return F.sigmoid(x)

    def get_params_prev(self, lidx):
        if lidx==0:
            for param in self.conv1.parameters():
                yield param
        else:
            for param in self.conv1.parameters():
                yield param
            for i in range(lidx):
                for param in self.conv2[i].parameters():
                    yield param
                
    def get_params_latt(self, lidx, eidx):
        for i in range(lidx, eidx):
            for param in self.conv2[i].parameters():
                yield param
        for param in self.conv1x1.parameters():
            yield param
         

class Encoder_conv_bam_noskip(nn.Module):
    def __init__(self, nlayers):
        super(Encoder_conv_bam_noskip, self).__init__()

        self.nlayers = nlayers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
        ) for _ in range(self.nlayers - 1)])

        self.bam = nn.ModuleList([BAM_GN(8) for _ in range(int(self.nlayers/2))])
        self.conv1x1 = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x_temp = x
        for i in range(self.nlayers - 1):
            if ((i+1)%2==0):
                x = self.bam[int((i-1)/2)](x)
            x = self.conv2[i](x)

        x = self.conv1x1(x)

        return F.sigmoid(x)

    def get_params_prev(self, lidx):
        if lidx==0:
            for param in self.conv1.parameters():
                yield param
        else:
            for param in self.conv1.parameters():
                yield param
            for i in range(lidx):
                for param in self.conv2[i].parameters():
                    yield param
                if ((i-1)%2==0):
                    for param in self.bam[int((i-1)/2)].parameters():
                        yield param

    def get_params_latt(self, lidx, eidx):
        for i in range(lidx, eidx):
            for param in self.conv2[i].parameters():
                yield param
        for param in self.conv1x1.parameters():
            yield param
        if ((i - 1) % 2 == 0):
            for param in self.bam[int((i - 1) / 2)].parameters():
                yield param

class Encoder_conv_nobam_noskip(nn.Module):
    def __init__(self, nlayers):
        super(Encoder_conv_nobam_noskip, self).__init__()

        self.nlayers = nlayers
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.ModuleList([nn.Sequential(
            nn.Conv2d(8, 8, kernel_size=5, padding=2),
            nn.GroupNorm(4, 8),
            nn.ReLU(inplace=True),
        ) for _ in range(self.nlayers - 1)])

        self.conv1x1 = nn.Conv2d(8, 1, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x_temp = x
        for i in range(self.nlayers - 1):
            x = self.conv2[i](x)

        x = self.conv1x1(x)

        return F.sigmoid(x)

    def get_params_prev(self, lidx):
        if lidx==0:
            for param in self.conv1.parameters():
                yield param
        else:
            for param in self.conv1.parameters():
                yield param
            for i in range(lidx):
                for param in self.conv2[i].parameters():
                    yield param

    def get_params_latt(self, lidx, eidx):
        for i in range(lidx, eidx):
            for param in self.conv2[i].parameters():
                yield param
        for param in self.conv1x1.parameters():
            yield param
