# coding=gbk
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MyDepthWiseConv(nn.Module):
    def __init__(self, ch, kernel_size, stride=1, padding=0, dilation=1):
        super(MyDepthWiseConv, self).__init__()
        self.unfold = nn.Unfold(kernel_size, dilation, padding, stride)
        self.k = kernel_size
        self.ch = ch
        self.w = nn.Parameter(torch.empty(ch, kernel_size, kernel_size))
        self.b = nn.Parameter(torch.zeros(1, ch, 1, 1))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.b, -bound, bound)

    def forward(self, x, ksa):
        n, c, h, w = ksa.shape
        ksa = ksa.unsqueeze(4).transpose(1, 4).view(n, 1, h, w, self.k, self.k)
        x = self.unfold(x).reshape(n, self.ch, self.k, self.k, h, w)
        x = x.transpose(2, 4).transpose(3, 5)
        x = torch.einsum('nchwkj,ckj->nchw', x*ksa, self.w)
        x = x + self.b
        return x


class KSA(nn.Module):
    def __init__(self, ch, kernel_size=5, dilation=1):
        super(KSA, self).__init__()
        self.conv = nn.Conv2d(ch, ch, (5, 5), padding=2*dilation, dilation=dilation, groups=ch)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, 2*kernel_size**2, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*kernel_size**2, kernel_size**2, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.mlp(x)
        return x


class KCA(nn.Module):
    def __init__(self, ch, dilation=1):
        super(KCA, self).__init__()
        self.conv = nn.Conv2d(ch, ch, (5, 5), padding=2*dilation, dilation=dilation, groups=ch)
        self.mlp = nn.Sequential(
            nn.Conv2d(ch, ch//4, (1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch//4, ch, (1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = self.mlp(x)
        return x


class UpConv(nn.Module):
    def __init__(self, chin, chout, dilation=1, kernel_size=5):
        super(UpConv, self).__init__()
        self.depthwise_conv = nn.Conv2d(chin, chin, (kernel_size, kernel_size),
                                        padding=dilation*(kernel_size//2),
                                        groups=chin, dilation=(dilation, dilation))
        self.conv1 = nn.Conv2d(chin, chout, (1, 1))

    def forward(self, x, kca):
        x = self.depthwise_conv(x)
        x = self.conv1(x * kca)
        return x


class LowConv(nn.Module):
    def __init__(self, chin, chout, dilation=1, kernel_size=5):
        super(LowConv, self).__init__()
        self.depthwise_conv = MyDepthWiseConv(chin, kernel_size, padding=dilation*(kernel_size//2),
                                              dilation=dilation)
        self.conv1 = nn.Conv2d(chin, chout, (1, 1))

    def forward(self, x, ksa):
        x = self.depthwise_conv(x, ksa)
        x = self.conv1(x)
        return x


class CIKAConv(nn.Module):
    def __init__(self, dilation=1):
        super(CIKAConv, self).__init__()
        self.lowconv = LowConv(32, 32, dilation)
        self.upconv = UpConv(32, 32, dilation)
        self.kca = KCA(32, dilation)
        self.ksa = KSA(32, 5, dilation)

    def forward(self, lower, upper):
        kca = self.kca(lower)
        ksa = self.ksa(upper)
        lower = self.lowconv(lower, ksa)
        upper = self.upconv(upper, kca)
        return lower, upper


class CIKABlk(nn.Module):
    def __init__(self, size):
        super(CIKABlk, self).__init__()
        self.conv1 = CIKAConv(1)
        self.conv2 = CIKAConv(2)
        self.ln = nn.LayerNorm([32, size, size], elementwise_affine=False)
        # self.ln3 = nn.LayerNorm([64, size, size], elementwise_affine=False)
        # self.ln4 = nn.LayerNorm([64, size, size], elementwise_affine=False)

    def forward(self, lower, upper):
        lower1, upper1 = self.conv1(lower, upper)
        lower1, upper1 = self.ln(lower1), self.ln(upper1)
        lower2, upper2 = self.conv2(lower1, upper1)
        lower, upper = F.relu(lower2 + lower), F.relu(upper2 + upper)
        return lower, upper


class CIKANet(nn.Module):
    def __init__(self, ch, size=64):
        super(CIKANet, self).__init__()
        self.upper = nn.Conv2d(ch, 32, (3, 3), padding=(1, 1))
        self.lower = nn.Conv2d(ch, 32, (3, 3), padding=(1, 1))
        self.blk1 = CIKABlk(size)
        self.blk2 = CIKABlk(size)
        self.blk3 = CIKABlk(size)
        self.blk4 = CIKABlk(size)
        self.fuse = nn.Conv2d(64, ch, (3, 3), padding=(1, 1))

    def forward(self, ms, pan):
        lms = F.interpolate(ms, scale_factor=4, mode='bicubic')
        _input = torch.subtract(pan, lms)
        upper_features = F.relu(self.upper(_input))
        lower_features = F.relu(self.lower(lms))
        lower_features, upper_features = self.blk1(lower_features, upper_features)
        lower_features, upper_features = self.blk2(lower_features, upper_features)
        lower_features, upper_features = self.blk3(lower_features, upper_features)
        lower_features, upper_features = self.blk4(lower_features, upper_features)
        features = torch.cat([upper_features, lower_features], dim=1)
        sr = self.fuse(features)+lms
        return sr
