import torch
import torch.nn as nn

class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class up_sample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(up_sample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = double_conv(in_channels, out_channels)

    def forward(self, a, b):
        a = self.up(a)
        x = torch.cat([a, b], 1)
        return self.conv(x)

class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, shortcut=None):
        super(residual_block, self).__init__()
        self.right = shortcut
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        return out + residual

class resnet34_unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resnet34_unet, self).__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer_1 = self.make_layer(64, 64, 3)
        self.layer_2 = self.make_layer(64, 128, 4, stride=2)
        self.layer_3 = self.make_layer(128, 256, 6, stride=2)
        self.layer_4 = self.make_layer(256, 512, 3, stride=2)

        self.bottleneck = self.make_layer(512, 1024, 1, stride=2)

        self.up_1 = up_sample(1024, 512)
        self.up_2 = up_sample(512, 256)
        self.up_3 = up_sample(256, 128)
        self.up_4 = up_sample(128, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def make_layer(self, in_channels, out_channels, block_num, stride=1):
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        ) if stride != 1 or in_channels != out_channels else None

        layers = [residual_block(in_channels, out_channels, stride, shortcut)]
        for _ in range(1, block_num):
            layers.append(residual_block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.pre(x)
        out = self.layer_1(out)
        down_1 = out
        out = self.layer_2(out)
        down_2 = out
        out = self.layer_3(out)
        down_3 = out
        out = self.layer_4(out)
        down_4 = out

        out = self.bottleneck(out)

        up_1 = self.up_1(out, down_4)
        up_2 = self.up_2(up_1, down_3)
        up_3 = self.up_3(up_2, down_2)
        up_4 = self.up_4(up_3, down_1)

        return self.final(up_4)


