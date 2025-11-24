import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

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
  def forward(self,x):
    return self.conv(x)

class down_sample(nn.Module):
  def __init__(self, in_channels, out_channels):
     super(down_sample, self).__init__()
     self.double_conv = double_conv(in_channels=in_channels, out_channels=out_channels)
     self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

  def forward(self,x):
    double_cov = self.double_conv(x)
    max_pool = self.maxpool(double_cov)  # for later concatenate
    return double_cov, max_pool

class up_sample(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(up_sample, self).__init__()
    self.up = nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels // 2, kernel_size=2, stride=2)
    self.double_conv = double_conv(in_channels=in_channels, out_channels=out_channels)

  def forward(self,a,b):
    a=self.up(a)
    tensor_cat = torch.cat((a,b),dim=1)
    return self.double_conv(tensor_cat)

class unet(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(unet, self).__init__()
    self.down_1 = down_sample(in_channels=in_channels, out_channels=64)
    self.down_2 = down_sample(in_channels=64, out_channels=128)
    self.down_3 = down_sample(in_channels=128, out_channels=256)
    self.down_4 = down_sample(in_channels=256, out_channels=512)

    self.bottleneck = double_conv(in_channels=512, out_channels=1024)

    self.up_1 = up_sample(in_channels=1024, out_channels=512)
    self.up_2 = up_sample(in_channels=512, out_channels=256)
    self.up_3 = up_sample(in_channels=256, out_channels=128)
    self.up_4 = up_sample(in_channels=128, out_channels=64)

    self.out = nn.Conv2d(in_channels=64, out_channels=out_channels, kernel_size=1)
    self.final = nn.Sigmoid()

  def forward(self,x):
    down_1, max_pool_1 = self.down_1(x)
    down_2, max_pool_2 = self.down_2(max_pool_1)
    down_3, max_pool_3 = self.down_3(max_pool_2)
    down_4, max_pool_4 = self.down_4(max_pool_3)

    bottleneck = self.bottleneck(max_pool_4)

    up_1 = self.up_1(bottleneck, down_4)
    up_2 = self.up_2(up_1, down_3)
    up_3 = self.up_3(up_2, down_2)
    up_4 = self.up_4(up_3, down_1)

    return self.final(self.out(up_4))
