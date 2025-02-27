import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import init

def make_layers(batch_norm=True):
    cfg = [64, 64, 'M', 64, 64, 'M', 128, 128, 'M', 128, 128]
    layers = []
    input_channel = 2
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue
        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]
        if batch_norm:
            layers += [nn.BatchNorm2d(l)]
        layers += [nn.ReLU(inplace=True)]
        input_channel = l
    return nn.Sequential(*layers)

def pyramid(im, n_levels=3):
    _, _, height, width = im.size()
    ims = [im]
    for i in range(1, n_levels):
        h = height // (2 ** i)
        w = width // (2 ** i)
        resized = F.interpolate(im, (h, w), mode='bilinear')
        ims.append(resized)
    return ims[::-1]

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)
        init.zeros_(m.bias.data)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, mid_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, mid_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        
        self.inc = DoubleConv(2, 32, 32)
        self.down1 = Down(32, 64, 64)
        self.down2 = Down(64, 64, 64)
        self.down3 = Down(64, 128, 128)
        self.out_layer = nn.Sequential(nn.AvgPool2d(kernel_size=4, stride=4), 
                                       nn.Dropout(0.2),
                                       nn.Conv2d(128, 8, 1))
                
    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        output = self.out_layer(x3).view(-1,4,2)
        return output
    
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()

        self.inc = DoubleConv(2, 32, 32)
        self.down1 = Down(32, 64, 64)
        self.down2 = Down(64, 64, 64)
        self.down3 = Down(64, 128, 128)
        self.down4 = Down(128, 128, 128)
        self.out_layer = nn.Sequential(nn.AvgPool2d(kernel_size=4, stride=4), 
                                       nn.Dropout(0.2),
                                       nn.Conv2d(128, 8, 1))

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        output = self.out_layer(x4).view(-1,4,2)
        return output
    
class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()

        self.inc = DoubleConv(2, 32, 32)
        self.down1 = Down(32, 64, 64)
        self.down2 = Down(64, 64, 64)
        self.down3 = Down(64, 128, 128)
        self.down4 = Down(128, 128, 128)
        self.down5 = Down(128, 256, 256)
        self.out_layer = nn.Sequential(nn.AvgPool2d(kernel_size=4, stride=4), 
                                       nn.Dropout(0.2),
                                       nn.Conv2d(256, 8, 1))

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        output = self.out_layer(x5).view(-1,4,2)
        return output


def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = torch.autograd.Variable(grid) + flo

    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :] / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :] / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).to(x.device)
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output * mask    

class Conv1(nn.Module):
    def __init__(self, input_dim = 145):
        super(Conv1, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_dim, 128, 1, padding=0, stride=1), nn.ReLU(), 
        )

    def forward(self, x):
        x = self.layer0(x)
        return x


class Conv3(nn.Module):
    def __init__(self, input_dim = 130):
        super(Conv3, self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(input_dim, 128, 3, padding=1, stride=1), nn.ReLU(), 
        )

    def forward(self, x):
        x = self.layer0(x)
        return x