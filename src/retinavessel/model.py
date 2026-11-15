"""U-Net and depthwise-separable lightweight U-Net (research baselines)."""
import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, lite=False):
        super().__init__()
        if lite:
            def conv(i, o):
                return nn.Sequential(nn.Conv2d(i, i, 3, padding=1, groups=i, bias=False), nn.Conv2d(i, o, 1, bias=False))
        else:
            def conv(i, o):
                return nn.Conv2d(i, o, 3, padding=1, bias=False)
        self.layers = nn.Sequential(
            conv(in_channels, out_channels), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            conv(out_channels, out_channels), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.layers(x)

class UNet(nn.Module):
    def __init__(self, base=16, lite=False):
        super().__init__()
        self.enc1 = ConvBlock(3, base, lite)
        self.enc2 = ConvBlock(base, base*2, lite)
        self.enc3 = ConvBlock(base*2, base*4, lite)
        self.enc4 = ConvBlock(base*4, base*8, lite)
        self.bridge = ConvBlock(base*8, base*16, lite)
        self.dec4 = ConvBlock(base*16+base*8, base*8, lite)
        self.dec3 = ConvBlock(base*8+base*4, base*4, lite)
        self.dec2 = ConvBlock(base*4+base*2, base*2, lite)
        self.dec1 = ConvBlock(base*2+base, base, lite)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))
        b = self.bridge(F.max_pool2d(e4, 2))
        d4 = self.dec4(torch.cat((F.interpolate(b, size=e4.shape[-2:], mode='bilinear', align_corners=False), e4), dim=1))
        d3 = self.dec3(torch.cat((F.interpolate(d4, size=e3.shape[-2:], mode='bilinear', align_corners=False), e3), dim=1))
        d2 = self.dec2(torch.cat((F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False), e2), dim=1))
        d1 = self.dec1(torch.cat((F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False), e1), dim=1))
        return self.out(d1)


def combined_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    intersection = (prob*target).sum(dim=(1,2,3))
    dice = (2*intersection+1) / (prob.sum(dim=(1,2,3))+target.sum(dim=(1,2,3))+1)
    return bce + (1-dice).mean()


def model_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
