from torch import nn
import torch 
import torch.nn.functional as F
from torchvision.models import resnet50

def conv3x3(in_planes, out_planes, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=dilation, bias=False, dilation=dilation)


class ResidualConvUnit(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.rcu = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            conv3x3(channels, channels)
        )

    def forward(self, x):
        rcu_out = self.rcu(x)
        return rcu_out + x


class RCUx2(nn.Sequential):
    def __init__(self, channels):
        super().__init__(
            ResidualConvUnit(channels),
            ResidualConvUnit(channels))


class MultiResolutionFusion(nn.Module):
    def __init__(self, out_channels, channels):
        super().__init__()
        self.resolve0 = conv3x3(channels[0], out_channels)
        self.resolve1 = conv3x3(channels[1], out_channels)

    def forward(self, *xs):
        f0 = self.resolve0(xs[0])
        f1 = self.resolve1(xs[1])
        if f0.shape[-1] < f1.shape[-1]:
            f0 = F.interpolate(f0, size=f1.shape[-2:], mode='bilinear', align_corners=True)
        else:
            f1 = F.interpolate(f1, size=f0.shape[-2:], mode='bilinear', align_corners=True)
        out = f0 + f1
        return out


class ChainedResidualPool(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.block1 = nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                                    conv3x3(channels, channels))
        self.block2 = nn.Sequential(nn.MaxPool2d(kernel_size=5, stride=1, padding=2),
                                    conv3x3(channels, channels))

    def forward(self, x):
        x = self.relu(x)
        out = x
        x = self.block1(x)
        out = out + x
        x = self.block2(x)
        out = out + x
        return out


class RefineNetBlock(nn.Module):
    def __init__(self, in_channels, channels):
        super(RefineNetBlock, self).__init__()
        self.rcu = nn.ModuleList([])
        for channel in channels:
            self.rcu.append(RCUx2(channel))

        self.mrf = MultiResolutionFusion(in_channels, channels) if len(channels) != 1 else None
        self.crp = ChainedResidualPool(in_channels)
        self.output_conv = ResidualConvUnit(in_channels)

    def forward(self, *xs):
        rcu_outs = [rcu(x) for (rcu, x) in zip(self.rcu, xs)]
        mrf_out = rcu_outs[0] if self.mrf is None else self.mrf(*rcu_outs)
        crp_out = self.crp(mrf_out)
        out = self.output_conv(crp_out)
        return out


class RefineNet(nn.Module):
    def __init__(self, num_classes=2,n_classes=1):
        super(RefineNet, self).__init__()
        self.n_classes = n_classes
        resnet = list(resnet50(True).children())
        self.layer1 = nn.Sequential(*resnet[:-5])
        self.layer2 = nn.Sequential(*resnet[-5])
        self.layer3 = nn.Sequential(*resnet[-4])
        self.layer4 = nn.Sequential(*resnet[-3])

        self.layer1_reduce = conv3x3(256, 256)
        self.layer2_reduce = conv3x3(512, 256)
        self.layer3_reduce = conv3x3(1024, 256)
        self.layer4_reduce = conv3x3(2048, 512)

        self.refinenet4 = RefineNetBlock(512, (512,))
        self.refinenet3 = RefineNetBlock(256, (512, 256))
        self.refinenet2 = RefineNetBlock(256, (256, 256))
        self.refinenet1 = RefineNetBlock(256, (256, 256))

        self.output_conv = nn.Sequential(
            RCUx2(256),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, img):
        x = self.layer1(img)
        layer1_out = self.layer1_reduce(x)
        x = self.layer2(x)
        layer2_out = self.layer2_reduce(x)
        x = self.layer3(x)
        layer3_out = self.layer3_reduce(x)
        x = self.layer4(x)
        layer4_out = self.layer4_reduce(x)

        refine4_out = self.refinenet4(layer4_out)
        refine3_out = self.refinenet3(refine4_out, layer3_out)
        refine2_out = self.refinenet2(refine3_out, layer2_out)
        refine1_out = self.refinenet1(refine2_out, layer1_out)

        out = self.output_conv(refine1_out)
        out = F.interpolate(out, size=img.shape[-2:], mode='bilinear', align_corners=True)
        return out