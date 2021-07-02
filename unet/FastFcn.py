import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
class ConvBnRelu(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(ConvBnRelu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__()
        self.conv = ConvBnRelu(in_channels, out_channels, 3, padding=dilation, dilation=dilation)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            ConvBnRelu(in_channels, out_channels, 1)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=True)


class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels=256, dilation_rates=(12, 24, 36)):
        super(ASPP, self).__init__()
        self.convs = nn.ModuleList([
            ConvBnRelu(in_channels, out_channels, 1),
            ASPPConv(in_channels, out_channels, dilation_rates[0]),
            ASPPConv(in_channels, out_channels, dilation_rates[1]),
            ASPPConv(in_channels, out_channels, dilation_rates[2]),
            ASPPPooling(in_channels, out_channels)
        ])

        self.project = ConvBnRelu(5 * out_channels, out_channels, 1)

    def forward(self, x):
        res = torch.cat([conv(x) for conv in self.convs], dim=1)
        return self.project(res)


class Head(nn.Module):
    def __init__(self, num_classes):
        super(Head, self).__init__()
        self.ASPP = ASPP(2048, dilation_rates=(6, 12, 18))

        self.reduce = ConvBnRelu(256, 48, 1)
        self.fuse_conv = nn.Sequential(
            ConvBnRelu(256 + 48, 256, 3, padding=1),
            ConvBnRelu(256, 256, 3, padding=1),
            nn.Dropout2d(0.1)
        )
        self.classifier = nn.Conv2d(256, num_classes, 1)

    def forward(self, low_features, high_features):
        high_features = self.ASPP(high_features)
        high_features = F.interpolate(high_features, size=low_features.shape[-2:], mode='bilinear', align_corners=True)

        low_features = self.reduce(low_features)
        f = torch.cat((high_features, low_features), dim=1)
        f = self.fuse_conv(f)

        predition = self.classifier(f)
        return predition


class JPU(nn.Module):
    def __init__(self):
        super(JPU, self).__init__()
        self.conv2 = ConvBnRelu(512, 512, 3, padding=1)
        self.conv3 = ConvBnRelu(1024, 512, 3, padding=1)
        self.conv4 = ConvBnRelu(2048, 512, 3, padding=1)

        self.dilated_convs = nn.ModuleList([
            ConvBnRelu(512 * 3, 512, 3, padding=1, dilation=1),
            ConvBnRelu(512 * 3, 512, 3, padding=2, dilation=2),
            ConvBnRelu(512 * 3, 512, 3, padding=4, dilation=4),
            ConvBnRelu(512 * 3, 512, 3, padding=8, dilation=8)
        ])

    def forward(self, f2, f3, f4):
        f2 = self.conv2(f2)
        f3 = self.conv3(f3)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=True)
        f4 = self.conv4(f4)
        f4 = F.interpolate(f4, size=f2.shape[-2:], mode='bilinear', align_corners=True)
        feat = torch.cat([f4, f3, f2], dim=1)
        dilat_out = torch.cat([conv(feat) for conv in self.dilated_convs], dim=1)
        return dilat_out


class FastFCN(nn.Module):
    def __init__(self, num_classes=2,n_classes=1):
        super(FastFCN, self).__init__()
        self.n_classes = n_classes
        resnet = list(resnet50(True, replace_stride_with_dilation=[False, False, True]).children())
        self.feature1 = nn.Sequential(*resnet[:5])
        self.feature2 = nn.Sequential(*resnet[5])
        self.feature3 = nn.Sequential(*resnet[6])
        self.feature4 = nn.Sequential(*resnet[7])

        self.jpu = JPU()
        self.head = Head(num_classes)

    def forward(self, x):
        f1 = self.feature1(x)
        f2 = self.feature2(f1)
        f3 = self.feature3(f2)
        f4 = self.feature4(f3)

        jpu_out = self.jpu(f2, f3, f4)
        pred = self.head(f1, jpu_out)

        output = F.interpolate(pred, size=x.shape[-2:], mode='bilinear', align_corners=True)
        return output