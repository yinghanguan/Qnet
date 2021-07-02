from torch import nn
import torch
from torchvision.models import resnet50
class FCN8(nn.Module):
    def __init__(self, num_classes=1,n_classes=1):
        super(FCN8, self).__init__()
        resnet = list(resnet50(True).children())
        self.n_classes = n_classes
        self.features1 = nn.Sequential(*resnet[:-4])
        self.features2 = nn.Sequential(*resnet[-4])
        self.features3 = nn.Sequential(*resnet[-3])

        self.score_pool1 = nn.Conv2d(512, num_classes, kernel_size=1)
        self.score_pool2 = nn.Conv2d(1024, num_classes, kernel_size=1)
        self.score_pool3 = nn.Conv2d(2048, num_classes, kernel_size=1)

        self.upscore2 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore2_ = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=2, padding=1, bias=False)
        self.upscore8 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=8, padding=4, bias=False)

    def forward(self, x):
        pool1 = self.features1(x)
        pool2 = self.features2(pool1)
        pool3 = self.features3(pool2)

        score_pool3 = self.score_pool3(pool3)
        upscore_pool3 = self.upscore2(score_pool3)

        score_pool2 = self.score_pool2(0.01 * pool2)
        upscore_pool2 = self.upscore2_(score_pool2 + upscore_pool3)

        score_pool1 = self.score_pool1(0.0001 * pool1)
        upscore_pool1 = self.upscore8(score_pool1 + upscore_pool2)
        return upscore_pool1
