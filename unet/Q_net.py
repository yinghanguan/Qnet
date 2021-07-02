import torch
from torch import nn
from .Q_net_parts import *


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result
class RepVGGBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=2, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(RepVGGBlock, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True, padding_mode=padding_mode)

        else:
            self.rbr_identity = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.cbam = CBAM(out_channels)
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, padding=padding_11, groups=groups)
            print('RepVGG Block, identity = ', self.rbr_identity)
    def forward(self, inputs):
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.rbr_reparam(inputs))

        if self.rbr_identity is None:
            id_out = 0
            return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs)+id_out)
        else:
            id_out = self.rbr_identity(inputs)
            # print(id_out.shape)
            return self.nonlinearity(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + self.cbam(id_out))

class Q_Net(nn.Module):
    def __init__(self, in_channel, out_channel, width_multiplier, num_blocks, num_layers,n_classes=1):
        super(Q_Net, self).__init__()
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.num_layers = num_layers]
        self.in_planes = min(64, int(64 * width_multiplier[0]))
        self.rdb0 = RDB(in_channel,in_channel,num_layers)
        self.n_classes = n_classes
        self.stage0 = RepVGGBlock(in_channels=3, out_channels=self.in_planes, kernel_size=3, stride=2, padding=1,
                                  deploy=False)
        self.cur_layer_idx = 1
        self.rdb1 = RDB(self.in_planes,self.in_planes,num_layers)
        self.stage1 = self._make_stage(int(64 * width_multiplier[1]), num_blocks[0], stride=2)
        self.rdb2 = RDB(int(64 * width_multiplier[1]),int(64 * width_multiplier[1]),num_layers)
        self.stage2 = self._make_stage(int(128 * width_multiplier[2]), num_blocks[1], stride=2)
        self.rdb3 = RDB(int(128 * width_multiplier[2]),int(128 * width_multiplier[2]),num_layers)
        self.stage3 = self._make_stage(int(256 * width_multiplier[3]), num_blocks[2], stride=2)
        self.rdb4 = RDB(int(256 * width_multiplier[3]),int(256 * width_multiplier[3]),num_layers)
        self.stage4 = self._make_stage(int(512 * width_multiplier[4]), num_blocks[3], stride=1)
        self.rdb5 = RDB(int(512 * width_multiplier[4]),int(512 * width_multiplier[4]),num_layers)
        self.stage5 = self._make_stage(int(2048 * width_multiplier[5]), num_blocks[4], stride=2)
        self.double33 = DoubleConv(int(2048 * width_multiplier[5]),int(2048 * width_multiplier[5]))
        self.up0 = Up(1024,512)
        self.cbam0 = CBAM(512)
        self.up1 = Up(256,128)
        self.cbam1 = CBAM(128)
        self.up2 = Up(64,32)
        self.cbam2 = CBAM(32)
        self.up3 = Up(16,8)
        self.cbam3 = CBAM(8)
        self.pixel = nn.PixelShuffle(2)
        self.out = OutConv(2,1)
    def forward(self, x):
      x1 = self.rdb0(x)
      #print('x1',x1.shape)
      x2 = self.stage0(x1)
      #print('x2',x2.shape)
      x3 = self.rdb1(x2)
      #print('x3',x3.shape)
      x4 = self.stage1(x3)
      #print('x4',x4.shape)
      x5 = self.rdb2(x4)
      #print('x5',x5.shape)
      x6 = self.stage2(x5)
      #print('x6',x6.shape)
      x7 = self.rdb3(x6)
      #print('x7',x7.shape)
      x8 = self.stage3(x7)
      #print('x8',x8.shape)
      x9 = self.rdb4(x8)
      #print('x9',x9.shape)
      x10 = self.stage4(x9)
      #print('x10',x10.shape)
      x11 = self.rdb5(x10)
      #print('x11',x11.shape)
      x12 = self.stage5(x11)
      #print('x12',x12.shape)
      x13 = self.double33(x12)
      #print('x13',x13.shape)
      x14 = self.up0(x13,self.cbam0(x9))
      #print('x14',x14.shape)
      x15 = self.up1(x14,self.cbam1(x7))
      #print('x15',x15.shape)
      x16 = self.up2(x15,self.cbam2(x5))
      #print('x16',x16.shape)
      x17 = self.up3(x16,self.cbam3(x3))
      #print('x17',x17.shape)
      x18 = self.pixel(x17)
      outc = self.out(x18)
      #print(outc.shape)
      return outc


    def _make_stage(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        blocks = []
        for stride in strides:
            blocks.append(RepVGGBlock(in_channels=self.in_planes, out_channels=planes, kernel_size=3,
                                      stride=stride, padding=1, groups=1, deploy=False))
            self.in_planes = planes
            self.cur_layer_idx += 1
        return nn.Sequential(*blocks)