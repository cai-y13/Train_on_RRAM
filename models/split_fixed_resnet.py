import torch
import torch.nn as nn
import math

from .rram import FixedModule
from .splitting import SplitConv, SplitFC


__all__ = ['split_fixed_resnet20', 'split_fixed_resnet32', 'split_fixed_resnet44',
           'split_fixed_resnet56', 'split_fixed_resnet110', 'split_fixed_resnet1202']


def conv3x3(in_planes, out_planes, stride=1, fixed_bits={'weight':8, 'input':8, 'output':8}):
    "3x3 convolution with padding"
    config = {'in_channel':in_planes, 'out_channel':out_planes, 'kernel_size':3, \
              'stride':stride, 'padding':1, 'bias':False, 'fixed_bits':fixed_bits}
    return SplitConv(config)

def conv1x1(in_planes, out_planes, stride=1, fixed_bits={'weight':8, 'input':8, 'output':8}):
    "1x1 convolution"
    config = {'in_channel':in_planes, 'out_channel':out_planes, 'kernel_size':1, \
              'stride':stride, 'padding':0, 'bias':False, 'fixed_bits':fixed_bits}
    return SplitConv(config)

class ShortcutA(nn.Module):

    def __init__(self, stride):
        super(ShortcutA, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.avgpool(x)
        out = torch.cat([out, torch.autograd.Variable(out.data.new().resize_as_(out.data).zero_(), requires_grad=False)], 1)
        return out


class FixedBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, fixed_bits={'weight':8, 'input':8, 'output':8}):
        super(FixedBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride, fixed_bits=fixed_bits)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = conv3x3(planes, planes, fixed_bits=fixed_bits)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


_depth = {
    20: [3, 3, 3],
    32: [5, 5, 5],
    44: [7, 7, 7],
    56: [9, 9, 9],
    110: [18, 18, 18],
    1202: [200, 200, 200]
}


class FixedResNet(nn.Module):

    def __init__(self, depth, fixed_bits={'weight':8, 'input':8, 'output':8}):
        assert depth in _depth
        self.depth = depth
        super(FixedResNet, self).__init__()
        self.inplanes = 16
        self.conv1 = conv3x3(3, self.inplanes, fixed_bits={'weight':fixed_bits['weight'], 'input':8, 'output':fixed_bits['output']})
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=False)
        self.maxpool = None
        self.layers = []
        depths = _depth[self.depth]
        self.layers.append(self._make_layer(self.inplanes, depths[0], fixed_bits=fixed_bits))
        for idx in range(1, len(depths)):
            self.layers.append(self._make_layer(self.inplanes*2, depths[idx], stride=2, fixed_bits=fixed_bits))
        self.avgpool = nn.AvgPool2d(8)
        self.fc = SplitFC(config={'in_neuron':64, 'out_neuron':10, 'fixed_bits':fixed_bits})
        self.layers = nn.Sequential(*self.layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, stride=1, fixed_bits={'weight':8, 'input':8, 'output':8}):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = ShortcutA(stride=stride)
            #downsample = nn.Sequential(
            #    conv1x1(self.inplanes, planes, stride),
            #    nn.BatchNorm2d(planes)
            #)
        layers = []
        layers.append(FixedBasicBlock(self.inplanes, planes, stride, downsample, fixed_bits=fixed_bits))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(FixedBasicBlock(self.inplanes, planes, fixed_bits=fixed_bits))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def split_fixed_resnet20(fixed_bits={'weight':8, 'input':8, 'output':8}):
    return FixedResNet(depth=20, fixed_bits=fixed_bits)

def split_fixed_resnet32(fixed_bits={'weight':8, 'input':6, 'output':6}):
    return FixedResNet(depth=32, fixed_bits=fixed_bits)

def split_fixed_resnet44(fixed_bits={'weight':8, 'input':6, 'output':6}):
    return FixedResNet(depth=44, fixed_bits=fixed_bits)

def split_fixed_resnet56(fixed_bits={'weight':8, 'input':6, 'output':6}):
    return FixedResNet(depth=56, fixed_bits=fixed_bits)

def split_fixed_resnet110(fixed_bits={'weight':8, 'input':6, 'output':6}):
    return FixedResNet(depth=110, fixed_bits=fixed_bits)

def split_fixed_resnet1202(fixed_bits={'weight':8, 'input':6, 'output':6}):
    return FixedResNet(depth=1202, fixed_bits=fixed_bits)
