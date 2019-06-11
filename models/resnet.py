import torch
import torch.nn as nn
import math


__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44',
           'resnet56', 'resnet110', 'resnet1202']


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class ShortcutA(nn.Module):

    def __init__(self, stride):
        super(ShortcutA, self).__init__()
        self.avgpool = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x):
        out = self.avgpool(x)
        out = torch.cat([out, torch.mul(out, 0)], 1)
        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


_depth = {
    'imagenet': {
        18: ([2, 2, 2, 2], BasicBlock),
        34: ([3, 4, 6, 3], BasicBlock),
        50: ([3, 4, 6, 3], Bottleneck),
        101: ([3, 4, 23, 3], Bottleneck),
        152: ([3, 8, 36, 3], Bottleneck),
    },
    'cifar': {
        20: ([3, 3, 3], BasicBlock),
        32: ([5, 5, 5], BasicBlock),
        44: ([7, 7, 7], BasicBlock),
        56: ([9, 9, 9], BasicBlock),
        110: ([18, 18, 18], BasicBlock),
        1202: ([200, 200, 200], BasicBlock)
    }
}


class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000):
        assert depth in _depth['imagenet'] or depth in _depth['cifar']
        self.depth = depth
        super(ResNet, self).__init__()
        if self.depth in _depth['imagenet']:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.layers = []
            layers, block = _depth['imagenet'][self.depth]
            self.layers.append(self._make_layer(block, self.inplanes, layers[0]))
            for idx in range(1, len(layers)):
                self.layers.append(self._make_layer(block, self.inplanes*(idx+1), layers[idx], stride=2))
            self.avgpool = nn.AvgPool2d(7)
            self.fc = nn.Linear(512 * block.expansion, num_classes)
        else:
            self.inplanes = 16
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1,
                                   bias=False)
            self.bn1 = nn.BatchNorm2d(self.inplanes)
            self.relu = nn.ReLU(inplace=True)
            self.maxpool = None
            self.layers = []
            layers, block = _depth['cifar'][self.depth]
            self.layers.append(self._make_layer(block, self.inplanes, layers[0]))
            for idx in range(1, len(layers)):
                self.layers.append(self._make_layer(block, self.inplanes*2, layers[idx], stride=2))
            self.avgpool = nn.AvgPool2d(8)
            self.fc = nn.Linear(64 * block.expansion, num_classes)
        self.layers = nn.Sequential(*self.layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if self.depth in _depth['cifar']:
            if self.inplanes != planes * block.expansion:
                downsample = ShortcutA(stride=stride)
        else:
            if stride != 1 or self.inplanes != planes * block.expansion:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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


def resnet20():
    return ResNet(depth=20, num_classes=10)

def resnet32():
    return ResNet(depth=32, num_classes=10)

def resnet44():
    return ResNet(depth=44, num_classes=10)

def resnet56():
    return ResNet(depth=56, num_classes=10)

def resnet110():
    return ResNet(depth=110, num_classes=10)

def resnet1202():
    return ResNet(depth=1202, num_classes=10)
