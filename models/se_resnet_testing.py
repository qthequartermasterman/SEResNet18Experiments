"""SE-ResNet in PyTorch
Based on preact_resnet.py
Author: Xu Ma.
Date: Apr/15/2019
Modified by Andrew Sansom on Jul/1/2019 to test how removal of each SEblock affects network performance
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['SEResNet18', 'SEResNet34', 'SEResNet50', 'SEResNet101', 'SEResNet152', 'SEResNet18WithMissingSEBlock',
           'SEResNet50WithMissingSEBlock', 'SEResNet18WithOneLayerOfSEBlock']


class Identity(nn.Module):
    # We will replace some SELayers with Identity Blocks
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEPreActBlock(nn.Module):
    """SE pre-activation of the BasicBlock"""
    expansion = 1 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(SEPreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.se = SELayer(planes,reduction)
        if stride !=1 or in_planes!=self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,self.expansion*planes,kernel_size=1,stride=stride,bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Add SE block
        out = self.se(out)
        out += shortcut
        return out


class SEPreActBootleneck(nn.Module):
    """Pre-activation version of the bottleneck module"""
    expansion = 4 # last_block_channel/first_block_channel

    def __init__(self,in_planes,planes,stride=1,reduction=16):
        super(SEPreActBootleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1=nn.Conv2d(in_planes,planes,kernel_size=1,bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,planes,kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,self.expansion*planes,kernel_size=1,bias=False)
        self.se = SELayer(self.expansion*planes, reduction)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self,'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Add SE block
        out = self.se(out)
        out +=shortcut
        return out


class NoSEPreActBlock(SEPreActBlock):
    expansion = 1 # last_block_channel/first_block_channel

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        # Do not add SE block
        # out = self.se(out)
        out += shortcut
        return out


class NoSEPreActBootleneck(SEPreActBootleneck):
    expansion = 4 # last_block_channel/first_block_channel

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out = self.conv3(F.relu(self.bn3(out)))
        # Do not add SE block
        # out = self.se(out)
        out += shortcut
        return out


class SEResNet(nn.Module):
    def __init__(self,block,num_blocks,num_classes=10,reduction=16):
        super(SEResNet, self).__init__()
        self.in_planes=64
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,stride=1,padding=1,bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,reduction=reduction)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,reduction=reduction)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,reduction=reduction)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,reduction=reduction)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    # block means SEPreActBlock or SEPreActBootleneck
    def _make_layer(self, block, planes, num_blocks, stride, reduction):
        strides = [stride] + [1]*(num_blocks-1) # like [1,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes,planes,stride,reduction))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        print(out.size())
        out = out.view(out.size(0), -1)
        print(out.size())
        out = self.linear(out)
        return out


class SEResNetWithMissingSEBlocks(nn.Module):
    def __init__(self, block, num_blocks, non_se_block, layer_number_to_remove, num_classes=10, reduction=16):
        super(SEResNetWithMissingSEBlocks, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.planes = [64, 128, 256, 512]  # Number of planes to put in each layer.
        self.strides = [1, 2, 2, 2]  # Strides to use in each layer
        self.layers = [self._make_layer(block, self.planes[i], num_blocks[i], stride=self.strides[i], reduction=reduction)
                       if i is not layer_number_to_remove
                       else self._make_layer(non_se_block, self.planes[i], num_blocks[i], stride=self.strides[i], reduction=reduction)
                       for i in range(0, 4)]
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.inner_layers = nn.Sequential(*self.layers)
        for i, module in enumerate(self.layers):
            self.add_module('Layer ' + str(i+1), module)

    # block means SEPreActBlock or SEPreActBootleneck
    def _make_layer(self,block, planes, num_blocks, stride, reduction):
        strides = [stride] + [1]*(num_blocks-1)  # like [1,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, reduction))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.inner_layers(out)
        # for layer in self.layers:
        #    out = layer(out)
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        # out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        print(out.size())
        out = out.view(out.size(0), -1)
        print(out.size())
        out = self.linear(out)
        return out


class SEResNetWithOneLayerOfSEBlocks(nn.Module):
    def __init__(self, block, num_blocks, non_se_block, layer_number_to_add_se_block, num_classes=10, reduction=16):
        super(SEResNetWithOneLayerOfSEBlocks, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.planes = [64, 128, 256, 512]  # Number of planes to put in each layer.
        self.strides = [1, 2, 2, 2]  # Strides to use in each layer
        self.layers = [self._make_layer(non_se_block, self.planes[i], num_blocks[i], stride=self.strides[i], reduction=reduction)
                       if i is not layer_number_to_add_se_block
                       else self._make_layer(block, self.planes[i], num_blocks[i], stride=self.strides[i], reduction=reduction)
                       for i in range(0, 4)]
        self.linear = nn.Linear(512 * block.expansion, num_classes)
        self.inner_layers = nn.Sequential(*self.layers)
        for i, module in enumerate(self.layers):
            self.add_module('Layer ' + str(i+1), module)

    # block means SEPreActBlock or SEPreActBootleneck
    def _make_layer(self,block, planes, num_blocks, stride, reduction):
        strides = [stride] + [1]*(num_blocks-1)  # like [1,1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, reduction))
            self.in_planes = planes*block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        # for layer in self.layers:
        #    out = layer(out)
        # out = self.layer0(out)
        # out = self.layer1(out)
        # out = self.layer2(out)
        # out = self.layer3(out)
        out = inner_layers(out)
        out = F.avg_pool2d(out, 4)
        print(out.size())
        out = out.view(out.size(0), -1)
        print(out.size())
        out = self.linear(out)
        return out


def SEResNet18(num_classes=10):
    return SEResNet(SEPreActBlock, [2,2,2,2],num_classes)


def SEResNet18WithMissingSEBlock(missing_layer, num_classes=10):
    # missing_layer is an integer between 0 and 3 (inclusive)
    return SEResNetWithMissingSEBlocks(SEPreActBlock, [2, 2, 2, 2], NoSEPreActBlock, missing_layer, num_classes=num_classes)


def SEResNet18WithOneLayerOfSEBlock(layer_with_se_blocks, num_classes=10):
    # missing_layer is an integer between 0 and 3 (inclusive)
    return SEResNetWithOneLayerOfSEBlocks(SEPreActBlock, [2, 2, 2, 2], NoSEPreActBlock, layer_with_se_blocks, num_classes=num_classes)


def SEResNet34(num_classes=10):
    return SEResNet(SEPreActBlock, [3,4,6,3],num_classes)


def SEResNet50(num_classes=10):
    return SEResNet(SEPreActBootleneck, [3,4,6,3],num_classes)


def SEResNet50WithMissingSEBlock(missing_layer, num_classes=10):
    # missing_layer is an integer between 0 and 3 (inclusive)
    return SEResNetWithMissingSEBlocks(SEPreActBootleneck, [2, 2, 2, 2], NoSEPreActBootleneck, missing_layer, num_classes=num_classes)


def SEResNet101(num_classes=10):
    return SEResNet(SEPreActBootleneck, [3,4,23,3],num_classes)


def SEResNet152(num_classes=10):
    return SEResNet(SEPreActBootleneck, [3,8,36,3],num_classes)


def test(n):
    net = SEResNet18WithMissingSEBlock(n)
    y = net((torch.randn(1,3,32,32)))
    print(y.size())


def test2():
    net = SEResNet18()
    y = net((torch.randn(1,3,32,32)))
    print(y.size())

# test()