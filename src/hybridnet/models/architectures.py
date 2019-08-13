import sys
import math
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable, Function

from .utils import export, parameter_count
from ..misc.tools import DotDict

@export
def cifar_resnet26(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet32x32(ResidualBlock,
                        layers=[4, 4, 4],
                        channels=96,
                        downsample='shift_conv', **kwargs)
    return model

@export
def cifar_shakeshake26(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet32x32(ShakeShakeBlock,
                        layers=[4, 4, 4],
                        channels=96,
                        downsample='shift_conv', **kwargs)
    return model

@export
def stl10_resnet26_hybrid(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet96x96Hybrid(ResidualBlock,
                              ResidualDeconvBlock,
                                layers=[4, 4, 4],
                                channels=96,
                                downsample='basic', **kwargs)
    return model

@export
def cifar_shakeshake26_basic(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet32x32(ShakeShakeBlock,
                        layers=[4, 4, 4],
                        channels=96,
                        downsample='basic', **kwargs)
    return model

@export
def cifar_shakeshake26_hybrid(options, pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet32x32Hybrid(ShakeShakeBlock,
                              ShakeShakeDeconvBlock,
                              layers=[4, 4, 4],
                              channels=96,
                              downsample='basic',
                              options=options, **kwargs)
    return model

@export
def resnext152(pretrained=False, **kwargs):
    assert not pretrained
    model = ResNet224x224(BottleneckBlock,
                          layers=[3, 8, 36, 3],
                          channels=32 * 4,
                          groups=32,
                          downsample='basic', **kwargs)
    return model



class ResNet224x224(nn.Module):
    def __init__(self, block, layers, channels, groups=1, num_classes=1000, downsample='basic'):
        super().__init__()
        assert len(layers) == 4
        self.downsample_mode = downsample
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(
            block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, channels * 4, groups, layers[2], stride=2)
        self.layer4 = self._make_layer(
            block, channels * 8, groups, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(block.out_channels(
            channels * 8, groups), num_classes)
        self.fc2 = nn.Linear(block.out_channels(
            channels * 8, groups), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)

class Print(nn.Module):
    def __init__(self, title=""):
        super().__init__()
        self.title = title

    def forward(self, input):
        print(self.title + " - " + str(input.shape))
        return input


class ResNet32x32(nn.Module):
    def __init__(self, block, layers, channels, groups=1, num_classes=1000, downsample='basic', **kwargs):
        super().__init__()
        assert len(layers) == 3
        self.downsample_mode = downsample
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.layer1 = self._make_layer(block, channels, groups, layers[0])
        self.layer2 = self._make_layer(
            block, channels * 2, groups, layers[1], stride=2)
        self.layer3 = self._make_layer(
            block, channels * 4, groups, layers[2], stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc1 = nn.Linear(block.out_channels(
            channels * 4, groups), num_classes)
        self.fc2 = nn.Linear(block.out_channels(
            channels * 4, groups), num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, groups, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x), self.fc2(x)


class ResNet32x32Hybrid(nn.Module):
    def __init__(self, block, blockdec, layers, channels, groups=1, num_classes=1000, downsample='basic',
                 options=DotDict({"type": "baseline",
                                  "pool_sup": "stride", "pool_unsup": "stride",
                                  "unpool_sup": "stride", "unpool_unsup": "stride",
                                  "common_enc": False, "short_unsup": False})):
        super().__init__()
        assert len(layers) == 3
        self.downsample_mode = downsample
        self.options = options
        self.inplanes = 16
        valid_options = {"type": ["baseline", "ae", "hybrid"],
                          "pool_sup": ["stride"], "pool_unsup": ["stride"],
                          "unpool_sup": ["stride"], "unpool_unsup": ["stride"],
                          "common_enc": [True, False], "short_unsup": [True, False]}
        for option in self.options:
            assert self.options[option] in valid_options[option], "invalid option "+option+", value "+self.options[option]

        # discriminative encoder
        self.Ec_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.Ec_layer1 = self._make_layer(block, channels, groups, layers[0])
        self.Ec_layer2 = self._make_layer(block, channels * 2, groups, layers[1], stride=2)
        self.Ec_layer3 = self._make_layer(block, channels * 4, groups, layers[2], stride=2)
        self.Ec_avgpool = nn.AvgPool2d(8)
        self.Ec_fc1 = nn.Linear(block.out_channels(channels * 4, groups), num_classes)
        self.Ec_fc2 = nn.Linear(block.out_channels(channels * 4, groups), num_classes)

        # self.Ec_pool1 = nn.MaxPool2d(2, return_indices=True)
        # self.Ec_pool2 = nn.MaxPool2d(2, return_indices=True)
        # self.Ec_layer2 = self._make_layer(block, channels * 2, groups, layers[1], stride=1)
        # self.Ec_layer3 = self._make_layer(block, channels * 4, groups, layers[2], stride=1)
        # problem, the pooling should be inside layer2|3 but it's a sequential so I can't get the pool inds

        # discriminative decoder
        if self.options.type == "ae" or self.options.type == "hybrid":
            self.Dc_layer3 = self._make_reverse_layer(blockdec, channels * 2, groups, layers[2], stride=2)
            self.Dc_layer2 = self._make_reverse_layer(blockdec, channels, groups, layers[1], stride=2)
            self.Dc_layer1 = self._make_reverse_layer(blockdec, 16, groups, layers[0])
            self.Dc_conv1 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=True)

        # discriminative encoder & decoder
        if self.options.type == "hybrid":
            # discriminative encoder
            if not self.options.common_enc:
                self.Er_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.Er_layer1 = self._make_layer(block, channels, groups, layers[0])
            self.Er_layer2 = self._make_layer(block, channels * 2, groups, layers[1], stride=2)
            if not self.options.short_unsup:
                self.Er_layer3 = self._make_layer(block, channels * 4, groups, layers[2], stride=2)
                self.Dr_layer3 = self._make_reverse_layer(blockdec, channels * 2, groups, layers[2], stride=2)
            self.Dr_layer2 = self._make_reverse_layer(blockdec, channels, groups, layers[1], stride=2)
            self.Dr_layer1 = self._make_reverse_layer(blockdec, 16, groups, layers[0])
            self.Dr_conv1 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=True)

        # weights init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, input):
        if self.options.pool_sup == "max":
            assert False

        x = input
        x = hc0 = self.Ec_conv1(x)
        x = hc1 = self.Ec_layer1(x)
        x = hc2 = self.Ec_layer2(x)
        x = hc = self.Ec_layer3(x)
        x = self.Ec_avgpool(x)
        x = x.view(x.size(0), -1)

        y1, y2 = self.Ec_fc1(x), self.Ec_fc2(x)
        out = {"y_hat": y1, "y_cons": y2}

        if self.options.type == "ae" or self.options.type == "hybrid":
            x = hc
            x = xhatc2 = self.Dc_layer3(x)
            x = xhatc1 = self.Dc_layer2(x)
            x = xhatc0 = self.Dc_layer1(x)
            x = xhatc = self.Dc_conv1(x)

            out["x_hat"] = xhatc
            out["x_hat_c"] = xhatc
            #out["inters_c"] = [F.mse_loss(xhatc0, hc0.detach()), F.mse_loss(xhatc1, hc1.detach()), F.mse_loss(xhatc2, hc2.detach())]
            out["inters_c"] = F.mse_loss(xhatc0, hc0.detach()) + F.mse_loss(xhatc1, hc1.detach()) + F.mse_loss(xhatc2, hc2.detach())

        if self.options.type == "hybrid":
            if not self.options.common_enc:
                x = input
                x = hr0 = self.Er_conv1(x)
            else:
                x = hr0 = hc0
            x = hr1 = self.Er_layer1(x)
            x = hr2 = self.Er_layer2(x)
            if not self.options.short_unsup:
                x = hr = self.Er_layer3(x)
                x = xhatr2 = self.Dr_layer3(x)
            x = xhatr1 = self.Dr_layer2(x)
            x = xhatr0 = self.Dr_layer1(x)
            x = xhatr = self.Dr_conv1(x)

            out["x_hat"] += xhatr
            out["x_hat_r"] = xhatr
            #out["inters_r"] = [F.mse_loss(xhatr0, hr0.detach()), F.mse_loss(xhatr1, hr1.detach())]
            out["inters_r"] = F.mse_loss(xhatr0, hr0.detach()) + F.mse_loss(xhatr1, hr1.detach())
            if not self.options.short_unsup:
                out["inters_r"] += F.mse_loss(xhatr2, hr2.detach())

        return out

    # class ResNetLayer(nn.Module):
    #     def __init__(self, layers, pooling=None):
    #         super().__init__()
    #         self.layers = layers
    #         self.pooling = pooling
    #
    #     def forward(self, x):
    #         pos = None
    #         for i in range(len(self.layers)):
    #             x = self.layers[i](x)
    #             if i == 0 and self.pooling is not None:
    #                 x, pos = self.pooling(x)
    #         return x, pos
    #
    # class ResNetLayerDecode(nn.Module):
    #     def __init__(self, layers, unpooling=None):
    #         super().__init__()
    #         self.layers = layers
    #         self.unpooling = unpooling
    #
    #     def forward(self, x, pos=None):
    #         for i in range(len(self.layers)):
    #             if i == len(self.layers) - 1 and self.unpooling:
    #                 x = self.unpooling(x, pos)
    #             x = self.layers[i](x)
    #         return x

    def _make_layer(self, block, planes, groups, blocks, stride=1, pooling=False):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        # prepare pooling layer if needed
        # pooling_layer = None
        # if pooling:
        #     pooling_layer = torch.nn.MaxPool2d(stride, return_indices=True)
        #     stride = 1

        # create blocks
        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)
        # return self.ResNetLayer(layers, pooling_layer)

    def _make_reverse_layer(self, block, planes, groups, blocks, stride=1, unpooling=False):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    # NOTE not correct if input not div by stride
                    nn.ConvTranspose2d(self.inplanes, block.out_channels(planes, groups),
                                       kernel_size=1, stride=stride, bias=False, output_padding=stride - 1),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                assert False, "not implemented yet"
                # downsample = ShiftConvDownsample(in_channels=self.inplanes,
                #                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        # prepare unpooling layer if needed
        # unpooling_layer = None
        # if unpooling:
        #     unpooling_layer = torch.nn.MaxUnpool2d(stride)
        #     stride = 1

        layers = []
        for i in range(blocks - 1, 0, -1):
            layers.append(block(self.inplanes, self.inplanes, groups))
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)

        return nn.Sequential(*layers)
        # return self.ResNetLayerDecode(layers, unpooling_layer)


class ResNet96x96Hybrid(nn.Module):
    def __init__(self, block, blockdec, layers, channels, groups=1, num_classes=1000, downsample='basic',
                 options=DotDict({"type": "baseline",
                                  "pool_sup": "stride", "pool_unsup": "stride",
                                  "unpool_sup": "stride", "unpool_unsup": "stride",
                                  "common_enc": False, "short_unsup": False})):
        super().__init__()
        assert len(layers) == 3
        self.downsample_mode = downsample
        self.options = options
        self.inplanes = 32
        valid_options = {"type": ["baseline", "ae", "hybrid"],
                          "pool_sup": ["stride"], "pool_unsup": ["stride"],
                          "unpool_sup": ["stride"], "unpool_unsup": ["stride"],
                          "common_enc": [True, False], "short_unsup": [True, False]}
        for option in self.options:
            assert self.options[option] in valid_options[option], "invalid option "+option+", value "+self.options[option]

        # discriminative encoder
        self.relu = nn.ReLU(inplace=True)
        self.Ec_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False) # 96x96x16
        self.Ec_bn1 = nn.BatchNorm2d(16)
        self.Ec_pool1 = nn.MaxPool2d(2, stride=2, return_indices=True) # 48x48x16
        self.Ec_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 48x48x32
        self.Ec_bn2 = nn.BatchNorm2d(32)
        self.Ec_pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)  # 24x24x32

        self.Ec_layer1 = self._make_layer(block, channels, groups, layers[0]) # 24x24xchannels
        self.Ec_layer2 = self._make_layer(block, channels * 2, groups, layers[1], stride=2) # 12x12xchannels*2
        self.Ec_layer3 = self._make_layer(block, channels * 4, groups, layers[2], stride=2) # 6x6xchannels*2
        self.Ec_avgpool = nn.AvgPool2d(6)
        self.Ec_fc1 = nn.Linear(block.out_channels(channels * 4, groups), num_classes)
        self.Ec_fc2 = nn.Linear(block.out_channels(channels * 4, groups), num_classes)

        # self.Ec_pool1 = nn.MaxPool2d(2, return_indices=True)
        # self.Ec_pool2 = nn.MaxPool2d(2, return_indices=True)
        # self.Ec_layer2 = self._make_layer(block, channels * 2, groups, layers[1], stride=1)
        # self.Ec_layer3 = self._make_layer(block, channels * 4, groups, layers[2], stride=1)
        # problem, the pooling should be inside layer2|3 but it's a sequential so I can't get the pool inds

        # discriminative decoder
        if self.options.type == "ae" or self.options.type == "hybrid":
            self.Dc_layer3 = self._make_reverse_layer(blockdec, channels * 2, groups, layers[2], stride=2)
            self.Dc_layer2 = self._make_reverse_layer(blockdec, channels, groups, layers[1], stride=2)
            self.Dc_layer1 = self._make_reverse_layer(blockdec, 32, groups, layers[0])
            self.Dc_unpool2 = nn.MaxUnpool2d(2, stride=2)
            self.Dc_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.Dc_bn2 = nn.BatchNorm2d(16)
            self.Dc_unpool1 = nn.MaxUnpool2d(2, stride=2)
            self.Dc_conv1 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=True)

        # discriminative encoder & decoder
        if self.options.type == "hybrid":
            # discriminative encoder
            if not self.options.common_enc:
                self.Er_conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)  # 96x96x16
                self.Er_bn1 = nn.BatchNorm2d(16)
                self.Er_pool1 = nn.MaxPool2d(2, stride=2, return_indices=True)  # 48x48x16
                self.Er_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=False)  # 48x48x32
                self.Er_bn2 = nn.BatchNorm2d(32)
                self.Er_pool2 = nn.MaxPool2d(2, stride=2, return_indices=True)  # 24x24x32
            self.Er_layer1 = self._make_layer(block, channels, groups, layers[0])
            self.Er_layer2 = self._make_layer(block, channels * 2, groups, layers[1], stride=2)
            if not self.options.short_unsup:
                self.Er_layer3 = self._make_layer(block, channels * 4, groups, layers[2], stride=2)
                self.Dr_layer3 = self._make_reverse_layer(blockdec, channels * 2, groups, layers[2], stride=2)
            self.Dr_layer2 = self._make_reverse_layer(blockdec, channels, groups, layers[1], stride=2)
            self.Dr_layer1 = self._make_reverse_layer(blockdec, 32, groups, layers[0])
            self.Dr_unpool2 = nn.MaxUnpool2d(2, stride=2)
            self.Dr_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
            self.Dr_bn2 = nn.BatchNorm2d(16)
            self.Dr_unpool1 = nn.MaxUnpool2d(2, stride=2)
            self.Dr_conv1 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1, bias=True)

        # weights init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, input):
        if self.options.pool_sup == "max":
            assert False

        # TODO completer le forward

        x = input

        x = self.Ec_conv1(x)
        x = self.Ec_bn1(x)
        x = self.relu(x)
        x, inds1 = self.Ec_pool1(x)
        x = self.Ec_conv2(x)
        x = self.Ec_bn2(x)
        x = self.relu(x)
        x, inds2 = self.Ec_pool2(x)
        hc0 = x

        x = hc1 = self.Ec_layer1(x)
        x = hc2 = self.Ec_layer2(x)
        x = hc = self.Ec_layer3(x)
        x = self.Ec_avgpool(x)
        x = x.view(x.size(0), -1)

        y1, y2 = self.Ec_fc1(x), self.Ec_fc2(x)
        out = {"y_hat": y1, "y_cons": y2}

        if self.options.type == "ae" or self.options.type == "hybrid":
            x = hc
            x = xhatc2 = self.Dc_layer3(x)
            x = xhatc1 = self.Dc_layer2(x)
            x = xhatc0 = self.Dc_layer1(x)

            x = self.Dc_unpool2(x, inds2)
            x = self.Dc_conv2(x)
            x = self.Dc_bn2(x)
            x = self.relu(x)
            x = self.Dc_unpool1(x, inds1)
            x = xhatc = self.Dc_conv1(x)

            out["x_hat"] = xhatc
            out["x_hat_c"] = xhatc
            out["inters_c"] = [F.mse_loss(xhatc0, hc0.detach()), F.mse_loss(xhatc1, hc1.detach()), F.mse_loss(xhatc2, hc2.detach())]

        if self.options.type == "hybrid":
            if not self.options.common_enc:
                x = input
                x = self.Er_conv1(x)
                x = self.Er_bn1(x)
                x = self.relu(x)
                x, inds1r = self.Er_pool1(x)
                x = self.Er_conv2(x)
                x = self.Er_bn2(x)
                x = self.relu(x)
                x, inds2r = self.Er_pool2(x)
                hr0 = x
            else:
                x = hr0 = hc0
                inds2r = inds2
                inds1r = inds1
            x = hr1 = self.Er_layer1(x)
            x = hr2 = self.Er_layer2(x)
            if not self.options.short_unsup:
                x = hr = self.Er_layer3(x)
                x = xhatr2 = self.Dr_layer3(x)
            x = xhatr1 = self.Dr_layer2(x)
            x = xhatr0 = self.Dr_layer1(x)

            x = self.Dr_unpool2(x, inds2r)
            x = self.Dr_conv2(x)
            x = self.Dr_bn2(x)
            x = self.relu(x)
            x = self.Dr_unpool1(x, inds1r)
            x = xhatr = self.Dr_conv1(x)

            out["x_hat"] += xhatr
            out["x_hat_r"] = xhatr
            out["inters_r"] = [F.mse_loss(xhatr0, hr0.detach()), F.mse_loss(xhatr1, hr1.detach())]
            if not self.options.short_unsup:
                out["inters_r"].append(F.mse_loss(xhatr2, hr2.detach()))

        return out

    # class ResNetLayer(nn.Module):
    #     def __init__(self, layers, pooling=None):
    #         super().__init__()
    #         self.layers = layers
    #         self.pooling = pooling
    #
    #     def forward(self, x):
    #         pos = None
    #         for i in range(len(self.layers)):
    #             x = self.layers[i](x)
    #             if i == 0 and self.pooling is not None:
    #                 x, pos = self.pooling(x)
    #         return x, pos
    #
    # class ResNetLayerDecode(nn.Module):
    #     def __init__(self, layers, unpooling=None):
    #         super().__init__()
    #         self.layers = layers
    #         self.unpooling = unpooling
    #
    #     def forward(self, x, pos=None):
    #         for i in range(len(self.layers)):
    #             if i == len(self.layers) - 1 and self.unpooling:
    #                 x = self.unpooling(x, pos)
    #             x = self.layers[i](x)
    #         return x

    def _make_layer(self, block, planes, groups, blocks, stride=1, pooling=False):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, block.out_channels(planes, groups),
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                downsample = ShiftConvDownsample(in_channels=self.inplanes,
                                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        # prepare pooling layer if needed
        # pooling_layer = None
        # if pooling:
        #     pooling_layer = torch.nn.MaxPool2d(stride, return_indices=True)
        #     stride = 1

        # create blocks
        layers = []
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups))

        return nn.Sequential(*layers)
        # return self.ResNetLayer(layers, pooling_layer)

    def _make_reverse_layer(self, block, planes, groups, blocks, stride=1, unpooling=False):
        downsample = None
        if stride != 1 or self.inplanes != block.out_channels(planes, groups):
            if self.downsample_mode == 'basic' or stride == 1:
                downsample = nn.Sequential(
                    # NOTE not correct if input not div by stride
                    nn.ConvTranspose2d(self.inplanes, block.out_channels(planes, groups),
                                       kernel_size=1, stride=stride, bias=False, output_padding=stride - 1),
                    nn.BatchNorm2d(block.out_channels(planes, groups)),
                )
            elif self.downsample_mode == 'shift_conv':
                assert False, "not implemented yet"
                # downsample = ShiftConvDownsample(in_channels=self.inplanes,
                #                                 out_channels=block.out_channels(planes, groups))
            else:
                assert False

        # prepare unpooling layer if needed
        # unpooling_layer = None
        # if unpooling:
        #     unpooling_layer = torch.nn.MaxUnpool2d(stride)
        #     stride = 1

        layers = []
        for i in range(blocks - 1, 0, -1):
            layers.append(block(self.inplanes, self.inplanes, groups))
        layers.append(block(self.inplanes, planes, groups, stride, downsample))
        self.inplanes = block.out_channels(planes, groups)

        return nn.Sequential(*layers)
        # return self.ResNetLayerDecode(layers, unpooling_layer)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv3x3deconv(in_planes, out_planes, stride=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, output_padding=stride-1, bias=False)


class BottleneckBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        if groups > 1:
            return 2 * planes
        else:
            return 4 * planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv_a1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, groups=groups)
        self.bn_a2 = nn.BatchNorm2d(planes)
        self.conv_a3 = nn.Conv2d(planes, self.out_channels(planes, groups), kernel_size=1, bias=False)
        self.bn_a3 = nn.BatchNorm2d(self.out_channels(planes, groups))

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, residual = x, x

        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = self.relu(a)
        a = self.conv_a2(a)
        a = self.bn_a2(a)
        a = self.relu(a)
        a = self.conv_a3(a)
        a = self.bn_a3(a)

        if self.downsample is not None:
            residual = self.downsample(residual)

        return self.relu(residual + a)


class ShakeShakeBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = conv3x3(inplanes, planes, stride)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.conv_b1 = conv3x3(inplanes, planes, stride)
        self.bn_b1 = nn.BatchNorm2d(planes)
        self.conv_b2 = conv3x3(planes, planes)
        self.bn_b2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, b, residual = x, x, x


        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)

        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        b = F.relu(b, inplace=False)
        b = self.conv_b1(b)
        b = self.bn_b1(b)
        b = F.relu(b, inplace=True)
        b = self.conv_b2(b)
        b = self.bn_b2(b)

        ab = shake(a, b, training=self.training)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + ab


class ShakeShakeDeconvBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = conv3x3deconv(inplanes, inplanes)
        self.bn_a1 = nn.BatchNorm2d(inplanes)
        self.conv_a2 = conv3x3deconv(inplanes, planes, stride)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.conv_b1 = conv3x3deconv(inplanes, inplanes)
        self.bn_b1 = nn.BatchNorm2d(inplanes)
        self.conv_b2 = conv3x3deconv(inplanes, planes, stride)
        self.bn_b2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, b, residual = x, x, x


        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)

        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        b = F.relu(b, inplace=False)
        b = self.conv_b1(b)
        b = self.bn_b1(b)
        b = F.relu(b, inplace=True)
        b = self.conv_b2(b)
        b = self.bn_b2(b)

        ab = shake(a, b, training=self.training)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + ab


class ResidualBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = conv3x3(inplanes, planes, stride)
        self.bn_a1 = nn.BatchNorm2d(planes)
        self.conv_a2 = conv3x3(planes, planes)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, residual = x, x

        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)
        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + a


class ResidualDeconvBlock(nn.Module):
    @classmethod
    def out_channels(cls, planes, groups):
        assert groups == 1
        return planes

    def __init__(self, inplanes, planes, groups, stride=1, downsample=None):
        super().__init__()
        assert groups == 1
        self.conv_a1 = conv3x3deconv(inplanes, inplanes)
        self.bn_a1 = nn.BatchNorm2d(inplanes)
        self.conv_a2 = conv3x3deconv(inplanes, planes, stride)
        self.bn_a2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        a, residual = x, x

        a = F.relu(a, inplace=False)
        a = self.conv_a1(a)
        a = self.bn_a1(a)

        a = F.relu(a, inplace=True)
        a = self.conv_a2(a)
        a = self.bn_a2(a)

        if self.downsample is not None:
            residual = self.downsample(x)

        return residual + a


class Shake(Function):
    @classmethod
    def forward(cls, ctx, inp1, inp2, training):
        assert inp1.size() == inp2.size()
        gate_size = [inp1.size()[0], *itertools.repeat(1, inp1.dim() - 1)]
        gate = inp1.new(*gate_size)
        if training:
            gate.uniform_(0, 1)
        else:
            gate.fill_(0.5)
        return inp1 * gate + inp2 * (1. - gate)

    @classmethod
    def backward(cls, ctx, grad_output):
        grad_inp1 = grad_inp2 = grad_training = None
        gate_size = [grad_output.size()[0], *itertools.repeat(1, grad_output.dim() - 1)]
        gate = Variable(grad_output.data.new(*gate_size).uniform_(0, 1))
        if ctx.needs_input_grad[0]:
            grad_inp1 = grad_output * gate
        if ctx.needs_input_grad[1]:
            grad_inp2 = grad_output * (1 - gate)
        assert not ctx.needs_input_grad[2]
        return grad_inp1, grad_inp2, grad_training


def shake(inp1, inp2, training=False):
    return Shake.apply(inp1, inp2, training)


class ShiftConvDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_channels=2 * in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              groups=2)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = torch.cat((x[:, :, 0::2, 0::2],
                       x[:, :, 1::2, 1::2]), dim=1)
        x = self.relu(x)
        x = self.conv(x)
        x = self.bn(x)
        return x
