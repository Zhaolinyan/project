# resnet_encoder.py
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, resnet34, resnet50, resnet101, resnet152,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights, ResNet101_Weights, ResNet152_Weights
)

class ResnetEncoder(nn.Module):
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()
        self.num_ch_enc = None

        # 创建 ResNet 模型
        self.encoder = self.resnet_multiimage_input(num_layers, pretrained, num_input_images)
        self.num_ch_enc = [64, 64, 128, 256, 512]
        if num_layers > 34:
            self.num_ch_enc = [64, 256, 512, 1024, 2048]

    def forward(self, input_image):
        self.features = []
        x = input_image
        # resnet 的 forward
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        x = self.encoder.relu(x)
        self.features.append(x)
        x = self.encoder.maxpool(x)
        x = self.encoder.layer1(x)
        self.features.append(x)
        x = self.encoder.layer2(x)
        self.features.append(x)
        x = self.encoder.layer3(x)
        self.features.append(x)
        x = self.encoder.layer4(x)
        self.features.append(x)
        return self.features

    def resnet_multiimage_input(self, num_layers, pretrained, num_input_images):
        # 支持 ResNet 各层和权重
        resnets = {
            18: (resnet18, ResNet18_Weights.IMAGENET1K_V1),
            34: (resnet34, ResNet34_Weights.IMAGENET1K_V1),
            50: (resnet50, ResNet50_Weights.IMAGENET1K_V1),
            101: (resnet101, ResNet101_Weights.IMAGENET1K_V1),
            152: (resnet152, ResNet152_Weights.IMAGENET1K_V1)
        }

        if num_layers not in resnets:
            raise ValueError(f"Unsupported number of layers: {num_layers}")

        resnet_fn, weights_enum = resnets[num_layers]

        if pretrained:
            model = resnet_fn(weights=weights_enum)
        else:
            model = resnet_fn(weights=None)

        # 如果输入通道 > 3，需要修改第一层卷积
        if num_input_images > 1:
            conv1 = model.conv1
            model.conv1 = nn.Conv2d(
                3 * num_input_images,
                conv1.out_channels,
                kernel_size=conv1.kernel_size,
                stride=conv1.stride,
                padding=conv1.padding,
                bias=conv1.bias is not None
            )
        return model
