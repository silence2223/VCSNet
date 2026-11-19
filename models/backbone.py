"""VGG-16-BN backbone and feature pyramid network."""
import torch
from torch import nn


class _VGGInitialization(nn.Module):
    """Initialize VGG layers in the reference construction order."""
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 4096), nn.ReLU(True), nn.Dropout(),
            nn.Linear(4096, 1000),
        )

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


class VGGBackbone(nn.Module):
    """Extract four feature levels at strides 2, 4, 8 and 16."""
    def __init__(self):
        super().__init__()
        layout = [64, 64, "M", 128, 128, "M", 256, 256, 256,
                  "M", 512, 512, 512, "M", 512, 512, 512, "M"]
        layers = []
        in_channels = 3
        for channels in layout:
            if channels == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
                layers.extend([conv, nn.BatchNorm2d(channels), nn.ReLU(inplace=True)])
                in_channels = channels
        vgg = _VGGInitialization(nn.Sequential(*layers))
        features = list(vgg.features.children())
        self.body1 = nn.Sequential(*features[:13])
        self.body2 = nn.Sequential(*features[13:23])
        self.body3 = nn.Sequential(*features[23:33])
        self.body4 = nn.Sequential(*features[33:43])
        self.num_channels = 256

    def forward(self, images):
        outputs = []
        for layer in (self.body1, self.body2, self.body3, self.body4):
            images = layer(images)
            outputs.append(images)
        return outputs


class FPN(nn.Module):
    """Fuse three backbone levels; the second output is the stride-8 P4 feature."""
    def __init__(self):
        super().__init__()

        self.P5_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P5_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.P4_1 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.P3_1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.P3_upsampled = nn.Upsample(scale_factor=2, mode="nearest")
        self.P3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, inputs):
        c3, c4, c5 = inputs
        p5 = self.P5_1(c5)
        p5_up = self.P5_upsampled(p5)
        p5 = self.P5_2(p5)
        p4 = p5_up + self.P4_1(c4)
        p4_up = self.P4_upsampled(p4)
        p4 = self.P4_2(p4)
        p3 = self.P3_2(self.P3_1(c3) + p4_up)
        return [p3, p4, p5]
