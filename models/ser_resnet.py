"""
语音情感识别专用ResNet模型
针对Mel频谱图优化的ResNet变体
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEResBlock(nn.Module):
    """Squeeze-and-Excitation残差块"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

        # SE模块
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # SE注意力
        se_weight = self.se(out)
        out = out * se_weight

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class SERResNet(nn.Module):
    """带Squeeze-and-Excitation的ResNet用于语音情感识别"""
    def __init__(self, input_size=128, num_class=4, layers=[2, 2, 2, 2],
                 base_channels=64, reduction=16):
        super().__init__()
        self.feature_dim = input_size

        # 初始卷积
        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层
        self.layer1 = self._make_layer(base_channels, base_channels, layers[0], reduction=reduction)
        self.layer2 = self._make_layer(base_channels, base_channels * 2, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(base_channels * 2, base_channels * 4, layers[2], stride=2, reduction=reduction)
        self.layer4 = self._make_layer(base_channels * 4, base_channels * 8, layers[3], stride=2, reduction=reduction)

        # 全局池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(base_channels * 8, num_class)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride=1, reduction=16):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        layers = [SEResBlock(in_channels, out_channels, stride, downsample, reduction)]
        for _ in range(1, blocks):
            layers.append(SEResBlock(out_channels, out_channels, reduction=reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def SERResNet18(input_size=128, num_class=4, **kwargs):
    """SEResNet18 - 较轻量级模型"""
    return SERResNet(input_size=input_size, num_class=num_class, layers=[2, 2, 2, 2], **kwargs)


def SERResNet34(input_size=128, num_class=4, **kwargs):
    """SEResNet34 - 中等规模模型"""
    return SERResNet(input_size=input_size, num_class=num_class, layers=[3, 4, 6, 3], **kwargs)


def SERResNet50(input_size=128, num_class=4, **kwargs):
    """SEResNet50 - 大规模高精度模型"""
    return SERResNet(input_size=input_size, num_class=num_class, layers=[3, 4, 6, 3],
                    base_channels=64, **kwargs)


class CNNResNet(nn.Module):
    """CNN + ResNet混合模型用于语音情感识别"""
    def __init__(self, input_size=128, num_class=4, hidden_size=256):
        super().__init__()
        self.feature_dim = input_size

        # CNN特征提取
        self.cnn_layers = nn.Sequential(
            nn.Conv1d(input_size, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),

            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Dropout(0.2),
        )

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_class)
        )

    def forward(self, x):
        # x: (batch, freq, time)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # CNN特征提取
        x = self.cnn_layers(x)  # (batch, 256, time/4)

        # Transformer
        x = x.permute(2, 0, 1)  # (time/4, batch, 256)
        x = self.transformer(x)
        x = x.mean(dim=0)  # (batch, 256)

        # 分类
        x = self.classifier(x)
        return x