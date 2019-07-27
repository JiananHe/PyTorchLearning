import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DenseLayer, self).__init__()

        self.layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, mid_channels, 1, bias=True),

            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=True)
        )

    def forward(self, x):
        out = self.layers(x)
        return torch.cat([x, out], 1)


class DenseBlock(nn.Sequential):
    def __init__(self, input_channels, layers_num, growth_rate, bn_size):
        """
        make dense block composed of dense bottle layers
        :param input_channels: input channels in the first bottle layer
        :param layers: number of bottle layers
        :param growth_rate: the output channels of bottle layers
        :param bn_size: bn_size*growth_rate is the middle channels of bottle layers
        """
        super(DenseBlock, self).__init__()
        self.layers_num = layers_num
        self.growth_rate = growth_rate
        self.in_channels = input_channels
        self.mid_channels = growth_rate*bn_size
        self.out_channels = growth_rate

        layers = []
        in_channels = self.in_channels
        for i in range(self.layers_num):
            layers.append(DenseLayer(in_channels, self.mid_channels, self.out_channels))
            in_channels = in_channels + self.growth_rate
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class DenseNet(nn.Module):
    def __init__(self, input_channels, growth_rate=12, block_config=(3, 5, 3, 2),
                 num_init_features=64, bn_size=4):
        super(DenseNet, self).__init__()

        self.channels = num_init_features
        self.growth_rate = growth_rate
        self.bn_size = bn_size

        self.head = nn.Sequential(
            nn.Conv2d(input_channels, num_init_features, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True)
        )
        self.body = self.make_blocks(block_config)
        self.classifier = nn.Sequential(
            nn.Linear(self.channels, 10),  # number of class in CIFAR10
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))  # use global pool
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x

    def make_blocks(self, layers_cfg):
        """
        init_num_feature: input feature map channels in first layer of first dense block
        layers_cfg: layers number in every dense block
        """
        blocks = []
        input_channels = self.channels
        for i, layers_num in enumerate(layers_cfg):
            # dense block
            blocks.append(DenseBlock(input_channels, layers_num, self.growth_rate, self.bn_size))
            input_channels = input_channels + layers_num * self.growth_rate

            # transition layer
            if i != len(layers_cfg) - 1:
                blocks.append(nn.Sequential(
                    nn.BatchNorm2d(input_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(input_channels, input_channels//2, 1, bias=True),
                    nn.AvgPool2d(kernel_size=2, stride=2)
                ))
                input_channels = input_channels // 2

        # final bn
        blocks.append(nn.BatchNorm2d(input_channels))

        self.channels = input_channels
        return nn.Sequential(*blocks)