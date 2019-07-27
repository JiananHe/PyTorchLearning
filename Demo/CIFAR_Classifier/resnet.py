import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, inChannel, midChannel, outChannel, stride, isShortCut=True):
        super(ResidualBlock, self).__init__()
        self.isShortCut = isShortCut
        self.relu = nn.ReLU(inplace=True)
        self.residual = nn.Sequential(
            nn.Conv2d(inChannel, midChannel, 1, bias=False),
            nn.BatchNorm2d(midChannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(midChannel, midChannel, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(midChannel),
            nn.ReLU(inplace=True),

            nn.Conv2d(midChannel, outChannel, 1, bias=False),
            nn.BatchNorm2d(outChannel)
        )

        if not isShortCut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inChannel, outChannel, 1, stride=stride, bias=False),  # down sample
                nn.BatchNorm2d(outChannel)
            )

    def forward(self, x):
        out = self.residual(x)
        if self.isShortCut:
            identity = x
        else:
            identity = self.shortcut(x)
            out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(self, input_channels):
        super(ResNet, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.body = self.make_layers([2, 3, 3, 2])
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.Linear(512, 10),
            nn.Softmax(1)
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = nn.AvgPool2d(4)(x)  # the size of conv layer output should be 4*4*2048
        x = x.view(-1, 2048)
        x = self.classifier(x)

        return x

    def make_layers(self, cfg):
        """
        cfg: res block numbers
        (64 --> 256 conv s=1) --> (64 --> 256 sc) * 2 -->
        (128 --> 512 conv s=2) --> (128 --> 512 sc) * 3 -->
        (256 --> 1024 conv s=2) --> (256 --> 1024 sc) * 3-->
        (512 --> 2048 conv s=2) --> (512 --> 2048 sc) *2
        """
        layers = []
        for index, num in enumerate(cfg):
            if index == 0:
                layers.append(ResidualBlock(64, 64, 256, stride=1, isShortCut=False))
            else:
                layers.append(ResidualBlock(256 * 2 ** (index - 1), 64 * 2 ** index, 256 * 2 ** index, stride=2,
                                            isShortCut=False))

            for i in range(num):
                layers.append(
                    ResidualBlock(256 * 2 ** index, 64 * 2 ** index, 256 * 2 ** index, stride=1, isShortCut=True))

        return nn.Sequential(*layers)
