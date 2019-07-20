import torch.nn as nn
import torch.nn.functional as F


vgg16_config = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG16(nn.Module):
    def __init__(self, input_channel):
        super(VGG16, self).__init__()
        self.input_channel = input_channel
        self.conv_layers = self.make_layers(vgg16_config)
        self.classifier = nn.Sequential(nn.Linear(512, 10),
                                        nn.Softmax(1))

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512)  # the output size of conv layers should be 512*1
        x = self.classifier(x)

        return x

    def make_layers(self, cfg):
        layers = []
        last_channel = self.input_channel
        for c in cfg:
            if c == 'M':
                layers += [nn.MaxPool2d(2)]
            else:
                layers += [nn.Conv2d(last_channel, c, kernel_size=3, padding=1),
                           nn.BatchNorm2d(c),
                           nn.ReLU(inplace=True)]
                last_channel = c
        return nn.Sequential(*layers)


