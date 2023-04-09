import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, k = 2, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.C1 = 4

        self.conv1 = nn.Conv2d(3, 2 ** self.C1, kernel_size=3, stride=1,
                            padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(2 ** self.C1)
        self.layer1 = self._make_layer(block, k * 2 ** self.C1, num_blocks[0], stride = 1)
        self.layer2 = self._make_layer(block, k * 2 ** (self.C1 + 1), num_blocks[1], stride = 2)
        self.layer3 = self._make_layer(block, k * 2 ** (self.C1 + 2), num_blocks[2], stride = 2)
        self.linear = nn.Linear(k * 2 ** (self.C1 + 2), num_classes)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8, 1, 0)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
