import torch
import torch.nn as nn
import torch.nn.functional as f

class VanillaResNet18_folded(nn.Module):
    def __init__(self, D, C):
        super().__init__()
        self.gen = GeneratorResNet18_folded()
        self.fc1 = nn.Linear(512, D)
        self.proto = nn.Linear(D, C)

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        x = self.proto(x)
        return x

class GeneratorResNet18_folded(nn.Module):
    def __init__(self):
        super().__init__()
        self.rn = resnet18_folded()

    def forward(self, x):
        x = self.rn(x)
        return x

# Changed conv bias to true so the bias from BN can be folded here.

def conv1x1(in_planes, out_planes, stride=1, padding=0):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=True)


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=True)


class BasicBlock_folded(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride=stride)
        # self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride=1)
        # self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(inplanes, self.expansion * planes, stride=stride),
                # nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        # out = f.relu(self.bn1(self.conv1(x)))
        # out = self.bn2(self.conv2(out))
        out = f.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = f.relu(out)
        return out

"""
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes, stride=1)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(inplanes, planes, stride=stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, self.expansion * planes, stride=1)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                conv1x1(inplanes, self.expansion * planes, stride=stride), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = f.relu(self.bn1(self.conv1(x)))
        out = f.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = f.relu(out)
        return out
"""

class ResNet_folded(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.inplanes = 64

        self.conv1 = conv3x3(3, self.inplanes, stride=1)
        # self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.layer1 = self._make_layer(block,  64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplanes, planes, stride))
            self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = f.relu(self.bn1(self.conv1(x)))
        out = f.relu(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = f.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out


def resnet18_folded():
    return ResNet_folded(BasicBlock_folded, [2, 2, 2, 2])
