import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import VanillaBase, WCANet_Base
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


class PreActBlock(nn.Module):
    """Pre-activation version of the BasicBlock."""
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2])


########################################################
#    PreActResNet18
########################################################

class GeneratorPreActResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.rn = PreActResNet18()

    def forward(self, x):
        x = self.rn(x)
        return x


class VanillaPreActResNet18(VanillaBase):
    def __init__(self, D, C):
        super().__init__()
        self.gen = GeneratorPreActResNet18()
        self.fc1 = nn.Linear(512, D)
        self.proto = nn.Linear(D, C)

    def forward(self, x):
        x = self.gen(x)
        x = F.relu(self.fc1(x))
        x = self.proto(x)
        return x


class PreActResNet18_StochasticBaseDiagonal(nn.Module):
    """ Zero mean, trainable variance. """

    def __init__(self, D, disable_noise=False):
        super().__init__()
        self.gen = GeneratorPreActResNet18()
        self.fc1 = nn.Linear(512, D)
        self.sigma = nn.Parameter(torch.rand(D), requires_grad=(not disable_noise))
        self.disable_noise = disable_noise

    def forward(self, x):
        x = self.gen(x)
        x = F.relu(self.fc1(x))
        if not self.disable_noise:
            dist = Normal(0., F.softplus(self.sigma))
            x_sample = dist.rsample()
            x = x + x_sample
        return x


class PreActResNet18_StochasticBaseMultivariate(nn.Module):
    """ Trainable lower triangular matrix L, so Sigma=LL^T. """

    def __init__(self, D, disable_noise=False):
        super().__init__()
        self.gen = GeneratorPreActResNet18()
        self.fc1 = nn.Linear(512, D)
        self.mu = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.L = nn.Parameter(torch.rand(D, D).tril(), requires_grad=(not disable_noise))
        self.disable_noise = disable_noise
        self.D = D

        # if not self.disable_noise:
        #     print('Generating stored points inside init.')
        #     self.num_points = 2**7
        #     dist = MultivariateNormal(self.mu, scale_tril=self.L, validate_args=False)
        #     self.stored_points = dist.sample(sample_shape=torch.Size([self.num_points // self.D])).flatten()

    @property
    def sigma(self):
        return self.L @ self.L.T

    def forward(self, x):
        x = self.gen(x)
        x = F.relu(self.fc1(x))
        if not self.disable_noise:
            dist = MultivariateNormal(self.mu, scale_tril=self.L, validate_args=False)
            x_sample = dist.rsample()
            # x_sample = self.stored_points[torch.randint(high=self.num_points,
            #                                             size=(x.shape[0], self.D)
            #                                             )
            # ].cuda()
            x = x + x_sample
        return x


class WCANet_PreActResNet18(WCANet_Base):
    def __init__(self, D, C, variance_type, disable_noise=False):
        super().__init__()
        if variance_type == 'isotropic':
            self.base = PreActResNet18_StochasticBaseDiagonal(D, disable_noise=disable_noise)
        elif variance_type == 'anisotropic':
            self.base = PreActResNet18_StochasticBaseMultivariate(D, disable_noise=disable_noise)
        self.proto = nn.Linear(D, C)

    @property
    def sigma(self):
        return self.base.sigma

    def forward(self, x):
        x = self.base(x)
        x = self.proto(x)
        return x
