import torch
import torch.nn as nn
from .common import VanillaBase, WCANet_Base
import torch.nn.functional as f
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=(5, 5), stride=(1, 1))
        self.mp_1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv_2 = nn.Conv2d(in_channels=4, out_channels=10, kernel_size=(5, 5), stride=(1, 1))
        self.mp_2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.fc_1 = nn.Linear(in_features=4 * 4 * 10, out_features=100)
        # self.fc_2 = nn.Linear(in_features=100, out_features=10)

    def forward(self, x):
        x = f.relu(self.conv_1(x))
        x = self.mp_1(x)
        x = f.relu(self.conv_2(x))
        x = self.mp_2(x)
        x = x.reshape(-1, 4 * 4 * 10)
        x = f.relu(self.fc_1(x))
        # x = self.fc_2(x)
        return x

class VanillaLeNet(VanillaBase):
    def __init__(self, D, C):
        super().__init__()
        self.gen = LeNet()
        self.fc1 = nn.Linear(100, D)
        self.proto = nn.Linear(D, C)

    def forward(self, x):
        x = self.gen(x)
        x = self.fc1(x)
        x = self.proto(x)
        return x


class LeNet_StochasticBaseDiagonal(nn.Module):
    """ Zero mean, trainable variance. """

    def __init__(self, D, disable_noise=False):
        super().__init__()
        self.gen = LeNet()
        self.fc1 = nn.Linear(100, D)
        self.sigma = nn.Parameter(torch.rand(D), requires_grad=(not disable_noise))
        self.disable_noise = disable_noise

    def forward(self, x):
        x = self.gen(x)
        x = self.fc1(x)
        if not self.disable_noise:
            dist = Normal(0., f.softplus(self.sigma))
            x_sample = dist.rsample()
            x = x + x_sample
        return x


class LeNet_StochasticBaseMultivariate(nn.Module):
    """ Trainable lower triangular matrix L, so Sigma=LL^T. """

    def __init__(self, D, disable_noise=False):
        super().__init__()
        self.gen = LeNet()
        self.fc1 = nn.Linear(100, D)
        self.mu = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.L = nn.Parameter(torch.rand(D, D).tril(), requires_grad=(not disable_noise))
        self.disable_noise = disable_noise

    @property
    def sigma(self):
        return self.L @ self.L.T

    def forward(self, x):
        x = self.gen(x)
        x = self.fc1(x)
        if not self.disable_noise:
            dist = MultivariateNormal(self.mu, scale_tril=self.L, validate_args=False)
            x_sample = dist.rsample()
            x = x + x_sample
        return x


class WCANet_LeNet(WCANet_Base):
    def __init__(self, D, C, variance_type, disable_noise=False):
        super().__init__()
        if variance_type == 'isotropic':
            self.base = LeNet_StochasticBaseDiagonal(D, disable_noise=disable_noise)
        elif variance_type == 'anisotropic':
            self.base = LeNet_StochasticBaseMultivariate(D, disable_noise=disable_noise)
        self.proto = nn.Linear(D, C)

    @property
    def sigma(self):
        return self.base.sigma

    def forward(self, x):
        x = self.base(x)
        x = self.proto(x)
        return x
