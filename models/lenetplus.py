import torch
import torch.nn as nn
from common import VanillaBase, WCANet_Base
import torch.nn.functional as f
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal

class Generator(nn.Module):
    """ LeNets++ architecture from: "A Discriminative Feature Learning Approach for Deep Face Recognition"
        The variant is used by PCL, i.e. no max pooling and no padding.
    """

    def __init__(self, D):
        super(Generator, self).__init__()
        self.conv1 = self._make_conv_layer(1, 32, 5)
        self.conv2 = self._make_conv_layer(32, 32, 5)
        self.conv3 = self._make_conv_layer(32, 64, 5)
        self.conv4 = self._make_conv_layer(64, 64, 5)
        self.conv5 = self._make_conv_layer(64, 128, 5)
        self.conv6 = self._make_conv_layer(128, 128, 5)

    def _make_conv_layer(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size), nn.PReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = torch.flatten(x, start_dim=1)
        return x


class VanillaCNN(VanillaBase):
    def __init__(self, D, C):
        super().__init__()
        self.gen = Generator(D)
        self.fc1 = nn.Linear(2048, D)
        self.proto = nn.Linear(D, C)

    def forward(self, x):
        x = self.gen(x)
        x = self.fc1(x)
        x = self.proto(x)
        return x


class StochasticBaseDiagonal(nn.Module):
    """ Zero mean, trainable variance. """

    def __init__(self, D, disable_noise=False):
        super().__init__()
        self.gen = Generator(D)
        self.fc1 = nn.Linear(2048, D)
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


class StochasticBaseMultivariate(nn.Module):
    """ Trainable lower triangular matrix L, so Sigma=LL^T. """

    def __init__(self, D, disable_noise=False):
        super().__init__()
        self.gen = Generator(D)
        self.fc1 = nn.Linear(2048, D)
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


class WCANet_CNN(WCANet_Base):
    def __init__(self, D, C, variance_type, disable_noise=False):
        super().__init__()
        if variance_type == 'isotropic':
            self.base = StochasticBaseDiagonal(D, disable_noise=disable_noise)
        elif variance_type == 'anisotropic':
            self.base = StochasticBaseMultivariate(D, disable_noise=disable_noise)
        self.proto = nn.Linear(D, C)

    @property
    def sigma(self):
        return self.base.sigma

    def forward(self, x):
        x = self.base(x)
        x = self.proto(x)
        return x