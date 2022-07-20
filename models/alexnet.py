import torch
import torch.nn as nn
import torch.nn.functional as f
from .common import VanillaBase, WCANet_Base
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal


class AlexNet(nn.Module):
    def __init__(self, num_classes: int = 10, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(11, 11), stride=(4, 4), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=(5, 5), padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            # nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class GeneratorAlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rn = AlexNet()

    def forward(self, x):
        x = self.rn(x)
        return x


class VanillaAlexNet(VanillaBase):
    def __init__(self, D, C):
        super().__init__()
        self.gen = GeneratorAlexNet()
        self.fc1 = nn.Linear(4096, D)
        self.proto = nn.Linear(D, C)

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        x = self.proto(x)
        return x


class AlexNet_StochasticBaseDiagonal(nn.Module):
    """ Zero mean, trainable variance. """

    def __init__(self, D, disable_noise=False):
        super().__init__()
        self.gen = GeneratorAlexNet()
        self.fc1 = nn.Linear(4096, D)
        self.sigma = nn.Parameter(torch.rand(D), requires_grad=(not disable_noise))
        self.disable_noise = disable_noise

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        if not self.disable_noise:
            dist = Normal(0., f.softplus(self.sigma))
            x_sample = dist.rsample()
            x = x + x_sample
        return x


class AlexNet_StochasticBaseMultivariate(nn.Module):
    """ Trainable lower triangular matrix L, so Sigma=LL^T. """

    def __init__(self, D, disable_noise=False):
        super().__init__()
        self.gen = GeneratorAlexNet()
        self.fc1 = nn.Linear(4096, D)
        self.mu = nn.Parameter(torch.zeros(D), requires_grad=False)
        self.L = nn.Parameter(torch.rand(D, D).tril(), requires_grad=(not disable_noise))
        self.disable_noise = disable_noise

    @property
    def sigma(self):
        return self.L @ self.L.T

    def forward(self, x):
        x = self.gen(x)
        x = f.relu(self.fc1(x))
        if not self.disable_noise:
            dist = MultivariateNormal(self.mu, scale_tril=self.L, validate_args=False)
            x_sample = dist.rsample()
            x = x + x_sample
        return x

class WCANet_AlexNet(WCANet_Base):
    def __init__(self, D, C, variance_type, disable_noise=False):
        super().__init__()
        if variance_type == 'isotropic':
            self.base = AlexNet_StochasticBaseDiagonal(D, disable_noise=disable_noise)
        elif variance_type == 'anisotropic':
            self.base = AlexNet_StochasticBaseMultivariate(D, disable_noise=disable_noise)
        self.proto = nn.Linear(D, C)

    @property
    def sigma(self):
        return self.base.sigma

    def forward(self, x):
        x = self.base(x)
        x = self.proto(x)
        return x
