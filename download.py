import torch
from torchvision import datasets

d = datasets.CIFAR100('./data', download=True)
e = datasets.SVHN('./data', download=True)

