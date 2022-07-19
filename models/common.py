import torch
from torch import nn


class VanillaBase(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))


class WCANet_Base(nn.Module):
    def __init__(self):
        super().__init__()

    def save(self, filename):
        torch.save(self.state_dict(), filename + ".pt")

    def load(self, filename):
        self.load_state_dict(torch.load(filename + ".pt"))

    def freeze_model_params(self):
        """ Freezes all model parameters except for the noise layer (used for ablation). """
        for param in self.base.gen.parameters():
            param.requires_grad = False
        for param in self.base.fc1.parameters():
            param.requires_grad = False
        for param in self.proto.parameters():
            param.requires_grad = False

    def unfreeze_model_params(self):
        """ Reverses `freeze_model_params`. """
        for param in self.base.gen.parameters():
            param.requires_grad = True
        for param in self.base.fc1.parameters():
            param.requires_grad = True
        for param in self.proto.parameters():
            param.requires_grad = True