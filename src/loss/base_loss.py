import torch
from torch import nn

class BaseLoss(nn.Module):
    def __init__(self, tag, coef=1.0):
        super().__init__()
        self.tag = tag
        self.coef = coef
