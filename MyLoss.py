import torch
import torch.nn as nn
import torch.nn.functional as F

class MSELoss(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, res, ref):
        return F.mse_loss(res, ref)

class BCELoss(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, res, ref):
        return F.binary_cross_entropy(res, ref)