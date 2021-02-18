import torch
import torch.nn as nn

"""
Single output version, for use w/ BCELoss, use_bias=False if bias is already a feature
"""
class LogRegModel(nn.Module):
    def __init__(self, input_dim, use_bias=False):
        super(LogRegModel, self).__init__()
        self.lin_1 = nn.Linear(input_dim, 1, bias=use_bias)

    def forward(self, x):
        return torch.sigmoid(self.lin_1(x)).squeeze()


class LinRegModel(nn.Module):
    def __init__(self, input_dim, use_bias=False):
        super(LinRegModel, self).__init__()
        self.lin_1 = nn.Linear(input_dim, 1, bias=use_bias)

    def forward(self, x):
        return self.lin_1(x).squeeze()
