import torch
import torch.nn as nn


class DSCLayer(nn.Module):
    def __init__(self, input_dim, out_dim, kernel_size=4):
        super().__init__()
        self.Resnet = nn.Sequential(
            Conv1d(input_dim, input_dim, kernel_size=kernel_size, groups=input_dim, padding='same'),
            nn.GELU()
        )
        self.Conv_1x1 = nn.Sequential(
            Conv1d(input_dim, input_dim, kernel_size=1),
            nn.GELU()
        )
        self.bn = nn.BatchNorm1d(input_dim)

    def forward(self, x):
        x = self.Conv_1x1(x + self.Resnet(x))
        b, n = x.shape[:2]
        x = x.flatten(0, 1)
        x = self.bn(x)
        x = x.reshape([b, n, -1, x.shape[-1]])
        return x

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels,groups=1, kernel_size=1,padding='same', dropout=0.1, actv=True):
        super(Conv1d, self).__init__()
        self.convolution = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, groups=groups, padding=padding)
        self.dropout = nn.Dropout(dropout)
        self.actv = actv

    def forward(self, x):
        if len(x.shape) == 4:
            b, n = x.shape[:2]
            x = x.flatten(0, 1)
            y = self.convolution(x)
            y = y.reshape([b, n, -1, y.shape[-1]])
        else:
            y = self.convolution(x)
        if self.actv:
            y =  torch.relu(self.dropout(y))
        return y