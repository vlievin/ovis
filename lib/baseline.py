import numpy as np
from torch import nn

from .utils import *


class Baseline(nn.Module):
    def __init__(self, xdim, nlayers, hdim, batch_norm=True, bias=True):
        super().__init__()

        # encoder
        layers = []
        h = int(np.prod(xdim))
        for i in range(nlayers):
            layers += [nn.Linear(h, hdim, bias=bias)]
            if batch_norm:
                layers += [nn.BatchNorm1d(hdim)]
            layers += [nn.ELU()]
            h = hdim
        layers += [nn.Linear(h, 1, bias=bias)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = flatten(x)
        return self.layers(x).squeeze(1)
