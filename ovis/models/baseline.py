import numpy as np
from torch import nn

from ..utils.utils import flatten


class Baseline(nn.Module):
    def __init__(self, xdim, nlayers, hdim, batch_norm=True, bias=True, center_x=True):
        super().__init__()
        self.center_x = center_x

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
        if self.center_x:
            x = 2. * x - 1.
        x = flatten(x)
        return self.layers(x).squeeze(1)
