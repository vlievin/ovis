from torch import nn

from .utils import prod, flatten


class Flatten(nn.Module):

    def forward(self, x):
        return flatten(x)


class MLP(nn.Module):

    def __init__(self, ninp, nhid, nout, nlayers=1, bias=True, act_in=False, normalization='batchnorm', dropout=0):
        super().__init__()
        Norm = {'batchnorm': nn.BatchNorm1d, 'layernorm': nn.LayerNorm, 'none': None, None: None}[normalization]
        layers = []
        if act_in:
            layers += [Norm(ninp), nn.Dropout(dropout), nn.ReLU()]
        h = ninp
        for i in range(nlayers):
            layers += [nn.Linear(h, nhid, bias=bias), nn.Dropout(dropout)]
            if Norm is not None:
                layers += [Norm(nhid)]
            layers += [nn.ReLU()]
            h = nhid
        layers += [Flatten(), nn.Linear(h, nout, bias=bias)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConvEncoder(nn.Module):

    def __init__(self, shp: tuple, nhid, nout, nlayers=1, bias=True, act_in=False, kernel_size=3, normalization='batchnorm'):
        super().__init__()
        Norm = {'batchnorm': nn.BatchNorm2d, 'layernorm': nn.LayerNorm, 'none': None, None: None}[normalization]
        layers = []

        if act_in:
            layers += [Norm(shp[0]), nn.ReLU()]

        for i in range(nlayers-1):
            layers += [nn.Conv2d(shp[0], nhid, kernel_size=kernel_size, bias=bias, padding=(kernel_size - 1) // 2, stride=2)]
            shp = [nhid, shp[1] // 2, shp[2] // 2]

            if Norm is not None:
                layers += [Norm(nhid)]
            layers += [nn.ReLU()]

        layers += [
            nn.Conv2d(shp[0], nout, kernel_size=kernel_size, bias=bias, padding=(kernel_size - 1) // 2, stride=2)]
        shp = [nhid, shp[1] // 2, shp[2] // 2]

        self.layers = nn.Sequential(*layers)
        self.output_shape = shp

    def forward(self, x):
        return self.layers(x)


class ConvDecoder(nn.Module):

    def __init__(self, shp: tuple, nhid, nout, nlayers=1, bias=True, act_in=False, kernel_size=3,
                 normalization='batchnorm'):
        super().__init__()
        Norm = {'batchnorm': nn.BatchNorm2d, 'layernorm': nn.LayerNorm, 'none': None, None: None}[normalization]
        layers = []

        if act_in:
            layers += [Norm(shp[0]), nn.ReLU()]

        for i in range(nlayers-1):
            layers += [
                nn.ConvTranspose2d(shp[0], nhid, kernel_size=kernel_size, bias=bias, padding=(kernel_size - 1) // 2, output_padding=1, stride=2)]
            shp = [nhid, shp[1] // 2, shp[2] // 2]

            if Norm is not None:
                layers += [Norm(nhid)]
            layers += [nn.ReLU()]

        layers += [
            nn.ConvTranspose2d(shp[0], nout, kernel_size=kernel_size, bias=bias, padding=(kernel_size - 1) // 2, output_padding=1, stride=2)]
        shp = [nhid, shp[1] // 2, shp[2] // 2]

        self.layers = nn.Sequential(*layers)
        self.output_shape = shp

    def forward(self, x):
        return self.layers(x)