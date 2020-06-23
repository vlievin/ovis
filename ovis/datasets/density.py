import math

import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


def density(u_type):
    w1 = lambda z: torch.sin(2. * math.pi * z / 4.0)

    w2 = lambda z: 3. * torch.exp(-0.5 * ((z - 1) / 0.6) ** 2)

    w3 = lambda z: 3. * torch.sigmoid((z - 1.0) / 0.3)

    if u_type == "quarters":
        L = 6
        M = 6

        def U(z):
            """
            energy function: 4 quarters
            """
            z1, z2 = torch.chunk(z, chunks=2, dim=1)
            norm = torch.sqrt(z1 ** 2 + z2 ** 2)

            exp1 = torch.exp(-0.5 * ((z1 - 2) / 0.8) ** 2)
            exp2 = torch.exp(-0.5 * ((z1 + 2) / 0.8) ** 2)
            u = 0.5 * ((norm - 4) / 0.4) ** 2 - torch.log(exp1 + exp2)

            return u

    elif u_type == "wave":
        L = 4
        M = 4
        _u_min = 0.
        _u_max = 500.

        def U(z):
            """
            energy function: wave
            """
            z1, z2 = torch.chunk(z, chunks=2, dim=1)
            u = 0.5 * ((z2 - w1(z1)) / 0.4) ** 2

            # clipping
            u = u + torch.exp(2 * F.relu(torch.abs(z1) - M)) - 1.0

            return u

    elif u_type == "wave-split1":
        L = 4
        M = 4
        _u_min = 0.
        _u_max = 500.

        def U(z):
            """
            energy function: wave
            """
            z1, z2 = torch.chunk(z, chunks=2, dim=1)
            u = -torch.log(torch.exp(- 0.5 * ((z2 - w1(z1)) / 0.35) ** 2) + torch.exp(
                - 0.5 * ((z2 - w1(z1) + w2(z1)) / 0.35) ** 2) + 1e-12)

            u = u + torch.exp(2 * F.relu(torch.abs(z1) - M)) - 1.0

            return u


    elif u_type == "wave-split2":
        L = 4
        M = 4
        _u_min = 0.
        _u_max = 500.

        def U(z):
            """
            energy function: wave
            """
            z1, z2 = torch.chunk(z, chunks=2, dim=1)
            u = -torch.log(torch.exp(- 0.5 * ((z2 - w1(z1)) / 0.35) ** 2) + torch.exp(
                - 0.5 * ((z2 - w1(z1) + w3(z1)) / 0.35) ** 2) + 1e-12)

            u = u + torch.exp(2 * F.relu(torch.abs(z1) - M)) - 1.0

            return u

    elif u_type == "half-moon":

        L = 4
        M = 4

        def U(z):

            """half moon"""

            z1, z2 = torch.chunk(z, chunks=2, dim=1)

            u = 0.5 * ((torch.sqrt(z1 ** 2 + z2 ** 2) - 2) / 0.04) ** 2

            a = (z1 - 2.0) / 0.6

            b = (z1 + 2.0) / 0.6

            u = u + -torch.log(torch.exp(-0.5 * a ** 2) + torch.exp(-0.5 * b ** 2))

            return u
    else:
        raise ValueError("Unknown u_type")

    return U


class Density(Dataset):
    def __init__(self, u_type, size, static=False):
        super().__init__()

        U = density
