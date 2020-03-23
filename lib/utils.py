import operator
from collections import defaultdict
from functools import reduce
from typing import *
import re
import torch


def print_summary(x, key):
    """print the summary of a variable"""
    print(
        f">>> {key}: avg = {x.mean().item():.3f}, min = {x.min().item():.3f}, max = {x.max().item():.3f}, std = {x.mean().item():.3f}")

def parse_numbers(s):
    return [eval(n) for n in re.findall("\d+", s)]

def notqdm(iterable, *args, **kwargs):
    """
    replacement for tqdm that just passes back the iterable
    useful to silence `tqdm` in tests
    """
    return iterable


def prod(x: Iterable):
    """return the product of an Iterable"""
    if len(x):
        return reduce(operator.mul, x)
    else:
        return 0


def flatten(x):
    return x.view(x.size(0), -1)


def batch_reduce(x):
    return flatten(x).sum(1)


class DataCollector(defaultdict):
    def __init__(self):
        super().__init__(list)

    def extend(self, data: Dict[str, List[Optional[torch.Tensor]]]) -> None:
        """Append new data item"""
        for key, d in data.items():
            self[key] += d

    def sort(self) -> Dict[str, List[Optional[torch.Tensor]]]:
        """sort data and return"""
        for key, d in self.items():
            d = d[::-1]
            self[key] = [t for t in d if t is not None]

        return self


class FreeBits():
    """
    free bits: https://arxiv.org/abs/1606.04934
    Assumes a each of the dimension to be one group
    """

    def __init__(self, min_KL: float):
        self.min_KL = min_KL

    def __call__(self, kls: torch.Tensor) -> torch.Tensor:
        """
        Apply freebits over tensor. The freebits budget is distributed equally among dimensions.
        The returned freebits KL is equal to max(kl, freebits_per_dim, dim = >0)
        :param kls: KL of shape [batch size, *dimensions]
        :return:  freebits KL of shape [batch size, *dimensions]
        """

        # equally divide freebits budget over the dimensions
        dimensions = prod(kls.shape[1:])
        min_KL_per_dim = self.min_KL / dimensions if len(kls.shape) > 1 else self.min_KL
        min_KL_per_dim = min_KL_per_dim * torch.ones_like(kls)

        # apply freebits
        freebits_kl = torch.cat([kls.unsqueeze(-1), min_KL_per_dim.unsqueeze(-1)], -1)
        freebits_kl = torch.max(freebits_kl, dim=-1)[0]

        return freebits_kl
