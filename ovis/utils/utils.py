import operator
import re
import sys
from collections import defaultdict
from functools import reduce
from typing import Iterable, Dict, List, Optional

import torch


class ManualSeed():
    """A simple class to execute a statement with a manual random seed
    without breaking the randomness. Another random seed is sampled and set when exiting the `with` statement. Usage:
    ```python
    with ManualSeed(seed=42):
        # code to execute with the random seed 42
        print(torch.rand((1,)))
    #  code to run independently of the seed 42
    print(torch.rand((1,)))
    ```
    """

    def __init__(self, seed: Optional[int] = 1):
        """define the manual seed (setting seed=None allows skipping the setting of the manual seed)"""
        self.seed = seed
        self.new_seed = None

    def __enter__(self):
        """set the random seed `seed`"""
        if self.seed is not None:
            self.new_seed = int(torch.randint(1, sys.maxsize, (1,)).item())
            torch.manual_seed(self.seed)

    def __exit__(self, type, value, traceback):
        """set the random random seed `new_seed`"""
        if self.seed is not None:
            torch.manual_seed(self.new_seed)


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
    """A small helper class to model a dictionary of lists: {key : [*values]}"""

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