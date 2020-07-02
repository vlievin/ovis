import operator
import re
import sys
from functools import reduce
from typing import Iterable, Optional

import torch
from booster.utils import logging_sep

BASE_ARGS_EXCEPTIONS = ['root', 'data_root', 'workers', 'silent', 'sequential_computation',
                        'epochs', 'nsteps', 'valid_bs', 'test_bs']


class Success():
    """handles the `success.txt` file generated for each run."""
    file = 'success.txt'
    success = f"Success."
    keyboard_interrupt = f"Failed. Interrupted (keyboard)."
    failure_base = "Failed."

    def failure(exception):
        return f"{Success.failure_base} Exception : \n{exception}\n\n{exception.__traceback__}"


class ManualSeed():
    """A simple class to execute a statement with a manual random seed without breaking the randomness.
    Another random seed is sampled and set when exiting the `with` statement. Usage:
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


def parse_numbers(s):
    return [eval(n) for n in re.findall("\d+", s)]


def notqdm(iterable, *args, **kwargs):
    """
    silent replacement for `tqdm`
    """
    return iterable


def prod(x: Iterable):
    """return the product of an Iterable"""
    if len(x):
        return reduce(operator.mul, x)
    else:
        return 0


def flatten(x):
    """return x.view(x.size(0), -1)"""
    return x.view(x.size(0), -1)


def batch_reduce(x):
    """return x.view(x.size(0), -1).sum(1)"""
    return flatten(x).sum(1)


class Header():
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        print(f"{logging_sep('=')}\n{self.message}\n{logging_sep('-')}")

    def __exit__(self, *args):
        print(f"{logging_sep('=')}")