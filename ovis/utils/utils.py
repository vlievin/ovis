import operator
import re
import sys
from functools import reduce
from logging import Logger
from numbers import Number
from typing import *
from typing import Iterable, Optional

import torch
from booster.utils import logging_sep
from torch import Tensor

BASE_ARGS_EXCEPTIONS = ['root', 'data_root', 'exp', 'workers', 'silent', 'sequential_computation',
                        'epochs', 'nsteps', 'valid_bs', 'test_bs', 'load', 'rf']


def print_info(logger: Optional[Logger] = None, run_id: str = None, logdir: str = None, device: str = None):
    print_fn = print if logger is None else logger.info

    with Header(f"Info [{sys.argv[0]}]"):
        if run_id is not None:
            print_fn(f"Run id: {run_id}")
        if logdir is not None:
            print_fn(f"Logging directory: {logdir}")
        if device is not None:
            print_fn(f"Device: {device}")
        print_fn(f"Pytorch version: {torch.__version__}")
        print_fn(f"cuDNN version: {torch.backends.cudnn.version()}")
        print_fn(f"Python version: {sys.version.splitlines()[0]}")


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


def parse_numbers(s: str):
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


def flatten(x: Tensor):
    """return x.view(x.size(0), -1)"""
    return x.view(x.size(0), -1)


def batch_reduce(x: Tensor):
    """return x.view(x.size(0), -1).sum(1)"""
    return flatten(x).sum(1)


def batch_average(x: Tensor):
    """return x.view(x.size(0), -1).sum(1)"""
    return flatten(x).mean(1)


def cast_tensor(value: Number, ref_tensor: torch.Tensor, **kwargs: Any):
    return torch.tensor(value, device=ref_tensor.device, dtype=ref_tensor.dtype, **kwargs)


class Header():
    def __init__(self, message: str):
        self.message = message

    def __enter__(self):
        print(f"{logging_sep('=')}\n{self.message}\n{logging_sep('-')}")

    def __exit__(self, *args):
        print(f"{logging_sep('=')}")
