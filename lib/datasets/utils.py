import random

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .air import PyroMultiMNIST
from .bernoulli_toy import get_bernoulli_toy_datasets
from .binmnist import get_binmnist_datasets
from .fashion import get_fashion_datasets
from .gaussian_toy import GaussianToyDataset
from .gmm import GaussianMixtureDataset
from .omniglot import get_omniglot_datasets
from .shapes import get_shapes_datasets

_train_mini = 5000
_valid_mini = 500
_test_mini = 500


class MiniDataset(Dataset):
    def __init__(self, dset, k, seed):
        super().__init__()

        if seed is not None:
            random.seed(seed)

        self.index = list(random.choices(list(range(len(dset))), k=k))
        self.dset = dset

    def __getitem__(self, item):
        return self.dset[self.index[item]]

    def __len__(self):
        return len(self.index)


def get_datasets(opt):
    transform = ToTensor()

    if "shapes" in opt.dataset:
        output = get_shapes_datasets(transform=transform)
    elif "gaussian-toy" in opt.dataset:
        output = GaussianToyDataset(), GaussianToyDataset(), GaussianToyDataset()
    elif "gmm-toy" in opt.dataset:
        _train_dset = GaussianMixtureDataset(N=10000, C=opt.N)
        _valid_dset = GaussianMixtureDataset(N=100, C=opt.N)
        _test_dset = GaussianMixtureDataset(N=100, C=opt.N)
        output = _train_dset, _valid_dset, _test_dset
    elif 'bernoulli-toy' in opt.dataset:
        output = get_bernoulli_toy_datasets(target=opt.toy_target, N=opt.N)
    elif "binmnist" in opt.dataset:
        output = get_binmnist_datasets(opt.data_root, transform=transform)
    elif "omniglot" in opt.dataset:
        output = get_omniglot_datasets(opt.data_root, transform=transform, dynamic=True)
    elif "fashion" in opt.dataset:
        output = get_fashion_datasets(opt.data_root, transform=transform, binarize=True)
    elif "air" in opt.dataset:
        path = 'lib/datasets/raw_data/multi_mnist_pyro.npz'
        train_dset = PyroMultiMNIST(path, train=True)
        test_dset = PyroMultiMNIST(path, train=False)
        output = train_dset, test_dset, test_dset
    else:
        raise ValueError(f"Unknown data: {opt.dataset}")

    if opt.only_train_set:
        def use_only_training(dset_train, dset_valid, dset_test):
            return dset_train, dset_train, dset_train

        output = use_only_training(*output)

    if opt.mini:
        def wrapper(dset_train, dset_valid, dset_test):
            return MiniDataset(dset_train, _train_mini, opt.seed), \
                   MiniDataset(dset_valid, _valid_mini, opt.seed), \
                   MiniDataset(dset_test, _test_mini, opt.seed + 1)

        return wrapper(*output)
    else:
        return output
