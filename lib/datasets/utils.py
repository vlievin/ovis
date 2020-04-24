import random

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from .binmnist import get_binmnist_datasets
from .fashion import get_fashion_datasets
from .gaussian_toy import GaussianToyDataset
from .omniglot import get_omniglot_datasets
from .shapes import get_shapes_datasets
from .gmm import GaussianMixtureDataset

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
    elif "gmm" in opt.dataset:
        output = GaussianMixtureDataset(N=10000, C=opt.N), GaussianMixtureDataset(N=100, C=opt.N), GaussianMixtureDataset(N=100, C=opt.N)
    elif "binmnist" in opt.dataset:
        output = get_binmnist_datasets(opt.data_root, transform=transform)
    elif "omniglot" in opt.dataset:
        output = get_omniglot_datasets(opt.data_root, transform=transform, dynamic=True)
    elif "fashion" in opt.dataset:
        output = get_fashion_datasets(opt.data_root, transform=transform, binarize=True)
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
