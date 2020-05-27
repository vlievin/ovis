import torch
from torch.distributions import Bernoulli
from torch.utils.data import Dataset


class BernoulliToyDataset(Dataset):
    def __init__(self, target, N=1, D=20, seed=42):
        super().__init__()
        self.N = N
        self.target = target
        target = torch.Tensor(1, N)
        target.fill_(self.target)
        self.data = [target]

    def __len__(self):
        return 1

    def __getitem__(self, item):
        return self.data[item]


def get_bernoulli_toy_datasets(target=0.499, N=1, **kwargs):
    train_dset = BernoulliToyDataset(target=target, N=N)
    return train_dset, train_dset, train_dset
