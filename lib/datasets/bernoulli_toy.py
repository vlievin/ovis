import torch
from torch.distributions import Bernoulli
from torch.utils.data import Dataset


class BernoulliToyDataset(Dataset):
    def __init__(self, target, N=1, bs=1):
        super().__init__()
        self.N = N
        self.bs = bs
        self.target = target
        target = torch.Tensor(1, N)
        target.fill_(self.target)
        self.data = [target] * bs

    def __len__(self):
        return self.bs

    def __getitem__(self, item):
        return self.data[item]


def get_bernoulli_toy_datasets(target=0.499, N=1, bs=1):
    train_dset = BernoulliToyDataset(target=target, N=N, bs=bs)
    return train_dset, train_dset, train_dset
