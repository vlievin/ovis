import torch
from torch.distributions import Normal
from torch.utils.data import Dataset


class GaussianToyDataset(Dataset):
    """
    Gaussian Toy dataset as described in
    `Tighter Variational Bounds are Not Necessarily Better` [https://arxiv.org/abs/1802.04537]
    """
    def __init__(self, N=1024, D=20, seed=42):
        super().__init__()
        torch.manual_seed(seed)

        self.mu = Normal(loc=torch.zeros((D,)), scale=torch.ones((D,))).sample()
        mus = self.mu[None].expand(N, D)
        z =  Normal(loc=mus, scale=torch.ones_like(mus)).sample()
        self.data = Normal(loc=z, scale=torch.ones_like(z)).sample()

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]


