import torch
from torch.utils.data import Dataset

from ..models import GaussianMixture


class GaussianMixtureDataset(Dataset):
    """
    Gaussian Mixture Dataset as described in
    `Revisiting Reweighted Wake-Sleep for Models with Stochastic Control Flow` [https://arxiv.org/abs/1805.10469]
    """

    def __init__(self, N=1024, C=20, seed=42, static=False):
        super().__init__()
        torch.manual_seed(seed)
        self.N = N
        self.static = static
        model = GaussianMixture(N=C, hdim=16)
        if static:
            self.data = self.sample(model, N)
        else:
            self.model = model

    def sample(self, model, N):
        return model.sample_from_prior(N=N, from_optimal=True)['px'].sample()

    def __len__(self):
        return self.N

    def __getitem__(self, item):
        if self.static:
            return self.data[item]
        else:
            return self.sample(self.model, 1)[0]
