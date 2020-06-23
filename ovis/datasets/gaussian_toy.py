from torch.utils.data import Dataset

from ..models import GaussianToyVAE


class GaussianToyDataset(Dataset):
    """
    Gaussian Toy dataset as described in
    `Tighter Variational Bounds are Not Necessarily Better` [https://arxiv.org/abs/1802.04537]
    """

    def __init__(self, N=1024, D=20):
        super().__init__()
        self.data = GaussianToyVAE(xdim=(D,), npoints=N).dset

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]
