import os
from urllib.request import urlretrieve

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from torchvision import transforms


def load_omniglot(datapath):
    dataset = os.path.join(datapath, "chardata.mat")

    print("# omniglot data path:", os.path.abspath(dataset))

    if not os.path.isfile(dataset):
        origin = (
            'https://github.com/yburda/iwae/raw/'
            'master/datasets/OMNIGLOT/chardata.mat'
        )
        print('Downloading data from %s' % origin)
        urlretrieve(origin, dataset)

    data = loadmat(dataset)

    train_x = data['data'].astype('float32').T
    test_x = data['testdata'].astype('float32').T

    return train_x, test_x


class Omniglot(Dataset):
    """Binary MNIST dataset"""

    def __init__(self, data, dynamic=False, transform=None):

        # load data
        h, w, c = 28, 28, 1
        self.data = data.reshape(-1, h, w)

        # base transform
        _transforms = []

        if transform is not None:
            _transforms += [transform]

        def sample_bernouilli_with_probs(x):
            return torch.distributions.bernoulli.Bernoulli(probs=x.float() / 255.).sample()

        if dynamic:
            _transforms += [transforms.Lambda(sample_bernouilli_with_probs)]
        else:
            _transforms += [transforms.Lambda(lambda x: x / 255.)]

        self.transform = transforms.Compose(_transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        sample = Image.fromarray(255 * sample)  # cannot read bytes directly: https://github.com/numpy/numpy/issues/5861

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_omniglot_datasets(root, dynamic=False, seed=42, **kwargs):
    path = os.path.join(root, 'omniglot/')
    if not os.path.exists(path):
        os.makedirs(path)
    x_train, x_test = load_omniglot(root)

    if dynamic:
        rng = np.random.RandomState(seed)
        x_test = rng.binomial(1, x_test).astype(np.float32)

    train_dset = Omniglot(x_train, dynamic=dynamic, **kwargs)
    test_dset = Omniglot(x_test, dynamic=False, **kwargs)
    return train_dset, test_dset, test_dset
