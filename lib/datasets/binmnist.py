import os
import pickle as pkl
from urllib.request import urlretrieve

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


def load_mnist_binarized(root):
    datapath = os.path.join(root, 'bin-mnist')
    if not os.path.exists(datapath):
        os.makedirs(datapath)
    dataset = os.path.join(datapath, "mnist.pkl.gz")

    if not os.path.isfile(dataset):

        datafiles = {
            "train": "http://www.cs.toronto.edu/~larocheh/public/"
                     "datasets/binarized_mnist/binarized_mnist_train.amat",
            "valid": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                     "binarized_mnist/binarized_mnist_valid.amat",
            "test": "http://www.cs.toronto.edu/~larocheh/public/datasets/"
                    "binarized_mnist/binarized_mnist_test.amat"
        }
        datasplits = {}
        for split in datafiles.keys():
            print("Downloading %s data..." % (split))
            datasplits[split] = np.loadtxt(urlretrieve(datafiles[split])[0])

        pkl.dump([datasplits['train'], datasplits['valid'], datasplits['test']], open(dataset, "wb"))

    x_train, x_valid, x_test = pkl.load(open(dataset, "rb"))
    return x_train, x_valid, x_test


class BinMNIST(Dataset):
    """Binary MNIST dataset"""

    def __init__(self, data, transform=None):
        h, w, c = 28, 28, 1
        self.data = np.ascontiguousarray(data.reshape(-1, h, w), dtype=np.ubyte)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]

        sample = Image.fromarray(255 * sample)  # cannot read bytes directly: https://github.com/numpy/numpy/issues/5861

        if self.transform:
            sample = self.transform(sample)

        return sample


def get_binmnist_datasets(root, **kwargs):
    x_train, x_valid, x_test = load_mnist_binarized(root)
    # x_train = np.append(x_train, x_valid, axis=0)  # https://github.com/casperkaae/LVAE/blob/master/run_models.py (line 401)
    train_dset = BinMNIST(x_train, **kwargs)
    valid_dset = BinMNIST(x_valid, **kwargs)
    test_dset = BinMNIST(x_test, **kwargs)
    return train_dset, valid_dset, test_dset
