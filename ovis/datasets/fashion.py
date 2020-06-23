from torch.distributions import Bernoulli
from torchvision.datasets import FashionMNIST
from torchvision.transforms import Lambda, Compose

class _Fashion(FashionMNIST):

    def __getitem__(self, item):
        x,*_ = super().__getitem__(item)
        return x


def get_fashion_datasets(root, dynamic=False, binarize=False, transform=None, **kwargs):
    assert not (binarize and dynamic)

    _transforms = []
    if transform is not None:
        _transforms += [transform]

    if dynamic:
        _transforms += [
            Lambda(lambda x: Bernoulli(probs=x).sample())]

    if binarize:
        _transforms += [
            Lambda(lambda x: (x >= 0.5).float())]

    transform = Compose(_transforms)
    train_dataset = _Fashion(root, train=True, transform=transform, download=True, **kwargs)
    valid_dataset = _Fashion(root, train=False, transform=transform, download=True, **kwargs)
    test_dataset = _Fashion(root, train=False, transform=transform, download=True, **kwargs)

    return train_dataset, valid_dataset, test_dataset
