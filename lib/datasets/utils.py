from torchvision.transforms import ToTensor

from .binmnist import get_binmnist_datasets
from .fashion import get_fashion_datasets
from .omniglot import get_omniglot_datasets
from .shapes import get_shapes_datasets


def get_datasets(opt):
    transform = ToTensor()
    if opt.dataset == "shapes":
        return get_shapes_datasets(transform=transform)
    elif opt.dataset == "binmnist":
        return get_binmnist_datasets(opt.data_root, transform=transform)
    elif opt.dataset == "omniglot":
        return get_omniglot_datasets(opt.data_root, transform=transform, dynamic=True)
    elif opt.dataset == "fashion":
        return get_fashion_datasets(opt.data_root, transform=transform, binarize=True)
    else:
        raise ValueError(f"Unknown data: {opt.dataset}")
