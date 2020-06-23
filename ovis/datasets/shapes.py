from math import pi

import numpy as np
from PIL import Image
from skimage.draw import polygon, ellipse
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

colors = np.array([
    [255, 255, 255],
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255],
    [255, 255, 0],
    [255, 0, 255],
    [0, 255, 255],
])

n_angles = 8
angles = [pi * k / n_angles for k in range(n_angles)]



# shape generation code stolen from
# https://github.com/addtt/multi-object-datasets/blob/master/utils/graphics.py

def get_ellipse(angle, color, scale, patch_size):
    img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    rr, cc = ellipse(patch_size / 2,
                     patch_size / 2,
                     r_radius=scale * patch_size / 2,
                     c_radius=scale * patch_size / 3,
                     shape=img.shape,
                     rotation=angle)
    img[rr, cc, :] = color[None, None, :]
    return img


def get_square(angle, color, scale, patch_size):
    num_vert = 4
    return get_regular_polygon(angle, num_vert, color, scale, patch_size)


def get_triangle(angle, color, scale, patch_size):
    num_vert = 3
    return get_regular_polygon(angle, num_vert, color, scale, patch_size)


def get_regular_polygon(angle, num_vert, color, scale, patch_size):
    # Coordinates of starting vertex
    def x1(a): return (1 + np.cos(a) * scale) * patch_size / 2

    def y1(a): return (1 + np.sin(a) * scale) * patch_size / 2

    # Loop over circle and add vertices
    angles = np.arange(angle, angle + 2 * np.pi - 1e-3, 2 * np.pi / num_vert)
    coords = list(([x1(a), y1(a)] for a in angles))

    # Create image and set polygon to given color
    img = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
    coords = np.array(coords)
    rr, cc = polygon(coords[:, 0], coords[:, 1], img.shape)
    img[rr, cc, :] = color[None, None, :]

    return img

def get_shape(shape_type, *args, **kwargs):
    if shape_type == 0:
        return get_triangle(*args, **kwargs)
    elif shape_type == 1:
        return get_square(*args, **kwargs)
    elif shape_type == 2:
        return get_ellipse(*args, **kwargs)
    else:
        raise ValueError(f"shape_type = {shape_type} must be in range [0..2]")


def get_shapes(patch_size=16,
               colors=colors,
               angles=angles,
               scales=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    for shape_type in range(3):
        for color in colors:
            for angle in angles:
                for scale in scales:
                    yield get_shape(shape_type, angle, color, scale, patch_size)


class ShapesDataset(Dataset):
    def __init__(self, shapes, transform=None):
        self.data = np.ascontiguousarray(np.concatenate([p[None].astype(np.ubyte) for p in shapes]))
        self.transform = transform

    def __getitem__(self, idx):
        sample = self.data[idx]
        sample = Image.fromarray(sample)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.data.shape[0]


def get_shapes_datasets(seed=1984, train_ratio=0.9, transform=ToTensor(), **kwargs):
    shapes = list(get_shapes(**kwargs))

    rgn = np.random.RandomState(seed)
    rgn.shuffle(shapes)

    n_train = int(train_ratio * len(shapes))
    train_shapes, test_shapes = shapes[:n_train], shapes[n_train:]

    train_dset = ShapesDataset(train_shapes, transform=transform)
    test_dset = ShapesDataset(test_shapes, transform=transform)

    return train_dset, test_dset, test_dset
