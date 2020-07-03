import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name='ovis',
    version='0.1.0',
    author="Valentin Lievin",
    author_email="valentin.lievin@gmail.com",
    description="Official code for the Optimal Variance Control of the Score Function Gradient Estimator for Importance Weighted Bounds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vlievin/ovis",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    install_requires=[
        'torch',
        'torchvision',
        'tensorboard',
        'tqdm',
        'numpy',
        'matplotlib',
        'seaborn',
        'booster-pytorch',
        'pandas',
        'scipy',
        'scikit-image',
        'tinydb',
        'tbparser',
        'GPUtil'
    ],
)
