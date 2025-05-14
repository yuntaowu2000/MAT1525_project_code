# MAT1525 project: Wasserstein Autoencoders

This is the project repository for MAT1525H (2025):Topics in Inverse Problems and Image Analysis. In this project, I replicate the Wasserstein Autoencoder (WAE) and compare its performance with Beta-VAE and WGAN-GP on MNIST and CelebA datasets.

WAE: [https://arxiv.org/abs/1711.01558](https://arxiv.org/abs/1711.01558)  
Beta-VAE: [https://openreview.net/forum?id=Sy2fzU9gl](https://openreview.net/forum?id=Sy2fzU9gl)  
WGAN-GP: [https://arxiv.org/abs/1704.00028](https://arxiv.org/abs/1704.00028)  


My [project report](./project_report.pdf)

## Code structure

- [./arch](./arch): PyTorch implementation of the WAE, Beta-VAE, and WGAN-GP.
- [./check_shape.py](./check_shape.py): Check the number of parameters and shapes of the models.
- [./dataset.py](./dataset.py): load MNIST and CelebA datasets.
- [./evaluation.py](./evaluation.py): code for computing and plotting evaluation metrics (FID score, KL divergence, loss).
- [./main_reconstruction.py](./main_reconstruction.py): code for reconstructing images using WAE and Beta-VAE.
- [./main_testing.py](./main_testing.py): main functions for computing evaluation metrics on MNIST and CelebA datasets.
- [./main_training.py](./main_training.py): main functions for training WAE, Beta-VAE, and WGAN-GP on MNIST and CelebA datasets.


The repository is configured with [poetry](https://python-poetry.org/) for dependency management and packaging. To install the dependencies and run the code:

```bash
poetry config virtualenvs.in-project true --local # this sets the virtual environment path to be in the local directory.
poetry shell # creates the virtual environment
poetry install --no-root # installs the dependencies

## You can now run the experiments by running main files
```