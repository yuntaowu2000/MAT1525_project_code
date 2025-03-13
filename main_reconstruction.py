import os
from typing import Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from arch.beta_vae import BetaVAE
from arch.wae_gan import WAEGAN
from arch.utils import set_seeds
from dataset import prepare_data
from evaluation import plot_generated_images


def vae_reconstruct(model: Union[BetaVAE, WAEGAN], dataset_type="mnist", 
             seed=0, train_ratio=0.8, batch_size=18, num_workers=0,
             output_dir="./models/beta_vae/evals"):
    os.makedirs(output_dir, exist_ok=True)
    train_set, test_set = prepare_data(dataset_type, train_ratio)
    set_seeds(seed)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    batch = next(iter(test_dataloader))[0]
    samples = model.generate(batch)
    res = torch.zeros((batch_size * 2, samples.shape[1], samples.shape[2], samples.shape[3]), device=samples.device)
    for i in range(batch_size // 6):
        res[2*i*6:(2*i+1)*6, ...] = batch[i*6:(i+1)*6,...]
        res[(2*i+1)*6:(2*i+2)*6,...] = samples[i*6:(i+1)*6,...]
    plot_generated_images(res, n_rows=6, fn=os.path.join(output_dir, f"reconstruct18.png"))

if __name__ == "__main__":
    betavae = BetaVAE(feat_size=(1, 32, 32),
                        latent_dim=32,
                        hidden_dims = [32, 32, 64, 64],
                        beta = 1e-3,)
    betavae.load_weights("./models/beta_vae_mnist/model.pt")
    vae_reconstruct(betavae, "mnist", output_dir="./models/beta_vae_mnist/evals")

    wae = WAEGAN(feat_size=(1, 32, 32),
                    latent_dim = 8,
                    hidden_dims = [32, 32, 64, 64],
                    discriminator_hidden = [64] * 4,
                    beta = 1,)
    wae.load_weights("./models/wae_gan_mnist/model.pt")
    vae_reconstruct(wae, "mnist", output_dir="./models/wae_gan_mnist/evals")

    betavae = BetaVAE(feat_size=(3, 64, 64),
                        latent_dim=32,
                        hidden_dims = [32, 32, 64, 64],
                        beta = 1e-3,)
    betavae.load_weights("./models/beta_vae_celeba/model.pt")
    vae_reconstruct(betavae, "celeba", output_dir="./models/beta_vae_celeba/evals")

    wae = WAEGAN(feat_size=(3, 64, 64),
                    latent_dim = 64,
                    hidden_dims = [128, 256, 512, 1024],
                    discriminator_hidden = [512] * 4,
                    beta = 1,)
    wae.load_weights("./models/wae_gan_celeba/model.pt")
    vae_reconstruct(wae, "celeba", output_dir="./models/wae_gan_celeba/evals")