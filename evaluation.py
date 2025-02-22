from typing import Dict

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from torchvision.utils import make_grid

from arch.utils import *

# perform model evaluation in terms of the accuracy and f1 score.
set_seeds(0)

def transform_back(batch: torch.Tensor):
    '''
    Note all images are scaled to [-1,1] by (x-0.5) * 2, we need to change back to [0,1]
    '''
    return torch.clamp(batch / 2 + 0.5, 0., 1.)

def interpolate(batch):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        lambda x: x * 255,
        lambda x: x.to(torch.uint8)
    ])
    batch = torch.stack([transform(img) for img in batch])
    if batch.shape[1] == 1:
        batch = torch.repeat_interleave(batch, 3, dim=1)
    return batch

def compute_fid_score(real_im: torch.Tensor, fake_im: torch.Tensor):
    '''
    Input images with be tensors in the range [0,1]
    '''
    real_im = interpolate(transform_back(real_im))
    fake_im = interpolate(transform_back(fake_im))
    metric = FrechetInceptionDistance()
    metric.update(real_im, real=True)
    metric.update(fake_im, real=False)
    res = metric.compute()
    return res.item()

def update_fid_score(metric: FrechetInceptionDistance, real_im: torch.Tensor, fake_im: torch.Tensor):
    '''
    Input images with be tensors in the range [0,1]
    '''
    real_im = interpolate(transform_back(real_im))
    fake_im = interpolate(transform_back(fake_im))
    metric.update(real_im, real=True)
    metric.update(fake_im, real=False)
    return metric


def compute_kl_divergence(encoded_z: torch.Tensor):
    '''
    Since the deterministic encoder always outputs the same z given an input x, 
    the posterior distribution collapses to a delta function centered at z. 
    The KL term can be interpreted as the cost of moving the deterministic encoding towards a Gaussian prior.
    '''
    return 0.5 * torch.sum(torch.square(encoded_z), dim=-1).mean()

def plot_generated_images(x: torch.Tensor, n_rows=4, fn: str=""):
    '''
    Input:
        x: (B, C, H, W) tensor
        n_rows: number of rows in the shown plot
    '''
    x = transform_back(x)
    grid = make_grid(x, nrow=n_rows).permute(1, 2, 0).cpu()

    plt.figure(figsize=(20, 20))
    plt.imshow(grid, vmin=0., vmax=1.)
    plt.tight_layout()
    if fn == "":
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()

def plot_loss(loss_df: pd.DataFrame, losses_to_plot: Dict[str, str], fn: str=""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    for k, label in losses_to_plot.items():
        ax.plot(loss_df["epoch"], loss_df[k], label=label)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend(loc="upper right")
    plt.tight_layout()
    if fn == "":
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()

def plot_fid_score(fid_df: pd.DataFrame, fn: str = ""):
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(fid_df["epoch"], fid_df["FID"], label="FID")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("FID")
    if "KLD" in fid_df.columns:
        twin_ax = ax.twinx()
        twin_ax.set_ylabel("KLD")
        twin_ax.plot(fid_df["epoch"], fid_df["KLD"], label="KLD", color="orange")
        ax_lines, ax_labels = ax.get_legend_handles_labels()
        twinax_lines, twinax_labels = twin_ax.get_legend_handles_labels()
        ax_lines.extend(twinax_lines)
        ax_labels.extend(twinax_labels)
        ax.legend(ax_lines, ax_labels, loc="upper right")
    plt.tight_layout()
    if fn == "":
        plt.show()
    else:
        plt.savefig(fn)
        plt.close()
