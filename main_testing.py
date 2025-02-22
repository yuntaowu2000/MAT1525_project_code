import gc
import os
from typing import Union, Dict, Tuple
import json

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from arch.beta_vae import BetaVAE
from arch.wae_gan import WAEGAN
from arch.wgan import WGANGP
from arch.utils import set_seeds
from dataset import prepare_data
from evaluation import (update_fid_score, plot_fid_score, plot_loss,
                        plot_generated_images)


def eval_vae(model: Union[BetaVAE, WAEGAN], dataset_type="celeba", 
             seed=0, train_ratio=0.8, batch_size=64, num_workers=0,
             output_dir="./models/beta_vae/evals"):
    os.makedirs(output_dir, exist_ok=True)
    train_set, test_set = prepare_data(dataset_type, train_ratio)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    set_seeds(seed)

    samples = model.sample(36)
    plot_generated_images(samples, n_rows=6, fn=os.path.join(output_dir, f"sample36.png"))

    metric = FrechetInceptionDistance().to(model.device)
    for step, batch in enumerate(tqdm(test_dataloader, desc=f"test")):
        batch = batch[0].to(model.device)
        fake_im = model.generate(batch)
        metric = update_fid_score(metric, batch, fake_im)
        gc.collect()
        torch.cuda.empty_cache()
    fid = metric.compute().item()
    return fid

def eval_wgan(model: WGANGP, seed=0, output_dir="./models/wgan_mnist/evals"):
    os.makedirs(output_dir, exist_ok=True)
    set_seeds(seed)
    samples = model.sample(36)
    plot_generated_images(samples, n_rows=6, fn=os.path.join(output_dir, f"sample36.png"))


def process_scores(betavae_generate_fid: Dict[str, float], 
                   wae_generate_fid: Dict[str, float],
                   output_dirs: Dict[Tuple[str, str], str],
                   score_output_dir: str):
    print("{0:=^80}".format("Processing scores"))
    indices = [[], []]
    for ds in ["MNIST", "CelebA"]:
        for model in ["VAE", "WAE", "WGAN-GP"]:
            indices[0].append(ds)
            indices[1].append(model)
    df_scores = pd.DataFrame(index=pd.MultiIndex.from_arrays(indices, names=["Dataset", "Model"]), columns=["KLD", "FID (Reconstruct)", "FID (Random Sample)"])
    for k, v in betavae_generate_fid.items():
        df_scores.loc[(k, "VAE"), "FID (Reconstruct)"] = f"{v:.2f}"
    for k, v in wae_generate_fid.items():
        df_scores.loc[(k, "WAE"), "FID (Reconstruct)"] = f"{v:.2f}"  
    for ds in ["MNIST", "CelebA"]:
        for model in ["VAE", "WAE", "WGAN-GP"]:
            res_df = pd.read_csv(os.path.join(output_dirs[(ds, model)], "model-100-scores.csv"))
            if "KLD" in res_df.columns:
                df_scores.loc[(ds, model), "KLD"] = "{0:.2f}".format(res_df["KLD"].values[-1])
            df_scores.loc[(ds, model), "FID (Random Sample)"] = "{0:.2f}".format(res_df["FID"].values[-1])

    os.makedirs(os.path.join(score_output_dir, "tables"), exist_ok=True)
    df_scores.reset_index().to_csv(os.path.join(score_output_dir, "tables", "scores.csv"))
    ltx = df_scores.style.to_latex(column_format="ll" + "c"*len(df_scores.columns), hrules=True, multirow_align="t")
    ltx = ltx.replace(" &  & KLD & FID (Reconstruct) & FID (Random Sample) \\\\\nDataset & Model &  &  &  \\\\\n\\midrule", 
r"""Dataset & Model & KLD & FID (Reconstruct) & FID (Random Sample) \\
\cmidrule(lr){1-2} \cmidrule(lr){3-5}""")
    ltx = ltx.replace("nan", "")
    with open(os.path.join(score_output_dir, "tables", "scores.tex"), "w") as f:
        f.write(ltx)

    
if __name__ == "__main__":
    if not os.path.exists("./models/generate_stats.json"):
        print("{0:=^80}".format("Generating stats"))
        betavae = BetaVAE(feat_size=(3, 64, 64),
                        latent_dim=32,
                        hidden_dims = [32, 32, 64, 64],
                        beta = 1e-3,)
        betavae.load_weights("./models/beta_vae_celeba/model.pt")
        wae = WAEGAN(feat_size=(3, 64, 64),
                        latent_dim = 64,
                        hidden_dims = [128, 256, 512, 1024],
                        discriminator_hidden = [512] * 4,
                        beta = 1,)
        wae.load_weights("./models/wae_gan_celeba/model.pt")
        
        fid_vae = eval_vae(betavae, "celeba", output_dir="./models/beta_vae_celeba/evals")
        fid_wae = eval_vae(wae, "celeba", output_dir="./models/wae_gan_celeba/evals")


        betavae = BetaVAE(feat_size=(1, 32, 32),
                        latent_dim=32,
                        hidden_dims = [32, 32, 64, 64],
                        beta = 1e-3,)
        betavae.load_weights("./models/beta_vae_mnist/model.pt")
        wae = WAEGAN(feat_size=(1, 32, 32),
                        latent_dim = 8,
                        hidden_dims = [32, 32, 64, 64],
                        discriminator_hidden = [64] * 4,
                        beta = 1,)
        wae.load_weights("./models/wae_gan_mnist/model.pt")
        fid_vae_mnist = eval_vae(betavae, "mnist", output_dir="./models/beta_vae_mnist/evals")
        fid_wae_mnist = eval_vae(wae, "mnist", output_dir="./models/wae_gan_mnist/evals")
        betavae_generate_fid = {"MNIST": fid_vae_mnist, "CelebA": fid_vae}
        wae_generate_fid = {"MNIST": fid_wae_mnist, "CelebA": fid_wae}
        with open("./models/generate_stats.json", "w") as f:
            f.write(json.dumps({"vae": betavae_generate_fid, "wae": wae_generate_fid}))
    else:
        print("{0:=^80}".format("Reading stats"))
        with open("./models/generate_stats.json", "r") as f:
            d = json.loads(f.read())
        betavae_generate_fid = d["vae"]
        wae_generate_fid = d["wae"]

    wgan = WGANGP(feat_size = (1, 32, 32),
                    latent_dim=128,
                    generator_hidden = [512, 256, 128],
                    discriminator_hidden = [128, 256, 512],
                    beta = 10,)
    wgan.load_weights("./models/wgan_mnist/model.pt")
    eval_wgan(wgan, output_dir="./models/wgan_mnist/evals")


    wgan = WGANGP(feat_size = (3, 64, 64),
                    latent_dim=128,
                    generator_hidden = [1024, 512, 256, 128],
                    discriminator_hidden = [128, 256, 512, 1024],
                    beta = 10,)
    wgan.load_weights("./models/wgan_celeba/model.pt")
    eval_wgan(wgan, output_dir="./models/wgan_celeba/evals")

    output_dirs = {
        ("MNIST", "VAE"): "./models/beta_vae_mnist",
        ("CelebA", "VAE"): "./models/beta_vae_celeba",
        ("MNIST", "WAE"): "./models/wae_gan_mnist",
        ("CelebA", "WAE"): "./models/wae_gan_celeba",
        ("MNIST", "WGAN-GP"): "./models/wgan_mnist",
        ("CelebA", "WGAN-GP"): "./models/wgan_celeba",
    }
    process_scores(
        betavae_generate_fid=betavae_generate_fid,
        wae_generate_fid=wae_generate_fid,
        output_dirs=output_dirs,
        score_output_dir="./models"
    )

    os.makedirs("./models/images", exist_ok=True)
    for m, model_dir in output_dirs.items():
        loss_df = pd.read_csv(os.path.join(model_dir, "model-100-loss.csv"))
        for k in ["train", "test"]:
            loss_df[f"{k}_total_loss"] = loss_df[f"{k}_total_loss"].abs()
        plot_loss(loss_df, {"train_total_loss": "train", "test_total_loss": "test"}, os.path.join("./models/images", f"{m[1]}_{m[0]}_loss.jpg"))
        
        score_df = pd.read_csv(os.path.join(model_dir, "model-100-scores.csv"))
        plot_fid_score(score_df, os.path.join("./models/images", f"{m[1]}_{m[0]}_scores.jpg"))
