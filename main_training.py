import gc
import os
import time
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

from arch.beta_vae import BetaVAE
from arch.utils import model_eval, save_loss_to_csv, set_seeds
from arch.wae_gan import WAEGAN
from arch.wgan import WGANGP
from dataset import prepare_data
from evaluation import (update_fid_score, compute_kl_divergence,
                        plot_generated_images)


def train_beta_vae(dataset_type="mnist", batch_size=64, train_ratio=0.8,
                feat_size=(1, 32, 32),
                latent_dim=32,
                hidden_dims = [32, 32, 64, 64],
                beta = 1e-3,
                lr=1e-4, epochs=100,
                eval_steps=10,
                num_workers=0,
                model_dir="./models/beta_vae/", filename="model"):
    train_set, test_set = prepare_data(dataset_type, train_ratio)
    set_seeds(0)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = BetaVAE(feat_size, latent_dim, hidden_dims, beta)
    optimizer = torch.optim.Adam(model.parameters(), lr)
        
    ## run for the specified number of epochs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if "." in filename:
        file_prefix = filename.split(".")[0]
    else:
        file_prefix = filename
    log_file = open(f"{model_dir}/{file_prefix}-{epochs}-log.txt", "w", encoding="utf-8")
    csv_file = f"{model_dir}/{file_prefix}-{epochs}-loss.csv"
    csv_file2 = f"{model_dir}/{file_prefix}-{epochs}-scores.csv"

    print("Model config: ", file=log_file)
    print(f"LR: {lr}", file=log_file)
    print(f"Epochs: {epochs}", file=log_file)
    print("", file=log_file)
    log_file.flush()
    start_time = time.time()

    kld_fid_scores = pd.DataFrame(columns=["epoch", "KLD", "FID"])
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_losses = defaultdict(float)
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"train-{epoch}")):
            batch = batch[0].to(model.device)
            losses = model.train_step(batch, optimizer)
            for k, v in losses.items():
                train_losses[k] += v.item()
            num_batches += 1
        for k, v in train_losses.items():
            train_losses[k] = v / (num_batches)
        gc.collect()
        torch.cuda.empty_cache()
        
        dev_losses = model_eval(model, test_dataloader)
        model.save_weights(model_dir, file_prefix)
        save_loss_to_csv(train_losses, dev_losses, epoch, csv_file)

        formatted_train_loss = ", ".join([f'{k}: {v:.3f}' for k, v in train_losses.items()])
        formatted_dev_loss = ", ".join([f'{k}: {v:.3f}' for k, v in dev_losses.items()])
        print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss}, \ndev loss :: {formatted_dev_loss}, \ntime elapsed :: {time.time() - epoch_start_time}")
        print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss}, \ndev loss :: {formatted_dev_loss}, \ntime elapsed :: {time.time() - epoch_start_time}", file=log_file, flush=True)

        if (epoch + 1) % eval_steps == 0:
            print("Evaluating...")
            samples = model.sample(16)
            plot_generated_images(samples, fn=os.path.join(model_dir, f"{filename}_epoch{epoch+1}.png"))

            metric = FrechetInceptionDistance().to(model.device)
            encoded_zs = torch.empty((0,), device=model.device)
            for step, batch in enumerate(tqdm(test_dataloader, desc=f"test-{epoch}")):
                batch = batch[0].to(model.device)
                encoded_z = model.encode_gaussian(batch)[0]
                encoded_zs = torch.cat([encoded_zs, encoded_z])
                fake_im = model.sample(batch.shape[0])
                metric = update_fid_score(metric, batch, fake_im)
                gc.collect()
                torch.cuda.empty_cache()
            fid = metric.compute().item()
            kld = compute_kl_divergence(encoded_zs).item()
            print(f"epoch: {epoch}, kld: {kld:.2f}, fid: {fid:.2f}")
            curr_df = pd.DataFrame({"epoch": [epoch], "KLD": [kld], "FID": [fid]})
            kld_fid_scores = curr_df if kld_fid_scores.empty else pd.concat([kld_fid_scores, curr_df], ignore_index=True)
        gc.collect()
        torch.cuda.empty_cache()    
    kld_fid_scores.to_csv(csv_file2, index=False)

    print(f"training finished, total time :: {time.time() - start_time}")
    print(f"training finished, total time :: {time.time() - start_time}", file=log_file)
    log_file.close()
    return train_losses, dev_losses

def train_wae_gan(dataset_type="mnist", batch_size=64, train_ratio=0.8,
                 feat_size=(1, 32, 32),
                latent_dim = 8,
                hidden_dims = [128, 256, 512, 1024],
                discriminator_hidden = [512] * 4,
                beta = 1,
                lr_enc_dec=3e-4, lr_dis=1e-3, 
                epochs=100, eval_steps=10,
                num_workers=0,
                model_dir="./models/wae_gan/", filename="model"):
    train_set, test_set = prepare_data(dataset_type, train_ratio)
    set_seeds(0)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = WAEGAN(feat_size, latent_dim, hidden_dims, discriminator_hidden, beta)

    optimizer_dec_enc = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr_enc_dec)
    optimizer_dis = torch.optim.Adam(model.discriminator.parameters(), lr_dis)
    
    ## run for the specified number of epochs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if "." in filename:
        file_prefix = filename.split(".")[0]
    else:
        file_prefix = filename
    log_file = open(f"{model_dir}/{file_prefix}-{epochs}-log.txt", "w", encoding="utf-8")
    csv_file = f"{model_dir}/{file_prefix}-{epochs}-loss.csv"
    csv_file2 = f"{model_dir}/{file_prefix}-{epochs}-scores.csv"

    print("Model config: ", file=log_file)
    print(f"LR: {lr_enc_dec}, {lr_dis}", file=log_file)
    print(f"Epochs: {epochs}", file=log_file)
    print("", file=log_file)
    log_file.flush()
    start_time = time.time()

    kld_fid_scores = pd.DataFrame(columns=["epoch", "KLD", "FID"])
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_losses = defaultdict(float)
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"train-{epoch}")):
            batch = batch[0].to(model.device)
            losses = model.train_step(batch, optimizer_dec_enc, optimizer_dis)
            for k, v in losses.items():
                train_losses[k] += v.item()
            num_batches += 1
        for k, v in train_losses.items():
            train_losses[k] = v / (num_batches)
        gc.collect()
        torch.cuda.empty_cache()

        dev_losses = model_eval(model, test_dataloader)
        model.save_weights(model_dir, file_prefix)
        save_loss_to_csv(train_losses, dev_losses, epoch, csv_file)

        formatted_train_loss = ", ".join([f'{k}: {v:.3f}' for k, v in train_losses.items()])
        formatted_dev_loss = ", ".join([f'{k}: {v:.3f}' for k, v in dev_losses.items()])
        print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss}, \ndev loss :: {formatted_dev_loss}, \ntime elapsed :: {time.time() - epoch_start_time}")
        print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss}, \ndev loss :: {formatted_dev_loss}, \ntime elapsed :: {time.time() - epoch_start_time}", file=log_file, flush=True)
        if (epoch + 1) % eval_steps == 0:
            print("Evaluating...")
            samples = model.sample(16)
            plot_generated_images(samples, fn=os.path.join(model_dir, f"{filename}_epoch{epoch+1}.png"))

            metric = FrechetInceptionDistance().to(model.device)
            encoded_zs = torch.empty((0,), device=model.device)
            for step, batch in enumerate(tqdm(test_dataloader, desc=f"test-{epoch}")):
                batch = batch[0].to(model.device)
                encoded_z = model.encode(batch)
                encoded_zs = torch.cat([encoded_zs, encoded_z])

                fake_im = model.sample(batch.shape[0])
                metric = update_fid_score(metric, batch, fake_im)
                gc.collect()
                torch.cuda.empty_cache()
            fid = metric.compute().item()
            kld = compute_kl_divergence(encoded_zs).item()
            print(f"epoch: {epoch}, kld: {kld:.2f}, fid: {fid:.2f}")
            curr_df = pd.DataFrame({"epoch": [epoch], "KLD": [kld], "FID": [fid]})
            kld_fid_scores =  curr_df if kld_fid_scores.empty else pd.concat([kld_fid_scores, curr_df], ignore_index=True)
        gc.collect()
        torch.cuda.empty_cache()    
    kld_fid_scores.to_csv(csv_file2, index=False)

    print(f"training finished, total time :: {time.time() - start_time}")
    print(f"training finished, total time :: {time.time() - start_time}", file=log_file)
    log_file.close()
    return train_losses, dev_losses


def train_wgan(dataset_type="mnist", batch_size=64, train_ratio=0.8,
                feat_size = (1, 32, 32),
                 latent_dim=128,
                 generator_hidden = [1024, 512, 256, 128],
                 discriminator_hidden = [128, 256, 512, 1024],
                 beta = 10,
                    lr_gen=2e-4, lr_dis=2e-4, 
                    epochs=100, n_critic=5, eval_steps=10,
                    num_workers=0,
                    model_dir="./models/wgan/", filename="model"):
    train_set, test_set = prepare_data(dataset_type, train_ratio)
    set_seeds(0)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    model = WGANGP(feat_size, latent_dim, generator_hidden, discriminator_hidden, beta)
    optimizer_gen = torch.optim.Adam(model.generator.parameters(), lr_gen, betas=(0.5, 0.9))
    optimizer_dis = torch.optim.Adam(model.discriminator.parameters(), lr_dis, betas=(0.5, 0.9))
    
    ## run for the specified number of epochs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if "." in filename:
        file_prefix = filename.split(".")[0]
    else:
        file_prefix = filename
    log_file = open(f"{model_dir}/{file_prefix}-{epochs}-log.txt", "w", encoding="utf-8")
    csv_file = f"{model_dir}/{file_prefix}-{epochs}-loss.csv"
    csv_file2 = f"{model_dir}/{file_prefix}-{epochs}-scores.csv"

    print("Model config: ", file=log_file)
    print(f"LR: {lr_gen}, {lr_dis}", file=log_file)
    print(f"Epochs: {epochs}", file=log_file)
    print("", file=log_file)
    log_file.flush()
    start_time = time.time()

    kld_fid_scores = pd.DataFrame(columns=["epoch", "FID"])
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        train_losses = defaultdict(float)
        num_batches = 0
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"train-{epoch}")):
            batch = batch[0].to(model.device)
            losses = model.train_step(batch, optimizer_dis, optimizer_gen, 
                                        generator_step=((step + 1) % n_critic == 0))

            for k, v in losses.items():
                train_losses[k] += v.item()
            num_batches += 1
        for k, v in train_losses.items():
            train_losses[k] = v / (num_batches)
        gc.collect()
        torch.cuda.empty_cache()

        dev_losses = model_eval(model, test_dataloader)
        model.save_weights(model_dir, file_prefix)
        save_loss_to_csv(train_losses, dev_losses, epoch, csv_file)

        formatted_train_loss = ", ".join([f'{k}: {v:.3f}' for k, v in train_losses.items()])
        formatted_dev_loss = ", ".join([f'{k}: {v:.3f}' for k, v in dev_losses.items()])
        print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss}, \ndev loss :: {formatted_dev_loss}, \ntime elapsed :: {time.time() - epoch_start_time}")
        print(f"epoch {epoch}: \ntrain loss :: {formatted_train_loss}, \ndev loss :: {formatted_dev_loss}, \ntime elapsed :: {time.time() - epoch_start_time}", file=log_file, flush=True)

        if (epoch + 1) % eval_steps == 0:
            print("Evaluating...")
            samples = model.sample(16)
            plot_generated_images(samples, fn=os.path.join(model_dir, f"{filename}_epoch{epoch+1}.png"))

            metric = FrechetInceptionDistance().to(model.device)
            for step, batch in enumerate(tqdm(test_dataloader, desc=f"test-{epoch}")):
                batch = batch[0].to(model.device)
                fake_im = model.sample(batch.shape[0])
                metric = update_fid_score(metric, batch, fake_im)
                gc.collect()
                torch.cuda.empty_cache()
            fid = metric.compute().item()
            print(f"epoch: {epoch}, fid: {fid:.2f}")
            curr_df = pd.DataFrame({"epoch": [epoch], "FID": [fid]})
            kld_fid_scores = curr_df if kld_fid_scores.empty else pd.concat([kld_fid_scores, curr_df], ignore_index=True)
        gc.collect()
        torch.cuda.empty_cache()    
    kld_fid_scores.to_csv(csv_file2, index=False)

    print(f"training finished, total time :: {time.time() - start_time}")
    print(f"training finished, total time :: {time.time() - start_time}", file=log_file)
    log_file.close()
    return train_losses, dev_losses


if __name__ == "__main__":
    print("{0:=^80}".format("MNIST VAE"))
    train_beta_vae(dataset_type="mnist", 
                   batch_size=64, train_ratio=0.8,
                feat_size=(1, 32, 32),
                latent_dim=32,
                hidden_dims = [32, 32, 64, 64],
                beta = 1e-3,
                lr=1e-4, 
                epochs=100, eval_steps=10,
                num_workers=0,
                model_dir="./models/beta_vae_mnist/", filename="model")
    
    print("{0:=^80}".format("MNIST WAE-GAN"))
    train_wae_gan(dataset_type="mnist", 
                  batch_size=64, train_ratio=0.8,
                 feat_size=(1, 32, 32),
                latent_dim = 64,
                hidden_dims = [32, 32, 64, 64],
                discriminator_hidden = [64] * 4,
                beta = 1,
                lr_enc_dec=3e-4, lr_dis=1e-3, 
                epochs=100, eval_steps=10,
                num_workers=0,
                model_dir="./models/wae_gan_mnist/", filename="model")
    
    print("{0:=^80}".format("MNIST WGAN-GP"))
    train_wgan(dataset_type="mnist", 
               batch_size=64, train_ratio=0.8,
                feat_size = (1, 32, 32),
                 latent_dim=128,
                 generator_hidden = [512, 256, 128],
                 discriminator_hidden = [128, 256, 512],
                 beta = 10,
                    lr_gen=2e-4, lr_dis=2e-4, 
                    epochs=100, n_critic=5, eval_steps=10,
                    num_workers=0,
                    model_dir="./models/wgan_mnist/", filename="model")
    
    print("{0:=^80}".format("CelebA VAE"))
    train_beta_vae(dataset_type="celeba", 
                   batch_size=64, train_ratio=0.8,
                feat_size=(3, 64, 64),
                latent_dim=32,
                hidden_dims = [32, 32, 64, 64],
                beta = 1e-3,
                lr=1e-4, 
                epochs=100, eval_steps=10,
                num_workers=8,
                model_dir="./models/beta_vae_celeba/", filename="model")
    
    print("{0:=^80}".format("CelebA WAE-GAN"))
    train_wae_gan(dataset_type="celeba", 
                  batch_size=64, train_ratio=0.8,
                 feat_size=(3, 64, 64),
                latent_dim = 64,
                hidden_dims = [128, 256, 512, 1024],
                discriminator_hidden = [512] * 4,
                beta = 1,
                lr_enc_dec=3e-4, lr_dis=1e-3, 
                epochs=100, eval_steps=10,
                num_workers=8,
                model_dir="./models/wae_gan_celeba/", filename="model")
    
    print("{0:=^80}".format("CelebA WGAN"))
    train_wgan(dataset_type="celeba", 
               batch_size=64, train_ratio=0.8,
                feat_size = (3, 64, 64),
                 latent_dim=128,
                 generator_hidden = [1024, 512, 256, 128],
                 discriminator_hidden = [128, 256, 512, 1024],
                 beta = 10,
                    lr_gen=2e-4, lr_dis=2e-4, 
                    epochs=100, n_critic=5, eval_steps=10,
                    num_workers=8,
                    model_dir="./models/wgan_celeba/", filename="model")