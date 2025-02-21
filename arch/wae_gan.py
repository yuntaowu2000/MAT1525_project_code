# https://github.com/schelotto/Wasserstein-AutoEncoders/blob/master/wae_gan.py
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class WAEGAN(nn.Module):
    def __init__(self,
                 feat_size: Tuple[int, int, int],
                 latent_dim: int = 64,
                 hidden_dims: List = [128, 256, 512, 1024],
                 discriminator_hidden: List = [512] * 4,
                 beta: float = 1) -> None:
        super(WAEGAN, self).__init__()
        self.feat_size = feat_size
        self.latent_dim = latent_dim
        self.beta = beta # lambda in the paper, regularization factor

        self.hidden_dims = hidden_dims
        self.discriminator_hidden = discriminator_hidden

        self.build_encoder()
        self.build_decoder()
        self.build_discriminator()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def build_encoder(self):
        # encoder is a non-random determinstic one, so we don't need to map to mu and sigma
        # it should just generate the latent code directly.
        modules = []
        in_feats = self.feat_size[0]
        im_size = self.feat_size[1] # assuming H=W
        # Build Encoder
        for h_dim in self.hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_feats, h_dim, kernel_size=4, stride=2, padding=1), # down sample
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_feats = h_dim
            im_size = im_size // 2
        modules.append(nn.Flatten())
        modules.append(nn.Linear(self.hidden_dims[-1] * im_size * im_size, self.latent_dim))
        self.encoder = nn.Sequential(*modules)
        self.encoder_modules = modules
        self.final_im_size = im_size

    def build_decoder(self):
        modules = []

        decoder_first_output_shape = self.hidden_dims[-1] * (self.final_im_size * 2) * (self.final_im_size * 2)
        self.decoder_input = nn.Linear(self.latent_dim, decoder_first_output_shape)

        in_feats = self.hidden_dims[-1]
        for h_dim in reversed(self.hidden_dims[:-1]):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_feats, h_dim, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU(),
                )
            )
            in_feats = h_dim
        modules.append(
            nn.Sequential(
                nn.Conv2d(self.hidden_dims[0], self.feat_size[0], kernel_size=3, padding=1)
            )
        )
        self.decoder = nn.Sequential(*modules)
        self.decoder_modules = modules
    
    def build_discriminator(self):
        modules = []
        in_dim = self.latent_dim
        for h_dim in self.discriminator_hidden:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_dim, h_dim),
                    nn.ReLU()
                )
            )
            in_dim = h_dim
        modules.append(
            nn.Sequential(
                nn.Linear(in_dim, 1),
                nn.Sigmoid()
            )
        )
        self.discriminator = nn.Sequential(*modules)
        self.discriminator_module = modules

    def encode(self, x: torch.Tensor, debug=False) -> torch.Tensor:
        if debug:
            for m in self.encoder_modules:
                x = m(x)
            return x
        z_tilde = self.encoder(x)
        return z_tilde
    
    def sample_prior_gaussian(self, B) -> torch.Tensor:
        return torch.randn((B, self.latent_dim), device=self.device)
    
    def decode(self, z: torch.Tensor, debug=False) -> torch.Tensor:
        im: torch.Tensor = self.decoder_input(z)
        im = im.reshape((-1, self.hidden_dims[-1], self.final_im_size * 2, self.final_im_size * 2))
        if debug:
            for m in self.decoder_modules:
                im = m(im)
            return im
        im = self.decoder(im)
        return im
    
    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map
        """
        
        z = self.sample_prior_gaussian(num_samples)
        samples = self.decode(z)
        return samples
    
    def discriminate(self, z: torch.Tensor, debug=False) -> torch.Tensor:
        if debug:
            for m in self.discriminator_module:
                z = m(z)
            return z
        d_z = self.discriminator(z)
        return torch.clamp(d_z, 1e-5, 1-1e-5) # add a clip to avoid nan
    
    def set_param_require_grad(self, module: nn.Module, require_grad: bool = False):
        for p in module.parameters():
            p.requires_grad_(require_grad)
    
    def train_step(self, x: torch.Tensor, 
                   optimizer_enc_dec: torch.optim.Optimizer, 
                   optimizer_dis: torch.optim.Optimizer,
                   debug=False):
        '''
        Inputs:
            x: tensor of shape (B, C, H, W)
            optimizer_enc_dec: optimizer for encoder and decoder
            optimizer_dis: optimizer for discriminator
        '''
        B = x.shape[0]

        optimizer_enc_dec.zero_grad()
        optimizer_dis.zero_grad()
        # firstly, only update the parameters for discriminator
        self.set_param_require_grad(self.encoder, False)
        self.set_param_require_grad(self.decoder, False)
        self.set_param_require_grad(self.discriminator, True)

        z = self.sample_prior_gaussian(B) # sample z from the prior (B, latent_dim)
        z_tilde = self.encode(x, debug) # sample z_tilde from q_{\phi}(z|x), (B, latent_dim)
        d_z = self.discriminate(z, debug) # (B, 1)
        d_ztilde = self.discriminate(z_tilde, debug) # (B, 1)

        loss_discriminator = torch.mean(
            torch.log(d_z) + torch.log(1 - d_ztilde)
        ) 
        (-self.beta * loss_discriminator).backward() # negative loss to perform gradient ascend
        optimizer_dis.step()

        # now update the parameters for encoder/decoder
        self.set_param_require_grad(self.encoder, True)
        self.set_param_require_grad(self.decoder, True)
        self.set_param_require_grad(self.discriminator, False)

        z_tilde = self.encode(x, debug) # sample z_tilde from q_{\phi}(z|x) (B, latent_dim)
        x_tilde = self.decode(z_tilde, debug) # (B, C, H, W)
        d_ztilde = self.discriminate(z_tilde, debug) # (B, 1)

        mse_loss = F.mse_loss(x_tilde, x)
        regularizer_loss = torch.mean(torch.log(d_ztilde))
        total_loss = mse_loss - self.beta * regularizer_loss
        total_loss.backward()
        optimizer_enc_dec.step()

        return {
            "loss_discriminator": loss_discriminator,
            "loss_reconstruction": mse_loss,
            "loss_regularizer": regularizer_loss,
            "total_loss": total_loss
        }
    
    def test_step(self, x: torch.Tensor):
        B = x.shape[0]
        
        z = self.sample_prior_gaussian(B) # sample z from the prior (B, latent_dim)
        z_tilde = self.encode(x) # sample z_tilde from q_{\phi}(z|x), (B, latent_dim)
        d_z = self.discriminate(z) # (B, 1)
        d_ztilde = self.discriminate(z_tilde) # (B, 1)

        loss_discriminator = torch.mean(
            torch.log(d_z) + torch.log(1 - d_ztilde)
        ) 

        z_tilde = self.encode(x) # sample z_tilde from q_{\phi}(z|x) (B, latent_dim)
        x_tilde = self.decode(z_tilde) # (B, C, H, W)
        d_ztilde = self.discriminate(z_tilde) # (B, 1)

        mse_loss = F.mse_loss(x_tilde, x)
        regularizer_loss = torch.mean(torch.log(d_ztilde))
        total_loss = mse_loss - self.beta * regularizer_loss

        return {
            "loss_discriminator": loss_discriminator,
            "loss_reconstruction": mse_loss,
            "loss_regularizer": regularizer_loss,
            "total_loss": total_loss
        }


    def save_weights(self, model_dir, filename):
        dict_to_save = {
            "model": self.state_dict(),
            "system_rng": random.getstate(),
            "numpy_rng": np.random.get_state(),
            "torch_rng": torch.random.get_rng_state(),
        }
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        torch.save(dict_to_save, f"{model_dir}/{filename}.pt")
    
    def load_weights(self, f: str=None, dict_to_load: dict=None):
        '''
            Input:
                f: path-like, pointing to the saved state dictionary
                dict_to_load: a dictionary, loaded from the saved file
                One of them must not be None
        '''
        assert (f is not None) or (dict_to_load is not None), "One of file path or dict_to_load must not be None"
        if f is not None:
            dict_to_load = torch.load(f, weights_only=False)
        self.load_state_dict(dict_to_load["model"])

    