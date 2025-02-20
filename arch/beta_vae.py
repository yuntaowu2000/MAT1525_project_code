import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class BetaVAE(nn.Module):
    def __init__(self,
                 feat_size: Tuple[int, int, int],
                 latent_dim: int = 32,
                 hidden_dims: List = [32, 32, 64, 64],
                 beta: float = 4) -> None:
        super(BetaVAE, self).__init__()
        self.feat_size = feat_size
        self.latent_dim = latent_dim
        self.beta = beta

        self.hidden_dims = hidden_dims

        self.build_encoder_conv()
        self.build_decoder_conv()

        self.forward = self.forward_gaussian
        self.sample = self.sample_gaussian

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)
    
    def build_encoder_conv(self):
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
        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(self.hidden_dims[-1] * im_size * im_size, self.latent_dim)
        self.fc_var = nn.Linear(self.hidden_dims[-1] * im_size * im_size, self.latent_dim)
        self.final_im_size = im_size

    def build_decoder_conv(self):
        # Build Decoder
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

    def encode_gaussian(self, x: torch.Tensor) -> List[torch.Tensor]:
        result = self.encoder(x)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return (mu, log_var)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.decoder_input(z)
        
        result = result.reshape((-1, self.hidden_dims[-1], self.final_im_size * 2, self.final_im_size * 2))
        result = self.decoder(result)
        return result

    def reparameterize_gaussian(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward_gaussian(self, input: torch.Tensor) -> torch.Tensor:
        mu, log_var = self.encode_gaussian(input)
        z = self.reparameterize_gaussian(mu, log_var)
        return self.decode(z), input, mu, log_var

    def sample_gaussian(self, num_samples: int) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map
        """
        
        z = torch.randn((num_samples, self.latent_dim), device=self.device)
        samples = self.decode(z)
        return samples

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input image x, returns the reconstructed image
        """
        return self.forward(x)[0]
    
    def train_step(self, x: torch.Tensor, optimizer: torch.optim.Optimizer):
        '''
        Input:
            x: tensor of shape (B, C, H, W)
        '''
        x = x.to(self.device)
        optimizer.zero_grad()
        recons, x, mu, log_var = self.forward_gaussian(x)
        recons_loss = F.mse_loss(recons, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var), dim = 1), dim = 0)
        loss = recons_loss + self.beta * kld_loss
        loss.backward()
        optimizer.step()
        return {"loss_reconstruction": recons_loss, "loss_kl": kld_loss, "total_loss": loss}
    
    def test_step(self, x: torch.Tensor):
        '''
        Input:
            x: tensor of shape (B, C, H, W)
        '''
        x = x.to(self.device)
        recons, x, mu, log_var = self.forward_gaussian(x)
        recons_loss = F.mse_loss(recons, x)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - torch.square(mu) - torch.exp(log_var), dim = 1), dim = 0)
        loss = recons_loss + self.beta * kld_loss
        return {'loss': loss, 'Reconstruction_Loss': recons_loss, 'KLD': kld_loss}
    
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