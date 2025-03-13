# https://github.com/igul222/improved_wgan_training/blob/master/gan_cifar.py
import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.func import vmap, jacrev

class WGANGP(nn.Module):
    def __init__(self, 
                 feat_size: Tuple[int, int, int],
                 latent_dim: int=128,
                 generator_hidden: List = [1024, 512, 256, 128],
                 discriminator_hidden: List = [128, 256, 512, 1024],
                 beta: float = 1):
        super(WGANGP, self).__init__()
        self.feat_size = feat_size
        self.latent_dim = latent_dim
        self.beta = beta # lambda in the paper, gradient penalty

        self.generator_hidden = generator_hidden
        self.discriminator_hidden = discriminator_hidden

        self.build_generator()
        self.build_discriminator()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def build_generator(self):
        self.init_image_size = self.feat_size[1] // 2 ** len(self.generator_hidden) * 2
        n_feats = self.init_image_size*self.init_image_size*self.generator_hidden[0]
        self.generator_input = nn.Sequential(
            nn.Linear(self.latent_dim, n_feats),
            nn.BatchNorm1d(n_feats),
            nn.ReLU(),
        )

        modules = []
        in_feats = self.generator_hidden[0]
        for h_dim in self.generator_hidden[1:]:
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
                nn.Conv2d(self.generator_hidden[-1], self.feat_size[0], kernel_size=3, padding=1)
            )
        )
        self.generator = nn.Sequential(*modules)
        self.generator_modules = modules
            
    def generate(self, z: torch.Tensor, debug=False):
        '''
        Input:
            z: (B, latent_dim) noise
        Given the noise, generate the image of shape (B, C, H, W)
        '''
        B = z.shape[0]
        generator_input = self.generator_input(z)
        generator_input = generator_input.reshape((B, self.generator_hidden[0], self.init_image_size, self.init_image_size))
        if debug:
            x = generator_input
            for m in self.generator_modules:
                x = m(x)
            return x
        return self.generator(generator_input)

    def build_discriminator(self):
        modules = []
        in_feats = self.feat_size[0]
        im_size = self.feat_size[1] # assuming H=W
        # Build Encoder
        for h_dim in self.discriminator_hidden:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_feats, h_dim, kernel_size=4, stride=2, padding=1), # down sample
                    nn.LeakyReLU(),
                )
            )
            in_feats = h_dim
            im_size = im_size // 2
        modules.append(nn.Flatten(start_dim=-3, end_dim=-1))
        modules.append(nn.Linear(self.discriminator_hidden[-1] * im_size * im_size, 1))
        self.discriminator = nn.Sequential(*modules)
        self.discriminator_modules = modules

    def discriminate(self, x: torch.Tensor, debug=False):
        '''
        Input:
            x: (B, C, H, W) generated images
        Given the generated images, compute the discrimination score
        '''
        if debug:
            for m in self.discriminator_modules:
                x = m(x)
            return x
        return self.discriminator(x)

    def calculate_gradient_penalty(self, real: torch.Tensor, fake: torch.Tensor):
        '''
        Input:
            real, fake: images of shape (B, C, H, W)
        '''
        B, C, H, W = real.shape
        eta = torch.rand((B, 1, 1, 1), device=self.device).expand(B, C, H, W)
        interpolated = (1 - eta) * real + eta * fake # (B, C, H, W)

        grads = vmap(jacrev(self.discriminate))(interpolated)
        grad_norm = torch.norm(grads.reshape((B, -1)), p=2, dim=1, keepdim=True)

        # NOTE: A modification from the original paper, we accept all gradients <=1, so we add a relu
        # only when grad_norm > 1, we penalize
        grad_penalty = torch.nn.functional.relu(grad_norm - 1.) # (B, 1)
        return torch.mean(torch.square(grad_penalty))
    
    def sample_noise(self, B: int):
        return torch.randn((B, self.latent_dim), device=self.device)
    
    def set_param_require_grad(self, module: nn.Module, require_grad: bool = False):
        for p in module.parameters():
            p.requires_grad_(require_grad)

    def sample(self, num_samples: int) -> torch.Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map
        """
        z = self.sample_noise(num_samples)
        return self.generate(z)
    
    def train_step(self, x: torch.Tensor, 
                    optimizer_dis: torch.optim.Optimizer,
                    optimizer_gen: torch.optim.Optimizer,
                    generator_step: bool=False,
                    debug = False):
        '''
        Input:
            x: tensor of shape (B, C, H, W)
        '''
        # critic step, optimize discriminator/kantorovich potential
        self.set_param_require_grad(self.discriminator, True)
        self.set_param_require_grad(self.generator, False)
        x = x.to(self.device)
        z = self.sample_noise(x.shape[0])

        optimizer_dis.zero_grad()
        fake = self.generate(z, debug)
        gp = self.calculate_gradient_penalty(x, fake)

        d_real = self.discriminate(x, debug) # Kantorovich potential
        d_fake = self.discriminate(fake, debug)  # Kantorovich potential

        dual_kantorovich = torch.mean(d_fake) - torch.mean(d_real)
        gen_loss = -torch.mean(d_fake)

        total_loss = dual_kantorovich + self.beta * gp
        total_loss.backward()
        optimizer_dis.step()

        if generator_step:
            self.set_param_require_grad(self.discriminator, False)
            self.set_param_require_grad(self.generator, True)
            optimizer_gen.zero_grad()
            fake = self.generate(z, debug)
            d_fake = self.discriminate(fake, debug)  # Kantorovich potential
            gen_loss = -torch.mean(d_fake)
            gen_loss.backward()
            optimizer_gen.step()

        return {
            "loss_dk": dual_kantorovich,
            "loss_gp": gp,
            "loss_gen": gen_loss,
            "total_loss": total_loss
        }

    def test_step(self, x: torch.Tensor):
        '''
        Input:
            x: tensor of shape (B, C, H, W)
        '''
        x = x.to(self.device)
        z = self.sample_noise(x.shape[0])
        fake = self.generate(z)
        gp = self.calculate_gradient_penalty(x, fake)

        d_real = self.discriminate(x) # Kantorovich potential
        d_fake = self.discriminate(fake)  # Kantorovich potential

        dual_kantorovich = torch.mean(d_fake) - torch.mean(d_real)
        gen_loss = -torch.mean(d_fake)

        total_loss = dual_kantorovich + self.beta * gp
        total_loss.backward()
        return {
            "loss_dk": dual_kantorovich,
            "loss_gp": gp,
            "loss_gen": gen_loss,
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
            dict_to_load = torch.load(f, weights_only=False, map_location=self.device)
        self.load_state_dict(dict_to_load["model"])

