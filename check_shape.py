from arch.beta_vae import BetaVAE
from arch.wae_gan import WAEGAN
from arch.wgan import WGANGP

def get_num_params(model):
    return sum(p.numel() for p in model.parameters())
betavae = BetaVAE(feat_size=(1, 32, 32),
                    latent_dim=32,
                    hidden_dims = [32, 32, 64, 64],
                    beta = 1e-3,)
print(betavae)
print(f"beta vae mnist: {get_num_params(betavae)}")
waegan = WAEGAN(feat_size=(1, 32, 32),
                    latent_dim = 8,
                    hidden_dims = [32, 32, 64, 64],
                    discriminator_hidden = [64] * 4,
                    beta = 1,)
print(waegan)
print(f"wae gan mnist: {get_num_params(waegan)}")
wgan = WGANGP(feat_size = (1, 32, 32),
                    latent_dim=128,
                    generator_hidden = [512, 256, 128],
                    discriminator_hidden = [128, 256, 512],
                    beta = 10,)
print(wgan)
print(f"wgan mnist: {get_num_params(wgan)}")

betavae = BetaVAE(feat_size=(3, 64, 64),
                    latent_dim=32,
                    hidden_dims = [32, 32, 64, 64],
                    beta = 1e-3,)
print(betavae)
print(f"beta vae celeba: {get_num_params(betavae)}")
waegan = WAEGAN(feat_size=(3, 64, 64),
                    latent_dim = 64,
                    hidden_dims = [128, 256, 512, 1024],
                    discriminator_hidden = [512] * 4,
                    beta = 1,)
print(waegan)
print(f"wae gan celeba: {get_num_params(waegan)}")
wgan = WGANGP(feat_size = (3, 64, 64),
                    latent_dim=128,
                    generator_hidden = [1024, 512, 256, 128],
                    discriminator_hidden = [128, 256, 512, 1024],
                    beta = 10,)
print(wgan)
print(f"wgan celeba: {get_num_params(wgan)}")