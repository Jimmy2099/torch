# vae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Shared Parameters ---
# Define these here so the main script can import them if needed,
# or just ensure consistency between this file and the main script.
IMAGE_SIZE = 64
CHANNELS_IMG = 3
LATENT_DIM = 64  # Match the TF example


class VAE(nn.Module):
    def __init__(self, channels_img=CHANNELS_IMG, latent_dim=LATENT_DIM, image_size=IMAGE_SIZE):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size
        ef_dim = 64  # Encoder filter dimension base (matches gf_dim in TF decoder)

        # --- Encoder --- similar to conv_anime_encoder
        encoder_layers = []
        # Input: (batch_size, channels_img, image_size, image_size) -> (N, 3, 64, 64)
        encoder_layers.append(nn.Conv2d(channels_img, ef_dim, kernel_size=5, stride=2, padding=2))  # -> (N, 64, 32, 32)
        encoder_layers.append(nn.BatchNorm2d(ef_dim))
        encoder_layers.append(nn.ReLU(True))

        encoder_layers.append(nn.Conv2d(ef_dim, ef_dim * 2, kernel_size=5, stride=2, padding=2))  # -> (N, 128, 16, 16)
        encoder_layers.append(nn.BatchNorm2d(ef_dim * 2))
        encoder_layers.append(nn.ReLU(True))

        encoder_layers.append(
            nn.Conv2d(ef_dim * 2, ef_dim * 4, kernel_size=5, stride=2, padding=2))  # -> (N, 256, 8, 8)
        encoder_layers.append(nn.BatchNorm2d(ef_dim * 4))
        encoder_layers.append(nn.ReLU(True))

        encoder_layers.append(
            nn.Conv2d(ef_dim * 4, ef_dim * 8, kernel_size=5, stride=2, padding=2))  # -> (N, 512, 4, 4)
        encoder_layers.append(nn.BatchNorm2d(ef_dim * 8))
        encoder_layers.append(nn.ReLU(True))

        self.encoder_conv = nn.Sequential(*encoder_layers)

        # Flatten and final layers for mean and logvar
        self.flatten = nn.Flatten()
        # Calculate the flattened size after conv layers: ef_dim*8 * (image_size//16) * (image_size//16)
        encoder_output_size = ef_dim * 8 * (image_size // 16) * (image_size // 16)  # 512 * 4 * 4 = 8192
        self.fc_mu = nn.Linear(encoder_output_size, latent_dim)
        self.fc_logvar = nn.Linear(encoder_output_size, latent_dim)

        # --- Decoder --- similar to conv_anime_decoder
        gf_dim = ef_dim  # Decoder filter dimension base
        decoder_input_size = gf_dim * 8 * (image_size // 16) * (
                    image_size // 16)  # Start size matches encoder output before FC

        self.decoder_fc = nn.Linear(latent_dim, decoder_input_size)

        # Reshape layer will be handled in forward pass
        self.decoder_reshape_channels = gf_dim * 8  # 512
        self.decoder_reshape_size = image_size // 16  # 4

        decoder_layers = []
        # Input: (N, 512, 4, 4)
        decoder_layers.append(nn.ConvTranspose2d(gf_dim * 8, gf_dim * 4, kernel_size=5, stride=2, padding=2,
                                                 output_padding=1))  # -> (N, 256, 8, 8)
        decoder_layers.append(nn.BatchNorm2d(gf_dim * 4))
        decoder_layers.append(nn.ReLU(True))

        decoder_layers.append(nn.ConvTranspose2d(gf_dim * 4, gf_dim * 2, kernel_size=5, stride=2, padding=2,
                                                 output_padding=1))  # -> (N, 128, 16, 16)
        decoder_layers.append(nn.BatchNorm2d(gf_dim * 2))
        decoder_layers.append(nn.ReLU(True))

        decoder_layers.append(nn.ConvTranspose2d(gf_dim * 2, gf_dim, kernel_size=5, stride=2, padding=2,
                                                 output_padding=1))  # -> (N, 64, 32, 32)
        decoder_layers.append(nn.BatchNorm2d(gf_dim))
        decoder_layers.append(nn.ReLU(True))

        # Final layer to get back to image channels, using stride=2 like the TF DeConv layer for the last upsampling step
        decoder_layers.append(nn.ConvTranspose2d(gf_dim, channels_img, kernel_size=5, stride=2, padding=2,
                                                 output_padding=1))  # -> (N, 3, 64, 64)
        # Activation matches TF example and [-1, 1] normalization
        decoder_layers.append(nn.Tanh())

        self.decoder_conv = nn.Sequential(*decoder_layers)

    def encode(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder_fc(z)
        # Reshape: (N, decoder_input_size) -> (N, channels, size, size)
        x = x.view(-1, self.decoder_reshape_channels, self.decoder_reshape_size, self.decoder_reshape_size)
        x = self.decoder_conv(x)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


# Define Loss Function Here (or keep it inline in the main script)
# This version matches the TF logic more closely (summing, then averaging batch)
def vae_loss_function_pytorch(recon_x, x, mu, logvar):
    """
    Computes the VAE loss function.
    loss = Reconstruction loss + KL divergence
    Assumes recon_x and x are in range [-1, 1] due to Tanh activation. Uses MSE.
    """
    # Reconstruction Loss (MSE Loss, summed over pixels and channels, averaged over batch)
    # MSE = F.mse_loss(recon_x, x, reduction='sum') / x.size(0) # Average over batch
    # Simpler: use reduction='mean' which averages over all elements, then multiply by num_elements
    recon_loss = F.mse_loss(recon_x, x, reduction='mean') * x.nelement() / x.size(0)  # Average over batch

    # KL Divergence (summed over latent dimensions, averaged over batch)
    # see Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld = kld / x.size(0)  # Average over batch

    total_loss = recon_loss + kld
    return total_loss, recon_loss, kld
