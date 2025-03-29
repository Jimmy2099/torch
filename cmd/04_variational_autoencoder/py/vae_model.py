import torch
import torch.nn as nn
import torch.nn.functional as F

# --- 共享的模型/数据参数 ---
# 这些参数需要在训练和生成时保持一致
IMAGE_SIZE = 64       # 图像尺寸
CHANNELS_IMG = 3      # 图像通道数 (RGB)
LATENT_DIM = 128      # 潜在空间维度 (需要与训练时一致)

class VAE(nn.Module):
    def __init__(self, channels_img=CHANNELS_IMG, latent_dim=LATENT_DIM, image_size=IMAGE_SIZE):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.image_size = image_size

        # --- 编码器 ---
        modules_enc = []
        hidden_dims = [32, 64, 128, 256] # 可以调整卷积层的通道数

        in_channels = channels_img
        current_size = image_size
        for h_dim in hidden_dims:
            modules_enc.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1), # Stride=2 halves spatial dimensions
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
            current_size //= 2 # Calculate final spatial size

        self.encoder = nn.Sequential(*modules_enc)

        # 计算展平后的维度
        self.flattened_size = hidden_dims[-1] * (current_size ** 2)

        self.fc_mu = nn.Linear(self.flattened_size, latent_dim)
        self.fc_log_var = nn.Linear(self.flattened_size, latent_dim)

        # --- 解码器 ---
        modules_dec = []
        self.decoder_input = nn.Linear(latent_dim, self.flattened_size)

        hidden_dims.reverse() # [256, 128, 64, 32]

        in_channels = hidden_dims[0] # Start with 256
        for i in range(len(hidden_dims) - 1):
            modules_dec.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels,
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
            in_channels = hidden_dims[i+1]
            current_size *= 2 # Track spatial size increase

        # 最后一个转置卷积层，输出通道为 channels_img
        modules_dec.append(
            nn.Sequential(
                nn.ConvTranspose2d(hidden_dims[-1], # Input channels = 32
                                   channels_img,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1,
                                   output_padding=1), # Should bring size back to image_size
                nn.Sigmoid()) # 输出范围 [0, 1]
        )
        current_size *= 2 # Final size check

        # 断言确保解码器最终输出尺寸与输入图像尺寸一致
        assert current_size == self.image_size, f"Decoder output size {current_size} does not match IMAGE_SIZE {self.image_size}"


        self.decoder_conv = nn.Sequential(*modules_dec)
        # 保存解码器输入卷积前的期望形状
        self.decoder_reshape_dims = (hidden_dims[0], image_size // (2**len(hidden_dims)), image_size // (2**len(hidden_dims)))


    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1) # 展平
        mu = self.fc_mu(result)
        log_var = self.fc_log_var(result)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std) # 从 N(0, I) 采样 epsilon
        return mu + eps * std

    def decode(self, z):
        result = self.decoder_input(z)
        # Reshape 回卷积层需要的形状
        result = result.view(-1, *self.decoder_reshape_dims)
        result = self.decoder_conv(result)
        return result

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        reconstruction = self.decode(z)
        return reconstruction, mu, log_var

# --- VAE 损失函数 (也放在这里方便共享，虽然主要在训练用) ---
def vae_loss_function(recon_x, x, mu, log_var, kld_weight=1.0, image_size=IMAGE_SIZE, channels_img=CHANNELS_IMG):
    # 重建损失 (BCE)
    recon_loss = F.binary_cross_entropy(recon_x.view(-1, image_size*image_size*channels_img),
                                        x.view(-1, image_size*image_size*channels_img),
                                        reduction='sum')

    # KL 散度损失 D_KL( N(mu, var) || N(0, 1) )
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    # 加权总损失
    loss = recon_loss + kld_weight * kld_loss

    # 平均到每个样本
    loss = loss / x.size(0)
    recon_loss = recon_loss / x.size(0)
    kld_loss = kld_loss / x.size(0)

    return loss, recon_loss, kld_loss