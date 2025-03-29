import os

import torch
from torchvision.utils import save_image

from vae_model import VAE

IMAGE_SIZE = 64
CHANNELS_IMG = 3
DEFAULT_LATENT_DIM = 128
model_path = './trained_vae.pth'
output_dir = 'generated_images_hardcoded'
num_images = 64
latent_dim = DEFAULT_LATENT_DIM  # Use imported default or set specific value (must match trained model!)
use_cuda_if_available = True  # Set to False to force CPU
seed = 42  # Set to None for random seed, or an integer for reproducible results
grid_rows = 8
output_name = 'generated_anime_faces_hardcoded.png'

use_cuda = use_cuda_if_available and torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

if seed is not None:
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed(seed)
    print(f"Using fixed random seed: {seed}")
else:
    print("Using random seed.")

print(f"Using device: {DEVICE}")
print(f"Loading model with Latent Dim: {latent_dim}")  # Use the configured latent_dim

# --- 创建输出目录 ---
os.makedirs(output_dir, exist_ok=True)

# --- 加载模型 ---
# 初始化模型结构 (确保 latent_dim 与训练时一致)
model = VAE(channels_img=CHANNELS_IMG, latent_dim=latent_dim, image_size=IMAGE_SIZE).to(
    DEVICE)  # Use the configured latent_dim

if model_path == './trained_vae.pth':
    print("=" * 50)
    print("ERROR: Please update the 'model_path' variable in the script")
    print("       to point to your trained VAE model (.pth file).")
    print("=" * 50)
    exit()

try:
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print(f"Model weights loaded successfully from {model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    # 常见错误：latent_dim 不匹配会导致 size mismatch
    print(
        f"Ensure the 'latent_dim' variable ({latent_dim}) in the script matches the dimension used for training the loaded model.")
    exit()

model.eval()

# --- 生成图像 ---
with torch.no_grad():  # 不需要计算梯度
    # 从标准正态分布采样潜在向量
    noise = torch.randn(num_images, latent_dim).to(DEVICE)

    # 使用解码器生成图像
    generated_images = model.decode(noise).cpu()  # 将结果移回 CPU

# --- 保存生成的图像网格 ---
output_path = os.path.join(output_dir, output_name)
save_image(generated_images, output_path, nrow=grid_rows)
print(f"Generated image grid saved to {output_path}")
