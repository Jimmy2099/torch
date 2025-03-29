import torch
from torchvision.utils import save_image, make_grid
import os
# import argparse # No longer needed
import matplotlib.pyplot as plt
import numpy as np

# 从模型文件中导入 VAE 类和共享参数
# Make sure vae_model.py is accessible and contains these definitions
try:
    from vae_model import VAE, IMAGE_SIZE, CHANNELS_IMG, LATENT_DIM as DEFAULT_LATENT_DIM
except ImportError:
    print("Error: Could not import from vae_model.py.")
    print("Please ensure vae_model.py is in the same directory or Python path,")
    print("and contains the VAE class and IMAGE_SIZE, CHANNELS_IMG, LATENT_DIM constants.")
    # Define placeholders if import fails, but this will likely cause errors later
    IMAGE_SIZE = 64
    CHANNELS_IMG = 3
    DEFAULT_LATENT_DIM = 128
    class VAE(torch.nn.Module): # Dummy class for placeholder
        def __init__(self, channels_img, latent_dim, image_size):
            super().__init__()
            print("WARNING: Using dummy VAE class due to import error.")
        def load_state_dict(self, state_dict):
            print("WARNING: Dummy load_state_dict called.")
        def eval(self):
            print("WARNING: Dummy eval called.")
        def decode(self, z):
            print("WARNING: Dummy decode called.")
            # Return dummy tensor of expected shape
            return torch.zeros(z.size(0), CHANNELS_IMG, IMAGE_SIZE, IMAGE_SIZE)


# --- Configuration Parameters ---
# MODIFY THESE VALUES AS NEEDED
# -----------------------------
model_path    = 'path/to/your/trained_vae.pth' # <<< IMPORTANT: SET THIS PATH
output_dir    = 'generated_images_hardcoded'
num_images    = 64
latent_dim    = DEFAULT_LATENT_DIM # Use imported default or set specific value (must match trained model!)
use_cuda_if_available = True     # Set to False to force CPU
seed          = 42               # Set to None for random seed, or an integer for reproducible results
grid_rows     = 8
output_name   = 'generated_anime_faces_hardcoded.png'
show_image    = False            # Set to True to display the image grid using matplotlib
# -----------------------------


# --- 设置设备 ---
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
print(f"Loading model with Latent Dim: {latent_dim}") # Use the configured latent_dim

# --- 创建输出目录 ---
os.makedirs(output_dir, exist_ok=True)

# --- 加载模型 ---
# 初始化模型结构 (确保 latent_dim 与训练时一致)
model = VAE(channels_img=CHANNELS_IMG, latent_dim=latent_dim, image_size=IMAGE_SIZE).to(DEVICE) # Use the configured latent_dim

if model_path == './trained_vae.pth':
    print("="*50)
    print("ERROR: Please update the 'model_path' variable in the script")
    print("       to point to your trained VAE model (.pth file).")
    print("="*50)
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
    print(f"Ensure the 'latent_dim' variable ({latent_dim}) in the script matches the dimension used for training the loaded model.")
    exit()

# 设置为评估模式 (关闭 dropout, batchnorm 使用运行统计数据)
model.eval()

# --- 生成图像 ---
with torch.no_grad(): # 不需要计算梯度
    # 从标准正态分布采样潜在向量
    noise = torch.randn(num_images, latent_dim).to(DEVICE) # Use configured num_images and latent_dim

    # 使用解码器生成图像
    generated_images = model.decode(noise).cpu() # 将结果移回 CPU

# --- 保存生成的图像网格 ---
output_path = os.path.join(output_dir, output_name) # Use configured output_dir and output_name
save_image(generated_images, output_path, nrow=grid_rows) # Use configured grid_rows
print(f"Generated image grid saved to {output_path}")

# --- (可选) 显示图像 ---
if show_image: # Use configured show_image
    try:
        grid_img = make_grid(generated_images, nrow=grid_rows).permute(1, 2, 0).numpy() # Use configured grid_rows
        # Adjust figure size dynamically based on rows and aspect ratio
        aspect_ratio = grid_img.shape[1] / grid_img.shape[0]
        num_cols = (num_images + grid_rows - 1) // grid_rows # Calculate number of columns
        fig_width = 10
        fig_height = fig_width / aspect_ratio * (grid_rows / num_cols) # Adjust height based on aspect ratio and grid layout

        plt.figure(figsize=(fig_width, fig_height))
        plt.imshow(grid_img)
        plt.title(f'Generated Anime Faces ({num_images} images)') # Use configured num_images
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not display image using matplotlib: {e}")