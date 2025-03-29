import torch
from torchvision.utils import save_image, make_grid
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

# 从模型文件中导入 VAE 类和共享参数
from vae_model import VAE, IMAGE_SIZE, CHANNELS_IMG, LATENT_DIM # 导入默认 LATENT_DIM 以防命令行未指定

# --- 生成参数 ---
parser = argparse.ArgumentParser(description='Generate Anime Faces using a trained VAE')
parser.add_argument('--model-path', type=str, required=True, help='Path to the trained VAE model (.pth file)')
parser.add_argument('--output-dir', type=str, default='generated_images', help='Directory to save generated images')
parser.add_argument('--num-images', type=int, default=64, help='Number of images to generate')
parser.add_argument('--latent-dim', type=int, default=LATENT_DIM, help='Dimension of the latent space (must match trained model)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA usage')
parser.add_argument('--seed', type=int, default=None, help='Random seed for generation (optional)')
parser.add_argument('--grid-rows', type=int, default=8, help='Number of rows in the output image grid')
parser.add_argument('--output-name', type=str, default='generated_anime_faces.png', help='Filename for the output grid image')
parser.add_argument('--show', action='store_true', default=False, help='Show the generated grid using matplotlib')


args = parser.parse_args()

# --- 设置设备 ---
use_cuda = not args.no_cuda and torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")

if args.seed is not None:
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)
    print(f"Using random seed: {args.seed}")

print(f"Using device: {DEVICE}")
print(f"Loading model with Latent Dim: {args.latent_dim}")

# --- 创建输出目录 ---
os.makedirs(args.output_dir, exist_ok=True)

# --- 加载模型 ---
# 初始化模型结构 (确保 latent_dim 与训练时一致)
model = VAE(channels_img=CHANNELS_IMG, latent_dim=args.latent_dim, image_size=IMAGE_SIZE).to(DEVICE)

try:
    # 加载训练好的权重
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    print(f"Model weights loaded successfully from {args.model_path}")
except FileNotFoundError:
    print(f"Error: Model file not found at {args.model_path}")
    exit()
except Exception as e:
    print(f"An error occurred while loading the model: {e}")
    # 常见错误：latent_dim 不匹配会导致 size mismatch
    print("Ensure the --latent-dim argument matches the dimension used for training the loaded model.")
    exit()

# 设置为评估模式 (关闭 dropout, batchnorm 使用运行统计数据)
model.eval()

# --- 生成图像 ---
with torch.no_grad(): # 不需要计算梯度
    # 从标准正态分布采样潜在向量
    noise = torch.randn(args.num_images, args.latent_dim).to(DEVICE)

    # 使用解码器生成图像
    generated_images = model.decode(noise).cpu() # 将结果移回 CPU

# --- 保存生成的图像网格 ---
output_path = os.path.join(args.output_dir, args.output_name)
save_image(generated_images, output_path, nrow=args.grid_rows)
print(f"Generated image grid saved to {output_path}")

# --- (可选) 显示图像 ---
if args.show:
    try:
        grid_img = make_grid(generated_images, nrow=args.grid_rows).permute(1, 2, 0).numpy()
        plt.figure(figsize=(10, 10 * (generated_images.size(0) // (args.grid_rows*args.grid_rows)))) # Adjust figure size
        plt.imshow(grid_img)
        plt.title(f'Generated Anime Faces ({args.num_images} images)')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Could not display image using matplotlib: {e}")