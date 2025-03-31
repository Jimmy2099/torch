import torch

USE_CUDA_IF_AVAILABLE = True
use_cuda = USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()

from vae_model import VAE, IMAGE_SIZE, CHANNELS_IMG, LATENT_DIM

DEVICE = torch.device("cuda" if use_cuda else "cpu")
model = VAE(channels_img=CHANNELS_IMG, latent_dim=LATENT_DIM, image_size=IMAGE_SIZE).to(DEVICE)

# TODO load_state_dict


print(model)
import numpy as np
import os

try:
    os.mkdir("data")
except Exception as e:
    pass
for name, param in model.named_parameters():
    # if name == "embedder.weight":
    #     continue
    param_np = param.detach().cpu().numpy()
    print(f"Layer: {name}, Shape: {param.shape}, dim: {param_np.ndim}")

    # 处理四维张量（卷积层）
    if param_np.ndim == 4:
        # 重新调整形状为二维: (输出通道数, 输入通道数 * 卷积核高度 * 卷积核宽度)
        param_np = param_np.reshape(param_np.shape[0], -1)

    # 保存参数
    np.savetxt(f"./data/{name}.csv", param_np, delimiter=",", fmt="%.16f")

print("所有参数已保存。")
