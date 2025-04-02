import os
import time

import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image


def main():
    # 基础配置
    data_dir = "data"
    batch_size = 64
    latent_dim = 64
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(time.time()))

    # ====================== 手动构建网络层 ======================
    # 全连接层 (修正输入输出方向)
    fc_layer = nn.Linear(64, 8192).to(device)  # in_features=64, out_features=8192

    # 转置卷积层
    convT_layers = [
        nn.ConvTranspose2d(512, 256, 5, 2, 2, output_padding=1).to(device),
        nn.ConvTranspose2d(256, 128, 5, 2, 2, output_padding=1).to(device),
        nn.ConvTranspose2d(128, 64, 5, 2, 2, output_padding=1).to(device),
        nn.ConvTranspose2d(64, 3, 5, 2, 2, output_padding=1).to(device)
    ]

    # BatchNorm层
    bn_layers = [
        nn.BatchNorm2d(256).to(device),
        nn.BatchNorm2d(128).to(device),
        nn.BatchNorm2d(64).to(device)
    ]

    # Tanh激活
    tanh = nn.Tanh().to(device)

    # ====================== 参数加载 ======================
    # 1. 全连接层参数（关键修正）
    fc_weight = np.loadtxt(os.path.join(data_dir, "decoder_fc.weight.csv"), delimiter=",")
    print("原始FC权重维度:", fc_weight.shape)  # 应为 (8192, 64)

    fc_layer.weight.data = torch.tensor(
        fc_weight,
        dtype=torch.float32
    ).to(device)

    fc_layer.bias.data = torch.tensor(
        np.loadtxt(os.path.join(data_dir, "decoder_fc.bias.csv"), delimiter=","),
        dtype=torch.float32
    ).to(device)

    # 2. 转置卷积参数加载（保持不变）
    convT_indices = [0, 3, 6, 9]
    for i, idx in enumerate(convT_indices):
        weight = np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.weight.csv"), delimiter=",")
        convT_layers[i].weight.data = torch.tensor(
            weight.reshape(convT_layers[i].in_channels, convT_layers[i].out_channels, 5, 5),
            dtype=torch.float32
        ).to(device)

        bias = np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.bias.csv"), delimiter=",")
        convT_layers[i].bias.data = torch.tensor(bias, dtype=torch.float32).to(device)

    # 3. BN参数加载（保持不变）
    bn_indices = [1, 4, 7]
    for i, idx in enumerate(bn_indices):
        bn_layers[i].weight.data = torch.tensor(
            np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.weight.csv"), delimiter=","),
            dtype=torch.float32
        ).to(device)

        bn_layers[i].bias.data = torch.tensor(
            np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.bias.csv"), delimiter=","),
            dtype=torch.float32
        ).to(device)

        bn_layers[i].running_mean.data = torch.tensor(
            np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.running_mean.csv"), delimiter=","),
            dtype=torch.float32
        ).to(device)

        bn_layers[i].running_var.data = torch.tensor(
            np.loadtxt(os.path.join(data_dir, f"decoder_conv.{idx}.running_var.csv"), delimiter=","),
            dtype=torch.float32
        ).to(device)

    # ====================== 验证维度 ======================
    print("\n=== 维度验证 ===")
    print("全连接层权重维度:", fc_layer.weight.shape)  # 应为 torch.Size([8192, 64])
    print("输入噪声维度:", torch.randn(batch_size, latent_dim).shape)  # 应为 torch.Size([64, 64])

    # ====================== 生成图像 ======================
    with torch.no_grad():
        z = torch.randn(batch_size, latent_dim).to(device)
        x = fc_layer(z)
        x = x.view(-1, 512, 4, 4)
        print(f"Shape after view: {x.shape}")  # 初始形状

        # 手动展开循环并打印每一步的shape
        # 第0层
        x = convT_layers[0](x)
        print(f"Shape after convT layer 0: {x.shape}")
        x = bn_layers[0](x)
        x = torch.relu(x)

        # 第1层
        x = convT_layers[1](x)
        print(f"Shape after convT layer 1: {x.shape}")
        x = bn_layers[1](x)
        x = torch.relu(x)

        # 第2层
        x = convT_layers[2](x)
        print(f"Shape after convT layer 2: {x.shape}")
        x = bn_layers[2](x)
        x = torch.relu(x)

        # 第3层（无BN和ReLU）
        x = convT_layers[3](x)
        print(f"Shape after convT layer 3: {x.shape}")

        output = tanh(x)
        output = (output * 0.5 + 0.5).clamp(0, 1)
        save_image(output, "generated.png", nrow=8)

if __name__ == "__main__":
    main()
