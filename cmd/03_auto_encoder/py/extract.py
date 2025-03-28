import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec

# 定义 Autoencoder 模型
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)  # 特征层
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 28 * 28),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoencoder.pth", map_location=device))
model.eval()

print(model)
import numpy as np
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
