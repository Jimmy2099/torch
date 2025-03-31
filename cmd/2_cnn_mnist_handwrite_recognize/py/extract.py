import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载测试集
testset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
test_loader = DataLoader(testset, batch_size=1, shuffle=False)

# CNN 模型
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(in_features=64 * 7 * 7, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=10)
        self.relu = torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
model.load_state_dict(torch.load('mnist_cnn.pth'))
model.eval()


import os
try:
    os.mkdir("data")
except Exception:
    pass

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
