import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=128)  # 输入层到隐藏层
        self.fc2 = nn.Linear(128, 10)   # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleNN()

# 打印fc1的权重和偏置的形状
print("fc1 权重 (weights) 的形状:", model.fc1.weight.shape)  # 打印fc1的权重形状
print("fc1 偏置 (bias) 的形状:", model.fc1.bias.shape)      # 打印fc1的偏置形状

# 查看fc1的权重和偏置的元素数量
print("fc1 权重 (weights) 的元素数量:", model.fc1.weight.numel())
print("fc1 偏置 (bias) 的元素数量:", model.fc1.bias.numel())


# 导出 fc1 的权重和偏置
fc1_weight = model.fc1.weight.detach().numpy()
fc1_bias = model.fc1.bias.detach().numpy()

import numpy as np
# 保存权重和偏置为 CSV 文件
np.savetxt("fc1_weight.csv", fc1_weight, delimiter=",")
np.savetxt("fc1_bias.csv", fc1_bias, delimiter=",")