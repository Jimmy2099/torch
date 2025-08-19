import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random
import os
import csv
from PIL import Image

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 加载测试集
testset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
# 注意这里用的是 testset 而非 test_loader，因为我们只抽取指定索引

# 随机抽取 10 张图片
indices = random.sample(range(len(testset)), 10)

# 创建保存图片的目录
save_dir = './mnist_images'
os.makedirs(save_dir, exist_ok=True)

# CSV 文件路径，用于保存图片文件名和标签的对应关系
csv_file = os.path.join(save_dir, 'labels.csv')
import numpy as np
# 打开 CSV 文件写入数据
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label'])  # CSV 表头
    for i, idx in enumerate(indices):
        image, label = testset[idx]
        # image: torch.Tensor, shape (1, H, W)
        # 将 tensor 转为 PIL image（注意 MNIST 是单通道）
        image_pil = transforms.ToPILImage()(image)
        filename = f'image_{i}.png'
        filepath = os.path.join(save_dir, filename)
        # 保存图片
        image_pil.save(filepath)
        np.savetxt(filepath+".csv",  image.squeeze().numpy(), delimiter=",", fmt="%.16f")
        # 写入 CSV 行
        writer.writerow([filename, label])
        print(f"Saved {filename} with label {label}")
