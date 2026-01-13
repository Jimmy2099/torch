import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import random
import os
import csv
from PIL import Image

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

testset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)

indices = random.sample(range(len(testset)), 10)

save_dir = './mnist_images'
os.makedirs(save_dir, exist_ok=True)

csv_file = os.path.join(save_dir, 'labels.csv')
import numpy as np
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'label'])
    for i, idx in enumerate(indices):
        image, label = testset[idx]
        image_pil = transforms.ToPILImage()(image)
        filename = f'image_{i}.png'
        filepath = os.path.join(save_dir, filename)
        image_pil.save(filepath)
        np.savetxt(filepath+".csv",  image.squeeze().numpy(), delimiter=",", fmt="%.16f")
        writer.writerow([filename, label])
        print(f"Saved {filename} with label {label}")
