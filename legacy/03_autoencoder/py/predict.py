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

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
model.load_state_dict(torch.load("autoencoder.pth", map_location=device))
model.eval()

transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x.view(-1))])

test_dataset = torchvision.datasets.MNIST(root='./dataset', train=False, transform=transform, download=True)
test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

data, labels = next(iter(test_loader))
data, labels = data.to(device), labels.cpu().numpy()

noise_factor = 0.2
noisy_data = data + noise_factor * torch.randn_like(data)
noisy_data = torch.clip(noisy_data, 0., 1.)

with torch.no_grad():
    encoded, reconstructed = model(noisy_data)

mse_loss = F.mse_loss(reconstructed, noisy_data, reduction='none').mean(dim=1).cpu().numpy()

encoded_features = encoded.cpu().numpy()
pca = PCA(n_components=2)
encoded_2d = pca.fit_transform(encoded_features)

fig = plt.figure(figsize=(14, 5))
gs = gridspec.GridSpec(2, 10, figure=fig, height_ratios=[1, 1])

for i in range(10):
    ax = plt.subplot(gs[0, i])
    ax.imshow(noisy_data[i].cpu().numpy().reshape(28, 28), cmap='gray')
    ax.axis('off')
    if i == 0:
        ax.set_title("Noisy Input", fontsize=10)

    ax = plt.subplot(gs[1, i])
    ax.imshow(reconstructed[i].cpu().numpy().reshape(28, 28), cmap='gray')
    ax.axis('off')
    if i == 0:
        ax.set_title("Reconstructed", fontsize=10)

plt.tight_layout()
plt.show()
