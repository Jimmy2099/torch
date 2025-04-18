import torch
import torch.nn as nn


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

print(model)
import numpy as np
import os

if not os.path.exists("data"):
    os.mkdir("data")

for name, param in model.named_parameters():
    param_np = param.detach().cpu().numpy()
    print(f"Layer: {name}, Shape: {param.shape}, dim: {param_np.ndim}")

    if param_np.ndim == 4:
        param_np = param_np.reshape(param_np.shape[0], -1)

    np.savetxt(f"./data/{name}.csv", param_np, delimiter=",", fmt="%.16f")

print("所有参数已保存。")
