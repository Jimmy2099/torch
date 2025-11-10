import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

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
        return decoded

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

train_dataset = torchvision.datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

noise_factor = 0.3

epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for data, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        data = data.to(device)
        noisy_data = data + noise_factor * torch.randn_like(data)
        noisy_data = torch.clip(noisy_data, 0., 1.)

        optimizer.zero_grad()
        outputs = model(noisy_data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)

    avg_loss = running_loss / len(train_dataset)
    print(f'Epoch [{epoch+1}/{epochs}], Average Loss: {avg_loss:.4f}')

torch.save(model.state_dict(), "autoencoder.pth")
print("训练完成，模型已保存！")
