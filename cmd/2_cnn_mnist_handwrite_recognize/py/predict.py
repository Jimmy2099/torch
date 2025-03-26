import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

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
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 128)
        self.fc2 = torch.nn.Linear(128, 10)
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

# 预测
def predict():
    with torch.no_grad():
        for i, (image, _) in enumerate(test_loader):
            image = image.to(device)
            output = model(image)
            _, predicted = torch.max(output, 1)
            print(f'Image {i + 1}: Predicted Digit: {predicted.item()}')

if __name__ == "__main__":
    predict()