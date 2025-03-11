from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


trainset = datasets.MNIST(root='./dataset', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)