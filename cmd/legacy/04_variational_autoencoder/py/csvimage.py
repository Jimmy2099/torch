import numpy as np
import torch
from torchvision.utils import save_image


def load_tensor_from_csv(file_path):
    with open(file_path, 'r') as f:
        header = f.readline().strip()
        if not header.startswith("Shape,"):
            raise ValueError("Invalid CSV format: missing shape header")

        shape = list(map(int, header.split(",")[1:]))
        data = np.loadtxt(f, delimiter=",")

    flattened = data.flatten()
    return torch.tensor(flattened, dtype=torch.float32).reshape(*shape)


def predict_plot(data_file, NUM_IMAGES=64):
    print(f"Attempting to load: {data_file}")
    image_data = load_tensor_from_csv(data_file).numpy()
    image_data = torch.from_numpy(image_data)
    image_data = image_data.reshape(NUM_IMAGES, 3, 64, 64)
    print(f"Shape: {image_data.shape}, dim: {image_data.ndim}")
    nrow = int(NUM_IMAGES ** 0.5)
    if nrow * nrow < NUM_IMAGES:
        nrow += 1
    save_image(image_data, "test.png", nrow=nrow)


def sv_image(image_data, NUM_IMAGES=64):
    image_data = image_data.reshape(NUM_IMAGES, 3, 64, 64)
    print(f"Shape: {image_data.shape}, dim: {image_data.ndim}")
    nrow = int(NUM_IMAGES ** 0.5)
    if nrow * nrow < NUM_IMAGES:
        nrow += 1
    save_image(image_data, "test.png", nrow=nrow)


if __name__ == "__main__":
    predict_plot("test.csv", NUM_IMAGES=64)
