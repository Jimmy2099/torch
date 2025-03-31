import numpy as np
import torch
from torchvision.utils import save_image


def predict_plot(data_file,NUM_IMAGES=64):
    print(f"Attempting to load: {data_file}")
    image_data = np.loadtxt(data_file, delimiter=',', dtype=np.float32)
    image_data = torch.from_numpy(image_data)
    image_data = image_data.reshape(NUM_IMAGES, 3, 64, 64)
    print(f"Shape: {image_data.shape}, dim: {image_data.ndim}")
    nrow = int(NUM_IMAGES ** 0.5)
    if nrow * nrow < NUM_IMAGES:
        nrow += 1
    save_image(image_data, "test.png", nrow=nrow)

if __name__ == "__main__":
    predict_plot("test.csv",NUM_IMAGES=64)
