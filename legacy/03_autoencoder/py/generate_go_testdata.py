import torch
from torchvision import transforms, datasets
import random
import os
import csv
from PIL import Image
import numpy as np

NOISE_FACTOR = 0.25
NUM_IMAGES = 10
SAVE_DIR = './mnist_noisy_images'
DATA_ROOT = './dataset'

print("Starting script to generate noisy MNIST samples...")

transform = transforms.Compose([
    transforms.ToTensor(),
])

try:
    print(f"Loading MNIST dataset from '{DATA_ROOT}'...")
    testset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    print(f"MNIST dataset loaded successfully. Size: {len(testset)}")
except Exception as e:
    print(f"Error loading/downloading MNIST dataset: {e}")
    exit()

if len(testset) < NUM_IMAGES:
    print(f"Warning: Requested {NUM_IMAGES} images, but dataset only has {len(testset)}. Using all available.")
    indices = list(range(len(testset)))
    NUM_IMAGES = len(testset)
else:
    indices = random.sample(range(len(testset)), NUM_IMAGES)
print(f"Selected {len(indices)} random indices.")

os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Ensured save directory exists: '{SAVE_DIR}'")

csv_label_file = os.path.join(SAVE_DIR, 'labels.csv')

print(f"Processing {NUM_IMAGES} images, adding noise (factor={NOISE_FACTOR}), and saving...")
try:
    with open(csv_label_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'label'])

        for i, idx in enumerate(indices):
            original_image, label = testset[idx]

            noise = NOISE_FACTOR * torch.randn_like(original_image)
            noisy_image = original_image + noise
            noisy_image = torch.clip(noisy_image, 0., 1.)

            base_filename = f'image_{i}'
            noisy_png_filename = f'{base_filename}.png'
            noisy_csv_filename = f'{base_filename}.png.csv'

            noisy_png_filepath = os.path.join(SAVE_DIR, noisy_png_filename)
            noisy_csv_filepath = os.path.join(SAVE_DIR, noisy_csv_filename) # Still used for saving tensor

            noisy_image_pil = transforms.ToPILImage()(noisy_image)

            noisy_image_pil.save(noisy_png_filepath)

            np.savetxt(noisy_csv_filepath, noisy_image.squeeze().numpy(), delimiter=",", fmt="%.8f")

            writer.writerow([noisy_png_filename, label])

            print(f"  Saved: {noisy_png_filename} (Label: {label}), Tensor: {noisy_csv_filename}")

except IOError as e:
    print(f"Error writing files: {e}")
except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")

print(f"\nFinished saving {NUM_IMAGES} noisy images and metadata.")
print(f"Noisy PNG images and their tensor CSVs are in: '{SAVE_DIR}'")
print(f"Label mapping (filename, label) is in: '{csv_label_file}'")