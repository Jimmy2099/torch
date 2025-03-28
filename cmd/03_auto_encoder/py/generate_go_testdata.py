import torch
from torchvision import transforms, datasets
# from torch.utils.data import DataLoader # DataLoader is not needed here as we sample directly
import random
import os
import csv
from PIL import Image
import numpy as np # Make sure numpy is imported

# --- Configuration ---
NOISE_FACTOR = 0.25 # Intensity of Gaussian noise to add (adjust as needed)
NUM_IMAGES = 10     # Number of images to sample and save
SAVE_DIR = './mnist_noisy_images' # Directory to save noisy images and CSVs
DATA_ROOT = './data' # Directory to download/load MNIST dataset
# --- End Configuration ---

print("Starting script to generate noisy MNIST samples...")

# 1. Data Preprocessing
#    - ToTensor(): Converts PIL Image or numpy.ndarray to FloatTensor, scales pixels to [0, 1].
#    - We do NOT Normalize here, as adding noise and clipping works best in the [0, 1] range,
#      and ToPILImage() for saving also expects [0, 1].
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 2. Load MNIST Test Set
try:
    print(f"Loading MNIST dataset from '{DATA_ROOT}'...")
    # Use train=False for the test set
    testset = datasets.MNIST(root=DATA_ROOT, train=False, download=True, transform=transform)
    print(f"MNIST dataset loaded successfully. Size: {len(testset)}")
except Exception as e:
    print(f"Error loading/downloading MNIST dataset: {e}")
    exit() # Exit if dataset cannot be loaded

# 3. Randomly Sample Indices
if len(testset) < NUM_IMAGES:
    print(f"Warning: Requested {NUM_IMAGES} images, but dataset only has {len(testset)}. Using all available.")
    indices = list(range(len(testset)))
    NUM_IMAGES = len(testset) # Adjust the number to actual available count
else:
    # Sample unique random indices from the range of the dataset length
    indices = random.sample(range(len(testset)), NUM_IMAGES)
print(f"Selected {len(indices)} random indices.")

# 4. Create Save Directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)
print(f"Ensured save directory exists: '{SAVE_DIR}'")

# 5. Prepare CSV file path for labels and filenames
csv_label_file = os.path.join(SAVE_DIR, 'labels.csv')

# 6. Process and Save Images with Noise
print(f"Processing {NUM_IMAGES} images, adding noise (factor={NOISE_FACTOR}), and saving...")
try:
    with open(csv_label_file, mode='w', newline='') as f:
        # Setup CSV writer and write header
        writer = csv.writer(f)
        # --- MODIFICATION: Changed header to only include filename and label ---
        writer.writerow(['filename', 'label'])

        # Loop through the selected indices
        for i, idx in enumerate(indices):
            # Get the original image tensor and its label
            original_image, label = testset[idx]
            # original_image: torch.Tensor, shape (1, 28, 28), range [0, 1]

            # --- Add Gaussian Noise ---
            # Generate noise with the same shape as the image
            noise = NOISE_FACTOR * torch.randn_like(original_image)
            # Add noise to the original image
            noisy_image = original_image + noise
            # Clip the values to stay within the valid pixel range [0, 1]
            noisy_image = torch.clip(noisy_image, 0., 1.)
            # --- Noise Added ---

            # Define filenames for the noisy image (PNG) and its tensor data (CSV)
            base_filename = f'image_{i}' # Base name for the i-th sampled image
            noisy_png_filename = f'{base_filename}.png'
            noisy_csv_filename = f'{base_filename}.png.csv' # Still used for saving tensor

            # Get the full paths for saving
            noisy_png_filepath = os.path.join(SAVE_DIR, noisy_png_filename)
            noisy_csv_filepath = os.path.join(SAVE_DIR, noisy_csv_filename) # Still used for saving tensor

            # Convert the noisy tensor to a PIL Image for saving as PNG
            # transforms.ToPILImage() expects CxHxW tensor in [0, 1] range.
            noisy_image_pil = transforms.ToPILImage()(noisy_image)

            # Save the noisy image as a PNG file
            noisy_image_pil.save(noisy_png_filepath)

            # Save the noisy image tensor data (pixel values) as a CSV file
            # .squeeze() removes the channel dimension (1, 28, 28) -> (28, 28) for saving 2D data
            # fmt="%.8f" saves floats with 8 decimal places (adjust if needed)
            # Note: We still save this file, just don't list it in the main CSV
            np.savetxt(noisy_csv_filepath, noisy_image.squeeze().numpy(), delimiter=",", fmt="%.8f")

            # --- MODIFICATION: Changed row data to only include png filename and label ---
            # Write the PNG filename and the corresponding label to the main CSV file
            writer.writerow([noisy_png_filename, label])

            # Print progress for each saved image (still mentioning the tensor file for clarity)
            print(f"  Saved: {noisy_png_filename} (Label: {label}), Tensor: {noisy_csv_filename}")

except IOError as e:
    print(f"Error writing files: {e}")
except Exception as e:
    print(f"An unexpected error occurred during processing: {e}")

print(f"\nFinished saving {NUM_IMAGES} noisy images and metadata.")
print(f"Noisy PNG images and their tensor CSVs are in: '{SAVE_DIR}'")
print(f"Label mapping (filename, label) is in: '{csv_label_file}'")