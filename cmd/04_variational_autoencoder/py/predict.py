import os
import random
import time

import torch


def save_tensor_to_csv(tensor, file_path):
    with open(file_path, 'w') as f:
        f.write("Shape," + ",".join(map(str, tensor.shape)) + "\n")
        tensor = tensor.reshape(-1, tensor.shape[0])
        np.savetxt(f, tensor.numpy(), delimiter=",", fmt="%.16f")


from vae_model import VAE, IMAGE_SIZE, CHANNELS_IMG, LATENT_DIM

if __name__ == "__main__":
    MODEL_PATH = 'output/vae_anime/models/vae_anime_final.pth'

    OUTPUT_DIR = 'output/vae_generated'
    OUTPUT_NAME = 'random_generated.png'
    NUM_IMAGES = 64
    # SEED = 42
    USE_CUDA_IF_AVAILABLE = True
    # --------------------------------

    use_cuda = USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")

    # torch.manual_seed(SEED)
    torch.manual_seed(int(time.time()))
    # if use_cuda:
    #     torch.cuda.manual_seed(SEED)
    # --------------------------

    print(f"Using device: {DEVICE}")
    print(f"Imported Latent Dim: {LATENT_DIM}")
    print(f"Imported Image Size: {IMAGE_SIZE}")
    print("-" * 20)
    print("Configuration:")
    print(f"  Model Path: {MODEL_PATH}")
    print(f"  Output Dir: {OUTPUT_DIR}")
    print(f"  Output Name: {OUTPUT_NAME}")
    print(f"  Num Images: {NUM_IMAGES}")
    print("  Seed: None (Random each run)")
    print("-" * 20)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = VAE(channels_img=CHANNELS_IMG, latent_dim=LATENT_DIM, image_size=IMAGE_SIZE).to(DEVICE)

    print(f"Loading model from: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at '{MODEL_PATH}'")
        print("Please update the MODEL_PATH variable in the script.")
        exit()

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Loaded model state_dict from checkpoint dictionary.")
            if 'epoch' in checkpoint:
                print(f"(Checkpoint likely from epoch {checkpoint['epoch']})")
        else:
            model.load_state_dict(checkpoint)
            print("Loaded model state_dict directly.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        print("Ensure the model architecture in vae_model.py matches the saved model.")
        exit()

    model.eval()
    import numpy as np

    print(f"Generating {NUM_IMAGES} random images...")
    with torch.no_grad():
        random_noise = torch.randn(NUM_IMAGES, LATENT_DIM).to(DEVICE)
        np.savetxt(f"noise.csv", random_noise, delimiter=",", fmt="%.16f")
        print(len(random_noise.numpy().shape))
        print(f"Shape: {random_noise.numpy().shape}, dim: {random_noise.numpy().ndim}")
        generated_images = model.decode(random_noise).cpu()
        generated_images = generated_images * 0.5 + 0.5
        generated_images = generated_images.clamp(0, 1)

        # output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        timestamp = int(time.time())
        random_suffix = random.randint(0, 9999)
        output_filename = f"random_generated_{timestamp}_{random_suffix:04d}.png"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        # ---------------------------------

    nrow = int(NUM_IMAGES ** 0.5)
    if nrow * nrow < NUM_IMAGES:
        nrow += 1

    save_tensor_to_csv(generated_images, "test.csv")
    exit(1)
    save_image(generated_images, output_path, nrow=nrow)

print(f"Generated images saved to '{output_path}'")
print("Generation finished.")
