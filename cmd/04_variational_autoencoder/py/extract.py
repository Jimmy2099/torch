import torch

from vae_model import VAE, IMAGE_SIZE, CHANNELS_IMG, LATENT_DIM
import os
import time



def model_save(model):
    import numpy as np
    import os
    print(model)
    try:
        os.mkdir("data")
    except Exception as e:
        pass
    for name, param in model.named_parameters():
        # if name == "embedder.weight":
        #     continue
        param_np = param.detach().cpu().numpy()
        print(f"param: {name}, Shape: {param.shape}, dim: {param_np.ndim}")

        if param_np.ndim == 4:
            param_np = param_np.reshape(param_np.shape[0], -1)

        np.savetxt(f"./data/{name}.csv", param_np, delimiter=",", fmt="%.16f")


    for name, buf in model.named_buffers():
        buf_np = buf.detach().cpu().numpy()
        print(f"buf: {name}, Shape: {buf.shape}, dim: {buf.ndim}")

        if buf_np.ndim == 0:
            buf_np = np.array([buf_np])

        np.savetxt(f"./data/{name}.csv", buf_np, delimiter=",", fmt="%.16f")



    print("所有参数已保存。")


if __name__ == "__main__":
    MODEL_PATH = 'output/vae_anime/models/vae_anime_final.pth'

    OUTPUT_DIR = 'output/vae_generated'
    OUTPUT_NAME = 'random_generated.png'
    NUM_IMAGES = 64
    # SEED = 42
    USE_CUDA_IF_AVAILABLE = True

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
    # print(f"  Seed: {SEED}")
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
    model_save(model)
