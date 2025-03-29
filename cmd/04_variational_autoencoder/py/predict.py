import os
import random
import time

import torch
from torchvision.utils import save_image

from vae_model import VAE, IMAGE_SIZE, CHANNELS_IMG, LATENT_DIM

if __name__ == "__main__":
    MODEL_PATH = 'output/vae_anime/models/vae_anime.pth'

    OUTPUT_DIR = 'output/vae_generated'  # 保存生成图片的目录
    OUTPUT_NAME = 'random_generated.png'  # 生成的图片网格文件名
    NUM_IMAGES = 64  # 要生成的图片数量
    # SEED = 42                           # 不再需要固定种子以获得不同的结果
    USE_CUDA_IF_AVAILABLE = True  # 如果可用，是否使用 CUDA
    # --------------------------------

    # --- 设置设备 ---
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
    # print(f"  Seed: {SEED}") # 可以移除或注释掉这行打印信息
    print("  Seed: None (Random each run)")  # 更新打印信息
    print("-" * 20)

    # --- 创建结果目录 ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 初始化模型 ---
    model = VAE(channels_img=CHANNELS_IMG, latent_dim=LATENT_DIM, image_size=IMAGE_SIZE).to(DEVICE)

    # --- 加载预训练模型 ---
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

    # --- 生成随机图像 ---
    print(f"Generating {NUM_IMAGES} random images...")
    with torch.no_grad():
        # 因为没有设置种子，每次运行时这里的噪声都会不同
        random_noise = torch.randn(NUM_IMAGES, LATENT_DIM).to(DEVICE)

        generated_images = model.decode(random_noise).cpu()
        generated_images = generated_images * 0.5 + 0.5
        generated_images = generated_images.clamp(0, 1)

        # --- 输出文件名 ---
        # output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
        timestamp = int(time.time())  # 获取当前时间的 Unix 时间戳 (整数)
        random_suffix = random.randint(0, 9999)  # 生成一个 0 到 9999 的随机数
        output_filename = f"random_generated_{timestamp}_{random_suffix:04d}.png"  # 格式化文件名
        # 使用 :04d 确保随机数至少有4位，不足的前面补0, e.g., 12 -> 0012
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        # ---------------------------------

    nrow = int(NUM_IMAGES ** 0.5)
    if nrow * nrow < NUM_IMAGES:
        nrow += 1

    save_image(generated_images, output_path, nrow=nrow)

print(f"Generated images saved to '{output_path}'")
print("Generation finished.")
