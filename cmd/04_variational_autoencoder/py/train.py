import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import time

# 从模型文件中导入 VAE 类和共享参数/函数
# 确保 vae_model.py 在同一个目录下或 Python 的搜索路径中
try:
    from vae_model import VAE, vae_loss_function, IMAGE_SIZE, CHANNELS_IMG, LATENT_DIM
except ImportError:
    print("Error: Could not import from vae_model.py.")
    print("Make sure vae_model.py is in the same directory or accessible in the Python path.")
    exit()

# --- Hardcoded Training Parameters ---
#https://www.kaggle.com/datasets/splcher/animefacedataset/data
DATA_DIR = 'dataset/images'
RESULTS_DIR = 'results_vae_anime_hardcoded' # 保存结果和模型的目录名
BATCH_SIZE = 128      # 批处理大小
EPOCHS = 50           # 训练轮数
LEARNING_RATE = 1e-3  # 学习率
KLD_WEIGHT = 0.00025  # KL散度损失的权重
# LATENT_DIM 在 vae_model.py 中定义并导入，这里直接使用导入的值
USE_CUDA_IF_AVAILABLE = True # 设置为 False 可强制使用 CPU
SEED = 42             # 随机种子
LOG_INTERVAL = 100    # 每隔多少个 batch 打印一次日志
SAVE_INTERVAL = 10    # 每隔多少个 epoch 保存一次模型

# --- 设置设备和种子 ---
use_cuda = USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()
DEVICE = torch.device("cuda" if use_cuda else "cpu")
torch.manual_seed(SEED)
if use_cuda:
    torch.cuda.manual_seed(SEED)

print(f"Using device: {DEVICE}")
print(f"Using Latent Dim: {LATENT_DIM}") # 确认使用的潜在维度 (从 vae_model 导入)

# --- 创建结果目录 ---
results_base_dir = RESULTS_DIR
reconstruction_dir = os.path.join(results_base_dir, 'reconstructed_train')
generated_dir = os.path.join(results_base_dir, 'generated_train')
models_dir = os.path.join(results_base_dir, 'models')
os.makedirs(reconstruction_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# --- 数据预处理 ---
# IMAGE_SIZE 从 vae_model 导入
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),  # [0, 1]
])

# --- 数据加载 ---
if not os.path.isdir(DATA_DIR):
    print(f"Error: Dataset directory not found at '{DATA_DIR}'")
    print("Please modify the DATA_DIR variable in the script to your actual dataset path.")
    exit()

try:
    dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
    # 检查数据集是否为空
    if len(dataset) == 0:
        print(f"Error: No images found in the dataset directory '{DATA_DIR}' or its subdirectories.")
        print("Please ensure your dataset is structured correctly for ImageFolder (e.g., data_dir/subdir/image.png).")
        exit()

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=use_cuda)
    print(f"Dataset loaded from {DATA_DIR}. Number of images: {len(dataset)}")
except Exception as e:
    print(f"An error occurred loading the dataset: {e}")
    exit()

# --- 初始化模型和优化器 ---
# 使用从 vae_model 导入的 CHANNELS_IMG, LATENT_DIM, IMAGE_SIZE
model = VAE(channels_img=CHANNELS_IMG, latent_dim=LATENT_DIM, image_size=IMAGE_SIZE).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- 训练函数 ---
def train(epoch):
    model.train()
    train_loss = 0
    train_recon_loss = 0
    train_kld_loss = 0
    start_epoch_batches = time.time() # Timer for batches within an epoch

    for batch_idx, (data, _) in enumerate(dataloader):
        start_batch_time = time.time() # Timer for a single batch
        data = data.to(DEVICE)
        optimizer.zero_grad()

        recon_batch, mu, log_var = model(data)
        # 使用从 vae_model 导入的 IMAGE_SIZE, CHANNELS_IMG 和上面定义的 KLD_WEIGHT
        loss, recon_loss, kld_loss = vae_loss_function(recon_batch, data, mu, log_var, KLD_WEIGHT, IMAGE_SIZE, CHANNELS_IMG)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_recon_loss += recon_loss.item()
        train_kld_loss += kld_loss.item()

        batch_process_time = time.time() - start_batch_time

        if batch_idx % LOG_INTERVAL == 0:
            batches_processed = batch_idx + 1
            avg_batch_time_so_far = (time.time() - start_epoch_batches) / batches_processed if batches_processed > 0 else 0
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / len(dataloader):.0f}%)]\t'
                  f'Loss: {loss.item():.4f} (Recon: {recon_loss.item():.4f}, KLD: {kld_loss.item():.4f})\t'
                  # f'Batch Time: {batch_process_time:.3f}s\t'
                  f'Avg Batch Time: {avg_batch_time_so_far:.3f}s')


    avg_loss = train_loss / len(dataloader)
    avg_recon_loss = train_recon_loss / len(dataloader)
    avg_kld_loss = train_kld_loss / len(dataloader)

    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} (Recon: {avg_recon_loss:.4f}, KLD: {avg_kld_loss:.4f})')
    return avg_loss, avg_recon_loss, avg_kld_loss

# --- 测试/评估函数 (用于生成样本) ---
def test(epoch, fixed_real_batch, fixed_noise):
    model.eval()
    with torch.no_grad():
        # 1. 重建图像
        recon_fixed, _, _ = model(fixed_real_batch)
        comparison = torch.cat([fixed_real_batch.cpu(), recon_fixed.cpu()])
        save_image(comparison,
                   os.path.join(reconstruction_dir, f'reconstruction_epoch_{epoch}.png'), nrow=8)

        # 2. 从固定噪声生成图像 (fixed_noise 在主循环开始时已根据 LATENT_DIM 创建)
        fixed_generated = model.decode(fixed_noise).cpu()
        save_image(fixed_generated,
                   os.path.join(generated_dir, f'fixed_generated_epoch_{epoch}.png'), nrow=8)

        # 3. 从随机噪声生成图像 (使用导入的 LATENT_DIM)
        sample = torch.randn(64, LATENT_DIM).to(DEVICE)
        random_generated = model.decode(sample).cpu()
        save_image(random_generated,
                   os.path.join(generated_dir, f'random_generated_epoch_{epoch}.png'), nrow=8)


if __name__ == "__main__":

    # --- 主训练循环 ---
    print("Starting training...")
    # 固定噪声和真实样本用于可视化
    # 使用导入的 LATENT_DIM
    fixed_noise = torch.randn(64, LATENT_DIM).to(DEVICE)

    # 尝试获取一个 batch 作为 fixed_real_batch
    try:
        fixed_real_batch, _ = next(iter(dataloader))
        fixed_real_batch = fixed_real_batch[:64].to(DEVICE) # Take first 64 samples
        save_image(fixed_real_batch.cpu(), os.path.join(reconstruction_dir, 'real_samples.png'), nrow=8)
    except StopIteration:
        print("Error: DataLoader is empty. Cannot get fixed real batch. Check dataset.")
        exit()


    total_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.time()
        train(epoch)
        test(epoch, fixed_real_batch, fixed_noise) # Generate samples after each epoch
        epoch_time = time.time() - epoch_start_time
        print(f"====> Epoch {epoch} completed in {epoch_time:.2f}s")

        # 保存模型
        # 使用上面定义的 SAVE_INTERVAL 和 EPOCHS
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            model_path = os.path.join(models_dir, f'vae_anime_epoch_{epoch}.pth')
            try:
                torch.save(model.state_dict(), model_path)
                print(f"Model saved to {model_path}")
            except Exception as e:
                print(f"Error saving model at epoch {epoch}: {e}")


    print("Training finished.")
    total_time = time.time() - total_start_time
    print(f"Total Training Time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

    # 保存最终模型 (以防万一)
    final_model_path = os.path.join(models_dir, 'vae_anime_final.pth')
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")