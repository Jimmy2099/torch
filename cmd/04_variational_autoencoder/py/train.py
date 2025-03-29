import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import os
import time

# Keep imports and potentially function/class definitions outside
# --- Function Definitions --- (Keep these outside the main block)
def train(epoch, model, dataloader, optimizer, device, log_interval, batch_size):
    model.train()
    train_loss = 0
    train_recon_loss_accum = 0
    train_kld_loss_accum = 0
    start_epoch_time = time.time()

    num_batches = len(dataloader)
    # Use the imported loss function directly if preferred, or calculate inline
    from vae_model import vae_loss_function_pytorch # Assuming it's defined there

    for batch_idx, (data, _) in enumerate(dataloader):
        start_batch_time = time.time()
        data = data.to(device)
        optimizer.zero_grad()

        recon_batch, mu, log_var = model(data)

        loss, recon_loss, kld_loss = vae_loss_function_pytorch(recon_batch, data, mu, log_var)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_recon_loss_accum += recon_loss.item()
        train_kld_loss_accum += kld_loss.item()

        batch_process_time = time.time() - start_batch_time

        if batch_idx % log_interval == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(dataloader.dataset)} ({100. * batch_idx / num_batches:.0f}%)]\t'
                  f'Loss: {loss.item() / len(data):.4f} \t'
                  f'Avg Batch Time: {batch_process_time:.3f}s \t'
                  f'LR: {current_lr:.1e}')

    avg_loss = train_loss / len(dataloader.dataset)
    avg_recon_loss = train_recon_loss_accum / len(dataloader.dataset)
    avg_kld_loss = train_kld_loss_accum / len(dataloader.dataset)
    epoch_time = time.time() - start_epoch_time

    print(f'====> Epoch: {epoch} Average loss: {avg_loss:.4f} (Recon: {avg_recon_loss:.4f}, KLD: {avg_kld_loss:.4f}) \t'
          f'Time: {epoch_time:.2f}s')
    return avg_loss, avg_recon_loss, avg_kld_loss

def test(epoch, model, fixed_real_batch, fixed_noise, device, reconstruction_dir, generated_dir, latent_dim, batch_size):
    model.eval()
    with torch.no_grad():
        recon_fixed, _, _ = model(fixed_real_batch)
        comparison = torch.cat([fixed_real_batch.cpu() * 0.5 + 0.5,
                                recon_fixed.cpu() * 0.5 + 0.5])
        save_image(comparison.clamp(0, 1),
                   os.path.join(reconstruction_dir, f'reconstruction_epoch_{epoch}.png'), nrow=8)

        fixed_generated = model.decode(fixed_noise).cpu()
        save_image(fixed_generated * 0.5 + 0.5,
                   os.path.join(generated_dir, f'fixed_generated_epoch_{epoch}.png'), nrow=8)

        sample = torch.randn(batch_size, latent_dim).to(device) # Use latent_dim directly
        random_generated = model.decode(sample).cpu()
        save_image(random_generated * 0.5 + 0.5,
                   os.path.join(generated_dir, f'random_generated_epoch_{epoch}.png'), nrow=8)

from vae_model import VAE, IMAGE_SIZE, CHANNELS_IMG, LATENT_DIM
# --- Hardcoded Training Parameters ---
DATA_DIR = 'dataset/images'
RESULTS_DIR = 'output/vae_anime'
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4
USE_CUDA_IF_AVAILABLE = True
SEED = 42
LOG_INTERVAL = 50
SAVE_INTERVAL = 2
NUM_WORKERS = 2 # Set number of workers for DataLoader

# --- Main Execution Block ---
if __name__ == "__main__":
    # --- 设置设备和种子 --- (Moved inside main block)
    use_cuda = USE_CUDA_IF_AVAILABLE and torch.cuda.is_available()
    DEVICE = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(SEED)
    if use_cuda:
        torch.cuda.manual_seed(SEED)

    print(f"Using device: {DEVICE}")
    print(f"Imported Latent Dim: {LATENT_DIM}")
    print(f"Imported Image Size: {IMAGE_SIZE}")

    # --- 创建结果目录 --- (Moved inside main block)
    results_base_dir = RESULTS_DIR
    reconstruction_dir = os.path.join(results_base_dir, 'reconstructed_train')
    generated_dir = os.path.join(results_base_dir, 'generated_train')
    models_dir = os.path.join(results_base_dir, 'models')
    os.makedirs(reconstruction_dir, exist_ok=True)
    os.makedirs(generated_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # --- 数据预处理 --- (Moved inside main block)
    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # --- 数据加载 --- (Moved inside main block)
    if not os.path.isdir(DATA_DIR):
        print(f"Error: Dataset directory not found at '{DATA_DIR}'")
        exit()

    try:
        dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
        if len(dataset) == 0:
            print(f"Error: No images found in the dataset directory '{DATA_DIR}' or its subdirectories.")
            exit()

        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS, pin_memory=use_cuda)
        print(f"Dataset loaded from {DATA_DIR}. Number of images: {len(dataset)}")
    except Exception as e:
        print(f"An error occurred loading the dataset: {e}")
        exit()

    # --- 初始化模型和优化器 --- (Moved inside main block)
    model = VAE(channels_img=CHANNELS_IMG, latent_dim=LATENT_DIM, image_size=IMAGE_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 主训练循环 --- (Already inside, but now follows setup)
    print("Starting training...")
    fixed_noise = torch.randn(64, LATENT_DIM).to(DEVICE)

    try:
        fixed_real_batch_iter = iter(dataloader)
        fixed_real_batch, _ = next(fixed_real_batch_iter)
        fixed_real_batch = fixed_real_batch[:64].to(DEVICE)
        save_image(fixed_real_batch.cpu() * 0.5 + 0.5,
                   os.path.join(reconstruction_dir, 'real_samples.png'), nrow=8)
    except StopIteration:
        print("Error: DataLoader is empty. Cannot get fixed real batch. Check dataset.")
        exit()
    except Exception as e:
        print(f"Error getting fixed real batch: {e}")
        # It might be okay to continue without the fixed batch for testing if needed
        fixed_real_batch = None
        print("Warning: Proceeding without fixed real batch for testing.")
        # exit() # Decide if this is critical

    total_start_time = time.time()

    for epoch in range(1, EPOCHS + 1):
        # Pass necessary variables to train/test functions
        train(epoch, model, dataloader, optimizer, DEVICE, LOG_INTERVAL, BATCH_SIZE)

        if (epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS) and fixed_real_batch is not None:
            test(epoch, model, fixed_real_batch, fixed_noise, DEVICE, reconstruction_dir, generated_dir, LATENT_DIM, BATCH_SIZE)

        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            model_path = os.path.join(models_dir, f'vae_anime_epoch_{epoch}.pth')
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, model_path)
                print(f"Model checkpoint saved to {model_path}")
            except Exception as e:
                print(f"Error saving model at epoch {epoch}: {e}")

    print("Training finished.")
    total_time = time.time() - total_start_time
    print(f"Total Training Time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")

    final_model_path = os.path.join(models_dir, 'vae_anime_final.pth')
    try:
        torch.save(model.state_dict(), final_model_path)
        print(f"Final model saved to {final_model_path}")
    except Exception as e:
        print(f"Error saving final model: {e}")