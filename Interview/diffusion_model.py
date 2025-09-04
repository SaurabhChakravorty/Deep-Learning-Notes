import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt


# --- 1. Noise Schedule ---
def get_beta_schedule(T, start=1e-4, end=0.02):
    return torch.linspace(start, end, T)

# --- 2. Simple UNet-like Model for Denoising ---
class SimpleUNet(nn.Module):
    def __init__(self, img_channels=1, base_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(img_channels, base_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(base_channels, img_channels, 3, padding=1)
    
    def forward(self, x, t):
        # t is the timestep, can be embedded and added to x for more advanced models
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# --- 3. Forward Diffusion Process ---
def forward_diffusion_sample(x_0, t, betas):
    """
    x_0: original image
    t: timestep
    betas: noise schedule
    Returns: noisy image x_t and the noise added
    """
    sqrt_alphas_cumprod = torch.sqrt(torch.cumprod(1 - betas, dim=0))
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - torch.cumprod(1 - betas, dim=0))
    noise = torch.randn_like(x_0)
    x_t = sqrt_alphas_cumprod[t] * x_0 + sqrt_one_minus_alphas_cumprod[t] * noise
    return x_t, noise

# --- 4. Training Step ---
def training_step(model, x_0, t, betas):
    x_t, noise = forward_diffusion_sample(x_0, t, betas)
    noise_pred = model(x_t, t)
    loss = F.mse_loss(noise_pred, noise)
    return loss

# --- 5. Sampling (Reverse Process) ---
@torch.no_grad()
def sample(model, x_t, T, t_start, betas, device):
    """
    Reverse process starting from a given noisy image x_t at timestep t_start.
    """
    x = x_t.clone()
    for t in reversed(range(t_start + 1)):
        noise_pred = model(x, torch.tensor([t], device=device))
        beta_t = betas[t]
        x = (x - beta_t * noise_pred) / torch.sqrt(1 - beta_t)
        if t > 0:
            x += torch.sqrt(beta_t)
    return x

def show_images(x_0, x_t, x_recon):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(x_0.squeeze().cpu().numpy(), cmap='gray')
    axs[0].set_title('Original')
    axs[1].imshow(x_t.squeeze().cpu().numpy(), cmap='gray')
    axs[1].set_title('Noisy')
    axs[2].imshow(x_recon.squeeze().cpu().numpy(), cmap='gray')
    axs[2].set_title('Reconstructed')
    for ax in axs:
        ax.axis('off')
    plt.show()
# --- 6. Usage Example ---
# ...existing code...

if __name__ == "__main__":
    T = 1000  # Number of diffusion steps
    betas = get_beta_schedule(T)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Create a constant image: a centered white square on black background
    img_size = 28
    x_0 = torch.zeros(1, 1, img_size, img_size).to(device)
    x_0[:, :, 8:20, 8:20] = 1.0  # white square in the center

    epochs = 300
    batch_size = 1  # single constant image
    t_const = 500   # fixed timestep for both training and visualization

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(10):  # 100 steps per epoch
            loss = training_step(model, x_0, t_const, betas)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/100:.4f}")

    # Visualize results for the constant image and fixed timestep
    x_t, _ = forward_diffusion_sample(x_0, t_const, betas)
    x_recon = sample(model, x_t, T=T, t_start=t_const, betas=betas, device=device)
    show_images(x_0, x_t, x_recon)