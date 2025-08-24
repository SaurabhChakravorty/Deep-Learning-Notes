import torch.nn as nn
import torch.optim as optim
import torch

# ----- Toy data -----
real = torch.normal(2.0, 1.0, size=(2000, 1))
noise_dim = 1

# ----- Discriminator neural network -----
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# ----- Generator neural network -----
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, z):
        return self.model(z)

D = Discriminator()
G = Generator()

d_optimizer = optim.Adam(D.parameters(), lr=0.001)
g_optimizer = optim.Adam(G.parameters(), lr=0.001)
bce = nn.BCELoss()

epochs = 5000
batch_size = 64

for epoch in range(epochs):
    # ----- Train Discriminator -----
    idx = torch.randint(0, real.size(0), (batch_size,))
    real_batch = real[idx]
    noise = torch.randn(batch_size, noise_dim)
    fake_batch = G(noise).detach()

    real_labels = torch.ones(batch_size, 1)
    fake_labels = torch.zeros(batch_size, 1)

    d_optimizer.zero_grad()
    real_pred = D(real_batch)
    fake_pred = D(fake_batch)
    d_loss_real = bce(real_pred, real_labels)
    d_loss_fake = bce(fake_pred, fake_labels)
    d_loss = d_loss_real + d_loss_fake
    d_loss.backward()
    d_optimizer.step()

    # ----- Train Generator -----
    noise = torch.randn(batch_size, noise_dim)
    g_optimizer.zero_grad()
    generated = G(noise)
    fake_pred = D(generated)
    g_loss = bce(fake_pred, real_labels)  # Generator wants D to output 1
    g_loss.backward()
    g_optimizer.step()

    if (epoch+1) % 200 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}: D loss={d_loss.item():.4f}, G loss={g_loss.item():.4f}")

# ----- Test the generator -----
test_noise = torch.randn(10, noise_dim)
generated_samples = G(test_noise).detach().numpy().flatten()
print("\nGenerated samples:", generated_samples)

# ...existing code...

# ----- Test the generator -----
test_noise = torch.randn(10, noise_dim)
generated_samples = G(test_noise).detach().numpy().flatten()
print("\nGenerated samples:", generated_samples)

# ----- Show original real samples for comparison -----
original_samples = real[:10].numpy().flatten()
print("Original real samples:", original_samples)



# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers, models
# # Discriminator Loss Function (Cross-Entropy)
# #
# # The discriminator tries to output a value close to 1 for real data and close to 0 for fake data.
# #
# # Mathematical formula:
# #
# # For real data x_real and fake data x_fake:
# #
# #     L_D = -E[log D(x_real)] - E[log(1 - D(x_fake))]
# #
# # Where:
# #   - D(x) is the discriminator output (probability x is real)
# #   - E[...] denotes the average over the dataset
# #
# # In code, this is approximated by:
# #
# #     real_loss = -mean(log(D(real)))
# #     fake_loss = -mean(log(1 - D(fake)))
# #     total_loss = real_loss + fake_loss
# #
# # This loss encourages the discriminator to output high values for real data and low

# # ----- Toy data -----
# rng = np.random.default_rng(0)
# real = rng.normal(2.0, 1.0, size=2000)   # real ~ N(2,1)
# fake = rng.normal(-2.0, 1.0, size=2000)  # fake ~ N(-2,1)

# print("Real data mean:", np.mean(real))

# # ----- Parameters (start simple) -----
# w, b = 0.4, 0.0

# # ----- TODO 1: sigmoid -----
# def sigmoid(x):
#     return 1/(1+ np.exp(-x))
    

# # ----- TODO 2: discriminator -----
# def D(x, w, b):
#     return sigmoid(w*x + b)


# # ----- TODO 3: discriminator loss (cross-entropy) -----
# def disc_loss(real, fake, w, b, eps=1e-8):
#     real_loss = -np.mean(np.log(D(real, w, b) + eps))
#     fake_loss = -np.mean(np.log(1 - D(fake, w, b)))
#     return real_loss + fake_loss

# # ----- TODO 4: compute loss -----
# loss = disc_loss(real, fake, w, b)import torch

# print("Discriminator loss:", loss)

# # Generator Loss Function (Cross-Entropy)
# #
# # The generator tries to fool the discriminator, so it wants the discriminator to output values close to 1 for fake data.
# #
# # Mathematical formula:
# #
# # For fake data x_fake:
# #
# #     L_G = -E[log D(x_fake)]
# #
# # Where:
# #   - D(x) is the discriminator output (probability x is real)
# #   - E[...] denotes the average over the dataset
# #
# # In code, this is approximated by:
# #
# #     gen_loss = -mean(log(D(fake)))
# #
# # This loss encourages the generator to produce data that the discriminator classifies as real.

# # ----- Generator loss function -----
# def gen_loss(fake, w, b, eps=1e-8):
#     return -np.mean(np.log(D(fake, w, b) + eps))

# # ----- Compute generator loss -----
# g_loss = gen_loss(fake, w, b)
# print("Generator loss:", g_loss)