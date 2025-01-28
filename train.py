import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
from utils.image_generator import Generator
from utils.discriminator import Discriminator
from utils.text_editor import encode_text



# Hyperparameters
img_size = 64
img_channels = 3
text_dim = 512  
noise_dim = 100
lr = 0.0002

# Initialize models
generator = Generator(text_dim, noise_dim, img_channels, img_size).to("cpu")
discriminator = Discriminator(img_channels, img_size, text_dim).to("cpu")

# Optimizers
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

# Loss function
criterion = nn.BCEWithLogitsLoss()


# Training loop
epochs = 100
batch_size = 16

for epoch in range(epochs):
    for i, text_prompt in enumerate(["A minimalist logo for a tech company"] * batch_size):
        # Encode text
        text_embedding = encode_text(text_prompt).mean(dim=1).to("cpu") 

        # Train Discriminator
        real_imgs = torch.randn(batch_size, img_channels, img_size, img_size).to("cpu")  
        noise = torch.randn(batch_size, noise_dim).to("cpu")
        fake_imgs = generator(text_embedding, noise)

        real_labels = torch.ones(batch_size, 1).to("cpu")
        fake_labels = torch.zeros(batch_size, 1).to("cpu")

        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs, text_embedding), real_labels)
        fake_loss = criterion(discriminator(fake_imgs.detach(), text_embedding), fake_labels)
        loss_D = (real_loss + fake_loss) / 2
        loss_D.backward(retain_graph=True) 
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        gen_loss = criterion(discriminator(fake_imgs, text_embedding), real_labels)
        gen_loss.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_D.item():.4f} | G Loss: {gen_loss.item():.4f}")
    if (epoch + 1) % 10 == 0:
        save_image(fake_imgs, f"logos_epoch_{epoch+1}.png", normalize=True)

torch.save(generator.state_dict(), "generator.pth")