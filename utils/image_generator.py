import torch.nn as nn
import torch

class Generator(nn.Module):
    def __init__(self, text_dim, noise_dim, img_channels, img_size):
        super(Generator, self).__init__()
        self.fc = nn.Linear(text_dim + noise_dim, 256 * (img_size // 8) * (img_size // 8))
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # Upsample to (img_size/4)
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),   # Upsample to (img_size/2)
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, img_channels, 4, stride=2, padding=1),  # Upsample to (img_size)
            nn.Tanh(),  # Output pixel values between -1 and 1
        )
    
    def forward(self, text_embedding, noise):
        x = torch.cat((text_embedding, noise), dim=1)
        x = self.fc(x).view(-1, 256, 8, 8)  # Reshape for deconvolution
        img = self.deconv(x)
        return img
