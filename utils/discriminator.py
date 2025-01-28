import torch.nn as nn
import torch


class Discriminator(nn.Module):
    def __init__(self, img_channels, img_size, text_dim):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(256 * (img_size // 8) * (img_size // 8) + text_dim, 1)
    
    def forward(self, img, text_embedding):
        img_features = self.conv(img).view(img.size(0), -1)
        combined = torch.cat((img_features, text_embedding), dim=1)
        validity = self.fc(combined)
        return validity
