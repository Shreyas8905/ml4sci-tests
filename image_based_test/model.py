import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.leaky_relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.leaky_relu(out)
        return out

class ResNetAutoencoder256(nn.Module):
    def __init__(self, latent_dim=512):
        super(ResNetAutoencoder256, self).__init__()
        
        self.encoder_initial = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2), 
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True)
        )
        
        self.encoder_blocks = nn.Sequential(
            ResidualBlock(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            
            ResidualBlock(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            
            ResidualBlock(128),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            
            ResidualBlock(256),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True)
        )
        
        self.flatten = nn.Flatten()
        self.encoder_fc = nn.Linear(256 * 8 * 8, latent_dim)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 8 * 8),
            nn.LeakyReLU(0.2, True)
        )
        
        self.decoder_blocks = nn.Sequential(
            ResidualBlock(256),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            
            ResidualBlock(128),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            
            ResidualBlock(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, True),
            
            ResidualBlock(32),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2, True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), 
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.encoder_initial(x)
        x = self.encoder_blocks(x)
        
        x = self.flatten(x)
        latent = self.encoder_fc(x)
        
        x = self.decoder_fc(latent)
        x = x.view(-1, 256, 8, 8) 
        reconstruction = self.decoder_blocks(x)
        
        return latent, reconstruction