import torch
from torch import optim
from torch.utils.data import DataLoader
from data_preprocessing import ALMADataset, train_transform, test_transform
from model import ResNetAutoencoder256
from split import split_data
from pytorch_msssim import ms_ssim

def train_ResNetAutoencoder256():
    device = torch.device("cpu")
    model = ResNetAutoencoder256(latent_dim=512).to(device)

    train_paths, test_paths = split_data()

    train_dataset = ALMADataset(train_paths, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    test_dataset = ALMADataset(test_paths, transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    num_epochs = 100

    print("Starting training...")
    model.train()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        
        for data, _ in train_dataloader:
            data = data.to(device)
            optimizer.zero_grad()

            _, reconstruction = model(data)

        loss_mse = criterion_mse(reconstruction, data)
        loss_l1 = criterion_l1(reconstruction, data)
        loss_ssim = 1 - ms_ssim(reconstruction, data, data_range=1.0, size_average=True)
        total_loss = loss_mse + loss_l1 + (0.1 * loss_ssim)
        
        total_loss.backward()
        optimizer.step()
        train_loss += total_loss.item()

        model.eval()
        val_mse = 0.0
        val_msssim = 0.0

        with torch.no_grad():
            for data, _ in test_dataloader:
                data = data.to(device)
                _, reconstruction = model(data)

                val_mse += criterion_mse(reconstruction, data).item()
                ssim_val = ms_ssim(reconstruction, data, data_range=1.0, size_average=True)
                val_msssim += ssim_val.item()
        avg_val_mse = val_mse / len(test_dataloader)
        avg_val_msssim = val_msssim / len(test_dataloader)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_dataloader):.4f}, Val MSE: {avg_val_mse:.4f}, Val MS-SSIM: {avg_val_msssim:.4f}")
    
    torch.save(model.state_dict(), "ResNetAutoencoder256.pth")
    print("Model saved!")