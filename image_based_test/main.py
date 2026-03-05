from train import train_ResNetAutoencoder256
from analyze import analyze
import torch
from data_preprocessing import ALMADataset, test_transform
from model import ResNetAutoencoder256
import glob
from torch.utils.data import DataLoader


def main():
    train_ResNetAutoencoder256()
    data_dir = "../data/*.fits"
    file_paths = glob.glob(data_dir)
    model = ResNetAutoencoder256(latent_dim=512).to(torch.device("cpu"))
    model.load_state_dict(torch.load('ResNetAutoencoder256.pth'))

    dataset = ALMADataset(file_paths, transform=test_transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    analyze(model, dataloader, torch.device("cpu"))

if __name__ == "__main__":
    main()