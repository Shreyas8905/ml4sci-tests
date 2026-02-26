from train import train_autoencoder
from feature_extraction import feature_extraction
from analyze import analyze_fingerprints

def main():
    train_autoencoder()
    latent_features, image_paths = feature_extraction()
    analyze_fingerprints(latent_features, image_paths)