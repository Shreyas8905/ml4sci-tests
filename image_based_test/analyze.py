import torch
import matplotlib.pyplot as plt


def analyze(model, dataloader, device):
    model.eval()
    reconstructions = []
    latent_vectors = []
    originals = []

    print("Analyzing reconstructions...")
    with torch.no_grad():
        for data, _ in dataloader:
            data = data.to(device)
            latent_batch, recon_batch = model(data)
            latent_vectors.extend(latent_batch.cpu().numpy())
            reconstructions.extend(recon_batch.cpu().numpy())
            originals.extend(data.cpu().numpy())
            break
        print(f"Successfully extracted {len(latent_vectors)} images")
        print(f"Shape of a single accessed latent vector: {latent_vectors[0].shape}")

        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    fig.suptitle('Qualitative Metric: Original FITS vs. Reconstructed Output', fontsize=16)
    
    for i in range(4):
        axes[0, i].imshow(originals[i][0], cmap='inferno', origin='lower')
        axes[0, i].set_title("Original (Log-Scaled)")
        axes[0, i].axis('off')
        axes[1, i].imshow(reconstructions[i][0], cmap='inferno', origin='lower')
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')
        
        plt.tight_layout()
        plt.show()