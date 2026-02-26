from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from astropy.io import fits
from sklearn.preprocessing import minmax_scale
import numpy as np
import matplotlib.pyplot as plt
import fits

class ALMADataset(Dataset):
    def __init__(self, file_paths, transforms=None):
        self.file_paths = file_paths
        self.transforms = transforms

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with fits.open(path) as image_data:
            data = image_data[0].data
            if data.ndim > 2:
                data = data.flatten()[:360000].reshape(600, 600)
            data = np.nan_to_num(data)
            data = (data - np.min(data)) / (np.ptp(data) + 1e-8)
            data = data.astype(np.float32)
            data = np.expand_dims(data, axis=0)
        if self.transforms:
            data = self.transforms(data)
        return data 

transform = transforms.Compose([
    transforms.RandomRotation(180, fill=0), 
    transforms.CenterCrop(512), 
    transforms.Resize((600, 600)), 
])    
