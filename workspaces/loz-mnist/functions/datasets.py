# datasets.py

from torch.utils.data import Dataset
from torchvision import datasets
import torch

class MNIST(Dataset):
    """Wrapper class that loads MNIST onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cuda", digits=None):
        """
        Args:
            train: whether to load training or test set
            download: whether to download if not present
            device: device to load data onto
            digits: list of digits to include (e.g., [0, 1] or [0,1,2,3]). 
                    If None, include all digits.
        """
        dataset = datasets.MNIST(root='./data', train=train, download=download)
        
        if digits is not None:
            # Create mask for selected digits
            mask = torch.zeros(len(dataset), dtype=torch.bool)
            for digit in digits:
                mask |= (dataset.targets == digit)
            
            self.x = dataset.data[mask].float().to(device).unsqueeze(1) / 255.0
            self.y = dataset.targets[mask].to(device)
        else:
            self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0
            self.y = dataset.targets.to(device)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)


class FMNIST(Dataset):
    """Wrapper class that loads F-MNIST onto the GPU for speed reasons."""
    def __init__(self, train=True, download=True, device="cuda", digits=None):
        dataset = datasets.FashionMNIST(root='./data', train=train, download=download)
        
        if digits is not None:
            mask = torch.zeros(len(dataset), dtype=torch.bool)
            for digit in digits:
                mask |= (dataset.targets == digit)
            
            self.x = dataset.data[mask].float().to(device).unsqueeze(1) / 255.0
            self.y = dataset.targets[mask].to(device)
        else:
            self.x = dataset.data.float().to(device).unsqueeze(1) / 255.0
            self.y = dataset.targets.to(device)
    
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)