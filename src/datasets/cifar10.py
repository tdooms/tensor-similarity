from torchvision import datasets
from src.datasets.base import Dataset, load_dataset
from torch import nn
import torch

class CIFAR10:
    classes = 10
    size = 32
    name = "cifar10"

    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.train, self.val = load_dataset(CIFAR10, device=device)
    
    @staticmethod
    def prepare():
        train = datasets.CIFAR10(root=f"{Dataset.root}/raw", train=True, download=True)
        val = datasets.CIFAR10(root=f"{Dataset.root}/raw", train=False, download=True)
        
        train_x = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous()
        val_x = torch.from_numpy(val.data).permute(0, 3, 1, 2).contiguous()

        train = Dataset(train_x, torch.tensor(train.targets))
        val = Dataset(val_x, torch.tensor(val.targets))
        return train, val
