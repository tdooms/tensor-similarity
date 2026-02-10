from torchvision import datasets
from src.datasets.base import Dataset, load_dataset
from kornia.augmentation import RandomGaussianNoise
from kornia.color import RgbToGrayscale
from torch import nn
import numpy as np
import torch

class SVHN:
    classes = 10
    size = (32, 32)
    rgb = True
    name = "svhn"

    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.train, self.val = load_dataset(SVHN, device=device)
    
    @staticmethod
    def prepare():
        train = datasets.SVHN(root=f"{Dataset.root}/raw", split="train", download=True)
        extra = datasets.SVHN(root=f"{Dataset.root}/raw", split="extra", download=True)
        test = datasets.SVHN(root=f"{Dataset.root}/raw", split="test", download=True)
        
        data = np.concatenate([train.data, extra.data], axis=0)
        labels = np.concatenate([train.labels, extra.labels], axis=0)

        train = Dataset(torch.from_numpy(data).contiguous(), torch.from_numpy(labels).contiguous())
        val = Dataset(torch.from_numpy(test.data).contiguous(), torch.from_numpy(test.labels).contiguous())
        return train, val

    @staticmethod
    def augment(noise=0.3, grayscale=True):
        train = nn.Sequential(
            RandomGaussianNoise(mean=0, std=noise, p=1),
            RgbToGrayscale() if grayscale else nn.Identity(),
        )
        return train, RgbToGrayscale() if grayscale else nn.Identity()

        
        
        
    