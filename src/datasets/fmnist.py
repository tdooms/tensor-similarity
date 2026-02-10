from torchvision import datasets
from src.datasets.base import Dataset, load_dataset

class FMNIST:
    classes = 10
    size = 28
    name = "fmnist"

    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.train, self.val = load_dataset(FMNIST, device=device)
    
    @staticmethod
    def prepare():
        train = datasets.FashionMNIST(root=f"{Dataset.root}/raw", train=True, download=True)
        val = datasets.FashionMNIST(root=f"{Dataset.root}/raw", train=False, download=True)

        train = Dataset(train.data.view(-1, 1, 28, 28), train.targets)
        val = Dataset(val.data.view(-1, 1, 28, 28), val.targets)
        return train, val

        
        
    