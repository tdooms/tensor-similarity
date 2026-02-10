from torchvision import datasets
from src.datasets.base import Dataset, load_dataset

class MNIST:
    classes = 10
    size = (28, 28)
    name = "mnist"
    rgb = False

    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.train, self.val = load_dataset(MNIST, device=device)
    
    @staticmethod
    def prepare():
        train = datasets.MNIST(root=f"{Dataset.root}/raw", train=True, download=True)
        val = datasets.MNIST(root=f"{Dataset.root}/raw", train=False, download=True)

        train = Dataset(train.data.view(-1, 1, 28, 28), train.targets)
        val = Dataset(val.data.view(-1, 1, 28, 28), val.targets)
        return train, val
