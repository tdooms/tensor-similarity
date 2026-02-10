from torchvision import datasets
from src.datasets.base import Dataset, load_dataset

class EMNIST:
    classes = 62
    size = (28, 28)
    name = "emnist"
    
    labels = [str(i) for i in range(10)] + [chr(i) for i in range(ord('A'), ord('Z') + 1)] + [chr(i) for i in range(ord('a'), ord('z') + 1)]
    rgb = False

    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.train, self.val = load_dataset(EMNIST, device=device)
    
    @staticmethod
    def prepare():
        train = datasets.EMNIST(root=f"{Dataset.root}/raw", split='byclass', train=True, download=True)
        val = datasets.EMNIST(root=f"{Dataset.root}/raw", split='byclass', train=False, download=True)

        train = Dataset(train.data.view(-1, 1, 28, 28).mT.contiguous(), train.targets)
        val = Dataset(val.data.view(-1, 1, 28, 28).mT.contiguous(), val.targets)
        return train, val
