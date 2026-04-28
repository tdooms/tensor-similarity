from safetensors.torch import save_file, load_file
from pathlib import Path
import torch


def load_dataset(cls, device="cuda"):
    if (Dataset.root / cls.name).exists():
        train = Dataset.from_file(cls.name, "train", device=device)
        val = Dataset.from_file(cls.name, "val", device=device)
    else:
        train, val = cls.prepare()
        (Dataset.root / cls.name).mkdir(parents=True, exist_ok=True)
        save_file(dict(x=train.x, y=train.y), str(Dataset.root / cls.name / "train.safetensors"))
        save_file(dict(x=val.x, y=val.y), str(Dataset.root / cls.name / "val.safetensors"))
    return train.to(device), val.to(device)


class Dataset(torch.utils.data.Dataset):
    """A simple dataset wrapper using safetensors."""
    root = Path(__file__).resolve().parents[2] / "_downloads"
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y
    
    @classmethod
    def from_file(cls, name, split, device="cuda"):
        dataset = load_file(str(Dataset.root / name / f"{split}.safetensors"), device)
        return cls(dataset["x"], dataset["y"])
    
    def to(self, device):
        return Dataset(self.x.to(device), self.y.to(device))

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.size(0)
    
    @staticmethod
    def prepare():
        raise NotImplementedError()
