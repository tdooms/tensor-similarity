import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from typing import Optional
from .tokenization import get_tokenizer, TOKENIZER_REPO

DATASET_REPO = "SimpleStories/SimpleStories"
TEXT_FIELD = "story"


class SimpleStoriesDataset(Dataset):
    """SimpleStories dataset wrapper for language modeling."""
    
    def __init__(
        self,
        split: str = "train",
        n_ctx: int = 256,
        tokenizer_name: str = TOKENIZER_REPO,
        max_samples: Optional[int] = None,
    ):
        self.n_ctx = n_ctx
        self.tokenizer = get_tokenizer(tokenizer_name)
        
        dataset = load_dataset(DATASET_REPO, split=split)
        
        if max_samples is not None:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        self.data = dataset
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx][TEXT_FIELD]
        
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.n_ctx,
            padding="max_length",
            return_tensors="pt",
        )
        
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }


def create_dataloaders(
    n_ctx: int = 256,
    batch_size: int = 16,
    tokenizer_name: str = TOKENIZER_REPO,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = 1000,
    num_workers: int = 0,
):
    """Create train and validation dataloaders for SimpleStories.
    
    Args:
        n_ctx: Context length
        batch_size: Batch size
        tokenizer_name: Name of tokenizer to use
        max_train_samples: Max training samples (None = all)
        max_val_samples: Max validation samples
        num_workers: DataLoader workers
        
    Returns:
        train_dataloader, val_dataloader
    """
    train_dataset = SimpleStoriesDataset(
        split="train",
        n_ctx=n_ctx,
        tokenizer_name=tokenizer_name,
        max_samples=max_train_samples,
    )
    
    val_dataset = SimpleStoriesDataset(
        split="test",
        n_ctx=n_ctx,
        tokenizer_name=tokenizer_name,
        max_samples=max_val_samples,
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    return train_dataloader, val_dataloader
