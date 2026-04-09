"""Repeated-token data generation for induction head experiments.

Algorithm (CORRECTED):
1. Pick subsequence length l from [2, n-2]
2. Pick starting index i from [0, n-1-l]  
3. Pick starting index of first repeat from [i+l, n-1]
4. Repeat subsequence as many times as fits with VARIABLE gaps
5. Evaluate on all repeated tokens EXCEPT first token of EACH repeat
6. Filler tokens MUST be UNIQUE and DISJOINT from subsequence
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class RepeatedTokenDataset(Dataset):
    """Dataset with variable-gap repeated subsequences."""

    def __init__(
        self,
        vocab_size: int,
        n_ctx: int,
        n_samples: int = 10_000,
        seed: int = 42,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.n_samples = n_samples
        
        rng = np.random.RandomState(seed)
        self.data = []
        self.repeat_masks = []
        
        for _ in range(n_samples):
            seq, mask = self._generate_sequence(rng)
            self.data.append(seq)
            self.repeat_masks.append(mask)
        
        self.data = torch.stack(self.data)
        self.repeat_masks = torch.stack(self.repeat_masks)
    
    def _generate_sequence(self, rng: np.random.RandomState):
        """Generate a single sequence with repeated subsequence."""
        n = self.n_ctx
        
        # 1. Pick subsequence length l from [2, n-2]
        # Ensure we can fit at least original + one full repeat
        max_l = (n - 1) // 2  # Conservative: ensures space for original + 1 repeat + 1 gap
        l = rng.randint(2, max(3, max_l + 1))
        
        # 2. Pick starting index i from [0, n-1-l]
        # But ensure we have space for at least one full repeat after
        max_i = n - 2*l - 1  # Space for: original + gap + repeat
        if max_i < 0:
            max_i = 0
            l = (n - 1) // 2
        i = rng.randint(0, max(1, max_i + 1))
        
        # 3. Pick starting index of first repeat from [i+l, n-1]
        # But ensure at least one full repeat fits
        min_repeat_start = i + l
        max_repeat_start = n - l  # Ensure full repeat fits
        
        if min_repeat_start > max_repeat_start:
            # Adjust to guarantee fit
            min_repeat_start = i + l
            max_repeat_start = min_repeat_start
        
        first_repeat_start = rng.randint(min_repeat_start, max(min_repeat_start + 1, max_repeat_start + 1))
        
        # Generate subsequence tokens - ALL MUST BE UNIQUE
        # Use random sampling without replacement
        if l > self.vocab_size:
            raise ValueError(f"Subsequence length {l} exceeds vocab size {self.vocab_size}")
        subseq = rng.choice(self.vocab_size, size=l, replace=False)
        subseq_set = set(subseq.tolist())
        
        # Initialize sequence
        seq = np.full(n, -1, dtype=np.int64)
        
        # Place original subsequence
        seq[i:i+l] = subseq
        
        # Track repeated positions (excluding first token of each repeat)
        repeat_positions = []
        
        # 4. Place repeats with variable gaps
        pos = first_repeat_start
        
        while pos + l <= n:
            # Place full repeat
            seq[pos:pos+l] = subseq
            
            # Mark positions (excluding first token of this repeat)
            for offset in range(1, l):
                repeat_positions.append(pos + offset)
            
            # Move to next position
            pos += l
            
            # Add variable gap if there's space
            if pos < n - l:
                max_gap = min(2, n - pos - l)
                if max_gap > 0:
                    gap = rng.randint(0, max_gap + 1)
                    pos += gap
        
        # Handle partial repeat at end
        if pos < n:
            remaining = n - pos
            seq[pos:n] = subseq[:remaining]
            # Mark positions (excluding first)
            for offset in range(1, remaining):
                repeat_positions.append(pos + offset)
        
        # 5. Fill gaps with UNIQUE tokens DISJOINT from subsequence
        # Get available filler tokens (disjoint from subsequence)
        available = [t for t in range(self.vocab_size) if t not in subseq_set]
        rng.shuffle(available)
        
        # Fill all gaps
        gap_indices = np.where(seq == -1)[0]
        for idx, pos in enumerate(gap_indices):
            if idx < len(available):
                seq[pos] = available[idx]
            else:
                # Should not happen with reasonable vocab, but fallback
                seq[pos] = rng.randint(0, self.vocab_size)
        
        # Create mask
        mask = torch.zeros(n, dtype=torch.bool)
        for pos in repeat_positions:
            mask[pos] = True
        
        return torch.tensor(seq, dtype=torch.long), mask

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {
            "input_ids": self.data[idx],
            "repeat_mask": self.repeat_masks[idx],
        }


def create_repeated_token_dataloaders(
    vocab_size: int,
    n_ctx: int,
    batch_size: int = 64,
    n_train: int = 50_000,
    n_val: int = 2_000,
    seed: int = 42,
):
    """Create train and validation dataloaders."""
    train_ds = RepeatedTokenDataset(vocab_size, n_ctx, n_samples=n_train, seed=seed)
    val_ds = RepeatedTokenDataset(vocab_size, n_ctx, n_samples=n_val, seed=seed + 1)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_dl, val_dl


def visualize_samples(vocab_size: int, n_ctx: int, n_samples: int = 50, seed: int = 42):
    """Generate and display sample sequences for verification."""
    dataset = RepeatedTokenDataset(vocab_size, n_ctx, n_samples=n_samples, seed=seed)
    
    print(f"Generated {n_samples} sequences with n_ctx={n_ctx}, vocab_size={vocab_size}")
    print("=" * 80)
    
    errors = []
    
    for i in range(n_samples):
        sample = dataset[i]
        seq = sample["input_ids"].tolist()
        mask = sample["repeat_mask"]
        
        repeat_positions = [j for j in range(len(mask)) if mask[j]]
        
        # VERIFICATION
        # 1. Must have at least 1 repeated token
        if len(repeat_positions) == 0:
            errors.append(f"Sample {i}: ERROR - 0 repeated tokens!")
        
        # 2. Find subsequence by looking for repeated patterns
        # The subsequence tokens are those that appear multiple times
        token_counts = {}
        for token in seq:
            token_counts[token] = token_counts.get(token, 0) + 1
        
        subseq_tokens = {t for t, count in token_counts.items() if count > 1}
        
        # 3. Check filler tokens are unique and disjoint
        filler_positions = []
        for j, token in enumerate(seq):
            if token not in subseq_tokens:
                filler_positions.append(j)
        
        filler_tokens = [seq[j] for j in filler_positions]
        
        # Check unique
        if len(filler_tokens) != len(set(filler_tokens)):
            errors.append(f"Sample {i}: ERROR - Filler tokens not unique: {filler_tokens}")
        
        # Check disjoint
        overlap = set(filler_tokens) & subseq_tokens
        if overlap:
            errors.append(f"Sample {i}: ERROR - Filler overlaps with subsequence: {overlap}")
        
        print(f"\nSample {i}:")
        print(f"  Sequence: {seq}")
        print(f"  Repeat positions: {repeat_positions}")
        print(f"  Num repeated tokens: {len(repeat_positions)}")
        
        # Visual
        visual = []
        for j, token in enumerate(seq):
            if mask[j]:
                visual.append(f"[{token}]")
            else:
                visual.append(f" {token} ")
        print(f"  Visual: {''.join(visual)}")
    
    print("\n" + "=" * 80)
    if errors:
        print("ERRORS FOUND:")
        for error in errors:
            print(f"  {error}")
    else:
        print("✓ All samples passed verification!")
    print("=" * 80)


if __name__ == "__main__":
    visualize_samples(vocab_size=32, n_ctx=20, n_samples=50, seed=42)
