"""Bigram score analysis for tracking model behavior.

Implements the bigram score metric from the paper:
B_k^S = -sum_i q~(t_i | t_k) * log p(t_i | t_k)

Where q~ is the empirical bigram distribution from training data
and p is the model's predicted distribution.
"""
import math
import torch
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class BigramAnalyzer:
    """Analyzes bigram statistics and computes bigram scores."""
    
    EPS = 1e-10  # numerical safety for log(p)
    LOG_EPS = math.log(1e-10)
    
    def __init__(
        self,
        vocab_size: int,
        device: str = "cpu",
    ):
        """Initialize the bigram analyzer.
        
        Args:
            vocab_size: Size of the vocabulary
            device: Device for computations
        """
        self.vocab_size = vocab_size
        self.device = device
        
        # (vocab_size, vocab_size) count matrix: row = context, col = next token
        # Always kept on CPU as int64 to avoid GPU memory waste
        self.count_matrix: Optional[torch.Tensor] = None
        self.total_bigrams = 0
        
        # Fixed sample indices for reproducible evaluation
        self._sample_indices: Optional[List[Tuple[int, int]]] = None
        
        self._is_fitted = False
    
    def fit(self, dataloader, max_samples: Optional[int] = None) -> "BigramAnalyzer":
        """Fit the empirical bigram distribution from training data.
        
        Args:
            dataloader: DataLoader yielding batches with 'input_ids'
            max_samples: Maximum number of samples to use (None = all)
            
        Returns:
            self for method chaining
        """
        V = self.vocab_size
        counts = torch.zeros(V, V, dtype=torch.long)
        
        n_samples = 0
        for batch in tqdm(dataloader, desc="Fitting bigram distribution"):
            input_ids = batch["input_ids"]  # (B, T)
            attn_mask = batch.get("attention_mask", None)  # (B, T) or None
            batch_size = input_ids.shape[0]
            
            # Determine how many sequences to use from this batch
            if max_samples is not None:
                take = min(batch_size, max_samples - n_samples)
                input_ids = input_ids[:take]
                if attn_mask is not None:
                    attn_mask = attn_mask[:take]
            
            # Ensure counting happens on CPU (count_matrix is CPU int64)
            input_ids = input_ids.cpu()
            if attn_mask is not None:
                attn_mask = attn_mask.cpu()
            
            # Vectorised bigram counting: pair consecutive tokens
            t1 = input_ids[:, :-1].reshape(-1)  # context tokens
            t2 = input_ids[:, 1:].reshape(-1)   # next tokens
            
            # Both tokens must be valid (non-padding) and in vocab range
            valid = (t1 >= 0) & (t1 < V) & (t2 >= 0) & (t2 < V)
            if attn_mask is not None:
                m1 = attn_mask[:, :-1].reshape(-1).bool()
                m2 = attn_mask[:, 1:].reshape(-1).bool()
                valid = valid & m1 & m2
            t1 = t1[valid]
            t2 = t2[valid]
            
            # Scatter-add into count matrix
            idx = t1 * V + t2
            counts.view(-1).scatter_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
            
            n_samples += input_ids.shape[0]
            if max_samples is not None and n_samples >= max_samples:
                break
        
        self.count_matrix = counts
        self.total_bigrams = int(counts.sum().item())
        self._is_fitted = True
        return self
    
    def get_conditional_distribution(self, context_token: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the sparse empirical conditional distribution q~(t' | t).
        
        Returns only the observed successors (nonzero counts), keeping q~
        sparse.  No additive smoothing is applied to q~ itself.
        
        Args:
            context_token: The conditioning token t
            
        Returns:
            (indices, probs) where:
              indices: int64 tensor of successor token ids with nonzero count
              probs:   float32 tensor of corresponding probabilities (sums to 1)
            Both tensors are on self.device.  Empty if context was never seen.
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before getting distribution")
        
        row = self.count_matrix[context_token]  # int64, CPU
        indices = row.nonzero(as_tuple=False).squeeze(-1)  # token ids with count > 0
        
        if indices.numel() == 0:
            empty = torch.tensor([], device=self.device)
            return empty.long(), empty.float()
        
        counts = row[indices].float()
        probs = counts / counts.sum()
        
        return indices.to(self.device), probs.to(self.device)
    
    def compute_bigram_entropy(self, context_token: int) -> float:
        """Compute the entropy of the empirical bigram distribution H(q~(·|t)).
        
        This is the best-achievable bigram score for this context.
        Computed only over the observed support — no smoothing distortion.
        
        Args:
            context_token: The conditioning token
            
        Returns:
            Entropy in nats (0.0 if context was never observed)
        """
        indices, q = self.get_conditional_distribution(context_token)
        if indices.numel() == 0:
            return 0.0
        # H(q) = -sum q * log q  (all q > 0 on the support by construction)
        entropy = -torch.sum(q * torch.log(q))
        return entropy.item()
    
    def compute_bigram_score(
        self,
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        position: int,
    ) -> float:
        """Compute the bigram score at a specific position.
        
        B_k^S = -sum_i q~(t_i | t_k) * log p(t_i | t_k)
        
        Args:
            model: The language model
            input_ids: Input token ids of shape (batch, seq_len)
            position: Position k to compute score at
            
        Returns:
            Bigram score (cross entropy between empirical and model distributions)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before computing scores")
        
        model.eval()
        with torch.no_grad():
            # Get model predictions at position k
            logits = model(input_ids)  # (batch, seq_len, vocab_size)
            
            # Get logits at position k (predicting position k+1)
            logits_k = logits[:, position, :]  # (batch, vocab_size)
            log_probs = F.log_softmax(logits_k, dim=-1)  # (batch, vocab_size)
            
            # Get context token at position k
            context_token = input_ids[0, position].item()
            
            # Get sparse empirical distribution
            supp_idx, q = self.get_conditional_distribution(context_token)
            if supp_idx.numel() == 0:
                return 0.0
            
            # Cross entropy on support only: -sum_i q(i) * log(max(p(i), eps))
            # Average over batch
            log_p_support = log_probs[:, supp_idx]  # (batch, |support|)
            log_p_safe = torch.clamp(log_p_support, min=self.LOG_EPS)
            cross_entropy = -torch.sum(q.unsqueeze(0) * log_p_safe, dim=-1)
            bigram_score = cross_entropy.mean().item()
        
        return bigram_score
    
    @staticmethod
    def _build_sample_indices(
        dataset: Dataset,
        n_samples: int,
        seed: int,
    ) -> List[Tuple[int, int]]:
        """Build a fixed list of (dataset_index, position) pairs.
        
        Samples directly from the dataset by integer index, so the result
        is independent of any DataLoader ordering or shuffling.
        
        A position k is valid only if mask[k]==1 AND mask[k+1]==1,
        ensuring both the context token and the next token are real.
        """
        rng = torch.Generator().manual_seed(seed)
        n_dataset = len(dataset)
        
        # Shuffle dataset indices deterministically
        perm = torch.randperm(n_dataset, generator=rng)
        
        indices: List[Tuple[int, int]] = []
        for dataset_idx in perm.tolist():
            sample = dataset[dataset_idx]
            input_ids = sample["input_ids"]
            attn_mask = sample.get("attention_mask", None)
            seq_len = input_ids.shape[0]
            
            if attn_mask is not None:
                # Valid bigram positions: mask[k]==1 AND mask[k+1]==1
                valid = attn_mask[:-1].bool() & attn_mask[1:].bool()
                valid_positions = valid.nonzero(as_tuple=False).squeeze(-1)
                if valid_positions.numel() == 0:
                    continue
                # Pick one random valid position
                pick = torch.randint(0, valid_positions.numel(), (1,), generator=rng).item()
                k = valid_positions[pick].item()
            else:
                if seq_len < 2:
                    continue
                k = torch.randint(0, seq_len - 1, (1,), generator=rng).item()
            
            indices.append((dataset_idx, k))
            if len(indices) >= n_samples:
                break
        
        return indices
    
    def compute_average_bigram_score(
        self,
        model: torch.nn.Module,
        val_dataset: Dataset,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """Compute the average bigram score over fixed random positions.
        
        B_bar = (1/n) * sum_i B_{k_i}^{S_i}
        
        On the first call, a fixed set of (dataset_index, position) pairs
        is sampled from val_dataset and cached.  Subsequent calls reuse
        the same pairs so that scores are comparable across checkpoints.
        
        Sampling is by dataset index — completely independent of any
        DataLoader ordering or shuffling.
        
        Args:
            model: The language model
            val_dataset: Validation dataset (supports __getitem__ and __len__)
            n_samples: Number of random (sequence, position) pairs
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (average_bigram_score, average_bigram_entropy)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before computing scores")
        
        # Build fixed sample indices once
        if self._sample_indices is None:
            self._sample_indices = self._build_sample_indices(
                val_dataset, n_samples, seed
            )
        
        # Fetch sequences by dataset index
        all_input_ids = []
        for dataset_idx, _ in self._sample_indices:
            sample = val_dataset[dataset_idx]
            all_input_ids.append(sample["input_ids"].to(self.device))
        
        model.eval()
        scores = []
        entropies = []
        
        with torch.no_grad():
            # Process in mini-batches for efficiency
            batch_seqs = []
            batch_positions = []
            
            for i, (_, k) in enumerate(self._sample_indices):
                batch_seqs.append(all_input_ids[i])
                batch_positions.append(k)
                
                if len(batch_seqs) == 64 or i == len(self._sample_indices) - 1:
                    stacked = torch.stack(batch_seqs)  # (B, T)
                    logits = model(stacked)  # (B, T, V)
                    
                    for j, pos in enumerate(batch_positions):
                        log_probs = F.log_softmax(logits[j, pos, :], dim=-1)
                        context_token = stacked[j, pos].item()
                        supp_idx, q = self.get_conditional_distribution(context_token)
                        
                        if supp_idx.numel() == 0:
                            continue
                        
                        # Score on support: -sum q * log(max(p, eps))
                        log_p_safe = torch.clamp(log_probs[supp_idx], min=self.LOG_EPS)
                        score = -torch.sum(q * log_p_safe).item()
                        entropy = self.compute_bigram_entropy(context_token)
                        
                        scores.append(score)
                        entropies.append(entropy)
                    
                    batch_seqs = []
                    batch_positions = []
        
        avg_score = sum(scores) / len(scores)
        avg_entropy = sum(entropies) / len(entropies)
        
        return avg_score, avg_entropy
    
    def save(self, path: str) -> None:
        """Save the fitted bigram distribution to disk.
        
        Args:
            path: File path to save to (e.g. 'cache/bigram.pt')
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before saving")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            "vocab_size": self.vocab_size,
            "count_matrix": self.count_matrix,
            "total_bigrams": self.total_bigrams,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "BigramAnalyzer":
        """Load a fitted bigram distribution from disk.
        
        Args:
            path: File path to load from
            device: Device for computations
            
        Returns:
            Fitted BigramAnalyzer instance
        """
        data = torch.load(path, weights_only=False)
        
        analyzer = cls(
            vocab_size=data["vocab_size"],
            device=device,
        )
        
        analyzer.count_matrix = data["count_matrix"].cpu()  # always CPU int64
        analyzer.total_bigrams = data["total_bigrams"]
        analyzer._is_fitted = True
        
        return analyzer
    
    def get_stats(self) -> Dict:
        """Get statistics about the fitted bigram distribution."""
        if not self._is_fitted:
            return {"fitted": False}
        
        unigram_counts = self.count_matrix.sum(dim=1)
        n_contexts = int((unigram_counts > 0).sum().item())
        n_bigrams = int((self.count_matrix > 0).sum().item())
        
        return {
            "fitted": True,
            "total_bigrams": self.total_bigrams,
            "unique_contexts": n_contexts,
            "unique_bigrams": n_bigrams,
        }
