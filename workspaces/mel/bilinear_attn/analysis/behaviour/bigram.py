"""Bigram score analysis for tracking model behavior.

Implements the bigram score metric from the paper:
B_k^S = -sum_i q~(t_i | t_k) * log p(t_i | t_k)

Where q~ is the empirical bigram distribution from training data
and p is the model's predicted distribution.
"""
import torch
import torch.nn.functional as F
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm


class BigramAnalyzer:
    """Analyzes bigram statistics and computes bigram scores."""
    
    def __init__(
        self,
        vocab_size: int,
        device: str = "cpu",
        smoothing: float = 1e-10,
    ):
        """Initialize the bigram analyzer.
        
        Args:
            vocab_size: Size of the vocabulary
            device: Device for computations
            smoothing: Smoothing constant for probability estimation
        """
        self.vocab_size = vocab_size
        self.device = device
        self.smoothing = smoothing
        
        # Bigram counts: counts[t1][t2] = count of t2 following t1
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.total_bigrams = 0
        
        # Cached conditional distribution (lazy computed)
        self._conditional_dist: Optional[torch.Tensor] = None
        self._is_fitted = False
    
    def fit(self, dataloader, max_samples: Optional[int] = None) -> "BigramAnalyzer":
        """Fit the empirical bigram distribution from training data.
        
        Args:
            dataloader: DataLoader yielding batches with 'input_ids'
            max_samples: Maximum number of samples to use (None = all)
            
        Returns:
            self for method chaining
        """
        self.bigram_counts = defaultdict(lambda: defaultdict(int))
        self.unigram_counts = defaultdict(int)
        self.total_bigrams = 0
        
        n_samples = 0
        for batch in tqdm(dataloader, desc="Fitting bigram distribution"):
            input_ids = batch["input_ids"]
            
            for seq in input_ids:
                seq = seq.tolist()
                for i in range(len(seq) - 1):
                    t1, t2 = seq[i], seq[i + 1]
                    self.bigram_counts[t1][t2] += 1
                    self.unigram_counts[t1] += 1
                    self.total_bigrams += 1
                
                n_samples += 1
                if max_samples is not None and n_samples >= max_samples:
                    break
            
            if max_samples is not None and n_samples >= max_samples:
                break
        
        self._conditional_dist = None
        self._is_fitted = True
        return self
    
    def get_conditional_distribution(self, context_token: int) -> torch.Tensor:
        """Get the empirical conditional distribution q~(t' | t).
        
        Args:
            context_token: The conditioning token t
            
        Returns:
            Tensor of shape (vocab_size,) with probabilities
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before getting distribution")
        
        dist = torch.zeros(self.vocab_size, device=self.device)
        
        total = self.unigram_counts[context_token]
        if total > 0:
            for next_token, count in self.bigram_counts[context_token].items():
                if next_token < self.vocab_size:
                    dist[next_token] = count / total
        
        # Add smoothing for unseen bigrams
        dist = dist + self.smoothing
        dist = dist / dist.sum()
        
        return dist
    
    def compute_bigram_entropy(self, context_token: int) -> float:
        """Compute the entropy of the empirical bigram distribution H(q~(·|t)).
        
        This is the best-achievable bigram score for this context.
        
        Args:
            context_token: The conditioning token
            
        Returns:
            Entropy in nats
        """
        q = self.get_conditional_distribution(context_token)
        # H(q) = -sum q * log q
        entropy = -torch.sum(q * torch.log(q + 1e-10))
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
            
            # Get empirical distribution
            q = self.get_conditional_distribution(context_token)  # (vocab_size,)
            
            # Compute cross entropy: -sum_i q(i) * log p(i)
            # Average over batch
            cross_entropy = -torch.sum(q.unsqueeze(0) * log_probs, dim=-1)
            bigram_score = cross_entropy.mean().item()
        
        return bigram_score
    
    def compute_average_bigram_score(
        self,
        model: torch.nn.Module,
        dataloader,
        n_samples: int = 5000,
        seed: int = 42,
    ) -> Tuple[float, float]:
        """Compute the average bigram score over random positions.
        
        B_bar = (1/n) * sum_i B_{k_i}^{S_i}
        
        Args:
            model: The language model
            dataloader: DataLoader for sampling sequences
            n_samples: Number of random (sequence, position) pairs
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (average_bigram_score, average_bigram_entropy)
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before computing scores")
        
        torch.manual_seed(seed)
        
        model.eval()
        scores = []
        entropies = []
        
        data_iter = iter(dataloader)
        samples_collected = 0
        
        with torch.no_grad():
            while samples_collected < n_samples:
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(dataloader)
                    batch = next(data_iter)
                
                input_ids = batch["input_ids"].to(self.device)
                batch_size, seq_len = input_ids.shape
                
                if seq_len < 2:
                    continue
                
                # Run model on the full batch once
                logits = model(input_ids)  # (batch_size, seq_len, vocab_size)
                
                # Sample one random position per sequence in batch
                for b in range(batch_size):
                    if samples_collected >= n_samples:
                        break
                    
                    # Random position k (must have k+1 valid)
                    k = torch.randint(0, seq_len - 1, (1,)).item()
                    
                    # Get model log-probs at position k from the batched logits
                    log_probs = F.log_softmax(logits[b, k, :], dim=-1)
                    
                    # Get empirical distribution for this context token
                    context_token = input_ids[b, k].item()
                    q = self.get_conditional_distribution(context_token)
                    
                    # Cross entropy: -sum_i q(i) * log p(i)
                    score = -torch.sum(q * log_probs).item()
                    entropy = self.compute_bigram_entropy(context_token)
                    
                    scores.append(score)
                    entropies.append(entropy)
                    samples_collected += 1
        
        avg_score = sum(scores) / len(scores)
        avg_entropy = sum(entropies) / len(entropies)
        
        return avg_score, avg_entropy
    
    def get_stats(self) -> Dict:
        """Get statistics about the fitted bigram distribution."""
        if not self._is_fitted:
            return {"fitted": False}
        
        n_contexts = len(self.bigram_counts)
        n_bigrams = sum(len(v) for v in self.bigram_counts.values())
        
        return {
            "fitted": True,
            "total_bigrams": self.total_bigrams,
            "unique_contexts": n_contexts,
            "unique_bigrams": n_bigrams,
        }
