"""N-gram score analysis for tracking model behavior.

Implements the n-gram score metric:
n-gram score = l_test^n / l_gram^n

Where l_test^n is the average loss on validation data at position n,
and l_gram^n is the average loss on predicting final tokens of common n-grams.
"""
import torch
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm


class NgramAnalyzer:
    """Analyzes n-gram statistics and computes n-gram scores."""
    
    def __init__(
        self,
        vocab_size: int,
        tokenizer=None,
        device: str = "cpu",
        max_common_ngrams: int = 1000,
    ):
        """Initialize the n-gram analyzer.
        
        Args:
            vocab_size: Size of the model's vocabulary
            tokenizer: Tokenizer for extracting n-grams
            device: Device for computations
            max_common_ngrams: Number of common n-grams to track per n
        """
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.device = device
        self.max_common_ngrams = max_common_ngrams
        
        # Common n-grams by n: {n: [(token_sequence, final_token), ...]}
        self.common_ngrams: Dict[int, List[Tuple[List[int], int]]] = {}
        
        # N-gram counts from training data
        self.ngram_counts: Dict[int, Dict[Tuple[int, ...], int]] = defaultdict(lambda: defaultdict(int))
        
        self._is_fitted = False
    
    def extract_common_ngrams_from_gpt2(
        self,
        gpt2_tokenizer,
        model_tokenizer,
        max_n: int = 5,
    ) -> "NgramAnalyzer":
        """Extract common n-grams by comparing with GPT-2 vocabulary.
        
        The GPT-2 vocabulary is constructed by iteratively merging frequent
        token pairs, so earlier entries tend to be more common n-grams.
        
        Args:
            gpt2_tokenizer: The full GPT-2 tokenizer
            model_tokenizer: The model's tokenizer (for re-tokenizing)
            max_n: Maximum n to consider
            
        Returns:
            self for method chaining
        """
        # Get all tokens from GPT-2 vocabulary
        gpt2_vocab = gpt2_tokenizer.get_vocab()
        sorted_tokens = sorted(gpt2_vocab.items(), key=lambda x: x[1])
        
        # Track n-grams by their length
        ngrams_by_n: Dict[int, List[Tuple[List[int], int]]] = defaultdict(list)
        
        for token_str, token_id in tqdm(sorted_tokens, desc="Extracting n-grams"):
            # Decode and re-tokenize with model tokenizer
            try:
                decoded = gpt2_tokenizer.decode([token_id])
                retokenized = model_tokenizer.encode(decoded, add_special_tokens=False)
            except:
                continue
            
            n = len(retokenized)
            if n < 2 or n > max_n:
                continue
            
            # Store as (context_tokens, final_token)
            if len(ngrams_by_n[n]) < self.max_common_ngrams:
                context = retokenized[:-1]
                final = retokenized[-1]
                ngrams_by_n[n].append((context, final))
        
        self.common_ngrams = dict(ngrams_by_n)
        self._is_fitted = True
        return self
    
    def extract_common_ngrams_from_data(
        self,
        dataloader,
        tokenizer=None,
        max_n: int = 5,
        max_samples: Optional[int] = None,
    ) -> "NgramAnalyzer":
        """Extract common n-grams from training data by frequency.
        
        N-grams containing special tokens (BOS, EOS, PAD, UNK) are excluded.
        
        Args:
            dataloader: DataLoader yielding batches with 'input_ids'
            tokenizer: HuggingFace tokenizer (used to identify special token ids).
                       If None, falls back to self.tokenizer.
            max_n: Maximum n to consider
            max_samples: Maximum samples to process (None = full corpus)
            
        Returns:
            self for method chaining
        """
        tok = tokenizer or self.tokenizer
        
        # Collect special token ids to exclude
        special_ids: Set[int] = set()
        if tok is not None:
            for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
                tid = getattr(tok, attr, None)
                if tid is not None:
                    special_ids.add(tid)
        
        # Count all n-grams using vectorised unfold per batch
        ngram_counts: Dict[int, Dict[Tuple[int, ...], int]] = {
            n: defaultdict(int) for n in range(2, max_n + 1)
        }
        
        n_samples = 0
        for batch in tqdm(dataloader, desc="Counting n-grams"):
            input_ids = batch["input_ids"]  # (B, T)
            attn_mask = batch.get("attention_mask", None)  # (B, T) or None
            batch_size = input_ids.shape[0]
            
            # Determine how many sequences to use from this batch
            if max_samples is not None:
                take = min(batch_size, max_samples - n_samples)
                input_ids = input_ids[:take]
                if attn_mask is not None:
                    attn_mask = attn_mask[:take]
            
            # Build a boolean mask: True where token is NOT special
            not_special = torch.ones_like(input_ids, dtype=torch.bool)
            for sid in special_ids:
                not_special &= (input_ids != sid)
            # Also respect attention_mask (padding)
            if attn_mask is not None:
                not_special &= attn_mask.bool()
            
            # Use unfold to extract all n-gram windows as tensors
            for n in range(2, max_n + 1):
                # windows: (B, T-n+1, n)
                windows = input_ids.unfold(1, n, 1)
                # validity: every position in the window must be non-special
                valid_windows = not_special.unfold(1, n, 1).min(dim=2).values.bool()
                valid = windows[valid_windows]  # (num_valid, n)
                
                for row in valid.tolist():
                    ngram_counts[n][tuple(row)] += 1
            
            n_samples += input_ids.shape[0]
            if max_samples is not None and n_samples >= max_samples:
                break
        
        # Get top common n-grams for each n
        self.common_ngrams = {}
        for n in range(2, max_n + 1):
            sorted_ngrams = sorted(
                ngram_counts[n].items(),
                key=lambda x: x[1],
                reverse=True
            )[:self.max_common_ngrams]
            
            # Store as (context_tokens, final_token)
            self.common_ngrams[n] = [
                (list(ngram[:-1]), ngram[-1])
                for ngram, count in sorted_ngrams
            ]
        
        self.ngram_counts = ngram_counts
        self._is_fitted = True
        return self
    
    def compute_ngram_loss(
        self,
        model: torch.nn.Module,
        n: int,
        bos_token_id: int,
        batch_size: int = 128,
    ) -> float:
        """Compute l_ngram(n): avg CE predicting t_n given [BOS, t_1, ..., t_{n-1}].
        
        For each n-gram (t_1, ..., t_n) in the probe set:
          - input:  [BOS, t_1, ..., t_{n-1}]   (length n)
          - target: t_n
          - loss:   CE(logits[:, -1, :], t_n)
        
        Args:
            model: The language model
            n: The n-gram size (e.g., 2 for bigrams)
            bos_token_id: BOS token id to prepend
            batch_size: Batch size for processing
            
        Returns:
            Average cross-entropy loss on n-gram final token prediction
        """
        if not self._is_fitted or n not in self.common_ngrams:
            raise RuntimeError(f"No common {n}-grams fitted. Call extract_common_ngrams first.")
        
        ngrams = self.common_ngrams[n]
        if len(ngrams) == 0:
            return float('nan')
        
        model.eval()
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for i in range(0, len(ngrams), batch_size):
                batch_ngrams = ngrams[i:i+batch_size]
                bs = len(batch_ngrams)
                
                # Build input: [BOS, t_1, ..., t_{n-1}]  (length n)
                inputs = []
                targets = []
                for context, final in batch_ngrams:
                    inputs.append([bos_token_id] + context)
                    targets.append(final)
                
                input_ids = torch.tensor(inputs, device=self.device)   # (bs, n)
                target_ids = torch.tensor(targets, device=self.device)  # (bs,)
                
                logits = model(input_ids)  # (bs, n, vocab)
                final_logits = logits[:, -1, :]  # (bs, vocab)
                
                loss = F.cross_entropy(final_logits, target_ids, reduction='sum')
                total_loss += loss.item()
                n_samples += bs
        
        return total_loss / n_samples if n_samples > 0 else float('nan')
    
    def compute_test_loss(
        self,
        model: torch.nn.Module,
        dataloader,
        n: int,
        bos_token_id: int,
        max_batches: Optional[int] = None,
    ) -> float:
        """Compute l_test(n): avg CE predicting x_n given [BOS, x_1, ..., x_{n-1}].
        
        For each validation sequence with at least n real tokens:
          - input:  [BOS, x_1, ..., x_{n-1}]   (length n)
          - target: x_n
          - loss:   CE(logits[:, -1, :], x_n)
        
        Args:
            model: The language model
            dataloader: Validation dataloader
            n: The n-gram size (number of real tokens needed)
            bos_token_id: BOS token id to prepend
            max_batches: Maximum batches to process (None = all)
            
        Returns:
            Average cross-entropy loss at position n
        """
        model.eval()
        total_loss = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if max_batches is not None and i >= max_batches:
                    break
                
                input_ids = batch["input_ids"].to(self.device)  # (B, T)
                attn_mask = batch.get("attention_mask")
                if attn_mask is not None:
                    attn_mask = attn_mask.to(self.device)
                
                B, T = input_ids.shape
                if T < n:
                    continue
                
                # Only keep sequences with at least n real tokens
                if attn_mask is not None:
                    real_len = attn_mask.sum(dim=1)  # (B,)
                    valid = real_len >= n
                    if not valid.any():
                        continue
                    input_ids = input_ids[valid]
                    B = input_ids.shape[0]
                
                # Build input: [BOS, x_1, ..., x_{n-1}]  (length n)
                first_n = input_ids[:, :n]  # (B, n)
                bos_col = torch.full((B, 1), bos_token_id, dtype=torch.long, device=self.device)
                inp = torch.cat([bos_col, first_n[:, :-1]], dim=1)  # (B, n)
                targets = first_n[:, -1]  # (B,)  i.e. x_n
                
                logits = model(inp)  # (B, n, V)
                loss = F.cross_entropy(logits[:, -1, :], targets, reduction='sum')
                total_loss += loss.item()
                n_samples += B
        
        return total_loss / n_samples if n_samples > 0 else float('nan')
    
    def compute_ngram_score(
        self,
        model: torch.nn.Module,
        val_dataloader,
        n: int,
        bos_token_id: int,
        max_val_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute the n-gram score: l_test^n / l_ngram^n.
        
        Args:
            model: The language model
            val_dataloader: Validation dataloader
            n: The n-gram size
            bos_token_id: BOS token id to prepend
            max_val_batches: Max batches for validation loss (None = all)
            
        Returns:
            Dict with ngram_loss, test_loss, and ngram_score
        """
        ngram_loss = self.compute_ngram_loss(model, n, bos_token_id)
        test_loss = self.compute_test_loss(
            model, val_dataloader, n, bos_token_id, max_batches=max_val_batches
        )
        
        if ngram_loss > 0:
            score = test_loss / ngram_loss
        else:
            score = float('nan')
        
        return {
            f"{n}gram_loss": ngram_loss,
            f"{n}gram_test_loss": test_loss,
            f"{n}gram_score": score,
        }
    
    def compute_all_ngram_scores(
        self,
        model: torch.nn.Module,
        val_dataloader,
        bos_token_id: int,
        max_val_batches: Optional[int] = None,
    ) -> Dict[str, float]:
        """Compute n-gram scores for all fitted n values.
        
        Args:
            model: The language model
            val_dataloader: Validation dataloader
            bos_token_id: BOS token id to prepend
            max_val_batches: Max batches for validation loss (None = all)
            
        Returns:
            Dict with all metrics
        """
        if not self._is_fitted:
            raise RuntimeError("Must fit n-grams first")
        
        results = {}
        for n in sorted(self.common_ngrams.keys()):
            scores = self.compute_ngram_score(
                model, val_dataloader, n, bos_token_id, max_val_batches
            )
            results.update(scores)
        
        return results
    
    def save(self, path: str) -> None:
        """Save the fitted n-gram data to disk.
        
        Args:
            path: File path to save to (e.g. 'cache/ngram.pt')
        """
        if not self._is_fitted:
            raise RuntimeError("Must fit n-grams before saving")
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        # Convert defaultdicts to regular dicts for pickling
        ngram_counts = {
            n: dict(counts) for n, counts in self.ngram_counts.items()
        }
        
        torch.save({
            "vocab_size": self.vocab_size,
            "max_common_ngrams": self.max_common_ngrams,
            "common_ngrams": dict(self.common_ngrams),
            "ngram_counts": ngram_counts,
        }, path)
    
    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NgramAnalyzer":
        """Load fitted n-gram data from disk.
        
        Args:
            path: File path to load from
            device: Device for computations
            
        Returns:
            Fitted NgramAnalyzer instance
        """
        data = torch.load(path, weights_only=False)
        
        analyzer = cls(
            vocab_size=data["vocab_size"],
            device=device,
            max_common_ngrams=data["max_common_ngrams"],
        )
        
        analyzer.common_ngrams = data["common_ngrams"]
        
        # Restore defaultdicts for ngram_counts
        analyzer.ngram_counts = defaultdict(lambda: defaultdict(int))
        for n, counts in data["ngram_counts"].items():
            analyzer.ngram_counts[n] = defaultdict(int, counts)
        
        analyzer._is_fitted = True
        
        return analyzer
    
    def get_stats(self) -> Dict:
        """Get statistics about fitted n-grams."""
        if not self._is_fitted:
            return {"fitted": False}
        
        stats = {"fitted": True}
        for n, ngrams in self.common_ngrams.items():
            stats[f"n={n}_count"] = len(ngrams)
        
        return stats
