"""N-gram score analysis for tracking model behavior.

Implements the n-gram score metric:
n-gram score = l_test^n / l_gram^n

Where l_test^n is the average loss on validation data at position n,
and l_gram^n is the average loss on predicting final tokens of common n-grams.
"""
import torch
import torch.nn.functional as F
from collections import defaultdict
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
        max_n: int = 5,
        max_samples: Optional[int] = 10000,
    ) -> "NgramAnalyzer":
        """Extract common n-grams from training data by frequency.
        
        Args:
            dataloader: DataLoader yielding batches with 'input_ids'
            max_n: Maximum n to consider
            max_samples: Maximum samples to process
            
        Returns:
            self for method chaining
        """
        # Count all n-grams
        ngram_counts: Dict[int, Dict[Tuple[int, ...], int]] = {
            n: defaultdict(int) for n in range(2, max_n + 1)
        }
        
        n_samples = 0
        for batch in tqdm(dataloader, desc="Counting n-grams"):
            input_ids = batch["input_ids"]
            
            for seq in input_ids:
                seq = seq.tolist()
                for n in range(2, max_n + 1):
                    for i in range(len(seq) - n + 1):
                        ngram = tuple(seq[i:i+n])
                        ngram_counts[n][ngram] += 1
                
                n_samples += 1
                if max_samples is not None and n_samples >= max_samples:
                    break
            
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
        batch_size: int = 32,
    ) -> float:
        """Compute average loss on predicting final tokens of common n-grams.
        
        Args:
            model: The language model
            n: The n-gram size (e.g., 2 for bigrams)
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
            # Process in batches
            for i in range(0, len(ngrams), batch_size):
                batch_ngrams = ngrams[i:i+batch_size]
                
                # Pad contexts to same length (n-1)
                max_ctx_len = n - 1
                contexts = []
                targets = []
                
                for context, final in batch_ngrams:
                    # Pad context if needed
                    padded = [0] * (max_ctx_len - len(context)) + context
                    contexts.append(padded)
                    targets.append(final)
                
                input_ids = torch.tensor(contexts, device=self.device)
                target_ids = torch.tensor(targets, device=self.device)
                
                # Get model predictions
                logits = model(input_ids)  # (batch, ctx_len, vocab)
                
                # Get logits for final position (predicting the n-gram's last token)
                final_logits = logits[:, -1, :]  # (batch, vocab)
                
                # Compute cross-entropy loss
                loss = F.cross_entropy(final_logits, target_ids, reduction='sum')
                total_loss += loss.item()
                n_samples += len(batch_ngrams)
        
        return total_loss / n_samples if n_samples > 0 else float('nan')
    
    def compute_position_loss(
        self,
        model: torch.nn.Module,
        dataloader,
        position: int,
        max_batches: Optional[int] = 100,
    ) -> float:
        """Compute average loss at a specific position in validation sequences.
        
        Args:
            model: The language model
            dataloader: Validation dataloader
            position: Position n to evaluate (0-indexed)
            max_batches: Maximum batches to process
            
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
                
                input_ids = batch["input_ids"].to(self.device)
                batch_size, seq_len = input_ids.shape
                
                if position >= seq_len - 1:
                    continue
                
                # Get model predictions
                logits = model(input_ids)
                
                # Get logits at position (predicting position+1)
                pos_logits = logits[:, position, :]
                targets = input_ids[:, position + 1]
                
                # Compute loss
                loss = F.cross_entropy(pos_logits, targets, reduction='sum')
                total_loss += loss.item()
                n_samples += batch_size
        
        return total_loss / n_samples if n_samples > 0 else float('nan')
    
    def compute_ngram_score(
        self,
        model: torch.nn.Module,
        val_dataloader,
        n: int,
        max_val_batches: int = 100,
    ) -> Dict[str, float]:
        """Compute the n-gram score: l_test^n / l_gram^n.
        
        Args:
            model: The language model
            val_dataloader: Validation dataloader
            n: The n-gram size
            max_val_batches: Max batches for validation loss
            
        Returns:
            Dict with ngram_loss, position_loss, and ngram_score
        """
        # Loss on predicting final tokens of common n-grams
        ngram_loss = self.compute_ngram_loss(model, n)
        
        # Loss at position n-1 on validation data (0-indexed, so position n-1 predicts token n)
        position_loss = self.compute_position_loss(
            model, val_dataloader, position=n-1, max_batches=max_val_batches
        )
        
        # N-gram score is the ratio
        if ngram_loss > 0:
            score = position_loss / ngram_loss
        else:
            score = float('nan')
        
        return {
            f"{n}gram_loss": ngram_loss,
            f"position_{n}_loss": position_loss,
            f"{n}gram_score": score,
        }
    
    def compute_all_ngram_scores(
        self,
        model: torch.nn.Module,
        val_dataloader,
        max_val_batches: int = 100,
    ) -> Dict[str, float]:
        """Compute n-gram scores for all fitted n values.
        
        Args:
            model: The language model
            val_dataloader: Validation dataloader
            max_val_batches: Max batches for validation loss
            
        Returns:
            Dict with all metrics
        """
        if not self._is_fitted:
            raise RuntimeError("Must fit n-grams first")
        
        results = {}
        for n in sorted(self.common_ngrams.keys()):
            scores = self.compute_ngram_score(model, val_dataloader, n, max_val_batches)
            results.update(scores)
        
        return results
    
    def get_stats(self) -> Dict:
        """Get statistics about fitted n-grams."""
        if not self._is_fitted:
            return {"fitted": False}
        
        stats = {"fitted": True}
        for n, ngrams in self.common_ngrams.items():
            stats[f"n={n}_count"] = len(ngrams)
        
        return stats
