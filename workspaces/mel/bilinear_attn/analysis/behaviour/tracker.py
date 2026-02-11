"""Behaviour tracker for monitoring model metrics during training.

Provides a unified interface for computing and logging various behavioral
metrics that can be toggled on/off during training.
"""
import json
import torch
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field

from .bigram import BigramAnalyzer
from .ngram import NgramAnalyzer


@dataclass
class TrackerConfig:
    """Configuration for the behaviour tracker."""
    # Bigram metrics
    bigram_enabled: bool = True
    bigram_compute_every: int = 500
    bigram_n_samples: int = 1000
    
    # N-gram metrics  
    ngram_enabled: bool = True
    ngram_compute_every: int = 500
    ngram_max_n: int = 4
    ngram_max_val_batches: int = 50
    
    # Loss tracking
    loss_enabled: bool = True
    
    # General
    seed: int = 42


class BehaviourTracker:
    """Tracks behavioral metrics during training.
    
    Provides modular tracking of:
    - Training/validation loss
    - Bigram scores
    - N-gram scores
    
    Metrics can be toggled on/off and computed at different intervals.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader,
        val_dataloader,
        vocab_size: int,
        device: str = "cpu",
        config: Optional[TrackerConfig] = None,
        run_dir: Optional[str] = None,
        tokenizer=None,
    ):
        """Initialize the behaviour tracker.
        
        Args:
            model: The language model to track
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            vocab_size: Vocabulary size
            device: Device for computations
            config: Tracker configuration
            run_dir: Directory to save metrics
            tokenizer: Tokenizer (optional, for n-gram extraction)
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.vocab_size = vocab_size
        self.device = device
        self.config = config or TrackerConfig()
        self.tokenizer = tokenizer
        
        # Set up run directory
        if run_dir is not None:
            self.run_dir = Path(run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.run_dir = None
        
        # Initialize analyzers
        self.bigram_analyzer: Optional[BigramAnalyzer] = None
        self.ngram_analyzer: Optional[NgramAnalyzer] = None
        
        # Metrics history
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Cached metrics file
        self._metrics_file = self.run_dir / "behaviour_metrics.jsonl" if self.run_dir else None
        
        self._is_fitted = False
    
    def fit(self, max_fit_samples: int = 10000) -> "BehaviourTracker":
        """Fit the analyzers on training data.
        
        This should be called before training starts.
        
        Args:
            max_fit_samples: Maximum samples for fitting distributions
            
        Returns:
            self for method chaining
        """
        if self.config.bigram_enabled:
            print("Fitting bigram analyzer...")
            self.bigram_analyzer = BigramAnalyzer(
                vocab_size=self.vocab_size,
                device=self.device,
            )
            self.bigram_analyzer.fit(self.train_dataloader, max_samples=max_fit_samples)
            print(f"  Bigram stats: {self.bigram_analyzer.get_stats()}")
        
        if self.config.ngram_enabled:
            print("Fitting n-gram analyzer...")
            self.ngram_analyzer = NgramAnalyzer(
                vocab_size=self.vocab_size,
                device=self.device,
                max_common_ngrams=1000,
            )
            self.ngram_analyzer.extract_common_ngrams_from_data(
                self.train_dataloader,
                max_n=self.config.ngram_max_n,
                max_samples=max_fit_samples,
            )
            print(f"  N-gram stats: {self.ngram_analyzer.get_stats()}")
        
        self._is_fitted = True
        return self
    
    def compute_metrics(self, step: int) -> Dict[str, Any]:
        """Compute all enabled metrics at the current step.
        
        Args:
            step: Current training step
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {"step": step}
        
        # Bigram metrics
        if (self.config.bigram_enabled and 
            self.bigram_analyzer is not None and
            step % self.config.bigram_compute_every == 0):
            
            bigram_score, bigram_entropy = self.bigram_analyzer.compute_average_bigram_score(
                self.model,
                self.val_dataloader,
                n_samples=self.config.bigram_n_samples,
                seed=self.config.seed,
            )
            metrics["bigram_score"] = bigram_score
            metrics["bigram_entropy"] = bigram_entropy
            metrics["bigram_gap"] = bigram_score - bigram_entropy
        
        # N-gram metrics
        if (self.config.ngram_enabled and
            self.ngram_analyzer is not None and
            step % self.config.ngram_compute_every == 0):
            
            ngram_metrics = self.ngram_analyzer.compute_all_ngram_scores(
                self.model,
                self.val_dataloader,
                max_val_batches=self.config.ngram_max_val_batches,
            )
            metrics.update(ngram_metrics)
        
        return metrics
    
    def should_compute(self, step: int) -> bool:
        """Check if any metrics should be computed at this step.
        
        Args:
            step: Current training step
            
        Returns:
            True if any metric should be computed
        """
        if self.config.bigram_enabled and step % self.config.bigram_compute_every == 0:
            return True
        if self.config.ngram_enabled and step % self.config.ngram_compute_every == 0:
            return True
        return False
    
    def log_metrics(self, step: int, additional_metrics: Optional[Dict] = None) -> Dict[str, Any]:
        """Compute and log metrics at the current step.
        
        Args:
            step: Current training step
            additional_metrics: Additional metrics to include (e.g., loss)
            
        Returns:
            All logged metrics
        """
        metrics = self.compute_metrics(step)
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        # Only log if we have meaningful metrics beyond step
        if len(metrics) > 1:
            self.metrics_history.append(metrics)
            
            # Write to file
            if self._metrics_file is not None:
                with open(self._metrics_file, "a") as f:
                    f.write(json.dumps(metrics) + "\n")
        
        return metrics
    
    def log_loss(self, step: int, train_loss: float, val_loss: Optional[float] = None):
        """Log loss values (convenience method).
        
        Args:
            step: Current training step
            train_loss: Training loss
            val_loss: Validation loss (optional)
        """
        metrics = {"step": step, "train_loss": train_loss}
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        
        self.metrics_history.append(metrics)
        
        if self._metrics_file is not None:
            with open(self._metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get the full metrics history.
        
        Returns:
            List of metric dictionaries
        """
        return self.metrics_history
    
    def get_metric_series(self, metric_name: str) -> tuple:
        """Extract a specific metric series from history.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Tuple of (steps, values) lists
        """
        steps = []
        values = []
        
        for entry in self.metrics_history:
            if metric_name in entry:
                steps.append(entry["step"])
                values.append(entry[metric_name])
        
        return steps, values
    
    def save_history(self, path: Optional[str] = None):
        """Save metrics history to a JSON file.
        
        Args:
            path: Output path (defaults to run_dir/behaviour_metrics_full.json)
        """
        if path is None:
            if self.run_dir is None:
                raise ValueError("No run_dir or path specified")
            path = self.run_dir / "behaviour_metrics_full.json"
        
        with open(path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)
    
    def load_history(self, path: str):
        """Load metrics history from a JSON file.
        
        Args:
            path: Path to the JSON file
        """
        with open(path, "r") as f:
            self.metrics_history = json.load(f)
    
    def toggle_bigram(self, enabled: bool):
        """Toggle bigram metrics on/off."""
        self.config.bigram_enabled = enabled
    
    def toggle_ngram(self, enabled: bool):
        """Toggle n-gram metrics on/off."""
        self.config.ngram_enabled = enabled
    
    def set_compute_interval(self, metric: str, interval: int):
        """Set the compute interval for a metric.
        
        Args:
            metric: 'bigram' or 'ngram'
            interval: Steps between computations
        """
        if metric == "bigram":
            self.config.bigram_compute_every = interval
        elif metric == "ngram":
            self.config.ngram_compute_every = interval
        else:
            raise ValueError(f"Unknown metric: {metric}")
