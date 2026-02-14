"""Behaviour tracker for monitoring model metrics during training.

Provides a unified interface for computing and logging various behavioral
metrics that can be toggled on/off during training or evaluated post-hoc
on saved checkpoints.
"""
import json
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field

from .bigram import BigramAnalyzer
from .ngram import NgramAnalyzer
from .ablation import compute_ablated_loss, compute_val_loss
from .icl import compute_icl_score


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
    
    # Position-ablated loss
    ablation_enabled: bool = True
    ablation_compute_every: int = 500
    ablation_max_val_batches: int = 50
    
    # ICL score
    icl_enabled: bool = True
    icl_compute_every: int = 500
    icl_k1: int = 50
    icl_k2: int = 500
    icl_max_val_batches: Optional[int] = None
    
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
        cache_dir: Optional[str] = None,
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
            cache_dir: Directory for cached bigram/ngram distributions.
                       If provided, fit() will load from cache when available
                       and save to cache after fitting.
            tokenizer: Tokenizer (optional, for n-gram extraction)
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.vocab_size = vocab_size
        self.device = device
        self.config = config or TrackerConfig()
        self.tokenizer = tokenizer
        
        # BOS token id for n-gram evaluation
        self.bos_token_id = getattr(tokenizer, "bos_token_id", 0) if tokenizer else 0
        
        # Set up run directory
        if run_dir is not None:
            self.run_dir = Path(run_dir)
            self.run_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.run_dir = None
        
        # Set up cache directory
        if cache_dir is not None:
            self.cache_dir = Path(cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        else:
            self.cache_dir = None
        
        # Initialize analyzers
        self.bigram_analyzer: Optional[BigramAnalyzer] = None
        self.ngram_analyzer: Optional[NgramAnalyzer] = None
        
        # Metrics history
        self.metrics_history: List[Dict[str, Any]] = []
        
        # Cached metrics file
        self._metrics_file = self.run_dir / "behaviour_metrics.jsonl" if self.run_dir else None
        
        self._is_fitted = False
    
    def _bigram_cache_path(self) -> Optional[Path]:
        """Return the cache file path for bigram data, or None."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / "bigram.pt"
    
    def _ngram_cache_path(self) -> Optional[Path]:
        """Return the cache file path for ngram data, or None."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / "ngram.pt"
    
    def fit(self, max_fit_samples: int = 10000) -> "BehaviourTracker":
        """Fit the analyzers on training data, or load from cache.
        
        If a cache_dir was provided and cached files exist, the analyzers
        are loaded from disk instead of re-fitting.  After a fresh fit the
        distributions are saved to cache_dir for next time.
        
        Args:
            max_fit_samples: Maximum samples for fitting distributions
            
        Returns:
            self for method chaining
        """
        bigram_path = self._bigram_cache_path()
        ngram_path = self._ngram_cache_path()
        
        # --- Bigram ---
        if self.config.bigram_enabled:
            if bigram_path is not None and bigram_path.exists():
                print(f"Loading cached bigram distribution from {bigram_path}")
                self.bigram_analyzer = BigramAnalyzer.load(str(bigram_path), device=self.device)
            else:
                print("Fitting bigram analyzer...")
                self.bigram_analyzer = BigramAnalyzer(
                    vocab_size=self.vocab_size,
                    device=self.device,
                )
                self.bigram_analyzer.fit(self.train_dataloader, max_samples=max_fit_samples)
                if bigram_path is not None:
                    self.bigram_analyzer.save(str(bigram_path))
                    print(f"  Saved bigram cache to {bigram_path}")
            print(f"  Bigram stats: {self.bigram_analyzer.get_stats()}")
        
        # --- N-gram ---
        if self.config.ngram_enabled:
            if ngram_path is not None and ngram_path.exists():
                print(f"Loading cached n-gram distribution from {ngram_path}")
                self.ngram_analyzer = NgramAnalyzer.load(str(ngram_path), device=self.device)
            else:
                print("Fitting n-gram analyzer...")
                self.ngram_analyzer = NgramAnalyzer(
                    vocab_size=self.vocab_size,
                    device=self.device,
                    max_common_ngrams=1000,
                )
                self.ngram_analyzer.extract_common_ngrams_from_data(
                    self.train_dataloader,
                    tokenizer=self.tokenizer,
                    max_n=self.config.ngram_max_n,
                    max_samples=max_fit_samples,
                )
                if ngram_path is not None:
                    self.ngram_analyzer.save(str(ngram_path))
                    print(f"  Saved n-gram cache to {ngram_path}")
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
                self.val_dataloader.dataset,
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
                bos_token_id=self.bos_token_id,
                max_val_batches=self.config.ngram_max_val_batches,
            )
            metrics.update(ngram_metrics)
        
        # ICL score
        if (self.config.icl_enabled and
            step % self.config.icl_compute_every == 0):
            
            icl_metrics = compute_icl_score(
                self.model,
                self.val_dataloader,
                bos_token_id=self.bos_token_id,
                k1=self.config.icl_k1,
                k2=self.config.icl_k2,
                device=self.device,
                max_batches=self.config.icl_max_val_batches,
            )
            metrics.update(icl_metrics)
        
        # Position-ablated loss
        if (self.config.ablation_enabled and
            step % self.config.ablation_compute_every == 0):
            
            val_loss = compute_val_loss(
                self.model, self.val_dataloader,
                device=self.device,
                max_batches=self.config.ablation_max_val_batches,
            )
            ablated_loss = compute_ablated_loss(
                self.model, self.val_dataloader,
                device=self.device,
                max_batches=self.config.ablation_max_val_batches,
            )
            metrics["val_loss"] = val_loss
            metrics["ablated_loss"] = ablated_loss
            metrics["ablation_gap"] = ablated_loss - val_loss
        
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
        if self.config.ablation_enabled and step % self.config.ablation_compute_every == 0:
            return True
        if self.config.icl_enabled and step % self.config.icl_compute_every == 0:
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
    
    # ------------------------------------------------------------------
    # Checkpoint evaluation
    # ------------------------------------------------------------------
    
    def evaluate_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """Load model weights from a checkpoint and compute all enabled metrics.
        
        The checkpoint must contain a 'model_state_dict' key (the format
        produced by Trainer.save_checkpoint).  The training step is read
        from the checkpoint's 'step' key when present.
        
        Args:
            checkpoint_path: Path to a .pt checkpoint file
            
        Returns:
            Dictionary of computed metrics (includes 'step' and 'checkpoint')
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before evaluating checkpoints")
        
        checkpoint_path = Path(checkpoint_path)
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        
        step = ckpt.get("step", 0)
        
        # Force-compute all enabled metrics regardless of interval
        metrics: Dict[str, Any] = {"step": step, "checkpoint": str(checkpoint_path)}
        
        if self.config.bigram_enabled and self.bigram_analyzer is not None:
            bigram_score, bigram_entropy = self.bigram_analyzer.compute_average_bigram_score(
                self.model,
                self.val_dataloader.dataset,
                n_samples=self.config.bigram_n_samples,
                seed=self.config.seed,
            )
            metrics["bigram_score"] = bigram_score
            metrics["bigram_entropy"] = bigram_entropy
            metrics["bigram_gap"] = bigram_score - bigram_entropy
        
        if self.config.ngram_enabled and self.ngram_analyzer is not None:
            ngram_metrics = self.ngram_analyzer.compute_all_ngram_scores(
                self.model,
                self.val_dataloader,
                bos_token_id=self.bos_token_id,
                max_val_batches=self.config.ngram_max_val_batches,
            )
            metrics.update(ngram_metrics)
        
        if self.config.ablation_enabled:
            val_loss = compute_val_loss(
                self.model, self.val_dataloader,
                device=self.device,
                max_batches=self.config.ablation_max_val_batches,
            )
            ablated_loss = compute_ablated_loss(
                self.model, self.val_dataloader,
                device=self.device,
                max_batches=self.config.ablation_max_val_batches,
            )
            metrics["val_loss"] = val_loss
            metrics["ablated_loss"] = ablated_loss
            metrics["ablation_gap"] = ablated_loss - val_loss
        
        if self.config.icl_enabled:
            icl_metrics = compute_icl_score(
                self.model,
                self.val_dataloader,
                bos_token_id=self.bos_token_id,
                k1=self.config.icl_k1,
                k2=self.config.icl_k2,
                device=self.device,
                max_batches=self.config.icl_max_val_batches,
            )
            metrics.update(icl_metrics)
        
        return metrics
    
    def evaluate_checkpoints(
        self,
        checkpoint_paths: Optional[List[Union[str, Path]]] = None,
        run_dir: Optional[Union[str, Path]] = None,
        save: bool = True,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """Evaluate behaviour metrics across multiple checkpoints.
        
        Either provide an explicit list of checkpoint paths, or a run_dir
        whose ``checkpoints/`` subdirectory will be scanned for ``*.pt``
        files (sorted by training step).
        
        Args:
            checkpoint_paths: Explicit list of checkpoint files
            run_dir: Run directory containing a checkpoints/ folder.
                     Ignored when checkpoint_paths is given.
            save: If True, save results to run_dir or self.run_dir
            verbose: Print progress
            
        Returns:
            List of metric dicts, one per checkpoint, sorted by step
        """
        if not self._is_fitted:
            raise RuntimeError("Must call fit() before evaluating checkpoints")
        
        # Resolve checkpoint list
        if checkpoint_paths is None:
            scan_dir = Path(run_dir) if run_dir else self.run_dir
            if scan_dir is None:
                raise ValueError("Provide checkpoint_paths or run_dir")
            ckpt_dir = scan_dir / "checkpoints"
            if not ckpt_dir.exists():
                raise FileNotFoundError(f"No checkpoints directory at {ckpt_dir}")
            checkpoint_paths = sorted(ckpt_dir.glob("*.pt"))
            if not checkpoint_paths:
                raise FileNotFoundError(f"No .pt files found in {ckpt_dir}")
        
        all_metrics: List[Dict[str, Any]] = []
        for i, ckpt_path in enumerate(checkpoint_paths):
            if verbose:
                print(f"[{i+1}/{len(checkpoint_paths)}] Evaluating {Path(ckpt_path).name} ...")
            metrics = self.evaluate_checkpoint(ckpt_path)
            all_metrics.append(metrics)
            if verbose:
                summary = {k: f"{v:.4f}" if isinstance(v, float) else v
                           for k, v in metrics.items() if k != "checkpoint"}
                print(f"  {summary}")
        
        # Sort by step
        all_metrics.sort(key=lambda m: m["step"])
        
        # Merge into history
        self.metrics_history.extend(all_metrics)
        
        # Save
        if save:
            save_dir = Path(run_dir) if run_dir else self.run_dir
            if save_dir is not None:
                save_dir.mkdir(parents=True, exist_ok=True)
                out_path = save_dir / "behaviour_metrics.jsonl"
                with open(out_path, "a") as f:
                    for m in all_metrics:
                        f.write(json.dumps(m) + "\n")
                if verbose:
                    print(f"Metrics appended to {out_path}")
        
        return all_metrics
    
    @classmethod
    def from_run_dir(
        cls,
        run_dir: Union[str, Path],
        config: Optional["TrackerConfig"] = None,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cfg_path: Optional[Union[str, Path]] = None,
    ) -> "BehaviourTracker":
        """Construct a BehaviourTracker from a saved training run.
        
        Rebuilds the model from the config, creates dataloaders, and
        sets up analyzer caches so that ``fit()`` can load from cache.
        
        The config YAML is located by searching (in order):
          1. ``cfg_path`` if provided
          2. ``<run_dir>/config.yaml``
          3. The first ``.yaml`` file found in ``run_dir``
        
        Args:
            run_dir: Path to the training run directory
            config: TrackerConfig override (uses defaults if None)
            device: Device string (auto-detected if None)
            cache_dir: Cache directory for analyzers (defaults to
                       ``<run_dir>/behaviour_cache``)
            cfg_path: Explicit path to the config YAML
            
        Returns:
            A BehaviourTracker ready for ``fit()`` and checkpoint evaluation
        """
        from models import AttentionLM
        from data import create_dataloaders
        
        run_dir = Path(run_dir)
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Locate config
        if cfg_path is not None:
            cfg_file = Path(cfg_path)
        elif (run_dir / "config.yaml").exists():
            cfg_file = run_dir / "config.yaml"
        else:
            yamls = list(run_dir.glob("*.yaml"))
            if not yamls:
                raise FileNotFoundError(
                    f"No config YAML found in {run_dir}. Pass cfg_path explicitly."
                )
            cfg_file = yamls[0]
        
        with open(cfg_file) as f:
            cfg = yaml.safe_load(f)
        
        model_cfg = cfg["model"]
        train_cfg = cfg.get("train", {})
        
        # Build model and dataloaders
        model = AttentionLM.from_config(cfg)
        model.to(device)
        
        train_dataloader, val_dataloader = create_dataloaders(
            n_ctx=model_cfg["n_ctx"],
            batch_size=train_cfg.get("batch_size", 16),
            max_train_samples=train_cfg.get("max_train_samples"),
            max_val_samples=train_cfg.get("max_val_samples", 1000),
        )
        
        cache_dir = cache_dir or str(run_dir / "behaviour_cache")
        
        tracker = cls(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            vocab_size=model_cfg["vocab_size"],
            device=device,
            config=config or TrackerConfig(),
            run_dir=str(run_dir),
            cache_dir=cache_dir,
        )
        
        return tracker
