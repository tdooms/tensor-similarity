import json
import math
import os
import yaml
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from analysis.behaviour import BehaviourTracker

from .losses import compute_loss
from .optim import create_optimizer, create_scheduler, Optimizers, _is_muon_param
from .eval import evaluate
from .plotting import plot_loss, plot_debug, plot_weight_norms, plot_activation_norms
from .hf_utils import CheckpointUploadConfig, HuggingFaceCheckpointUploader

# Supported dtype strings → (torch dtype, whether to use GradScaler)
_DTYPE_MAP = {
    "float32": (torch.float32, False),
    "float16": (torch.float16, True),
    "bfloat16": (torch.bfloat16, False),
}


def _build_log_linear_checkpoint_steps(max_steps: int, checkpoint_count: int, alpha: float) -> list[int]:
    """Build a log-linear schedule of checkpoint steps.

    Generates ``checkpoint_count`` evenly-spaced points in ``[0, 1]``, bends
    them toward 0 with ``log1p((e-1)u)`` and mixes the curve in with the
    linear schedule via ``alpha`` (``1.0`` = pure log, ``0.0`` = pure
    linear). Steps are then scaled to ``[0, max_steps]``, rounded to ints,
    and deduplicated.

    Note: after rounding+deduplication the returned list may contain
    **fewer than** ``checkpoint_count`` entries (early log-spaced steps can
    collide, especially for small ``max_steps`` or large ``alpha``). Step
    0 and ``max_steps`` are always included.
    """
    if max_steps < 0:
        raise ValueError(f"max_steps must be >= 0, got {max_steps}")
    if checkpoint_count <= 0:
        raise ValueError(f"checkpoint_count must be > 0, got {checkpoint_count}")
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"checkpoint_log_linear_alpha must be in [0, 1], got {alpha}")

    steps: list[int] = []
    denom = max(1, checkpoint_count - 1)
    for i in range(checkpoint_count):
        u = i / denom
        log_u = math.log1p((math.e - 1.0) * u)
        mix_u = alpha * log_u + (1.0 - alpha) * u
        steps.append(int(round(mix_u * max_steps)))

    steps = sorted(set(steps))
    if 0 not in steps:
        steps.insert(0, 0)
    if max_steps not in steps:
        steps.append(max_steps)
    return steps


class Trainer:
    """Training loop for AttentionLM."""
    
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataloader,
        val_dataloader=None,
        cfg: dict = None,
        run_dir: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        behaviour_tracker: Optional["BehaviourTracker"] = None,
        wandb_run=None,
    ):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.cfg = cfg or {}
        self.device = device
        
        train_cfg = self.cfg.get("train", {})
        loss_cfg = self.cfg.get("loss", {})
        
        self.max_steps = int(float(train_cfg.get("max_steps", 1000)))
        self.warmup_steps = int(float(train_cfg.get("warmup_steps", 100)))
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.loss_type = loss_cfg.get("type", "next_token_ce")
        self.label_smoothing = loss_cfg.get("label_smoothing", 0.0)
        lr_decay_frac = train_cfg.get("lr_decay_frac", 0.1)
        
        # ---- precision (P4) ----
        dtype_str = train_cfg.get("dtype", "float32")
        # Backwards compat: old configs may have "amp: true" instead of dtype
        if "amp" in train_cfg and "dtype" not in train_cfg:
            dtype_str = "float16" if train_cfg["amp"] else "float32"
        self.pt_dtype, use_scaler = _DTYPE_MAP[dtype_str]
        self.use_amp = dtype_str != "float32" and device == "cuda"
        
        # ---- optimizer (P0 / P3) ----
        use_muon = train_cfg.get("use_muon", True)
        opt_result = create_optimizer(
            model,
            lr=train_cfg.get("lr", 3e-4),
            muon_lr=train_cfg.get("muon_lr", 0.02),
            weight_decay=train_cfg.get("weight_decay", 0.1),
            betas=tuple(train_cfg.get("betas", (0.9, 0.95))),
            use_muon=use_muon,
        )
        
        if isinstance(opt_result, Optimizers):
            self.optimizer = opt_result.muon
        else:
            self.optimizer = opt_result
        
        self.scheduler = create_scheduler(
            self.optimizer,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            lr_decay_frac=lr_decay_frac,
        )
        
        self.scaler = torch.amp.GradScaler("cuda") if (use_scaler and device == "cuda") else None
        
        if run_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            name = self.cfg.get("name", "run")
            run_dir = f"runs/{timestamp}_{name}"
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir = self.run_dir / "checkpoints"
        self.checkpoints_dir.mkdir(exist_ok=True)
        (self.run_dir / "errors").mkdir(exist_ok=True)
        
        # Save config for reproducibility and post-hoc analysis
        with open(self.run_dir / "config.yaml", "w") as f:
            yaml.dump(self.cfg, f, default_flow_style=False)
        
        self.step = 0
        self.metrics_file = self.run_dir / "metrics.jsonl"

        self.checkpoint_mode = str(train_cfg.get("checkpoint_schedule", "interval"))
        self.checkpoint_count = int(float(train_cfg.get("checkpoint_count", 1000)))
        digits_max_steps = len(str(max(1, int(self.max_steps))))
        digits_checkpoint_count = len(str(max(1, self.checkpoint_count)))
        self._checkpoint_name_width = max(4, digits_max_steps, digits_checkpoint_count)
        self.checkpoint_log_linear_alpha = float(train_cfg.get("checkpoint_log_linear_alpha", 0.5))
        self.checkpoint_steps: set[int] = set()
        if self.checkpoint_mode == "log_linear":
            self.checkpoint_steps = set(
                _build_log_linear_checkpoint_steps(
                    max_steps=self.max_steps,
                    checkpoint_count=self.checkpoint_count,
                    alpha=self.checkpoint_log_linear_alpha,
                )
            )
        self._retcon_checkpoint_names()

        self.hf_upload_checkpoints = bool(train_cfg.get("hf_upload_checkpoints", False))
        self.hf_upload_every_n_checkpoints = int(float(train_cfg.get("hf_upload_every_n_checkpoints", 100)))
        self.hf_checkpoint_repo_id = train_cfg.get("hf_checkpoint_repo_id") or os.getenv("HF_CHECKPOINT_REPO_ID")
        self.hf_checkpoint_repo_type = str(train_cfg.get("hf_checkpoint_repo_type", "dataset"))
        self.hf_checkpoint_repo_private = bool(train_cfg.get("hf_checkpoint_repo_private", True))
        self.hf_checkpoint_repo_path_prefix = train_cfg.get("hf_checkpoint_repo_path_prefix", "checkpoints")
        self._hf_uploader: HuggingFaceCheckpointUploader | None = None
        self._hf_uploaded_manifest = self.checkpoints_dir / ".hf_uploaded.txt"
        self._uploaded_checkpoint_names: set[str] = set()
        if self.hf_upload_checkpoints:
            self._load_uploaded_checkpoint_manifest()
        if self.hf_upload_checkpoints:
            if self.hf_upload_every_n_checkpoints <= 0:
                raise ValueError("hf_upload_every_n_checkpoints must be > 0 when hf_upload_checkpoints=True")
            if not self.hf_checkpoint_repo_id:
                raise ValueError("hf_checkpoint_repo_id (or HF_CHECKPOINT_REPO_ID) is required when hf_upload_checkpoints=True")
            cfg = CheckpointUploadConfig(
                repo_id=self.hf_checkpoint_repo_id,
                repo_type=self.hf_checkpoint_repo_type,
                private=self.hf_checkpoint_repo_private,
                path_prefix=self.hf_checkpoint_repo_path_prefix,
            )
            self._hf_uploader = HuggingFaceCheckpointUploader(cfg)
        
        # ---- debug logging ----
        self.debug = train_cfg.get("debug", False)
        self.use_muon = use_muon
        if self.debug:
            # Pre-compute param groups for efficient per-step logging
            self._muon_params = []
            self._adam_params = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if _is_muon_param(name, p):
                    self._muon_params.append(p)
                else:
                    self._adam_params.append(p)
        
        # ---- activation norm hooks ----
        self._activation_stats = {}
        self._hooks = []
        for layer_name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                hook = module.register_forward_hook(
                    self._make_activation_hook(layer_name)
                )
                self._hooks.append(hook)
        
        # Behaviour tracker for analysis metrics
        self.behaviour_tracker = behaviour_tracker
        
        # Weights & Biases run (optional)
        self.wandb_run = wandb_run
    
    def log_metrics(self, metrics: dict):
        """Append metrics to jsonl file."""
        metrics["step"] = self.step
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")
        if self.wandb_run is not None:
            self.wandb_run.log(metrics, step=self.step)
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error to errors directory."""
        error_file = self.run_dir / "errors" / f"error_step{self.step}.txt"
        with open(error_file, "w") as f:
            f.write(f"Step: {self.step}\n")
            f.write(f"Context: {context}\n")
            f.write(f"Error: {type(error).__name__}: {error}\n")
    
    def save_checkpoint(self, name: str = None):
        """Save model checkpoint."""
        if name is None:
            name = self._format_step_checkpoint_name(self.step)
        path = self.checkpoints_dir / f"{name}.pt"
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)
        if self._hf_uploader is not None:
            self._upload_pending_checkpoints()

    def _upload_pending_checkpoints(self, force: bool = False) -> None:
        """Upload buffered checkpoints to Hugging Face when enough new files exist."""
        if self._hf_uploader is None:
            return

        pending_paths = self._pending_checkpoint_paths()
        if not pending_paths:
            return

        batch_size = max(1, self.hf_upload_every_n_checkpoints)
        if not force:
            full_batches = len(pending_paths) // batch_size
            if full_batches == 0:
                return
            pending_paths = pending_paths[: full_batches * batch_size]

        for chunk in self._chunk_paths(pending_paths, batch_size):
            if not chunk:
                continue
            message = self._format_checkpoint_commit_message(chunk)
            self._hf_uploader.upload_batch(chunk, commit_message=message)
            for checkpoint_path in chunk:
                self._record_uploaded_checkpoint(checkpoint_path)

    def _record_uploaded_checkpoint(self, checkpoint_path: Path) -> None:
        name = checkpoint_path.name
        if name in self._uploaded_checkpoint_names:
            return
        self._uploaded_checkpoint_names.add(name)
        try:
            with open(self._hf_uploaded_manifest, "a") as f:
                f.write(f"{name}\n")
        except OSError as exc:
            tqdm.write(f"Warning: failed to log uploaded checkpoint {name}: {exc}")

    def _load_uploaded_checkpoint_manifest(self) -> None:
        if not self._hf_uploaded_manifest.exists():
            return
        try:
            with open(self._hf_uploaded_manifest, "r") as f:
                for line in f:
                    checkpoint_name = line.strip()
                    if checkpoint_name:
                        self._uploaded_checkpoint_names.add(checkpoint_name)
        except OSError as exc:
            print(f"Warning: failed to load uploaded checkpoint manifest: {exc}")

    @staticmethod
    def _infer_step_from_name(checkpoint_path: Path) -> int:
        stem = checkpoint_path.stem
        if stem.startswith("step_"):
            try:
                return int(stem.split("_", 1)[1])
            except ValueError:
                return -1
        if stem == "final":
            return -1
        return -1

    def _pending_checkpoint_paths(self) -> list[Path]:
        pending = [
            p
            for p in self.checkpoints_dir.glob("*.pt")
            if p.is_file() and p.name not in self._uploaded_checkpoint_names
        ]
        pending.sort(key=self._checkpoint_sort_key)
        return pending

    def _checkpoint_sort_key(self, path: Path) -> tuple[float, float]:
        step = self._infer_step_from_name(path)
        sort_step = float(step) if step >= 0 else float("inf")
        return (sort_step, path.stat().st_mtime)

    @staticmethod
    def _chunk_paths(paths: list[Path], chunk_size: int) -> list[list[Path]]:
        if chunk_size <= 0:
            return [paths]
        return [paths[i : i + chunk_size] for i in range(0, len(paths), chunk_size)]

    def _format_checkpoint_commit_message(self, paths: list[Path]) -> str:
        if not paths:
            return "Upload checkpoints"
        steps = [self._infer_step_from_name(p) for p in paths if self._infer_step_from_name(p) >= 0]
        count = len(paths)
        if steps:
            start, end = min(steps), max(steps)
            if start == end:
                return f"Upload checkpoint step {start}"
            return f"Upload checkpoints steps {start}-{end} ({count} files)"
        first, last = paths[0].stem, paths[-1].stem
        if first == last:
            return f"Upload checkpoint {first}"
        return f"Upload checkpoints {first}..{last} ({count} files)"

    def train(self, eval_every: int = 100, save_every: int = 1000):
        """Run training loop."""
        self.model.train()
        data_iter = iter(self.train_dataloader)
        self._ensure_step0_checkpoint()
        
        # Evaluate at step 0 before training starts
        if self.val_dataloader is not None:
            val_loss = evaluate(self.model, self.val_dataloader, self.device)
            self.log_metrics({"val_loss": val_loss})
            self.model.train()
        
        pbar = tqdm(total=self.max_steps, desc="Training")
        
        while self.step < self.max_steps:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_dataloader)
                batch = next(data_iter)
            
            input_ids = batch["input_ids"].to(self.device)
            
            try:
                self.optimizer.zero_grad()
                
                if self.use_amp:
                    with torch.amp.autocast("cuda", dtype=self.pt_dtype):
                        logits = self.model(input_ids)
                        loss = compute_loss(
                            logits, input_ids,
                            loss_type=self.loss_type,
                            label_smoothing=self.label_smoothing,
                        )
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        preclip_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        # bfloat16 path: no scaler needed
                        loss.backward()
                        preclip_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()
                else:
                    logits = self.model(input_ids)
                    loss = compute_loss(
                        logits, input_ids,
                        loss_type=self.loss_type,
                        label_smoothing=self.label_smoothing,
                    )
                    loss.backward()
                    preclip_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.step += 1
                
                if self.step % 10 == 0:
                    metrics = {
                        "train_loss": loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                    if self.debug:
                        metrics.update(self._compute_debug_metrics(preclip_norm.item()))
                        metrics.update(self._compute_weight_norms())
                        metrics.update(self._compute_activation_norms())
                    self.log_metrics(metrics)
                
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)
                
                if self.step % eval_every == 0 and self.val_dataloader is not None:
                    val_loss = evaluate(self.model, self.val_dataloader, self.device)
                    self.log_metrics({"val_loss": val_loss})
                    self.model.train()
                
                if self.checkpoint_mode == "log_linear":
                    if self.step in self.checkpoint_steps:
                        self.save_checkpoint()
                elif self.step % save_every == 0:
                    self.save_checkpoint()
                
                # Log behaviour metrics if tracker is enabled
                if self.behaviour_tracker is not None and self.behaviour_tracker.should_compute(self.step):
                    behaviour_metrics = self.behaviour_tracker.log_metrics(
                        self.step,
                        additional_metrics={"train_loss": loss.item()}
                    )
                    # Forward behaviour metrics to wandb via log_metrics
                    wandb_behaviour = {k: v for k, v in behaviour_metrics.items() if k != "step"}
                    if wandb_behaviour:
                        self.log_metrics(wandb_behaviour)
                    if "bigram_score" in behaviour_metrics:
                        pbar.set_postfix(
                            loss=f"{loss.item():.4f}",
                            bigram=f"{behaviour_metrics['bigram_score']:.2f}"
                        )
                    
            except Exception as e:
                self.log_error(e, "training step")
                raise
        
        pbar.close()
        if self.checkpoint_mode != "log_linear":
            self.save_checkpoint("final")
        self._generate_plots()
        self._upload_pending_checkpoints(force=True)

    def _ensure_step0_checkpoint(self) -> None:
        step0_name = self._format_step_checkpoint_name(0)
        step0_path = self.checkpoints_dir / f"{step0_name}.pt"
        if step0_path.exists():
            return
        original_step = self.step
        self.step = 0
        try:
            self.save_checkpoint(step0_name)
        finally:
            self.step = original_step

    def _format_step_checkpoint_name(self, step: int) -> str:
        """Return zero-padded checkpoint name, e.g., step_0001."""
        step_int = max(0, int(step))
        return f"step_{str(step_int).zfill(self._checkpoint_name_width)}"

    def _retcon_checkpoint_names(self) -> None:
        """Rename any pre-existing checkpoints to match the current padding."""
        for path in sorted(self.checkpoints_dir.glob("*.pt")):
            step = self._infer_step_from_name(path)
            if step < 0:
                continue
            expected = f"{self._format_step_checkpoint_name(step)}{path.suffix}"
            if path.name == expected:
                continue
            target = path.with_name(expected)
            if target.exists():
                continue
            path.rename(target)

    def _compute_debug_metrics(self, preclip_norm: float = 0.0) -> dict:
        """Compute debug metrics: per-group LRs, grad norms, param max-abs."""
        m = {}
        
        # Per-group learning rates
        lrs = self.scheduler.get_last_lr()
        if self.use_muon and len(lrs) >= 3:
            m["lr_muon"] = lrs[0]
            m["lr_adamw"] = lrs[1]  # adam_decay group
        else:
            m["lr_adamw"] = lrs[0]
        
        # Pre-clip grad norm (captured from clip_grad_norm_ return value)
        m["grad_norm_preclip"] = preclip_norm
        
        # Post-clip grad norms (per group and total)
        total_norm = 0.0
        muon_norm = 0.0
        adam_norm = 0.0
        for p in self._muon_params:
            if p.grad is not None:
                pn = p.grad.data.norm(2).item() ** 2
                muon_norm += pn
                total_norm += pn
        for p in self._adam_params:
            if p.grad is not None:
                pn = p.grad.data.norm(2).item() ** 2
                adam_norm += pn
                total_norm += pn
        m["grad_norm_postclip"] = total_norm ** 0.5
        if self._muon_params:
            m["grad_norm_muon"] = muon_norm ** 0.5
        if self._adam_params:
            m["grad_norm_adamw"] = adam_norm ** 0.5
        
        # Param max absolute values
        if self._muon_params:
            m["max_param_abs_muon"] = max(p.data.abs().max().item() for p in self._muon_params)
        if self._adam_params:
            m["max_param_abs_adamw"] = max(p.data.abs().max().item() for p in self._adam_params)
        
        return m
    
    def _make_activation_hook(self, layer_name: str):
        """Create a forward hook that records activation norm mean and variance.
        
        Computes the L2 norm along the last dimension (per token), then
        records the mean and variance of those norms across the batch.
        """
        def hook(module, input, output):
            with torch.no_grad():
                out = output.float()
                # L2 norm per token: shape (B, T)
                norms = out.norm(dim=-1)
                self._activation_stats[layer_name] = {
                    "mean": norms.mean().item(),
                    "var": norms.var().item(),
                }
        return hook
    
    def _compute_weight_norms(self) -> dict:
        """Compute L2 norm of each named parameter."""
        m = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                m[f"weight_norm/{name}"] = p.data.norm(2).item()
        return m
    
    def _compute_activation_norms(self) -> dict:
        """Return activation norm mean/var captured by forward hooks."""
        m = {}
        for layer_name, stats in self._activation_stats.items():
            m[f"act_norm_mean/{layer_name}"] = stats["mean"]
            m[f"act_norm_var/{layer_name}"] = stats["var"]
        return m
    
    def _generate_plots(self) -> None:
        """Generate end-of-training plots."""
        try:
            plot_loss(self.metrics_file, self.run_dir / "loss.png")
            if self.debug:
                plot_debug(self.metrics_file, self.run_dir)
                plot_weight_norms(self.metrics_file, self.run_dir / "weight_norms.png")
                plot_activation_norms(self.metrics_file, self.run_dir / "activation_norms.png")
        except Exception as e:
            print(f"Warning: plot generation failed: {e}")
