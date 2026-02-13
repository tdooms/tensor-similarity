import json
import torch
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from .losses import compute_loss
from .optim import create_optimizer, create_scheduler, Optimizers, _is_muon_param
from .eval import evaluate
from .plotting import plot_loss, plot_debug

_DTYPE_MAP = {
    "float32": (torch.float32, False),
    "float16": (torch.float16, True),
    "bfloat16": (torch.bfloat16, False),
}


class Trainer:
    """Training loop for AttentionLM."""

    def __init__(self, model, train_dataloader, val_dataloader=None,
                 cfg=None, run_dir=None, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.cfg = cfg or {}
        self.device = device

        train_cfg = self.cfg.get("train", {})
        loss_cfg = self.cfg.get("loss", {})

        self.max_steps = train_cfg.get("max_steps", 1000)
        self.warmup_steps = train_cfg.get("warmup_steps", 100)
        self.grad_clip = train_cfg.get("grad_clip", 1.0)
        self.loss_type = loss_cfg.get("type", "next_token_ce")
        self.label_smoothing = loss_cfg.get("label_smoothing", 0.0)
        lr_decay_frac = train_cfg.get("lr_decay_frac", 0.1)

        dtype_str = train_cfg.get("dtype", "float32")
        if "amp" in train_cfg and "dtype" not in train_cfg:
            dtype_str = "float16" if train_cfg["amp"] else "float32"
        self.pt_dtype, use_scaler = _DTYPE_MAP[dtype_str]
        self.use_amp = dtype_str != "float32" and device == "cuda"

        use_muon = train_cfg.get("use_muon", True)
        opt_result = create_optimizer(
            model, lr=train_cfg.get("lr", 3e-4), muon_lr=train_cfg.get("muon_lr", 0.02),
            weight_decay=train_cfg.get("weight_decay", 0.1),
            betas=tuple(train_cfg.get("betas", (0.9, 0.95))), use_muon=use_muon,
        )
        if isinstance(opt_result, Optimizers):
            self.optimizer = opt_result.muon
        else:
            self.optimizer = opt_result

        self.scheduler = create_scheduler(
            self.optimizer, warmup_steps=self.warmup_steps,
            max_steps=self.max_steps, lr_decay_frac=lr_decay_frac,
        )

        self.scaler = torch.amp.GradScaler("cuda") if (use_scaler and device == "cuda") else None

        if run_dir is None:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            name = self.cfg.get("name", "run")
            run_dir = f"runs/{timestamp}_{name}"
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "checkpoints").mkdir(exist_ok=True)
        (self.run_dir / "errors").mkdir(exist_ok=True)

        self.step = 0
        self.metrics_file = self.run_dir / "metrics.jsonl"
        self.debug = train_cfg.get("debug", False)
        self.use_muon = use_muon
        if self.debug:
            self._muon_params = []
            self._adam_params = []
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if _is_muon_param(name, p):
                    self._muon_params.append(p)
                else:
                    self._adam_params.append(p)

        self.eval_callbacks = []

    def log_metrics(self, metrics):
        metrics["step"] = self.step
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(metrics) + "\n")

    def log_error(self, error, context=""):
        error_file = self.run_dir / "errors" / f"error_step{self.step}.txt"
        with open(error_file, "w") as f:
            f.write(f"Step: {self.step}\nContext: {context}\nError: {type(error).__name__}: {error}\n")

    def save_checkpoint(self, name=None):
        if name is None:
            name = f"step_{self.step}"
        path = self.run_dir / "checkpoints" / f"{name}.pt"
        torch.save({
            "step": self.step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }, path)

    def train(self, eval_every=100, save_every=1000):
        self.model.train()
        data_iter = iter(self.train_dataloader)
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
                        loss = compute_loss(logits, input_ids, loss_type=self.loss_type,
                                            label_smoothing=self.label_smoothing)
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                        self.scaler.unscale_(self.optimizer)
                        preclip_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        loss.backward()
                        preclip_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                        self.optimizer.step()
                else:
                    logits = self.model(input_ids)
                    loss = compute_loss(logits, input_ids, loss_type=self.loss_type,
                                        label_smoothing=self.label_smoothing)
                    loss.backward()
                    preclip_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()

                self.scheduler.step()
                self.step += 1

                if self.step % 10 == 0:
                    metrics = {"train_loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]}
                    if self.debug:
                        metrics.update(self._compute_debug_metrics(preclip_norm.item()))
                    self.log_metrics(metrics)

                pbar.set_postfix(loss=f"{loss.item():.4f}")
                pbar.update(1)

                if self.step % eval_every == 0 and self.val_dataloader is not None:
                    val_loss = evaluate(self.model, self.val_dataloader, self.device)
                    self.log_metrics({"val_loss": val_loss})
                    for cb in self.eval_callbacks:
                        cb_metrics = cb(self.model, self.step)
                        if cb_metrics:
                            self.log_metrics(cb_metrics)
                    self.model.train()

                if self.step % save_every == 0:
                    self.save_checkpoint()

            except Exception as e:
                self.log_error(e, "training step")
                raise

        pbar.close()
        self.save_checkpoint("final")
        self._generate_plots()

    def _compute_debug_metrics(self, preclip_norm=0.0):
        m = {"grad_norm_preclip": preclip_norm}
        lrs = self.scheduler.get_last_lr()
        if self.use_muon and len(lrs) >= 3:
            m["lr_muon"] = lrs[0]
            m["lr_adamw"] = lrs[1]
        else:
            m["lr_adamw"] = lrs[0]
        total_norm = muon_norm = adam_norm = 0.0
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
            m["max_param_abs_muon"] = max(p.data.abs().max().item() for p in self._muon_params)
        if self._adam_params:
            m["grad_norm_adamw"] = adam_norm ** 0.5
            m["max_param_abs_adamw"] = max(p.data.abs().max().item() for p in self._adam_params)
        return m

    def _generate_plots(self):
        try:
            plot_loss(self.metrics_file, self.run_dir / "loss.png")
            if self.debug:
                plot_debug(self.metrics_file, self.run_dir)
        except Exception as e:
            print(f"Warning: plot generation failed: {e}")
