import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from bilinear_icl.data import sample_episodes, to_sequence
from bilinear_icl.eval.runner import build_eval_bundle, eval_runner
from bilinear_icl.figures import (
    make_attention,
    make_embedding,
    make_learning_dynamics,
    make_ood_input,
    make_ood_task,
    make_residual,
)
from bilinear_icl.io import init_wandb, maybe_log, push_run_to_hf
from bilinear_icl.models import RegressionTransformer
from bilinear_icl.train.checkpoint import build_schedule, save_checkpoint, write_manifest
from bilinear_icl.train.loss import mean_mse
from bilinear_icl.train.optim import build_optimizer, build_scheduler


def _dtype_from_cfg(name: str):
    lookup = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    return lookup[name]


def _make_run_dir(base_dir: Path, name: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    run_dir = base_dir / f"{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _save_run_readme(cfg: dict, run_dir: Path):
    model = cfg["model"]
    data = cfg["data"]
    train = cfg["train"]
    lines = [
        f"# {cfg['name']}",
        "",
        "## Model Spec",
        "",
        f"- D: {data['D']}",
        f"- K: {data['K']}",
        f"- n_layers: {model['n_layers']}",
        f"- n_head: {model['n_head']}",
        f"- d_model: {model['d_model']}",
        f"- d_mlp: {model['d_mlp']}",
        f"- noise_variance: {data['noise_variance']}",
        f"- max_steps: {train['max_steps']}",
        "",
        "## Load",
        "",
        "```python",
        "from huggingface_hub import hf_hub_download",
        "import torch, yaml",
        "REPO = \"<repo-id>\"",
        "cfg = yaml.safe_load(open(hf_hub_download(REPO, \"config.yaml\"), encoding=\"utf-8\"))",
        "ckpt = torch.load(hf_hub_download(REPO, \"checkpoints/step_500000.pt\"), map_location=\"cpu\")",
        "from bilinear_icl.models import RegressionTransformer",
        "m = RegressionTransformer(D=cfg[\"data\"][\"D\"], K=cfg[\"data\"][\"K\"], **cfg[\"model\"])",
        "m.load_state_dict(ckpt[\"model_state_dict\"])",
        "m.eval()",
        "```",
    ]
    (run_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def train(cfg: dict, run_dir: str | None = None):
    base_runs = Path(run_dir) if run_dir else Path("runs")
    if run_dir:
        run_path = base_runs
        run_path.mkdir(parents=True, exist_ok=True)
    else:
        run_path = _make_run_dir(base_runs, cfg["name"])

    (run_path / "checkpoints").mkdir(parents=True, exist_ok=True)
    with (run_path / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    device = torch.device(cfg["train"]["device"])
    model = RegressionTransformer(
        D=cfg["data"]["D"],
        K=cfg["data"]["K"],
        **cfg["model"],
    ).to(device)
    dtype = _dtype_from_cfg(cfg["train"]["dtype"])

    opt = build_optimizer(
        model,
        muon_lr=cfg["train"]["muon_lr"],
        adamw_lr=cfg["train"]["adamw_lr"],
        weight_decay=cfg["train"]["weight_decay"],
        betas=tuple(cfg["train"]["betas"]),
        allow_adamw_fallback=cfg["train"].get("allow_adamw_fallback", False),
    )
    sch = build_scheduler(
        opt,
        max_steps=cfg["train"]["max_steps"],
        warmup_frac=cfg["train"]["warmup_frac"],
        lr_decay_frac=cfg["train"]["lr_decay_frac"],
    )

    schedule = build_schedule(cfg["train"]["max_steps"], cfg["checkpoint"]["n_log"], cfg["checkpoint"]["n_linear"])
    schedule_set = set(schedule)
    write_manifest(schedule, run_path / "manifest.json")
    bundle = build_eval_bundle(cfg, device, run_path)
    wb_run = init_wandb(cfg, run_path)

    gen = torch.Generator(device=device).manual_seed(cfg["seed"])
    rows = []
    metrics_jsonl = run_path / "metrics.jsonl"

    def save_eval(step: int):
        metrics = eval_runner(model, bundle, cfg)
        payload = {"step": step, **metrics}
        rows.append(payload)
        with metrics_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")

        save_checkpoint(
            {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sch.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            step,
            run_path / "checkpoints",
        )
        maybe_log(wb_run, metrics, step=step)

    if 0 in schedule_set:
        save_eval(0)

    for step in tqdm(range(1, cfg["train"]["max_steps"] + 1), desc="train"):
        xs, ys, _ = sample_episodes(
            cfg["train"]["batch_size"],
            cfg["data"]["K"],
            cfg["data"]["D"],
            cfg["data"]["noise_variance"],
            generator=gen,
            device=device,
        )
        raw = to_sequence(xs, ys)

        use_amp = device.type == "cuda" and dtype in (torch.float16, torch.bfloat16)
        if use_amp:
            ctx = torch.autocast(device_type="cuda", dtype=dtype)
        else:
            ctx = torch.autocast(device_type=device.type, enabled=False)

        with ctx:
            y_hat = model(raw)
            loss = mean_mse(y_hat, ys)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])
        opt.step()
        sch.step()

        if step % 10 == 0:
            lrs = sch.get_last_lr()
            log_row = {
                "train/step": step,
                "train/loss": float(loss.item()),
                "train/lr_muon": float(lrs[0]),
                "train/lr_adamw": float(lrs[1] if len(lrs) > 1 else lrs[0]),
                "train/grad_norm": float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm),
            }
            with metrics_jsonl.open("a", encoding="utf-8") as f:
                f.write(json.dumps(log_row) + "\n")
            maybe_log(wb_run, log_row, step=step)

        if step in schedule_set:
            save_eval(step)

    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    df.to_parquet(run_path / "metrics.parquet", index=False)

    if cfg.get("figures", {}).get("run_at_end", False):
        make_learning_dynamics(df, run_path)
        make_ood_input(df, run_path)
        make_ood_task(df, run_path)
        make_embedding(df, run_path)
        make_attention(df, run_path)
        make_residual(df, run_path)

    _save_run_readme(cfg, run_path)

    if cfg.get("hf", {}).get("enabled", False):
        push_run_to_hf(run_path, repo_id=cfg["hf"].get("repo_id"), private=cfg["hf"].get("private", True))

    if wb_run is not None:
        wb_run.finish()

    return run_path
