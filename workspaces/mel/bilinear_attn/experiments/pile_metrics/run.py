"""Evaluate behaviour-analysis metrics over every checkpoint of a HuggingFace
dataset-style checkpoint repo.

Usage
-----
From ``workspaces/mel/bilinear_attn``::

    python -m experiments.pile_metrics.run \
        --hf-repo melephant/2l-bilinear-attn \
        --stride 1000

The script is resumable: already-evaluated steps are skipped by reading
``analysis_metrics.jsonl``. Each checkpoint is downloaded to a temp
location, evaluated, appended to the jsonl, then deleted.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import EntryNotFoundError

# Allow running either as ``python -m experiments.pile_metrics.run`` or
# directly from this directory.
_WORKSPACE = Path(__file__).resolve().parents[2]
if str(_WORKSPACE) not in sys.path:
    sys.path.insert(0, str(_WORKSPACE))

from analysis.behaviour import load_metrics  # noqa: E402
from analysis.behaviour.tracker import BehaviourTracker, TrackerConfig  # noqa: E402
from data.pile import create_pile_dataloaders, _load_tokenizer  # noqa: E402
from models import AttentionLM  # noqa: E402


_STEP_RE = re.compile(r"step_(\d+)\.pt$")
_DEFAULT_OUT_DIR = object()


def _parse_step(name: str) -> int | None:
    m = _STEP_RE.search(name)
    return int(m.group(1)) if m else None


def _sanitize_component(value: str) -> str:
    """Return a filesystem-safe component (used for run directories)."""
    value = value.strip().replace(os.sep, "_").replace("/", "__")
    value = re.sub(r"[^0-9A-Za-z._-]+", "_", value)
    return value or "default"


def _default_run_dir(hf_repo: str, path_prefix: str) -> Path:
    runs_root = Path(__file__).resolve().parent / "runs"
    repo_component = _sanitize_component(hf_repo)
    prefix_component = _sanitize_component(path_prefix)
    return runs_root / repo_component / prefix_component


def _extract_series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    steps: list[int] = []
    values: list[float] = []
    for row in rows:
        if key in row and "step" in row:
            steps.append(int(row["step"]))
            values.append(row[key])
    # Sort by step to ensure correct plotting order
    sorted_pairs = sorted(zip(steps, values), key=lambda x: x[0])
    if sorted_pairs:
        steps, values = zip(*sorted_pairs)
        return list(steps), list(values)
    return [], []


def _first_key_with_prefix(rows: list[dict], prefix: str) -> str | None:
    for row in rows:
        for key in row:
            if key.startswith(prefix):
                return key
    return None


def _plot_series_or_note(ax, steps: list[int], values: list[float], label: str, **plot_kwargs) -> bool:
    if steps:
        ax.plot(steps, values, label=label, **plot_kwargs)
        return True
    return False


def _annotate_empty(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, fontsize=10, color="0.4")
    ax.set_axis_off()


def _generate_summary_plots(metrics_path: Path, plots_dir: Path) -> None:
    if not metrics_path.exists():
        print(f"Skipping plot generation (missing {metrics_path})")
        return

    metrics = load_metrics(metrics_path)
    if not metrics:
        print(f"Skipping plot generation (no entries in {metrics_path})")
        return

    try:
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib optional in CI
        print(f"Skipping plot generation (matplotlib unavailable: {exc})")
        return

    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(12, 16), sharex=False)
    fig.suptitle("Behaviour metrics overview", fontsize=16)

    # 1) N-gram scores
    ax = axes[0]
    ngram_keys: list[tuple[int, str]] = []
    for row in metrics:
        for key in row:
            match = re.match(r"^(\d+)gram_score$", key)
            if match:
                order = int(match.group(1))
                ngram_keys.append((order, key))
        if ngram_keys:
            break
    ngram_keys = sorted(set(ngram_keys))
    if ngram_keys:
        for order, key in ngram_keys:
            steps, values = _extract_series(metrics, key)
            if steps:
                ax.plot(steps, values, marker="o", markersize=3, label=f"{order}-gram")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_ylabel("Score (ratio)")
        ax.set_title("N-gram scores")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        _annotate_empty(ax, "No n-gram metrics yet")

    # 2) Position-ablated loss
    ax = axes[1]
    plotted = False
    val_steps, val_loss = _extract_series(metrics, "val_loss")
    plotted |= _plot_series_or_note(ax, val_steps, val_loss, "Val loss", linewidth=1.2)
    abl_steps, abl_loss = _extract_series(metrics, "ablated_loss")
    plotted |= _plot_series_or_note(ax, abl_steps, abl_loss, "Ablated loss", linestyle="--", linewidth=1.2)
    if plotted:
        ax.set_ylabel("Loss")
        ax.set_title("Validation vs position-ablated loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        _annotate_empty(ax, "No ablation metrics yet")

    # 3) ICL score
    ax = axes[2]
    icl_key = _first_key_with_prefix(metrics, "icl_")
    if icl_key:
        icl_steps, icl_values = _extract_series(metrics, icl_key)
        if icl_steps:
            ax.plot(icl_steps, icl_values, color="tab:purple", linewidth=1.2, label=icl_key.replace("_", " "))
            ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_ylabel("Δ loss")
            ax.set_title("ICL score")
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            _annotate_empty(ax, "No ICL data yet")
    else:
        _annotate_empty(ax, "No ICL data yet")

    # 4) Bigram score vs entropy baseline
    ax = axes[3]
    score_steps, score_values = _extract_series(metrics, "bigram_score")
    entropy_steps, entropy_values = _extract_series(metrics, "bigram_entropy")
    plotted = False
    plotted |= _plot_series_or_note(ax, score_steps, score_values, "Bigram score", linewidth=1.2)
    plotted |= _plot_series_or_note(ax, entropy_steps, entropy_values, "Bigram entropy", linestyle="--", linewidth=1.2)
    if plotted:
        ax.set_xlabel("Step")
        ax.set_ylabel("nats")
        ax.set_title("Bigram score vs entropy baseline")
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        _annotate_empty(ax, "No bigram metrics yet")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path = plots_dir / "behaviour_metrics.png"
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved behaviour plots to {output_path}")


def load_repo_config(
    repo_id: str,
    repo_type: str = "dataset",
    cache_dir: Path | None = None,
) -> dict:
    """Download the training config for a checkpoint repo.

    Looks for (in order):
      1. ``config.json``  — produced by ``scripts/upload_metadata.py``.
      2. ``config.yaml``  — produced by ``publish_run_to_hf``.

    Raises FileNotFoundError if neither is present. The returned dict has
    the same shape as the training-config YAML (``data``, ``model``,
    ``train``, ``init``, ``name``, ``seed``).
    """
    download_kwargs = dict(
        repo_id=repo_id,
        repo_type=repo_type,
    )
    if cache_dir is not None:
        download_kwargs["local_dir"] = str(cache_dir)

    for filename, parser in (("config.json", json.load), ("config.yaml", yaml.safe_load)):
        try:
            path = hf_hub_download(filename=filename, **download_kwargs)
        except EntryNotFoundError:
            continue
        except Exception as exc:  # network, auth, etc.
            print(f"  config fetch of {filename} failed ({type(exc).__name__}: {exc})")
            continue
        with open(path, "r") as f:
            return parser(f)

    raise FileNotFoundError(
        f"Neither config.json nor config.yaml found in {repo_type} {repo_id}. "
        f"Pass --config to provide one locally."
    )


def list_remote_checkpoints(api: HfApi, repo_id: str, path_prefix: str = "checkpoints") -> List[str]:
    """Return ``checkpoints/step_XXXXX.pt`` paths sorted by step."""
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    ckpt_files = [f for f in files if f.startswith(path_prefix + "/") and f.endswith(".pt")]
    ckpt_files = [f for f in ckpt_files if _parse_step(f) is not None]
    ckpt_files.sort(key=lambda f: _parse_step(f))
    return ckpt_files


def load_done_steps(jsonl_path: Path) -> set[int]:
    """Return the set of steps already present in ``jsonl_path``."""
    if not jsonl_path.exists():
        return set()
    done: set[int] = set()
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "step" in entry:
                done.add(int(entry["step"]))
    return done


def select_checkpoints(
    all_ckpts: List[str],
    stride: int,
    min_step: int | None,
    max_step: int | None,
    limit: int | None,
) -> List[str]:
    """Subset the checkpoint list by stride / min / max / limit.
    
    Always includes the last checkpoint regardless of stride.
    """
    selected: List[str] = []
    for i, path in enumerate(all_ckpts):
        step = _parse_step(path)
        if min_step is not None and step < min_step:
            continue
        if max_step is not None and step > max_step:
            continue
        if i % max(1, stride) != 0:
            continue
        selected.append(path)
    # Always include the last checkpoint if it passed min/max filters
    if all_ckpts:
        last_path = all_ckpts[-1]
        last_step = _parse_step(last_path)
        if (min_step is None or last_step >= min_step) and \
           (max_step is None or last_step <= max_step) and \
           last_path not in selected:
            selected.append(last_path)
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_tracker(
    cfg: dict,
    device: str,
    cache_dir: Path,
    eval_batch_size: int | None,
    tracker_config: TrackerConfig,
) -> BehaviourTracker:
    """Build model + Pile dataloaders + a fitted BehaviourTracker.

    ``fit()`` is called here so that the bigram / n-gram distributions are
    cached once at the start and reused for every checkpoint.
    """
    model_cfg = cfg["model"]
    train_cfg = cfg.get("train", {})
    data_cfg = cfg.get("data", {})

    model = AttentionLM.from_config(cfg).to(device)

    batch_size = eval_batch_size or train_cfg.get("batch_size", 64)
    train_dl, val_dl = create_pile_dataloaders(
        n_ctx=model_cfg["n_ctx"],
        batch_size=batch_size,
        vocab_size=model_cfg["vocab_size"],
        val_cache_size=data_cfg.get("val_cache_size", 500),
        max_val_samples=data_cfg.get("val_cache_size", 500),
        tokenizer_name=data_cfg["tokenizer"],
        dataset_repo=data_cfg["repo"],
        text_field=data_cfg.get("text_field", "contents"),
        train_split=data_cfg.get("train_split", "train"),
        val_split=data_cfg.get("val_split", "heldout"),
        dataset_revision=data_cfg.get("revision"),
        train_shuffle=data_cfg.get("train_shuffle", True),
        train_shuffle_seed=data_cfg.get("train_shuffle_seed", 0),
        train_shuffle_buffer_size=data_cfg.get("train_shuffle_buffer_size", 10_000),
        insert_eos_between_documents=data_cfg.get("insert_eos_between_documents", True),
        holdout_fraction=data_cfg.get("holdout_fraction", 0.01),
        holdout_seed=data_cfg.get("holdout_seed", 0),
    )

    tokenizer = _load_tokenizer(data_cfg["tokenizer"])

    tracker = BehaviourTracker(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        vocab_size=model_cfg["vocab_size"],
        device=device,
        config=tracker_config,
        run_dir=None,
        cache_dir=str(cache_dir),
        tokenizer=tokenizer,
    )
    tracker.fit(max_fit_samples=10_000)
    return tracker


def _pile_tracker_config() -> TrackerConfig:
    """Defaults tuned for a Pile-DSIR run (contiguous training, no BOS)."""
    return TrackerConfig(
        bigram_enabled=True,
        ngram_enabled=True,
        ablation_enabled=True,
        icl_enabled=True,
        ngram_prepend_bos=False,
        icl_prepend_bos=False,
        icl_k1=50,
        icl_k2=500,
        ngram_max_val_batches=None,  # evaluate on the full val cache
        ablation_max_val_batches=None,
        icl_max_val_batches=None,
    )


def evaluate_one_remote_checkpoint(
    tracker: BehaviourTracker,
    api: HfApi,
    repo_id: str,
    remote_path: str,
    download_dir: Path,
) -> dict:
    """Download one checkpoint from HF, evaluate it, delete the local file.

    Returns the metrics dict (augmented with ``remote_path``).
    """
    local_path: Path | None = None
    try:
        local = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=remote_path,
            local_dir=str(download_dir),
        )
        local_path = Path(local)
        metrics = tracker.evaluate_checkpoint(local_path)
        metrics["remote_path"] = remote_path
        metrics.pop("checkpoint", None)  # path is meaningless once deleted
        return metrics
    finally:
        if local_path is not None and local_path.exists():
            try:
                local_path.unlink()
            except OSError:
                pass


def append_jsonl(jsonl_path: Path, entry: dict) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate behaviour metrics across HF-hosted checkpoints."
    )
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Optional training config YAML/JSON override. If omitted, the "
             "config is downloaded from the HF repo (config.json, then "
             "config.yaml). This is the correct default: the repo-side "
             "config matches the checkpoint weights exactly.",
    )
    parser.add_argument(
        "--hf-repo", type=str, default="melephant/2l-bilinear-attn-v2",
        help="HuggingFace dataset repo holding the checkpoints.",
    )
    parser.add_argument(
        "--hf-path-prefix", type=str, default="checkpoints",
        help="Path prefix inside the repo (matches Trainer upload prefix).",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=_DEFAULT_OUT_DIR,
        help=(
            "Root directory to hold run artifacts. Defaults to "
            "<script>/runs/<repo>/<path_prefix> with repo/prefix sanitized."
        ),
    )
    parser.add_argument(
        "--device", type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--stride", type=int, default=1,
                        help="Evaluate every Nth checkpoint (by list order).")
    parser.add_argument("--min-step", type=int, default=None)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of checkpoints to evaluate.")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Override the batch size used for eval "
                             "(defaults to train batch_size).")
    parser.add_argument("--dry-run", action="store_true",
                        help="List the checkpoints that would be evaluated "
                             "and exit.")
    args = parser.parse_args()

    if args.out_dir is _DEFAULT_OUT_DIR:
        out_dir = _default_run_dir(args.hf_repo, args.hf_path_prefix)
    else:
        out_dir = args.out_dir

    metadata_dir = out_dir / "metadata"
    metrics_dir = out_dir / "metrics"
    cache_dir = out_dir / "cache"
    download_dir = out_dir / "downloads"
    plots_dir = out_dir / "plots"

    for path in (metadata_dir, metrics_dir, cache_dir, download_dir, plots_dir):
        path.mkdir(parents=True, exist_ok=True)

    print(f"Writing artifacts to {out_dir}")

    # Load config: prefer the one stored alongside the checkpoints in the HF
    # repo (the training-time config) unless the user explicitly overrides.
    if args.config is not None:
        print(f"Loading config from local file: {args.config}")
        with open(args.config) as f:
            cfg = (
                json.load(f) if args.config.suffix == ".json"
                else yaml.safe_load(f)
            )
        cfg_source = str(args.config)
    else:
        print(f"Fetching config from HF repo: {args.hf_repo}")
        cfg = load_repo_config(args.hf_repo, repo_type="dataset")
        cfg_source = f"hf://{args.hf_repo}"
        # Persist a copy alongside the jsonl for provenance.
        with open(metadata_dir / "resolved_config.yaml", "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    jsonl_path = metrics_dir / "analysis_metrics.jsonl"

    api = HfApi()
    all_ckpts = list_remote_checkpoints(api, args.hf_repo, args.hf_path_prefix)
    if not all_ckpts:
        print(f"No checkpoints found in {args.hf_repo}")
        return

    done_steps = load_done_steps(jsonl_path)
    selected = select_checkpoints(
        all_ckpts, args.stride, args.min_step, args.max_step, args.limit
    )
    pending = [p for p in selected if _parse_step(p) not in done_steps]

    print(f"Repo {args.hf_repo}: {len(all_ckpts)} total checkpoints")
    print(f"Selected {len(selected)} (stride={args.stride}), {len(pending)} pending "
          f"(already done: {len(selected) - len(pending)})")
    if args.dry_run:
        for p in pending[:20]:
            print(f"  {p}")
        if len(pending) > 20:
            print(f"  ... ({len(pending) - 20} more)")
        return

    tracker_config = _pile_tracker_config()
    tracker = build_tracker(
        cfg=cfg,
        device=args.device,
        cache_dir=cache_dir,
        eval_batch_size=args.eval_batch_size,
        tracker_config=tracker_config,
    )

    # Write a sidecar describing the run settings (overwritten each invocation)
    with open(metadata_dir / "run_settings.yaml", "w") as f:
        yaml.safe_dump(
            {
                "config_source": cfg_source,
                "hf_repo": args.hf_repo,
                "device": args.device,
                "tracker_config": asdict(tracker_config),
            },
            f,
            sort_keys=False,
        )

    for i, remote_path in enumerate(pending):
        step = _parse_step(remote_path)
        print(f"[{i+1}/{len(pending)}] step={step}  ({remote_path})")
        try:
            metrics = evaluate_one_remote_checkpoint(
                tracker, api, args.hf_repo, remote_path, download_dir,
            )
        except Exception as exc:
            print(f"  !! failed: {type(exc).__name__}: {exc}")
            continue
        append_jsonl(jsonl_path, metrics)
        summary = {
            k: (f"{v:.4f}" if isinstance(v, float) else v)
            for k, v in metrics.items()
            if k not in ("remote_path",)
        }
        print(f"    {summary}")

    _generate_summary_plots(jsonl_path, plots_dir)


if __name__ == "__main__":
    main()
