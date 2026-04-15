#!/usr/bin/env python3
"""Compute and cache n-gram + repeat-ICL scores for induction checkpoints."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from analysis.behaviour.ngram import NgramAnalyzer
from experiments.induction_heads.data import create_repeated_token_dataloaders
from experiments.tn_sim_induction.heatmap import find_config, load_checkpoint


_DEFAULT_N_TRAIN = 50_000
_DEFAULT_N_VAL = 2_000
_DEFAULT_MAX_FIT_SAMPLES = 10_000
_DEFAULT_MAX_VAL_BATCHES = 50


def resolve_checkpoint_dir(
    npz_path: Path,
    checkpoint_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Infer checkpoint directory from a similarity npz path."""
    if checkpoint_dir is not None:
        return Path(checkpoint_dir)

    if npz_path.parent.name == "heatmap":
        candidate = npz_path.parent.parent / "checkpoints"
        if candidate.exists():
            return candidate

    if npz_path.parent.name == "checkpoints":
        return npz_path.parent

    candidate = npz_path.parent / "checkpoints"
    if candidate.exists():
        return candidate

    return None


def load_score_records(metrics_path: Path) -> list[Dict[str, float]]:
    """Load score records from a JSONL file."""
    if not metrics_path.exists():
        return []

    records = []
    with open(metrics_path, "r") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def append_score_records(metrics_path: Path, records: Iterable[Dict[str, float]]) -> None:
    """Append score records to a JSONL file."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "a") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")


def align_score_records(
    records: Sequence[Dict[str, float]],
    steps: np.ndarray,
) -> Optional[Dict[str, np.ndarray]]:
    """Align score records to the provided steps array."""
    if not records:
        return None

    step_to_record = {int(r["step"]): r for r in records if "step" in r}
    metric_keys = sorted({
        key for record in records for key in record.keys() if key != "step"
    })
    if not metric_keys:
        return None

    series = {}
    for key in metric_keys:
        series[key] = np.array([
            step_to_record.get(int(step), {}).get(key, np.nan) for step in steps
        ], dtype=float)

    return {
        "steps": steps,
        "series": series,
    }


def prepare_ngram_analyzer(
    train_dataloader,
    vocab_size: int,
    cache_path: Path,
    device: str,
    max_n: int = 3,
    max_fit_samples: int = _DEFAULT_MAX_FIT_SAMPLES,
    max_common_ngrams: int = 1000,
) -> NgramAnalyzer:
    """Load or fit a common n-gram cache."""
    if cache_path.exists():
        analyzer = NgramAnalyzer.load(str(cache_path), device=device)
        return analyzer

    analyzer = NgramAnalyzer(
        vocab_size=vocab_size,
        device=device,
        max_common_ngrams=max_common_ngrams,
    )
    analyzer.extract_common_ngrams_from_data(
        train_dataloader,
        max_n=max_n,
        max_samples=max_fit_samples,
    )
    analyzer.save(str(cache_path))
    return analyzer


def compute_ngram_loss_no_bos(
    model: torch.nn.Module,
    analyzer: NgramAnalyzer,
    n: int,
    device: str,
    batch_size: int = 128,
) -> float:
    """Compute n-gram loss without prepending a BOS token."""
    if n not in analyzer.common_ngrams:
        return float("nan")

    ngrams = analyzer.common_ngrams[n]
    if not ngrams:
        return float("nan")

    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i in range(0, len(ngrams), batch_size):
            batch = ngrams[i:i + batch_size]
            contexts = [context for context, _ in batch]
            targets = [final for _, final in batch]

            input_ids = torch.tensor(contexts, device=device, dtype=torch.long)
            target_ids = torch.tensor(targets, device=device, dtype=torch.long)

            logits = model(input_ids)
            final_logits = logits[:, -1, :]
            loss = F.cross_entropy(final_logits, target_ids, reduction="sum")
            total_loss += loss.item()
            total_samples += len(batch)

    return total_loss / total_samples if total_samples > 0 else float("nan")


def compute_test_loss_no_bos(
    model: torch.nn.Module,
    dataloader,
    n: int,
    device: str,
    max_batches: Optional[int] = None,
) -> float:
    """Compute validation loss at position n without BOS."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            if input_ids.shape[1] < n:
                continue

            first_n = input_ids[:, :n]
            inputs = first_n[:, :-1]
            targets = first_n[:, -1]

            logits = model(inputs)
            loss = F.cross_entropy(logits[:, -1, :], targets, reduction="sum")
            total_loss += loss.item()
            total_samples += input_ids.shape[0]

    return total_loss / total_samples if total_samples > 0 else float("nan")


def compute_ngram_score_no_bos(
    model: torch.nn.Module,
    analyzer: NgramAnalyzer,
    val_dataloader,
    n: int,
    device: str,
    max_val_batches: Optional[int] = _DEFAULT_MAX_VAL_BATCHES,
) -> Dict[str, float]:
    """Compute n-gram score without BOS prefixing."""
    ngram_loss = compute_ngram_loss_no_bos(model, analyzer, n, device=device)
    test_loss = compute_test_loss_no_bos(
        model,
        val_dataloader,
        n,
        device=device,
        max_batches=max_val_batches,
    )

    score = test_loss / ngram_loss if ngram_loss > 0 else float("nan")
    return {
        f"{n}gram_loss": ngram_loss,
        f"{n}gram_test_loss": test_loss,
        f"{n}gram_score": score,
    }


def _first_occurrence_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Mask first occurrence positions for tokens that repeat."""
    input_cpu = input_ids.cpu().tolist()
    mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for row_idx, seq in enumerate(input_cpu):
        counts = Counter(seq)
        seen = set()
        for pos, token in enumerate(seq):
            if token in seen:
                continue
            seen.add(token)
            if counts[token] > 1:
                mask[row_idx, pos] = True

    return mask.to(input_ids.device)


def compute_repeat_icl_score(
    model: torch.nn.Module,
    dataloader,
    device: str,
    max_batches: Optional[int] = _DEFAULT_MAX_VAL_BATCHES,
) -> Dict[str, float]:
    """Compare loss at first vs repeated occurrences of repeated tokens."""
    model.eval()
    total_first_loss = 0.0
    total_repeat_loss = 0.0
    total_first_tokens = 0
    total_repeat_tokens = 0

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if max_batches is not None and i >= max_batches:
                break

            input_ids = batch["input_ids"].to(device)
            repeat_mask = batch["repeat_mask"].to(device)

            logits = model(input_ids)
            shift_logits = logits[:, :-1, :]
            shift_targets = input_ids[:, 1:]
            bsz, seq_len, vocab = shift_logits.shape

            losses = F.cross_entropy(
                shift_logits.reshape(bsz * seq_len, vocab),
                shift_targets.reshape(bsz * seq_len),
                reduction="none",
            ).reshape(bsz, seq_len)

            first_mask = _first_occurrence_mask(input_ids)[:, 1:]
            repeat_mask = repeat_mask[:, 1:]

            if first_mask.any():
                total_first_loss += losses[first_mask].sum().item()
                total_first_tokens += first_mask.sum().item()
            if repeat_mask.any():
                total_repeat_loss += losses[repeat_mask].sum().item()
                total_repeat_tokens += repeat_mask.sum().item()

    first_loss = total_first_loss / total_first_tokens if total_first_tokens > 0 else float("nan")
    repeat_loss = total_repeat_loss / total_repeat_tokens if total_repeat_tokens > 0 else float("nan")

    return {
        "repeat_first_loss": first_loss,
        "repeat_loss": repeat_loss,
        "repeat_icl": repeat_loss - first_loss,
    }


def load_or_compute_scores(
    steps: np.ndarray,
    npz_path: Path,
    checkpoint_dir: Optional[Path] = None,
    device: Optional[str] = None,
    n_train: int = _DEFAULT_N_TRAIN,
    n_val: int = _DEFAULT_N_VAL,
    ngram_ns: Sequence[int] = (2, 3),
) -> Optional[Dict[str, np.ndarray]]:
    """Load cached scores, computing any missing checkpoint metrics."""
    checkpoint_dir = resolve_checkpoint_dir(npz_path, checkpoint_dir)
    if checkpoint_dir is None:
        print("Warning: could not resolve checkpoint dir for n-gram/ICL scores")
        return None

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    score_dir = checkpoint_dir.parent / "score_metrics"
    scores_path = score_dir / "ngram_icl_scores.jsonl"
    records = load_score_records(scores_path)
    existing_steps = {int(r["step"]) for r in records if "step" in r}

    missing_steps = [int(step) for step in steps if int(step) not in existing_steps]
    if not missing_steps:
        return align_score_records(records, steps)

    config_path = find_config(checkpoint_dir)
    with open(config_path, "r") as handle:
        cfg = yaml.safe_load(handle)

    torch.manual_seed(cfg.get("seed", 42))

    vocab_size = cfg["model"]["vocab_size"]
    n_ctx = cfg["model"]["n_ctx"]
    batch_size = cfg.get("train", {}).get("batch_size", 64)

    train_dl, val_dl = create_repeated_token_dataloaders(
        vocab_size=vocab_size,
        n_ctx=n_ctx,
        batch_size=batch_size,
        n_train=n_train,
        n_val=n_val,
        seed=cfg.get("seed", 42),
    )

    ngram_cache_path = score_dir / "ngram_cache.pt"
    analyzer = prepare_ngram_analyzer(
        train_dl,
        vocab_size=vocab_size,
        cache_path=ngram_cache_path,
        device=device,
        max_n=max(ngram_ns),
    )

    new_records = []
    for step in sorted(missing_steps):
        ckpt_path = checkpoint_dir / f"step_{step}.pt"
        if not ckpt_path.exists():
            print(f"Warning: checkpoint not found: {ckpt_path}")
            continue

        model, _ = load_checkpoint(cfg, ckpt_path)
        model = model.to(device)

        row = {"step": step}
        for n in ngram_ns:
            row.update(compute_ngram_score_no_bos(
                model,
                analyzer,
                val_dl,
                n,
                device=device,
            ))

        row.update(compute_repeat_icl_score(model, val_dl, device=device))
        new_records.append(row)

        model = model.cpu()
        del model
        if device != "cpu":
            torch.cuda.empty_cache()

    if new_records:
        append_score_records(scores_path, new_records)
        records.extend(new_records)

    return align_score_records(records, steps)
