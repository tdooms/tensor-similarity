#!/usr/bin/env python3
"""Train a DSIR-Pile tokenizer with AutoTokenizer.train_new_from_iterator()."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoTokenizer
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_eos_pad_only(tokenizer) -> None:
    if tokenizer.eos_token is None:
        raise ValueError("Tokenizer has no eos_token; cannot enforce eos/pad special-token policy")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.additional_special_tokens = []

    # Remove other special-token slots; byte-level BPE doesn't need unk for coverage.
    for attr in ("bos_token", "unk_token", "cls_token", "sep_token", "mask_token"):
        try:
            setattr(tokenizer, attr, None)
        except Exception:
            pass


def _special_tokens_summary(tokenizer) -> dict[str, Any]:
    return {
        "eos_token": tokenizer.eos_token,
        "pad_token": tokenizer.pad_token,
        "bos_token": tokenizer.bos_token,
        "unk_token": tokenizer.unk_token,
        "cls_token": tokenizer.cls_token,
        "sep_token": tokenizer.sep_token,
        "mask_token": tokenizer.mask_token,
        "additional_special_tokens": list(tokenizer.additional_special_tokens),
        "all_special_tokens": list(tokenizer.all_special_tokens),
    }


def _validate_special_tokens_policy(tokenizer) -> None:
    s = _special_tokens_summary(tokenizer)
    eos = s["eos_token"]
    pad = s["pad_token"]
    if eos is None:
        raise ValueError("Expected eos_token to be set")
    if pad != eos:
        raise ValueError(f"Expected pad_token == eos_token, got pad={pad!r}, eos={eos!r}")

    disallowed = [
        ("bos_token", s["bos_token"]),
        ("unk_token", s["unk_token"]),
        ("cls_token", s["cls_token"]),
        ("sep_token", s["sep_token"]),
        ("mask_token", s["mask_token"]),
    ]
    bad = [(k, v) for k, v in disallowed if v is not None]
    if bad:
        raise ValueError(f"Unexpected non-null special tokens: {bad}")

    extras = set(s["all_special_tokens"])
    allowed = {eos, pad}
    if not extras.issubset(allowed):
        raise ValueError(
            f"Unexpected special-token content: {sorted(extras)} (allowed subset: {sorted(allowed)})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a tokenizer on DSIR-filtered Pile")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pile_dsir.yaml",
        help="YAML config path (data/model/seed). Used for defaults, including revision.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tokenizers/pile_dsir_4096",
        help="Directory to save the tokenizer",
    )
    parser.add_argument(
        "--base-tokenizer",
        type=str,
        default="gpt2",
        help="Base fast tokenizer to clone tokenization pipeline from",
    )
    parser.add_argument(
        "--dataset-repo",
        type=str,
        default=None,
        help="HuggingFace dataset repo",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split",
    )
    parser.add_argument(
        "--text-field",
        type=str,
        default=None,
        help="Text field in dataset examples",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="Target tokenizer vocab size",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=1_000_000,
        help="Number of documents to use for training",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of documents per iterator batch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed",
    )
    parser.add_argument(
        "--shuffle-buffer-size",
        type=int,
        default=100_000,
        help="Streaming shuffle buffer size (set 0 to disable shuffling)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=50_000,
        help="Progress logging interval in documents",
    )
    parser.add_argument(
        "--test-run",
        action="store_true",
        help="Use a small corpus for smoke testing",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output directory if it exists",
    )
    parser.add_argument(
        "--smoke-texts",
        type=int,
        default=8,
        help="How many real texts to use in encode/decode smoke checks",
    )
    parser.add_argument(
        "--hf-tokenizer-repo-id",
        type=str,
        default=None,
        help="Hugging Face model repo id for tokenizer push (e.g., user/pile-dsir-4096-tokenizer)",
    )
    return parser.parse_args()


def _read_cfg(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        return {}
    with open(cfg_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def _resolve_dataset_revision(dataset_repo: str, cfg_revision: str | None) -> tuple[str, str]:
    if cfg_revision:
        return cfg_revision, "config"
    sha = HfApi().dataset_info(dataset_repo).sha
    if not sha:
        raise ValueError(f"Could not resolve dataset SHA for repo={dataset_repo!r}")
    return str(sha), "huggingface_api"


def main() -> None:
    args = parse_args()
    cfg = _read_cfg(args.config)
    data_cfg = cfg.get("data", {}) if isinstance(cfg.get("data", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}

    if args.test_run:
        args.max_docs = min(args.max_docs, 10_000)
        args.output_dir = f"{args.output_dir}_test"

    dataset_repo = args.dataset_repo or data_cfg.get("repo", "stanford-crfm/DSIR-filtered-pile-50M")
    split = args.split or data_cfg.get("train_split") or data_cfg.get("split", "train")
    text_field = args.text_field or data_cfg.get("text_field", "contents")

    raw_vocab = args.vocab_size if args.vocab_size is not None else model_cfg.get("vocab_size", 4096)
    vocab_size = int(raw_vocab)

    raw_seed = args.seed if args.seed is not None else cfg.get("seed", 42)
    seed = int(raw_seed)

    revision, revision_source = _resolve_dataset_revision(
        dataset_repo=dataset_repo,
        cfg_revision=data_cfg.get("revision"),
    )

    output_dir = Path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output dir exists: {output_dir}. Use --overwrite or choose another --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)

    print(f"Loading base tokenizer: {args.base_tokenizer}")
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_tokenizer, use_fast=True)
    if not getattr(base_tokenizer, "is_fast", False):
        raise ValueError("train_new_from_iterator requires a fast tokenizer")

    print(
        f"Loading dataset stream: repo={dataset_repo} split={split} revision={revision} ({revision_source})"
    )
    ds = load_dataset(
        dataset_repo,
        split=split,
        streaming=True,
        revision=revision,
    )
    if args.shuffle_buffer_size > 0:
        ds = ds.shuffle(seed=seed, buffer_size=args.shuffle_buffer_size)

    stats = {"docs": 0}
    smoke_texts: list[str] = []

    def training_corpus():
        batch: list[str] = []
        for ex in ds:
            if stats["docs"] >= args.max_docs:
                break

            text = ex.get(text_field)
            if not isinstance(text, str) or not text:
                continue

            if len(smoke_texts) < args.smoke_texts:
                smoke_texts.append(text)

            batch.append(text)
            stats["docs"] += 1

            if args.log_every > 0 and stats["docs"] % args.log_every == 0:
                print(f"  prepared {stats['docs']:,} docs")

            if len(batch) >= args.batch_size:
                yield batch
                batch = []

        if batch:
            yield batch

    print(
        f"Training tokenizer with train_new_from_iterator on up to {args.max_docs:,} docs "
        f"(vocab_size={vocab_size})"
    )
    tokenizer = base_tokenizer.train_new_from_iterator(
        training_corpus(),
        vocab_size=vocab_size,
    )

    _set_eos_pad_only(tokenizer)
    _validate_special_tokens_policy(tokenizer)

    if not smoke_texts:
        raise RuntimeError("No texts collected for smoke testing")

    # Encode/decode smoke checks on real examples + compare length against GPT-2.
    smoke_results = []
    id_violation_count = 0
    for i, text in enumerate(smoke_texts):
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids_gpt2 = base_tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        invalid_ids = [tid for tid in ids if tid < 0 or tid >= vocab_size]
        if invalid_ids:
            id_violation_count += len(invalid_ids)

        smoke_results.append(
            {
                "index": i,
                "text_chars": len(text),
                "new_len": len(ids),
                "gpt2_len": len(ids_gpt2),
                "len_ratio_new_over_gpt2": (len(ids) / max(1, len(ids_gpt2))),
                "max_id": max(ids) if ids else None,
                "has_invalid_ids": bool(invalid_ids),
                "decoded_preview": decoded[:240],
                "source_preview": text[:240],
            }
        )

    if id_violation_count > 0:
        raise ValueError(f"Found {id_violation_count} token IDs outside [0, {vocab_size})")

    tokenizer.save_pretrained(output_dir)

    hf_tokenizer_repo_id = args.hf_tokenizer_repo_id or data_cfg.get("tokenizer_hf_repo_id")
    hf_tokenizer_private = bool(data_cfg.get("tokenizer_hf_private", True))
    should_push_to_hf = bool(hf_tokenizer_repo_id) and not args.test_run
    if should_push_to_hf:
        api = HfApi()
        api.create_repo(repo_id=hf_tokenizer_repo_id, repo_type="model", private=hf_tokenizer_private, exist_ok=True)
        tokenizer.push_to_hub(hf_tokenizer_repo_id)

    smoke_report = {
        "special_tokens": _special_tokens_summary(tokenizer),
        "id_range_check": {
            "target_vocab_size": vocab_size,
            "violations": id_violation_count,
        },
        "examples": smoke_results,
    }
    with open(output_dir / "smoke_test_report.json", "w", encoding="utf-8") as f:
        json.dump(smoke_report, f, indent=2, ensure_ascii=False)

    metadata = {
        "config_path": args.config,
        "dataset_repo": dataset_repo,
        "split": split,
        "revision": revision,
        "revision_source": revision_source,
        "text_field": text_field,
        "base_tokenizer": args.base_tokenizer,
        "target_vocab_size": vocab_size,
        "trained_docs": stats["docs"],
        "batch_size": args.batch_size,
        "seed": seed,
        "shuffle_buffer_size": args.shuffle_buffer_size,
        "test_run": args.test_run,
        "smoke_texts": args.smoke_texts,
        "hf_tokenizer_repo_id": hf_tokenizer_repo_id,
        "pushed_to_hf": should_push_to_hf,
        "result_vocab_size": int(tokenizer.vocab_size),
        "result_len": int(len(tokenizer)),
        "special_tokens": _special_tokens_summary(tokenizer),
    }
    with open(output_dir / "training_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Done.")
    print(f"  tokenizer saved to: {output_dir}")
    print(f"  trained docs: {stats['docs']:,}")
    print(f"  tokenizer.vocab_size: {tokenizer.vocab_size}")
    print(f"  len(tokenizer): {len(tokenizer)}")
    if should_push_to_hf:
        print(f"  pushed tokenizer to hf://model/{hf_tokenizer_repo_id}")
    print(f"  smoke report: {output_dir / 'smoke_test_report.json'}")


if __name__ == "__main__":
    main()
