#!/usr/bin/env python3
"""Measure induction opportunity in SimpleStories vs FineWeb.

For each context window (512 tokens), we measure how much a model could
benefit from induction heads — i.e., copying bigram patterns that appeared
earlier in the same context.

Metrics per dataset:
  1. raw_repeat_rate: fraction of positions where bigram (t_{i-1}, t_i)
     appeared earlier in the same context window
  2. unique_repeated_bigrams: avg number of distinct bigrams that repeat
     per context window
  3. rare_repeat_rate: same as (1) but only counting bigrams with below-median
     global frequency (these are the ones induction actually helps with —
     common bigrams can be learned by MLPs/embeddings)
  4. induction_value: for each repeated bigram, weight by log(1/global_freq)
     so rare repeats count more. This approximates the "bits saved by induction".

Usage:
    python measure_induction_opportunity.py
"""

import torch
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path


CACHE_DIR = Path(__file__).parent / "cached_tokens"
N_CTX = 512


def load_simplestories(split="train", max_docs=10000):
    """Load cached SimpleStories tokens."""
    path = CACHE_DIR / f"{split}_perstory.pt"
    data = torch.load(path, weights_only=True).to(torch.long)[:, :N_CTX]
    if max_docs and data.shape[0] > max_docs:
        data = data[:max_docs]
    return data.numpy()


def load_fineweb(max_docs=10000, vocab_size=4096):
    """Load FineWeb-Edu sample, tokenize with same tokenizer, chunk to N_CTX."""
    from datasets import load_dataset
    from transformers import AutoTokenizer

    print("Loading FineWeb-Edu sample...")
    tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-V2-1.25M")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                       split="train", streaming=True)

    all_tokens = []
    n_docs = 0
    for example in ds:
        tokens = tokenizer.encode(example["text"])
        # Chunk into N_CTX windows
        for i in range(0, len(tokens) - N_CTX + 1, N_CTX):
            all_tokens.append(tokens[i:i + N_CTX])
            n_docs += 1
            if n_docs >= max_docs:
                break
        if n_docs >= max_docs:
            break
        if n_docs % 1000 == 0 and n_docs > 0:
            print(f"  {n_docs}/{max_docs} windows...")

    print(f"  Got {n_docs} windows from FineWeb")
    return np.array(all_tokens, dtype=np.int64)


def compute_global_bigram_freqs(data):
    """Compute global bigram frequencies across all context windows."""
    counts = Counter()
    total = 0
    for row in data:
        for i in range(1, len(row)):
            if row[i] == 0:  # padding
                break
            bg = (int(row[i-1]), int(row[i]))
            counts[bg] += 1
            total += 1
    # Convert to frequency
    freqs = {bg: c / total for bg, c in counts.items()}
    return freqs, counts


def measure_induction_opportunity(data, global_freqs, label=""):
    """Measure induction metrics for a dataset.

    For each context window, scan left-to-right tracking bigrams seen so far.
    When a bigram repeats, that's an "induction opportunity".
    """
    n_docs = data.shape[0]

    # Per-document metrics
    raw_repeat_rates = []
    unique_repeated_counts = []
    rare_repeat_rates = []
    induction_values = []
    doc_lengths = []

    # Compute median global frequency for rare/common split
    freq_vals = np.array(list(global_freqs.values()))
    median_freq = np.median(freq_vals)

    for doc_idx in range(n_docs):
        row = data[doc_idx]

        # Find actual length (before padding)
        length = len(row)
        for i in range(len(row)):
            if row[i] == 0:
                length = i
                break

        if length < 3:
            continue

        doc_lengths.append(length)
        seen_bigrams = set()
        n_positions = 0      # total bigram positions
        n_repeats = 0        # positions where bigram was seen before
        n_rare_repeats = 0   # same, but only for globally-rare bigrams
        repeated_bigrams = set()
        total_induction_value = 0.0

        for i in range(1, length):
            bg = (int(row[i-1]), int(row[i]))
            n_positions += 1

            if bg in seen_bigrams:
                n_repeats += 1
                repeated_bigrams.add(bg)

                gf = global_freqs.get(bg, 1e-8)
                if gf < median_freq:
                    n_rare_repeats += 1
                # Induction value: bits saved ≈ log2(1/global_freq)
                total_induction_value += np.log2(1.0 / max(gf, 1e-10))

            seen_bigrams.add(bg)

        if n_positions > 0:
            raw_repeat_rates.append(n_repeats / n_positions)
            unique_repeated_counts.append(len(repeated_bigrams))
            rare_repeat_rates.append(n_rare_repeats / n_positions)
            induction_values.append(total_induction_value / n_positions)

    raw_repeat_rates = np.array(raw_repeat_rates)
    unique_repeated_counts = np.array(unique_repeated_counts)
    rare_repeat_rates = np.array(rare_repeat_rates)
    induction_values = np.array(induction_values)
    doc_lengths = np.array(doc_lengths)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Documents analyzed:        {len(raw_repeat_rates):,}")
    print(f"  Avg document length:       {doc_lengths.mean():.0f} tokens")
    print(f"  Median global bigram freq: {median_freq:.6f}")
    print(f"")
    print(f"  --- Bigram Repetition ---")
    print(f"  Raw repeat rate:           {raw_repeat_rates.mean():.4f} "
          f"(±{raw_repeat_rates.std():.4f})")
    print(f"  Unique repeated bigrams:   {unique_repeated_counts.mean():.1f} "
          f"(±{unique_repeated_counts.std():.1f}) per window")
    print(f"  Rare repeat rate:          {rare_repeat_rates.mean():.4f} "
          f"(±{rare_repeat_rates.std():.4f})")
    print(f"")
    print(f"  --- Induction Value ---")
    print(f"  Avg bits/position from induction: {induction_values.mean():.4f} "
          f"(±{induction_values.std():.4f})")
    print(f"")
    print(f"  Percentiles (raw repeat rate):")
    for p in [10, 25, 50, 75, 90]:
        print(f"    p{p}: {np.percentile(raw_repeat_rates, p):.4f}")

    return {
        "raw_repeat_rate": raw_repeat_rates,
        "unique_repeated_bigrams": unique_repeated_counts,
        "rare_repeat_rate": rare_repeat_rates,
        "induction_value": induction_values,
        "doc_lengths": doc_lengths,
    }


def main():
    max_docs = 10000

    # --- SimpleStories ---
    print("Loading SimpleStories...")
    ss_data = load_simplestories("train", max_docs=max_docs)
    print(f"  Shape: {ss_data.shape}")

    print("Computing global bigram frequencies (SimpleStories)...")
    ss_freqs, ss_counts = compute_global_bigram_freqs(ss_data)
    print(f"  Unique bigrams: {len(ss_freqs):,}")

    ss_results = measure_induction_opportunity(ss_data, ss_freqs, "SimpleStories")

    # --- FineWeb ---
    fw_data = load_fineweb(max_docs=max_docs)
    print(f"  Shape: {fw_data.shape}")

    print("Computing global bigram frequencies (FineWeb)...")
    fw_freqs, fw_counts = compute_global_bigram_freqs(fw_data)
    print(f"  Unique bigrams: {len(fw_freqs):,}")

    fw_results = measure_induction_opportunity(fw_data, fw_freqs, "FineWeb-Edu")

    # --- Comparison ---
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    ss_rr = ss_results["raw_repeat_rate"].mean()
    fw_rr = fw_results["raw_repeat_rate"].mean()
    ss_rare = ss_results["rare_repeat_rate"].mean()
    fw_rare = fw_results["rare_repeat_rate"].mean()
    ss_iv = ss_results["induction_value"].mean()
    fw_iv = fw_results["induction_value"].mean()

    print(f"  {'Metric':<35} {'SimpleStories':>14} {'FineWeb':>14} {'Ratio':>8}")
    print(f"  {'-'*71}")
    print(f"  {'Raw repeat rate':<35} {ss_rr:>14.4f} {fw_rr:>14.4f} {ss_rr/fw_rr:>8.2f}x")
    print(f"  {'Rare repeat rate':<35} {ss_rare:>14.4f} {fw_rare:>14.4f} {ss_rare/max(fw_rare,1e-8):>8.2f}x")
    print(f"  {'Induction value (bits/pos)':<35} {ss_iv:>14.4f} {fw_iv:>14.4f} {ss_iv/max(fw_iv,1e-8):>8.2f}x")
    print(f"  {'Unique repeated bigrams/window':<35} {ss_results['unique_repeated_bigrams'].mean():>14.1f} {fw_results['unique_repeated_bigrams'].mean():>14.1f}")


if __name__ == "__main__":
    main()
