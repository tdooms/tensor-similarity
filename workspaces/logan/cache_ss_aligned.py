#!/usr/bin/env python3
"""Cache SimpleStories with each chunk starting at a story boundary.

Each story is tokenized independently. Stories longer than n_ctx are truncated,
shorter stories are padded with the pad token (EOS). Very short stories (<32 tokens)
are skipped.
"""
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset

N_CTX = 512
MIN_TOKENS = 32  # skip very short stories
CACHE_DIR = Path(__file__).parent / "cached_tokens"

tokenizer = AutoTokenizer.from_pretrained("SimpleStories/SimpleStories-V2-1.25M")
pad_id = tokenizer.eos_token_id or 1

CACHE_DIR.mkdir(parents=True, exist_ok=True)

for split in ["train", "test"]:
    print(f"Processing {split}...")
    ds = load_dataset("SimpleStories/SimpleStories", split=split)

    chunks = []
    skipped = 0
    truncated = 0

    # Batch tokenize
    batch_size = 10000
    for start in range(0, len(ds), batch_size):
        end = min(start + batch_size, len(ds))
        batch_stories = ds[start:end]["story"]
        encoded = tokenizer(batch_stories, add_special_tokens=False)

        for tokens in encoded["input_ids"]:
            if len(tokens) < MIN_TOKENS:
                skipped += 1
                continue

            if len(tokens) >= N_CTX:
                # Truncate to exactly N_CTX
                chunk = tokens[:N_CTX]
                truncated += 1
            else:
                # Pad with pad_id to N_CTX
                chunk = tokens + [pad_id] * (N_CTX - len(tokens))

            chunks.append(chunk)

        if start % 100000 == 0 and start > 0:
            print(f"  processed {start}/{len(ds)} stories, {len(chunks)} chunks so far")

    data = torch.tensor(chunks, dtype=torch.int16)
    out_path = CACHE_DIR / f"ss_aligned_{split}_ctx{N_CTX}.pt"
    torch.save(data, out_path)
    mb = os.path.getsize(out_path) / 1e6
    total_tokens = data.numel()

    print(f"  {split}: {data.shape[0]} stories -> {data.shape} chunks")
    print(f"  Tokens: {total_tokens:,} ({total_tokens/1e9:.3f}B)")
    print(f"  Truncated: {truncated}, Skipped (<{MIN_TOKENS} tokens): {skipped}")
    print(f"  Saved: {out_path} ({mb:.0f}MB)")
