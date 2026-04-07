#!/usr/bin/env python3
"""Verify that saved checkpoints load correctly and produce expected induction results."""
import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
sys.path.insert(0, ".")
from train_run_D_1k_ckpts import AttentionLM, eval_toy_repeated, eval_indist_bigrams, _find_repeated_bigrams

DEVICE = "cuda:0"
N_CTX = 512
VOCAB_SIZE = 5000
CKPT_DIR = "runs/run_D_1k_ckpts/checkpoints"
METRICS_FILE = "runs/run_D_1k_ckpts/metrics.jsonl"

# Load val data for bigram eval
val_data = torch.load("cached_tokens/dsir_pile_val_ctx512.pt", weights_only=True).to(torch.long)[:, :N_CTX]
bigram_positions = _find_repeated_bigrams(val_data[:100], N_CTX)

# Load logged metrics for comparison
logged = {}
with open(METRICS_FILE) as f:
    for line in f:
        d = json.loads(line)
        if "toy_loss_diff" in d:
            logged[d["step"]] = d

# Test a few checkpoints across the range
test_steps = [0, 1, 10, 100, 1000, 10000, 50000, 101725]

def load_model(ckpt_path):
    model = AttentionLM(
        vocab_size=VOCAB_SIZE, n_ctx=N_CTX, d_model=512, n_head=16,
        n_layers=2, attn_scale=1.0, rope_base=10000,
        std_embed=0.02, std_qkv=0.02, std_o=0.01, norm_type="layernorm",
    ).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    return model

print("Verifying checkpoints can be loaded and produce correct induction results...\n")
all_ok = True

for step in test_steps:
    ckpt_path = f"{CKPT_DIR}/step_{step:06d}.pt"
    try:
        model = load_model(ckpt_path)
        toy = eval_toy_repeated(model, VOCAB_SIZE, DEVICE, n_samples=100, half_len=16)
        indist = eval_indist_bigrams(model, val_data[:100], N_CTX, bigram_positions, DEVICE)

        # Compare to logged metrics if available
        status = "OK"
        if step in logged:
            log_toy = logged[step]["toy_loss_diff"]
            log_indist = logged[step]["indist_loss_diff"]
            toy_err = abs(toy["toy_loss_diff"] - log_toy)
            indist_err = abs(indist["indist_loss_diff"] - log_indist)
            if toy_err > 0.01 or indist_err > 0.01:
                status = f"MISMATCH (toy_err={toy_err:.4f}, indist_err={indist_err:.4f})"
                all_ok = False
            else:
                status = f"OK (toy_err={toy_err:.6f}, indist_err={indist_err:.6f})"

        print(f"  step {step:>6d}: toy_diff={toy['toy_loss_diff']:+.4f}  "
              f"indist_diff={indist['indist_loss_diff']:+.4f}  "
              f"toy_acc={toy['toy_accuracy']:.4f}  [{status}]")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"  step {step:>6d}: FAILED to load — {e}")
        all_ok = False

print(f"\n{'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
print(f"Total checkpoints on disk: {len(list(__import__('pathlib').Path(CKPT_DIR).glob('*.pt')))}")
