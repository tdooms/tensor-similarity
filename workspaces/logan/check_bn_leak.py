#!/usr/bin/env python3
"""Compare train-mode vs eval-mode CE to check if BN leaks future info."""
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, ".")

# We need the model class from the training scripts
from train_bilinear_bn_nobias_ctx512_d512 import AttentionLM as AttentionLM_Pile
from train_bilinear_bn_nobias_ss_ctx512_d512 import AttentionLM as AttentionLM_SS

device = "cuda:0"

def compute_ce(model, input_ids):
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids)
    shift_logits = logits[:, :-1, :].contiguous().float()
    shift_labels = input_ids[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    return F.cross_entropy(shift_logits.view(B * T, V), shift_labels.view(B * T)).item()


def check_run(name, model_cls, vocab_size, ckpt_path, data_path, n_ctx=512):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    # Load data
    data = torch.load(data_path, weights_only=True).to(torch.long)[:, :n_ctx]

    # Load model (uncompiled)
    model = model_cls(
        vocab_size=vocab_size, n_ctx=n_ctx, d_model=512, n_head=16,
        n_layers=2, attn_scale=1.0, rope_base=10000,
        std_embed=0.02, std_qkv=0.02, std_o=0.01, norm_type="layernorm",
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Strip _orig_mod. prefix from compiled checkpoints
    state = {}
    for k, v in ckpt["model_state_dict"].items():
        state[k.replace("_orig_mod.", "")] = v
    model.load_state_dict(state)

    # Use batches of 16 sequences, up to available data
    batch_size = 16
    n_batches = min(30, data.shape[0] // batch_size)

    train_losses, eval_losses = [], []
    for i in range(n_batches):
        batch = data[i * batch_size:(i + 1) * batch_size].to(device)

        model.train()
        train_losses.append(compute_ce(model, batch))

        model.eval()
        eval_losses.append(compute_ce(model, batch))

    avg_train = sum(train_losses) / len(train_losses)
    avg_eval = sum(eval_losses) / len(eval_losses)
    diff = avg_train - avg_eval

    print(f"  model.train() CE:  {avg_train:.4f}")
    print(f"  model.eval()  CE:  {avg_eval:.4f}")
    print(f"  Difference:        {diff:.4f}  ({'train lower' if diff < 0 else 'eval lower'})")
    print(f"  Relative diff:     {abs(diff) / avg_eval * 100:.2f}%")


# Run D: DSIR Pile
check_run(
    "Run D (DSIR Pile) — step 80k",
    AttentionLM_Pile, 5000,
    "runs/2026-02-13_211725_bilinear_bn_nobias_ctx512_d512/checkpoints/step_80000.pt",
    "cached_tokens/dsir_pile_val_ctx512.pt",
)

# Run F: SimpleStories
check_run(
    "Run F (SimpleStories) — step 80k",
    AttentionLM_SS, 4019,
    "runs/2026-02-13_221438_bilinear_bn_nobias_ss_ctx512_d512/checkpoints/step_80000.pt",
    "cached_tokens/ss_test_ctx512.pt",
)
