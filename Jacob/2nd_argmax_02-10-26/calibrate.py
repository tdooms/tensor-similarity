"""Quick calibration: train one seed per distribution, log accuracy at many checkpoints."""

import sys
import importlib
import torch
import torch.nn as nn

# Import the hyphenated module
spec = importlib.util.spec_from_file_location("bilinear", "bilinear-2nd-argmax.py")
bilinear = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bilinear)

BilinearStack = bilinear.BilinearStack
task_2nd_argmax = bilinear.task_2nd_argmax
N = bilinear.N
BATCH_SIZE = bilinear.BATCH_SIZE

DISTRIBUTIONS = {
    'gaussian': bilinear.gaussian,
    'half_gaussian': bilinear.half_gaussian,
    'bimodal': bilinear.bimodal,
    'uniform': bilinear.uniform,
    'rademacher': bilinear.rademacher,
    'sparse_spikes': bilinear.sparse_spikes,
    'permutation': bilinear.permutation,
    'correlated_gaussian': bilinear.correlated_gaussian,
}

EVAL_STEPS = [10, 25, 50, 100, 200, 500, 1000, 2000, 3000, 5000, 7500, 10000]
EVAL_SAMPLES = 50000

torch.manual_seed(0)

for dist_name, dist_fn in DISTRIBUTIONS.items():
    print(f"\n=== {dist_name} ===")
    model = BilinearStack(N, num_layers=3, rank=32, use_linear=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for step in range(1, EVAL_STEPS[-1] + 1):
        x = dist_fn(BATCH_SIZE, N)
        targets = task_2nd_argmax(x)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step in EVAL_STEPS:
            model.eval()
            with torch.no_grad():
                x_eval = dist_fn(EVAL_SAMPLES, N)
                targets_eval = task_2nd_argmax(x_eval)
                preds = model(x_eval).argmax(-1)
                acc = (preds == targets_eval).float().mean().item()
            print(f"  step {step:>5d}: {acc:.1%}")
            model.train()
