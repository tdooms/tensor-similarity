"""Full pipeline (train + similarity matrices + plot) on grokking hyperparams,
20 log-linear checkpoints. Output goes to results_grokking_summary/.
"""
import subprocess
import sys
import time
from pathlib import Path

import train as train_mod

OUT_DIR = Path(__file__).parent / "results_grokking_summary"

CFG = train_mod.ExperimentConfig(
    val_fraction=0.7,
    weight_decay=0.3,
    lr=1e-3,
    total_steps=100_000,
    d_hidden=64,
    P=113,
    seed=1337,
    schedule_mode="power",
    schedule_exponent=5.0,
    n_checkpoints=30,
    output_dir=str(OUT_DIR),
)

print("=" * 60)
print(f"Grokking full pipeline: lr={CFG.lr}, wd={CFG.weight_decay}, "
      f"val_fraction={CFG.val_fraction}, n_checkpoints={CFG.n_checkpoints}")
print("=" * 60)
sys.stdout.flush()

t0 = time.time()
checkpoints, history = train_mod.train_with_checkpoints(CFG)
t1 = time.time()
print(f"\n[timing] train: {t1-t0:.1f}s ({len(checkpoints)} ckpts)", flush=True)

tn_sim, act_sim, js_div = train_mod.compute_similarity_matrices(checkpoints, CFG)
t2 = time.time()
print(f"[timing] similarity: {t2-t1:.1f}s", flush=True)

train_mod.save_results(checkpoints, history, tn_sim, act_sim, js_div, CFG)
print(f"[timing] TRAIN+SAVE TOTAL: {time.time()-t0:.1f}s", flush=True)

# Run plot.py against the new results dir
print(f"\nRunning plot.py against {OUT_DIR}...", flush=True)
subprocess.run([sys.executable, "plot.py", str(OUT_DIR)], check=True)
print(f"\nAll outputs in: {OUT_DIR}/")
