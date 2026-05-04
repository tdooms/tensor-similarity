import json
import math
from pathlib import Path

import pandas as pd
import yaml

from bilinear_icl.train.trainer import train


def test_smoke_train_end_to_end(tmp_path):
    with Path("configs/smoke.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg["wandb"]["enabled"] = False
    cfg["hf"]["enabled"] = False

    run_dir = tmp_path / "smoke_run"
    out_dir = train(cfg, run_dir=str(run_dir))

    assert (out_dir / "config.yaml").exists()
    assert (out_dir / "manifest.json").exists()
    assert (out_dir / "metrics.parquet").exists()
    assert (out_dir / "metrics.jsonl").exists()
    assert (out_dir / "eval_episodes.pt").exists()
    assert len(list((out_dir / "checkpoints").glob("step_*.pt"))) >= 1

    df = pd.read_parquet(out_dir / "metrics.parquet")
    assert not df.empty
    required_cols = {
        "step",
        "test_loss",
        "icl_1_4",
        "icl_4_8",
        "pred_sq_magnitude",
        "rav",
        "erank",
    }
    assert required_cols.issubset(set(df.columns))
    assert df["test_loss"].map(math.isfinite).all()
    assert df["rav"].map(math.isfinite).all()
    assert df["erank"].map(math.isfinite).all()
    assert any(c.startswith("ood_x_") and c.endswith("_raw_mse") for c in df.columns)
    assert any(c.startswith("ood_t_") and c.endswith("_raw_mse") for c in df.columns)
    assert any(c.startswith("attn_L") and c.endswith("_entropy_norm") for c in df.columns)
    assert any(c.startswith("attn_L") and c.endswith("_entropy_unnormalized") for c in df.columns)

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            assert df[c].map(lambda x: not (isinstance(x, float) and math.isnan(x))).all()

    train_steps = []
    with (out_dir / "metrics.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "train/step" in row:
                train_steps.append(row["train/step"])
    assert len(train_steps) > 0
    assert max(train_steps) > 0
