import argparse
from pathlib import Path

import pandas as pd
import torch
import yaml

from bilinear_icl.eval.runner import eval_runner
from bilinear_icl.models import RegressionTransformer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with (run_dir / "config.yaml").open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg["train"]["device"])
    model = RegressionTransformer(D=cfg["data"]["D"], K=cfg["data"]["K"], **cfg["model"]).to(device)
    bundle = torch.load(run_dir / "eval_episodes.pt", map_location=device)

    rows = []
    for ckpt in sorted((run_dir / "checkpoints").glob("step_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        step = state["step"]
        metrics = eval_runner(model, bundle, cfg)
        rows.append({"step": step, **metrics})

    pd.DataFrame(rows).sort_values("step").to_parquet(run_dir / "metrics.parquet", index=False)


if __name__ == "__main__":
    main()
