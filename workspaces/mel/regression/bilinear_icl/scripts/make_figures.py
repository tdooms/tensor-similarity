import argparse
from pathlib import Path

import pandas as pd

from bilinear_icl.figures import (
    make_attention,
    make_embedding,
    make_learning_dynamics,
    make_ood_input,
    make_ood_task,
    make_residual,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    df = pd.read_parquet(run_dir / "metrics.parquet")

    make_learning_dynamics(df, run_dir)
    make_ood_input(df, run_dir)
    make_ood_task(df, run_dir)
    make_embedding(df, run_dir)
    make_attention(df, run_dir)
    make_residual(df, run_dir)


if __name__ == "__main__":
    main()
