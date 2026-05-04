import argparse
from pathlib import Path

import yaml

from bilinear_icl.train.trainer import train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--run-dir", default=None)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train(cfg, run_dir=args.run_dir)


if __name__ == "__main__":
    main()
