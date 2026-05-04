from pathlib import Path

import yaml

from bilinear_icl.train.trainer import train


def main():
    cfg_path = Path("configs/smoke.yaml")
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    train(cfg)


if __name__ == "__main__":
    main()
