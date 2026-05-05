import argparse
import copy
from pathlib import Path
import traceback

import yaml

from bilinear_icl.train.trainer import train


VARIANTS = [
    {"name": "mup", "init_type": "mup", "attn_type": "bilinear", "norm_places": ["pre_unembed"]},
    {"name": "softmax", "init_type": "normal", "attn_type": "softmax", "norm_places": ["pre_unembed"]},
    {"name": "norm_mlp_unembed_mup", "init_type": "mup", "attn_type": "bilinear", "norm_places": ["pre_mlp", "pre_unembed"]},
    {"name": "norm_attn_unembed_mup", "init_type": "mup", "attn_type": "bilinear", "norm_places": ["pre_attn", "pre_unembed"]},
    {
        "name": "norm_all_mup",
        "init_type": "mup",
        "attn_type": "bilinear",
        "norm_places": ["pre_attn", "pre_mlp", "pre_unembed"],
    },
]


def run_variant(base_cfg: dict, variant: dict, group: str):
    cfg = copy.deepcopy(base_cfg)
    cfg["model"]["init_type"] = variant["init_type"]
    cfg["model"]["attn_type"] = variant["attn_type"]
    cfg["model"]["norm_places"] = variant["norm_places"]
    cfg["model"]["norm_type"] = "tok0"
    cfg["name"] = f"{base_cfg['name']}_{variant['name']}"
    cfg.setdefault("wandb", {})["group"] = group
    try:
        train(cfg)
    except Exception as e:
        err_dir = Path("runs") / "sweep_errors"
        err_dir.mkdir(parents=True, exist_ok=True)
        (err_dir / f"{cfg['name']}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[FAILED] {cfg['name']}: {type(e).__name__}: {e}")
    finally:
        try:
            import wandb
            if wandb.run is not None:
                wandb.finish()
        except Exception:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--group", default="sweep_arch_smoke")
    args = ap.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    for variant in VARIANTS:
        run_variant(base_cfg, variant, args.group)


if __name__ == "__main__":
    main()
