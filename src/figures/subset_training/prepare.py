"""Preparation stage for the Laurence-derived subset-training figure family."""

import json

import polars as pl
from loguru import logger
from safetensors.torch import load_file
from tqdm import tqdm

from src.components.similarity import precompile, similarity_parts
from src.figures import CACHE_DIR, EXPERIMENT_DIR, cosine_from_parts
from src.figures.subset_training.train import SEEDS, SUBSET_CONFIGS, build_model

OUT = EXPERIMENT_DIR / "subset_training"
CACHE = CACHE_DIR / "subset_training"
REFERENCE_SEED = 42


def run_dir(seed: int, config_name: str):
    return OUT / f"seed_{seed}" / config_name


def load_run_checkpoints(seed: int, config_name: str):
    manifest = json.loads((run_dir(seed, config_name) / "checkpoints.json").read_text(encoding="utf-8"))
    return [
        {
            "batch": item["batch"],
            "epoch": item["epoch"],
            "state_dict": load_file(run_dir(seed, config_name) / item["file"]),
        }
        for item in manifest
    ]


def main():
    reference_checkpoints = load_run_checkpoints(REFERENCE_SEED, "all")
    reference_model = build_model(REFERENCE_SEED)
    reference_model.load_state_dict(reference_checkpoints[-1]["state_dict"])
    reference_model = reference_model.eval()

    second_model = None
    for seed in SEEDS:
        for name in SUBSET_CONFIGS:
            checkpoints = load_run_checkpoints(seed, name)
            if checkpoints:
                second_model = build_model(seed)
                second_model.load_state_dict(checkpoints[0]["state_dict"])
                second_model = second_model.eval()
                if seed != REFERENCE_SEED or name != "all" or checkpoints[0]["batch"] != reference_checkpoints[-1]["batch"]:
                    break
        if second_model is not None:
            break
    if second_model is not None:
        precompile(reference_model, second_model)

    rows = []
    for seed in SEEDS:
        for config_name in SUBSET_CONFIGS:
            history = pl.read_ipc(run_dir(seed, config_name) / "history.feather")
            history_by_batch = {
                int(row["batch"]): row
                for row in history.iter_rows(named=True)
            }
            checkpoints = load_run_checkpoints(seed, config_name)
            for checkpoint in tqdm(checkpoints, desc=f"{seed}/{config_name}"):
                model = build_model(seed)
                model.load_state_dict(checkpoint["state_dict"])
                model = model.eval()
                metrics = history_by_batch[int(checkpoint["batch"])]
                rows.append(
                    {
                        "seed": seed,
                        "config": config_name,
                        "batch": int(checkpoint["batch"]),
                        "epoch": float(checkpoint["epoch"]),
                        "train_acc": float(metrics["train_acc"]),
                        "test_acc": float(metrics["val_acc"]),
                        "similarity": cosine_from_parts(*similarity_parts(model, reference_model)),
                    }
                )

    CACHE.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_ipc(CACHE / "evolution.feather")
    (CACHE / "metadata.json").write_text(
        json.dumps(
            {
                "seeds": SEEDS,
                "configs": list(SUBSET_CONFIGS),
                "reference_seed": REFERENCE_SEED,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info(f"Saved subset training figure cache to {CACHE}")
