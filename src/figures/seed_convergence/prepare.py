"""Preparation stage for the seed-convergence figure family."""
import json

import polars as pl
from safetensors.torch import load_file
from tqdm import tqdm

from src.components.similarity import precompile, similarity_parts
from src.figures import CACHE_DIR, FIGURES, cosine_from_parts, resolve
from src.figures.seed_convergence.train import OUT, REFERENCE_SEED, SEEDS, build_model, seed_dir

CACHE = CACHE_DIR / "seed_convergence"


def load_model(seed, state_dict):
    model = build_model(seed).eval()
    model.load_state_dict(state_dict)
    return model


def load_seed_checkpoints(seed):
    manifest = json.loads((seed_dir(seed) / "checkpoints.json").read_text(encoding="utf-8"))
    return [{"batch": item["batch"], "state_dict": load_file(seed_dir(seed) / item["file"])} for item in manifest]


def main():
    if not all((seed_dir(seed) / "checkpoints.json").exists() for seed in SEEDS):
        if "train" in FIGURES["seed-convergence"]:
            resolve("seed-convergence", "train")()

    ref_checkpoints = load_seed_checkpoints(REFERENCE_SEED)
    ref_model = load_model(REFERENCE_SEED, ref_checkpoints[-1]["state_dict"])
    second_model = None
    for seed in SEEDS:
        checkpoints = load_seed_checkpoints(seed)
        if checkpoints:
            second_model = load_model(seed, checkpoints[0]["state_dict"])
            if seed != REFERENCE_SEED or checkpoints[0]["batch"] != ref_checkpoints[-1]["batch"]:
                break
    if second_model is not None:
        precompile(ref_model, second_model)

    similarity_rows = []
    for seed in SEEDS:
        checkpoints = load_seed_checkpoints(seed)
        for checkpoint in tqdm(checkpoints, desc=f"Seed {seed}"):
            model = load_model(seed, checkpoint["state_dict"])
            similarity_rows.append({
                "seed": seed,
                "batch": checkpoint["batch"],
                "similarity": cosine_from_parts(*similarity_parts(model, ref_model)),
            })

    history_rows = [pl.read_ipc(seed_dir(seed) / "history.feather").with_columns(pl.lit(seed).alias("seed")) for seed in SEEDS]

    CACHE.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(similarity_rows).write_ipc(CACHE / "similarity.feather")
    pl.concat(history_rows).write_ipc(CACHE / "history.feather")
    (CACHE / "metadata.json").write_text(
        json.dumps({"seeds": SEEDS, "reference_seed": REFERENCE_SEED}, indent=2),
        encoding="utf-8",
    )
