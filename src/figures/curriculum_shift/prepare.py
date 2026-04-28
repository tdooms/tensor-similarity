"""Preparation stage for the curriculum-shift figure family."""
import json

import polars as pl
import torch
from safetensors.torch import load_file
from tqdm import tqdm

from src.components.similarity import precompile, similarity_parts
from src.figures import CACHE_DIR, FIGURES, cosine_from_parts, resolve
from src.figures.curriculum_shift.train import OUT, SEED, build_model

CACHE = CACHE_DIR / "curriculum_shift"
N_HEATMAP_SAMPLES = 100


def stage_dir(name):
    return OUT / name


def load_model(state_dict):
    model = build_model(SEED).eval()
    model.load_state_dict(state_dict)
    return model


def load_stage_checkpoints(name):
    manifest = json.loads((stage_dir(name) / "checkpoints.json").read_text(encoding="utf-8"))
    return [{"batch": item["batch"], "state_dict": load_file(stage_dir(name) / item["file"])} for item in manifest]


def main():
    if not (OUT / "curriculum.json").exists():
        if "train" in FIGURES["curriculum-shift"]:
            resolve("curriculum-shift", "train")()

    curriculum = json.loads((OUT / "curriculum.json").read_text(encoding="utf-8"))
    all_checkpoints = []
    cumulative = 0
    for stage in curriculum:
        name = stage["name"]
        checkpoints = load_stage_checkpoints(name)
        for checkpoint in checkpoints:
            all_checkpoints.append({
                "batch": cumulative + checkpoint["batch"],
                "state_dict": checkpoint["state_dict"],
                "stage": name,
            })
        if checkpoints:
            cumulative += checkpoints[-1]["batch"]

    ref_model = load_model(all_checkpoints[-1]["state_dict"])
    if len(all_checkpoints) >= 2:
        precompile(load_model(all_checkpoints[0]["state_dict"]), ref_model)

    trajectory_rows = []
    for checkpoint in tqdm(all_checkpoints, desc="Trajectory"):
        model = load_model(checkpoint["state_dict"])
        trajectory_rows.append({
            "batch": checkpoint["batch"],
            "similarity": cosine_from_parts(*similarity_parts(model, ref_model)),
            "stage": checkpoint["stage"],
        })

    n = min(N_HEATMAP_SAMPLES, len(all_checkpoints))
    indices = sorted({int(i) for i in torch.linspace(0, len(all_checkpoints) - 1, n).round().tolist()})
    sampled = [all_checkpoints[i] for i in indices]
    models = [load_model(checkpoint["state_dict"]) for checkpoint in sampled]
    heatmap_rows = []
    for i in tqdm(range(n), desc="Heatmap"):
        for j in range(n):
            similarity = cosine_from_parts(*similarity_parts(models[i], models[j]))
            heatmap_rows.append({
                "batch_i": sampled[i]["batch"],
                "batch_j": sampled[j]["batch"],
                "stage_i": sampled[i]["stage"],
                "stage_j": sampled[j]["stage"],
                "similarity": similarity,
            })

    accuracy_rows = []
    cumulative = 0
    for stage in curriculum:
        name = stage["name"]
        history = pl.read_ipc(stage_dir(name) / "history.feather")
        history = history.with_columns(
            (pl.col("batch") + cumulative).alias("batch"),
            pl.lit(name).alias("stage"),
        )
        accuracy_rows.append(history.select("stage", "batch", "val_acc"))
        if history.height > 0:
            cumulative = int(history["batch"][-1])

    CACHE.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(trajectory_rows).write_ipc(CACHE / "trajectory.feather")
    pl.concat(accuracy_rows).write_ipc(CACHE / "accuracy.feather")
    pl.DataFrame(heatmap_rows).write_ipc(CACHE / "heatmap.feather")
    (CACHE / "curriculum.json").write_text(json.dumps(curriculum, indent=2), encoding="utf-8")
