import traceback
from pathlib import Path

import torch

from bilinear_icl.data import sample_episodes
from . import attention_mass, behavioral, embedding, ood, residual


def build_eval_bundle(cfg, device, run_dir):
    g = torch.Generator(device=device).manual_seed(cfg["eval"]["fixed_seed"])
    N = cfg["eval"]["episodes"]
    K = cfg["data"]["K"]
    D = cfg["data"]["D"]
    s2 = cfg["data"]["noise_variance"]
    grid = [10 ** lg for lg in cfg["data"]["ood_log10_grid"]]

    bundle = {
        "id": sample_episodes(N, K, D, s2, generator=g, device=device),
        "ood_x": {gx: sample_episodes(N, K, D, s2, x_scale=gx, generator=g, device=device) for gx in grid},
        "ood_t": {gt: sample_episodes(N, K, D, s2, t_scale=gt, generator=g, device=device) for gt in grid},
        "grid": grid,
    }
    torch.save(bundle, run_dir / "eval_episodes.pt")
    return bundle


@torch.no_grad()
def eval_runner(model, bundle, cfg, step=None, run_dir=None):
    model.eval()
    out = {}
    ood_failures = 0

    def _record_ood_error(kind: str, g, exc: Exception):
        nonlocal ood_failures
        ood_failures += 1
        if run_dir is not None:
            errs = Path(run_dir) / "errors"
            errs.mkdir(parents=True, exist_ok=True)
            step_tag = -1 if step is None else step
            g_tag = str(g).replace(".", "p").replace("-", "m")
            (errs / f"ood_{kind}_g{g_tag}_step{step_tag}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[OOD FAILED] {kind} g={g}: {type(exc).__name__}: {exc}")

    def _update(tag: str, payload: dict):
        from bilinear_icl.train.sanity import check_finite_metrics

        out.update(payload)
        check_finite_metrics(
            tag,
            out,
            step=-1 if step is None else step,
            run_dir=run_dir,
            enabled=cfg.get("train", {}).get("nan_check", True),
        )

    _update("eval_metrics", behavioral.compute(model, bundle["id"]))
    for g, ep in bundle["ood_x"].items():
        try:
            _update("eval_metrics", ood.compute_input(model, ep, g))
        except Exception as exc:
            _record_ood_error("x", g, exc)
            continue
    for g, ep in bundle["ood_t"].items():
        try:
            _update("eval_metrics", ood.compute_task(model, ep, g))
        except Exception as exc:
            _record_ood_error("t", g, exc)
            continue

    out["eval/ood_failures"] = float(ood_failures)
    _update("eval_metrics", embedding.compute(model))
    _update("eval_metrics", attention_mass.compute(model, bundle["id"]))
    _update("eval_metrics", residual.compute(model, bundle["id"]))
    model.train()
    return out
