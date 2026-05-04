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
def eval_runner(model, bundle, cfg):
    model.eval()
    out = {}
    out.update(behavioral.compute(model, bundle["id"]))
    for g, ep in bundle["ood_x"].items():
        out.update(ood.compute_input(model, ep, g))
    for g, ep in bundle["ood_t"].items():
        out.update(ood.compute_task(model, ep, g))
    out.update(embedding.compute(model))
    out.update(attention_mass.compute(model, bundle["id"]))
    out.update(residual.compute(model, bundle["id"]))
    model.train()
    return out
