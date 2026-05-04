import os


def init_wandb(cfg, run_dir):
    if not cfg["wandb"]["enabled"]:
        return None
    if not os.getenv("WANDB_API_KEY"):
        return None
    try:
        import wandb
    except Exception:
        return None
    return wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"].get("entity"),
        name=cfg["name"],
        config=cfg,
        dir=str(run_dir),
    )


def maybe_log(run, payload, step=None):
    if run is None:
        return
    run.log(payload, step=step)
