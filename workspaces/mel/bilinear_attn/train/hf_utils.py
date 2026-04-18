from __future__ import annotations

import json
import shutil
import tempfile
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import yaml
from huggingface_hub import HfApi


_DEFAULT_CARD_TAGS = ("language-modeling", "causal-lm", "pytorch", "attention")


@dataclass(slots=True)
class CheckpointUploadConfig:
    """Configuration for streaming checkpoint uploads."""

    repo_id: str
    repo_type: str = "dataset"
    private: bool = True
    path_prefix: str = "checkpoints"


class HuggingFaceCheckpointUploader:
    """Small wrapper around HfApi to upload checkpoints safely."""

    def __init__(self, config: CheckpointUploadConfig) -> None:
        self.config = config
        self.api = HfApi()
        self.api.create_repo(
            repo_id=config.repo_id,
            repo_type=config.repo_type,
            private=config.private,
            exist_ok=True,
        )

    def upload_batch(self, checkpoint_paths: Sequence[Path], commit_message: str | None = None) -> None:
        """Upload multiple checkpoints in a single commit by staging them into a folder."""

        if not checkpoint_paths:
            return

        path_prefix = (self.config.path_prefix or ".").strip("/") or "."
        message = commit_message or self._default_commit_message(checkpoint_paths)

        try:
            with tempfile.TemporaryDirectory(prefix="hf_ckpts_") as tmpdir:
                staging_dir = Path(tmpdir)
                for checkpoint_path in checkpoint_paths:
                    shutil.copy2(checkpoint_path, staging_dir / checkpoint_path.name)

                self.api.upload_folder(
                    folder_path=str(staging_dir),
                    path_in_repo=path_prefix,
                    repo_id=self.config.repo_id,
                    repo_type=self.config.repo_type,
                    commit_message=message,
                )

            print(
                f"Uploaded {len(checkpoint_paths)} checkpoints to hf://{self.config.repo_type}/"
                f"{self.config.repo_id}/{path_prefix}"
            )
        except Exception as exc:  # pragma: no cover - network failures best-effort
            print(f"Warning: failed to upload checkpoints to Hugging Face: {exc}")

    @staticmethod
    def _default_commit_message(checkpoint_paths: Sequence[Path]) -> str:
        first = checkpoint_paths[0].stem
        last = checkpoint_paths[-1].stem
        if first == last:
            return f"Upload {first}"
        return f"Upload {len(checkpoint_paths)} checkpoints ({first}–{last})"


def publish_run_to_hf(
    run_dir: Path,
    cfg: dict,
    project_root: Path,
    *,
    repo_id: str,
    repo_type: str = "model",
    private: bool = True,
    branch: str | None = None,
    tags: Sequence[str] | None = None,
    summary: str | None = None,
    include_model_code: bool = True,
    tokenizer_repo: str | None = None,
    dataset_repo: str | None = None,
    commit_message: str | None = None,
) -> Path:
    """Bundle a completed run directory and push it to the Hugging Face Hub.

    Returns:
        Path to the temporary export directory that was uploaded.
    """

    run_dir = Path(run_dir)
    project_root = Path(project_root)
    export_dir = run_dir / "hf_export"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    checkpoint_path = _select_latest_checkpoint(run_dir / "checkpoints")
    weights_name = "pytorch_model.bin" if checkpoint_path.suffix != ".safetensors" else "model.safetensors"
    shutil.copy2(checkpoint_path, export_dir / weights_name)

    config_yaml_path = run_dir / "config.yaml"
    if config_yaml_path.exists():
        shutil.copy2(config_yaml_path, export_dir / "config.yaml")

    metrics_file = run_dir / "metrics.jsonl"
    metrics_summary = {}
    if metrics_file.exists():
        shutil.copy2(metrics_file, export_dir / "metrics.jsonl")
        metrics_summary = _load_latest_metrics(metrics_file)

    if include_model_code:
        models_src = project_root / "models"
        if models_src.exists():
            models_dst = export_dir / "models"
            if models_dst.exists():
                shutil.rmtree(models_dst)
            _copy_minimal_model_code(models_src, models_dst)

    _write_minimal_readme(
        export_dir / "README.md",
        run_name=run_dir.name,
        cfg=cfg,
        metrics=metrics_summary,
        dataset_repo=dataset_repo,
        tokenizer_repo=tokenizer_repo,
        weights_name=weights_name,
    )

    api = HfApi()
    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    revision = branch or "main"
    if branch and branch != "main":
        try:
            api.create_branch(
                repo_id=repo_id,
                repo_type=repo_type,
                branch=branch,
                exist_ok=True,
            )
        except Exception:
            # Branch might already exist or the server might not support it; ignore.
            pass

    api.upload_folder(
        folder_path=str(export_dir),
        repo_id=repo_id,
        repo_type=repo_type,
        path_in_repo=".",
        commit_message=commit_message or f"Upload run {run_dir.name}",
        revision=revision,
    )
    print(f"Published run artifacts to hf://{repo_type}/{repo_id}@{revision}")
    return export_dir


def _select_latest_checkpoint(checkpoints_dir: Path) -> Path:
    checkpoint_dir = Path(checkpoints_dir)
    candidates = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.bin")) + list(
        checkpoint_dir.glob("*.safetensors")
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _load_latest_metrics(metrics_file: Path) -> dict:
    metrics: dict = {}
    with open(metrics_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                metrics.update(json.loads(line))
            except json.JSONDecodeError:
                continue
    return metrics


def _write_minimal_readme(
    readme_path: Path,
    *,
    run_name: str,
    cfg: dict,
    metrics: dict,
    dataset_repo: str | None,
    tokenizer_repo: str | None,
    weights_name: str,
) -> None:
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})
    model_name = cfg.get("name") or run_name

    lines = [f"# {model_name}", ""]
    lines.append("Artifacts in this repo:")
    lines.append(f"- `{weights_name}` (latest checkpoint)")
    lines.append("- `config.yaml`")
    lines.append("- `metrics.jsonl`")
    lines.append("- `models/` (minimal code to load the checkpoint)")
    lines.append("")

    if dataset_repo or tokenizer_repo:
        lines.append("Data references:")
        if dataset_repo:
            lines.append(f"- Dataset: `{dataset_repo}`")
        if tokenizer_repo:
            lines.append(f"- Tokenizer: `{tokenizer_repo}`")
        lines.append("")

    summary_items = {
        "max_steps": train_cfg.get("max_steps"),
        "batch_size": train_cfg.get("batch_size"),
        "lr": train_cfg.get("lr"),
        "dtype": train_cfg.get("dtype"),
        "n_layers": model_cfg.get("n_layers"),
        "d_model": model_cfg.get("d_model"),
        "n_ctx": model_cfg.get("n_ctx"),
    }
    summary_yaml = yaml.safe_dump({k: v for k, v in summary_items.items() if v is not None}, sort_keys=False).strip()
    if summary_yaml:
        lines.append("Training summary:")
        lines.append("```yaml")
        lines.append(summary_yaml)
        lines.append("```")
        lines.append("")

    if metrics:
        lines.append("Latest logged metrics:")
        for key, value in metrics.items():
            lines.append(f"- {key}: {value}")
        lines.append("")

    lines.append("To load:")
    lines.append("```python")
    lines.append("import torch")
    lines.append("from models.transformer import AttentionLM")
    lines.append("")
    lines.append(f"state = torch.load('{weights_name}', map_location='cpu')")
    lines.append("model = AttentionLM.from_config(yaml.safe_load(open('config.yaml')))\n")
    lines.append("model.load_state_dict(state['model_state_dict'])")
    lines.append("model.eval()")
    lines.append("```")

    readme_path.write_text("\n".join(lines).strip() + "\n")


def _copy_minimal_model_code(models_src: Path, models_dst: Path) -> None:
    """Copy just enough code to reload AttentionLM checkpoints."""

    models_dst.mkdir(parents=True, exist_ok=True)

    required_files = ("__init__.py", "transformer.py")
    missing = [name for name in required_files if not (models_src / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required model files under {models_src}: {missing}")

    for name in required_files:
        shutil.copy2(models_src / name, models_dst / name)

    attn_src = models_src / "attention_kernels"
    if not attn_src.is_dir():
        raise FileNotFoundError(
            f"Missing attention_kernels directory under {models_src}; cannot export model"
        )

    shutil.copytree(
        attn_src,
        models_dst / "attention_kernels",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
    )

