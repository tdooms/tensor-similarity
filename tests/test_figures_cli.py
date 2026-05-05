import torch
import pytest

from src.components.similarity import precompile, similarity_parts
from src.figures import ARTIFACT_DIR, CACHE_DIR, DOWNLOAD_DIR, EXPERIMENT_DIR, FIGURE_DIR, FIGURES, REPO_ROOT, cosine_from_parts
from src.figures.cli import plot_main, prepare_main, train_main
from src.figures.style import COLORWAY, CURRICULUM_COLORS, SUBSET_COLORS
from src.models.checkpoint_transformer import CheckpointTransformer
from src.models.deep_mlp import DeepMLP


def test_artifact_directories_live_at_repo_root():
    assert REPO_ROOT.name == "tensor-mars"
    assert DOWNLOAD_DIR == REPO_ROOT / "_downloads"
    assert ARTIFACT_DIR == REPO_ROOT / "artifacts"
    assert CACHE_DIR == ARTIFACT_DIR / "cache"
    assert EXPERIMENT_DIR == ARTIFACT_DIR / "experiments"
    assert FIGURE_DIR == ARTIFACT_DIR / "figures"


def test_registry_contains_public_figure_families():
    assert set(FIGURES) == {"seed-convergence", "curriculum-shift", "language-similarity", "grokking-similarity", "svhn-backdoor", "svhn-backdoor-focused", "svhn-forgetting", "svhn-diffing", "subset-training"}


def test_deep_mlp_similarity_smoke():
    torch.manual_seed(0); a = DeepMLP(d_input=64, d_model=16, d_hidden=32, d_output=4, n_layers=1).eval()
    torch.manual_seed(1); b = DeepMLP(d_input=64, d_model=16, d_hidden=32, d_output=4, n_layers=1).eval()
    precompile(a, b)
    cosine = cosine_from_parts(*similarity_parts(a, b))
    assert -1.0 <= cosine <= 1.0


def test_checkpoint_transformer_from_hf_config_smoke():
    cfg = {"model": {"vocab_size": 32, "d_model": 16, "n_head": 2, "n_layers": 2,
                     "attn_scale": 0.35, "use_bias_qk": True, "attn_type": "bilinear"}}
    a = CheckpointTransformer.from_hf_config(cfg, n_ctx=4).eval()
    b = CheckpointTransformer.from_hf_config(cfg, n_ctx=4).eval()
    precompile(a, b)
    cosine = cosine_from_parts(*similarity_parts(a, b))
    assert -1.0 <= cosine <= 1.0


def test_official_figures_code_does_not_import_from_workspaces():
    """Reading workspace data from prepare.py is fine (e.g. logan's grokking bundle);
    importing workspace *code* into the official pipeline is not."""
    for path in (REPO_ROOT / "src" / "figures").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from workspaces" not in text, f"{path} imports from workspaces"
        assert "import workspaces" not in text, f"{path} imports workspaces"


@pytest.mark.parametrize("entrypoint", [train_main, prepare_main, plot_main])
def test_top_level_entrypoints_reject_missing_family(entrypoint):
    with pytest.raises(SystemExit):
        entrypoint([])


@pytest.mark.parametrize("entrypoint", [train_main, prepare_main, plot_main])
def test_top_level_entrypoints_reject_unknown_family(entrypoint):
    with pytest.raises(SystemExit):
        entrypoint(["nonexistent"])


def test_train_rejects_hf_sourced_family():
    """language-similarity has no train stage; argparse rejects it for `train`."""
    with pytest.raises(SystemExit):
        train_main(["language-similarity"])


def test_color_palettes_share_one_source():
    """Per-family palettes should be a slice of `COLORWAY`, not parallel hex literals."""
    assert set(CURRICULUM_COLORS.values()) <= set(COLORWAY)
    assert set(SUBSET_COLORS.values()) <= set(COLORWAY)


def test_no_numpy_in_official_figures_code():
    """Math primitive is torch; numpy is allowed only in prepare.py at the .npy/.npz
    I/O boundary (e.g. logan's grokking bundle ships .npy files)."""
    paths = [p for p in (
        *(REPO_ROOT / "src" / "figures").rglob("*.py"),
        *(REPO_ROOT / "src" / "models").rglob("*.py"),
    ) if p.name != "prepare.py"]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "import numpy" not in text, f"{path} imports numpy"
        assert "from numpy" not in text, f"{path} from-imports numpy"
