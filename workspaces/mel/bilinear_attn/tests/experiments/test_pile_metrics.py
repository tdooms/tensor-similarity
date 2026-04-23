"""Tests for the pile_metrics experiment utilities.

These tests do *not* hit the network or run a real evaluation; they cover
the pure-Python helpers: checkpoint listing / selection, the resumable
jsonl logic, and that the plotting functions produce a file without
error when given synthetic data.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

from experiments.pile_metrics.plots import (  # noqa: E402
    load_metrics,
    plot_ablation_and_icl,
    plot_ngrams,
)
from experiments.pile_metrics.run import (  # noqa: E402
    _parse_step,
    append_jsonl,
    list_remote_checkpoints,
    load_done_steps,
    load_repo_config,
    select_checkpoints,
)


# ---------------------------------------------------------------------------
# parse_step / select_checkpoints
# ---------------------------------------------------------------------------

def test_parse_step_variants():
    assert _parse_step("checkpoints/step_00042.pt") == 42
    assert _parse_step("step_1000.pt") == 1000
    assert _parse_step("junk.pt") is None


def test_select_checkpoints_stride_min_max_limit():
    ckpts = [f"checkpoints/step_{i:05d}.pt" for i in range(0, 100, 10)]  # 0,10,..,90
    # stride only: strided picks + the always-included last checkpoint
    assert select_checkpoints(ckpts, stride=2, min_step=None, max_step=None, limit=None) == [
        ckpts[0], ckpts[2], ckpts[4], ckpts[6], ckpts[8], ckpts[-1],
    ]
    # min + max (last checkpoint is 90, which is outside max_step=70, so not added)
    got = select_checkpoints(ckpts, stride=1, min_step=20, max_step=70, limit=None)
    assert [_parse_step(p) for p in got] == [20, 30, 40, 50, 60, 70]
    # limit (applied after appending the last)
    got = select_checkpoints(ckpts, stride=1, min_step=None, max_step=None, limit=3)
    assert len(got) == 3


def test_select_checkpoints_always_includes_last():
    ckpts = [f"checkpoints/step_{i:05d}.pt" for i in (0, 100, 250, 999)]
    # stride=10 would skip everything except index 0, but the last must be kept.
    got = select_checkpoints(ckpts, stride=10, min_step=None, max_step=None, limit=None)
    assert _parse_step(got[0]) == 0
    assert _parse_step(got[-1]) == 999
    # When the last already falls on the stride, no duplicate.
    got = select_checkpoints(ckpts, stride=1, min_step=None, max_step=None, limit=None)
    assert len(got) == len(ckpts)


# ---------------------------------------------------------------------------
# Resumable jsonl
# ---------------------------------------------------------------------------

def test_append_and_load_done_steps(tmp_path: Path):
    jsonl = tmp_path / "analysis_metrics.jsonl"
    assert load_done_steps(jsonl) == set()
    append_jsonl(jsonl, {"step": 0, "val_loss": 1.0})
    append_jsonl(jsonl, {"step": 100, "val_loss": 0.9})
    append_jsonl(jsonl, {"step": 100, "val_loss": 0.9})  # duplicate OK
    done = load_done_steps(jsonl)
    assert done == {0, 100}


def test_load_done_steps_skips_malformed_lines(tmp_path: Path):
    jsonl = tmp_path / "m.jsonl"
    jsonl.write_text(
        '{"step": 5, "x": 1}\n'
        'not-json\n'
        '\n'
        '{"no_step": true}\n'
        '{"step": 7}\n'
    )
    assert load_done_steps(jsonl) == {5, 7}


# ---------------------------------------------------------------------------
# list_remote_checkpoints uses the repo listing — stub the API client.
# ---------------------------------------------------------------------------

def test_load_repo_config_prefers_json(tmp_path: Path, monkeypatch):
    """config.json must be tried first; config.yaml is a fallback."""
    json_file = tmp_path / "config.json"
    json_file.write_text('{"model": {"vocab_size": 4096}, "name": "x"}')

    calls: list[str] = []

    def fake_download(filename, repo_id, repo_type, **kwargs):
        calls.append(filename)
        if filename == "config.json":
            return str(json_file)
        from huggingface_hub.utils import EntryNotFoundError
        raise EntryNotFoundError(f"no {filename}")

    monkeypatch.setattr(
        "experiments.pile_metrics.run.hf_hub_download", fake_download
    )

    cfg = load_repo_config("x/y", repo_type="dataset")
    assert calls == ["config.json"]  # yaml not even attempted
    assert cfg["model"]["vocab_size"] == 4096


def test_load_repo_config_falls_back_to_yaml(tmp_path: Path, monkeypatch):
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text("model:\n  vocab_size: 2048\nname: y\n")

    def fake_download(filename, repo_id, repo_type, **kwargs):
        from huggingface_hub.utils import EntryNotFoundError
        if filename == "config.json":
            raise EntryNotFoundError("missing")
        return str(yaml_file)

    monkeypatch.setattr(
        "experiments.pile_metrics.run.hf_hub_download", fake_download
    )
    cfg = load_repo_config("x/y")
    assert cfg["model"]["vocab_size"] == 2048


def test_load_repo_config_raises_when_absent(monkeypatch):
    def fake_download(filename, repo_id, repo_type, **kwargs):
        from huggingface_hub.utils import EntryNotFoundError
        raise EntryNotFoundError("missing")

    monkeypatch.setattr(
        "experiments.pile_metrics.run.hf_hub_download", fake_download
    )
    import pytest

    with pytest.raises(FileNotFoundError):
        load_repo_config("x/y")


def test_list_remote_checkpoints_sorts_by_step():
    fake_api = SimpleNamespace(
        list_repo_files=lambda repo_id, repo_type: [
            "README.md",
            "checkpoints/step_00100.pt",
            "checkpoints/step_00000.pt",
            "checkpoints/step_01234.pt",
            "checkpoints/other.txt",
            "other/step_00050.pt",
        ]
    )
    got = list_remote_checkpoints(fake_api, repo_id="x/y")
    assert got == [
        "checkpoints/step_00000.pt",
        "checkpoints/step_00100.pt",
        "checkpoints/step_01234.pt",
    ]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _fake_entries():
    return [
        {
            "step": step,
            "val_loss": 3.0 - 0.01 * step,
            "ablated_loss": 3.5 - 0.005 * step,
            "ablation_gap": 0.5 + 0.005 * step,
            "2gram_loss": 4.0,
            "2gram_test_loss": 3.9 - 0.01 * step,
            "2gram_score": (3.9 - 0.01 * step) / 4.0,
            "3gram_loss": 3.5,
            "3gram_test_loss": 3.4 - 0.01 * step,
            "3gram_score": (3.4 - 0.01 * step) / 3.5,
            "loss_50": 3.2 - 0.01 * step,
            "loss_500": 2.9 - 0.012 * step,
            "icl_50_500": -0.3 - 0.002 * step,
        }
        for step in range(0, 101, 20)
    ]


def test_load_metrics_sorts_by_step(tmp_path: Path):
    jsonl = tmp_path / "m.jsonl"
    with open(jsonl, "w") as f:
        for e in reversed(_fake_entries()):  # write out-of-order
            f.write(json.dumps(e) + "\n")
    got = load_metrics(jsonl)
    assert [e["step"] for e in got] == sorted([e["step"] for e in got])


def test_plot_ngrams_writes_file(tmp_path: Path):
    out = tmp_path / "ngrams.png"
    plot_ngrams(_fake_entries(), out)
    assert out.exists() and out.stat().st_size > 0


def test_plot_ablation_and_icl_writes_file(tmp_path: Path):
    out = tmp_path / "ablation_icl.png"
    plot_ablation_and_icl(_fake_entries(), out)
    assert out.exists() and out.stat().st_size > 0


def test_plots_tolerate_missing_keys(tmp_path: Path):
    # Only val_loss present - functions must not crash.
    minimal = [{"step": 0, "val_loss": 2.0}, {"step": 10, "val_loss": 1.9}]
    plot_ablation_and_icl(minimal, tmp_path / "a.png")
    plot_ngrams(minimal, tmp_path / "b.png")
    assert (tmp_path / "a.png").exists()
    assert (tmp_path / "b.png").exists()
