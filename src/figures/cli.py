"""Public CLIs for preparing and plotting figure families.

Three top-level entrypoints registered in `pyproject.toml`:

    uv run train   <family>   # only the training step (where applicable)
    uv run prepare <family>   # cache step (auto-triggers train if checkpoints absent)
    uv run plot    <family>   # render figures (auto-triggers prepare if cache absent)

Plus the legacy combined CLI for back-compat:

    uv run figures <action> <family>
"""
from __future__ import annotations

import argparse

from loguru import logger

from src.figures import FIGURES, resolve, stages


def _per_action_parser(action: str, help_text: str):
    parser = argparse.ArgumentParser(prog=f"uv run {action}", description=help_text)
    parser.add_argument("family", choices=sorted(FIGURES), help="Figure family to run.")
    return parser


def _run_stage(family: str, action: str):
    if action not in FIGURES[family]:
        raise SystemExit(f"{family} does not support {action}. Available stages: {', '.join(stages(family))}")
    resolve(family, action)()


def train_main(argv: list[str] | None = None):
    args = _per_action_parser("train", "Create checkpoints for a figure family.").parse_args(argv)
    _run_stage(args.family, "train")


def prepare_main(argv: list[str] | None = None):
    args = _per_action_parser("prepare", "Prepare local inputs and figure caches.").parse_args(argv)
    _run_stage(args.family, "prepare")


def plot_main(argv: list[str] | None = None):
    """`plot` auto-triggers `prepare` if the prepared cache is missing.

    The hand-off is data-driven: each family's plot reads its cache files at start.
    A `FileNotFoundError` raised there is interpreted as "cache missing"; we then
    invoke the family's prepare and re-call plot. One try, no flags.
    """
    args = _per_action_parser("plot", "Render figures from prepared cache.").parse_args(argv)
    try:
        _run_stage(args.family, "plot")
    except FileNotFoundError:
        logger.info(f"plot cache missing for {args.family}; running prepare first")
        _run_stage(args.family, "prepare")
        _run_stage(args.family, "plot")


def _legacy_parser():
    families = "\n".join(f"  {name:<24} {config['description']}" for name, config in FIGURES.items())
    parser = argparse.ArgumentParser(
        prog="uv run figures",
        description="Prepare local inputs and render paper figures.",
        epilog=f"Available figure families:\n{families}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="action", required=True)
    for action, help_text in (
        ("train", "Create checkpoints for a figure family."),
        ("prepare", "Prepare local inputs and figure caches."),
        ("plot", "Render figures from prepared cache."),
    ):
        subparser = subparsers.add_parser(action, help=help_text)
        subparser.add_argument("family", choices=sorted(FIGURES), help="Figure family to run.")
    return parser


def main(argv: list[str] | None = None):
    parser = _legacy_parser()
    args = parser.parse_args(argv)
    {"train": train_main, "prepare": prepare_main, "plot": plot_main}[args.action]([args.family])
