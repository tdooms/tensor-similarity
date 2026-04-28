"""Per-action CLIs for figure families.

    uv run train   <family>   # training step (where applicable)
    uv run prepare <family>   # cache step
    uv run plot    <family>   # render figures from prepared cache
"""
import argparse
import importlib

from src.figures import FIGURES


def _entry(action, argv):
    families = sorted(family for family, spec in FIGURES.items() if action in spec)
    parser = argparse.ArgumentParser(prog=f"uv run {action}")
    parser.add_argument("family", choices=families)
    args = parser.parse_args(argv)
    importlib.import_module(FIGURES[args.family][action]).main()


def train_main(argv=None):   _entry("train", argv)
def prepare_main(argv=None): _entry("prepare", argv)
def plot_main(argv=None):    _entry("plot", argv)
