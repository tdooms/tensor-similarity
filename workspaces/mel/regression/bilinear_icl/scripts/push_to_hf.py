import argparse
from pathlib import Path

from bilinear_icl.io import push_run_to_hf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action="store_true")
    args = parser.parse_args()

    push_run_to_hf(Path(args.run_dir), repo_id=args.repo_id, private=args.private)


if __name__ == "__main__":
    main()
