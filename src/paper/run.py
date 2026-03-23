"""Run all paper experiments end-to-end.

Usage:
  python -m src.paper.run              # everything
  python -m src.paper.run convergence  # just convergence
  python -m src.paper.run perturbation # just perturbation
"""
import sys
import time


def run(name, fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    fn()
    print(f"\n  Done in {time.time()-t0:.1f}s")


def convergence():
    from src.paper.convergence.train import main as train
    from src.paper.convergence.experiment import main as experiment
    from src.paper.convergence.plot import main as plot

    run("convergence/train", train)
    run("convergence/experiment", experiment)
    run("convergence/plot", plot)


def perturbation():
    from src.paper.perturbation.train import main as train
    from src.paper.perturbation.experiment import main as experiment
    from src.paper.perturbation.plot import main as plot

    run("perturbation/train", train)
    run("perturbation/experiment", experiment)
    run("perturbation/plot", plot)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    pipelines = {"convergence": convergence, "perturbation": perturbation}

    if which == "all":
        for p in pipelines.values():
            p()
    elif which in pipelines:
        pipelines[which]()
    else:
        print(f"Unknown: {which}. Options: all, convergence, perturbation")
        sys.exit(1)


if __name__ == "__main__":
    main()
