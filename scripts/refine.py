"""Iterative code-taste refinement via independent Claude agents.

Each cycle on a target file:
  1. Rater (fresh `claude -p` session, read-only) scores 1-10 + JSON cuts.
  2. Fixer (fresh `claude -p` session, Read+Edit) applies the cuts.
  3. Tests run (`uv run pytest -q`); on failure the file is reverted.
  4. Verifier is the next round's rater — naturally independent of both
     the prior rater (different session) and the fixer.

Stops when score >= TARGET, no cuts proposed, fixer produced no diff, or
MAX_CYCLES exhausted. Per-cycle log written to artifacts/review/<file>.json.

Usage:
    uv run python scripts/refine_taste.py <file>...
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

CLAUDE = os.environ["CLAUDE_CODE_EXECPATH"]
REPO = Path(__file__).resolve().parents[1]
TARGET_SCORE = 9
MAX_CYCLES = 6
MODEL = "sonnet"
OUT_DIR = REPO / "artifacts" / "review"

CONVENTIONS = """\
Project conventions (CLAUDE.md):
- "Conditions are code killers". Kill avoidable if/else.
- Let it crash. No defensive `if X is not None`, `if X in dict`, try/except for impossible cases.
- Comments: only WHY (when non-obvious), never WHAT.
- No `[N/M]` progress narration in logger calls (tqdm narrates).
- No single-use wrapper functions. No redundant intermediate variables.
- Polars > pandas, torch > numpy, loguru > print, uv > pip.
- Tuples for module constants, not lists.
- Decorators over context managers (`@torch.no_grad()` not `with torch.no_grad():`).
- No backwards-compat shims, no removed-code stubs."""

RATER = """\
You are a senior code reviewer obsessed with minimalism for research code.

Read {path}. Rate its TASTE 1-10. 10 = every line earns its place; 5 = typical OK code; 1 = codex-mess.

{conv}

Output STRICT JSON (one object, nothing else, no code fences):
{{"score": <int 1-10>, "praise": "<one sentence>", "cuts": [{{"location": "<line N or function foo>", "issue": "<what>", "fix": "<minimal alternative>"}}]}}

If score >= {target}, return cuts: []. Don't invent cuts to seem rigorous."""

FIXER = """\
Apply these specific minimalism cuts to {path} via the Edit tool. ONLY edit {path}.

Be surgical: apply only the listed cuts, do not introduce abstractions, rename, or reformat. Preserve behavior.

{conv}

Cuts:
{cuts}

When done, output a single line: DONE"""


def claude(prompt: str, tools: tuple[str, ...]) -> str:
    out = subprocess.run(
        [CLAUDE, "-p", prompt,
         "--output-format", "json",
         "--model", MODEL,
         "--permission-mode", "bypassPermissions",
         "--add-dir", str(REPO),
         "--allowed-tools", ",".join(tools)],
        capture_output=True, text=True, check=True, timeout=900, cwd=REPO,
    )
    return json.loads(out.stdout)["result"]


def parse_rating(text: str) -> dict:
    m = re.search(r"\{[\s\S]*\}", text.strip())
    if not m:
        raise ValueError(f"no JSON in rater output: {text[:300]}")
    return json.loads(m.group(0))


def file_hash(path: Path) -> str:
    return subprocess.run(["git", "hash-object", str(path)],
                          capture_output=True, text=True, check=True, cwd=REPO).stdout.strip()


def tests_pass() -> bool:
    r = subprocess.run(["uv", "run", "pytest", "-q", "--no-header", "-x"],
                       capture_output=True, text=True, cwd=REPO)
    return r.returncode == 0


def revert(path: Path) -> None:
    subprocess.run(["git", "checkout", "--", str(path)], check=True, cwd=REPO)


def cycle(path: Path) -> dict:
    rate = parse_rating(claude(RATER.format(path=path, conv=CONVENTIONS, target=TARGET_SCORE),
                               tools=("Read",)))
    if rate["score"] >= TARGET_SCORE or not rate["cuts"]:
        return {"phase": "stop", "rate": rate}

    pre = file_hash(path)
    claude(FIXER.format(path=path, conv=CONVENTIONS,
                        cuts=json.dumps(rate["cuts"], indent=2)),
           tools=("Read", "Edit", "Bash"))
    post = file_hash(path)

    if pre == post:
        return {"phase": "no_diff", "rate": rate}

    if not tests_pass():
        revert(path)
        return {"phase": "revert", "rate": rate}

    return {"phase": "ok", "rate": rate, "post_hash": post}


def refine(path: Path) -> list[dict]:
    log = []
    for n in range(MAX_CYCLES):
        logger.info(f"  cycle {n + 1}/{MAX_CYCLES}")
        try:
            r = cycle(path)
        except subprocess.TimeoutExpired as e:
            log.append({"cycle": n, "error": f"timeout: {e}"})
            break
        except Exception as e:
            log.append({"cycle": n, "error": f"{type(e).__name__}: {e}"})
            break
        log.append({"cycle": n, "ts": datetime.now(timezone.utc).isoformat(), **r})
        score = r["rate"]["score"]
        n_cuts = len(r["rate"]["cuts"])
        logger.info(f"    score={score}/10 cuts={n_cuts} phase={r['phase']}")
        if r["phase"] in ("stop", "no_diff", "revert"):
            break
    return log


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for arg in sys.argv[1:]:
        path = Path(arg).resolve().relative_to(REPO)
        logger.info(f"=== {path} ===")
        log = refine(path)
        out = OUT_DIR / f"{path.name}.json"
        out.write_text(json.dumps(log, indent=2))
        final = log[-1]["rate"]["score"] if log and "rate" in log[-1] else None
        logger.info(f"    final score: {final}, log: {out}")


if __name__ == "__main__":
    main()
