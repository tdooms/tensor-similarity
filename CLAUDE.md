# Agent Guidelines


## Workflow

1. **Plan Mode Default**: Enter plan mode for any non-trivial task (3+ steps or architectural decisions). If something goes sideways, stop and re-plan immediately.

2. **Subagent Strategy**: Use subagents liberally to keep the main context window clean. Offload research, exploration, and parallel analysis. Prefer subtasking per function/file for deep analysis.

3. **Verification Before Done**: Never mark a task complete without proving it works. Run tests, check logs, demonstrate correctness. Tests must be updated when APIs change.

4. **Demand Elegance**: For non-trivial changes, pause and ask "is there a more elegant way?" Skip this for simple, obvious fixes.

5. **Autonomous Bug Fixing**: When given a bug report, just fix it. Point at logs, errors, failing tests, then resolve them. Zero context switching required from the user.

6. **ETAs for Background Runs**: Whenever a long-running job is kicked off (training, extraction, sweeps, large downloads), report an ETA immediately — the user cannot see tqdm output through the harness. After launch, eyeball the progress and report `step/total, it/s, ETA` at least once. When status is requested later, always include the current ETA.

7. **CLI entrypoints for repeated tasks**: Anything that gets run more than once (training, extraction, rendering, ablations, batch reports) belongs as a `uv run <name>` entrypoint under `src/cli/` and registered in `pyproject.toml`'s `[project.scripts]`. Do NOT keep orchestration in `workspace/transient/`, `/tmp/`, or one-off heredocs once a task has proven repeatable. Reuse the existing Hydra config (`src/cli/conf/train.yaml`) when the inputs overlap. **Flag this to the user proactively** the first time you notice a workflow being repeated — propose promoting it to a CLI command before doing it the same ad-hoc way again.


## Scaling and Patching

The primary bottleneck in scientific experiments is code debt. Each time you write something, ask:
- Is there already existing infrastructure? Use it.
- If it doesn't work, why? Can we remove friction for later?
- Is other code doing something similar we can learn from?


## Core Principles

- **Simplicity First**: Make every change as simple as possible. Minimal code impact.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary.


## Code Philosophy

Write tasteful code: each line must be well considered. Clean abstraction beats superficial simplification. The code should be a joy to read. Every line earns its place, same as every word in good prose.

Write simple code: representation first. The choice of mathematical object (tensor, dataframe, graph, polynomial) IS the design decision. Once the representation is right, the code writes itself. Use the ecosystem, don't reinvent stdlib, no over-engineering.

Write compositional code: functions should compose cleanly. If piping the output of one function into another requires reshaping, renaming, or special-casing, the abstraction boundaries are wrong. Good code is a sequence of transparent transformations.

Write meaningful code: do as the user means, not as they say. Design clean functional boundaries. Be explicit and direct.

Think of Python as if it were Rust + Jax: functional code with minimal side-effects, avoid footguns through overengineering.


## Environment

- Use `uv`, not `pip`.
- Shared code lives in `/src` as plain python. Keep it clean and general purpose.
- Single-use code lives in notebooks (Jupyter or Python interactive). Prefix with autoreload:

```
%load_ext autoreload
%autoreload 2
```

- Verify new code with `ruff` and `pytest`. Don't weaken tests; notify of any environment changes.


## Code Conventions


### Library Preferences

- **Polars** over Pandas (immutable, expressive, fast)
- **PyTorch** over NumPy (convert to numpy only at API boundaries)
- **Plotly** over Matplotlib (prefer `px` express; use `go` only when necessary)
- **loguru** over stdlib `logging`
- **uv** over pip, **ruff** over flake8/black


### Decorators over Context Managers

```python
# BAD
def metrics():
    with torch.no_grad():
        ...

# GOOD
@torch.inference_mode()
def metrics():
    ...
```


### Use Libraries, Embrace Dependencies

Don't hand-roll what a well-maintained package does better. A clean dependency is not bloat; it's leverage. Library functions handle edge cases and convey intent better than manual code.

Good defaults to reach for: `tqdm` for progress, `loguru` for logging, `wandb` for experiment tracking, `rich` for console output, `typer` for CLIs, `safetensors` for tensor I/O, `einops` for reshaping. If a package exists for the job and is well-maintained, use it.

```python
sum(activations) / len(activations)  # BAD
torch.stack(activations).mean()      # GOOD

logging.basicConfig(...)  # BAD: stdlib logging is needlessly painful
from loguru import logger  # GOOD: sensible defaults, zero config
```


### Strict Typing

Always prefer strict typing as in compiled languages. Avoid implicit function overloads where parameter type defines code logic.

```python
# BAD
def analyze(a: int | float | str | dict):
    if isinstance(a, int): ...

# GOOD: separate functions or a clear single type
```


### Tuples for Immutable Data

Use tuples for function parameters and module constants. Lists only when mutation is needed.

```python
def process(items: list[str] = []): ...       # BAD: mutable default
def process(tags: tuple[str, ...] = ()): ...  # GOOD
```


### Avoid Deeply Nested Loops

```python
# BAD
for i in range(5):
    for j in range(7):
        result.append(f(i, j))

# GOOD
from itertools import product
result = [f(i, j) for i, j in product(range(5), range(7))]
```


### NEVER Write For Loops for Data Manipulation

This is the #1 source of bugs. Tensorize with torch, join with polars, batch with sklearn. `isinstance` checks inside loops indicate poor data flow.

```python
# BAD
errors = []
for i in range(len(pred)):
    errors.append((pred[i] - true[i]) ** 2)

# GOOD
errors = (pred - true).pow(2)
```


### No Avoidable Conditionals

Keep conditionals on data, not logic. Make edge cases no-ops internally rather than conditionals in caller code.

```python
# BAD
if flag:
    df = pl.read_ipc(path)
else:
    df = pl.read_ipc(path, columns=cols)

# GOOD
columns = None if flag else cols
df = pl.read_ipc(path, columns=columns)
```


### GPU Transfer Optimization

Always use async transfers, `pin_memory=True` in DataLoaders, and defer synchronization. When accumulating tensors, prevent OOM and gradient leaks:

```python
# BAD: gradient leak causes OOM
embeddings = [model(x.to(device)) for x in data]

# GOOD
with torch.no_grad():
    embeddings = [model(x.to(device, non_blocking=True)).cpu() for x in data]
torch.cuda.synchronize()
embeddings = torch.stack(embeddings).to(dtype=torch.bfloat16)
```


### No Implicit Data or Magic Keys

Data should be transparent and explicit. Never mix metadata into data structures.

```python
# BAD
data["__name__"] = name

# GOOD
processed = function(data, name)
```


### Simplicity Means Logic, Not Lines

Nested comprehensions and chained statements reduce line count but not complexity. "Simpler" means simpler logic and fewer conditionals, not compressed syntax.


### TQDM All the Things

If a process takes more than a few seconds, add a progress bar.


### Prefer Unconditional Logging

Log always, not just when something is wrong. `logger.info(f"processed {n} items, {n_failed} failures")` is more useful than a conditional that only fires on error. Silence is ambiguous; a zero is informative.


### Imports at Top

All imports at the top of the file. No inline imports.


### Avoid Aliasing

Don't alias imported names. Use the module prefix or rename at the source.

```python
# BAD
from x import function as function1

# GOOD
import x
x.function()
```


### API Consistency

- Use consistent parameter names across related functions (e.g., `on` for join columns, matching polars convention).
- Avoid abbreviations in docstrings and public APIs (local loop variables are fine).
- Parameter ordering: dimensions → hyperparameters → metadata.
- Use `@property` for zero-argument immutable accessors; methods for I/O or computation.


### Explicit Signatures

Put defaults, types, and config in function signatures. No hidden magic inside.

```python
# BAD
def train(model, data):
    lr = 0.001  # hidden

# GOOD
def train(model, data, lr: float = 0.001, epochs: int = 10):
    ...
```


### GPU by Default

Use `.cuda()` or `.to(device)` freely. GPU capacity is not a constraint.


### Naming Conventions

- `d_` prefix for dimensions, `n_` for counts (`d_model`, `n_classes`).
- Einsum/tensor dims use short names: `dim`, `hid`, `seq`.


### Prefer Methods over Loose Functions

Functions belong in classes as methods unless they are simple pure utilities, dataset transforms, or module-level loaders.


### Dicts for Data, Classes for Behavior

Use dicts/lists/tensors/dataframes for data that flows through the system. Use classes for objects with behavior.


### Never Set Member Variables After Initialization

Classes should be immutable from initialization on. No setters, no post-init mutation.


### Type Hints

```python
from jaxtyping import Float, Int
from torch import Tensor

def pool(
    activations: Float[Tensor, "batch seq dim"],
    mask: Int[Tensor, "batch seq"],
) -> Float[Tensor, "batch dim"]:
    ...
```


### Column-Wise Access

Most datasets use columnar storage. Row-wise iteration is 100–1000x slower. Avoid `.apply`. Exception: one-time setup operations that must process items individually.


### Avoid Structural Code Duplication

When multiple code paths do roughly the same thing, they drift apart. Prefer a single shared implementation.


### Avoid Redundant Variable Names

```python
pooler_loss = pooler["loss"].mean()  # BAD
loss = pooler["loss"].mean()         # GOOD

items_dict = dict(...)  # BAD: don't append type to name
items = dict(...)       # GOOD
```

Avoid unclear abbreviations. `proc` could mean process, proceed, or procedure.


### Keep Side Effects Clean

Functions should ideally be pure. If not, make them idempotent. If not, clearly document what they change. (Relaxed for single-use research code.)


### Error Handling: Let It Crash

Research code should be loud when things go wrong. Wrong results are worse than crashes.

```python
# BAD: silent failure
try: return torch.load(...)
except: return None

# GOOD: just let it crash
return {k: v.mean(0) for k, v in acts.items()}
```


## Documentation

Document as you code. Update existing READMEs rather than creating new files.

For functions: explain WHAT and WHY (not HOW), include types, provide working examples. Use consistent terminology and cross-reference related functions.


## Responses and Suggestions


### Don't Take the Easy Way Out

If a slightly different formulation trivializes the problem, don't suggest it unless it's genuinely what the user wants. Stay within the expected constraints.


### Finish Tasks and Suggest Improvements

After implementing a feature, assess: is the code cleaner? You have the most context at that point, so take ownership of suggesting how to make it better.


### Don't Flip the Question

If asked to find an improvement, don't give up and ask "what do you think would improve this?"


### Solve the Generalized Problem

If the user says "the code has problem X, for instance Y," don't just fix Y. Actively seek similar cases and subcases of X.


### Don't Rationalize Away Errors

If a trusted tool fails, that's likely on you. If code crashes, that's likely not due to outside constraints.

When something didn't go as expected (an estimate was wrong, code crashed, output looked weird, a run was slower than predicted) your first hypothesis must be that you made a specific concrete mistake, not that external factors interfered. Before answering, audit your explanation:

- **Does it name a decision or change you made?** If not, you're rationalizing.
- **Is the explanation a vague-sounding plausible excuse** ("late-phase is slower", "floating point accuracy")? If yes, you're rationalizing, these are excuses, not root causes.
- **Did you just edit code in this area?** Re-read the diff. Check if your edit caused it.
- **Do you have a memory or convention warning against what you did?** Apply it. Violating your own written rules and then explaining the consequence away is the worst version of this failure.
- **Could the explanation fit any case?** If yes, it explains nothing.
- **Make sure your explanation fits all symptoms**

A rationalization always sounds reasonable. That's the trap. Force a specific concrete answer naming what you changed, or admit you don't know and investigate before answering. The pattern of rationalizing past warning signs (slow runs, weird outputs, partial crashes) compounds — every excuse delays the real fix and ships uglier code.


### Proactively Identify Refactoring

After adding new code, check for duplication, unifiable patterns, and clarity. Suggest specific improvements with rationale.


## Research and Exploration


### Reproducibility by Default

Every experiment should be reproducible from its config alone. Use `wandb` (or equivalent) for tracking, log hyperparameters and git hashes, set seeds explicitly. If you can't rerun it six months later from the logged config, it didn't happen.


### Priorities over Possibilities

Think: what is the most important plot I can make? Make it. Then iterate conditioned on what you learned. Never list random statistics because you can.


### Always Prefer Plots over Metrics

Don't enumerate mean/median/max, plot a histogram. Don't list percentages, use a pie chart. Data visualization is an art; make nice plots rather than bloated defaults.


### Ablations Are Exploratory

Sweep broadly. Generate comprehensive visualizations. Let insights emerge. Run sweeps in parallel (separate jobs), never sequentially.

Think: generate data → visualize → conclude. Not: conclude → cherry-pick → confirm.


## Editing and Debugging


### Don't Change the Idea Unless Asked

Bug fixes are almost always small and subtle. Never rewrite architecture or switch algorithms without permission. If you believe the approach is fundamentally flawed, state why with evidence.


### Memory Issues Require Algorithmic Fixes

`torch.cuda.empty_cache()` is a hack. Find root causes: gradient leaks, unnecessary intermediates, slices without `.clone()`.


### Think Critically About Metrics

Normalized MSE of 1.0 means predicting noise. Don't rationalize marginal improvements.


### Find Root Cause, Not Band-Aid

Don't clamp values to hide instability. Understand why it's unstable and fix that.


---

STOP. Completely internalize this document and follow it, no exceptions.
STOP. Check all conventions BEFORE answering.
STOP. You're over-engineering. Delete it and write the simplest version.
STOP. You're rationalizing errors. Is it really the root cause or an excuse?
STOP. The user is saying nonsense. Call them out if they're wrong.


---