# Paper figures

Figure modules in this folder are the public paper-facing entrypoints. They are
the canonical figure pipelines. Each family stays within the three-stage shape
`train -> prepare -> plot`, with at most three experiment-specific files. EDA
does not live here — promote durable figure code to this folder, push everything
else to `workspaces/<user>/transient/`.

## Workflow

```bash
uv run train   <family>   # training step (where applicable)
uv run prepare <family>   # cache step
uv run plot    <family>   # render figures from prepared cache
```

Available figure families:

- `language-similarity`
- `grokking-similarity`
- `svhn-backdoor`
- `svhn-forgetting` (also renders the `svhn-progress` companion line plot)
- `svhn-diffing`

Shared design + export helpers live in `style.py`. Per-experiment logic lives
in `src/figures/<family>/{train,prepare,plot}.py`.

Repo-local inputs live in `_downloads/`, prepared tabular caches live in
`artifacts/cache/<family>/`, and rendered figures live in `artifacts/figures/`
as `.pdf`. Pipelines prefer `.safetensors`, `.feather`, and `.json` for I/O.
