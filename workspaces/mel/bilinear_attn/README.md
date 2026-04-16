# language-modelling

bilinear attention, trains on SimpleStories.

## setup

```
uv sync
uv pip install -e .
```

if you dont wanna use my uv.lock:
```
uv sync --refresh
git restore uv.lock
uv pip install -e .
```

## train

```
python -m scripts.run_train --config configs/main256.yaml
```

see `models/README.md` for architecture details and config options.

## track bigram and skip trigram scores over training

```
python -m scripts.run_train --config configs/main256.yaml --track-behaviour
```

## evaluate

```
python -m scripts.run_eval --checkpoint runs/<run>/checkpoints/final.pt --config configs/main256.yaml
```

## tests

```
pytest
```
