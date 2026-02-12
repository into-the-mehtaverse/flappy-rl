# Flappy RL

Flappy Bird RL env for [PufferLib](https://puffer.ai): high-throughput reinforcement learning (millions of steps/second).

## Recommended workflow (use this)

For current results, use the curriculum setup in `variations/flappy/` for training and eval.
The best checkpoints in this repo were produced with this path, not the base `src/flappy_rl/flappy/` setup.

## Quick start

```bash
uv sync
uv run python -m flappy_rl.train --train.env flappy_curriculum --train.total-timesteps 150000000
uv run python -m variations.flappy.run_eval --model experiments/177087020156/model_018310.pt --episodes 50 --no-render
uv run python -m variations.flappy.eval_last_checkpoints --run-id 177087020156 --last 5 --episodes 50 --difficulty 1.0
```

## Setup

**Requires:** Python 3.10+, [uv](https://docs.astral.sh/uv/). From repo root: `uv sync`.

*(If `uv sync` fails in the IDE due to cache permissions, run it in your system terminal.)*

## Current project status (latest code path)

- The most up-to-date training setup is the curriculum environment in `variations/flappy/`.
- Best checkpoints so far:
  - `experiments/177086914642/model_009765.pt` (80M-step run)
  - `experiments/177087020156/model_018310.pt` (150M-step run, current best)
- Base env in `src/flappy_rl/flappy/` is still used and maintained, but the latest curriculum logic lives in `variations/flappy/curriculum.py`.
- Train latest curriculum setup:
  - `uv run python -m flappy_rl.train --train.env flappy_curriculum --train.total-timesteps 150000000`
- Eval latest curriculum checkpoints:
  - `uv run python -m variations.flappy.run_eval --model experiments/177087020156/model_018310.pt --episodes 50 --no-render`
  - `uv run python -m variations.flappy.eval_last_checkpoints --run-id 177087020156 --last 5 --episodes 50 --difficulty 1.0`

## What's in this repo

| Path | Description |
|------|-------------|
| **`src/flappy_rl/env.py`** | Envs: Flappy Grid, Gymnasium wrapper, native PufferEnv |
| **`src/flappy_rl/flappy/`** | Flappy Bird (C + raylib): physics, binding, build |
| **`src/flappy_rl/train.py`** | Train with PuffeRL (PPO-style); `--train.env flappy` for C Flappy |
| **`src/flappy_rl/run_eval_flappy.py`** | Eval a checkpoint (render or headless stats) |
| **`src/flappy_rl/sweep.py`** | Sweep lr × clip_coef; reports final entropy |
| **`scripts/eval_all_checkpoints.py`** | Run eval on all main Flappy checkpoints (50 ep each) |
| **`docs/`** | [env-instructions.md](docs/env-instructions.md), [rl-basics.md](docs/rl-basics.md), [env-checklist.md](docs/env-checklist.md), [blog.md](docs/blog.md) |

## Base env (reference)

- **Build:** See [src/flappy_rl/flappy/README.md](src/flappy_rl/flappy/README.md) (raylib, `make`, assets in `resources/flappy/`).
- **Train:** `uv run python -m flappy_rl.train --train.env flappy` (checkpoints in `experiments/`).
- **Eval (render):** `uv run python -m flappy_rl.run_eval_flappy --model experiments/<run>/model_XXXXXX.pt`
- **Eval (headless, 50 ep):** `uv run python -m flappy_rl.run_eval_flappy --model <path> --episodes 50 --no-render`  
  Use `--random-seed` to vary episodes between runs; default seed 42 is reproducible.
- **Eval all checkpoints:** `uv run python scripts/eval_all_checkpoints.py` (expects existing `experiments/` run dirs).

## Sweep (Flappy Grid)

```bash
uv run python -m flappy_rl.sweep
```

Runs lr × clip_coef grid, 500k steps per run; entropy &lt; 0.01 = deterministic. Edit `sweep.py` for grid and timesteps.

## Docs

- **PufferLib:** https://puffer.ai/docs.html
- **This repo:** [docs/env-instructions.md](docs/env-instructions.md) (writing envs, API), [docs/rl-basics.md](docs/rl-basics.md) (RL intro), [docs/env-checklist.md](docs/env-checklist.md) (native PufferEnv checklist).
