# Flappy RL

Flappy Bird RL env for [PufferLib](https://puffer.ai): high-throughput reinforcement learning (millions of steps/second).

## Quick start

```bash
uv sync
uv run python -m flappy_rl                         # run env demo (mock envs)
uv run python -m flappy_rl.train                   # train on Flappy Grid (2-row)
uv run python -m flappy_rl.train --train.env flappy # train on Flappy (C + raylib)
```

## Setup

**Requires:** Python 3.10+, [uv](https://docs.astral.sh/uv/). From repo root: `uv sync`.

*(If `uv sync` fails in the IDE due to cache permissions, run it in your system terminal.)*

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

## Flappy (C + raylib)

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
