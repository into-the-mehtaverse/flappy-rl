# Curly-succotash

First RL env for [PufferLib](https://puffer.ai): high-throughput reinforcement learning (millions of steps/second).

## Setup

**Requires:** Python 3.10+, [uv](https://docs.astral.sh/uv/).

From the repo root:

```bash
uv sync
```

*(If `uv sync` fails in the IDE due to cache permissions, run it in your system terminal.)*

## What’s in this repo

- **`src/curly_succotash/env.py`** — Two minimal envs:
  1. **Gymnasium + wrapper:** `SampleGymnasiumEnv` + `make_gymnasium_env()` (easiest).
  2. **Native PufferEnv:** `SamplePufferEnv` (in-place buffer updates for vectorization).
- **`ENV_INSTRUCTIONS.md`** — Step-by-step instructions (from PufferLib docs) for writing envs and using the API.

## Run the env demo

```bash
uv run python -m curly_succotash
```

*(If you see a Gym deprecation message, it comes from PufferLib’s dependencies; this project uses Gymnasium only. A RuntimeWarning about `curly_succotash.env` is avoided by using `python -m curly_succotash` instead of `python -m curly_succotash.env`.)*

## Docs and API

- Full PufferLib docs: **https://puffer.ai/docs.html**
- Env-writing tutorial and API summary: **`ENV_INSTRUCTIONS.md`** in this repo.
