# Writing an RL env for PufferLib

Instructions below are based on the [PufferLib docs](https://puffer.ai/docs.html). This repo is set up as a **uv** project with `pufferlib` and `gymnasium`; run **`uv sync`** in the repo root to install (from your terminal; the IDE may not have write access to uv's cache).

---

## Two ways to provide an env

### 1. Gymnasium (or PettingZoo) + emulation wrapper — easiest

- Implement a normal **Gymnasium** (or **PettingZoo**) env.
- Wrap it in one line with PufferLib's emulation layer:
  - **Single-agent:** `pufferlib.emulation.GymnasiumPufferEnv(your_gymnasium_env)`
  - **Multi-agent:** `pufferlib.emulation.PettingZooPufferEnv(your_pettingzoo_env)`
- PufferLib flattens observation/action spaces and handles padding so the same vectorization and training code works. You don't manage batching or structured buffers yourself.

**Example in this repo:** `src/flappy_rl/env.py` — `SampleGymnasiumEnv` + `make_gymnasium_env()`.

### 2. Native PufferEnv — vector-friendly, best throughput

- Subclass **`pufferlib.PufferEnv`**.
- The constructor receives an optional **`buf`** (buffer). When vectorized, PufferLib passes **slices of shared memory**; your env writes **in-place** into `self.observations`, `self.rewards`, `self.terminals`, `self.truncations` (and reads actions from the step argument).
- Define:
  - **`single_observation_space`** and **`single_action_space`** (Gymnasium spaces for one agent).
  - **`num_agents`** (e.g. 1 for single-agent).
- Implement **`reset(seed=0)`** → return `(observations, infos)` and **`step(action)`** → return `(observations, rewards, terminals, truncations, infos)`. All arrays are batched; use in-place updates (e.g. `self.observations[:] = ...`) and avoid allocating new arrays each step.
- **Important:** Calling `step` again overwrites the previous step's data; the trainer/vectorizer expects this.

**Example in this repo:** `src/flappy_rl/env.py` — `SamplePufferEnv`.

For a **pure Python** reference that stays close to this API (single-agent, in-place buf), see PufferLib's [pysquared](https://github.com/PufferAI/PufferLib/blob/3.0/pufferlib/ocean/pysquared/pysquared.py). For **C + Python binding** and much higher steps/second, see the [Squared](https://github.com/PufferAI/PufferLib/tree/3.0/pufferlib/ocean/squared) tutorial in the docs.

---

## Checklist for native PufferEnv (from PufferLib docs)

- **Spaces and dtypes:** Python `observation_space` / `action_space` (and buffer shapes/dtypes) must match what you write in C/Python (zero or wrong obs/actions is a common bug).
- **Resets:** Env should handle resets internally; for envs that don't "end," consider respawning agents (e.g. after 500 steps without reward).
- **Zeroing:** At the start of each step, zero `rewards`, `terminals` (and any observation slots you don't fully overwrite). Otherwise values from the previous step leak through.
- **Scale:** Observations and rewards roughly in **[-1, 1]** tend to behave better.
- **Binding (C envs):** If you add a C backend, the binding must pass the same init args as your C code and call your C init.

**Detailed explanation of each item:** see [env-checklist.md](env-checklist.md).

---

## How to use the API and what it's for

### Emulation (`pufferlib.emulation`)

- **Purpose:** Let existing Gymnasium/PettingZoo envs work with PufferLib's vectorization and training without rewriting them.
- **What it does:** Wraps your env so that:
  - Observation and action spaces are **flattened** for storage and batching.
  - Unflattening happens at the policy (e.g. PyTorch) so you can still use structured spaces there.
  - Multi-agent envs are padded to a fixed number of agents where needed.
- **When to use:** Whenever you have or want a standard Gymnasium/PettingZoo env and want to train with PufferLib (e.g. `puffer train`, PuffeRL, or custom training scripts).

### Vectorization (`pufferlib.vector`)

- **Purpose:** Run many envs in parallel (synchronous or asynchronous) for fast data collection.
- **API (conceptually):**
  - **`pufferlib.vector.make(env_creator_or_creators, env_args=None, env_kwargs=None, backend=PufferEnv, num_envs=1, seed=0, **kwargs)`**  
    Builds a vectorized env. You pass an env **creator** (e.g. a function that returns your PufferEnv or wrapped Gymnasium env).
  - **Sync:** `vecenv.reset()`, `vecenv.step(actions)`.
  - **Async (EnvPool-style):** `vecenv.async_reset(seed=None)`, `vecenv.send(actions)`, `vecenv.recv()` — more envs than batch size, return batches as soon as ready.
- **When to use:** Always when training; PufferLib's multiprocessing backend is optimized (shared memory, fewer copies, optional async). Use **Serial** backend for debugging.

### PuffeRL (training)

- **Purpose:** Train policies (e.g. PPO-style) at high steps/second with a single, readable training script.
- **API (conceptually):**
  - **`pufferl.PuffeRL(config, vecenv, policy)`** — trainer object.
  - **`evaluate()`** — collect a batch of env interactions.
  - **`train()`** — update on one batch.
  - **`mean_and_log()`** — aggregate logs, optional WandB/Neptune.
  - **`save_checkpoint()`**, **`close()`**, **`print_dashboard()`**.
  - **`pufferl.train(env_name, args=None, vecenv=None, policy=None, logger=None)`** — high-level train entry (loads config by env name).
  - **`pufferl.load_config(env_name)`**, **`pufferl.load_env(env_name, args)`**, **`pufferl.load_policy(args, vecenv)`** — load defaults and env/policy for that env.
- **When to use:** For training agents with minimal boilerplate; use **custom policy** and/or **custom env** by passing your own `vecenv` and `policy` (see [pufferl.py example](https://github.com/PufferAI/PufferLib/blob/3.0/examples/pufferl.py)).

### CLI

- **`puffer train <env_name> [OPTIONS]`** — train (e.g. `puffer train puffer_breakout`).
- **`puffer eval <env_name> [OPTIONS]`** — evaluate/render (e.g. `--load-model-path path/model.pt` or `latest`).
- **`puffer sweep <env_name> [OPTIONS]`** — hyperparameter sweeps (e.g. with `--wandb` or `--neptune`).
- Override config: `--train.device cuda`, `--train.learning-rate 0.001`, `--env.*`, `--vec.*`, etc.

To expose your **own** env to this CLI (like Ocean envs), you'd register it and add a config (see PufferLib's Ocean "template" and `pufferlib/config/ocean` in the repo). For a first env, using **`pufferlib.vector.make`** and **`pufferl.PuffeRL`** (or **`pufferl.train`** with a custom env name/loader) in your own script is enough.

---

## Quick reference: flow

1. **Implement env** — either Gymnasium + `GymnasiumPufferEnv`, or native `pufferlib.PufferEnv` with in-place buf updates.
2. **Vectorize** — `vecenv = pufferlib.vector.make(your_env_creator, num_envs=..., backend=...)`.
3. **Train** — use **PuffeRL** (e.g. `pufferl.PuffeRL(config, vecenv, policy)`) or **`pufferl.train`** with your env/policy loaded by name or by function.
4. **Eval / sweep** — use CLI or the same `vecenv`/policy in a small script.

You can run the demo in this repo with:

```bash
uv run python -m flappy_rl
```

(or `uv run python src/flappy_rl/env.py` if not installed in editable mode).
