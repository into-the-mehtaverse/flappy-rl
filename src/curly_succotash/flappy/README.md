# Flappy (C + raylib)

Flappy Bird-style environment implemented in C with raylib rendering. Uses PufferLib Ocean-style binding.

## Build

1. **Install raylib** (required for rendering):
   - macOS: `brew install raylib`
   - Linux: install `libraylib-dev` (or build from [raylib](https://www.raylib.com/)).

2. **Build the C extension** from the project root:
   ```bash
   cd src/curly_succotash/flappy
   make PYTHON=../../../.venv/bin/python
   ```
   Or from the `flappy` directory with a custom Python:
   ```bash
   make PYTHON=/path/to/your/venv/bin/python
   ```

3. If raylib is not in `/opt/homebrew` or `/usr/local`, set:
   ```bash
   RAYLIB_INC="-I/path/to/raylib/include" RAYLIB_LIB="-L/path/to/raylib/lib -lraylib" make PYTHON=...
   ```

## Assets

Place `bird.png` and `pipe.png` in `resources/flappy/` (relative to the process CWD when running). The game uses them for rendering; run from the project root so `resources/flappy/` is found.

## Train / eval

- Train (from project root): `uv run python -m curly_succotash.train --train.env flappy`
- Eval with render: run a small script that creates `Flappy(num_envs=1)`, calls `reset()`, then in a loop `step(actions)` and `render()`.


uv run python -m curly_succotash.train --train.env flappy
uv run python -m curly_succotash.run_eval_flappy

uv run python -m curly_succotash.train --train.env flappy --train.total-timesteps 50000000
