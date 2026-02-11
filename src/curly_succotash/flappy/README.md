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

uv run python -m curly_succotash.train --train.env flappy --train.total-timesteps 10000000

## Fine-tune from fixed-gap checkpoint (random gaps)

Find latest checkpoint: `ls -t experiments/*/model_*.pt 2>/dev/null | head -1`
Latest checkpoint (fixed-gap run): `experiments/177077321939/model_001220.pt`

```bash
# Rebuild after changing FIXED_GAP_DEBUG in flappy.h
cd src/curly_succotash/flappy && make clean && make && cd ../../..

# Fine-tune from fixed-gap checkpoint (random gaps)
uv run python -m curly_succotash.train --train.env flappy --train.load-checkpoint experiments/177077321939/model_001220.pt
```
