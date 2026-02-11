# Flappy (C + raylib)

Flappy Bird-style environment implemented in C with raylib rendering. Uses PufferLib Ocean-style binding.

## Build

1. **Install raylib** (required for rendering):
   - macOS: `brew install raylib`
   - Linux: install `libraylib-dev` (or build from [raylib](https://www.raylib.com/)).

2. **Build the C extension** from the project root:
   ```bash
   cd src/flappy_rl/flappy
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

From project root:

- **Train:** `uv run python -m flappy_rl.train --train.env flappy`  
  Optional: `--train.total-timesteps 10000000`, `--train.load-checkpoint path/model.pt`
- **Eval with render:** `uv run python -m flappy_rl.run_eval_flappy --model path/to/model.pt`
- **Eval headless (stats):** `uv run python -m flappy_rl.run_eval_flappy --model path/to/model.pt --episodes 50 --no-render`

## Eval (watch the trained bird)

**Latest eval checkpoint:** `experiments/177077412473/model_002441.pt` (fine-tuned, random gaps, ~655 episode length)

```bash
uv run python -m flappy_rl.run_eval_flappy --model experiments/177077412473/model_002441.pt
```

Or use latest in experiments: `uv run python -m flappy_rl.run_eval_flappy` (no `--model` = picks newest `model_*.pt`)

## Fine-tune from fixed-gap checkpoint (random gaps)

Find latest checkpoint: `ls -t experiments/*/model_*.pt 2>/dev/null | head -1`
Latest checkpoint (fixed-gap run): `experiments/177077321939/model_001220.pt`

```bash
# Rebuild after changing FIXED_GAP_DEBUG in flappy.h
cd src/flappy_rl/flappy && make clean && make && cd ../../..

# Fine-tune from a checkpoint (random gaps)
uv run python -m flappy_rl.train --train.env flappy --train.load-checkpoint experiments/177077321939/model_001220.pt   # fixed-gap
uv run python -m flappy_rl.train --train.env flappy --train.load-checkpoint experiments/177077412473/model_002441.pt   # ~655 ep length
uv run python -m flappy_rl.train --train.env flappy --train.load-checkpoint experiments/177077598031/model_002441.pt   # ~684 ep, smaller LR
uv run python -m flappy_rl.train --train.env flappy --train.load-checkpoint experiments/177078209826/model_002441.pt   # biased fine-tune
```
