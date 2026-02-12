# Env variations

Copy of the current Flappy (C + raylib) environment. Edit files here to try different approaches; the main env stays in `src/flappy_rl/flappy/`.

## Curriculum Flappy

Smooth difficulty ramp (0.0 → 1.0) with NO cliff transitions:

- d 0.00–0.25 : gap range widens from center-only to full [0.25, 0.75]
- d 0.25–0.55 : full range + increasing extreme-bias (peaks ~45 %)
- d 0.55–0.85 : full range + decreasing extreme-bias
- d 0.85–1.00 : pure uniform [0.25, 0.75] — matches eval distribution

Learning rate decays linearly alongside difficulty (3e-4 → 5e-5) to avoid overwriting earlier skills.

**Source of truth:** `variations/flappy/curriculum.py`. Difficulty is computed by `compute_difficulty(global_step, total_timesteps)` and stored in a shared `multiprocessing.Value("f")`. The trainer sets it each epoch; the env reads it at each step and pushes it into the C envs.

1. **Build** (from repo root): `cd variations/flappy && make`
2. **Train** (from repo root): `uv run python -m flappy_rl.train --train.env flappy_curriculum`

Eval stays on the standard random-gap game.

Example run:
`uv run python -m flappy_rl.train --train.env flappy_curriculum --train.total-timesteps 150000000`

Best checkpoint so far (simplified reward/obs + transitional curriculum, 80M-step run):
`experiments/177086914642/model_009765.pt`
New best checkpoint from 150M-step run: `experiments/177087020156/model_018310.pt`
