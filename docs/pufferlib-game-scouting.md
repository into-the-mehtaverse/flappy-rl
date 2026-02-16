# PufferLib Game Inventory + New Env Ideas

This note summarizes:

1. What is already present in your local PufferLib install (`resources/` and `pufferlib/ocean/`).
2. What that implies for PR opportunities.
3. Five candidate arcade-style environments that are more complex than Flappy Bird.

## Context from this repo

From `README.md` and `docs/blog2.md`, your Flappy project converged best when you:

- Kept rewards sparse (`+1` pass, `-1` death).
- Simplified observations to core state.
- Added memory (LSTM) for temporal dependencies.

That profile is a strong fit for building a harder arcade task with:

- Longer horizons,
- Delayed credit assignment,
- More strategic action timing than pure reflexes.

## Full game/module list discovered

### `resources/` top-level directories

Source: `resources/` (symlinked to `.../.venv/lib/python3.12/site-packages/pufferlib/resources`)

- `blastar`
- `breakout`
- `cartpole`
- `connect4`
- `convert`
- `cpr`
- `drone`
- `enduro`
- `flappy`
- `freeway`
- `go`
- `gpudrive`
- `impulse_wars`
- `moba`
- `nmmo3`
- `pacman`
- `pong`
- `robocode`
- `rware`
- `school`
- `shared`
- `snake`
- `terraform`
- `tetris`
- `tower_climb`
- `trash_pickup`
- `tripletriad`

### Parent folder above `target.h` (`pufferlib/ocean/`) directories

Source: `.venv/lib/python3.12/site-packages/pufferlib/ocean/`

- `blastar`
- `boids`
- `breakout`
- `cartpole`
- `checkers`
- `connect4`
- `convert`
- `cpr`
- `drone`
- `enduro`
- `freeway`
- `go`
- `gpudrive`
- `grid`
- `impulse_wars`
- `moba`
- `nmmo3`
- `pacman`
- `pong`
- `pysquared`
- `robocode`
- `rocket_lander`
- `rware`
- `school`
- `snake`
- `squared`
- `tactical`
- `target`
- `tcg`
- `template`
- `terraform`
- `tetris`
- `tower_climb`
- `trash_pickup`
- `tripletriad`

## Combined inventory view

### Present in both places

- `blastar`
- `breakout`
- `cartpole`
- `connect4`
- `convert`
- `cpr`
- `drone`
- `enduro`
- `freeway`
- `go`
- `gpudrive`
- `impulse_wars`
- `moba`
- `nmmo3`
- `pacman`
- `pong`
- `robocode`
- `rware`
- `school`
- `snake`
- `terraform`
- `tetris`
- `tower_climb`
- `trash_pickup`
- `tripletriad`

### `resources/` only

- `flappy`
- `shared`

### `ocean/` only

- `boids`
- `checkers`
- `grid`
- `pysquared`
- `rocket_lander`
- `squared`
- `tactical`
- `target`
- `tcg`
- `template`

## 5 recommendations for your next PufferLib environment

All options below are not in your discovered inventory and are meaningfully harder than Flappy.

### 1) `bomberman_lite` (single-agent survival + planning)

Why it is a strong next step:

- Delayed reward: bomb placement pays off seconds later.
- Spatiotemporal planning: avoid self-traps while routing enemies.
- Easily scalable curriculum: map size, enemy count, blast radius, destructible density.

PR angle: broad RL interest, arcade recognizable, strong benchmark value.

### 2) `asteroids_plus` (inertia + shooting + target prioritization)

Why it is a strong next step:

- Continuous-feeling dynamics from thrust/inertia, unlike Flappy's simple vertical impulse.
- Multi-objective behavior: survive, rotate/aim, split asteroids efficiently.
- Hard partial observability at higher speeds and clutter.

PR angle: classic arcade control challenge with richer physics and action semantics.

### 3) `donkey_kong_classic` (platforming with moving hazards and ladders)

Why it is a strong next step:

- Long-horizon route planning with timing windows.
- Dynamic obstacle patterns (rolling hazards, moving enemies).
- Good testbed for memory + risk-aware navigation.

PR angle: very interpretable progression and failure modes for RL research demos.

### 4) `qix_territory` (area-capture under pursuit pressure)

Why it is a strong next step:

- Sparse strategic reward (captured area) with high-risk action segments.
- Non-trivial geometry and boundary interactions.
- Requires balancing greed vs safety under adversarial motion.

PR angle: unusual mechanics compared to standard control tasks; likely novel in RL env suites.

### 5) `digdug_like` (maze digging + enemy state transitions)

Why it is a strong next step:

- Hybrid tactical game loop: path carving, chase/escape, enemy manipulation.
- Rich state machine behavior for enemies -> harder policy learning than Flappy timing.
- Natural difficulty knobs: enemy types, speed, map density, oxygen/time pressure.

PR angle: deeper interaction mechanics without requiring huge rendering/physics complexity.

## Practical shortlist (if you want one to start now)

- Fastest to prototype with high novelty: `bomberman_lite`
- Best for "physics + strategy" showcase: `asteroids_plus`
- Best for benchmark-style curriculum research: `donkey_kong_classic`
