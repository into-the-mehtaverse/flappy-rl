# Mr Driller Spec (PufferLib) - v1

This spec is for a **Mr. Driller-inspired** RL environment that is:

- Non-maze-like
- Repeatable and seedable
- Harder than Flappy Bird
- Friendly to PPO + LSTM in PufferLib

This is an inspired environment design, not an exact recreation.

## 1) Design Goals

- **Primary objective:** descend as deep as possible before timeout/death.
- **Core tensions:** oxygen depletion, block collapse, route commitment, short-horizon reflex + medium-horizon planning.
- **Training properties:** stable signals with sparse-first rewards and minimal shaping.
- **PR-readiness:** deterministic core loop, clean metrics, configurable difficulty.

## 2) Gameplay Rules (v1)

### World

- Grid: `width=12`, `height_visible=20`, `height_total=200` (scrolling downward as agent descends).
- Cell types:
  - `EMPTY`
  - `COLOR_A`, `COLOR_B`, `COLOR_C`, `COLOR_D` (diggable solid blocks)
  - `HARD_BLOCK` (not diggable in v1)
  - `OXYGEN_CAPSULE`
- Gravity:
  - Unsupported blocks fall.
  - Falling groups can crush player.

### Agent

- Starts near top center: `(x=width//2, y=2)`.
- Occupies one grid cell.
- Can move and/or dig according to action space.
- Dies if:
  - oxygen reaches 0, or
  - crushed by falling blocks.

### Oxygen

- Starts at `100`.
- Depletes by `oxygen_decay_per_step = 0.12`.
- Picking oxygen capsule restores `+20` (capped at 100).
- Capsules are sparse and seed-controlled.

### Dig/clear logic

- Digging into color blocks removes target cell.
- If connected group size of same color meets threshold (`>=4`), clear entire group and award score/reward.
- `HARD_BLOCK` cannot be dug.

### Episode termination

- Terminal on death (crush/oxygen).
- Truncation on `max_steps` (e.g. `3000`) for runtime control.

## 3) Action Space

Use `Discrete(7)`:

- `0`: NOOP
- `1`: MOVE_LEFT
- `2`: MOVE_RIGHT
- `3`: MOVE_DOWN
- `4`: DIG_LEFT
- `5`: DIG_RIGHT
- `6`: DIG_DOWN

Notes:

- Keep actions discrete and explicit for easier PPO training.
- No jump action in v1.

## 4) Observation Space (vector, not pixels)

Use a compact structured vector first, then expand only if needed.

### v1 observation (~48 dims)

- **Agent state (8)**
  - `x_norm`, `y_norm`
  - `vx_norm`, `vy_norm` (or last movement dir one-hot reduced form)
  - `oxygen_norm`
  - `is_falling`, `is_grounded`, `recent_damage`

- **Progress/context (6)**
  - `depth_norm`
  - `max_depth_norm`
  - `steps_remaining_norm`
  - `time_since_last_oxygen_norm`
  - `seed_phase_sin`, `seed_phase_cos` (optional)

- **Local occupancy window (5x5 around player = 25)**
  - Encode each cell into normalized scalar IDs:
    - empty=0.0
    - color blocks mapped into (0.2-0.5)
    - hard block=0.7
    - oxygen capsule=1.0
  - Flatten in fixed order (row-major around player).

- **Nearest-object features (9)**
  - nearest oxygen `dx`, `dy`, `dist_norm`
  - nearest falling mass `dx`, `dy`, `dist_norm`
  - nearest hard-block column offset
  - left_clearance, right_clearance

### Why this shape

- Local window captures tactical dig/fall interactions.
- Global features preserve long-horizon survival/progress context.
- Works with MLP; improves materially with LSTM.

## 5) Reward Function

Start sparse-first and only add tiny shaping:

- `+1.0` per new depth row reached (first time only).
- `+0.2` for valid color-group clear (`>=4`).
- `+0.15` for oxygen capsule pickup.
- `-1.0` on death.
- Optional tiny survival penalty: `-0.001` per step to reduce stalling.

Avoid over-shaping early. If policy camps, use small anti-stall penalty or progress timeout.

## 6) Difficulty / Curriculum

Use fixed phases by environment parameterization (not reward tricks):

- **Phase 0 (easy):**
  - 3 colors
  - low hard-block density
  - high oxygen spawn rate
  - low gravity update frequency

- **Phase 1 (medium):**
  - 4 colors
  - medium hard-block density
  - normal oxygen spawn
  - normal gravity

- **Phase 2 (hard):**
  - 4 colors + denser hard blocks
  - lower oxygen spawn
  - faster falling groups
  - occasional dead-end pockets

- **Phase 3 (target):**
  - final distribution matching intended benchmark.

Training schedule suggestion:

- 10% phase 0
- 25% phase 1
- 30% phase 2
- 35% phase 3

## 7) Repeatability and Seeds

- Deterministic generation from episode seed:
  - block colors
  - hard block placement
  - oxygen capsule placement
- Keep a fixed seed set for evaluation (e.g. 100 seeds).
- Separate train/eval seed ranges.

## 8) Baseline Metrics

Track per-episode:

- `episode_return`
- `depth_reached`
- `max_depth_reached`
- `group_clears`
- `oxygen_pickups`
- `death_by_crush` / `death_by_oxygen`
- `steps_survived`

Primary benchmark metric:

- `mean_depth_reached` over fixed eval seed set.

Secondary:

- success rate above depth thresholds (e.g. >=50, >=100, >=150).

## 9) PufferLib Training Defaults (starting point)

- PPO with LSTM (hidden size 128)
- `gamma=0.99`
- `gae_lambda=0.95`
- clip range `0.2`
- entropy coef small but non-zero
- learning-rate decay over run
- medium rollout horizon (enough for delayed oxygen/depth effects)

Keep this simple first. Tune only after verifying env sanity and learning curve shape.

## 10) Env Sanity Checklist (before long runs)

- No impossible starts (spawn not enclosed by hard blocks).
- Oxygen capsules reachable from spawn in early depth bands.
- Dig + gravity logic cannot create silent invalid states.
- Collision/death handling is deterministic.
- Reward accounting matches events exactly once.
- Reset returns identical initial state for identical seed.

## 11) Suggested v1 Milestones

- **M1:** single-map deterministic prototype, no rendering polish.
- **M2:** seed-based generator + metrics + headless training.
- **M3:** curriculum phases + stable convergence.
- **M4:** rendering polish + docs + reproducible eval script for PR.

## 12) Scope guardrails

Do not add these in v1:

- Enemies
- Bomb mechanics
- Complex combo multipliers
- Pixel observation mode

Ship a stable core first, then iterate.
