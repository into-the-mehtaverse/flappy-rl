"""
First RL env for PufferLib.

Envs:
- FlappyGridEnv: 2-row grid (floor/ceiling), up/down actions, wall obs, -1 reward for hitting ceiling.
- SampleGymnasiumEnv / SamplePufferEnv: minimal mock envs for API demo.
"""

import numpy as np
import gymnasium
import pufferlib
import pufferlib.emulation


# ---------------------------------------------------------------------------
# Flappy Grid: 2-row strip, up/down, wall obs, -1 for hitting ceiling
# ---------------------------------------------------------------------------

class FlappyGridEnv(pufferlib.PufferEnv):
    """
    Stripped-down Flappy Bird on a 2-block-tall grid.
    - Rows: 0 = floor, 1 = ceiling.
    - Actions: 0 = down, 1 = up.
    - Observation: (position, wall_roof, wall_floor) in [-1, 1]. position: -1 = floor, +1 = ceiling.
    - Reward: -1 for hitting the ceiling (or floor), 0 otherwise.
    """

    MAX_STEPS = 2000

    def __init__(self, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.num_agents = 1
        super().__init__(buf)
        self._rng = np.random.default_rng(seed)
        self._y = 0
        self._step_count = 0
        self._wall_roof = 0
        self._wall_floor = 0

    def _obs(self):
        # (position, wall_roof, wall_floor); scale 0/1 -> -1/1 for checklist
        position_scaled = 2.0 * self._y - 1.0  # 0 -> -1 (floor), 1 -> +1 (ceiling)
        return np.array(
            [
                position_scaled,
                2.0 * self._wall_roof - 1.0,
                2.0 * self._wall_floor - 1.0,
            ],
            dtype=np.float32,
        )

    def _write_obs(self):
        # Batched observations shape (num_agents, obs_dim) = (1, 3)
        self.observations[0] = self._obs()

    def _sample_walls(self):
        # One gap: either wall at roof or at floor (not both)
        if self._rng.random() < 0.5:
            self._wall_roof, self._wall_floor = 1, 0
        else:
            self._wall_roof, self._wall_floor = 0, 1

    def reset(self, seed=0):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._y = self._rng.integers(0, 2)
        self._step_count = 0
        self._sample_walls()
        self._write_obs()
        return self.observations, []

    def step(self, action):
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False

        # Flatten action for single-agent batch shape (1,)
        a = int(action.flat[0]) if hasattr(action, "flat") else int(action)

        hit_ceiling = self._y == 1 and a == 1
        hit_floor = self._y == 0 and a == 0
        if hit_ceiling or hit_floor:
            self.rewards[:] = -1.0
            self.terminals[:] = True
            self._write_obs()
            return self.observations, self.rewards, self.terminals, self.truncations, [{}]

        self._y = 1 if a == 1 else 0
        self._step_count += 1
        self._sample_walls()
        # Small reward per step survived so "stay alive" has a positive signal (kept in [-1,1])
        self.rewards[:] = 0.01
        if self._step_count >= self.MAX_STEPS:
            self.truncations[:] = True
        self._write_obs()
        return self.observations, self.rewards, self.terminals, self.truncations, [{}]

    def close(self):
        """Required by PufferLib vectorizer (driver_env.close()). No resources to release."""
        pass


def flappy_grid_env_creator(buf=None, seed=0):
    """Env creator for pufferlib.vector.make."""
    return FlappyGridEnv(buf=buf, seed=seed)


# ---------------------------------------------------------------------------
# Option 1: Gymnasium env — write a normal Gymnasium env, then wrap it
# ---------------------------------------------------------------------------

class SampleGymnasiumEnv(gymnasium.Env):
    """Minimal Gymnasium env. Wrap with pufferlib.emulation.GymnasiumPufferEnv to use with PufferLib."""

    def __init__(self):
        self.observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.action_space = gymnasium.spaces.Discrete(2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.observation_space.sample()
        return obs, {}

    def step(self, action):
        obs = self.observation_space.sample()
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info


def make_gymnasium_env():
    """Create a PufferLib-compatible env from our Gymnasium env (1-line wrapper)."""
    base = SampleGymnasiumEnv()
    return pufferlib.emulation.GymnasiumPufferEnv(base)


# ---------------------------------------------------------------------------
# Option 2: Native PufferEnv — observations/rewards/terminals from buf, in-place updates
# ---------------------------------------------------------------------------

class SamplePufferEnv(pufferlib.PufferEnv):
    """
    Minimal native PufferEnv. Uses a shared buffer (buf): observations, rewards,
    terminals, truncations are written in-place. Required for fast vectorization
    (vectorization passes slices of shared memory).
    """

    def __init__(self, buf=None, seed=0):
        self.single_observation_space = gymnasium.spaces.Box(low=-1.0, high=1.0, shape=(2,))
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.num_agents = 1
        super().__init__(buf)

    def reset(self, seed=0):
        self.observations[:] = self.observation_space.sample()
        return self.observations, []

    def step(self, action):
        # Tutorial: zero rewards/terminals/truncations at start so they don't retain previous values
        self.rewards[:] = 0
        self.terminals[:] = False
        self.truncations[:] = False
        self.observations[:] = self.observation_space.sample()
        infos = [{}]
        return self.observations, self.rewards, self.terminals, self.truncations, infos


# Run demo via: uv run python -m flappy_rl
# (Running this file as __main__ can trigger a double-import warning.)
