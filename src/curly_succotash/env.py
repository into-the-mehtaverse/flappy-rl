"""
First RL env for PufferLib.

Two options:
1. Gymnasium env wrapped with GymnasiumPufferEnv (easiest).
2. Native PufferEnv with in-place buf updates (vector-friendly, faster when vectorized).
"""

import gymnasium
import pufferlib
import pufferlib.emulation


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
        self.observations[:] = self.observation_space.sample()
        infos = [{}]
        return self.observations, self.rewards, self.terminals, self.truncations, infos


# Run demo via: uv run python -m curly_succotash
# (Running this file as __main__ can trigger a double-import warning.)
