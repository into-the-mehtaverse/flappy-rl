"""Flappy Bird-style env: C + raylib, PufferLib Ocean binding."""

import gymnasium
import numpy as np
import pufferlib

from flappy_rl.flappy import binding

OBS_DIM = 9


class Flappy(pufferlib.PufferEnv):
    """Single-agent Flappy. Actions: 0 = no flap, 1 = flap. Obs: bird y, vy, next pipe dist/gap, signed gap error (above/below)."""

    def __init__(
        self,
        num_envs=1,
        render_mode=None,
        log_interval=128,
        width=400,
        height=600,
        max_steps=5000,
        buf=None,
        seed=0,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        if binding is None:
            raise ImportError(
                "Flappy C extension not loaded. Build it from the flappy directory: "
                "cd src/flappy_rl/flappy && make"
            )
        super().__init__(buf)
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            width=width,
            height=height,
            max_steps=max_steps,
        )
        self._tick = 0

    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self._tick = 0
        return self.observations, []

    def step(self, actions):
        self._tick += 1
        self.actions[:] = actions
        binding.vec_step(self.c_envs)
        info = []
        if self._tick % self.log_interval == 0:
            log = binding.vec_log(self.c_envs)
            if log:
                info.append(log)
        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)


def flappy_env_creator(**kwargs):
    """Picklable env creator for pufferlib.vector.make. Pass num_envs, width, height, etc. via env_kwargs."""
    return Flappy(
        num_envs=kwargs.get("num_envs", 1),
        width=kwargs.get("width", 400),
        height=kwargs.get("height", 600),
        max_steps=kwargs.get("max_steps", 5000),
        seed=kwargs.get("seed", 0),
        buf=kwargs.get("buf"),
        **{k: v for k, v in kwargs.items() if k not in ("num_envs", "width", "height", "max_steps", "seed", "buf")},
    )
