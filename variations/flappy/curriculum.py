"""
Curriculum Flappy: smooth difficulty ramp with NO cliff transitions.

difficulty is a float in [0.0, 1.0] that controls the gap distribution:
  0.00–0.25 : range widens from center-only to full [0.25, 0.75]
  0.25–0.55 : full range + increasing extreme-bias (peaks ~45 %)
  0.55–0.85 : full range + decreasing extreme-bias
  0.85–1.00 : pure uniform [0.25, 0.75]  (matches eval)

Difficulty is stored in a multiprocessing.Value("f") shared between trainer
and env.  Trainer sets it each epoch; env reads it every step and pushes it
into the C envs so auto-resets use the current value.
"""

import gymnasium
import numpy as np
import pufferlib

from . import binding

OBS_DIM = 5


WARMUP_FRAC = 0.10  # hold difficulty at 0.0 for the first 10 % of training


def compute_difficulty(global_step: int, total_timesteps: int) -> float:
    """Warmup hold then linear ramp 0.0 → 1.0.

    First WARMUP_FRAC of training stays at 0.0 (fixed center gaps) so the
    policy can solidify basic flight before the distribution starts shifting.
    """
    total = max(1, total_timesteps)
    warmup_steps = int(total * WARMUP_FRAC)
    if global_step <= warmup_steps:
        return 0.0
    remaining = total - warmup_steps
    return min(1.0, (global_step - warmup_steps) / max(1, remaining))


class FlappyCurriculum(pufferlib.PufferEnv):
    """Flappy with gap difficulty from curriculum_difficulty_value (shared Value)."""

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
        curriculum_difficulty_value=None,
    ):
        self.single_observation_space = gymnasium.spaces.Box(
            low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32
        )
        self.single_action_space = gymnasium.spaces.Discrete(2)
        self.render_mode = render_mode
        self.num_agents = num_envs
        self.log_interval = log_interval
        self.difficulty_value = curriculum_difficulty_value
        if binding is None:
            raise ImportError(
                "Curriculum Flappy C extension not loaded. Build it from the variations/flappy directory: "
                "cd variations/flappy && make"
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

    def reset(self, seed=None):
        if seed is None:
            seed = int(np.random.default_rng().integers(0, 2**31))
        difficulty = float(self.difficulty_value.value) if self.difficulty_value is not None else 0.0
        binding.vec_reset(self.c_envs, seed, difficulty)
        self._tick = 0
        return self.observations, []

    def step(self, actions):
        self._tick += 1
        self.actions[:] = actions
        # Push current difficulty into C envs so auto-resets use it
        difficulty = float(self.difficulty_value.value) if self.difficulty_value is not None else 0.0
        binding.vec_step(self.c_envs, difficulty)
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


def curriculum_env_creator(**kwargs):
    """Picklable env creator. Trainer must pass curriculum_difficulty_value (multiprocessing.Value)."""
    return FlappyCurriculum(
        num_envs=kwargs.get("num_envs", 1),
        width=kwargs.get("width", 400),
        height=kwargs.get("height", 600),
        max_steps=kwargs.get("max_steps", 5000),
        seed=kwargs.get("seed", 0),
        buf=kwargs.get("buf"),
        curriculum_difficulty_value=kwargs.get("curriculum_difficulty_value"),
        **{k: v for k, v in kwargs.items() if k not in (
            "num_envs", "width", "height", "max_steps", "seed", "buf", "curriculum_difficulty_value"
        )},
    )
