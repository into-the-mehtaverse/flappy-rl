"""Flappy RL: Flappy Bird env for PufferLib."""

import warnings

# Suppress Gym deprecation message from pufferlib's dependencies (we only use Gymnasium)
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")

from flappy_rl.env import (
    FlappyGridEnv,
    SamplePufferEnv,
    flappy_grid_env_creator,
    make_gymnasium_env,
)

__all__ = [
    "FlappyGridEnv",
    "SamplePufferEnv",
    "flappy_grid_env_creator",
    "make_gymnasium_env",
]
