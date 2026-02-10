"""Flappy Bird C + raylib env and binding. Build the C extension with make in this dir first."""

try:
    from curly_succotash.flappy import binding
except ImportError:
    binding = None  # Build with: make PYTHON=../../../.venv/bin/python in this directory

from curly_succotash.flappy.flappy import Flappy, flappy_env_creator

__all__ = ["Flappy", "flappy_env_creator", "binding"]
