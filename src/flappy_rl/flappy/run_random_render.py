"""
Run Flappy with a random policy and rendering (like PufferLib eval).
Run from repo root so C code finds resources/flappy/:
  uv run python -m flappy_rl.flappy.run_random_render
Press ESC in the game window to exit.
"""
import time
import numpy as np
import pufferlib.vector

from flappy_rl.flappy import flappy_env_creator

FPS = 60


def main():
    vecenv = pufferlib.vector.make(
        flappy_env_creator,
        env_kwargs={"num_envs": 1, "width": 400, "height": 600},
        backend=pufferlib.vector.Serial,
        num_envs=1,
        seed=42,
    )
    driver = vecenv.driver_env
    obs, info = vecenv.reset(seed=42)
    while True:
        driver.render()
        action = np.array([driver.single_action_space.sample()])
        obs, rewards, terms, truncs, info = vecenv.step(action)
        time.sleep(1 / FPS)


if __name__ == "__main__":
    main()
