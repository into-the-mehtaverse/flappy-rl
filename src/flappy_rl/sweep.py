"""
Sweep over learning rate and clip_coef to see when the policy becomes deterministic (entropy → 0).

Usage:
  uv run python -m flappy_rl.sweep
  uv run python -m flappy_rl.sweep --sweep.steps 1000000   # longer per run
"""

import copy
import warnings

warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.*")

import torch
import pufferlib
import pufferlib.vector
from pufferlib import pufferl

from flappy_rl.env import flappy_grid_env_creator
from flappy_rl.train import FlappyGridPolicy


# Grid: at which (lr, clip_coef) does entropy go to ~0?
LEARNING_RATES = [1e-3, 3e-3, 0.01, 0.03]
CLIP_COEFS = [0.2, 0.35, 0.5, 0.7]

# Steps per run (shorter so sweep finishes in reasonable time)
SWEEP_TIMESTEPS = 500_000


def run_one(args, learning_rate, clip_coef):
    """Run training with given lr and clip_coef; return final entropy and epoch."""
    args = copy.deepcopy(args)
    args["train"]["learning_rate"] = learning_rate
    args["train"]["clip_coef"] = clip_coef
    args["train"]["total_timesteps"] = SWEEP_TIMESTEPS

    vec_kwargs = dict(args["vec"])
    if vec_kwargs.get("num_workers") == "auto":
        vec_kwargs["num_workers"] = 2
    if vec_kwargs.get("num_envs") in (None, "auto") or vec_kwargs.get("num_envs", 0) < 128:
        vec_kwargs["num_envs"] = 128

    vecenv = pufferlib.vector.make(flappy_grid_env_creator, **vec_kwargs)
    policy = FlappyGridPolicy(vecenv.driver_env).to(args["train"]["device"])
    trainer = pufferl.PuffeRL(args["train"], vecenv, policy)

    last_entropy = None
    last_epoch = 0
    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        logs = trainer.train()
        if logs is not None:
            last_entropy = logs.get("losses/entropy")
            last_epoch = trainer.epoch
        trainer.print_dashboard()

    trainer.close()
    return last_entropy, last_epoch


def main():
    args = pufferl.load_config("default")
    args["train"]["env"] = "flappy_grid"
    args["train"]["optimizer"] = "adam"
    if not torch.cuda.is_available():
        args["train"]["device"] = "cpu"

    results = []
    for lr in LEARNING_RATES:
        for clip in CLIP_COEFS:
            print(f"\n--- lr={lr}, clip_coef={clip} ---")
            ent, epoch = run_one(args, lr, clip)
            results.append((lr, clip, ent, epoch))
            deterministic = ent is not None and ent < 0.01
            print(f"  -> entropy={ent:.4f} (epoch {epoch})  {'DETERMINISTIC' if deterministic else ''}")

    # Summary table
    print("\n" + "=" * 60)
    print("Sweep summary (final entropy per lr × clip_coef)")
    print("=" * 60)
    print(f"{'lr':<10} {'clip_coef':<10} {'entropy':<12} {'epoch':<8}  deterministic")
    print("-" * 60)
    for lr, clip, ent, epoch in results:
        ent_str = f"{ent:.4f}" if ent is not None else "N/A"
        det = "yes" if (ent is not None and ent < 0.01) else "no"
        print(f"{lr:<10} {clip:<10} {ent_str:<12} {epoch:<8}  {det}")
    print("=" * 60)


if __name__ == "__main__":
    main()
