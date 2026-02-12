"""
Batch-evaluate the last N checkpoints from a run and report the best one.

Uses the curriculum variation env at a fixed difficulty (default 1.0).

Examples (from repo root):
  uv run python -m variations.flappy.eval_last_checkpoints
  uv run python -m variations.flappy.eval_last_checkpoints --run-id 177087020156 --last 5 --episodes 50
  uv run python -m variations.flappy.eval_last_checkpoints --difficulty 0.7
"""

import argparse
import glob
import multiprocessing
import os
import re

import numpy as np
import pufferlib.pytorch
import pufferlib.vector
import torch

from flappy_rl.train import FlappyGridPolicy
from variations.flappy import curriculum_env_creator


def find_latest_run(experiments_root: str) -> str | None:
    runs = [p for p in glob.glob(os.path.join(experiments_root, "*")) if os.path.isdir(p)]
    if not runs:
        return None
    runs.sort(key=os.path.getmtime, reverse=True)
    return os.path.basename(runs[0])


def checkpoint_step(path: str) -> int:
    name = os.path.basename(path)
    m = re.match(r"model_(\d+)\.pt$", name)
    return int(m.group(1)) if m else -1


def run_episode(vecenv, policy, device, seed: int):
    obs, _ = vecenv.reset(seed=seed)
    steps = 0
    pipes_passed = 0
    with torch.no_grad():
        while True:
            ob = torch.as_tensor(obs).to(device)
            logits, _ = policy.forward_eval(ob)
            action = logits.argmax(dim=-1).cpu().numpy().reshape(vecenv.action_space.shape)
            obs, rewards, terms, truncs, _ = vecenv.step(action)
            if float(rewards.flat[0]) >= 1.0:
                pipes_passed += 1
            steps += 1
            if terms.any() or truncs.any():
                break
    return pipes_passed, steps


def eval_checkpoint(vecenv, policy, model_path: str, episodes: int, seed: int, device: str):
    state_dict = torch.load(model_path, map_location=device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    pipes = []
    lengths = []
    for ep in range(episodes):
        p, l = run_episode(vecenv, policy, device, seed + ep)
        pipes.append(p)
        lengths.append(l)

    pipes_np = np.array(pipes)
    lengths_np = np.array(lengths)
    return {
        "model_path": model_path,
        "mean_pipes": float(pipes_np.mean()),
        "std_pipes": float(pipes_np.std()),
        "min_pipes": int(pipes_np.min()),
        "max_pipes": int(pipes_np.max()),
        "mean_length": float(lengths_np.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate last N checkpoints and pick best")
    parser.add_argument("--run-id", type=str, default=None, help="Experiment run id under experiments/")
    parser.add_argument("--last", type=int, default=5, help="How many latest checkpoints to evaluate")
    parser.add_argument("--episodes", type=int, default=50, help="Episodes per checkpoint")
    parser.add_argument("--difficulty", type=float, default=1.0, help="Eval difficulty in [0,1]")
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    experiments_root = "experiments"
    run_id = args.run_id or find_latest_run(experiments_root)
    if run_id is None:
        raise SystemExit("No runs found in experiments/")

    run_dir = os.path.join(experiments_root, run_id)
    checkpoints = glob.glob(os.path.join(run_dir, "model_*.pt"))
    checkpoints = [p for p in checkpoints if checkpoint_step(p) >= 0]
    checkpoints.sort(key=checkpoint_step)
    if not checkpoints:
        raise SystemExit(f"No checkpoints found in {run_dir}")

    selected = checkpoints[-max(1, args.last):]

    difficulty_value = multiprocessing.Value("f", float(args.difficulty))
    vecenv = pufferlib.vector.make(
        curriculum_env_creator,
        env_kwargs={
            "num_envs": 1,
            "width": 400,
            "height": 600,
            "curriculum_difficulty_value": difficulty_value,
        },
        backend=pufferlib.vector.Serial,
        num_envs=1,
        seed=args.seed,
    )
    policy = FlappyGridPolicy(vecenv.driver_env).to(args.device)

    print(f"Run: {run_id}")
    print(f"Eval difficulty: {args.difficulty:.2f}")
    print(f"Episodes/checkpoint: {args.episodes}")
    print(f"Evaluating {len(selected)} checkpoints:")
    for p in selected:
        print(f"  - {p}")
    print("")

    results = []
    for ckpt in selected:
        res = eval_checkpoint(
            vecenv=vecenv,
            policy=policy,
            model_path=ckpt,
            episodes=args.episodes,
            seed=args.seed,
            device=args.device,
        )
        results.append(res)
        print(
            f"{os.path.basename(ckpt)} | pipes mean {res['mean_pipes']:.2f} "
            f"(std {res['std_pipes']:.2f}, min {res['min_pipes']}, max {res['max_pipes']}) "
            f"| len mean {res['mean_length']:.1f}"
        )

    vecenv.close()

    best = max(results, key=lambda r: (r["mean_pipes"], r["mean_length"]))
    print("\nBest checkpoint:")
    print(f"  {best['model_path']}")
    print(
        f"  mean pipes {best['mean_pipes']:.2f}, std {best['std_pipes']:.2f}, "
        f"min {best['min_pipes']}, max {best['max_pipes']}, mean len {best['mean_length']:.1f}"
    )


if __name__ == "__main__":
    main()
