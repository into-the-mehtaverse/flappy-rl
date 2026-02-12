"""
Eval a curriculum-trained Flappy policy. Uses the curriculum C binding at
difficulty=1.0 (pure uniform [0.25, 0.75] — same distribution as the standard
eval environment).

Run from repo root:

  uv run python -m variations.flappy.run_eval
  uv run python -m variations.flappy.run_eval --model experiments/<run_id>/model_009765.pt
  uv run python -m variations.flappy.run_eval --episodes 50 --no-render
  uv run python -m variations.flappy.run_eval --difficulty 0.5   # test at mid-curriculum

Actions are always argmax (greedy). Press ESC in the game window to exit when rendering.
"""
import argparse
import glob
import multiprocessing
import os
import time

import numpy as np
import torch
import pufferlib.vector
import pufferlib.pytorch

from variations.flappy import curriculum_env_creator
from flappy_rl.train import FlappyGridPolicy

FPS = 60


def find_latest_checkpoint():
    pattern = "experiments/*/model_*.pt"
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getctime)


def run_episode(vecenv, policy, device, seed=None):
    """Run one episode; return (pipes_passed, episode_length). Uses argmax (greedy)."""
    obs, _ = vecenv.reset(seed=seed)
    steps = 0
    pipes_passed = 0
    with torch.no_grad():
        while True:
            ob = torch.as_tensor(obs).to(device)
            logits, _ = policy.forward_eval(ob)
            action = logits.argmax(dim=-1).cpu().numpy().reshape(vecenv.action_space.shape)
            obs, rewards, terms, truncs, info = vecenv.step(action)
            r = float(rewards.flat[0])
            if r >= 1.0:
                pipes_passed += 1
            steps += 1
            if terms.any() or truncs.any():
                break
    return pipes_passed, steps


def main():
    parser = argparse.ArgumentParser(description="Eval curriculum-trained Flappy policy")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to checkpoint .pt (default: latest in experiments/)",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed (default 42 = reproducible)")
    parser.add_argument(
        "--random-seed",
        action="store_true",
        help="Use a random seed (stats will vary between runs)",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="If > 0, run this many episodes headless and print stats (no render)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="Run headless (use with --episodes for numerical eval)",
    )
    parser.add_argument(
        "--difficulty",
        type=float,
        default=1.0,
        help="Curriculum difficulty for eval (default 1.0 = pure uniform, matching standard env)",
    )
    args = parser.parse_args()

    if args.random_seed:
        args.seed = int(np.random.default_rng().integers(0, 2**31))
        print(f"Using random seed: {args.seed}")

    model_path = args.model or find_latest_checkpoint()
    if not model_path or not os.path.isfile(model_path):
        print("No checkpoint found. Train first or pass --model path/to/model_XXXXXX.pt")
        return

    # Shared difficulty value — fixed for eval (not ramped)
    difficulty_value = multiprocessing.Value("f", args.difficulty)
    print(f"Eval difficulty: {args.difficulty:.2f}")

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
    driver = vecenv.driver_env
    policy = FlappyGridPolicy(driver).to(args.device)
    state_dict = torch.load(model_path, map_location=args.device)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    policy.load_state_dict(state_dict, strict=True)
    policy.eval()

    print(f"Checkpoint: {model_path}")
    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")

    if args.episodes > 0:
        # Numerical eval: run N episodes, report pipes passed and length
        scores = []
        lengths = []
        for ep in range(args.episodes):
            pipes, length = run_episode(vecenv, policy, args.device, seed=args.seed + ep)
            scores.append(pipes)
            lengths.append(length)
            if (ep + 1) % 10 == 0:
                print(f"  [{ep+1}/{args.episodes}] last: {pipes} pipes, {length} steps")
        vecenv.close()
        scores = np.array(scores)
        lengths = np.array(lengths)
        print(f"\n--- Results ({args.episodes} episodes, difficulty={args.difficulty:.2f}) ---")
        print(f"Pipes passed — mean: {scores.mean():.2f}, std: {scores.std():.2f}, min: {scores.min()}, max: {scores.max()}")
        print(f"Length       — mean: {lengths.mean():.1f}, std: {lengths.std():.1f}, min: {lengths.min()}, max: {lengths.max()}")
        return

    # Interactive render mode
    episode_num = 0
    obs, info = vecenv.reset(seed=args.seed + episode_num)
    total_pipes = 0
    ep_pipes = 0
    with torch.no_grad():
        while True:
            if not args.no_render:
                driver.render()
            ob = torch.as_tensor(obs).to(args.device)
            logits, _ = policy.forward_eval(ob)
            action = logits.argmax(dim=-1).cpu().numpy().reshape(vecenv.action_space.shape)
            obs, rewards, terms, truncs, info = vecenv.step(action)
            r = float(rewards.flat[0])
            if r >= 1.0:
                ep_pipes += 1
            time.sleep(1 / FPS if not args.no_render else 0)
            if terms.any() or truncs.any():
                total_pipes += ep_pipes
                episode_num += 1
                print(f"Episode {episode_num}: {ep_pipes} pipes")
                ep_pipes = 0
                obs, _ = vecenv.reset(seed=args.seed + episode_num)


if __name__ == "__main__":
    main()
