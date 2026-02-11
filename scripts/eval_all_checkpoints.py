#!/usr/bin/env python3
"""
Run numerical eval (50 episodes, no render) on the main Flappy checkpoints.
Run from repo root:  uv run python scripts/eval_all_checkpoints.py
"""
import subprocess
import sys

CHECKPOINTS = [
    ("177077321939/model_001220.pt", "fixed-gap"),
    ("177077412473/model_002441.pt", "655 ep length"),
    ("177077598031/model_002441.pt", "684 ep length"),
    ("177078209826/model_002441.pt", "biased fine-tune"),
]

def main():
    for path_suffix, label in CHECKPOINTS:
        path = f"experiments/{path_suffix}"
        print(f"\n{'='*60}")
        print(f"  {label}")
        print(f"  {path}")
        print("="*60)
        r = subprocess.run(
            [sys.executable, "-m", "flappy_rl.run_eval_flappy", "--model", path, "--episodes", "50", "--no-render", "--random-seed"],
            cwd=".",
            capture_output=False,
        )
        if r.returncode != 0:
            print(f"  (exit code {r.returncode})")
    print("\nDone.")

if __name__ == "__main__":
    main()
