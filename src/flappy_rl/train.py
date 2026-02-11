"""
Train an agent on FlappyGridEnv or Flappy (C+raylib) with PufferLib's PuffeRL.

Usage:
  uv run python -m flappy_rl.train
  uv run python -m flappy_rl.train --train.env flappy
  uv run python -m flappy_rl.train --train.device cuda --train.total_timesteps 1000000
"""

import argparse
import torch
import pufferlib
import pufferlib.vector
from pufferlib import pufferl

from flappy_rl.env import flappy_grid_env_creator, FlappyGridEnv


class FlappyGridPolicy(torch.nn.Module):
    """Simple MLP policy for FlappyGrid or Flappy (small obs, discrete actions)."""

    def __init__(self, env, logit_temperature=2.0):
        super().__init__()
        obs_size = env.single_observation_space.shape[0]
        n_actions = env.single_action_space.n
        hidden = 128
        self.logit_temperature = logit_temperature
        self.net = torch.nn.Sequential(
            pufferlib.pytorch.layer_init(torch.nn.Linear(obs_size, hidden)),
            torch.nn.ReLU(),
            pufferlib.pytorch.layer_init(torch.nn.Linear(hidden, hidden)),
            torch.nn.ReLU(),
        )
        self.action_head = torch.nn.Linear(hidden, n_actions)
        self.value_head = torch.nn.Linear(hidden, 1)

    def forward_eval(self, observations, state=None):
        hidden = self.net(observations)
        logits = self.action_head(hidden)
        values = self.value_head(hidden)
        return logits, values

    def forward(self, observations, state=None):
        logits, values = self.forward_eval(observations, state)
        logits = logits / self.logit_temperature
        return logits, values


def main():
    # Parse --train.env and --train.total-timesteps before load_config; strip from argv so PufferLib's parser doesn't reject
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--train.env", default="flappy_grid", dest="train_env")
    parser.add_argument("--train.total-timesteps", type=int, default=None, dest="train_total_timesteps")
    parser.add_argument("--train.load-checkpoint", type=str, default=None, dest="train_load_checkpoint")
    parser.add_argument("--train.learning-rate", type=float, default=None, dest="train_learning_rate")
    known, _ = parser.parse_known_args()
    env_name = known.train_env
    total_timesteps_override = known.train_total_timesteps
    if "--train.env" in sys.argv:
        i = sys.argv.index("--train.env")
        del sys.argv[i]
        if i < len(sys.argv):
            del sys.argv[i]  # value
    if "--train.total-timesteps" in sys.argv:
        i = sys.argv.index("--train.total-timesteps")
        del sys.argv[i]
        if i < len(sys.argv):
            del sys.argv[i]  # value
    if "--train.load-checkpoint" in sys.argv:
        i = sys.argv.index("--train.load-checkpoint")
        del sys.argv[i]
        if i < len(sys.argv):
            del sys.argv[i]  # value
    if "--train.learning-rate" in sys.argv:
        i = sys.argv.index("--train.learning-rate")
        del sys.argv[i]
        if i < len(sys.argv):
            del sys.argv[i]  # value

    # Load default PufferLib config (train + vec sections)
    args = pufferl.load_config("default")
    args["train"]["env"] = env_name
    args["train"]["total_timesteps"] = total_timesteps_override if total_timesteps_override is not None else 5_000_000
    args["train"]["optimizer"] = "adam"
    # Tuned for Flappy (and fine-tuning): lower LR and standard clip to avoid collapse
    args["train"]["learning_rate"] = getattr(known, "train_learning_rate", None) or 3e-4
    args["train"]["clip_coef"] = 0.2
    args["train"]["ent_coef"] = 0.01
    if not torch.cuda.is_available():
        args["train"]["device"] = "cpu"

    vec_kwargs = dict(args["vec"])
    if vec_kwargs.get("num_workers") == "auto":
        vec_kwargs["num_workers"] = 2
    if vec_kwargs.get("num_envs") in (None, "auto") or vec_kwargs.get("num_envs", 0) < 128:
        vec_kwargs["num_envs"] = 128

    if env_name == "flappy":
        from flappy_rl.flappy import flappy_env_creator

        # One Flappy(num_envs=1) per vector slot; pass the function (picklable), not its return value
        vecenv = pufferlib.vector.make(
            flappy_env_creator,
            env_kwargs={"num_envs": 1, "width": 400, "height": 600},
            **vec_kwargs,
        )
    else:
        vecenv = pufferlib.vector.make(
            flappy_grid_env_creator,
            **vec_kwargs,
        )
    policy = FlappyGridPolicy(vecenv.driver_env).to(args["train"]["device"])

    load_checkpoint = getattr(known, "train_load_checkpoint", None)
    if load_checkpoint:
        state_dict = torch.load(load_checkpoint, map_location=args["train"]["device"])
        policy.load_state_dict(state_dict, strict=True)
        print(f"Loaded policy from {load_checkpoint} (fine-tuning)")

    trainer = pufferl.PuffeRL(args["train"], vecenv, policy)

    while trainer.epoch < trainer.total_epochs:
        trainer.evaluate()
        trainer.train()
        trainer.print_dashboard()

    trainer.close()
    print("Training finished. Check experiments/ for checkpoints.")


if __name__ == "__main__":
    main()
