"""Run the env demo. Use: uv run python -m curly_succotash"""

import warnings

# Gym deprecation comes from pufferlib's dependencies; we only use Gymnasium
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")


def _run_demo():
    from curly_succotash.env import SamplePufferEnv, make_gymnasium_env

    print("=== Gymnasium wrapper ===")
    env = make_gymnasium_env()
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    print("obs:", obs, "reward:", reward, "term:", term, "trunc:", trunc)

    print("\n=== Native PufferEnv ===")
    env = SamplePufferEnv()
    obs, infos = env.reset()
    actions = env.action_space.sample()
    obs, rewards, terms, truncs, infos = env.step(actions)
    print("obs:", obs, "rewards:", rewards, "terms:", terms, "truncs:", truncs)


if __name__ == "__main__":
    _run_demo()
