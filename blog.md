---
title: "Building my first RL environment."
description: "Pt. 3 of building undeniable technical ability"
pubDate: 2026-02-03
---

I find reinforcement learning to be the most interesting sector of ML at the moment for its unique attributes of requiring less compute than supervised learning policies, relevance to the next paradigm of AI (robotics / world models), and for that it is largely neglected relative to LLMs and other forms of ML.

Today I built my first environment, namely a two-grid flappy bird where the agent has two actions, up or down, and receives a -1 reward for hitting the floor or roof, and 0 reward for staying alive.

My first version of the env just returns three values for the observation, namely, [position of agent, roof pos, floor pos].

**Mistakes and iterations.** Initially I didn’t include the agent’s position in the observation—only whether there was a wall on the roof or floor. The agent had to infer “am I at floor or ceiling?” from history, which made learning much harder. I fixed that. Reward was only -1 on death and 0 otherwise, so the signal was sparse; I added a small +0.01 per step survived so “stay alive” had a positive return. Training hit a few snags: the vectorizer calls `driver_env.close()`, and the base PufferEnv doesn’t implement it, so I added a no-op `close()`. Default config gave `batch_size < minibatch_size` with only two envs; I increased `num_envs` so the batch was large enough for PuffeRL.

**Why wasn’t it learning in 10 steps?** The rule is trivial: at roof don’t go up, at floor don’t go down. I expected the policy to overfit that in a handful of steps. What I got was entropy stuck near 0.69 (max for two actions) for 500k–2M steps. The reason isn’t the task—it’s the algorithm. RL doesn’t get “correct action” labels; it explores, gets rewards, and slowly reinforces better actions. PPO batches thousands of steps per update and clips policy changes, so each bad (or good) experience only nudges the policy a little. The setup is built for stability in hard envs, not “memorize this rule in 10 steps.”

**Making it learn faster.** I bumped the learning rate (3e-4 → 0.01) and loosened the PPO clip (0.2 → 0.5). The policy went deterministic (entropy → 0) within 2M steps. So the agent *can* learn the rule; it just needed a more aggressive update.

**Sweep: where does it become deterministic?** I added a small grid sweep over learning rate and clip coefficient, ran 500k steps per (lr, clip) pair, and recorded final entropy. Result: at lr = 0.03 the policy goes deterministic for every clip value tried (0.2, 0.35, 0.5, 0.7). At lr = 0.01 entropy drops to ~0.05–0.14 but doesn’t cross the “deterministic” threshold in 500k steps. At 0.001 and 0.003 the policy stays exploratory (entropy ~0.6). So LR is the main lever; clip_coef barely mattered in this grid.

Sweep summary (final entropy per lr × clip_coef, 500k steps per run):


| lr     | clip_coef | entropy | epoch | deterministic |
|--------|-----------|---------|-------|----------------|
| 0.001  | 0.2       | 0.6721  | 55    | no             |
| 0.001  | 0.35      | 0.6807  | 55    | no             |
| 0.001  | 0.5       | 0.6823  | 60    | no             |
| 0.001  | 0.7       | 0.6495  | 61    | no             |
| 0.003  | 0.2       | 0.6151  | 61    | no             |
| 0.003  | 0.35      | 0.6352  | 58    | no             |
| 0.003  | 0.5       | 0.6229  | 55    | no             |
| 0.003  | 0.7       | 0.5921  | 54    | no             |
| 0.01   | 0.2       | 0.0552  | 54    | no             |
| 0.01   | 0.35      | 0.1178  | 60    | no             |
| 0.01   | 0.5       | 0.1123  | 55    | no             |
| 0.01   | 0.7       | 0.1359  | 55    | no             |
| 0.03   | 0.2       | 0.0000  | 58    | yes            |
| 0.03   | 0.35      | 0.0000  | 60    | yes            |
| 0.03   | 0.5       | 0.0000  | 61    | yes            |
| 0.03   | 0.7       | 0.0000  | 60    | yes            |
