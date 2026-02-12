---
title: "Building my first RL environment."
description: "Pt. 3 of building undeniable technical ability"
pubDate: 2026-02-03
---

I find reinforcement learning to be the most interesting sector of ML at the moment for its unique attributes of requiring less compute than supervised learning policies, relevance to the next paradigm of AI (robotics / world models), and for that it is largely neglected relative to LLMs and other forms of ML.

Today I built my first environment, namely a two-grid flappy bird where the agent has two actions, up or down, and receives a -1 reward for hitting the floor or roof, and 0 reward for staying alive.

My first version of the env just returns three values for the observation, namely, [position of agent, roof pos, floor pos].

**Mistakes and iterations.** Initially I didn't include the agent's position in the observation—only whether there was a wall on the roof or floor. The agent had to infer "am I at floor or ceiling?" from history, which made learning much harder. I fixed that. Reward was only -1 on death and 0 otherwise, so the signal was sparse; I added a small +0.01 per step survived so "stay alive" had a positive return. Training hit a few snags: the vectorizer calls `driver_env.close()`, and the base PufferEnv doesn't implement it, so I added a no-op `close()`. Default config gave `batch_size < minibatch_size` with only two envs; I increased `num_envs` so the batch was large enough for PuffeRL.

**Why wasn't it learning in 10 steps?** The rule is trivial: at roof don't go up, at floor don't go down. I expected the policy to overfit that in a handful of steps. What I got was entropy stuck near 0.69 (max for two actions) for 500k–2M steps. The reason isn't the task—it's the algorithm. RL doesn't get "correct action" labels; it explores, gets rewards, and slowly reinforces better actions. PPO batches thousands of steps per update and clips policy changes, so each bad (or good) experience only nudges the policy a little. The setup is built for stability in hard envs, not "memorize this rule in 10 steps."

**Making it learn faster.** I bumped the learning rate (3e-4 → 0.01) and loosened the PPO clip (0.2 → 0.5). The policy went deterministic (entropy → 0) within 2M steps. So the agent *can* learn the rule; it just needed a more aggressive update.

**Sweep: where does it become deterministic?** I added a small grid sweep over learning rate and clip coefficient, ran 500k steps per (lr, clip) pair, and recorded final entropy. Result: at lr = 0.03 the policy goes deterministic for every clip value tried (0.2, 0.35, 0.5, 0.7). At lr = 0.01 entropy drops to ~0.05–0.14 but doesn't cross the "deterministic" threshold in 500k steps. At 0.001 and 0.003 the policy stays exploratory (entropy ~0.6). So LR is the main lever; clip_coef barely mattered in this grid.

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

---

**Flappy (C + raylib): reward, obs, and env iterations.** After porting to a proper Flappy Bird–style env (pipes, gap, continuous-ish physics), the agent often couldn't get through the first pipe. Below are the changes I tried, what broke, and how things improved to where they are now.

**Env / physics tweaks**
- **Slower pipes** – Reduced `PIPE_SPEED_RATIO` from 0.012 → 0.006 so the bird had more time to line up. Without this, the agent rarely got +1 for passing a pipe and learning didn't take off.
- **Lower flap velocity** – Started at 0.055; the bird overshot and hit the top pipe. I stepped it down (0.04 → 0.032 → 0.022 → 0.02). Each reduction gave finer control and less "flap into ceiling." This was one of the biggest levers for actually getting through pipes.

**Rewards**
- **Survival bonus** – Already had +0.01 per step alive so the policy had a positive signal for "don't die."
- **In-gap bonus** – +0.02 when the bird is *inside* the gap, scaled by distance to the pipe (closer = more). Helps with "stay in the safe zone" once you're there.
- **Alignment bonus** – Added a small reward for being *near* the gap center *before* entering the gap (e.g. 0.008 per step, decaying with vertical distance over a tolerance of 0.2). Goal: encourage lining up early instead of last-second flapping. This helped.
- **Streak bonus** – Pipe-pass reward became 1.0 for first pipe, 1.1 for second, 1.2 for third, etc., so the agent is incentivized to keep going for later pipes.
- **Flap penalty** – Tried a tiny cost per flap (0.001) to discourage flapping when already high. In practice it didn't seem to change behavior much. I didn't push it higher because **too much flap penalty makes the bird too passive and it deterministically hits the ground**—the policy stops flapping enough to stay in the air.
- **Penalty for being above the gap** – Considered a small negative reward when the bird is above the gap center to reduce "flap into top pipe." Didn't implement it because **strong "above gap" penalties also risk the bird preferring to fall and hit the ground** rather than risk being "too high." Same failure mode as an oversized flap penalty.

**Observations**
- **Original 7-D** – bird y, bird vy, distance to next pipe, gap center, gap height, "is there a next pipe?", and signed gap error (gap_center − bird_y, clamped to ±1). The clamp on gap error meant "slightly above" and "way above" both looked similar (both negative), so the policy didn't get a clear "don't flap, you're way too high" signal.
- **Added clearance from top and bottom of gap (9-D)** – Two new dims: signed clearance from the *top* of the gap and from the *bottom*, in half-gap units, clamped to ±1. So the agent sees explicitly "how far am I from the top pipe?" and "how far am I from the bottom pipe?" That made "way above" vs "slightly above" learnable and helped with consistency on the first pipe and sometimes the second.



**Where things stand (as of yesterday)** – With slower pipes, reduced flap velocity, alignment + streak rewards, a small flap penalty, and the extra clearance obs, the bird reliably gets the first pipe and sometimes the second. Episode length hovers around ~200 steps; more training (e.g. 40M steps) didn't push it much past that—so we're at a plateau. Perhaps a local minima that just gets the first pipe and doesn't really try for later ones. Trying different checkpoints (earlier saves from the same run) can sometimes yield a slightly more consistent policy than the final one.

## Day 2

**Bugs fixed along the way**
- **"Next" pipe could be wrong** – "Next" was the first pipe in *array* order that was still in front of the bird. After recycling, the leftmost pipe moves to the right so array order ≠ left-to-right; the agent could get obs for a pipe it wasn't actually approaching. Fixed by choosing the pipe in front with the *smallest x* (closest).
- **Pipe recycling broke spacing** – On recycle we set the new pipe's x to `rightmost + spacing`, then called `spawn_pipe()`, which overwrote x with the screen width. That put two pipes almost on top of each other. Fixed by having `spawn_pipe()` only set gap and scored; the caller keeps responsibility for x.

**Where things stood (plateau)**
- With slower pipes, reduced flap velocity, alignment + streak rewards, a small flap penalty, and the extra clearance obs, the bird **reliably got the first pipe and sometimes the second**. Episode length hovered around ~200 steps; more training didn't push it much past that.


**Curriculum and fine-tuning**

1. **Shorter run-up to the first pipe** – The distance from bird spawn to the first pipe was longer than the spacing between later pipes, so the agent saw many more "approach first pipe" steps than "approach second pipe" and never got enough positive examples for pipes 2+. I cut that initial distance in half (first pipe at 0.5× width instead of 0.8×). That helped: return and score improved.

2. **Sanity check: fixed gap** – To verify the env and training stack, I removed gap randomness so every pipe had its gap in the middle (same y). My reasoning: if everything is correct, the agent should learn to stay in the middle and fly forever. After a 10M-step run with this "fixed gap" env, the agent reached my max terminal (50 pipes) and kept going—so the env and PufferLib were fine. The difficulty with random gaps was task difficulty, not a bug.

3. **Fine-tune from fixed-gap, then catastrophic failure** – I used the fixed-gap checkpoint as a starting point and reintroduced random gaps, then fine-tuned. After more steps, the policy **collapsed**: episode length dropped to ~5–6, the bird was effectively suiciding. I'd left the hyperparameters from the simple grid task in place (high LR, loose clip, high entropy). I rolled them back to values better suited for a harder, stochastic task:
   - **Learning rate** – How big each gradient step is; too high and the policy overshoots and can forget good behavior (I went 0.02 → 3e-4).
   - **Clip coefficient** – PPO's limit on how much the policy can change per update; too loose and updates are noisy (0.5 → 0.2).
   - **Entropy coefficient** – Bonus for exploring; too high and the policy stays too random and doesn't commit to a strategy (0.2 → 0.01).

   With LR 3e-4, clip 0.2, and ent_coef 0.01, I fine-tuned again from the *previous* good checkpoint (before the collapse) for 20M steps. That produced a policy with **average episode return ~2.76 and episode length ~655**—a clear improvement. On eval, the bird consistently gets several pipes, with some runs reaching the high teens or twenties, but it's still inconsistent and there's room to improve.

4. **Further fine-tuning** – Additional fine-tuning runs (another 5M or 20M steps from that best checkpoint) gave similar or slightly worse metrics (e.g. length ~601–608 vs 655). So I'm sticking with the checkpoint that achieved ~655 episode length as my best model for eval. The lesson: once you have a good checkpoint on a stochastic task, more training with the same setup doesn't always help; sometimes it's noise or a slight regression, and the best thing is to keep that checkpoint for deployment and only change hyperparams or curriculum if you want to push further.

5. Just tried another fine tuning with a smaller learning rate for 50M steps and it gave a worse result. I think the 655 run was our local optimum.

6. Ran bias training / fine tuned the best checkpoint thus far on 20M steps. The bias mechanism only sampled extreme gap locations (super high or super low end of the range), so that the bird got used to those situations and would fail less. I anticipate that doing this for a variety of bias levels will improve the policy. Here's a summary of improvements from our best policies:

Checkpoint	Label	Seed	Pipes (mean ± std)	Min–max pipes	Length (mean ± std)	Min–max length
177077321939 model_001220.pt	fixed-gap	91486274	4.76 ± 3.62	1–17	363.0 ± 271.4	82–1281
177077412473 model_002441.pt	655 ep	1994498143	8.96 ± 6.82	1–24	678.3 ± 511.4	82–1807
177077598031 model_002441.pt	684 ep	1275270359	8.20 ± 8.22	1–50	621.4 ± 616.7	79–3757
177078209826 model_002441.pt	biased fine-tune	1865502367	10.94 ± 9.21	1–41	827.5 ± 690.5	78–3082



## Day 3:

Today I focused on solidifying the curriculum into a single, repeatable training loop and making refinements to hyperparams, and simplifying / improving the environment and rewards.

1. First iteration involved breaking the environment into four distinct phases which were:
-> Phase 0: Non-deviating pipe gaps (one straight line, pipes in same place)
-> Phase 1: Slight deviation (0.35 - 0.65)
-> Phase 2: Working on extremes (0.25 / 0.75 only)
-> Phase 3: Standard env with random pipes in full range

The learning rate stayed constant through all four phases.

The end result of a run with 100M steps was worse than my previous best checkpoint and plateaued at 200 episode length (2 pipes). I watched training - beat the game in Phase 0 (50 pipes), dropped to 8 pipes in Phase 1, 2 pipes in Phase 3, and 4 pipes at the end.

I figured some things that might be making this worse than before were:
   1. Hard cliffs between phases were throwing off learning because the envs were too different
   2. Learning rate too large late into training meant forgetting what was previously learned

Also, I looked at some other envs in PufferLib more complex than mine, but their reward signal was drastically simpler. For example, in the Target env, which is multi-agent and has a more extensive set of obs, the reward signal was only +1 for getting the coin and 0 otherwise. I had five reward signals including +1 for passing a pipe, -1 for dying, a streak multiplier to incentivize staying alive for later pipes, +0.01 surival bonus for every time step, and -0.001 for flapping to reduce hitting the top pipe. It was clear that I had overcomplicated the absolute @#$* out of my rewards, creating noise and making it more difficult for a small policy to learn the right relationships. Same with the the observations. I had 9 dimensions of which four were derivatives that should've just been able to be infered by a good policy and were making it harder to learn.


2. I cut reward to +1 for pipe passed and -1 for dying. I cut observations down to bird pos (x, y), distance to next pipe, gap center, and gap height. I made the ramp up of difficulty continous (removed discrete phases) for the first half of training and then had the standard environment for the later half. I also incorporated learning rate decay so we dont unlearn things from before.

I knew this was an improvement but realized that by setting the ramp up to be continous from the get-go, the agent always had a moving target and never learned the fundamentals of how to fly, so this resulted in a poor outcome in an 80M step run of 4 pipes.

So the last change I made, was making sure that the first 10% of the run is always in the fixed pipe, easy environment.

With this, I ran another 80M step run and patiently awaited. The result showed victory - 9.5 pipes on average, beating my previous best checkpoint. I ran some evals, and it was clear that training was still getting better with more steps. So finally, I ran it again for 150M steps total, and got the best agent which is where we are at today, and I'm calling this environment solved. Mean number of pipes = 15.8!. I ran evals again and saw that the agent was only improving with more training still. Since I'm on a CPU and I need to use my laptop, I'm stopping here. But the key lessons from this process:

- Simplicity wins in RL reward design. I don't a bunch of different signals that add noise - I just need the ones that directly correlate to the desired outocme.
- Simplicity also wins in the observation design. It should be exactly the base, objective data points that the agent needs. From there, a good policy can make its own inferences on how these data poitns relate. By including these over-engineered data points, I'm making it harder for the policy to learn all the relationships. At least a small policy - if I had more compute / more params, it probably would've been fine. But there's certainly a tradeoff here and it's one to keep in mind.
- Curriculum matters. The same way we learn to walk before we run, throwing the agent into a super hard version of the environment off the getgo with randomly initialized weights is a recipe for failure. Letting the agent learn baseline params from easier envs and baking this into your training curriculum is the right way to design training envs for RL.
- Hyperparams - always run a sweep, ensure you're exploring enough in the early stages of training, but also ensure you're not unlearning things in the later stages of learning.

## High-level training run summary

| Run ID | Checkpoint | Setup (high-level) | Train steps | Eval mean pipes | Eval mean length | Notes |
|---|---|---|---:|---:|---:|---|
| 177077321939 | model_001220.pt | Fixed-gap baseline | N/A | 4.76 | 363.0 | First strong baseline |
| 177077412473 | model_002441.pt | Fine-tune on random gaps | N/A | 8.96 | 678.3 | Best pre-curriculum checkpoint |
| 177077598031 | model_002441.pt | Lower-LR fine-tune | N/A | 8.20 | 621.4 | Slight regression vs prior best |
| 177078209826 | model_002441.pt | Biased fine-tune (extreme gaps) | N/A | 10.94 | 827.5 | Best Day 2 checkpoint |
| 177086914642 | model_009765.pt | Simplified reward/obs + smooth curriculum (80M) | 80M | 13.02 | 991.3 | First run to clearly beat Day 2 |
| 177087020156 | model_018310.pt | Same setup, longer training + warmup hold | 150M | 15.58 | 1176.1 | Current best checkpoint |

