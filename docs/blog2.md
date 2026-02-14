## I built an RL environment for Flappy Bird and trained an agent to beat the game

I just finished designing my first RL environment and training a policy that beats it 99% of the time.

Reinforcement learning, simplified for those who are unfamiliar, involves having an environment (like a video game) which has a state (like the position of your characters and things around you), and an agent that interacts with the environment through actions (up, down, shoot, jump, etc.). Your goal is to train a "policy" (some neural net) that takes the environment's state as an input and learns the preferred actions to beat the env (win the game) from reward signals. To train this policy, the env returns rewards which are positive for good actions and neutral / negative for other actions. Your policy just learns to maximize the reward it receives over time by picking the right actions over and over again.

You can appreciate the difficulties faced in reinforement learning: how do you develop a strategy that correctly maps actions taken early in a game where rewards are sparse and given over a long time horizon, for example, or building strategy for a constantly changing environment where there are many different players acting independently?

RL has been used to solve a number of complex problems historically, like OpenAI beating the best players of Dota 2 in 2019 (1), or AlphaGo defeating the world champions of Go in 2016 (2). Today, RL is used beyond beating video games. In bleeding edge fields, RL is being used to train robots how to wash dishes (the world is just one big env after all) and improve large language models (ie when ChatGPT asks you "which response do you prefer"), amongst other things.

As someone who's been diving into ML the last 107 days (see my daily status updates on X), RL has stood out as a point of interest not just for its applications in a variety of industries, but also because it is super accessible. **Training good policies for RL does not require large amounts of labelled data nor does it require an insane amount of compute to get started with meaningful work.**

### The Flappy Bird Process

I'll walk you through some of the iterations I went through before reaching the final checkpoint that beat the game fully. At the end, I'll include a chart with the metrics for each one and my key takeaways / learnings. If you want to skip ahead, click here.

Version 1: 2 grid Flappy Bird - literally just played in the terminal, where the only options are roof and floor. If the bird is up, it should go down, and if it's down, it should go up. +1 for correct, -1 if it dies. This converged pretty quickly on a tiny feed-forward policy of ~17.7K params.

Version 2: Since that worked, I was getting the hang of it. It was time to design the full game. +1 for passing a pipe and -1 for dying. For observations, the env returned bird position (x,y), bird velocity / direction, gap center point, gap height, distance from next gap. The possible actions were flap or don't flap. **My policy arhitecture remained the same from above - I had figured it didn't need to be any bigger since the game was seemingly simple. This was a mistake I paid for throughout** (we'll get there). My first 8 training runs didn't even make it past the first pipe due to some critical env flaws.

Version 3: Repeatedly corrected the flaws and kept trying to train. Fixed things like flaps sending the bird to the ground instead of flying up, each flap sending the bird at an impossible distance / speed, and a huge lead-up to the first pipe which resulted in bird learning the wrong pipe gap early on. Still not making it past 4 pipes on average. I even ran a hyper parameter sweep and trained for 30-40M steps on some of these runs. Since nothing was working, I figured maybe the bird needed stronger rewards and more information from the env. **We expanded the rewards to include a survival bonus, in-gap alignment bonus, pre-gap centering bonus, and streak bonus. Also expanded the observations to include signed gap error, "is next pipe valid", top-gap clearance, and bottom-gap clearance.** Still no improvement.

Version 4: Time to try something else. I needed a sanity check to make sure there wasn't some critical bug that was preventing this from working. I simplified the environment to have normalized gaps (every pipe gap is the same). It converged within 10M steps, beating the game every time. Sanity check passed. I took the model checkpoints from this run and kept fine-tuning ~20M steps on marginally harder environments until we reached the standard env. This got me to a policy that could hit ~6 pipes. I ran an eval and saw it kept dying on pipes which were at the ends of the range, so I ran a bias training with an env that only had pipes at the ends of the gap range (high or low). This fine-tuned policy was hitting ~8.5 pipes on average, the best checkpoint so far.

Version 5: I now started to understand this concept of a **curriculum**. It's like teaching a baby to walk - you wouldn't just throw it into a marathon from day one. So I solidified the fine-tunes I ran into a special training environment with four phases: Phase 0 was easiest with normalized pipe gaps, Phase 1 with some variation, Phase 2 with super hard pipes, and then Phase 3 with the standard game. After a couple of runs, I saw that the policy was drastically deteriorating each phase (expected at first) but not recovering (not expected). So, I changed the curriculum: first 10% is the easy environment (don't want to start with a moving target), then 11-45% is a gradual ramp up, after which the last 55% is the standard. **I also introduced a learning rate decay and tweaked hyper params so the policy woulnd't unlearn things in later stages of training.** Running this for 150M steps got me a policy that could consistently hit 15 pipes (I capped the env at around 65 pipes before the game restarts).

# Version 6: THE FINAL VERSION.

Something wasn't right about this still. Hitting 15 pipes was a big step up from where we started, but when I trained other environments in the PufferLib library, some of which were more complex than Flappy, they converged in less steps than mine and with simpler reward design, no curriculum, and no special tweaks. I then realized my mistake. Though an MLP could learn useful behavior with the observations I gave it,**without some sort of memory mechanism, my policy was poorly suited to model how a previous flap affects future state.** Additionally, **including redudant, overengineered observation features also defeated the purpose of having a neural net which should understand the implied relationships** between things like bird pos, gap height, etc. Lastly, the **additional rewards like streak and survival bonus were teaching the bird to optimize for the wrong things** and adding noise to the policy. So, I went back to the original +1, -1 reward system and 5 base observations. I removed the curriculum entirely. **And, the special ingridient which eluded me this whole time, I added a 128-hidden LSTM (total policy ~133K params) on top of my existing policy.** And voila, there we had it - I trained a near-ceiling policy that hits the game cap (~65 pipes) consistently.

| Version | Env setup | Reward + observations | Policy / training | Outcome |
|---|---|---|---|---|
| 1 | 2-grid terminal Flappy (roof/floor) | Simple signal, trivial state | Small MLP (~17.7K params) | Converged quickly (sanity baseline) |
| 2 | First full Flappy game | +1 pass, -1 death; base game obs/action | Same small MLP | First 8 runs failed before first pipe due to env issues |
| 3 | Fixed physics/pipe spawn bugs, many retries | Added dense bonuses + engineered obs features | Same small MLP; 30-40M step attempts | Plateaued (typically <= 4 pipes) |
| 4 | Simplified fixed-gap sanity env + staged fine-tunes | Gradually reintroduced randomness + extreme-gap biasing | Checkpoint transfer + fine-tuning | Reached ~8.5 to ~10.9 pipes depending on checkpoint |
| 5 | Curriculum Flappy (smooth ramp + warmup hold) | Simplified back to +1/-1 and leaner obs | Added LR decay and tuned PPO hyperparams | 150M-step run reached ~15.6 pipes mean (best pre-v6) |
| 6 | Target-like setup: no curriculum, full random gap range | Kept sparse +1/-1 and 5 core obs | Default+LSTM (128 hidden, ~133K params), simplified hyperparams | Near-ceiling policy: ~65 pipes cap consistently, score ~61 at 100M steps |

My key takeaways moving forward:

1. Simplicity is king in RL. More complexity can be more noise unless it's addressing a specific problem or gap in the existing implementation.

2. Setting a policy architecture that reflects the nature of the problem. Beating Flappy Bird involves solving a temporal problem, ie timing your flaps, so we need to capture temporal relationships. A lesson I learned by banging my head against a wall repeatedly.

3. Curriculum is a useful trick. You can test a lot of interesting things with a carefully crafted curriculum and fine-tune your agent to solve increasingly harder tasks. Though it wasn't needed for my final best setup, I'm glad I took the scenic route to my solution and stumbled upon this concept.


Onto Day 108 of diving into ML 🫡
