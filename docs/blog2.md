## I built an RL environment for Flappy Bird and trained an agent to beat the game

I just finished designing my first RL environment and training a policy that beats it 99% of the time.

Reinforcement learning, simply put for those who are unfamiliar, involves having an environment (like a video game) which has a state (like the position of your characters and things around you), and an agent that interacts with the environment through actions (up, down, shoot, jump, etc.). Your goal is to train a "policy" (some neural net) that takes the environment's state as an input and gives the correct action as an output to beat the env (win the game). To train this policy, the env returns rewards which are positive for good actions and neutral / negative for other actions. Your policy just learns to maximize the reward it receives over time by picking the right actions over and over again.

You can appreciate the difficulties faced in reinforement learning: how do you develop a strategy that correctly maps actions taken early in a game where rewards are sparse and given over a long time horizon, for example, or building strategy for a constantly changing environment where there are many different players acting independently?

RL has been used to solve a number of complex problems historically, like OpenAI beating the best players of Dota 2 in 2019, or AlphaGo defeating the world champions of Go in 2016. Today, RL is used beyond beating video games. In bleeding edge fields, RL is being used to train robots how to wash dishes (the world is just one big env after all) and improve large language models (ie when ChatGPT asks you "which response do you prefer").

As someone who's been diving into ML the last 107 days (see my daily status updates on X), RL has stood out as a point of interest not just for its applications in a variety of industries, but also because it is super accessible. **Training good policies for RL does not require large amounts of labelled data nor does it require an insane amount of compute to get started with meaningful work.**

Below I'm going to document how my process designing Flappy Bird RL went and some key takeaways.

My final architecture:
    - Reward: +1 for passing a pipe, -1 for dying
