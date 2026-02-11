# Reinforcement learning basics (and how this repo fits in)

You're new to RL — here's the big picture, then how the env and PufferLib fit in.

---

## 1. What is reinforcement learning?

You have an **agent** (e.g. a neural network) that has to learn how to **act** in a **world** so that it gets as much **reward** as possible over time.

- **World** = the **environment** (a game, a robot in a sim, a custom problem you model).
- **Act** = at each moment, the agent picks an **action** (e.g. "move left", "press button 1").
- **Reward** = a number the environment gives back (e.g. +1 for scoring, -1 for dying, 0 for nothing). The agent's goal is to get more reward in the long run.

So: **agent sees the world → chooses action → world updates and gives reward → repeat.** The agent has to learn which actions lead to more reward, even when the effect is delayed (e.g. "press jump now, get reward when you land on the platform").

---

## 2. The loop (one episode)

Conceptually, every run looks like this:

1. **Reset** the environment → you get an **observation** (what the agent "sees" right now).
2. **Loop:**
   - Agent looks at the **observation** and picks an **action**.
   - You send that **action** to the environment.
   - Environment does one **step**: the world changes, and it gives you:
     - the **next observation**,
     - a **reward** (number),
     - whether the episode **ended** (terminated or truncated).
   - If the episode ended, **reset** again and start a new episode; otherwise repeat the loop.

So the core contract is:

- **reset()** → `(observation, info)`
- **step(action)** → `(observation, reward, terminated, truncated, info)`

The **policy** (e.g. a neural network) is the thing that turns **observation → action**. Training is the process of improving that policy so it gets more reward over time.

---

## 3. Observations and actions (spaces)

The environment has to declare the **shape and type** of observations and actions so the agent (and the library) know what to expect.

- **Observation space**  
  "What does the agent see?"  
  Example: a vector of 2 floats in [-1, 1] → `Box(low=-1, high=1, shape=(2,))`.  
  Could also be an image (pixels), a dict of features, etc.

- **Action space**  
  "What can the agent do?"  
  Example: choose one of 2 options (e.g. 0 or 1) → `Discrete(2)`.  
  Could also be continuous (e.g. "steering angle") or multi-discrete.

In our mock env we use:

- **Observation:** 2 floats in [-1, 1] (no real meaning; just for the API).
- **Action:** 0 or 1 (we ignore it in the mock; in a real env it would change the world).

---

## 4. What the environment does (in this repo)

Our **env** is the "world" in that loop. It doesn't train anything — it only:

- **reset()** → returns an observation (and info).
- **step(action)** → returns next observation, reward, terminated, truncated, info.

So the env is just the **interface** to the world: "here's what's happening (obs), here's what I did (action), tell me what happened next (obs, reward, done)." The **policy** and **training algorithm** live outside the env.

---

## 5. How training works (high level)

Training = many, many loops of "see obs → take action → get reward" and then using that **experience** to update the policy (e.g. PPO: improve probability of actions that led to more reward, reduce probability of actions that led to less).

- You run **many steps** (often millions).
- You run **many envs in parallel** (e.g. 64 copies of the same env) so you get 64 steps per "tick" — that's **vectorization**.
- A **training step** usually: collect a batch of (obs, action, reward, …) from the vectorized envs, then do one update of the policy on that batch.

So:

- **Env** = single world (reset + step).
- **Vectorized env** = many envs run in parallel, returning batches of obs/rewards/etc.
- **Trainer** (e.g. PuffeRL) = runs the loop, collects batches, updates the policy.

---

## 6. Where PufferLib fits

- **Your code:** You implement the **environment** (reset, step, observation space, action space). That's what we did in `env.py`.
- **PufferLib:**
  - **Emulation** (optional): If your env is Gymnasium-style, one wrapper makes it "PufferLib-ready" (batched, flattened if needed).
  - **Vectorization:** Runs many copies of your env in parallel and gives you batches (for speed).
  - **PuffeRL:** The training algorithm (collect experience, update policy). You give it a vectorized env and a policy; it runs the training loop.

So: **you define the world (env); PufferLib runs many copies of it and trains a policy on the experience.**

---

## 7. How our mock env fits the loop

In `env.py`:

- **SampleGymnasiumEnv** (or **SamplePufferEnv**): implements the world.
  - **reset()** → one (or a batch of) observation(s).
  - **step(action)** → next observation(s), reward(s), terminated, truncated, info.
- Our mock is **trivial**: observations are random, reward is 0, episode never ends. So there's nothing to learn — it only shows the **API** (the contract above).
- A **real** env would: keep internal state (e.g. position, health), use `action` to change that state, compute `reward` and `terminated`/`truncated` from the state.

So "how this all works":

1. **RL** = agent learns to act in an env to maximize reward.
2. **Env** = reset + step + spaces; our file defines that contract (with dummy logic).
3. **Vectorization** = many envs at once for speed.
4. **PufferLib** = vectorization + training (PuffeRL); we just provide the env.

Once you plug in a **real** env (real state, real reward, real termination), the same loop and the same library train a policy that learns from your world.
