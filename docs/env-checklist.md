# PufferLib env checklist — what each requirement means

The [PufferLib docs](https://puffer.ai/docs.html) give a short checklist for native PufferEnv envs. Here's what each item means in plain language and how to satisfy it.

---

## 1. Spaces and dtypes must match what you write

**What it means:** The spaces you declare (e.g. `Box(shape=(2,), dtype=float32)`) describe the **contract** between your env and the rest of the stack. Whatever you write into `self.observations` (and whatever actions you read in `step`) must have the **same shape and dtype** as those spaces. If you declare "obs is 2 floats" but then write 3 numbers, or use int instead of float, the policy and trainer get wrong or misaligned data and training can fail or behave oddly.

**What to do:** When you set `self.observations[:] = ...` (or write to reward/terminal buffers), use arrays that match `single_observation_space` / `observation_space` in shape and dtype. If you use C, the C buffers and the Python space definition must describe the same layout.

**In our mock:** We use `Box(low=-1, high=1, shape=(2,))` and assign `self.observation_space.sample()`, so shape and dtype match.

---

## 2. Resets: handle them internally; respawn if agents get stuck

**What it means:** The trainer will call `reset()` when an episode ends (terminated or truncated). Your env must correctly re-initialize state so the next episode is a fresh run. For envs that **never** "end" (e.g. infinite runner, or no clear terminal condition), the trainer might never call reset. In that case, if an agent is stuck (e.g. no reward for hundreds of steps), you should **respawn** or reset that agent yourself inside `step()` so training can make progress.

**What to do:** In `reset()`, set all internal state to a valid starting state and fill the observation buffer. In long-horizon envs without natural termination, consider a rule like "if reward == 0 for 500 steps, treat this as done / respawn the agent" so the algorithm can start new episodes and learn.

**In our mock:** We have no real state; `reset()` just fills observations. We don't implement respawn logic because we don't have a real game.

---

## 3. Zeroing: zero rewards, terminals (and any obs you don't overwrite)

**What it means:** The buffers (`self.rewards`, `self.terminals`, `self.truncations`, and sometimes parts of `self.observations`) are **reused** every step. If you don't overwrite them, they still hold **last step's values**. So if you only set `rewards` for agent 0 and leave the rest unchanged, other agents might still have old reward values. Same for terminals/truncations: if you don't set them to False at the start of the step, a previous "True" can leak into the next step and the trainer will think the episode ended when it didn't.

**What to do:** At the **start** of `step()`, before writing new logic, set:

- `self.rewards[:] = 0` (or the appropriate "no reward" value),
- `self.terminals[:] = False`,
- `self.truncations[:] = False`.

If your observation has slots you don't always fill (e.g. one-hot that you only set for one index), clear those slots too so they're not left over from the previous step.

**In our mock:** We zero rewards, terminals, and truncations at the start of `step()`, then write observations.

---

## 4. Data scale: observations and rewards roughly in [-1, 1]

**What it means:** Neural networks and many RL algorithms behave better when inputs and reward magnitudes are in a bounded range. If observations are in the thousands or rewards are +1000 sometimes, learning can be unstable or slow. Scaling (or clipping) so that observations and rewards are roughly in **[-1, 1]** (or at least not huge) usually helps.

**What to do:** When you compute the observation and reward, scale or clip them (e.g. divide by max value, or use a tanh-like mapping) so they sit roughly in [-1, 1]. You don't have to be exact; "roughly" is fine.

**In our mock:** Observation space is `Box(low=-1, high=1, shape=(2,))` and we sample from it; reward is 0. So we're in range.

---

## 5. Binding args (C envs only)

**What it means:** If you implement the env in **C** and expose it to Python via a binding, the **Python-side** code that builds the env (e.g. passes `buf`, size, seed) must pass the **same** arguments and call the **same** init function as your C code expects. Mismatched args (e.g. wrong size, wrong order) can cause wrong behavior or crashes.

**What to do:** Only relevant if you add a C backend. Keep the binding in sync with your C API: same parameters, same init call.

**In our mock:** We're pure Python; no binding.

---

## Summary table

| Requirement | In short |
|-------------|----------|
| **Spaces/dtypes** | What you write into obs/action buffers must match the shapes and dtypes you declared in the spaces. |
| **Resets** | `reset()` must start a fresh episode; if the env never "ends," respawn stuck agents (e.g. after 500 steps with no reward). |
| **Zeroing** | At the start of each `step()`, zero rewards, terminals, truncations (and any obs slots you don't fully overwrite). |
| **Scale** | Keep observations and rewards roughly in [-1, 1]. |
| **Binding (C)** | If you use C, the Python binding must pass the same args and call the same init as the C code. |
