# Agentic RL Study Guide (built on this repo)

A step-by-step path from "what is a policy gradient?" to training tool-using, multi-turn LLM agents. It uses the 16 chapters in [`experiments/`](../experiments/README.md) as the backbone. Each phase gives you:

- **Read**: the chapter and paper to read.
- **Code**: the exact functions in this repo that carry the idea.
- **Lab**: a CPU-only toy in [`labs/`](./labs) that runs in about a second. It needs only `numpy`, with no GPU and no API key.
- **Experiment**: a real run on PyTRIO, with a hypothesis to test.
- **Check yourself**: questions you should be able to answer before moving on.

> The chapter write-ups in `experiments/*/readme.md` are mostly in Chinese. There is an English version of the handbook at <https://datawhalechina.github.io/agentic-rl-lab/en/>. The code comments are in Chinese too, but the variable names and docstrings are easy to follow, and this guide points you to the right lines.

---

## 0. The mental model (read this first)

**Agentic RL = treat the LLM as a policy, its tokens as actions, and the outside world (tools, sandboxes, games, judges) as the environment.**

```
            ┌──────────────────────────── rollout.py ────────────────────────────┐
 prompt ──► │ LLM samples tokens ──► parse (protocol.py) ──► tool / env step ──┐  │
            │      ▲                                                          │  │
            │      └──────────── observation tokens appended ◄────────────────┘  │
            └──────────────────────────────┬─────────────────────────────────────┘
                                           ▼ final trajectory
                               reward.py  (verifier / judge / teacher logprobs)
                                           ▼
                               advantages (group-relative, turn-level, critic…)
                                           ▼
                train.py: build Datum (tokens, old logprobs, advantage, MASK)
                          → forward_backward(loss_fn) → optim_step → new sampler
```

Every chapter in this repo is a variation on one or more of these boxes:

| Box | What changes across chapters |
|---|---|
| **Rollout** | single-turn (GRPO, DAPO, GSPO) → multi-turn with tools (Search-R1, ReTool, Spec-o3) → long-horizon environments (ALFWorld, TEMPO, AgentOPSD) |
| **Reward** | rule-based verifiers (GSM8K `\boxed{}`, `math_verify`) → exact match on search answers → LLM judge tournaments (Tournament-GRPO) → teacher logprobs (OPD/OPSD) |
| **Advantage** | `reward − group_mean` (GRPO) → `/std` → turn-level credit (AgentOPSD) → a generative critic with TD targets (TEMPO) |
| **Loss** | importance_sampling / PPO / CISPO (token-level) → GSPO (sequence-level) → DAPO tweaks (clip-higher, token-level averaging) |

**Why this repo is a good starting point:** it uses a *Train API* (PyTRIO, similar to Thinking Machines' Tinker). The GPU work happens remotely behind four calls:

```python
sampling_client = training_client.save_weights_and_get_sampling_client()   # current policy
sampling_client.sample(prompt, num_samples=G, ...)                         # rollouts + logprobs
training_client.forward_backward(datums, loss_fn="ppo")                     # gradients
training_client.optim_step(adam_params)                                     # update
```

You never touch FSDP, vLLM or Ray. What's left is the algorithm: the data, reward, advantage, mask and loss. That is what you're here to learn.

**Using Tinker instead of PyTRIO?** See [`tinker/`](./tinker/README.md). It has a working Tinker port of the GRPO experiment and a cheat sheet for porting the other chapters.

**Cost note:** the labs here are free. The real experiments need a PyTRIO account and credits. The chapter titles give cost hints: "two Luckin coffees for ReTool", "a Crazy-Thursday KFC meal for OPSD", "a McDonald's breakfast for Jev", "5 RMB for Vision GRPO". Always do the smoke-test commands in each `start.md` first.

---

## Setup

```bash
# labs only (any Python 3.10+ with numpy)
python3 study-guide/labs/lab01_policy_gradient.py
```

```bash
# full repo experiments
uv sync
```

```bash
trio login
```

```bash
swanlab login
```

Run the labs in order. Read the docstring at the top of each lab before you run it. It tells you what to look for and which flags to change.

---

## Phase 0: Prerequisites (1–2 weeks, skip what you know)

| Topic | Minimum you need | Suggested resource |
|---|---|---|
| LLM basics | tokenization, chat templates, autoregressive sampling, logprobs, temperature/top-p, LoRA | Karpathy "Let's build GPT" + HF LLM course (chat templates chapter) |
| RL basics | MDP, policy, return, advantage, on-/off-policy, exploration | Sutton & Barto ch. 1–3 and 13; OpenAI *Spinning Up*, "Intro to Policy Optimization" |
| PyTorch-level math | log-softmax gradients, `detach()` / stop-gradient | any autograd tutorial |

**Check yourself**
1. What does `sequence.logprobs` from the sampler contain, and why do we need it later?
2. What is the difference between `apply_chat_template(..., add_generation_prompt=True)` and `False`? (You'll need this in Phase 5. See `experiments/search-r1/protocol.py:100`.)

---

## Phase 1: Policy-gradient losses (2–3 days)

**Read**: [`experiments/loss-functions/readme.md`](../experiments/loss-functions/readme.md) (chapter 0). Paper: PPO ([1707.06347](https://arxiv.org/abs/1707.06347)).

**Key ideas**
- REINFORCE: raise the log-prob of tokens with positive advantage and lower it for tokens with negative advantage.
- **Baselines** reduce variance without adding bias.
- `ratio = exp(target_logprob − sampling_logprob)` corrects for the sampler and the learner being slightly different policies.
- PPO clips the *objective*: once a token has moved enough, its gradient drops to zero. CISPO clips the *weight* but keeps every token's gradient.

**Labs**
```bash
python3 study-guide/labs/lab01_policy_gradient.py
```

```bash
python3 study-guide/labs/lab01_policy_gradient.py --reward-offset 5
```

```bash
python3 study-guide/labs/lab02_loss_functions.py
```

- Lab 01: adding a constant to every reward makes gradient variance about 100× worse without a baseline, and leaves it unchanged with one. A per-prompt (group) baseline has lower variance than a batch baseline.
- Lab 02: a per-token gradient table for IS / PPO / CISPO, plus an experiment that takes many optimizer steps on one stale batch. PPO stops after roughly one trust-region step. IS and CISPO drift far from the policy that collected the data.

**Check yourself**
1. Why is there a minus sign in `loss = -(ratio * advantages).sum()`?
2. For A>0 and ratio=1.5 with ε=0.2, what gradient does each of IS, PPO and CISPO give? (Check with lab02 part 1.)
3. Why does async/off-policy training (e.g. `grpo/02-demo-async.py`) make clipping more important?

---

## Phase 2: GRPO with verifiable rewards (1 week)

**Read**: [`grpo`](../experiments/grpo/readme.md) → [`vision-grpo`](../experiments/vision-grpo/readme.md) → [`pytrio-jev`](../experiments/pytrio-jev/readme.md). Paper: DeepSeekMath / GRPO ([2402.03300](https://arxiv.org/abs/2402.03300)).

**Code to study (read these closely)**
| What | Where |
|---|---|
| sample G answers per prompt, score them, `advantage = reward − mean` | `experiments/grpo/03-grpo.py:227` (`run_rollout_group`), line 277 |
| right-shifted datum: `input = prompt + completion[:-1]`, zero advantage on the prompt | `experiments/grpo/03-grpo.py:283` (`build_grpo_datum`) |
| skip "degenerate" groups and log `frac_degenerate` | `experiments/grpo/03-grpo.py:477-518` |
| rule-based reward from the last `\boxed{}` | `experiments/grpo/03-grpo.py` `grade_answer` |
| the CISPO custom loss (when the server has no built-in) | `experiments/grpo/01-demo-sync.py` (`LOSS_FNS`) |

**Key ideas**
- GRPO = PPO-style loss + **group-relative advantage** instead of a learned value network.
- A group where every reward is the same carries **no signal**. That happens on prompts that are too hard (all 0) or too easy (all 1), so data difficulty matters.
- RL mostly *sharpens* behaviour the base model can already produce sometimes (pass@k → pass@1).
- Vision GRPO: the same loop with images in `ModelInput`. Jev: one forward pass, one decision token, a restricted softmax over candidate labels, so RL/SFT can train classifiers and routers too.

**Lab**
```bash
python3 study-guide/labs/lab03_grpo_toy.py                 # baseline
```

```bash
python3 study-guide/labs/lab03_grpo_toy.py --group-size 2
```

```bash
python3 study-guide/labs/lab03_grpo_toy.py --group-size 32
```

```bash
python3 study-guide/labs/lab03_grpo_toy.py --std-norm
```

```bash
python3 study-guide/labs/lab03_grpo_toy.py --partial-credit
```

What you should see: hard prompts learn last. `--std-norm` amplifies rare successes and learns hard prompts much faster. `--partial-credit` gives a higher "reward" but *lower* exact-match success, because the policy optimizes the proxy instead of the goal.

**Experiment (PyTRIO)**: *hypothesis: PPO/CISPO are more stable than plain IS*
```bash
uv run python experiments/grpo/01-demo-sync.py --steps 10 --batch-size 4 --group-size 8 \
    --max-tokens 512 --loss-fn importance_sampling --swanlab-mode online
```
Run it three times, changing only `--loss-fn` (`importance_sampling`, `ppo`, `cispo`). Compare `reward`, `frac_degenerate`, `rollout/avg_gen_len` and `loss_mean` in SwanLab. Then try `--group-size 4` vs `16` with everything else fixed.

**Check yourself**
1. Why is the prompt part of `target_tokens` padded with 0 and its advantage set to 0.0?
2. In the GSM8K run, what does a rising `frac_degenerate` late in training tell you?
3. Why might GRPO with `--std-norm` over-weight prompts that are almost solved or almost impossible? (Search for "Dr. GRPO".)

---

## Phase 3: Making GRPO work at scale: DAPO and GSPO (1 week)

**Read**: [`dapo`](../experiments/dapo/readme.md) ([2503.14476](https://arxiv.org/abs/2503.14476)), [`gspo`](../experiments/gspo/readme.md) ([2507.18071](https://arxiv.org/abs/2507.18071)).

| Fix | Problem it solves | Code |
|---|---|---|
| **Clip-higher** (ε_high=0.28 > ε_low=0.2) | entropy collapse: low-probability "exploration" tokens can't grow | `experiments/dapo/train.py:56-77` (presets), `:172` (loss) |
| **Dynamic sampling** | degenerate groups waste the batch, so keep sampling until groups have signal | `experiments/dapo/rollout.py:181` (`is_effective_group`), `train.py:691` |
| **Token-level loss averaging** | long answers get diluted in sequence-mean averaging | `experiments/dapo/train.py` (`make_ppo_loss_fn`) |
| **Soft overlong punishment** | truncated answers get a noisy reward | `--overlong-cache` in `experiments/dapo/train.py` |
| **Sequence-level ratio (GSPO)** | token-level ratios are noisy, especially for MoE / long sequences | `experiments/gspo/loss.py:133-134` (geometric mean of token ratios) |

**Lab**: `python3 study-guide/labs/lab02_loss_functions.py --eps 0.28` (watch how clip-higher changes the clipped fraction).

**Experiment**: *hypothesis: dynamic sampling costs wall-clock time but improves sample efficiency*
```bash
uv run python experiments/dapo/prepare_data.py
```

```bash
uv run python experiments/dapo/train.py --algorithm grpo --max-steps 10
```

```bash
uv run python experiments/dapo/train.py --algorithm dapo --max-steps 10
```
Compare reward per step **and** reward per wall-clock minute. The author's chapter title, "I got burned by DAPO", is about exactly this trade-off.

**Check yourself**
1. Why is GSPO's clip range (3e-4 / 4e-4) so much smaller than PPO's 0.2?
2. Dynamic sampling changes the *data distribution*. What bias could that introduce?

---

## Phase 4: Distillation as RL: OPD, OPSD (1 week)

**Read**: [`general-opd`](../experiments/general-opd/readme.md) → [`medical-opd`](../experiments/medical-opd/readme.md) → [`opsd`](../experiments/opsd/readme.md). Papers: GKD ([2306.13649](https://arxiv.org/abs/2306.13649)), MiniLLM ([2306.08543](https://arxiv.org/abs/2306.08543)), OPSD ([2601.18734](https://arxiv.org/abs/2601.18734)).

**Key ideas**
- **On-policy distillation**: the student samples, the teacher scores every token, and `advantage_t = −(log p_student − log p_teacher)` (`experiments/general-opd/01-demo-sync.py:344-345`). This is the same RL machinery with a **dense, per-token reward**.
- Reverse KL is **mode-seeking**: the student commits to what the teacher considers good. SFT (forward KL) is mode-covering and can blur strategies together.
- The student trains on *its own* states, which removes the train/test mismatch of SFT. That matters even more for agents, where one early mistake leads into states SFT never showed.
- Medical OPD: specialize in a domain while keeping general ability, using SAR-OPD and IDT-OPD with multiple teachers.
- OPSD: the teacher is a frozen copy of the student itself (step 0) given privileged context.

**Lab**
```bash
python3 study-guide/labs/lab06_distillation_kl.py
```

```bash
python3 study-guide/labs/lab06_distillation_kl.py --init-mu 0.5 --init-sigma 3   # bad init
```

**Experiment**: run `general-opd/01-demo-sync.py` with its small-cost flags (in the docstring). Watch `opd/reverse_kl_mean` fall. Then change `--kl-penalty-coef` and see how fast it falls.

**Check yourself**
1. GRPO gives one number per *sequence*. How many numbers does OPD give per sequence, and why does that matter for sample efficiency?
2. When would you choose OPD over GRPO? (Hint: do you have a good teacher, or a good verifier?)

---

## Phase 5: Becoming an agent: multi-turn tool use (2 weeks) ⭐

This is where "LLM RL" becomes "agentic RL".

**Read**: [`search-r1`](../experiments/search-r1/readme.md) ([2503.09516](https://arxiv.org/abs/2503.09516)) → [`retool`](../experiments/retool/readme.md) ([2504.11536](https://arxiv.org/abs/2504.11536)).

**Code to study, in this order**
| What | Where |
|---|---|
| tool schema, system prompt, regex parser for `<tool_call>` | `experiments/search-r1/protocol.py:1-60`, `:147` (`parse_assistant`) |
| **keeping exact sampled tokens** when appending tool results (prefix extension) | `experiments/search-r1/protocol.py:100` (`build_next_prompt`) |
| the multi-turn rollout loop, concurrent tool calls, budget limits | `experiments/search-r1/rollout.py` (`RolloutConfig`, `fit_tool_content`) |
| **the loss mask**: advantage only on assistant tokens, 0 on observations | `experiments/search-r1/train.py:62` (`build_datum`) |
| reward: +1 exact match / 0 wrong / −0.1 bad format | `experiments/search-r1/reward.py` |
| a code sandbox as the tool (timeouts, output caps, sanitizing) | `experiments/retool/sandbox.py` |
| ±1 reward with `math_verify` equivalence | `experiments/retool/reward.py` |

**Key ideas**
- **Loss masking**: never train on tokens the environment wrote. If you do, the model learns to *hallucinate tool outputs* instead of calling the tool.
- **Token alignment**: re-tokenizing text is *not* identity. Keep the sampled token IDs, or the old logprobs no longer line up with the tokens.
- Budgets (max turns, tool calls, tokens) are part of the environment design.
- Tool use **emerges** from outcome reward alone, but reward shaping can kill it (see lab 04).

**Lab**
```bash
python3 study-guide/labs/lab04_multiturn_tool_agent.py
```

```bash
python3 study-guide/labs/lab04_multiturn_tool_agent.py --search-cost 0.1
```

```bash
python3 study-guide/labs/lab04_multiturn_tool_agent.py --search-cost 0.1 --cost-start-step 150 --steps 400
```

```bash
python3 study-guide/labs/lab04_multiturn_tool_agent.py --search-acc 0.5 --max-turns 4
```
Part A prints the token mask for a real-looking Search-R1 trajectory. Part B shows four things:
1. Tool use emerges, and it is adaptive: the agent searches on unfamiliar questions and skips search on known ones.
2. A tool-call penalty from step 0 stops tool use from ever emerging.
3. The same penalty switched on later only trims wasteful calls.
4. With a flaky tool, the agent learns to retry.

**Experiments (PyTRIO)**
- *Hypothesis: more search budget → more searches and higher EM, with diminishing returns.* Run `search-r1/train.py` with `--max-search-calls 1` vs `4`. Use `search-r1/analyse.py` to compare searches per trajectory.
- *Hypothesis: the format penalty matters early.* Edit `reward.py` to give 0.0 instead of −0.1 for invalid format. Watch the invalid-format rate.
- ReTool: compare `eval.py --mode retool` before and after 20 steps. Read the sandbox metrics in SwanLab (`images/swanlab-sandbox.png`).

**Check yourself**
1. Draw the token layout of a two-search trajectory and mark which positions get advantage. (Compare with lab 04 part A.)
2. Why does `build_datum` raise an error if a turn's prompt is not a prefix extension of the previous tokens?
3. What could go wrong security-wise in `retool/sandbox.py` if you ran it on a shared machine?

---

## Phase 6: Long horizons and credit assignment (2 weeks)

**Read**: [`alfworld`](../experiments/alfworld/readme.md) ([2010.03768](https://arxiv.org/abs/2010.03768)) → [`agentopsd`](../experiments/agentopsd/readme.md) ([2608.05987](https://arxiv.org/abs/2608.05987)) → [`tempo`](../experiments/tempo/readme.md) ([Dots blog](https://studio.dots.ai/dots/tempo-blog.html)).

| Chapter | Credit-assignment idea | Code |
|---|---|---|
| ALFWorld | trajectory-level group-relative advantage over K plays of the *same* game, with 12K-token trajectories | `experiments/alfworld/advantages.py` |
| AgentOPSD | a self-teacher with a skill library scores each **turn**, and the sequence advantage is reshaped into turn-level credit (γ, λ, weight bound) | `experiments/agentopsd/advantages.py:60-160` |
| TEMPO | cut long trajectories into **macro-steps**, branch N times from saved states, and train a **generative critic** (the LLM itself estimates value by reasoning) with TD targets | `experiments/tempo/tempo.py:58` (actor signals), `:105` (critic signals) |

**Key ideas**
- With hundreds of turns, one final reward can't tell you *which* decision mattered.
- Fixes: per-turn credit from a teacher (AgentOPSD), value bootstrapping (TEMPO), and environment save/restore to branch from mid-episode states (TEMPO section 4, the "hardest engineering problem").

**Experiments**
- ALFWorld smoke test (`--max-steps 1 --games-per-batch 1`), then compare `--group-size 4` vs `8`.
- AgentOPSD: sweep `--reshape-lambda 0` (plain GRPO) vs `0.5`. Does turn-level credit help on long games?

**Check yourself**
1. Why does TEMPO need to "save and load" the environment, and why is that hard for real-world APIs?
2. What is the difference between a scalar value head and a generative critic? When is the second worth its cost?

---

## Phase 7: Open-ended rewards and real domains (1–2 weeks)

**Read**: [`tournament-grpo`](../experiments/tournament-grpo/readme.md) ([2605.26958](https://arxiv.org/abs/2605.26958)) and [`spec-o3`](../experiments/spec-o3/readme.md) ([2601.06498](https://arxiv.org/abs/2601.06498)).

- **Tournament-GRPO**: research reports have no ground truth. Responses in a group play knockout matches judged by an LLM, and wins become the reward (`experiments/tournament-grpo/reward.py`, `race.py`). Pairwise judging is more robust than absolute scores.
- **Spec-o3**: an end-to-end domain pipeline. SFT warm start → GRPO with tools on astronomy spectra. This is the template for applying agentic RL to *your* field.

**Lab**
```bash
python3 study-guide/labs/lab05_reward_hacking.py
```
A substring grader gets "hacked" by hedging (training reward 1.0, true accuracy about 0). A length-biased judge leaves accuracy unchanged but makes outputs 5× longer. **Always log a strict held-out metric next to the training reward.**

**Check yourself**
1. List three ways a policy could game an LLM-judge reward, and one mitigation for each.
2. Why does Spec-o3 do SFT *before* GRPO? (Connect to lab 03 `--hard-only` and lab 06's bad init.)

---

## Phase 8: Capstone: build your own agentic RL task (2–4 weeks)

Planning a domain research agent? See the worked example for catalysis: [`catalyst-research-agent/`](./catalyst-research-agent/README.md).

Copy the Search-R1 layout (`protocol.py`, `rollout.py`, `reward.py`, `train.py`, `eval.py`, `analyse.py`) and swap in your own environment:

1. **Pick a task with a checkable outcome**: SQL against a test DB, unit tests passing, an API returning the right value, a game score, a schema-valid JSON.
2. **Measure the base model first** (pass@1 and pass@k with `eval.py`). If pass@k ≈ 0, do SFT or distillation first. If pass@1 is already high, RL won't add much.
3. **Write the reward** and try to hack it yourself before the model does (lab 05).
4. **Smoke test** with 1 step and 1 batch. Print one full trajectory with its mask, like lab 04 part A.
5. **Train** and log: reward, held-out strict metric, `frac_degenerate`, average length, tool calls per trajectory, invalid-format rate.
6. **Ablate one thing**: group size, the loss, a reward term, or the turn budget.
7. **Write it up** like the chapters here. Contributing a chapter back (see [`CONTRIBUTING_EN.md`](../CONTRIBUTING_EN.md)) is a strong portfolio piece. The README mentions a reader who got a job with an AgentOPSD reproduction on their résumé.

Capstone ideas, sorted by difficulty:
- ⭐ **Text-to-SQL agent** on SQLite: a `run_sql` tool, reward = result set matches the reference.
- ⭐ **Calculator / unit-conversion tool** for GSM8K-style problems (ReTool-lite).
- ⭐⭐ **Docs QA agent** over a local doc corpus with a BM25 `search` tool (Search-R1 with your own backend; `search.py` already has a backend switch).
- ⭐⭐ **Bug-fix agent**: `read_file` / `edit` / `run_tests` tools on small Python repos, reward = tests pass.
- ⭐⭐⭐ **Web/GUI agent** in a simulated website (WebShop-style), with turn-level credit from AgentOPSD.
- ⭐⭐⭐ **Distill** a big model's tool-use behaviour into a 4B model with OPD on multi-turn trajectories.

---

## Experiment menu (quick reference)

| # | Where | Question | Change | Watch |
|---|---|---|---|---|
| L1 | lab01 | why baselines? | `--reward-offset 5` | grad variance |
| L2 | lab02 | IS vs PPO vs CISPO on stale data | `--inner-steps`, `--eps`, `--reward-noise` | KL, clipped fraction |
| L3 | lab03 | group size, std-norm, shaping | `--group-size`, `--std-norm`, `--partial-credit` | P(success) easy vs hard, degenerate % |
| L4 | lab04 | emergent and adaptive tool use | `--search-cost`, `--cost-start-step`, `--search-acc` | searches (known/unknown) |
| L5 | lab05 | reward hacking | `--skill` | train reward vs TRUE accuracy |
| L6 | lab06 | forward vs reverse KL | `--gap`, `--init-*` | mass in valley, teacher-likes-my-samples |
| R1 | grpo | loss comparison | `--loss-fn` | reward, loss_mean, avg_gen_len |
| R2 | grpo | group size | `--group-size 4/8/16` | frac_degenerate, reward per token spent |
| R3 | dapo | GRPO vs DAPO | `--algorithm grpo/dapo` | reward per step and per minute |
| R4 | gspo | sequence vs token ratio | compare with R3's GRPO | seq_ratio_mean, AIME25 |
| R5 | general-opd | KL coefficient | `--kl-penalty-coef` | reverse_kl_mean |
| R6 | search-r1 | search budget | `--max-search-calls 1/4` | EM, searches per trajectory |
| R7 | retool | code budget | `--max-code-calls` | pass rate, sandbox errors |
| R8 | alfworld | group size on long tasks | `--group-size` | success rate, degenerate groups |
| R9 | agentopsd | turn-level credit | `--reshape-lambda 0 / 0.5` | success rate |
| R10 | tempo | critic samples | `--critic-samples`, `--branches` | critic error, TD metrics |

**Debugging checklist for any run**
- Reward flat at 0 → task too hard, broken parser, or stop sequences cutting answers off. Print 5 raw samples.
- `frac_degenerate` near 100% → wrong difficulty. Filter data or use dynamic sampling.
- Average length exploding → length-biased reward or no overlong handling.
- Average length collapsing → entropy collapse. Try clip-higher, temperature 1.0, or check the reward.
- Reward rising but the held-out metric isn't → reward hacking (lab 05).
- `token/logprob length mismatch` errors → chat-template or re-tokenization bug (Phase 5).

---

## Where you can apply agentic RL skills

Agentic RL is worth using when **(1)** success can be checked automatically or by a reliable judge, **(2)** the base model succeeds *sometimes*, and **(3)** you can run the environment many times cheaply. Otherwise, start with prompting, SFT or distillation.

| Area | Concrete use cases | Closest chapter(s) |
|---|---|---|
| **Search and research agents** | enterprise knowledge-base QA, deep-research reports, competitive intelligence, legal/patent search | Search-R1, Tournament-GRPO |
| **Code and data agents** | data-analysis assistants that run Python, text-to-SQL, spreadsheet automation, scientific computation | ReTool |
| **Software-engineering agents** | bug fixing with unit tests as reward, test generation, migrations, CI repair | ReTool + capstone bug-fix agent |
| **Web, GUI and computer-use agents** | form filling, booking, back-office RPA, browser QA testing | ALFWorld, TEMPO, AgentOPSD |
| **Workflow and customer-support agents** | multi-step tool workflows (CRM, refunds, ticket triage) against simulated users | Search-R1 skeleton + AgentOPSD credit |
| **Domain specialization without forgetting** | medical, legal, finance assistants that keep general ability | Medical OPD |
| **Cheaper, faster models** | distill a frontier model's agent behaviour into a 4B model for on-device or high-volume serving | General OPD, OPSD |
| **Decision and routing models** | intent routing, moderation, tool/model selection, triage | PyTrio-Jev |
| **Multimodal reasoning** | charts and documents, geometry/maths tutoring, scientific data (spectra, imaging) | Vision GRPO, Spec-o3 |
| **Games, simulation, planning** | game NPCs, puzzle solvers, logistics and planning environments | ALFWorld, TEMPO |
| **Maths and reasoning** | tutoring with verifiable answers, competition maths, formal proofs (Lean) | GRPO, DAPO, GSPO |

**Roles this prepares you for:** LLM post-training / RL research engineer, agent engineer, evaluation and reward-model engineer, RL infrastructure engineer. Even if you never train a model yourself, knowing how rewards, credit assignment and environment design shape agent behaviour makes you much better at *building and evaluating* agents with prompting and tools.

---

## Suggested 10-week schedule

| Week | Phase | Deliverable |
|---|---|---|
| 1 | 0 + 1 | labs 01–02 done, and the three losses explained in your own words |
| 2 | 2 | lab 03 + GRPO run (R1), SwanLab screenshots |
| 3 | 3 | DAPO vs GRPO (R3) notes |
| 4 | 4 | lab 06 + OPD run (R5) |
| 5–6 | 5 | lab 04 + Search-R1 smoke + one ablation (R6), a hand-drawn token-mask diagram |
| 7 | 6 | ALFWorld smoke, read AgentOPSD and TEMPO code |
| 8 | 7 | lab 05 + Tournament-GRPO reading |
| 9–10 | 8 | capstone: your own environment, one ablation, a write-up |

---

## Reading list (in order)

1. PPO: [1707.06347](https://arxiv.org/abs/1707.06347)
2. DeepSeekMath / GRPO: [2402.03300](https://arxiv.org/abs/2402.03300)
3. DAPO: [2503.14476](https://arxiv.org/abs/2503.14476)
4. GSPO: [2507.18071](https://arxiv.org/abs/2507.18071)
5. GKD (on-policy distillation): [2306.13649](https://arxiv.org/abs/2306.13649), MiniLLM: [2306.08543](https://arxiv.org/abs/2306.08543)
6. OPSD: [2601.18734](https://arxiv.org/abs/2601.18734)
7. Search-R1: [2503.09516](https://arxiv.org/abs/2503.09516)
8. ReTool: [2504.11536](https://arxiv.org/abs/2504.11536)
9. ALFWorld: [2010.03768](https://arxiv.org/abs/2010.03768)
10. AgentOPSD: [2608.05987](https://arxiv.org/abs/2608.05987), TEMPO: [Dots blog](https://studio.dots.ai/dots/tempo-blog.html)
11. Tournament-GRPO: [2605.26958](https://arxiv.org/abs/2605.26958), Spec-o3: [2601.06498](https://arxiv.org/abs/2601.06498)

All links are taken from the chapter readmes in this repo, so each chapter also explains how its reproduction differs from the paper.
