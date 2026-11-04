"""Lab 04 - From single-turn RL to an agent: multi-turn tool calls, loss masks, adaptive tool use.

Pairs with experiments/search-r1 (and retool / alfworld, which share the same skeleton).

PART A - the loss mask (the #1 implementation detail of agentic RL)
  A multi-turn trajectory interleaves tokens the MODEL wrote (assistant turns) with tokens the
  ENVIRONMENT wrote (system prompt, user question, tool results). We only train on the former.
  We rebuild the exact datum layout of experiments/search-r1/train.py:62 (build_datum) on a
  fake trajectory and print which positions carry an advantage.

PART B - RL on a toy search environment
  Each episode the agent gets a question. Some questions it already "knows" (parametric memory
  leans towards the right answer); others are unfamiliar, with a fresh random answer each time.
  Per turn the agent chooses: SEARCH, or ANSWER_k.
    SEARCH -> returns the true answer with prob --search-acc, otherwise "no result".
    Reward: +1 correct, 0 wrong, -0.1 if it never answers within --max-turns (format penalty,
    same as search-r1/reward.py), minus --search-cost per search call.
  Trajectory-level group-relative advantage, applied to every agent action (turn) in it.

What to notice
  1. (default) Tool use EMERGES from outcome reward alone - nobody tells the agent to search -
     and it is ADAPTIVE: ~0 searches on known questions, ~1 on unfamiliar ones.
  2. --search-cost 0.1 from step 0: tool use never emerges! At the start the agent can't yet
     read search results, so searching = paying a cost for nothing -> it is punished away
     before it can become useful. An exploration trap you will meet in real agent RL
     (e.g. penalising tool calls or long outputs too early).
  3. Same cost, but switched on later (--cost-start-step 150): the skill is learned first,
     and the cost now only trims wasteful calls. Curriculum / reward scheduling matters.
  4. Flaky tool (--search-acc 0.5 --max-turns 4): it learns to retry, ~1.8 searches per question,
     and occasionally runs out of turns (the "no answer" column).

Run
  python lab04_multiturn_tool_agent.py
  python lab04_multiturn_tool_agent.py --search-cost 0.1
  python lab04_multiturn_tool_agent.py --search-cost 0.1 --cost-start-step 150 --steps 400
  python lab04_multiturn_tool_agent.py --search-acc 0.5 --max-turns 4
"""

import argparse

import numpy as np


# --------------------------------------------------------------------------------------
# PART A: build a multi-turn datum and show the mask
# --------------------------------------------------------------------------------------

def part_a() -> None:
    print("PART A: which tokens get trained in a multi-turn trajectory?\n")
    # (source, text). In the real code these are token ids; here 1 word = 1 token.
    segments = [
        ("env", "<system> answer with help of search </system> <user> Where was the author of The Little Prince born? </user> <assistant>"),
        ("model", "I need the author. <tool_call> search: Little Prince author </tool_call>"),
        ("env", "</assistant> <tool> Antoine de Saint-Exupery wrote it... </tool> <assistant>"),
        ("model", "<tool_call> search: Saint-Exupery birthplace </tool_call>"),
        ("env", "</assistant> <tool> born in Lyon, France </tool> <assistant>"),
        ("model", "Answer: France"),
    ]
    trajectory_advantage = 0.75  # reward(1.0) - group mean(0.25), shared by every model token
    tokens, advantages = [], []
    for source, text in segments:
        words = text.split()
        tokens += words
        advantages += [trajectory_advantage if source == "model" else 0.0] * len(words)

    # Right-shift exactly like the repo: input = tokens[:-1], target = tokens[1:], adv = adv[1:]
    inputs, targets, adv = tokens[:-1], tokens[1:], advantages[1:]
    assert len(inputs) == len(targets) == len(adv)
    line_tok, line_mask = [], []
    for t, a in zip(targets, adv):
        w = max(len(t), 1)
        line_tok.append(t.ljust(w))
        line_mask.append(("#" if a != 0 else ".").ljust(w))
    width = 100
    tok_str, mask_str = " ".join(line_tok), " ".join(line_mask)
    for i in range(0, len(tok_str), width):
        print("  target: " + tok_str[i:i + width])
        print("  train?: " + mask_str[i:i + width])
    n_train = sum(1 for a in adv if a != 0)
    print(f"\n  {n_train}/{len(adv)} target positions are trained ('#'); the rest have advantage 0.")
    print("  If you forget the mask, the model is trained to *predict tool outputs* - i.e. to")
    print("  hallucinate search results instead of calling the tool. Classic agentic-RL bug.")
    print("  Also see search-r1/protocol.py:100 (build_next_prompt): the real code keeps the")
    print("  EXACT sampled tokens and appends observations, so old logprobs stay aligned.\n")


# --------------------------------------------------------------------------------------
# PART B: RL in a toy search environment
# --------------------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def part_b(args: argparse.Namespace) -> None:
    print("PART B: RL on a toy search environment\n")
    rng = np.random.default_rng(args.seed)
    V = args.vocab
    SEARCH = V  # action index; answers are 0..V-1
    n_actions = V + 1
    # The policy sees a "context", not a question id, so it cannot simply memorise answers:
    #   context k in 0..V-1 : a question whose answer the model already knows is k (parametric memory)
    #   context V           : an unfamiliar question; its answer is random every episode
    n_ctx = V + 1
    UNKNOWN = V
    # observation states: 0 = nothing seen yet, 1 = searched but no result, 2+k = search said k
    n_obs = 2 + V
    logits = np.zeros((n_ctx, n_obs, n_actions))
    for k in range(V):
        logits[k, 0, k] = args.memory_strength  # the base model leans towards what it knows

    def rollout(known: bool, search_cost: float) -> tuple[int, list[tuple[int, int]], float, int, bool, bool]:
        answer = int(rng.integers(0, V))
        ctx = answer if known else UNKNOWN
        obs, steps, searches = 0, [], 0
        for _ in range(args.max_turns):
            a = rng.choice(n_actions, p=softmax(logits[ctx, obs]))
            steps.append((obs, a))
            if a == SEARCH:
                searches += 1
                obs = 2 + answer if rng.random() < args.search_acc else 1
                continue
            correct = a == answer
            return ctx, steps, float(correct) - search_cost * searches, searches, True, correct
        return ctx, steps, -0.1 - search_cost * searches, searches, False, False

    print(f"{'step':>5} | {'correct':>7} | {'no answer':>9} | {'searches (known)':>16} | {'searches (unknown)':>18}")
    for step in range(args.steps):
        search_cost = args.search_cost if step >= args.cost_start_step else 0.0
        grad = np.zeros_like(logits)
        correct, no_answer, s_known, s_unknown = [], [], [], []
        for _ in range(args.batch_size):
            known = rng.random() < args.memorised_frac
            group = [rollout(known, search_cost) for _ in range(args.group_size)]
            rewards = np.array([g[2] for g in group])
            adv = rewards - rewards.mean()
            for (ctx, steps, r, searches, answered, ok), A in zip(group, adv):
                correct.append(ok)
                no_answer.append(not answered)
                (s_known if known else s_unknown).append(searches)
                # Only the agent's own actions get gradient. Observations (tool results) are
                # inputs to the next decision, never targets - this IS the loss mask.
                for obs, a in steps:
                    p = softmax(logits[ctx, obs])
                    g = -p
                    g[a] += 1.0
                    grad[ctx, obs] += A * g / args.group_size
        logits += args.lr * grad / args.batch_size * 4
        if step == args.cost_start_step and step > 0:
            print(f"----- search cost {args.search_cost} switched on -----")
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            print(f"{step:5d} | {np.mean(correct):7.2%} | {np.mean(no_answer):9.2%} | "
                  f"{np.mean(s_known) if s_known else float('nan'):16.2f} | "
                  f"{np.mean(s_unknown) if s_unknown else float('nan'):18.2f}")
    print("\nExpected: 'searches (known)' stays ~0 (answer from memory, save the cost) while")
    print("'searches (unknown)' climbs to >= 1: the agent learned WHEN a tool call is worth it.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab", type=int, default=8)
    parser.add_argument("--memorised-frac", type=float, default=0.5)
    parser.add_argument("--memory-strength", type=float, default=2.0)
    parser.add_argument("--search-acc", type=float, default=0.9)
    parser.add_argument("--search-cost", type=float, default=0.0)
    parser.add_argument("--cost-start-step", type=int, default=0,
                        help="charge --search-cost only from this step on (a simple curriculum)")
    parser.add_argument("--max-turns", type=int, default=3)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-part-a", action="store_true")
    args = parser.parse_args()
    if not args.skip_part_a:
        part_a()
    part_b(args)


if __name__ == "__main__":
    main()
