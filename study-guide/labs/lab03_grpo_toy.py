"""Lab 03 - A complete GRPO loop on a toy "math" task (mirrors experiments/grpo/03-grpo.py).

Task: the prompt is a target number T. The model "writes" L digits (0-9), one token at a time.
Reward = 1 if the digits sum to T, else 0. Easy targets (around L*4.5) are hit by chance often;
extreme targets (e.g. 0 or 9*L) are almost never hit by a random policy -> sparse reward.

Policy: an autoregressive-ish logit table logits[target, position, digit] (no neural net needed;
the RL logic is identical to what PyTRIO does for a 4B model).

The loop is the same as the repo:
  for step:
      pick a batch of prompts
      for each prompt: sample G completions  -> rewards
      advantage = reward - group_mean   (optionally / group_std, as in agentopsd/advantages.py:60)
      skip groups where every reward is equal  ("degenerate", no signal)
      policy-gradient step on completion tokens

Experiments to run (one flag at a time!)
  python lab03_grpo_toy.py                          # baseline
  python lab03_grpo_toy.py --group-size 2           # tiny groups: many degenerate groups
  python lab03_grpo_toy.py --group-size 32          # big groups: more signal, more compute
  python lab03_grpo_toy.py --std-norm               # Dr.GRPO vs GRPO debate: divide by std?
  python lab03_grpo_toy.py --hard-only              # only extreme targets: does it learn at all?
  python lab03_grpo_toy.py --partial-credit         # reward shaping: 1 - |sum-T|/T_max
"""

import argparse

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def reward_fn(digits: np.ndarray, target: int, max_sum: int, partial: bool) -> float:
    s = int(digits.sum())
    if partial:
        return 1.0 - abs(s - target) / max_sum
    return float(s == target)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=3)
    parser.add_argument("--hard-margin", type=int, default=2, help="targets within this of 0 or 9*L count as hard")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--std-norm", action="store_true", help="divide advantage by group std (original GRPO)")
    parser.add_argument("--hard-only", action="store_true", help="train only on extreme targets")
    parser.add_argument("--partial-credit", action="store_true", help="dense reward instead of 0/1")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    L, V = args.length, 10
    max_sum = 9 * L
    targets = np.arange(0, max_sum + 1)
    if args.hard_only:
        train_targets = targets[(targets <= args.hard_margin) | (targets >= max_sum - args.hard_margin)]
    else:
        train_targets = targets
    logits = np.zeros((max_sum + 1, L, V))

    hard_mask = (targets <= args.hard_margin) | (targets >= max_sum - args.hard_margin)

    def success_prob(T: int) -> float:
        """Exact P(sum of sampled digits == T) under the current policy (convolve per-position dists)."""
        probs = softmax(logits[T])
        dist = np.array([1.0])
        for i in range(L):
            dist = np.convolve(dist, probs[i])
        return float(dist[T])

    def evaluate() -> tuple[float, float]:
        p = np.array([success_prob(T) for T in targets])
        return float(p[~hard_mask].mean()), float(p[hard_mask].mean())

    print(f"{'step':>5} | {'batch reward':>12} | {'degenerate':>10} | {'P(success) easy':>15} | {'P(success) hard':>15}")
    for step in range(args.steps):
        batch = rng.choice(train_targets, size=args.batch_size)
        grad = np.zeros_like(logits)
        rewards_all, n_degenerate, n_groups = [], 0, 0
        for T in batch:
            probs = softmax(logits[T])  # (L, V)
            samples = np.stack([[rng.choice(V, p=probs[i]) for i in range(L)] for _ in range(args.group_size)])
            rewards = np.array([reward_fn(s, T, max_sum, args.partial_credit) for s in samples])
            rewards_all.extend(rewards)
            adv = rewards - rewards.mean()
            if args.std_norm:
                adv = adv / (rewards.std() + 1e-4)
            if np.allclose(adv, 0.0):
                n_degenerate += 1
                continue
            n_groups += 1
            for s, A in zip(samples, adv):
                for i, tok in enumerate(s):  # every completion token shares the sequence advantage
                    g = -probs[i].copy()
                    g[tok] += 1.0
                    grad[T, i] += A * g / args.group_size
        logits += args.lr * grad
        if step % max(1, args.steps // 10) == 0 or step == args.steps - 1:
            easy_acc, hard_acc = evaluate()
            print(f"{step:5d} | {np.mean(rewards_all):12.3f} | {n_degenerate / args.batch_size:10.0%} | "
                  f"{easy_acc:15.3f} | {hard_acc:15.3f}")

    print("\nThings to look for:")
    print(" * 'degenerate' = groups where all G rewards were equal -> thrown away. High early (all 0)")
    print("   and late (all 1). DAPO's 'dynamic sampling' re-samples to replace these (experiments/dapo).")
    print(" * Hard targets learn last: the model must stumble on a success before GRPO can reinforce it.")
    print("   RL mostly sharpens what the base model can already do *sometimes*; that is why SFT")
    print("   warm-starts (spec-o3/train_sft.py) or teacher signals (OPD) help on very hard tasks.")
    print(" * --std-norm: a lone success in a group of 8 gets advantage ~2.6 instead of ~0.9 -> rare")
    print("   successes are amplified and hard targets learn much faster. (Critics argue it also")
    print("   over-weights near-solved / near-impossible prompts; see the Dr. GRPO paper.)")
    print(" * --partial-credit: no degenerate groups and high 'batch reward', yet exact-match success")
    print("   is LOWER. The model optimised the proxy (be close) instead of the goal (be exact).")
    print("   Reward design is half of agentic RL. See lab05_reward_hacking.py.")
    print(" * Caveat: this toy has one independent table per prompt, so nothing transfers between")
    print("   prompts (--hard-only never improves easy ones). A real LLM shares weights and does generalise.")

if __name__ == "__main__":
    main()
