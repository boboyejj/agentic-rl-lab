"""Lab 01 - Policy gradient from scratch: why we need a baseline (and why GRPO uses a group).

Setting: a "one-token LLM". For each prompt the model picks one of V answers
(think: the token after "The answer is"). Exactly one answer is correct -> reward 1, else 0.
The policy is a softmax over a logit table: pi(a | prompt) = softmax(logits[prompt]).

We compare three ways to turn a reward into a gradient:
  1. "none"  : REINFORCE,           grad = r * grad log pi(a)
  2. "mean"  : batch-mean baseline, grad = (r - mean_batch(r)) * grad log pi(a)
  3. "group" : GRPO-style,          sample G answers *for the same prompt* and use
               r - mean_group(r)   (this is exactly experiments/grpo/03-grpo.py:277)

What to notice
  - All three estimators point the same way *on average*; they differ in VARIANCE.
  - With --reward-offset 5 (add a constant to every reward) "none" breaks: its gradient
    variance explodes ~100x and learning on hard prompts collapses, while "mean" and
    "group" are unchanged. A baseline makes the update invariant to reward shifts.
  - "group" has lower variance than the batch-mean baseline because it is a *per-prompt*
    baseline (easy prompts don't inflate the baseline for hard ones) - and it needs no
    value network. That is the whole trick behind GRPO.
  - But "group" is slightly SLOWER on the hard prompts at offset 0. Why? When all G samples
    of a hard prompt fail, every advantage is 0 -> no signal at all ("degenerate group",
    the `frac_degenerate` metric in experiments/grpo/03-grpo.py:518). Try --group-size 32.

Run
  python lab01_policy_gradient.py
  python lab01_policy_gradient.py --vocab 50 --steps 300     # harder: sparser reward
  python lab01_policy_gradient.py --reward-offset 5           # every reward +5: watch "none" suffer
"""

import argparse

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def run(method: str, args: argparse.Namespace) -> tuple[list[float], float]:
    rng = np.random.default_rng(args.seed)
    n_prompts, vocab, group = args.prompts, args.vocab, args.group_size
    correct = rng.integers(0, vocab, size=n_prompts)
    logits = np.zeros((n_prompts, vocab))
    # Mixed difficulty: half the prompts are "easy" (the base model already likes the right answer).
    easy = np.arange(n_prompts) < int(n_prompts * args.easy_frac)
    logits[easy, correct[easy]] = args.easy_bonus
    accuracy_curve: list[float] = []
    grad_estimates: list[np.ndarray] = []

    for _ in range(args.steps):
        prompts = rng.integers(0, n_prompts, size=args.batch_size)
        grad = np.zeros_like(logits)
        all_rewards, samples = [], []
        for p in prompts:
            probs = softmax(logits[p])
            actions = rng.choice(vocab, size=group, p=probs)
            rewards = (actions == correct[p]).astype(float) + args.reward_offset
            samples.append((p, probs, actions, rewards))
            all_rewards.extend(rewards)
        batch_mean = float(np.mean(all_rewards))

        for p, probs, actions, rewards in samples:
            if method == "none":
                adv = rewards
            elif method == "mean":
                adv = rewards - batch_mean
            else:  # group
                adv = rewards - rewards.mean()
            for a, A in zip(actions, adv):
                # d log softmax(logits)[a] / d logits = onehot(a) - probs
                g = -probs.copy()
                g[a] += 1.0
                grad[p] += A * g
        grad /= args.batch_size * group
        logits += args.lr * grad

        acc = softmax(logits)[np.arange(n_prompts), correct]
        accuracy_curve.append(float(np.mean(acc[~easy])) if (~easy).any() else float(np.mean(acc)))
    return accuracy_curve


def gradient_variance(method: str, args: argparse.Namespace, repeats: int = 300) -> float:
    """Estimate the variance of the one-batch gradient estimator at initialisation.

    Lower variance = each step is a more reliable signal = you can use bigger steps / fewer samples.
    """
    rng = np.random.default_rng(args.seed + 1)
    n_prompts, vocab, group = args.prompts, args.vocab, args.group_size
    correct = rng.integers(0, vocab, size=n_prompts)
    logits = np.zeros((n_prompts, vocab))
    easy = np.arange(n_prompts) < int(n_prompts * args.easy_frac)
    logits[easy, correct[easy]] = args.easy_bonus
    estimates = []
    for _ in range(repeats):
        prompts = rng.integers(0, n_prompts, size=args.batch_size)
        grad = np.zeros_like(logits)
        samples, all_rewards = [], []
        for p in prompts:
            probs = softmax(logits[p])
            actions = rng.choice(vocab, size=group, p=probs)
            rewards = (actions == correct[p]).astype(float) + args.reward_offset
            samples.append((p, probs, actions, rewards))
            all_rewards.extend(rewards)
        batch_mean = float(np.mean(all_rewards))
        for p, probs, actions, rewards in samples:
            adv = {"none": rewards, "mean": rewards - batch_mean, "group": rewards - rewards.mean()}[method]
            for a, A in zip(actions, adv):
                g = -probs.copy()
                g[a] += 1.0
                grad[p] += A * g
        estimates.append((grad / (args.batch_size * group)).ravel())
    est = np.stack(estimates)
    return float(est.var(axis=0).sum())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", type=int, default=20)
    parser.add_argument("--vocab", type=int, default=10)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--reward-offset", type=float, default=0.0,
                        help="add a constant to every reward (should NOT matter, but does without a baseline)")
    parser.add_argument("--easy-frac", type=float, default=0.5, help="fraction of prompts that start easy")
    parser.add_argument("--easy-bonus", type=float, default=4.0, help="initial logit bonus on the right answer for easy prompts")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    checkpoints = [s for s in (10, 50, 100, args.steps) if s <= args.steps]
    print(f"{'method':<8} | " + " | ".join(f"step {s:>4}" for s in checkpoints) + " | grad variance @ init")
    print("-" * 76)
    for method in ("none", "mean", "group"):
        curve = run(method, args)
        var = gradient_variance(method, args)
        cols = [curve[s - 1] for s in checkpoints]
        print(f"{method:<8} | " + " | ".join(f"{c:9.3f}" for c in cols) + f" | {var:.2e}")
    print("\nColumns = mean probability of the correct answer on the HARD prompts.")
    print("Last column = variance of the gradient estimate (lower = less noisy updates).")
    print("Try --reward-offset 5 (shifting rewards by a constant should not matter... but it does without a baseline).")


if __name__ == "__main__":
    main()
