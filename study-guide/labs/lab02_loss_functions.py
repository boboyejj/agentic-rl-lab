"""Lab 02 - importance_sampling vs PPO vs CISPO, token by token.

Pairs with experiments/loss-functions/readme.md. Each loss sees three numbers per token:
  advantage A, sampling_logprob (old policy q), target_logprob (current policy p_theta).
  ratio r = exp(target_logprob - sampling_logprob)

The only thing that matters for learning is the gradient each token sends back to
its own log-prob, d(-loss)/d(log p). We compute it analytically:

  IS    : objective = r * A                              -> grad = r * A
  PPO   : objective = min(r*A, clip(r, 1-e, 1+e) * A)     -> grad = r * A  or 0 (if the clipped branch wins)
  CISPO : objective = sg(clip(r, 1-e, 1+e)) * log p * A   -> grad = clip(r) * A   (never 0)

Part 1 prints the per-token gradient table.
Part 2 runs a small "off-policy" experiment: collect ONE batch of samples, then take many
optimizer steps on it (like many PPO epochs, or an async trainer with stale rollouts).
Watch how far each loss lets the policy drift from the policy that produced the data,
and whether that drift actually helps when the batch rewards are noisy.

Run
  python lab02_loss_functions.py
  python lab02_loss_functions.py --inner-steps 50 --lr 3   # more staleness, bigger steps
  python lab02_loss_functions.py --eps 0.28                 # DAPO-style "clip-higher"
"""

import argparse

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def token_grad(loss: str, ratio: np.ndarray, adv: np.ndarray, eps_low: float, eps_high: float) -> np.ndarray:
    """d objective / d log p_theta(token) for each token (objective = -loss)."""
    clipped = np.clip(ratio, 1 - eps_low, 1 + eps_high)
    if loss == "is":
        return ratio * adv
    if loss == "ppo":
        # min(r*A, clip(r)*A): the gradient is zero whenever the clipped term is the smaller one
        # i.e. A>0 and r>1+eps_high, or A<0 and r<1-eps_low.
        clipped_active = ((adv > 0) & (ratio > 1 + eps_high)) | ((adv < 0) & (ratio < 1 - eps_low))
        return np.where(clipped_active, 0.0, ratio * adv)
    if loss == "cispo":
        return clipped * adv
    raise ValueError(loss)


def part1(eps: float) -> None:
    print("PART 1: gradient each token sends to its own log-prob (eps = %.2f)" % eps)
    print(f"{'ratio':>6} {'A':>5} | {'IS':>7} {'PPO':>7} {'CISPO':>7}")
    for A in (+1.0, -1.0):
        for r in (0.5, 0.8, 1.0, 1.2, 1.5, 3.0):
            row = [token_grad(l, np.array([r]), np.array([A]), eps, eps)[0] for l in ("is", "ppo", "cispo")]
            print(f"{r:6.2f} {A:5.1f} | " + " ".join(f"{v:7.2f}" for v in row))
    print("  -> PPO zeroes the gradient once a token has 'moved enough' in the helpful direction.")
    print("  -> CISPO keeps every token in the update but caps its weight (good for rare, important tokens).")
    print("  -> IS has no cap: ratio 3.0 means 3x the push - unstable when data is stale.\n")


def part2(args: argparse.Namespace) -> None:
    print(f"PART 2: {args.inner_steps} optimizer steps on ONE stale batch (lr={args.lr}, eps={args.eps})")
    rng = np.random.default_rng(args.seed)
    vocab, n = args.vocab, args.batch
    reward_per_token = rng.normal(size=vocab)  # some tokens are better than others

    print(f"{'loss':<6} | {'KL(old||new)':>12} | {'max ratio':>9} | {'expected reward':>15} | {'frac tokens w/ zero grad':>24}")
    for loss in ("is", "ppo", "cispo"):
        logits = np.zeros(vocab)
        old_probs = softmax(logits)
        actions = rng.choice(vocab, size=n, p=old_probs)
        old_logp = np.log(old_probs[actions])
        # observed reward = true quality + noise (like a flaky verifier or a lucky sample)
        rewards = reward_per_token[actions] + args.reward_noise * rng.normal(size=n)
        adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        zero_frac = 0.0
        for _ in range(args.inner_steps):
            probs = softmax(logits)
            ratio = np.exp(np.log(probs[actions]) - old_logp)
            g_tok = token_grad(loss, ratio, adv, args.eps, args.eps)
            zero_frac = float(np.mean(g_tok == 0.0))
            grad = np.zeros(vocab)
            for a, g in zip(actions, g_tok):
                d = -probs.copy()
                d[a] += 1.0
                grad += g * d
            logits += args.lr * grad / n
        new_probs = softmax(logits)
        kl = float(np.sum(old_probs * (np.log(old_probs) - np.log(new_probs))))
        max_ratio = float(np.max(new_probs[actions] / old_probs[actions]))
        exp_reward = float(new_probs @ reward_per_token)
        print(f"{loss:<6} | {kl:12.4f} | {max_ratio:9.2f} | {exp_reward:15.3f} | {zero_frac:24.2f}")
    print(f"(expected reward of the starting policy = {softmax(np.zeros(vocab)) @ reward_per_token:.3f};"
          f" best possible = {reward_per_token.max():.3f})")
    print("  -> 'expected reward' uses the TRUE token quality, not the noisy batch rewards.")
    print("  -> IS / CISPO drift far from the policy that collected the data (large KL, huge ratios):")
    print("     they keep squeezing the same 64 noisy samples, so they partly fit the noise.")
    print("  -> PPO stops after ~1 trust-region step (most tokens clipped -> zero grad). Safe, but it")
    print("     needs FRESH rollouts to keep improving - which is why GRPO resamples every step.")
    print("  -> Try --reward-noise 0 (IS looks great) vs --reward-noise 3 (IS overfits), and")
    print("     --seed 1..5 to see how much the IS result varies between runs.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=0.2)
    parser.add_argument("--vocab", type=int, default=20)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--inner-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--reward-noise", type=float, default=1.5, help="std of noise added to each observed reward")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    part1(args.eps)
    part2(args)


if __name__ == "__main__":
    main()
