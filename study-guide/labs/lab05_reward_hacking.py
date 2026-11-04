"""Lab 05 - Reward hacking: the policy optimises the grader, not your intention.

The model can answer in one of several STYLES. Its true skill is fixed (it knows the right
answer with probability --skill); what RL can change is which style it picks.

  honest    : "Answer: <its best guess>"                     correct w.p. skill
  hedge     : "Answer: A or B or C or D" (lists every option)
  verbose   : long essay, then "Answer: <best guess>"        correct w.p. skill, 5x the tokens
  no_format : rambles, never writes an Answer line

Graders (all of these exist in the wild):
  strict     : exactly one 'Answer:' line, normalised exact match.       (search-r1/reward.py)
  substring  : 1 if the reference string appears anywhere in the output.  (a tempting shortcut)
  judge_len  : an "LLM judge" that likes correct answers but also likes long, thorough-looking
               responses: 0.7*correct + 0.3*length_score.                  (known judge bias)

What to notice
  - Under 'substring' the policy converges to 'hedge'. Training reward -> ~1.0 while TRUE
    accuracy (always measured with the strict grader) stays at chance. Classic hack.
  - Under 'judge_len' true accuracy is unchanged, but the policy goes 'verbose': same answers,
    5x the tokens. In real runs this shows up as avg_gen_len creeping up (watch
    `rollout/avg_gen_len` in experiments/grpo). Tournament-GRPO (experiments/tournament-grpo)
    uses pairwise judging partly to reduce such absolute-score biases.
  - Always log a held-out, strict metric next to the training reward.

Run
  python lab05_reward_hacking.py
  python lab05_reward_hacking.py --skill 0.8
"""

import argparse

import numpy as np

STYLES = ["honest", "hedge", "verbose", "no_format"]
TOKENS = {"honest": 20, "hedge": 25, "verbose": 100, "no_format": 60}
N_OPTIONS = 4


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def simulate(style: str, skill: float, rng: np.random.Generator) -> dict:
    """Return what a grader can observe about one response."""
    knows = rng.random() < skill
    guess_correct = knows or rng.random() < 1 / N_OPTIONS
    if style in ("honest", "verbose"):
        return {"has_answer_line": True, "single_answer": True, "correct_single": guess_correct,
                "reference_in_text": guess_correct, "tokens": TOKENS[style]}
    if style == "hedge":
        return {"has_answer_line": True, "single_answer": False, "correct_single": False,
                "reference_in_text": True, "tokens": TOKENS[style]}
    # no_format: the right answer may still be mentioned somewhere in the rambling
    return {"has_answer_line": False, "single_answer": False, "correct_single": False,
            "reference_in_text": guess_correct and rng.random() < 0.5, "tokens": TOKENS[style]}


def grade(grader: str, obs: dict) -> float:
    if grader == "strict":
        if not obs["has_answer_line"]:
            return -0.1
        return float(obs["single_answer"] and obs["correct_single"])
    if grader == "substring":
        return float(obs["reference_in_text"])
    if grader == "judge_len":
        correct = float(obs["single_answer"] and obs["correct_single"])
        return 0.7 * correct + 0.3 * min(obs["tokens"] / 100, 1.0)
    raise ValueError(grader)


def train(grader: str, args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    logits = np.zeros(len(STYLES))
    for _ in range(args.steps):
        grad = np.zeros_like(logits)
        for _ in range(args.batch_size):
            probs = softmax(logits)
            styles = rng.choice(len(STYLES), size=args.group_size, p=probs)
            obs = [simulate(STYLES[s], args.skill, rng) for s in styles]
            rewards = np.array([grade(grader, o) for o in obs])
            adv = rewards - rewards.mean()
            for s, A in zip(styles, adv):
                g = -probs.copy()
                g[s] += 1.0
                grad += A * g
        logits += args.lr * grad / (args.batch_size * args.group_size)

    probs = softmax(logits)
    eval_rng = np.random.default_rng(123)
    train_r, true_acc, toks = [], [], []
    for _ in range(4000):
        s = STYLES[eval_rng.choice(len(STYLES), p=probs)]
        o = simulate(s, args.skill, eval_rng)
        train_r.append(grade(grader, o))
        true_acc.append(float(o["single_answer"] and o["correct_single"]))
        toks.append(o["tokens"])
    mix = "  ".join(f"{name}={p:.2f}" for name, p in zip(STYLES, probs))
    print(f"{grader:<10} | train reward {np.mean(train_r):5.2f} | TRUE accuracy {np.mean(true_acc):5.2f} | "
          f"avg tokens {np.mean(toks):5.1f} | {mix}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skill", type=float, default=0.4, help="P(model genuinely knows the answer)")
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    honest_acc = args.skill + (1 - args.skill) / N_OPTIONS
    print(f"An always-honest policy would have TRUE accuracy {honest_acc:.2f}.\n")
    for grader in ("strict", "substring", "judge_len"):
        train(grader, args)


if __name__ == "__main__":
    main()
