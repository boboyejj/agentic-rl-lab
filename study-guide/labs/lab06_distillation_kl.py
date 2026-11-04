"""Lab 06 - On-policy distillation (OPD) vs SFT: reverse KL vs forward KL.

Pairs with experiments/general-opd (and medical-opd, opsd, agentopsd).

The teacher's output distribution is BIMODAL (two good ways to answer). The student is
too small to represent both: it is a single bump (discretised Gaussian, params mu and sigma).

  SFT / forward KL  KL(teacher || student):
      sample x ~ TEACHER, maximise log student(x).                     (off-policy, teacher data)
  OPD / reverse KL  KL(student || teacher):
      sample x ~ STUDENT, advantage = -(log student(x) - log teacher(x)),
      policy-gradient step.  This is literally general-opd/01-demo-sync.py:344-345.

What to notice
  - Forward KL is "mode covering": the student spreads out to cover both modes and puts a lot
    of mass in the valley between them, where the teacher would NEVER answer. In LLM terms:
    a blurry average of two strategies -> incoherent outputs / hallucination.
  - Reverse KL is "mode seeking": the student commits to ONE mode and samples things the
    teacher considers good. That is why on-policy distillation produces sharp, high-quality
    students, and why it fixes the train/test mismatch of SFT (the student is trained on its
    OWN samples, the states it will actually visit - crucial for multi-turn agents).
  - Per-token dense signal: OPD gives every token its own advantage, unlike GRPO's single
    sequence-level reward. Much more information per sample.

  - Reverse KL is a non-convex, on-policy objective: WHICH mode it finds depends on where the
    student starts (the student only sees its own samples). Start it wide and centred
    (--init-mu 0.5 --init-sigma 3) and it can get stuck straddling both modes. This is why
    OPD usually starts from an SFT'd student (experiments/medical-opd/02-medical-sft.py).

Run
  python lab06_distillation_kl.py
  python lab06_distillation_kl.py --gap 2                       # modes close: methods agree
  python lab06_distillation_kl.py --init-mu 0.5 --init-sigma 3  # bad init for reverse KL
"""

import argparse

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max()
    e = np.exp(z)
    return e / e.sum()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=int, default=61)
    parser.add_argument("--gap", type=float, default=6.0, help="distance between the teacher's two modes")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--init-mu", type=float, default=1.5)
    parser.add_argument("--init-sigma", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    xs = np.linspace(-10, 10, args.bins)
    teacher = 0.5 * np.exp(-((xs + args.gap / 2) ** 2) / 2) + 0.5 * np.exp(-((xs - args.gap / 2) ** 2) / 2)
    teacher = teacher / teacher.sum()
    log_t = np.log(teacher + 1e-12)

    def student_probs(mu: float, log_sigma: float) -> np.ndarray:
        return softmax(-((xs - mu) ** 2) / (2 * np.exp(2 * log_sigma)))

    def grad_log_student(mu: float, log_sigma: float, idx: np.ndarray) -> np.ndarray:
        """d log s(x_idx) / d(mu, log_sigma), analytic."""
        s = student_probs(mu, log_sigma)
        var = np.exp(2 * log_sigma)
        dlogit_mu = (xs - mu) / var
        dlogit_ls = (xs - mu) ** 2 / var
        g_mu = dlogit_mu[idx] - s @ dlogit_mu
        g_ls = dlogit_ls[idx] - s @ dlogit_ls
        return np.stack([g_mu, g_ls], axis=-1)

    results = {}
    for method in ("forward_kl_sft", "reverse_kl_opd"):
        rng = np.random.default_rng(args.seed)
        mu, log_sigma = args.init_mu, np.log(args.init_sigma)  # base student leans towards one strategy
        for _ in range(args.steps):
            s = student_probs(mu, log_sigma)
            if method == "forward_kl_sft":
                idx = rng.choice(args.bins, size=args.samples, p=teacher)
                weights = np.ones(args.samples)  # plain maximum likelihood on teacher samples
            else:
                idx = rng.choice(args.bins, size=args.samples, p=s)
                reverse_kl = np.log(s[idx] + 1e-12) - log_t[idx]
                weights = -reverse_kl
                weights = weights - weights.mean()  # baseline, same trick as GRPO
            g = (weights[:, None] * grad_log_student(mu, log_sigma, idx)).mean(axis=0)
            mu += args.lr * g[0]
            log_sigma += args.lr * g[1]
        s = student_probs(mu, log_sigma)
        valley = np.abs(xs) < args.gap / 4
        results[method] = (mu, np.exp(log_sigma), float(s @ teacher / teacher.max()), float(s[valley].sum()),
                           float(np.sum(s * (np.log(s + 1e-12) - log_t))),
                           float(np.sum(teacher * (log_t - np.log(s + 1e-12)))))

    print(f"teacher modes at {-args.gap / 2:+.1f} and {+args.gap / 2:+.1f}\n")
    print(f"{'method':<16} | {'mu':>6} | {'sigma':>5} | {'teacher-likes-my-samples':>24} | {'mass in valley':>14} | "
          f"{'KL(s||t)':>8} | {'KL(t||s)':>8}")
    for m, (mu, sig, quality, valley, rkl, fkl) in results.items():
        print(f"{m:<16} | {mu:6.2f} | {sig:5.2f} | {quality:24.2f} | {valley:14.2%} | {rkl:8.3f} | {fkl:8.3f}")
    print("\n'teacher-likes-my-samples' = E_student[teacher(x)] / max teacher(x)  (higher = better outputs)")
    print("Each method wins on the KL it optimises - but only reverse KL yields outputs the teacher endorses.")
    print("\nASCII view (T = teacher, F = forward-KL student, R = reverse-KL student):")
    for name, dist in (("T", teacher), ("F", student_probs(*_params(results["forward_kl_sft"]))),
                       ("R", student_probs(*_params(results["reverse_kl_opd"])))):
        peak = dist.max()
        row = "".join(" .:-=+*#%@"[min(9, int(9 * v / peak))] for v in dist)
        print(f"  {name} |{row}|")


def _params(result: tuple) -> tuple[float, float]:
    mu, sigma = result[0], result[1]
    return mu, float(np.log(sigma))


if __name__ == "__main__":
    main()
