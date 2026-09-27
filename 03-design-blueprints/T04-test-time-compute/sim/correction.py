"""Self-correction, with and without an oracle outside the model.

The corpus's negative result is qualitative: "intrinsic self-correction often
fails without external feedback. And um performance can even degrade uh quite
frequently if you're just asking it to self-correct itself" -- and the lecture
gives no benchmark, no accuracy figure and no author attribution for the
degradation. So this module does NOT model an amount of degradation.

What it does instead is state the mechanism as two rates and derive the
CONDITION under which accuracy falls, so the design can be argued without a
number that the corpus does not supply:

    a correct answer is flipped to wrong with probability f_c
    a wrong answer is flipped to correct with probability f_w

    acc' = acc(1 - f_c) + (1 - acc) f_w
    acc' < acc   <=>   f_c / f_w  >  (1 - acc) / acc

The right-hand threshold is small when the model is accurate. At acc = 0.8 a
corrector that damages a correct answer at a quarter the rate it repairs a wrong
one is already net-negative. That is the whole argument for gating correction
on an external oracle, and it needs no measured constant.

The corpus's asymmetry is the reason f_c is not zero: "things that are difficult
for the models to do are things that are also difficult for them to check", plus
"confirmation bias... a tendency to reinforce initial reasoning".
"""


def accuracy_after(acc, f_c, f_w, rounds=1):
    """acc after `rounds` of intrinsic self-correction."""
    for _ in range(rounds):
        acc = acc * (1.0 - f_c) + (1.0 - acc) * f_w
    return acc


def breaks_even(acc):
    """The f_c / f_w ratio above which intrinsic self-correction loses accuracy.

    Returns the ratio. Below it, correction helps; above it, it hurts.
    """
    if acc <= 0.0:
        return float("inf")
    if acc >= 1.0:
        return 0.0
    return (1.0 - acc) / acc


def oracle_accuracy_after(acc, recovery, rounds=1):
    """Self-Debugging: only wrong answers are touched, so f_c is structurally 0.

    A wrong answer becomes correct with probability `recovery`; a correct answer
    is never modified, because the executor is an oracle the model is not. This
    is why accuracy here can only rise.
    """
    for _ in range(rounds):
        acc = acc + (1.0 - acc) * recovery
    return acc


def cost_of_correction(samples_per_round, rounds):
    """Self-consistency needs many samples; self-debugging needs one plus a run.

    The corpus's comparison is self-consistency at 16 samples against
    self-debugging at 1 -- so the interesting quantity is not samples per round
    but total generations across the whole attempt.
    """
    return samples_per_round * rounds


def compare_strategies(acc, f_c, f_w, recovery, rounds):
    """The arms the design chooses between, on identical inputs."""
    intrinsic = accuracy_after(acc, f_c, f_w, rounds)
    oracle = oracle_accuracy_after(acc, recovery, rounds)
    return {
        "base": acc,
        "intrinsic": intrinsic,
        "oracle": oracle,
        "intrinsic_delta": intrinsic - acc,
        "oracle_delta": oracle - acc,
        "break_even_ratio": breaks_even(acc),
        "f_c_over_f_w": (f_c / f_w) if f_w > 0.0 else float("inf"),
    }
