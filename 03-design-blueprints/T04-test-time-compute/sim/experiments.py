"""The experiments run.py executes.

Every count printed here is this simulation's own, over toy problems with hand-
chosen probabilities and a fixed RNG seed. The corpus's figures -- 100 samples
= 100x cost, the 0.95 threshold, alpha = 3 giving 0.5/0.25/0.25, v^100 latent
terms, the R1 AIME curve -- are quoted and attributed in HLD.md and are NOT
reproduced by this model.
"""

from . import adaptive, cot, correction, length

SEED = 20260927
TRIALS = 4000
# Each adaptive experiment runs TRIALS // SAMPLE_DIVISOR draws per row.
SAMPLE_DIVISOR = 4


# ------------------------------------------------------------------ 1

def exp_latent_marginal():
    """Greedy, joint argmax and the true marginal disagree on the corpus's case.

    The corpus's counterexample is 3 ft in inches. On this fixture the highest-
    probability chain is the centimetres one and the highest-probability joint
    (z, y) pair is on it too -- but the sum over chains prefers 36, because the
    two inch chains agree with each other.
    """
    rows = []
    for name in ("3ft_in_inches", "2plus3"):
        m = cot.marginal(name)
        g_chain, g_ans = cot.greedy(name)
        j_chain, j_ans, j_p = cot.joint_argmax(name)
        m_ans = cot.marginal_argmax(name)
        correct = cot.correct_answer(name)
        rows.append({
            "problem": name,
            "chains": [(z, cot.chain_prob(name, z)) for z in sorted(cot.chains(name))],
            "marginal": m,
            "correct": correct,
            "greedy": (g_chain, g_ans),
            "joint": (j_chain, j_ans, j_p),
            "marginal_argmax": m_ans,
            "greedy_right": g_ans == correct,
            "joint_right": j_ans == correct,
            "marginal_right": m_ans == correct,
        })
    return rows


# ------------------------------------------------------------------ 2

def exp_self_consistency():
    """Majority vote over n samples: accuracy rises toward the marginal, at n x cost.

    The corpus states the price plainly: 100 samples is 100x the inference cost.
    This measures what accuracy is bought for it on a contested question and on
    an easy one.
    """
    # More trials than the adaptive experiments: the interesting differences
    # between adjacent n are only a couple of points, so 1,000 draws would leave
    # the table dominated by sampling noise.
    trials = 3000
    rows = []
    for name in ("3ft_in_inches", "2plus3"):
        correct = cot.correct_answer(name)
        for n in (1, 2, 4, 8, 16, 32, 64):
            rng = cot.make_rng(SEED + n)
            hits = 0
            for _ in range(trials):
                ans, _ = cot.self_consistency(name, rng, n)
                if ans == correct:
                    hits += 1
            rows.append({
                "problem": name,
                "n": n,
                "accuracy": hits / float(trials),
                "cost": adaptive.cost_multiplier(n),
                "trials": trials,
            })
    return rows


# ------------------------------------------------------------------ 3

def exp_dirichlet_worked_example():
    """The corpus's own arithmetic: alpha = 3 with counts 1,0,0 gives 0.5/0.25/0.25.

    Also the boundary case the corpus names: "If alpha is zero this is maximum
    likelihood estimation."
    """
    counts = {"A": 1, "B": 0, "C": 0}
    rows = []
    for alpha in (0.0, 1.0, 3.0, 10.0, 30.0):
        post, pseudo = adaptive.dirichlet_posterior(counts, alpha)
        rows.append({
            "alpha": alpha,
            "pseudo": pseudo,
            "posterior": post,
            "is_mle": alpha == 0.0,
        })
    return rows


# ------------------------------------------------------------------ 4

def exp_adaptive_stopping():
    """What the Beta rule actually saves, per problem.

    The finding worth carrying: the rule is conservative at alpha = 3, so it
    stops early only on a near-unanimous run -- which easy questions produce and
    contested questions by definition do not. The saving therefore lands on the
    questions that did not need it, and the contested ones run to the cap. The
    corpus gives no samples-saved figure; this is the model's own.
    """
    rows = []
    for name in ("2plus3", "3ft_in_inches", "trap"):
        correct = cot.correct_answer(name)
        rng = cot.make_rng(SEED)
        samples = 0
        hits = 0
        early = 0
        early_wrong = 0
        for _ in range(TRIALS // 4):
            r = adaptive.adaptive_self_consistency(name, rng, threshold=0.95, cap=16)
            samples += r["samples"]
            if r["answer"] == correct:
                hits += 1
            if r["stopped_early"]:
                early += 1
                if r["answer"] != correct:
                    early_wrong += 1
        trials = TRIALS // 4
        rows.append({
            "problem": name,
            "mean_samples": samples / float(trials),
            "accuracy": hits / float(trials),
            "early_rate": early / float(trials),
            "early_wrong_rate": early_wrong / float(trials),
            "cap_fraction": 1.0 - early / float(trials),
            "saving_vs_16": 1.0 - (samples / float(trials)) / 16.0,
        })
    return rows


# ------------------------------------------------------------------ 5

def exp_confidently_wrong():
    """The blind spot, and whether a sample floor fixes it.

    On the trap fixture the wrong answer sits in one high-probability chain, so
    a short unlucky run is unanimous and wrong. The Beta rule reads unanimity as
    confidence and stops. Raising the floor buys protection at a cost in
    samples; it does not remove the failure, because a long enough unlucky run
    still looks unanimous.
    """
    name = "trap"
    correct = cot.correct_answer(name)
    rows = []
    for floor in (2, 4, 8, 12, 16):
        rng = cot.make_rng(SEED)
        samples = 0
        hits = 0
        wrong_confident = 0
        for _ in range(TRIALS // 4):
            r = adaptive.adaptive_self_consistency(
                name, rng, threshold=0.95, cap=max(floor, 16),
                min_batches=max(1, floor // 2))
            samples += r["samples"]
            if r["answer"] == correct:
                hits += 1
            elif r["confidence"] >= 0.95:
                wrong_confident += 1
        trials = TRIALS // 4
        rows.append({
            "floor": floor,
            "mean_samples": samples / float(trials),
            "accuracy": hits / float(trials),
            "wrong_and_confident": wrong_confident / float(trials),
        })
    return rows


# ------------------------------------------------------------------ 6

def exp_self_correction():
    """When intrinsic self-correction loses, and why the oracle does not.

    The corpus's negative result is qualitative -- no accuracy figure is given --
    so the model fixes the RATES and derives the CONDITION instead:

        acc' = acc(1 - f_c) + (1 - acc) f_w
        accuracy falls  <=>  f_c / f_w > (1 - acc) / acc

    The oracle arm sets f_c = 0 structurally, because the executor never
    damages a correct answer.
    """
    acc = 0.80
    threshold = correction.breaks_even(acc)
    rows = []
    for ratio in (0.05, 0.10, 0.25, 0.50, 1.00):
        f_w = 0.30
        f_c = ratio * f_w
        out = correction.compare_strategies(acc, f_c, f_w, recovery=0.30, rounds=1)
        rows.append({
            "ratio": ratio,
            "f_c": f_c,
            "f_w": f_w,
            "intrinsic": out["intrinsic"],
            "delta": out["intrinsic_delta"],
            "loses": out["intrinsic_delta"] < 0.0,
        })
    oracle = {
        "base": acc,
        "r1": correction.oracle_accuracy_after(acc, 0.30, 1),
        "r2": correction.oracle_accuracy_after(acc, 0.30, 2),
        "r3": correction.oracle_accuracy_after(acc, 0.30, 3),
    }
    return {"threshold": threshold, "rows": rows, "oracle": oracle}


# ------------------------------------------------------------------ 7

def exp_length_budget():
    """The exceed rate, the clipped answer, and what a conclude instruction buys.

    The corpus's crash: traces that exceed the maximum output length are "getting
    all of the problems wrong basically because the final answer was getting
    clipped". At inference the analogue is a budget that always leaves room for
    the answer, so a clipped generation is never returned as one.
    """
    base = 0.80
    budgets = (200, 400, 600, 800, 1000, 1600, 2000, 2400)
    no_instruction = length.budget_table(budgets, base, conclude_prob=0.0)
    with_instruction = length.budget_table(budgets, base, conclude_prob=0.80)
    rewards = length.reward_table((200, 600, 1200, 2000))
    return {
        "mean_length": length.mean_length(),
        "no_instruction": no_instruction,
        "with_instruction": with_instruction,
        "rewards": rewards,
    }
