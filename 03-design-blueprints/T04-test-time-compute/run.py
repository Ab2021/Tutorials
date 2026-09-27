#!/usr/bin/env python3
"""T04 -- Test-Time Compute. Runnable core.

Proves the design's central mechanisms:

  * chain of thought is a latent variable, and the three ways to read it --
    greedy, joint argmax and the true marginal -- do not agree, which is why
    sampling is the recommended direction and search over chains is not;
  * self-consistency is a sampled estimate of that marginal, and it converges to
    it at n times the inference cost;
  * the adaptive version's prior arithmetic is the corpus's own: alpha = 3 with
    counts 1,0,0 gives 0.5/0.25/0.25, and alpha = 0 is maximum likelihood;
  * the Beta stopping rule saves samples on EASY questions and very few on
    contested ones -- the opposite of the intuition, and a direct input to the
    capacity model;
  * the rule reads agreement as confidence, so a unanimous wrong run stops early;
    a sample floor mitigates but does not remove this;
  * intrinsic self-correction loses accuracy under a derivable condition, and an
    oracle sets that condition to zero structurally;
  * a budget that always leaves room for the answer is what prevents the corpus's
    exceed-rate crash.

WHAT THIS IS NOT: a benchmark. Every probability is hand-chosen, no model was
run, and the corpus's figures (100 samples = 100x, the 0.95 threshold, alpha = 3,
v^100 latent terms) are quoted and attributed in HLD.md and are NOT reproduced
here.

Stdlib only. Offline. Exits 0.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sim import experiments as ex  # noqa: E402

RULE = "=" * 78


def header(title):
    print()
    print(RULE)
    print(title)
    print(RULE)


def main():
    print(RULE)
    print("T04 Test-Time Compute -- marginal, vote, stop, correct, budget")
    print(RULE)
    print("This is a MODEL OF A MECHANISM, not a measurement.")
    print("Toy problems with hand-chosen probabilities; seed %d." % ex.SEED)

    header("1. Chain of thought as a latent variable")
    for r in ex.exp_latent_marginal():
        print()
        print("problem: %s   (correct answer: %s)" % (r["problem"], r["correct"]))
        print("  chains and P(z|x):")
        for z, p in r["chains"]:
            print("    %-14s %.2f" % (z, p))
        print("  P(y|x) = sum_z P(z|x) P(y|x,z):")
        for a in sorted(r["marginal"]):
            print("    y=%-4s %.4f" % (a, r["marginal"][a]))
        print("  greedy        (%s) -> %-4s  correct: %s" % (
            r["greedy"][0], r["greedy"][1], r["greedy_right"]))
        print("  joint argmax  (%s) -> %-4s  correct: %s  (p=%.4f)" % (
            r["joint"][0], r["joint"][1], r["joint_right"], r["joint"][2]))
        print("  marginal      argmax y -> %-4s  correct: %s" % (
            r["marginal_argmax"], r["marginal_right"]))
    print()
    print("On 3ft_in_inches the highest-probability chain is the centimetres one,")
    print("and it also carries the highest-probability JOINT (z, y) pair -- but the")
    print("sum over chains prefers 36, because the two inch chains agree with each")
    print("other. That is the corpus's counterexample shape: the jointly-most-")
    print("likely path is not the most likely answer, so search over chains is the")
    print("wrong tool and sampling is the recommended one.")

    header("2. Self-consistency: a sampled estimate of that marginal")
    print("%-16s %-6s %-12s %s" % ("problem", "n", "accuracy", "cost (x greedy)"))
    for row in ex.exp_self_consistency():
        print("%-16s %-6d %-12.3f %.0fx" % (
            row["problem"], row["n"], row["accuracy"], row["cost"]))
    print()
    print("Majority vote converges toward the marginal's argmax and pays n times")
    print("the inference cost for it -- the corpus's '100 samples, 100 times the")
    print("inference cost'. On the easy problem the vote is already right at n=1,")
    print("so every further sample is spent confirming an answer that was never in")
    print("doubt. That is the case adaptive stopping exists for.")

    header("3. The prior arithmetic, checked against the corpus")
    print("%-8s %-22s %-28s %s" % ("alpha", "pseudo-counts", "posterior",
                                   "reading"))
    for r in ex.exp_dirichlet_worked_example():
        pseudo = ", ".join("%.2f" % r["pseudo"][k] for k in sorted(r["pseudo"]))
        post = ", ".join("%.3f" % r["posterior"][k] for k in sorted(r["posterior"]))
        reading = ("MLE (corpus: 'if alpha is zero this is maximum "
                   "likelihood estimation')" if r["is_mle"] else "prior-weighted")
        print("%-8.1f %-22s %-28s %s" % (r["alpha"], pseudo, post, reading))
    print()
    print("Counts 1,0,0 over three answers. At alpha = 3 the pseudo-counts are")
    print("2, 1, 1 and the posterior is 0.500 / 0.250 / 0.250 -- the corpus's own")
    print("worked example, reproduced. At alpha = 0 the estimate collapses to the")
    print("raw counts, which is the MLE the corpus warns about: one observation of")
    print("A and the model is certain B and C are impossible.")

    header("4. What adaptive stopping actually saves")
    print("%-16s %-14s %-10s %-12s %-12s %s" % (
        "problem", "mean samples", "accuracy", "stopped early", "hit the cap",
        "saving"))
    for r in ex.exp_adaptive_stopping():
        print("%-16s %-14.2f %-10.3f %-12.2f %-12.2f %.0f%%" % (
            r["problem"], r["mean_samples"], r["accuracy"], r["early_rate"],
            r["cap_fraction"], r["saving_vs_16"] * 100))
    print()
    print("The finding worth carrying is the opposite of the intuition. The Beta")
    print("rule at alpha = 3 is conservative: a run has to be near-unanimous before")
    print("the leader's posterior clears 0.95. Easy questions produce unanimous")
    print("runs immediately, so they stop in a handful of samples. Contested")
    print("questions by definition do not, so they run to the cap and save almost")
    print("nothing. The saving lands on the questions that did not need it.")
    print()
    print("This is the model's own arithmetic, not a corpus figure: the lecture")
    print("calls adaptive self-consistency a way to save compute and reports no")
    print("samples-saved number at all. A capacity model that assumes a mean of 5")
    print("samples is assuming the easy case.")

    header("5. The blind spot: agreement is not correctness")
    print("%-8s %-14s %-12s %s" % ("floor", "mean samples", "accuracy",
                                   "wrong and confident"))
    for r in ex.exp_confidently_wrong():
        print("%-8d %-14.2f %-12.3f %.3f" % (
            r["floor"], r["mean_samples"], r["accuracy"],
            r["wrong_and_confident"]))
    print()
    print("On the trap fixture the wrong answer lives in one high-probability")
    print("chain, so a short unlucky run is unanimous and wrong. The rule reads")
    print("unanimity as confidence and stops early on it.")
    print()
    print("The first two floors are inert, and that is not a bug: two unanimous")
    print("samples give the leader a Beta(3.5, 1.5) posterior, whose probability of")
    print("exceeding one half is 0.8395 -- short of the 0.95 threshold. The rule")
    print("needs a 5-0 run (Beta(5.5, 1.5) = 0.9527) before it will fire at all.")
    print("floor can only bite once it exceeds the point at which the rule would")
    print("have fired anyway.")
    print()
    print("Raising it further buys protection and costs samples; it does not")
    print("remove the failure, because a long enough unlucky run still looks")
    print("unanimous. At floor 16 the residue is the majority-wrong rate, which no")
    print("stopping rule can fix.")
    print()
    print("This is why the design never treats the posterior as a correctness")
    print("signal. It is an AGREEMENT signal. Correctness comes from a verifier,")
    print("which is the handoff to T05.")

    header("6. Self-correction: the condition, not an amount")
    s = ex.exp_self_correction()
    print("base accuracy 0.80 -> intrinsic self-correction loses accuracy when")
    print("f_c / f_w > (1 - acc) / acc = %.2f" % s["threshold"])
    print()
    print("%-10s %-10s %-14s %-12s %s" % ("f_c/f_w", "f_c", "accuracy", "delta",
                                          "net effect"))
    for r in s["rows"]:
        print("%-10.2f %-10.3f %-14.4f %-12.4f %s" % (
            r["ratio"], r["f_c"], r["intrinsic"], r["delta"],
            "LOSES" if r["loses"] else "helps"))
    print()
    print("The corpus gives no accuracy figure for the degradation -- it is a")
    print("qualitative result. So this model does not invent one: it fixes the two")
    print("rates and derives the boundary. The boundary is unforgiving. At 80%")
    print("accuracy a corrector that damages a correct answer at a quarter the")
    print("rate it repairs a wrong one is already net-negative.")
    print()
    print("The oracle arm, for contrast (accuracy after each round):")
    print("  round 0 %.4f -> round 1 %.4f -> round 2 %.4f -> round 3 %.4f" % (
        s["oracle"]["base"], s["oracle"]["r1"], s["oracle"]["r2"],
        s["oracle"]["r3"]))
    print("An executor never damages a correct answer, so f_c is structurally")
    print("zero and the curve can only rise. That asymmetry -- 16 samples versus")
    print("1 in the corpus's own comparison -- is the whole reason the repair bot")
    print("gets a correction loop and the tutor does not.")

    header("7. Length: the exceed rate and the clipped answer")
    L = ex.exp_length_budget()
    print("Mean trace length in the model: %.0f tokens." % L["mean_length"])
    print()
    print("%-10s %-14s %-16s %-14s %s" % (
        "budget", "exceed rate", "accuracy (raw)", "accuracy (conclude)",
        "clipped wrong (raw)"))
    for a, b in zip(L["no_instruction"], L["with_instruction"]):
        print("%-10d %-14.3f %-16.4f %-14.4f %.3f" % (
            a["budget"], a["exceed_rate"], a["accuracy"], b["accuracy"],
            a["clipped_wrong"]))
    print()
    print("A clipped trace cannot emit its final answer, so it is counted wrong --")
    print("the corpus's crash, where exceeding the maximum output length got 'all")
    print("of the problems wrong basically because the final answer was getting")
    print("clipped'. Raising the budget lowers the exceed rate but costs tokens on")
    print("every trace. An instruction to conclude before the budget buys the same")
    print("accuracy at a much smaller budget.")
    print()
    print("The cosine length term, as the corpus describes it (1.0 at length 0,")
    print("0.0 at the maximum output length):")
    print("%-10s %-10s %-16s %s" % ("length", "cosine", "reward if right",
                                    "reward if wrong"))
    for r in L["rewards"]:
        print("%-10d %-10.4f %-16.4f %.4f" % (
            r["length"], r["cosine"], r["reward_correct"], r["reward_wrong"]))
    print()
    print("Both stated behaviours fall out of the shape: a wrong answer is")
    print("punished hardest when it is short ('if the answer is wrong we basically")
    print("give a larger negative reward when the answer is short'), and a correct")
    print("answer is pushed shorter ('if you're getting it right, make it")
    print("shorter'). At inference there is no gradient, so the design uses a")
    print("budget and a conclude instruction rather than a reward.")

    header("Summary")
    print("  * the marginal over chains is the target; greedy and joint argmax")
    print("    both miss it on the corpus's own counterexample")
    print("  * self-consistency estimates it by sampling, at n times the cost")
    print("  * the Dirichlet prior arithmetic reproduces the corpus's 0.5/0.25/0.25")
    print("  * adaptive stopping saves on easy questions, not contested ones")
    print("  * the posterior measures agreement, not correctness -- a verifier is")
    print("    the only correctness signal, and that is the T05 handoff")
    print("  * intrinsic self-correction loses accuracy above a derivable")
    print("    f_c / f_w threshold; an oracle sets it to zero")
    print("  * the exceed rate, not latency, is what makes a clipped trace wrong")
    print()
    print("OK -- T04 simulation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
