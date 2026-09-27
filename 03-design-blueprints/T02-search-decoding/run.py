#!/usr/bin/env python3
"""T02 -- Beam, A* and Best-First Search. Runnable core.

Proves the design's central mechanism: that greedy, beam search, uniform-cost
search and A* are four settings of ONE priority-queue decoder, parameterised by

    (comparator, beam constraint K, heuristic h)

and that which setting is legal depends entirely on whether an admissible
heuristic exists -- which, over an open-ended decoder, it does not.

WHAT THIS IS NOT: a benchmark. The graph is a hand-built four-token bigram
model chosen to contain a memorised sweet spot. Every step count below is this
simulation's own. The corpus's figures (the 10x best-first speedup, the 16x16
KL matrix, the 4/6/9/8 step counts on the lecture's toy graph) are quoted and
attributed in HLD.md section 9 and are NOT reproduced here.

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
    print("T02 Beam, A* and Best-First Search -- one decoder, four settings")
    print(RULE)
    print("This is a MODEL OF A MECHANISM, not a measurement.")
    print("Toy vocabulary: a, b, c, EOS. Max length: %d." % ex.MAX_LEN)
    print("Costs are -log p in nats; lower is better.")

    header("1. The same task through one machine, four ways")
    rows, optimum = ex.exp_decoder_zoo()
    print("Exhaustive optimum: %s  (cost %.4f)" % (
        ex.render(optimum["path"]), optimum["cost"]))
    print()
    print("%-28s %-22s %-10s %-11s %s" % (
        "setting", "output", "cost", "expansions", "optimal?"))
    for r in rows:
        print("%-28s %-22s %-10.4f %-11d %s" % (
            r["decoder"], r["path"], r["cost"], r["expansions"],
            "yes" if r["optimal"] else "NO"))
    print()
    print("One loop, five settings, all scored by the same objective (raw")
    print("log-probability). Greedy commits token by token and settles on the")
    print("suboptimal a-c-EOS. Beam and uniform-cost find the raw optimum --")
    print("the graph is small -- but beam pays in expansions, and the bill")
    print("rises with K (28 -> 68) without changing the answer. Best-first beam")
    print("returns greedy's answer at 6 expansions against beam K=4's 68: the")
    print("beam constraint bought the search and cost it the guarantee. The")
    print("parameters differ; the machine does not.")

    header("2. Length normalization decides which candidate you ship")
    print("The beam controls the candidate SET. The normalizer picks the member.")
    print()
    print("%-22s %-22s %-8s %-10s %s" % (
        "normalization", "selected", "length", "raw_cost", "score"))
    for r in ex.exp_length_normalization():
        print("%-22s %-22s %-8d %-10.4f %.4f" % (
            r["normalization"], r["selected"], r["length"],
            r["raw_cost"], r["score"]))
    print()
    print("Raw log-probability prefers the shortest sequence -- EOS gets 'an")
    print("unfairly good chance' [T]. Each step down the table trades a worse")
    print("raw score for a better score per token, and the shipped output gets")
    print("longer. That is the whole effect, and it is the first thing to")
    print("check when width up makes quality down.")

    header("3. A* where the length is known")
    print("A glossary-constrained segment has a bounded length, so a")
    print("row-minimum heuristic is computable AND admissible.")
    print()
    print("%-28s %-10s %-11s %s" % ("search", "cost", "expansions", "output"))
    for r in ex.exp_bounded_astar():
        print("%-28s %-10.4f %-11d %s" % (
            r["search"], r["cost"], r["expansions"], r["path"]))
    print()
    print("Identical answer, a third of the expansions. In bounded mode the")
    print("row minimum really is a lower bound on what is left to pay, because")
    print("there is no EOS to stop early and no length to guess.")

    header("4. Why that same heuristic is illegal one line over")
    print("Carry the row-minimum function to the OPEN-ENDED decoder -- the one")
    print("where EOS is an available action -- and it stops being a lower")
    print("bound. EOS out of the model's cheap state costs less than that")
    print("state's cheapest non-EOS arc, so the row minimum now OVERestimates")
    print("what remains. The guarantee is void.")
    print()
    rows, opt4 = ex.exp_heuristic_breaks_on_memorised_span()
    print("True open-ended optimum: %s  (cost %.4f)" % (
        ex.render(opt4["path"]), opt4["cost"]))
    print()
    print("%-32s %-20s %-10s %-11s %s" % (
        "search", "output", "cost", "expansions", "optimal?"))
    for r in rows:
        print("%-32s %-20s %-10.4f %-11d %s" % (
            r["search"], r["path"], r["cost"], r["expansions"],
            "yes" if r["optimal"] else "NO"))
    print()
    print("The heuristic that was correct in section 3 now returns a worse")
    print("answer, and the pessimistic one returns a worse answer still while")
    print("expanding a single node. This is the corpus's warning made")
    print("concrete: 'you might deviate from the optimal solution. You're not")
    print("guaranteed to deviate... but you might' [T].")

    header("5. Recombination: n-gram clustering against KL")
    rows, k = ex.exp_recombination()
    print("%d live hypotheses." % k)
    print()
    print("%-38s %-10s %-16s %s" % (
        "criterion", "clusters", "largest_cluster", "pairwise_ops"))
    for r in rows:
        ops = "-" if r["pairwise_ops"] is None else str(r["pairwise_ops"])
        print("%-38s %-10d %-16d %s" % (
            r["criterion"], r["clusters"], r["largest_cluster"], ops))
    print()
    print("n-gram clustering is free and composes with a cache; it can merge")
    print("hypotheses that 'were very different previously but similar for the")
    print("most recent words' [T]. KL clustering is semantically grounded but")
    print("costs a pairwise matrix -- at beam 16 that is a 16x16 matrix of KL")
    print("divergences per step [T], which is why the design keeps it for the")
    print("bounded path where the candidate set is small.")

    header("6. Diverse beam search: the step accounting")
    d = ex.exp_diverse_beam()
    print("  groups g              : %d" % d["groups"])
    print("  beams per group       : %d" % d["beams_per_group"])
    print("  steps t               : %d" % d["steps"])
    print("  sequential g*t        : %d" % d["sequential_steps"])
    print("  batched  t+g-1        : %d" % d["batched_steps"])
    print("  saving factor         : %.2fx" % d["saving_factor"])
    print()
    dup = ex.exp_duplicate_check()
    print("  plain beam outputs    : %s" % ", ".join(dup["plain_outputs"]))
    print("  diverse beam outputs  : %s" % ", ".join(dup["diverse_outputs"]))
    print()
    print("The saving is structural: g groups can be advanced together, so the")
    print("cost is t+g-1 steps rather than g*t [T]. The duplicate promise is")
    print("NOT demonstrated here and this model does not claim it -- a")
    print("hypothesis in this representation is its token tuple, so two outputs")
    print("are never the same object. On a 128k vocabulary the promise is not")
    print("trivial, because many distinct token paths render to the same")
    print("string, and that is the case the deduplication step exists for.")
    print("This model measures the step accounting and nothing else.")

    header("Summary")
    print("  * one priority queue; comparator, K and h are the parameters")
    print("  * beam search's K is a budget, not a quality dial -- the curse of")
    print("    beam search is real, and length normalization is the first")
    print("    suspect when width up makes BLEU go down")
    print("  * A* is usable only where the length is bounded and h is")
    print("    computable; over an open-ended decoder it is not constructible")
    print("  * recombination is what makes A*-style search affordable, and the")
    print("    cheap n-gram form is what the design actually deploys")
    print()
    print("OK -- T02 simulation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
