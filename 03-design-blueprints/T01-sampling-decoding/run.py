#!/usr/bin/env python3
"""T01 -- Sampling & Decoding. Runnable core.

Proves the design's central mechanism: the logit-processor chain. It shows

  1. what temperature does at its limits (T=1 identity, T->0 one-hot, T->inf uniform)
  2. why a fixed top-k is a different constraint at every decoding step
  3. what each truncation rule keeps, on a peaked and on a long-tailed step
  4. why the schema mask must run *before* truncation
  5. the diversity/surprise trade-off that the policy service has to pick a point on
  6. why temperature 0 is not a reproducibility contract
  7. that exactly tied logits give a uniform draw, not an argmax
  8. what target-perplexity control buys over a fixed threshold

WHAT THIS IS NOT: a benchmark. It is a model of a mechanism. There is no GPU
here and no real model. Every distribution is synthetic (see sim/distribution.py).
The real-world figures this illustrates are quoted and attributed in HLD.md
section 9; the numbers below are this simulation's own output under its stated
assumptions, and are labelled as such.

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


def sub(title):
    print()
    print("-- " + title)


def main():
    print(RULE)
    print("T01 Sampling & Decoding -- simulation of the logit-processor chain")
    print(RULE)
    print("This is a MODEL OF A MECHANISM, not a measurement.")
    print("Toy vocabulary: %d tokens. Real corpus vocabulary: 128k [T]." % ex.VOCAB)
    print("All randomness is seeded, so this output is reproducible.")

    header("1. Temperature at its limits")
    print("T=1 must be the identity; T->0 one-hot; T->inf uniform.")
    print()
    print("%-12s %-12s %-14s %-8s %s" % ("temperature", "max_prob", "entropy_nats", "argmax", "tied_at_max"))
    for r in ex.exp_temperature_limits():
        print("%-12.1f %-12.6f %-14.6f %-8d %d" % (
            r["temperature"], r["max_prob"], r["entropy_nats"],
            r["argmax"], r["tied_at_max"]))

    header("2. A fixed top-k is a different constraint at every step")
    print("Corpus figures: the top 6 tokens hold 68% of the mass after 'the'")
    print("and 99% after 'the car' [T]. Synthetic heads in that regime:")
    print()
    print("%-32s %-14s %-14s %s" % ("step", "requested", "realised_top6", "mass_beyond_top50"))
    for r in ex.exp_topk_drift():
        print("%-32s %-14.2f %-14.4f %.4f" % (
            r["step"], r["requested_head_mass"],
            r["realised_top6_mass"], r["mass_beyond_top50"]))
    print()
    print("Same k=6, two different constraints. This is why top-p is the default.")

    header("3. What each truncation rule keeps")
    print("%-12s %-22s %-11s %-14s %s" % (
        "step shape", "rule", "survivors", "retained_mass", "max_kept_prob"))
    for r in ex.exp_truncation_compare():
        print("%-12s %-22s %-11d %-14.6f %.6f" % (
            r["shape"], r["rule"], r["survivors"],
            r["retained_mass"], r["max_kept_prob"]))
    print()
    print("locally_typical can keep a token that is unlikely but typical, and")
    print("can cut one that is likely but atypical. That is the design intent.")

    header("4. Processor order: mask before truncate")
    print("Grammar permits 10 of 1000 tokens, all of them in the tail.")
    print()
    print("%-26s %-11s %-14s %s" % ("order", "survivors", "retained_mass", "outcome"))
    for r in ex.exp_processor_order():
        print("%-26s %-11d %-14.6f %s" % (
            r["order"], r["survivors"], r["retained_mass"], r["outcome"]))
    print()
    print("Truncating first spends the probability budget on tokens the schema")
    print("will reject, and can leave nothing to draw from. Mask first.")

    header("5. Diversity against surprise")
    print("200 generations x 100 tokens, i.i.d. from one fixed step distribution.")
    print()
    print("%-12s %-12s %-24s %s" % (
        "temperature", "distinct_1", "mean_token_perplexity", "survivors"))
    for r in ex.exp_diversity_vs_temperature():
        print("%-12.1f %-12.4f %-24.3f %d" % (
            r["temperature"], r["distinct_1"],
            r["mean_token_perplexity"], r["survivors"]))
    print()
    print("distinct_1 is the corpus's deterministic diversity metric [T].")
    print("Greedy collapses it. Raising temperature buys diversity and pays in")
    print("surprise -- the curve the policy service picks a point on.")

    header("6. Why temperature 0 is not a reproducibility contract")
    rows = ex.exp_reduction_order()
    print("The same argmax, re-run 200 times per step shape, on a distribution")
    print("perturbed by a reduction-order-sized jitter (relative magnitude 1e-7).")
    print()
    print("%-30s %-12s %-10s %-12s %s" % (
        "step shape", "top1-top2", "agreement", "longest_run", "flips"))
    for r in rows:
        print("%-30s %-12.1e %-10.4f %-12d %d" % (
            r["step"], r["top1_minus_top2"], r["agreement_rate"],
            r["longest_agreeing_streak"], r["flips"]))
    print()
    print("The flip rate depends on the margin, not on the code. That is why the")
    print("failure is intermittent: most steps are confident and stable, and the")
    print("ones that are near-ties flip. The corpus describes the symptom as")
    print("'you'll get it the same like five times, but then you'll get a")
    print("different one' [T] -- a streak, then a flip. In an autoregressive")
    print("sequence one early flip changes every token after it.")

    header("7. Tied logits")
    t = ex.exp_tie_behaviour()
    print("Three tokens with identical logits, temperature 0:")
    print("  frozen max prob : %.6f" % t["frozen_max_prob"])
    for k, v in t["share_of_draws_on_each_tied_token"].items():
        print("  share on token %-2d: %.4f" % (k, v))
    print()
    print("T -> 0 is one-hot over the argmax, but a tie is uniform over the")
    print("tied set [T]. Greedy implementations that take index 0 get this wrong.")

    header("8. Target-perplexity control versus a fixed threshold")
    m = ex.exp_mirostat()
    print("  target perplexity          : %.3f" % m["target_perplexity"])
    print("  mirostat mean perplexity   : %.3f" % m["mirostat_mean_perplexity"])
    print("  fixed top_p(0.95) mean ppl : %.3f" % m["fixed_topp_mean_perplexity"])
    print()
    print("Mirostat tracks its target; the fixed threshold's perplexity is")
    print("whatever the distribution happens to give. The price is a running")
    print("estimate that converges poorly on short outputs [T].")

    header("Summary")
    print("The mechanism this design rests on:")
    print("  * temperature reshapes; it does not truncate, and T=1 is the no-op")
    print("  * truncation is a deliberate bias, and its rule matters more than")
    print("    its parameter -- the same k means different things at each step")
    print("  * order is load-bearing: mask, then temperature, then truncate")
    print("  * temperature 0 is argmax with unspecified tie-breaking, not a")
    print("    reproducibility guarantee")
    print()
    print("Policy consequence: the sampler is cheap and the *policy* is the")
    print("product. That is why the design puts policy in a versioned registry")
    print("and never in application code.")
    print()
    print("OK -- T01 simulation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
