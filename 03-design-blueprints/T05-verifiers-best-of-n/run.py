"""T05 -- verifiers and best-of-N: why the VERIFIER is the system.

Run:  python run.py

Stdlib only, no network, no GPU, no model. Every scorer below is an explicit arithmetic
function whose biases are PARAMETERS, so each experiment can turn a bias on and off and
report the DIFFERENCE. A simulated unbiased oracle would make the package vacuous.

WHAT THIS PROVES
  1. the KL bound is log n - (n-1)/n, it is a CEILING, and it is not tight
  2. n = 32 is a SYSTEMS constant: it fits one batch, and the 33rd sample costs a second
  3. a pool of n candidates is NOT n distinct candidates, and the gate saves the verifier bill
  4. a length-biased judge selects LONGER, WORSE candidates -- with no quality gain
  5. the tiered cascade cuts the JUDGE call count from n to k (26 -> 4 here), which is where
     the dominant cost lives -- while inheriting the ORM's recall at k
  6. a same-family judge systematically prefers its own family's outputs

WHAT THIS IS NOT: a benchmark. Nothing here measured a real system, and no corpus number is
asserted. The corpus supplies the mechanisms and the formulas; this file shows the DESIGN's
claims follow from its own arithmetic. Corpus figures are attributed in HLD.md.

Provenance: [T] transcript, [R] repo, [D] derived.
"""
from __future__ import annotations

from sim import experiments as ex

RULE = "=" * 76


def section(n: int, title: str) -> None:
    print()
    print(RULE)
    print(f"  {n}. {title}")
    print(RULE)


def main() -> None:
    print(RULE)
    print("  T05 - VERIFIERS AND BEST-OF-N: the verifier is the system")
    print(RULE)
    print("  Best-of-N is trivial. Scoring is where the value and every failure mode lives: the")
    print("  scorer has biases, provenance questions and a cost, and it can be gamed.")
    print()
    print("  This models MECHANISMS, not a machine. Every cost unit is NORMALISED, never")
    print("  currency -- the corpus asserts no vendor price, and a currency default would be a")
    print("  fabricated number carrying the authority of a config file.")

    section(1, "The KL bound: a ceiling, not a target")
    ex.exp_kl_bound()

    section(2, "The batch boundary: why n = 32")
    ex.exp_batch_boundary()

    section(3, "Diversity: the pool is smaller than you think")
    ex.exp_diversity()

    section(4, "Verbosity bias: the direction that reverses")
    ex.exp_verbosity_bias()

    section(5, "The tiered cascade: exact, then ORM, then judge on top-k")
    ex.exp_tiered_cost()

    section(6, "Self-preference: cross-family judging")
    ex.exp_judge_family()

    print()
    print(RULE)
    print("  DONE. Six relations, no benchmark numbers.")
    print(RULE)
    print("  Carry away: the verifier IS the system; exact checks come first and are free;")
    print("  n = 32 because it fits a batch; verbosity bias reverses the objective; a verifier")
    print("  you have not measured is not a verifier; and the KL bound is a ceiling, not a goal.")


if __name__ == "__main__":
    main()
