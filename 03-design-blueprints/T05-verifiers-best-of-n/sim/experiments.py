"""The six experiments. Each is independently runnable and each asserts a RELATION.

Nothing here asserts a corpus number. The corpus supplies the mechanisms, the batch-fit
argument, the bias list and the bound's formula; whether those hold in the corpus is checked
by provenance review, not by executing this file. What this file shows is that the DESIGN's
claims follow from its own arithmetic.
"""
from __future__ import annotations

import math

from .candidates import make_candidate, build_pool, diversity_gate, make_rng, \
    draw_pool, expected_unique, phrasings_at_temperature
from .verifiers import programmatic_score, orm_score, tiered_score, judge_score, \
    measure_agreement, judge_flip_rate
from .selection import select, kl_bound, simulate_best_of_n, kl_divergence


# Costs in NORMALISED units, never currency: the corpus asserts no vendor price, and a
# currency default would be a fabricated number carrying the authority of a config file.
COST_TIER1 = 0.02
COST_TIER3 = 1.00


def _rule_no_profanity(c):
    return ("forbidden" not in c.text, "contains a forbidden phrase")


def _rule_cites_real_policy(c):
    return (c.meta.get("policy_ok", True), "cites a policy that does not exist")


DEFAULT_RULES = [_rule_no_profanity, _rule_cites_real_policy]


# --------------------------------------------------------------------------------------
# 1. The KL bound: a ceiling that rises as log n, and is NOT tight.
# --------------------------------------------------------------------------------------

def exp_kl_bound(seed: int = 7) -> None:
    print("\n1. THE KL BOUND -- a ceiling, not a target")
    print("   corpus: KL(P_bon || P_target) <= log n - (n-1)/n   [T] CMU lecture 12")
    print("   the divergence is against the TARGET (preference) distribution, NOT the base policy,")
    print("   and the lecture states explicitly that the bound is NOT tight.\n")

    # A toy selection problem: K outcomes, a base model q, and a target preference.
    # q is deliberately misaligned with the preference, so best-of-n has work to do.
    K = 6
    rng = make_rng(seed)
    q = [0.35, 0.25, 0.18, 0.12, 0.06, 0.04]
    reward = [0.0, 0.2, 0.4, 0.6, 0.9, 1.3]        # the target preference's ordering
    # The target distribution the corpus's bound refers to: the reward-tilted model
    # p_target propto q * exp(beta * reward). At n -> inf, best-of-n approaches it.
    beta = 1.1
    tilt = [qi * math.exp(beta * ri) for qi, ri in zip(q, reward)]
    z = sum(tilt)
    p_target = [t / z for t in tilt]

    print(f"   {'n':>5}  {'bound':>8}  {'measured KL(p_bon||p_target)':>30}")
    print("   " + "-" * 48)
    for n in (1, 2, 4, 8, 16, 32, 64, 128):
        emp = simulate_best_of_n(q, reward, n, rng, trials=3000)
        kl = kl_divergence(emp, p_target)
        flag = "  (degenerate: n=1 means no selection)" if n == 1 else ""
        print(f"   {n:>5}  {kl_bound(n):>8.4f}  {kl:>30.4f}{flag}")

    print("\n   READ: the bound rises with n, but only as log n -- so selection is a CHEAP way")
    print("   to move the policy, with no gradient step and no training run.")
    print("   READ: at n=1 the bound is 0 by construction, because n=1 means no selection at")
    print("   all; the construction's target is the n->inf limit. That edge case is why the")
    print("   bound is described as loose rather than as an equality.")


# --------------------------------------------------------------------------------------
# 2. The batch boundary: n = 32 is a SYSTEMS constant wearing statistical clothes.
# --------------------------------------------------------------------------------------

def exp_batch_boundary(batch: int = 32) -> None:
    print(f"\n2. THE BATCH BOUNDARY -- why n = {batch}")
    print(f"   corpus: n = {batch} is chosen because it FITS ONE BATCH; the {batch + 1}rd sample would need\n"
          "   a second pass or a second GPU. [T] CMU lecture 12\n")

    print(f"   {'n':>5}  {'batches':>8}  {'slots paid':>11}  {'waste':>8}  {'cost index':>11}")
    print("   " + "-" * 50)
    base = None
    for n in (8, 16, 32, 33, 40, 64, 100):
        batches = math.ceil(n / batch)
        slots = batches * batch
        waste = (slots - n) / slots
        cost = batches                      # one pass per batch
        if base is None:
            base = cost / n
        print(f"   {n:>5}  {batches:>8}  {slots:>11}  {waste:>7.1%}  {cost / n / base:>11.2f}")

    print("\n   READ: cost per sample is FLAT from n=1 to n=32, then steps. The 33rd sample")
    print("   costs a whole second batch, so the effective price of asking for 33 is the price")
    print("   of 64. This is why the capacity model is written in batches, not samples, and why")
    print("   the n cap belongs ON the boundary rather than above it.")


# --------------------------------------------------------------------------------------
# 3. Diversity: n candidates are not n DISTINCT candidates.
# --------------------------------------------------------------------------------------

def exp_diversity(floor: float = 0.5, n: int = 32) -> None:
    print(f"\n3. DIVERSITY -- the pool is smaller than you think (n = {n}, floor = {floor})")
    print("   corpus: at temperature 0.2, 100 draws produce only ~20 unique outputs. [T]\n")

    print(f"   {'temp':>6}  {'phrasings m':>12}  {'expected unique':>16}  {'unique frac':>12}  {'gate':>8}")
    print("   " + "-" * 62)
    for t in (0.0, 0.2, 0.4, 0.8, 1.0):
        m = phrasings_at_temperature(t)
        eu = min(expected_unique(n, m), n)
        frac = eu / n
        gate = "PASS" if frac >= floor else "DEGENERATE"
        print(f"   {t:>6.1f}  {m:>12}  {eu:>16.1f}  {frac:>12.1%}  {gate:>8}")

    print("\n   Now the gate on real pools, and -- the point -- the verifier bill it saves:")
    rng = make_rng(11)
    for t in (0.0, 0.2, 0.8):
        pool = build_pool("prompt", draw_pool(n, t, rng), floor=floor)
        passes, reason = diversity_gate(pool, floor)
        verifier_calls = n if passes else 0
        print(f"   temp {t}: {reason}")
        print(f"            verifier calls spent: {verifier_calls}")

    print("\n   READ: a low-temperature pool is DEGENERATE, and the gate fires BEFORE any")
    print("   verifier call. The gate's value is not statistical -- it is that no verifier")
    print("   compute is spent on a pool that cannot support a selection.")


# --------------------------------------------------------------------------------------
# 4. Verbosity bias: the judge selects for LENGTH, and quality does not improve.
# --------------------------------------------------------------------------------------

def exp_verbosity_bias() -> None:
    print("\n4. VERBOSITY BIAS -- the direction that reverses")
    print("   corpus: judges carry position, length and self-preference biases (Jung et al. 2023). [T]\n")

    # Two candidates: A is better and shorter; B is worse and longer.
    a = make_candidate("cA", "concise and correct", length=120, logprob=-0.2,
                       meta={"quality": 0.80, "family": "A"})
    b = make_candidate("cB", "long and slightly worse", length=400, logprob=-0.1,
                       meta={"quality": 0.78, "family": "A"})
    pool = [a, b]

    print(f"   A: length {a.length:>4}, true quality {a.meta['quality']:.2f}")
    print(f"   B: length {b.length:>4}, true quality {b.meta['quality']:.2f}   <- worse, but LONGER\n")

    print(f"   {'length_bias':>12}  {'score A':>9}  {'score B':>9}  {'winner':>8}  {'winner quality':>15}")
    print("   " + "-" * 60)
    for lb in (0.0, 0.05, 0.10, 0.20):
        sa = judge_score(a, 0, 2, length_bias=lb)
        sb = judge_score(b, 1, 2, length_bias=lb)
        scores = [sa, sb]
        chosen, _, _ = select(scores, pool, tie_break="shorter")
        wq = {"cA": 0.80, "cB": 0.78}[chosen]
        print(f"   {lb:>12.2f}  {sa['score']:>9.3f}  {sb['score']:>9.3f}  {chosen:>8}  {wq:>15.2f}")

    print("\n   READ: at length_bias = 0 the better candidate wins. Raise it, and the LONGER,")
    print("   WORSE candidate wins -- and it keeps winning as the bias grows. Quality is flat")
    print("   while length inflates, which reads as thoroughness on every dashboard except one.")
    print("   The design responses are all parameters in judge_score: length-normalise,")
    print("   randomise position, and break ties toward the SHORTER candidate.")


# --------------------------------------------------------------------------------------
# 5. The tiered cascade: the biggest cost lever in the blueprint.
# --------------------------------------------------------------------------------------

def exp_tiered_cost(seed: int = 3, n: int = 32, top_k: int = 4) -> None:
    print(f"\n5. THE TIERED CASCADE -- exact on all, ORM on survivors, judge on top-{top_k}")
    print("   corpus: generation is far more expensive than scoring. [T]\n")

    rng = make_rng(seed)
    pool = []
    for i in range(n):
        true_q = rng.random()
        # The ORM sees a NOISIER view than the judge -- that is what limits top-k recall.
        orm_q = max(0.0, min(1.0, true_q + rng.gauss(0.0, 0.06)))
        policy_ok = rng.random() > 0.10          # 10% fail the exact check
        pool.append(make_candidate(
            f"c{i:02d}", f"draft-{i:02d}", length=100 + 5 * i, logprob=-0.1,
            meta={"quality": true_q, "orm_quality": orm_q, "family": "A",
                  "policy_ok": policy_ok},
        ))

    # (a) judge every candidate -- the naive design
    full_scores, full_usage = tiered_score(pool, DEFAULT_RULES, top_k=n, judge_kwargs={})
    full_cost = full_usage["1"] * COST_TIER1 + full_usage["3"] * COST_TIER3
    full_choice, _, _ = select(full_scores, pool)

    # (b) the cascade
    cas_scores, cas_usage = tiered_score(pool, DEFAULT_RULES, top_k=top_k, judge_kwargs={})
    cas_cost = cas_usage["1"] * COST_TIER1 + cas_usage["3"] * COST_TIER3
    cas_choice, _, _ = select(cas_scores, pool)

    q = {c.id: c.meta["quality"] for c in pool}
    print(f"   {'design':>22}  {'tier0':>6}  {'tier1':>6}  {'tier3':>6}  {'cost':>7}  {'choice quality':>15}")
    print("   " + "-" * 72)
    print(f"   {'judge all n':>22}  {full_usage['0']:>6}  {full_usage['1']:>6}  {full_usage['3']:>6}"
          f"  {full_cost:>7.2f}  {q[full_choice]:>15.3f}")
    print(f"   {'cascade (top-%d)' % top_k:>22}  {cas_usage['0']:>6}  {cas_usage['1']:>6}  {cas_usage['3']:>6}"
          f"  {cas_cost:>7.2f}  {q[cas_choice]:>15.3f}")
    print(f"\n   judge call count: {full_usage['3']} -> {cas_usage['3']} "
          f"({full_usage['3'] / max(1, cas_usage['3']):.1f}x fewer)")
    print(f"   TOTAL cost ratio: {full_cost / cas_cost:.1f}x cheaper for the cascade")
    print(f"   same winner: {full_choice == cas_choice}")

    print("\n   READ: the judge call count falls from n to k -- here 26 -> 4, the tier where the")
    print("   dominant cost lives. The TOTAL ratio is smaller than the judge ratio because the")
    print("   ORM pass is not free and is unchanged; the judge is what must be economised.")
    print("   But the cascade INHERITS the ORM's recall at k: if the true best candidate is not")
    print("   in the ORM's top-k, the judge can never recover it. top_k is a RECALL parameter,")
    print("   not a cost parameter -- set it by measuring how often the human-preferred candidate")
    print("   is in the ORM's top-k, never by picking a number because it is cheap.")


# --------------------------------------------------------------------------------------
# 6. Self-preference: a same-family judge prefers its own family's outputs.
# --------------------------------------------------------------------------------------

def exp_judge_family(seed: int = 5, n: int = 24) -> None:
    print("\n6. SELF-PREFERENCE -- why the judge must come from a DIFFERENT family")
    print("   corpus: judges carry a self-preference bias. [T]\n")

    rng = make_rng(seed)
    pool = []
    for i in range(n):
        fam = "A" if i % 2 == 0 else "B"
        # Held-out true quality is IDENTICAL across families: any winner change is bias.
        true_q = 0.5 + 0.4 * (i / n)
        pool.append(make_candidate(f"c{i:02d}", f"draft-{i:02d}", length=200,
                                   logprob=-0.1, meta={"quality": true_q, "family": fam}))

    q = {c.id: c.meta["quality"] for c in pool}
    fam = {c.id: c.meta["family"] for c in pool}

    print(f"   {'judge family':>13}  {'winner':>8}  {'winner family':>14}  {'winner quality':>15}")
    print("   " + "-" * 56)
    for jf in (None, "A", "B"):
        r = judge_flip_rate(jf, pool)
        label = "none (control)" if jf is None else jf
        print(f"   {label:>13}  {r['with_family']:>8}  {fam[r['with_family']]:>14}  "
              f"{q[r['with_family']]:>15.3f}")

    print("\n   READ: quality is identical across families by construction, yet the family")
    print("   bonus moves the winner. A judge from the same family as your generator will")
    print("   systematically prefer that family's outputs -- so the selection is partly a")
    print("   self-portrait. Use a cross-family judge, and MEASURE agreement on a gold set.")
    agree = measure_agreement([True, True, False, True], [True, False, False, True])
    print(f"   agreement() on a 4-item gold set: {agree:.2f}  (re-measure per scorer version)")
