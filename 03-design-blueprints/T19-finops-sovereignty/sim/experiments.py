"""T19 -- the six demonstrations. Each prints a table and then a FINDING.

A demonstration that only prints numbers is a dashboard. Each function here also states what the
numbers mean for a design decision, so the HLD can quote a run rather than an opinion.

The sixth is deliberately about the PRECONDITION rather than the optimisation, because attribution
is the step the corpus says everything else depends on, and it is the one metric in the topic that
degrades as the business succeeds.
"""
from __future__ import annotations

from . import attribution as A
from . import levers as L
from . import sovereignty as V
from . import stack as S
from . import units as U

LINE = "-" * 78


def _hdr(n, title: str) -> None:
    print("\n" + "=" * 78)
    print(f"{n}. {title}")
    print("=" * 78)


def _pct(x: float) -> str:
    return f"{x * 100.0:.1f}%"


def _pct2(x: float) -> str:
    return f"{x * 100.0:.2f}%"


def _mult(reduction: float) -> str:
    """A reduction expressed as the multiple it implies. 0.9 -> '10.00x'."""
    return f"{1.0 / (1.0 - reduction):.2f}x" if reduction < 1.0 else "inf"


# ----------------------------------------------------------------------------------------------
def exp_cost_stack() -> None:
    """1. The discount curve, its asymptote, and the write premium nobody computes."""
    _hdr(1, "THE COST STACK -- a discount is a price, a saving is an amount, and the curve ends")
    print("""
[corpus] "Optimization without measurement is just guessing."                    -- Tokenomics [T]
[corpus] "Cache reads are roughly ten times cheaper than fresh tokens."          -- tokenomics [T]
[corpus] "Cache something that changes each call and you gain nothing."          -- tokenomics [T]
[repo]   system prompts are ~69% of INPUT tokens; only ~28% of calls use prompt caching.
         Input is ~45% of the bill, so the prefix line is 0.69 x 0.45 = 31.05% of it.   [R]
""")
    print("The bill, in eight lines. Shares are of the TOTAL inference bill [D, illustrative]:")
    print(f"  {'layer':<20}{'share':>8}  {'lever':<38}driver")
    print("  " + LINE[:74])
    for l in S.layer_table():
        print(f"  {l['name']:<20}{_pct(l['share']):>8}  {l['lever']:<38}{l['driver']}")
    print(f"  {'TOTAL':<20}{_pct(S.BILL_TOTAL):>8}")

    print("\nThe cache discount, swept. `line` is the reduction on the PREFIX LINE; `banked` is what")
    print("that is worth against the whole bill -- and the two differ by a factor of three because")
    print("the prefix is only 31.05% of it:")
    print(f"  {'discount':>9}{'line now':>10}{'line fixed':>12}{'line red.':>11}{'banked':>9}")
    print("  " + LINE[:52])
    for r in S.discount_sweep():
        print(f"  {r['discount']:>8.1f}x{r['prefix_line_now']:>10.4f}{r['prefix_line_fixed']:>12.4f}"
              f"{_pct(r['line_reduction']):>11}{_pct(r['blended_saving']):>9}")
    a = S.asymptotic_reduction()
    print(f"  {'INFINITE':>9}{'':>10}{'':>12}{_pct(a['line_reduction']):>11}{_pct(a['blended_saving']):>9}"
          f"   <-- the asymptote")

    be = S.write_premium_break_even(10.0, 1.25)
    print(f"\nThe write premium [R]: a cached prefix costs {be['write_price']:.2f}x a fresh token to")
    print(f"WRITE. At a {be['discount']:.0f}x read discount it needs {be['n_reads']:.2f} reads to pay")
    print(f"for itself -- {be['n_reads_ceil']} in whole reads. A prefix read once is a bill INCREASE:")
    for n in (1, 2, 5):
        c = S.caching_npv(n, 10.0, 1.25)
        print(f"  {n} read(s): uncached {c['uncached']:.2f}  cached {c['cached']:.3f}  "
              f"saving {c['saving']:+.3f}  {'worth it' if c['worth_it'] else 'NOT worth it'}")

    r = S.reasoning_amplification(300, 2000)
    print(f"\nReasoning tokens are billed at the output rate and absent from the response. A call")
    print(f"whose visible answer is {r['visible']} tokens and which thinks for {r['reasoning']} bills")
    print(f"{r['total_billed']} -- an amplification of {r['amplification']:.2f}x, while the metric")
    print("'tokens per output' reports 1.0x for this call by construction.")

    print("\nFINDING." + """
  The discount is not the saving and the gap is not a rounding error: a 10x read discount cuts the
  prefix line by 74.6% and the bill by 23.2%. Three numbers matter and only one is usually quoted.
  First, the BANKED number -- 23.2% -- because the prefix is 31% of the bill, not all of it.
  Second, the ASYMPTOTE -- 26.7%, reached at an infinite discount. A free cache read cannot do
  better than 26.7% of the bill, because the uncached share of calls (a 10% target) still pays full
  price and everything outside the prefix is untouched. Any business case claiming more from caching
  alone has confused the two units. Third, the WRITE PREMIUM, which is the only number here that can
  be negative: at a 10x discount and a 1.25x write price, a prefix must be read 1.39 times to break
  even, so a rollout that enables caching on single-shot traffic is a bill INCREASE with a dashboard
  that says caching is on. Turn it on for stable system prompts and tool definitions, measure the
  read-to-write ratio per prefix, and disable it per prefix -- not globally -- when the ratio is
  below 2.
""")


# ----------------------------------------------------------------------------------------------
def exp_lever_ladder() -> None:
    """2. The ladder's claimed order, its achievable order, and its ceiling."""
    _hdr(2, "THE LEVER LADDER -- 'roughly a tenth of the cost' is a claim with a number in it")
    print("""
[corpus] "Stack the techniques, do not look for a hero optimization... before you touch the model
          at all, exhaust the free wins."                                       -- tokenomics [T]
[corpus] "...from a naive baseline to roughly a tenth of the cost, with no change to the model
          itself."                                                              -- tokenomics [T]
[repo]   distillation: 5-40x per-token cut, and it "wins on narrow, high-volume tasks and fails on
         open-ended long-tail work".                                                       [R]
[repo]   difficulty routing / cascades: 45-85% cost cut at ~95% quality.                   [R]
""")
    print("The ladder, ranked by the largest multiple each claim implies -- which is what reading it")
    print("gives you. Three claims are UNQUANTIFIED in the corpus and are marked; the multiple shown")
    print("for those is the author's reading, not a parse:")
    print(f"  {'#':>3}  {'lever':<20}{'claim':>8}  {'notation':<11}claim, as the corpus states it")
    print("  " + LINE[:74])
    for i, r in enumerate(L.rank_by_claim(), 1):
        note = "  (UNQUANTIFIED in corpus)"
        print(f"  {i:>3}  {r['lever']:<20}{r['claim_multiple']:>7.2f}x  {r['notation']:<11}"
              f"{r['claim'][:44]}{note if not r['quantified'] else ''}")

    print("\nThe same levers ranked by ACHIEVABLE saving -- what a programme can actually bank, which")
    print("is `discount x share_of_bill`, not the discount:")
    print(f"  {'#':>3}  {'lever':<20}{'banked':>9}  {'kind':<10}{'ease':>5}   moved")
    print("  " + LINE[:60])
    d = L.ordering_disagreement()
    for i, r in enumerate(L.rank_by_achievable(), 1):
        p = d["positions"][r["lever"]]
        delta = p["claim"] - i
        arrow = "" if delta == 0 else (f"up {delta}" if delta > 0 else f"down {-delta}")
        print(f"  {i:>3}  {r['lever']:<20}{_pct(r['saving']):>9}  {r['kind']:<10}{r['ease']:>5}   {arrow}")

    h = L.headline_check()
    print(f"\nThe headline, tested. 'A tenth of the cost' is a {_mult(0.90)} reduction of the bill.")
    print(f"  ladder ceiling (every lever at its full claimed discount, disjoint shares): "
          f"{_pct2(h['ceiling'])}  = {_mult(h['ceiling'])}")
    print(f"  headline:                                                                  "
          f"{_pct2(h['headline'])}  = {_mult(h['headline'])}")
    print(f"  GAP:                                                                       "
          f"{_pct2(h['gap'])}  --  {'reachable' if h['reachable'] else 'NOT REACHABLE'}")
    t = h["top_claim"]
    print(f"\nThe most aggressive number in the ladder is {t['lever']}: a {t['discount']:.0%} cut on")
    print(f"{_pct(t['share'])} of the {t['layer']} line. It is the only claim whose widest reach")
    print("approaches the headline -- and the same repo that supplies it says it fails on the long")
    print("tail, which is where the agentic workload is going (see experiment 4).")
    without = L.ceiling([x for x in L.LEVERS if x.name != "distillation"])
    print(f"  ceiling WITHOUT distillation:  {_pct2(without['reduction'])}  = {_mult(without['reduction'])}")
    print(f"  so the whole 40x claim is worth {_pct2(h['ceiling'] - without['reduction'])} of the ceiling,")
    print(f"  against the {_pct2(h['gap'])} the headline still needs after it.")

    print("\nFINDING." + """
  The ladder is a list of DISCOUNTS and the headline is an AMOUNT, and the two orderings disagree.
  Distillation is #1 by claim and #8 by bankable saving, because 40x on a 15% share is 2.4 points;
  prompt caching is #3 by claim and #1 by bankable saving, because 10x on a 31% share is 23.2 points.
  Eight of the ten levers move position between the two orderings. This is not a criticism of the
  ladder -- it is the arithmetic that turns it into a plan, and the plan starts somewhere other than
  the top of the list.
  The headline does not survive the test honestly applied. With every lever at its full claimed
  discount, taking disjoint shares so nothing is counted twice, the ceiling is 60.6% -- a 2.54x
  reduction, not 10x -- and the headline needs 90%. The 29.4-point gap is not a missing technique;
  it is a missing ARGUMENT. The ladder's own most aggressive number, distillation's 40x, is worth
  2.0 points of the ceiling at its stated reach, and would have to be applied to the whole non-prefix
  bill to move the ceiling meaningfully -- which is exactly the application the same source calls a
  failure mode. Quote the ceiling, not the headline. 60.6% off, no model change, is an excellent
  programme.
""")


# ----------------------------------------------------------------------------------------------
def exp_ordering() -> None:
    """3. Three defensible orderings of the same ten levers, and the price of the rule."""
    _hdr(3, "THE ORDERING -- 'do the easy things first' is the most expensive of three rules")
    print("""
[corpus] "Before you touch the model at all, exhaust the free wins."             -- tokenomics [T]
[repo]   batch: ~50% discount with a ~24h ceiling. Reserved capacity: 15-70% on sustained load.
         Both price the SAME tokens -- they are one saving, not two.                        [R]
""")
    oc = L.ordering_comparison()
    for name, r in oc["orders"].items():
        print(f"\n{name}  ->  {_pct(r['reduction'])} off (final bill {_pct(r['final'])})")
        for s in r["steps"]:
            mark = f"   <-- SKIPPED: {s['skipped']}" if s["skipped"] else ""
            print(f"    {s['lever']:<20} saved {_pct(s['saved']):>7}{mark}")
    print(f"\n  best {_pct(oc['best'])}   worst {_pct(oc['worst'])}   spread {_pct2(oc['spread'])}")
    print(f"  free wins (ease >= 4): {', '.join(L.free_wins())}")
    print("\n  conflicts -- levers that cannot both apply to the same traffic:")
    for c in L.conflict_pairs():
        print(f"    {c['a']}  x  {c['b']}")

    print("\nFINDING." + """
  All three orderings are defensible from the corpus's own instruction, they differ by 4.2 points of
  the bill, and the worst of them is the one that follows the most natural reading -- do the easy
  things first. The mechanism is not that easy levers are weak; it is that a PRICING lever applies to
  whatever is left, so there is one pricing slot, and ease picks the cheaper occupant of it.
  batch_lane (ease 4) beats reserved_capacity (ease 2) to the slot; once batch_lane is applied,
  reserved_capacity is skipped as conflicted -- 15.8 points of the bill lost to an ordering rule, and
  the same ordering also loses distillation behind its routing conflict. The saving-first order does
  the reverse: it takes reserved_capacity first (15.8 points) and drops batch_lane, which is the
  correct trade, and it reaches 59.3%.
  The design rule is therefore not an ordering at all -- it is a rule about WHICH LEVERS COMPETE.
  Price levers are mutually exclusive per traffic slice and must be decided once, on value; reduction
  levers are mostly independent and can be ordered by ease among themselves. A plan that sorts all
  ten levers on one axis is choosing its 4.2 points by accident.
""")


# ----------------------------------------------------------------------------------------------
def exp_unit() -> None:
    """4. The unit that decides the sign, and the mix that decides the order."""
    _hdr(4, "THE UNIT -- 30c to 16c is a 47% saving, a 22% increase and a 270% increase")
    print("""
[corpus] "Between February and April the cost of a 10,000-token knowledge artifact fell from 30c
          to 16c, while lines of code drafted fell from 630 to 91 and files touched from 8.2 to 3.6,
          and thinking tokens rose."                                        -- Token Raj [T]
[corpus] "Companies are doing tokens per output as one of the metrics -- you need to change the
          metrics. You need to start looking at it as a business outcome rather than the token
          spend."                                                           -- Token Raj [T]
[repo]   an agentic multi-step task can run from tens of cents to several dollars; a chat turn is
         cents.                                                                             [R]
""")
    n = U.normalised(U.FEB, U.APR)
    print("The same two rows under three normalisations. A ratio above 1.00 means MORE expensive:")
    print(f"  {'unit':<14}{'ratio':>8}{'change':>10}  {'reading':<16}note")
    print("  " + LINE[:74])
    for r in U.unit_ranking():
        print(f"  {r['unit']:<14}{r['ratio']:>8.4f}{r['change_pct']:>+9.1f}%  {r['reading']:<16}{r['note']}")
    print(f"\n  unit costs before:  artifact {n['per_unit_cost_before']['artifact']:.4f}c   "
          f"file {n['per_unit_cost_before']['file']:.4f}c   line {n['per_unit_cost_before']['line']:.4f}c")
    print(f"  unit costs after:   artifact {n['per_unit_cost_after']['artifact']:.4f}c   "
          f"file {n['per_unit_cost_after']['file']:.4f}c   line {n['per_unit_cost_after']['line']:.4f}c")
    print(f"  signs disagree: {n['sign_disagreement']}")
    f = U.deflation_check()
    print(f"\n  Was it the same task? cost ratio {f['cost_ratio']:.4f}  vs  lines ratio "
          f"{f['lines_ratio']:.4f}   ->  {f['verdict']}")
    print(f"  real cost per line of delivered work: {f['real_cost_change']:.2f}x")

    print("\nAnd a time axis on the same shape -- the average per call against the population that")
    print("dominates the bill:")
    m = U.mix_summary()
    print(f"  {'workload':<12}{'calls':>10}{'vol share':>11}{'cost':>11}{'bill share':>12}")
    print("  " + LINE[:56])
    for r in m["rows"]:
        print(f"  {r['name']:<12}{r['calls']:>10,}{_pct(r['call_share']):>11}{r['cost']:>11.2f}"
              f"{_pct(r['cost_share']):>12}")
    t = U.tail_ratio()
    print(f"  mean cost per call {t['mean_per_call']:.6f}  vs  an agentic call {t['costly_call']:.4f}"
          f"   ->  {t['ratio']:.1f}x")
    g = U.growth_scenario()
    for k in ("chat", "agentic"):
        print(f"  {k:<10} bill share  {_pct(g['before_share'][k]['cost_share'])}  ->  "
              f"{_pct(g['after_share'][k]['cost_share'])}   after {g['factor']}x growth")
    print(f"  total {g['before']['total_cost']:.0f}  ->  {g['after']['total_cost']:.0f}")

    print("\nFINDING." + """
  One decision -- the unit -- produces three readings of one pair of numbers and they do not share a
  sign. Per artifact the price fell 46.7%; per file it rose 21.5%; per line of delivered work it rose
  269.2%. The reason is visible in the raw data and invisible in the headline: the artifact shrank
  faster than the price did (lines fell to 14.4% of their February value while cost fell to 53.3%),
  so the per-unit cost of DELIVERED WORK tripled while the per-request cost halved. Both statements
  are true. The one that goes on the slide is the one whose denominator is what was asked for, and
  the one that predicts a business outcome is the one whose denominator is what was delivered.
  This is the most directly transferable finding in the topic, because the metric is not the problem
  -- the DENOMINATOR is, and the team being measured usually picks it. 'Tokens per output' is the
  sharpest case: reasoning tokens are billed at the output rate and never appear in the response, so
  a call with a 300-token answer and 2,000 thinking tokens scores 1.0x on the metric and costs 7.7x.
  Pick the denominator first, in writing, before the programme starts -- and pick one whose numerator
  the team cannot shrink by doing less work.
  The mix adds the time axis. Today the expensive population is 1.0% of volume and 29.2% of the bill,
  and an agentic call costs 29.2x the mean call. Grow that population 4.3x -- the scenario's own rate
  -- and it becomes 63.9% of the bill with the total rising from 4,194 to 8,233. So the optimisation
  order derived from TODAY's average is the wrong order for NEXT YEAR's bill, and it is wrong in a
  predictable direction: the tail is where the money is going.
""")


# ----------------------------------------------------------------------------------------------
def exp_sovereignty() -> None:
    """5. Four dimensions with prerequisites, a mean and a floor, and the attestation price."""
    _hdr(5, "SOVEREIGNTY -- a vector, not a binary; a floor, not a mean; a filter, not a score")
    print("""
[corpus] "People look at sovereignty as something near-binary -- it is a sovereign setup or a
          non-sovereign setup. I think that is wrong... you need to start looking at it as a system
          property, and this property has various dimensions."              -- sovereign AI [T]
[corpus] "You cannot legislate a memory dump."                                   -- sovereign AI [T]
[corpus] "It is a trade-off. It is not going to run as fast as what it was running before. But it
          is going to run more securely."                                        -- sovereign AI [T]
[corpus] "The choice of accelerator is extremely important... it should be part of the implementation
          strategy."                                                             -- sovereign AI [T]
""")
    print("The four dimensions and what each one actually asks:")
    for d in V.DIMENSIONS:
        print(f"  {d:<11}{V.DIMENSION_QUESTION[d]}")
    print("\nPrerequisites -- a dimension cannot exceed the one it depends on:")
    for k, v in V.PREREQ.items():
        print(f"  {k} <= {v}")

    print("\nPostures, claimed against effective. `capped` is where the prerequisite bit:")
    print(f"  {'posture':<18}{'claimed':>9}{'effective':>11}  {'capped':<12}weakest")
    print("  " + LINE[:66])
    for r in V.posture_table():
        print(f"  {r['posture']:<18}{r['claimed_mean']:>9.3f}{r['effective_mean']:>11.3f}  "
              f"{str(r['capped']):<12}{r['weakest_link']}")

    print("\nThe same four scores read as a mean and read as a floor. The mean is what a slide shows;")
    print("the floor is what an adversary or a regulator gets to choose:")
    print(f"  {'posture':<18}{'mean':>8}{'min':>8}{'gap':>8}")
    print("  " + LINE[:42])
    for p in V.POSTURES:
        mm = V.mean_vs_min(p)
        print(f"  {mm['posture']:<18}{mm['mean']:>8.3f}{mm['min']:>8.3f}{mm['gap']:>8.3f}")

    print("\nThe attestation arithmetic -- the one trade-off in the corpus that CAN be quantified:")
    print(f"  {'throughput penalty':>19}{'cost uplift':>13}   and blended over part of the traffic")
    print("  " + LINE[:60])
    for r in V.tee_budget_table(penalties=(0.05, 0.10, 0.20, 0.30, 0.40), shares=(0.40, 1.00)):
        print(f"  {_pct(r['penalty']):>19}{_pct(r['uplift_per_token']):>13}   "
              f"at {_pct(r['attested_share'])} attested -> {_pct(r['blended_uplift'])} on the total bill")

    print("\nResidency as a SCORE, against residency as a FILTER. The in-region option must win by")
    print("`gap` against a scoring noise of `noise`:")
    print(f"  {'gap':>6}{'noise':>7}{'leak rate':>11}{'leaks / 10k':>13}{'as a filter':>13}")
    print("  " + LINE[:50])
    for gap, noise in ((2.0, 1.0), (1.0, 1.0), (0.5, 1.0)):
        r = V.residency_leaks(gap, noise, 10_000)
        print(f"  {gap:>6.1f}{noise:>7.1f}{_pct(r['leak_rate']):>11}{r['leaks']:>13.1f}{0.0:>13.1f}")
    c = V.residency_compliance(0.159)
    print(f"\n  a 99.84% correct router is {c['verdict']}")
    print(f"  the alternative wording, for the deck: {'compliant' if c['compliant_as_filter'] else 'NOT compliant'}")

    print("\nContinuity, over four layers. Two independent options at a layer is the threshold:")
    print(f"  {'accelerator':>12}{'model':>7}{'engine':>8}{'region':>8}{'floor':>8}{'mean':>8}"
          f"  {'binding':<14}")
    print("  " + LINE[:72])
    for r in V.continuity_examples():
        s = r["sources"]
        print(f"  {s['accelerator']:>12}{s['model_family']:>7}{s['serving_engine']:>8}"
              f"{s['cloud_region']:>8}{r['continuity']:>8.1f}{r['mean']:>8.2f}  "
              f"{r['binding_layer']:<14}")

    print("\n'Open weights' as a spectrum rather than a checkbox -- the corpus's own objection is")
    print("that 'when people say open source model they just publish open weights' [T]:")
    print(f"  {'name':<32}{'weights':>9}{'code':>7}{'data':>7}{'repro':>7}{'openness':>10}")
    print("  " + LINE[:64])
    for r in V.open_weights_spectrum():
        print(f"  {r['name']:<32}{str(r['weights']):>9}{str(r['training_code']):>7}"
              f"{str(r['training_data']):>7}{str(r['reproducible']):>7}"
              f"{V.spectrum_position(r['name'])['openness']:>10.2f}")

    print("\nFINDING." + """
  Sovereignty is a vector, and the two readings of a vector disagree in the direction that matters.
  compliance_only has the largest mean-to-floor gap in the table (0.338) and api_with_dpa the second
  (0.238): the postures that sound most defensible on a slide are the ones whose weakest dimension is
  nearest zero, because a contract and a regional endpoint do not make the model yours. A review that
  scores the four dimensions and averages them is measuring the slide.
  The prerequisite rule is not decoration. vendor_tee_lease claims 0.85 trust and can evidence 0.50,
  because the silicon and the model are not yours -- you are attesting a property of an environment
  you did not choose. That is a 0.087 cut in the effective mean and it is invisible unless the cap is
  computed rather than assumed.
  Two things here are arithmetic rather than judgement, and both are usually got wrong. A 20%
  throughput penalty for confidential computing is a 25% cost uplift, not 20% -- capacity is what you
  buy, and 1/(1-0.20) = 1.25. And a residency rule has NO error tolerance: a score-based router leaks
  15.9% of requests at a modest gap and noise (1,587 per 10,000), a filter leaks zero, and 99.84%
  correct residency is not 99.84% compliant, it is non-compliant. Residency must be implemented by
  removing the out-of-region option, never by discounting it -- the same conclusion T14 reaches about
  routing quality, arriving here as a legal requirement instead of a performance one.
  Continuity is a minimum over layers and not a mean. Three of the four examples below score a mean
  of 0.75 and a floor of zero: two accelerators, two clouds and two model families with ONE serving
  engine is a single-vendor system, and the mean will not say so. The cheapest continuity purchase in
  most stacks is not a second cloud -- it is a second serving runtime on the same weights.
""")


# ----------------------------------------------------------------------------------------------
def exp_attribution() -> None:
    """6. The precondition: coverage that falls as the thing it measures grows."""
    _hdr(6, "ATTRIBUTION -- the metric that degrades as the business succeeds")
    print("""
[repo]   "Tag every call by team, feature, tenant, model, route and environment. Without attribution
         there is no way to compute unit economics." Showback before chargeback, and chargeback only
         "once the tags are trustworthy".                                                   [R]
[repo]   "Runaway agents have burned tens of thousands of dollars over a single weekend."   [R]
""")
    c = A.coverage()
    print("Coverage, and the two readings of it:")
    print(f"  {'team':<18}{'spend':>11}{'tagged':>9}{'untagged':>11}{'growth':>8}  note")
    print("  " + LINE[:74])
    for r in c["rows"]:
        print(f"  {r['team']:<18}{r['spend']:>11,.0f}{_pct(r['tagged_share']):>9}"
              f"{r['untagged']:>11,.0f}{r['growth']:>7.1f}x  {r['note']}")
    print(f"  {'TOTAL':<18}{c['total_spend']:>11,.0f}{c['tagged_spend']:>9,.0f}"
          f"{c['untagged_spend']:>11,.0f}")
    print(f"\n  spend-weighted coverage {_pct2(c['spend_weighted_coverage'])}   "
          f"simple mean of the percentages {_pct2(c['simple_mean_coverage'])}")
    print(f"  worst-instrumented team: {c['worst_team']['team']} at "
          f"{_pct(c['worst_team']['tagged_share'])}, growing {c['worst_team']['growth']:.1f}x")

    g = A.untagged_share_of_growth()
    print(f"\nNow let each team grow at its own rate, scaled so that nothing shrinks (the slowest-")
    print(f"growing team is the reference, {g['growth_reference']:.1f}x):")
    print(f"  coverage          {_pct2(g['coverage_before'])}  ->  {_pct2(g['coverage_after'])}"
          f"   ({g['coverage_delta'] * 100:+.2f} points)")
    print(f"  untagged share    {_pct2(g['untagged_share_before'])}  ->  "
          f"{_pct2(g['untagged_share_after'])}")
    print(f"  untagged spend    {g['untagged_before']:>10,.0f}  ->  {g['untagged_after']:,.0f}")
    print(f"  total bill        {g['before']['total_spend']:>10,.0f}  ->  "
          f"{g['after']['total_spend']:,.0f}   ({g['total_growth']:.2f}x)")

    print("\nThe governance ladder, and what each rung needs before it works:")
    for r in A.governance_ladder():
        print(f"  {r['stage']:<20}needs {r['needs']:<38}pressure {r['behavioural_pressure']:.1f}"
              f"   fails: {r['failure']}")

    s = A.shadow_gap(880_000, 1_000_000)
    print(f"\nThe reconciliation that detects traffic leaving the gateway. Provider invoices of")
    print(f"  {s['provider_invoice']:,} against gateway spend of {s['gateway_spend']:,} leave a")
    print(f"  residual of {s['residual']:,} -- {_pct(s['shadow_share'])} of spend the governance")
    print(f"  plane cannot see. Visible within 1%? {s['visible']}. A residual that size is not a")
    print("  rounding error, it is traffic that left the gateway.")

    cap = A.cap_from_mean_vs_tail(0.0042, 0.1224, 5.00)
    print(f"\nA per-run ceiling, drawn from the mean or from the tail:")
    print(f"  mean task cost {cap['mean_task_cost']:.4f}   p99 task cost {cap['p99_task_cost']:.4f}"
          f"   ceiling {cap['ceiling']:.2f}")
    print(f"  -> {cap['verdict']}")

    print("\nFINDING." + """
  Attribution is the precondition the corpus puts first, and it is the one metric in this topic whose
  coverage FALLS as the business succeeds. Total coverage starts at 90.5%, and the reason a single
  growth cycle takes it to 82.4% is not that anyone got worse at tagging -- it is that the untagged
  spend is concentrated in the team growing 6.5x. Untagged spend is not a random sample of the bill,
  so a 90% coverage figure is not '90% of the picture', it is the whole picture minus the part that is
  about to matter. Untagged share doubles (9.5% to 17.6%) over the same period.
  Two design consequences follow, and neither is about tagging more diligently. First, coverage must
  be reported per team WITH the growth rate beside it, because the pair is the signal and either alone
  is not: a team at 55% coverage growing 6.5x is the finding. Second, the failure mode chargeback
  introduces has to be instrumented before chargeback is enabled -- once spend is billed to a team,
  the cheapest route for that team is an account the gateway never sees, and the only detector is
  reconciliation. A 12% residual between gateway spend and provider invoices is what routing around
  the gateway looks like from the inside.
  The ceiling rule is the same shape one level down. A run cap drawn from the MEAN terminates
  legitimate work, and a cap drawn from the tail catches exactly the runaway the corpus warns about.
  With a mean task cost of 0.4c and a p99 of 12.2c, a $5 ceiling kills nothing legitimate and still
  catches a runaway two orders of magnitude above the mean. Set the ceiling from the tail; the mean is
  not a smaller version of the tail, it is a different population.
""")


# ----------------------------------------------------------------------------------------------
def run_all() -> None:
    exp_cost_stack()
    exp_lever_ladder()
    exp_ordering()
    exp_unit()
    exp_sovereignty()
    exp_attribution()


if __name__ == "__main__":
    run_all()
