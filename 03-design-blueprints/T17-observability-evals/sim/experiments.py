"""The nine demonstrations. Each prints a table and then a FINDING.

A demonstration that only prints numbers is a dashboard. The requirement here is that each function
also states what the numbers mean for a design decision, so the HLD can quote a run rather than an
opinion.
"""
from __future__ import annotations

from . import gate as G
from . import judge as J
from . import latency as L
from . import loop as LP
from . import sampling as S
from . import stats as ST

LINE = "-" * 78


def _hdr(n, title: str) -> None:
    print("\n" + "=" * 78)
    print(f"{n}. {title}")
    print("=" * 78)


# ----------------------------------------------------------------------------------------------
def exp_latency_decomposition() -> None:
    """1. A trace is a decomposition, not a measurement."""
    _hdr(1, "LATENCY DECOMPOSITION -- the corpus's 1.2 s request, and where the time went")
    print("""
[corpus] "Here is one real request drawn as a stack of spans. The total was 1.2 seconds. Retrieval
          took 90 milliseconds. Generation took over a second. The bottleneck is obvious the moment
          you can see it." -- LLM Observability: Traces, Spans, OTel [T]
""")
    root = L.chat_request()
    print("  THE TRACE, as a UI would render it:")
    for line in L.render_tree(root):
        print("    " + line)

    chk = L.decomposition_check(root)
    print(f"\n  end-to-end      {chk['root_ms']:8.1f} ms")
    print(f"  leaf sum        {chk['leaf_sum_ms']:8.1f} ms")
    print(f"  residual        {chk['residual_ms']:8.1f} ms  ({chk['residual_pct']:.2f}%)"
          f"   consistent={chk['consistent']}")

    att = L.attribution(root, target_ms=300.0)
    print(f"\n  rank by SELF time   {' > '.join(att['rank_by_self'][:5])}")
    print(f"  rank by SPAN COUNT  {' > '.join(att['rank_by_count'][:5])}")
    print(f"  does a count ranking discriminate here? {att['count_discriminates']}"
          f"   (every span occurs once, so it cannot)")
    print(f"\n  top span by self time: {att['top_by_self']} owning {att['top_share_pct']:.1f}% "
          f"of end-to-end")
    print(f"  a 300 ms target is reachable by attacking it alone: {att['target_reachable']}")

    print("\n  NOW THE SAME ANALYSIS ON AN AGENTIC REQUEST (12 tools, 5 LLM turns):")
    ag = L.agentic_request()
    att2 = L.attribution(ag, target_ms=500.0)
    print(f"    end-to-end {ag.ms:.0f} ms across {len(L.leaves(ag))} leaf spans")
    print(f"    rank by SELF    {' > '.join(att2['rank_by_self'][:5])}")
    print(f"    rank by COUNT   {' > '.join(att2['rank_by_count'][:5])}")
    print(f"\n    {'name':<16}{'count':>7}{'self ms':>11}{'pct e2e':>10}")
    print("    " + LINE[:44])
    for a in att2["by_self"][:7]:
        pct = a["self_ms"] / ag.ms * 100.0
        print(f"    {a['name']:<16}{a['count']:>7}{a['self_ms']:>11.1f}{pct:>10.1f}")

    print(f"\n    the two rankings disagree: {att2['disagree']}")
    print(f"    LLM spans own {sum(a['self_ms'] for a in att2['by_self'] if a['name'] in ('ttft', 'decode')):.0f} ms;"
          f"  tool calls own "
          f"{sum(a['self_ms'] for a in att2['by_self'] if a['name'] == 'tool_call'):.0f} ms")

    print("""
  FINDING: the two rankings DISAGREE on the agentic request and cannot even be formed on the chat
  turn. In the agentic case `tool_call` wins on count (12 spans) and loses by more than an order of
  magnitude on self time, because the LLM spans dominate the wall clock. A metrics-only dashboard
  cannot make this distinction at all -- a metric has no name attached, so "p95 latency regressed"
  and "tool calls are slow" are the same sentence. This is the concrete argument for traces over
  metrics, and it gets STRONGER as workloads become agentic, which the corpus says is where the
  traffic went ("about 70% of the inference traffic these days is agentic" [T], llm-d).

  TWO TRAPS IN BUILDING THIS TABLE, both avoided above and both producing a confidently wrong answer:
  the root span owns 100% of the wall clock by construction, and a parent's total already contains
  its children's. Rank by total and include the root and the answer is "your request is slow, and
  also your LLM call is slow twice". Rank by SELF time with the root excluded and self times sum to
  the end-to-end total exactly -- the `residual` line is the check that the instrument works.""")


# ----------------------------------------------------------------------------------------------
def exp_span_sum_invariants() -> None:
    """1b. The two invariants a trace must satisfy, checked rather than assumed."""
    _hdr("1b", "TRACE INVARIANTS -- does the instrument sum, and does it nest")
    root = L.chat_request()
    chk = L.decomposition_check(root)
    agg = L.by_name(root)
    self_sum = sum(a["self_ms"] for a in agg.values())
    print(f"  chat request: root {chk['root_ms']:.1f} ms, leaf sum {chk['leaf_sum_ms']:.1f} ms, "
          f"residual {chk['residual_pct']:.2f}%")
    print(f"  self-time sum across all span names: {self_sum:.1f} ms "
          f"(must equal the root: {abs(self_sum - root.ms) < 1e-6})")
    ag = L.agentic_request()
    agg2 = L.by_name(ag)
    self_sum2 = sum(a["self_ms"] for a in agg2.values())
    print(f"  agentic request: root {ag.ms:.1f} ms, self-time sum {self_sum2:.1f} ms "
          f"(equal: {abs(self_sum2 - ag.ms) < 1e-6})")
    print("""
  FINDING: self time is the ONLY aggregation that sums to the end-to-end total without double
  counting, which is why `attribution` ranks by it. A trace view built on total time looks plausible
  and over-reports every nested span -- and because the error is proportional to nesting depth, it
  over-reports the agentic requests (deep) far more than the chat turns (flat). The instrument
  degrades exactly where the workload got complicated.""")


# ----------------------------------------------------------------------------------------------
def exp_sampling_frontier() -> None:
    """2. Every sampling policy's cost and coverage, side by side."""
    _hdr(2, "TRACE SAMPLING -- cost against coverage, and the cell the cost dashboard hides")
    print("""
[corpus] "In development, trace everything. ... In high-traffic production, sample the successes,
          but always keep 100% of the errors. And redact personal data at the boundary before it is
          ever written." -- LLM Observability [T]
[corpus] the collector example: errors 100%, latency > 10 s 100%, baseline probabilistic 5% [R]
""")
    n = 100_000
    err, slow = 0.012, 0.008
    rows = S.frontier(n, err, slow)
    print(f"  workload: {n:,} requests/release, {err:.1%} errored, {slow:.1%} slow-but-not-errored\n")
    print(f"  {'policy':<14}{'kept':>9}{'kept%':>8}{'err cov':>9}{'slow cov':>10}{'MB/rel':>9}{'tail?':>7}")
    print("  " + LINE[:70])
    for r in rows:
        print(f"  {r['policy']:<14}{r['kept']:>9,.0f}{r['kept_pct']:>8.1f}"
              f"{r['error_coverage'] * 100:>8.1f}%{r['slow_coverage'] * 100:>9.1f}%"
              f"{r['storage_gb'] * 1000:>9.1f}{'yes' if r['tail'] else 'no':>7}")

    base = next(r for r in rows if r["policy"] == "uniform_5pct")
    tail = next(r for r in rows if r["policy"] == "tail_sample")
    allr = next(r for r in rows if r["policy"] == "trace_all")
    print(f"""
  THE CELL THAT MATTERS: uniform_5pct and tail_sample keep ALMOST THE SAME NUMBER OF SPANS
  ({base['kept_pct']:.1f}% vs {tail['kept_pct']:.1f}%) and have completely different error coverage
  ({base['error_coverage'] * 100:.0f}% vs {tail['error_coverage'] * 100:.0f}%). A cost dashboard shows the
  first column; an incident review needs the second. Tail sampling costs
  {tail['storage_gb'] / allr['storage_gb'] * 100:.1f}% of trace-all storage to keep {tail['error_coverage'] * 100:.0f}%
  of the errors, which is the entire frontier in one line.

  [corpus] "redact personal data at the boundary before it is ever written" [T] -- and sampling does
  not substitute for it. A 1% sample of unredacted prompts is still an incident, just a smaller one.
  `redaction_is_orthogonal()` returns that as a struct so a config validator can refuse a
  non-redacting collector at ANY sampling rate.""")


# ----------------------------------------------------------------------------------------------
def exp_head_sampling_blind_spot() -> None:
    """3. Why 'keep 100% of the errors' is about RARITY, not about cost."""
    _hdr(3, "THE BLIND SPOT IN HEAD SAMPLING -- rarity, not cost, is the argument")
    window = 10_000
    print(f"  window: {window:,} requests (roughly an hour at 3 rps)\n")
    print(f"  {'incident rate':>15}{'keep':>7}{'P(no trace)':>13}{'P(caught)':>11}"
          f"{'reqs for 95% conf':>20}")
    print("  " + LINE[:70])
    for rate, keep, label in [(0.10, 0.05, ""), (0.01, 0.05, ""), (0.001, 0.05, ""),
                              (0.0001, 0.05, ""), (0.001, 1.00, ""), (0.0001, 1.00, "")]:
        pz = S.p_zero_captured(rate, keep, window)
        need = S.requests_until_first_trace(rate, keep, 0.95)
        need_s = f"{need:,.0f}" if need != float("inf") else "never"
        print(f"  {rate:>15.4%}{keep:>7.0%}{pz:>13.4f}{1 - pz:>11.4f}{need_s:>20}")

    print("\n  At a 3 requests/s arrival rate:")
    for rate in (0.01, 0.001, 0.0001):
        need5 = S.requests_until_first_trace(rate, 0.05, 0.95)
        need100 = S.requests_until_first_trace(rate, 1.00, 0.95)
        print(f"    rate {rate:.4%}: uniform 5% needs {need5 / 3 / 3600:8.1f} h for one trace;  "
              f"keep-100% needs {need100 / 3 / 3600:6.2f} h")

    print("""
  FINDING: head sampling's blind spot is not "fewer traces", it is "for a rare failure, NO trace,
  exactly when one is needed". P(no trace) is monotone in rarity, so the failures that matter most
  are the ones most likely to be invisible -- and at a 1-in-10,000 rate with 5% sampling, a
  full hour of traffic produces a trace with probability 0.5%.

  This is why the corpus's rule is stated the way it is: "always keep 100% of the errors". The
  reason is not that errors are cheap to store (1-2% of traffic here) -- it is that a sampled error
  trace is an error trace you do not have. **Tail sampling is a correctness requirement for
  debuggability, not a cost optimization.**""")


# ----------------------------------------------------------------------------------------------
def exp_judge_corrections() -> None:
    """4. Each of the three corrections, alone, so their value is separable."""
    _hdr(4, "JUDGE BIAS -- the three corrections, and what each one actually buys")
    print("""
[corpus] "the judge itself is biased. It tends to favor the first answer it sees. It rewards length
          even when length adds nothing, and it flatters outputs from its own model family. So,
          randomize the answer order, cap or normalize length, and never let a model be the sole
          judge of its own family." -- How to Evaluate LLM Apps, citing Jung et al. 2023 [T]
""")
    gold = J.synthetic_gold_set()
    print(f"  gold set: {gold.n} pairs, mean length {gold.length_mean:.0f} tokens (sd {gold.length_sd:.0f})")
    print(f"  family mix: {gold.family_mix()}")
    print(f"  label balance: {gold.label_balance():.3f} of truths are slot `a`"
          f"   degenerate={gold.is_degenerate()}")
    print(f"  judge: position_bias={J.DEFAULT_JUDGE.position_bias}, "
          f"verbosity_weight={J.DEFAULT_JUDGE.verbosity_weight}, "
          f"self_pref={J.DEFAULT_JUDGE.self_pref}, noise={J.DEFAULT_JUDGE.noise}\n")
    rows = J.correction_ladder(gold)
    print(f"  {'variant':<22}{'agreement':>11}{'kappa':>9}{'close-call agr':>16}")
    print("  " + LINE[:58])
    for r in rows:
        print(f"  {r['variant']:<22}{r['agreement'] * 100:>10.1f}%{r['kappa']:>9.3f}"
              f"{r['close_agreement'] * 100:>15.1f}%")

    base = rows[0]["agreement"]
    best = rows[-1]["agreement"]
    singles = [(r["variant"], r["agreement"] - base) for r in rows[1:-1]]
    ranked = sorted(singles, key=lambda x: -x[1])
    print(f"\n  naive -> all three: {base * 100:.1f}% -> {best * 100:.1f}% "
          f"({(best - base) * 100:+.1f} points)")
    for name, d in singles:
        print(f"    {name:<22} alone changes agreement by {d * 100:+.1f} points")
    print(f"    {'+ all three':<22} changes it by {(best - base) * 100:+.1f} points")
    print(f"\n  ranked by what each single correction buys: "
          + " > ".join(f"{n.split('+ ')[-1]} ({d * 100:+.1f})" for n, d in ranked))
    top_name, top_delta = ranked[0]
    print(f"  best single ({top_name.split('+ ')[-1]}) = {(base + top_delta) * 100:.1f}%   "
          f"all three = {best * 100:.1f}%   "
          f"marginal value of the other two once the best one is applied: "
          f"{(best - base - top_delta) * 100:+.1f} points")

    print(f"""
  FINDING: the three corrections are NOT interchangeable, the ranking is measured above rather than
  assumed, and the ranking is not the one the corpus's sentence order would suggest. Randomising the
  order is the cheapest fix -- it is free, it is one line -- but on THIS gold set the largest single
  correction is {top_name.split('+ ')[-1]} ({top_delta * 100:+.1f} points), because the judge's length and
  self-preference terms are continuous and fire on every pair, while the position term only matters
  for pairs whose margin is smaller than the bias. Experiment 5 shows why position is the weakest of
  the three even though the corpus lists it first: a random reordering decorrelates the bias but does
  not remove it. **A team that applies "the three fixes" as a bundle cannot tell which one is carrying
  the result, and these three cost very different amounts** -- order randomisation is free, length
  normalization changes the judge prompt, and a second model family is a second vendor and a second
  bill.

  **The corrections are also not ADDITIVE, and that line is the one to read before buying the third
  one.** The best single correction gets {(base + top_delta) * 100:.1f}% and all three together get
  {best * 100:.1f}% -- so once the dominant bias is fixed, the other two buy
  {(best - base - top_delta) * 100:+.1f} points. That is not a defect in the corrections; it is a statement about
  the judge. Two of the three biases were pulling the same verdicts in the same direction on this gold
  set, so fixing either one recovers most of the error and fixing both recovers little more. A team
  that measures each correction ALONE learns this; a team that ships the bundle pays for a second
  vendor to find out it bought almost nothing.

  The `close-call` column is the one to read. A biased judge is accurate on the easy pairs and near
  chance on the close ones -- and close calls are the ONLY ones a release decision turns on. An
  overall agreement number hides this completely.

  AND READ `label_balance` BEFORE TRUSTING ANY OF IT. If the gold set puts the better answer in slot
  `a` almost always, a position bias aimed at `a` scores as ACCURACY, kappa collapses to 0.000, and
  the correction table inverts -- the fixes look like they make the judge worse. That is not a
  hypothetical: it was a real defect in an earlier version of this gold set, and `is_degenerate()`
  is the check that catches it.""")


# ----------------------------------------------------------------------------------------------
def exp_bias_sweeps() -> None:
    """5. Position bias as a curve, under three handling strategies, plus verbosity as a correlation."""
    _hdr(5, "BIAS SWEEPS -- position as a curve, verbosity as a correlation")
    gold = J.synthetic_gold_set()
    print("  POSITION BIAS: agreement vs the bias magnitude, under three handling strategies\n")
    print("    fixed        = run the pair once, slot `a` always shown first")
    print("    randomised   = run the pair once, presentation order drawn at random  [corpus's fix, T]")
    print("    both/tie=wrong  = run BOTH orders, count a disagreement as an error")
    print("    both/decided    = run BOTH orders, score only the pairs the two runs agreed on")
    print("    ties         = the share of pairs the two orders disagreed on\n")
    print(f"  {'beta':>6}{'fixed':>9}{'randomised':>12}{'both/tie=wrong':>17}{'both/decided':>14}"
          f"{'ties':>8}")
    print("  " + LINE[:66])
    sweep = J.position_bias_sweep(gold)
    for r in sweep:
        print(f"  {r['beta']:>6.2f}{r['agreement_off'] * 100:>8.1f}%{r['agreement_on'] * 100:>11.1f}%"
              f"{r['agreement_both'] * 100:>16.1f}%{r['decided_both'] * 100:>13.1f}%"
              f"{r['tie_rate'] * 100:>7.1f}%")

    v = J.verbosity_effect(gold)
    print(f"\n  VERBOSITY: correlation between the score margin and the LENGTH margin")
    print(f"    biased judge         r = {v['r_length_score_biased']:+.3f}")
    print(f"    length-normalized    r = {v['r_length_score_corrected']:+.3f}")
    print(f"    baseline (length vs QUALITY margin, sample) r = {v['r_length_quality_true']:+.3f}")

    last = sweep[-1]
    worst = max(sweep, key=lambda r: r["recovered_both"])
    print(f"""
  FINDING: the corpus prescribes "randomize the answer order" [T], and the measured value of doing
  that is SMALL. Randomising decorrelates the bias from the label, so the SIGNED error disappears --
  but the bonus is still applied, at random, on every call, and on a close pair a random bonus the
  size of the margin is a coin flip. It trades a systematic error for extra variance. Across the
  whole sweep the gain is {min(r['recovered'] for r in sweep) * 100:+.1f} to {max(r['recovered'] for r in sweep) * 100:+.1f} points, and at
  beta={last['beta']:.2f} -- where the bias is largest -- it is only {last['recovered'] * 100:+.1f}.

  **Running BOTH orders cancels the bonus exactly instead of in expectation**, and it buys something
  randomising cannot: an explicit UNDECIDED. Read the last three columns together, because none of
  them means anything alone:

    * scored with ties as errors, both-orders looks WORSE than fixed order, and increasingly so
      ({last['agreement_both'] * 100:.1f}% vs {last['agreement_off'] * 100:.1f}% at beta={last['beta']:.2f}). That number is
      honest and it is not the reason to adopt the method.
    * scored over the pairs it DID decide, both-orders is better than fixed order at EVERY beta and
      the margin grows with the bias ({sweep[0]['decided_both'] * 100:.1f}% vs {sweep[0]['agreement_off'] * 100:.1f}% at
      beta=0.00, {worst['decided_both'] * 100:.1f}% vs {worst['agreement_off'] * 100:.1f}% at beta={worst['beta']:.2f}).
    * and the tie rate is the number that makes the other two interpretable. It is not noise -- it
      RISES monotonically with the bias ({sweep[0]['tie_rate'] * 100:.1f}% -> {last['tie_rate'] * 100:.1f}%) because the pairs the bias
      flips are exactly the pairs whose verdict changes when the order changes. **The abstention set
      is a calibrated measurement of the judge's unreliability**, and it is the only one of the three
      strategies that reports one at all.

  So the operational answer is not "randomise". It is: run both orders, publish the decided accuracy
  AND the tie rate side by side, and route the ties to a human or a second judge -- because a judge
  that silently calls a coin flip and a judge that says "I cannot call this" produce the same accuracy
  number and completely different decisions.

  **Verbosity bias is not fixed by either** -- it is a correlation between length and score, and it
  survives any ordering. That is why "cap or normalize length" is a separate sentence in the corpus's
  list, and it is a separate correction in the ladder above.

  The baseline line is the honesty check, and it matters: length is generated independently of
  quality in the gold set, so the TRUE association is 0 by construction -- but the SAMPLE's is not
  exactly 0, and {v['r_length_quality_true']:+.3f} is the baseline the biased judge's {v['r_length_score_biased']:+.3f} must be
  compared against. The corrected judge lands at {v['r_length_score_corrected']:+.3f}, which is at the sample's own
  incidental level. A judge whose r merely MATCHED the baseline would not be exhibiting bias at all;
  it would be reading the length-quality association that happens to be in the sample.""")


# ----------------------------------------------------------------------------------------------
def exp_pointwise_vs_pairwise() -> None:
    """6. The scoring mode, and where the difference is concentrated."""
    _hdr(6, "POINTWISE vs PAIRWISE -- and that the difference lives in the close calls")
    print("""
[corpus] "Pointwise is scoring as the judge to rate one answer from one to five. It is simple but
          noisy because absolute scores wander. Pairwise scoring instead shows the judge two answers
          and asks which is better. Pairwise is far more reliable for close calls." [T]
""")
    gold = J.synthetic_gold_set()
    corr = J.Correction(randomize_order=True, length_controlled=True, different_family=True)
    close = gold.close_pairs(0.5)
    print(f"  gold set: {gold.n} pairs, of which {len(close)} are CLOSE calls "
          f"(|quality margin| <= 0.5)\n")
    print(f"  {'mode':<12}{'scope':<18}{'n':>6}{'agreement':>12}{'kappa':>9}{'close-call subset':>20}")
    print("  " + LINE[:76])
    for mode in ("pairwise", "pointwise"):
        for scope, pairs in (("all pairs", gold.pairs), ("close calls only", close)):
            r = J.evaluate(gold, J.DEFAULT_JUDGE, corr, mode=mode, pairs=pairs)
            ca = r["close_agreement"] if scope == "all pairs" else r["agreement"]
            label = "within this set" if scope == "close calls only" else "within all pairs"
            print(f"  {mode:<12}{scope:<18}{r['n']:>6}{r['agreement'] * 100:>11.1f}%"
                  f"{r['kappa']:>9.3f}{ca * 100:>19.1f}%")
    print(f"\n  (the rightmost column is the close-call agreement: for the 'all pairs' rows it is the "
          f"close-call\n   slice of that run; for the 'close calls only' rows the whole run IS the slice)")

    pw_all = J.evaluate(gold, J.DEFAULT_JUDGE, corr, mode="pairwise")
    pt_all = J.evaluate(gold, J.DEFAULT_JUDGE, corr, mode="pointwise")
    pw_cl = J.evaluate(gold, J.DEFAULT_JUDGE, corr, mode="pairwise", pairs=close)
    pt_cl = J.evaluate(gold, J.DEFAULT_JUDGE, corr, mode="pointwise", pairs=close)
    print(f"""
  FINDING: the gap between the two modes is {(pw_all['agreement'] - pt_all['agreement']) * 100:+.1f} points on all
  pairs and {(pw_cl['agreement'] - pt_cl['agreement']) * 100:+.1f} points on the close calls -- the difference is
  CONCENTRATED exactly where the corpus says it is.

  The mechanism is not that pointwise is biased; the bias terms are IDENTICAL in both modes. It is
  that pointwise has to place an answer against a remembered rubric instead of against a visible
  alternative, so the per-call noise is larger and the same true margin is harder to resolve.
  **Both modes suffer the same biases and only one suffers the extra noise**, which is the whole
  recommendation: pointwise to gate, pairwise to choose.""")


# ----------------------------------------------------------------------------------------------
def exp_eval_set_size() -> None:
    """7. The three statistical consequences of a small eval set."""
    _hdr(7, "EVAL SET SIZE -- power, false confidence, and tail representation")
    print("""
[corpus] "With 20 examples, one lucky run looks like real progress. ... And a mean of 4.2 can still
          hide the 5% of answers that leak data or invent facts. Always inspect the worst cases, not
          just the average." [T]
[corpus] "A curated 200 examples often beats a random 10,000." [T]
""")
    sigma = ST.win_rate_sigma()
    print(f"  metric: pairwise win rate; per-example sd = {sigma} (Bernoulli worst case)\n")
    print(f"  {'n':>6}{'std err':>10}{'MDE (80% power)':>18}{'P(false +0.20)':>17}"
          f"{'power vs +0.20':>17}{'P(contains 5%)':>16}")
    print("  " + LINE[:84])
    for r in ST.eval_set_sizing(sigma):
        print(f"  {r['n']:>6}{r['se']:>10.3f}{r['mde_80']:>18.3f}"
              f"{r['p_false_improve'] * 100:>16.1f}%{r['power_0.2'] * 100:>16.1f}%"
              f"{r['p_contains_5pct'] * 100:>15.1f}%")

    n20 = ST.eval_set_sizing(sigma, sizes=[20])[0]
    n200 = ST.eval_set_sizing(sigma, sizes=[200])[0]
    print(f"""
  FINDING: three different problems, three different rates of improvement, and one of them binds.

  1. POWER. At n=20 the smallest detectable win-rate change is {n20['mde_80']:.3f} -- so a REAL
     regression of 0.20 is detected only {n20['power_0.2'] * 100:.0f}% of the time. The break ships green.
  2. FALSE CONFIDENCE. At n=20 a change that did NOTHING reports a >= 0.20 improvement
     {n20['p_false_improve'] * 100:.0f}% of the time. This is the corpus's "one lucky run", and it is the
     arm that merges a bad change.
  3. TAIL REPRESENTATION. At n=20 a set contains an example of a 5% failure class only
     {n20['p_contains_5pct'] * 100:.0f}% of the time. **A mean cannot report a problem in a class the set does
     not contain**, and this is the corpus's "mean of 4.2 hiding the 5%".

  At n=200 all three are adequate ({n200['p_false_improve'] * 100:.1f}%, {n200['power_0.2'] * 100:.0f}%,
  {n200['p_contains_5pct'] * 100:.1f}%) -- which is why the corpus's "curated 200" is not a round number
  pulled from the air. n=100 is where (1) and (2) become acceptable; (3) is usually the binding
  constraint and it is the one nobody computes.

  THE DESIGN CONSEQUENCE: the size that satisfies power is not the size that satisfies tail
  representation, so "how big should the eval set be?" has no single answer -- and the largest of the
  three requirements binds.""")


# ----------------------------------------------------------------------------------------------
def exp_gate_choice() -> None:
    """8. Three gates over the same scores, and where they disagree."""
    _hdr(8, "THE GATE -- mean, p5 or worst-class, over the same release scores")
    print("""
[corpus] "Most importantly, watch the fifth percentile, your worst cases. A great average with an
          ugly tail is exactly the profile that produces embarrassing screenshots." [T]
""")
    gates = [G.Gate("mean", 4.00), G.Gate("percentile", 3.20, percentile=5.0),
             G.Gate("worst_class", 3.60)]
    releases = [
        G.make_release("v1-baseline", seed=5),
        G.make_release("v2-reranker", easy_mu=4.70, hard_mu=3.65, safety_bad_frac=0.03, seed=6),
        G.make_release("v3-aggressive", easy_mu=4.85, hard_mu=3.75, safety_bad_frac=0.10, seed=7),
        G.make_release("v4-safe", easy_mu=4.62, hard_mu=3.62, safety_bad_frac=0.01, seed=8),
    ]
    print(f"  {'release':<16}{'mean':>7}{'p5':>7}{'worst class':>13}{'class':>9}"
          f"{'mean gate':>11}{'p5 gate':>9}{'class gate':>12}")
    print("  " + LINE[:82])
    for r in releases:
        wc, wv = r.worst_class_mean()
        m = gates[0].evaluate(r)["pass"]
        p = gates[1].evaluate(r)["pass"]
        c = gates[2].evaluate(r)["pass"]
        mark = lambda b: "PASS" if b else "FAIL"
        print(f"  {r.name:<16}{r.mean():>7.2f}{r.percentile(5):>7.2f}{wv:>13.2f}{wc:>9}"
              f"{mark(m):>11}{mark(p):>9}{mark(c):>12}")

    d = G.disagreement(releases, gates)
    print(f"\n  releases where the gates disagree: {d['n_split']} of {d['n_releases']}")
    print(f"  releases where MEAN passes and BOTH tail gates fail: {d['mean_passes_tail_fails']}")
    print(f"  pass rate by gate: "
          + "  ".join(f"{k} {v * 100:.0f}%" for k, v in d["rates"].items()))

    v1, v3 = releases[0], releases[2]
    h = G.regression_hidden_by_mean(v1, v3)
    print(f"\n  REGRESSION HIDDEN BY THE MEAN -- {h['baseline']} -> {h['candidate']}:")
    print(f"    mean        {h['mean_baseline']:.2f} -> {h['mean_candidate']:.2f}   "
          f"delta {h['mean_delta']:+.2f}   (IMPROVED)")
    print(f"    p5          {h['p5_baseline']:.2f} -> {h['p5_candidate']:.2f}   "
          f"delta {h['p5_delta']:+.2f}")
    print(f"    safety slice {h['slice_baseline']:.2f} -> {h['slice_candidate']:.2f}   "
          f"delta {h['slice_delta']:+.2f}   (WORSE)")
    print(f"    hidden regression detected: {h['hidden_regression']}")

    print("""
  FINDING: three gates over the SAME scores give different answers, and the disagreement is
  one-directional -- the mean passes releases the tail gates fail. That is the corpus's "great
  average with an ugly tail" as a decision table, and it is the same tail-over-mean pattern this
  knowledge base finds independently in T08's goodput, T09's p99 and T10's SNR.

  The v3 row is the one to study: it has the HIGHEST mean of the four releases and the worst tail.
  A team gating on the mean ships it and calls it the best release of the quarter.

  AND NOTE WHICH GATE IS STRICTEST. `worst_class` fails v3 where p5 does not have to, because a
  percentile is a property of the whole distribution while a slice is a property of the population
  that gets hurt. **A percentile gate assumes every example is exchangeable; a slice gate refuses
  that assumption.** For a safety slice, the refusal is correct.""")


# ----------------------------------------------------------------------------------------------
def exp_feedback_loop() -> None:
    """9. Open loop vs closed loop, over releases."""
    _hdr(9, "THE FEEDBACK LOOP -- traces into storage, vs traces into the eval set")
    print("""
[corpus] "the magic is the feedback loop. Score live traces with a judge, alert when the score
          drops, and sample the failures into a dataset. ... Yesterday's production failure becomes
          today's test case, and your eval set gets stronger every week on its own." [T]
[corpus] "Worst of all is the dashboard nobody owns. If no alert fires and no eval reads the traces,
          you have paid for storage, not for insight." [T]
""")
    res = LP.compare(releases=24)
    o, c = res["open"], res["closed"]
    print(f"  failure taxonomy: {res['n_classes']} classes, heavy-tailed "
          f"(rare classes cost more)\n")
    print(f"  {'release':>8}{'open escaped':>15}{'closed escaped':>17}{'open cover':>13}"
          f"{'closed cover':>15}{'closed eval set':>18}")
    print("  " + LINE[:84])
    for ro, rc in zip(res["open_run"].per_release, res["closed_run"].per_release):
        if ro["release"] % 4 == 1 or ro["release"] == 24:
            print(f"  {ro['release']:>8}{ro['escaped']:>15,.0f}{rc['escaped']:>17,.0f}"
                  f"{ro['coverage'] * 100:>12.1f}%{rc['coverage'] * 100:>14.1f}%"
                  f"{rc['eval_size']:>18}")

    lr = res["last_release_ratio"]
    lr_s = "n/a (zero escapes)" if lr == float("inf") else f"{lr:.1f}x"
    fc = res["full_coverage_release"]
    print(f"""
  TOTALS OVER {res['releases']} RELEASES
    open loop   escaped {o['escaped_total']:>10,.0f}   severity-weighted {o['escaped_weighted']:>12,.0f}   coverage {o['final_coverage'] * 100:.1f}%
    closed loop escaped {c['escaped_total']:>10,.0f}   severity-weighted {c['escaped_weighted']:>12,.0f}   coverage {c['final_coverage'] * 100:.1f}%
    ratio (raw)      {res['escaped_ratio_total']:.1f}x
    ratio (weighted) {res['escaped_ratio_weighted']:.1f}x
    ratio on the LAST release alone: {lr_s}
    first release where the closed loop halved the escapes: {res['crossover_release']}
    severity per escaped occurrence: open {res['open_severity_per_escape']:.2f}   closed {res['closed_severity_per_escape']:.2f}
    first release at FULL coverage: {fc}   (closed-loop escapes are confined to releases 1-{fc})""")

    cost = LP.open_loop_cost(res["open_run"], storage_gb_per_release=48.0,
                             gpu_hours_per_release=1.5)
    print(f"""
  THE OPEN LOOP'S RETURN: {cost['storage_gb_total']:,.0f} GB stored over {cost['releases']} releases,
  {cost['defects_prevented']:.0f} defects prevented, coverage gained {cost['coverage_gained']:.1f}.
  {cost['note']}.

  FINDING: the two arms are identical in infrastructure and differ in ONE edge -- whether an observed
  failure becomes a test case. Three properties of the result matter:

  1. **The ratio grows without bound.** The open loop's escapes are FLAT ({o['first_release_escaped']:,.0f}
     -> {o['last_release_escaped']:,.0f}); the closed loop's FALL ({c['first_release_escaped']:,.0f} ->
     {c['last_release_escaped']:,.0f}), reaching zero at release {fc} and staying there. So the value of the
     loop is not a constant factor, it is compounding -- which is what "gets stronger every week on
     its own" means mechanically. The last-release ratio is not a number at all: it is a divide by
     zero, and that IS the result.

  2. **The WEIGHTED ratio is LOWER than the raw one** ({res['escaped_ratio_weighted']:.1f}x vs
     {res['escaped_ratio_total']:.1f}x), and that is a finding rather than a detail. The closed loop's residual
     escapes carry a HIGHER mean severity than the open loop's ({res['closed_severity_per_escape']:.2f} vs
     {res['open_severity_per_escape']:.2f}), so the classes that resist closure LONGEST are the rare, expensive ones.
     **The loop closes the head first and the tail last**, because a rare class produces few traced
     occurrences and the noticing probability falls with its rate. It is the same tail-over-mean
     pattern this knowledge base finds in T08's goodput, T09's p99 and T10's SNR -- reproduced
     inside the loop that exists to fix it, which is why the ramp cannot be read as the steady state.
     The fix is not a bigger sample rate for its own sake: it is targeting the tail, by seeding the
     eval set with the rare classes from incident review rather than waiting for 5% sampling to
     stumble on them.

  3. **The open loop's cost is not the problem -- its return is.** 48 GB/release is affordable. The
     defects prevented are zero, so the return per GB is zero and any price is too high.

  **The design consequence:** instrumentation that does not feed a gate is a cost centre with no
  signal, and the fix is one edge in a diagram, not more storage.""")


def run_all() -> None:
    exp_latency_decomposition()
    exp_span_sum_invariants()
    exp_sampling_frontier()
    exp_head_sampling_blind_spot()
    exp_judge_corrections()
    exp_bias_sweeps()
    exp_pointwise_vs_pairwise()
    exp_eval_set_size()
    exp_gate_choice()
    exp_feedback_loop()
