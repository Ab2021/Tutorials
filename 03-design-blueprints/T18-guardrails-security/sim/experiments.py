"""T18 -- the six demonstrations. Each prints a table and then a FINDING.

A demonstration that only prints numbers is a dashboard. Each function here also states what the
numbers mean for a design decision, so the HLD can quote a run rather than an opinion.
"""
from __future__ import annotations

import math

from . import gating as G
from . import policy as P
from . import rails as R
from . import decay as D

LINE = "-" * 78


def _hdr(n, title: str) -> None:
    print("\n" + "=" * 78)
    print(f"{n}. {title}")
    print("=" * 78)


def _pct(x: float) -> str:
    return f"{x * 100.0:.1f}%"


def _pct3(x: float) -> str:
    """For the scalar arithmetic, where 99.986% printed as 100.0% would read as a rounding artifact
    instead of as the 17.7-point overstatement it is."""
    return f"{x * 100.0:.3f}%"


# ----------------------------------------------------------------------------------------------
def exp_rail_stack() -> None:
    """1. Four rails are not 1-(1-r)^4."""
    _hdr(1, "THE RAIL STACK -- the layering claim, and the arithmetic that misstates it")
    print("""
[corpus] "Stacked up, defense in depth looks like this. Four rails each catching a different
          failure... No single rail is enough. It is the layering, the redundancy, that makes the
          system genuinely hard to break."                                  -- Guardrails video [T]
[corpus] "Teams reliably add the obvious input and output rails, and reliably skip the two that
          matter most: filtering the retrieved context and gating the tool calls."        [T]
""")

    m = R.mix_summary()
    print("  THE ATTACK MIX (technique x path, with a severity nothing in the corpus asks you to")
    print("  attach -- which is the point of experiments 1 and 5):")
    print(f"\n    {'class':<24}{'technique':<15}{'path':<14}{'share':>7}{'sev':>5}")
    print("    " + LINE[:65])
    for a in R.ATTACK_MIX:
        print(f"    {a.name:<24}{a.technique:<15}{a.path:<14}{_pct(a.weight):>7}{a.severity:>5}")
    print(f"\n    by path:      " + "  ".join(
        f"{k} {_pct(v)}" for k, v in sorted(m["by_path"].items(), key=lambda kv: -kv[1])))
    print(f"    by technique: " + "  ".join(
        f"{k} {_pct(v)}" for k, v in sorted(m["by_technique"].items(), key=lambda kv: -kv[1])))
    print(f"    mean severity {m['mean_severity']:.2f}"
          f"   the DIRECT path is {_pct(m['direct_share'])} of attempts")

    print("\n  THE ARITHMETIC A DESIGN REVIEW PRODUCES:")
    for label, stack in (("the obvious two rails", R.OBVIOUS_TWO),
                         ("the full four rails", R.FULL_STACK)):
        print(f"    {label:<24} 1-prod(1-r_i) = {_pct3(R.scalar_catch(stack))}")

    print("\n  THE SAME STACKS, MEASURED AGAINST THE MIX:")
    print(f"\n    {'configuration':<34}{'scalar':>10}{'measured':>10}{'gap':>8}")
    print("    " + LINE[:62])
    configs = [
        ("input pattern only", (R.INPUT_PATTERN,)),
        ("the obvious two (input rails)", R.OBVIOUS_TWO),
        ("the obvious two + output", R.OBVIOUS_TWO + (R.OUTPUT_VALIDATOR,)),
        ("+ retrieval rail (the full stack)", R.FULL_STACK),
        ("the full stack minus retrieval", (R.INPUT_PATTERN, R.INPUT_CLASSIFIER, R.OUTPUT_VALIDATOR)),
    ]
    for label, stack in configs:
        sc, mc = R.scalar_catch(stack), R.measured_catch(stack)
        print(f"    {label:<34}{_pct3(sc):>10}{_pct(mc):>10}{_pct(sc - mc):>8}")

    print(f"\n    the full stack's own CEILING (every r set to 1.0): {_pct(R.ceiling_catch(R.FULL_STACK))}")
    print(f"    headroom left in the detection approach:            "
          f"{_pct(R.ceiling_catch(R.FULL_STACK) - R.measured_catch(R.FULL_STACK))}")
    solo = max((r for r in R.FULL_STACK), key=lambda r: R.measured_catch((r,)))
    print(f"    the single best rail ALONE: {solo.name} at {_pct(R.measured_catch((solo,)))}"
          f" -- one rail with full coverage beats four with partial coverage")

    print("\n  WHERE THE MISSES ARE -- every class, worst first:")
    print(f"\n    {'class':<24}{'path':<14}{'sev':>5}{'share':>8}{'catch':>9}{'escaped':>9}")
    print("    " + LINE[:69])
    for r in R.catch_by_class(R.FULL_STACK):
        print(f"    {r['name']:<24}{r['path']:<14}{r['severity']:>5}{_pct(r['weight']):>8}"
              f"{_pct(r['catch']):>9}{_pct(r['escaped_share']):>9}")

    print("\n  BY TECHNIQUE AND BY PATH -- competence is not coverage:")
    print(f"\n    {'technique':<16}{'share':>8}{'catch':>9}      {'path':<16}{'share':>8}{'catch':>9}")
    print("    " + LINE[:72])
    bt, bp = R.catch_by_technique(R.FULL_STACK), R.catch_by_path(R.FULL_STACK)
    for i in range(max(len(bt), len(bp))):
        left = f"{bt[i]['technique']:<16}{_pct(bt[i]['share']):>8}{_pct(bt[i]['catch']):>9}" if i < len(bt) else " " * 33
        right = f"{bp[i]['path']:<16}{_pct(bp[i]['share']):>8}{_pct(bp[i]['catch']):>9}" if i < len(bp) else ""
        print(f"    {left}      {right}")

    print("\n  WHAT EACH RAIL IS WORTH, GIVEN THE OTHERS:")
    for r in R.leave_one_out(R.FULL_STACK):
        print(f"    remove {r['rail']:<20} catch falls to {_pct(r['without']):<8}"
              f" marginal value {_pct(r['marginal'])}")

    print("\n  THE GREEDY BUILD ORDER (the corpus's answer [T] is 'the tool-call rail first' --")
    print("  this is the order if you only ever build DETECTION rails, which is the mistake):")
    for s in R.build_order():
        print(f"    {s['step']}. {s['rail']:<20} +{_pct(s['gain']):<8} -> {_pct(s['catch'])}")

    sk = R.severity_skew(R.FULL_STACK)
    print(f"\n  SEVERITY WEIGHTING, the check nobody runs:")
    print(f"    catch, weighted by FREQUENCY     {_pct(sk['mean_catch'])}")
    print(f"    catch, weighted by CONSEQUENCE   {_pct(sk['severity_catch'])}")
    print(f"    skew                             {_pct(sk['skew'])}"
          f"   ({abs(sk['skew']) * 100:.1f} points)")
    print(f"    escape rate: frequency {_pct(sk['mean_catch_rate'])}"
          f"  vs consequence {_pct(sk['severity_escape_rate'])}")

    ev = R.evasion_sensitivity(R.FULL_STACK)
    print(f"\n  IF EVERY ATTACKER SWITCHED TO A NOVEL TECHNIQUE TOMORROW:")
    print(f"    today's catch  {_pct(ev['current'])}   ->   all-novel catch  {_pct(ev['all_novel'])}")

    lat = R.rail_latency(R.FULL_STACK, k_chunks=5, cached_frac=0.0)
    lat_cached = R.rail_latency(R.FULL_STACK, k_chunks=5, cached_frac=0.8)
    print(f"\n  LATENCY, because the cheap rail is the one nobody skips and the expensive one is:")
    print(f"    {'rail':<20}{'calls':>7}{'p50 ms':>9}{'p99 ms':>9}")
    print("    " + LINE[:45])
    for r in lat["rows"]:
        print(f"    {r['rail']:<20}{r['calls']:>7.1f}{r['ms_p50']:>9.1f}{r['ms_p99']:>9.1f}")
    print(f"    {'TOTAL':<20}{'':>7}{lat['p50']:>9.1f}{lat['p99']:>9.1f}"
          f"    (retrieval verdict cached at 80%: p50 {lat_cached['p50']:.1f} ms)")

    print(f"""
  FINDING: the layering claim is true and the arithmetic people attach to it is wrong by a wide
  margin. The scalar formula gives {_pct3(R.scalar_catch(R.FULL_STACK))} for the full stack; the measured
  value against the mix is {_pct(R.measured_catch(R.FULL_STACK))}, a gap of
  {_pct(R.scalar_catch(R.FULL_STACK) - R.measured_catch(R.FULL_STACK))}. Two mechanisms produce the gap and both are design-relevant:

    * COVERAGE. The two input rails see only the {_pct(m['direct_share'])} of attempts that arrive on
      the direct path. Their scalar arithmetic says {_pct3(R.scalar_catch(R.OBVIOUS_TWO))}; measured against
      the mix they catch {_pct(R.measured_catch(R.OBVIOUS_TWO))}. The corpus's "teams skip the two that
      matter most" [T] is not a discipline problem, it is a {_pct3(R.scalar_catch(R.OBVIOUS_TWO))} vs {_pct(R.measured_catch(R.OBVIOUS_TWO))} measurement gap that
      nobody closes because the honest number was never computed. The sharpest form of the coverage
      point is above: {solo.name} ALONE catches {_pct(R.measured_catch((solo,)))} -- more than the two
      input rails together, and more than any other single rail -- purely because it is the only one
      that covers every path. Coverage beats competence, and a stack of four well-tuned rails that
      share a coverage hole is one rail's worth of protection.
    * TECHNIQUE. Every rail has a residual on techniques it was not built for, so a class the stack
      cannot see contributes nothing from ANY rail. Those classes are the ones with the highest
      severity, which is why consequence-weighted catch ({_pct(sk['severity_catch'])}) is
      {abs(sk['skew']) * 100:.1f} points BELOW frequency-weighted catch ({_pct(sk['mean_catch'])}) -- the fifth independent topic in this
      knowledge base where a frequency-weighted average reads better than the consequence-weighted
      one (after T08's goodput, T09's p99, T10's SNR and T17's gate).

  And the number that decides the whole design: the ceiling is {_pct(R.ceiling_catch(R.FULL_STACK))}. The full
  stack is already within {_pct(R.ceiling_catch(R.FULL_STACK) - R.measured_catch(R.FULL_STACK))} of the best it can EVER do, because the ceiling is set by the
  novel-technique residual and not by any rail's r. "Improve the classifier" is therefore not a
  lever here; it is a rounding error. That is the argument for experiment 2.
""")


# ----------------------------------------------------------------------------------------------
def exp_which_lever() -> None:
    """2. Which factor: better detection, or a capability gate?"""
    _hdr(2, "WHICH LEVER -- detection is near its ceiling; the gate is not")
    print("""
[corpus] "above all, gate the tool calls that can send money, email a customer, or delete a record.
          That last layer has the biggest blast radius."                        -- video [T]
[corpus] "Capability gating is the most underused defense; many teams add a guardrail classifier
          and stop there."                                    -- security guide, via the corpus [R]
""")

    catch = R.measured_catch(R.FULL_STACK)
    ceiling = R.ceiling_catch(R.FULL_STACK)

    print(f"  harm = (1 - catch) x gate_failure, where gate_failure is measured, not assumed.")
    print(f"  today's catch {_pct(catch)}, the detectors' ceiling {_pct(ceiling)},"
          f" headroom {_pct(ceiling - catch)}")

    print(f"\n  THE GATE LADDER (what each level of the corpus's recipe [T] is worth):")
    print(f"\n    {'gate level':<34}{'gate_failure':>14}{'harm':>12}{'vs nothing':>12}")
    print("    " + LINE[:72])
    for level, label in (("none", "no gate"), ("schema", "allowlist + strict schema"),
                         ("gated", "schema + capability gating")):
        gf = G.gate_failure(level, tag_lost=0.03, fail_open=True)
        h = G.harm_rate(catch, level, tag_lost=0.03, fail_open=True)
        print(f"    {label:<34}{gf:>14.4f}{h:>12.5f}{_pct(1 - h / (1 - catch)):>12}")

    print(f"\n  NOW PUT THE TWO LEVERS ON ONE SCALE (harm per attack attempt):")
    print(f"\n    {'option':<44}{'harm':>12}{'reduction':>12}")
    print("    " + LINE[:68])
    for row in G.lever_table(catch, ceiling, tag_lost=0.03):
        print(f"    {row['option']:<44}{row['harm_per_attack']:>12.5f}"
              f"{_pct(row['reduction_vs_nothing']):>12}")

    tbl = {r["option"]: r for r in G.lever_table(catch, ceiling, tag_lost=0.03)}
    gate_and_ceiling = tbl["detectors at CEILING + gating (closed)"]["harm_per_attack"]

    gate_closed = G.gate_failure("gated", 0.90, 0.0, False)
    print(f"\n  THE DECOMPOSITION, stated as a product:")
    print(f"    detection factor (1-catch):   {1 - catch:.4f}   -> at ceiling {1 - ceiling:.4f}"
          f"   (a factor of {(1 - catch) / (1 - ceiling):.2f})")
    print(f"    gate factor:                  1.0000   -> gated (closed) {gate_closed:.4f}"
          f"   (a factor of {1.0 / gate_closed:.1f})")

    print(f"""
  FINDING: the two levers MULTIPLY -- harm = (1-catch) x gate_failure -- so neither substitutes for
  the other, and asking "which one" is the wrong question. The right question is which factor still
  has room, and the measurement answers it decisively.

  Detection has {_pct(ceiling - catch)} of headroom, total, forever, because the ceiling is set by the residual
  the rails carry on techniques nobody has a rule for. A gate has {1.0 / gate_closed:.0f}x, and it is a
  factor that does not care what technique the attacker used, because it never tries to recognise
  the attack -- it removes the capability the attack needs. That is what "structurally immune to
  persuasion" (case study 4.2 [R]) means as a number.

  The consequence for the build order: the corpus says tool-call gating is "mandatory the moment the
  system can take an action" [T], and this run says something stronger. Compare the two ladders:

    detection 0%  + gating    harm {tbl['detectors at ZERO + gating (closed)']['harm_per_attack']:.5f}
    detection 82% + no gate   harm {tbl['do nothing (pre-incident)']['harm_per_attack']:.5f}
    detection 83% + no gate   harm {tbl['detectors at their CEILING, no gate']['harm_per_attack']:.5f}
    detection 83% + gating    harm {gate_and_ceiling:.5f}

  Turning the detectors completely OFF and adding a gate ({tbl['detectors at ZERO + gating (closed)']['harm_per_attack']:.5f}) beats running the
  detectors at their theoretical MAXIMUM with no gate ({tbl['detectors at their CEILING, no gate']['harm_per_attack']:.5f}) by
  {tbl['detectors at their CEILING, no gate']['harm_per_attack'] / tbl['detectors at ZERO + gating (closed)']['harm_per_attack']:.1f}x. Build the gate first. The classifiers then improve it by a further
  {tbl['detectors at ZERO + gating (closed)']['harm_per_attack'] / gate_and_ceiling:.1f}x -- they are worth having, and they are not what makes the system safe.
""")


# ----------------------------------------------------------------------------------------------
def exp_false_refusal() -> None:
    """3. The false-refusal trade, and why one threshold is the wrong architecture."""
    _hdr(3, "THE FALSE-REFUSAL OPTIMUM -- a number with a cost on both sides")
    print("""
[corpus] "Tune the rails too tight and you frustrate real users with false refusals, which is its
          own kind of failure."                                                      [T]
[corpus] "a system that blocks everything is safe and useless."                        [T]
""")

    print(f"  Detector separation: cheap filter AUC {P.CHEAP_FILTER.auc:.3f}"
          f", precise check AUC {P.PRECISE_CHECK.auc:.3f}")

    print(f"\n  THE COST-OPTIMAL THRESHOLD, PER SCENARIO:")
    print(f"\n    {'scenario':<16}{'L/F ratio':>11}{'theta*':>9}{'catch':>9}{'fpr':>9}{'cost':>12}")
    print("    " + LINE[:66])
    opts = {}
    for sc in P.SCENARIOS:
        o = P.optimal_threshold(P.CHEAP_FILTER, sc)
        opts[sc.name] = o
        b = o["optimum"]
        print(f"    {sc.name:<16}{o['ratio']:>11.1f}{b['theta']:>9.2f}{_pct(b['catch']):>9}"
              f"{_pct(b['fpr']):>9}{b['total']:>12.4f}")

    print(f"\n  THE TWO ENDPOINTS, which is the corpus's 'safe and useless' sentence priced:")
    print(f"\n    {'scenario':<16}{'allow-all cost':>15}{'block-all cost':>15}{'block worse?':>14}")
    print("    " + LINE[:60])
    for name, o in opts.items():
        print(f"    {name:<16}{o['allow_all']['total']:>15.2f}{o['block_all']['total']:>15.2f}"
              f"{str(o['block_all_is_worse_than_allow_all']):>14}")

    h = opts["health_payer"]["optimum"]
    print(f"""
  The health payer's optimum is at theta*={h['theta']:.2f}: catch {_pct(h['catch'])} and a false-refusal rate of
  {_pct(h['fpr'])}. That is not a recommendation, it is a proof by construction: at a leak-to-refusal
  cost ratio of {opts['health_payer']['ratio']:.0f}:1 the arithmetic genuinely says "block almost everything", and no
  product team would ship it. So the single-threshold model is not mis-tuned -- it is the WRONG
  ARCHITECTURE, and the fix is to stop asking one detector to be both high-recall and precise.
""")

    print("  THE CASCADE -- cheap high-recall filter, then an expensive precise check on what it flags:")
    print(f"\n    {'architecture':<26}{'theta':>8}{'catch':>10}{'fpr':>9}{'det cost':>10}{'total':>10}")
    print("    " + LINE[:73])
    for sc in P.SCENARIOS:
        cv = P.cascade_vs_single(sc=sc, target_fpr=0.02)
        s = cv["single"]
        p = cv["precise_only"]
        c = cv["cascade"]
        print(f"    {sc.name + ' / single':<26}{cv['cheap_theta']:>8.2f}{_pct3(s['catch']):>10}"
              f"{_pct(s['fpr']):>9}{s['detector_cost']:>10.1f}{s['total']:>10.2f}")
        print(f"    {sc.name + ' / precise only':<26}{P.find_operating_point(P.PRECISE_CHECK, sc, 0.02):>8.2f}"
              f"{_pct3(p['catch']):>10}{_pct(p['fpr']):>9}{p['detector_cost']:>10.1f}{p['total']:>10.2f}")
        print(f"    {sc.name + ' / cascade':<26}{c['theta_precise']:>8.2f}{_pct3(c['catch']):>10}"
              f"{_pct(c['fpr']):>9}{c['detector_cost']:>10.1f}{c['total']:>10.2f}")

    cv = P.cascade_vs_single(sc=P.SCENARIOS[0], target_fpr=0.02)
    print(f"\n  at the SAME 2% false-refusal budget on the health payer:")
    for k, v in cv["catch_at_fixed_fpr"].items():
        print(f"    {k:<16} catch {_pct3(v)}")

    theta_5pct = P.find_operating_point(P.CHEAP_FILTER, P.SCENARIOS[0], 0.05)
    print(f"\n  THE SEGMENT SKEW -- one global threshold is not uniformly harsh:")
    print(f"    global operating point theta={theta_5pct:.3f} (a 5% false-refusal rate on average)")
    print(f"\n    {'segment':<30}{'share':>8}{'fpr':>9}{'x overall':>11}")
    print("    " + LINE[:58])
    for r in P.segment_fpr(P.CHEAP_FILTER, theta_5pct):
        mark = "  <-- worst" if r["worst"] else ""
        print(f"    {r['segment']:<30}{_pct(r['share']):>8}{_pct(r['fpr']):>9}"
              f"{r['multiple_of_overall']:>10.1f}x{mark}")

    worst = max(P.segment_fpr(P.CHEAP_FILTER, theta_5pct), key=lambda r: r["fpr"])
    print(f"""
  FINDING: "balance safety against false refusals" [T] is three separate measurements, not one.

  1. The optimum is a ratio you must name. The health payer's cost-optimal operating point is
     {_pct(h['fpr'])} false refusals -- operationally absurd -- because its leak-to-refusal ratio is
     {opts['health_payer']['ratio']:.0f}:1. The marketing bot's ratio is {opts['marketing_bot']['ratio']:.1f}:1 and its optimum moves by
     {abs(h['theta'] - opts['marketing_bot']['optimum']['theta']):.1f} standard deviations. "Tighten the rails" is not a policy; it is a choice between
     two costs, and the costs are different businesses.
  2. The endpoints are not symmetric, and they do not even agree on which one is safe. For the
     marketing bot, blocking everything costs {opts['marketing_bot']['block_all']['total']:.2f} against {opts['marketing_bot']['allow_all']['total']:.2f} for allowing
     everything -- the corpus's "safe and useless" [T] is the WRONG side of the arithmetic for that
     business by {opts['marketing_bot']['block_all']['total'] / opts['marketing_bot']['allow_all']['total']:.1f}x. For the health payer the same comparison inverts: blocking everything
     costs {opts['health_payer']['block_all']['total']:.2f} against {opts['health_payer']['allow_all']['total']:.2f}, a {opts['health_payer']['allow_all']['total'] / opts['health_payer']['block_all']['total']:.0f}x saving. One
     sentence, two opposite correct answers, decided entirely by L/F.
  3. A global threshold is an average over a distribution that is not uniform. The same 5% global
     false-refusal rate lands as {_pct(worst['fpr'])} on the {worst['segment']} segment --
     {worst['multiple_of_overall']:.1f}x. The cost model treats F as one number; the users who pay it are not one
     population.

  And the architectural finding, which is the one worth carrying into a design review: the
  cost-optimal threshold for the regulated case is unusable, so the answer is not a better threshold
  but a cascade. At an identical 2% false-refusal budget the cascade lifts catch from
  {_pct3(cv['catch_at_fixed_fpr']['single'])} to {_pct3(cv['catch_at_fixed_fpr']['cascade'])} while running the expensive detector on only
  {_pct(cv['cascade']['precise_run_share'])} of traffic -- the same "multiply independent errors, pay per use" move that
  makes the rail stack work, applied to cost instead of to coverage. Note the precise detector alone
  at the same budget only reaches {_pct3(cv['catch_at_fixed_fpr']['precise_only'])}: paying 20x for a better detector is not a substitute
  for putting it in the right place.
""")


# ----------------------------------------------------------------------------------------------
def exp_trust_tags() -> None:
    """4. Fail-open versus fail-closed: a plumbing decision wearing a security costume."""
    _hdr(4, "TRUST TAGS -- the gate's input is a label that has to survive the pipeline")
    print("""
[corpus] "The trust level is data, not metadata: it travels in the same channel as the content, so
          the model itself can reason about it."                     -- security guide [R]
[corpus] "the attack is hidden inside a document your RAG system retrieves. So the user never typed
          it."                                                                -- video [T]
""")

    print(f"  tag survival = (1 - loss_per_hop) ^ hops")
    print(f"\n    {'hops':>5}{'1% loss':>10}{'3% loss':>10}{'5% loss':>10}{'10% loss':>11}")
    print("    " + LINE[:46])
    for hops in (1, 2, 3, 5, 8):
        cells = "".join(f"{R.tag_survival(hops, l):>10.3f}" for l in (0.01, 0.03, 0.05))
        print(f"    {hops:>5}{cells}{R.tag_survival(hops, 0.10):>11.3f}")

    print(f"\n  WHAT A LOST TAG MEANS, by pipeline posture (3 hops, 3% loss):")
    for fail_open in (True, False):
        eff = R.effective_privilege(3, 0.03, fail_open)
        label = "FAIL OPEN  (missing tag -> trusted)" if fail_open else "FAIL CLOSED (missing tag -> untrusted)"
        print(f"\n    {label}")
        print(f"      tag survival                          {_pct(eff['survival'])}")
        print(f"      untrusted content arriving PRIVILEGED  {_pct(eff['privileged_share'])}")
    print(f"      content RESTRICTED by the fail-closed rule  "
          f"{_pct(R.tag_loss_false_refusal(3, 0.03, 0.9))} of legitimate traffic (fail-closed only)")
    print(f"      -- the same {_pct(1 - R.tag_survival(3, 0.03))} of content, converted into either an unchecked privilege or a")
    print(f"         measured refusal, depending on one word in the policy")

    catch = R.measured_catch(R.FULL_STACK)
    print(f"\n  THE EFFECT ON HARM, which is the only place it shows up:")
    print(f"\n    {'posture':<28}{'gate_failure':>14}{'harm':>12}{'vs closed':>12}")
    print("    " + LINE[:66])
    closed = G.gate_under_tag_loss(3, 0.03, False, catch)
    opened = G.gate_under_tag_loss(3, 0.03, True, catch)
    for label, r in (("fail-closed", closed), ("fail-open", opened)):
        print(f"    {label:<28}{r['gate_failure']:>14.4f}{r['harm']:>12.5f}"
              f"{_pct(r['harm'] / closed['harm'] - 1.0 if closed['harm'] else 0.0):>12}")

    print(f"\n  AND IT IS INVISIBLE TO EVERY DETECTION METRIC:")
    print(f"    catch rate is IDENTICAL under both postures: {_pct(catch)}")
    print(f"    nothing was detected, nothing was missed -- a metadata field was absent,")
    print(f"    and the gate widened by {opened['gate_failure'] / closed['gate_failure']:.2f}x")

    print(f"""
  FINDING: the corpus requires the trust level to travel with the content as data [R], and the
  measurement shows that requirement is load-bearing in a way the requirement itself does not say.
  At 3 hops and a 3% per-hop loss the tag survives {_pct(closed['tag_survival'])} of the time. Under fail-open
  the {_pct(1 - closed['tag_survival'])} that lost it arrive with INSTRUCTION-LEVEL PRIVILEGE, and because the gate's
  external-tool term is driven by exactly that label, harmful actions rise
  {(opened['harm'] / closed['harm'] - 1.0) * 100:.0f}% -- while the injection catch rate does not move a single point.

  Three design consequences:

    * FAIL CLOSED is the only defensible default, and it is not free: it converts the same tag loss
      into a {_pct(R.tag_loss_false_refusal(3, 0.03, 0.9))} false-refusal rate on legitimate content. The two options are
      "silently privileged" and "loudly refused", and the second is the one you can measure.
    * ASSERT THE TAG AT THE MODEL BOUNDARY, not at ingestion. Every hop between them is a chance for
      the label to disappear, and nothing downstream of the loss can tell that it ever existed.
    * The tag is not metadata [R] because a sidecar cannot be reasoned about by the model -- and it
      is not metadata for a second reason this run makes visible: a sidecar lookup keyed on content
      is one more hop, and every hop is {_pct(0.03)} of your untrusted content.
""")


# ----------------------------------------------------------------------------------------------
def exp_decay() -> None:
    """5. The rail that worked in March."""
    _hdr(5, "RAIL DECAY -- 'a rail you tested in March may be bypassed by June'")
    print("""
[corpus] "The subtlest trap is treating safety as done. A rail you tested in March may be bypassed
          by June. So, red teaming has to be a recurring schedule, not a launch checkbox."   [T]
[corpus] "Run a red team suite and track the injection catch rate over time."              [T]
""")

    d = D.run_decay()
    print(f"  in-library catch {_pct(d['r_library'])}   residual (novel-technique) catch "
          f"{_pct(d['r_residual'])}")
    print(f"  24 monthly releases; attackers rotate {_pct(d['params'].rotation)} of the "
          f"in-library population per month; a red-team run finds {_pct(d['params'].discovery)} of the novel stock")

    print(f"\n  THE CADENCE TABLE:")
    print(f"\n    {'cadence':<26}{'mean catch':>12}{'trough':>9}{'sev mean':>10}{'sev trough':>12}{'exposure':>10}")
    print("    " + LINE[:84])
    for r in D.cadence_table():
        print(f"    {r['cadence']:<26}{_pct(r['mean_catch']):>12}{_pct(r['trough']):>9}"
              f"{_pct(r['severity_mean']):>10}{_pct(r['severity_trough']):>12}"
              f"{r['exposure_pct']:>9.1f}%")

    print(f"\n  THE SAWTOOTH, one row per quarter (novel share -> mean catch), for the worst cadence:")
    never = d["series"]["never (launch checkbox)"]["rows"]
    for i in range(0, len(never), 3):
        r = never[i]
        bar = "#" * int(r["mean_catch"] * 50)
        print(f"    month {r['period']:>2}  novel {_pct(r['novel_share']):>6}  "
              f"catch {_pct(r['mean_catch']):>6}  |{bar}")

    print(f"\n  THE SAME CURVE, SEVERITY-WEIGHTED, next to the published metric:")
    print(f"\n    {'month':>6}{'novel share':>13}{'mean catch':>12}{'severity catch':>16}{'gap':>8}")
    print("    " + LINE[:57])
    for i in range(0, len(never), 4):
        r = never[i]
        print(f"    {r['period']:>6}{_pct(r['novel_share']):>13}{_pct(r['mean_catch']):>12}"
              f"{_pct(r['severity_catch']):>16}{_pct(r['mean_catch'] - r['severity_catch']):>8}")

    mb = D.metric_blindness()
    print(f"\n  WHY THE PRESCRIBED METRIC CANNOT SEE THIS:")
    print(f"    over {mb['periods']} months with NO red teaming:")
    print(f"      mean catch          {_pct(mb['mean_first'])} -> {_pct(mb['mean_last'])}"
          f"    (moves {_pct(mb['mean_drop'])})")
    print(f"      severity catch      {_pct(mb['severity_first'])} -> {_pct(mb['severity_last'])}"
          f"    (moves {_pct(mb['severity_drop'])})")
    print(f"      novel-class catch   {_pct(mb['novel_first'])} -> {_pct(mb['novel_last'])}"
          f"    (moves {_pct(mb['novel_drop'])})  <-- NEVER MOVES")

    for target in (0.90, 0.75, 0.70, 0.60):
        c = D.cadence_for_trough(target)
        print(f"\n    to keep the trough above {_pct(target)}: {c['cadence']}"
              f"  (trough {_pct(c['trough'])}, every {c['every']} month(s))")

    print(f"""
  FINDING: "red team on a schedule" [T] is right and, taken literally, still under-specified -- the
  schedule is the whole decision. The same rotation rate that is harmless at monthly cadence
  produces a trough of {_pct(d['series']['annual']['trough'])} at annual cadence and
  {_pct(d['series']['never (launch checkbox)']['trough'])} at none, against a continuous baseline whose worst moment is
  {_pct(d['series']['continuous (every release)']['trough'])}. Exposure -- the catch you did not have, accumulated over two years -- is
  {d['series']['never (launch checkbox)']['exposure_pct_of_periods']:.0f}% of periods with no programme against
  {d['series']['quarterly']['exposure_pct_of_periods']:.0f}% quarterly.

  A HONEST NOTE ON WHERE THIS DISAGREES WITH THE REST OF THE KNOWLEDGE BASE. This experiment does NOT
  reproduce the average-hides-the-tail signature -- and saying so matters, because the temptation is
  to report a fifth instance and the run refuses. Here the decay is fast enough that the published
  mean is not hiding anything: it falls {_pct(mb['mean_drop'])} over two years, from {_pct(mb['mean_first'])} to
  {_pct(mb['mean_last'])}, and it falls because the attackers are winning, not because a population drifted out from
  under an average. The severity-weighted gap NARROWS slightly, from {_pct(abs(never[0]['mean_catch'] - never[0]['severity_catch']))} to
  {_pct(abs(never[-1]['mean_catch'] - never[-1]['severity_catch']))}, rather than widening -- so the signature found statically in experiment 1 does not
  compound over time. The drift dominates it.

  What this experiment DOES show, and it is a different and sharper failure:

    * ONE READING IS PERMANENTLY DEAD. The novel-class catch moves {_pct(mb['novel_drop'])} across two years of
      accelerating decay, because it starts at the residual ({_pct(mb['novel_first'])}) and STAYS there. Whatever
      instrument you point at novel techniques, it was already at its floor on day one, so it cannot
      cross a threshold, so it cannot fire. A monitor on the population that is entirely severity-5
      is therefore not merely insensitive -- it is silent by construction. What can fire is a
      count of ATTEMPTS, an unexplained tool-call pattern, or a new-source signal: something that
      measures the adversary rather than the classifier.
    * A LAGGING MEAN IS NOT THE SAME AS A LATE ONE. The mean's {_pct(mb['mean_drop'])} fall is visible in the chart
      long before anybody acts on it, because nothing in the number says whether it is decaying
      fast or slow. A rate of change would say so; a level will not. Reporting a red-team result
      without its DATE is the same class of error.
    * THE TROUGH IS WHEN YOU GET BREACHED, NOT THE MEAN. Between runs there is a window in which the
      deployed defences are measurably weaker than the last report said, and the report is what
      everybody is steering by. The interval between runs is a control setting, and it is the one
      nobody writes down next to a date.

  The design consequence: a red-team cadence is a control with a failure mode, and the failure mode
  is the interval. It should be sized like every other control here -- against a target, with the
  TROUGH measured rather than the average -- and it has to be paired with a detection aimed at the
  adversary, because the classifier metric is provably incapable of seeing this.
""")


# ----------------------------------------------------------------------------------------------
def exp_approval_queue() -> None:
    """6. Human approval is a resource, and the gate's floor is made of it."""
    _hdr(6, "THE APPROVAL QUEUE -- the gate's floor is a service with a queue")
    print("""
[corpus] "Irreversible actions like a refund or a delete require human approval."            [T]
[corpus] (edge case) "Human approval as a rubber stamp: if approvers approve 99.9% of requests,
          the control has become a latency tax."
""")

    q = G.ApprovalQueue()
    print(f"  approver pool: {q.approvers} reviewers x {q.decisions_per_hour:.0f} decisions/hour"
          f" = {q.capacity:.0f}/hour over a {q.horizon_hours:.0f}h day")

    print(f"\n  AS DEMAND APPROACHES CAPACITY, THE CONTROL DEGRADES IN THREE WAYS AT ONCE:")
    print(f"\n    {'utilisation':>12}{'depth':>9}{'latency h':>11}{'approver acc':>14}{'gate floor':>12}{'stamp?':>8}")
    print("    " + LINE[:69])
    for r in G.sweep_queue():
        depth = "inf" if r["depth"] == float("inf") else f"{r['depth']:.1f}"
        lat = "inf" if r["latency_h"] == float("inf") else f"{r['latency_h']:.2f}"
        print(f"    {r['utilisation']:>12.2f}{depth:>9}{lat:>11}{r['approver_accuracy']:>14.3f}"
              f"{r['gate_failure']:>12.4f}{str(r['rubber_stamp']):>8}")

    print(f"\n  WHERE THE RUBBER STAMP POINT ACTUALLY IS:")
    quiet = G.approver_accuracy_at_depth(0.0)
    break_even = -G.APPROVER_ACCURACY_DECAY_DEPTH * math.log(0.5 / G.BASE_APPROVER_ACCURACY)
    print(f"    approver accuracy starts at {quiet:.3f} and halves at a queue depth of "
          f"{break_even:.1f} items")
    print(f"    for comparison, a utilisation of 0.95 on this pool means a depth of "
          f"{G.ApprovalQueue().queue_depth(76.0):.1f} -- still above half accuracy,")
    print(f"    and the pool is unstable above utilisation 1.0, where accuracy goes to zero.")
    print(f"    the control degrades CONTINUOUSLY, which is why it needs an alert rather than a")
    print(f"    configuration flag: nothing in the setup ever says 'I am weaker than you think'.")

    print(f"\n  AND THE GATE'S FLOOR MOVES WITH IT -- the quantity experiment 2 held constant at "
          f"{G.gate_failure('gated', 0.90, 0.0, False):.4f}:")
    print(f"\n    {'queue depth':>13}{'approver acc':>14}{'gate failure':>14}")
    print("    " + LINE[:43])
    for depth in (0.0, 5.0, 20.0, 60.0, 150.0, 400.0):
        acc = G.approver_accuracy_at_depth(depth)
        print(f"    {depth:>13.0f}{acc:>14.3f}{G.P_TARGET_IRREVERSIBLE * (1 - acc):>14.4f}")

    print(f"\n  THE SCALING QUESTION -- approval demand grows with agent volume:")
    print(f"\n    {'irreversible/day':>19}{'per hour':>10}{'utilisation':>13}{'latency h':>11}{'servable':>10}")
    print("    " + LINE[:66])
    for demand in (50, 200, 500, 1000, 2000, 5000):
        a = G.approval_surface(demand, G.P_TARGET_IRREVERSIBLE)
        lat = "inf" if a["latency_h"] == float("inf") else f"{a['latency_h']:.2f}"
        print(f"    {demand:>19}{a['per_hour']:>10.1f}{a['utilisation']:>13.2f}{lat:>11}"
              f"{str(a['servable']):>10}")

    print(f"\n  THE TWO ANSWERS TO AN UNAFFORDABLE QUEUE:")
    print(f"\n    {'demand/day':>12}{'approvers needed':>18}{'multiple of today':>19}")
    print("    " + LINE[:51])
    for demand in (200, 1000, 5000, 20000):
        u = G.undo_vs_approve(demand)
        print(f"    {demand:>12}{u['approvers_needed']:>18}{u['headcount_multiple']:>18.1f}x")

    print(f"""
  FINDING: the corpus makes approval mandatory for irreversible actions [T] and treats it as a
  control. The measurement says it is a RESOURCE, and its failure mode is not refusal -- it is
  approval. Queue depth rises with utilisation, approver accuracy falls with queue depth, and the
  gate's floor -- the quantity experiment 2 held constant at {G.gate_failure('gated', 0.90, 0.0, False):.4f} -- is a function of
  how busy the humans are: {G.P_TARGET_IRREVERSIBLE * (1 - G.approver_accuracy_at_depth(0.0)):.4f} on a quiet day and
  {G.P_TARGET_IRREVERSIBLE * (1 - G.approver_accuracy_at_depth(400.0)):.4f} at a depth of four hundred items. A {G.P_TARGET_IRREVERSIBLE * (1 - G.approver_accuracy_at_depth(400.0)) / (G.P_TARGET_IRREVERSIBLE * (1 - G.approver_accuracy_at_depth(0.0))):.0f}x swing in the
  strength of the control, with nothing in the control's own configuration having changed.

  Three consequences:

    * THE GATE HAS A CAPACITY, and it is a headcount, not a flag. Every architecture that scales
      agent volume without scaling the queue converts a hard control into a rubber stamp at a
      predictable point: the point where approvers stop reading -- a queue depth of
      {break_even:.0f} items on this model. The degradation is CONTINUOUS, so there is no moment at which
      the configuration becomes wrong; it becomes wrong gradually, and the only instrument that can
      see it is the approval rate.
    * THE APPROVAL RATE IS THE METRIC, not the request rate. A rising approval rate with a stable
      request rate is the signature of a degraded reviewer, and no detection metric in this topic
      can see it -- the same class of blindness as experiment 4's missing tag and experiment 5's
      novel class. In all three cases the control weakens without any control reporting it.
    * THE OTHER ANSWER IS TO REMOVE THE IRREVERSIBILITY. The corpus never names this option and it
      is the only one that scales: approver headcount grows linearly with agent volume, while an
      UNDO converts an irreversible action into a reversible one and takes it off the queue
      entirely. At 20,000 irreversible actions a day the queue needs {G.undo_vs_approve(20000)['approvers_needed']} reviewers on this model -- which is
      the point at which "add an undo path" stops being a nicety and becomes the design.
""")


# ----------------------------------------------------------------------------------------------
def run_all() -> None:
    exp_rail_stack()
    exp_which_lever()
    exp_false_refusal()
    exp_trust_tags()
    exp_decay()
    exp_approval_queue()


if __name__ == "__main__":
    run_all()
