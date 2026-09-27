# T19 — FinOps, Token Economics & Sovereignty: High-Level Design

> `T19` · **Transcript coverage:** primary · [LLD](LLD.md) · [Cheat sheet](../../00-cheat-sheets/T19-finops-sovereignty.md) · [Case study](../../01-case-studies/T19-finops-sovereignty.md) · [Interview bank](../../02-interview-questions/T19-finops-sovereignty.md) · [Runnable core](run.py) · [Production](production/README.md) · [Sequences](docs/SEQUENCES.md)

Every number in this document is produced by `python run.py`. Corpus figures carry `[T]`, supporting-repository figures `[R]`, and anything this design derives is marked `[D]` — including the bill decomposition and the team mix, which are **synthetic and say so**. Nothing here is a fabricated benchmark.

---

## 1. System context

### 1.1 The claim this design is built on

This topic looks like two topics. It is one, and the corpus says so twice without noticing:

> "Optimization without measurement is just guessing." — tokenomics talk **[T]**

> "You can't legislate a memory dump." — sovereign-AI talk **[T]**

The first sentence is about cost. The second is about sovereignty. Both are statements of the same
rule: **a claim about a system is worth nothing until it is measured, and a claim about trust is worth
nothing until it is evidenced.** The cost half is measured in a unit; the sovereignty half is
evidenced by an attestation. Both of those are chosen before the work starts, and both of them decide
the answer.

So the design claim is:

> **Cost and sovereignty are the same question asked twice — *who controls the layer that turns my
> data into value* — and both are settled by a unit and an evidence type, each chosen before any
> measurement happens. Every failure in this topic is a failure to notice which one was chosen.**

Four consequences follow, and they organise the whole design.

**A discount is a price and a saving is an amount.** The corpus's caching lever is stated as a price
("cache reads ~10× cheaper" **[T]**); the programme banks an amount (`line_reduction × share_of_bill`).
The two differ by a factor of three in this stack, and the price is always the larger number.

**The unit precedes the measurement.** The corpus's own two data points — 30¢→16¢, 630→91 lines **[T]**
— read as a 46.7% saving, a 21.5% increase, or a 269.2% increase depending on the denominator. The
choice is made by whoever writes the dashboard, and the team being measured usually writes it.

**Sovereignty is a vector with prerequisites, and its floor is not its mean.** The corpus rejects the
binary ("I think that's wrong… you need to start looking at it as a system property, and this property
has various dimensions" **[T]**) and names four: control, trust, economics, continuity. Trust cannot
exceed control — an attestation of an environment you did not choose is a statement about somebody
else's property. A design that averages four dimensions is measuring the slide.

**A legal property has no error tolerance.** Residency is a filter, never a score. A 99.84%-correct
router is non-compliant, not 99.84% compliant. This is the same conclusion T14 reaches about routing
quality, arriving here as a compliance requirement instead of a performance one.

### 1.2 The four planes of this design

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  SPEND PLANE                                          ── what the bill is made of   │
│                                                                                      │
│   8 bill layers (system_prompt 31.1% · model_tier 20% · output 16% · reasoning 15%   │
│   · retrieved 8% · memory 7% · retries 3%)                                           │
│   10 levers, each targeting (layer, share_of_layer, discount)                        │
│   Two lever KINDS: reduction (removes work) · pricing (reprices work)                │
│   Outputs: banked saving per lever · the stack · the ceiling · the ordering spread   │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  MEASUREMENT PLANE                                    ── what the number means      │
│                                                                                      │
│   The unit (per artifact · per file · per line · per call · per task · per resolution)│
│   The workload mix (the cheap majority vs the expensive tail)                        │
│   The denominator rule: whose numerator can the measured team move?                   │
│   Outputs: a sign · a trend · the growth axis                                        │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  ATTRIBUTION PLANE                                    ── who spent it                 │
│                                                                                      │
│   Tags by team / feature / tenant / model / route / environment                      │
│   Coverage (spend-weighted, and per team, beside that team's growth)                 │
│   Showback → chargeback → per-run caps                                                │
│   Reconciliation: gateway spend vs provider invoices                                 │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  SOVEREIGNTY PLANE                                    ── who controls the layer       │
│                                                                                      │
│   4 dimensions, with PREREQUISITES  (trust ≤ control, continuity ≤ control)          │
│   2 readings: mean (the slide) and floor (what an adversary gets to choose)          │
│   Attestation arithmetic (penalty → uplift) · residency as a filter                  │
│   Continuity as a minimum over 4 layers                                              │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

The planes are ordered, and the order is the design. A spend decision that has not passed through
the measurement plane has no unit and therefore no meaning. A measurement that has not passed through
attribution has no owner. A governance decision that has not passed through the sovereignty plane may
be illegal regardless of what it saves. Teams consistently build the top plane first — dashboards,
discounts, an optimisation backlog — and then discover that the number they optimised was the wrong
one, in the wrong unit, with no owner, in a jurisdiction they cannot use.

### 1.3 What this design is *not*

| Not this | Because |
|---|---|
| A cost-reduction programme | The lever ladder is a **bound**, not a plan. Its ceiling is 60.6%, and its best ordering is one of three defensible ones that differ by 4.2 points **[D]**. A programme that begins with a target has already chosen its unit. |
| A procurement decision | Self-host vs API is decided by a **break-even the sovereignty plane constrains**, not by $/token. The cheapest posture in §6 has the widest claim-to-evidence gap in the table. |
| A dashboard | Every metric in §4.5 is a ratio whose denominator the measured team chooses. The dashboard is the **second** artifact; the unit memo is the first. |
| A compliance checkbox | §6.3 shows four dimensions with a floor of zero under a mean of 0.75. A checkbox is a statement about one dimension. |
| A one-time project | §5.5 shows coverage **falling** from 90.5% to 82.4% under one growth cycle with nobody changing anything **[D]**. Cost governance is a running system, not a deliverable. |

---

## 2. The spend plane: what the bill is made of

### 2.1 The eight layers

The supporting guide's decomposition **[R]** supplies the map and its anchor statistic: **system prompts
are ~69% of input tokens and only ~28% of calls use prompt caching**. Input is ~45% of the bill, so the
prefix line is `0.69 × 0.45 = 31.05%` — the largest single line in most stacks, sitting in the most
mature, least risky lever available.

| Layer | Share | Driver | Lever |
|---|---|---|---|
| `system_prompt` | 31.1% | fixed scaffolding, tool defs, few-shot | prompt / prefix caching |
| `model_tier` | 20.0% | frontier vs mid vs small / self-hosted | right-sizing, routing, distillation |
| `output_length` | 16.0% | verbosity, format, `max_tokens` | caps, terse output contracts |
| `reasoning_tokens` | 15.0% | extended thinking | gate thinking by task complexity |
| `retrieved_context` | 8.0% | RAG chunks, injected documents | context discipline |
| `conversation_memory` | 7.0% | chat history, agent scratchpad | windowing, compaction |
| `retry_overhead` | 3.0% | transient errors, guardrail re-runs | bounded retries, circuit breakers |
| **TOTAL** | **100%** | | |

The decomposition matters because it is the **only** thing that converts a discount into a saving, and
the corpus's own lever descriptions are all discounts. The share column is `[D]`; the 31.05% line is
pinned to the corpus statistic and is `[R]`.

### 2.2 The discount, its asymptote, and the write premium

A cache read at a **10× discount** cuts the prefix line by **74.6%** and the bill by **23.2%** **[D]**.
Three numbers matter and usually only the first is quoted.

| Discount | Prefix line, now | Prefix line, fixed | Line reduction | **Banked** |
|---|---|---|---|---|
| 2× | 0.8600 | 0.5500 | 36.0% | 11.2% |
| 5× | 0.7760 | 0.2800 | 63.9% | 19.8% |
| 10× | 0.7480 | 0.1900 | 74.6% | **23.2%** |
| 20× | 0.7340 | 0.1450 | 80.2% | 24.9% |
| 50× | 0.7256 | 0.1180 | 83.7% | 26.0% |
| 100× | 0.7228 | 0.1090 | 84.9% | 26.4% |
| ∞ | — | — | 86.1% | **26.7%** |

**The asymptote is 26.7%** **[D]**. A free cache read cannot do better than 26.7% of the bill, because
the uncached share of calls (a 10% target) still pays full price and nothing outside the prefix is
touched. This is the bound nobody quotes, and it is what a business case should be capped at.

**The write premium is the only number here that can be negative.** A cached prefix costs ~1.25× a
fresh token to *write* **[R]**. At a 10× read discount the break-even is:

```
n > write_price / (1 - 1/discount)  =  1.25 / (1 - 0.10)  =  1.39 reads
```

So a prefix read once is a **bill increase** of 0.35 units; read twice it saves 0.55 **[D]**. A rollout
that enables caching globally on single-shot traffic has made the bill worse and left a dashboard that
says caching is on. **Design consequence:** caching is enabled *per prefix*, and the read-to-write
ratio is a first-class metric (§4.4, metric 4).

### 2.3 Two kinds of lever, and they compose differently

| Kind | What it does | Composition | Examples |
|---|---|---|---|
| **reduction** | removes work from the bill | mostly independent across layers; **conflicts** where two levers reach the same layer share | prompt_caching, reasoning_gating, context_discipline, output_caps, bounded_retries, distillation, quantisation, difficulty_routing |
| **pricing** | reprices work that still happens | multiplies with everything left; **conflicts with other pricing levers on the same traffic** | batch_lane (−50%, 24h ceiling), reserved_capacity (15–70%) |

This distinction is the design's most load-bearing structural idea, and it is not in the corpus. A
pricing lever applies to **whatever remains** — so there is exactly one pricing slot per traffic slice,
and two pricing levers on the same tokens is one saving counted twice. The declared conflicts in the
runnable core are:

```
difficulty_routing  ×  distillation          (both claim the easy path; distil it OR route it)
batch_lane          ×  reserved_capacity     (both reprice the same tokens)
difficulty_routing  ×  reserved_capacity     (committing to capacity for traffic you intend to route away)
```

### 2.4 The ladder, ranked two ways

The corpus's ladder, ranked by the largest multiple each claim implies — which is what reading it
gives you:

| # | Lever | Claim | Notation | Banked |
|---|---|---|---|---|
| 1 | `distillation` | 5–40× per-token cut **[R]** | multiple | **2.4%** (#8) |
| 2 | `reasoning_gating` | 3–15× multipliers **[R]** | multiple | 9.0% (#4) |
| 3 | `prompt_caching` | ~10× cheaper reads **[T]** | multiple | **23.2%** (#1) |
| 4 | `reserved_capacity` | 15–70% **[R]** | percentage | 15.8% (#2) |
| 5 | `difficulty_routing` | "often cuts total spend by half" **[T]** | percentage | 5.5% (#5) |
| 6 | `quantisation` | unquantified **[T]** | none | 2.4% (#9) |
| 7 | `batch_lane` | ~50% **[R]** | percentage | 10.0% (#3) |
| 8 | `output_caps` | 20–40% token cut **[R]** | percentage | 4.8% (#6) |
| 9 | `context_discipline` | unquantified **[T]** | none | 4.5% (#7) |
| 10 | `bounded_retries` | unquantified **[R]** | none | 1.5% (#10) |

**Eight of the ten levers move position between the two orderings** **[D]**. Three claims are
unquantified in the corpus and are marked as the author's reading rather than a parse: the runner
stores `claim_multiple` and `claim_notation` as explicit fields precisely because a regex over the
prose cannot tell "$N× cheaper" from "$N% quality", and reading "~95% quality" as a 20× saving is the
error that motivated the field.

### 2.5 The ceiling, and the headline

> "…from a naive baseline to roughly a tenth of the cost, with no change to the model itself." — tokenomics **[T]**

That is a **10× claim**. Tested honestly — every lever at its full claimed discount, taking disjoint
shares within a layer so nothing is double-counted, then the single best pricing lever applied to what
remains:

| | Reduction | Multiple |
|---|---|---|
| Ladder ceiling **[D]** | **60.61%** | **2.54×** |
| Headline **[T]** | 90.00% | 10.00× |
| **Gap** | **29.39 points** | |

The gap is not a missing technique. It is a **missing argument**. Consider what the ladder's single
most aggressive number contributes: distillation's 40×, at its stated reach of 15% of `model_tier`,
is worth **2.0 points of the ceiling** (and the ceiling without it is 58.59%) **[D]**. To move the
ceiling meaningfully, that 40× would have to be applied to the whole non-prefix bill — which is exactly
the application the same source calls a failure mode:

> distillation "wins on narrow, high-volume tasks and fails on open-ended long-tail work" **[R]**

**Design consequence: quote the ceiling.** 60.6% off with no model change is an excellent programme,
and it is defensible in front of a reviewer. A 90% target is not achievable from this ladder, and a
programme that commits to it will either miss or start inventing savings.

### 2.6 The ordering, and why the natural one is the worst

Three orderings, every one of them defensible from the corpus's instruction ("before you touch the
model at all, exhaust the free wins" **[T]**):

| Ordering | Rule | Result | Skipped |
|---|---|---|---|
| `saving_first` | highest saving per unit of ease, pricing competing | **59.3%** | `batch_lane` (conflict) |
| `free_first` | easy reduction levers first, best pricing lever last | 56.5% | — |
| `ease_first` | ease, everything on one axis | **55.2%** | `reserved_capacity`, `distillation` |

**Spread: 4.15 points** **[D]**. The worst is `ease_first` — "do the easy things first" — and the
mechanism is not that easy levers are weak. It is that `batch_lane` (ease 4) beats `reserved_capacity`
(ease 2) to the single pricing slot; once `batch_lane` applies, `reserved_capacity` is skipped as
conflicted, and **15.8 points of the bill are lost to an ordering rule**. The same ordering also loses
distillation behind its routing conflict.

**Design consequence:** the rule is not an ordering at all. It is a rule about **which levers compete**.
Sort *reduction* levers by ease among themselves; decide the *pricing* lever once, on value, before
anything else in the plan is fixed. A plan that sorts all ten levers on one axis is choosing its 4.2
points by accident.

---

## 3. The measurement plane: the unit decides the answer

### 3.1 Three signs from two rows

> "Between February and April the cost of a 10,000-token knowledge artifact fell from 30¢ to 16¢,
> while lines of code drafted fell from 630 to 91 and files touched from 8.2 to 3.6, and thinking
> tokens rose." — Token Raj **[T]**

| Unit | Ratio | Change | Reading | Note |
|---|---|---|---|---|
| per artifact | 0.5333 | **−46.7%** | cheaper | what was *asked for* — the unit the headline uses |
| per file | 1.2148 | **+21.5%** | more expensive | what *changed* — a proxy for coupling and review load |
| per line | 3.6923 | **+269.2%** | more expensive | what was *delivered* — the unit cheapest to inflate |

The signs disagree, and the reason is visible in the raw data: **the deliverable shrank faster than the
price did.** Cost fell to 53.3% of its February value; lines fell to 14.4%. So the per-unit cost of
delivered work **tripled** (3.69×) while the per-request cost halved **[D]**. Both statements are true,
and both will be made by someone.

### 3.2 The denominator rule

This is the most transferable finding in the topic, because the metric is not the problem — the
**denominator** is, and the team being measured usually picks it.

> "Companies are doing tokens per output as one of the metrics — you need to change the metrics. You
> need to start looking at it as a business outcome rather than the token spend." — Token Raj **[T]**

| Metric | Denominator | Gameable by | Cannot see |
|---|---|---|---|
| cost per call | calls | splitting one task into more calls | the tail — 1% of calls can be 29% of the bill |
| **tokens per output** | visible output tokens | moving work into reasoning tokens | reasoning billed at the output rate, absent from the response |
| cost per task | tasks | doing less work per task (Feb→Apr) | whether the task got smaller |
| cost per resolved case | resolutions | reclassifying resolutions | much — the corpus's preferred direction **[R]** |
| lines of code / PRs | commits | generation volume | value — a model "tends to make a shallow copy of the whole file" **[T]** |

`tokens per output` is the sharpest case, and it is **systematically optimistic** rather than merely
uninformative: a call with a 300-token visible answer and 2,000 thinking tokens scores 1.0× on the
metric and costs **7.67×** **[D]**. The corpus reports reasoning multipliers "anywhere from ~3× to ~15×
depending on the task" **[R]**.

**Design consequence: the unit memo comes first.** Before any optimisation work, every programme names
its denominator in writing, names who can move it, and names the consequence-weighted companion metric.
A programme with no unit memo is measuring whatever the last person to edit the dashboard chose.

### 3.3 The mix: an average that is the wrong basis for an order

| Workload | Calls | Volume share | Cost | Bill share |
|---|---|---|---|---|
| chat | 990,000 | 99.0% | 2,970 | 70.8% |
| agentic | 10,000 | 1.0% | 1,224 | **29.2%** |

An agentic call costs **29.2× the mean call** **[D]** — the same tail-over-mean shape as T08's goodput,
T09's p99 and T17's gate, here with a growth axis. Grow the tail population at the scenario's own
**4.3×** **[D]** and the shares invert:

| | Before | After 4.3× growth |
|---|---|---|
| chat bill share | 70.8% | 36.1% |
| **agentic bill share** | 29.2% | **63.9%** |
| total | 4,194 | 8,233 |

**Design consequence:** an optimisation order derived from *today's* average is the wrong order for
*next year's* bill, and it is wrong in a predictable direction. The tail is where the money is going,
which is why distillation's long-tail failure (§2.5) is a much larger problem than its 2.4-point
contribution suggests.

### 3.4 Reasoning tokens: invisible by construction

Reasoning is billed at the **output rate** and never appears in the response **[R]**. Three places this
matters:

1. **Cost.** It is its own 15% layer here, and the multiplier literature puts it at 3–15× **[R]**.
2. **Measurement.** It is the mechanism behind the one metric that hides spend (§3.2).
3. **Latency.** T06's ITL and T08's goodput both treat output tokens as the stream a user waits on;
   reasoning tokens are output tokens for billing and *pre*-output tokens for experience, and no
   dashboard in the corpus models that asymmetry.

**Design consequence:** `reasoning_tokens` is a bill layer with its own lever (`reasoning_gating`,
worth 9.0% at a 60% reduction on the whole layer). Gate thinking by task complexity, and **measure the
quality cost on the tasks you gate *off*, not on the ones you leave on** — the failure mode of this
lever is a task that needed to think and did not.

---

## 4. The attribution plane: who spent it

### 4.1 The precondition

> "Tag every call by team, feature, tenant, model, route and environment. Without attribution there is
> no way to compute unit economics." **[R]** — showback before chargeback, and chargeback only once the
> tags are trustworthy.

Coverage is a **fraction**, and uncovered spend is **invisible, not neutral**. The two readings of one
coverage figure already disagree:

| | Value |
|---|---|
| spend-weighted coverage | 90.46% |
| simple mean of the per-team percentages | 84.60% |

Here the simple mean *understates* — but that direction is not the point, and it is not stable. The
point is that either figure is an average over a non-uniform population.

### 4.2 Coverage falls as the business succeeds

| Team | Spend | Tagged | Untagged | Growth |
|---|---|---|---|---|
| agent_pilot | 100,000 | **55.0%** | 45,000 | **6.5×** |
| data_science | 140,000 | 85.0% | 21,000 | 3.0× |
| customer_assist | 260,000 | 95.0% | 13,000 | 2.1× |
| platform_copilot | 420,000 | 98.0% | 8,400 | 1.4× |
| finance_ops | 80,000 | 90.0% | 8,000 | 1.2× |

Now let each team grow at its own rate, scaled so nothing shrinks **[D]**:

| | Before | After |
|---|---|---|
| coverage | 90.46% | **82.43%** (−8.03 pts) |
| untagged share | 9.54% | **17.57%** |
| untagged spend | 95,400 | 336,800 |
| total bill | 1,000,000 | 1,916,667 (1.92×) |

Nobody tagged worse. **The untagged spend is concentrated in the team growing 6.5×**, so it is not a
random sample of the bill — a 90% coverage figure is not "90% of the picture", it is the whole picture
*minus the part about to matter*. This is the same tail-over-mean shape as §3.3, applied to a
governance metric.

**Design consequence:** coverage is reported **per team with that team's growth rate beside it**. The
pair is the signal; either alone is not. A team at 55% coverage growing 6.5× is the finding, and it is
invisible in both the global coverage figure and the per-team percentage.

### 4.3 The governance ladder

| Stage | Needs | Behavioural pressure | Fails when |
|---|---|---|---|
| no attribution | nothing | 0.0 | the first time the bill doubles |
| showback | tags | 0.3 | nobody acts on it |
| chargeback | **trusted** tags | 0.8 | teams route around the gateway |
| hard per-run caps | a gateway on the critical path | 1.0 | the cap is set from a mean rather than a tail |

Showback-before-chargeback is a sequencing instruction and the reason is political as well as
technical: chargeback on untrusted tags produces a fight about the numbers instead of a plan to reduce
them.

### 4.4 The failure mode chargeback introduces

Once spend is billed to a team, **the cheapest route for that team is an account the gateway never
sees.** The only detector is reconciliation:

| | Value |
|---|---|
| provider invoices | 1,000,000 |
| gateway spend | 880,000 |
| **residual** | **120,000 (12.0%)** |

A 12% residual is not a rounding error; it is traffic that left the gateway **[D]**. It belongs on the
governance dashboard, not in a quarterly audit, because it is also the only signal that fires on the
failure mode chargeback itself creates.

### 4.5 The cap rule

> "Runaway agents have burned tens of thousands of dollars over a single weekend." **[R]**

That is an argument for a ceiling. It is not an argument for a *small* ceiling, and the difference is
which statistic it is drawn from:

| | Mean task | p99 task | Ceiling | Verdict |
|---|---|---|---|---|
| drawn from the tail | 0.42¢ | 12.24¢ | $5.00 | legitimate long tasks survive; runaway caught |

A cap drawn from the **mean** terminates legitimate work. A cap drawn from the **tail** catches
exactly the runaway the corpus warns about, because with a mean of 0.42¢ and a p99 of 12.24¢ a $5
ceiling kills nothing legitimate and still catches a runaway two orders of magnitude above the mean
**[D]**. The mean is not a smaller version of the tail; it is a different population.

---

## 5. The sovereignty plane

### 5.1 A vector, not a binary

> "People look at sovereignty as something near-binary — it is a sovereign setup or a non-sovereign
> setup. I think that is wrong… you need to start looking at it as a system property, and this
> property has various dimensions." — sovereign AI **[T]**

| Dimension | The question it asks |
|---|---|
| **control** | Where does inference run, on which model, on which accelerator? |
| **trust** | What code and artifacts are actually executing? |
| **economics** | Who controls the token cost? |
| **continuity** | Can you operate without a single vendor? |

And the corpus's closing definition of sovereignty — **freedom to choose, control, verify and operate**
**[T]** — maps onto exactly those four.

### 5.2 The prerequisites, and the cap that hides

A dimension **cannot exceed the one it depends on** **[D]**:

```
trust      ≤ control      — you cannot cryptographically attest an environment you did not choose
continuity ≤ control      — you cannot switch vendors if the vendor picks the model
```

| Posture | Claimed | Effective | Capped | Weakest |
|---|---|---|---|---|
| `compliance_only` | 0.438 | 0.438 | — | trust |
| `regional_attested` | 0.825 | 0.812 | trust | economics |
| `own_accelerators` | 0.887 | 0.887 | — | trust |
| `own_stack` | 0.975 | 0.975 | — | trust |
| `api_with_dpa` | 0.288 | 0.288 | — | trust |
| `vendor_tee_lease` | 0.500 | **0.413** | **trust** | continuity |

`vendor_tee_lease` is the posture the rule exists for: it claims **0.85 trust** and can evidence
**0.50**, because the silicon *and* the model are not yours — the attestation proves a property of an
environment you did not choose. That is a 0.087 cut in the effective mean and it is invisible unless
the cap is computed rather than assumed.

### 5.3 The mean and the floor

| Posture | Mean | Floor | Gap |
|---|---|---|---|
| `compliance_only` | 0.438 | **0.100** | **0.338** |
| `api_with_dpa` | 0.288 | 0.050 | 0.238 |
| `vendor_tee_lease` | 0.413 | 0.300 | 0.113 |
| `regional_attested` | 0.812 | 0.750 | 0.062 |
| `own_accelerators` | 0.887 | 0.850 | 0.037 |
| `own_stack` | 0.975 | 0.950 | 0.025 |

The **postures that sound most defensible on a slide have the largest mean-to-floor gap**, because a
contract and a regional endpoint do not make the model yours **[D]**. A review that scores four
dimensions and averages them is measuring the slide; the floor is what an adversary or a regulator
gets to choose.

### 5.4 The attestation arithmetic

> "It is a trade-off. It is not going to run as fast as what it was running before. But it is going to
> run more securely." — sovereign AI **[T]**

An unquantified trade-off is a trade-off nobody can approve. The arithmetic:

```
cost uplift = 1/(1 − throughput penalty) − 1
```

| Throughput penalty | Cost uplift | At 40% of traffic attested |
|---|---|---|
| 5% | 5.3% | 2.1% |
| 10% | 11.1% | 4.4% |
| **20%** | **25.0%** | **10.0%** |
| 30% | 42.9% | 17.1% |
| 40% | 66.7% | 26.7% |

**A 20% throughput penalty is a 25% cost uplift, not 20%** **[D]** — capacity is what you buy, and
`1/(1 − 0.20) = 1.25`. Getting this wrong in the direction of the penalty undersizes every TEE budget
by about a fifth. The right-hand column is the design's answer to "we cannot afford to attest
everything": attest the regulated slice (40% of traffic here costs 10% on the total bill) and leave the
rest on the cheaper posture.

### 5.5 Residency is a filter, never a score

| Gap | Noise | Leak rate | Leaks / 10,000 | As a filter |
|---|---|---|---|---|
| 2.0 | 1.0 | 2.3% | 227.5 | 0 |
| 1.0 | 1.0 | **15.9%** | **1,586.6** | 0 |
| 0.5 | 1.0 | 30.9% | 3,085.4 | 0 |

A **score**-based residency router discounts the out-of-region option; a **filter** removes it. Any
non-zero leak rate is the whole argument, and residency has **no error tolerance**: a 99.84%-correct
router is **non-compliant**, because residency is a legal property rather than a quality score **[D]**.

**Design consequence:** residency is implemented by removing the out-of-region option from the
candidate set, never by penalising it. This is T14's "routing is a filter for compliance requirements"
arriving as a legal requirement instead of a performance one.

### 5.6 Continuity is a minimum, not a mean

| Accelerator | Model families | Engines | Regions | Floor | Mean | Binding |
|---|---|---|---|---|---|---|
| 2 | 2 | 2 | 2 | 1.0 | 1.00 | accelerator |
| 2 | 3 | 2 | 1 | **0.0** | 0.75 | cloud_region |
| 2 | 2 | 1 | 3 | **0.0** | 0.75 | serving_engine |
| 1 | 4 | 3 | 3 | **0.0** | 0.75 | accelerator |

Three of the four examples score a mean of **0.75** and a floor of **zero** **[D]**: two accelerators,
two clouds and two model families with **one serving engine** is a single-vendor system, and the mean
will not say so.

**Design consequence:** the cheapest continuity purchase in most stacks is **not a second cloud — it is
a second serving runtime on the same weights**. Accelerator choice is explicitly part of the
implementation strategy **[T]**, so the layer that is fixed *first* (silicon) is also the one that is
hardest to change later.

### 5.7 Three-way trust, and what each party can prove

The corpus names three exposures, each of which has a different closer **[T]**:

| Party exposed to | Exposure | Closer | Kind | Proves |
|---|---|---|---|---|
| model owner | putting weights on someone else's infrastructure exposes "years of training and the millions of dollars they are spending" | confidential computing on the lease | cryptographic | weights are not readable by the host |
| infrastructure provider | "a model may ship malicious code along with the model weights — the first time Qwen came out, they installed it, it was trying to do a netconnection back to the servers" | Kata containers, network isolation | isolation | malicious code is contained to that layer |
| consumer | the provider trains on their data, or builds a competing service from it | **TEE + attestation** | cryptographic | "only you as the data owner can see that in plain text, but you can attest that" |

The corpus's own assessment of the state of the art is the sentence to carry into a design review:
*"right now the only frameworks which are protecting sovereignty are the legal frameworks… how can you
prove cryptographically that none of the data stored in your CPUs and GPUs can be stolen? You need a
cryptographic attestation"* **[T]**. A DPA proves an intention; a regional endpoint proves where the
request was routed, not what ran; only an attestation proves a state.

### 5.8 "Open weights" is a spectrum, not a checkbox

> "When people say open source model they'll just publish open weights. They don't show the source code
> which was used to train, and they don't showcase the kind of silicon they used to train that, so that
> you can replicate it and reproduce it." — Token Raj **[T]**

| Position | Weights | Training code | Training data | Reproducible | Openness |
|---|---|---|---|---|---|
| closed API | ✗ | ✗ | ✗ | ✗ | 0.00 |
| open weights (the common case) | ✓ | ✗ | ✗ | ✗ | 0.25 |
| open weights + training code | ✓ | ✓ | ✗ | ✗ | 0.50 |
| open source (rare) | ✓ | ✓ | ✓ | ✓ | 1.00 |

**Design consequence:** a procurement question that asks "is it open source?" has three possible
answers and usually gets the middle one. Ask for the four columns. Only the last row is a continuity
answer, and it is rare enough that a continuity plan built on it is a plan built on a supplier's
goodwill.

---

## 6. Where the planes meet: the governance contract

The four planes are not four projects. They are four **contracts** that must be signed in order, and
the sovereignty plane constrains the spend plane rather than the other way round.

| Contract | From | To | Terms |
|---|---|---|---|
| **the unit memo** | finance + the owning team | every dashboard | the denominator, who can move it, and the consequence-weighted companion. Signed **before** any optimising. |
| **the tag contract** | platform | every team | which tags are mandatory, what "trustworthy" means numerically, and coverage reported beside growth |
| **the residency filter** | legal + platform | routing | residency is a filter; the out-of-region option is removed, not discounted; the leak rate is a tested invariant, not a monitored metric |
| **the posture decision** | exec + security | architecture | the four dimensions with a **floor** threshold, the attested share, and the throughput penalty converted to a cost uplift before approval |
| **the lever plan** | platform | finance | levers ranked by `discount × share_of_bill`, the pricing lever chosen once on value, and the **ceiling** quoted rather than the headline |

Two of these are the ones teams skip. The **unit memo** is skipped because it feels bureaucratic and is
actually the decision. The **residency filter** is skipped because a score is easier to write than a
filter and passes the test suite either way — the leak only shows up in an audit.

---

## 7. Deployment topology

```
                        ┌──────────────────────────────────────────┐
                        │  GATEWAY  (attribution + enforcement)    │
                        │                                          │
   ┌─────────┐          │  · tags every call (§4.1)                │          ┌────────────┐
   │ clients │─────────▶│  · routes by difficulty tier             │─────────▶│ tier: small│
   └─────────┘          │  · RESIDENCY FILTER, not a score (§5.5)  │          │  (cheap)   │
                        │  · per-run ceiling + budget caps (§4.5)  │          └────────────┘
                        │  · emits the shadow-reconciliation pair   │          ┌────────────┐
                        └──────────────┬───────────────────────────┘─────────▶│ tier: mid  │
                                       │                                        └────────────┘
             ┌─────────────────────────┼─────────────────────────┐              ┌────────────┐
             │                         │                         │             │tier:frontier│
             ▼                         ▼                         ▼             └────────────┘
   ┌───────────────────┐   ┌───────────────────────┐   ┌──────────────────┐
   │ PREFIX STORE      │   │ METERING / LEDGER     │   │ ATTESTATION      │
   │                   │   │                       │   │                  │
   │ · per-prefix      │   │ · cost per team/      │   │ · attestation    │
   │   read/write      │   │   feature/tenant      │   │   verification   │
   │   ratio (§2.2)    │   │ · unit-normalised     │   │ · boot check:    │
   │ · disable per     │   │   cost (§3.1)         │   │   refuse to      │
   │   prefix, not     │   │ · coverage × growth   │   │   serve in a     │
   │   globally        │   │   (§4.2)              │   │   bad posture    │
   │ · stable prefix   │   │ · reconciliation vs   │   │ · TEE uplift     │
   │   contract        │   │   provider invoices   │   │   applied to     │
   └───────────────────┘   └───────────────────────┘   │   the budget     │
                                                       └──────────────────┘
             ┌──────────────────────────────────────────────────────┐
             │  POLICY STORE  (versioned with the scenario)         │
             │  bill shares · lever parameters · unit definitions   │
             │  · posture thresholds (floor, not mean)              │
             │  · attested share · pricing commitments              │
             └──────────────────────────────────────────────────────┘
```

Four properties of this topology carry design weight:

**The gateway is the only place all four planes touch.** It tags (§4), filters (§5.5), caps (§4.5) and
routes (§2.3's `difficulty_routing`). That is why it is also the failure domain: T14's gateway design
and this one are the same component seen from two directions, and §8.1 records that.

**The prefix store is a cost component with its own SLO.** Per-prefix read/write ratio is what makes
§2.2's write premium visible. A cache that cannot report per-prefix read counts cannot be operated
correctly — it can only be switched on.

**Metering is a ledger, not a dashboard.** It must survive a team disputing a number, which means
per-call records with tags, not aggregates.

**Attestation is enforced at boot, not monitored.** A posture that has degraded is not an alert; it is
a refusal to serve the traffic that requires it. This is T18's boot-check pattern, applied to the
evidence plane instead of the detection plane.

---

## 8. Failure domains

### 8.1 The table

| # | Failure | Observable signature | Blast radius | Design answer |
|---|---|---|---|---|
| 1 | **The wrong unit is chosen** | cost per request falls; cost per delivered outcome rises | the whole programme optimises the wrong number | the unit memo (§6), signed before work starts |
| 2 | **Prompt caching enabled globally** | caching is "on"; bill rose ~2% | the largest line in the bill | per-prefix read/write ratio; disable per prefix (§2.2) |
| 3 | **The headline is committed to** | quarter-on-quarter misses against a 90% target | credibility of the whole programme | quote the **ceiling**, 60.6% (§2.5) |
| 4 | **The pricing slot is filled by ease** | `batch_lane` deployed, `reserved_capacity` "not needed" | 15.8 points of the bill | decide the pricing lever once, on value (§2.6) |
| 5 | **Reasoning gated by cost, not complexity** | token spend falls; quality falls on long-tail tasks | exactly the tasks §3.3 shows are growing | measure quality on the tasks gated **off** (§3.4) |
| 6 | **Coverage reported as one number** | 90% coverage, "healthy" | the 17.6% about to matter is invisible | coverage beside growth, per team (§4.2) |
| 7 | **Chargeback drives traffic off-gateway** | provider invoice exceeds gateway spend | governance plane cannot see the spend at all | reconciliation on the dashboard (§4.4) |
| 8 | **Cap drawn from the mean** | legitimate long tasks terminated | user trust, and the tail workload | cap from the **tail** (§4.5) |
| 9 | **Residency implemented as a score** | "99.8% compliant" | a legal property, non-compliant | implement as a filter (§5.5) |
| 10 | **Trust claimed above control** | attestation in place; posture unreviewable | the attestation proves the wrong environment | prerequisites computed, not assumed (§5.2) |
| 11 | **Continuity read as a mean** | "three of four layers covered" | single-vendor system, unmanaged | read as a minimum (§5.6) |
| 12 | **TEE penalty read as its own uplift** | budget undersized ~20% | every attested path | `1/(1−p) − 1` before approval (§5.4) |
| 13 | **"Open source" accepted as a claim** | weights published, nothing reproducible | any continuity plan resting on it | the four columns (§5.8) |
| 14 | **A lever claimed twice** | two pricing levers in one plan | the saving is counted twice | declared conflicts (§2.3) |

### 8.2 The three that are silent

Most of the table announces itself: a bill that rose is a bill that rose. Three do not.

**The wrong unit (#1)** produces a *falling* number. Every dashboard is green. The programme is
succeeding at the metric and failing at the business, and the gap only closes when somebody asks what
the per-outcome cost did — which is the question the unit memo exists to have already asked.

**Coverage (#6)** falls by 8 points under growth with nobody changing anything (90.46% → 82.43%). No
alert fires, because no threshold was crossed; the number was always under the target and it drifted.

**The trust cap (#10)** is a *claimed* number that is not a *measurable* one. The posture table looks
satisfied. The `vendor_tee_lease` row claims 0.85 trust and can evidence 0.50, and the difference is
invisible unless the prerequisite is computed — which means the failure mode is a review that passes.

These three are the same failure as T18's, T17's and T08's: **a metric that averages away the
population that matters.** The remedy is the same shape in all four topics — measure the thing next to
the thing that hides it.

### 8.3 The one failure that is arithmetic

The TEE penalty (#12) is not a judgement call. `1/(1 − 0.20) − 1 = 0.25`. A business case written at a
20% uplift is wrong by 20% of itself, every time, and the error is always in the same direction: the
sovereign option looks cheaper than it is.

---

## 9. Build vs buy

| Component | Build | Buy | Decide by |
|---|---|---|---|
| **Gateway** | high control, high cost | a managed AI gateway | T14: the gateway is the attribution and enforcement point. **Buy the transport, build the policy** — the tags and the budget rules are yours. |
| **Prefix store** | you control the read/write accounting | provider caching (50–90% advertised **[R]**) | the **break-even is 1.39 reads** (§2.2). Buy it, but only where the ratio clears 2, and require per-prefix read counts from the vendor. |
| **Metering / ledger** | necessary | nothing good enough | this is the attribution plane. It cannot be outsourced, because the dispute resolution is the product. |
| **Attestation** | not viable for most | TEE leases (cost: §5.4) | the uplift is a **known number**; buy it for the slice that is regulated, not for everything. |
| **Routing tier** | a small classifier | a cascade service **[R]** | 45–85% at ~95% quality **[R]**; the eval that proves the 95% is yours. |
| **Accelerators** | 1.35× cost index, no elasticity | API at 0.85× | see the decision table below; the sovereignty plane constrains this, not $/token. |

### 9.1 Self-host vs API, as the sovereignty plane sees it

| Posture | Cost index | Control | Trust | Economics | Continuity | Floor |
|---|---|---|---|---|---|---|
| `api_with_dpa` | **0.85** | 0.55 | 0.05 | 0.30 | 0.25 | **0.05** |
| `compliance_only` | 1.00 | 0.70 | 0.10 | 0.40 | 0.55 | **0.10** |
| `vendor_tee_lease` | 1.05 | 0.50 | 0.85 → **0.50** | 0.35 | 0.30 | 0.30 |
| `regional_attested` | 1.10 | 0.85 | 0.90 → **0.85** | 0.75 | 0.80 | 0.75 |
| `own_accelerators` | 1.35 | 0.95 | 0.85 | 0.85 | 0.90 | 0.85 |
| `own_stack` | 1.60 | 1.00 | 0.95 | 0.95 | 1.00 | 0.95 |

The cheapest posture has a floor of **0.05** and the widest claim-to-evidence gap among the API-based
options. The design rule that falls out: **choose the posture by its floor against a written threshold,
then pay the cost index.** Choosing by cost index and then checking the floor produces a decision made
in the wrong order — which is how organisations end up with a 0.85-cost posture they cannot use for the
traffic that made the programme worth doing.

---

## 10. Capacity and cost model, worked

The base case is a 1,000,000-unit monthly bill with the §2.1 decomposition.

**Step 1 — the free win.** Enable prompt caching on stable prefixes only, at a provider discount of
10×, moving the cached share of calls from 28% to 90%:

```
line, now    = 0.28 × 0.10 + 0.72      = 0.7480
line, fixed  = 0.90 × 0.10 + 0.10      = 0.1900
line reduction = (0.7480 − 0.1900) / 0.7480 = 74.6%
banked         = 0.746 × 0.3105         = 23.2%  →  232,000 saved
```

Cost: near zero; risk: a prefix that changes per call gains nothing **[T]**, so the cache key is a
contract (§7, prefix store). **The asymptote is 26.7% (267,000)** — no further discount can beat it.

**Step 2 — the pricing decision, taken once.** `reserved_capacity` on sustained load at 45% share /
35% discount, applied last:

```
saving = 1.0 (whole bill) × 0.45 × 0.35 = 15.75%  →  157,500
```

`batch_lane` is now unavailable (conflict). If instead `batch_lane` at 20% share / 50% were taken
first, this drops to **10.0% (100,000)** and `reserved_capacity` is skipped — the 4.15-point ordering
spread in §2.6, priced.

**Step 3 — the reduction levers, by banked saving, disjoint shares, avoiding declared conflicts.**
Best plan (`saving_first`) = **59.3%**, final bill **40.7%** of the start:

```
reserved_capacity    15.8%   ← pricing, decided first, on value
prompt_caching       19.5%
distillation          2.0%
reasoning_gating      7.6%
difficulty_routing    4.1%
quantisation          1.3%
context_discipline    3.8%
output_caps           4.0%
bounded_retries       1.3%
batch_lane            0.0%   ← skipped: conflicts with reserved_capacity
```

**Step 4 — the sovereignty cost, quantified.** Attest 40% of traffic at a 20% throughput penalty:

```
per-token uplift    = 1/(1 − 0.20) − 1     = 25.0%
on those tokens     = 0.40 × 25.0%         = 10.0%
applied to the POST-optimisation bill (0.4070) = 0.0407   →  4.07 points
```

**Step 5 — the net.** `59.3%` off, then `4.07` points on for the attested slice: the bill lands at
**44.8%** of the starting level. The sovereignty cost is **6.9% of the saving** — small enough to
approve, large enough that a business case omitting it is wrong. Note the direction of the arithmetic:
the uplift is applied to what the attested traffic *costs after optimisation*, not to the original
bill, because optimisation and attestation apply to the same tokens.

**Step 6 — sensitivity, in the two directions that matter.**

| Change | Effect |
|---|---|
| provider cache discount 10× → 5× | banked 23.2% → 19.8% (−3.4 pts) |
| cached share target 90% → 70% | banked 23.2% → **15.7%** (−7.5 pts) |
| agentic tail grows 4.3× (§3.3) | tail becomes 63.9% of the bill; the lever order changes |
| TEE penalty 20% → 30% | uplift 25.0% → 42.9%; the sovereign add goes 4.07 → 6.98 points |

The first two are the ones a business case usually assumes its way past. The third is why §2.6's
ordering matters more every quarter.

---

## 11. What changes at 10×

| At 1× | At 10× | Why |
|---|---|---|
| tags by hand, in code | tag contract enforced at the gateway, coverage tested in CI | coverage already falls 8 points under 1.9× growth (§4.2) |
| one pricing lever, chosen by feel | pricing decided as a portfolio, per traffic slice | there is exactly one slot per slice (§2.3) |
| the lever list is a backlog | levers ranked by `discount × share_of_bill`, re-ranked quarterly | the mix inverts within one growth cycle (§3.3) |
| unit memo is a document | unit definitions versioned with the scenario, like policy | a unit that cannot be re-derived cannot be defended in a dispute |
| residency implemented in the router | residency is a **filter** with a tested leak invariant of zero | a score's leak rate scales with traffic (§5.5) |
| attest the pilot | attested share is a budget line, and attested and unattested traffic are separate SLOs | the uplift is 25% per token; mixing them hides it (§5.4) |
| continuity is "we have a second region" | continuity is a **minimum** across four layers, reviewed | three of four examples score a floor of zero (§5.6) |
| cost reviews are monthly | reconciliation runs continuously | a 12% residual is traffic that left the gateway (§4.4) |

The one that is not a scaling change: **the ceiling does not move at 10×.** The bill gets bigger and
the levers get the same percentages. 60.6% is the bound at any scale, and a 10× business is still
looking at a 90% headline it cannot reach.

---

## 12. Six takeaways

1. **A discount is a price; a saving is an amount.** A 10× cache read cuts the prefix line 74.6% and
   the bill 23.2%, and the curve ends at 26.7% because the uncached share still pays. Compute
   `discount × share_of_bill`, and never quote a discount as a saving.

2. **The write premium is the number that can be negative.** At 1.25× write and 10× read, a prefix
   must be read **1.39 times** to break even. Caching is enabled per prefix, with the read/write ratio
   on a dashboard, and disabled per prefix when it falls below 2.

3. **The ladder's ceiling is 60.6%, and the headline is 90%.** The 29.4-point gap is a missing
   argument, not a missing technique — the ladder's most aggressive claim contributes 2.0 points of
   the ceiling at its stated reach, and its stated failure mode is the workload that is growing. Quote
   the ceiling.

4. **The unit decides the sign, and the denominator is the decision.** 30¢→16¢ is −46.7% per artifact,
   +21.5% per file and +269.2% per line delivered. Write the unit memo before the programme starts,
   and pick a denominator the measured team cannot shrink by doing less work.

5. **Sovereignty is a vector with prerequisites and a floor.** `trust ≤ control`, `continuity ≤
   control`; three of four continuity examples score a mean of 0.75 and a floor of zero; a 20%
   throughput penalty is a **25%** cost uplift; and residency is a **filter**, because 99.84% correct
   is non-compliant.

6. **The measuring instrument degrades in silence.** Coverage falls 90.5% → 82.4% under one growth
   cycle with nobody changing anything, because the untagged spend is where the growth is. Report
   coverage beside growth, per team, and instrument the gateway-invoice residual before enabling
   chargeback.

---

## Sources

| File under `refs/` | Used for |
|---|---|
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/The_Token_Raj_Rethinking_the_AI_Inference_Stack.txt` | the subsidy ending and budget exhaustion; the 30¢→16¢ knowledge-artifact study (630→91 lines, 8.2→3.6 files, rising thinking tokens); "optimization without measurement is just guessing"; vendor-default incentives; the three-way trust problem; plaintext-in-memory; silicon signing keys; TEEs and attestation; "you can't legislate a memory dump"; open-weights-vs-open-source; the Qwen netconnect example; Kata containers; "stop asking your vendors the contracts of sovereignty" |
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Sovereign_AI_Inference_Own_Your_AI._Control_Your_Data.txt` | the four dimensions (control, trust, economics, continuity); "sovereignty is a system property"; the train-once/infer-constantly asymmetry; the five-layer stack; accelerator choice as implementation strategy; hermetic builds; freedom to choose, control, verify and operate |
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Scaling_AI_Inference_at_NxtGen_Indias_Best_Sovereign_Cloud_AI_Powerhouse.txt` | the ₹10 lakh → ₹5 lakh/month routing result; ~700 government controls; sovereign-cloud archetypes |
| `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` | the stacked cost ladder; "roughly a tenth of the cost"; cache reads ~10× cheaper; difficulty routing "often cuts total spend by half"; buy-vs-build guidance; the quantisation accuracy warning; "before you touch the model at all, exhaust the free wins" |
| `Agentic_AI_Infra_transcripts_2/Saurabh_Tiwary_-_From_Models_to_Agents_to_Discovery_Building_the_Full_Stack_of_A.txt` | the 10–100× agentic compute multiplier; 3.2 quadrillion tokens/month; agent identity/registry/gateway; the build/scale/govern/optimize platform decomposition |
| `ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/04-finops-and-token-economics.md` | the eight-layer cost stack; the 69%/28% anchor; reasoning and agent multipliers; provider caching discount ranges; batch and provisioned pricing; the FinOps discipline; structural cost decisions and their reported ranges; the cost anti-pattern table |
| `ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/07-cost-optimization-playbook.md` | the cascade tiering example (0.5B → 8B → frontier → thinking); SLM economics; distillation's long-tail failure; the DeepSeek V4 pricing floor |
| `ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/03-ai-gateways-and-model-routing.md` | the gateway as the attribution and budget-enforcement point |
| `ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/01-llm-infrastructure.md` | the multi-vendor capacity picture; "no senior architect designs a serious AI product around a single vendor anymore" |

Related blueprints: [T14 routing & gateways](../T14-routing-gateways/HLD.md) ·
[T15 autoscaling & SLO](../T15-autoscaling-slo/HLD.md) · [T17 observability & evals](../T17-observability-evals/HLD.md) ·
[T18 guardrails & security](../T18-guardrails-security/HLD.md)
