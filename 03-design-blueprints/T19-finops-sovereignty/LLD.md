# T19 — FinOps, Token Economics & Sovereignty: low-level design

> `T19` · **Transcript coverage:** primary · [HLD](HLD.md) · [Cheat sheet](../../00-cheat-sheets/T19-finops-sovereignty.md) · [Case study](../../01-case-studies/T19-finops-sovereignty.md) · [Interview bank](../../02-interview-questions/T19-finops-sovereignty.md) · [Runnable core](run.py) · [Production](production/README.md) · [Sequences](docs/SEQUENCES.md)

This document specifies the **interfaces, data structures, state and failure behaviour** of the five components that make this topic's design decide something: the cost stack, the lever algebra, the unit, the sovereignty posture and the attribution ledger. It is written against the runnable core in [`sim/`](sim/) and every number quoted here is produced by `python run.py`.

---

## 1. Module map

```
03-design-blueprints/T19-finops-sovereignty/
├── run.py                  six demonstrations + the summary            (executable spec)
├── sim/
│   ├── __init__.py         the organising claim, and the public surface
│   ├── stack.py            the bill, the discount curve, the asymptote, the write premium
│   ├── levers.py           the ladder, the algebra, the ordering, the ceiling
│   ├── units.py            the three normalisations, the metric families, the workload mix
│   ├── sovereignty.py      four dimensions, prerequisites, TEE arithmetic, residency, continuity
│   ├── attribution.py      tags, coverage, the governance ladder, reconciliation, caps
│   └── experiments.py      the six demonstrations, each ending in a FINDING
├── production/             real configs, reference-grade
└── docs/SEQUENCES.md       twelve sequences, each drawn so the failure is visible
```

| Module | One-line responsibility | The mistake it exists to prevent |
|---|---|---|
| `stack.py` | Convert a **discount** into a **share of the bill**, and bound the result | quoting a line reduction as a saving |
| `levers.py` | Rank, compose and bound a set of levers | double-counting a pricing lever, and ordering by the wrong axis |
| `units.py` | Hold the denominator decision, and the mix that moves under it | a falling headline against a rising per-outcome cost |
| `sovereignty.py` | Score a posture on four dimensions with prerequisites, and check its **floor** | averaging four dimensions into a slide |
| `attribution.py` | Attribute spend, and report coverage beside growth | a governance metric that degrades silently |

### 1.1 What is deliberately not here

* **No forecasting.** The mix's growth `factor` is an input, not a model. A forecast that is wrong in
  a parameter is wrong; a mix that is wrong in *structure* is wrong in the conclusion, and this design
  is about the structure.
* **No currency.** Every figure is a share of a bill, or in cents where the corpus states cents. A
  programme that converts to currency before comparing levers has introduced an FX assumption into a
  ratio.
* **No vendor price list.** Provider discounts enter as a single `discount` scalar **[R]**, because
  the arithmetic that matters (`discount × share_of_bill`, the write premium, the asymptote) is
  invariant to which provider supplies the number.
* **No per-model routing policy.** That is T14. This design consumes a router's *tier* as a lever
  target and does not implement the router.

### 1.2 Why the bill is modelled as layers with shares, not as a token count

The alternative model — tokens × $/token — cannot represent the thing this topic is about. Every
lever in the ladder acts on a **different part of the request**: caching acts on the prefix, output
caps on the completion, reasoning gating on a token class that never appears in the response, routing
on which model runs at all, batch and reserved pricing on the *whole* request at a different price.
A single token count collapses those into one number, and then `discount × tokens` produces a saving
that is arithmetically valid and operationally meaningless — it is how a 40× claim on a 2%-of-bill
line reads as a 40% programme.

Modelling the bill as **seven named layers that sum to 1.0**, with each lever declaring which layers
it can reach and what fraction of each, is what makes the two rank orders (§2.2) fall out. It is also
what makes §2.3's pricing conflict expressible: a pricing lever targets the layer `"*"`, meaning
"everything that is left", and two of them on the same traffic is one saving counted twice.

---

## 2. Data structures

### 2.1 `Layer` — one line of the bill (`stack.py`)

```python
@dataclass(frozen=True)
class Layer:
    name: str          # system_prompt | retrieved_context | conversation_memory | model_tier
                       # | output_length | reasoning_tokens | retry_overhead
    share: float       # share of the TOTAL bill; the seven shares sum to 1.00
    driver: str        # what makes the line as big as it is
    lever: str         # the lever class that reaches it
```

```python
BILL = (
    Layer("system_prompt",       0.3105, ...),   # pinned: 0.69 x 0.45  [R]
    Layer("retrieved_context",   0.0795, ...),
    Layer("conversation_memory", 0.0700, ...),
    Layer("model_tier",          0.2000, ...),
    Layer("output_length",       0.1600, ...),
    Layer("reasoning_tokens",    0.1500, ...),
    Layer("retry_overhead",      0.0300, ...),
)
BILL_TOTAL = sum(l.share for l in BILL)   # asserted == 1.0 in the test suite
LAYER_BY_NAME = {l.name: l for l in BILL}
```

**Why `share` is of the total bill and not of a category.** A lever's worth is
`reachable_share × discount × layer_share`, and the last factor must be a share of the thing the
programme is trying to reduce. A share-of-input-tokens model would make caching look 2.2× better than
it is, which is precisely the error §3.1 of the HLD documents.

**Why `system_prompt` is 0.3105 and the rest are round.** The prefix line is the corpus's own anchor
statistic (69% of input tokens, input being 45% of the bill **[R]**) and is derived rather than chosen.
The other six are `[D]` illustrative and are the first thing a real engagement replaces.

### 2.2 `Lever` — a claim, its reach, and its caveat (`levers.py`)

```python
@dataclass(frozen=True)
class Lever:
    name: str
    claim: str                # the corpus's own words [T]/[R]
    claim_multiple: float     # the largest multiple the claim implies, READ BY THE AUTHOR
    claim_notation: str       # "multiple" | "percentage" | "none"
    kind: str                 # "reduction" | "pricing"
    targets: tuple            # ((layer, share_of_layer, discount), ...); layer == ALL for pricing
    ease: int                 # 5 = free, 1 = a project
    caveat: str = ""
    conflicts: tuple = field(default_factory=tuple)   # lever names it cannot stack with
```

The two fields that carry the design:

**`claim_multiple` + `claim_notation` are separate on purpose.** The corpus states some levers as
multiples ("~10× cheaper", "5–40×") and others as percentages ("15–70%", "45–85%"), and three are not
quantified at all. An earlier version of this module parsed the prose with a regex; it read "≈95%
quality" — a *quality* figure quoted beside a cost one — as a 20× cost saving and ranked
`difficulty_routing` first. The fields exist so the reading is an explicit, reviewable act rather than
a parse, and `experiments.py` prints `(UNQUANTIFIED in corpus)` against the three that are.

**`conflicts` is declared, not inferred, and `apply_order()` enforces it.** Without it, `batch_lane`
and `reserved_capacity` both apply to the same tokens and the stack reports a 66.98% reduction that
exceeds its own ceiling — a number that looks like synergy and is arithmetic.

The declared set:

| Pair | Why they cannot both apply |
|---|---|
| `difficulty_routing` × `distillation` | both claim the easy path; distil it **or** route it |
| `batch_lane` × `reserved_capacity` | both reprice the same tokens; one slot |
| `difficulty_routing` × `reserved_capacity` | committing to capacity for traffic you intend to route away |

### 2.3 `Artifact` / `Workload` — the unit and the mix (`units.py`)

```python
@dataclass(frozen=True)
class Artifact:      # the corpus's Feb/Apr pair, verbatim  [T]
    period: str
    cost: float      # cents
    lines: float
    files: float

FEB = Artifact("Feb", 0.30, 630.0, 8.2)
APR = Artifact("Apr", 0.16,  91.0, 3.6)
```

```python
@dataclass(frozen=True)
class Workload:
    name: str
    calls: int
    cost_per_call: float
    note: str = ""

MIX = (Workload("chat",    990_000, 0.0030, "a chat turn is cents [R]"),
       Workload("agentic",  10_000, 0.1224, "tens of cents to several dollars [R]"))
```

**Why `Artifact` holds raw counts rather than ratios.** Every unit ratio in the topic is derivable from
these four numbers per row, and deriving them is the point: `normalised()` computes all three, and
`sign_disagreement` is `True` because one ratio is below 1.0 and two are above. A struct holding
pre-computed ratios could not have produced that flag.

**Why the workload carries `cost_per_call` and not a token count.** The mix exists to show that the
*mean* is the wrong basis for an order. A mean over a long tail is only computable from per-population
costs; token counts would reintroduce the single-unit collapse §1.2 rejects.

### 2.4 `Posture` — a sovereignty position and its claimed scores (`sovereignty.py`)

```python
@dataclass(frozen=True)
class Posture:
    name: str
    scores: dict        # dimension -> 0..1, as CLAIMED
    cost_index: float   # 1.00 = a managed API at list price
    note: str

DIMENSIONS = ("control", "trust", "economics", "continuity")

PREREQ = {"trust": "control",       # you cannot attest an environment you did not choose
          "continuity": "control"}  # you cannot switch vendors if the vendor picks the model
```

**`PREREQ` is data, not code in a scoring function, and that is the design decision.** A rule living
inside `effective()` is a rule nobody argues with; a rule living in a dict is a rule a reviewer can
change, cite, or veto. It is also the field that makes §5.2 of the HLD computable: without it,
`vendor_tee_lease` reports 0.85 trust honestly and the *claim* passes review.

**`scores` is deliberately named "claimed".** The struct does not know what a posture can evidence.
`effective()` is the function that caps each dimension by its prerequisite, and the pairing
(claimed, effective) is what every downstream table prints.

### 2.5 `Team` — an attribution unit (`attribution.py`)

```python
@dataclass(frozen=True)
class Team:
    name: str
    spend: float         # true spend
    tagged_share: float  # fraction carrying a usable tag
    growth: float        # YoY multiple
    note: str = ""
```

**The design point is the correlation between `tagged_share` and `growth`,** which is what makes a
coverage average misleading rather than merely incomplete. If the two were independent, §4.2 of the
HLD would not exist: coverage would be stable under growth. `untagged_share_of_growth()` scales each
team by `growth / min(growth)` so that nothing shrinks and the total bill grows, and the coverage
*ratio* is then invariant to the reference chosen while the untagged *spend* and its share of a
growing bill are not — which are the two numbers worth quoting.

**Why `growth` is a multiple and not a rate.** The corpus's figure is a multiple ("4.3×" in the
scenario), and the mix's growth story (§3.3 HLD) is a single multiplication. A rate would make the
unit of the scenario's own numbers ambiguous.

---

## 3. Interface contracts

### 3.1 `stack.py` — the four measurements

| Function | Signature | Contract |
|---|---|---|
| `percent_to_multiple(pct)` | `float → float` | a "90% cheaper" discount **is** a 10× multiple. Raises on `pct ≥ 1`. |
| `cache_read_price(discount)` | `float → float` | `1/discount`. The *only* place a discount becomes a price. |
| `prefix_line(cached_share, discount)` | `(float, float) → float` | cost of the prefix per call, in units where an uncached call is 1.00 |
| `caching_saving(discount, today, target, prefix_share)` | `→ dict` | returns **both** `line_reduction` and `blended_saving`, because they differ by 3.2× and only the second is bankable |
| `asymptotic_reduction(...)` | `→ dict` | the bound at `discount → ∞`. Returns `uncached_floor_*` so the reason is visible |
| `write_premium_break_even(discount, write_price)` | `→ dict` | `n > write_price / (1 − 1/discount)`; returns the fractional and ceiling values |
| `reasoning_amplification(visible, reasoning)` | `(int, int) → dict` | `total/visible`; the metric that hides it returns 1.0 by construction |

**Invariant enforced in tests:** `blended_saving == line_reduction × prefix_share` for every discount,
and `asymptotic_reduction()["blended_saving"]` is the supremum of `discount_sweep()` — no finite
discount may exceed the asymptote.

### 3.2 `levers.py` — the algebra

| Function | Contract |
|---|---|
| `achievable(lever, bill)` | the bankable saving as a share of the **total** bill; `kind == "pricing"` targets the whole remaining bill |
| `rank_by_achievable` / `rank_by_claim` | two orderings of one set; `ordering_disagreement()` returns both plus a `positions` map |
| `apply_order(order, levers, bill)` | three rules: no double-claiming a layer share; **pricing applies to what is LEFT**; a lever whose declared conflict is applied is **skipped and reported** |
| `ceiling(levers, bill)` | a **bound**, not a plan: disjoint shares per layer in discount order, then the single best pricing lever |
| `headline_check(...)` | `ceiling` against a headline; returns `reachable` as a boolean, and the top claim beside it |
| `order_saving_first` / `order_free_first` / `order_ease_first` | three defensible orderings; `ordering_comparison()` returns their spread |

**The three rules of `apply_order` are each load-bearing**, and each was added in response to a number
that did not survive scrutiny:

1. Without *disjoint shares*, two reduction levers on `model_tier` both claim 15% of it.
2. Without *pricing-applies-to-what-is-left*, a 50% batch discount is taken against the original bill
   after a 60% reduction has already happened — the saving exceeds what is left to save.
3. Without *conflict enforcement*, the stack (66.98%) exceeds the ceiling (60.6%), which is not a
   close call but a contradiction.

**Contract for `ceiling`:** it must be `≥` any achievable `apply_order` result. This is asserted in the
test suite for all three orderings, because it was false before rule 3 existed.

### 3.3 `units.py` — the denominator

| Function | Contract |
|---|---|
| `normalised(a, b)` | all three ratios from two rows, plus `sign_disagreement` |
| `unit_ranking(a, b)` | best-looking unit first; **the ranking is the incentive**, so it is printed rather than the raw ratios |
| `deflation_check(a, b)` | not "did the price fall" but "did the deliverable fall **faster**" |
| `metric_families()` | every metric with its denominator, how it is gamed, and what it cannot see |
| `tail_ratio(mix)` | the mean per call against the costly population — the basis for the ordering critique |
| `growth_scenario(factor, grows, mix)` | grow one population, re-read the shares |

**Contract for `metric_families()`:** every entry must name a `denominator` the measured team can move.
A metric whose denominator is chosen by finance is not a governance hazard and does not belong in the
table.

### 3.4 `sovereignty.py` — the posture and the arithmetic

| Function | Contract |
|---|---|
| `effective(posture)` | each dimension capped by `PREREQ`. Non-mutating; `scores` is never overwritten |
| `mean_vs_min(posture)` | both readings plus the gap, because the gap **is** the finding |
| `tee_uplift(penalty)` | `1/(1−p) − 1`. Raises outside `[0, 1)` |
| `attested_blended_uplift(penalty, share)` | the uplift on the *attested portion*; the caller applies it to the post-optimisation bill |
| `residency_score_leak(gap, noise)` | `P(in-region loses)` for a **score**; returns exactly 0.0 for a filter by construction |
| `residency_compliance(leak_rate)` | compliant **iff** `leak_rate == 0.0`; no tolerance, by design |
| `continuity(sources)` | returns the **minimum** over layers as `continuity`, with `mean` beside it and `binding_layer` named |
| `spectrum_position(name)` | `claimable_as_open_source` is true only when `reproducible` |

**Contract for `continuity`:** `continuity` is the min, never the mean. The struct returns `mean` for
comparison only; every downstream consumer that quotes a single number must quote the floor.

### 3.5 `attribution.py` — the ledger

| Function | Contract |
|---|---|
| `coverage(teams)` | spend-weighted **and** simple-mean, plus `worst_team` and `largest_untagged` |
| `untagged_share_of_growth(teams)` | scales by `growth / min(growth)`; returns coverage before/after, untagged spend before/after, and total-bill growth |
| `governance_ladder()` | four stages with the *precondition* each needs and the failure each introduces |
| `shadow_gap(gateway, invoice)` | `residual`, `shadow_share`, and a `visible` boolean at a 1% tolerance |
| `cap_from_mean_vs_tail(mean, p99, ceiling)` | a verdict, not a number: the ceiling's source statistic is the decision |

**Contract for `shadow_gap`:** a positive residual is spend the governance plane cannot see. The
function does not attempt to attribute it — an unattributed residual is exactly the finding.

---

## 4. State — where a cost decision lives

### 4.1 The five state locations

| # | State | Lives in | Lifetime | Must be |
|---|---|---|---|---|
| 1 | **The unit definition** | the policy store, versioned with the scenario | the programme | written, owned, and re-derivable from raw counts |
| 2 | **Bill shares** | the policy store | per planning cycle | replaced wholesale, never edited in place |
| 3 | **Lever parameters** (`share`, `discount`, `ease`, `conflicts`) | the lever registry | per review | reviewed against the claim they encode |
| 4 | **Tag coverage**, per team | the metering ledger | continuous | reported beside growth, never aggregated alone |
| 5 | **Pricing commitments** (reserved, batch lanes) | the procurement system, **mirrored** into the policy store | contract term | mirrored, because a commitment is a cost even when unused |

Location 5 is the one teams miss. A reserved-capacity commitment is a cost that exists whether or not
the traffic does, so a policy store that models only *usage* will report a saving from routing traffic
away from capacity that has already been paid for.

### 4.2 The decision order, and why the common order is wrong

```
CORRECT:   unit  ─▶  attribution  ─▶  posture (floor ≥ threshold)  ─▶  pricing lever  ─▶  reduction levers
COMMON:    pricing ─▶  reduction  ─▶  dashboard  ─▶  unit (never)  ─▶  posture (never)
```

The common order is wrong in three specific ways, and each maps to a failure in HLD §8.1:

* **Pricing first** fills the single pricing slot before the reduction set is known, so a conflict
  drops the larger lever (#4, worth 15.8 points).
* **Dashboard before unit** means every number on it has a denominator somebody else chose (#1).
* **Posture never** means the cheapest posture is chosen and then found unusable for the traffic that
  justified the programme (#9, #10).

### 4.3 The state that must be asserted, not assumed

Three values in this design are **claims that pass review unless a rule is applied**:

| Value | Claimed | Computed | Rule |
|---|---|---|---|
| a posture's trust score | 0.85 (`vendor_tee_lease`) | **0.50** | `PREREQ["trust"] = "control"` |
| a lever's saving | the discount | `discount × share_of_bill` | `achievable()` |
| "open source" | weights published | `reproducible` | `spectrum_position()` |

All three are the same design pattern: **a value that is stated by a party with an interest in it, and
must be recomputed from a rule before it is used.** This is the LLD-level statement of the topic's
organising claim, and it is why each of the three lives behind a function rather than in a literal.

---

## 5. Failure behaviour

### 5.1 The four failures that matter, and their signatures

| Failure | Signature | Detected by | Silent? |
|---|---|---|---|
| **Wrong unit** | headline falls, per-outcome cost rises | `sign_disagreement` in `normalised()` | **yes** |
| **Coverage drift** | global coverage falls under growth, no threshold crossed | `untagged_share_of_growth()["coverage_delta"]` | **yes** |
| **Trust claimed above control** | posture table satisfied; attestation proves the wrong environment | `claimed_vs_effective()["capped"]` | **yes** |
| **Pricing double-count** | stacked reduction exceeds the ceiling | `ceiling() ≥ apply_order()` assertion | no — it is arithmetic |

Three of the four are silent, and the reason is common: **each is a comparison the design must make
against a quantity that is not on the dashboard.** The wrong unit needs the companion metric; coverage
drift needs the growth rate beside the coverage; the trust cap needs the prerequisite rule. There is no
threshold to alert on, because nothing crossed one.

### 5.2 The pricing double-count, worked

Without conflict enforcement, from the same lever set:

```
apply_order(saving_first, all levers)   = 66.98%   >   ceiling(all levers) = 60.61%
```

The stack exceeding its own bound is the contradiction, and it is not subtle once both numbers exist.
The fix is one line in `apply_order` — track `applied`, skip a lever whose declared conflict is in it,
and **report the skip in the step record** rather than dropping it. The reporting matters: a silently
skipped lever is a lever a reviewer will later "re-add", and the second addition is the one that
double-counts.

### 5.3 The residency leak, as arithmetic rather than policy

```
score-based router, gap 1.0 / noise 1.0   →  leak rate 0.1587  →  1,586.6 leaks per 10,000
the same routing as a FILTER              →  0 leaks
```

`residency_score_leak()` returns `1 − Φ(gap / noise)`. The function has no "acceptable" threshold and
no tolerance parameter, and that is deliberate: `residency_compliance()` declares compliant **iff** the
rate is exactly zero. The design encodes the legal property in the type of the answer rather than in a
policy document, so a future engineer cannot configure an acceptable leak rate without noticing that
the function has no parameter for one.

### 5.4 The failure-domain contract

| Component | Fails | Behaviour | Recovery |
|---|---|---|---|
| Gateway | unreachable | **fail to the last-known-good policy**, never to "no policy": an unattributed request is worse than a rejected one | T14's gateway semantics; tags are re-emitted |
| Prefix store | cold | every request pays full prefix price; bill rises with **no metric changing** | the read/write ratio per prefix is the detector |
| Metering | lagging | coverage *appears* to fall; the finding must distinguish lag from drift | compare against provider invoices; a lagging ledger and a drifting one differ in the reconciliation residual |
| Attestation | fails at boot | **refuse to serve** the traffic that requires it | degrade to the unattested posture for non-regulated traffic only; never silently |
| Policy store | unavailable | serve with the cached policy, **and freeze the pricing commitments** | a lever applied against a stale bill is the double-count in §5.2 |

---

## 6. Concurrency and determinism

### 6.1 Determinism

Every module here is pure: given the same `BILL`, `LEVERS`, `MIX` and `TEAMS`, `run.py` produces
byte-identical output. There is no clock, no random source and no I/O inside `sim/`. The only
non-obvious source of nondeterminism would be **set iteration order**, so the one place a set is used
(`applied` in `apply_order`) is never iterated to produce output — `applied` is sorted before it is
returned.

### 6.2 The one place ordering is load-bearing

`apply_order` is order-dependent **by construction**, and this is the design rather than a limitation:
the topic's finding is that the ordering is worth 4.15 points of the bill. A version of the function
that sorted its inputs internally would destroy the finding. The order-dependence is therefore
documented in the signature, asserted in the test suite (the three orderings must differ), and printed
by `ordering_comparison()`.

`ceiling()` is deliberately **order-independent** — it sorts levers by discount within each layer — so
that it is a true bound and can be compared against any ordering.

---

## 7. Interfaces

### 7.1 The cost event — the only durable output

Every attributed call emits one record. This is the schema the metering plane is built around, and the
four fields in **bold** are the ones a general-purpose observability pipeline omits.

```json
{
  "schema": "t19.cost_event.v1",
  "ts": "2026-09-27T10:14:02.113Z",
  "request_id": "req_01J...",
  "tenant": "acme",
  "team": "agent_pilot",
  "feature": "support_agent",

  "route": { "tier": "mid", "residency": "in-region", "filtered_out": ["frontier", "out-of-region"] },

  "tokens": {
    "input_uncached": 1840,
    "input_cached_read": 61200,
    "input_cache_write": 61200,
    "output_visible": 300,
    "output_reasoning": 2000
  },

  "unit": { "name": "per_resolved_case", "numerator": 0.0042, "denominator_label": "cases" },

  "pricing": { "lane": "reserved", "commitment_id": "rc-2026-q3", "batch_eligible": false },

  "sovereignty": {
    "attested": true,
    "attestation_id": "att_...",
    "attested_share_of_traffic": 0.40,
    "throughput_penalty": 0.20,
    "cost_uplift_applied": 0.25
  },

  "cost": { "input": 0.0082, "output": 0.0104, "uplift": 0.0047, "total": 0.0233 },
  "tags_complete": true
}
```

The four load-bearing fields:

* **`route.filtered_out`** — the residency decision is a filter, so the record must show what was
  *removed*, not merely where the request went. A score-based router cannot emit this field, which is
  the point (§5.3).
* **`tokens.output_reasoning`** — the token class that is billed and invisible. Without it,
  `tokens_visible` is a denominator that makes the call look 7.67× cheaper than it is.
* **`unit`** — the denominator travels **with the event**, not in a dashboard definition. This is what
  makes the unit memo enforceable rather than aspirational.
* **`sovereignty.cost_uplift_applied`** — the uplift stored per event, so that a post-hoc "what did
  sovereignty cost us" question is answerable from the ledger rather than from a spreadsheet.

### 7.2 The metrics contract

| # | Metric | Reads | Alert when |
|---|---|---|---|
| 1 | `cost_per_unit{unit=...}` | one line per unit | **never directly** — see #2 |
| 2 | `unit_sign_disagreement` | whether the units disagree in sign | the flag is `true` for two consecutive periods |
| 3 | `cache_read_write_ratio{prefix=...}` | per prefix | any prefix below **2** for a full period |
| 4 | `prefix_cached_share` | the 28% → 90% move **[R]** | below target; the asymptote caps the gain at 26.7% |
| 5 | `lever_banked_saving{lever=...}` | `discount × share_of_bill` per lever | a lever whose banked saving is below its claim by > 5 pts |
| 6 | `ladder_ceiling_gap` | the ceiling against the committed target | the gap widens — i.e. the target is drifting from the bound |
| 7 | `pricing_lever_conflicts_skipped` | the skipped count from `apply_order` | **any** non-zero value in a deployed plan |
| 8 | `attribution_coverage{team=...}` **beside** `team_growth{team=...}` | the pair, never alone | coverage < 0.90 **and** growth > 2× |
| 9 | `gateway_invoice_residual` | provider invoices − gateway spend | above **1%** (§4.4 HLD) |
| 10 | `runaway_terminations{source=mean\|tail}` | which statistic the cap came from | any termination from a mean-derived cap |
| 11 | `sovereignty_floor` | `min(effective(posture))` | below the written threshold, **not** below a mean |
| 12 | `claimed_vs_effective_gap{posture=...}` | `capped` dimensions | any posture with a non-empty `capped` in production |
| 13 | `residency_leak_rate` | the filter's invariant | **any** non-zero value is a defect, not a warning |
| 14 | `continuity_floor{layer=...}` | the **minimum** over the four layers | floor at 0 while the mean is ≥ 0.70 (the §5.6 signature) |
| 15 | `reasoning_share_of_output_billing` | reasoning ÷ (visible + reasoning) | a rising share with a flat visible-token metric |

Metrics 2, 7, 8, 12, 13 and 14 are the six that no general-purpose cost dashboard emits, and each of
them is the *only* signal for one of HLD §8.1's silent failures. Metric 1 carries the explicit
instruction **never alert on it directly**: it is the metric the whole topic warns about, and it is
present so that #2 has something to compare against.

---

## 8. Configuration surface

### 8.1 The bill (`production/bill.yaml`)

```yaml
# production/bill.yaml
# The decomposition IS the model. Every lever's worth is computed against these shares.
# Replace all six illustrative lines with measured values BEFORE ranking any lever.  [D]
version: 2026-Q3
assert_sums_to: 1.00          # enforced at load; a bill that does not sum is a modelling error

layers:
  - name: system_prompt
    share: 0.3105             # DERIVED [R]: 0.69 of input tokens x 0.45 input share of bill
    derived: true             # a derived line is re-derived at load, never trusted as typed
  - name: model_tier
    share: 0.2000
    derived: false
  - name: output_length
    share: 0.1600
  - name: reasoning_tokens
    share: 0.1500
  - name: retrieved_context
    share: 0.0795
  - name: conversation_memory
    share: 0.0700
  - name: retry_overhead
    share: 0.0300

cache:
  read_discount: 10.0         # [T] "roughly ten times cheaper"; providers 2x-10x equivalent [R]
  write_premium: 1.25         # [R] Anthropic-class write price
  cached_share_today: 0.28    # [R]
  cached_share_target: 0.90
  min_read_write_ratio: 2.0   # break-even is 1.39; operate at 2, and disable below it
```

`derived: true` is the field that keeps the model honest. A derived line is recomputed from its
inputs at load, so the 31.05% cannot drift away from the 69%/45% that produce it.

### 8.2 The lever registry (`production/levers.yaml`)

```yaml
# production/levers.yaml
# ORDER OF OPERATIONS, and it is not negotiable:
#   1. decide the PRICING lever (there is one slot; see conflicts)
#   2. sort the REDUCTION levers by `banked`, not by `claim`
#   3. quote the CEILING, not the headline

# MEASURED [D], this bill:
#   ceiling (all levers, disjoint shares, best pricing)      60.61%   = 2.54x
#   headline [T]                                             90.00%   = 10.00x   GAP 29.39 pts
#   best ordering (saving_first)                             59.30%
#   worst ordering (ease_first)                              55.17%   SPREAD 4.15 pts
#
#   claim rank -> banked rank: prompt_caching #3 -> #1 ; distillation #1 -> #8
#
# pricing_slot: ONE lever per traffic slice. batch_lane and reserved_capacity are one saving.

pricing:
  - name: reserved_capacity
    share: 0.45
    discount: 0.35            # 15-70% [R]
    chosen: true              # the value decision: 15.75 pts vs batch_lane's 10.00 pts
  - name: batch_lane
    share: 0.20
    discount: 0.50
    chosen: false             # skipped by conflict, and the skip is REPORTED

reduction:
  - name: prompt_caching
    target: [[system_prompt, 1.00, 0.746]]
    ease: 5
    banked: 0.2316            # #1 by banked, #3 by claim
    caveat: "only a STABLE prefix; a nonce or a reordered tool list invalidates everything"
  - name: difficulty_routing
    target: [[model_tier, 0.55, 0.50]]
    ease: 3
    banked: 0.0550
    conflicts: [distillation]
  - name: distillation
    target: [[model_tier, 0.15, 0.80]]
    ease: 1
    banked: 0.0240            # #1 by CLAIM (40x) and #8 by BANKED -- the whole point
    conflicts: [difficulty_routing]
  - name: reasoning_gating
    target: [[reasoning_tokens, 1.00, 0.60]]
    ease: 4
    banked: 0.0900
    eval_on: tasks_gated_off   # measure quality where you turned thinking OFF
```

### 8.3 The sovereignty policy (`production/sovereignty.yaml`)

```yaml
# production/sovereignty.yaml
# The FLOOR is the threshold, not the mean.  [D]
#   MEASURED: compliance_only   mean 0.438  floor 0.100   gap 0.338
#             api_with_dpa      mean 0.288  floor 0.050   gap 0.238
#             vendor_tee_lease  claims 0.85 trust, evidences 0.50 (control caps trust)

prerequisites:                 # data, not code -- so a reviewer can veto a rule
  trust: control
  continuity: control

thresholds:
  floor_min: 0.60              # the posture fails if ANY dimension is below this
  mean_min: null               # deliberately null: a mean threshold is the failure mode

residency:
  mode: filter                 # NOT "score". A score leaks; a filter has no parameter for leak.
  filter_field: allowed_regions
  leak_rate_allowed: 0.0       # exactly zero. enforced in test, not monitored in prod.

continuity:
  reading: minimum             # not mean
  layers: [accelerator, model_family, serving_engine, cloud_region]
  min_options_per_layer: 2     # 1 is not continuity
  note: "the cheapest fix is usually a second SERVING ENGINE on the same weights"

attestation:
  attested_share: 0.40         # attest the regulated slice only
  throughput_penalty: 0.20
  cost_uplift: 0.25            # = 1/(1-0.20) - 1 . NEVER enter the penalty here.
  open_weights_requires: [weights, training_code, training_data, reproducible]
```

`cost_uplift` is written as a **separate key from `throughput_penalty`** on purpose. Every business
case error in this area is entering the penalty where the uplift belongs, undersizing the budget by
about a fifth. Two keys force the conversion to be an act.

### 8.4 The attribution policy (`production/attribution.yaml`)

```yaml
# production/attribution.yaml
# MEASURED [D]: coverage 90.46% -> 82.43% under one growth cycle; untagged share 9.54% -> 17.57%
# Nobody tagged worse. The untagged spend is where the GROWTH is.

required_tags: [team, feature, tenant, model, route, environment]
coverage:
  min_spend_weighted: 0.90
  report_beside: team_growth   # NON-OPTIONAL. Coverage alone is the failure mode.
governance:
  stage: showback              # showback -> chargeback -> caps, in that order [R]
  chargeback_requires: trusted_tags
  reconciliation:
    enabled: true
    cadence: hourly            # not quarterly -- this is the only off-gateway detector
    alert_residual_pct: 0.01
caps:
  per_run_ceiling: 5.00
  derive_from: p99_task_cost   # NOT the mean. A mean-derived cap kills legitimate long tasks.
```

---

## 9. Test strategy

### 9.1 The five tests that are actually regression tests for this design

| # | Test | Asserts | The defect it would have caught |
|---|---|---|---|
| 1 | `ceiling ≥ apply_order` for **all three** orderings | the bound holds | the 66.98%-vs-60.6% contradiction |
| 2 | `BILL` sums to 1.00 | the model is closed | a lever's worth computed against an open bill |
| 3 | `blended_saving == line_reduction × prefix_share` for every discount | the two units are distinguished | a discount quoted as a saving |
| 4 | the three orderings **differ**, and `ease_first` is the minimum | order-dependence is a finding, not a bug | a "helpful" sort inside `apply_order` |
| 5 | `claimed_vs_effective(posture)["capped"]` is non-empty for at least one posture | the prerequisite rule bites | `PREREQ` present but never applied |

Test 5 is unusual and deliberate. A rule that no test forces to fire can silently stop applying — the
first version of `sovereignty.py` had `PREREQ` populated and no posture whose trust exceeded its
control, so the cap never triggered and the rule was invisible in every table. Adding the
`vendor_tee_lease` posture was the fix, and the test is the guard against its removal.

### 9.2 Property tests worth their cost

* **Monotonicity:** `caching_saving(d)` increases with `d` and is bounded above by
  `asymptotic_reduction()`. Catches a discount curve that has been "improved" past its own asymptote.
* **Sign stability:** for any `Artifact` pair, `normalised()` returns at least one ratio above and one
  below 1.0 whenever `lines_ratio < cost_ratio`. This is the Feb/Apr finding as a property, and it
  would have caught an early version that computed `per_line` the wrong way round.
* **Conflict symmetry:** `conflict_pairs()` returns each pair once, from either direction.
* **No tolerance:** `residency_compliance(r)` is compliant only at exactly `0.0`, for a swept range of
  `r`. Guards the legal property against a future "practical" threshold.

### 9.3 What is not tested, and why

* **Vendor prices.** They change; the arithmetic does not.
* **Forecast accuracy.** The mix's growth factor is an input (§1.1).
* **The `[D]` shares.** They are illustrative. Testing them would give them a false authority; the
  test suite asserts only that they sum to 1.00.

---

## 10. Build order

The order is the design's own claim (§4.2): a unit before a measurement, an attribution before an
optimisation, a posture floor before a posture cost.

| Step | Deliverable | Needs | Why first |
|---|---|---|---|
| 1 | **The unit memo** | finance + the owning team | every later number is meaningless without it; it is the only artifact that cannot be back-filled |
| 2 | **`bill.yaml`** with measured shares | the metering ledger (a day of tagged calls) | the lever ranks cannot be computed without it |
| 3 | **The tag contract** | platform | attribution is the precondition **[R]** |
| 4 | **`sovereignty.yaml` floor** | legal + security | a floor decided after the spend plan is a floor negotiated down |
| 5 | **The pricing lever decision** | procurement + the mix | one slot; decide on value, before the reduction plan is fixed |
| 6 | **Reduction levers, ranked by `banked`** | steps 2 and 5 | the ordering is worth 4.15 points |
| 7 | **The ceiling, quoted** | step 6 | replaces the headline as the committed target |
| 8 | **Per-prefix caching rollout** | the prefix store's read/write accounting | the only lever in the ladder that can raise the bill |
| 9 | **Reconciliation** | provider invoices | must exist **before** chargeback, not after |
| 10 | **Per-run caps from the tail** | step 9 | the last rung of the governance ladder |

Steps 1–4 involve no engineering, and they are the four that decide whether steps 5–10 are measuring
the right things. Every one of them is skipped by teams that start at step 6.

---

## Sources

| File under `refs/` | Used for |
|---|---|
| `LLMOps_Agentic_AIOps_The_Hands-On_Playlist_2026_transcripts/Cut_LLM_Cost_Latency_KV_Cache_Batching_Quantization_vLLM.txt` | the lever ladder and its ordering instruction; "exhaust the free wins"; "roughly a tenth of the cost"; cache reads ~10× cheaper; the stable-prefix caveat; difficulty routing "often cuts total spend by half"; the quantisation accuracy warning; "optimization without measurement is just guessing" |
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/The_Token_Raj_Rethinking_the_AI_Inference_Stack.txt` | the Feb→Apr artifact study; "tokens per output" as the metric to change; the three-way trust problem; attestation; open-weights-vs-open-source; Kata containers |
| `vLLM_Inference_Meetup_Bengaluru_2026_transcripts/Sovereign_AI_Inference_Own_Your_AI._Control_Your_Data.txt` | the four dimensions; the system-property framing; the TEE trade-off sentence; accelerator choice; freedom to choose, control, verify and operate |
| `Agentic_AI_Infra_transcripts_2/Saurabh_Tiwary_-_From_Models_to_Agents_to_Discovery_Building_the_Full_Stack_of_A.txt` | the 10–100× agentic multiplier behind the mix's tail population |
| `ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/04-finops-and-token-economics.md` | the eight-layer decomposition; the 69%/28% anchor; provider caching discount ranges; batch and provisioned pricing; the FinOps discipline; the attribution-first rule; showback-before-chargeback; the runaway-agent case |
| `ai-system-design-guide-main/ai-system-design-guide-main/04-inference-optimization/07-cost-optimization-playbook.md` | cascade tiering and its 45–85% range; distillation's long-tail failure; the reasoning multiplier range |
| `ai-system-design-guide-main/ai-system-design-guide-main/11-infrastructure-and-mlops/03-ai-gateways-and-model-routing.md` | the gateway as budget-enforcement and attribution point |

Related: [HLD](HLD.md) · [Production configs](production/README.md) · [Sequence diagrams](docs/SEQUENCES.md) · [Runnable core](run.py) · [Case study](../../01-case-studies/T19-finops-sovereignty.md)
