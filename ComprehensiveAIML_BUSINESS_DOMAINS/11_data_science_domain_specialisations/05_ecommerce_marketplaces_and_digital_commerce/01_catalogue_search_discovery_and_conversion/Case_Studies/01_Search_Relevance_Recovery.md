---
case_id: "CASE-05-01-001"
case_title: "Search relevance recovery after an offline-NDCG rollout"
case_version: "2.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Digital commerce / catalogue search and ranking"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["rollout-regression", "human-in-the-loop"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "ranking quality manager"
source_index: "INDEX.md"
review_trigger: "judgement pool composition, query-mix shift, ranking model version, or conversion definition changes"
---

# Search relevance recovery after an offline-NDCG rollout

## 1. The One-Line Framing

> Given one search session record (query, locale, ranked result list, and the product
> availability snapshot at request time), decide whether to keep the incumbent ranker,
> apply a bounded re-ranking correction, or roll the candidate ranker back for a named
> query slice, for the ranking quality manager, to protect completed-order conversion in
> availability-constrained slices without losing the aggregate gain, subject to
> judgement-pool coverage, review capacity, and a reversible rollout fence, compared with
> the incumbent lexical ranker that is still serving all traffic.

**Evidence declaration.** This is an ILLUSTRATIVE design exercise built on constructed
numbers. It reports no company deployment, no measured relevance gain, and no incident.
Every figure below is either labelled `ILLUSTRATIVE` (input) or `DERIVED` (an arithmetic
consequence of those illustrative inputs, recomputed in this document).

The obvious framing — "the offline NDCG@10 went up, so ship it" — fails for four separate
reasons that this case keeps apart:

1. **Pooling mismatch.** Judgement sets are built from the incumbent ranker's top results.
   A candidate that retrieves documents the pool never contained is scored as if those
   documents were irrelevant, so the offline metric is biased against exactly the
   retrieval it was meant to reward.
2. **Objective mismatch.** NDCG grades ordered relevance labels. The business outcome is
   a completed, kept, profitable order. A result that is relevant but out of stock, or
   relevant but shipping beyond the promise date, contributes to NDCG and subtracts from
   the business outcome.
3. **Aggregation mismatch.** A uniform mean over queries lets a large improvement on head
   queries hide a severe regression on a small tail slice, because the slice carries under
   10% of sessions.
4. **Reversibility mismatch.** A ranker that changes candidate generation is not a
   configuration flag; rolling back needs the index, the analyser version, and the
   feature snapshot to be restorable together.

**Takeaway:** the decision the owner actually faces is not "is the new ranker better on
average" but "which traffic is safe to move, and what is the smallest reversible step that
proves it".

**Primary pattern:** `evaluation-mismatch`. **Secondary patterns:**
`rollout-regression` (the release degraded a live guardrail while the primary metric
improved) and `human-in-the-loop` (a bounded review queue adjudicates the disputed slice).

**Decision frequency.** Every ranking release, every analyser change, and every
relevance-model retrain. In a busy catalogue this is a weekly-to-fortnightly decision, so
the cost of a wrong rule is paid repeatedly rather than once.

## 2. Context & Background

### 2.1 What the system does

A shopper types a query. A query-understanding stage normalises it (lowercasing,
locale-specific tokenisation, spelling correction, unit normalisation, and optional
attribute extraction such as colour or size). A retrieval stage produces candidates from
an inverted index plus a vector index. A ranking stage orders them with a learning-to-rank
model over lexical, behavioural, and availability features. A policy stage removes
out-of-stock, restricted, and non-compliant items and applies business rules. The page is
rendered and the session either converts or does not.

The decision in this case sits between ranking and policy: which ordering is allowed to
serve a given query slice.

### 2.2 Workflow table

| Stage | What happens | Artifact produced | Who owns it | Failure if this stage is wrong |
|---|---|---|---|---|
| Query capture | Raw query string, locale, device, session ID | `query_event` with event time | Search platform | Locale-specific normalisation applied to the wrong language |
| Query understanding | Tokenise, correct, expand, extract attributes | `parsed_query` with analyser version | Search relevance | Expansion silently drops a required term |
| Retrieval | Lexical + vector candidate generation | Candidate set with retrieval scores | Search platform | Candidate set excludes the item the shopper wanted |
| Ranking | LTR ordering over features | Ranked list with model version | Ranking quality | Ordering optimises a proxy that harms availability |
| Policy | Availability, compliance, restriction filters | Final list plus exclusion reasons | Merchandising ops | Stock-out or restricted item shown and cancelled later |
| Presentation | Snippets, price, delivery promise | Rendered page | Storefront | Promise text contradicts inventory truth |
| Outcome | Impression, click, add-to-cart, order, cancellation, return | Funnel events | Analytics | Denominators disagree across stages |

### 2.3 Stakeholders

| Role | Interest | Authority | Failure they experience |
|---|---|---|---|
| Ranking quality manager | Relevance quality across slices | Approve, reject, or defer a ranker release | Cannot defend a release that hurt a slice |
| Search platform engineer | Latency, availability, index freshness | Owns serving capacity and rollback mechanics | Paged for a regression they did not cause |
| Merchandising operations | Assortment coverage and stock truth | Owns the availability feed contract | Blamed for showing unavailable items |
| Category manager (affected slice) | Revenue in their category | Can escalate a slice regression | Sees fewer orders without an explanation |
| Analytics / experimentation | Valid measurement | Signs off on the test design | Asked to certify a broken assignment |
| Customer | Finds a usable product | None inside the system | Gets a relevant but unorderable result |

### 2.4 Baseline and why it persists

The incumbent is a lexical ranker with hand-tuned field boosts (title, brand, category),
a BM25-style term-weighting core, and a small set of deterministic business rules. Its
documented weaknesses are real but bounded:

- It cannot generalise across paraphrases ("running shoes" versus "jogging footwear"), so
  recall on descriptive queries is limited.
- Its boosts are global, so a category with unusual vocabulary is mis-served.
- It has no learned availability feature, so it cannot prefer a comparable in-stock item.

It persists because it is *predictable*: an engineer can explain any ordering by reading
the boosts, rollback is a configuration change, and no slice has ever regressed without a
matching configuration diff. A learned ranker trades that explainability for lift, and the
organisation has not yet built the per-slice instrumentation that would make the trade
safe.

### 2.5 Scale (illustrative, not asserted)

| Quantity | Illustrative value | Unit | Why this value matters |
|---|---|---|---|
| Search sessions | 2,000,000 | sessions/day | Denominator for the aggregate test |
| Queries with results | 96.5 | % of sessions | The rest hit zero-result pages and must be excluded from ranking metrics |
| Disputed slice share | 9.5 | % of sessions | Small enough to hide inside a mean |
| Slice sessions | 190,000 | sessions/day | 0.095 × 2,000,000 |
| Judged query pool | 4,800 | queries | Pool comes from incumbent top-10 only |
| Human relevance reviewers | 6 | reviewers | Caps how much new pool can be built per week |
| Reviews per reviewer-hour | 180 | judgements/hour | 4-grade scale, 1 query per 30 seconds |
| Rollback budget | 10 | minutes | Business tolerance for a bad ordering |

These are teaching inputs. Section 6 shows which conclusions survive a change in each.

### 2.6 Prior attempts

Two prior ranker releases are described here as *considered context*, not as measured
history: a field-boost retune and an embedding-based candidate generator. Neither release
built per-slice conversion instrumentation, which is why this case treats slice-level
measurement as a prerequisite rather than an afterthought. No claim is made about what
those releases achieved.

## 3. The Business Question

### 3.1 Decision contract

| Field | Content |
|---|---|
| Question | Should the candidate ranker replace the incumbent for all search traffic, for one named query slice only, or not yet — and if it replaces anything, what reversible evidence must exist first? |
| Decision owner | Ranking quality manager, with platform engineering concurrence on rollback feasibility |
| Trigger and deadline | Candidate ranker completes offline evaluation; the decision is needed before the next merchandising cycle, within 5 business days |
| Available actions | (a) retain incumbent everywhere; (b) serve candidate on one slice behind a fence; (c) apply a bounded re-ranking correction to the incumbent; (d) defer and build pool coverage; (e) roll back candidate if already partially live |
| Value and downside | Upside: aggregate completed-order conversion; downside: a slice regression that is invisible in the mean, plus reputational and rollback cost |
| Objective | Maximise completed-order conversion per eligible search session, subject to no slice regressing beyond its guardrail |
| Hard constraints | Point-in-time availability features only; ≤10-minute rollback; judgement-pool coverage disclosure; review capacity of 6 reviewers |
| Scope | One locale, one device class, 28-day window, queries with at least one result |
| Out of scope | Pricing, paid placement policy, legal compliance determinations, and catalogue onboarding |

### 3.2 Why the proxy and the objective diverge

NDCG@10 measures whether graded relevance labels appear in a good order. Three concrete
divergences:

1. **Availability.** A judged-relevant item that is out of stock still earns its full
   graded credit while producing a cancelled order. The metric cannot see the difference.
2. **Substitution.** Two items with the same label may differ in whether the shopper
   accepts a substitute. A ranker that promotes the substitutable one looks identical
   offline and better online.
3. **Position bias.** Labels are collected on presentations produced by the incumbent. A
   candidate that would place a different item first never gets that item judged, so the
   measured value is conditional on the incumbent's exposure pattern.

The guardrail that keeps the two aligned is *slice-level completed-order conversion*,
measured on the same session population with the same denominator. Where that guardrail
cannot be measured, the correct action is deferral, not optimism.

### 3.3 What an expensive wrong decision looks like

- **False ship.** Candidate improves the mean by 0.06 percentage points while the slice
  loses 17% of its conversion. At 190,000 slice sessions/day that is a large daily order
  loss concentrated in a category with its own revenue owner.
- **False hold.** Candidate is genuinely better but is rejected because the offline pool
  was built from incumbent results. The organisation loses a quarter of a year of
  improvement, and the next release repeats the same bias.
- **Unrecoverable ship.** Candidate ships with a new index format, rollback needs a
  reindex, and the 10-minute budget is exceeded — turning a reversible experiment into an
  outage.

**Takeaway:** the decision is a choice among evidence-gated partial moves, not a binary
ship/hold vote.

## 4. Assumptions & Constraints

| ID | Assumption / constraint | Type | Verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Availability state used at ranking time was knowable at that instant | Data | Join rank-time timestamp against inventory event log | unverified | Learned ranker is optimising on future stock | Drop availability feature, retrain | Inventory feed contract change / data owner |
| A-02 | Judgement labels are complete for the candidate's candidate set | Data | Measure pool recall on a stratified re-pool sample | violated (pool built from incumbent top-10) | NDCG biased against new retrieval | Report pooled and unpooled metrics separately | Any candidate-generation change / relevance lead |
| A-03 | Completed-order conversion is stable absent ranking change | Business | Pre-period variance and seasonality check | unverified | Attributing seasonality to the ranker | Use a held-out slice as control | Promotional calendar change / analytics |
| A-04 | Session-level assignment is sticky for the whole session | Systems | Check assignment ID persistence across requests | unverified | Within-session mixing contaminates both arms | Session-scoped bucketing key | Assignment-service change / platform |
| A-05 | The candidate's latency is within the serving budget | Systems | Load test at peak query rate | unverified | Timeout-driven fallback silently reverts to incumbent | Keep candidate behind a timeout-tolerant path | Model or index size change / platform |
| C-01 | Rollback within 10 minutes | Operations | Timed rollback drill on a staging fence | unverified | Bad ordering cannot be withdrawn | Pre-build parallel index and dual-read | Index format change / platform |
| C-02 | Reviewer capacity is 6 reviewers × 180 judgements/hour | Operations | Rota and throughput log | assumed | Pool cannot be extended in time | Restrict the slice being judged | Headcount change / ops lead |
| C-03 | Only one locale and device class in scope | Scope | Query log filter audit | confirmed by construction | Results do not transfer to other locales | Re-run before extending | Locale launch / relevance lead |
| C-04 | No paid-placement change during the window | Business | Merchandising change log | assumed | Revenue mixing confounds conversion | Freeze placements during the test | Any placement policy change / category manager |
| A-06 | Zero-result sessions are excluded, not counted as failures | Measurement | Compare denominators across stages | unverified | Conversion rate computed on two different populations | Fix denominator to queries-with-results | Metric definition review / analytics |

**Two assumptions most likely to reverse the recommendation.** `A-02` (pool
completeness) and `A-03` (conversion stability). The first attacks the evidence used to
rank the candidate; the second attacks the evidence used to judge it in production. Both
are cheap to test and expensive to ignore.

**Explicitly not assumed.** That offline lift predicts online lift in magnitude. The case
treats offline NDCG as a *screening* signal only; the decision rule requires a
slice-level online guardrail that is measured independently.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Detail |
|---|---|
| Unit of analysis | One search session: `session_id` + `query_id`; a session may contain several queries, each retained separately |
| Keys | `session_id` (stable, assignment key), `query_id` (unique per submitted query), `impression_id` (one rendered result position) |
| Volume and window | 2,000,000 sessions/day illustrative; 28-day evaluation window; UTC timestamps with locale display conversion |
| Inputs | Raw query, locale, device class, ranked list with positions, item availability snapshot reference, model version, analyser version |
| Outcome/label | `order_kept`: an order placed from the session that survives the cancellation window and is not returned within 30 days; maturity is 30 days, so the last window day is censored |
| Population and exclusions | Queries with ≥1 result; bot traffic excluded by a documented rule; staff sessions excluded; restricted-category queries excluded with reason code |
| Join semantics | Session→query is 1:N; query→order is 0:N; impressions are counted with `COUNT(DISTINCT impression_id)`; duplicate order events reconciled to the earliest terminal state |
| Splits | Time-based: 21 days fit/observe, 7 days held out untouched; the online test uses a fresh 14-day window after the offline decision |
| Censoring | Orders placed on day 29–30 of the window are immature at analysis time and reported in a separate censored count |

**Grain warning.** The most common defect in this data is joining `order` to `session`
without deduplicating retries, which double-counts orders and inflates conversion
identically in both arms — a failure that looks like noise until someone checks the count
against the order service.

### 5b. Tools & Stack

| Component | Role | Version / configuration | Claim status |
|---|---|---|---|
| Inverted index | Lexical retrieval, positional phrase support | analyser pinned per field | proposed |
| Vector index | Semantic candidate generation | embedding model version pinned in the index metadata | proposed |
| Learning-to-rank model | Ordering over lexical, behavioural, availability features | gradient-boosted trees, fixed feature list | proposed |
| Feature store | Point-in-time feature retrieval | event-time snapshots, not latest-state lookups | proposed |
| Judgement store | Human relevance labels | 4-grade scale, pooling provenance recorded | proposed |
| Metrics service | Offline NDCG, online conversion, slice decomposition | versioned metric definitions | proposed |
| Experiment assignment | Session-scoped bucketing | hash of `session_id` with a salted key | proposed |
| Rollback fence | Dual index read plus version pin | parallel index, 10-minute target | proposed |

Vendors, hardware classes, and model sizes are deliberately unstated: the case is about
the decision logic, and naming a product would imply a benchmark that was never run.

### 5c. Method Summary

```text
raw query -> normalise (analyser v) -> retrieve (lexical + vector)
          -> filter (availability, compliance, restriction)
          -> rank (LTR v) -> render -> session outcome
```

The offline screen computes NDCG@10 per judged query, then averages. The case's central
methodological claim is that this average is *insufficient* and must be paired with:

- **Pooled and unpooled metric reporting**, so the bias from incumbent-derived labels is
  visible rather than absorbed.
- **Slice decomposition** with a fixed, pre-registered slice definition.
- **A mixture model** that separates the aggregate change into slice and non-slice
  contributions, so a masked regression cannot hide.
- **An online guardrail** on completed-order conversion at slice level.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Incumbent lexical ranker | Production configuration | Explainable, rollback is a config diff | No cross-paraphrase generalisation | Baseline, retained for fallback | Vocabulary shifts heavily |
| Candidate LTR ranker | Offline NDCG on existing pool | Higher mean NDCG | Pool built from incumbent, no availability term | Evaluated, not shipped | Pool rebuilt with candidate results |
| Bounded re-ranking correction | Slice diagnostic | Reversible per slice, keeps incumbent explainability | Cannot fix retrieval recall | Chosen first step | Slice fix proves insufficient |
| Defer and re-pool | Reviewer capacity | Removes the bias at its source | Costs 6 reviewers for ~2 weeks | Considered, not executed here | Next release cycle |
| Full replacement | — | Simplest operationally long-term | No rollback without reindex | Rejected for now | Dual-index rollback proven |

### 5d. Data provenance, quality, exclusions and leakage controls

- **Provenance.** Query logs, impression logs, order events, inventory snapshots, and
  judgement exports are described as *proposed* artifacts. None were accessed for this
  exercise; all values are constructed.
- **Freshness.** Availability joins use the inventory state as of the rank timestamp, not
  the analysis date. A `latest_state` join is the single most dangerous defect in this
  dataset because inventory changes constantly and the join would leak post-decision stock.
- **Missingness.** Results with no availability record are treated as unavailable with a
  reason code, not dropped; dropping them would overstate the availability feature's value.
- **Duplicates.** Order retries are deduplicated by `order_id` before conversion is
  counted. Impression duplicates are removed by `impression_id`.
- **Skew.** A few head queries dominate volume; per-query weighting would let 40 queries
  decide the release. The metric is session-weighted, and head-query behaviour is reported
  separately as a diagnostic rather than as the decision metric.
- **Leakage controls.** The LTR feature list is audited so that no feature uses a
  post-request event. Specifically excluded: click-through on the *current* impression,
  order events, cancellation flags, and any aggregate computed over a window that includes
  the decision timestamp.
- **Analyser consistency.** Offline evaluation must use the same analyser version as
  serving; a mismatch changes tokenisation and therefore the retrieved set, which is a
  silent and common source of "offline improved, online regressed".

**Unavailable evidence (OPEN).** No primary company relevance study, no production query
log, and no human-judgement export were reviewed. The case therefore cannot and does not
claim any real ranking improvement.

### 5e. Reproducibility and method-selection ledger

| Artifact | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|
| `ILL-01` | NDCG and mixture arithmetic in Section 6 | reproducible from this text | constructed inputs only |
| `ILL-02` | Power/sample-size calculation | reproducible from this text | normal approximation, not a guarantee |
| `ILL-03` | Gate economics (precision, over-trigger) | reproducible from this text | assumes a fixed recall for a hypothetical gate |
| `ORACLE-01` | Reference-history replay of rank decisions | proposed, not executed | no immutable log available here |
| `SRC-OPEN` | Primary relevance literature and company publications | unavailable | no attribution made |

**Takeaway.** Reproducibility of the Section 6 arithmetic is established; production
performance and source claims are not.

## 6. Analysis Walkthrough

**Decisive tradeoff.** The candidate improves the session-weighted mean while a 9.5%
slice regresses by 17%. Any rule that reads only the mean ships the regression; any rule
that vetoes on any slice regression at any size never ships anything. The decision
therefore needs a *sized* slice guardrail and a decomposition that makes the masking
arithmetically explicit.

### Step 1 — Compute the offline screen honestly

Two judged queries, graded relevance on a 0–3 scale, cut-off at 5.

```text
Query A candidate ordering, grades by position: [3, 0, 2, 1, 0]
Ideal ordering for the same grade multiset:     [3, 2, 1, 0, 0]

DCG@5 = sum over positions i of (2^rel_i - 1) / log2(i + 1)
      = (2^3-1)/log2(2) + (2^0-1)/log2(3) + (2^2-1)/log2(4)
        + (2^1-1)/log2(5) + (2^0-1)/log2(6)
      = 7/1 + 0/1.5849625 + 3/2 + 1/2.3219281 + 0/2.5849625
      = 7 + 0 + 1.5 + 0.4306766 + 0
      = 8.9306766

IDCG@5 (ideal) = 7/1 + 3/1.5849625 + 1/2 + 0/2.3219281 + 0/2.5849625
               = 7 + 1.8927893 + 0.5
               = 9.3927893

NDCG@5(A) = 8.9306766 / 9.3927893 = 0.950801...  (0.9508009 at full precision)
```

```text
Query B candidate ordering, grades by position: [1, 3, 0, 2, 1]
Ideal ordering for the same grade multiset:     [3, 2, 1, 1, 0]

DCG@5(B) = 1/1 + 7/1.5849625 + 0 + 3/2.3219281 + 1/2.5849625
         = 1 + 4.4165194 + 0 + 1.2919858 + 0.3868528
         = 7.0953580

IDCG@5(B) = 7/1 + 3/1.5849625 + 1/2 + 1/2.3219281 + 0
          = 7 + 1.8927893 + 0.5 + 0.4306766
          = 9.8234659

NDCG@5(B) = 7.0953908 / 9.8234659 = 0.722290...  (0.7222905 at full precision)
```

Session-weighted mean over these two queries:

```text
FORMULA [SR-01]: NDCG_bar = (1/N) * sum_{q=1..N} DCG@k(q) / IDCG@k(q)

WHERE: N judged queries; DCG@k as above; IDCG@k the DCG of the grade-sorted ideal
       list for the same query's grade multiset; grades on a fixed ordinal scale.

CONSTRAINT: the grade multiset is fixed per query; the pool for that query contains
            every document that could be credited. If the pool is incomplete, the
            quantity computed is a pooled NDCG, not the true NDCG.

NUMERICAL EXAMPLE: (0.950801 + 0.722290) / 2 = 1.673091 / 2 = 0.8365455.

INTERPRETATION: 0.8365 is a screening number for two queries. It says ordering
                quality is imperfect on B; it says nothing about conversion.
```

**Independent check.** Recompute A from the ideal side: the sum of gains is
7 + 3 + 1 = 11 across five positions; the achieved 8.9306766 must be ≤ IDCG 9.3927893.
It is. For B, gains are 1 + 7 + 3 + 1 = 12; the achieved 7.0953580 ≤ 9.8234659. Both
inequalities hold, which is the structural check that no grade was mis-mapped.

**What this step establishes.** The candidate can be scored. It does not establish that
the score is the right score, because the pool contains only incumbent top-10 documents
(`A-02` violated). The corrective is to report pooled and unpooled values separately and
to re-pool a stratified sample before any *magnitude* claim.

### Step 2 — Decompose the production result and expose the masking

| Arm | Slice conversion | Non-slice conversion | Slice share | Aggregate conversion |
|---|---|---|---|---|
| Incumbent | 3.05% | 4.2323% | 9.5% | 4.1200% |
| Candidate | 2.52% | 4.3542% | 9.5% | 4.1800% |
| Difference | −0.53 pp | +0.1219 pp | — | +0.06 pp |

Mixture check:

```text
FORMULA [SR-02]: c_agg = s * c_slice + (1 - s) * c_rest

WHERE: c_agg aggregate conversion [orders/session]; s slice share of sessions
       [dimensionless, 0 <= s <= 1]; c_slice, c_rest conversion within groups.

CONSTRAINT: the two groups partition the population and both use the same
            denominator definition and the same outcome maturity.

NUMERICAL EXAMPLE (incumbent): 0.095 * 0.0305 + 0.905 * 0.042323
                             = 0.0028975 + 0.03830 = 0.041200.
INTERPRETATION: 4.12% aggregate, matching the reported arm value.
```

```text
FORMULA [SR-03]: Delta_c_agg = s * Delta_c_slice + (1 - s) * Delta_c_rest
                              + Delta_share * (c_slice - c_rest)

WHERE: Delta_share is the change in slice share between arms (zero here by design).

NUMERICAL EXAMPLE: 0.095 * (-0.0053) + 0.905 * (0.001219)
                 = -0.0005035 + 0.0011032 = +0.0005997 ≈ +0.06 pp.
INTERPRETATION: the aggregate gain is entirely carried by the non-slice 90.5%,
                and it is 1.8x larger in absolute terms than the slice loss.
```

The masking ratio is the number that should drive the guardrail:

```text
FORMULA [SR-04]: M = |s * Delta_c_slice| / |Delta_c_agg|

NUMERICAL EXAMPLE: M = 0.0005035 / 0.0005997 = 0.8396.
INTERPRETATION: a slice holding 9.5% of traffic contributes 84% as much loss as
                the entire aggregate gain. The mean is not "fine"; it is a near
                cancellation of two large opposite movements.
```

**Why this matters decision-theoretically.** With `M` close to 1, the aggregate is
numerically unstable: a 10% change in either component flips its sign. A release gated on
the aggregate sign is therefore gated on noise. The correct read is "two populations moved
strongly in opposite directions", which is a *segmentation* finding, not a ranking finding.

### Step 3 — Size the slice guardrail instead of judging by vibe

A guardrail needs a sampling distribution. Under a two-proportion normal approximation:

```text
FORMULA [SR-05]: n_per_arm = (z_{1-alpha/2} + z_{power})^2
                            * (p1(1-p1) + p2(1-p2)) / (p1 - p2)^2

WHERE: n sessions per arm; p1, p2 conversion rates as proportions; alpha two-sided
       significance level; power 1 - beta.

CONSTRAINT: independent Bernoulli outcomes per session; large-sample normal
            approximation; no within-session correlation. Sessions with multiple
            queries violate independence and require a clustered variance.

NUMERICAL EXAMPLE (slice): p1 = 0.0305, p2 = 0.0252, alpha = 0.05, power = 0.80.
  z_{0.975} = 1.9600, z_{0.80} = 0.8416
  numerator  = (2.8016)^2 * (0.0305*0.9695 + 0.0252*0.9748)
             = 7.8490 * (0.0295698 + 0.0245650) = 7.8490 * 0.0541348 = 0.424905
  denominator= (-0.0053)^2 = 0.00002809
  n         = 0.424905 / 0.00002809 = 15,127 sessions per arm.

  At 190,000 slice sessions/day split across two arms: 95,000 per arm/day,
  so 15,127 / 95,000 = 0.16 days ≈ 4 hours of traffic.
INTERPRETATION: a 17% relative slice regression is detectable in hours, so a
                "we cannot know" defence is false at this slice size.
```

```text
FORMULA [SR-06]: n_corrected = n_per_arm * (z_{1-alpha/(2m)} / z_{1-alpha/2})^2  [approx.]

WHERE: m simultaneous slice comparisons; the ratio rescales the per-comparison
       sample size to an approximate Bonferroni-corrected level.

CONSTRAINT: approximate; exact correction depends on the joint distribution of the
            slice statistics, which are not independent when slices overlap.

NUMERICAL EXAMPLE: m = 12 slices, alpha = 0.05.
  z_{1-0.05/24} = z_{0.9979167} = 2.8653
  ratio^2 = (2.8653/1.9600)^2 = (1.4619)^2 = 2.1371
  n_corrected = 15,127 * 2.1371 = 32,318 per arm ≈ 0.34 days of slice traffic.
INTERPRETATION: even with 12 slices tested simultaneously, the guardrail is
                affordable at this slice size. That is a fact about the slice's
                volume, not about the ranker.
```

Overall-metric power, for contrast:

```text
p1 = 0.0412, p2 = 0.0418, alpha = 0.05, power = 0.80
  numerator  = 7.8490 * (0.0412*0.9588 + 0.0418*0.9582)
             = 7.8490 * (0.0395026 + 0.0400528) = 7.8490 * 0.0795554 = 0.624430
  denominator= (0.0006)^2 = 0.00000036
  n         = 1,734,528 per arm.
  At 1,000,000 sessions per arm/day: 1.73 days.
```

So the aggregate needs about 2 days of traffic to resolve a 6-basis-point move, while the
slice needs about 4 hours. **The guardrail is *more* measurable than the primary metric.**
This asymmetry is the case's practical lever: the owner can gate on the slice without
waiting for the aggregate to become decisive.

### Step 4 — Price a targeted gate rather than a full rollback

A gate that routes a suspected slice to the incumbent costs recall and precision. Suppose
the gate's recall on truly-affected sessions is 86% and it fires on 14.8% of all sessions.

| Quantity | Computation | Result |
|---|---|---|
| Share of traffic gated | given | 14.8% |
| Slice sessions correctly gated | 0.095 × 0.86 | 8.17% of all sessions |
| Slice sessions left on candidate | 9.5% − 8.17% | 1.33% |
| Non-slice sessions gated (false positives) | 14.8% − 8.17% | 6.63% |
| Non-slice sessions left on candidate | 90.5% − 6.63% | 83.87% |
| Gate precision | 8.17% / 14.8% | 55.2% |
| Non-slice over-trigger rate | 6.63% / 90.5% | 7.33% |

```text
FORMULA [SR-07]: Orders_kept_gate = S * [ f_uncaught * (c_slice^C - c_slice^B)
                                       + f_rest^C * (c_rest^C - c_rest^B) ]

WHERE: S sessions/day; f_uncaught slice share not gated and served by candidate;
       f_rest^C non-slice share served by candidate; c^C, c^B conversion in
       candidate and baseline.

NUMERICAL EXAMPLE: S = 2,000,000.
  0.0133 * (-0.0053) = -0.00007049
  0.8387 * (+0.001219) = +0.00102237
  sum = +0.00095188
  Orders/day = 2,000,000 * 0.00095188 / 100 * 100 ... see unit note below.

Unit note: conversion is a proportion (orders per session), so multiplying sessions by
a proportion yields orders directly:
  2,000,000 * 0.00095188 = 1,903.8 orders/day kept.
```

Compare with the ungated candidate:

```text
2,000,000 * (0.095 * -0.0053 + 0.905 * 0.001219)
= 2,000,000 * (-0.0005035 + 0.00110320)
= 2,000,000 * 0.00059970 = 1,199.4 orders/day kept.
```

**Result.** The targeted gate keeps about 1,904 modelled orders/day versus about 1,199 for
the ungated candidate — roughly 59% more modelled retained value — while leaving 1.33% of
slice sessions exposed. That residual exposure is the gate's honest boundary: recall is
not 100%, so the gate is a mitigation, not a repair.

**Where the gate is the wrong tool.** If the root cause is a retrieval defect that also
affects queries misclassified out of the slice, the gate hides a broader problem and
manufactures a false sense of containment. The gate's own precision of 55.2% means nearly
half its actions are unnecessary, which is acceptable when the action is cheap (serve the
incumbent) and unacceptable when the action is expensive (reindex).

### Step 5 — Quantify the rollback risk that no metric shows

Rollback to the incumbent requires the incumbent's index, analyser, and feature snapshot
to still exist. If the candidate shipped a new index format, "rollback" is a reindex of
the whole catalogue. With a 10-minute budget and a reindex measured in hours, the honest
conclusion is that the candidate was never reversible, and a non-reversible change must
not be rolled out on an aggregate-only signal.

```text
FORMULA [SR-08]: R_feasible = 1 if (t_restore <= t_budget) else 0

WHERE: t_restore total time to restore the previous serving state, including
       index, analyser, model, and configuration; t_budget the business tolerance.

CONSTRAINT: t_restore must be measured by an actual drill, not estimated from
            component documentation. Summing component estimates understates
            restoration because steps are serialized by dependency.

NUMERICAL EXAMPLE: t_restore = 41 min (drill), t_budget = 10 min -> R_feasible = 0.
INTERPRETATION: dual-index read is a prerequisite, not an optimisation.
```

### Step 6 — Sensitivity

| Stress | Original | Alternative | Conclusion change | Limitation | Decision consequence |
|---|---|---|---|---|---|
| Slice share | 9.5% | 3.0% | Aggregate gain rises to +1.16 pp while slice loss shrinks to −0.16 pp; masking ratio falls to 0.14 | Small slices are harder to measure | Gate still preferred; guardrail power drops |
| Slice share | 9.5% | 25% | Aggregate becomes −0.18 pp: the release is net negative | Requires 25% of traffic in one slice | Veto outright, not gate |
| Slice regression size | −17.4% rel | −4% rel | Aggregate +0.22 pp; masking ratio 0.17 | Smaller effects need more traffic | Gate optional; monitor |
| Non-slice gain | +2.88% rel | +0.5% rel | Aggregate +0.004 pp; masking ratio 12.6 | Aggregate indistinguishable from zero | Defer, do not ship |
| Gate recall | 86% | 60% | Uncaught slice rises to 3.8%; modelled kept orders ≈ 1,540/day | Recall depends on the classifier | Prefer full slice exclusion |
| Gate false-positive rate | 7.33% of non-slice | 20% of non-slice | Non-slice candidate coverage falls to 72%; kept orders ≈ 1,600/day | Opportunity cost of gating grows | Tighten the gate or scope by query pattern |
| Order maturity | 30 days | 7 days | Early conversion overstates kept orders; cancellation-heavy slices look better | Censoring bias | Keep 30-day maturity for the final claim |
| Assignment stickiness | session | request | Both arms contaminated; effect attenuated toward zero | Within-session mixing | Fix assignment before trusting any number |

**Boundary that matters most.** At a slice share of 25%, the same per-slice movement flips
the aggregate negative. No amount of offline NDCG protects against that; only the
decomposition does.

### 6.7 What was tried first and why it did not settle the question

`Not tested.` No alternative ranking configuration was actually run for this exercise. The
incumbent and candidate are described as alternatives, not as historical runs. The
honest statement is: the arithmetic shows how the decision must be structured; the
organisation still owes the measurement.

## 7. The Finding

The candidate ranker must not replace the incumbent for all traffic on the strength of an
aggregate mean, because the aggregate is a near-cancellation of a −0.53 pp movement in a
9.5% slice and a +0.12 pp movement in the remaining 90.5%, with a masking ratio of 0.84.
The recommended action is to exclude the affected slice from the candidate (or gate it
with a measured classifier) and re-evaluate on the remaining population, where the
modelled retained value is about 1,904 orders/day against 1,199 for the ungated release.

The evidence is illustrative and derived, not measured: the slice numbers are constructed,
the gate recall is a stipulated 86%, and the power figures are normal-approximation
estimates. The finding holds only while point-in-time availability features, session-scoped
assignment, and a demonstrated sub-10-minute rollback all hold. If rollback cannot be
demonstrated, the correct action degrades to "defer", regardless of the arithmetic.

**Uncertainty.** The largest unquantified risk is pool incompleteness: the offline metric
was computed on labels drawn from incumbent results, so the candidate's true offline
quality is unknown in either direction. The second is conversion stability across the
window.

## 8. System/Solution Design

### 8.1 Architecture

```text
                    +---------------------------+
  query ----------->| query understanding (v_a) |--- parsed_query + analyser version
                    +-------------+-------------+
                                  |
                    +-------------v-------------+
                    |  retrieval: lexical + vec |--- candidates + retrieval scores
                    +-------------+-------------+
                                  |
                    +-------------v-------------+        +----------------------+
                    | availability / compliance |<-------| inventory snapshot   |
                    |        filter             |        | (event-time, pinned) |
                    +-------------+-------------+        +----------------------+
                                  |
                    +-------------v-------------+
                    |  ranker (model v_r)       |--- ranked list + feature trace
                    +-------------+-------------+
                                  |
              +-------------------v--------------------+
              | slice router (session-keyed, fenced)   |--- arm decision recorded
              +---------+--------------------+---------+
                        |                    |
             incumbent ordering        candidate ordering
                        |                    |
                        +---------+----------+
                                  |
                    +-------------v-------------+
                    | render + impression log   |
                    +-------------+-------------+
                                  |
                    +-------------v-------------+
                    | outcome matcher (30-day)  |--- order_kept, cancellation, return
                    +---------------------------+
```

### 8.2 Key choices and their justification

| Choice | Alternative rejected | Why | Boundary where the choice fails |
|---|---|---|---|
| Session-keyed assignment | Request-keyed | Keeps one session in one arm, so conversion is attributable | A user with two devices gets two assignments |
| Slice router instead of global flag | Global feature flag | Contains a known regression without discarding the gain | Slice definition itself can be wrong |
| Dual index read for rollback | Single index with config flip | Meets the 10-minute budget | Doubles index memory during the overlap |
| Offline NDCG as screen only | NDCG as decision metric | Pool bias and objective mismatch make it insufficient | Useful when the pool is candidate-complete |
| 30-day outcome maturity | 7-day proxy | Cancellations and returns concentrate after day 7 | Slows every decision by three weeks |

### 8.3 Successful path

1. Query arrives; analyser version is recorded with the parsed query.
2. Retrieval returns candidates with scores and the index snapshot ID.
3. Availability filter consults the snapshot pinned at request time; excluded items carry
   a reason code.
4. Ranker produces an ordering and a feature trace keyed by `(query_id, model_version)`.
5. The slice router assigns the session to an arm using a salted hash of `session_id`, and
   writes the assignment *before* rendering.
6. The impression log records positions, item IDs, and arm.
7. The outcome matcher joins orders to sessions after maturity and writes `order_kept`.

### 8.4 Failed path and recovery

| Failure | Detection | Immediate action | Recovery | Verification |
|---|---|---|---|---|
| Availability snapshot missing | Filter emits `availability_unknown` count above threshold | Serve incumbent for affected queries | Re-pin snapshot, replay filter | Compare served availability against the inventory authority for sampled queries |
| Router writes assignment after render | Ordering of audit timestamps in the log | Stop the release fence | Fix write order, replay the window | Every impression has exactly one prior assignment |
| Candidate latency exceeds budget | Serving p99 and timeout counters | Automatic fallback to incumbent | Reduce candidate list size or model cost | Timeout rate returns below threshold and arm labels remain correct |
| Rollback exceeds 10 minutes | Timed drill | Do not ship; declare non-reversible | Pre-build parallel index, dual-read | Drill passes twice consecutively |
| Outcome matcher double counts | Order count differs from order service | Freeze metric reporting | Deduplicate by `order_id` | Counts reconcile to the authority |

**Takeaway.** The recovery path is a designed component with its own budget; a ranker
without a *tested* restore path is a production risk independent of its quality.

## 9. Evaluation & Validation

### 9.1 Metric definitions

| Metric | Numerator | Denominator | Population | Window | Comparator | Guardrail | Status |
|---|---|---|---|---|---|---|---|
| NDCG@10 | Sum of discounted gains | Sum of discounted ideal gains | Judged queries | Fixed offline snapshot | Incumbent on the same pool | Pooled/unpooled gap | not measured |
| Completed-order conversion | Sessions with a kept order | Sessions with ≥1 result | Same population both arms | 14-day online window, 30-day maturity | Incumbent arm | Slice regression | not measured |
| Slice regression | Candidate slice conversion − incumbent slice conversion | Incumbent slice conversion | The 9.5% slice | 14 days | Incumbent arm | ≤2% relative, pre-registered | not measured |
| Zero-result rate | Queries with no result | All queries | All queries | 14 days | Incumbent arm | ≤ baseline + 0.5 pp | not measured |
| Rollback time | Drill start to restored serving | — | Staging fence | Per drill | 10-minute budget | Hard | not measured |
| Availability accuracy | Items correctly marked available | Items shown | Sampled impressions | Per day | Inventory authority | ≥99% | not measured |

### 9.2 Validation protocol

1. **Pre-register** the slice definition, the guardrail threshold (2% relative), and the
   stopping rule before unblinding any result.
2. **Balance check** on assignment: session counts per arm within 0.5%, and no
   systematic difference in locale, device, or query-length distribution.
3. **Point-in-time audit**: for a random sample of 500 rank decisions, verify that every
   feature value was computable at the recorded decision timestamp.
4. **Pool audit**: re-pool 200 queries with candidate top-10 added, and report the pooled
   and unpooled NDCG delta. If the sign flips, the offline screen was measuring the pool.
5. **Rollback drill** twice, then again during the first hour of the release.
6. **Sequential monitoring** with a pre-declared alpha-spending rule so that peeking does
   not inflate the error rate.

### 9.3 What this evaluation cannot establish

- It cannot establish causality for the *slice* fix if the slice router correlates with
  user intent (for example, if the slice is defined by query language and language
  correlates with device class).
- It cannot establish transfer to other locales or to the vector index's other languages.
- It cannot establish that the offline screen would pass on a re-pooled judgement set.

**Takeaway.** The protocol must be able to *fail* the candidate; a protocol whose only
outcome is a ship decision is theatre.

## 10. Action Taken

No production action was taken. This is an illustrative exercise.

The proposed action set, in order:

1. **Nominate** the ranking quality manager as decision owner and record the decision
   contract of Section 3 in the release checklist.
2. **Declare the candidate non-reversible** until the rollback drill passes twice.
3. **Exclude the affected slice** from the candidate instead of shipping globally; record
   the exclusion rule and its version.
4. **Re-pool** a stratified sample of 200 queries with candidate results added, and report
   pooled versus unpooled NDCG before making any magnitude claim.
5. **Run the fenced slice test** with the pre-registered guardrail, sequential monitoring,
   and the 30-day outcome maturity.
6. **Publish the gate's** recall and precision alongside the conversion result, so the
   residual exposure is visible rather than implied.

**Takeaway.** The first action is a *contract*, not a model.

## 11. Outcome & Results

**Result: not-measured.**

| Claim | Value | Label | What it would take to measure |
|---|---|---|---|
| Offline mean NDCG@5 on two judged queries | 0.8365455 | DERIVED (illustrative inputs) | A real judgement export with documented pooling |
| Aggregate conversion change | +0.06 pp | DERIVED (illustrative inputs) | A balanced online test with 30-day maturity |
| Slice conversion change | −0.53 pp | DERIVED (illustrative inputs) | Slice-level test with pre-registered guardrail |
| Masking ratio | 0.84 | DERIVED | Recomputed from measured components |
| Gate-kept orders/day | 1,903.8 | DERIVED (stipulated 86% recall) | Measured gate recall and precision |
| Ungated kept orders/day | 1,199.4 | DERIVED | The same online test without the gate |
| Rollback feasibility | 0 (41 min vs 10 min budget) | ILLUSTRATIVE drill value | An actual timed drill |

A future outcome record must state population, numerator, denominator, window,
comparator, uncertainty interval, and guardrail status. Projected orders/day must never be
restated as observed orders.

**Takeaway.** Modelling a decision and observing its consequence are different epistemic
acts; conflating them is how a plausible design becomes a false claim.

## 12. Failure Modes & What Went Wrong

### FM-01 — Pool built from incumbent results makes the candidate look worse than it is

**Mechanism.** Judgement sets are assembled from the top-k of the live ranker. Documents
the candidate would surface were never judged, so they are scored as grade 0. The bias is
proportional to how much the candidate changes retrieval, which is exactly the quantity
the evaluation is supposed to measure.

**Signal.** Unpooled NDCG improves when the candidate's own top-10 are added to the pool,
and the improvement is concentrated in queries where the candidate's candidate set differs
most from the incumbent's.

**Discriminating check.** Take 200 queries, union both arms' top-10, re-judge, and compare
the *delta* between pooled and unpooled. If |delta_pooled| > |delta_unpooled|, the original
screen was measuring pooling, not quality.

**Correction.** Re-pool, then re-screen. **Mitigation risk.** Judging costs reviewer hours
and delays the release; a rushed pool reproduces the same bias with more confidence.

**Verification.** The pooled and unpooled deltas agree in sign after re-pooling.

### FM-02 — Global metric masks a slice regression

**Mechanism.** Two populations move in opposite directions; the weighted sum nearly
cancels. The aggregate is a low-signal statistic whose sign is unstable.

**Signal.** Masking ratio near 1, and the aggregate's confidence interval spanning zero
while both component intervals exclude zero.

**Discriminating check.** Compute `SR-03` with the measured components. If
`|s * Delta_slice| / |Delta_agg| > 0.5`, the aggregate is not a decision statistic.

**Correction.** Gate or exclude the slice; report components. **Mitigation risk.** Gating
costs the non-slice gain wherever the gate falsely fires.

**Verification.** Kept-order value recomputed with measured gate precision exceeds the
ungated value.

### FM-03 — Availability feature leaks post-request stock state

**Mechanism.** A `latest_state` lookup substitutes the analysis-time inventory for the
rank-time inventory. Offline, out-of-stock items appear rankable; online, they are filtered
later, so offline and online disagree about which items were even candidates.

**Signal.** Offline NDCG improves sharply on queries whose items had large inventory
mobility in the window; the improvement vanishes when the feature is ablated with a
point-in-time join.

**Discriminating check.** Re-join with event-time semantics and recompute. A materially
lower offline score confirms the leak.

**Correction.** Rebuild the feature store snapshot at decision time. **Mitigation risk.**
Point-in-time joins are slower and more expensive; a cached "close enough" join silently
reintroduces the leak.

**Verification.** Feature timestamps are ≤ decision timestamps for every sampled record.

### FM-04 — Analyser version mismatch between offline and serving

**Mechanism.** Tokenisation, stemming, and synonym expansion differ between the offline
evaluation build and the serving cluster. The retrieved sets differ, so the offline list
is not the list that will be served.

**Signal.** The same query returns a different candidate count offline and online with no
model change.

**Discriminating check.** Replay 50 queries through both analysers and diff the token
streams. Any diff is disqualifying.

**Correction.** Pin the analyser version in the index metadata and assert it at query
time. **Mitigation risk.** A pinned analyser blocks urgent linguistic fixes; the escape is
a versioned reindex, not an in-place change.

**Verification.** Token streams are identical for the replayed set.

### FM-05 — Request-scoped assignment splits a session across arms

**Mechanism.** Assignment keyed on request rather than session puts two queries from the
same shopping journey into different arms. Conversion is then attributed to whichever arm
served the last query, attenuating the effect toward zero and making the variance
estimate wrong.

**Signal.** The number of sessions with more than one distinct arm label is non-zero.

**Discriminating check.** Query the assignment log for `COUNT(DISTINCT arm) > 1` per
session. Any non-zero value invalidates the analysis.

**Correction.** Key on `session_id` with a salted hash and persist the assignment.
**Mitigation risk.** Sticky assignment reduces effective randomness and can produce
imbalance if the salt correlates with time.

**Verification.** Every session has exactly one arm label, and arm balance is within 0.5%.

### FM-06 — Gate recall is asserted rather than measured

**Mechanism.** The gate's benefit is computed from a stipulated recall. If the gate fires
on the wrong sessions, the modelled kept orders are fiction, and the residual exposure is
larger than reported.

**Signal.** Gate precision on a labelled sample is far from the stipulated value, and the
distribution of gated sessions does not resemble the slice.

**Discriminating check.** Label 300 gated and 300 ungated sessions by hand; compute recall
and precision against that label.

**Correction.** Replace the stipulated recall with the measured value and recompute
`SR-07`. **Mitigation risk.** A hand-labelled sample of 600 is small; report its interval.

**Verification.** The recomputed kept-order value still exceeds the ungated value, or the
gate is abandoned.

### FM-07 — Timeout fallback silently reverts the treatment arm

**Mechanism.** When the candidate exceeds its latency budget, the serving layer falls back
to the incumbent. The fallback is correct for availability and fatal for measurement: the
"candidate" arm becomes a mixture whose mixing rate depends on load.

**Signal.** Fallback counters rise with peak traffic, and the arm's effect size shrinks
during peak hours.

**Discriminating check.** Compare conversion in candidate-arm sessions with and without a
recorded fallback. A large difference means the arm label is not the treatment.

**Correction.** Either exclude fallback sessions from the analysis or size the candidate
to meet the budget. **Mitigation risk.** Excluding fallback sessions biases the population
toward low-load hours.

**Verification.** Fallback rate is below the pre-registered threshold, and the primary
estimate is reported both with and without fallback sessions.

### FM-08 — Rollback restores configuration but not the index

**Mechanism.** The candidate required a new index format. Rolling back the model version
leaves a new-format index that the incumbent cannot read, so restoration requires a full
reindex that exceeds the budget.

**Signal.** The rollback drill completes the config step in seconds and then blocks on the
index step for tens of minutes.

**Discriminating check.** Time each restoration step separately; the maximum step, not the
sum of estimates, determines feasibility.

**Correction.** Dual-write both formats during the overlap, or dual-read with a version
pin. **Mitigation risk.** Dual-format storage doubles index memory and complicates merges.

**Verification.** Two consecutive drills complete within budget with the candidate's index
in place.

**Takeaway.** Each failure above has a *different* discriminating check. "Check the logs"
discriminates nothing; the value of the list is that each check can fail in only one way.

## 13. Transferable Lesson

The transferable mechanism is the **decomposition requirement**, not the NDCG arithmetic.
Any decision whose aggregate is a weighted sum over sub-populations inherits the same
hazard: a large local movement can be nearly cancelled by an opposite movement elsewhere,
and the aggregate's sign becomes a coin flip. The general procedure is:

```text
STEP 1  Declare the population partition before seeing results.
STEP 2  Report every component with its own denominator and its own interval.
STEP 3  Compute the masking ratio |s * Delta_component| / |Delta_aggregate|.
STEP 4  If the ratio is large, treat the aggregate as non-decisional.
STEP 5  Size each guardrail from its own sampling distribution, not from the aggregate's.
STEP 6  Price the mitigation (gate, exclusion, rollback) against the same ledger.
STEP 7  Require a demonstrated restore path before any irreversible move.
```

**Where this transfers directly.** Marketplace seller-risk thresholds (a small
high-risk cohort with a large effect), marketing attribution (a small converting segment),
claims triage (a rare catastrophic class), and infrastructure rollouts (a small latency-
sensitive key range).

**Where it does not transfer.** When the partition is *endogenous* — defined by the
treatment itself, or by a variable the treatment affects — the decomposition is not a
valid causal split. In this case the slice is defined by query language, which the ranker
does not change, so it is exogenous enough; a slice defined by "sessions where the
candidate returned fewer results" would not be.

**Prerequisites.** A stable partition key, per-component denominators, a measured
mitigation rather than an assumed one, and a tested restore path.

**Counterexample to the lesson.** If the slice is 0.1% of traffic and the effect is small,
per-component intervals will not exclude zero and the decomposition buys nothing; the
correct action there is to pool the evidence over a longer window, not to gate on noise.

**Takeaway.** Transfer the contract — partition first, components measured, mitigation
priced, restore proven — not the scenario's figures.

## 14. Connections & Cross-References

### 14.1 Related methods and cases

| Relation | Artifact | Why it is related | Difference that matters |
|---|---|---|---|
| Same pattern, different domain | `../02_marketplace_seller_trust_and_economics/Case_Studies/01_Seller_Risk_Scoring.md` | Both face evaluation mismatch between an offline score and a business consequence | Seller risk has delayed labels; search has immediate but censored orders |
| Method reuse | `../Cheat_Sheets/01_Ecommerce_Search_Scorecard.md` | Defines the metric suite and cut-offs | The sheet owns the definitions; this case owns the decision |
| Contrast | `03_Recommendation_Cold_Start.md` | Both rank items for a shopper | Cold start has no interaction history; this case has a full pool |
| Systems dependency | `06_Authorized_Near_Real_Time_Index.md` | Index freshness determines candidate sets | That case owns the indexing contract |
| Measurement dependency | `../03_digital_commerce_data_platform_and_operations/Case_Studies/02_Experiment_Assignment_Failure.md` | Assignment correctness underpins the online claim | That case owns assignment semantics |

### 14.2 Open questions (explicitly unresolved)

1. **Pool completeness.** Does re-pooling reverse the candidate's offline sign? Unknown;
   requires reviewer hours that were not available.
2. **Gate generalisation.** Is the 86% recall stable across query paraphrases, or does it
   depend on exact-match patterns? Unknown.
3. **Availability feature value.** How much of the candidate's offline gain is the
   availability feature, and how much survives a point-in-time rebuild? Unknown; the
   ablation was not run.
4. **Long-run effect.** Does excluding the slice train users to abandon the category?
   Outside the 14-day window.
5. **Locale transfer.** Does the decomposition hold in a second locale with different
   morphology? Untested.

### 14.3 Interview pack (14 questions, each fully answered)

#### Interview 01 — State the decision in 30 seconds

**Prompt.** You have ten minutes with the category manager. What is the decision?

**Answer.** "We have a candidate ranker that raises the average while hurting one slice of
about 9.5% of sessions by roughly 17% relative. I am proposing we exclude that slice and
run a fenced test on the rest, because the average is a near-cancellation of two opposite
movements and its sign is not stable. The test needs a pre-registered 2% slice guardrail
and a rollback drill that completes in ten minutes; today we cannot demonstrate the
rollback, so the honest status is 'defer the global release, proceed with the slice fence
once rollback is proven'."

**Reasoning.** The answer names the population, the decision, the two constraints that
could invalidate it, and the fallback. It does not lead with the model.

**Follow-up.** "What if the category manager says the slice is unimportant?" Then the
decision changes: an unimportant slice with a large local loss may still matter if the
loss is concentrated in high-margin items. Ask for contribution per session, not orders.

**Acceptable alternative.** "Run the global test and monitor the slice" is defensible *if*
the slice guardrail has enough power — which Section 6 shows it does at this volume.

**Red flag.** Leading with "NDCG improved by 0.04" without a population or a denominator.

**Scoring.** 0 = no population; 1 = names the metric only; 2 = names population,
guardrail, and fallback; 3 = also names the reversibility prerequisite and its price.

**Answer support.** Sections 1, 3, 6 (Steps 2–4), and 10.

#### Interview 02 — Why is the incumbent baseline credible?

**Prompt.** Justify keeping a lexical ranker as the comparator.

**Answer.** Because it is the actual counterfactual. It serves all traffic today, its
behaviour is explainable from configuration, and its rollback path is a config diff. The
comparator's job is to represent what happens without the intervention, not to be
impressive. A candidate must beat the thing that will otherwise run.

**Reasoning.** A weak-looking baseline that is genuinely deployed dominates a strong-looking
one that is hypothetical. The risk is that the baseline's weakness (no cross-paraphrase
generalisation) makes the candidate's advantage look larger than the *deployable* advantage,
because some of the gap is a retrieval change that the rollback path cannot undo.

**Follow-up.** "What if the baseline is already bad?" Then the case for change is stronger
but the decision rule does not relax: you still need a guardrail and a restore path.

**Acceptable alternative.** Using a *rule-only* ranker as a second comparator isolates how
much of the gain comes from learning versus from the new retrieval stage.

**Red flag.** Comparing against an "ideal" ranker or against an earlier offline experiment
that was never deployed.

**Scoring.** 0 = names a competitor product; 1 = names the incumbent; 2 = explains
counterfactual role; 3 = also explains the retrieval-versus-ranking confound.

**Answer support.** Sections 2.4 and 5c.

#### Interview 03 — Recompute the aggregate from components

**Prompt.** Slice is 9.5% of sessions at 3.05% converting; the rest converts at 4.2323%.
The candidate gives 2.52% and 4.3542%. What is the aggregate change and what does it hide?

**Answer.** Incumbent aggregate is 0.095 × 0.0305 + 0.905 × 0.042323 = 0.041200. Candidate
is 0.095 × 0.0252 + 0.905 × 0.043542 = 0.041800. Change is +0.0006, i.e. +0.06 pp. It
hides a −0.0053 pp slice movement (17.4% relative) offset by +0.001219 pp on the rest. The
masking ratio is 0.0005035 / 0.0005997 = 0.84.

**Reasoning.** The two terms have opposite signs and similar magnitudes; the sum is a small
residual. Reporting the residual without the terms makes an unstable quantity look stable.

**Follow-up.** "At what slice share does the aggregate turn negative?" Solve
`s × (−0.0053) + (1−s) × 0.001219 = 0`, giving `s = 0.001219 / 0.001219 + 0.0053` ≈ 0.187.
At about 18.7% slice share the release is net zero; above it, net negative.

**Acceptable alternative.** Reporting per-component intervals instead of a point aggregate
is strictly better and should be accepted as a full answer.

**Red flag.** Computing the aggregate as an unweighted mean of the two rates (3.785%),
which is wrong because the groups have different sizes.

**Scoring.** 0 = averages the rates; 1 = correct aggregate, no decomposition; 2 = correct
decomposition; 3 = also computes the break-even share.

**Answer support.** Section 6, Step 2, `SR-02`/`SR-03`/`SR-04`.

#### Interview 04 — Which input would leak, and how would you detect it?

**Prompt.** Name the most dangerous leakage risk in this dataset and a check that
distinguishes it from an honest improvement.

**Answer.** The availability feature joined as `latest_state`. Detection: re-join with
event-time semantics and recompute the offline metric. If the score drops materially, the
original used post-decision information. A second check is timestamp arithmetic on a
sample: every feature timestamp must be ≤ the decision timestamp.

**Reasoning.** Inventory is the fastest-moving input in the system, so the gap between
"state now" and "state at ranking" is largest there. The check discriminates because an
honest feature is invariant to the join method.

**Follow-up.** "What if the event-time join is unavailable?" Then drop the feature and
report the model without it; a feature you cannot join honestly is a feature you cannot
claim.

**Acceptable alternative.** Ablating the feature and comparing against a rule-only variant
also isolates its contribution, though it does not prove the leak.

**Red flag.** "Availability is known at request time because the feed is near-real-time" —
proximity is not a timestamp.

**Scoring.** 0 = names clicks; 1 = names availability; 2 = names the join defect; 3 = gives
the invariance check.

**Answer support.** Sections 5d and FM-03.

#### Interview 05 — How does capacity change the selection?

**Prompt.** Reviewer capacity is 6 reviewers at 180 judgements/hour. Does that change the
plan?

**Answer.** Yes, in the evaluation stage, not the serving stage. Extending the pool by
unioning both arms' top-10 over 200 queries needs 200 × 20 = 4,000 judgements, which is
4,000 / (6 × 180) = 3.7 reviewer-hours, or about 0.6 hours of wall-clock across six
reviewers — affordable. Extending it to 2,000 queries needs 40,000 judgements = 37
reviewer-hours ≈ 6.2 hours wall-clock, still affordable but it consumes the week's budget.
What is *not* affordable is re-pooling the entire 4,800-query pool: 96,000 judgements ≈ 89
reviewer-hours.

**Reasoning.** Capacity converts an open question ("is the pool biased?") into a sized
sample ("we can afford 200 queries and an interval on the delta"). It also bounds the
claim.

**Follow-up.** "What if reviewers are the bottleneck and the release window is 5 days?"
Then scope the pool audit to the slice only, and state the unpooled result as unverified.

**Acceptable alternative.** Stratified sampling by query frequency gives more information
per judgement than a uniform sample; it is defensible and preferable when head queries
dominate.

**Red flag.** "We'll just judge everything" without an hours calculation.

**Scoring.** 0 = ignores capacity; 1 = mentions cost; 2 = computes hours; 3 = also states
which claim the budget can and cannot support.

**Answer support.** Sections 2.5, 4 (`C-02`), 5d.

#### Interview 06 — When does the conclusion reverse?

**Prompt.** Give two distinct conditions under which you would recommend the opposite.

**Answer.** First, if the slice share rises above about 18.7%, the aggregate turns
negative and the release should be vetoed outright rather than gated. Second, if the
non-slice gain shrinks to +0.5% relative, the aggregate moves to about +0.004 pp, which is
indistinguishable from zero at this volume, so the release should be deferred — an
immeasurable gain does not justify a reversible-only rollout.

A third: if re-pooling shows the candidate's offline quality is *lower* than the
incumbent's on a candidate-complete pool, the entire premise fails and no amount of online
gating fixes a worse ranker.

**Reasoning.** Each reversal attacks a different term in `SR-03` or a different premise,
so the responses are not interchangeable.

**Follow-up.** "What if both happen together?" Then the decision is veto, and the reason is
the slice, not the gain.

**Acceptable alternative.** A cost-based reversal — if the candidate's infrastructure cost
exceeds the value of the retained orders — is equally valid and should be priced.

**Red flag.** "It reverses if p > 0.05" without saying which comparison.

**Scoring.** 0 = one vague condition; 1 = one precise condition; 2 = two distinct
conditions; 3 = three, with the arithmetic.

**Answer support.** Section 6, Step 6.

#### Interview 07 — Which guardrail blocks rollout, and why that one?

**Prompt.** You can pre-register only one guardrail. Which?

**Answer.** Slice-level completed-order conversion with a 2% relative regression threshold,
because it is the only candidate that (a) shares the decision's objective, (b) is
measurable at this slice volume in hours, and (c) can fail independently of the primary
metric. Zero-result rate is a useful secondary but it is an input to the outcome, not the
outcome.

**Reasoning.** A guardrail must be able to veto. If it cannot fail while the primary metric
passes, it is decoration. The slice metric can: Section 6 shows the slice resolves in about
4 hours while the aggregate needs about 2 days.

**Follow-up.** "What threshold?" 2% relative is about 0.06 pp at a 3.05% base — roughly
1.5× the slice's detectable effect at 4 hours of traffic, so it is a real constraint rather
than a formality.

**Acceptable alternative.** An absolute pp threshold is defensible when base rates vary
across slices; the relative one is preferred here because the base is stable.

**Red flag.** "Latency" as the only guardrail — it protects availability, not the decision.

**Scoring.** 0 = no guardrail; 1 = names a metric; 2 = justifies with measurability;
3 = also prices the threshold.

**Answer support.** Sections 6 (Step 3) and 9.1.

#### Interview 08 — How does the recovery path prevent duplicate effects?

**Prompt.** The outcome matcher reprocesses a day of orders. How do you prevent double
counting?

**Answer.** By making the matcher idempotent on `order_id` with a terminal-state rule:
each order contributes exactly one `order_kept` value, and a later cancellation supersedes
an earlier "kept" only by writing a new version with a higher sequence number, not by
incrementing a counter. Conversion is then computed as
`COUNT(DISTINCT order_id WHERE terminal_state = kept)` over sessions.

**Reasoning.** Counting is where replays break. A versioned terminal state is monotone and
replay-safe; an additive counter is neither.

**Follow-up.** "What if two cancellations arrive for one order?" The second is a no-op
because the state is already terminal; record it as a duplicate observation, not a new
event.

**Acceptable alternative.** A deduplication window keyed on `(order_id, event_type)` also
works but breaks if the window is shorter than the retry horizon.

**Red flag.** "Wrap the job in a transaction" — the job spans an external system.

**Scoring.** 0 = retries the job; 1 = mentions dedupe; 2 = specifies the key; 3 = specifies
the terminal-state rule and the replay test.

**Answer support.** Sections 5a and 8.4.

#### Interview 09 — What is the strongest simpler alternative?

**Prompt.** Argue for a solution that involves no new model.

**Answer.** Fix the availability filter and the field boosts for the affected slice only.
The slice's regression is plausibly driven by showing items that are relevant but not
orderable; a bounded re-ranking correction that demotes unavailable items within the slice
is a configuration change, is explainable, and rolls back in seconds. It cannot fix
retrieval recall, so it will not recover the non-slice gain — but it addresses the harm
the case actually identifies.

**Reasoning.** The case's finding is about a *slice regression*, not about ranking quality
in general. A correction aimed at the mechanism of the regression is cheaper and safer than
a ranker replacement aimed at the mean.

**Follow-up.** "What evidence would make you prefer the model?" If the slice regression
persists after the availability correction, the mechanism is retrieval/ranking, not
availability, and the model becomes the right tool.

**Acceptable alternative.** Excluding the slice entirely (no correction) is even simpler
and is the action this case recommends as step one.

**Red flag.** "Just retrain with more data" — it does not address the identified mechanism.

**Scoring.** 0 = no alternative; 1 = names one; 2 = ties it to the mechanism; 3 = states
the evidence that would change the preference.

**Answer support.** Sections 5c, 7, 10.

#### Interview 10 — How would you test causal value?

**Prompt.** Design the minimal experiment that establishes the candidate's effect.

**Answer.** Session-randomised, two-arm, 14-day window, pre-registered slice guardrail,
30-day outcome maturity, sequential monitoring with a pre-declared alpha-spending function,
and stratification by device class. The estimand is the difference in completed-order
conversion per eligible session, with the slice reported separately and the gate's
exclusion recorded as a protocol deviation rather than silently dropped.

**Reasoning.** Randomisation handles the seasonality and mix confounds that observational
comparison cannot. The pre-registered guardrail prevents post-hoc slice selection, which is
the main way this design gets abused.

**Follow-up.** "What if randomisation is impossible?" Then use a matched pre-period with a
difference-in-differences design and state that the parallel-trends assumption is untested;
do not present it as equivalent.

**Acceptable alternative.** A switchback design is defensible at very high volume but
introduces time-of-day confounding that must be modelled.

**Red flag.** Comparing pre-release and post-release periods with no control.

**Scoring.** 0 = "A/B test"; 1 = names randomisation; 2 = adds estimand and guardrail;
3 = adds alpha spending, maturity, and the deviation rule.

**Answer support.** Section 9.2.

#### Interview 11 — What red flag stops automation?

**Prompt.** You are asked to auto-ship rankers that pass the offline screen. What stops you?

**Answer.** Three: the pool is incumbent-derived, so the screen is biased in a direction
that depends on the change itself; the offline metric does not contain availability, so it
cannot see the dominant failure; and there is no demonstrated restore path. Any one of
these makes an automatic ship decision unsound. Automation would need candidate-complete
pooling, an online guardrail, and a passing drill — at which point the decision is cheap
enough to automate.

**Reasoning.** Automation moves the decision from "is this evidence sufficient" to "is this
rule sufficient". A rule built on a biased screen is not made safer by being automatic.

**Follow-up.** "Could you automate only the *hold* direction?" Yes — auto-hold on a
guardrail breach is asymmetric and safe, because the fallback is the incumbent.

**Acceptable alternative.** Automating with a mandatory human acknowledgement for slice
breaches is a reasonable middle ground.

**Red flag.** "The model is trained on millions of judgements" — volume does not fix
selection bias.

**Scoring.** 0 = "need more data"; 1 = one reason; 2 = two reasons; 3 = asymmetric
automation proposal.

**Answer support.** Sections 4, 6, 12.

#### Interview 12 — What evidence is still missing?

**Prompt.** List the gaps that block a real decision, in priority order.

**Answer.** (1) A candidate-complete judgement pool, because the screen's direction is
unknown. (2) A point-in-time availability audit, because the feature may be leaking. (3) A
timed rollback drill, because reversibility is a precondition, not a nice-to-have. (4)
Measured gate precision and recall, because the kept-order figure rests on a stipulated
86%. (5) Session-level assignment integrity, because a within-session split attenuates
everything.

**Reasoning.** Each gap maps to a specific number in Section 6 that would be recomputed;
none is a general "more data" request.

**Follow-up.** "Which is cheapest to close?" The rollback drill — a few hours of
engineering — and it is also a hard constraint, so it should be first operationally.

**Acceptable alternative.** Closing (1) first is defensible if the release cannot be
delayed; the pool is the largest threat to the *claim*.

**Red flag.** "We need a bigger model."

**Scoring.** 0 = generic; 1 = one specific gap; 2 = three specific gaps; 3 = prioritised
with the number each would change.

**Answer support.** Sections 4, 5d, 5e, 9.3, 14.2.

#### Interview 13 — Explain the masking ratio to a non-technical stakeholder

**Prompt.** No formulas. Why does +0.06 pp not mean "small improvement"?

**Answer.** "Imagine two groups of shoppers. The big group got a bit better and the small
group got much worse. Because the small group is small, the two effects nearly cancel in
the overall number. The overall number looks like a tiny gain, but it is really two large
changes pointing in opposite directions — and if the small group were slightly bigger, the
overall number would be a loss. So I do not want to use that overall number to decide."

**Reasoning.** The explanation preserves the mechanism (weighted sum, opposite signs), the
consequence (sign instability), and the decision (do not gate on the aggregate), without
notation.

**Follow-up.** "How much bigger would flip it?" About 18.7% instead of 9.5%.

**Acceptable alternative.** A two-bar chart with a weighted-average line is equally good
and is often better received.

**Red flag.** "The average is fine, the slice is an edge case" — that is the failure the
ratio was built to detect.

**Scoring.** 0 = repeats the number; 1 = mentions a slice; 2 = explains cancellation;
3 = also gives the flip point.

**Answer support.** Section 6, `SR-04`.

#### Interview 14 — Change one constraint and re-derive the action

**Prompt.** Suppose rollback is proven at 8 minutes but the pool cannot be re-judged for
six weeks. What now?

**Answer.** The reversibility constraint is satisfied, so a fenced slice test can proceed
now. The pool gap changes the *claim*, not the action: report the offline screen as
pool-biased and unverified, rely on the online slice guardrail as the decision statistic,
and pre-commit to re-pooling before any magnitude statement about relevance. The gate's
recall must be measured, because it replaces the offline screen as the safety argument.

**Reasoning.** Constraints bind different parts of the decision. Reversibility binds the
*action*; pool completeness binds the *offline claim*. Satisfying one does not satisfy the
other, and the correct response is to narrow what is claimed rather than to delay
everything.

**Follow-up.** "What if the online guardrail also cannot be measured because the slice is
too small?" Then the action is defer: with neither a valid offline screen nor a measurable
online guardrail, there is no evidence to act on.

**Acceptable alternative.** Running the test with the guardrail and accepting a wider
interval is defensible if the decision is cheap to reverse.

**Red flag.** Proceeding while claiming the offline number is validated.

**Scoring.** 0 = no change; 1 = one constraint addressed; 2 = both, with narrowed claims;
3 = also states what would force deferral.

**Answer support.** Sections 4, 6, 9.3, 10.

### 14.4 Claim ledger

| Claim ID | Exact claim and scope | Evidence type | Source/artifact | Supports | Does not establish |
|---|---|---|---|---|---|
| `CLM-01` | Masked aggregate with ratio 0.84 on constructed components | illustrative, arithmetic checked | `ILL-01`, Section 6 Step 2 | The aggregate is non-decisional here | Any company's conversion effect |
| `CLM-02` | Slice test detectable in ~4 h at 190k sessions/day | illustrative, derived | `ILL-02`, `SR-05` | Guardrail feasibility | That the guardrail threshold is correct |
| `CLM-03` | Gate keeps 1,904 vs 1,199 orders/day | illustrative, derived from stipulated recall | `ILL-03`, `SR-07` | A priced mitigation | Real gate performance |
| `CLM-04` | Rollback infeasible at 41 min vs 10 min budget | illustrative drill value | `ILL-01` | Reversibility precondition | Any real system's restore time |
| `CLM-05` | Pool derived from incumbent top-10 biases the screen | derived from pooling mechanics | `A-02`, FM-01 | Why offline magnitude is unverified | The size of the bias |

## 15. Quick Reference Card

| Field | Content |
|---|---|
| Decision | Exclude the 9.5% slice, prove rollback, then run a fenced test on the remainder — not a global release |
| Core arithmetic | `c_agg = s·c_slice + (1−s)·c_rest`; masking ratio `0.84`; aggregate `+0.06 pp` = `−0.53 pp` slice + `+0.12 pp` rest |
| Binding constraints | Point-in-time availability; session-scoped assignment; ≤10-minute demonstrated rollback; pre-registered 2% slice guardrail |
| Evidence status | All numbers ILLUSTRATIVE/DERIVED; no company data, no measured lift; pool bias unquantified (OPEN) |
| Failure to watch | Pool bias (FM-01), masked slice (FM-02), availability leak (FM-03), analyser mismatch (FM-04), request-scoped assignment (FM-05), asserted gate recall (FM-06), timeout reversion (FM-07), config-only rollback (FM-08) |
| First action | Record the decision contract and run the rollback drill twice |
| Stop condition | Guardrail breach, unmeasurable slice, or any irreversible effect without a proven restore path |
| Go deeper | `../Cheat_Sheets/01_Ecommerce_Search_Scorecard.md` for metric definitions; `06_Authorized_Near_Real_Time_Index.md` for the indexing contract |
