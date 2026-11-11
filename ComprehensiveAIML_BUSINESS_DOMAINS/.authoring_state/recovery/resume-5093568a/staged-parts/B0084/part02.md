## 5. Data, Tools & Method

### 5a Data Specification

The grain is one **bay-wave observation**: a single bay, in a single store, observed on a single date,
evaluated against a single planogram version, and dispositioned once in a wave. Every join in the
pipeline is an as-of join keyed on the decision timestamp, not a conventional equi-join on the latest
record. That single choice is what makes the rest of the case reproducible.

| Field group | Fields | Units / type | Notes |
|---|---|---|---|
| Keys | store_id, bay_id, wave_id, sku_id, planogram_version | string, integer | bay_id unique within store; version is a string, never a date |
| Entity | cluster_id, fixture_type, shelf_level, position_index | string, enum, int | position_index counts from the left fixture edge |
| Observation | observed_at, capture_method, observer_id, image_id | timestamp, enum, string | observed_at is event time, not ingest time |
| Layout observed | sku_id, facing_count, block_order, depth_units | int, int, int, int | facing_count integer and ≥ 0 |
| Layout planned | planned_facings, planned_order, planned_level, planned_depth | int, int, int, int | from the version bound to observed_at |
| Commercial | sku_share_of_category, category_sales_4w, unit_margin | fraction, currency, fraction | share used as the importance weight |
| Outcome (late) | verification_observed_at, verification_match | timestamp, boolean | never available at decision time |
| Decision | decided_at, action, reason_code, operator_id, labour_minutes | timestamp, enum, enum, string, decimal | labour_minutes measured, not estimated, for completed fixes |

- **Cut-off timestamp.** `decided_at` is the wave scheduling moment, taken at 06:00 local on the first
  day of the wave. Every field used in ranking must have `observed_at < decided_at`. The verification
  observation is by construction after the fix and is therefore excluded from all ranking inputs.
- **Population.** All bays in the 28 stores of the cluster that are active (not under refit), carry a
  planogram version effective on `decided_at`, and have at least one usable observation inside the
  14 days before `decided_at`. ILLUSTRATIVE: 28 stores × 380 bays = 10,640 bays, of which 380 are
  excluded for refit or missing version, leaving 10,260 eligible.
- **Keys and uniqueness.** The observation table is unique on (store_id, bay_id, observed_at,
  image_id). The decision table is unique on (store_id, bay_id, wave_id). A duplicate decision key
  is a defect, not a weighting, and section 12 treats it as a failure mechanism because the
  duplicate silently doubles a bay's weight in any aggregate.
- **Units.** Facings are unit-width counts. Position index is a rank from the left, not a distance.
  Dollars are local currency, unadjusted, at a 28% unit margin. Labour is minutes at $26.00 per hour
  loaded. Sales are per bay per week, not per store per period, because the ranking unit is the bay.
- **Not available at decision time.** Any post-fix verification, any planogram version published after
  `decided_at`, any sales figure for the wave being scheduled, any out-of-stock scan produced by a
  later replenishment cycle, and the crew's actual minutes on the bay. The last of these is the
  easiest to leak, because a completed-fix log from a prior wave is legitimately available and a
  completed-fix log from the *current* wave is not.
- **Known-missing by design.** Bays under refit have no observation and are not imputed. Bays whose
  only observation is older than 14 days are counted as stale, reported separately, and excluded
  from the primary numerator. Both exclusions are counted, never silently dropped, because the
  exclusion rate is the first thing a reviewer should ask about.

### 5b Tools & Stack

The stack is deliberately boring and every component is substitutable. No product version is
asserted, and no performance characteristic of any named vendor is claimed.

| Layer | Component | Responsibility | Failure it must survive |
|---|---|---|---|
| Capture | Handheld or fixed camera, image store | Produce a photograph per bay per wave with a stable identifier | Duplicate or mis-bound images |
| Version binding | Planogram service with effective dates | Resolve exactly one version per bay per timestamp | Overlapping effective dates |
| Layout extraction | Vision model plus human review queue | Emit facing_count, block_order, level per SKU | Low-confidence reads on occluded blocks |
| Prevalence | Walk-audit sampling job | Estimate true non-compliance per store per wave | Small samples, clustered bays |
| Scoring | Deterministic rule, versioned | Combine deviation, value, and confidence into a rank | Silent rule change between waves |
| Gate | Capacity and eligibility filter | Truncate the queue at fix capacity | Capacity drift, stale capacity |
| Decision log | Append-only store | Record rank, inputs, reason code, operator | Retrospective edits |
| Outcome join | Late-arriving verification job | Attach verification_match to the decision | Censoring, late arrival |

Three properties of this stack carry most of the reliability.

First, **the scoring rule is deterministic and versioned**. A change to the rule produces a new
version string, and the decision log records the version with every row. Without that, two waves are
not comparable and any improvement claim is unfalsifiable.

Second, **the gate is upstream of the human queue, not inside it**. Capacity is applied by the
system, which means the tail is truncated with a recorded rank rather than dropped by a supervisor
who has run out of crew. The recorded tail is what makes the deferred value visible in section 11.

Third, **the outcome join is separate from the scoring job** and runs on a different schedule. A
scoring job that waits for verification will either stall or, worse, read the verification it is
waiting for. Separate schedules make that mistake structurally harder.

### 5c Method Summary

The method has four steps, each of which can fail independently and each of which is validated
separately in section 9.

1. **Measure prevalence.** From a probability sample of bays per store, estimate the true proportion
   of non-compliant bays using a correction that inverts the detector's error rates. Without this
   step the queue's precision cannot be computed, because precision is a function of prevalence and
   of error rates together (section 6, derivation 2).
2. **Score each bay twice.** Once for severity — how far the observed layout deviates from planned,
   using a facing-weighted distance — and once for value — the margin at risk implied by that
   deviation under a stated space elasticity. The two scores rank differently and the difference is
   the subject of section 7.
3. **Allocate capacity.** Rank by expected value per labour minute, take bays in that order until the
   measured fix capacity is exhausted, and record the rank at which truncation occurred. This is a
   greedy allocation of a knapsack with unit sizes in bays but unequal fix times in minutes.
4. **Verify a sample.** Re-observe a random subset of fixed bays at the next wave and compute the
   realised persistence. Feed that persistence back into the value term rather than assuming it.

Baseline comparators are explicit and are not straw men. Three are used throughout: the unchanged
operator rule (fix whatever the store manager notices), the severity-only rule (fix the largest
deviation first, ignoring value), and the coverage rule (fix every flagged bay in walking order
until the crew runs out of time). Each is a real practice, each is cheap, and each beats the
proposed rule on at least one dimension that matters to somebody.

Alternatives considered but not tested: inferred compliance from sales anomalies alone, and a
threshold tuned to maximise F1 against the auditor's labels. The first is rejected because it cannot
distinguish a design error from an execution error, which is the distinction the fix budget depends
on. The second is tested in section 7 as the classification objective, and it is shown to allocate
capacity worse than the value objective while looking better on its own metric.

### 5d Provenance, Quality, Exclusions, Leakage

Four timestamps are retained for every record, and the difference between them is diagnostic rather
than decorative.

| Timestamp | Meaning | Typical defect it exposes |
|---|---|---|
| observed_at | Event time of the photograph or walk | Images re-ingested later inheriting the ingest date |
| ingested_at | When the record landed in the warehouse | Pipeline lag masquerading as freshness |
| version_effective_from | When the planogram version binds | Bay compared to a version not yet in force |
| decided_at | Wave scheduling moment | Ranking computed after the crew was briefed |

- **Quality gates before scoring.** Facing counts must be non-negative integers; the summed facing
  widths must fit the fixture; block_order must be a permutation without duplicates; every SKU in the
  observation must exist in the bound version's assortment or be flagged as unplanned. A record
  failing a gate is quarantined with a code, and the quarantine rate is reported per store. A store
  with a 20% quarantine rate has a capture problem, not a compliance problem, and treating its
  remaining records as representative is the mistake that makes the whole cluster look worse than it is.
- **Exclusions and their direction.** Refit bays are excluded (no shelf to observe). Stale
  observations (>14 days) are excluded from the primary numerator. Ambiguous versions are excluded
  from the numerator but counted in a separate bucket. None of these exclusions is random with
  respect to compliance: refits cluster in stores scheduled for investment, and stale observations
  cluster in stores with labour shortages, which are precisely the stores most likely to be
  non-compliant. The exclusions therefore bias the reported number *upward*, and the size of the
  bias is bounded only by the exclusion rate. That is why the exclusion rate is a reported metric
  and not a footnote.
- **Leakage checks, three of them.** (i) Rebuild the feature table using only records with
  `observed_at < decided_at` and confirm the ranking is identical; a difference means a later record
  reached the ranker. (ii) Confirm the bound version was effective before `decided_at`; a version
  published after the decision is the most common leak in reset windows. (iii) Confirm no
  verification field appears in the scoring job's input schema at all — schema-level exclusion is
  stronger than discipline, because discipline degrades under deadline.
- **Leakage that looks like skill.** If a wave's ranking is rebuilt after the wave closes, the
  rebuilt ranking will look better than the live one, because the rebuild can see which bays the
  crew actually fixed and which fixes stuck. This is not a subtle effect; it typically flatters the
  rule by several percentage points of apparent precision. The decision log exists so that the live
  ranking, with its version and timestamp, is the only thing that can be evaluated.

### 5e Reproducibility & Method-Selection Ledger

An evaluation is reproducible when a second analyst, given the same snapshot, gets the same queue.
That requires recording eight things, and omitting any one of them makes the result unreproducible
in a way that will not be discovered until someone tries.

| Item | Recorded value (ILLUSTRATIVE) | Why it changes the answer |
|---|---|---|
| Snapshot identifier | wave 2026-W45, cut 06:00 local | Different cut changes eligibility |
| Prevalence estimate and method | 0.22, walk sample, error-inverted | Drives queue precision |
| Detector error rates | Se 0.91, Sp 0.91 | Drive expected true and false fixes |
| Eligibility filter | active, version-bound, ≤14 days | Changes the denominator |
| Scoring rule version | rule-v3 (deviation × margin) | Two rules rank differently |
| Elasticity assumption | 0.35, constant, borrowed | Scales every dollar equally |
| Persistence assumption | 0.90 recovery, 4-week discount | Multiplies value per fix |
| Capacity | 65 fixes, 331.5 minutes | Sets queue length and truncation rank |

- **Rule selection argument.** The proposed rule was chosen over three alternatives on a stated
  criterion: expected net value per wave at fixed capacity. It was not chosen on classification
  accuracy, and section 7 shows that on classification accuracy it loses to a threshold tuned for F1.
  Recording the criterion is part of the ledger, because a later reviewer with a different criterion
  is entitled to a different rule — but only if the criterion is visible.
- **What cannot be reproduced.** The photographs themselves, if the image store is not archived with
  the wave. The case therefore requires image retention for at least two waves, and treats the
  image store as part of the record rather than as an operational cache.
- **Status of the numbers.** No executable run is claimed. The arithmetic in section 6 was performed
  with a calculator and is reproducible from the stated inputs; the inputs themselves are constructed.
  There is no measured performance claim anywhere in this manuscript, and section 11 says so in the
  result field rather than in a hedge.
