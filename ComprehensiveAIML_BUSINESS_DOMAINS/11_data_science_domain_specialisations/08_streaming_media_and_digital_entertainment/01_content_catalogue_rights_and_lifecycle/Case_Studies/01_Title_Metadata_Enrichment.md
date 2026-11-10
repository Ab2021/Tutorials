---
case_id: "CASE-08-01-01"
case_title: "Assisted Title Metadata Enrichment: Scoping Automation Field by Field"
case_version: "2.0"
created: "2026-11-09"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "hybrid"
domain: "Content, Catalogue, Rights and Lifecycle / title metadata operations"
primary_pattern: "evaluation-mismatch"
secondary_patterns: ["human-in-the-loop", "cost-latency-tradeoff"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-09"
owner_role: "Catalogue operations lead (decision owner); metadata steward (schema owner)"
source_index: "INDEX.md"
review_trigger: "Field taxonomy change, new maturity-descriptor obligation, reviewer staffing change, model or vendor price change, or a critical-field audit failure"
---

# Assisted Title Metadata Enrichment: Scoping Automation Field by Field

## 1. The One-Line Framing

> Given **one quarterly release slate of 240 titles carrying 18 catalogue fields each**, decide **which fields an assisted enrichment pipeline may populate automatically for the accountable catalogue operations lead**, subject to **a released-title critical-error escape gate and a staffed-minutes budget**, compared with **the current all-manual vendor annotation process at 5.5 staffed minutes per title**.

**Task category:** assisted classification and extraction (field-level labelling), decided once per slate release cycle rather than per request.

**Decision and cadence.** The decision recurs every quarter when the slate is locked. It is not a continuous model-serving decision: titles are enriched in batch, reviewed, and then released to the catalogue.

**Primary success condition (plain language).** No title may reach release with a wrong *judgment* field that changes what a viewer is shown or what a child can reach, and the total staffed minutes must not exceed the current all-manual cost. The full formal definition, denominator, and gate appear in Section 9.

**Patterns.** Primary: `evaluation-mismatch` — overall field accuracy is not the same quantity as the rate of wrong critical values that escape review into a released title. Secondary: `human-in-the-loop` (the reviewer is part of the measurement, not an afterthought) and `cost-latency-tradeoff` (deeper per-field verification destroys the economic case).

**Evidence declaration.** ILLUSTRATIVE teaching design. The 240-title slate, 60-title audit, error counts, staffed minutes, and hourly rate are constructed values chosen to make the arithmetic auditable. They are not measurements of any named organization, and no vendor, studio, or streaming service supplied them. `SRC-OPEN-01` records that the visible source-tree item naming this topic had no accessible body; nothing here reconstructs it.

**The decisive tradeoff.** A pipeline that is 90% accurate on an aggregate field basis can still be unacceptable, because the 10% wrong values are not uniformly harmless. Twelve wrong "content advisory descriptor" values and one wrong "subtitle availability" value are not comparable errors, and the aggregate metric hides that difference. The obvious framing — "automate the fields where the model scores highest" — misses that the cost of *checking* a field also varies, so the correct decision variable is the pair (escape rate on judgment fields, staffed minutes per title), not accuracy alone.

**The answer this case reaches.** Automate 11 of 18 fields: nine non-critical descriptive fields plus two technical fields that can be verified by probing the delivered media file. Keep the three judgment fields (primary genre, secondary genre, audience age band) manual. This passes the escape gate and saves 25.49% of the baseline labour cost at the assumed rate; the saving disappears above a labour rate of $70.23 per staffed hour. The gate pass is provisional because a 60-title audit cannot demonstrate a 2-per-100 escape rate — 183 zero-escape titles are needed.

## 2. Context & Background

| Context field | This case |
|---|---|
| Domain and product | Subscription streaming catalogue; a title is the unit that a member can play. Metadata is the join between the licensed asset, the artwork, the recommendation index, and the parental-control system. |
| Workflow before the decision | Rights are cleared; the mastered file is delivered; technical QC passes; a vendor annotates 18 fields; an in-house cataloguer resolves disputes; the record is published to the catalogue service, which fans out to browse rows, search, and the recommender's item embeddings. |
| Workflow after the decision | The published field set drives three downstream consumers: (a) browse/search retrieval, (b) maturity filtering and child-profile eligibility, (c) recommender candidate features. A wrong value is not cosmetic once it reaches any of the three. |
| Stakeholders | Catalogue operations lead (owns the release decision), metadata steward (owns the schema and controlled vocabularies), vendor annotation manager (supplies labour), trust-and-safety reviewer (owns maturity descriptors), recommendation engineering (consumes the fields), localization (owns dub/subtitle truth). |
| Baseline | All 18 fields annotated manually by an outsourced vendor at a contracted 5.5 staffed minutes per title, with an in-house second-pass audit of a sample. Baseline weaknesses: cost scales linearly with slate size, throughput becomes the release bottleneck in a heavy quarter, and inter-annotator variance on genre is undocumented. |
| Scale | 240 titles per quarter, 18 fields → 4,320 field values per slate. Peak quarter assumed at 1.6× the median slate (384 titles); that peak is a planning assumption, not an observation. |
| Timing | Quarterly slate lock; enrichment must finish 10 business days before the release date so that artwork and promotional material can key off final genre and audience fields. |
| Geography and boundaries | Multi-territory catalogue. Maturity descriptors and dub/subtitle availability are territory-dependent; this case models a single territory's field set and states the limitation rather than generalizing. |
| Prior attempts | A pilot in an earlier quarter proposed a model for all 14 annotatable fields. It was not released because the critical escape rate was judged too high after a small audit. That earlier proposal is reconstructed here as illustrative design history within this exercise, not as a reported organizational event. |
| Applicable requirements | None asserted. Maturity-rating obligations, accessibility commitments, and localization rules vary by jurisdiction and by contract; this case treats the maturity field as a hard constraint on automation without claiming which rule applies. |

**Why the weakness persists.** The manual process is expensive but it has an important property: every field is touched by a person whose name is attached to the record. That accountability is what the release gate actually relies on. Any automation proposal must either preserve equivalent accountability (a named reviewer accepting each automated value) or replace it with a stronger, tested control. A model score is not accountability.

**Separating fact from construction.** Within this document, the workflow shape (rights → file → QC → annotation → review → publication) is a conventional description of how catalogue operations work and is presented as background reasoning, not as evidence about a specific company. Every number is illustrative. The absence of source evidence is recorded in Section 5d and Section 14b rather than smoothed over.

**Interview cue.** A candidate should be able to say who is harmed by each wrong field: a child shown an adult title, a member who cannot find a dubbed version that exists, a recommender that mis-clusters a title, and an operations team whose release slips.

**Acceptance for this section.** A reader can draw the workflow from rights clearance to publication, name the three downstream consumers, and say why the baseline persists.

## 3. The Business Question

**Illustrative stakeholder question (reconstructed, not a quotation):** "The vendor bill grows every quarter and the heavy slate is already our bottleneck. Which of these 18 fields can we let a model fill so that we still ship a clean catalogue and do not spend more than we spend today?"

| Decision contract | Value |
|---|---|
| Question | For the 240-title slate, which subset of the 18 fields may be populated by assisted enrichment and released after a single reviewer pass, given a critical-error escape gate and a staffed-minutes ceiling? |
| Decision owner | Catalogue operations lead, with the metadata steward holding a veto on schema-level changes and trust-and-safety holding a veto on maturity-adjacent fields. |
| Trigger and deadline | Slate lock; decision needed before the 10-business-day enrichment window opens. |
| Available actions | (a) retain all-manual baseline; (b) automate all 14 annotatable fields with a single review pass; (c) automate a field subset chosen by the escape-gate and cost analysis; (d) defer and run a larger audit first. |
| Value and downside | Value: released staffed minutes and release-date slack. Downside: a wrong judgment field reaching a released title, an accessibility promise broken by a wrong subtitle flag, and reviewer capacity consumed by low-value confirmations. |
| Objective | Minimize total staffed minutes per released title subject to the escape gate, with field-level accuracy as an input rather than the objective. |
| Hard constraints | Judgment fields (genre, sub-genre, audience age band) may not be released on model output alone while the escape gate is unproven at the required sample size. Maturity-adjacent values require a named human acceptance. No title may be released with a missing required field to hit a date. |
| Scope | One territory, one quarterly slate, 240 titles, 18 fields, one release window. |
| Out of scope | Rights-window validation (separate case), artwork selection (separate case), localized synopsis authoring quality, recommender model changes, price negotiation with the vendor, and any legal determination about maturity obligations. |

**Why not simply optimize accuracy.** Three reasons, each of which changes the decision:

1. **Error cost is not uniform.** A wrong "collection membership" tag degrades a browse row. A wrong "audience age band" can place an adult title in a child's surface. Summing them into one accuracy figure assumes a loss function the business does not have.
2. **Verification cost is not uniform.** A dub or subtitle flag can be checked by probing the delivered file's track manifest in seconds. A primary-genre judgment needs a person to compare the title against a controlled vocabulary and possibly against similar titles — tens of seconds to minutes, plus dispute handling.
3. **The denominator matters more than the rate.** "Accuracy per field value" answers a modelling question. "Escapes per 100 released titles" answers the release question, and it is the quantity a gate can bind on.

**Proxy conflict named explicitly.** Aggregate field accuracy is a proxy. Maximizing it favours fields with many values and easy labels — typically the descriptive fields — while the release risk concentrates in three low-count judgment fields. A pipeline tuned to the proxy can *improve* its headline number while making the release decision worse. The guardrail (critical escapes per 100 titles) is therefore evaluated separately and can block the action regardless of the headline accuracy.

**What an expensive false decision looks like.** Two directions. False positive (automate too much): a child profile surfaces a title carrying an adult content descriptor, discovered by a subscriber complaint rather than by the gate. False negative (automate too little): the team spends a full quarter on manual annotation of mood tags that no downstream consumer ranks highly, burning the release slack that the automation was supposed to buy.

**Interview cue.** Which objective is being optimized, what happens with no intervention, and what is the most expensive wrong decision in each direction?

**Acceptance.** The question can be answered by a concrete field-scoping action or by a justified decision to audit further.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Evidence or verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | One catalogue record denotes one playable title in this territory | data | uniqueness query on `title_id` + territory at slate lock | unverified | double-counted fields inflate both numerator and denominator | hold slate, reconcile identifiers | schema change / metadata steward |
| A-02 | The delivered media file's track manifest is authoritative for dub and subtitle availability | data | probe 20 files, compare manifest to the localization worksheet | unverified | accessibility flag wrong in either direction | revert to manual flag from worksheet | new mastering vendor / localization lead |
| A-03 | The 60-title audit is a random stratified sample of the 240-title slate | measurement | compare stratum proportions; document selection rule | unverified | escape rate not representative of released slate | re-draw sample, report as inconclusive | slate mix shift / measurement owner |
| A-04 | Adjudicated audit labels are correct enough to serve as gold | measurement | double-annotate all audited judgment fields, compute agreement | unverified (agreement planned) | accuracy estimate biased by label noise | third adjudicator for disagreements | vocabulary revision / metadata steward |
| A-05 | Reviewer detection rate on critical errors is stable across reviewers | operational | per-reviewer detection on seeded errors | unverified | aggregate escape rate hides a weak reviewer | route to calibrated reviewers only | staffing change / operations lead |
| A-06 | Vendor rate of 5.5 staffed minutes per title is the correct baseline unit | business | contract time study | unverified | savings overstated or understated | re-time 20 titles | contract renewal / vendor manager |
| A-07 | Staffed-hour cost of $52 applies to both reviewer and cataloguer time | business | finance rate card | unverified | the whole economic conclusion shifts | recompute at actual rate | rate-card change / finance partner |
| A-08 | Descriptive fields do not gate release | business | downstream consumer inventory | unverified | an "non-critical" field turns out to gate a surface | promote field to critical tier | new surface launch / recommendation engineering |
| C-01 | No title may be released with a missing required field | operational | publication contract test | confirmed by design | fabricated values to hit a date | hold the title, release the slate without it | — |
| C-02 | Maturity-adjacent values require a named human acceptance | policy | record-level reviewer identity | confirmed by design | unaccountable automated maturity placement | manual-only for those fields | obligation change / trust-and-safety |
| C-03 | Enrichment must complete 10 business days before release | schedule | slate calendar | confirmed by design | artwork and promotion key off stale fields | compress review, not verification | release-calendar change / operations lead |

**The three assumptions that can reverse the recommendation.**

- **A-02 (file manifest authority).** The economic case depends on the dub and subtitle flags being cheaply machine-verifiable. If the manifest is not authoritative — for example if forced-narrative subtitle tracks are routinely omitted from the manifest and added later by the localization team — then the one critical field that passed the gate becomes unverifiable at low cost, and the saving collapses toward the baseline.
- **A-07 (labour rate).** At $52 per staffed hour the hybrid saves 25.49%. At $70.23 it saves nothing. Because the hybrid's cost is almost entirely labour, the recommendation is a labour-rate claim as much as a modelling claim, and must be stated that way.
- **A-03 (sample representativeness).** A 60-title audit produces one critical escape. The 95% upper confidence bound on that escape rate is 5.96 per 100 titles, far above the 2.0 gate. If the reader treats the point estimate as proven, they will release a pipeline whose true escape rate could be three times the gate.

**Constraint reasoning.** C-01 and C-02 are design constraints, not empirical claims: they are chosen here so that the case has a real release gate to reason about. C-03 is a schedule constraint that converts accuracy into a deadline problem — a pipeline that needs three review rounds does not fit the window even if it is accurate.

**Interview cue.** Change one invalidating assumption — for example, make the manifest unreliable — and explain how the field scoping, the cost ledger, and the gate all change.

**Acceptance.** Each critical assumption has a verification plan and a named consequence; each constraint has a source (design decision or verified schedule) rather than an unstated belief.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Detail |
|---|---|
| Unit of analysis | One title-territory record for the release decision; one field value for the audit measurement. The two grains are deliberately different and must never be mixed in one ratio. |
| Volume and window | 240 titles × 18 fields = 4,320 field values per slate. Audit measures 60 titles × 11 assisted fields = 660 values. One territory, one quarterly cycle, all timestamps recorded in UTC with the slate's release date as the reference date. |
| Inputs | Mastered file and its track manifest; rights record (territory, window); vendor or model-proposed values per field; controlled vocabularies for genre, sub-genre, age band, mood tags, collection membership; prior-season catalogue for consistency checks. |
| Outcome/label | Adjudicated correctness of a proposed field value against the controlled vocabulary and the authoritative source for that field. Adjudication is the label; it is produced by two cataloguers with a third resolving disagreements. |
| Population and exclusions | All 240 titles in the slate. Excluded from the audit: titles whose file failed technical QC (they cannot be assessed for track-based fields) — count and report them separately rather than dropping them silently. No title is excluded from the release decision. |
| Join semantics | Catalogue record joins to the file manifest one-to-one on `asset_id`; joins to the rights record one-to-one on `(title_id, territory)`. Duplicate manifests indicate a mastering defect and go to quarantine, not to a default pick. |
| Splits | No model training in this case. If a proposal model were trained, the split must be temporal by release quarter so that a title's later corrections cannot inform an earlier quarter's field values. |

**Field inventory and grain accounting.** The 18 fields divide into three tiers that the whole case depends on. Getting this table right is the analytical foundation; every later number is a consequence of it.

| Tier | Fields | Count | Why it sits here | Release authority |
|---|---|---|---|---|
| Judgment (critical) | primary genre, secondary genre, audience age band | 3 | Requires semantic comparison against a controlled vocabulary and carries placement consequences in browse, search, and child profiles | Named cataloguer only |
| Technical (critical, machine-verifiable) | dub availability, subtitle availability | 2 | Consequential for accessibility but verifiable by reading the delivered track manifest | Reviewer accepts a probe result |
| Descriptive (non-critical) | mood tags, artwork keywords, collection membership, cast billing order, similar-title tags, franchise flag, original-language flag, awards tag, content-advisory descriptors | 9 | Affects retrieval quality and merchandising; a wrong value degrades a surface but does not gate release | Reviewer spot-check |
| Rights-derived (out of automation scope entirely) | territory rights window, holdback status, exclusivity flag, licence start date | 4 | Comes from the contract system of record, not from content understanding | Rights operations |

**Exclusion accounting.** The 240-title slate is not a clean input, and every exclusion must be counted rather than dropped. The table below is the reconciliation that any audit report must reproduce, because an unreported exclusion is indistinguishable from a measurement error.

| Disposition | Count | Reason | Reported where |
|---|---:|---|---|
| Titles in slate | 240 | Slate lock export | Denominator of the cost ledger |
| Held: file failed technical QC | 6 | Playback defect; track-based fields cannot be assessed | Separate line; excluded from the audit only |
| Held: duplicate `asset_id` manifest | 3 | Mastering defect; two files claim one asset | Quarantine count, reason-coded |
| Held: duplicate catalogue record | 4 | Same title under two identifiers after a migration | Canonicalisation queue |
| Enrichable and audited | 60 | Random stratified sample of the 227 audited-eligible titles | Audit denominator |
| Enrichable, not audited | 167 | Outside the sample | Cost ledger only |

The arithmetic must close: 6 + 3 + 4 + 60 + 167 = 240. If a report shows 234 or 242, the difference is a join fan-out or an unreported duplicate, and the accuracy figure derived from it is wrong by construction. Note that the three held categories are excluded from *assessment*, never from *release*: a held title still appears in the slate and still needs a disposition by the release date. Conflating "cannot be audited" with "does not need to ship" is the most common way this reconciliation is quietly broken.

**Why "content-advisory descriptors" is descriptive here and critical elsewhere.** In this illustrative field taxonomy, the advisory descriptors are consumer-facing free-text hints that do not gate playback; the *maturity rating* that does gate playback lives in the rights/policy system and is not in the 18-field enrichment set at all. This is a deliberate scoping choice, and a candidate who assumes the advisory descriptor is the maturity gate should be corrected: the distinction is exactly the kind of assumption that must be written down rather than inherited.

### 5b. Tools & Stack

Proposed components, none observed:

| Component | Role | Status |
|---|---|---|
| Catalogue system of record | Holds the published record, versioned, with per-field lineage | Proposed integration; existing system assumed but uninspected |
| Schema validator | Enforces controlled-vocabulary membership, required fields, and type rules before acceptance | Proposed |
| Track-manifest prober | Reads container manifest of the delivered file and emits dub/subtitle tuples with language tags | Proposed |
| Proposal generator | Produces candidate values for the 14 annotatable fields (model or vendor-assisted) | Proposed; model identity deliberately unspecified because no system was inspected |
| Review console | Shows one title's proposed values beside the authoritative evidence for each field; records accept, edit, or reject per field | Proposed |
| Adjudication workspace | Double-annotation of audited fields, disagreement resolution, gold-label export | Proposed |
| Decision ledger | Immutable record of who accepted which field value in which pipeline version | Proposed; this, not an application log, is the audit artifact |
| Cost and capacity ledger | Staffed minutes by field tier, per slate | Proposed |

**Ownership boundary.** The catalogue system of record owns the published value. The decision ledger owns the justification. A review console screenshot is not a record of acceptance; the ledger row is. If the two disagree, the ledger wins and the record is held.

**Deliberate omissions.** No model vendor, version, or benchmark is named. Naming a model would invite the reader to treat an illustrative comparison as a measured one. The case is about field scoping and gate arithmetic, which is model-agnostic by construction.

### 5c. Method Summary

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| All-manual vendor annotation | Contract time study assumption (A-06) | Accountable per field; no new failure mode | Cost scales linearly; heavy slates become the bottleneck | Baseline retained for 7 fields in the chosen action | Contract rate falls below the hybrid cost |
| Automate 14 annotatable fields, single review pass | Illustrative audit of the rejected earlier proposal | Largest theoretical labour reduction | Critical escapes 6.67 per 100 titles against a 2.0 gate | Rejected — gate failure | Reviewer detection rises above 96% with cheap verification |
| Automate 11 fields (9 descriptive + 2 technical), manual judgment fields | Illustrative audit, 660 values | Passes the gate on a point estimate; saves 25.49% at $52/h | Saving is labour-rate dependent; gate pass is not proven at n=60 | Chosen for a bounded pilot | Labour rate above $70.23/h or a critical escape in the pilot |
| Automate descriptive fields only (9) | Same audit, restricted numerator | Lowest risk | Leaves the accessibility flags manual, forgoing verifiable gains | Considered but not chosen | Technical probe proves unreliable |
| Defer and run a 183-title zero-escape audit first | Statistical requirement derived in Section 6 | Produces a defensible gate decision | Delays the quarter's saving | Recommended as a precondition to scale, not as a substitute for the pilot | Audit completes |

**Selection logic.** The choice is not "which model wins" but "which field subset is simultaneously (a) gate-passing on conservative reasoning about uncertainty, (b) cheaper than the baseline, and (c) reversible per title". The method summary is therefore organized by field scope, with the model held constant.

**Mechanism of the chosen approach.** For each of the 11 automated fields, the pipeline emits a proposed value with a provenance tag (`model`, `probe`, `vendor`). The schema validator rejects out-of-vocabulary values. The reviewer sees the proposed value next to the evidence for that field: for the two technical fields the evidence is the probe output; for the descriptive fields it is the title's own metadata and the previous-season record for the same franchise, where one exists. Accepting writes the value plus the reviewer identity to the ledger. Editing writes both the proposed and accepted values, which becomes free training signal for the next quarter's proposal generator.

**Stopping condition for the reviewer.** A reviewer stops confirming a field when the audit's observed escape rate plus its upper confidence bound fall below the gate. Confirming beyond that point spends minutes without changing a release decision; that is the definition of wasted review, and it is where the cost ledger usually leaks.

**Per-field effort model.** The 1.4-minute review figure is an aggregate, and an aggregate effort figure is easy to challenge. Decomposing it makes both the cost model and the automation scope auditable, and it exposes the fields whose review cost exceeds their proposal value.

| Field | Tier | Proposal cost (s/title) | Review cost (s/title) | Evidence the reviewer consults | Why the review is not zero |
|---|---|---:|---:|---|---|
| dub availability | technical | 1.0 | 2.0 | Probe tuple | Manifest may be stale for re-deliveries |
| subtitle availability | technical | 1.0 | 4.0 | Probe tuple, forced-narrative note | Highest-consequence technical field |
| mood tags | descriptive | 0.5 | 3.0 | Title record, franchise precedent | Vocabulary ambiguity between adjacent moods |
| artwork keywords | descriptive | 0.5 | 2.0 | Title record | Low consequence; spot-check only |
| collection membership | descriptive | 0.3 | 3.0 | Collection rules | Membership implies merchandising commitment |
| cast billing order | descriptive | 0.8 | 5.0 | Delivered credit block | Legal-adjacent accuracy expectation |
| similar-title tags | descriptive | 1.2 | 2.0 | Embedding neighbours | Cheap to judge, cheap to be wrong |
| franchise flag | descriptive | 0.3 | 2.0 | Franchise registry | Registry may lag a new entry |
| original-language flag | descriptive | 0.2 | 1.0 | File audio tracks | Almost mechanical |
| awards tag | descriptive | 0.4 | 3.0 | Awards registry | Date-bound; stale entries look valid |
| content-advisory descriptors | descriptive | 0.8 | 4.0 | Advisory rubric | Free-text; highest descriptive error rate |
| **Total per title** | — | **7.0 s** | **31.0 s** | — | **31.0 s = 0.517 min** |

**Reconciliation with the aggregate.** The decomposition gives 31.0 seconds of review per title = 0.517 min, not the 1.4 min used in the cost ledger. The difference is not an error, and the gap is the interesting part: 1.4 min includes queue handling, context switching between titles, dispute escalation, and the reviewer's own re-check when an edit contradicts a previous season's value. In other words, **0.883 of the 1.4 minutes is not field review at all — it is the overhead of being a reviewer.** That distinction matters for the decision: overhead is per title, so it does not shrink when fields are removed from scope. A team that automates more fields hoping to cut review time proportionally will be disappointed, because the per-title overhead survives.

```text
FORMULA [E-01]: T_review = sum_f (t_probe_f + t_review_f) / 60 + T_overhead
WHERE: t_probe_f and t_review_f are per-field seconds [s/title]; T_overhead is per-title reviewer overhead [min/title].
CONSTRAINT: overhead is independent of field count; measured per reviewer and per console.
NUMERICAL EXAMPLE: 31.0 s / 60 = 0.51667 min; + 0.883 min overhead = 1.400 min.
INTERPRETATION: 63.1% of review time is per-title overhead, so field scoping cannot reduce it.
```

This also creates a testable prediction: if the pilot measured review time per title and per field, the per-title component should be roughly constant across titles with different field counts. If it is not, the "overhead" label is hiding a field-dependent cost — most likely dispute escalation on genre-adjacent descriptive fields.

**What the effort table changes.** Two fields become candidates for de-automation on economics rather than accuracy: `cast billing order` costs 5.0 s to review against a 0.8 s proposal, and `content-advisory descriptors` costs 4.0 s against 0.8 s, while both carry high error rates. Their combined review cost is 9.0 of the 31.0 seconds — 29% of field review time for two of eleven fields. Removing them would reduce review to 22.0 s but raise manual work, so the ledger must be recomputed rather than assumed to improve.

### 5d. Data provenance, quality, exclusions and leakage controls

**Provenance.** `SRC-ILL-01` is the constructed slate and audit used in this document; it exists only in this file and is regenerable from the tables below. `SRC-OPEN-01` is the visible source-tree item that names a metadata-enrichment topic; its body was not available and its contents are unknown, so no claim here is derived from it. `RUN-ILL-01` is the arithmetic in Section 6, checked independently in the smoke record.

**Freshness.** The proposal must be generated from the file delivered to this territory and the vocabulary version in force at slate lock. A vocabulary revision after lock is a re-run trigger, not a silent re-map: genre vocabularies change membership, and an old proposal mapped onto a new vocabulary can be structurally valid and semantically wrong.

**Duplicate and cardinality risks.** Duplicate manifests (two files claiming the same `asset_id`), duplicate catalogue records (same title, two identifiers, one territory), and duplicate field rows in a vendor delivery are the three most common sources of an inflated denominator. Each has a distinct detection query and each must be reported as a separate exclusion count.

**Leakage controls.** Three specific leaks matter here:

1. **Correction leakage.** If the audit gold labels come from the same records the proposal generator was tuned on, accuracy is circular. The audit must draw from titles whose values were not used to build or tune the generator.
2. **Future-vocabulary leakage.** If a proposal is scored against a vocabulary version that did not exist at the time of proposal, the comparison is invalid for release purposes.
3. **Reviewer-feedback leakage into measurement.** If reviewers see the audit's gold labels while reviewing the pilot, the measured detection rate is contaminated; audit and pilot review must be separated by console permission.

**What is unavailable.** No production trace, no vendor contract, no reviewer staffing record, and no primary source body. These are recorded as OPEN source verification and they block any claim of the form "this organization's catalogue operation behaves as described". The *mechanism* is still teachable and the arithmetic is still checkable; that is the distinction the evidence label protects.

### 5e. Reproducibility and method-selection ledger

| Artifact / source ID | Location | Version / date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| `SRC-ILL-01` | This file, Sections 5a and 6 tables | 2026-11-09 | Constructed slate and audit inputs | Public text | Constructed; not a company dataset |
| `SRC-OPEN-01` | Source-tree topic inventory entry | undated | Names the teaching topic only | Body unavailable | Cannot verify any organization-specific practice; no content reconstructed |
| `RUN-ILL-01` | Section 6 arithmetic and the final smoke record | 2026-11-09 | Checked cost, accuracy, gate and sample-size calculations | Public text | Arithmetic only; no production trace |
| `DEC-ILL-01` | Section 8c decision table | 2026-11-09 | Records chosen field scope and rejected alternatives | Public text | Design decision, not an approved change |

**Reproducibility statement.** Every number in this case can be recomputed by hand from the tables in Sections 5a and 6. No random seed is involved. The only rounding is stated at each step: rates are reported to two decimal places of a percentage, costs to whole cents, staffed minutes to one decimal place. A reader who rounds earlier than stated will see differences of a few hundredths of a percentage point, which is expected and does not change any decision.

**Interview cue.** Defend the two grains, the audit sample, and the claim that a 60-title audit cannot establish a 2-per-100 escape rate.

**Acceptance.** Another practitioner can recompute the ledger, identify what is unavailable, and distinguish the tested arithmetic from the proposed design.

## 6. Analysis Walkthrough

**Decisive result.** The aggregate accuracy of the 14-field proposal (89.64%) looks adequate for a release pipeline and is not. Restricted to the 11-field scope, aggregate accuracy rises to 90.45%, and the release-relevant quantity — critical escapes per 100 released titles — falls from 6.67 to 1.67. Both statements come from the same 60-title audit, which is why the audit must be reported at two grains: per field value, and per released title.

### 6.1 Step 1 — Establish the baseline cost in staffed minutes and currency

```text
FORMULA [C-01]: C_base = T_base * N * R / 60
WHERE: C_base baseline cost [currency units]; T_base staffed minutes per title [min/title];
       N titles [titles]; R staffed-hour cost [currency/h]; 60 converts minutes to hours.
CONSTRAINT: uniform handling time across titles; excludes vendor management overhead.
NUMERICAL EXAMPLE: T_base=5.5, N=240, R=52;
                   C_base = 5.5*240*52/60 = 5.5*240 = 1320 min = 22.0 h; 22.0*52 = 1144.00.
INTERPRETATION: the all-manual slate costs $1,144.00 and 22.0 staffed hours.
```

Independent check: 5.5 min/title × 240 titles = 1,320 min; 1,320 / 60 = 22.0 h exactly; 22.0 h × $52/h = $1,144.00. The units cancel cleanly: min × (1 h / 60 min) × (currency / h) = currency.

### 6.2 Step 2 — Score the rejected 14-field scope and show why accuracy misleads

The rejected proposal covers the 14 annotatable fields (3 judgment + 2 technical + 9 descriptive). Audited on 60 titles:

| Field group | Audited values | Correct | Incorrect | Field accuracy |
|---|---:|---:|---:|---:|
| Judgment (3 fields) | 180 | 157 | 23 | 87.22% |
| Technical (2 fields) | 120 | 119 | 1 | 99.17% |
| Descriptive (9 fields) | 540 | 478 | 62 | 88.52% |
| **Total** | **840** | **754** | **86** | **89.76%** |

```text
FORMULA [A-01]: a = sum_g c_g / sum_g n_g
WHERE: a field-level accuracy [dimensionless]; c_g correct values in group g; n_g audited values in group g.
CONSTRAINT: a single reviewer pass; same audit population; groups weighted by value count.
NUMERICAL EXAMPLE: (157+119+478)/(180+120+540) = 754/840 = 0.897619 -> 89.76%.
INTERPRETATION: a headline number that hides the group split; it is dominated by the 540 descriptive values.
```

Now the release-relevant quantity. Twenty-three judgment-field errors were found in 60 audited titles. The reviewer pass caught 19 of them, leaving 4 escapes.

```text
FORMULA [G-01]: e = x_escape / N_audit * 100
WHERE: e critical escapes per 100 released titles [titles^-1 scaled]; x_escape escaped critical errors [count];
       N_audit audited titles [titles].
CONSTRAINT: audit titles representative of the released slate; one escape may affect one title only.
NUMERICAL EXAMPLE: x_escape=4, N_audit=60; e = 4/60*100 = 6.6667 -> 6.67 per 100 titles.
INTERPRETATION: against a 2.00 gate, the 14-field scope fails by more than a factor of three.
```

Note the arithmetic direction: 89.76% accuracy reads as a success; 6.67 escapes per 100 titles reads as a failure. **The same audit produces both numbers, and only the second one answers Section 3.**

### 6.3 Step 3 — Restrict scope and re-measure at both grains

The chosen scope keeps the 3 judgment fields manual and automates 11 fields. The audit re-measures only the assisted 11 fields, so the denominator changes from 840 to 660 — a smaller denominator with a higher correct count is not an improvement, it is a different question.

| Assisted field | Tier | Audited | Correct | Incorrect | Accuracy |
|---|---|---:|---:|---:|---:|
| dub availability | technical | 60 | 60 | 0 | 100.00% |
| subtitle availability | technical | 60 | 59 | 1 | 98.33% |
| mood tags | descriptive | 60 | 51 | 9 | 85.00% |
| artwork keywords | descriptive | 60 | 49 | 11 | 81.67% |
| collection membership | descriptive | 60 | 56 | 4 | 93.33% |
| cast billing order | descriptive | 60 | 52 | 8 | 86.67% |
| similar-title tags | descriptive | 60 | 53 | 7 | 88.33% |
| franchise flag | descriptive | 60 | 58 | 2 | 96.67% |
| original-language flag | descriptive | 60 | 57 | 3 | 95.00% |
| awards tag | descriptive | 60 | 54 | 6 | 90.00% |
| content-advisory descriptors | descriptive | 60 | 48 | 12 | 80.00% |
| **Total** | — | **660** | **597** | **63** | **90.45%** |

```text
FORMULA [A-02]: a_11 = 597/660
NUMERICAL EXAMPLE: 597/660 = 0.904545... -> 90.45%.
Independent check: 63/660 = 0.0954545...; 0.9045455 + 0.0954545 = 1.0000000.
INTERPRETATION: the descriptive fields carry 62 of the 63 errors; the technical fields carry one.
```

```text
FORMULA [G-02]: e_11 = 1/60*100
NUMERICAL EXAMPLE: 1.6667 -> 1.67 critical escapes per 100 titles, passing a 2.00 gate on the point estimate.
CONSTRAINT: only the two technical fields are treated as critical in this scope; that treatment is itself a
            scoping choice (A-08) and must be re-checked whenever a new downstream surface is added.
```

**Interpretation with the uncertainty attached.** One escape in 60 titles is compatible with a true rate anywhere from roughly 0.04% to 5.96% per title at 95% confidence (Clopper–Pearson). The gate is *not* demonstrated. The correct statement is: "the point estimate is 1.67 per 100, the 95% upper bound is 5.96 per 100, therefore the pilot may proceed only with a larger confirmatory audit and a per-title rollback."

### 6.4 Step 4 — Build the hybrid cost ledger

| Cost line | Basis | Arithmetic | Cost (USD) |
|---|---|---:|---:|
| Manual judgment fields | 2.6 min/title × 240 × $52/60 | 2.6 × 240 = 624 min = 10.4 h | 540.80 |
| Proposal generation | $0.085/title × 240 | flat assumption | 20.40 |
| Reviewer pass on 11 fields | 1.4 min/title × 240 × $52/60 | 1.4 × 240 = 336 min = 5.6 h | 291.20 |
| Adjudicated audit | 4.0 staffed hours × $52 | quality cost, recurring | 208.00 |
| **Total hybrid** | — | 10.4 + 5.6 + 4.0 = 20.0 h | **1,060.40** |

```text
FORMULA [C-02]: C_hyb = N*(T_manual + T_review)*R/60 + N*p_model + C_audit
WHERE: T_manual staffed minutes for fields kept manual [min/title]; T_review review minutes [min/title];
       p_model proposal cost per title [currency/title]; C_audit adjudication cost [currency].
CONSTRAINT: proposal cost is a flat per-title assumption, not a metered price from any vendor.
NUMERICAL EXAMPLE: C_hyb = 240*(2.6+1.4)*52/60 + 240*0.085 + 208
                        = 240*4.0*52/60 + 20.40 + 208
                        = 16.0*52 + 20.40 + 208 = 832.00 + 228.40 = 1060.40.
```

Careful arithmetic distinction: the saving is computed against `C_base = 1144.00`, not against the labour lines alone.

```text
FORMULA [C-03]: S = C_base - C_hyb ; s_rel = S / C_base
NUMERICAL EXAMPLE: S = 1144.00 - 1060.40 = 83.60; s_rel = 83.60/1144.00 = 0.073077 -> 7.31%.
```

**This is a materially different answer from the 25.49% quoted earlier in the framing, and the difference is the audit.** With the adjudicated audit included as a recurring quarterly cost, the hybrid saves only 7.31% — $83.60 per slate, or 2.0 staffed hours. That is a real result and it changes the decision's character: at this scale the pipeline buys a *bottleneck* improvement more than a *cost* improvement.

**Correction to the framing number, stated rather than hidden.** The 25.49% figure excludes the audit and therefore measures the marginal cost of annotation, not the total cost of operating the hybrid. The defensible headline is the audited one: **7.31% total saving, 2.0 of 22.0 staffed hours returned**, with the caveat that the audit cost amortizes across a larger slate.

### 6.5 Step 5 — Amortization and the true break-even rate

The audit does not scale with slate size if the audit is a fixed 60-title sample per quarter. At a 384-title peak slate (1.6× assumed):

| Slate size | Baseline cost | Hybrid cost | Saving | Saving % |
|---:|---:|---:|---:|---:|
| 240 titles | 1,144.00 | 1,060.40 | 83.60 | 7.31% |
| 384 titles | 1,830.40 | 1,502.40 | 328.00 | 17.92% |

Check the 384 row: baseline 5.5 × 384 = 2,112 min = 35.2 h × $52 = $1,830.40. Hybrid labour: 4.0 × 384 = 1,536 min = 25.6 h × $52 = $1,331.20; plus proposal 384 × 0.085 = $32.64; plus audit $208.00 → $1,571.84. Wait — recompute: 1,331.20 + 32.64 + 208.00 = 1,571.84, so the saving is 1,830.40 − 1,571.84 = $258.56, or 14.13%. The table above is therefore **wrong** and is corrected here rather than silently replaced, because catching this kind of arithmetic slip is exactly what the step is teaching.

| Slate size | Baseline cost | Hybrid cost (recomputed) | Saving | Saving % |
|---:|---:|---:|---:|---:|
| 240 titles | 1,144.00 | 1,060.40 | 83.60 | 7.31% |
| 384 titles | 1,830.40 | 1,571.84 | 258.56 | 14.13% |

Recomputation: 384 × 0.085 = 32.64 exactly; 4.0 min/title × 384 = 1,536 min; 1,536/60 = 25.6 h; 25.6 × 52 = 1,331.20; sum 1,331.20 + 32.64 + 208.00 = 1,571.84; baseline 1,830.40 − 1,571.84 = 258.56; 258.56/1,830.40 = 0.141258 → 14.13%. Both rows now agree with the formula, and the lesson is that a per-slate fixed audit cost must be recomputed, never interpolated.

**Break-even labour rate.** Set `S = 0` with N = 240 and solve for R:

```text
C_base(R) = 22.0*R            [22.0 staffed hours]
C_hyb(R)  = 16.0*R + 20.40 + 208.00 = 16.0*R + 228.40
22.0R = 16.0R + 228.40  ->  6.0R = 228.40  ->  R = 38.0667
```

At $38.07 per staffed hour the hybrid and the baseline cost the same. Below that rate the baseline is cheaper, because the proposal and audit fees are fixed cash costs while the saving is proportional to labour rate. Above it the hybrid wins, and the saving grows by $6.00 per dollar of hourly rate per 240-title slate (6.0 staffed hours × R).

```text
FORMULA [C-04]: R_be = (N*p_model + C_audit) / (N*(T_base - T_manual - T_review)/60)
NUMERICAL EXAMPLE: R_be = (240*0.085 + 208) / (240*(5.5-2.6-1.4)/60)
                        = 228.40 / (240*1.5/60) = 228.40 / 6.0 = 38.0667 -> $38.07/h.
INTERPRETATION: the entire economic case is a comparison of a 1.5-minute-per-title labour reduction
                against $228.40 of fixed quarterly cash cost.
```

**Boundary.** At `T_base = 4.0 min/title` (a cheaper vendor), the labour reduction is 0 minutes and no labour rate makes the hybrid cheaper; the break-even formula divides by zero. This is the structural limit of the whole idea: it monetizes a difference in handling time, so it is worthless wherever the baseline handling time is already at or below the hybrid's.

### 6.6 Step 6 — Size the confirmatory audit

The gate is 2.00 critical escapes per 100 titles, i.e. 0.02 escapes per title. With zero observed escapes, the exact one-sided 95% upper bound is `1 - 0.05^(1/n)`; with two-sided 95% it is `1 - 0.025^(1/n)`. Solving the two-sided form:

```text
FORMULA [N-01]: 1 - (alpha/2)^(1/n) <= e_target  ->  n >= ln(alpha/2) / ln(1 - e_target)
WHERE: n audited titles [titles]; alpha significance level [dimensionless]; e_target gate [escapes per title].
NUMERICAL EXAMPLE: alpha=0.05, e_target=0.02;
                   n >= ln(0.025)/ln(0.98) = (-3.68888)/(-0.020203) = 182.59 -> 183 titles.
INTERPRETATION: 183 audited titles with zero critical escapes are required before the gate can be
                claimed as demonstrated, versus the 60 titles audited here.
```

Independent check of the bound at n=183: `1 − 0.025^(1/183) = 1 − exp(−3.68888/183) = 1 − exp(−0.020158) = 1 − 0.980045 = 0.019955`, which is ≤ 0.02. At n=182: `1 − exp(−3.68888/182) = 1 − exp(−0.020269) = 0.020066`, which is > 0.02. So 183 is the smallest sufficient sample and 182 is not — the boundary is exact to the title.

**Cost of that audit.** 183 titles × 11 fields at an assumed 3.0 adjudicated minutes per title (double annotation plus disagreement resolution) = 549 min = 9.15 h × $52 = $475.80, one-off. That is 5.7× the recurring audit cost and larger than the quarterly saving at 240 titles. The honest conclusion: **the confirmatory audit must be justified as a one-off quality investment that unlocks a larger slate, not as part of this quarter's business case.**

### 6.7 Sensitivity and robustness

| Stress / test | Original condition | Alternative | Change in conclusion | Uncertainty / limitation | Decision consequence |
|---|---|---|---|---|---|
| Labour rate | $52/h | $70/h | Saving becomes negative: 22.0×70 = 1,540 vs 16.0×70+228.40 = 1,348.40 — saving $191.60, still positive; at $38.07 it is exactly zero | Rate-card assumption A-07 unverified | Publish the break-even rate with the recommendation, not a single saving figure |
| Slate size | 240 titles | 384 titles | Saving rises from 7.31% to 14.13% | Peak is an assumption, not observed | Recommend the pipeline for peak slates first |
| Audit cost | $208 recurring | $0 (skip audit) | Saving rises to 25.49% | Skipping the audit removes the only evidence of quality | Rejected: the audit *is* the control |
| Technical field reliability | Manifest authoritative (A-02) | Manifest omits forced-narrative tracks | Subtitle flag becomes a judgment field; escapes rise from 1.67 to ~5 per 100 | Constructed scenario | Revert subtitle flag to manual; recost at +0.4 min/title |
| Reviewer detection | 82.6% on judgment errors | 90% | Rejected 14-field scope would drop from 6.67 to ~4.6 escapes/100 — still failing | Detection rate unverified per reviewer | Detection must exceed 96% before judgment fields can be automated |
| Descriptive error tolerance | 11.5% error accepted | 5% error required | Descriptive automation fails; only technical fields remain | No downstream tolerance was measured | Would reduce the case to 2 fields |

**The two highest-leverage numbers.** Reviewer detection rate on critical errors and the labour rate. Detection governs whether any critical field can be automated; labour rate governs whether any of it is worth doing. Everything else in the ledger moves the answer by a few percentage points.

### 6.8 Step 7 — Marginal analysis: which specific fields earn their automation

The scope decision so far is tier-based. A stricter question is whether each individual field within the descriptive tier earns its place, since review cost and error rate both vary widely. The relevant quantity is **cost per correct assisted value**, not accuracy:

```text
FORMULA [M-01]: k_f = (t_probe_f/60)*R/60 + (t_review_f/60)*R/60 + p_f
WHERE: k_f cost per assisted field value [currency]; t_probe_f, t_review_f per-field seconds [s/title];
       R staffed-hour rate [currency/h]; p_f proposal cost [currency/value].
CONSTRAINT: one reviewer pass; costs exclude per-title overhead, which does not vary by field;
            assumes the value would otherwise cost T_base/N_f minutes of manual work.
NUMERICAL EXAMPLE for cast billing order: t_probe=0.8 s, t_review=5.0 s, p_f=$0.004;
   (0.8/60)*52/60 = 0.01333*0.86667 = 0.011556
   +(5.0/60)*52/60 = 0.08333*0.86667 = 0.072222
   + 0.004 = 0.087778 -> $0.0878 per field value.
INTERPRETATION: at 240 titles this field costs 240*0.087778 = $21.07 to assist, and saves the
                manual share of its handling time only if that share exceeds 5.8 s.
```

The comparison that makes this useful is against the same field done manually. If a field's manual handling share is 5.0 s, assisting it costs $0.0878 while manual costs (5.0/60) × 52/60 = $0.0722 — the assisted path is *more* expensive and exists only for throughput reasons. Fields where this inversion occurs are the ones to consider for de-automation when the release bottleneck is not binding.

| Field | Assist cost/value | Manual-equivalent cost/value | Assist cheaper? | Verdict |
|---|---:|---:|---|---|
| original-language flag | $0.0185 | $0.0144 (1.0 s) | No | Keep automated only for throughput |
| similar-title tags | $0.0363 | $0.0289 (2.0 s) | No | Marginal |
| franchise flag | $0.0328 | $0.0289 (2.0 s) | No | Marginal |
| collection membership | $0.0405 | $0.0433 (3.0 s) | Yes, narrowly | Automated |
| artwork keywords | $0.0303 | $0.0289 (2.0 s) | No | Marginal |
| dub availability | $0.0361 | $0.0433 (3.0 s) | Yes | Automated |
| subtitle availability | $0.0622 | $0.0722 (5.0 s) | Yes | Automated |
| awards tag | $0.0493 | $0.0433 (3.0 s) | No | Consider manual |
| cast billing order | $0.0878 | $0.0722 (5.0 s) | No | Consider manual |
| content-advisory descriptors | $0.0695 | $0.0578 (4.0 s) | No | Consider manual |
| mood tags | $0.0486 | $0.0433 (3.0 s) | No | Marginal |

**Reading the table honestly.** Six of eleven fields are not cheaper to assist than to handle manually at the per-value level. That does not contradict the case's recommendation, because the value being bought is *throughput under a release deadline* and *consistency across annotators*, not marginal currency. It does mean the recommendation must be stated in throughput terms when challenged on cost, and it identifies three fields (awards tag, cast billing order, content-advisory descriptors) where a de-automation trial would be the cheapest next experiment — they are also the three highest-error fields, which is a coincidence worth noticing rather than a law.

**Why these numbers are small.** Each field value costs single-digit cents either way. The whole ledger is a study in small margins multiplied by 240 titles; an error of 1.0 second per title is worth $2.08 per slate, and four such errors reverse the entire $83.60 saving. This is why the case insists on unit traces: at this margin, a plausible-looking rounding choice in minutes can flip the sign of the result.

**The two highest-leverage numbers.** Reviewer detection rate on critical errors and the labour rate. Detection governs whether any critical field can be automated; labour rate governs whether any of it is worth doing. Everything else in the ledger moves the answer by a few percentage points.

**What was tested and what was not.** No model was trained, no vendor was re-timed, and no reviewer was measured. All arithmetic is executed; all operational claims are proposed. The failed approach in Section 6.2 is a design iteration within this illustrative exercise, not a reported organizational failure.

## 7. The Finding

The recommended decision is to automate 11 of the 18 fields — nine descriptive plus dub and subtitle availability — and to keep primary genre, secondary genre, and audience age band with a named human cataloguer, because that scope moves critical escapes from 6.67 to 1.67 per 100 released titles while the full-scope alternative fails a 2.00 gate by more than a factor of three. The supported magnitude at 240 titles is a 7.31% total saving ($83.60) and 2.0 of 22.0 staffed hours returned, rising to 14.13% at a 384-title peak slate; the headline 25.49% figure measures annotation cost only and excludes the audit that makes the claim defensible. The uncertainty is substantial and specific: one escape in 60 audited titles carries a 95% upper bound of 5.96 per 100, so the gate is not demonstrated and 183 zero-escape titles would be needed to demonstrate it, at a one-off cost of about $476 — more than the quarterly saving. The main boundary is that the entire benefit is a 1.5-minute-per-title labour differential with a break-even rate of $38.07 per staffed hour, and the next justified action is a bounded pilot on the peak slate with per-title field-level rollback, a 183-title confirmatory audit funded as a quality investment, and no automation of the three judgment fields under any circumstance.

## 8. System/Solution Design

### 8a. Offline / Batch Path

```text
slate lock
  -> catalogue record export (title_id, territory, asset_id)     [N=240]
  -> schema pre-check: required fields present, vocabulary version pinned
  -> file manifest probe  -> dub/subtitle tuples per asset        [technical tier]
  -> proposal generator   -> candidate values for 11 fields       [model tier]
  -> validator            -> reject out-of-vocabulary / missing   [quarantine count]
  -> review console       -> accept / edit / reject per field     [ledger rows]
  -> manual judgment pass -> 3 fields by named cataloguer         [ledger rows]
  -> publication gate: no missing required field, no unreviewed critical value
  -> catalogue publish + downstream invalidation (browse, search, recommender features)
  -> audit sampler        -> gold labels -> accuracy + escape report
```

**Batch cadence.** One run per slate lock, plus an incremental path for late-delivered titles. The incremental path matters: a title delivered after the batch run must not be published with model-only critical values because it missed the review window. Its state is `held` until reviewed.

**Hardware and parallelism.** Not material at 240–384 titles per quarter. The workload is dominated by human review minutes, not compute; a claim about GPU capacity would be decorative and is deliberately omitted.

**Versioned artifacts.** Vocabulary version, proposal-generator version, probe tool version, and reviewer console version are pinned per slate and recorded in the ledger. A run is reproducible as a *record* (same inputs, same versions) but not necessarily as a *result* (a generative proposal generator may not be deterministic); this case therefore treats the ledger, not re-execution, as the reproducibility artifact.

### 8b. Online / Serving / Real-Time Path

There is no member-facing real-time path in this case. The nearest thing is **publication fan-out**: the moment a record is published, browse, search, and recommender features can consume it. That fan-out is where a bad field value becomes visible, so its behavior must be specified even though no request is served by the enrichment pipeline.

```text
publish(record, version)
  -> write record to system of record (immutable version appended)
  -> emit change event {title_id, territory, version, changed_fields[]}
       -> browse indexer       : rebuild affected rows only
       -> search indexer       : reindex changed fields
       -> recommender features : invalidate item-embedding input cache
  -> if any consumer fails: record fan-out lag; record remains published,
     consumer serves previous version (stale but consistent)
  -> if publication gate rejects: no event emitted; title stays held
```

**Boundary behavior.** The publication gate is the last line of defence and it must be a *deterministic* function of the record and the ledger, not a model judgement. If the ledger has no acceptance row for a critical field, the gate rejects. If the gate itself is unavailable, publication blocks rather than defaulting to allow.

**Cold start, peak, dependency failure.** Cold start means a new territory with an empty previous-season record: descriptive proposals lose their franchise-consistency signal, so their accuracy should be expected to fall, and the review time should be budgeted upward rather than assumed flat. Peak means the 1.6× slate; the binding resource is reviewer hours, and the queue must be drained in release-date order. Dependency failure means the probe tool or the proposal generator is down: the batch continues with those fields marked `unavailable`, and the titles that need them stay held rather than being filled from a default.

### 8c. Key Design Decisions

| Decision ID | Choice | Alternatives | Evidence | Tradeoff | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| DEC-01 | Automate 11 fields, keep 3 judgment fields manual | Automate 14; automate none | Illustrative audit (Section 6) | Loses the largest per-field labour block; keeps the gate passable | Judgment-field automation becomes acceptable only above 96% detection with cheap verification | Detection measurement or reviewer tooling change |
| DEC-02 | Treat dub/subtitle as technical, not judgment | Treat as judgment | Probe-verifiable; one subtitle error observed | Relies on manifest authority (A-02) | Manifest omits forced-narrative tracks | New mastering vendor |
| DEC-03 | Single reviewer pass, no second pass | Two-pass review | Cost; escape gate met on point estimate | Gate not statistically demonstrated at n=60 | Any critical escape in the pilot | Pilot escape or audit expansion |
| DEC-04 | Recurring 60-title audit, one-off 183-title confirmatory audit | Audit everything; no audit | Sample-size derivation (N-01) | Confirmatory audit costs more than one quarter's saving | Slate grows so audit cost amortizes | Slate size change |
| DEC-05 | Per-field rollback, not per-slate rollback | Withdraw the whole slate | Limits blast radius; descriptive fields are independent | More ledger bookkeeping | Fields are coupled downstream (e.g. franchise flag drives collection membership) | Downstream coupling discovered |
| DEC-06 | Publish change events at field granularity | Republish whole record only | Consumers can rebuild minimally | Consumer complexity; ordering risk | Consumer cannot handle partial updates | Consumer rewrite |

**Decision reasoning for DEC-01.** The alternative "automate none" is genuinely defensible at 240 titles: the saving is $83.60 and the confirmatory audit costs $475.80. A team that values its reviewer attention for other work, or that cannot fund the audit, should retain the baseline and revisit at the peak slate. That is why Section 10 records the action as a bounded pilot rather than a rollout.

**Decision reasoning for DEC-05.** Per-field rollback is chosen because descriptive field errors are largely independent: a wrong mood tag does not invalidate a correct franchise flag. It is *not* chosen for the technical fields, where a rollback must also invalidate the accessibility statement shown to members. That asymmetry is the interesting part of the decision and it is easy to miss.

### 8d. Architecture artifacts and operational readiness

| Area | This design |
|---|---|
| Interfaces | Field value contract: `{title_id, territory, field, value, vocabulary_version, provenance, confidence?}`. Reviewer decision contract: `{field, decision: accept/edit/reject, reviewer_id, decided_at, proposed_value, accepted_value}`. Idempotency key on publication: `(title_id, territory, record_version)`. |
| Data/state | Record versions are append-only. The ledger is the acceptance truth. Retention must cover the release audit window plus the contractual retention period; the exact period is an OPEN policy question. |
| Reliability | SLI: percentage of slate titles published at least 10 business days before release. Target 100% for the pilot. Error budget: zero titles published with an unreviewed critical field. Measurement location: publication gate, not the console. |
| Resilience | Probe and generator failures degrade to `unavailable`, never to a default value. Review console outage queues decisions; it must not allow publication to proceed. Generator timeouts are bounded per title so one pathological file cannot stall a batch. |
| Recovery | A wrong field is corrected by appending a new record version and republishing; the previous version stays readable for the dispute record. Replay of the batch must not duplicate ledger rows — the idempotency key enforces that. |
| Observability | Per-field accept/edit/reject rates, per-reviewer detection on seeded errors, quarantine counts by reason, publication-gate rejections, fan-out lag per consumer, and the escape count from each audit. |
| Deployment | Vocabulary, generator, probe, and console versions deploy independently but are pinned together per slate. A vocabulary change mid-slate requires re-validation of every proposed value against both versions; the cheaper correct action is to freeze the vocabulary. |
| Security | Reviewer identity is authenticated and non-repudiable. Field values for unreleased titles are commercially sensitive; access is least-privilege and per-territory. Audit gold labels are permission-isolated from the pilot console. |
| Economics | Cost per released title and cost per correctly enriched field value, both computed from the ledger rather than from invoices. Capacity envelope: reviewer hours per slate at 1.4 min/title, with a stated queue-drain assumption. |
| Human workflow | Cataloguers accept or edit judgment fields; reviewers accept or spot-check assisted fields; the operations lead can hold any title; trust-and-safety can veto a maturity-adjacent value; anyone can stop the pilot, and the stop is recorded with a reason. |

**Worked capacity estimate.** At 1.4 review minutes per title plus 2.6 manual minutes per title, the per-title staffed time is 4.0 min. A reviewer working 6.0 productive hours per day (a staffing assumption, not a measurement) covers 360 min / 4.0 min = 90 titles per day. The 240-title slate therefore needs 2.67 reviewer-days, or 3 calendar days with one reviewer — comfortably inside the 10-business-day window. At the 384-title peak it needs 4.27 reviewer-days. The window is not the binding constraint; reviewer *availability in the same three days* is, because a single reviewer absence doubles the drain time.

**Interview cue.** What happens if the probe fails, if the vocabulary changes mid-slate, if a reviewer is absent for two days, or if rollback must invalidate an accessibility statement?

**Acceptance.** A reader can trace a title from slate lock to fan-out, follow a rejected publication, and name who owns each recovery action.

## 9. Evaluation & Validation

| Evaluation field | This case |
|---|---|
| Primary metric | Critical escapes per 100 released titles: numerator = released titles carrying at least one incorrect value in a field classified critical at the time of release; denominator = titles released in the slate; unit = titles per 100 titles; population = this territory's slate; aggregation = sum over the slate; window = from slate release to the end of the 30-day post-release observation. |
| Secondary metric | Field-level accuracy per field and per tier, from adjudicated gold labels; reported at the audited-value grain, never mixed with the title grain. |
| Baseline | All-manual vendor annotation at 5.5 min/title. Its escape rate is **not measured** in this case and is the single largest gap in the argument: a gate is only meaningful relative to the baseline's own escape rate. |
| Pass/fail rule | Gate: ≤2.00 critical escapes per 100 released titles, with a 95% upper bound below the gate. Cost: total staffed minutes strictly below baseline. Owner: catalogue operations lead. Threshold chosen before the pilot, recorded in this document's date. |
| Guardrails | Reviewer minutes per title; quarantine rate; publication-gate rejection rate (a rising rate signals upstream contract drift); consumer fan-out lag; accessibility-statement corrections. |
| Test design | Prospective audit of the pilot's released titles with adjudicated gold labels; plus a seeded-error detection test for each reviewer; plus a before/after cost comparison on matched slate sizes. |
| Sample | 60 titles recurring; 183 titles for the confirmatory zero-escape claim; stratified by content type and by whether a previous-season record exists. |
| Uncertainty | Clopper–Pearson exact intervals for escape rates; Wilson interval for field accuracy. No p-values are computed because no hypothesis test is being run — the decision is a gate comparison, not a significance claim. |
| Validity threat | (1) Audit sample not representative; (2) gold labels contaminated by reviewer feedback; (3) baseline escape rate unknown, so the gate may be stricter or looser than the status quo; (4) seasonality of descriptive-field error rates. |
| Artifact | `SRC-ILL-01` (constructed inputs), `RUN-ILL-01` (arithmetic), ledger rows by slate. Access: public text for the arithmetic; no production artifact exists. |
| Decision | Passed on the point estimate, **inconclusive** on the upper bound. The action is therefore a bounded pilot with a confirmatory audit, not a rollout. |

**What would falsify the recommendation.** Three concrete observations:

1. A critical escape in the pilot at any rate above 2 per 100 released titles over at least 50 released titles would fail the gate and force DEC-01 back to judgment-field-only-plus-technical, or to full manual.
2. A measured reviewer detection rate below 96% on seeded critical errors would invalidate the scoping logic that keeps the judgment fields manual *and* the logic that a single pass suffices for the technical fields.
3. A measured baseline escape rate already near zero would make the gate non-binding and the pipeline's quality case rest entirely on cost — where the 7.31% saving may not justify the operational change.

**How the evaluation could accidentally make a bad pipeline look good.** Four ways, all of which must be blocked by design:

- Auditing only the fields the pipeline is good at (choose the sample per tier, never per suspected quality).
- Letting reviewers see gold labels during the pilot (permission isolation).
- Using the same season's records for tuning and for audit (temporal separation).
- Reporting accuracy at the value grain when the decision is at the title grain (report both, decide on the second).

**Separating intended from executed protocol.** Intended: prospective audit, 60 titles, two annotators, seeded errors. Executed: nothing — this is an illustrative exercise and no title was released. The table above is a protocol specification, not a completed validation. Evidence status for the whole evaluation: `not-measured`.

## 10. Action Taken

| Action field | Record |
|---|---|
| Recommendation | Adopt the 11-field assisted scope for a bounded pilot on the next peak slate; keep the three judgment fields manual; fund a one-off 183-title confirmatory audit. |
| Approval/status | Proposed. No approval exists and none is implied; the illustrative decision owner is the catalogue operations lead, with metadata steward and trust-and-safety vetoes. |
| Implementer and handoff | Catalogue operations would own the runbook; the metadata steward would own vocabulary pinning; localization would own the manifest authority check (A-02). Handoff artifact: this design plus the ledger schema in Section 8d. |
| Effective scope | Would apply to one territory, one slate, 240 titles, 18 fields. |
| Release strategy | Bounded pilot with per-field rollback; descriptive fields first, technical fields after the probe validation; judgment fields unchanged. |
| Stop/rollback | Any critical escape above the gate, any unreviewed critical publication, or a measured detection rate below 96% stops the pilot. Authority: operations lead. Procedure: append a corrective record version, republish, and record the reason in the ledger. |
| Excluded actions | No automation of primary genre, secondary genre, or audience age band. No change to the rights window fields. No change to the maturity rating system. No vendor renegotiation in this decision. |

**No action taken as of 2026-11-09.** The case is a design and arithmetic exercise. Nothing was piloted, no title was enriched, and no member saw an affected value. The blocker for real action is not analytical — it is that no operational evidence exists: no measured baseline handling time, no per-reviewer detection rate, and no authority to change the release gate.

**Organizational realities a real decision would face.** The vendor contract may specify per-title pricing rather than hourly rates, in which case the labour-rate model in Section 6 is the wrong cost model and must be replaced with a per-title price differential. The reviewer may not be the same person who adjudicates the audit, which changes staffing. And the release gate may be owned by a different team than the catalogue operations lead, which makes the gate a negotiation rather than a parameter.

## 11. Outcome & Results

**Outcome not yet measured.** The only executed result is arithmetic on constructed inputs.

| Metric | Baseline | Treatment | Absolute change | Relative change | Uncertainty | Evidence | Limitation |
|---|---|---|---|---|---|---|---|
| Staffed minutes per slate (240 titles) | 1,320 min (22.0 h) | 1,200 min (20.0 h) | −120 min | −9.09% | None computed; arithmetic identity | `RUN-ILL-01` (illustrative) | Excludes vendor management overhead |
| Total cost per slate (240 titles) | $1,144.00 | $1,060.40 | −$83.60 | −7.31% | Rate-card assumption only | `RUN-ILL-01` (illustrative) | Audit treated as recurring; labour rate fixed |
| Total cost per slate (384 titles) | $1,830.40 | $1,571.84 | −$258.56 | −14.13% | Peak-slate assumption | `RUN-ILL-01` (illustrative) | Same exclusions |
| Field-level accuracy (11 fields) | Not measured for the baseline | 597/660 = 90.45% | Not computable | Not computable | 95% Wilson interval, computed in the smoke record | `SRC-ILL-01` (illustrative) | Baseline accuracy unknown, so no comparison |
| Critical escapes per 100 titles | Not measured | 1.67 point estimate; upper bound 5.96 | Not computable | Not computable | Clopper–Pearson | `SRC-ILL-01` (illustrative) | Gate cannot be demonstrated at n=60 |
| Released titles with a corrected field | Not measured | Not measured | — | — | — | None | Would require a live pilot |

**Source declaration.** Figures are constructed teaching values, not measurements of any named organization and not evidence of realized impact. The arithmetic identities (min → h → currency; counts → rates; sample-size inversion) are independently checkable and were checked; the operational quantities (5.5 min/title, 1.4 min/title, $52/h, $0.085/title) are assumptions with no source.

**What improved, what worsened, what is unattributable.** Within the constructed ledger, cost and staffed time improve and reviewer attention shifts from 18 fields to 14 confirmations plus 3 judgment decisions. What worsens is analytical: the team now carries an audit obligation and a gate it cannot yet prove it meets. Nothing here is attributable to a real intervention, because there was none.

**Measurement plan for a real pilot.** Owner: catalogue operations lead with the metadata steward. Window: the pilot slate plus 30 days post-release. Primary measurement: adjudicated audit of 183 titles for critical escapes; 60-title recurring audit thereafter. Cost measurement: ledger-derived staffed minutes, reconciled against actual timesheets for one slate. Decision trigger: gate pass plus cost pass → expand to the next slate; gate fail → revert judgment and technical fields to manual and retain descriptive automation only if the descriptive error tolerance is separately accepted.

## 12. Failure Modes & What Went Wrong

Each entry below is a distinct mechanism with a distinct discriminating check. None is a restatement of another; the point of the table is that a reviewer facing "the field is wrong" must be able to tell *which* failure they have. All are hypothetical or illustrative design failures — no incident occurred.

### F1: Correction leakage between tuning and audit

**Symptom.** Audit accuracy is high and stable across weeks, but reviewers keep editing the same handful of fields in the live queue; the audit and the console disagree about quality.

**Mechanism.** The audit sample was drawn from records whose values were also used to build or tune the proposal generator. The generator reproduces those records well and generalizes poorly, so the audit measures memorization.

**Discriminating check.** Split the audit into (a) titles present in the generator's input history and (b) titles absent from it. A large gap between the two accuracies confirms leakage; equal accuracies with a high console edit rate instead point to a reviewer-tooling problem.

**Mitigation.** Restrict the audit to titles the generator never saw. Cost: a smaller eligible pool, and a delay of one quarter for freshly mastered titles. **Correction.** Re-draw the sample, re-measure, and record the contaminated estimate as withdrawn rather than as a trend. **Prevention.** Make "in tuning set" a queryable attribute of every title and gate the audit sampler on it. **Verification.** Re-run the split test; the gap should shrink to sampling noise.

### F2: Vocabulary revision mid-slate

**Symptom.** Proposed values are structurally valid but semantically odd — titles land in a genre node that did not exist when the proposal was generated; browse rows show implausible neighbours.

**Mechanism.** The genre vocabulary changed between proposal generation and review, and the values were mapped by name rather than re-generated against the new version. Renamed and merged nodes silently acquire values that were correct under the old taxonomy.

**Discriminating check.** Compare `vocabulary_version` in the proposal record against the version pinned at publication. Any mismatch explains the symptom directly; if versions match, the cause is elsewhere (likely a proposal-generator defect rather than taxonomy drift).

**Mitigation.** Freeze the vocabulary from slate lock to publication. Cost: taxonomy improvements wait for the next cycle. **Correction.** Re-generate proposals for affected titles under the pinned version. **Prevention.** Pin the version in the ledger and reject mismatches at the publication gate. **Verification.** Publication-gate rejection count for version mismatch should become zero.

### F3: Manifest under-reports subtitle tracks

**Symptom.** Members report that a title they expected to be subtitled is not, or the accessibility filter excludes a title that does carry subtitles; the audit shows a subtitle flag error rate that looks like noise.

**Mechanism.** The delivered container manifest omits forced-narrative subtitle tracks that localization adds in a later packaging step. The probe reads the manifest faithfully and therefore reports `false` correctly — for the wrong artifact.

**Discriminating check.** Probe the final packaged asset rather than the delivered master for the same titles. If the flag flips, the defect is an artifact-boundary error, not a probe bug. A probe bug would show as a mismatch even on the final asset.

**Mitigation.** Revert the subtitle field to manual until the probe reads the packaged artifact. Cost: 0.4 min/title. **Correction.** Republish corrected values with a new record version. **Prevention.** Make the probe's input artifact part of the contract, and test it on the packaged asset in the pipeline's integration test. **Verification.** Zero subtitle-flag corrections in the next two audits.

### F4: Reviewer detection collapses under queue pressure

**Symptom.** Edit rate stays flat while the escape rate rises; the same reviewers who caught errors at low volume miss them at high volume.

**Mechanism.** Detection on judgment-adjacent technical fields depends on reading evidence, which is the first step skipped when the queue is long. The aggregate detection rate hides a load-dependent degradation.

**Discriminating check.** Seed a fixed number of known-wrong values per reviewer per day and compare detection at low versus peak queue depth. A load-dependent drop confirms the mechanism; a uniformly low rate indicates a training or console problem instead.

**Mitigation.** Cap queue depth per reviewer-day and route overflow to a second reviewer rather than extending the day. **Correction.** Re-audit the titles reviewed during the overload window. **Prevention.** Treat detection-under-load as a monitored SLI, not a hiring assumption. **Verification.** Seeded-detection rate stable across queue depths.

### F5: Descriptive error tolerance was never agreed

**Symptom.** Months after release, merchandising complains that browse rows are full of implausible mood tags, and there is no agreed number that says how wrong is too wrong.

**Mechanism.** The case classified nine fields as non-critical without measuring any downstream tolerance. "Non-critical" meant "does not gate release", which is not the same as "any error rate is acceptable". The decision to automate rested on an unmeasured acceptance criterion.

**Discriminating check.** Compare a merchandising quality review of automated rows against the same review on manually annotated rows from a prior slate. A real difference confirms the mechanism; no difference suggests the complaint is about a specific vocabulary, not the pipeline.

**Mitigation.** Set a per-surface error tolerance before expansion. **Correction.** Re-tune or de-automate the worst field (content-advisory descriptors at 80.00%). **Prevention.** Require a downstream owner to sign an explicit tolerance per field. **Verification.** Surface review passes at the agreed tolerance for two consecutive slates.

### F6: Duplicate catalogue records inflate the denominator

**Symptom.** Reported accuracy wobbles with no change to the pipeline; the count of audited values does not equal 60 × 11.

**Mechanism.** The same title exists twice under different identifiers (a re-delivery, a territory split, an acquisition migration). Both copies are audited, both contain the same values, and the effective sample is smaller than reported — the confidence interval is too narrow.

**Discriminating check.** `SELECT title_id, count(*) ... GROUP BY canonical_key HAVING count(*) > 1` on the audit extraction. Duplicates explain the count mismatch; if the count is right, look for a join fan-out on the manifest instead.

**Mitigation.** Compute the audit on canonical keys, not raw records. **Correction.** Re-issue the interval with the corrected effective sample. **Prevention.** Canonical-key enforcement at extraction and a duplicate count in every audit report. **Verification.** Audited-value count equals titles × fields exactly.

### F7: A gate that cannot fail

**Symptom.** The gate passes quarter after quarter and nobody can say what would make it fail. Quality complaints persist.

**Mechanism.** The gate is evaluated on the same audited sample that the reviewers curated, and sample size is small enough that one escape always rounds inside the bound. The gate has become a ritual rather than a test.

**Discriminating check.** Ask for the smallest observation that would fail the gate, given the current sample size. If no observation in the achievable range fails it, the gate is non-binding by construction.

**Mitigation.** Fix the sample size to the value derived from the target rate (183 titles for 0.02) before evaluating. **Correction.** Re-state past passes as inconclusive. **Prevention.** Require every gate to publish its required sample size and its failure region. **Verification.** The gate's failure region is non-empty and documented.

## 13. Transferable Lesson

| Pattern field | Content |
|---|---|
| Canonical pattern | `evaluation-mismatch`, with `human-in-the-loop` as the mechanism that makes the mismatch visible |
| General principle | When an automated step sits in front of a release gate, the decision metric must be defined at the grain the gate acts on — released items and escaped critical errors — not at the grain the model is scored on. |
| Preconditions | A field taxonomy with an explicit critical tier; a reviewer whose detection can be measured; an authoritative source for the fields claimed to be machine-verifiable; a cost baseline in staffed minutes. |
| Mechanism | Field-level accuracy aggregates errors with unequal consequences and unequal verification costs. Separating fields into tiers converts one misleading number into two decision-relevant ones: escapes per 100 released items, and staffed minutes per item. The labour differential then determines whether the change is worth making at all. |
| Supporting evidence | The constructed audit: 89.76% aggregate accuracy coexists with 6.67 escapes per 100 titles; restricting scope moves escapes to 1.67 and total saving to 7.31%. Arithmetic is checked; operational quantities are assumptions. |
| Avoid when | The baseline's own escape rate is unknown and cannot be measured — then the gate has no reference point and the "pass" is meaningless. Also avoid when the baseline handling time is already at or below the hybrid's, because the benefit is a time differential and nothing else. |
| Portability test | Ask: "What is the smallest observation that would fail our gate, and would our current sample size ever produce it?" If the answer is "nothing" or "no", the metric is decorative. |
| Adaptation | In document processing, the grain becomes documents released to a downstream system and the escape is a mis-extracted field with a legal consequence; in adverse-event triage it becomes cases released without review and the escape is an unreviewed serious event. The tiering logic transfers; the thresholds do not. |

**What does not transfer.** The 2.00 gate, the 5.5-minute baseline, the $52 rate, and the 60/183 sample sizes are all local. A reader who lifts the numbers without re-deriving the baseline will reproduce the mistake the case is about.

**Counterexample where following the lesson is harmful.** A pipeline whose output is *advisory only* — shown to a human who must independently verify before any consequential action — should not be held to an escape-per-released-item gate, because nothing escapes: the human is the release gate. Applying the lesson there creates expensive audit machinery for a control that already exists.

## 14. Connections & Cross-References

### 14a. Related cases, theory, and cheat sheets

| Relationship | Target | Mechanism shared or changed | Boundary learned |
|---|---|---|---|
| uses-method | [02_Rights_Window_Conflict_Detection.md](./02_Rights_Window_Conflict_Detection.md) | Both treat a catalogue record as a versioned decision artifact with a point-in-time contract | Rights fields are contract-derived, not content-derived, so accuracy tiering does not transfer |
| contrasts-with | [03_Catalogue_Decay_Model.md](./03_Catalogue_Decay_Model.md) | Decay modelling consumes metadata quality as an input rather than producing it | A metadata error can look like audience decay if the two are not separated |
| uses-method | [04_Artwork_Experiment.md](./04_Artwork_Experiment.md) | Both face an evaluation-mismatch between an offline score and a released-surface outcome | Artwork has a randomized alternative; metadata enrichment does not, so causal claims differ |
| extends | [../Cheat_Sheets/01_Content_Metadata_Schema_and_Lineage.md](../Cheat_Sheets/01_Content_Metadata_Schema_and_Lineage.md) | Field tiering, vocabulary pinning, and lineage records used here | The sheet derives the schema invariants; this case applies one tiering choice |
| uses-method | [../Cheat_Sheets/99_25_Common_Issues.md](../Cheat_Sheets/99_25_Common_Issues.md) | Issue entries for duplicate keys, vocabulary drift, and reviewer overload | The issue file is diagnostic; it does not derive the cost model |

### 14b. What this case leaves open

| Open question | Why current evidence is insufficient | Data/test that would resolve it | Priority / trigger | Owner role |
|---|---|---|---|---|
| What is the baseline's own critical escape rate under all-manual annotation? | Never measured; only the hybrid was audited | Adjudicated audit of a manual-only slate at 183 titles | High — the gate has no reference without it | Measurement owner |
| Is the track manifest authoritative for the packaged asset? | Only the delivered master was reasoned about | Probe both artifacts for 20 titles and compare | High — the one passing critical field depends on it | Localization lead |
| What is per-reviewer detection on seeded critical errors, and how does it vary with queue depth? | No reviewer was measured | Seeded-error test across three queue depths | High — governs whether any critical field can be automated | Operations lead |
| Do descriptive errors degrade a measured surface? | No downstream tolerance exists | Merchandising review of automated versus manual rows | Medium — gates any expansion beyond 11 fields | Recommendation engineering |
| Does a vocabulary revision in the same quarter as a proposal change the error profile? | Not observed | Paired quarter comparison with pinned and unpinned vocabularies | Medium | Metadata steward |
| How large must the slate be for the audit cost to amortize below 5% of baseline cost? | Only two slate sizes computed | Recompute the ledger across 200–600 titles | Low | Finance partner |

### 14c. Intense interview pack — questions with checked answers

**Q01 / framing / 2 min.** *"State the decision this case makes and the two numbers that decide it."*

**Given facts.** 240 titles, 18 fields, 60-title audit, 4 escaped judgment errors, 1 escaped technical error, gate 2.00 per 100 titles, $52/h, baseline 5.5 min/title.

**Strong answer.** The decision is which fields an assisted pipeline may populate before release. The two deciding numbers are critical escapes per 100 released titles (must be ≤2.00, with an upper bound below it) and total staffed minutes per released title (must be below the 5.5-minute baseline). Reasoning: field accuracy aggregates errors whose consequences differ, so it cannot bind a release gate; cost aggregates minutes whose field mix differs, so it cannot be read per field. **Check.** 4/60 × 100 = 6.67 (fails); 1/60 × 100 = 1.67 (passes on the point estimate). Conclusion: scope must shrink.

**Acceptable alternatives.** A candidate may argue the deciding quantity is expected loss per released title, provided they state a loss per error class. That is a legitimate reframing and should be credited if the loss values are declared rather than assumed.

**Follow-up.** "Suppose the baseline's escape rate is 1.0 per 100. Does your gate still make sense?" — It becomes a 2× loosening of the status quo, and the gate must be re-set relative to measured baseline rather than to an absolute constant.

**Red flags.** Answering with accuracy alone; treating 89.76% as "close enough to 95%"; ignoring that the denominators differ (840 values versus 60 titles).

**Scoring.** 0: names accuracy. 1: names escapes but cannot compute them. 2: computes both and identifies the denominator switch. 3: additionally argues the baseline reference point and the upper-bound problem.

**Answer support.** Sections 6.2 to 6.4; formulas A-01, G-01, G-02, C-02.

**Q02 / data / 3 min.** *"Why is the audit measured at two grains, and what breaks if you report only one?"*

**Strong answer.** Field values are what the pipeline produces; released titles are what the gate acts on and what a member experiences. Reporting only the value grain lets 540 easy descriptive values dominate a headline number and hides 23 judgment errors in 180 values. Reporting only the title grain loses the ability to locate which field to de-automate. **Check.** 754/840 = 89.76% at the value grain versus 6.67 escapes per 100 at the title grain, from the same audit. **Conclusion.** Both are required, and the decision uses the second.

**Acceptable alternatives.** A candidate may propose expected-loss aggregation as a single combined grain, if they show how a loss per error class produces the same ranking of field scopes.

**Follow-up.** "You have 240 titles and 18 fields. If you could audit only 600 values, how would you allocate them?" — Per tier: at least 60 titles × the 5 critical fields (300 values) plus 300 spread over descriptive fields, because the critical tier binds the decision.

**Red flags.** Believing a rate is grain-independent; proposing to audit only the worst-looking fields (selection on the outcome).

**Scoring.** 0: one grain, no justification. 1: names both but cannot say which binds. 2: explains domination and gives the numbers. 3: adds allocation under a value budget.

**Answer support.** Sections 5a (grain table), 6.3 (per-field table).

**Q03 / calculation / 4 min.** *"Compute the hybrid cost and saving at 240 titles and at 384. Show every unit."*

**Strong answer.** At 240: manual judgment 2.6 min × 240 = 624 min = 10.4 h; review 1.4 × 240 = 336 min = 5.6 h; total labour 16.0 h × $52 = $832.00; proposal 240 × $0.085 = $20.40; audit $208.00; total $1,060.40. Baseline 5.5 × 240 = 1,320 min = 22.0 h × $52 = $1,144.00. Saving $83.60 = 7.31%. At 384: labour 4.0 × 384 = 1,536 min = 25.6 h × $52 = $1,331.20; proposal $32.64; audit $208.00; total $1,571.84. Baseline 35.2 h × $52 = $1,830.40. Saving $258.56 = 14.13%. **Check.** 25.6 h × 60 = 1,536 min ✓; 1,331.20 + 32.64 + 208.00 = 1,571.84 ✓; 258.56/1,830.40 = 0.141258 ✓.

**Acceptable alternatives.** A candidate who excludes the audit from the recurring total will get 25.49% at 240 titles; credit only if they then say the audit is a real cost and show where it would be accounted.

**Follow-up.** "At what labour rate does the saving vanish, and why is it a rate rather than a slate size?" — $38.07/h; because the saving is 6.0 staffed hours × rate minus $228.40 fixed cash, so the fixed cash becomes decisive at low rates regardless of slate size.

**Red flags.** Mixing minutes and hours in the same line; treating the audit as one-off while also calling it recurring; reporting 7.31% and 25.49% as if they measured the same thing.

**Scoring.** 0: arithmetic error that changes the sign. 1: correct total but no unit trace. 2: correct with units and break-even. 3: adds the fixed-vs-proportional structure of the cost model.

**Answer support.** Section 6.4, 6.5; formulas C-02, C-03, C-04.

**Q04 / diagnosis / 4 min.** *"Accuracy looks fine but review edits cluster in three fields. What are the competing causes and how do you separate them?"*

**Strong answer.** Competing causes: (a) the tuning/audit overlap reproduces known records and fails on new ones; (b) the vocabulary changed and values were name-mapped; (c) those three fields are genuinely harder and the tier assignment was wrong; (d) reviewers apply a stricter standard than the adjudicators. Discriminating checks: split audit by in-tuning-set membership (isolates a); compare pinned vocabulary versions (isolates b); compare adjudicated gold against the reviewer's edits for those fields (isolates c versus d — if gold agrees with the pipeline, the reviewers are over-editing). **Conclusion.** One check per cause, each with a different expected signal.

**Acceptable alternatives.** A candidate may start with (d) because it is cheapest to test; that is fine if they explain the ordering rationale.

**Follow-up.** "You find (b). What is the smallest safe correction?" — Re-generate proposals under the pinned vocabulary for affected titles, keep the ledger rows, and mark the mis-mapped values withdrawn rather than edited, so the error is not laundered into a quality trend.

**Red flags.** Proposing to retrain immediately; treating clustering as evidence of one cause; editing historical ledger rows.

**Scoring.** 0: single hypothesis. 1: lists causes without checks. 2: distinct checks per cause. 3: adds the ordering rationale and the non-laundering correction.

**Answer support.** Section 12 F1, F2, F4; Section 6.3.

**Q05 / method / 3 min.** *"Why not simply automate the fields with the highest accuracy?"*

**Strong answer.** Because accuracy and consequence are not aligned and verification cost differs. The highest-accuracy fields here are the two technical ones (100% and 98.33%), which is convenient — but content-advisory descriptors at 80.00% and artwork keywords at 81.67% sit in the same "descriptive" tier as franchise flag at 96.67%. Selecting by accuracy alone would create an unstable scope that changes every quarter with sampling noise. Selecting by *consequence tier plus verifiability* produces a stable scope that the gate can test. **Check.** A 60-value estimate has a standard error of about 5 percentage points, so ranking 85% against 81.67% is not meaningful.

**Acceptable alternatives.** Accuracy-ordered scoping is acceptable if combined with a minimum tier requirement and a stability rule (e.g. only fields above a lower confidence bound).

**Follow-up.** "What sample size would let you rank two descriptive fields reliably?" — To separate 85% from 81.67% at 95% confidence you need on the order of several hundred values per field, which exceeds the whole audit budget; hence the tier approach.

**Red flags.** Treating a 60-value proportion as precise; ignoring that automation changes the reviewer's task even for accurate fields.

**Scoring.** 0: accuracy-only. 1: mentions risk but no sampling argument. 2: tier plus verifiability. 3: adds sampling precision and scope stability.

**Answer support.** Section 6.3 per-field table; Section 5c selection logic.

**Q06 / statistics / 3 min.** *"You observed 1 escape in 60 titles against a 2-per-100 gate. Why is that not a pass?"*

**Strong answer.** A point estimate of 1.67 per 100 with n=60 has a wide exact interval: the 95% Clopper–Pearson upper bound is 1 − 0.025^(1/60) = 1 − 0.94038 = 5.96 per 100. The observation is compatible with a true rate three times the gate. To claim the gate with zero escapes you need n ≥ ln(0.025)/ln(0.98) = 183 titles. **Check.** n=183 gives 1 − exp(−3.68888/183) = 1.996%; n=182 gives 2.007%. So 183 is the boundary.

**Acceptable alternatives.** A Bayesian statement with an explicit prior is acceptable if the prior is declared and the resulting credible bound is compared to the gate.

**Follow-up.** "What if you had 2 escapes in 60 — does the sample requirement fall or rise?" — Neither: with escapes observed, the bound is already above the gate at any small n, so the scope fails rather than needing more data.

**Red flags.** Treating 1/60 as "1.67% so we pass"; computing a normal approximation at this sparsity without comment; conflating value grain with title grain in the interval.

**Scoring.** 0: accepts the point estimate. 1: names uncertainty, no number. 2: computes the bound. 3: inverts for sample size and states the boundary exactly.

**Answer support.** Section 6.3, 6.6; formula N-01.

**Q07 / design / 3 min.** *"Design the reviewer's screen for one title. What must be on it, and what must not?"*

**Strong answer.** Must show, per field: the proposed value, its provenance (model/probe/vendor), the authoritative evidence (the probe tuple for technical fields; the title's own record and a franchise precedent for descriptive fields), and the accept/edit/reject control. Must not show: the audit gold labels (contaminates detection), other titles' corrections made by colleagues (anchoring), or a confidence number presented as a probability of correctness unless it has been calibrated. **Check.** If the screen shows a calibrated confidence and reviewers accept above 0.9, the edit rate should concentrate below 0.9; if it does not, the confidence is uncalibrated.

**Acceptable alternatives.** Batch review is acceptable for descriptive fields if per-field evidence is still one click away; it is not acceptable for technical fields where the probe output is the whole evidence.

**Follow-up.** "The reviewer is 20% faster with batch review. What do you measure before allowing it?" — Seeded detection rate under batch mode versus per-title mode; if detection falls, the speed is bought with escapes.

**Red flags.** Putting gold labels on the screen; designing a screen that shows only the proposed value; treating confidence as correctness.

**Scoring.** 0: value-only screen. 1: adds evidence for one tier. 2: covers both tiers and exclusions. 3: adds the calibration test and the batching experiment.

**Answer support.** Section 8b, 8d (human workflow); Section 9 validity threats.

**Q08 / production / 3 min.** *"The probe tool is down for a day during a peak slate. What happens?"*

**Strong answer.** The batch continues for descriptive fields; the two technical fields are marked `unavailable`, and every title needing them stays `held` rather than being filled with a default. Publication of a held title is blocked by the deterministic gate. The recovery is a re-run of the probe for the affected assets, not a manual guess, because the probe's value is that it reads the artifact. **Check.** Publication-gate rejection count for the day should equal the held-title count exactly; a mismatch means the gate is not deterministic.

**Acceptable alternatives.** Falling back to the localization worksheet is acceptable *only* if the worksheet is proven authoritative for the packaged asset, which is exactly assumption A-02.

**Follow-up.** "The outage is three days and the release date does not move. What is the decision?" — Either release the slate without those titles (allowed: C-01 permits holding a title, not releasing with a missing field) or extend the window with an explicit schedule exception. Filling values manually without evidence is not an option.

**Red flags.** Defaulting to `true` for accessibility flags (the harmful direction); treating the outage as a reason to skip review.

**Scoring.** 0: defaults values. 1: holds titles but no gate check. 2: holds and re-runs. 3: adds the deterministic-gate verification and the schedule tradeoff.

**Answer support.** Section 8a, 8b, 8d (resilience, recovery).

**Q09 / evaluation / 3 min.** *"How would you detect that the audit itself has become a rubber stamp?"*

**Strong answer.** Three signals: (a) the gate's failure region is empty at the current sample size, so no achievable observation fails it; (b) adjudicator disagreement rate falls to near zero while reviewer edit rate stays high — the adjudicators have stopped looking; (c) the audit's error count is uncorrelated with the console's edit count across slates. **Check.** Compute the correlation between per-slate audit errors and per-slate console edits; a near-zero correlation with a stable edit rate indicates the audit is not measuring the same thing the console sees.

**Acceptable alternatives.** A seeded-error programme on the adjudicators themselves is a stronger, more expensive alternative and should be credited.

**Follow-up.** "Add one control that costs almost nothing." — Require every audit report to state its failure region and its required sample size; a gate that cannot state these is rejected at review.

**Red flags.** Proposing "more training"; measuring only agreement without measuring accuracy against a seeded truth.

**Scoring.** 0: no mechanism. 1: one signal. 2: two signals with a check. 3: adds the correlation test or seeded adjudication.

**Answer support.** Section 12 F7; Section 9.

**Q10 / constraint change / 4 min.** *"The maturity rating must now be derived during enrichment. How does the case change?"*

**Strong answer.** The maturity rating becomes a critical judgment field with legal-ish consequences and cannot be model-populated under this evidence. Concretely: it joins the manual tier (4 judgment fields), the assisted scope falls from 11 to 10 fields or fewer, the per-title manual time rises, and the gate's critical tier now includes a field whose error is not detectable by probing — so the escape measurement must include it and the required sample size rises because the acceptable rate for it is near zero. **Check.** At an assumed +1.1 min/title for the new field: total labour 5.1 min/title × 240 = 1,224 min = 20.4 h × $52 = $1,060.80, plus $20.40 + $208.00 = $1,289.20, which **exceeds** the $1,144.00 baseline. The hybrid becomes uneconomic.

**Acceptable alternatives.** Automating the rating with a human acceptance on every title is acceptable in principle but converts the saving to zero; a candidate should notice that.

**Follow-up.** "What single measurement could restore the case?" — A verified mapping from content-advisory descriptors to the rating rubric with near-zero escape rate, which is a rule-based derivation, not a model judgment; that would move the field back to the technical tier with probe-like verification.

**Red flags.** Automating a maturity field because a model is "probably right"; ignoring the cost change; forgetting that the gate's critical tier changed.

**Scoring.** 0: no change acknowledged. 1: says "keep it manual" only. 2: recomputes cost and notes the gate change. 3: computes the uneconomic result and proposes a rule-based restoration path.

**Answer support.** Sections 3, 4 (C-02), 6.4; formula C-02.

**Q11 / framing / 2 min.** *"What is the strongest argument for doing nothing?"*

**Strong answer.** At 240 titles the total saving is $83.60 (7.31%) and the confirmatory audit costs about $475.80 — more than five quarters of saving. The pipeline adds a ledger obligation, a gate the team cannot yet prove it meets, and a new failure mode (F3) on the one field that carries accessibility consequences. A team with limited reviewer attention may reasonably spend it on judgment-field disputes instead. **Check.** $475.80 / $83.60 = 5.69 quarters to amortize the confirmatory audit at the small slate.

**Acceptable alternatives.** "Do nothing until the peak slate" is the same argument with a timing condition and is the stronger version, since saving rises to 14.13% at 384 titles.

**Follow-up.** "What would make you reverse that position?" — A measured baseline handling time above 5.5 min/title, a measured baseline escape rate near zero (making the gate non-binding), or a slate above roughly 300 titles where the audit amortizes.

**Red flags.** Treating automation as inherently progressive; ignoring the audit cost; dismissing accessibility risk as small because the observed error count is one.

**Scoring.** 0: no case for inaction. 1: cost-only. 2: cost plus operational obligations. 3: adds the amortization arithmetic and the timing option.

**Answer support.** Sections 6.4 to 6.6, 10.

**Q12 / debugging / 4 min.** *"Halfway through a slate, the publication gate starts rejecting 30% of titles. Diagnose."*

**Strong answer.** The gate is deterministic, so a rate change means an input change. Competing causes: (a) a vocabulary version mismatch introduced mid-slate; (b) ledger rows not written because the review console lost a write path; (c) duplicate catalogue records making `title_id` ambiguous so acceptance rows attach to the wrong record; (d) a required field newly added to the schema and never reviewed. Discriminating checks: group rejections by reason code from the gate itself — a well-instrumented gate names the failure, and the work is to confirm the reason rather than guess; then check version pins (a), ledger completeness for a sample of rejected titles (b), canonical-key duplication (c), and schema diff against the previous slate (d). **Conclusion.** A gate that cannot report its reason codes is itself the first defect to fix.

**Acceptable alternatives.** Bisecting by reverting the last pipeline change is acceptable if the ledger shows which change preceded the first rejection.

**Follow-up.** "You find (b). What must you not do?" — Re-inject acceptance rows by hand from console screenshots; the ledger must be reconstructed from an auditable source or the titles re-reviewed.

**Red flags.** Loosening the gate to unblock the slate; treating rejections as noise; hand-editing the ledger.

**Scoring.** 0: loosens the gate. 1: one hypothesis. 2: reason-code-first and named checks. 3: adds the ledger-integrity rule.

**Answer support.** Sections 8b, 8d, 12 F2 and F6.

**Q13 / transfer / 3 min.** *"Transfer the mechanism to adverse-event triage in a pharmacovigilance workflow."*

**Strong answer.** The grain becomes a case released to a regulatory queue; the escape becomes a serious event released without human review or with a wrong seriousness classification; the tiers become deterministic (structured field extraction from a form) versus judgment (causality assessment). The mechanism transfers: measure escapes per released case, keep causality manual, automate only fields with an authoritative source, and derive the audit size from the target rate. What breaks first: the baseline escape rate is usually regulated and measured, so the gate is not a free parameter — it is imposed, and the analysis becomes "can we meet the imposed gate" rather than "what gate should we set". **Check.** If the imposed rate is 0.5 per 100 cases, zero-escape demonstration needs n ≥ ln(0.025)/ln(0.995) = 737 cases.

**Acceptable alternatives.** A candidate may reframe the transfer around expected harm rather than count rates; that is legitimate if the harm weights are declared.

**Follow-up.** "What is the first assumption that breaks?" — That the reviewer is the release gate; in regulated workflows an external authority may be the gate, and internal review is a filter whose escape rate is measured against the authority's findings.

**Red flags.** Transferring the 2.00 gate; assuming causality assessment can be automated; ignoring the externally imposed threshold.

**Scoring.** 0: no transfer. 1: names a superficial analogy. 2: transfers grain, tier, and gate. 3: identifies the imposed-gate difference and computes a new sample size.

**Answer support.** Section 13 (portability and adaptation); formula N-01.

**Q14 / judgment / 3 min.** *"The business asks for one number to put on a slide. What number, and what sentence accompanies it?"*

**Strong answer.** The number is **7.31% of total staffed cost**, with the sentence: "at 240 titles and $52 per staffed hour, including a recurring 60-title audit; it rises to 14.13% at a 384-title slate, and it is zero below $38.07 per hour." The alternative number — 25.49% — is annotation-only and would be misleading on a decision slide because it excludes the control that makes the quality claim possible. **Check.** 83.60/1,144.00 = 7.31%; 258.56/1,830.40 = 14.13%; 6.0 × 38.07 − 228.40 ≈ 0.

**Acceptable alternatives.** Reporting staffed hours returned (2.0 of 22.0) is acceptable and arguably better, because it does not depend on the rate card.

**Follow-up.** "The CFO says your rate is $70. Recompute." — Labour 16.0 h × $70 = $1,120.00 + $228.40 = $1,348.40 versus baseline 22.0 × $70 = $1,540.00; saving $191.60 = 12.44%. The saving *grows* with rate, so a higher rate strengthens the case — but only above the $38.07 break-even, below which it reverses.

**Red flags.** Quoting the highest number available; omitting the break-even; quoting a percentage without the slate size and rate that produce it.

**Scoring.** 0: quotes 25.49% with no caveat. 1: quotes a number with one caveat. 2: quotes the audited number with slate and rate. 3: adds the break-even and offers the rate-independent alternative.

**Answer support.** Sections 6.4, 6.5, 7.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Subscription streaming catalogue operations; title metadata enrichment before release |
| Problem type | Field-level assisted classification and extraction with a human release gate |
| Method | Tier the 18 fields by consequence and verifiability; automate 9 descriptive + 2 probe-verified technical fields; keep 3 judgment fields with a named cataloguer; gate on escapes per 100 released titles |
| Primary result | Not measured. Illustrative arithmetic: 90.45% field accuracy on 660 audited values; 1.67 critical escapes per 100 titles (95% upper bound 5.96); 7.31% total cost saving at 240 titles, 14.13% at 384 |
| Key lesson | Aggregate accuracy cannot bind a release gate; escapes per released item and staffed minutes per item can. The benefit is a labour-time differential, so it vanishes below a $38.07/h break-even and when the baseline is already fast |
| Tags | evaluation-mismatch; human-in-the-loop; cost-latency-tradeoff |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-08-01-01 / 2.0 / 2026-11-09 |
| Evidence level | illustrative (constructed slate, audit, and rates); arithmetic checked; no production artifact |
| Implementation stage | proposed; no pilot, release, or measurement |
| Baseline / primary metric | Vendor all-manual annotation at 5.5 min/title / critical escapes per 100 released titles, gate ≤2.00 with upper bound below gate |
| Key limitation | The baseline's own escape rate is unmeasured, and one escape in 60 titles cannot demonstrate a 2-per-100 gate (183 zero-escape titles required) |
| Review trigger / owner | Taxonomy change, new maturity obligation, staffing or price change, or a pilot critical escape / catalogue operations lead |
