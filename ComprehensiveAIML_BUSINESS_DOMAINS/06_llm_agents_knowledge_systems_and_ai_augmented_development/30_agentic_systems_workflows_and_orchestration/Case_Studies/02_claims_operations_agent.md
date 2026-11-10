---
case_id: "CASE-2026-032"
case_title: "Claims operations agent: promote a release bundle only when its guardrails survive a canary"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "llm-agentic"
domain: "Insurance claims operations / first-notice-of-loss triage"
primary_pattern: "atomic-release-bundle"
secondary_patterns: ["rollout-regression", "human-in-the-loop"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Claims operations director with model risk sign-off"
source_index: "INDEX.md"
review_trigger: "Any change to claim packet schema, coverage policy, model, prompt, retrieval index, or reviewer capacity"
---

# Claims Operations Agent: Release Only the Bundle That Survives Its Canary

**Evidence declaration.** This is an ILLUSTRATIVE design exercise.
The insurer, claim counts, staffing costs, model prices, and guardrail rates are constructed teaching values.
They are arithmetically derived and independently checked, but no organization measured them.
The source inventory named a claims-agent release and incident brief; its body was unavailable, so mechanism claims below are general design reasoning, not source reconstruction.
Implementation stage is `proposed`. Document state is `draft`. Editorial review is `not reviewed`.

## 1. The One-Line Framing

> Given one first-notice-of-loss claim packet and a versioned agent release bundle, decide whether to promote that bundle from canary to full traffic for **[stakeholder]** so that triage time falls without increasing missed-coverage-signal rate, subject to a hard rule that no claim decision is made without a licensed adjuster, compared with a rules-plus-human baseline.

The unit of analysis is one claim packet at the moment of triage.
The input is a structured FNOL packet plus attachments, an authenticated adjuster identity, and a release identifier.
The output is a triage proposal: coverage-route suggestion, missing-document list, urgency band, and a citation set for each suggestion.
The beneficiary is the claims operations team and, indirectly, the claimant waiting for a decision.
The decision owner is the claims operations director with model risk sign-off.
The binding constraint is that the model output is a proposal attached to the adjuster's own decision record; it never authorises payment or denial.
The baseline is a rule-based routing table applied by an adjuster who reads the packet.
The reusable pattern is an atomic release bundle: model, prompt, tool schemas, retrieval index, policy table, and evaluation suite are promoted and rolled back as one immutable set.
The secondary patterns are rollout regression and human-in-the-loop review.
The decisive tradeoff is that a release which improves average handling time while raising the missed-signal rate is not promotable, no matter how attractive its labour arithmetic looks.

**Decision frequency.** One promotion decision per release candidate; roughly one candidate per month in the illustrative operating rhythm.
**Success condition.** Median triage minutes fall by at least the agreed margin while the missed-coverage-signal guardrail stays inside its upper bound and no claim is decided without an adjuster.
**Evidence limit.** Nothing below demonstrates that a real bundle improves anything; it demonstrates how the promotion decision is computed and bounded.
**Takeaway.** Promotion is a bundle-compatibility decision with a statistical gate, not a model-quality opinion.

## 2. Context & Background

First-notice-of-loss handling begins when a claimant reports damage, injury, or loss.
An intake step creates a claim record with policy number, loss date, loss type, and free-text description.
An adjuster then reads the packet, decides which coverage section applies, identifies missing documents, and sets urgency.
The triage step is narrow but consequential: a mis-routed claim delays a valid payment, and a missed fraud or subrogation signal costs the insurer money later.

The baseline routing table maps loss type and a handful of codes to a queue.
It is fast and deterministic.
It fails on mixed-cause losses, on descriptions that mention a second peril, and on documents that arrive in the wrong slot.
Adjusters compensate by reading everything, which is where the fourteen minutes go.

The illustrative workload contains 18,000 claims per staffed week.
Five staffed days at nine staffed hours each give 45 staffed hours, or 162,000 staffed seconds.
The mean arrival rate is 18,000 / 162,000 = 0.1111 claims per staffed second, equivalent to 3,600 claims per staffed day.
These are calendar-independent staffed minutes; a claim arriving at 02:00 waits until the next staffed window, and the queue arithmetic below must not be read as a 24-hour service level.

| Context field | Illustrative definition | Boundary |
|---|---|---|
| Domain | Personal lines property and motor claims | Not commercial liability |
| Workflow stage | Triage after intake, before adjudication | No payment authority |
| Decision owner | Claims operations director | Not the model provider |
| Users | Licensed adjusters | Proposal, not instruction |
| Baseline | Rule table plus full manual read | Retained as fallback |
| Proposed aid | Bundle-proposed routing and document gaps | Requires adjuster acceptance |
| Scale | 18,000 claims per staffed week | Constructed |
| Timing | Weekly arrivals, daily peaks | Peak unmeasured |
| Geography | Single jurisdiction, single language | Not multilingual |
| Prior attempt | Keyword rules only | No learned routing deployed |
| Applicable requirements | Internal model-risk policy | Not legal advice |

The manual baseline persists for three reasons that a promotion case must respect.
First, the adjuster is accountable, and accountability without reading is not achievable.
Second, an adjuster notices absences: a missing photograph, a policy endorsement, a prior claim.
Third, the rule table is cheap to audit; every route is explainable by one rule identifier.

The proposed bundle does not replace the adjuster.
It changes the starting point: the adjuster receives a proposed route, a document gap list, and citations into the packet.
The adjuster can accept, change, or reject each field.
Every rejection is a labelled observation about the bundle, which is why the rejection rate is a first-class release metric rather than a nuisance.

**Stakeholder tension.** Operations wants throughput; model risk wants bounded exposure; adjusters want proposals that save reading rather than add a second reading; claimants want speed and correctness.
These goals conflict at the margin: a bundle that proposes aggressively saves more minutes and creates more false urgency bands.

**Takeaway.** The case is about a governed promotion of a bundle inside an existing accountable workflow, not about autonomy.

## 3. The Business Question

The illustrative stakeholder question, labelled as a paraphrase rather than a quotation: "Can we move the new claims triage bundle from a five-percent canary to full traffic without raising missed coverage signals or shipping a bundle we cannot roll back cleanly?"

| Decision-contract field | Case contract |
|---|---|
| Question | Promote bundle R2 to full triage traffic, hold at canary, or roll back to R1 |
| Owner | Claims operations director with model-risk co-signature |
| Trigger | R2 canary completed its planned exposure window on the claims population |
| Available actions | Promote; extend canary; hold; roll back; promote to a narrower segment |
| Value | Lower triage minutes per claim; fewer missing-document round trips |
| Downside | Missed coverage signal, wrong urgency band, adjuster rework, rollback ambiguity |
| Objective | Reduce median triage minutes while keeping the guardrail inside its upper bound |
| Hard constraint | No claim decision without an adjuster; bundle components promoted together |
| Scope | Personal lines property and motor, single jurisdiction, staffed hours |
| Out of scope | Fraud scoring thresholds, payment amounts, reserving, subrogation referral |

The objective is measured in staffed minutes per accepted proposal, not per offered proposal.
A bundle that proposes on fewer claims can look fast per proposal while helping almost nobody; the denominator must be eligible claims, and both numbers must be reported.
The guardrail is defined over eligible claims where the packet contains a coverage signal that the bundle failed to surface and the adjuster later found it.
The guardrail's numerator is claims with at least one such miss; the denominator is eligible claims in the window; the window is the canary exposure period.

**Counterfactual.** Holding the release at R1 costs nothing measurable in labour terms and defers the improvement.
Extending the canary costs calendar time and consumes reviewer attention, because every canary claim still receives the full labelled review.
Rolling back costs a release cycle and, if R2 already produced triage proposals that adjusters accepted, some accepted proposals may need re-checking for the specific fields that changed.
Narrow promotion, for example property-only, is a real option when the regression is concentrated in one loss family.

**Why the obvious framing fails.** The obvious framing asks "is the new model better?" That question cannot be answered usefully, because the bundle changes six things at once and the canary only observes the assembled behaviour.
A model-only comparison would hold the prompt and index fixed, which is not the release artefact anyone would deploy.
The promotion question is therefore about the composed bundle's behaviour under the guardrail, plus the ability to identify which component to change if the guardrail fails.

**Takeaway.** Answer a promotion question with a bundle-level comparison, a predeclared guardrail, and a rollback that restores every component.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | Adjuster accept/reject labels are captured per field | Data | Field-level audit of 200 sampled claims | unverified | Rejection analysis impossible; cannot attribute regressions | Manual sample review only | Any UI change / ops analytics |
| A-02 | The coverage-signal guardrail can be adjudicated from the packet | Data | Dual review of 120 claims with disagreement adjudication | unverified | Guardrail noisy, promotion decision unstable | Widen to team lead adjudication | Policy wording change / model risk |
| A-03 | Claim packets in canary match full-traffic mix | Data | Compare loss-type and channel mix canary vs trailing four weeks | unverified | Canary result does not transfer | Stratified promotion | Seasonal shift / marketing campaign |
| A-04 | Bundle components can be pinned and rolled back together | Infrastructure | Staged rollback drill on a shadow lane | unverified | Partial rollback leaves mixed versions | Freeze promotions until drill passes | Registry change / platform team |
| A-05 | Reviewer capacity exists for labelled canary review | Business | Capacity ledger against arrival rate | unverified | Canary underpowered or delayed | Extend window, reduce scope | Staffing change / ops |
| A-06 | LLM price and token profile stay fixed during the window | Business | Dated price sheet and token histogram | unverified | Cost arithmetic drifts | Reprice and re-decide | Provider pricing change / finance |
| A-07 | Missed-signal rate is stationary within the window | Model | Split-window comparison | unverified | Guardrail crossing is time-confounded | Longer window, sequential test | Peril season / catastrophe event |
| A-08 | No claim decision is produced without an adjuster record | Safety | Executor-side assertion and audit query | unverified | Regulatory and ethical exposure | Disable proposal path | Any workflow change / compliance |
| C-01 | Promotion requires both operations and model-risk approval | Governance | Approval record with two identities | confirmed by scenario | Single-signature promotion possible | None; block | Policy revision |
| C-02 | Canary share is capped at 5% of eligible claims | Governance | Assignment service configuration | confirmed by scenario | Blast radius grows | Reduce share | Risk appetite change |
| C-03 | Full-traffic promotion must be reversible within one staffed hour | Operations | Rollback drill timing | unverified | Containment slower than promised | Pre-stage artifact pins | Registry change |
| C-04 | Claim packets contain special-category data under retention limits | Privacy | Data-classification record | confirmed by scenario | Logging design invalid | Redact and shorten retention | Retention policy change |
| C-05 | Guardrail upper bound is 1.20% of eligible claims | Governance | Signed threshold in evaluation manifest | confirmed by scenario | Promotion rule undefined | Re-derive with owners | Annual threshold review |
| C-06 | Reviewers cannot see model identity during adjudication | Evaluation | Blinded review harness check | unverified | Reviewer bias inflates apparent quality | Re-run blinded | Tooling change |

The three assumptions most likely to reverse the recommendation are A-03, A-04, and A-07.
If canary mix differs materially from full traffic, the guardrail estimate transfers poorly, and a stratified or narrower promotion is the correct action.
If rollback cannot restore all six bundle components, a successful canary does not justify promotion, because containment is part of the promotion contract.
If the missed-signal rate is non-stationary within the window, for example because a storm event changes the loss mix mid-window, the estimate is confounded and the window must be re-run or analysed by peril stratum.

**Takeaway.** Each critical assumption has a check and a consequence; an unverified assumption is a reason to extend the canary, not a reason to promote.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Specification |
|---|---|
| Unit of analysis | One claim packet version at triage time; one row per claim per bundle |
| Volume and window | 18,000 claims per staffed week (illustrative); canary window expressed in eligible claims |
| Keys | `claim_id`, `packet_version`, `bundle_id`, `adjuster_id` (pseudonymous), `review_id` |
| Inputs | Loss type code, loss date, policy effective window, free-text description, attachment manifest, channel |
| Outcome | Adjuster final route, document requests raised, urgency band, accepted/changed/rejected proposal fields |
| Guardrail label | Binary: at least one coverage signal present in packet but absent from proposal, adjudicated by two reviewers |
| Population | Personal lines property and motor claims in the jurisdiction |
| Exclusions | Claims with no packet at triage; duplicate submissions; claims already in litigation; test accounts |
| Join semantics | Packet version joined to the review record produced at the same version; later edits never overwrite the reviewed version |
| Splits | Assignment is by hashed claim identifier with a fixed salt, so the same claim always lands in the same arm |
| Maturity | Guardrail label is available only after reviewer adjudication, typically next staffed day |

The exclusion list matters for the denominator.
Claims with no packet at triage are not evidence about the bundle; they are an intake defect and belong to a separate metric.
Duplicate submissions inflate the numerator if they share a root cause and are counted twice; deduplicate on `claim_id` plus loss date.
Litigation claims follow a different route with legal involvement and are out of scope for this promotion.

The split uses a hash of the claim identifier with a salt fixed for the release.
Hashing rather than random assignment prevents a claim from being re-triaged into the other arm after a packet correction, which would contaminate the comparison.
The salt is part of the bundle manifest, so a re-run reproduces the same assignment.

### 5b. Tools & Stack

| Component | Role in this case | Version pin required |
|---|---|---|
| Prompt registry | Stores reviewer-facing instructions per bundle | Prompt hash |
| Model endpoint | Produces structured triage proposal | Model identifier and snapshot |
| Tool schemas | Define proposal fields and citation retrieval | Schema hash |
| Retrieval index | Supplies policy wording fragments for citations | Index build identifier |
| Policy table | Coverage rules used for validation | Policy version |
| Evaluation suite | Computes metrics and thresholds | Suite identifier |
| Assignment service | Applies hash-based canary assignment | Salt and configuration |
| Review UI | Captures field-level accept/change/reject | UI release |
| Audit store | Append-only records of proposals and reviews | Retention policy |
| Rollback harness | Restores the previous bundle pins | Drill record |

None of these components is optional, because the promotion contract is a statement about the assembled behaviour.
A prompt change without an index change still alters citations; an index change without a prompt change still alters which wording the proposal can cite.
Treating the bundle as one artefact is what makes rollback meaningful.

### 5c. Method Summary

The method has four stages: assemble and pin, shadow, canary, and decide.

```text
[Claim packet v] -> [Validate schema + policy fields]
                          |
                          v
                  [Bundle R2 pinned]
                          |
             +------------+-------------+
             v                          v
   [Shadow lane: no reviewer]   [Canary lane: 5% + labelled review]
             |                          |
   compare proposals offline      compute guardrail + minutes
             |                          |
             +------------+-------------+
                          v
             [Promotion decision with predeclared rule]
                          |
        +-----------------+------------------+
        v                 v                  v
   [Promote]        [Extend canary]     [Roll back]
```

| Candidate | Evidence considered | Strength | Limitation / avoid when | Status | Revisit if |
|---|---|---|---|---|---|
| Rule table only (R1) | Three years of routing rules | Auditable, cheap, deterministic | Misses mixed-cause losses | Baseline | Never; it is the comparator |
| Rule table plus LLM summary | Shadow comparison | Cheap, low authority | Summary does not route | Rejected: no routing gain measured in design review | If summaries reduce reading independently |
| Bundle R2 with citations | Canary | Composed behaviour, explainable fields | Needs labelled review capacity | Chosen for canary | Guardrail breach or rollback failure |
| Model-only upgrade, fixed prompt | Offline suite | Isolates model contribution | Not a deployable artefact | Considered, not tested | If attribution is required after a regression |
| Manual triage with no rules | Operations log | Maximally careful | Slow, inconsistent | Rejected | If guardrail cannot be measured |

The method deliberately does not fit a model-comparison experiment.
The comparison is between two operational policies: R1 plus full manual read, and R2 plus assisted read.
That is why the canary is the unit of evidence and why the offline evaluation suite is used as a gate rather than a verdict.

### 5d. Data provenance, quality, exclusions and leakage controls

Provenance begins with the claim record and its packet version.
Packets are immutable once submitted for triage; corrections create a new version with a new review.
The guardrail label is derived by two reviewers independently reading the packet and the proposal, then recording whether a coverage signal present in the packet was absent from the proposal.
Disagreements go to a team lead, and the adjudicated label is stored with both original labels so agreement can be measured.

Leakage risks in this case are subtle rather than conventional.
The first risk is label leakage through the proposal text: a reviewer who sees the proposal first may fail to notice signals the proposal omitted, because the proposal anchors attention. The mitigation is that reviewers read the packet first, then the proposal, with the packet read timed and recorded.
The second risk is outcome leakage through later claim history. A reviewer who knows the claim was paid may infer that coverage applied. The mitigation is that the review record is created before payment activity is visible in the review tool.
The third risk is assignment leakage through packet corrections; the hash-based split prevents re-randomisation after a correction.

Missingness is handled explicitly: claims with an incomplete attachment manifest are reported as a separate stratum, not silently dropped, because the bundle's document-gap feature is most useful exactly there.
Outliers in description length are retained; truncation policy is part of the bundle and its effect is measured by comparing truncated and untruncated shadow proposals on the same packet.

### 5e. Reproducibility and method-selection ledger

| Artifact/source ID | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| SRC-01 | Internal model-risk policy (illustrative) | current | Defines dual approval | restricted | Cannot be quoted or verified here |
| SRC-02 | Claims routing rule table (illustrative) | v7 | Baseline comparator | restricted | Rule precedence not reproduced |
| RUN-01 | Canary guardrail computation | release candidate R2 | Guardrail estimate | constructed | No real claim data used |
| RUN-02 | Labour arithmetic | this file, Section 6 | Cost comparison | constructed | Prices illustrative |
| RUN-03 | Rollback drill plan | this file, Section 8d | Containment check | not executed | Drill outcome unknown |
| EVAL-01 | Evaluation suite manifest | suite 4 | Threshold and power | constructed | Suite composition not reproduced |

**Claim ledger**

| Claim ID | Exact claim and scope | Evidence type | Source/artifact | Supports | Does not establish |
|---|---|---|---|---|---|
| CLM-01 | Illustrative canary of 180 eligible claims per day at 5% share | illustrative | RUN-01 | Sizing arithmetic in 6.3 | Any real exposure |
| CLM-02 | Baseline labour of 4,200 staffed hours per week at 14 minutes per claim | illustrative | RUN-02 | Economics in 6.4 | Real staffing |
| CLM-03 | Break-even review time of 13.34 minutes | derived | RUN-02 | Sensitivity in 6.5 | Provider performance |
| CLM-04 | Power-based sample need of 2,905 claims per arm | derived | EVAL-01 | Guardrail gate | Real effect size |
| CLM-05 | Rollback contract requires six pinned components | derived | manifest design | Containment rule | Actual drill success |

**Takeaway.** The evidence chain is explicit: constructed inputs, derived arithmetic, and an unexecuted drill are labelled as such wherever they influence the recommendation.

## 6. Analysis Walkthrough

### 6.1 Bundle inventory and compatibility surface

The first step is to enumerate what actually changes between R1 and R2.

| Component | R1 | R2 | Compatibility risk |
|---|---|---|---|
| Model | Endpoint A snapshot a1 | Endpoint A snapshot a2 | New tokenizer; length accounting shifts |
| Prompt | p11 | p14 | Adds citation requirement; output schema tightened |
| Tool schema | t5 | t6 | `urgency_band` gains a fourth value |
| Retrieval index | i3 | i4 | 12% of policy fragments re-chunked |
| Policy table | pol2025-11 | pol2026-02 | One coverage exclusion clarified |
| Evaluation suite | e2 | e3 | Guardrail definition unchanged; minutes metric adds field-level latency |

Two compatibility facts follow immediately.
First, the fourth urgency band is a schema change: any downstream consumer that enumerates three bands will either reject or misinterpret the new value.
Second, the policy table change alters the meaning of a correct proposal for one exclusion, so proposals generated under R1 and reviewed under pol2026-02 are not comparable without restating them.

The compatibility matrix must therefore be part of promotion, not an afterthought.

```text
[New bundle] -> [Consumer inventory] -> [Schema diff] -> [Compatible consumers] -> promote
                        |                     |
                        v                     v
                 [Incompatible list] -> block or adapt
```

The count of consumers is small but not zero: the adjuster UI, the document-request service, the reporting extract, and the audit schema all read proposal fields.
A promotion that ignores the reporting extract produces silent nulls in a weekly report, which is a failure that no guardrail on claims would catch.

### 6.2 Canary share and exposure arithmetic

The canary share is 5% of eligible claims.
With 3,600 claims per staffed day and 62% eligible, eligible claims per day are 3,600 × 0.62 = 2,232.
At a 5% share, the canary receives 2,232 × 0.05 = 111.6 eligible claims per day.
Reviewer capacity in the illustrative ledger is 180 labelled reviews per staffed day, so the canary is reviewable with headroom of 68.4 reviews per day.

The 62% eligibility rate is itself an assumption, not a measurement.
If eligibility is 55%, canary volume is 3,600 × 0.55 × 0.05 = 99 claims per day, and the time needed for a fixed sample grows proportionally.

### 6.3 Statistical gate on the guardrail

The guardrail is a small-proportion comparison: missed-signal claims per eligible claim, R1 versus R2.
A decision rule that says "promote if the canary rate is below 1.20%" is a threshold rule on a noisy estimate; it does not control the probability of promoting a bundle whose true rate is above the threshold.

A two-proportion design is more defensible. Using a one-sided test at 5% with 80% power, and illustrative true rates of 0.8% for R1 and 1.6% for R2, the required per-arm sample is

```text
FORMULA [C-01]: n = (z_alpha + z_beta)^2 * (p1(1-p1) + p2(1-p2)) / (p2 - p1)^2
WHERE: n is eligible claims per arm; p1, p2 are guardrail proportions;
       z_alpha = 1.959964 (two-sided 5%); z_beta = 0.841621 (80% power).
CONSTRAINT: independent claims, stable rates within the window, no interim peeking.
NUMERICAL EXAMPLE: p1=0.008, p2=0.016;
  (2.801585)^2 = 7.848881;
  0.008*0.992 = 0.007936; 0.016*0.984 = 0.015744; sum = 0.023680;
  n = 7.848881 * 0.023680 / 0.000064 = 0.1858647 / 0.000064 = 2904.1 -> 2,905 per arm.
INTERPRETATION: about 2,905 canary claims and 2,905 comparable baseline claims are needed
to distinguish these two rates with the stated error rates.
```

The baseline arm does not need to be contemporaneous at the same rate of randomisation, but it does need the same adjudication protocol.
Using a historical baseline with a different reviewer pool is a validity threat, not a shortcut.

At 111.6 canary claims per staffed day, 2,905 canary claims require 2,905 / 111.6 = 26.0 staffed days, about five staffed weeks.
If the baseline arm must also be reviewed concurrently at the same 5% share, the window doubles in calendar terms unless the baseline review is drawn from the existing audit sample.

A cheaper sequential design is available: a group-sequential or sequential probability ratio test can stop early when the evidence is decisive.
It changes the decision rule and must be declared before the window opens; a post-hoc "we looked at day nine and it looked fine" is not a decision rule.
The illustrative case keeps the fixed-sample design for clarity and records the sequential alternative as untested.

### 6.4 Labour economics of the bundle

The economics must be computed on staffed minutes, not on model tokens, because tokens are a rounding error next to adjuster time in this workload.

```text
FORMULA [C-02]: H_base = N * m_base / 60
WHERE: H_base staffed hours per week; N claims per week; m_base minutes per claim; 60 minutes/hour.
NUMERICAL EXAMPLE: N=18,000; m_base=14; H_base = 18,000*14/60 = 4,200 hours.
INTERPRETATION: 4,200 staffed hours per week at 36 productive hours per FTE is 116.67 FTE-weeks.
```

The assisted path applies only to eligible claims, and its saving depends on how much reading the proposal removes.

```text
FORMULA [C-03]: H_assist = [N_e * m_assist + (N - N_e) * m_base] / 60
WHERE: N_e eligible claims per week; m_assist assisted minutes per eligible claim.
NUMERICAL EXAMPLE: N_e = 18,000*0.62 = 11,160; m_assist = 6;
  H_assist = [11,160*6 + 6,840*14]/60 = [66,960 + 95,760]/60 = 162,720/60 = 2,712 hours.
INTERPRETATION: the assisted path saves 4,200 - 2,712 = 1,488 staffed hours per week.
```

At an illustrative loaded cost of 52 currency units per staffed hour, the labour difference is 1,488 × 52 = 77,376 per week.
The model cost for the same week, using 6,200 input tokens and 900 output tokens per eligible claim, priced at 3.00 per million input tokens and 15.00 per million output tokens, is

```text
FORMULA [C-04]: c_claim = t_in * p_in / 1e6 + t_out * p_out / 1e6
NUMERICAL EXAMPLE: 6,200*3.00/1e6 = 0.0186; 900*15.00/1e6 = 0.0135;
  c_claim = 0.0186 + 0.0135 = 0.0321 per eligible claim.
  Weekly model cost = 11,160 * 0.0321 = 358.24.
INTERPRETATION: model spend is 0.46% of the labour difference; the decision is not token-price-driven.
```

Adding an illustrative platform and review-tooling charge of 6,000 per week, the net difference is 77,376 − 358.24 − 6,000 = 71,017.76 per week.
That number is a modelled difference under constructed assumptions, not a saving anyone realised.

**Check.** 1,488 hours × 52 = 77,376. 11,160 × 0.0321 = 358.236, which rounds to 358.24. 77,376 − 6,358.24 = 71,017.76. All three lines reconcile.

### 6.5 Sensitivity and the break-even review time

The most fragile input is `m_assist`, because it is the quantity the bundle actually changes and the one no offline suite estimates credibly.

```text
FORMULA [C-05]: m* = m_base - (c_fixed * 60) / (c_hour * N_e)
WHERE: m* break-even assisted minutes; c_fixed weekly fixed model plus platform cost;
       c_hour loaded hourly cost; N_e eligible claims per week.
NUMERICAL EXAMPLE: c_fixed = 6,358.236; c_hour = 52; N_e = 11,160;
  m* = 14 - (6,358.236*60)/(52*11,160) = 14 - 381,494.16/580,320 = 14 - 0.6574 = 13.34 minutes.
INTERPRETATION: the assisted path stops paying for itself only if it costs more than
about 13.3 minutes per eligible claim - that is, if it barely saves any reading at all.
```

| Assisted minutes per eligible claim | Weekly net difference | Reading of the result |
|---|---|---|
| 4.5 | 85,525.76 | Optimistic; requires near-perfect proposal acceptance |
| 6.0 | 71,017.76 | Base case |
| 8.0 | 51,673.76 | Still positive; plausible if adjusters re-read documents |
| 10.0 | 32,329.76 | Positive but modest |
| 12.0 | 12,985.76 | Thin margin; platform cost assumption dominates |
| 13.34 | 0 | Break-even |
| 14.0 | -381.49 | Bundle adds cost with no time benefit |

The economics are robust over a wide range, which is exactly why they must not be the gate.
A bundle can be strongly economic and still unpromotable because it raises the missed-signal rate.
Conversely, a bundle with a thin margin and a clean guardrail may be worth promoting for consistency reasons rather than cost.

The second fragility is the review cost of the canary itself: 2,905 labelled reviews at, illustratively, 9 minutes each is 435.75 hours, or about 12.1 FTE-weeks at 36 hours.
That is a real cost that must be staffed, and it is not in the per-claim economics above.

### 6.6 Capacity and queue consequences

Promotion changes arrival-to-first-touch time if reviewer capacity does not scale with the new workload.
Under the assisted path, an adjuster handles an eligible claim in 6 minutes instead of 14, so service capacity per staffed hour rises from 60/14 = 4.29 claims to 60/6 = 10 claims for eligible work.
With 62% eligibility, blended capacity per adjuster-hour is 0.62 × 10 + 0.38 × 4.29 = 6.20 + 1.63 = 7.83 claims, against a baseline of 4.29 claims.

```text
FORMULA [C-06]: rho = lambda / (c * mu_s)
WHERE: rho utilisation; lambda arrivals per staffed hour; c adjusters on shift;
       mu_s claims per adjuster-hour.
NUMERICAL EXAMPLE: lambda = 3,600/9 = 400 claims per staffed hour;
  baseline: c=100, mu_s=4.29 -> capacity 429/hour -> rho = 0.932;
  assisted: mu_s=7.83 -> capacity 783/hour -> rho = 0.511.
INTERPRETATION: the bundle creates large headroom; it does not by itself reduce queueing delay,
because the binding constraint becomes document round-trips, not triage minutes.
```

A utilisation of 0.932 in the baseline implies long queues even before any model change, which is consistent with the operational complaint that started the project.
The assisted path's 0.511 utilisation is not a service-level guarantee; it is a fluid approximation that assumes constant arrivals, no abandonment, and no document wait.

**Takeaway.** The bundle's labour case is robust, its capacity effect is large, and neither fact licenses promotion without the guardrail.

### 6.7 What we tried first and why it did not settle the question

The first design compared model snapshots on an offline suite of 4,000 historical claims with fixed prompts and indices.
That comparison is reported as `not tested` for promotion purposes, because it answers a different question: it isolates a model effect that will not exist in the deployed bundle.
Its legitimate use is as a screening gate: if the offline suite shows a catastrophic guardrail regression, the canary is unnecessary.
Its illegitimate use is as a promotion verdict, because the composed bundle with the new prompt and index can behave differently in either direction.

A second design considered comparing R2 against R1 on the same claims in a shadow lane.
That is useful for cost and latency, and it cannot observe adjuster behaviour, which is where the minutes actually change.
Shadow therefore informs the canary design rather than replacing it.

### 6.8 Robustness table

| Stress or variation | Original condition | Alternative | Change in conclusion | Limitation | Decision consequence |
|---|---|---|---|---|---|
| Canary share | 5% | 10% | Sample reached in ~13 staffed days | Blast radius doubles | Risk sign-off required |
| Eligibility | 62% | 55% | Canary 99/day; window ~29 days | Constructed | Extend window |
| Guardrail bound | 1.20% | 1.00% | Sample need rises; promotion stricter | Power-dependent | More review capacity |
| Baseline rate | 0.8% | 1.2% | Smaller detectable difference; larger n | Rate uncertainty | Sequential design |
| Reviewer minutes | 9 | 6 | Review cost 290.5 hours | Reviewer skill | Staffing plan |
| Rollback drill | passes | fails | No promotion regardless of canary | Containment | Fix registry first |
| Loss mix drift | stable | storm week | Guardrail confounded | Stationarity | Stratify by peril |

### 6.9 Document-gap precision as a second, cheaper gate

The document-gap field has a different error structure from the coverage-signal field, and it deserves its own arithmetic rather than being folded into the guardrail.
A missing-document flag creates a request to the claimant; a correct flag shortens the claim, and a wrong flag costs a needless round trip and annoys the claimant.
The asymmetry is real but it is not the same as the coverage asymmetry, so the gap field can carry a numeric gate even when the coverage field cannot.

Illustratively, 38% of eligible claims are missing at least one document at triage: 11,160 × 0.38 = 4,240.8 claims per week.
The baseline rule table catches 55% of those, so it raises 2,332.44 correct requests.
Suppose the bundle flags a broader set: it raises a request on 71% of the truly-gap claims, and its precision - the share of its flags that are genuinely missing documents - is the unknown that decides whether the field helps.

If the bundle flags `f` claims and its precision is `p`, then the correctly flagged claims are `f × p` and its false flags are `f × (1 − p)`.
With `f = 4,240.8 × 0.71 = 3,011.0` claims per week, the incremental correct flags over baseline are `3,011.0 × p − 2,332.44`.

```text
FORMULA [C-08]: net_gap = (f*p - t_base) * c_miss - f*(1-p) * c_false
WHERE: f flagged claims/week; p precision in [0,1]; t_base baseline correct flags/week;
       c_miss staffed minutes saved per correctly flagged gap (3.5);
       c_false staffed minutes lost per false flag (4.0).
CONSTRAINT: flags are approximately independent across claims; costs are constant;
            c_false > c_miss because a false request consumes claimant and staff time.
NUMERICAL EXAMPLE: f=3,011.0, t_base=2,332.44, c_miss=3.5, c_false=4.0.
  At p=0.90: correct 2,709.9, incremental 377.5 -> +1,321.2 min;
             false 301.1 -> -1,204.4 min; net +116.8 min/week.
  At p=0.85: correct 2,559.3, incremental 226.9 -> +794.2 min;
             false 451.6 -> -1,806.4 min; net -1,012.2 min/week.
INTERPRETATION: the gap field pays only above 89.5% precision at these costs;
below that, extra flags cost more in round trips than they save.
```

| Precision `p` | Correct flags | False flags | Gross saving (min) | False cost (min) | Net (min/week) |
|---|---|---|---|---|---|
| 0.70 | 2,107.7 | 903.3 | −786.6 | −3,613.2 | −4,399.8 |
| 0.80 | 2,408.8 | 602.2 | +267.3 | −2,408.8 | −2,141.5 |
| 0.835 | 2,514.2 | 496.8 | +636.2 | −1,987.2 | −1,351.0 |
| 0.85 | 2,559.3 | 451.6 | +794.2 | −1,806.4 | −1,012.2 |
| 0.90 | 2,709.9 | 301.1 | +1,321.2 | −1,204.4 | +116.8 |
| 0.95 | 2,860.5 | 150.6 | +1,848.2 | −602.4 | +1,245.8 |

The exact break-even solves `(f p − t_base) × 3.5 = f (1 − p) × 4.0`.
Expanding, `3.5 f p − 3.5 t_base = 4.0 f − 4.0 f p`, so `7.5 f p = 4.0 f + 3.5 t_base`, and

```text
p* = (4.0*f + 3.5*t_base) / (7.5*f)
   = (12,044.0 + 8,163.5) / 22,582.5
   = 20,207.5 / 22,582.5 = 0.8948.
```

That threshold sits between the tabulated 0.85 and 0.90 rows, which is why neither row should be described as "the break-even point".
Reporting "the field helps above 89.5% precision" is the honest form; rounding to "about 85%" would overstate the margin by roughly five points and would promote a field that loses about a thousand staffed minutes per week at that precision.

The sensitivity to the false-flag cost is strong and worth stating explicitly.
If a false request costs only 2.5 minutes because a clerk rather than the claimant resolves most of them, the threshold falls to 86.9% precision, and the 0.85 row moves from −1,012.5 to −335.0 minutes per week: closer to viability but still negative.
At 6.0 minutes per false flag, which is plausible when the claimant must be re-contacted, the threshold rises to 91.7% and the 0.90 row turns negative.
The gate therefore depends on an operational assumption about who absorbs the rework, which is exactly the kind of assumption that a promotion case must verify rather than assume.

### 6.10 Review capacity during the canary

The canary consumes reviewer capacity that would otherwise be available for quality audit.
With 2,905 canary claims to review and a capacity of 180 labelled reviews per staffed day, of which 111.6 are consumed by new canary arrivals, the spare capacity for clearing the initial sample is 68.4 reviews per day.

```text
FORMULA [C-09]: T_days = n / (c_capacity - lambda_canary)
WHERE: n reviews required; c_capacity reviews/day; lambda_canary arrivals/day.
NUMERICAL EXAMPLE: n=2,905; c_capacity=180; lambda_canary=111.6;
  T_days = 2,905 / 68.4 = 42.5 staffed days.
INTERPRETATION: if the sample must be accumulated before adjudication, the window is
42.5 staffed days, not the 26.0 that dividing by arrivals alone suggests.
```

The difference between 26.0 and 42.5 staffed days is the difference between an unbounded backlog and a staffing plan.
Dividing by arrivals assumes that reviews happen instantly and that the sample can be analysed continuously.
Dividing by spare capacity assumes a queue that must be drained, which is the operational reality when the same reviewers also do routine audit.

Utilisation of the review pool is `111.6 / 180 = 0.62` during the window.
At that utilisation a fluid queue is stable, and a modest increase in arrival rate or a reviewer absence can push it toward instability.
The staffing consequence is that the canary should not start in a week with planned reviewer leave, which is a scheduling constraint that the arithmetic makes visible.

### 6.11 What the arithmetic does and does not decide

The labour difference, the break-even assisted time, the gap-precision threshold, and the review-capacity window are four independent quantities.
They point in different directions under plausible shifts:
the labour case is robust, the gap field is fragile at low precision, and the evidence window is staffing-bound.

| Quantity | Base value | Dominant sensitivity | Decision role |
|---|---|---|---|
| Weekly labour difference | 1,488 hours | Assisted minutes per claim | Economic viability |
| Break-even assisted time | 13.34 minutes | Fixed cost | Economic boundary |
| Gap precision threshold | 89.5% | Cost of a false request | Field-level gate |
| Canary window | 42.5 staffed days | Reviewer spare capacity | Evidence feasibility |
| Guardrail bound | 1.20% upper bound | True miss rate | Promotion gate |

**Takeaway.** Every material variation either lengthens the evidence window or narrows the promotion, and none turns a guardrail breach into a promotion.

## 7. The Finding

The bundle's labour arithmetic is strongly favourable under constructed inputs: 1,488 staffed hours per week, a net modelled difference of 71,017.76 per week, and a break-even assisted time of 13.34 minutes that leaves a wide margin.
That arithmetic does not answer the business question, because the promotion rule is dominated by the guardrail and by containment.
The canary as designed needs about 2,905 eligible claims per arm, or roughly 26 staffed days at a 5% share, and the illustrative guardrail rates would be distinguishable at that size.
No outcome is measured here: the case recommends the staged promotion protocol and its stop conditions, not a promotion.
The main boundary is that a canary result transfers to full traffic only if the mix is comparable and the rollback drill has passed; if either fails, extend or narrow the promotion rather than promoting on schedule.

**Interview cue.** State the answer for an operations director in one sentence, then state the one number that could reverse it.

## 8. System/Solution Design

### 8a. Offline / Training / Batch Path

There is no training step in this design; the bundle uses a hosted model and a retrieval index built from policy documents.
The offline path is a build-and-gate pipeline.

```text
[Policy documents] -> [Chunk + embed] -> [Index i4] --+
[Prompt p14] ------------------------------------------+--> [Bundle manifest]
[Tool schema t6] --------------------------------------+         |
[Policy table pol2026-02] -----------------------------+         v
                                                        [Offline suite e3]
                                                                 |
                                                    +------------+------------+
                                                    v                         v
                                            [Gate passes]             [Gate fails]
                                                    |                         |
                                            [Shadow lane]              [Block release]
```

Batch steps and their artefacts:

| Step | Input | Output | Gate |
|---|---|---|---|
| Index build | Policy corpus snapshot | Index i4 with chunk manifest | Chunk coverage check; no empty sections |
| Prompt assembly | Prompt p14 source | Compiled prompt hash | Schema conformance; citation instruction present |
| Suite run | 4,000 historical packets | Field-level scores | No catastrophic regression band |
| Shadow run | Live packets, no reviewer | Proposal log | Latency and cost profile within envelope |
| Manifest seal | All component hashes | Immutable bundle record | Six components present, none floating |

The suite is a gate, not a verdict: a pass admits the bundle to canary, and a fail blocks it.
The suite's composition is versioned with the bundle so that a later reviewer can tell which claims were used.

### 8b. Online / Serving / Real-Time Path

```text
[Claim packet + adjuster identity] -> [Authenticate + authorize adjuster]
                                              |
                                              v
                                   [Assign arm by claim hash]
                                              |
                     +------------------------+------------------------+
                     v                                                 v
              [R1 path: rules]                                  [R2 path: bundle]
                     |                                                 |
                     |                                    [Retrieve policy fragments]
                     |                                                 |
                     |                                    [Propose route + gaps + citations]
                     |                                                 |
                     +------------------------+------------------------+
                                              v
                                  [Validate proposal against pol2026-02]
                                              |
                                  +-----------+-----------+
                                  v                       v
                          [Invalid: no proposal]   [Valid: show to adjuster]
                                                          |
                                                  [Adjuster accept/change/reject]
                                                          |
                                                  [Record review + decision]
                                                          |
                                                          v
                                              [Queue assignment + audit]
```

Timeouts and degradation are explicit.
If retrieval exceeds 2,500 ms, the proposal is returned without citations and the UI marks it `unverified`; the adjuster then sees a route suggestion with no supporting text, which is a weaker but non-blocking state.
If the model endpoint fails, the claim falls back to the rule path and the event is counted in a `bundle_unavailable` counter; the adjuster is never blocked by the bundle.
If policy validation fails, the proposal is suppressed rather than shown with a warning, because a route that contradicts the current policy table is worse than no route.

Authorization happens at the proposal boundary: the adjuster identity must be entitled to the claim's line of business, and the proposal record stores the identity that saw it.
The model endpoint receives a packet rendered without direct identifiers where the task permits; the redaction step is part of the bundle and its misses are audited.

### 8c. Key Design Decisions

| Decision ID | Choice | Alternatives | Evidence | Tradeoff | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| DEC-01 | Promote six components atomically | Independent component rollout | Compatibility matrix in 6.1 | Coarser rollback granularity | Any consumer schema mismatch | Registry supports partial pinning |
| DEC-02 | Hash-based 5% canary assignment | Random per-request assignment | Reproducibility requirement | Slightly uneven daily split | Claim re-triaged after correction | Assignment service change |
| DEC-03 | Guardrail at 1.20% with power-based sizing | Point-estimate threshold | Statistical reasoning in 6.3 | Longer window | Rate non-stationarity | Threshold review |
| DEC-04 | Adjuster sees packet before proposal | Show proposal first | Anchoring risk in 5d | Slower per review | Reviewer time pressure | Blinded study result |
| DEC-05 | Suppress policy-invalid proposals | Show with warning | Safety preference | Fewer proposals | Over-strict validation | Validation error analysis |
| DEC-06 | Rollback target under one staffed hour | Same-day rollback | Containment promise | Requires pre-staged pins | Registry cold start | Infrastructure change |
| DEC-07 | No autonomous decision path | Auto-route low-risk claims | Accountability requirement | Higher cost | None permitted | Regulatory change |

### 8d. Architecture artifacts and operational readiness

| Area | Design answer |
|---|---|
| Interfaces | Versioned proposal schema; explicit `unverified` flag; 2,500 ms retrieval timeout; audit record per proposal |
| Data/state | Packet versions immutable; review records append-only; bundle manifest in registry |
| Reliability | SLI: share of eligible claims receiving a valid proposal within 4 s; target 99% over staffed hours |
| Resilience | Fallback to rules; circuit breaker on model endpoint; bounded retries with no proposal regeneration |
| Recovery | Rollback restores six pins; in-flight claims finish under their recorded bundle |
| Observability | Per-bundle guardrail dashboard; acceptance rate by field; `bundle_unavailable` counter |
| Deployment | Canary by hash; promotion by registry switch; rollback by pin restore |
| Security | Adjuster identity at proposal boundary; redaction audit; proposal access logged |
| Economics | 0.0321 per eligible claim model cost; 6,358.24 weekly fixed |
| Human workflow | Adjuster authority; team-lead adjudication; model-risk co-signature |

The rollback drill is a specific artefact: restore R1 pins in a shadow lane, replay 200 recorded packets, and confirm that the proposal schema, citations, and queue assignment match the R1 record exactly.
Passing the drill means the pins restore and the outputs reproduce; it does not prove that a live rollback is instantaneous under load.

**Takeaway.** The design's containment claim is only as good as the drill that demonstrates pin restoration.

## 9. Evaluation & Validation

| Evaluation field | Definition |
|---|---|
| Primary metric | Median staffed minutes per eligible claim from packet open to route confirmed |
| Guardrail | Share of eligible claims with at least one packet-present coverage signal absent from the proposal, adjudicated |
| Secondary | Field-level acceptance rate; document-gap precision; citation coverage; `bundle_unavailable` rate |
| Baseline | R1 with full manual read, trailing four staffed weeks |
| Pass rule | Guardrail upper confidence bound ≤ 1.20% and acceptance ≥ 65% with no field below 50% |
| Test design | Hash-assigned canary with blinded dual adjudication |
| Sample | 2,905 per arm for the stated difference; actual canary per day 111.6 |
| Uncertainty | Wilson interval per arm; paired difference where the same claim is reviewed under both bundles in shadow |
| Validity threat | Anchoring, mix drift, reviewer pool change, packet corrections |
| Decision | Not executed; canary is proposed |

The pass rule uses an upper confidence bound rather than a point estimate, which encodes the asymmetry between the two errors.
Promoting a bundle that breaches the guardrail is worse than extending a canary that would have passed, because the breach affects claimants and creates rework.

```text
FORMULA [C-07]: p_hat +/- z * sqrt(p_hat(1-p_hat)/n)
NUMERICAL EXAMPLE: 24 misses in 2,905 -> p_hat = 0.008262;
  se = sqrt(0.008262*0.991738/2905) = sqrt(0.000002820) = 0.001679;
  upper 95% (one-sided z=1.645) = 0.008262 + 0.002762 = 0.011024 = 1.10%.
INTERPRETATION: 24 misses in 2,905 is compatible with a true rate below the 1.20% bound.
```

```text
BOUNDARY CHECK: 36 misses in 2,905 -> p_hat = 0.012392;
  se = sqrt(0.012392*0.987608/2905) = sqrt(0.000004213) = 0.002053;
  upper = 0.012392 + 0.003377 = 0.015769 = 1.58% -> fails the 1.20% rule.
```

The boundary between 24 and 36 misses is 12 claims, which is why the sample size cannot be reduced casually.
At 2,905 claims the interval half-width is about 0.28 percentage points one-sided, comparable to the entire gap between the base rate and the threshold.

Guardrails beyond quality:

| Guardrail | Threshold | Why |
|---|---|---|
| Adjuster override of urgency band | ≤ 12% | Signals calibration failure |
| Citation precision (sample audit) | ≥ 90% | Wrong citations destroy trust |
| `bundle_unavailable` rate | ≤ 0.5% of eligible | Availability regression |
| Rollback drill | pass | Containment |
| Review backlog | ≤ 1 staffed day | Evidence freshness |
| Special-category exposure | zero unredacted fields in logs | Privacy |

**Takeaway.** The evaluation is falsifiable: a stated sample, a stated bound, and a stated consequence for failure.

## 10. Action Taken

No action taken as of 2026-11-10.
The case remains a recommendation.

| Action field | Record |
|---|---|
| Recommendation | Run the canary as specified, then decide with the predeclared rule |
| Approval status | Not submitted; would require operations and model-risk signatures |
| Implementer | Claims platform team, with evaluation owner |
| Effective scope | None; no bundle deployed |
| Release strategy | Proposed: 5% hash canary, then staged 25/50/100 |
| Stop/rollback | Guardrail upper bound breach, drill failure, or `bundle_unavailable` above 0.5% |
| Excluded actions | No autonomous routing, no payment effect, no multilingual expansion |

The exclusion list is part of the decision, not a limitation of the write-up.
A reviewer who asks why the agent does not simply route low-value claims should be answered with the accountability constraint: the adjuster's record is the decision, and the bundle informs it.

## 11. Outcome & Results

Outcome not yet measured.

| Metric | Baseline | After | Absolute change | Relative change | Uncertainty | Evidence | Limitation |
|---|---|---|---|---|---|---|---|
| Median triage minutes | 14 (illustrative) | not measured | — | — | — | constructed | No canary run |
| Guardrail rate | 0.8% (illustrative) | not measured | — | — | — | constructed | No adjudication |
| Field acceptance | not measured | not measured | — | — | — | none | UI not built |
| Model cost per claim | 0 | 0.0321 (modelled) | +0.0321 | undefined from zero | price-dependent | derived | Illustrative price |
| Net weekly difference | 0 | 71,017.76 (modelled) | +71,017.76 | undefined from zero | assumption-dependent | derived | Not realised |
| Review cost of canary | 0 | 435.75 hours (modelled) | +435.75 | undefined from zero | capacity-dependent | derived | Not staffed |

Relative change from a zero baseline is reported as undefined rather than as an infinite percentage.
The modelled net difference is not a realised saving; it is the arithmetic that would follow if the assisted minutes equalled six and prices held.

A measurement plan belongs here because the section is otherwise empty of evidence.
The plan: run the canary for 26 staffed days, adjudicate every canary claim, sample 2,905 baseline claims from the existing audit pool under the same protocol, compute Wilson bounds, and report median minutes with an interquartile range rather than a mean, because triage time is right-skewed.
Owner: evaluation lead. Decision trigger: guardrail upper bound crosses 1.20% at any interim analysis with the sequential rule declared in advance.

## 12. Failure Modes & What Went Wrong

| Failure ID | Trigger and symptom | Discriminating test | Cause status | Blast radius | Containment | Prevention / owner | Verification |
|---|---|---|---|---|---|---|---|
| F-01 | Proposals omit a coverage signal present in the packet | Dual adjudication on 40 sampled claims | hypothetical | Claimant delay | Revert to R1 for the affected loss type | Guardrail gate; model risk | Canary bound within 1.20% |
| F-02 | Urgency band value rejected by reporting extract | Schema diff against consumer inventory | hypothetical | Weekly report nulls | Restore R1 reporting mapping | Compatibility gate; platform | Consumer contract test |
| F-03 | Rollback leaves index i4 active with prompt p11 | Replay 200 packets after pin restore | hypothetical | Mixed behaviour | Freeze promotions; manual pin audit | Drill as promotion prerequisite | Byte-identical proposal replay |
| F-04 | Canary slower than baseline because adjusters double-read | Time-on-task instrumentation | hypothetical | Adoption stalls | Coaching, then reconsider design | Measure first-read acceptance | Median minutes decline |
| F-05 | Anchoring suppresses reviewer detection of omissions | Packet-first reading protocol audit | hypothetical | Guardrail understated | Re-run blinded subset | Review UI enforces order | Disagreement rate stable |
| F-06 | Packet mix drifts during a storm week | Stratified guardrail by peril | hypothetical | Wrong promotion decision | Extend window by stratum | Predeclare stratification | Stratum bounds reported |
| F-07 | Model endpoint outage blocks triage | `bundle_unavailable` counter | hypothetical | Queue growth | Rule fallback | Circuit breaker; capacity headroom | Fallback latency measured |
| F-08 | Citations point at superseded policy text | Citation audit against index i4 manifest | hypothetical | Wrong advice to adjuster | Suppress citations for affected section | Index build gate | Citation precision ≥ 90% |
| F-09 | Review backlog exceeds one day | Review queue depth | hypothetical | Stale evidence | Add reviewers or cut share | Capacity ledger | Depth ≤ 1 day |
| F-10 | Special-category text reaches logs | Log sampling and pattern scan | hypothetical | Privacy breach | Purge, restrict, notify per policy | Redaction gate | Zero unredacted fields |
| F-11 | Promotion decided on point estimate | Decision record review | hypothetical | Unsafe promotion | Recompute with bound | Predeclared rule in manifest | Bound recorded |
| F-12 | Same claim re-triaged into both arms | Assignment audit on corrected packets | hypothetical | Contaminated comparison | Exclude corrected claims | Hash with fixed salt | No claim in two arms |

F-03 deserves emphasis because it is the failure that makes every other claim about the bundle meaningless.
If rollback restores the prompt but not the index, the system runs a new retrieval surface under old instructions, and the resulting behaviour matches neither release.
The drill must compare proposal bytes, not merely confirm that a configuration value changed.

A second mechanism worth separating from F-01 is F-04, the double-read failure.
The bundle saves time only if the proposal replaces part of the read.
If adjusters distrust it enough to read everything and then check the proposal, minutes rise, and the economic case collapses while the quality metrics look excellent.
Time-on-task distinguishes this from a quality failure; a bundle with high acceptance and rising minutes is a trust-and-usability failure, not an accuracy failure.

**Takeaway.** The dangerous failures are composition failures and behavioural failures; neither is visible in a model-only metric.

## 13. Transferable Lesson

| Pattern field | Content |
|---|---|
| Canonical pattern | `atomic-release-bundle`: promote and revert a pinned set of model, prompt, tools, index, policy, and evaluation versions |
| General principle | When a deployed behaviour is produced by several versioned artefacts, treat the set as the unit of release and the unit of rollback |
| Preconditions | Immutable artefact registry; consumer inventory; reproducible assignment; adjudicated guardrail |
| Mechanism | Compatibility checks catch schema drift; pinned manifests make rollback meaningful; canary with power controls promotion risk |
| Supporting evidence | Constructed arithmetic in Sections 6.2-6.6; no measured promotion |
| Avoid when | The behaviour is produced by one component, or consumers are dynamically discovered and cannot be enumerated |
| Portability test | Ask whether a rollback that restores one artefact would leave a mixed behaviour; if yes, the pattern applies |
| Adaptation | For a retrieval product, the index and embedding model join the bundle; for a coding agent, the tool schema set and repository instruction files join it |

The lesson's boundary is important: bundling creates coarser control.
A team that wants to fix a single prompt typo must still promote a whole bundle, which costs a canary or a justified exception path.
The pattern trades release agility for recoverability, and it is the wrong trade when the components genuinely do not interact.

**Interview cue.** Give a domain where the bundle is too coarse, then say what would have to be true to make it necessary.

## 14. Connections & Cross-References

### 14a. Related cases, theory, and cheat sheets

| Relationship | Target | Shared or changed mechanism | Boundary learned |
|---|---|---|---|
| extends | [Approval gate for an enterprise agent](01_enterprise_agent_approval_gate.md) | Both separate proposal from authorised effect | That case binds one action; this one binds a release |
| uses-method | [Durable workflows and idempotent execution](../Cheat_Sheets/05_durable_workflows_and_idempotent_execution.md) | Effect keys and reconciliation support canary replay | A replay is not a rollback |
| uses-method | [Agent boundaries and request ownership](../Cheat_Sheets/01_agent_boundaries_and_request_ownership.md) | Request-path ownership identifies the promotion owner | Ownership is not a model property |
| contrasts-with | [ERP action agent](../../31_tools_mcp_protocols_and_enterprise_integrations/Case_Studies/01_erp_action_agent.md) | Both use exact-payload binding | Procurement binds a payload; this binds a bundle |
| theory | [NIST least privilege](https://csrc.nist.gov/glossary/term/least_privilege) | Scope minimisation | Not evidence for this design |

### 14b. What this case leaves open

| Open question | Why evidence is insufficient | Test that would resolve it | Priority | Owner |
|---|---|---|---|---|
| Does the bundle reduce minutes without double-reading? | No time-on-task data | Instrumented canary with first-read acceptance | High | Evaluation lead |
| Is the guardrail stable across peril strata? | No canary | Stratified canary analysis | High | Model risk |
| Can a live rollback meet one staffed hour? | Drill not executed | Timed drill under shadow load | High | Platform |
| Does the 62% eligibility rate hold? | Assumed | Claim-type census over four weeks | Medium | Ops analytics |
| Do citations improve acceptance or create false confidence? | No UI study | A/B on citation presence | Medium | Product |

### 14c. Intense interview pack

#### Q-01 / foundation / Framing
**Prompt.** In one minute, state the decision this case asks for and the constraint that dominates it.
**Strong answer.** The decision is whether to move bundle R2 from a five-percent canary to full claims-triage traffic. The dominating constraint is that the bundle informs a licensed adjuster and never decides; a promotion that raises missed coverage signals is disqualified regardless of labour savings.
**Reasoning.** The unit of release is the bundle, but the unit of accountability is the adjuster's record, so the promotion test must measure signal completeness rather than average quality.
**Follow-up.** If the bundle were restricted to document-gap detection only, would the constraint change? Only in strength: gaps are still a completeness claim, but a missed gap delays a request rather than mis-routing coverage.
**Acceptable alternatives.** Framing the decision as a staged rollout with stop conditions is acceptable if the guardrail remains the gate.
**Red flags.** "The model is more accurate, so ship it." Accuracy is not measured for the composed bundle and does not address the guardrail.
**Scoring.** 0 no decision; 1 decision without constraint; 2 decision plus constraint; 3 constraint plus why labour cannot override it.
**Support.** Sections 1, 3, 7.

#### Q-02 / foundation / Framing
**Prompt.** Why is a model-only comparison not sufficient evidence for this promotion?
**Strong answer.** Because the deployable artefact changes six components, and the composed behaviour can differ from a model swap with fixed prompt and index. A model-only comparison answers a question nobody deploys, and it cannot reveal schema incompatibilities with downstream consumers.
**Reasoning.** Behaviour is a function of the whole pipeline; retrieval changes alter which policy text is citable, and the prompt change alters output shape.
**Follow-up.** When would a model-only comparison be the right test? When the bundle changes only the model and every other artefact is provably identical, verified by hashes.
**Acceptable alternatives.** Using the offline suite as a screening gate is acceptable and is what the case does.
**Red flags.** Claiming the offline suite proves production behaviour.
**Scoring.** 0 model quality assertion; 1 mentions prompt; 2 explains composition; 3 adds the consumer-schema consequence.
**Support.** Sections 5c, 6.1, 6.7.

#### Q-03 / intermediate / Data
**Prompt.** You are told the canary is at 5% and reaches 180 claims per day. What must you check before trusting that number?
**Strong answer.** Check the eligible-claim denominator: 3,600 claims per day at 62% eligibility gives 2,232 eligible, and 5% of that is 111.6, not 180. A figure of 180 implies either a different share, a different eligibility rate, or counting ineligible claims.
**Reasoning.** 180/2,232 = 8.07% share, or 180/0.05 = 3,600 claims assumed fully eligible. Both are inconsistent with the stated parameters, so one input is wrong.
**Follow-up.** If the share were really 8%, what happens to the evidence window? It shortens to about 16 staffed days but doubles the exposed population.
**Acceptable alternatives.** Reporting both interpretations is acceptable if the inconsistency is flagged.
**Red flags.** Accepting 180 without reconciling the denominator.
**Scoring.** 0 accepts; 1 notices mismatch; 2 reconciles arithmetic; 3 states the decision consequence.
**Support.** Sections 6.2, 5a.

#### Q-04 / intermediate / Data
**Prompt.** Design the guardrail label. What makes it hard to measure?
**Strong answer.** The label is "a coverage signal present in the packet but absent from the proposal", adjudicated by two blinded reviewers with team-lead adjudication of disagreements. It is hard because it requires reading the packet independently of the proposal to avoid anchoring, and because reviewers must agree on what counts as a signal rather than a generic remark.
**Reasoning.** A label defined by absence is vulnerable to reviewer attention and to proposal anchoring; measuring agreement is therefore part of the label's validity.
**Follow-up.** How would you detect drifting label standards? Track disagreement rate and the distribution of adjudicated positives over time; a jump suggests a standards change rather than a behaviour change.
**Acceptable alternatives.** Using a structured signal checklist is acceptable if the checklist is versioned with the bundle.
**Red flags.** "Ask the adjuster if the proposal was good."
**Scoring.** 0 vague; 1 definition only; 2 definition plus blinding; 3 adds agreement and drift monitoring.
**Support.** Sections 5a, 5d, 9.

#### Q-05 / advanced / Data
**Prompt.** The canary runs during a storm week and the guardrail rises. What can you conclude?
**Strong answer.** Almost nothing about the bundle. The loss mix changed, so the comparison is confounded by stratum composition. The correct action is to report the guardrail by peril stratum and either extend the window or promote only within strata that were adequately represented.
**Reasoning.** A composition shift can raise the aggregate rate while every stratum stays flat, which is Simpson's paradox in operational form.
**Follow-up.** What if strata are too small for bounds? Then aggregate with stratification weights from the full-traffic mix and state the assumption.
**Acceptable alternatives.** Pausing the canary is acceptable but delays evidence without resolving confounding.
**Red flags.** Concluding the bundle is worse from an aggregate movement.
**Scoring.** 0 accepts aggregate; 1 mentions mix; 2 names confounding; 3 adds stratified reporting and weighted aggregation.
**Support.** Sections 4, 6.8, 9.

#### Q-06 / intermediate / Calculation
**Prompt.** Compute the weekly labour difference if the assisted path takes six minutes on 62% of 18,000 claims and the baseline takes fourteen minutes on all claims.
**Strong answer.** Baseline 18,000 × 14 / 60 = 4,200 hours. Assisted 11,160 × 6 / 60 = 1,116 hours plus 6,840 × 14 / 60 = 1,596 hours, total 2,712 hours. Difference 1,488 hours per week.
**Reasoning.** The saving applies only to eligible claims; applying six minutes to all claims would overstate the benefit by 6,840 claims' worth of work.
**Follow-up.** At 52 per hour, what is that worth? 1,488 × 52 = 77,376 per week before model and platform cost.
**Acceptable alternatives.** Expressing the answer in FTE-weeks at 36 hours is acceptable: 116.67 to 75.33.
**Red flags.** Applying the assisted time to the whole population.
**Scoring.** 0 wrong denominator; 1 correct total without split; 2 correct split; 3 correct split plus cost.
**Support.** Sections 6.4, 6.5.

#### Q-07 / advanced / Calculation
**Prompt.** Derive the break-even assisted time and interpret it.
**Strong answer.** Break-even sets the labour difference equal to fixed cost: (14 − m) × 11,160 / 60 × 52 = 6,358.236, so 14 − m = 0.6574 and m = 13.34 minutes. The margin is wide: the bundle would have to save less than a minute per eligible claim before it stops paying.
**Reasoning.** Fixed cost is small relative to labour, so the economics are insensitive across a broad range of assisted times.
**Follow-up.** Why is the wide margin not a reason to promote? Because the guardrail is the binding constraint, and a cost-positive bundle can still harm claimants.
**Acceptable alternatives.** Working in per-claim currency is acceptable: 12.13 saved per eligible claim against 0.57 fixed per eligible claim.
**Red flags.** Treating break-even as sufficient evidence.
**Scoring.** 0 no derivation; 1 algebra only; 2 algebra plus interpretation; 3 adds why economics cannot gate promotion.
**Support.** Sections 6.4, 6.5.

#### Q-08 / advanced / Calculation
**Prompt.** How many canary claims do you need to distinguish 0.8% from 1.6% at 5% one-sided with 80% power? What does that imply for the window?
**Strong answer.** Using n = (1.959964 + 0.841621)² (p1q1 + p2q2)/(p2 − p1)², the numerator is 7.848881 × 0.023680 = 0.1858647 and the denominator is 0.000064, giving 2,904.1, so 2,905 per arm. At 111.6 canary claims per staffed day that is 26.0 staffed days.
**Reasoning.** The squared difference dominates; halving the detectable difference quadruples the sample.
**Follow-up.** If the true baseline is 1.2%, the detectable difference shrinks and the sample grows; what do you do? Use a sequential design or accept a wider margin.
**Acceptable alternatives.** A Bayesian decision rule with an explicit prior is acceptable if declared before the window.
**Red flags.** Using the threshold rule and calling it a test.
**Scoring.** 0 no formula; 1 formula only; 2 correct number; 3 adds window and design consequence.
**Support.** Sections 6.3, 9.

#### Q-09 / intermediate / Design
**Prompt.** What has to be in the bundle manifest, and why does each field matter?
**Strong answer.** Model identifier and snapshot, prompt hash, tool-schema hash, index build identifier, policy-table version, and evaluation-suite identifier. Each matters because behaviour depends on it: the model changes tokenisation and quality; the prompt changes output shape; the schema gates downstream compatibility; the index determines citable text; the policy table defines correctness; the suite defines the gate.
**Reasoning.** Rollback is only meaningful if restoring the manifest restores behaviour, which requires every behaviour-determining artefact to be pinned.
**Follow-up.** What if the model endpoint cannot be pinned to a snapshot? Then the bundle cannot guarantee reproducibility, and the release contract must say so explicitly or move to a pinned deployment.
**Acceptable alternatives.** Pinning by digest instead of version string is acceptable and stronger.
**Red flags.** Listing the model only.
**Scoring.** 0 model only; 1 model plus prompt; 2 four or more components; 3 all six with the rollback rationale.
**Support.** Sections 5b, 6.1, 8c.

#### Q-10 / advanced / Design
**Prompt.** Design rollback and prove it works.
**Strong answer.** Rollback restores all six pins in the registry, invalidates in-flight assignments so new claims use R1, and lets already-reviewed claims finish under the bundle recorded on their review record. The proof is a drill: restore R1 in a shadow lane, replay 200 recorded packets, and require byte-identical proposals to the R1 record, timed under shadow load against a one-staffed-hour target.
**Reasoning.** Mixed bundling is the characteristic failure; a drill that checks configuration rather than outputs cannot detect it.
**Follow-up.** What if replay is not byte-identical because the endpoint is non-deterministic? Then pin temperature and seeds, or define equivalence at the field level with a stated tolerance, and record that the contract is equivalence-based.
**Acceptable alternatives.** Canary-first rollback with traffic shifting is acceptable if pins move atomically.
**Red flags.** "Redeploy the old image."
**Scoring.** 0 no procedure; 1 pins only; 2 pins plus in-flight rule; 3 adds drill and equivalence definition.
**Support.** Sections 8b, 8d, 12 F-03.

#### Q-11 / advanced / Design
**Prompt.** Where does authorization belong in the serving path, and what does the audit record need?
**Strong answer.** At the proposal boundary: the adjuster's identity must be entitled to the claim's line of business before any packet content is assembled, and the resulting proposal record stores the authenticated identity, the claim and packet version, the bundle identifier, and the policy version used for validation.
**Reasoning.** If authorization happens only at login, a stale session can read a claim reassigned to another team; if the audit omits the bundle, no later analysis can attribute behaviour.
**Follow-up.** The model endpoint is third-party. What changes? Minimise and redact direct identifiers before egress, record what was sent, and treat the endpoint as untrusted for authority.
**Acceptable alternatives.** Delegated token exchange is acceptable if the failure mode (identity provider outage) has a defined behaviour.
**Red flags.** Service-account-only audit trails.
**Scoring.** 0 login-only; 1 boundary check; 2 boundary plus audit fields; 3 adds egress minimisation.
**Support.** Sections 8b, 8d, 12 F-10.

#### Q-12 / intermediate / Evaluation
**Prompt.** Why use an upper confidence bound rather than a threshold on the point estimate?
**Strong answer.** The point estimate is noisy, and the two errors are asymmetric: promoting a breaching bundle harms claimants, while extending a canary costs time. An upper bound at 1.20% keeps the probability of promoting a truly breaching bundle bounded, whereas a point-estimate rule promotes roughly half the time when the truth sits at the threshold.
**Reasoning.** At p = 0.012 with n = 2,905, the estimate is below 0.012 about half the time by construction, so a threshold rule has approximately 50% error at the boundary.
**Follow-up.** What if the bound straddles the threshold after the planned sample? Predeclare whether that means extend or fail; the case treats it as fail-and-extend.
**Acceptable alternatives.** A sequential test with declared boundaries is acceptable and cheaper.
**Red flags.** "We hit 1.1%, so we pass" without an interval.
**Scoring.** 0 threshold only; 1 mentions noise; 2 explains asymmetry; 3 quantifies boundary error.
**Support.** Sections 9.

#### Q-13 / intermediate / Evaluation
**Prompt.** A reviewer says the bundle improved acceptance from 61% to 66%. Why is that not sufficient?
**Strong answer.** Acceptance measures whether adjusters kept the proposal, which can rise for reasons unrelated to accuracy: anchoring, time pressure, or a more conservative proposal style. It is a necessary operational signal but it is not the guardrail, and it can move opposite to signal completeness.
**Reasoning.** Acceptance is a behavioural measure with a strategic component; adjusters under load may accept plausible proposals without verification.
**Follow-up.** What would make acceptance trustworthy? Pairing it with blinded adjudication of a sample and with time-on-task, so acceptance cannot rise purely from reduced scrutiny.
**Acceptable alternatives.** Treating acceptance as a leading indicator with a floor is acceptable.
**Red flags.** Presenting acceptance as accuracy.
**Scoring.** 0 accepts; 1 questions; 2 names anchoring; 3 adds the paired measurement design.
**Support.** Sections 8d, 9, 12 F-05.

#### Q-14 / advanced / Debugging
**Prompt.** Adjusters report "the agent got slower" after promotion, but latency dashboards are flat. Diagnose.
**Strong answer.** The dashboards measure service latency, not staffed minutes. The likely mechanism is double-reading: adjusters read the packet, read the proposal, then re-read documents to check citations. Instrument time-on-task and first-read acceptance; if minutes rise while service latency is flat, the failure is trust or usability, not infrastructure.
**Reasoning.** The economic claim depends on replacing part of the reading, and nothing in the serving path measures that.
**Follow-up.** What would you change? Reduce citation volume, require packet-first order with a timed gate, and measure per-field acceptance to find which field triggers verification.
**Acceptable alternatives.** Suspecting retrieval latency is acceptable as a hypothesis, but it is falsified by flat service latency.
**Red flags.** Adding capacity.
**Scoring.** 0 blames infrastructure; 1 suspects UI; 2 identifies double-read; 3 adds instrumentation design.
**Support.** Sections 6.6, 12 F-04, 9.

#### Q-15 / advanced / Debugging
**Prompt.** After a rollback, some claims show citation text from the new index. What happened, and what is the immediate containment?
**Strong answer.** The index pin did not restore, so prompt p11 is running against index i4. Containment is to disable citation rendering and fall back to rule routing while pins are audited; then restore i4's predecessor and replay recorded packets to prove equivalence.
**Reasoning.** Partial pin restoration produces a bundle that never passed evaluation, which is the composed-failure mechanism the manifest exists to prevent.
**Follow-up.** How do you prevent recurrence? Make the drill compare output bytes, and make the registry reject a manifest with any unpinned component.
**Acceptable alternatives.** Freezing all promotions pending audit is acceptable but broader than needed.
**Red flags.** "Restart the service."
**Scoring.** 0 restart; 1 notices index; 2 identifies mixed bundle; 3 adds containment and drill fix.
**Support.** Sections 8c, 8d, 12 F-03.

#### Q-16 / advanced / Transfer
**Prompt.** Transfer the bundle pattern to a coding agent. What joins the bundle, and what breaks?
**Strong answer.** The bundle would include the model, the instruction files, the tool schema set, the repository index or embedding snapshot, and the policy that governs permitted commands. It breaks in two places: repository content changes continuously, so the index is a moving input rather than a release artefact, and the tool surface may be discovered dynamically, so consumers cannot be enumerated.
**Reasoning.** The pattern requires an enumerable consumer set and a pinnable input; both are weaker in a coding environment.
**Follow-up.** What is the minimum viable adaptation? Pin the instruction files and tool schema, treat repository state as an input with a recorded commit, and gate on sandboxed tests.
**Acceptable alternatives.** Treating the bundle as a deployable configuration rather than a data snapshot is acceptable if data versioning is separate.
**Red flags.** Claiming the pattern transfers unchanged.
**Scoring.** 0 no transfer; 1 lists components; 2 names both breaks; 3 adds the adaptation.
**Support.** Section 13.

## 15. Quick Reference Card

| Field | Value |
|---|---|
| Domain | Insurance claims operations; first-notice-of-loss triage |
| Problem type | Governed release decision for an assembled agent bundle |
| Method | Pinned six-component bundle, hash canary, dual-adjudicated guardrail, power-based gate |
| Primary result | Not measured; constructed economics show 1,488 staffed hours/week and 71,017.76 net |
| Key lesson | Promote the composed bundle only when the guardrail's upper bound and the rollback drill both pass |
| Tags | `atomic-release-bundle`; `rollout-regression`; `human-in-the-loop` |

| Supplemental field | Value |
|---|---|
| Case ID / version / as-of | CASE-2026-032 / 1.0 / 2026-11-10 |
| Evidence level | Illustrative; arithmetic derived and checked, no operational measurement |
| Implementation stage | Proposed |
| Baseline / primary metric | Rule table with full manual read; median staffed minutes per eligible claim |
| Key limitation | Canary transfer depends on unverified mix comparability and an unexecuted rollback drill |
| Review trigger / owner | Any change to packet schema, policy, model, prompt, index, or reviewer capacity; claims operations director |
