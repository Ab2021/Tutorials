---
case_id: "CASE-04-17-01"
case_title: "Visual quality inspection: route each manufactured part to accept, review, or reject"
case_version: "1.0"
created: "2026-11-10"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Manufacturing vision / inline visual quality control"
primary_pattern: "cost-sensitive triage with human review"
secondary_patterns: ["distribution-shift monitoring", "capacity-constrained routing"]
evidence_level: "illustrative"
source_basis: "illustrative"
as_of_date: "2026-11-10"
owner_role: "Plant quality manager"
source_index: "INDEX.md"
review_trigger: "Any camera, illumination, part revision, lot mix, reviewer workflow, defect-cost, or line-rate change."
---

# Visual Quality Inspection: a bounded pass/reject/review design

> **Evidence declaration:** This is an illustrative design exercise. All parts counts, defect rates, scores, costs, capacities, and model behavior below are constructed teaching inputs, not a production result, vendor benchmark, or claim about a named plant. **Implementation stage:** proposed. **Editorial status:** draft; independent technical and publication review are OPEN.

## 1. The One-Line Framing

Given one image set for each finished part at the final inspection station, decide **accept**, **hold for human review**, or **reject/rework** for the quality supervisor, so that suspect defects are contained without making the line wait, subject to a 480-parts/minute conveyor and six-reviewer staffed capacity, compared with manual inspection of every part.

This is a cost-sensitive visual triage decision, not a claim that a neural network can authorize disposition by itself. The primary success condition is that the proposed routing plan fits the review queue while its explicitly estimated quality and correction costs are preferable to the specified manual comparator; completed validation is not claimed.

| Contract element | Bounded choice |
| --- | --- |
| Input | One part ID plus six synchronized images captured at final inspection. |
| Output | Accept, review, or reject/rework recommendation with score, localization, and reason code. |
| Beneficiary | Quality manager who owns the disposition policy; reviewers retain final authority. |
| Decision cadence | One decision per part; nominal arrival 480 parts/minute. |
| Baseline | Humans visually screen every part before disposition. |
| Binding constraint | Inference must not block conveyor; review workload must not exceed staffed capacity. |
| Evidence | Illustrative arithmetic and proposed validation only. |

The tempting formulation—binary defect classification—is incomplete. A score can be accurate on a curated test set yet be operationally unusable when a camera drifts, when a rare part family is absent from training, when localization points to the wrong surface, or when every low-confidence alert overwhelms reviewers. The operational product is a governed routing policy with a physical recovery path.

## 2. Context & Background

The illustrative station inspects a finished molded-and-printed component after assembly and before packing. A part passes beneath a fixed enclosure containing two area cameras, two grazing-light cameras, one diffuse-light camera, and one code/identity camera. The inspection can observe surface and print appearance; it cannot measure internal dimensions, material chemistry, or downstream assembly function.

| Workflow step | Current manual baseline | Proposed intervention boundary |
| --- | --- | --- |
| 1. Build and identify | Part is assigned a part ID, product revision, lot, station, and event time. | Same identifier and event time enter an acquisition manifest. |
| 2. Capture | Inspector looks at the part under station light. | Cameras capture six images only after trigger confirmation. |
| 3. Assess | Inspector decides accept or direct rework. | Model produces class scores and regions; policy routes accept/review/reject. |
| 4. Disposition | Inspector records an outcome when workflow permits. | Reviewer verifies routed parts; manufacturing execution system records final disposition. |
| 5. Correct | Rejected parts go to rework or scrap decision. | Same physical correction workflow; no automatic actuator is proposed. |
| 6. Learn | Supervisors investigate recurring issues. | Adjudicated outcomes form a delayed, versioned candidate-label set. |

The quality manager owns the disposition policy and escalation rules. The manufacturing engineer owns camera mounts, trigger timing, and lighting maintenance. The vision engineer owns the model artifact and monitoring specification. Reviewers own the final decision for routed items. The line leader owns the stop/slow decision. No model score transfers these authorities.

The numerical scenario assumes two 225-minute staffed blocks in an eight-hour calendar shift, for 450 staffed inspection minutes after planned breaks and handover. At 480 parts/minute, the modeled shift population is 216,000 parts. Calendar minutes must not be substituted for staffed minutes in the reviewer calculation.

Manual visual screening persists because it handles novel appearances and physical handling naturally, provides a fallback when cameras fail, and lets a reviewer inspect ambiguous texture. Its stated limitation in this scenario is capacity: six reviewers cannot inspect every part at the stated line rate. This is an illustrative constraint, not an observation about a real workforce.

## 3. The Business Question

| Decision contract | Illustrative definition |
| --- | --- |
| Question | Should the plant quality manager pilot a review-gated vision routing policy for one product revision, and which score threshold can be evaluated without exceeding review capacity? |
| Decision owner | Plant quality manager; may approve a shadow-mode and controlled pilot, or retain manual screening. |
| Trigger and deadline | A 480-parts/minute line is planned; staffing is fixed at six reviewers for the initial scenario. |
| Available actions | Retain manual screening; add reviewers; use rules only; run vision in shadow mode; pilot review-gated vision; stop the pilot. |
| Value and downside | Value is lower review demand and earlier defect containment; downside is escaped defects, false rejects, delay, and unjustified automation. |
| Objective | Minimize illustrative per-shift decision cost while satisfying quality and review-capacity guardrails. |
| Hard constraints | No conveyor-blocking inference; no unattended final rejection; routing must be auditable by part ID. |
| Scope | One final station, one product revision, one shift model; no supplier, process-control, or safety certification decision. |
| Out of scope | Replacing dimensional inspection, redesigning the product, and production-wide efficacy claims. |

The business proxy is an error-and-workload ledger, not classification accuracy. A high accuracy number can conceal poor defect capture when defects are rare: an always-accept system would be 98.8% accurate at the illustrative 1.2% defect prevalence while detecting no defects. The guardrails are defect escape count, review load, false-reject burden, image acquisition integrity, and an escalation path for new appearance modes.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Type | Check / evidence plan | Status | Impact if wrong | Fallback | Owner / trigger |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A-01 | Defect prevalence is 1.2% over the modeled shift. | Data | Compare adjudicated samples by lot and shift. | Unverified | Cost and PPV change. | Recompute threshold policy. | Quality analyst / lot mix change |
| A-02 | Model sensitivity is 92% and FPR is 3.5% at review threshold. | Model | Frozen blind evaluation by grouped lot and time. | Unverified | Escapes or review load differ. | Shadow mode or manual screen. | Vision lead / validation failure |
| A-03 | Reviewer sensitivity is 98% and specificity is 90% for routed parts. | Human process | Blind repeat-read study with adjudication. | Unverified | Final escape/rework ledger changes. | Increase verification or retain baseline. | Quality manager / disagreement rise |
| A-04 | Six reviewers work 15 parts/minute for 450 staffed minutes. | Capacity | Time study during representative operation. | Unverified | Queue may grow. | Throttle routing; add staff; manual fallback. | Line leader / queue threshold |
| A-05 | Capture covers all required surfaces. | Physical | Golden-part and occlusion test per shift. | Unverified | Invisible defect families escape. | Hold affected product; manual screen. | Manufacturing engineer / capture alarm |
| C-01 | Inference cannot hold conveyor. | Latency | Measure p99 from trigger to route event. | Required | Missed routing or blocked line. | Default-to-review buffer or manual lane. | Controls engineer / p99 breach |
| C-02 | Final reject requires human authorization. | Governance | Workflow configuration review. | Required scenario rule | Unattended false dispositions. | Make model output review-only. | Quality manager / policy change |
| C-03 | Images and decisions must retain part, lot, revision, time, camera, and model version. | Traceability | Manifest reconciliation. | Required scenario rule | Cannot audit a decision. | Quarantine records and stop learning loop. | Data steward / missing fields |

The assumptions most likely to reverse the recommendation are the real false-positive rate under lighting and part-mix drift, the reviewer throughput measured with physical handling, and whether labels distinguish cosmetic variation from customer-relevant defects. Each is a testable assumption, not a fact implied by a vision model.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Proposed specification |
| --- | --- |
| Unit of analysis | One physical part inspection event, keyed by immutable `part_id`; six image records are children of one event. |
| Volume and window | Illustrative workload: 216,000 events in 450 staffed minutes; development data quantity is intentionally unspecified and must be recorded before training. |
| Inputs | Six images, camera exposure/gain, lighting setting, part revision, station ID, event time, and capture-complete flag. |
| Outcome | Adjudicated final quality disposition plus defect taxonomy, severity, and region where visible; label maturity is delayed until review/rework confirmation. |
| Population | Finished parts of one revision captured at the named station; exclude missing required-view events from automated accept. |
| Join semantics | `part_id` joins one inspection event to 1..6 images and later disposition; duplicate `part_id` events require a sequence number and reconciliation. |
| Splits | Group by production lot and time block; never scatter near-duplicate parts from one lot across train and test. |
| Time basis | Event time is camera trigger time in station clock; label time is final adjudication time and is unavailable at prediction. |

A “defect” label needs a written taxonomy. For this case, class `surface_defect` means an adjudicated visible surface anomaly under the approved inspection condition; `print_defect` means an unreadable, incomplete, or materially misplaced print; `cosmetic_variation` means visible variation accepted by the product specification; and `unknown` means no final adjudication. The model must not learn that every reworked part was truly defective, because rework can reflect routing caution or a downstream process decision.

Localization is a different target from classification. A part-level positive can be correct while a heat map highlights glare instead of a scratch. For reviewer usability, proposed evaluation records whether a localized region overlaps an adjudicated defect region under a documented annotation protocol. Region evidence is unavailable for defect types that cannot be visually bounded; those types are excluded from localization scoring rather than counted as misses without explanation.

### 5b. Tools & Stack

| Component | Proposed role | Owner | Version/evidence boundary |
| --- | --- | --- | --- |
| PLC trigger and encoder | Create capture event and associate physical sequence. | Controls engineer | Configuration to be versioned; no actual controller claim. |
| Camera and lighting controller | Acquire six views and report health telemetry. | Manufacturing engineer | Device settings must be recorded per event. |
| Edge acquisition service | Validate manifest, store images, and publish inspection request. | Platform engineer | Proposed; latency to be measured. |
| Vision inference service | Return score vector, localization candidate, and model version. | Vision engineer | Candidate architecture unspecified until evaluation. |
| Policy service | Apply threshold and route recommendation with reason code. | Quality systems engineer | Threshold is a versioned policy, not embedded in weights. |
| Review workbench | Show images, localization, and history; collect decision and reason. | Quality operations | Human final decision. |
| Event store and object storage | Retain manifest and image references under approved retention rules. | Data steward | Access and retention requirements not determined here. |
| Monitoring job | Compute capture, score, queue, and label-drift indicators. | Vision engineer | Proposed calculations, not production telemetry. |

### 5c. Method Summary

Input -> acquisition validation -> image normalization using training-derived parameters -> representation/model score and optional region proposal -> policy threshold -> accept or review queue -> human final accept/rework decision -> delayed adjudication and monitoring. Any missing required image, unhealthy light reading, unknown revision, or stale model/policy pairing routes to manual handling rather than automated accept.

| Candidate | Evidence considered | Strength | Limitation / avoid when | Decision status | Revisit if |
| --- | --- | --- | --- | --- | --- |
| Manual screen every part | Illustrative capacity and error ledger. | Handles novelty and is explainable. | Needs 32 reviewers at stated rate. | Baseline comparator. | Line rate or staffing changes. |
| Fixed image rules | Considered only; no test artifact. | Simple for stable, high-contrast features. | Brittle to pose, glare, and new defect forms. | Not tested. | A single geometric defect dominates. |
| Part-level classifier | Illustrative confusion matrix. | Direct triage score and lower serving complexity. | May not explain location; label ambiguity remains. | Candidate for shadow evaluation. | Localization need is weak. |
| Detector / segmenter | Considered; no test artifact. | Can localize multiple visible defects. | Region labels cost more and may miss diffuse defects. | Not selected before data audit. | Reviewer needs validated regions. |
| Anomaly detector | Considered; no test artifact. | May surface novel appearance. | Normal variation can create high alert load. | Supplementary candidate only. | Known-defect labels are scarce. |

The illustrative chosen policy uses a calibrated defect-risk score `s` on [0,1] only after calibration is demonstrated on held-out, group-separated data. It routes `s < 0.55` to accept and `s >= 0.55` to review. “Reject” is a reviewer outcome in the initial design; automatic reject is explicitly not selected. Score calibration is assessed separately from ranking: a useful ranking can still be numerically overconfident.

### 5d. Data provenance, quality, exclusions and leakage controls

| Risk | Why it invalidates an apparent result | Control |
| --- | --- | --- |
| Part leakage | Several images of one part, or serially similar parts, in different splits make recall look better. | Split by `part_id`; deduplicate exact and near-duplicate images before split. |
| Lot leakage | One lot shares material, tooling, illumination, and defect mechanism. | Assign every lot wholly to one split; report leave-one-lot-out results. |
| Time leakage | Later inspection disposition or downstream rework code may encode the answer. | Use only fields available at trigger time; retain feature availability timestamps. |
| Revision leakage | Product revision can act as a proxy for known quality outcome. | Separate revisions or document a supported compatibility mapping. |
| Reviewer leakage | A reviewer overlay or prior disposition in an image/workbench can reveal label. | Mask dispositions and annotation layers during model input creation. |
| Preprocessing leakage | Fitting normalization across train and test lets test distribution influence training. | Fit transforms on training fold only inside each grouped split. |
| Label ambiguity | Cosmetic variation, rework routing, and true customer-relevant defect can differ. | Use specification-linked adjudication with `uncertain` and disagreement fields. |
| Missing view bias | Failed captures may cluster by equipment state or part appearance. | Track missingness by camera, shift, lot, and final disposition; do not silently drop. |
| Correction contamination | Post-rework images do not represent decision-time appearance. | Store original capture immutable and separate correction images. |

The data freeze requires a manifest hash, image object checksums, label export timestamp, taxonomy version, split assignment version, and exclusion report. A record with an unavailable label remains available for unsupervised drift monitoring but is not silently treated as a negative. Inaccessible source bodies for the original course planning entry remain OPEN; this manuscript uses no claimed external system data.

### 5e. Reproducibility and method-selection ledger

| Artifact/source ID | Location | Version/date | Purpose | Accessibility | Reproduction limitation |
| --- | --- | --- | --- | --- | --- |
| SRC-01 | Local Case_Studies/INDEX.md | Read 2026-11-10 | Original planned topic: visual quality inspection. | Local/public course file. | Planning row supplies no dataset or outcome. |
| STD-01 | Course authoring standards | Effective 2026-11-06 | Structure and numerical rules. | Local/public course file. | Not evidence of manufacturing performance. |
| CFG-01 | Proposed `inspection-policy` record | Uncreated; must be versioned. | Threshold, allowed revisions, queue thresholds. | Not yet available. | No executable configuration exists. |
| RUN-01 | Illustrative arithmetic ledger in Section 6 | 2026-11-10 | Check counts and capacity. | Reproducible by hand. | Uses constructed inputs, not a model run. |
| DATA-01 | Proposed immutable image manifest | Uncreated. | Prediction-time joins and split audit. | Not yet available. | No source images or labels were accessed. |

A reproducible evaluation would pin code revision, dependency lockfile, camera configuration, illumination setpoint, taxonomy version, split manifest, random seeds, trained model digest, calibration artifact, policy version, and reviewer-workbench configuration. Seeds do not make a camera stream or GPU execution reproducible by themselves.

## 6. Analysis Walkthrough

The decisive illustrative tradeoff is not “92% recall is good.” At the stated prevalence, the 0.55 review threshold sends 9,853.92 parts to review in one shift, far below the 40,500-part staffed review capacity, while the error ledger predicts 255.0528 escaped defects after reviewer errors. That escape count is an estimate conditional on assumed independent rates; it is not a safety or quality guarantee.

### Worked numerical ledger: inputs and validity

**Validity before use:** The following are expected-count calculations for a fixed 216,000-part population. They assume a stable 1.2% prevalence, model true-positive rate (TPR) 0.92 and false-positive rate (FPR) 0.035 at the review threshold, reviewer sensitivity 0.98 and reviewer specificity 0.90 for every routed item, and no queue abandonment. Rates may vary by lot, defect type, light state, and reviewer; the calculation is not a stochastic confidence interval.

```text
FORMULA [Q-01]: N_defect = N * p; N_good = N * (1 - p)
WHERE: N is parts/shift; p is defect prevalence [defective parts/part].
CONSTRAINT: fixed modeled population and prevalence; p is illustrative.
NUMERICAL EXAMPLE: N=216000 parts; p=0.012.
N_defect=216000*0.012=2592 defective parts.
N_good=216000*(1-0.012)=213408 good parts.
INTERPRETATION: prevalence is a population denominator, not a model accuracy.

FORMULA [Q-02]: TP=N_defect*TPR; FN=N_defect*(1-TPR); FP=N_good*FPR; TN=N_good*(1-FPR)
WHERE: TPR=Pr(route review | defect); FPR=Pr(route review | good).
CONSTRAINT: rates apply to the modeled population at one fixed threshold.
NUMERICAL EXAMPLE: TPR=0.92; FPR=0.035.
TP=2592*0.92=2384.64; FN=2592*0.08=207.36.
FP=213408*0.035=7469.28; TN=213408*0.965=205938.72.
CHECK: TP+FN=2592; FP+TN=213408; all four cells sum to 216000.
INTERPRETATION: expected counts are fractional because they are rates times a population.

FORMULA [Q-03]: PPV=TP/(TP+FP); NPV=TN/(TN+FN)
WHERE: PPV is defective fraction among routed parts; NPV is good fraction among accepted parts.
CONSTRAINT: prevalence and both conditional rates must match the target population.
NUMERICAL EXAMPLE: PPV=2384.64/(2384.64+7469.28)=0.2420=24.20%.
NPV=205938.72/(205938.72+207.36)=0.998994=99.8994%.
INTERPRETATION: despite 92% TPR, about 75.80% of review alerts are good parts at 1.2% prevalence.
```

| Actual condition / routed state | Review (`s >= 0.55`) | Accept (`s < 0.55`) | Row total | Metric meaning |
| --- | --- | --- | --- | --- |
| Defective | TP = 2,384.64 | FN = 207.36 | 2,592.00 | Recall/TPR = 92.0% |
| Good | FP = 7,469.28 | TN = 205,938.72 | 213,408.00 | Specificity = 96.5%; FPR = 3.5% |
| Column total | 9,853.92 | 206,146.08 | 216,000.00 | Review rate = 4.562% |

A confusion matrix must name its positive action. Here “positive” means **route to human review**, not “final reject.” If a report instead calls every alert a reject, it conflates model triage, human disposition, and physical correction. Classification, detection, and localization have different denominators: part-level classification counts parts; detection counts annotated instances and false detections; localization counts region matches under an explicit overlap rule.

### Reviewer and line capacity

```text
FORMULA [Q-04]: C_review = R * r * m; R_required_manual = ceil(N / (r*m))
WHERE: C_review is review capacity [parts/shift]; R reviewers; r parts/(reviewer*minute);
m staffed minutes/shift. `ceil` rounds required whole people upward.
CONSTRAINT: constant rate, all six reviewers assigned, no rework handling in this service rate.
NUMERICAL EXAMPLE: R=6; r=15 parts/(reviewer*min); m=450 min.
C_review=6*15*450=40500 parts/shift.
R_required_manual=ceil(216000/(15*450))=ceil(32)=32 reviewers.
INTERPRETATION: six reviewers can review 40500, not manually screen 216000, parts in this scenario.

FORMULA [Q-05]: q = TP+FP; utilization = q/C_review
WHERE: q routed review load [parts/shift].
CONSTRAINT: same shift denominator for q and capacity.
NUMERICAL EXAMPLE: q=2384.64+7469.28=9853.92; utilization=9853.92/40500=0.2433067=24.33%.
HEADROOM: 40500-9853.92=30646.08 review slots/shift.
INTERPRETATION: mean capacity appears sufficient; burstiness and reviewer interruptions still require a queue test.
```

The capacity boundary is useful because it ties the classifier to staffing. With TPR fixed at 0.92, the largest FPR compatible with 40,500 review slots is `(40,500 - 2,592*0.92) / 213,408 = 0.178603...`, or **17.8603%**. At a higher FPR, expected routed volume exceeds modeled capacity even before allowing for bursts. This is a mean-shift boundary, not a service-level guarantee.

### Final disposition and cost ledger

For a review-routed defective part, assumed reviewer sensitivity 98% gives `2,384.64*0.98=2,336.9472` correctly rejected/reworked and `47.6928` reviewer misses. For a routed good part, assumed reviewer specificity 90% gives `7,469.28*0.90=6,722.352` correctly cleared and `746.928` false rejects. Automated accept already contains `207.36` missed defects; estimated final escapes are `207.36+47.6928=255.0528` parts.

The escape ledger deliberately keeps model misses (`207.36`) and reviewer misses (`47.6928`) distinct: the former never reached review, whereas the latter reached review but received an incorrect final disposition. Their sum is the final escape quantity used in the cost table.

| Ledger item | Formula | Illustrative unit cost | Shift quantity | Illustrative cost |
| --- | --- | --- | --- | --- |
| Image/inference processing | N * $0.015/part | $0.015 / part | 216,000 | $3,240.00 |
| Human review | q * $0.45/review | $0.45 / routed part | 9,853.92 | $4,434.264 |
| False reject correction | FP * (1-reviewer specificity) * $6 | $6 / good part wrongly rejected | 746.928 | $4,481.568 |
| Escaped defect consequence | final escapes * $90 | $90 / escaped defective part | 255.0528 | $22,954.752 |
| Total proposed ledger | sum above | mixed | — | $35,110.584 |

The manual comparator assumes all 216,000 parts receive one $0.45 screen, sensitivity 98%, specificity 90%, and the same $6 false-reject correction and $90 escape consequence. Its expected screen labor is `$97,200`; good false rejects are `213,408*0.10=21,340.8` and cost `$128,044.80`; escaped defects are `2,592*0.02=51.84` and cost `$4,665.60`; total is **$229,910.40**. This comparator is deliberately simplistic and cannot establish savings; it makes labor/error assumptions inspectable.

The proposed-minus-manual ledger difference is `$35,110.584 - $229,910.40 = -$194,799.816` per modeled shift. The negative sign means lower estimated cost under these constructed assumptions. It does not include capital, integration, maintenance, downtime, disposition delay, customer harm beyond the $90 proxy, or any production outcome; those omissions are why the action is a validation plan rather than a rollout.

### Sensitivity / robustness checks

| Stress/test | Original condition | Alternative | Exact or stated consequence | Limitation | Decision consequence |
| --- | --- | --- | --- | --- | --- |
| Prevalence lower | p=1.2% | p=0.3%, same TPR/FPR | TP=596.16; FP=7,537.32; PPV=7.330%; review=8,133.48. | FPR assumed stable despite changed mix. | Reviewers see mostly good parts; inspect alert fatigue and calibration. |
| Prevalence higher | p=1.2% | p=3.0%, same TPR/FPR | TP=5,961.60; FP=7,333.20; review=13,294.80; PPV=44.841%. | Defect types may differ at high prevalence. | Capacity still fits mean shift; validate lot containment. |
| Stricter threshold | TPR=.84, FPR=.015 | Same N,p | TP=2,177.28; FP=3,201.12; review=5,378.40; model misses=414.72. | Rates are hypothetical threshold point. | Less workload but more escapes; not automatically preferable. |
| Looser threshold | TPR=.96, FPR=.08 | Same N,p | TP=2,488.32; FP=17,072.64; review=19,560.96. | Reviewer accuracy may fall with volume. | Fits mean capacity, but queue/burst test needed. |
| Capacity boundary | TPR=.92 | FPR=17.8603% | Expected review approximately 40,500. | Rounding and bursts can exceed. | Do not set policy at this boundary. |
| Overload example | TPR=.92 | FPR=50% | Review=2,384.64+106,704=109,088.64 > 40,500. | No queue abandonment modeled. | Disable automated routing or increase capacity. |
| Reviewer degradation | Sensitivity=.98 | Sensitivity=.90 | Final escapes=207.36+238.464=445.824. | Assumes same model routing. | Reviewer training/ergonomics are quality controls. |
| Light drift | FPR=.035 | FPR=.12, TPR=.92 | Review=2,384.64+25,608.96=27,993.60. | No causal model for drift. | Still fits mean capacity; trigger image-quality investigation. |

The threshold is selected by expected cost and guardrails, not by maximizing F1 alone. A calibration check bins scores and compares predicted defect risk with adjudicated frequency by part revision, lot, camera, and light state. If the score is miscalibrated, an operating point based on “0.55 means 55% risk” is invalid even if ranking metrics look acceptable.

## 7. The Finding

For this illustrative one-station scenario, a review-gated vision policy is a reasonable **candidate for shadow-mode validation**, not a deployment conclusion. At the stated assumptions, its 9,853.92 routed parts are below six reviewers’ 40,500-part staffed capacity and its constructed ledger is lower than the constructed manual comparator; however, it estimates more final escapes than the assumed all-manual screen and depends on unverified model, label, and reviewer rates. The quality manager should therefore authorize only an auditable, reversible evaluation with manual disposition authority, grouped validation, and stop criteria. The key tradeoff is not a single accuracy score: lower review load can increase escaped defects, while a loose threshold can turn lighting variation into an unserviceable queue.

## 8. System/Solution Design

The proposed architecture has an **online decision path** and an **offline learning path**. They share immutable identifiers but do not let late labels change a past decision. All components below are proposed interfaces, not a description of a deployed plant.

```text
ONLINE SUCCESS PATH
part + trigger -> acquisition manifest -> capture validation -> image store reference
  -> inference(score, regions, model_digest) -> policy(score, health, revision, policy_digest)
  -> accept event OR review-queue event -> reviewer final disposition -> MES correction work order

ONLINE FAILURE / RECOVERY PATH
missing image OR camera health bad OR unknown revision OR inference timeout OR policy mismatch
  -> manual-handling event -> physical marked lane / review queue -> operator acknowledgement
  -> incident record -> recover capture/inference -> reconcile every part_id before release

OFFLINE PATH
immutable images + manifests + later adjudication -> label audit -> grouped split manifest
  -> train/calibrate candidate -> blind evaluation -> approval packet -> versioned policy release
```

| Interface | Input contract | Output contract | Invariant / failure handling |
| --- | --- | --- | --- |
| Acquisition | `part_id`, sequence, trigger time, camera configuration. | Exactly one manifest with six required-view statuses. | No capture-complete claim unless each required view has checksum or explicit failure state. |
| Inference | Validated image references and approved revision. | Score vector, localization candidates, latency, model digest. | Never silently substitute a different model digest. |
| Policy | Inference output, health, revision, threshold policy. | `accept`, `review`, or `manual_handling` plus reason. | Policy cannot emit final physical rejection in this initial design. |
| Review workbench | Part images, score, regions, policy reason. | Human disposition, defect code, confidence, reviewer ID, time. | Reviewer can disagree; disagreement is data, not an error to overwrite. |
| MES handoff | Final authorized disposition and part ID. | Rework/scrap/accept workflow state. | Retry must be idempotent by disposition event ID. |
| Monitor | Events, capture health, queue, delayed labels. | Alerts and investigation tickets. | Monitor failure itself routes to manual oversight. |

Latency budget is an acquisition-to-route budget, not merely GPU inference time. A proposed budget is trigger-to-last-image 40 ms, manifest validation 5 ms, image transfer 20 ms, inference p99 35 ms, policy 5 ms, and event write 10 ms, totaling 115 ms. This is a design allocation whose sum is `40+5+20+35+5+10=115 ms`; it has not been measured. The conveyor interface must specify the physical distance and speed that make 115 ms safe, or the routing point must be redesigned.

Queue state is explicit: `READY` after valid manifest; `INFERRED` after digest-bearing score; `ROUTED_ACCEPT`, `ROUTED_REVIEW`, or `ROUTED_MANUAL`; `HUMAN_DECIDED`; then `MES_ACKED`. A retry can repeat a request but may not create two final dispositions. The reconciliation job finds any part that reached `READY` without a terminal state and holds it for manual resolution.

## 9. Evaluation & Validation

No production accuracy is asserted. The following is a proposed validation protocol that must be completed before any claim about performance, savings, or suitability. The test set is held out from threshold selection, calibration fitting, augmentation tuning, and taxonomy revisions.

| Evaluation question | Population / denominator | Metric and calculation | Pass/decision use | Status |
| --- | --- | --- | --- | --- |
| Capture integrity | All triggered parts by camera and shift. | Required-view completion; duplicated/missing manifest rate. | Missing required views must route manual. | Not measured. |
| Part triage | Adjudicated parts in held-out lots and later time blocks. | TPR, FPR, PPV, NPV with counts and uncertainty. | Choose candidate policy point. | Not measured. |
| Defect taxonomy | Each defect type with sufficient adjudicated support. | Recall and false-alert rate by type; report unsupported types as unknown. | Prevent aggregate masking. | Not measured. |
| Localization | Region-annotated visible defects only. | Predefined region-overlap success plus reviewer usefulness rating. | Use overlay only if it helps and is reliable. | Not measured. |
| Calibration | Held-out scored and adjudicated parts. | Reliability bins, calibration error, and score histogram by slice. | Do not interpret score as risk without support. | Not measured. |
| Queue behavior | Representative line-rate trace including bursts. | Arrival/service traces, p95 wait, maximum queue, abandonment. | Set staffing and alert thresholds. | Not measured. |
| Human performance | Blinded routed sample with independent adjudication. | Sensitivity, specificity, disagreement, handling time. | Validate final-disposition ledger. | Not measured. |
| Drift resilience | Light/camera/lot/revision slices and perturbation tests. | Score, capture, and error changes relative to reference. | Define hold or recalibration trigger. | Not measured. |

Use a grouped, temporal split: train on earlier lots, tune on held-out training-era lots, and reserve later unseen lots for final evaluation. If a defect is so rare that its interval is too wide to support a decision, say so and retain manual control for that defect family. Do not pool images from the same part or lot to manufacture a stable-looking result.

The validation confusion matrix reports raw integer counts as well as rates, with the exact positive definition. A secondary analysis compares human-only, model-review-plus-human, and rules-only alternatives on the same adjudicated population. Any comparison must hold the label source and inspection conditions constant; otherwise it measures different tasks rather than a model advantage.

Acceptance criteria for a limited pilot are proposed, not passed: 100% traceable disposition events for sampled parts; zero automated accepts on missing required views; review queue below an agreed operational alert level in representative bursts; no uninvestigated material degradation by lot/camera/defect slice; and quality-manager approval of the residual escape risk. A model threshold cannot approve itself.

## 10. Action Taken

**Action actually taken:** this manuscript specifies an illustrative evaluation plan only. No camera was installed, no model was trained, no threshold was deployed, no parts were routed, and no quality outcome was observed.

| Phase | Proposed controlled action | Decision gate | Rollback / recovery |
| --- | --- | --- | --- |
| 0. Instrument | Audit part IDs, clock sync, camera health, and current manual labels without changing disposition. | Traceability and capture audit complete. | Continue manual baseline. |
| 1. Shadow | Run candidate inference after capture; hide score from disposition reviewer. | Grouped evaluation and calibration reviewed. | Disable inference; preserve manifests. |
| 2. Assisted review | Show score/regions only for a sampled routed workflow; human screens all selected parts. | Reviewer usability and queue safety reviewed. | Hide overlay and return to manual screen. |
| 3. Limited triage | Permit automatic accept only under approved policy and fallback rules; human decides reviews. | Quality manager accepts documented pilot guardrails. | Route all parts to manual handling. |
| 4. Review | Compare pilot evidence with predeclared metrics and stop criteria. | Independent quality/engineering review. | Pause, recalibrate, retrain, or retire candidate. |

The initial operational threshold should be treated as a configuration record with effective time, model digest, calibration digest, score threshold, allowed part revisions, and queue-alert settings. A threshold change requires a documented validation rationale and makes a new analysis segment; silently changing it destroys comparability.

## 11. Outcome & Results

**Observed outcome:** not measured. There is no deployment, customer outcome, production accuracy, throughput improvement, cost saving, or vendor comparison to report.

**Illustrative calculated result:** Under the stated fixed-rate inputs, 9,853.92 review routes use 24.33% of the 40,500-part mean staffed review capacity, while the constructed proposed cost ledger is $35,110.584 versus $229,910.40 for the constructed manual comparator. These are conditional estimates, not outcomes, and omit capital, integration, maintenance, downtime, and actual customer impact.

| Claim ID | Exact claim and scope | Evidence type | Source / calculation | Supports | Does not establish | Check trigger |
| --- | --- | --- | --- | --- | --- | --- |
| CLM-01 | At N=216,000 and p=1.2%, TPR=.92/FPR=.035 yields 9,853.92 review routes. | illustrative | Q-01 and Q-02 in Section 6. | Arithmetic workload estimate. | Observed alert volume or model quality. | Any population/rate change. |
| CLM-02 | Six reviewers at 15 parts/min for 450 staffed min provide 40,500 review slots. | illustrative | Q-04 in Section 6. | Mean capacity calculation. | Burst wait time or actual throughput. | Time study or staffing change. |
| CLM-03 | The constructed proposed ledger is lower than constructed manual ledger by $194,799.816. | illustrative | Section 6 cost ledger. | Which assumptions dominate a decision. | Savings, ROI, or business case. | Any cost/error assumption change. |
| CLM-04 | A candidate should enter shadow-mode validation, not autonomous operation. | not-measured | Sections 4, 9, and 10. | A cautious next action. | Fitness for production. | Validation result or governance change. |

The most important result is therefore a **decision boundary**: the candidate cannot be judged from accuracy alone, and no numerical result survives without measurement of capture completeness, grouped generalization, calibration, human review, and queue bursts.

## 12. Failure Modes & What Went Wrong

| Failure mode | Symptom | Mechanism / competing causes | Diagnostic check | Recovery | Verification |
| --- | --- | --- | --- | --- | --- |
| Part-level leakage | Unexpectedly high held-out metrics. | Same part or near-identical lot appears across splits. | Hash/embedding duplicate audit and lot membership audit. | Rebuild grouped split; invalidate affected result. | No group overlaps; rerun blinded evaluation. |
| Time leakage | Metrics collapse in later lots. | Later disposition or rework field entered training features. | Feature availability timestamp audit. | Remove late fields; retrain. | Every feature predates trigger time. |
| Label ambiguity | Reviewers disagree or model “errors” are subjective. | Taxonomy insufficient; cosmetic variation treated as defect. | Adjudication disagreement matrix and spec review. | Add uncertain class or clarify acceptance criterion. | Agreement/uncertainty reported by category. |
| Camera defocus | Localization becomes diffuse; score distribution shifts. | Vibration, focus drift, lens contamination. | Sharpness/control-chart by camera and golden-part test. | Hold automated accepts for affected view; clean/refocus. | Golden-part capture returns within approved reference. |
| Lighting drift | False alerts rise on one shift. | LED aging, power variation, reflector position, ambient leak. | Brightness/color/score distributions by light state. | Restore setpoint; investigate enclosure. | Reference panel and alert rate stabilize. |
| Trigger mismatch | Images belong to adjacent part. | Encoder/PLC delay or dropped trigger. | Sequence reconciliation against part ID and physical marks. | Route affected interval to manual; repair timing. | No unmatched sequence events. |
| Missing view silently accepted | Defects on unseen side escape. | Manifest bug or degraded camera health. | Required-view completeness test and alarm drill. | Change default to manual handling. | Injected missing view never reaches accept. |
| Revision mix shift | New product revision generates alarms. | Geometry/print changes outside training coverage. | Score and error slices by revision. | Block revision or collect labeled evaluation data. | Explicit approval mapping exists. |
| Lot/process shift | One supplier lot has novel texture. | Material/tooling/process change changes appearance. | Lot-level alert, capture, and defect monitoring. | Contain lot; increase review; diagnose process. | Adjudicated lot analysis closes incident. |
| Defect taxonomy drift | Rework code no longer maps to visual defect. | Process changes or staff use codes differently. | Audit label lineage and reviewer notes. | Version taxonomy; relabel sample. | Old/new labels are not mixed without mapping. |
| Calibration decay | A score of .55 no longer corresponds to intended risk. | Population or model serving changes. | Reliability bins by recent adjudicated slice. | Recalibrate or revert to review-all/manual. | Held-out calibration check passes. |
| Review backlog | Review wait grows despite mean capacity. | Burst arrivals, breaks, rework interruptions, slower handling. | Queue age, arrival/service trace, staff state. | Throttle auto-routing; reassign staff; manual lane. | Queue drains below agreed alert threshold. |
| Reviewer automation bias | Humans accept highlighted errors or over-trust score. | Overlay anchoring and time pressure. | Blind A/B workbench study; disagreement analysis. | Hide score/overlay; retrain reviewers. | Independent adjudication does not worsen. |
| False reject correction overload | Rework area fills with good parts. | FPR high or reviewer specificity low. | False-reject count by cause and reviewer. | Tighten threshold or add second verification. | Correction queue returns to operating limit. |
| Inference timeout | Route missing at physical decision point. | Network, hardware, service overload, bad image payload. | p50/p95/p99 and timeout reason codes. | Use manual lane; fail closed from auto-accept. | Timeout injection creates manual event. |
| Model/policy mismatch | Score interpreted with wrong threshold/calibration. | Partial rollout or stale cache. | Digest pair audit per event. | Quarantine mismatched events; roll back atomically. | Every event has approved digest pair. |
| Storage retention gap | Cannot investigate a disputed disposition. | Object lifecycle or permission configuration changed. | Sample manifest-to-object retrieval audit. | Freeze deletion; restore approved retention. | Audit sample retrieves immutable evidence. |
| Localization illusion | Box highlights glare but part classification is correct. | Weak labels or saliency artifact. | Region overlap and reviewer usefulness evaluation. | Remove overlay; use part-level review. | Overlay only shown when validated. |
| Novel defect | Low score accept contains a new defect family. | Closed-set classifier lacks representation. | Escaped-defect taxonomy review and novelty audit. | Contain affected lot; add manual checks; collect labels. | New family is separately evaluated before use. |

Recovery priority is physical containment first, data preservation second, model diagnosis third. A failed camera, missing trace, or queue overflow does not call for “best effort” acceptance: it requires a documented manual route or a line decision by the accountable operations role.

## 13. Transferable Lesson

Visual inspection is a coupled sensing, classification, human-factors, and queueing problem. A model score is only one intermediate signal; the reliable unit of design is the part-level decision record that binds images, physical identity, timing, policy version, reviewer decision, and later adjudication.

The same reasoning transfers to document review, fraud investigation, and medical-image prioritization only after changing the harm model, decision authority, label maturity, and workload geometry. A manufacturing review queue can be capacity-modeled in parts/minute; a clinical task cannot inherit this case’s thresholds, cost proxy, or autonomy boundary.

A useful counterexample is a stable, high-contrast single defect that a calibrated physical gauge catches deterministically. In that situation a learned visual classifier can add cost and failure modes rather than value. The strongest alternative remains the simpler instrument or manual check when it meets the actual decision contract.

## 14. Connections & Cross-References

Related module topics include vision evaluation and error taxonomy, CNNs/attention/transfer learning, detection and segmentation, and self-supervised representation learning. Those materials can explain model mechanics; this case focuses on the operational decision contract. The sibling planned case, medical-image triage evaluation, remains a distinct future case and is not covered by this manufacturing manuscript.

### Open questions

- Which defect families are visible at the final station, and which require a different sensing modality?
- What independent adjudication process defines a true defect, cosmetic variation, and uncertain case?
- What are actual arrival bursts, review interruptions, and physical lane capacities?
- What evidence supports reviewer sensitivity and specificity under real ergonomic conditions?
- How long may images and disposition records be retained, and who can access them?
- Which lighting/camera telemetry measurements predict material score drift?
- What capital, maintenance, downtime, and correction costs belong in a complete business case?
- What pilot stop criteria are acceptable to the quality manager and line leader?

### Interview pack

**Q01. Why is accuracy inadequate for this case?**
- **Checked answer:** At p=1.2%, always accepting is 98.8% accurate yet captures no defects. Use confusion-matrix counts, defect recall, false-alert rate, queue load, and disposition cost instead.
- **Reasoning:** At p=1.2%, always accepting is 98.8% accurate yet captures no defects. Use confusion-matrix counts, defect recall, false-alert rate, queue load, and disposition cost instead.
- **Follow-up:** Ask for the positive-class definition and prevalence.
- **Acceptable alternatives:** AUC may be reported for ranking, but it cannot select a staffed operating point alone.
- **Red flags:** Quoting one accuracy number or ignoring denominator.
- **Scoring:** Full credit names prevalence and operational consequences.
- **Answer support:** Section 3 and Q-01/Q-02.

**Q02. Recompute the modeled review volume at the stated threshold.**
- **Checked answer:** Defects: 216,000*.012=2,592; TP=2,592*.92=2,384.64. Good: 213,408; FP=213,408*.035=7,469.28. Review total=9,853.92.
- **Reasoning:** Defects: 216,000*.012=2,592; TP=2,592*.92=2,384.64. Good: 213,408; FP=213,408*.035=7,469.28. Review total=9,853.92.
- **Follow-up:** What does a fractional count mean here?
- **Acceptable alternatives:** It is an expected count; integer simulation would vary around it.
- **Red flags:** Adding TPR and FPR or applying FPR to all parts.
- **Scoring:** Full credit shows both population denominators.
- **Answer support:** Section 6 Q-01/Q-02.

**Q03. Why must parts and lots be grouped in the split?**
- **Checked answer:** Views of one part and serially similar lot parts share appearance and process conditions. Random image splits leak these correlations, creating optimistic held-out results.
- **Reasoning:** Views of one part and serially similar lot parts share appearance and process conditions. Random image splits leak these correlations, creating optimistic held-out results.
- **Follow-up:** How would you test for leakage?
- **Acceptable alternatives:** Group by part and lot, audit overlap, and deduplicate near images.
- **Red flags:** Only randomizing image filenames.
- **Scoring:** Full credit identifies the evaluation invalidation mechanism.
- **Answer support:** Section 5d.

**Q04. A model identifies the correct defective part but highlights glare. Is it correct?**
- **Checked answer:** It is correct for part-level classification but fails the separate localization requirement. Report localization only on region-annotated visible defects with a predetermined criterion.
- **Reasoning:** It is correct for part-level classification but fails the separate localization requirement. Report localization only on region-annotated visible defects with a predetermined criterion.
- **Follow-up:** When would you hide the overlay?
- **Acceptable alternatives:** Hide it if it misleads reviewers or lacks validated utility.
- **Red flags:** Calling saliency a verified defect location.
- **Scoring:** Full credit separates task definitions.
- **Answer support:** Sections 5a and 9.

**Q05. What is the reviewer capacity, and why is staffed time important?**
- **Checked answer:** Six reviewers * 15 parts/(reviewer*min) * 450 staffed min = 40,500 reviews/shift. Breaks and handover mean calendar eight hours are not the service denominator.
- **Reasoning:** Six reviewers * 15 parts/(reviewer*min) * 450 staffed min = 40,500 reviews/shift. Breaks and handover mean calendar eight hours are not the service denominator.
- **Follow-up:** How many reviewers for manual inspection?
- **Acceptable alternatives:** ceil(216,000/(15*450))=32.
- **Red flags:** Using 480 calendar minutes without saying so.
- **Scoring:** Full credit keeps units visible.
- **Answer support:** Section 6 Q-04.

**Q06. What FPR boundary follows from the stated staffing constraint?**
- **Checked answer:** With TPR=.92, FPR_max=(40,500-2,592*.92)/213,408=17.8603%. Above it, mean review demand exceeds mean capacity.
- **Reasoning:** The boundary leaves exactly 40,500 expected review slots under the stated fluid model, but it provides no headroom for bursts, breaks, or rate-estimation error.
- **Follow-up:** Is operating exactly there safe?
- **Acceptable alternatives:** No; bursts, interruptions, and rounding need headroom.
- **Red flags:** Treating it as a service-level guarantee.
- **Scoring:** Full credit distinguishes mean boundary from queue guarantee.
- **Answer support:** Section 6.

**Q07. Why is score calibration relevant after choosing a ranking model?**
- **Checked answer:** A threshold chosen from expected risk/cost assumes scores correspond to observed risk. Ranking can remain good while predicted probabilities are overconfident or shift by lot.
- **Reasoning:** A threshold chosen from expected risk/cost assumes scores correspond to observed risk. Ranking can remain good while predicted probabilities are overconfident or shift by lot.
- **Follow-up:** How do you check it?
- **Acceptable alternatives:** Reliability bins on held-out adjudicated data by relevant slices.
- **Red flags:** Using training-set calibration only.
- **Scoring:** Full credit names held-out, slice-aware evidence.
- **Answer support:** Sections 5c and 9.

**Q08. A new lot doubles defect prevalence but rates are assumed unchanged. What changes?**
- **Checked answer:** At p=3%, review becomes 13,294.80 and PPV rises to 44.841% in the stated sensitivity. The quality consequence still needs defect-type validation.
- **Reasoning:** At p=3%, review becomes 13,294.80 and PPV rises to 44.841% in the stated sensitivity. The quality consequence still needs defect-type validation.
- **Follow-up:** What may not stay unchanged?
- **Acceptable alternatives:** TPR/FPR, calibration, and reviewer behavior can change with the lot.
- **Red flags:** Assuming prevalence changes only PPV with certainty.
- **Scoring:** Full credit recalculates and questions stability.
- **Answer support:** Section 6 sensitivity.

**Q09. A camera loses focus during a shift. What should the system do?**
- **Checked answer:** Detect health degradation with reference/golden-part checks, prevent automated accept for affected views, route to manual handling, preserve the incident interval, then clean/refocus and verify before release.
- **Reasoning:** Detect health degradation with reference/golden-part checks, prevent automated accept for affected views, route to manual handling, preserve the incident interval, then clean/refocus and verify before release.
- **Follow-up:** Why not keep accepting low scores?
- **Acceptable alternatives:** The score distribution is no longer validated for degraded sensing.
- **Red flags:** Merely retraining without physical recovery.
- **Scoring:** Full credit prioritizes containment and traceability.
- **Answer support:** Section 12.

**Q10. What is label ambiguity here, and how do you manage it?**
- **Checked answer:** A cosmetic variation, rework route, and customer-relevant defect may differ. Use specification-linked adjudication, an uncertain state, disagreement records, and taxonomy versions.
- **Reasoning:** A cosmetic variation, rework route, and customer-relevant defect may differ. Use specification-linked adjudication, an uncertain state, disagreement records, and taxonomy versions.
- **Follow-up:** Can rework code be the label?
- **Acceptable alternatives:** Not without audit; it can encode caution or downstream routing.
- **Red flags:** Forcing disagreement into a binary label.
- **Scoring:** Full credit protects label provenance.
- **Answer support:** Sections 5a and 5d.

**Q11. Why is an automatic reject intentionally excluded?**
- **Checked answer:** The scenario retains human final authority because false rejects and correction decisions carry physical and business consequences, and no validation supports autonomy.
- **Reasoning:** The scenario retains human final authority because false rejects and correction decisions carry physical and business consequences, and no validation supports autonomy.
- **Follow-up:** When might the policy change?
- **Acceptable alternatives:** Only after governed evidence and explicit authority, outside this initial case.
- **Red flags:** Equating high confidence with authorization.
- **Scoring:** Full credit separates model confidence from authority.
- **Answer support:** Sections 4 and 8.

**Q12. What does an inference timeout mean operationally?**
- **Checked answer:** It means the part lacks a timely trustworthy route. Emit a manual-handling event, use the physical fallback lane, reconcile the part ID, and investigate latency reason codes.
- **Reasoning:** It means the part lacks a timely trustworthy route. Emit a manual-handling event, use the physical fallback lane, reconcile the part ID, and investigate latency reason codes.
- **Follow-up:** What must not happen?
- **Acceptable alternatives:** Silent accept or duplicate final disposition.
- **Red flags:** Retrying until a physical part has passed the decision point.
- **Scoring:** Full credit gives a safe state transition.
- **Answer support:** Section 8.

**Q13. Compare a detector and a part-level classifier for this task.**
- **Checked answer:** A classifier may offer simpler triage; a detector/segmenter may support visible-region review. Neither is selected by fashion: use label availability, defect geometry, reviewer need, and grouped evaluation.
- **Reasoning:** A classifier may offer simpler triage; a detector/segmenter may support visible-region review. Neither is selected by fashion: use label availability, defect geometry, reviewer need, and grouped evaluation.
- **Follow-up:** What if defects are diffuse?
- **Acceptable alternatives:** Region labels/overlap may be inappropriate; evaluate part-level decision instead.
- **Red flags:** Assuming localization always improves quality.
- **Scoring:** Full credit gives selection conditions.
- **Answer support:** Section 5c.

**Q14. What plausible wrong method could make the project look successful?**
- **Checked answer:** Randomly splitting all images lets near-duplicate views and lot appearance leak across train/test. It can inflate metrics while failing on a future lot.
- **Reasoning:** Randomly splitting all images lets near-duplicate views and lot appearance leak across train/test. It can inflate metrics while failing on a future lot.
- **Follow-up:** What is the correction?
- **Acceptable alternatives:** Part/lot/time grouped splits plus overlap audits and a later-lot final holdout.
- **Red flags:** Only adding more augmentations.
- **Scoring:** Full credit explains why the original result is invalid.
- **Answer support:** Section 5d.

**Q15. The review queue is below mean capacity but waits spike at breaks. What changes?**
- **Checked answer:** Model arrivals and service by time block, preserve headroom, set queue-age alerts, redistribute staff or throttle routing, and route overflow to manual handling.
- **Reasoning:** Model arrivals and service by time block, preserve headroom, set queue-age alerts, redistribute staff or throttle routing, and route overflow to manual handling.
- **Follow-up:** Does 24.33% utilization prove no waits?
- **Acceptable alternatives:** No; averages hide bursts and interruptions.
- **Red flags:** Declaring capacity solved from shift totals.
- **Scoring:** Full credit distinguishes fluid calculation from queue behavior.
- **Answer support:** Sections 6 and 9.

**Q16. How would you decide whether an alert-cost threshold should tighten?**
- **Checked answer:** Compare the incremental reduction in escaped-defect consequence against review labor, false-reject correction, capacity, and human-factor guardrails using held-out evidence. Recompute under changed prevalence and costs.
- **Reasoning:** Compare the incremental reduction in escaped-defect consequence against review labor, false-reject correction, capacity, and human-factor guardrails using held-out evidence. Recompute under changed prevalence and costs.
- **Follow-up:** What evidence is missing here?
- **Acceptable alternatives:** Actual costs, conditional rates, queue data, and quality authority acceptance.
- **Red flags:** Optimizing F1 without decision costs.
- **Scoring:** Full credit states both quantitative and governance constraints.
- **Answer support:** Sections 3, 6, and 9.

### Operational control cards

#### Part identity handshake
- **Failure boundary:** PLC sequence, readable code, and manifest ID can disagree.
- **Diagnostic signal:** Sequence gaps and code/manifest mismatch rate.
- **Immediate recovery:** Route unmatched events to manual; never borrow adjacent identity.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Clock synchronization
- **Failure boundary:** Late or non-monotonic timestamps corrupt time splits and reconciliation.
- **Diagnostic signal:** Offset and monotonicity checks across station devices.
- **Immediate recovery:** Mark time-uncertain records and repair synchronization before evaluation.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Required-view manifest
- **Failure boundary:** One missing view can hide a defect-bearing surface.
- **Diagnostic signal:** Completion fraction by camera and revision.
- **Immediate recovery:** Make any absent required view a manual route.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Exposure saturation
- **Failure boundary:** Clipped highlights or shadows erase texture evidence.
- **Diagnostic signal:** Pixel-extrema fraction by camera.
- **Immediate recovery:** Hold affected view and restore approved exposure setting.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Lens contamination
- **Failure boundary:** Dust and oil become a sensing fault, not a defect class.
- **Diagnostic signal:** Sharpness trend and fixed-pattern residue.
- **Immediate recovery:** Clean under procedure and verify with a reference part.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Mechanical vibration
- **Failure boundary:** Mount movement changes registration and blur.
- **Diagnostic signal:** Fiducial displacement and blur during conveyor operation.
- **Immediate recovery:** Hold automated accepts, secure mount, and realign.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Illumination aging
- **Failure boundary:** Changing light output shifts appearance and scores.
- **Diagnostic signal:** Reference-panel intensity trend.
- **Immediate recovery:** Replace/recalibrate light and version the changed setting.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Ambient-light intrusion
- **Failure boundary:** External light leaks can create shift-specific alerts.
- **Diagnostic signal:** Score change correlated with enclosure state/time.
- **Immediate recovery:** Repair shield and retain manual handling until stable.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Camera replacement
- **Failure boundary:** Same camera model can have different color and optics.
- **Diagnostic signal:** Serial-number/configuration fingerprint mismatch.
- **Immediate recovery:** Treat replacement as unvalidated sensing configuration.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Firmware change
- **Failure boundary:** Firmware can change trigger timing or image encoding.
- **Diagnostic signal:** Firmware delta in manifest.
- **Immediate recovery:** Freeze automated policy pending compatibility test.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Part orientation
- **Failure boundary:** A flipped or rotated part can make target area invisible.
- **Diagnostic signal:** Pose confidence and fiducial geometry.
- **Immediate recovery:** Send out-of-pose part to fixture correction/manual review.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Occlusion
- **Failure boundary:** Fixture, hand, debris, or packaging can cover the region.
- **Diagnostic signal:** Required-region visibility flag.
- **Immediate recovery:** Manual route and capture only after obstruction removal.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Compression change
- **Failure boundary:** Lossy encoding can erase small scratches.
- **Diagnostic signal:** Codec/version and high-frequency artifact audit.
- **Immediate recovery:** Restore approved encoding and re-evaluate candidate.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Color pipeline
- **Failure boundary:** Auto white balance confounds print and discoloration analysis.
- **Diagnostic signal:** Color-patch error and white-balance setting.
- **Immediate recovery:** Lock approved pipeline or use manual color check.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Crop registration
- **Failure boundary:** Offset crop can omit the feature under inspection.
- **Diagnostic signal:** Fiducial-to-crop displacement.
- **Immediate recovery:** Reject crop config and use full-frame/manual route.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Resolution mismatch
- **Failure boundary:** Serving scale may not resolve training-scale defect.
- **Diagnostic signal:** Input dimensions and pixels-per-millimeter proxy.
- **Immediate recovery:** Block mismatched deployment and recapture/revalidate.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Exact duplicate image
- **Failure boundary:** Repeated bytes can signal trigger or storage fault.
- **Diagnostic signal:** Checksum collision across event IDs.
- **Immediate recovery:** Quarantine pair and reconcile physical sequence.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Near duplicate image
- **Failure boundary:** Similarity can leak across splits or reveal stalled flow.
- **Diagnostic signal:** Embedding similarity with sequence audit.
- **Immediate recovery:** Keep in one split and investigate repeat capture.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Lot leakage
- **Failure boundary:** Shared process conditions make randomized splits optimistic.
- **Diagnostic signal:** Lot membership in each split.
- **Immediate recovery:** Assign each lot wholly to one split and rerun.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Time leakage
- **Failure boundary:** Later rework/disposition fields encode the answer.
- **Diagnostic signal:** Feature availability timestamp audit.
- **Immediate recovery:** Remove late fields and retrain/evaluate.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Revision leakage
- **Failure boundary:** Revision ID can proxy known quality outcome.
- **Diagnostic signal:** Error and score by revision.
- **Immediate recovery:** Separate revisions or document supported mapping.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Reviewer overlay leakage
- **Failure boundary:** Annotations or old disposition reveal the label.
- **Diagnostic signal:** Input-pipeline visual audit.
- **Immediate recovery:** Mask layers and regenerate training inputs.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Preprocessing leakage
- **Failure boundary:** Test distribution fitted into normalization inflates result.
- **Diagnostic signal:** Fold-specific transform digest.
- **Immediate recovery:** Fit transforms on training fold only.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Label ambiguity
- **Failure boundary:** Cosmetic variation and defect have disputed boundary.
- **Diagnostic signal:** Adjudicator disagreement matrix.
- **Immediate recovery:** Version taxonomy and retain uncertain label.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Delayed label
- **Failure boundary:** Unadjudicated records are not known negatives.
- **Diagnostic signal:** Label-delay distribution and censored fraction.
- **Immediate recovery:** Freeze cutoff and report immature-label exclusion.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Rework-label bias
- **Failure boundary:** Rework route is not always true defect evidence.
- **Diagnostic signal:** Cross-tab rework versus adjudication.
- **Immediate recovery:** Use adjudicated target; retain rework separately.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Selection bias
- **Failure boundary:** Only alerts may receive rich labels.
- **Diagnostic signal:** Label coverage by route.
- **Immediate recovery:** Randomly audit accepted parts under approved plan.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Class imbalance
- **Failure boundary:** Rare defects hide beneath high average accuracy.
- **Diagnostic signal:** Counts and rates by defect type.
- **Immediate recovery:** Use stratified raw-count reporting.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Hard-negative drift
- **Failure boundary:** Permitted variation and glare can dominate alerts.
- **Diagnostic signal:** False-positive taxonomy trend.
- **Immediate recovery:** Collect representative negatives into training only.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Unphysical augmentation
- **Failure boundary:** Synthetic changes may not match optics or process.
- **Diagnostic signal:** Recipe review and sample audit.
- **Immediate recovery:** Remove unsupported transform and compare real holdout.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Threshold change
- **Failure boundary:** Policy alters workload and error tradeoff.
- **Diagnostic signal:** Policy digest, approver, effective time.
- **Immediate recovery:** Require approved change record and new report segment.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Calibration decay
- **Failure boundary:** Risk meaning changes even when ranking remains useful.
- **Diagnostic signal:** Recent reliability bins by slice.
- **Immediate recovery:** Recalibrate on held-out labels or return manual.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Model update
- **Failure boundary:** New weights change behavior behind same endpoint.
- **Diagnostic signal:** Model digest and release ledger.
- **Immediate recovery:** Shadow/canary and atomic rollback.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Bundle mismatch
- **Failure boundary:** Model, calibrator, and threshold can be incompatible.
- **Diagnostic signal:** Approved tuple audit.
- **Immediate recovery:** Manual route mismatched tuple and restore bundle.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Inference saturation
- **Failure boundary:** Concurrent images raise latency and timeout.
- **Diagnostic signal:** Queue depth, utilization, p99 latency.
- **Immediate recovery:** Throttle/scale or send overflow manual.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Network partition
- **Failure boundary:** Storage, inference, or policy dependency can disappear.
- **Diagnostic signal:** Dependency health and timeout codes.
- **Immediate recovery:** Fail to manual handling with traceable incident.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Event retry
- **Failure boundary:** At-least-once delivery can duplicate dispositions.
- **Diagnostic signal:** Duplicate event-ID counter.
- **Immediate recovery:** Deduplicate with immutable idempotency key.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### MES acknowledgement
- **Failure boundary:** A digital decision can fail to reach correction workflow.
- **Diagnostic signal:** Acknowledgement age and terminal-state gaps.
- **Immediate recovery:** Hold/reconcile and retry idempotently.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Physical lane divergence
- **Failure boundary:** Digital route may not match actual location.
- **Diagnostic signal:** Lane sensor/event-state reconciliation.
- **Immediate recovery:** Stop affected lane and manually reconcile.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Workbench latency
- **Failure boundary:** Slow screens lower reviewer service rate.
- **Diagnostic signal:** Page-load and handling-time distribution.
- **Immediate recovery:** Simplify rendering or revise capacity plan.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Automation bias
- **Failure boundary:** Reviewers may over-trust a score or heat map.
- **Diagnostic signal:** Blind A/B disagreement and error analysis.
- **Immediate recovery:** Hide/simplify overlay and retrain reviewers.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Reviewer fatigue
- **Failure boundary:** Service and error rates vary by time and alert load.
- **Diagnostic signal:** Handling time/disagreement by staffed block.
- **Immediate recovery:** Rotate staff, add breaks, or lower routed volume.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Correction queue
- **Failure boundary:** False rejects can clog rework capacity.
- **Diagnostic signal:** Rework WIP, age, and reason.
- **Immediate recovery:** Tighten policy or add verification.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Escaped defect discovery
- **Failure boundary:** Downstream finding needs causal trace, not assumption.
- **Diagnostic signal:** Part ID, route, evidence, and defect-family trace.
- **Immediate recovery:** Contain lot and classify sensing/model/human/process cause.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Novel defect
- **Failure boundary:** Closed-set score can accept a new appearance family.
- **Diagnostic signal:** Escaped-defect taxonomy and novelty review.
- **Immediate recovery:** Increase manual sampling and evaluate new family separately.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Drift alert
- **Failure boundary:** A score shift can arise from mix, light, or software.
- **Diagnostic signal:** Joint telemetry, lot/revision, score, and label view.
- **Immediate recovery:** Investigate cause before retraining.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Metric denominator
- **Failure boundary:** A rate can hide exclusions and censorship.
- **Diagnostic signal:** Report numerator, denominator, window, exclusion fields.
- **Immediate recovery:** Reject incomplete dashboard calculation.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Cost scope
- **Failure boundary:** Labor, correction, capital, and downtime differ in scope.
- **Diagnostic signal:** Dated source and included/excluded cost fields.
- **Immediate recovery:** Mark unknowns; do not call estimate ROI.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Prevalence shift
- **Failure boundary:** PPV and expected costs change with defect mix.
- **Diagnostic signal:** Recent adjudicated prevalence by lot/revision.
- **Immediate recovery:** Recompute policy sensitivity.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Burst arrival
- **Failure boundary:** Shift average hides batch-release spikes.
- **Diagnostic signal:** Per-minute arrival and queue-age trace.
- **Immediate recovery:** Reserve headroom or route burst overflow manual.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Fallback staffing
- **Failure boundary:** A manual fallback fails without available trained people.
- **Diagnostic signal:** Roster, training, and lane-readiness check.
- **Immediate recovery:** Pause automation when fallback unavailable.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Stop criterion
- **Failure boundary:** Pilot needs predeclared pause conditions.
- **Diagnostic signal:** Missing-view, queue-age, slice-error, trace-gap thresholds.
- **Immediate recovery:** Pause, contain population, and review authority.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Audit retrieval
- **Failure boundary:** Disputed disposition needs original evidence and versions.
- **Diagnostic signal:** Sample manifest-to-object retrieval.
- **Immediate recovery:** Repair retention/index permissions.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Access control
- **Failure boundary:** Images and reviewer records require least privilege.
- **Diagnostic signal:** Role review and access-log anomaly.
- **Immediate recovery:** Remove excess grant and investigate incident.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Retention lifecycle
- **Failure boundary:** Too-early deletion blocks investigation; over-retention breaks policy.
- **Diagnostic signal:** Lifecycle configuration against approved schedule.
- **Immediate recovery:** Hold deletion for review and apply approved schedule.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Operator communication
- **Failure boundary:** Route meaning changes require procedural understanding.
- **Diagnostic signal:** Acknowledged training/version bulletin.
- **Immediate recovery:** Pause cutover until briefing and drill complete.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Independent release review
- **Failure boundary:** Model author alone cannot certify operational evidence.
- **Diagnostic signal:** Approval roles and unresolved objections.
- **Immediate recovery:** Escalate disagreement and retain baseline.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Retirement trigger
- **Failure boundary:** Process or product change can obsolete the candidate.
- **Diagnostic signal:** Overrides, unsupported mix, maintenance burden.
- **Immediate recovery:** Retire to manual baseline and archive artifacts.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Golden-part control
- **Failure boundary:** A reference part tests whole sensing path, not model quality.
- **Diagnostic signal:** Scheduled golden-part result by camera/view.
- **Immediate recovery:** Manual route while control fails; repair sensing path.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Queue-age alarm
- **Failure boundary:** Oldest review matters more than average queue count.
- **Diagnostic signal:** Maximum and percentile review wait.
- **Immediate recovery:** Escalate staffing or throttle incoming review work.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Reconciliation sweep
- **Failure boundary:** Every READY state must reach a terminal state exactly once.
- **Diagnostic signal:** Nonterminal age and duplicated terminal state count.
- **Immediate recovery:** Hold/reconcile missing records before release.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Change freeze
- **Failure boundary:** Untracked changes make evaluation comparison uninterpretable.
- **Diagnostic signal:** Configuration-change ledger completeness.
- **Immediate recovery:** Freeze policy/config during measurement window.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.

#### Human escalation
- **Failure boundary:** Uncertain reviewer cases need accountable resolution.
- **Diagnostic signal:** Escalation rate and resolution time.
- **Immediate recovery:** Route to designated quality authority with evidence.
- **Accountable owner:** The quality manager assigns the responsible controls, engineering, platform, or operations role in the release record.
- **Evidence to retain:** Preserve the part/lot/time scope, configuration versions, diagnostic result, recovery action, and closure decision.
- **Verification:** Demonstrate that affected parts are contained or reconciled and that the stated diagnostic returns to its approved operating range.
## 15. Quick Reference Card

| Field | Decision-ready content |
| --- | --- |
| Decision | Pilot only an auditable review-gated visual triage policy; do not claim deployment or autonomous rejection. |
| Owner and authority | Quality manager owns policy; reviewer owns final disposition; line leader owns stop/slow response. |
| Input / output | Six-view part inspection event -> accept, review, or manual-handling route; reviewer may reject/rework. |
| Baseline | Manual screen every part; illustrative six-reviewer staffing cannot cover 216,000 parts at 15 parts/min for 450 staffed min. |
| Illustrative operating point | N=216,000; p=1.2%; TPR=.92; FPR=.035; review=9,853.92; capacity=40,500. |
| Critical metric | Report raw confusion counts, PPV/NPV, capture completeness, calibration, queue behavior, reviewer performance, and slices. |
| Do not confuse | Part classification is not localization; alert is not final reject; accuracy is not defect capture; mean capacity is not burst safety. |
| Fallback | Missing view, health fault, timeout, unknown revision, or policy mismatch -> manual handling and reconciliation. |
| Evidence record | Illustrative/not-measured calculations only; no production result, vendor benchmark, or observed savings. |
| Stage / review | Proposed intervention; draft manuscript; source-body and independent publication review remain OPEN. |

**Final validity condition:** The numerical ledger is conditional on stated prevalence, model rates, reviewer rates, costs, and staffed capacity. It supports only a proposed validation decision; human authorization and measured grouped evaluation are required before any operational conclusion.
