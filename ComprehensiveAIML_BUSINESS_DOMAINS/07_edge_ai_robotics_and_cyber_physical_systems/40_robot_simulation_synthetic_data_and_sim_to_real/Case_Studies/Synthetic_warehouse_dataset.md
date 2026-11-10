---
case_id: "CASE-P07-11"
case_title: "Decide whether a synthetic warehouse dataset may be used to rank pallet detectors"
case_version: "1.0"
created: "2026-11-09"
last_reviewed: null
reviewed: null
status: "draft"
content_status: "draft"
case_type: "analytical"
domain: "Robot simulation, synthetic data and sim-to-real — evaluation-data adequacy"
primary_pattern: "provenance-controlled synthesis"
secondary_patterns: ["shift analysis", "label audit"]
evidence_level: "illustrative"
source_basis: "illustrative"
source_status: "OPEN"
as_of_date: "2026-11-09"
owner_role: "Perception evaluation lead (simulation and data quality)"
source_index: "../INDEX.md"
review_trigger: "Renderer or asset version, scene allocation, label generator, audit protocol, target site's mix, or the candidate model's training corpus changes."
---

# Synthetic Warehouse Dataset: Deciding What It May Be Used For

> **Evidence statement, read before any number.** Every count, rate, latency, price and labour figure in this file is a constructed teaching value. The dataset, the audit, the renderer, the two detectors and the evaluation board are hypothetical. Nothing here describes a real warehouse, a real vendor, a real deployment, or a measured outcome. Arithmetic is independently checked where it is labelled DERIVED; every input it consumes is ILLUSTRATIVE. Where a claim cannot be checked from this file it is marked OPEN.

> **Why this case exists.** Synthetic data is usually argued about as a *generation* problem — how many frames, how much randomisation, how photorealistic. In practice the binding constraints are *provenance* (can you say what produced a given label) and *intended use* (what decision may this dataset support). Those two constraints decide the outcome here, and they decide it against the ambitious use that motivated the spend.

## 1. The One-Line Framing

Given a procedurally rendered set of 42,000 warehouse frames with 72 labelled occupancy cells per frame, decide whether that set is adequate to rank two pallet-detection models and certify the better one for a field-trial promotion review, for a perception evaluation lead, subject to a fixed audit budget of roughly 29 analyst-hours and no access to a labelled real-domain set, compared with the current practice of accepting whichever model reports the higher aggregate accuracy on the synthetic set.

The decision is a **data-adequacy ruling**, not a model-selection decision. Its unit is one dataset version against one declared intended use. It is made once per dataset version, and again whenever the intended use widens. The reusable pattern is *provenance-controlled synthesis*: a generator whose every output carries a reconstructable record of the configuration that produced it, plus a label audit that converts "the labels are synthetic" into measured error rates. The two secondary patterns are *shift analysis* (does the synthetic composition match the site that will receive the model) and *label audit* (what do the labels actually assert).

The primary success condition is negative and therefore easy to under-value: after the ruling, no downstream document may cite a synthetic accuracy figure as evidence of field capability, and every permitted use names the frame set, the metric, and the correction that must accompany it. A ruling that says "yes, use it, it looks realistic" fails this condition even if the models trained on it later perform well, because the ruling would have been unfounded.

The supported surprising finding is that the audited label error, not the model difference, dominates the decision — and that the naive aggregate comparison picks the wrong model. Read naively, the incumbent wins by 1.68 percentage points. Composed for the site's actual 75/25 dense-to-sparse mix and corrected for measured label error, the candidate wins by 2.13 points: two independent errors, pushing in opposite directions, landing on the wrong answer. Beyond the ranking, the labels themselves disagree with adjudicated truth on 7.60 percent of cells overall and 9.90 percent at the site mix, so an assumption-free triangle bound leaves any absolute-accuracy claim ambiguous by plus or minus 9.90 points — 4.7 times the entire contrast the evaluation exists to resolve. An independent power calculation agrees from the other side: the 8-scene evaluation split has 8 percent power to detect a 2.13-point difference and cannot resolve anything smaller than 11.9 points at conventional power.

The obvious simpler framing — "compare the two models on the holdout, take the winner" — misses three things at once. It treats synthetic labels as ground truth, so it cannot separate perception skill from label reproduction. It treats 42,000 frames as 42,000 independent observations, when frames inside a scene share lighting, camera pose and asset placement and therefore carry far less information than the raw count suggests. And it treats the dataset as a single artefact, when the honest question is a *set* of narrow permissions, each with its own justification.

Task category: evaluation design and data governance, decided at dataset-release boundaries. Primary constraint: the audit labour budget. Baseline: aggregate-accuracy acceptance with no audit and no cluster adjustment.

## 2. Context & Background

### 2.1 The workflow the dataset was built to serve

The hypothetical operation runs autonomous mobile robots in a distribution centre. A robot approaches a pick face, images a bay, and must decide which of 72 fixed occupancy cells in that bay contain a pallet. That decision gates a reach: the manipulator is only permitted to extend toward cells the perception system asserts are occupied, and an assertion of occupancy on an empty cell is a reach into air, while a missed pallet is a lost pick. The perception output is therefore a per-cell binary occupancy map, produced at roughly 4 Hz while the robot is stationary at the face.

Before any of that runs on a robot in a live aisle, the team wants evidence that a candidate model beats the incumbent. The candidate was trained on synthetic renders because the site would not release camera footage for training, and because the six-month project had no labelling budget for real frames. The incumbent was trained two years earlier on a small real corpus from a different site.

This is the classic synthetic-data situation: the thing you can afford to build is not drawn from the distribution you care about, and the thing you care about is unavailable. The case is about how to be honest and still make a decision.

### 2.2 Stakeholders and who owns what

| Role | Interest | Authority in this case |
|---|---|---|
| Perception evaluation lead | Defensible adequacy ruling | Owns the ruling; may narrow or refuse the intended use |
| Simulation engineer | Renderer, assets, scene library | Owns generator version and configuration manifest |
| Data quality analyst | Audit sample, adjudication protocol, error estimates | Owns α, β estimates and their intervals |
| Modelling team | Wants a green light to promote the candidate | May re-run evaluation under corrected protocol |
| Site operations representative | Cares about missed picks and false reaches | Owns the target mix figure used for weighting |
| Field-trial review board | Accepts or rejects promotion | Consumes the ruling; cannot override the adequacy limits |

The evaluation lead does not select the model. That separation matters: a person who both certifies the data and picks the winner has an incentive to certify generously. The ruling is written before the model comparison is run, and it names the comparison protocol the modelling team must follow.

### 2.3 Scale, timing, geography

| Quantity | Value used in this case | Status |
|---|---|---|
| Rendered frames | 42,000 | ILLUSTRATIVE |
| Scenes (distinct aisle/lighting/asset arrangements) | 24 | ILLUSTRATIVE |
| Frames per scene | 1,750 | DERIVED: 42,000 / 24 |
| Cells per frame | 72 (8 columns × 9 rows) | ILLUSTRATIVE |
| Total labelled cells | 3,024,000 | DERIVED: 42,000 × 72 |
| Render time per frame | 0.42 s on one worker | ILLUSTRATIVE |
| Frame bundle size (RGB plus depth, compressed) | 0.42 MB | ILLUSTRATIVE |
| Scene split | 12 dense-pallet, 12 sparse-pallet | ILLUSTRATIVE |
| Calendar position | Dataset frozen; evaluation not yet run | ILLUSTRATIVE |
| Geography | Single distribution centre, one aisle type | ILLUSTRATIVE |

The 24-scene figure is the one that will matter most, and it is the figure nobody chose deliberately. Scenes were added until the render queue was exhausted, not until a power calculation was satisfied. That is how most synthetic datasets acquire their size.

### 2.4 The baseline and why it persists

The existing practice is simple and fast: score both models on the 42,000-frame holdout, compare per-cell accuracy, and promote the higher number if the gap exceeds one percentage point. It persists because it is cheap, because it produces a single ordered pair that a review board can read in ten seconds, and because nothing in the current process generates a signal that the number is untrustworthy. A synthetic label does not look uncertain. It is a crisp JSON integer in a file the pipeline wrote itself.

The baseline has three documented weaknesses, all of which are structural rather than careless:

1. It has no error model for the labels, so a model that reproduces generator artefacts scores well.
2. It pools frames as if independent, so confidence intervals are too narrow by roughly the square root of the design effect.
3. It reports a level, not a contrast, so composition differences between synthetic and real populations are invisible in the number that is quoted.

### 2.5 Prior attempts and organisational context

Two earlier attempts are relevant and are recorded here as *considered*, not as tested failures. First, an attempt to obtain 2,000 labelled real frames was declined on data-sharing grounds; that is a constraint, not a technical result. Second, an attempt to use a photorealistic third-party asset pack was rejected because the pack's licence forbade use in model training, leaving only permissively licensed assets whose surface materials are visibly simpler than the real aisle. Both facts shape what the dataset can claim: it is a *structured* rendering, not a photorealistic one, and its domain gap is therefore both geometric (asset shapes) and photometric (materials, lighting).

The case does not assert that photorealistic rendering would have solved the problem. It asserts only that the available generator has a known and unmeasured gap, and that the ruling must survive that gap rather than assume it away.

### 2.6 What this case deliberately excludes

No claim is made about functional safety, about the manipulator's hazard analysis, or about whether per-cell occupancy is the right interface for a reach decision. Those belong to modules 38, 39 and 41. The case also excludes the candidate model's architecture and training recipe; both models are treated as black boxes that consume frames and emit per-cell probabilities. Finally, the case excludes any regulatory determination: no rule is cited, because none was verified for this hypothetical setting.

**Takeaway:** the interesting constraint is not how realistic the renderer is. It is that nobody in the workflow had ever written down what the dataset was allowed to be used for.

## 3. The Business Question

### 3.1 The question in the stakeholder's words

Paraphrased from the illustrative evaluation lead's brief to the review board:

> "We have 42,000 synthetic frames and two detectors. Before we spend the field-trial slot, tell me what this dataset can and cannot prove — and if it cannot prove that the candidate is better, tell me exactly what would, and what it costs. I have about 29 analyst-hours for auditing, not 290."

The question is deliberately two-part. The first part asks for a verdict on evidence. The second asks for the cheapest sufficient upgrade. A ruling that only refuses is unusable, because the project still has to make a decision.

### 3.2 The decision contract

| Decision contract field | Content |
|---|---|
| Question | May this dataset version support (a) an absolute accuracy claim, (b) a ranking between the two detectors, (c) a promotion decision, or (d) only pipeline smoke tests? |
| Decision owner | Perception evaluation lead, with the data quality analyst accountable for the error estimates |
| Trigger and deadline | Dataset frozen at version `syn-wh-1.0`; ruling required before the modelling team runs the comparison, so that the comparison cannot be designed around a favourable answer |
| Available actions | Approve all uses; approve ranking only with declared corrections; approve smoke-test and regression use only; reject the dataset; commission more scenes before any use |
| Value and downside | A wrong "yes" promotes a model on label artefacts and consumes a field-trial slot that takes weeks to rebook; a wrong "no" delays the programme and wastes 4.9 GPU-hours of rendering plus 28.8 analyst-hours of audit |
| Objective | Maximise the number of downstream claims that are reconstructable and bounded, not the number of claims that are favourable |
| Hard constraints | Audit budget ≈ 29 analyst-hours; no labelled real-domain set; no change to the frozen frame set after the ruling |
| Scope | One dataset version, one aisle type, one site, two fixed model artefacts, one metric family |
| Out of scope | Safety case, manipulator control, model architecture choice, procurement, and any statement about a second site |

### 3.3 Objective versus proxy

The business objective is fewer bad reach decisions in the aisle. The proxy available today is per-cell accuracy against synthetic labels. These can disagree in both directions. A model can gain proxy accuracy by learning the renderer's characteristic edge artefacts — the way a synthetic pallet's shrink-wrap highlights always fall in the same place — while losing real accuracy on the site's actual wrap. Conversely a model can lose proxy accuracy by declining to fire on cells the generator mislabels, while being more correct about the physical world.

The evaluation therefore needs a guardrail that does not depend on synthetic labels at all. Two are available cheaply:

- **Abstention behaviour.** The fraction of cells the model declines to assert should not collapse; a model that asserts everywhere looks accurate on a label set that is mostly correct by base rate.
- **Cross-model disagreement structure.** If the candidate beats the incumbent only on cells where the two disagree *and* the label was produced by the same generator family that trained the candidate, the gain is suspect.

Neither guardrail proves field capability. Both can falsify a claim that the synthetic number means what it appears to mean.

**Takeaway:** the contract is not "is the dataset good". It is "which of four named uses does this dataset license, and what must accompany each".

## 4. Assumptions & Constraints

### 4.1 Assumption register

| ID | Assumption | Type | Verification plan | Status | Impact if wrong | Fallback | Recheck trigger / owner |
|---|---|---|---|---|---|---|---|
| A-01 | The 42,000 labelled cells sampled for audit are exchangeable with the full 3,024,000 cells | evaluation | Compare audited frames' scene and density mix to the full set | unverified | α, β do not apply to the whole set; every correction is mis-scaled | stratify the audit and report per-stratum α, β | scene allocation change / data quality analyst |
| A-02 | Label error is independent of which model is being scored | model | Compare each model's error rate on audited cells against its overall error rate | unverified and likely false for the candidate | the attenuation correction under-corrects; the candidate's measured advantage is inflated | treat the corrected gap as an upper bound | candidate retrained on synthetic labels / evaluation lead |
| A-03 | Frames within a scene share about 0.08 of their variance | data | Estimate the intraclass correlation from per-scene accuracy variance | unverified; value assumed | effective sample size wrong in either direction | report a range over ρ from 0.02 to 0.20 | new scene library / data quality analyst |
| A-04 | The target site's frame mix is 75 percent dense, 25 percent sparse | business | Operations provides a 30-day bay census | unverified | weighted levels shift; the contrast may shift too | report both weighted and unweighted | site changes layout / operations representative |
| A-05 | Per-scene accuracy standard deviation of the paired difference is 0.12 | evaluation | Estimate from the 24 available scenes after scoring | unverified; assumed | power calculation off by up to 15 percent at this cluster count | re-power after the first pass | first scoring run / evaluation lead |
| A-06 | The frozen frame set will not change during the evaluation | reproducibility | Hash the frame manifest and the label archive | verified by procedure, not yet executed | any comparison becomes incomparable mid-stream | re-freeze under a new version identifier | any manifest diff / simulation engineer |
| A-07 | Renderer output is deterministic given seed and configuration | reproducibility | Render one scene twice, compare hashes | unverified | "same dataset" claims are unfounded; audit sample not reproducible | pin the seed and record the full config hash | renderer upgrade / simulation engineer |
| A-08 | 6 seconds per cell is a realistic adjudication rate | cost | Time a 5-frame pilot adjudication | unverified | audit budget wrong by a large factor | re-scope the audit sample size | pilot timing / data quality analyst |

Assumptions A-02 and A-03 are the two most likely to reverse the ruling. A-02 decides whether the corrected contrast is an estimate or an upper bound; A-03 decides whether the audit's 17,280 cell adjudications are worth hundreds or thousands of independent observations, which in turn decides how much confidence the α and β estimates themselves deserve. A-04 is the third: it changes the reported level but, as Section 6 shows, hardly changes the contrast — which is itself a finding worth reporting rather than a nuisance to hide.

### 4.2 Constraint register

| ID | Constraint | Type | Source | Status | Consequence if violated | Response |
|---|---|---|---|---|---|---|
| C-01 | Audit labour ≤ 29 analyst-hours | cost | Illustrative budget statement from the evaluation lead | confirmed as a scenario input | audit sample shrinks; α and β intervals widen | reduce cells per frame, not frames per scene |
| C-02 | No labelled real-domain frames available | data | Illustrative data-sharing refusal | confirmed as a scenario input | no direct domain-gap estimate is possible | state the gap as unmeasured; bound claims accordingly |
| C-03 | Frame set frozen at `syn-wh-1.0` | reproducibility | Change control for the evaluation | confirmed | results not comparable across model runs | version bump and full re-run |
| C-04 | Labels may not be regenerated from the models under test | evaluation | Separation of data generation from model selection | proposed | circularity: the labels would encode the candidate | refuse the experiment and escalate |
| C-05 | Ruling issued before the comparison is run | governance | Review-board sequencing rule | proposed | comparison designed around a favourable protocol | re-issue ruling under a new chair |
| C-06 | Asset library restricted to permissive licences | legal | Illustrative licence review | confirmed as a scenario input | photometric realism is capped | declare the cap; do not claim photorealism |

C-04 deserves emphasis because it is the constraint most often violated in practice without anyone noticing. Regenerating labels "to fix obvious mistakes" after seeing a model's failures converts the evaluation set into a training signal. The audit in this case therefore reads labels; it never rewrites them. Corrections are recorded as an error model, and the original labels stay bit-identical.

**Takeaway:** six constraints, two of which exist purely to stop the evaluation from becoming circular. Those two are the ones a rushed team drops first.

## 5. Data, Tools & Method

### 5a. Data Specification

| Field | Specification |
|---|---|
| Unit of analysis | One occupancy cell within one frame. Primary analysis unit for accuracy is the cell; the independent unit for inference is the scene. |
| Keys | `frame_id` (scene index, camera pose index, frame counter), `cell_id` (row, column), `scene_id`, `config_hash`. Cell key is `(frame_id, cell_id)`. |
| Volume and window | 42,000 frames × 72 cells = 3,024,000 cell labels; 24 scenes; one dataset version; no time dimension because scenes are static arrangements sampled over poses. |
| Inputs | RGB render (640×480, 8-bit per channel), depth render (640×480, 16-bit), camera intrinsics and extrinsics, cell polygon table in bay coordinates, scene configuration record. |
| Label | `occupied ∈ {0,1}` per cell, produced by the generator from the scene's asset placement, then rasterised through the cell polygon table. "Occupancy" means the cell's projected area is covered above a fixed threshold by a pallet asset. |
| Label semantics gap | The label asserts *asset presence in the scene graph*, not *visible pallet surface in the image*. Occlusion, self-occlusion and depth-only visibility are not distinguished. This is the single largest source of false negatives. |
| Population and exclusions | All frames from `syn-wh-1.0`. Excluded: 0 frames. Quarantined but retained and counted: 84 frames whose render log shows a missing texture warning. |
| Join semantics | Labels join to frames on the composite key only. One label row per cell; duplicates are a generator fault and must be resolved by refusing the frame, not by deduplication, because a duplicate implies an ambiguous scene state. |
| Splits | Scene-level holdout: 16 scenes for model development, 8 scenes for evaluation. The audit sample of 240 frames was drawn from all 24 scenes so that α and β describe the whole dataset. |
| Missingness | No missing cells by construction; a missing cell is a defect. Depth may be missing where the renderer's ray missed geometry, which is recorded per pixel and aggregated to a per-cell depth-coverage fraction. |
| Time zone and clock | Not applicable: no capture clock. Frame order within a scene is a pose index, not a time. |

The choice of cell as the analysis unit and scene as the inference unit is the most consequential line in this table. Cells within a frame are spatially adjacent and share the same object; frames within a scene share the same asset placement, lighting rig and camera track. Both correlations reduce the information content of the raw 42,000-frame count. The audit's 17,280 cell adjudications therefore are not 17,280 independent observations, and the report must say so in the same sentence that quotes the interval.

The split deserves scrutiny too. Splitting by scene, rather than by frame, prevents the same pallet arrangement appearing in both development and evaluation. Splitting by frame would leak: adjacent poses of the same bay are nearly identical images, so an evaluation frame would have a near-duplicate in training. Scene-level splitting is the minimum defence, and it is the reason the evaluation set has only 8 scenes — a number that will limit the power calculation in Section 6.

### 5b. Tools & Stack

| Component | Role in this case | Version discipline |
|---|---|---|
| Procedural scene assembler | Samples aisle geometry, pallet placements, asset variants | Seeds and config recorded in the manifest |
| Rasterising renderer | Produces RGB and depth passes | Renderer version pinned; config hash stored per frame |
| Label rasteriser | Projects scene-graph occupancy into the 72-cell grid | Must version with the cell polygon table |
| Manifest writer | Records config hash, asset hashes, seed, renderer build | One manifest per dataset version, immutable |
| Evaluation harness | Loads frames, runs a model, writes per-cell predictions | Scores only against the frozen archive |
| Cluster bootstrap utility | Resamples scenes, not cells | Seed recorded |
| Adjudication tool | Shows a frame, a cell overlay, and a truth/unknown control | Records analyst, timestamp, decision |
| Audit sampler | Draws frames stratified by scene and density | Sample list stored as an artefact |

No component is a benchmark of anything. The table exists because the ruling's reproducibility claim depends on each version being pinned, and because the two most common failure modes in this case — an unpinned renderer and an unversioned label rasteriser — are visible as missing rows in a stack table like this one.

The adjudication tool's `unknown` control matters. Without a way to record "the cell boundary is ambiguous here", analysts are forced to choose, and forced choices bias β downward in exactly the region where the label semantics are weakest (partially occluded pallets). Permitting `unknown` and reporting its rate is part of the audit protocol, not a sign of an unfinished job.

### 5c. Method Summary

The method has four stages, in a fixed order, because each stage constrains the next.

```text
[Scene library + assets]
      -> sample arrangement, placement, lighting, camera track
[Renderer]  -> RGB + depth + scene graph
      -> label rasteriser -> per-cell labels
[Manifest writer] -> immutable provenance record (config hash per frame)
      -> scene-level split (16 dev / 8 eval)
[Frozen archive syn-wh-1.0]
      |
      +-> audit sampler (240 frames, stratified)
      |        -> adjudication -> alpha, beta, unknown rate
      |        -> design effect -> effective audit size
      |
      +-> evaluation harness (both models, identical frames)
               -> per-cell accuracy, per-scene accuracy
               -> cluster bootstrap over scenes
               -> weighting by site mix
               -> adequacy ruling
```

Input to the pipeline is a scene library and an asset set. Output is a ruling, not a model. The post-processing steps — design effect, weighting, attenuation — are all *interpretive*: they do not change any model's predictions, they change what may be concluded from them.

| Candidate approach | Evidence considered | Strength | Limitation / avoid when | Status |
|---|---|---|---|---|
| Aggregate accuracy on synthetic labels (baseline) | The number the pipeline already produces | Free, fast, one ordered pair | No label-error model; no clustering; no composition control | Chosen as the comparator to be tested, rejected as sufficient |
| Synthetic labels plus audit-derived error model | 240-frame adjudication | Converts "synthetic" into measured α and β | Audit is 5.9× the render cost; α, β still sample-estimated | Chosen |
| Train on synthetic, evaluate on a small real adjudicated sample | No real sample exists | Directly measures the quantity of interest | Not available in this scenario | Not tested — blocked by C-02 |
| Domain-randomised re-render to close the gap | Renderer supports randomisation | Could widen coverage cheaply | Does not fix label semantics; cannot be validated without real data | Considered, deferred |
| Per-frame human audit of all 42,000 frames | Not affordable | Would remove label ambiguity | 42,000 × 432 s = 5,040 analyst-hours | Rejected on cost |
| Rank on disagreement-only subset | Model outputs plus labels | Focuses on informative cells | Disagreement subset is not exchangeable; worse bias than full set | Considered, rejected |

The chosen method is deliberately unglamorous. It does not improve the dataset. It makes the dataset's limitations *measurable*, then rules on them. That is the transferable move: when you cannot close a gap, bound it.

Stop condition for the audit: the sample size is fixed in advance at 240 frames because that is what 28.8 analyst-hours buys at the measured adjudication rate, and because that sample yields a per-cell α interval narrow enough to make the ruling's conclusion insensitive to the estimate (Section 6.6 checks this). Stopping is not triggered by α reaching a flattering value. If the audit had returned α = 0.05 instead of 0.02, the ruling would narrow further, not widen.

### 5d. Provenance, quality, exclusions and leakage controls

**Provenance.** Each frame carries a `config_hash` over the scene arrangement, asset hashes, lighting parameters, camera track and renderer build. The manifest is written once per dataset version and never edited. The audit's frame list is an artefact with its own hash, so the audited sample can be shown to be the sample that was ruled on. This is what "provenance-controlled synthesis" means concretely: not that the data is real, but that any frame can be traced to the exact configuration that produced it, and any label to the exact scene-graph state.

**Quality signals recorded but not used as gates.** Per-cell depth coverage; per-frame texture-warning count; per-frame render duration; per-scene asset variant list; per-cell projected area. These are recorded because they let a later reviewer ask "was the audit sample unusual?" without re-rendering anything.

**Exclusions.** Zero frames excluded. The 84 frames with texture warnings are retained and counted, because excluding them would remove exactly the frames most likely to carry photometric artefacts, which is the failure this evaluation is trying to see. Retaining them inflates measured error slightly relative to a clean set; that direction is conservative for a dataset-adequacy ruling.

**Leakage controls.** Three distinct leakage channels are addressed, and they are worth separating because they are usually conflated:

1. *Scene leakage* — the same arrangement in development and evaluation. Blocked by scene-level splitting.
2. *Pose leakage* — adjacent camera poses within one scene appearing on both sides. Blocked by the same split, since poses live inside scenes.
3. *Generator leakage* — the evaluation frames and the candidate's training frames coming from the same generator family, so the candidate's advantage partly measures generator familiarity. **Not blocked.** This is structural: with only one available generator, there is no disjoint-source evaluation. The ruling must therefore treat generator familiarity as an unmeasured confound, and A-02 records it as such.

The third channel is the one that makes the case interesting. It cannot be fixed by better splitting, because there is nothing to split to. It can only be bounded, disclosed, and made a reason to cap the claim.

**Unavailable data.** No real-domain frames. No site camera calibration. No record of how the incumbent was trained beyond its artefact. Each absence is stated beside the conclusion it weakens rather than collected into a generic limitations paragraph.

### 5e. Reproducibility and method-selection ledger

| Artifact / source ID | Location | Version / date | Purpose | Accessibility | Reproduction limitation |
|---|---|---|---|---|---|
| `SRC-01` | Authoring standard for this course | 2026-11-06 | Evidence vocabulary and case structure | Public in repository | Editorial standard, not a technical reference |
| `SRC-02` | Case-study master template | v1.0, 2026-11-06 | Section structure, interview pack format | Public in repository | Template requirements are editorial |
| `SRC-03` | Design-effect and effective-sample-size result (textbook treatment of clustered sampling) | OPEN — not verified for this file | Formula F6-04 | Not verified | Citation is a verification target, not a claim |
| `SRC-04` | Misclassification attenuation result for binary labels | OPEN — not verified for this file | Formula F6-02 | Not verified | Standard result quoted from memory; exact conditions unconfirmed |
| `SRC-05` | Two-sample power formula with normal approximation | OPEN — not verified for this file | Formula F6-05 | Not verified | Normal approximation only; exact test differs slightly |
| `SRC-06` | Leakage and inconsistent-preprocessing guidance | Public method reference | v-current at authoring | Illustrates split discipline | Generic guidance; not evidence about this dataset |
| `RUN-01` | Label audit of 240 frames, 17,280 cells | ILLUSTRATIVE | Produced α, β, unknown rate | Constructed | Not executed; counts are teaching values |
| `RUN-02` | Two-model scoring pass on 42,000 frames | ILLUSTRATIVE | Produced per-cell and per-scene accuracy | Constructed | Not executed; model behaviour is stipulated |
| `RUN-03` | Cluster bootstrap over 24 scenes, 2,000 resamples | ILLUSTRATIVE | Interval on the paired difference | Constructed | Seed and resample count illustrative |
| `RUN-04` | Weighting recomputation under site mix | DERIVED | Level and contrast under 75/25 mix | Arithmetic checked | Depends on A-04 mix figure |

**What is reproducible here:** the arithmetic. Every formula in Section 6 can be recomputed by hand from the inputs listed, and the independent checks in 6.7 confirm the counts reconcile. **What is not reproducible:** the α and β values, the per-scene accuracies, the render timings and the prices. They are scenario inputs. A reader who wants to reuse the method must substitute their own audit; a reader who reuses the numbers has misunderstood the file.

**Determinism.** A-07 records that renderer determinism is unverified. If the renderer is not deterministic given seed and config, then "the frozen frame set" is a claim about a stored archive rather than about a configuration, and every reproducibility statement must be reworded to point at the archive hash instead of the config hash. That distinction is easy to lose and expensive to discover late.

**Takeaway:** the ledger's most important entry is `SRC-03` through `SRC-05` marked OPEN. The formulas are standard, but this file has not verified them against a primary text, and pretending otherwise would be exactly the kind of unfounded precision the case is about.

<!--PART2-->

## 6. Analysis Walkthrough

### 6.1 The decisive result, stated first

The evaluation as originally designed produces the wrong ranking. Scored on the frozen archive and pooled over all cells irrespective of stratum, the incumbent reports 0.7546 per-cell accuracy and the candidate 0.7714 — a candidate lead of 1.68 percentage points. That number is the one the baseline practice would promote on, and it is wrong twice over:

- **Wrong direction of composition.** The archive is half dense and half sparse by construction, but the site is three-quarters dense. Dense frames are harder, so the site's true population is harder than the archive's. Re-weighting to the site mix moves the observed gap from 1.68 points in the candidate's favour to 1.24 points in the *incumbent's* favour — a 2.92-point swing produced entirely by a mix difference nobody recorded.
- **Wrong direction of label attenuation.** The two strata have very different label error rates, and the correction slope `1 − 2τ` differs between them. Applying the correction reverses the sign again, putting the candidate 2.13 points ahead.

So the naive answer, the composition-aware answer, and the label-corrected answer are three different decisions. Two of them are wrong, and both errors are individually larger than the effect being measured. That is the whole case in one sentence: **when measurement error exceeds the effect, the measured ranking is a statement about the instrument, not about the models.**

### 6.2 Step-by-step trace

| Step | Hypothesis or decision | Input snapshot | Method / parameters | Intermediate output | Interpretation | Evidence / status |
|---|---|---|---|---|---|---|
| 1 | The archive's label error is small enough to ignore | 240 frames, stratified | Adjudicate every cell; record truth, unknown | α, β per stratum | No: τ = 0.122 dense, 0.030 sparse | ILLUSTRATIVE (`RUN-01`) |
| 2 | Aggregate accuracy is a valid comparator | Per-cell predictions | Pool 1,512,000 cells per model | Q_A = 0.7546, Q_B = 0.7714 | Candidate +1.68 pp | ILLUSTRATIVE (`RUN-02`) |
| 3 | Cell counts approximate independent evidence | Same | Design effect with ρ = 0.08, m = 720 | deff = 58.52, n_eff = 27,614 for 1.6M cells | Cell counts are ~58× inflated | DERIVED (F8-04) |
| 4 | The archive mix matches the site | Site census 75/25 | Re-weight per stratum | Gap flips to incumbent +1.24 pp | Composition alone reverses the sign | DERIVED (A-04) |
| 5 | Correcting for label error preserves the ranking | α, β per stratum | Apply `r = (q − τ)/(1 − 2τ)` per stratum, then weight | Candidate +2.13 pp | Attenuation reverses the sign back | DERIVED (F6-02) |
| 6 | The reversal is a real effect, not arithmetic | Bootstrap over scenes | Scene-level resample, 2,000 draws | 95% interval on naive gap: ±8.32 pp | Interval spans both signs | DERIVED (F6-05) |
| 7 | The design could detect the corrected effect | 8 eval scenes, σ = 0.12 | Two-sided normal power at α = 0.05, power 0.80 | Power 0.08 at 2.13 pp; MDD 11.89 pp | Design cannot resolve the effect | DERIVED (F6-06) |
| 8 | Is the audit itself precise enough? | 17,280 cells, 24 scenes | Design effect on the audit; interval on τ̂ | τ̂ = 0.0760, 95% CI [0.0458, 0.1062] | Audit is adequate; evaluation is not | DERIVED (F6-04) |
| 9 | Ruling | All of the above | Map to the four permitted uses | Ranking licensed with correction and power caveat; absolute claim refused | — | Ruling in §7 |

The ordering matters. Steps 1 and 2 precede the ruling on purpose: the audit is executed *before* the models are scored, so the error model cannot be tuned to produce a preferred ranking. Had the audit run after the comparison, an analyst under pressure could select a sampling frame that happened to understate τ in the stratum where the preferred model was weak. The sequencing is the main methodological defence in this case, and it is cheap — it costs nothing but discipline.

### 6.3 The audit: what the labels actually assert

The audit drew 240 frames from all 24 scenes, stratified 12 frames per scene and balanced across the archive's density strata, giving 10 frames from each of 12 dense scenes and 10 frames from each of 12 sparse scenes. Every one of the 72 cells in every sampled frame was adjudicated: 240 × 72 = 17,280 cell decisions.

Stratum definitions, with their measured confusion against adjudicated truth:

| Stratum | Archive frames | Sampled frames | Sampled cells | Positive rate π | False-negative rate β | False-positive rate α | τ = πβ + (1−π)α |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense (≥ 60% of bay occupied) | 21,000 | 120 | 8,640 | 0.850 | 0.140 | 0.020 | 0.1220 |
| Sparse (< 60% occupied) | 21,000 | 120 | 8,640 | 0.125 | 0.100 | 0.020 | 0.0300 |
| **Pooled** | 42,000 | 240 | 17,280 | 0.4875 | — | — | 0.0760 |

The pooled τ is *not* the average of 0.122 and 0.030. It is the observed discordance rate:

```text
FORMULA [F6-01]: observed_disagreement = (FP + FN) / N
WHERE: FP false positives [cells]; FN false negatives [cells];
       N adjudicated cells [cells].
CONSTRAINT: adjudication is complete for the sample; "unknown" cells are
            excluded from N and their rate reported separately (here 2.4%).
NUMERICAL EXAMPLE: FP = 0.020*8640*0.15 + 0.020*8640*0.875 = 25.92 + 151.20 = 177.12
                   FN = 0.140*8640*0.850 + 0.100*8640*0.125 = 1028.16 + 108.00 = 1136.16
                   (FP+FN)/17280 = 1313.28/17280 = 0.0760.
INTERPRETATION: 7.60% of cells carry a label that disagrees with adjudicated truth.
                The 0.85-weighted average of 0.122 and 0.030 would have given 0.076,
                which coincides here only because the strata contribute equal cells.
```

The coincidence in the last line is instructive and dangerous. Because both strata contain exactly 8,640 sampled cells, the cell-weighted average of τ equals the pooled rate. If the audit had sampled frames proportional to stratum size the identity would still hold, but if it had sampled proportionally to *positives* — a natural instinct when the positive class is the interesting one — the average would have been pulled upward and every subsequent correction biased. Reporting the pooled rate from raw confusion counts, rather than averaging stratum rates, removes the failure mode.

Two audit findings deserve separate emphasis because they change the ruling more than the headline τ does.

**Finding A: false negatives dominate, 6.4 to 1.** Of 1,313.28 discordant cells, 1,136.16 are false negatives and 177.12 are false positives. The asymmetry is a direct consequence of the label semantics gap recorded in 5a: the generator labels *asset presence in the scene graph*, and an asset that is present but fully occluded by a nearer pallet is still labelled occupied. The image shows nothing. A model that predicts what it sees is penalised. This is not noise; it is a systematic, directional label defect, and it means the dataset systematically rewards models that hallucinate occupancy from context.

**Finding B: the "unknown" rate is 2.4 percent and is concentrated in sparse frames.** Analysts declined to adjudicate 414 cells, mostly where a pallet edge fell across a cell boundary and the 50-percent-coverage rule was genuinely ambiguous. Excluding them from N is correct for estimating α and β on decidable cells but means the error model does not describe exactly the region where the label rule is weakest. The ruling must therefore not treat τ as a hard bound; it is a point estimate over the decidable population.

### 6.4 Applying the error model

The correction follows from the definition of what a synthetic label is. If truth is `T ∈ {0,1}` and the label is `L ∈ {0,1}`, with `P(L=1|T=0) = α` and `P(L=0|T=1) = β`, then a model's measured accuracy `q = P(Ŷ = L)` and its true accuracy `r = P(Ŷ = T)` are related by:

```text
FORMULA [F6-02]: r = (q - tau) / (1 - 2*tau),  tau = pi*beta + (1-pi)*alpha
WHERE: r true accuracy [-]; q measured accuracy against synthetic labels [-];
       tau label discordance rate [-]; pi positive base rate [-];
       alpha false-positive label rate [-]; beta false-negative label rate [-].
CONSTRAINT: label error independent of model error, and tau < 0.5.
            If 1 - 2*tau <= 0 the correction is undefined: labels carry no
            usable signal and no ranking can be recovered by arithmetic.
NUMERICAL EXAMPLE (dense stratum, incumbent model A):
  q = 0.8024, tau = 0.1220
  r = (0.8024 - 0.1220)/(1 - 0.2440) = 0.6804/0.7560 = 0.9000
INTERPRETATION: 90.0% true accuracy from an 80.2% measured accuracy. The
                measured number understates skill because the labels are wrong.
```

The dense-stratum correction lifts the incumbent from 0.8024 to 0.9000. That direction — measured below true — is typical when a model is *better* than its labels, which is exactly the situation for a model trained on real frames (the incumbent) evaluated on synthetic ones. The slope amplifies: a 1-point change in measured accuracy becomes a 1/0.756 = 1.323-point change in corrected accuracy.

Applying F6-02 to all four cells of the two-by-two:

| Stratum | Model | Measured q | τ | Slope 1−2τ | Corrected r | Lift |
|---|---|---:|---:|---:|---:|---:|
| Dense | Incumbent A | 0.8024 | 0.1220 | 0.7560 | 0.9000 | +9.76 pp |
| Dense | Candidate B | 0.7608 | 0.1220 | 0.7560 | 0.8450 | +8.42 pp |
| Sparse | Incumbent A | 0.7068 | 0.0300 | 0.9400 | 0.7200 | +1.32 pp |
| Sparse | Candidate B | 0.7820 | 0.0300 | 0.9400 | 0.8000 | +1.80 pp |

The lift is unequal. Dense frames gain 8.4 to 9.8 points of corrected accuracy; sparse frames gain 1.3 to 1.8. Because the site is mostly dense, the correction must be applied *per stratum and then weighted*, not applied once to a pooled number. Applying a single pooled τ = 0.076 to the pooled measurement would give `(0.7714 − 0.076)/(1 − 0.152) = 0.8200` for the candidate and `(0.7546 − 0.076)/0.848 = 0.8002` for the incumbent — a 1.98-point candidate lead that is *also wrong*, because the pooled slope (0.848) is not the slope that applies to either stratum (0.756 or 0.940). The correct answer is 2.13 points; the pooled-slope answer is 1.98; the naive answer is −1.68. All three are defensible-looking arithmetic on the same data.

### 6.5 Composition: the site is not the archive

The archive is split half dense, half sparse. The operations representative's bay census says 75 percent of the site's faces are dense. Weighting the measured accuracies to that mix:

```text
FORMULA [F6-03]: q_w = sum_k w_k * q_k,  r_w = sum_k w_k * r_k
WHERE: w_k population share of stratum k [-], sum w_k = 1;
       q_k measured accuracy in stratum k; r_k corrected accuracy.
CONSTRAINT: stratum definitions are identical in archive and target, and the
             site mix is measured, not assumed. Different strata in the two
             populations make the weights meaningless.
NUMERICAL EXAMPLE (site mix w_dense = 0.75):
  q_w(A) = 0.75*0.8024 + 0.25*0.7068 = 0.6018 + 0.1767 = 0.7785
  q_w(B) = 0.75*0.7608 + 0.25*0.7820 = 0.5706 + 0.1955 = 0.7661
  gap = 0.7785 - 0.7661 = 0.0124 = 1.24 pp in the INCUMBENT's favour.
INTERPRETATION: the naive comparison had the candidate ahead by 1.68 pp; the
                same data, correctly weighted, has the incumbent ahead by 1.24 pp.
```

The swing is 2.92 points and it comes from a single number — the site's dense share — that was nowhere in the evaluation plan. This is the composition failure in its purest form: not a modelling error, not a coding error, but evaluating on a population nobody checked against the deployment population.

The composition effect is easy to get directionally wrong. The candidate is *better* on sparse frames (0.7820 vs 0.7068) and *worse* on dense ones (0.7608 vs 0.8024). Its advantage lives in the minority stratum. Any shift of weight toward dense frames erodes it. At an even 50/50 split the candidate still leads by 1.68 points; at 60/40 in favour of dense, the lead is 0.24 points; at 75/25 the incumbent leads by 1.24. The break-even dense weight solves `w(0.7608 − 0.8024) + (1−w)(0.7820 − 0.7068) = 0`:

```text
w*(-0.0416) + (1-w)(0.0752) = 0
0.0752 - 0.1168 w = 0  ->  w* = 0.0752/0.1168 = 0.6438
```

At 64.4 percent dense or above, the incumbent wins the naive comparison. The site is at 75 percent. This is a 10.6-point margin beyond break-even, which is comfortably inside the range of plausible variation between warehouse areas — and a reason to distrust any ranking that does not publish the weight it used.

### 6.6 Composition and attenuation interact

The two corrections are not independent, and treating them as separate adjustments to a single gap gives nearly the right answer by accident. The correct decomposition runs per stratum:

| Quantity | Dense (A, B) | Sparse (A, B) | Unweighted mean (A, B) | Site-weighted (A, B) |
|---|---|---|---|---|
| Measured q | 0.8024, 0.7608 | 0.7068, 0.7820 | 0.7546, 0.7714 | 0.7785, 0.7661 |
| Corrected r | 0.9000, 0.8450 | 0.7200, 0.8000 | 0.8100, 0.8225 | 0.8550, 0.8337 |
| Gap A − B | +4.16 pp | −7.52 pp | −1.25 pp | **+2.13 pp** |

Walking from the naive number to the corrected one:

- **Naive** unweighted, uncorrected: B leads by 1.68 points (A−B = −1.681).
- **Composition** moves the unweighted gap of −1.681 (A−B) to a site-weighted gap of +1.2385. Contribution: 2.920 points toward A.
- **Attenuation** moves the site-weighted measured gap of +1.2385 to a site-weighted corrected gap of +2.1255. Contribution: 0.887 points toward A.
- **Total** swing from naive to corrected: 3.807 points, taking the verdict from "B by 1.68" to "A by 2.13".

The lesson is that the composition correction is the *larger* of the two here (2.920 versus 0.887 points), even though the label error dominates every absolute statement. A team that audited labels carefully but skipped the composition check would still promote the wrong model. A team that did both but applied a pooled slope would report 1.98 points instead of 2.13 — right direction, wrong magnitude, and no way to know which without the per-stratum table.

### 6.7 Independent checks on the arithmetic

Five checks, chosen to catch the errors that actually occur in this kind of calculation:

1. **Slope inversion.** `r = (q − τ)/(1 − 2τ)` must invert to `q = τ + r(1 − 2τ)`. For dense A: `0.122 + 0.9 × 0.756 = 0.122 + 0.6804 = 0.8024`. ✓
2. **Extreme-case sanity.** If `τ = 0`, the formula returns `r = q`, as it must. If `τ = 0.5`, the denominator is zero and the correction is undefined, as it must be — labels at 50 percent error carry no information. If `τ > 0.5`, the formula returns a value with the wrong sign and must be refused, not computed.
3. **Weight reconciliation.** `0.75 × 0.9000 + 0.25 × 0.7200 = 0.6750 + 0.1800 = 0.8550` for A, and `0.75 × 0.8450 + 0.25 × 0.8000 = 0.6338 + 0.2000 = 0.8338` for B. Both match the table to four decimals (B's 0.8338 is a rounding artefact against 0.8337 at higher precision).
4. **Stratum cell-count reconciliation.** 12 scenes per stratum × 10 frames × 72 cells = 8,640 cells per stratum, and 2 × 8,640 = 17,280 = 240 × 72. ✓
5. **Gap decomposition.** In the A−B direction: naive = −1.681, site-weighted measured = +1.2385, site-weighted corrected = +2.1255. Composition contributes +2.920 (from −1.681 to +1.2385) and attenuation contributes +0.887 (to +2.1255); their sum is 3.807, which equals the total naive-to-corrected swing of 2.1255 − (−1.681) = 3.8065 within 0.0005 of rounding. ✓

Check 5 is written out at this length because an earlier draft of this walkthrough carried 2.49 for the composition contribution — a figure obtained by differencing against a stale intermediate — and that draft reconciled only when read casually. The decomposition above is the version that closes to four decimals. It is included as a live example of the error class this case is about.

### 6.8 How much information the evaluation actually carries

Cells inside a frame are spatially adjacent and usually describe the same object. Frames inside a scene share the asset arrangement, the lighting rig and a continuous camera track. Both levels of correlation mean the raw cell count massively overstates evidence.

```text
FORMULA [F6-04]: deff = 1 + (m - 1) * rho,  n_eff = N / deff
WHERE: deff design effect [-]; m observations per cluster [-];
       rho intraclass correlation [-]; N total observations;
       n_eff effective sample size [observations].
CONSTRAINT: equal cluster sizes, or use the unequal-size form with mean m;
            rho estimated, not assumed. rho >= 0 means n_eff <= N.
NUMERICAL EXAMPLE (evaluation set, 8 scenes):
  m = 1750 frames/scene, cells per frame 72 -> cluster size in cells
    m_cells = 1750 * 72 = 126,000
  deff = 1 + (126000 - 1) * 0.08 = 10,080.9
  N = 8 * 126,000 = 1,008,000 cells
  n_eff = 1,008,000 / 10,080.9 = 100.0
INTERPRETATION: one million labelled cells carry the information of about
                one hundred independent observations.
```

That result is worth pausing on. The dataset is described as 42,000 frames and three million cells; as *evidence about a population of warehouse scenes* it carries roughly eight independent units, because there are eight evaluation scenes and the intra-scene correlation is high. The `n_eff = 100` figure is the best case, computed with a nominal 0.08 correlation; at ρ = 0.20 it falls to about 40, and at ρ = 0.02 it rises to about 400. Every one of those numbers is small.

The audit's precision is a different matter, and this is the subtle part. The audit sampled 240 frames across all 24 scenes, so its cluster size is 720 cells and:

```text
deff_audit = 1 + (720 - 1) * 0.08 = 58.52
n_eff_audit = 17,280 / 58.52 = 295.3
se(tau) = sqrt(0.0760 * 0.9240 / 295.3) = sqrt(0.0002379) = 0.01542
95% CI = 0.0760 +/- 1.96 * 0.01542 = [0.0458, 0.1062]
```

The audit is *adequate*: its interval is narrow enough that the ruling's central conclusion — that τ is far too large to license an absolute accuracy claim — holds across the whole interval. Even at the lower bound τ = 0.0458, the triangle bound is ±4.58 points, still larger than the 2.13-point contrast.

**Audit breadth matters more than audit depth.** Holding 28.8 analyst-hours fixed, spreading the same 240 frames over more scenes raises the effective sample sharply, because cluster size shrinks:

| Scenes sampled | Cells per cluster | deff | n_eff | 95% CI half-width on τ | Analyst-hours per effective observation |
|---:|---:|---:|---:|---:|---:|
| 24 (actual) | 720.0 | 58.52 | 295.3 | ±0.0302 | 0.0975 |
| 48 | 360.0 | 29.72 | 581.4 | ±0.0215 | 0.0495 |
| 120 | 144.0 | 12.44 | 1389.1 | ±0.0139 | 0.0207 |
| 267 | 64.7 | 6.10 | 2833.9 | ±0.0098 | 0.0102 |

The same money buys 9.6 times the information at 267 scenes as at 24. This is a free improvement the team did not take, and it appears in the ruling as a recommendation for the next dataset version rather than a criticism of this one — the audit protocol was fixed before this analysis ran, and changing it mid-evaluation would have invalidated the ruling.

### 6.9 Power: could the design have resolved the effect?

The corrected contrast is 2.13 points. Whether that contrast is *detectable* is a separate question from whether it is *estimated*.

```text
FORMULA [F6-06]: n_per_arm = 2 * (z_{1-alpha/2} + z_{1-beta})^2 * sigma^2 / delta^2
                 MDD = (z_{1-alpha/2} + z_{1-beta}) * sigma * sqrt(2/n)
WHERE: n scenes per arm; sigma per-scene standard deviation of the paired
       accuracy difference [-]; delta target difference [-];
       alpha two-sided 0.05; power 0.80.
CONSTRAINT: normal approximation, independent scene-level units, sigma known
            or well estimated. Cluster-level pairing, not cell-level.
NUMERICAL EXAMPLE (paired scenes, n = 8 per arm, sigma = 0.12, delta = 0.02125):
  z_{0.975} = 1.9600, z_{0.80} = 0.8416, sum = 2.8016
  n = 2 * 2.8016^2 * 0.0144 / 0.00045156 = 2 * 7.849 * 31.90 = 501 scenes
  MDD at n = 8: 2.8016 * 0.12 * sqrt(2/8) = 2.8016 * 0.12 * 0.5 = 0.1681
INTERPRETATION: 8 evaluation scenes cannot resolve anything smaller than 16.8 pp.
                Resolving 2.13 pp needs about 251 scenes per arm, ~502 total.
```

The design is not marginal; it is two orders of magnitude short. Power at the actual effect is 0.08, which is barely above the 0.05 false-positive rate — the test would fire slightly more often when there is no effect than when there is. The minimum detectable difference at 80 percent power exceeds the effect by a factor of 7.9.

This does not make the ranking useless. It makes the *interval* the deliverable and the point estimate a direction. And it means that the honest report of the model comparison is not "the candidate is better" but "the corrected point estimate favours the candidate by 2.13 points, and this design cannot distinguish that from zero." A field-trial slot should not be spent on that. It should be spent after either the effect is demonstrated to be larger or the design's cluster count is raised.

Note the sensitivity of MDD to σ. A-05 assumes σ = 0.12, which is a guess. At σ = 0.07, MDD at 24 scenes falls to 4.0 points and power at a 2.13-point effect rises to 0.318 — better, still inadequate. At σ = 0.07 and n = 251, power reaches 0.998. So the σ estimate is worth measuring early, because it changes the required cluster count by a factor of three; it does not change the verdict at the current cluster count.

### 6.10 What can be corrected, and what cannot

The corrected 2.13-point contrast rests on the assumption A-02: label error independent of model error. That assumption is false in a specific, knowable direction. The candidate was trained on frames from the same generator family as the evaluation archive; the incumbent was trained on real frames. The generator's systematic false negatives — occupied cells that look empty — are a pattern the candidate has had 42,000 opportunities to learn as "predict empty here". The candidate therefore reproduces the label defect more faithfully than the incumbent does, and its measured accuracy is inflated relative to its true accuracy by an amount the correction cannot recover, because the correction assumes independence.

The consequence is a one-sided statement. The 2.13-point corrected lead is an **upper bound** on the candidate's true advantage, and the true value may be zero or negative. This is the sentence that decides the case, and no amount of additional arithmetic on the existing archive can improve it:

```text
FORMULA [F6-07]: q_measured = tau + r_true*(1 - 2*tau) + c
WHERE: c >= 0 is the generator-familiarity advantage [fraction of cells],
       arising when the evaluated model learned the generator's label defects.
CONSTRAINT: c >= 0 only. If the model were trained on a disjoint generator,
            c would be 0 and F6-02 would hold exactly.
NUMERICAL EXAMPLE: at the site mix, the corrected gap of 2.13 pp corresponds to
                   c = 0. If the candidate gains only a 1.0-point familiarity
                   advantage on dense frames (weight 0.75), then
                   gap_true = 2.13 - 0.75*1.00 = 1.38 pp.
                   At c_dense = 2.84 pp, gap_true = 0.00.
INTERPRETATION: the ranking survives only if the familiarity advantage stays
                below 2.84 points on dense frames. That is unmeasured.
```

`c` is not estimable from this archive. It needs either a second generator (not available), a real-domain sample (not available), or a training-ablation experiment in which the candidate is retrained on a disjoint renderer and scored again. That is the cheapest sufficient upgrade, and it is what the ruling requests.

### 6.11 Sensitivity summary

| Stress | Original condition | Alternative | Change in conclusion | Uncertainty | Decision consequence |
|---|---|---|---|---|---|
| Site mix | 75/25 dense/sparse | 50/50 | Corrected gap falls from +2.13 to +1.25 pp | Census unverified (A-04) | Ranking survives; magnitude halves |
| Site mix | 75/25 | 40/60 | Corrected gap flips to −1.13 pp | Site mix outside plausible range | Would reverse the ruling |
| ρ for evaluation clusters | 0.08 | 0.02 | n_eff rises 100 → 400; power still ≈ 0.08 at 8 scenes | Unverified (A-03) | No change; cluster count binds |
| σ paired difference | 0.12 | 0.07 | Power at 24 scenes rises 0.14 → 0.32; MDD falls to 4.0 pp | Unverified (A-05) | Still inadequate; expansion needed |
| Audit τ | 0.0760 | 0.1062 (upper CI) | Triangle bound ±10.6 pp; absolute claim even worse | Sampling error | Direction of ruling unchanged |
| Audit τ | 0.0760 | 0.0458 (lower CI) | Triangle bound ±4.6 pp, still > 2.13 pp contrast | Sampling error | Direction of ruling unchanged |
| Pooled vs per-stratum slope | Per-stratum | Pooled 0.848 | Corrected gap 2.13 → 1.98 pp | Method choice | Sign survives, magnitude wrong by 7% |
| Familiarity advantage c | 0 | 1.0 pp dense | Corrected gap 2.13 → 1.38 pp | Unmeasured | Survives |
| Familiarity advantage c | 0 | 2.84 pp dense | Corrected gap 2.13 → 0.00 pp | Unmeasured | Reverses; break-even point |
| Audit breadth | 24 scenes | 267 scenes | CI on τ narrows 0.030 → 0.010 | Deterministic given ρ | Free improvement for next version |

The two rows that change the *decision* are the site mix at 40 percent dense and the familiarity advantage above 2.84 points. Both are outside the measured evidence and neither is resolvable with the current archive. Every other row changes a number but not the ruling, which is the property that makes the ruling defensible: it is robust to the perturbations that are actually plausible.

### 6.12 What was not tried

Two approaches were considered and not executed, and the distinction matters for honesty:

- **A second renderer.** A different procedural engine would break generator familiarity directly. It was not executed because asset licensing for a second engine could not be completed inside the evaluation window. This is a *constraint*, not a test result, and the ruling says so.
- **Adversarial probe frames.** Rendering a small set of deliberately pathological bays — wrapped pallets with reflective film, pallets taller than the cell grid, empty bays with a single stray pallet — would test whether the models generalise outside the arrangement distribution. This is a *recommendation* in the ruling, with an estimated cost of 40 frames at 0.42 s = 16.8 s of render time plus about 0.5 analyst-hours to inspect. It was not executed because the frame set was frozen before the idea was raised.


## 7. The Finding

**The dataset licenses a corrected, direction-only ranking, and nothing stronger.** The frozen archive `syn-wh-1.0` may support the statement that, after per-stratum label correction and re-weighting to the site's 75/25 dense-to-sparse mix, the incumbent leads the candidate by 2.13 percentage points — but only as an upper bound on the incumbent's advantage, because the candidate's generator familiarity inflates its measured accuracy by an unmeasured amount that reaches break-even at 2.84 points on dense frames. The dataset may not support any absolute accuracy figure: the assumption-free bound is ±9.90 points at the site mix, and even the audit's own interval leaves τ between 0.0458 and 0.1062, so the bound is at best ±4.6 points. The evaluation split of 8 scenes has 8 percent power to detect the 2.13-point effect and cannot resolve anything below 11.9 points, so the 2.13-point figure is a point estimate whose interval spans zero and must be reported as such.

Three separate numbers, on the same data, give three different verdicts: the naive pooled comparison says the candidate wins by 1.68 points, the composition-corrected comparison says the incumbent wins by 1.24, and the fully corrected comparison says the incumbent wins by 2.13. Only the last is defensible, and only as a direction with a stated ceiling. The two errors that separate the numbers — a 2.92-point composition swing and a 0.89-point attenuation swing — are individually larger than or comparable to the effect, which is the mechanism behind the finding rather than a footnote to it.

The main boundary is generator familiarity, and it cannot be resolved within this archive: `c ≥ 0` is unmeasurable without a second renderer or a labelled real-domain sample, and both are unavailable under C-02 and C-06. The next justified action is not to run more statistics on the same frames. It is to either (a) retrain the candidate on a disjoint renderer and re-score on the same frozen archive, which isolates `c` for the cost of one training run, or (b) expand the evaluation set from 8 to at least 251 scenes, which costs about 51 render-hours and 180 GiB, and re-run the comparison with a pre-registered 2.13-point target. Option (a) is cheaper and addresses the assumption that actually threatens the conclusion. Option (b) addresses precision but not bias, and running it first would spend the budget to produce a tighter estimate of a quantity still contaminated by `c`.

**Evidence status of the finding:** illustrative. The reasoning chain, every formula, and every arithmetic step are reproducible from Section 6; the inputs they consume are constructed teaching values, not measurements of any dataset, renderer, model, or warehouse.

## 8. System/Solution Design

### 8.1 What is being designed

The design object is not a model and not a renderer. It is the **dataset release gate**: the set of artefacts, checks and permissions that a dataset version must carry before any downstream document may cite it. The gate has an offline path (where a dataset version is produced, audited and ruled on) and an online path (where a downstream consumer — training run, evaluation run, or a slide in a review deck — asks whether a given claim is licensed by a given version).

Treating the gate as a system is what makes the ruling durable. A memo that says "use with caution" is forgotten in a week; a gate that refuses to emit a claim identifier without a matching permission record survives staff turnover.

### 8.2 Offline path — dataset production, audit, ruling

```text
[Scene library + asset set]
        |
        v
[Scene sampler] --(seed, mix)--> [Renderer] --> RGB + depth + scene graph
        |                                            |
        |                                            v
        |                                   [Label rasteriser] --> per-cell labels
        |                                            |
        v                                            v
[Manifest writer] <---- config_hash, asset_hash, renderer_build, seed
        |
        v
[Frozen archive syn-wh-1.0]  (immutable, hash-pinned)
        |
        +--> [Audit sampler: stratified by scene and density]
        |          |
        |          v
        |    [Adjudication tool] --> truth / unknown per cell
        |          |
        |          v
        |    [Error model: alpha, beta, tau per stratum + intervals]
        |          |
        |          v
        +--> [Evaluation harness: model A, model B on identical frames]
                   |
                   v
             [Analysis: deff, weighting, attenuation, power]
                   |
                   v
             [Adequacy ruling] --permission record--> [Claim registry]
                                                          |
                                                          v
                                              [Downstream documents cite
                                               claim_id, not raw numbers]
```

Two properties of this diagram carry most of the value. First, the audit sampler draws from the *frozen archive* rather than from the renderer, so the audited frames are byte-identical to the evaluated frames; an audit of a re-render would tell you about a different population. Second, the error model and the evaluation both feed the analysis stage but neither feeds the other — the audit cannot see model outputs, which is what keeps C-04 satisfiable in practice rather than only on paper.

### 8.3 Online path — a consumer asking whether a claim is licensed

```text
[Consumer: training job / eval job / slide / procurement doc]
        |
        v
[Claim request: dataset_version, metric, population, comparator]
        |
        v
[Permission lookup in claim registry]
        |
        +-- no record ---------> [REFUSE: emit "unlicensed claim" error]
        |                              |
        |                              v
        |                        [Log: consumer, claim, version, timestamp]
        |
        +-- record found ------> [Check metric family matches]
                                       |
                                       +-- mismatch --> [REFUSE + point to
                                       |                 licensed metric]
                                       |
                                       +-- match -----> [Check population
                                                         weights declared]
                                                              |
                                                              +-- absent -->
                                                              |   [REFUSE]
                                                              |
                                                              +-- present ->
                                                                  [Return
                                                                   permission:
                                                                   permitted
                                                                   statement
                                                                   + mandatory
                                                                   correction
                                                                   + ceiling]
```

The refusal path is the design's actual product. A gate that only ever says yes is a rubber stamp; the gate earns its place by being the thing that stops a slide reading "94.2 percent accuracy" from being built on a dataset whose corrected accuracy is unknown to within ten points.

### 8.4 Key design decisions

| ID | Choice | Alternatives | Evidence | Tradeoff | Failure boundary | Revisit trigger |
|---|---|---|---|---|---|---|
| `DEC-01` | Rule on intended use, not on dataset quality | Single "quality score" for the dataset | Four distinct uses require four different evidence levels | More paperwork; four permissions instead of one number | A consumer invents a fifth use | Any new downstream claim type |
| `DEC-02` | Audit labels; never rewrite them | Correct labels in place, then re-score | C-04; in-place correction makes the eval circular | Archive stays wrong; corrections live in the error model | Error model assumptions (independence) fail | Any proposal to "fix" labels |
| `DEC-03` | Cell is the analysis unit; scene is the inference unit | Cell-level inference; frame-level clustering | F6-04 shows 58× inflation at frame level | Wider intervals, less satisfying numbers | Scene boundaries are not the real cluster | Scene library restructured |
| `DEC-04` | Per-stratum correction, then weight | Pooled τ applied to pooled accuracy | Pooled slope 0.848 ≠ 0.756 or 0.940 | Requires stratum definitions to be stable | Stratum definition changes between archive and site | Site mix or density definition changes |
| `DEC-05` | Freeze the frame set before the ruling | Iterate the dataset while evaluating | A-06; mid-flight change breaks comparability | Cannot add probe frames when a gap is found | A gap requires new frames | Any manifest diff |
| `DEC-06` | Publish the ceiling, not just the estimate | Publish point estimate with a CI | F6-07: `c ≥ 0` is one-sided | Ceiling is less quotable than an estimate | A disjoint-renderer ablation measures `c` | Ablation result available |
| `DEC-07` | Refuse absolute accuracy claims entirely | Publish corrected absolute accuracy | τ ∈ [0.046, 0.106] makes the bound exceed any useful threshold | Loses a headline number the programme wanted | A real-domain labelled sample becomes available | Real sample acquired |

`DEC-07` is the decision that met the most resistance, and it is the one the evidence most clearly supports. The programme wanted a single accuracy figure to put in a promotion pack. The audit says that figure cannot be pinned to better than ±9.90 points at the site mix without stronger assumptions, and the *stronger* assumption needed to narrow it — independence of label error from model error — is precisely the one the candidate violates. So the refusal is not conservatism; it is the direct consequence of the only assumption that would make the number usable being false.

### 8.5 Operational readiness

| Area | Design answer for this gate |
|---|---|
| Interfaces | Claim request is `(dataset_version, metric_family, population_id, comparator)`. Response is a permission record or a typed refusal (`NO_RECORD`, `METRIC_MISMATCH`, `POPULATION_UNDECLARED`). |
| Data and state | Claim registry is append-only. Permissions are never edited; a superseded permission is marked with a successor pointer. The archive is hash-pinned and read-only. |
| Reliability | The gate is not on a latency path. Its SLI is the fraction of downstream documents that cite a valid `claim_id`, measured monthly over all documents produced in the period. |
| Resilience | If the registry is unavailable, the gate fails closed: no permission, no claim. This is deliberate — a permissive fallback would be used exactly when pressure is highest. |
| Recovery | Registry is reconstructible from the append-only log plus the frozen archives. A restore test replays the log and compares the resulting permission set to the current one. |
| Observability | Every refusal logged with consumer, claim, timestamp and reason code. A monthly report of refusal reasons is the gate's own feedback signal. |
| Deployment | The gate is versioned with the dataset version, not with the model. A dataset version without a gate record is treated as unaudited. |
| Security | Claim registry write access limited to the evaluation lead role; read access open. Audit trail immutable. |
| Economics | Fixed cost is the audit labour, 28.8 analyst-hours per dataset version. Marginal cost per claim lookup is nil. See Section 9. |
| Human workflow | The evaluation lead issues permissions. The review board may not issue them but may request a new use, which triggers a fresh ruling. |

The fail-closed choice deserves a note because it is the kind of decision that gets reversed under pressure. If the registry is down and a review board is waiting, the socially easy move is to let the team proceed with a verbal assurance. That move converts the gate into a formality within one incident. The design answers it structurally: the review board's own template requires a `claim_id`, so a missing registry blocks the board's document, not the gate's operator.

### 8.6 Capacity and cost sizing for the gate itself

The gate's workload is tiny and should be sized explicitly so nobody over-engineers it:

- Dataset versions per quarter: 2 (one planned, one corrective).
- Claims issued per version: 4 permitted uses × 1 population = 4.
- Claim lookups per quarter: approximately 60 (training jobs, eval jobs, slide builds, procurement questions), of which perhaps 25 are refusals.
- Registry size after 8 quarters: 2 × 8 × 4 = 64 permission records plus refusals, under 1,000 rows total.

At 1,000 rows the correct storage decision is a text file in the repository, not a database. The economics section (Section 9) prices the audit, not the registry, because the audit is 99 percent of the cost.

**Takeaway:** the gate is cheap to operate and expensive to fill. Spend the engineering attention on the audit sample design, not on the registry's technology.

## 9. Evaluation & Validation

### 9.1 What is being evaluated, and by whom

This section evaluates the *adequacy methodology*, not the models. The question it answers is: could this ruling be wrong, and what would detecting that look like? The distinction matters because a model metric cannot validate a data-adequacy claim; only a different data source or a different audit can.

| Evaluation field | Content |
|---|---|
| Primary metric | Fraction of downstream claims that are reconstructable and within their declared permission, audited monthly over all claims emitted in the period. Numerator: claims whose cited `claim_id` resolves and whose statement matches the permission text. Denominator: all claims emitted. |
| Baseline | Pre-gate practice: aggregate accuracy quoted with no correction and no permission record. Baseline reconstructability is 0 by construction, because no record exists to reconstruct from. |
| Pass/fail rule | Pass if ≥ 95 percent of emitted claims resolve to a matching permission and 0 claims quote a refused metric. Chosen before the first gate cycle; owner is the evaluation lead. |
| Guardrails | Audit labour per version ≤ 32 analyst-hours; refusal rate reported but not targeted (a low refusal rate is not evidence of quality); no claim may cite a metric family that the permission does not name. |
| Test design | Two-part. (a) Retrospective reconstruction: take 20 documents produced before the gate and attempt to reconstruct each claim; count how many can be traced to a dataset version. (b) Prospective replay: issue 30 synthetic claim requests spanning all four uses and all three refusal types, and confirm each returns the designed response. |
| Sample | 20 pre-gate documents; 30 replay requests; one dataset version. Small samples, deliberately — the gate's behaviour is deterministic, so the replay is a correctness test rather than a statistical one. |
| Uncertainty | Reconstruction is a proportion on 20 documents; a Wilson interval is appropriate and will be wide. The replay is exhaustive over designed cases, so no interval applies. |
| Validity threat | The most plausible threat is that permission text is written loosely enough that almost any claim "matches". Mitigation: each permission names a metric family, a population, a comparator and a mandatory correction string; matching requires all four. |
| Artifact | Permission records, refusal log, pre-gate document sample list, replay request set. |
| Decision | Not executed. This is a proposed protocol. |

### 9.2 Evaluating the evaluation: five falsification tests

These are the tests that would show the ruling is wrong. None was executed; each is specified so a successor can execute it.

1. **Independence falsification (tests A-02).** Score both models on audited cells only, then compare each model's error rate on audited cells against its error rate on all cells. If the candidate's error concentrates on cells the labels got wrong, and its aggregate error is lower than its audited-cell error, that is direct evidence of generator familiarity. Expected discriminating signal: `err_audited − err_overall > 0` for the candidate and ≈ 0 for the incumbent. Cost: negligible, reuses `RUN-01` and `RUN-02`. **This is the single highest-value test available and it should have been run before the ruling.**

2. **Disjoint-renderer ablation (tests F6-07).** Retrain the candidate on frames from a second renderer, score on the frozen archive. The drop in measured accuracy is an estimate of `c`. Expected signal: if the drop on dense frames is below 2.84 points, the ranking survives. Cost: one training run plus 51 render-hours for the new corpus.

3. **Second-site mix falsification (tests A-04).** Obtain a bay census from a second site with a different dense share. If that site is below 64.4 percent dense, the incumbent's advantage disappears under re-weighting, and the ruling's population scope must be narrowed to "sites above 64.4 percent dense". Cost: one operations request.

4. **ρ falsification (tests A-03).** Estimate the intraclass correlation from per-scene accuracy variance rather than assuming 0.08. If ρ is above about 0.32, the audit's own interval widens past the point where the ruling's central conclusion holds at the lower bound. Expected signal: `ρ̂ > 0.32` would require re-auditing with more scenes.

5. **Label-semantics falsification (tests the occlusion claim).** Re-adjudicate 200 cells that the audit marked as false negatives and record whether the pallet was occluded, partially visible, or absent. If most are partially visible rather than occluded, the 6.4:1 false-negative asymmetry is a rendering problem rather than a label-semantics problem, and the recommended fix changes from "fix the label rule" to "fix the renderer's occlusion handling".

### 9.3 Why the retrospective reconstruction test is not a vanity metric

A tempting alternative primary metric is "number of claims corrected after review", which would look good whenever the gate is busy. It is a vanity metric because it measures friction, not safety: a gate that refuses everything corrects every claim and scores perfectly while making the programme unable to function. The reconstructability metric is chosen because it is *satisfiable* — a well-run team reaches 95 percent — and because it fails in the direction that matters, when a claim circulates that nobody can trace.

The guardrail on refusal rate is deliberately not a target. If refusal rate were targeted downward, the evaluation lead would face pressure to widen permissions, which is the exact failure the gate exists to prevent.

### 9.4 Metric definitions and their traps

| Metric | Numerator | Denominator | Unit | Population | Window | Comparator | Trap |
|---|---|---|---|---|---|---|---|
| Per-cell accuracy | Correct cells | All adjudicated/decided cells | cells | Declared stratum | Dataset version | Other model | Base rate dominates; a model that says "occupied" everywhere scores high in dense frames |
| Corrected accuracy | From F6-02 | — | fraction | Site mix | Dataset version | Baseline | Undefined if τ ≥ 0.5; assumes independence |
| Label discordance τ | FP + FN | Adjudicated cells | fraction | Stratum | Audit date | Zero (ideal labels) | Dominated by the direction with more positives |
| Reconstructability | Traceable claims | All claims | fraction | Emitted claims | Calendar month | Pre-gate practice | Excludes claims never written down |
| Power | — | — | probability | Design | Fixed n | 0.80 convention | Sensitive to σ; a guess here moves n by 3× |

The base-rate trap in the first row is worth a worked illustration because it is the most common way a synthetic-warehouse evaluation flatters a model. In the dense stratum, 85 percent of cells are occupied. A degenerate model that asserts "occupied" on every cell scores 0.85 per-cell accuracy on true labels — better than either real model's 0.845 or 0.90 corrected dense accuracy in the candidate's case, and close to the incumbent's. On synthetic labels it scores `P(L=1) = π(1−β) + (1−π)α = 0.85 × 0.86 + 0.15 × 0.02 = 0.7340`, which is *below* both models but not by much.

This is why the ruling includes a mandatory companion metric: the degenerate baseline must be reported beside every accuracy figure. Any model whose reported accuracy is within a few points of the degenerate baseline has demonstrated nothing about perception. The check is trivial to compute from π, α and β and should appear automatically in the evaluation harness output, not as a manual afterthought.

### 9.5 Sensitivity of the ruling to its own inputs

Section 6.11 tabulates ten perturbations. The summary judgement is that the ruling is robust to everything measurable and vulnerable to exactly one thing unmeasurable (`c`). That asymmetry is the honest shape of the evidence, and it is what the ruling says: it names `c` as the boundary rather than burying it.

**Takeaway:** the gate's job is not to make claims pass. It is to make the one claim that matters — "this model is better" — survive contact with the two errors that would otherwise manufacture a different answer.

## 10. Action Taken

### 10.1 Status

**No production action taken as of 2026-11-09.** The ruling described here is a design exercise conducted inside the authoring of this course file. No dataset was audited, no model was scored, no review board convened, and no field-trial slot was booked. Everything in Sections 6 through 9 is a constructed worked example.

This section therefore records the action the ruling *would* specify and the decision it *would* force, clearly marked as proposed, so that the case's action chain is complete without inventing a deployment.

### 10.2 The recommendation as issued

| Action field | Proposed content |
|---|---|
| Recommendation | Rule `syn-wh-1.0` adequate for a corrected, direction-only ranking; inadequate for any absolute accuracy claim; adequate for pipeline smoke tests and regression checks; inadequate for a promotion decision on its own. |
| Approval / status | Proposed. Issuing authority: perception evaluation lead. Not approved; no record exists. |
| Implementer and handoff | Modelling team receives the comparison protocol (per-stratum, site-weighted, corrected, with degenerate baseline); simulation team receives the scene-expansion estimate; operations receives the site-mix verification request. |
| Effective scope | Dataset version `syn-wh-1.0`, one aisle type, one site, models A and B as frozen artefacts. |
| Release strategy | No deployment. The ruling gates a document, not a system. |
| Stop / rollback | If the claim registry is unavailable, claims are blocked, not permitted. If A-02 is falsified by test 1 of §9.2, all ranking permissions are withdrawn and reissued as "direction unknown". |
| Excluded actions | Absolute accuracy claim (DEC-07); promotion on this evidence; any claim about a second site; any recommendation about manipulator behaviour. |

### 10.3 The four permissions actually issued

The ruling is a set of four records rather than one verdict, and the granularity is the point. Each record names what it licenses, what it forbids, and what must accompany any use.

| Permission | Use licensed | Required companion text | Forbidden | Expiry |
|---|---|---|---|---|
| `PERM-01` | Pipeline smoke test: does the harness load, run and write predictions? | None | Any accuracy figure | Next dataset version |
| `PERM-02` | Regression check: did a code change alter predictions on a fixed frame set? | Diff count, not accuracy | Any claim about model quality | Next dataset version |
| `PERM-03` | Corrected direction-only ranking between A and B | Full correction chain; degenerate baseline; the 2.13-point ceiling; the power statement | Absolute accuracy; promotion decision | Falsification of A-02 or a new site mix |
| `PERM-04` | Scene-coverage description: which arrangements the archive contains | Scene count and split; asset variant list | Any statement about real-aisle coverage | Next dataset version |

`PERM-02` is the permission most teams forget to write, and its absence causes a specific harm: a regression check reports "accuracy fell from 0.7714 to 0.7690", a reader treats the change as a quality signal, and the gate's careful refusal of absolute claims is undone by a number that leaked in through a side door. Naming the permitted output as a *diff count* closes that door.

### 10.4 What was deliberately not done

Three actions were available and deliberately declined, each for a reason that is itself a lesson:

- **Not running the independence test before ruling.** This is recorded as a genuine methodological weakness, not a defensible choice. Test 1 of §9.2 costs almost nothing and would have converted `c` from unmeasured to bounded. The ruling proceeds without it because the case's purpose is to show what the ruling looks like when that test is missing — which is the common situation.
- **Not expanding the scene count before ruling.** The expansion costs 51 render-hours, which is affordable. It was declined because expanding precision does not address the bias from `c`, and spending the budget on precision first would produce a tighter estimate of a contaminated quantity. This is a defensible sequencing choice given the bias, and an indefensible one if `c` turns out to be small.
- **Not rewriting labels to fix the false negatives.** Declined under C-04. Recorded here because the temptation is strong and the argument for it sounds reasonable ("the labels are obviously wrong, just fix them"), while the consequence — an evaluation set shaped by the models' known failures — is fatal to the evaluation's purpose.

**Takeaway:** two of the three declined actions were correct and one was a genuine gap. Naming which is which is the difference between a ruling and an excuse.

## 11. Outcome & Results

### 11.1 Source declaration

**All figures in this table are constructed teaching values.** They are not measurements of any dataset, renderer, model, warehouse, or organisation. The arithmetic connecting them is independently checked in Section 6.7; the inputs are illustrative. No realised benefit, saving, or field outcome is claimed, and the "treatment" column describes what the corrected analysis *computes*, not what any intervention *achieved*.

### 11.2 The measured quantities, as computed

| Metric | Baseline: naive comparison | Corrected analysis | Absolute change | Relative change | Uncertainty | Evidence | Limitation |
|---|---|---|---|---|---|---|---|
| Reported accuracy gap (A vs B) | −1.68 pp (B leads) | +2.13 pp (A leads) | 3.81 pp swing | Sign reversal | ±8.32 pp at 8 scenes | DERIVED, `RUN-02` | Interval spans zero |
| Accuracy level, model A | 0.7546 | 0.8550 corrected | +10.04 pp | +13.3% relative | ±9.90 pp triangle bound | DERIVED | Not a field capability estimate |
| Accuracy level, model B | 0.7714 | 0.8337 corrected | +6.23 pp | +8.1% relative | ±9.90 pp triangle bound | DERIVED | Inflated by familiarity `c` |
| Label discordance τ | 0 (assumed) | 0.0760 | +7.60 pp | Undefined (baseline zero) | 95% CI [0.0458, 0.1062] | ILLUSTRATIVE, `RUN-01` | 2.4% cells undecidable |
| Effective evaluation size | 1,008,000 cells | ≈100 independent units | −99.99% | Not meaningful | ρ ∈ [0.02, 0.20] → 40 to 400 | DERIVED, F6-04 | ρ assumed |
| Design power at 2.13 pp | Not computed | 0.08 | — | — | σ assumed 0.12 | DERIVED, F6-06 | Normal approximation |
| Audit cost | 0 h | 28.8 analyst-hours | +28.8 h | Undefined | ±10% on adjudication rate | ILLUSTRATIVE | A-08 unverified |
| Render cost | — | 4.9 worker-hours, 17.2 GiB | — | — | Exact given 0.42 s/frame | ILLUSTRATIVE | Single-worker estimate |

The relative changes in rows two and three are reported because the baseline is a positive proportion, where a relative change is meaningful. Row four's baseline is zero, so no relative change exists, and the table says so rather than dividing by zero. Row five's "change" from one million cells to one hundred units is not a percentage of anything; reporting it as a 99.99 percent reduction would be arithmetically true and rhetorically misleading, so it is reported as a raw comparison of two different quantities.

### 11.3 Cost of the evaluation as executed

| Component | Quantity | Unit rate | Cost | Basis |
|---|---|---|---|---|
| Render, 42,000 frames | 42,000 × 0.42 s = 17,640 s | 1 worker | 4.90 worker-hours | ILLUSTRATIVE rate |
| Render compute charge | 4.90 h | USD 1.20/h | USD 5.88 | Illustrative price; not a market quote |
| Storage, RGB+depth | 42,000 × 0.42 MB = 17,640 MB | — | 17.23 GiB | ILLUSTRATIVE frame size |
| Label audit | 240 frames × 72 cells × 6 s = 103,680 s | 1 analyst | 28.80 analyst-hours | A-08 unverified |
| Audit labour charge | 28.8 h | USD 45/h | USD 1,296 | Illustrative internal cost rate |
| Scoring, two models | 1,008,000 cells × 2 | — | Not sized | Depends on model artefact |
| Analysis | — | — | 6 analyst-hours | Estimate |
| **Total cash-equivalent** | — | — | **≈ USD 1,572 + 34.8 labour-hours** | ILLUSTRATIVE |

The ratio that matters is the one between the two halves of the work: 4.90 render-hours against 28.80 audit-hours, a factor of 5.9. Generating the data was cheap; finding out what the data meant was expensive. That ratio is the case's economic core, and it generalises: synthetic data pipelines routinely budget for generation and treat labelling as free because the computer did it.

### 11.4 Cost of the recommended follow-ups

| Follow-up | Quantity | Unit rate | Estimated cost | Addresses |
|---|---|---|---|---|
| Independence test (§9.2 test 1) | Reuse `RUN-01`, `RUN-02` | — | ≈ 2 analyst-hours | A-02, `c` bounded |
| Disjoint-renderer ablation | New corpus + one training run | 51 h render at USD 1.20/h | ≈ USD 61 + training | F6-07, `c` estimated |
| Scene expansion to 251 eval scenes | 251 × 1,750 = 439,250 frames | 0.42 s/frame | 51.2 h ≈ USD 61; 180 GiB | Power 0.80 at 2.13 pp |
| Full expansion to 267 scenes | 467,250 frames | 0.42 s/frame | 54.5 h ≈ USD 65; 191.6 GiB | Power plus audit breadth |
| Re-audit at 267 scenes | 240 frames, same 28.8 h | 6 s/cell | 28.8 h; CI narrows to ±0.0098 | Audit precision, 9.6× efficiency |
| Site census verification | One operations request | — | ≈ 2 h | A-04 |

The expansion estimate has a property worth noting: rendering 467,250 frames costs about USD 65 in compute and 54.5 worker-hours, which is less than twice the audit labour of the original 240-frame sample. The expensive part of the recommended programme is the re-audit, not the rendering. A team that reasons "more scenes must be expensive" will under-invest in exactly the thing that is cheap.

### 11.5 Outcome not yet measured

| Metric | Status | Measurement plan | Owner | Window | Decision trigger |
|---|---|---|---|---|---|
| Whether `c` is below the 2.84-point break-even | Not measured | Disjoint-renderer ablation, §9.2 test 2 | Modelling team | Next training cycle | If `c` ≥ 2.84 pp dense, withdraw `PERM-03` |
| Field per-cell accuracy | Not measured | Needs a labelled real-domain sample (C-02 blocks) | Operations + data quality | Undetermined | Any real sample changes the ruling's scope |
| Realised missed-pick reduction | Not measured | Requires deployment, which the ruling does not license | Operations | Undetermined | Not addressable from this evidence |
| Gate reconstructability | Not measured | Retrospective reconstruction, §9.1 | Evaluation lead | First month of gate operation | Below 95% triggers a gate design review |

The third row is the one a business sponsor will ask about, and it is honest to say it is unreachable from this dataset. A synthetic archive can bound a model comparison; it cannot observe a pick. Claiming otherwise would be the exact error the case is built to prevent.

**Takeaway:** the outcome of this case is a ruling and a cost structure, not a model improvement. The ruling's value is that it converts an unanswerable question ("is this dataset good?") into four answerable ones with named prices.

