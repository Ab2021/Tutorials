# Instructor Kit

> **Purpose:** Everything an instructor needs to run the course without re-deriving the design.
>
> **Status:** Teaching kit, not a validated curriculum. External review and pilot are pending. See `GOVERNANCE_AND_EXTERNAL_REVIEW.md` and `PILOT_PROTOCOL.md`.

---

## 1. Delivery models

| Model | Duration | Weekly load | What to cut |
|---|---|---:|---|
| Full program | 40 weeks | 17–19 h | Nothing core; 14B optional |
| Intensive | 12 weeks | 40–45 h | Cut 10, 12, 14A depth; keep 00–06, 08, 14, 15–19 |
| Part-time | 48–52 weeks | 10–12 h | Spread labs; keep sequence |
| Corporate | 6–10 weeks | 8–12 h | Use one track, one real project, modules 00, 02, 03, 06, 14, 17–19 |
| MedTech short course | 8 weeks | 10 h | 00, 01A, 11, 14, 14A, 14B, 25 |

## 2. Before the first class

- [ ] Confirm the 40-week calendar and assessment dates.
- [ ] Create the LMS shell and module folders.
- [ ] Issue data-access instructions; start PhysioNet/CITI credentialing early.
- [ ] Test every lab in a clean environment.
- [ ] Prepare the synthetic data and fallback datasets.
- [ ] Brief clinical, regulatory, and hardware guest reviewers.
- [ ] Confirm the AI-assistance and data-use policies.
- [ ] Run the accessibility check on all materials.
- [ ] Assemble the instructor answer-key folder (private).
- [ ] Schedule the midterm, playbook, and capstone reviews.

## 3. Module teaching notes

| Module | Prep | Common learner difficulty | Instructor move |
|---|---|---|---|
| 00 | Industry examples | Confusing research and product | Use the charter; ask “who acts?” |
| 01 | Clinical vocabulary | Too many terms | Use one patient journey across the module |
| 01A | Units and assumptions | Math anxiety | Start with unit errors, not proofs |
| 02 | Data landscape | Treating a dataset as truth | Compare EHR vs claims on one question |
| 03 | Pipelines | Notebook habits | Enforce a clean-checkout run |
| 04 | Causal inference | Confusing prediction and causation | Draw DAGs before code |
| 05 | Study design | Design after data | Start from the claim |
| 06 | ML | AUROC obsession | Calibration and decision curve first |
| 07 | Deep learning/LLM | Hype | Require a simple baseline |
| 08 | EHR/claims | Workflow blindness | Shadow a clinic or use a workflow map |
| 09 | NLP | Mention = diagnosis | Annotation and assertion exercises |
| 10 | Imaging | Random splits | Patient/site split drill |
| 11 | Signals | Treating device output as truth | Signal-quality lab first |
| 12 | Omics | p-values only | Effect sizes, batch, validation |
| 13 | Pharma | Trial data as clean | Show screen failures and missing data |
| 14 | Regulation | Self-declaring CDS | Triage with counsel |
| 14A | Hardware | “The sensor is the truth” | Signal-chain teardown |
| 14B | Compression | Accuracy on original test only | Target-hardware HIL |
| 15 | MLOps | “Deploy = notebook” | Incident and rollback drills |
| 16 | Privacy/ethics | Generic caveats | Threat model and subgroup audit |
| 17 | Product | No buyer | Stakeholder map |
| 18 | Quality | Report after analysis | Pre-specify and model card |
| 19 | Capstone | Scope creep | Kill criteria and weekly gates |
| 20 | Career | Listing tools | Decision-first case study |
| 21 | Datasets | Download without license | Five-gate audit |
| 22 | Sources | Citation dumping | Annotated bibliography |
| 23/24 | Specialty/labs | Breadth vs depth | Choose one elective, not all |
| 25/26 | Operations/cloud | Tool tourism | One end-to-end lab |

## 4. Assessment schedule

| Week | Artifact | Feedback |
|---:|---|---|
| 4 | Project charter | Written + 10-min review |
| 8 | Case critique 1–3 | Rubric |
| 12 | Midterm evidence dossier | Panel review |
| 16 | Case critique 4–6 | Rubric |
| 20 | Track playbook draft | Peer + instructor |
| 26 | Track playbook final | External reviewer |
| 30 | Capstone charter and data plan | Gate |
| 34 | Capstone validation review | Gate |
| 38 | Capstone dossier | Reviewer panel |
| 40 | Demo and defense | Panel |

## 5. Anchor lab answer keys

These are solution outlines for instructor calibration. They are not full student solutions.

### 5.1 Unit conversion (01A.1)

Expected:

- Glucose 126 mg/dL → 126 × 0.0555 = **6.99 mmol/L** (report ~7.0).
- Creatinine 1.2 mg/dL → 1.2 × 88.4 = **106.1 µmol/L**.
- Cholesterol 200 mg/dL → 200 × 0.02586 = **5.17 mmol/L**.
- BP 120/80 → MAP = 80 + (120−80)/3 = **93.3 mmHg**.
- Weight 154 lb → 154 × 0.453592 = **69.9 kg**.
- BMI 69.9 kg / (1.75 m)² = **22.8 kg/m²**.
- BSA Mosteller = √(175 × 69.9 / 3600) = **1.84 m²**.
- NNT: control 20%, treatment 15%, ARR 5% → **NNT 20**.

Common errors: multiplying instead of dividing, using 18.0 vs 18.0182, forgetting units, applying adult ranges to children.

### 5.2 Calibration and decision curve (06.3)

Expected reasoning:

- A model with higher AUROC can have worse calibration and lower net benefit.
- Brier score decomposes into reliability/resolution/uncertainty; lower is better.
- Calibration slope < 1 suggests overfitting or shift; > 1 suggests underconfidence.
- Decision curve: net benefit = TP/N − FP/N × threshold/(1−threshold).
- At a 20% threshold, a false positive costs 0.25 of a true positive.
- The recommended threshold must reflect harm ratio and capacity, not 0.5.

### 5.3 Cohort pipeline (03.1)

Expected:

- Raw zone immutable.
- Patient/encounter spine deduplicated.
- Index event and time zero explicit.
- Lookback only before time zero.
- Group split by patient.
- Tests: uniqueness, referential integrity, temporal order, ranges.
- Rerun from clean checkout yields identical counts.

### 5.4 Leakage audit (06.2)

Expected finds:

- Post-outcome lab or note.
- Same patient in train and test.
- Imputation/scaling fit before split.
- Target encoding across folds.
- Discharge disposition as a feature.
- Duplicate encounters.

Fix: time-zero feature cutoff, group split, pipeline fit inside CV, remove post-outcome fields, deduplicate.

### 5.5 ECG quality (01.2/11.1)

Expected:

- Baseline removal and band-pass 0.5–40 Hz for monitoring; 0.05–150 Hz for diagnostic ECG.
- R-peak detection on the cleanest lead.
- HR = 60 / mean RR (seconds).
- Flag clipping, lead-off, motion, and 50/60 Hz noise.
- Do not compare automated and manual intervals without noting lead and calibration.

### 5.6 Pharmacokinetics (01A.5)

Expected:

- First-order elimination: \(C(t)=C_0e^{-kt}\).
- \(t_{1/2}=\ln 2/k\).
- Steady state ≈ 4–5 half-lives.
- Loading dose = \(V_d \times C_{\text{target}}/F\).
- Maintenance dose = \(CL \times C_{\text{target}} \times \tau / F\).
- State assumptions: one compartment, first-order, stable CL/Vd.

### 5.7 Quantization (14B.2)

Expected:

- Symmetric int8 scale: \(S=\max|x|/127\).
- Asymmetric: \(S=(x_{\max}-x_{\min})/255\), \(Z=\text{round}(-x_{\min}/S)\).
- Quantized value: \(q=\text{clamp}(\text{round}(x/S)+Z,0,255)\).
- Dequantized: \(\hat{x}=S(q-Z)\).
- PTQ requires representative calibration data; QAT uses fake quantization and STE.
- Report accuracy, calibration, subgroup performance, latency, memory, and energy on target.

### 5.8 HIL latency/energy (14B.7)

Expected:

- Measure cold and warm start.
- Report p50/p95/p99, not mean only.
- Measure joules/inference with a power monitor; estimate battery life.
- Run sustained load and record thermal throttling.
- Test power loss, watchdog, and rollback.
- A desktop emulator is not target validation.

### 5.9 Capstone rubric calibration (19.12)

Use three anchor submissions:

- **Novice:** model-first, random split, AUROC only, no limitations.
- **Competent:** clear decision, temporal split, calibration, subgroup, limitations.
- **Expert:** external validation or credible plan, decision curve, workflow, regulatory path, monitoring, stop criteria.

Score independently, then reconcile. A 10-point difference between reviewers triggers discussion.

## 6. Grading calibration

See `GRADING_CALIBRATION.md`. Minimum practice:

1. Two graders score the same five artifacts independently.
2. Compare criterion-level scores.
3. Discuss differences > 1 point on a 4-point scale.
4. Agree on anchor examples.
5. Re-score after calibration.
6. Record the calibration memo.

## 7. Starter notebooks

Located in `labs/`:

- `01A_unit_conversion.ipynb`
- `03_cohort_pipeline.ipynb`
- `06_calibration_decision_curve.ipynb`
- `11_ecg_quality.ipynb`
- `14B_quantization.ipynb`

Each notebook contains:

- Objective and data instructions.
- Setup cell.
- TODO cells.
- Acceptance checks.
- Reflection prompts.

## 8. Answer-key policy

- Keep answer keys in a private instructor repository.
- Do not distribute solution notebooks to learners before the due date.
- Rotate datasets and thresholds each cohort.
- Publish rubrics, not answers.
- Record model solutions and known ambiguities.
- Update keys when libraries or data change.

## 9. AI-assistance policy for assessments

- Disclosure required for material LLM/coding-assistant use.
- Learner accountable for every claim, citation, and line of code.
- No identifiable data in external tools.
- No fabricated sources or results.
- Oral defense may cover any part of the submission.
- AI-generated code must be tested and explained.

## 10. Accessibility and inclusion

- Provide materials in accessible HTML/PDF with alt text.
- Caption videos; provide transcripts.
- Avoid color-only encoding.
- Allow extra time where not clinically meaningfully timed.
- Offer multiple formats for outputs.
- Test with disabled and neurodivergent learners.
- Use plain-language summaries.

## 11. Lab logistics

- Provide a fallback synthetic dataset for every credentialed dataset.
- Pin environments and test on Windows, macOS, and Linux.
- Budget cloud credits and GPU time.
- Require data-use agreements before access.
- Keep PHI out of shared drives and chat.
- Have a safety plan for hardware labs.

## 12. Weekly instructor checklist

- [ ] Learning outcomes and agenda posted.
- [ ] Lab tested.
- [ ] Dataset access confirmed.
- [ ] Case critique rubric ready.
- [ ] Guest reviewer confirmed.
- [ ] Accessibility check.
- [ ] Assessment feedback returned within 7 days.
- [ ] Risk/issue log updated.

## 13. File map

| File | Use |
|---|---|
| `README.md` | Program guide |
| `COURSE_OUTLINE.md` | Master TOC |
| `COMPETENCY_MAP.md` | Outcome mapping |
| `GOVERNANCE_AND_EXTERNAL_REVIEW.md` | Review plan |
| `REGULATORY_VERIFICATION_LOG.md` | Claim log |
| `GRADING_CALIBRATION.md` | Grader training |
| `PILOT_PROTOCOL.md` | Pilot |
| `labs/` | Starter notebooks |
| `labs/solutions/` | Private sample solutions |
| `labs/cloud/` | Terraform, Kubernetes, Kafka reference stack |
| `labs/mlops/` | FastAPI, CI, K8s, monitoring, runbook reference stack |
| `_research/` | Source packs and audit |
