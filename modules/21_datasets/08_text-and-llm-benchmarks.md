# Text and LLM Benchmarks

> Module 21 — Datasets, Tools, and Benchmarks
> Topic 08 of 12
> Estimated study time: 4 hours
> Prerequisites: Topic 01, Topic 02
> Version 2026.3

## 1. Why this topic exists

Clinical text is the modality where a benchmark is easiest to publish and hardest to interpret. A question set can be released in a week, a model can be scored in an afternoon, and a leaderboard can be cited in a press release. None of that establishes that the model is safe, useful, or better than the alternative in a clinical workflow.

The failures are specific:

- A model scores highly on a medical question set whose items appeared in its pretraining data, and the score is reported as reasoning.
- A summarization model is evaluated against reference summaries written by clinicians for a different purpose, and factual errors that matter clinically are not penalised.
- A rubric-based benchmark ranks two models differently when the judge model changes, and the ranking is reported without the judge.
- An agent benchmark rewards a correct tool call and does not penalise an unsafe near-miss.
- A benchmark built on synthetic patients is treated as evidence of clinical behaviour.
- A leaderboard difference of two points is reported as progress when the confidence intervals overlap.
- A dataset licensed for research is used to evaluate a commercial product.

Text benchmarks are measurement instruments. An instrument has a construct, a sample, a scoring rule, and an error. Most clinical LLM benchmarks publish the sample and the scoring rule and leave the construct and the error implicit. This topic makes them explicit.

The resources covered are MIMIC-IV-Note, the i2b2 and n2c2 corpora, MedQA, MedMCQA, PubMedQA, HealthBench, MedAgentBench, and the MIMIC-derived note collections. The focus is on contamination, rubric dependence, judge validation, and the claims a benchmark cannot support.

The practical consequence is that text benchmarks require two audits, not one. The first asks what the benchmark measures; the second asks whether the measurement is clean. A benchmark that passes the first and fails the second produces a number that is precise, reproducible, and wrong about the model.

A second consequence is that the judge is part of the result. Two teams can run the same model on the same items and report different scores because they used different rubric prompts. The report must include the judge, the prompt, the temperature, and the agreement with human grades.

> **Note:** Benchmark releases, leaderboards, and scores change frequently. This topic does not report 2026 scores or winners. It describes what each benchmark measures and how to audit a result. Verify the current release and terms at the official page before citing.

## 2. Learning outcomes

By the end of this topic you can:

1. **Distinguish** knowledge QA, clinical NLP tasks, summarization, rubric-graded advice, and agent benchmarks by their construct and unit. Assessed by Exercise 1 and self-check item 1.
2. **Audit** a benchmark for pretraining contamination using n-gram overlap, date cutoffs, and paraphrase probes. Assessed by Exercise 2.
3. **Validate** an LLM judge against clinician grades and report agreement. Assessed by Exercise 3.
4. **Compute** a confidence interval for a benchmark score and decide whether a ranking is supported. Assessed by Exercise 4.
5. **Map** a benchmark's licence and access tier to a product evaluation plan. Assessed by Exercise 5.
6. **Design** a rubric that separates factual accuracy, completeness, and safety, and test its reliability. Assessed by Exercise 6.
7. **Evaluate** an agent benchmark for unsafe near-misses and tool-call correctness. Assessed by Exercise 7.
8. **Write** a benchmark validity memo that states the construct, the sample, the judge, the contamination checks, and the prohibited claims. Assessed by Exercise 8.

## 3. The core mental model

A text benchmark is a chain from a construct to a number. Each link adds error, and the number at the end is only as meaningful as the weakest link.

```text
   CONSTRUCT
   knowledge | reasoning | summarization | advice quality | tool use
        |
        v
   ITEMS
   exam questions | clinical notes | dialogues | tasks | synthetic patients
        |
        v
   REFERENCE
   gold answer | clinician summary | rubric | expected tool call
        |
        v
   SCORING
   exact match | overlap | rubric grade | LLM judge | human review
        |
        v
   AGGREGATION
   accuracy | pass rate | mean rubric score | win rate
        |
        v
   CLAIM
   capability | safety | clinical utility | deployment readiness
```

**The construct rule.** A benchmark measures the construct its items and scoring rule operationalise. A medical exam measures exam performance, which correlates with knowledge and is not clinical judgement. A rubric measures agreement with the rubric author, which correlates with quality and is not patient outcome.

**The contamination rule.** If the items, the reference answers, or close paraphrases appeared in pretraining, the score is a measure of memorisation as well as capability. Contamination is not binary; it is a spectrum from exact overlap to conceptual familiarity.

**The judge rule.** An LLM judge is a measurement instrument. It has a prompt, a model, a temperature, and an agreement rate with human grades. A benchmark that reports a score without the judge configuration is not reproducible.

**The interval rule.** A pass rate on 500 items has a confidence interval of several percentage points. A two-point difference between models can be noise. Report intervals and seed variance before ranking.

**The claim rule.** A benchmark score supports a claim about the benchmark. It does not support a claim about clinical utility, safety, or deployment readiness without a separate study.

### Claims and the benchmarks that cannot support them

| Claim | Tempting benchmark | Why it fails | What is needed |
|---|---|---|---|
| The model is clinically safe | HealthBench | Rubric advice is not patient outcome | Red-teaming and safety review |
| The model can diagnose | MedQA | Exam items are not clinical cases | Case-based human evaluation |
| The model summarizes accurately | ROUGE on notes | Overlap is not factual accuracy | Fact-level audit |
| The model is ready to deploy | A leaderboard | Benchmark construct differs | Prospective workflow study |
| The model reasons | High exam score | Contamination and format | Temporal holdout and probes |
| The model is fair | Aggregate score | No subgroup reporting | Stratified evaluation |
| The model is robust | One prompt | Prompt sensitivity | Prompt and seed variance |

### What each benchmark cannot answer

| Benchmark | Cannot answer | Why |
|---|---|---|
| MIMIC-IV-Note | Commercial deployment | Research DUA |
| i2b2/n2c2 | Open-ended summarization | Task-specific annotations |
| MedQA | Clinical judgement | Exam format and contamination |
| MedMCQA | Global practice | Single-country exam |
| PubMedQA | Patient-specific advice | Literature-only evidence |
| HealthBench | Patient outcome | Rubric advice, not care |
| MedAgentBench | Safe autonomy | Simulated EHR |
| A leaderboard | Local workflow fit | Different population and judge |

| Benchmark type | Construct | Unit | What it can support | What it cannot support |
|---|---|---|---|---|
| Knowledge QA | Exam-style knowledge | Item | Relative knowledge | Clinical reasoning or safety |
| Clinical NLP | Extraction or classification | Note or entity | Task performance | Workflow benefit |
| Summarization | Factual consistency and coverage | Summary | Summary quality | Clinical decision quality |
| Rubric advice | Advice quality against criteria | Conversation turn | Rubric agreement | Patient outcome |
| Agent | Tool use and task completion | Task | Tool-call competence | Safe autonomy |
| Human evaluation | Clinician preference | Comparison | Preference | Objective correctness |

## 4. Definitions and vocabulary

### 4.1 Benchmark and task terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Benchmark | A fixed task, dataset, and metric. | Treated as clinical validation. | Task specification |
| Item | One question or task instance. | Counted as a patient. | Item |
| Construct | The capability the benchmark intends to measure. | Left implicit. | Definition |
| Gold answer | The reference answer. | Treated as uniquely correct. | Answer |
| Reference summary | A clinician-written summary. | Treated as the only valid summary. | Text |
| Rubric | Criteria and weights for grading. | Treated as objective truth. | Criteria |
| Judge | The model or person assigning grades. | Not reported. | Model or human |
| Pass rate | The share of items scored correct. | Reported without an interval. | Proportion |
| Win rate | The share of pairwise comparisons won. | Reported without ties or rater agreement. | Proportion |
| Saturation | Top models cluster within noise. | Reported as a new record. | Interval overlap |

### 4.2 Contamination and leakage terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Contamination | Test items in training data. | Checked only by exact match. | Proportion |
| Near-duplicate | A paraphrase or close variant. | Ignored. | Similarity |
| Canary string | A marker inserted to detect leakage. | Omitted. | Probe |
| Date cutoff | The model's training cutoff. | Assumed from the release date. | Date |
| Paraphrase probe | A reworded item to test sensitivity. | Treated as a new item. | Probe |
| Option-order probe | Reordering multiple-choice options. | Omitted. | Probe |
| Private holdout | Items never released. | Replaced by a public test set. | Split |
| Temporal holdout | Items published after the cutoff. | Not constructed. | Split |
| Memorisation | Reproduction of training text. | Confused with reasoning. | Behaviour |
| Benchmark leakage | The benchmark itself in training. | Ignored. | Contamination |

### 4.3 Evaluation and statistics terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Cohen's kappa | Chance-corrected agreement. | Used for more than two raters without adjustment. | Statistic |
| Agreement rate | Raw proportion of agreement. | Reported without chance correction. | Proportion |
| Confidence interval | Uncertainty around an estimate. | Omitted. | Interval |
| Wilson interval | A binomial interval robust at extremes. | Replaced by a normal approximation. | Interval |
| Seed variance | Variation across random seeds. | Ignored. | Standard deviation |
| Judge prompt | The instruction given to the judge. | Not reported. | Text |
| Temperature | Sampling randomness. | Not reported. | Parameter |
| Position bias | Preference for the first option. | Ignored in pairwise judging. | Bias |
| Verbosity bias | Preference for longer answers. | Ignored. | Bias |
| Self-preference | A judge favouring its own outputs. | Ignored. | Bias |

### 4.4 Text and model terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Token | A unit of text processed by a model. | Treated as a word. | Token |
| Context window | The maximum tokens a model can process. | Ignored in long notes | Tokens |
| Prompt | The instruction and context given to a model. | Not reported | Text |
| System prompt | A persistent instruction. | Changed silently | Text |
| Few-shot example | An example included in the prompt. | Treated as training | Count |
| Temperature | Sampling randomness. | Not reported | Parameter |
| Hallucination | A fabricated fact. | Treated as creativity | Error type |
| Grounding | Traceability to a source. | Assumed from fluency | Score |
| Redaction | Removing identifiers. | Confused with de-identification | Process |
| Note | A clinical document. | Treated as a complete record | Document |
| Discharge summary | A summary written at discharge. | Treated as the full record | Document |

### 4.5 Resource terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| MIMIC-IV-Note | De-identified notes linked to MIMIC-IV. | Treated as general clinical text. | Dataset |
| i2b2 | Informatics for Integrating Biology and the Bedside. | Treated as one corpus. | Programme |
| n2c2 | National NLP Clinical Challenges. | Treated as a single task. | Challenge series |
| MedQA | US medical licensing exam-style QA. | Treated as clinical reasoning. | Dataset |
| MedMCQA | Indian medical entrance exam QA. | Treated as a global benchmark. | Dataset |
| PubMedQA | QA over biomedical abstracts. | Treated as clinical decision support. | Dataset |
| HealthBench | Clinician-written conversations with rubrics. | Treated as patient outcome. | Benchmark |
| MedAgentBench | Agent tasks on a FHIR server. | Treated as clinical autonomy. | Benchmark |

## 5. The mechanics, step by step

The procedure has 18 steps in four phases: define the construct and the claim, audit the items and contamination, design the scoring and judge, and report with uncertainty and limits.

### Phase A — Define the construct and the claim

**Step 1. Write the claim the benchmark is meant to support.**
Input: the product or research question.
Output: a claim labelled as capability, safety, workflow, or outcome.
Rationale: the benchmark choice follows from the claim, not the reverse.
Branch: if the claim is clinical utility, no public text benchmark is sufficient; a prospective study is needed.
Edge case: a regulatory claim about a device function needs evidence beyond benchmark accuracy.

**Step 2. Identify the construct.**
Input: the claim and the candidate benchmarks.
Output: the construct each benchmark operationalises.
Rationale: a knowledge QA benchmark does not measure summarization, and a summarization benchmark does not measure safety.
Branch: if the construct and the claim differ, the benchmark is at most supporting evidence.
Edge case: a benchmark can measure several constructs and report one aggregate; read the task definitions.

**Step 3. Fix the unit of analysis.**
Input: the benchmark structure.
Output: item, note, conversation, summary, or task.
Rationale: the unit determines the denominator and the interval.
Branch: if a patient contributes many items, the items are not independent.
Edge case: a multi-turn conversation can be scored per turn or per conversation; the choice changes the result.

**Step 4. Choose the reference and the scoring rule.**
Input: the benchmark documentation.
Output: gold answer, reference summary, rubric, or expected tool call.
Rationale: the reference defines what counts as correct.
Branch: if the reference is a single clinician summary, the scoring measures similarity to that summary, not all valid summaries.
Edge case: a rubric can encode the author's preferences; test reliability across raters.

### Rubric dimensions

| Dimension | What it captures | Scale | Failure if omitted |
|---|---|---|---|
| Factual accuracy | Correctness against the source | 0 to 3 | Fluent errors pass |
| Completeness | Coverage of required facts | 0 to 3 | Omissions pass |
| Safety | Harmful or dangerous advice | Gate | Unsafe output scores well |
| Grounding | Traceability to the source | 0 to 2 | Fabrication passes |
| Clarity | Understandability | 0 to 2 | Style dominates |
| Conciseness | Absence of padding | 0 to 2 | Verbosity rewarded |
| Actionability | Whether the reader can act | 0 to 2 | Advice is useless |
| Uncertainty | Appropriate hedging | 0 to 2 | False confidence passes |

A rubric with an aggregate score and no safety gate can reward a model that is fluent, complete, and dangerous. The gate is a design choice, not a statistical one.

### Judge validation

| Check | Method | Threshold | Consequence if failed |
|---|---|---|---|
| Agreement | Kappa or ICC against clinicians | Study-specific | Score not interpretable |
| Position bias | Swap A and B | Small difference | Ranking unreliable |
| Verbosity bias | Compare length-controlled pairs | Small difference | Long answers win |
| Self-preference | Use a different judge family | Small difference | Own model wins |
| Prompt sensitivity | Vary wording | Small difference | Score is prompt artefact |
| Temperature | Repeat at zero and non-zero | Small difference | Score is sampling noise |
| Calibration | Compare grades across bins | Monotonic | Threshold unusable |
| Failure detection | Inject bad answers | High detection | Judge misses errors |

The judge is a model with its own failure modes. A benchmark report that names the judge but not its validation is incomplete.

### Human evaluation design

| Choice | Options | Effect | Recommendation |
|---|---|---|---|
| Rater expertise | Trainee, specialist, multi-disciplinary | Agreement and cost | Match the claim |
| Blinding | Blind, partially blind, open | Bias | Blind to model identity |
| Anchoring | None, examples, reference | Consistency | Calibrate with examples |
| Scale | Binary, Likert, pairwise | Discrimination | Pairwise for preference |
| Ties | Allowed or forced | Win rate | Allow ties |
| Sample size | Fixed or sequential | Interval width | Pre-specify |

### Contamination arithmetic

| Scenario | Items | Overlap | Interpretation |
|---|---:|---:|---|
| Exact match | 500 | 25 | 5% of items copied |
| Near-duplicate | 500 | 60 | 12% paraphrased |
| Temporal holdout | 200 | 0 | Cleaner but smaller |
| Private holdout | 100 | 0 | Clean but expensive |

A 5 percent exact overlap can move a pass rate by several points if the copied items are easy. Report the overlap and the score with and without the flagged items.

### Reporting checklist

| Item | What to report | Why |
|---|---|---|
| Benchmark and version | Name, release, hash | Reproducibility |
| Construct | What it measures | Interpretation |
| Items | Count and source | Denominator |
| Reference | Gold, summary, or rubric | Correctness |
| Judge | Model, prompt, temperature | Measurement |
| Agreement | Kappa or ICC | Error |
| Contamination | Probes and results | Validity |
| Interval | Confidence interval | Uncertainty |
| Seeds | Runs and variance | Stability |
| Subgroups | Specialty and safety | Hidden failures |
| Cost | Tokens and compute | Practicality |
| Licence | Terms and access | Compliance |

### Prompt sensitivity

| Prompt element | Variation | Effect | Control |
|---|---|---|---|
| System instruction | Persona or role | Large | Fix and report |
| Few-shot examples | Number and content | Large | Pre-specify |
| Output format | Free text or JSON | Medium | Match the task |
| Chain-of-thought | On or off | Medium | Report |
| Temperature | 0 to 1 | Medium | Report |
| Max tokens | Truncation | Large | Check truncation |

A benchmark score is a property of the model, the prompt, and the decoding settings together. Reporting only the model name is not enough.

### Safety failure taxonomy

| Failure | Example | Detection | Severity |
|---|---|---|---|
| Omitted red flag | Misses a critical finding | Clinician review | High |
| Fabricated fact | Invents a lab value | Source audit | High |
| Wrong dose | Incorrect medication dose | Rule check | Critical |
| Delayed action | Advice that postpones care | Case review | High |
| Unsafe reassurance | "No need to seek care" | Safety rubric | Critical |
| Hallucinated citation | A paper that does not exist | Citation check | Medium |
| Privacy leak | Repeats identifiable text | Redaction check | Critical |

The safety categories should be gates in the rubric and separate rows in the report. An aggregate score that averages a critical failure with a style gain is misleading.

### Aggregation choices

| Aggregation | What it does | Risk |
|---|---|---|
| Mean rubric score | Averages dimensions | Hides a zero |
| Pass rate | Share above a threshold | Threshold arbitrary |
| Win rate | Pairwise preference | Ties and rater bias |
| Worst-case score | Minimum across categories | Sensitive to one item |
| Gate plus score | Safety gate, then quality | Requires a gate definition |

A benchmark report should state the aggregation and show the gate result separately. A mean score is a summary; the gate is the decision.

The probe sequence is a screen, not a proof. It identifies benchmarks that cannot support a clean capability claim and cannot certify the ones that remain. A clean probe means the score is more interpretable, not that the model is safe.

### Item provenance and difficulty

| Source | Typical contamination risk | Difficulty | Claim it supports |
|---|---|---|---|
| Public exam questions | High | Easy to hard | Knowledge proxy |
| Published guidelines | High | Medium | Guideline recall |
| Clinician-written cases | Medium | Hard | Case reasoning |
| Private hospital notes | Low | Hard | Local workflow |
| Synthetic cases | Low | Variable | Method development |
| Adversarial cases | Low | Hard | Safety |

The source determines both the contamination risk and the difficulty. A benchmark built from public exam items is cheap and easy to contaminate; a benchmark built from private cases is expensive and harder to interpret across sites.

### Evaluation budget

| Item | Unit | Typical range | Planning implication |
|---|---|---|---|
| Model calls | Per item | 1 to 5 | Cost scales with items |
| Judge calls | Per item | 1 to 5 | Often exceeds model cost |
| Human ratings | Per item | 1 to 3 | Dominates the budget |
| Adjudication | Per disagreement | 1 extra rating | Add 20 to 40 percent |
| Re-runs | Seeds or prompts | 3 to 5 | Multiply the total |
| Annotation tooling | Fixed | Variable | Often omitted |

A 500-item benchmark with three seeds, three judge calls, and a 10 percent human sample is not a cheap experiment. The budget should be estimated before the evaluation, not discovered after.

### Phase B — Audit the items and contamination

**Step 5. Audit the item source and publication date.**
Input: the benchmark paper and release notes.
Output: the date range and the provenance of the items.
Rationale: items published after a model's cutoff are less likely to be contaminated.
Branch: if all items predate the cutoff, build a temporal holdout from newer items.
Edge case: a benchmark can be released after the cutoff while its items are old.

**Step 6. Audit exact overlap with public corpora.**
Input: the items and a searchable corpus.
Output: exact matches and their sources.
Rationale: exact overlap is the easiest contamination to detect.
Branch: if exact matches are found, exclude or flag the items.
Edge case: a match in a public corpus does not prove the model saw it, but it raises the risk.

**Step 7. Audit near-duplicates and paraphrases.**
Input: the items and embedding or n-gram similarity.
Output: near-duplicate candidates and a similarity distribution.
Rationale: models can memorise paraphrases, not only exact strings.
Branch: if near-duplicates are common, the benchmark is contaminated for that model family.
Edge case: clinical text has standard phrases; high similarity is not always contamination.

**Step 8. Run behavioural probes.**
Input: the benchmark and the model.
Output: scores under canary insertion, option reordering, and paraphrase.
Rationale: a large drop under paraphrase indicates fragility; a large drop under option reordering indicates position bias.
Branch: if the score is stable under paraphrase and reordering, the evidence for genuine capability is stronger.
Edge case: a probe can change difficulty; compare with a control model.

**Step 9. Audit the licence and access tier.**
Input: the benchmark terms.
Output: whether the benchmark can be used for a commercial evaluation and whether items can be retained.
Rationale: a research-only benchmark cannot be the primary evidence for a product.
Branch: if the licence is research-only, use it for methods and find a commercial-permissive or partner set.
Edge case: the items can be open while the reference annotations are restricted.

### Phase C — Design the scoring and judge

**Step 10. Define the rubric dimensions.**
Input: the construct and the clinical risk.
Output: separate dimensions for factual accuracy, completeness, safety, and style.
Rationale: an aggregate score hides a safety failure behind a style gain.
Branch: if safety is a hard constraint, score it as a gate, not a weighted dimension.
Edge case: a rubric can reward verbosity; include a conciseness criterion or control for length.

**Step 11. Validate the rubric with human raters.**
Input: a sample of items and two or more clinicians.
Output: agreement statistics and a list of ambiguous criteria.
Rationale: a rubric that clinicians cannot apply consistently is not a measurement instrument.
Branch: if agreement is low, revise the criteria or the scale before scoring models.
Edge case: the raters can be anchored by examples; record the calibration.

**Step 12. Choose the judge and report its configuration.**
Input: the rubric and the budget.
Output: a human judge, an LLM judge, or a hybrid, with model, prompt, and temperature.
Rationale: the judge is part of the measurement.
Branch: if an LLM judge is used, validate it against human grades on a sample.
Edge case: a judge can favour its own model family; use a different judge or a panel.

**Step 13. Measure judge agreement.**
Input: the judge scores and the human scores.
Output: Cohen's kappa or ICC, and the disagreement cases.
Rationale: a judge with low agreement produces a score with unknown error.
Branch: if agreement is low, do not report the benchmark score as a quality estimate.
Edge case: agreement can be high on easy items and low on the hard items that matter.

### Phase D — Report with uncertainty and limits

**Step 14. Compute the interval.**
Input: the score and the number of items.
Output: a confidence interval appropriate to the design.
Rationale: a point estimate without an interval invites over-interpretation.
Branch: if items are clustered by patient or conversation, use a cluster-robust interval.
Edge case: a 100-item benchmark has a wide interval even at a high score.

**Step 15. Report seed and prompt variance.**
Input: repeated runs with different seeds and prompts.
Output: the distribution of scores.
Rationale: a single run can be a lucky sample.
Branch: if variance is large, report the range and do not rank close models.
Edge case: temperature zero reduces but does not eliminate variance.

**Step 16. Evaluate subgroups and failure modes.**
Input: the item metadata.
Output: performance by specialty, item difficulty, and safety category.
Rationale: an average can hide a dangerous subgroup.
Branch: if a safety category fails, the aggregate score is not a defence.
Edge case: subgroup sample sizes are often small; report intervals.

**Step 17. Score the gates and record the evidence.**
Input: the outputs of steps 1 to 16.
Output: a gate table with evidence and residual uncertainty.
Rationale: the decision must be reviewable.
Branch: a zero on any gate is a veto for the claim as written.
Edge case: a benchmark can be admissible for a capability claim and inadmissible for a product claim.

**Step 18. Re-run on benchmark, model, or judge change.**
Input: the release notes and the model versions.
Output: an updated record with a new date.
Rationale: benchmarks, models, and judges change.
Branch: re-run when the benchmark version, the model checkpoint, the judge, or the prompt changes.
Edge case: a leaderboard can change without a version number.

### Benchmark profiles

| Benchmark | Custodian | Unit | Access and licence | Construct | What it cannot answer |
|---|---|---|---|---|---|
| MIMIC-IV-Note | PhysioNet and MIT | Note | Credentialed; research DUA | Clinical text and summarization | General clinical text; commercial product rights |
| i2b2 and n2c2 | Harvard DBMI and challenge organisers | Note or entity | DUA; some open | Clinical NLP tasks | Open-domain reasoning; product deployment |
| MedQA | Open research release | Question | Open; verify terms | Exam-style medical knowledge | Clinical reasoning; workflow safety |
| MedMCQA | Open research release | Question | Open; verify terms | Exam-style knowledge | Global clinical practice |
| PubMedQA | Open research release | Question | Open; verify terms | Literature comprehension | Patient-specific decisions |
| HealthBench | OpenAI | Conversation | Open release; verify terms | Advice quality against rubrics | Patient outcome; clinical utility |
| MedAgentBench | Stanford | Task | Open code and data; verify licence | Tool use on a FHIR server | Safe autonomy; real EHR behaviour |

The profiles are orientation, not endorsement. Each benchmark's release and terms should be checked at the official page before use.

### Construct-to-evidence table

| Claim | Benchmark that helps | What it still lacks | Required addition |
|---|---|---|---|
| The model knows medical facts | MedQA, MedMCQA | Clinical judgement | Case-based human evaluation |
| The model summarizes accurately | MIMIC-IV-Note, n2c2 | Factual error taxonomy | Clinician fact audit |
| The model gives safe advice | HealthBench | Patient outcome | Safety review and red-teaming |
| The model uses tools correctly | MedAgentBench | Near-miss safety | Unsafe-action rate |
| The model extracts entities | i2b2 and n2c2 | Generalization | External corpus |
| The model is deployable | None | Workflow and outcome | Prospective study |

### Contamination probe table

| Probe | What it detects | Cost | Limitation |
|---|---|---|---|
| Exact n-gram match | Copied text | Low | Misses paraphrases |
| Near-duplicate search | Paraphrases | Medium | False positives |
| Date-based holdout | Older training data | Medium | Requires dates |
| Canary insertion | Leakage of a marker | Low | Only detects marker |
| Option reordering | Position bias | Low | Not contamination |
| Paraphrase rewrite | Fragility and memorisation | Medium | Changes difficulty |
| Private holdout | Tuning overfitting | High | Requires new items |

### Edge cases that change the procedure

**Licence forbids commercial use.** MIMIC-IV-Note and i2b2 corpora are research resources under data use agreements. A commercial product evaluation needs a commercial-permissive benchmark or partner data. A benchmark's open download is not a commercial licence.

**Benchmark withdrawn or changed.** A benchmark can be revised, retracted, or replaced. Pin the version and keep a copy of the item set where the licence permits.

**Credentialing takes months.** Clinical note corpora can require training and a DUA. A timeline shorter than the access path fails Gate 3.

**Version drift.** A benchmark update can change items, references, and scoring. A score from one version is not comparable with another.

**Label errors discovered.** Reference answers and clinician summaries can be wrong. Re-audit a sample, report the error, and test sensitivity. Do not relabel only the test set.

**Benchmark contamination.** If items appear in pretraining, the score is not a clean capability estimate. Replace the test set with a temporal or private holdout.

## 6. Worked example

### The case

A 35-person clinical documentation company wants to sell a discharge-summary generator to hospital systems. The model drafts a summary from the admission record, and the clinician edits and signs it. The company claims a 30 percent reduction in editing time without an increase in factual errors. It has 12 months to a pilot.

**Decision statement.** "At discharge, draft a summary that a clinician can edit and sign in less than the current median time, with no increase in factual errors or omissions; a factual error can propagate into the next episode of care, and a slow draft is ignored."

**Construct.** The claim is workflow and factual consistency, not medical knowledge. The relevant benchmark must contain discharge summaries, reference facts, and a way to measure editing and error.

**Unit.** The discharge summary, clustered by patient and by clinician.

### Candidate long list

| Candidate | Custodian | Unit | Access and licence | Construct | Why it is on the list |
|---|---|---|---|---|---|
| MIMIC-IV-Note | PhysioNet and MIT | Note | Credentialed; research DUA | Clinical notes and summaries | Real discharge summaries |
| i2b2 and n2c2 | Harvard DBMI and organisers | Note or entity | DUA; some open | Clinical NLP tasks | Annotation quality |
| HealthBench | OpenAI | Conversation | Open; verify terms | Advice quality | Rubric-based clinician evaluation |
| MedAgentBench | Stanford | Task | Open code and data | Tool use on FHIR | Agent evaluation |
| MedQA | Open research release | Question | Open | Exam knowledge | Common baseline |
| Partner notes | Two hospital systems | Note | Contract and IRB | Target workflow | Exact setting |

### Gate scores for the commercial claim

| Gate | MIMIC-IV-Note | i2b2/n2c2 | HealthBench | MedAgentBench | MedQA | Partner |
|---|---:|---:|---:|---:|---:|---:|
| 1 Decision fit | 2 | 1 | 1 | 1 | 0 | 2 |
| 2 Population fit | 1 | 1 | 1 | 1 | 1 | 2 |
| 3 Access fit | 2 | 1 | 2 | 2 | 2 | 0 |
| 4 Licence fit | 0 | 0 | 1 | 1 | 2 | 2 |
| 5 Benchmark fit | 1 | 1 | 1 | 1 | 1 | 2 |
| Veto | licence | licence | none | none | decision | access |

**MIMIC-IV-Note, Gate 1 and Gate 4.** The notes include discharge summaries, which match the construct. The data are credentialed and governed by a research data use agreement; commercial product evaluation is outside it. Score 2 on decision fit and 0 on licence. Veto for the product claim.

**i2b2 and n2c2, Gate 1 and Gate 4.** The corpora support specific NLP tasks such as entity recognition and relation extraction, not open-ended summarization. Access is task-specific and often research-only. Veto for the product claim.

**HealthBench, Gate 1.** The benchmark measures advice quality against clinician-written rubrics. It is adjacent to the claim and does not contain discharge summaries or editing time. Score 1, admissible as secondary evidence about communication quality.

**MedAgentBench, Gate 1.** The benchmark measures tool use on a synthetic FHIR server. It is relevant if the product retrieves facts through tools and not relevant to summary quality. Score 1.

**MedQA, Gate 1.** Exam-style knowledge QA does not measure summarization, factual consistency, or editing. Score 0, veto.

**Partner, Gate 3.** Two hospital systems are interested, but no contract, IRB approval, or data-sharing agreement exists. Twelve months is not credible. Veto for the timeline, retained as the validation path.

### The two diagnostics

**Diagnostic 1: contamination.** A knowledge benchmark's items can appear in pretraining. In the planning audit, 8-gram overlap against a public corpus flagged a subset of items with near-complete overlap, and the overlap was concentrated in the easiest items. The model's score on those items cannot be attributed to reasoning. The diagnostic is not the aggregate score; it is the overlap distribution and the score with and without the flagged items.

**Diagnostic 2: rubric and judge dependence.** HealthBench-style scoring depends on a rubric and a judge. In a calibration exercise, two judges agreed on 83 percent of items with a Cohen's kappa of 0.62, which is substantial and not high. Changing the judge changed the ranking of two model outputs. The benchmark score is therefore a property of the model, the rubric, and the judge together, and a report that names only the model is incomplete.

A third diagnostic is reference-summary variance. Two clinicians writing a discharge summary from the same record produce different but valid summaries. A metric that measures similarity to one reference penalises valid alternatives. The evaluation needs a fact-level audit, not only text similarity.

### The selected route

The admissible set is HealthBench and MedAgentBench, and neither supports the summarization claim. The plan uses them as secondary evidence and makes the partner notes the primary evaluation.

| Track | Steps | Owner | Duration | Planning cost |
|---|---|---|---|---|
| Benchmark audit | Contamination and judge calibration | Evaluation lead | 4 weeks | Staff time |
| Secondary evaluation | HealthBench and MedAgentBench | ML team | 4 weeks | Compute |
| Partner contract | DUA, IRB, data transfer | Legal and clinical | 6 to 9 months | Legal and site fees |
| Fact audit | Clinician review of 300 summaries | Clinical lead | 6 weeks | About $18,000 |
| Editing study | Time-and-motion study | Human factors | 6 weeks | Staff time |
| Pilot | Silent mode and clinician sign-off | Product | 3 months | Site costs |

### Time and cost

| Item | Detail | Calendar time | Planning cost |
|---|---|---|---|
| Secondary benchmarks | Contamination and judge checks | 8 weeks | Compute and staff |
| Partner data | Contract, IRB, transfer | 6 to 9 months | Legal and site fees |
| Fact audit | 300 summaries, two clinicians | 6 weeks | About $18,000 |
| Editing study | Timed sessions | 6 weeks | Staff time |
| Total to a pilot decision | Excludes product build | About 12 months | Dominated by partner access |

The partner access is the critical path. The public benchmarks are fast and cannot support the product claim.

### Substitute plan if the partner route fails

If the hospital systems cannot provide data, the plan changes in three ways:

1. The claim narrows to a research evaluation on public notes, with no workflow or editing claim.
2. The fact audit uses public discharge summaries and a clinician panel, which is weaker evidence for the target setting.
3. The product launch is deferred until partner data exist, because the workflow claim cannot be tested otherwise.

The fallback preserves the methods and loses the commercial claim. It is not equivalent.

### Vary one assumption: MIMIC-IV-Note permitted commercial use

Suppose the data use agreement were amended to permit commercial evaluation. Gate 4 rises to 2 for MIMIC-IV-Note, and the plan changes:

- MIMIC-IV-Note becomes the primary development and fact-audit set.
- The partner route remains necessary for workflow and editing time, because MIMIC has no workflow data.
- The timeline shortens by the partner data-collection phase for the scientific claim, but not for the product claim.
- The licence amendment must be in writing and must cover model training, evaluation, and any release.

If the variation is benchmark size, a smaller item set widens the confidence interval and weakens any ranking. If the variation is judge quality, a lower kappa makes the rubric score uninterpretable and forces human review.

### Conditions under which the case would fail

- **No partner data.** The workflow and editing claims cannot be tested, and the product claim is deferred.
- **The judge does not agree with clinicians.** The rubric score is uninterpretable, and the evaluation reverts to human review.
- **Contamination is widespread.** The secondary benchmark cannot support even a capability claim.
- **The licence does not permit commercial evaluation.** The public evidence is unusable in diligence.
- **The editing study shows no time saving.** The value claim fails even if the summaries are accurate.
- **The fact audit finds a high error rate.** The safety gate fails, and the product needs a narrower use.

### Gate evidence log

| Gate | Candidate | Evidence inspected | Verdict | Residual uncertainty |
|---|---|---|---|---|
| 1 | MIMIC-IV-Note | Discharge summaries present | 2 | De-identification removes clinical tokens |
| 1 | HealthBench | Advice rubric, not summaries | 1 | Adjacent construct |
| 1 | MedAgentBench | Tool-use tasks | 1 | Not summary quality |
| 2 | MIMIC-IV-Note | Single-center notes | 1 | Not the target hospitals |
| 3 | MIMIC-IV-Note | Credentialing route | 2 | Project approval time |
| 3 | Partner | No contract or IRB | 0 | Timeline not credible |
| 4 | MIMIC-IV-Note | Research DUA | 0 | No commercial evaluation |
| 4 | HealthBench | Open release terms | 1 | Verify commercial use |
| 5 | MIMIC-IV-Note | No fact-level benchmark | 1 | Fact audit must be built |
| 5 | HealthBench | Rubric and judge | 1 | Judge dependence |
| 5 | Partner | Workflow and editing measurable | 2 | Not started |

### Decision timeline

| Month | Benchmark track | Partner track | Clinical track | Deliverable |
|---:|---|---|---|---|
| 1 | Contamination audit | Identify sites | Draft fact taxonomy | Audit report |
| 2 | Judge calibration | DUA negotiation | Clinician recruitment | Calibrated judge |
| 3 | Secondary evaluation | IRB submission | Fact-audit protocol | Secondary results |
| 4-5 | Prompt sensitivity | Data transfer | Fact audit | Internal model v1 |
| 6-7 | Subgroup analysis | Site integration | Editing study | Evaluation report |
| 8-12 | Monitoring plan | Pilot | Sign-off | Pilot decision |

### What the case demonstrates

- The benchmark with the right construct was unavailable for commercial use, and the available benchmarks measured adjacent constructs.
- Contamination was concentrated in easy items, which is the pattern that most inflates a score.
- Judge agreement was substantial and not high, and the ranking changed with the judge.
- The reference-summary variance made text similarity a weak proxy for factual correctness.
- The partner route was the only path to the workflow claim and the longest path in the plan.
- The fallback lost the commercial claim, not only the dataset.

### Sensitivity of the decision

| Change | Gate affected | Does the source change? | Plan response |
|---|---|---|---|
| MIMIC DUA permits commercial use | 4 | No | Use as primary development set |
| Partner signs in month two | 3 and 5 | Yes, becomes primary | Shift evaluation to partner |
| Judge kappa rises above 0.8 | 5 | No | Rubric score becomes interpretable |
| A public discharge-summary benchmark is released | 1 and 5 | Possibly | Audit licence and contamination |
| The model is used only for drafting, not advice | 1 | No | Narrow the claim to editing support |
| The editing study shows no time saving | 5 | No | Stop the workflow claim |

The case ends with a plan that is honest about its evidence: two public benchmarks for secondary support, one partner study for the product claim, and a fact audit that can fail the model. That is a stronger position than a high score on an adjacent benchmark presented as validation.

The plan is deliberately more expensive than a benchmark run and more defensible than a leaderboard claim. The cost is the price of a claim that survives diligence.

## 7. Code, query, or calculation

The three blocks below use the Python standard library only. They implement an n-gram contamination check, a judge agreement calculation, and a Wilson confidence interval.

### 7.1 N-gram contamination check

```python
def ngrams(text, n=8):
    tokens = text.lower().split()
    return {tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)}

corpus = [
    "a 62 year old man with hypertension presents with chest pain and diaphoresis",
    "the patient was started on aspirin and nitroglycerin for suspected acs",
    "ecg showed st elevation in the inferior leads and troponin was elevated",
]
corpus_ngrams = set()
for doc in corpus:
    corpus_ngrams |= ngrams(doc)

test_items = [
    "a 62 year old man with hypertension presents with chest pain and diaphoresis",
    "a 45 year old woman with diabetes presents with polyuria and weight loss",
    "ecg showed st elevation in the inferior leads and troponin was elevated",
]
print(f"{'Test item':<4}{'8-grams':>9}{'Overlap':>9}{'Share':>8}")
for i, item in enumerate(test_items, 1):
    grams = ngrams(item)
    overlap = len(grams & corpus_ngrams)
    share = overlap / len(grams) if grams else 0
    print(f"{i:<4}{len(grams):>9}{overlap:>9}{share:>8.0%}")
```

Expected output:

```text
Test item  8-grams  Overlap   Share
1           6        6    100%
2           6        0      0%
3           5        5    100%
```

The check is deliberately simple. In a real audit, the corpus is a large pretraining proxy, the n-gram size is varied, and the flagged items are reviewed. A high overlap on an easy item is more damaging than a high overlap on a hard one.

### 7.2 Judge agreement

```python
def cohen_kappa(matrix):
    n = sum(sum(row) for row in matrix)
    po = sum(matrix[i][i] for i in range(len(matrix))) / n
    rows = [sum(row) for row in matrix]
    cols = [sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix))]
    pe = sum(rows[i] * cols[i] for i in range(len(matrix))) / (n * n)
    return (po - pe) / (1 - pe), po

matrix = [
    [180, 20],
    [30, 70],
]
k, po = cohen_kappa(matrix)
print(f"Items: {sum(sum(r) for r in matrix)}")
print(f"Observed agreement: {po:.2f}")
print(f"Cohen's kappa: {k:.2f}")
```

Expected output:

```text
Items: 300
Observed agreement: 0.83
Cohen's kappa: 0.62
```

Observed agreement of 0.83 looks high; kappa is 0.62 because the marginal distributions are unbalanced. Report both, and inspect the disagreement cases.

### 7.3 Confidence interval for a pass rate

```python
import math

def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return centre - half, centre + half

for k, n in ((320, 500), (160, 500), (48, 50)):
    lo, hi = wilson(k, n)
    print(f"{k}/{n} = {k/n:.0%}  95% CI {lo:.1%} to {hi:.1%}")
print()
print("A 64% pass rate on 500 items has a 95% interval of roughly 60% to 68%.")
print("Ranking two systems that differ by 2 points on 500 items is not supported.")
```

Expected output:

```text
320/500 = 64%  95% CI 59.7% to 68.1%
160/500 = 32%  95% CI 28.1% to 36.2%
48/50 = 96%  95% CI 86.5% to 98.9%

A 64% pass rate on 500 items has a 95% interval of roughly 60% to 68%.
Ranking two systems that differ by 2 points on 500 items is not supported.
```

The interval is a property of the item count, not of the model. A benchmark with 100 items cannot resolve small differences, and reporting a rank without the interval is an overclaim.

## 8. Data, standards, and regulatory context

Clinical text benchmarks sit at the intersection of privacy law, device regulation, and reporting standards. The table separates binding requirements from conventions.

| Instrument | Status | What it constrains | What it does not do |
|---|---|---|---|
| HIPAA Privacy Rule | Binding in the US | Use and disclosure of protected health information | Define a benchmark construct |
| GDPR | Binding in the EU | Personal data processing and transfer | Define clinical quality |
| Common Rule | Binding for federally funded research | Consent and IRB review | Grant commercial rights |
| FDA device regulation | Binding when applicable | Claims, validation, labeling | Certify language quality |
| EU AI Act | Binding, phased | Risk management and documentation | Replace clinical evaluation |
| MDR and IVDR | Binding in the EU | Device evidence | Address benchmark validity |
| HTI-1 | Binding for certified health IT | Source attributes and fairness information | Certify a model |
| TRIPOD+AI | Reporting guideline | Prediction-model reporting | Cover generative models explicitly |
| EQUATOR reporting guidelines | Community standards | Study reporting | Validate a benchmark |
| FHIR | Exchange standard | Resource structure | Define note quality |
| UMLS | Terminology licence | Concept mapping | Define correctness |
| SNOMED CT | Clinical terminology | Concept identity | Define a summary standard |

**De-identification is not adequacy.** A note corpus can be de-identified and still contain copy-forward text, rare events, and institutional phrasing that make it unsuitable for a general claim.

**A benchmark is not a medical device.** A benchmark score does not establish that a product meets a device standard, and a device clearance does not establish that a benchmark is valid.

**Reporting guidelines lag generative models.** TRIPOD+AI addresses prediction models and does not explicitly cover large language model outputs. A benchmark report should borrow its structure and add items for prompts, judges, contamination, and safety.

### Text standards that affect evaluation

| Standard | Domain | What it gives | What it does not give |
|---|---|---|---|
| FHIR DocumentReference | Document exchange | Resource structure for notes | Note quality |
| HL7 CDA | Clinical documents | Structured document format | Semantic correctness |
| SNOMED CT | Clinical concepts | Concept identity | Summary completeness |
| LOINC | Observations | Test identity | Narrative quality |
| UMLS | Terminology integration | Concept mapping | Licence-free use |
| RxNorm | Medications | Drug identity | Dose correctness |
| ICD-10-CM | Diagnoses | Code identity | Clinical timing |

A summary can be fluent and use the right codes and still omit the fact that changes the plan. The standards make the content machine-readable; they do not make it complete.

### Note structure and what it hides

| Section | Typical content | Evaluation risk |
|---|---|---|
| Chief complaint | Reason for visit | Template text |
| History | Symptoms and context | Copy-forward |
| Medications | List and changes | Reconciliation errors |
| Labs | Values and trends | Missing units |
| Assessment | Clinician reasoning | Hedging and uncertainty |
| Plan | Actions and follow-up | Omissions |
| Discharge instructions | Patient guidance | Readability |
| Addenda | Late corrections | Not reflected in the summary |

A summary can be complete against the note structure and still miss an addendum that changes the plan. The evaluation should include addenda and late corrections where they exist.

### Text data quality checks

| Check | Method | Frequency |
|---|---|---|
| De-identification | Manual and automated review | Dataset build |
| Copy-forward text | Duplicate sentence detection | Dataset build |
| Note type | Header and template review | Dataset build |
| Length distribution | Histogram by site | Dataset build |
| Template variation | Cluster templates | Dataset build |
| Language | Language identification | Dataset build |
| Negation | Annotation spot check | Dataset build |
| Section structure | Parser output review | Dataset build |
| Temporal coverage | Date distribution | Dataset build |
| Duplicate patients | Identifier audit | Dataset build |

De-identification removes tokens that can change the clinical meaning. A note that loses a medication name or a date can be unusable for the task. The audit should measure the loss, not only the privacy gain.

### Reporting an evaluation

| Section | Content | Common omission |
|---|---|---|
| Model | Name, version, checkpoint | Checkpoint |
| Prompt | System, examples, format | Prompt text |
| Decoding | Temperature, top-p, max tokens | Settings |
| Benchmark | Name, version, item count | Version |
| Judge | Model, prompt, agreement | Agreement |
| Contamination | Probes and results | Results |
| Interval | Confidence interval | Interval |
| Subgroups | Specialty, difficulty, safety | Safety |
| Cost | Tokens and compute | Cost |
| Licence | Terms and access | Terms |

An evaluation report that omits the prompt and the judge is not reproducible, even if it reports a precise score.

### Regulatory artefacts for a text product

- **Intended use** names the user, the task, the setting, and the decision.
- **Training data** describes the corpus, the licence, and the de-identification.
- **Evaluation** describes the benchmark, the judge, the contamination checks, and the subgroups.
- **Human factors** describes how the clinician reviews and edits the output.
- **Safety** describes the failure taxonomy and the red-teaming.
- **Monitoring** describes drift and incident reporting.

## 9. Industry practice and case evidence

### What organisations actually do

- **Separate capability from safety.** A capability benchmark and a safety evaluation are different instruments.
- **Report the judge.** Mature evaluations publish the judge model, prompt, temperature, and agreement.
- **Build a private holdout.** Public items are treated as contaminated until tested.
- **Use clinician fact audits.** Text similarity is a screen; fact-level review is the evidence.
- **Measure editing time.** A workflow claim needs a time-and-motion study, not a rubric score.
- **Track prompt versions.** A score is a property of the model and the prompt.
- **Report intervals.** Close rankings are treated as ties.

### Case evidence

| Case | Documented fact | Inference |
|---|---|---|
| HealthBench | The benchmark consists of clinician-written conversations with rubrics and was released with an open description and a physician-authored criteria set, as summarised in the module survey. | Rubric-based advice quality is measurable and depends on the rubric and the judge. |
| MedAgentBench | The benchmark provides agent tasks on a FHIR server with synthetic patients and open code, as described by its authors. | Tool-use competence can be measured in a simulated EHR and does not establish safe autonomy. |
| MedQA | The benchmark is a medical exam-style question set released for research. | Exam performance is a knowledge proxy and is vulnerable to pretraining contamination. |
| MIMIC-IV-Note | The notes are de-identified and linked to MIMIC-IV under a credentialed research agreement. | Real notes support methods and are not a commercial evaluation set. |
| i2b2 and n2c2 | The corpora are task-specific clinical NLP datasets released under data use agreements. | Annotation quality is high and the tasks are narrow. |

The first three rows are release-dependent. Verify the current benchmark version and terms before citing. The structural conclusions, that judges and contamination determine interpretability, are stable.

### What good looks like

- The construct and the claim are written before the benchmark is chosen.
- The item source and publication dates are reported.
- Exact and near-duplicate contamination checks are run.
- The judge, prompt, temperature, and agreement are reported.
- The interval and seed variance are reported.
- Safety is a gate, not a weighted dimension.
- The licence covers the evaluation use.
- The data card states the prohibited claims.

### Documented fact versus inference

- **Documented fact:** a benchmark paper, a release page, a licence, or a dataset card.
- **Inference:** a planning consequence for a different product or population.
- **Unverified summary:** a secondary account of a score or a release.

A case table should carry a source and a status. Benchmark pages change, and a stale score can misstate both the capability and the ranking.

### Why leaderboards mislead procurement

A buyer asks whether the product reduces editing time and errors in their workflow. A leaderboard answers whether a model scores well on a fixed item set under a fixed judge. The gap is a difference in construct, and it is closed only by a local evaluation.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Pretraining contamination | Scores are highest on the easiest items | Test items in training | Inflated capability | Temporal or private holdout |
| Exact-match scoring | Valid paraphrases marked wrong | Reference is one string | Understated performance | Rubric or fact-level scoring |
| Judge dependence | Ranking changes with the judge | Judge bias | Unreliable comparison | Validate and report judge |
| Rubric ambiguity | Raters disagree on criteria | Underspecified dimensions | Noise | Calibrate and revise |
| Position bias | First option wins | Pairwise judging | Spurious preference | Swap positions |
| Verbosity bias | Longer answers win | Judge preference | Padding rewarded | Length control |
| Self-preference | Judge favours its own family | Same-model judging | Inflated score | Different judge or panel |
| Unsafe near-miss | Pass rate high, unsafe actions present | Binary task success | False safety | Unsafe-action rate |
| Synthetic patients | Behaviour looks clinical | Simulation | Unsupported claim | Real workflow study |
| Small item set | Wide interval | Sampling | Over-ranked models | Report interval |
| Prompt sensitivity | Score changes with wording | Prompt artefact | Non-reproducible | Fix and report prompt |
| Licence assumption | Research benchmark used for product | Access tier ignored | Compliance breach | Clause review |
| Reference error | Gold answer is wrong | Annotation error | Mismeasured quality | Re-audit sample |
| Benchmark saturation | Top models cluster | Task too easy | Ranking noise | Harder tasks and intervals |
| Copy-forward leakage | Duplicate sentences across notes | Template reuse | Inflated overlap | Deduplicate |
| Note template shortcut | Header predicts the label | Template artefact | Spurious performance | Template audit |
| Retrieval leakage | Retrieved text contains the answer | Test-set overlap | Inflated score | Retrieval audit |
| Prompt injection | Input overrides instructions | Adversarial text | Unsafe output | Injection testing |
| Refusal miscalibration | Refuses benign requests | Over-conservative safety | Poor utility | Refusal audit |
| Truncation | Long notes cut off | Context limit | Missing facts | Chunking and checks |

Red flags that should stop a text benchmark claim:

- The score is reported without the judge.
- The benchmark has no contamination check.
- The item count is under 200 and models are ranked.
- The claim is clinical utility and the benchmark is exam QA.
- The licence is research-only and the use is commercial.
- The pass rate hides unsafe actions.
- The prompt is not reported.
- The reference is a single summary treated as truth.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test |
|---|---|---|---|
| The construct matches the claim | Interpretation depends on it | Adjacent benchmark | Construct audit |
| The items are uncontaminated | Capability estimate depends on it | Pretraining overlap | N-gram and temporal probes |
| The judge is valid | Scores depend on it | Low agreement | Human calibration |
| The rubric is reliable | Scores depend on it | Ambiguous criteria | Inter-rater study |
| The reference is correct | Correctness depends on it | Annotation error | Re-audit sample |
| The item count is adequate | Intervals depend on it | Small set | Report interval |
| The prompt is fixed | Reproducibility depends on it | Prompt drift | Version control |
| The decoding settings are fixed | Reproducibility depends on it | Temperature change | Record settings |
| The licence covers the use | Compliance depends on it | Research-only terms | Clause review |
| The benchmark version is pinned | Comparability depends on it | Silent update | Record version |
| The workflow is represented | Utility depends on it | No time study | Prospective evaluation |
| The subgroup is large enough | Equity analysis depends on it | Small counts | Confidence intervals |
| The notes are not truncated | Completeness depends on it | Context limit | Length audit |
| Retrieval is clean | Grounding depends on it | Test overlap | Retrieval audit |
| Templates are not shortcuts | Validity depends on it | Header predicts label | Template audit |
| Refusals are calibrated | Utility depends on it | Over-conservative | Refusal audit |
| Prompts resist injection | Safety depends on it | Adversarial input | Injection test |
| The judge is not the model | Independence depends on it | Self-preference | Different judge |
| The rubric is length-neutral | Style balance depends on it | Verbosity bias | Length control |
| The test set is private | Overfitting control depends on it | Repeated tuning | Holdout |
| The benchmark is current | Relevance depends on it | Retired items | Version check |

The deepest limit is that a benchmark measures a construct, not a patient outcome. A model can improve a benchmark score and fail to improve care, and a model can regress on a benchmark and still be useful in a narrow workflow.

## 12. Exercises

**Exercise 1. Classify benchmark constructs.**
Deliverable: a table mapping six benchmarks to constructs, units, and claims.
Acceptance criteria: six benchmarks; one construct mismatch identified; one claim rejected.
Time: 35 minutes.

**Exercise 2. Run a contamination audit.**
Deliverable: an n-gram overlap report with flagged items and a score with and without them.
Acceptance criteria: reproducible script; overlap distribution; a decision on the flagged items.
Time: 50 minutes.

**Exercise 3. Validate a judge.**
Deliverable: a human-rated sample, a judge run, and a kappa with disagreement cases.
Acceptance criteria: at least 100 items; judge configuration reported; limitations stated.
Time: 75 minutes.

**Exercise 4. Compute intervals.**
Deliverable: a table of pass rates and Wilson intervals for three item counts.
Acceptance criteria: arithmetic correct; a ranking decision justified; assumptions stated.
Time: 30 minutes.

**Exercise 5. Map licences.**
Deliverable: a clause matrix for MIMIC-IV-Note, i2b2, and HealthBench covering training, evaluation, and release.
Acceptance criteria: three resources; each cell marked; the blocking clause named.
Time: 45 minutes.

**Exercise 6. Design a rubric.**
Deliverable: dimensions, scales, a safety gate, and a calibration plan.
Acceptance criteria: separate accuracy and safety; gate defined; reliability test proposed.
Time: 50 minutes.

**Exercise 7. Audit an agent benchmark.**
Deliverable: a table of task success, unsafe near-misses, and tool-call errors.
Acceptance criteria: unsafe actions counted separately; near-miss taxonomy; a safety recommendation.
Time: 45 minutes.

**Assessment note.** Exercises 2, 3, 4, and 5 produce checkable artefacts: an overlap report, a judge agreement table, an interval table, and a licence matrix. Exercises 1, 6, 7, and 8 are design tasks and are marked against the rubric. Exercise 8 is the integration task.

**Exercise 8. Write a validity memo.**
Deliverable: a two-page memo with construct, items, judge, contamination, intervals, and prohibited claims.
Acceptance criteria: every element present; one quantified limitation; no unsupported ranking.
Time: 60 minutes.

## 13. Self-check answers

**Item 1. Why is a medical exam benchmark not a clinical reasoning benchmark?**
It measures performance on exam-style items, which correlates with knowledge and does not test the integration of history, examination, and context under uncertainty. A strong answer names the construct gap.

**Item 2. What is contamination, and why does it matter?**
Test items or close paraphrases in training data. It matters because the score then measures memorisation as well as capability, and the two cannot be separated without a clean holdout.

**Item 3. Why is exact-match scoring weak for summarization?**
Many valid summaries differ in wording. Exact match penalises valid alternatives and rewards copying the reference. A fact-level or rubric evaluation is needed.

**Item 4. What does a Cohen's kappa of 0.62 mean?**
Substantial agreement after chance correction, and not high agreement. It means the judge and the human disagree on a meaningful fraction of items, so the score carries error.

**Item 5. Why report a confidence interval for a pass rate?**
Because the observed rate is an estimate from a finite item set. The interval shows how much the estimate would vary with a different sample and whether a ranking is supported.

**Item 6. Why is an LLM judge a measurement instrument?**
It has a prompt, a model, a temperature, and a bias profile. Its output is not ground truth. It must be validated against human grades and reported with its configuration.

**Item 7. What does an agent benchmark measure, and what does it not?**
It measures task completion and tool-call correctness in a simulated environment. It does not measure safe autonomy, clinical judgement, or behaviour in a real EHR.

**Item 8. Why is a research-only benchmark unsuitable for a product claim?**
Because the licence does not permit commercial evaluation or the retention of items for product development. The evidence would be unusable in diligence or a regulatory file.

**Item 9. What is benchmark saturation?**
Top models cluster within noise, so the ranking is unstable and a new record is not meaningful. The response is harder tasks and interval reporting, not a new leaderboard.

**Item 10. When is a synthetic patient benchmark acceptable?**
For method development, tool-use testing, and interface evaluation. It is not acceptable for clinical behaviour, safety, or outcome claims.

**Item 11. Why is a high score on a public benchmark not evidence of generalization?**
The benchmark's items, population, and scoring rule define a narrow setting. Generalization requires a different sample, a different site, and often a different judge. A high score is evidence about the benchmark.

**Item 12. What is the difference between a rubric score and a fact audit?**
A rubric score is a judgement against criteria; a fact audit checks each claim against the source. The first can be fluent and wrong; the second is slower and more accurate for factual errors.

**Item 13. Why does copy-forward text matter for summarization evaluation?**
Copy-forward duplicates text across notes, so a model can reproduce it without understanding. It also creates near-duplicates between training and test and inflates overlap metrics.

**Item 14. What is the difference between a capability failure and a safety failure?**
A capability failure is a wrong or incomplete answer; a safety failure is an answer that could cause harm. A model can be capable and unsafe, and the evaluation must report them separately.

**Item 15. Why should ties be allowed in pairwise evaluation?**
Many outputs are equivalent in quality, and forcing a choice creates noise and position bias. Allowing ties produces a more honest preference estimate.

**Item 16. What does a private holdout protect against?**
Repeated tuning against a public test set. It does not protect against contamination in pretraining, which requires a temporal or genuinely unseen set.

**Item 17. Why is an item count of 100 a problem for ranking?**
The confidence interval is wide, often plus or minus several percentage points, so small differences between models are not resolvable. Ranking requires more items or a paired design.

**Item 18. Why is a refusal rate a safety metric and a utility metric?**
A model that refuses too often is safe and useless; a model that never refuses is useful and unsafe. The rate must be evaluated against the task's tolerance for refusal.

**Item 19. What is prompt injection in a clinical text pipeline?**
Text in a note or retrieved document instructs the model to ignore its rules or reveal data. The model may comply because it cannot distinguish data from instructions. The defence is input sanitisation and instruction hierarchy.

**Item 20. Why does truncation matter for note summarization?**
A long note that exceeds the context window is cut, and facts after the cut are invisible. The summary may be fluent and omit the most important late finding.

**Item 21. What is the difference between a benchmark score and a decision curve?**
A benchmark score summarises accuracy on items; a decision curve evaluates net benefit across thresholds for a clinical decision. The first does not encode the costs of false positives and negatives.

**Rubric for the design exercises (2, 3, 6, 8).** Full marks require a named construct, a contamination check, a validated judge, and a prohibited-claim statement. Partial marks for a complete structure without judge validation. No marks for a leaderboard rank presented as clinical utility.

## 14. Interview questions

**1. How would you evaluate a clinical summarization model?**
Define the construct, use real notes under a suitable licence, build a fact-level audit, measure editing time, and report intervals. A strong answer separates factual accuracy from style. A weak answer reports ROUGE.

**2. A vendor reports a high MedQA score. What do you ask?**
The benchmark version, the contamination checks, the prompt, the item count, the interval, and whether the claim is knowledge or clinical judgement. A strong answer asks for a temporal holdout. A weak answer accepts the score.

**3. Why is an LLM judge not ground truth?**
It is a model with biases and error. It must be validated against clinicians and reported with its configuration. A strong answer reports kappa. A weak answer says the judge is objective.

**4. What is the difference between capability and safety evaluation?**
Capability asks whether the model can do the task; safety asks whether it fails dangerously. A strong answer uses separate instruments and a safety gate. A weak answer treats a high capability score as safety evidence.

**5. How do you detect contamination?**
N-gram overlap, near-duplicate search, date-based holdout, canary strings, and paraphrase probes. A strong answer reports the overlap distribution. A weak answer checks exact matches only.

**6. Why can a two-point leaderboard difference be meaningless?**
The confidence interval on a finite item set can be wider than the difference, and seed and prompt variance add more. A strong answer computes the interval. A weak answer ranks by point estimate.

**7. What does MedAgentBench not measure?**
Safe autonomy, clinical judgement, and real EHR behaviour. A strong answer names the simulated environment. A weak answer treats task success as deployment readiness.

**8. How do you handle a research-only note corpus for a product?**
Use it for methods, and obtain commercial-permissive or partner data for the product evaluation. A strong answer maps the licence to activities. A weak answer assumes de-identification grants commercial rights.

**9. What is rubric dependence?**
The score depends on the rubric's dimensions and weights. A different rubric can rank models differently. A strong answer tests reliability and reports the rubric. A weak answer treats the score as objective.

**10. When would you stop using a benchmark?**
When it is saturated, contaminated, or construct-mismatched, or when its licence no longer fits. A strong answer names the trigger. A weak answer keeps using it because it is familiar.

**11. How do you report a negative LLM evaluation?**
Report the construct, the items, the judge, the interval, and the failure categories. A strong answer distinguishes absence of capability from absence of evidence. A weak answer reports only the mean score.

**12. What belongs in a benchmark validity memo?**
Construct, items, reference, judge, agreement, contamination, interval, subgroups, licence, and prohibited claims. A strong answer is specific about the weakest link. A weak answer summarizes the leaderboard.

**13. How do you evaluate an LLM that retrieves from a guideline?**
Check retrieval accuracy, grounding of each claim, completeness against the guideline, and safety. A strong answer separates retrieval from generation. A weak answer scores the final text only.

**14. What would you ask before using a benchmark in a regulatory file?**
The version, the licence, the construct, the contamination checks, the judge validation, the interval, and the subgroup results. A strong answer asks whether the benchmark is fit for the intended use. A weak answer asks for the score.

**15. Why is a synthetic-patient benchmark useful and insufficient?**
It is useful for tool use, interface testing, and edge cases, and insufficient for clinical behaviour and safety. A strong answer names both. A weak answer treats simulation as deployment.

**16. How do you handle a benchmark that has been retired?**
Pin the version, keep the results, and state the retirement. A strong answer does not compare across a retired and a current benchmark without a task audit. A weak answer keeps citing the old number.

**17. What is the difference between a rubric and a reference answer?**
A rubric lists criteria and scales; a reference answer is one example of a correct response. A rubric can accommodate multiple valid answers; a reference answer cannot.

**18. When is human evaluation worth the cost?**
When the claim depends on judgement, safety, or workflow, and when an automated judge has not been validated. A strong answer uses a sample and a pre-specified protocol. A weak answer uses human review without a design.

## 15. Key terms

- **Agreement rate** — raw proportion of agreement between raters.
- **Benchmark** — a fixed task, dataset, and metric.
- **Canary string** — a marker inserted to detect leakage.
- **Cohen's kappa** — chance-corrected agreement.
- **Confidence interval** — uncertainty around an estimate.
- **Construct** — the capability a benchmark intends to measure.
- **Contamination** — test items present in training data.
- **Date cutoff** — the model's training cutoff.
- **Gold answer** — the reference answer.
- **HealthBench** — clinician-written conversations with rubrics.
- **i2b2** — clinical informatics research programme.
- **Item** — one question or task instance.
- **Judge** — the model or person assigning grades.
- **MedAgentBench** — agent tasks on a FHIR server.
- **MedMCQA** — medical entrance exam QA dataset.
- **MedQA** — US medical licensing exam-style QA.
- **MIMIC-IV-Note** — de-identified notes linked to MIMIC-IV.
- **n2c2** — clinical NLP challenge series.
- **Near-duplicate** — a paraphrase or close variant.
- **Pass rate** — share of items scored correct.
- **Position bias** — preference for an option's position.
- **Private holdout** — items never released.
- **Prompt sensitivity** — score change from prompt wording.
- **PubMedQA** — QA over biomedical abstracts.
- **Rubric** — criteria and weights for grading.
- **Saturation** — top models cluster within noise.
- **Self-preference** — a judge favouring its own outputs.
- **Temporal holdout** — items published after the cutoff.
- **Verbosity bias** — preference for longer answers.
- **Wilson interval** — a binomial confidence interval.
- **Win rate** — share of pairwise comparisons won.
- **Abstention** — declining to answer.
- **Alignment** — agreement with human preferences.
- **Benchmark leakage** — the benchmark present in training.
- **Chain-of-thought** — intermediate reasoning text.
- **Context window** — maximum tokens processed.
- **Copy-forward** — duplicated text across notes.
- **De-identification** — removing identifiers.
- **Discharge summary** — summary written at discharge.
- **Factuality** — correctness of stated facts.
- **Few-shot** — examples included in a prompt.
- **Fine-tuning** — further training on task data.
- **Grounding** — traceability to a source.
- **Hallucination** — fabricated content.
- **Human evaluation** — clinician or expert review.
- **Instruction tuning** — training on instructions.
- **Item difficulty** — the share of models answering correctly.
- **Judge prompt** — instruction given to a judge.
- **Length control** — adjusting for answer length.
- **Likert scale** — an ordered rating scale.
- **Model card** — documentation of a model.
- **Note type** — the category of a clinical note.
- **Pairwise comparison** — choosing between two outputs.
- **Position bias** — preference by order.
- **Precision** — positive predictive value.
- **Prompt injection** — malicious instructions in input.
- **Red-teaming** — adversarial safety testing.
- **Recall** — sensitivity.
- **Reference standard** — the best available truth.
- **Refusal** — declining an unsafe request.
- **Retrieval-augmented generation** — generation grounded in retrieved text.
- **ROUGE** — overlap-based summarization metric.
- **Safety gate** — a pass/fail safety criterion.
- **Seed variance** — variation across random seeds.
- **Self-consistency** — agreement across samples.
- **System prompt** — persistent instruction.
- **Temperature** — sampling randomness.
- **Token** — a unit of text.
- **Toxicity** — harmful language.
- **Unsafe action rate** — harmful actions per task.
- **Verbosity** — answer length.
- **Zero-shot** — no examples in the prompt.
- **Addendum** — a late addition to a note.
- **Assessment** — the clinician's reasoning section.
- **Chief complaint** — the reason for the visit.
- **Discharge instruction** — patient guidance at discharge.
- **Evaluation protocol** — the pre-specified evaluation plan.
- **Fact audit** — checking each claim against the source.
- **Item bank** — the collection of benchmark items.
- **Length-neutral** — not biased by answer length.
- **Note template** — the structure imposed by the EHR.
- **Pre-specified** — defined before seeing results.
- **Ranking stability** — whether the order survives resampling.
- **Refusal audit** — measuring refusals on benign requests.
- **Safety gate** — a pass/fail safety criterion.
- **Truncation** — cutting input at the context limit.
- **Unsafe near-miss** — an action close to harm.
- **Validity memo** — a document assessing benchmark validity.
- **Workflow study** — a study of use in real work.
- **Clinician panel** — a group of expert raters.
- **Preference elicitation** — collecting comparative judgements.
- **Benchmark card** — documentation of a benchmark.
- **Ceiling effect** — scores near the maximum.
- **Floor effect** — scores near the minimum.
- **Item exposure** — how widely items are known.
- **Judge ensemble** — multiple judges combined.
- **Label noise** — error in the reference.
- **Parsing error** — failing to extract the answer.
- **Prompt template** — the reusable prompt structure.
- **Rubric weight** — the importance of a dimension.
- **Score aggregation** — combining item scores.
- **Test-retest reliability** — stability across runs.
- **Unsafe action** — an action that could harm.
- **Validity** — whether the benchmark measures the construct.
- **Variance decomposition** — separating item, model, and judge variance.
- **Workflow fit** — compatibility with clinical work.
- **Benchmark retirement** — withdrawing a benchmark from use.

## 16. Sources

All URLs accessed 26 October 2026. Benchmark versions and terms change; verify at the official page.

1. PhysioNet. MIMIC-IV-Note. https://physionet.org/content/mimic-iv-note/
2. Harvard DBMI. i2b2 and n2c2 projects. https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
3. MedQA repository. https://github.com/jind11/MedQA
4. MedMCQA. https://medmcqa.github.io/
5. PubMedQA. https://pubmedqa.github.io/
6. OpenAI. HealthBench. https://openai.com/index/healthbench/
7. Stanford. MedAgentBench. https://stanfordmlgroup.github.io/projects/medagentbench/
8. US Department of Health and Human Services. HIPAA. https://www.hhs.gov/hipaa/
9. European Parliament and Council. Regulation (EU) 2016/679 (GDPR). https://eur-lex.europa.eu/eli/reg/2016/679/oj
10. US Food and Drug Administration. AI-enabled device software functions guidance, draft January 2025. https://www.fda.gov/media/184856/download
11. European Parliament and Council. Regulation (EU) 2024/1689 (AI Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
12. Collins GS, Moons KGM, Dhiman P, et al. TRIPOD+AI statement. *BMJ* 2024;385:e078378. https://doi.org/10.1136/bmj-2023-078378
13. HL7. FHIR R4. https://hl7.org/fhir/R4/
14. National Library of Medicine. UMLS. https://www.nlm.nih.gov/research/umls/
15. SNOMED International. SNOMED CT. https://www.snomed.org/
16. Regenstrief Institute. LOINC. https://loinc.org/
17. Office of the National Coordinator for Health Information Technology. HTI-1 final rule. https://www.healthit.gov/rule/hti-1
18. Module 21 survey file, section 10. [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)

## 17. Links

- Module map and topic table: [`_index.md`](_index.md)
- Previous topic, pathology datasets: [`07_pathology-datasets.md`](07_pathology-datasets.md)
- Omics datasets: [`06_omics-datasets.md`](06_omics-datasets.md)
- Access and licensing: [`02_access-and-licensing.md`](02_access-and-licensing.md)
- Root survey file: [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
- Labs, including the benchmark validity audit: [`../../labs/`](../../labs/)

