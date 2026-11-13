# Grading Calibration

> **Purpose:** Make grading consistent across instructors, cohorts, and terms.
> **Version:** 2026.5.0
> **Use with:** `INSTRUCTOR_KIT.md`, the module rubrics, and `COMPETENCY_MAP.md`

---

## 1. Why calibration is necessary

Two instructors grading the same domain project can differ by a full letter grade without disagreeing about the facts. One rewards model sophistication; another rewards decision relevance. One treats a missing baseline as a minor omission; another treats it as disqualifying.

Calibration aligns three things:

1. **What the rubric means** at each level.
2. **What evidence** counts as proof.
3. **What the program values** when criteria conflict.

The program values, in order: a real decision, honest evidence, feasible action, and technical craft. A sophisticated model on a fabricated decision is not a strong submission.

## 2. The anchor principle

Before grading a cohort, grade two anchor submissions together:

- **Anchor A — competent:** correct decision framing, credible baseline, one defensible method, honest limits.
- **Anchor B — expert:** the same, plus a counterfactual, sensitivity analysis, and a clear operational path.

Instructors score both independently, compare, and resolve differences above one rubric level. Record the resolved score as the cohort anchor. Every later submission is judged relative to the anchors, not to an abstract ideal.

## 3. Level definitions

| Level | Score | What it means |
|---|---:|---|
| Exemplary | 90–100 | Could be presented to an executive or review board with minor edits |
| Strong | 80–89 | Decision-ready; small gaps in evidence or craft |
| Competent | 70–79 | Correct core; missing depth, sensitivity, or adoption detail |
| Developing | 60–69 | Partial; the decision or evidence is incomplete |
| Insufficient | below 60 | Wrong decision, fabricated evidence, or no reproducible analysis |

A submission at 70 is not a failure. It is a correct piece of work that would need a senior reviewer before use.

## 4. Criterion-level anchors

### 4.1 Decision framing

| Level | Evidence |
|---|---|
| Exemplary | Names the decision, owner, cadence, cost of error, and the counterfactual; the model changes a specific choice |
| Strong | Names the decision and owner; links the model output to an action |
| Competent | Names a business problem but not a decision |
| Developing | Describes a dataset and a method with no decision |
| Insufficient | The stated decision is not one the organization can make |

### 4.2 Data fitness

| Level | Evidence |
|---|---|
| Exemplary | Grain, coverage, censoring, latency, and provenance are documented; known gaps are quantified |
| Strong | Grain and coverage documented; major gaps named |
| Competent | Data described but grain or censoring unexamined |
| Developing | Data used without a data card |
| Insufficient | Data is fabricated, mislabeled, or used outside its license |

### 4.3 Baseline and validation

| Level | Evidence |
|---|---|
| Exemplary | Naive and current-process baselines; rolling-origin or temporal validation; uncertainty reported; leakage checked |
| Strong | A credible baseline and a temporal holdout |
| Competent | A baseline, but validation is a random split or single holdout |
| Developing | No baseline, or only an in-sample score |
| Insufficient | Validation leaks future information and the error is not acknowledged |

### 4.4 Method and craft

| Level | Evidence |
|---|---|
| Exemplary | Method fits the decision; assumptions tested; sensitivity analysis; code reproducible |
| Strong | Appropriate method; main assumptions stated; code runs |
| Competent | Method works but is over- or under-engineered |
| Developing | Method applied mechanically; results not reproducible |
| Insufficient | Method cannot answer the question, or the code does not run |

### 4.5 Impact and feasibility

| Level | Evidence |
|---|---|
| Exemplary | Value ranges with assumptions; implementation cost and adoption path; failure modes and guardrails |
| Strong | A defensible value estimate and an owner |
| Competent | An estimate without sensitivity or adoption |
| Developing | Claims savings without a mechanism |
| Insufficient | Fabricated or double-counted value |

### 4.6 Communication and defense

| Level | Evidence |
|---|---|
| Exemplary | Decision-ready memo; answers challenges with evidence; states what would change the conclusion |
| Strong | Clear memo; answers most questions |
| Competent | Understands the work but cannot defend choices |
| Developing | Reads slides without ownership |
| Insufficient | Cannot explain the analysis or its sources |

### 4.7 Domain fulfillment design

| Level | Evidence |
|---|---|
| Exemplary | Names the node, promise, cost-to-serve, and returns loop; quantifies the service-cost frontier and the failure mode at peak |
| Strong | A defensible node placement or routing policy with a promise and cost estimate |
| Competent | A fulfillment model with no promise or unit-economics link |
| Developing | A generic network diagram relabeled as e-commerce |
| Insufficient | Assumes infinite capacity, free delivery, or zero returns |

### 4.8 Recommendation and ranking quality

| Level | Evidence |
|---|---|
| Exemplary | Offline and online evaluation; popularity and random baselines; constraint-aware re-ranking; feedback-loop and cold-start analysis |
| Strong | A ranking model with Recall@k or NDCG and a business constraint |
| Competent | A model evaluated only on offline accuracy |
| Developing | A recommender with no baseline or no exposure analysis |
| Insufficient | Optimizes clicks while ignoring stockouts, margin, or fairness |

## 5. Conflict rules

When criteria conflict, apply these rules in order:

1. **Fabrication outranks craft.** A fabricated citation, dataset, or result is an automatic fail regardless of presentation.
2. **Decision outranks method.** A simple method on a real decision beats a complex method on no decision.
3. **Validation outranks accuracy.** A modest, honestly validated result beats an impressive unvalidated one.
4. **Feasibility outranks optimality.** An implementable plan beats a mathematically optimal one that cannot be executed.
5. **Honesty outranks confidence.** A stated limitation is a strength, not a weakness.

## 6. Common grading errors

| Error | Why it happens | Fix |
|---|---|---|
| Rewarding complexity | Sophistication is visible; relevance is not | Grade the decision first |
| Accepting a random split | It looks like validation | Check the split by time |
| Ignoring censoring | Stockout rows look like low demand | Ask how zero sales were treated |
| Over-crediting dashboards | Visual polish reads as insight | Ask what decision changed |
| Penalizing simple methods | Simple looks unsophisticated | Ask whether it beat the baseline |
| Inconsistent domain-project scope | Teams choose different sizes | Use the scope card |
| Grading the team, not the individual | Group work hides contribution | Use the individual defense |
| Late-rubric drift | Criteria reinterpreted mid-term | Anchor before grading |

## 7. Individual contribution

For team work, require:

- A contribution log with artifacts, not hours.
- One section of the final memo owned by each member.
- A short individual defense: "What did you decide, and what would you do differently?"

Score the team artifact on the team rubric and the defense on individual understanding. A student who cannot explain the model did not learn the model.

## 8. Feedback standard

Every grade above 69 includes:

1. One thing that worked, with the evidence.
2. One thing that would raise the level, with the evidence.
3. One question the student should be able to answer.

Feedback that could apply to any submission is not feedback.

## 9. Re-grading and appeals

- Re-grade requests are accepted within five working days.
- The reviewer regrades from the rubric, not from the original score.
- If the two scores differ by more than one level, a third reviewer anchors.
- The outcome and reason are recorded in the cohort log.

## 10. Calibration log template

| Date | Cohort | Artifact | Reviewers | Anchor score | Variance | Resolution |
|---|---|---|---|---:|---:|---|
| | | | | | | |

## 11. Review cycle

Recalibrate when:

- A new instructor joins.
- A rubric changes.
- More than 20% of submissions fall in one level.
- A domain-project archetype changes.

Record changes in `CHANGELOG.md`.

> **Next:** Use the module rubrics with these anchors, and record cohort outcomes in the calibration log.
