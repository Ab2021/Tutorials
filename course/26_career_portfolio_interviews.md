# Module 26 — Career, Portfolio, and Interviews

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 14 hours (4 lecture, 4 workshop, 6 portfolio)
> **Track fit:** Common core
> **Prerequisites:** Modules 00–10; at least one completed capstone-style project
> **Tools:** Git, a portfolio site or repository, a resume, a case-study template
> **Industry question:** "Can you prove, in twenty minutes and with three artifacts, that you can improve a supply chain decision?"

---

## 1. Why this module exists

Supply chain analytics hiring is unusual. A candidate may be interviewed by a demand planner, a data engineering lead, an operations research scientist, a finance partner, and a hiring manager in the same week. Each tests a different competence:

- The planner wants to know whether you understand the planning cycle and the exceptions.
- The engineer wants to know whether your pipeline is reproducible and your SQL is correct.
- The OR scientist wants to know whether your formulation is valid and your solver result is feasible.
- Finance wants to know whether you can value an improvement without double counting.
- The manager wants to know whether you can scope, communicate, and finish.

A generic "I built a model with 95% accuracy" answer fails all five. A portfolio that shows a decision, a baseline, a validation, an impact estimate, and a deployment path passes all five.

This module turns the program into a credible professional narrative. It is not about marketing. It is about **evidence selection and clear communication**.

## 2. Learning outcomes

By the end of this module, you can:

- Map your experience to the supply chain analytics role families and their actual work.
- Select and structure three portfolio artifacts that demonstrate different competencies.
- Write a case study in decision → data → method → evidence → impact → limits format.
- Tailor a resume and LinkedIn profile to a specific role without overclaiming.
- Answer SQL, forecasting, optimization, case, and behavioral questions with structured reasoning.
- Design and defend a take-home analysis under a time limit.
- Explain a planning-system architecture at a whiteboard.
- Negotiate a role, scope, and compensation with accurate market context.
- Build a 90-day onboarding plan that creates value before the first review.

## 3. Mental model: the evidence portfolio

```
Role family → Competencies → Artifacts → Stories → Interview answers
```

Work backward. Start with the role you want, list the competencies the role actually uses, choose artifacts that prove them, and prepare stories that make the artifacts legible.

### 3.1 Role families

| Role | Core work | What interviewers test | Portfolio artifact |
|---|---|---|---|
| Demand planner | Build and review forecasts; manage exceptions | Planning judgment, forecast review, stakeholder handling | Forecast review pack with FVA |
| Supply chain analyst | Reporting, analysis, process improvement | SQL, Excel, metrics, business framing | Decision-support analysis |
| Data scientist | Forecast, classify, optimize, deploy | Python, statistics, validation, communication | End-to-end model with backtest |
| Operations research scientist | Formulate and solve planning problems | LP/MIP, algorithms, feasibility, sensitivity | Optimization study with solver log |
| ML engineer | Build and serve models | Engineering, pipelines, monitoring, latency | Deployed service and monitoring |
| Product manager | Define and deliver analytics products | Discovery, prioritization, value case | Product brief and roadmap |
| Consultant | Frame, analyze, recommend | Structuring, client communication, speed | Case study and executive memo |
| Supply chain analytics lead | Build the function | Strategy, hiring, governance, adoption | Program design and value case |

### 3.2 The three-artifact rule

Choose three artifacts, each proving a different axis:

1. **Depth**: a rigorous forecast, optimization, or causal analysis.
2. **Engineering**: a reproducible pipeline, service, or deployment.
3. **Impact**: a value case, adoption story, or operational decision that changed.

If all three artifacts are notebooks, you look like a student. If all three are dashboards, you look like a reporter. If all three are slide decks, you look like a consultant who cannot build.

## 4. Portfolio construction

### 4.1 Artifact template

Each artifact should have a one-page README:

```markdown
# Project title

## Decision
Who acted, on what cadence, at what grain and horizon?

## Data
Source, period, grain, known defects, license.

## Baseline
Current method and its measured performance.

## Method
Why this method; what alternatives were rejected.

## Validation
Split, backtest, subgroup, stress test.

## Result
Primary metric with uncertainty; cost/service/cash impact.

## Limits
What would break it; what you would do next.

## Artifacts
Notebook, SQL, model card, dashboard, memo.
```

### 4.2 What to show

| Show | Do not show |
|---|---|
| A decision and a baseline | A model with no comparator |
| A temporal backtest | A random split on time-series data |
| Error analysis and bias | A single aggregate metric |
| A feasible plan | An optimization with no constraints |
| A cost or service estimate | "Accuracy improved by 12%" |
| A deployment sketch | A notebook with no path to use |
| Honest limits | A claim that cannot be falsified |

### 4.3 Repository hygiene

- A README that states the decision and how to run the project.
- A pinned environment or container.
- Data instructions that do not redistribute restricted data.
- Tests for the data pipeline and key calculations.
- A `results/` folder with the exact metrics and dates.
- A model card for any predictive model.
- No secrets, credentials, or personal data.

## 5. The case study

### 5.1 Six-part structure

1. **Situation**: the supply chain, the decision, the cost of failure.
2. **Complication**: what was wrong with the current method.
3. **Question**: the specific decision and metric.
4. **Approach**: data, baseline, method, validation.
5. **Evidence**: results with uncertainty and a counterfactual.
6. **Action**: what you recommended, what changed, what you would do next.

### 5.2 A worked outline

> A grocery retailer had a 91% fill rate on fresh produce and 11% waste. The demand plan was a spreadsheet average with manual overrides. Planners overrode 38% of SKU-store forecasts, mostly upward, which raised waste. We built a 14-day quantile forecast at SKU-store-day, backtested over 26 weeks, and converted the 80th-percentile forecast into a shelf-life-constrained replenishment policy. At constant fill rate, simulated waste fell 1.8 percentage points; at constant waste, fill rate rose 1.1 points. The pilot is limited to two distribution centres and does not yet include supplier lead-time variability.

This outline is credible because it names the decision, the baseline, the override behavior, the validation, the trade-off, and the limit.

### 5.3 Common case-study mistakes

| Mistake | Why it fails | Fix |
|---|---|---|
| Starting with the model | Interviewer cannot see the decision | Start with the decision and cost |
| No baseline | No way to judge improvement | Reproduce current method |
| Random split on time series | Leakage | Rolling-origin backtest |
| One metric | Hides bias and tail risk | Report bias, quantiles, cost |
| No constraints | Plan is infeasible | Show capacity, MOQ, lead time |
| Hero narrative | Implausible and unverifiable | State team role and limits |
| No impact | Looks academic | Estimate service, cost, or cash |
| No next step | Looks finished but untested | Propose a pilot or monitoring plan |

## 6. Resume and LinkedIn

### 6.1 Resume structure for supply chain analytics

```
Name and contact
Summary: 2 lines, role + domain + evidence
Skills: SQL, Python, forecasting, optimization, tools
Experience: 3–5 bullets per role, each with decision + method + result
Projects: 2–3 portfolio artifacts with links
Education and certifications
```

### 6.2 Bullet formula

> **Action + decision + method + quantified result + constraint.**

Example:

> Rebuilt SKU-store replenishment for 1,200 stores using a hierarchical LightGBM forecast and a shelf-life-constrained policy; reduced waste 1.8 points at constant fill rate in a two-DC pilot.

Weak version:

> Used machine learning to improve forecasting.

### 6.3 Honesty rules

- Do not claim a production deployment if it was a pilot.
- Do not claim a dollar saving you did not measure.
- Name your role on a team project.
- State the data limitation.
- Do not list a tool you cannot use in an interview.

## 7. Interview types

### 7.1 SQL

Expect joins, window functions, CTEs, deduplication, date spines, and slowly changing dimensions.

**Sample question:** "Given orders, shipments, and inventory snapshots, calculate weekly fill rate by DC for the last 12 weeks."

A strong answer:

1. Clarify the fill-rate definition: units shipped / units demanded, or orders filled complete.
2. Check grain and duplicates.
3. Build a date spine to include weeks with zero demand.
4. Use a left join from demand to shipments.
5. Handle returns and cancellations explicitly.
6. Show the query and a sanity check.

### 7.2 Forecasting

**Sample question:** "How would you forecast demand for a new SKU with eight weeks of history?"

A strong answer:

- Check whether the series is launch, substitution, or replenishment.
- Use analogous products and attribute-based methods.
- Set a wide interval; do not pretend to precision.
- Backtest on other launches, not on the new SKU.
- Define the decision: initial buy, allocation, or replenishment.
- State the cost of over- and under-forecasting.

### 7.3 Optimization

**Sample question:** "Design a model to assign orders to DCs."

A strong answer:

- Define sets, parameters, and decision variables.
- Objective: minimize transportation plus handling cost, or maximize service.
- Constraints: capacity, inventory, service window, split-shipment rules.
- Discuss integrality and solve time.
- Discuss infeasibility and sensitivity.
- Validate the solution against current practice.

### 7.4 Case

**Sample question:** "A retailer has 20% stockouts and 60 days of inventory. What do you do?"

A strong answer:

- Segment by ABC/XYZ and channel.
- Check master data and inventory accuracy.
- Separate forecast bias from allocation and replenishment policy.
- Check lead-time variability and DC-store allocation.
- Check shelf execution and substitution.
- Propose a diagnostic plan before a model.

### 7.5 Behavioral

Use STAR, but keep it specific and quantitative.

**Sample question:** "Tell me about a time your analysis was rejected."

A strong answer shows:

- The stakeholder's real objection (often trust, workload, or incentives).
- What you changed (grain, explanation, pilot).
- The result.
- What you learned about adoption.

### 7.6 Take-home

A typical take-home gives a dataset and asks for a forecast, analysis, or recommendation.

Rules:

- Read the brief twice; restate the decision.
- Ask clarifying questions early.
- Build a baseline first.
- Timebox the model; spend time on validation and communication.
- Write a one-page memo with limits.
- Do not overfit to the provided sample.
- Submit a reproducible notebook and a short README.

## 8. System design for planning systems

### 8.1 Whiteboard structure

```
Sources → Ingestion → Warehouse → Feature/planning layer → Model → Decision service → Planning tool → Monitoring
```

Cover:

- Batch versus streaming and the planning cadence.
- Data contracts and late-arriving data.
- Master-data resolution.
- Training and scoring schedule.
- Model registry and approval.
- Human override and audit trail.
- Monitoring and rollback.
- Security and access.

### 8.2 Common follow-ups

- How do you handle a late shipment file?
- How do you backfill a corrected forecast?
- How do you prevent a planner from overriding without a reason code?
- How do you detect drift when actuals arrive late?
- How do you roll back a bad model on a Tuesday morning?

## 9. Negotiation and onboarding

### 9.1 Negotiation

Prepare:

- Your evidence: three artifacts and a value story.
- Your scope: the decision you want to own.
- Your market range: use dated, sourced comparisons, not a single salary site.
- Your trade-offs: title, scope, remote, learning, compensation.

Ask:

- What does success look like in 6 and 12 months?
- Which decisions will I own?
- What data access will I have?
- How is the team measured?
- What is the on-call and planning-cycle expectation?

Do not:

- Invent a competing offer.
- Negotiate without knowing the scope.
- Optimize only for base salary and ignore scope and learning.

### 9.2 Ninety-day plan

| Phase | Days | Focus | Output |
|---|---:|---|---|
| Learn | 1–30 | Shadow planners, map data, read the plan | Decision and data map |
| Diagnose | 31–60 | Baseline, quality audit, quick wins | Baseline report |
| Deliver | 61–90 | One decision improved, one artifact shipped | Pilot or decision change |

A strong 90-day plan names the decision, the metric, the stakeholders, and the evidence you will produce.

## 10. Labs

### Lab 26.1 — Role and competency map

**Objective:** Map your experience to a target role.

**Steps:** Choose three job postings; extract competencies; rate yourself 1–5 with evidence; identify two gaps; choose artifacts to close them.

**Deliverable:** One-page role map and gap plan.

**Acceptance criteria:** Every rating has evidence; gaps have a specific artifact and deadline.

### Lab 26.2 — Portfolio README

**Objective:** Turn one project into a hiring artifact.

**Steps:** Fill the artifact template; add a baseline, validation, impact, and limits; test it on a peer who is not in your domain.

**Deliverable:** README and a two-minute walkthrough.

**Acceptance criteria:** A non-specialist can state the decision and the result; no unsupported claim.

### Lab 26.3 — SQL interview

**Objective:** Solve a fill-rate query under time pressure.

**Steps:** Clarify definitions; write the query; test on a small dataset; explain joins, grain, and nulls.

**Deliverable:** Query, test output, and a five-minute explanation.

**Acceptance criteria:** Correct grain, no duplicate inflation, explicit handling of zero-demand weeks.

### Lab 26.4 — Case interview

**Objective:** Structure an ambiguous supply chain problem.

**Steps:** Clarify the decision; segment the problem; propose a diagnostic plan; estimate the value; state what you would not do.

**Deliverable:** One-page case structure and a verbal walkthrough.

**Acceptance criteria:** MECE structure; explicit assumptions; a recommendation with a next step.

### Lab 26.5 — Mock panel

**Objective:** Practice a mixed panel.

**Steps:** Run a 45-minute panel with a planner, a technical interviewer, and a manager; record feedback; revise your stories.

**Deliverable:** Feedback log and revised answers.

**Acceptance criteria:** Each answer has a decision, evidence, and a limit; no more than two minutes per answer.

## 11. Assignment

**Portfolio and interview plan (individual, 10 hours).**

Produce:

1. A target-role map with competency ratings and evidence.
2. Three artifact READMEs.
3. A tailored resume and LinkedIn summary.
4. A one-page case study.
5. A 90-day onboarding plan for a target role.
6. A mock-interview feedback log with three revised answers.

## 12. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Role targeting | 15% | Generic interest | Specific role and competencies | Evidence-based gap plan |
| Artifact quality | 30% | Notebooks only | Three artifacts across depth, engineering, impact | Each artifact is decision-ready and reproducible |
| Case study | 20% | Model-first narrative | Decision, baseline, evidence, impact | Honest limits and a credible next step |
| Interview readiness | 20% | Vague answers | Structured, specific answers | Handles challenge and uncertainty well |
| Professional communication | 15% | Dense or overclaiming | Clear resume and memo | Tailored, credible, and concise |

## 13. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Notebook portfolio | No README, no baseline | Student habits | Rejected at screen | Add decision, baseline, impact |
| Metric inflation | "95% accuracy" | Wrong metric for imbalanced data | Credibility loss | Report base rate and cost |
| Claimed production | "Deployed" but local only | Ambition outruns evidence | Reference risk | Say pilot or prototype |
| Tool list without depth | Ten libraries, no project | Keyword optimization | Fails technical screen | Depth over breadth |
| No domain vocabulary | Cannot explain OTIF or MOQ | No operations context | Planner distrust | Module 01 and interviews |
| Case-study hero | "I alone..." | Team erasure | Cultural red flag | State role and team |
| Ignoring constraints | Plan is infeasible | Textbook habits | Operational rejection | Show capacity and MOQ |
| No questions | Accepts scope blindly | Disengagement | Poor fit | Ask about decision rights and data |
| Salary-only negotiation | Ignores scope | Short-term view | Bad fit | Negotiate scope and learning |
| No 90-day plan | "I will learn the business" | Vague | Slow start | Name a decision and metric |

## 14. Interview and discussion questions

1. **Tell me about a supply chain decision you improved.** Hint: decision, baseline, method, evidence, impact, limit.
2. **How do you choose between a forecast and a policy?** Hint: the decision determines the required output; a forecast is an input, not the answer.
3. **What is your experience with messy master data?** Hint: examples of item/location joins, duplicates, and reconciliation.
4. **How do you explain a model to a planner?** Hint: decision-relevant explanation, not feature importance alone.
5. **How do you handle a stakeholder who distrusts the model?** Hint: shadow mode, reason codes, small pilot, shared metric.
6. **What would you do in your first 90 days?** Hint: learn, diagnose, deliver one decision.
7. **How do you value an inventory reduction?** Hint: holding cost, working capital, service risk, and avoided markdown.
8. **What is the hardest optimization problem you have solved?** Hint: formulation, constraints, solve time, feasibility, sensitivity.
9. **How do you keep a model from causing a stockout?** Hint: bias monitoring, lower quantile for safety stock, human review, guardrails.
10. **Why this company and this supply chain?** Hint: specific decision, data, and operating model, not generic praise.

## 15. Key terms

| Term | Definition |
|---|---|
| Portfolio artifact | A project packaged to demonstrate a competency. |
| Baseline | The current method used for comparison. |
| FVA | Forecast value added; improvement over a naive benchmark. |
| STAR | Situation, task, action, result; a behavioral answer structure. |
| MECE | Mutually exclusive, collectively exhaustive; a case structure. |
| Take-home | A time-limited analysis used in hiring. |
| System design | A whiteboard exercise on architecture and trade-offs. |
| Decision rights | Who can approve, override, or stop a decision. |
| 90-day plan | A first-quarter plan with deliverables. |
| Scope negotiation | Agreeing on decision ownership, data, and success metrics. |
| Reason code | A structured reason for a manual override. |
| Shadow mode | Running a model without acting on it. |
| Model card | A summary of a model's purpose, data, performance, and limits. |
| Reproducibility | The ability to rerun and obtain the same result. |
| Evidence grade | A rating of source authority and proximity. |

## 16. Sources and further reading

- Hyndman, R. J., Athanasopoulos, G., *Forecasting: Principles and Practice*. https://otexts.com/fpp3/
- Kohavi, R., Tang, D., Xu, Y., *Trustworthy Online Controlled Experiments*.
- ASCM, CPIM and CSCP bodies of knowledge. https://www.ascm.org/
- CSCMP career resources. https://www.cscmp.org/
- INFORMS career center. https://www.informs.org/
- Gartner supply chain research, 2025–2026. https://www.gartner.com/en/supply-chain
- O*NET supply chain and operations occupations. https://www.onetonline.org/

> **Next:** [Module 27](27_datasets_tools_benchmarks.md) gives you the datasets, tools, and benchmark cautions to build the portfolio artifacts.
