# Time Series Forecasting - Case Study Collection

> **Mandatory authoring contract:** Every authored case follows the [course authoring and numerical standards](../../../../00_AUTHORING_AND_NUMERICAL_STANDARDS.md) and linked case master. Evidence, implementation stage and editorial status remain separate; a draft is not a reviewed or published case.

## 1. Collection metadata

| Field | Value |
|---|---|
| Collection owner | Curriculum maintainer |
| Last reviewed | 2026-11-06 (planning catalog only; new case unreviewed) |
| Review cadence | Quarterly and on material market/regulatory change |
| Index version | 2.1-draft |
| Canonical case-ID format | CASE-01-10-NNN |
| Canonical tags | Use the controlled vocabulary below; add synonyms only after review |
| Source of truth | This directory |

## 2. Case catalog

| Case ID | Case title / relative link | Domain | Case type | Decision / business question | Primary pattern | Secondary pattern(s) | Evidence | Stage | Last reviewed | Key result or limitation |
|---|---|---|---|---|---|---|---|---|---|---|
| CASE-01-10-001 | [Retail CPG demand forecasting at scale](01_Retail_CPG_Demand_Forecasting_At_Scale.md) | Retail CPG | predictive-ml | Daily bounded store-SKU replenishment order | evaluation-mismatch | data-leakage; model-monitoring | not-measured | proposed | unreviewed | Illustrative arithmetic only; no backtest, pilot, or business outcome measured; editorial status draft. Source: `DS_Domain/Case_Studies_Production_ML_Systems/TimeSeries_Forecasting/01_Retail_CPG_Demand_Forecasting_At_Scale.md` (formerly PLAN-01-10-01). |
| CASE-01-10-002 | [Ride-hailing demand forecasting for operational planning](02_Ride_Hailing_Demand_Forecasting.md) | Ride-hailing marketplace | predictive-ml | Next-hour zone request forecast for optional planning | evaluation-mismatch | data-leakage; model-monitoring | not-measured | proposed | unreviewed | Draft; historical Uber article plus illustrative design and arithmetic, no measured accuracy or causal impact. ETA remains planned. |
| PLAN-01-10-03 | Financial volatility forecast | Time Series Forecasting | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |
| CASE-01-10-003 | [Energy load forecasting for day-ahead planning](03_Energy_Load_Forecasting.md) | Utility operations | predictive-ml | Day-ahead hourly net-load forecast for human planning | evaluation-mismatch | data-leakage; model-monitoring | not-measured | proposed | unreviewed | Draft; illustrative assumptions and arithmetic only, no measured outcomes; source title `DS_Domain/Case_Studies_Production_ML_Systems/TimeSeries_Forecasting/04_Energy_Load_Forecasting.md` (formerly PLAN-01-10-04). |
| PLAN-01-10-05 | Server-metric anomaly detection | Time Series Forecasting | hybrid | Decision to be framed before writing | evaluation-mismatch | model-monitoring | not-measured | proposed | 2026-11-06 | |

## 3. Canonical reusable patterns

| Tag | Diagnostic signature | Use when | Boundary |
|---|---|---|---|
| data-leakage | Information unavailable at decision time inflates performance | Unexpectedly strong predictive result | Feature is truly available before the cutoff |
| label-delay | Outcome truth arrives late or changes | Fraud, claims, retention, downstream operational labels | Immediate, stable labels |
| distribution-shift | Operating population differs from training/evaluation | New market, season, channel, or rollout | Change is sampling noise only |
| evaluation-mismatch | Offline metric does not represent the business decision | Ranking, forecast, LLM, agent, or multi-objective products | Metric captures decision and guardrails |
| cost-latency-tradeoff | Quality, cost, and response time conflict | Online, real-time, or edge operation | Cost/latency are non-binding |
| human-in-the-loop | Qualified review changes risk/quality trade-off | High-impact, uncertain, or regulated work | Review is infeasible or ineffective |
| rollout-regression | Release degrades a live guardrail despite tests | Canary, A/B, feature-flag, or migration work | No rollout comparison exists |
| model-monitoring | Production signals reveal drift or degradation | Deployed model/system with feedback | No meaningful feedback signal |
| causal-confounding | Observational relationship is not intervention effect | Attribution, uplift, promotion, policy decisions | Valid randomization addresses concern |
| incident-postmortem | Operational impact requires causal reconstruction | Actual incident or replay exercise | No failure event/replay exists |

## 4. Similarity and contrast map

| Pattern | Cases | Theory/method links | Boundary to test |
|---|---|---|---|
| evaluation-mismatch | CASE-01-10-001; CASE-01-10-002; CASE-01-10-003 | [Rolling-origin evaluation](https://otexts.com/fpp3/tscv.html) | Forecast error need not reflect inventory, marketplace, or utility planning consequences. |
| distribution-shift | Market, season, policy, or user changes; planned cases | Forecasting and MLOps | Population and operational conditions |
| human-in-the-loop | CASE-01-10-001; high-impact operational cases planned | Governance and operations workflows | Review capacity and override quality |

## 5. Questions and future-case backlog

- Establish how much demand is censored during stockouts; availability data and a validated estimation approach are needed.
- Reconcile inventory position and supplier lead times at historical cutoffs before backtesting.
- Test whether a forecast-assisted policy improves cost and service against the operational baseline.
- Establish whether ride-hailing request events, corrections, and zone maps can be reconstructed as of historical forecast origins.
- Validate the request-count target and design a spillover-aware test of driver guidance before claiming policy impact.
- Evaluate interval coverage and width by zone and time after mature labels become available.
- Author ride-hailing ETA, financial volatility, and server-metric cases as separate planned decisions.
- Establish archived weather-vintage availability, load revisions, peak calibration, and an operationally credible comparison before evaluating the energy-load proposal.

## 6. Review and retirement log

| Date | Item | Event | Evidence and consequence |
|---|---|---|---|
| 2026-11-06 | Collection | Planning catalog reviewed | Does not constitute review of later authored content. |
| 2026-11-07 | CASE-01-10-001 | Draft added | Illustrative, proposed, unreviewed; review on changes to cutoff, inventory semantics, snapshots, or lead times. |
| 2026-11-07 | CASE-01-10-002 | Draft added | Published historical context and illustrative proposed design; unreviewed and not measured. |
| 2026-11-07 | CASE-01-10-003 | Draft added | Proposed and not measured; review if weather availability, target boundary, dispatch workflow or label revisions change. |

## 7. Pre-publish checks


- [ ] Case ID is unique and linked.
- [ ] Evidence maturity is distinct from implementation stage.
- [ ] Decision, metric, action, and outcome refer to the same population.
- [ ] Assumptions, constraints, failure modes, and a quick-reference card are present.
- [ ] Company-specific claims have primary/public evidence or are labelled illustrative.
- [ ] Local links resolve and the review trigger is actionable.