---
case_id: "TS-VOL-04"
case_title: "Point-in-time financial volatility forecasts for a risk review"
case_version: "1.0"
created: "2026-11-07"
last_reviewed: null
status: "proposed"
content_status: "draft"
case_type: "predictive-ml"
domain: "Financial risk / time-series forecasting"
primary_pattern: "time-series forecasting"
secondary_patterns: []
evidence_level: "not-measured"
source_basis: "illustrative"
as_of_date: "2026-11-07"
owner_role: "Market risk manager"
source_index: "INDEX.md"
review_trigger: "Change in market calendar, data vendor, target, or intended decision"
---

# Financial Volatility Forecasting for a Risk Review

## 1. The One-Line Framing

Given point-in-time daily close-to-close returns for one liquid hypothetical instrument, forecast next-session conditional variance for a market-risk manager's morning review, comparing a proposed exponentially weighted moving average (EWMA) with a rolling historical-variance baseline while preventing future-data leakage.
The decision is whether to bring an exposure to manual review, not whether to buy, sell, or size a position automatically.
The primary success condition is improved out-of-sample variance-forecast quality without unacceptable underprediction or operational failures.
This is a hypothetical design exercise: the original source file is unavailable, and all data, schedules, examples, thresholds, and calculations are constructed teaching inputs, not observed market or trading performance.
No named institution, real instrument, historical event, live deployment, or investment recommendation is implied.
A one-number price forecast is not the same question: the stakeholder needs uncertainty about returns, including the consequences of sudden risk changes.
The evidence-supported conclusion is a design tradeoff, not evidence that EWMA is superior on real markets.

## 2. Context & Background

A hypothetical market-risk team reviews the risk of an existing exposure once per trading session.
The team receives a vendor's timestamped adjusted closes after the exchange session, stores versions, and runs the next-session forecast after a data-quality deadline.
The risk manager reads the variance and volatility forecasts alongside exposure information and decides whether further manual investigation is warranted.
The baseline process displays the average squared returns over the most recent 20 valid sessions, with a fixed mean of zero.
That baseline is transparent and easy to audit, but its equal weights can react slowly to new shocks and can drop an old shock abruptly at the window edge.
The proposed scope is one hypothetical liquid instrument, one exchange calendar, and next-session close-to-close returns.
There is no claim that these operating assumptions describe any actual institution or market.
The data engineer owns price ingestion, market-calendar mapping, and correction reconciliation.
The quantitative analyst owns formulas, evaluation, and model artifacts.
The risk manager owns review decisions and may reject the forecast.
An independent model-risk reviewer would approve promotion criteria and review overrides.
No customer or investment transaction is automatically initiated by this system.
Geography, legal entity, exchange, vendor contract, and regulation are deliberately unspecified; legal applicability requires separate jurisdiction-specific review.
Project stage is proposed; no prior attempts, production history, incident, or outcome are known.

## 3. The Business Question

The illustrative stakeholder question is: can a next-session risk estimate improve the priority of manual exposure review compared with a trailing-20-session variance estimate, without hiding uncertainty during regime changes?
The owner is the market-risk manager; the trigger is availability of a validated closing-price snapshot.
The proposed deadline is 18:00 exchange-local time after the session; this is a scenario constraint, not a market fact.
Actions are to review exposure, defer pending data, or keep the existing baseline.
The forecast affects attention allocation; it does not directly authorize trading, hedging, leverage changes, or capital treatment.
A missed high-volatility period may leave risk unexamined; excessive flags consume analyst time and can displace other reviews.
The proxy objective is lower mean variance-forecast loss; the business outcome is better prioritized review and requires separate workflow evidence.
A model can score better on average while repeatedly missing the largest moves; tail underprediction is therefore a disqualifying guardrail.
The comparison covers the same eligible dates and data snapshots for all methods.
Portfolio aggregation, intraday forecasts, options-implied volatility, trading P&L, and regulatory capital decisions are outside this case.

## 4. Assumptions & Constraints

| ID | Assumption or constraint | Status and verification | Impact and fallback | Owner / recheck |
|---|---|---|---|---|
| A-01 | Correct exchange calendar and completed closes exist before cutoff | Unverified; compare vendor timestamps and calendar | Invalid return sequence; defer affected forecast | Data engineering / calendar change |
| A-02 | Adjustment factors are available as of each historical decision | Unverified; reconstruct vintage snapshots | Retrospective adjustments can leak; quarantine and obtain point-in-time data | Data engineering / vendor change |
| A-03 | A 20-session baseline and zero mean are acceptable references | Proposed; evaluate across regimes | Biased reference or unstable estimate; compare alternative window and mean | Quant / quarterly review |
| A-04 | Previous squared return is informative for next-session variance | Unverified; rolling holdout | Dynamics may change; baseline or no forecast | Quant / regime alert |
| A-05 | Forecast relates to the same close-to-close horizon as its label | Required design invariant; test timestamps | Horizon mismatch invalidates score; reject affected rows | Quant / horizon change |
| C-01 | Human authorization is required for any consequential action | Hypothetical policy constraint | Automation cannot act; defer to manager | Risk manager / policy change |
| C-02 | Forecast must use only information available by cutoff | Required for valid replay; as-of audit | Leakage invalidates comparisons; halt evaluation | Model risk / data correction |
| C-03 | Sensitive exposure data and vendor data need access controls | Proposed operating requirement; verify contracts | Unauthorized use; restrict access or do not deploy | Security / contract change |

The two assumptions most likely to reverse the model decision are availability of authentic historical data vintages and stability of the relationship between recent and subsequent volatility.
If point-in-time versions are unavailable, a retrospective backtest cannot establish deployable performance, even if it produces plausible charts.
If review costs exceed the benefit of earlier flags, lower statistical loss alone does not justify changing workflow.
No budget, staffing, infrastructure capacity, legal rule, or production service level has been verified.

## 5. Data, Tools & Method

### 5a. Data Specification

The decision grain is `(instrument_id, decision_session, cutoff_timestamp, horizon)`.
Prices are per share in a consistent quote currency; prices, corporate-action records, and calendar sessions have event and availability timestamps.
A forecast at cutoff after session `t` predicts the next completed session's close-to-close return `r_(t+1)`.
For a consistent adjusted total-return close `P_t`, the log return is `r_t = ln(P_t/P_(t-1))`, expressed as a decimal.
The illustrative target is `y_(t+1) = r_(t+1)^2`, a noisy proxy for next-session conditional variance, not observed latent variance.
We assume conditional mean zero for the forecast and loss examples; under a nonzero conditional mean, the conditional second moment is variance plus squared mean.
The proxy is only known after the next session's close and the validated label-vintage timestamp.
Labels with missing or unvalidated closes are unavailable rather than zero.
The planned dataset size, start/end dates, and missingness are unknown until an eligible licensed vintage archive is identified.
Only instruments with continuous, validated data under the declared inclusion policy are eligible; selecting surviving instruments retrospectively would create survivorship bias.
For this single-instrument exercise, no cross-sectional performance or generalizability is claimed.
Duplicate vendor records must be resolved using source identity, revision, and availability, never by selecting a correction published after the forecast cutoff.
Join calendar and corporate-action data using instrument key and an as-of condition on availability; verify one effective record per key and cutoff.
A split is by chronological decision session, with training and tuning preceding untouched final test dates and labels matured before use.

### 5b. Tools & Stack

The proposed stack uses a versioned object store for snapshots and artifacts, a scheduled batch process, and a review interface.
Python and numerical data-frame libraries are candidate implementation tools, not tools reported to have been used in a real project.
Exact dependency versions, source revision, storage technology, hardware, and vendor remain unselected and must be pinned before an actual run.
The scheduler and quant team own jobs and models respectively; security owns access policy.
The examples in this document are arithmetic constructions, not outputs of an executed backtest.

### 5c. Method Summary

The control is `v_base,t+1 = (1/20) × sum(r_(t-i)^2, i=0..19)` after the current return is known.
The candidate is `v_EWMA,t+1 = λ v_EWMA,t + (1−λ) r_t²`, with `0 < λ < 1`, and a documented initial variance.
Here `v_EWMA,t` is the forecast made at the previous cutoff for session `t`; it must never be initialized with future observations.
The daily volatility forecast is `sqrt(v)`; variance has squared-return units while volatility has return units.
For a positive forecast, Gaussian quasi-likelihood loss omitting constants is `L(v,y) = ln(v) + y/v`.
The minimum forecast floor is a predeclared numerical safeguard and must be applied identically during evaluation and serving.
A conditional-mean model, GARCH(1,1), and a simple historical 20-session estimator are plausible alternatives, but none has been evaluated here.
GARCH can represent mean reversion but requires parameter estimation and stability checks; it is not automatically better under changing regimes.
Select `λ` only on rolling development folds using a prespecified candidate set, then freeze it before the final test.
The proposed initial development grid is `0.90, 0.94, 0.97`; the selected value is unknown until validation.
No optimizer, stopping rule, fitted preprocessing, trained parameters, or performance comparison exists yet.
The human reviewer receives the forecast, its timestamp, baseline, data health, and warning flags, not a trading instruction.

| Candidate | Strength | Limitation | Status |
|---|---|---|---|
| Trailing-20 mean squared return | Simple, auditable comparator | Abrupt window turnover | Proposed baseline, not tested |
| EWMA | More weight on recent squared returns | Persistence depends on fixed decay; shock response may miscalibrate | Candidate, not selected by evidence |
| GARCH(1,1) | Separates persistent variance and shock response | Fit instability and extra tuning risk | Considered, not tested |
| Constant long-run variance | Low implementation complexity | Poor adaptation to changing volatility | Considered, not tested |

### 5d. Data Provenance, Quality, Exclusions and Leakage Controls

Each close has market event time, vendor publication time, ingestion time, revision ID, adjustment basis, and immutable snapshot ID.
The availability time is no earlier than both vendor publication and local ingestion; the evaluation cutoff is the actual decision deadline.
At every historical cutoff, replay selects only versions available by then; later corrected closes remain in a separate vintage for later decisions.
Never backfill a corrected value into a past feature snapshot as though the correction were known at the time.
A later correction can be used for final adjudicated labels only under a frozen label-vintage policy, after label maturity; report sensitivity to first-published labels.
For splits, cash dividends, symbol changes, mergers, and delistings, record the declared return convention and the time each adjustment became available.
A total-return series and a price-return series answer different questions; mixing them across dates or across a corporate action creates a spurious return.
If a corporate-action adjustment is late or uncertain, quarantine the affected return and defer the forecast rather than infer the adjustment.
For a market holiday, advance on the exchange session calendar, not a calendar-day offset.
A weekend-spanning close-to-close return remains one session return; this does not imply all session intervals have identical physical duration.
Do not forward-fill missing closes to create artificial zero returns; missing data breaks the return chain until valid endpoints can be established.
A multi-session gap must not be silently treated as a one-session return.
Stale prices, duplicate records, unexpected zero returns, nonpositive prices, unmatched symbols, and extreme adjustment factors trigger investigation.
Extreme genuine returns are not automatically removed as outliers: filtering them would conceal precisely the risk the forecast is intended to capture.
All fitted transformations and parameter selections use only training data within each chronological fold.
Access, retention, encryption, licensing, and deletion terms need explicit review; no real source is identified here.

### 5e. Reproducibility and Method-Selection Ledger

| ID | Reference or artifact | Scope and limitation |
|---|---|---|
| SRC-01 | [RiskMetrics Technical Document, fourth edition, 1996](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a) | Method reference for exponentially weighted risk modeling, not evidence for this case or a universal decay value |
| SRC-02 | [Bollerslev, Generalized Autoregressive Conditional Heteroskedasticity, 1986](https://doi.org/10.1016/0304-4076(86)90063-1) | Original GARCH formulation; no result for the hypothetical instrument |
| SRC-03 | [scikit-learn: Common pitfalls](https://scikit-learn.org/stable/common_pitfalls.html) | Method reference for leakage and consistent preprocessing; does not validate this design |
| SRC-04 | [Patton, Volatility forecast comparison using imperfect volatility proxies, 2011](https://doi.org/10.1016/j.jeconom.2010.03.034) | Method reference for evaluating volatility forecasts with imperfect proxies, not a case result |
| ART-01 | This document's worked arithmetic | Constructed examples only; no market dataset or executed run |
| RUN-01 | Not created | No code revision, dependency lock, seed, data snapshot, or fitted model exists |

References were identified for method context as of 2026-11-07; their availability and exact edition should be checked before operational use.
A future run must retain code commit, runtime and library versions, data vintage, calendar version, formula configuration, fold boundaries, and model artifact hash.
It must also retain the review decision, exclusion counts, label vintage, and complete prediction ledger needed to reconstruct each score.
If a licensed snapshot cannot be shared, reproduction requires equivalent rights and cannot be claimed from the document alone.

## 6. Analysis Walkthrough

The decisive tradeoff is that EWMA can react immediately to a new shock, but the correct decay and resulting real-world loss advantage are unknown.
The following arithmetic is illustrative and establishes neither market behavior nor model performance.
Suppose two validated adjusted closes available by cutoff are 100.00 and 102.00.
The log return is `ln(102/100) ≈ 0.0198026`, or approximately 1.9803%.
Its squared value is approximately `0.00039214` in decimal-return-squared units.
If yesterday's variance forecast was `0.00010000` and `λ=0.94`, the next forecast is `0.94×0.00010000 + 0.06×0.00039214 ≈ 0.00011753`.
The corresponding daily volatility is `sqrt(0.00011753) ≈ 0.010841`, or 1.0841% per session.
If the next 251 session returns were uncorrelated, had zero conditional mean, and each had this same variance, the 252-session variance would be `252×0.00011753 ≈ 0.029618`.
The corresponding annualized volatility would be `sqrt(252×0.00011753) ≈ 0.17210`, or 17.21%; 252 is an assumed number of trading sessions, not a calendar-day count.
This is a square-root-of-time scaling illustration, not a forecast that all future conditional variances are constant.
More generally the variance of a multi-session sum includes covariances; for varying conditional variances it requires a model for their future sum.
Volatility scales by `sqrt(252)` only under the stated simplifying assumptions; multiplying daily volatility by 252 is incorrect.
Suppose the trailing-20 squared returns sum to `0.00240000`; the baseline forecast is `0.00240000/20 = 0.00012000`, or 1.0954% daily volatility.
For a hypothetical next-day squared return of `0.000225`, the proxy is positive and can be scored after the close; it does not reveal the true conditional variance.
For EWMA, quasi-likelihood loss is approximately `ln(0.00011753)+0.000225/0.00011753 ≈ -7.1344`; for baseline it is approximately `-7.1530`.
Lower is better, so this one fabricated realization favors the baseline by roughly 0.0186 loss units; it proves nothing about expected or out-of-sample ranking.
The return, recurrence, and baseline calculations are traceable as ART-01; no rejected trial was actually run.

### Sensitivity and robustness

| Stress | Change | Consequence | Evidence |
|---|---|---|---|
| Decay | `λ=0.90` instead of `0.94` with the same inputs | Forecast rises to approximately `0.00012921`, daily volatility 1.137% | Illustrative arithmetic; faster shock response is not proof of improved forecast |
| Annualization | 250 instead of 252 sessions | Same daily forecast gives roughly 17.14% rather than 17.21% | Illustrative scaling; both rely on zero covariance and constant variance |
| Return definition | Price-only instead of total return around a dividend | The return and target can differ materially | Untested; use a consistent decision-relevant convention |
| Price correction | A vendor revision arrives after historical cutoff | It cannot be inserted into that cutoff's features | Required replay constraint; evaluate label-vintage sensitivity |
| Shift | Persistence or shock size changes | Fixed decay can adapt too slowly or overreact | Untested; analyze chronologically by regime |

The highest-leverage step is point-in-time reconstruction: no subsequent statistical technique repairs a forecast built from future corrections.
The competing explanation for any eventual improvement is better data handling or accidental differences in eligible dates rather than superior model dynamics.

## 7. The Finding

Do not replace the baseline on the available evidence; propose a point-in-time rolling evaluation of EWMA and retain human review.
The constructed one-day example yields variance forecasts of approximately `0.00011753` versus `0.00012000`, but it is not a market evaluation and has no measured business benefit.
Both label noise and changing regimes could reverse a real comparison, and annualized values depend on explicit assumptions.
The next justified action is to obtain licensed vintage data and execute the preregistered evaluation and operational checks described below.

## 8. System/Solution Design

### 8a. Offline / Training / Batch Path

Proposed architecture; no component has been deployed:

```text
Vendor event + publication time -> immutable vintage store -> calendar and adjustment validation
                                      |                         |
                                      |                         +-> quarantine -> data owner review
                                      v
                        as-of replay -> rolling training/tuning -> frozen final test
                                                     |
                                                     v
                                  model + schema + config registry -> promotion review
```

Ingestion records the original payload, source ID, receipt time, checksum, and revision without overwriting prior vintages.
A validation job creates eligible close and return records using a versioned exchange calendar and corporate-action policy.
A replay job generates forecasts at historical cutoffs using only available vintages and writes immutable forecast and label ledgers.
A quant selects hyperparameters on development folds and freezes code, configuration, and population before the final holdout.
A separate reviewer signs the evaluation and release decision; failure leaves the baseline in place.
Training state includes the previous EWMA variance, the timestamp to which it applies, initialization method, and data snapshot lineage.
A restart reconstructs state from the last verified checkpoint and replays eligible events in timestamp order.

### 8b. Online / Serving / Real-Time Path

The proposed service is scheduled batch rather than intraday real-time streaming.

```text
Scheduled cutoff -> authorize job -> as-of validated return + prior state -> variance update
                                          |                                  |
                                          v                                  v
                                  missing/stale -> defer             compare baseline + gates
                                                                             |
                                                      audit record <- review dashboard -> human decision
```

A successful cycle checks job identity, cutoff, calendar, source revision, price quality, and state compatibility before calculating forecasts.
It writes the result with forecast ID, instrument, target session, cutoff, model and data versions, variance, daily volatility, annualization convention, and quality status.
The dashboard shows a baseline comparison and a clear warning that annualized volatility is conditional on the scaling assumption.
An invalid price, untrusted adjustment, stale state, or missed deadline results in no candidate forecast and a manual data-quality escalation.
An independently available, valid baseline may be displayed as a fallback, explicitly labeled stale or current as appropriate; otherwise no numeric forecast is displayed.
The job does not send orders, change limits, or suppress an existing escalation.
A cold start uses a predeclared training-only initial variance with an explicit warm-up; insufficient history means defer rather than use future data.
The idempotency key is `(instrument_id, target_session, cutoff, model_version)`; reruns cannot silently overwrite an earlier issued forecast.
Corrected outputs use a distinct revision and preserve the original decision-time record.

### 8c. Key Design Decisions

| ID | Choice | Alternative and tradeoff | Revisit trigger |
|---|---|---|---|
| DEC-01 | Next-session variance with squared-return proxy | Intraday realized variance could reduce proxy noise but requires different inputs and label controls | Access to reliable intraday data |
| DEC-02 | Historical 20-session baseline | Other windows may represent a stronger local control | Rolling validation shows window dependence |
| DEC-03 | EWMA as candidate | GARCH offers explicit mean reversion but has more estimation risk | Stable evidence that added complexity helps |
| DEC-04 | No automatic trade | Automated action would require a separate control and authorization framework | Explicit new approved scope |
| DEC-05 | Immutable prediction and vintage ledger | Mutable tables are cheaper to query but destroy replayability | Storage or vendor terms change |

### 8d. Architecture Artifacts and Operational Readiness

The input contract requires a positive adjusted close, instrument key, exchange session, event time, vendor publication time, ingestion time, revision ID, and adjustment basis.
The output contract includes status, target session, data freshness, forecast variance, volatility unit, annualization assumption, baseline, and model/data lineage.
Failures return a typed reason rather than an invented value; only authorized risk and operations identities can retrieve sensitive exposures.
The owner retains immutable snapshots and audit logs under approved vendor and internal retention policies; actual retention duration is not yet specified.
Proposed service-level indicator: fraction of eligible session cutoffs with either a valid forecast or an explicit deferral delivered before the deadline, measured at the dashboard.
A candidate target is 99% over a rolling 60 eligible sessions; it is an unapproved scenario target, not an observed service level.
Forecast accuracy and timeliness are separate indicators: timely delivery of invalid risk estimates is not success.
On budget exhaustion or repeated data-quality failure, stop candidate promotion, investigate, and route review to the existing manual process.
Use bounded retries only before the cutoff and only for transient delivery errors; retries may not turn late data into an on-time historical input.
Back up versioned state and raw snapshots; rehearse restoration and compare replayed forecast IDs and values against the ledger before production approval.
Deploy a shadow candidate before any workflow change, then require model-risk and risk-manager approval; rollback restores a compatible previous model and schema.
Audit overrides, access, failed validations, vintage corrections, forecast availability, and latency; do not log secrets or unnecessary position details.
Sizing illustration: 1 instrument × 252 sessions/year × 2 forecast variants × 1 KiB/record is 504 KiB/year before indexes, replicas, and raw data.
At an illustrative 100 instruments the same forecast records would be 50,400 KiB/year; this is a linear estimate, not a measured capacity benchmark.
Raw vendor history, replicas, and audit retention may dominate this estimate; no infrastructure or license prices are available, so cost per useful review is unpriced.
A tenfold increase in instruments requires measured batch time and vendor rate-limit checks; no latency guarantee follows from record-size arithmetic.

## 9. Evaluation & Validation

The proposed primary metric is mean `ln(v_t)+y_t/v_t` over all eligible, matured final-test sessions, using decimal-squared returns and positive floored forecasts.
The denominator is the number of common eligible target sessions, not the number of generated forecasts or days on a calendar.
Evaluate the baseline and candidate on exactly the same rows, target horizon, cutoff, correction policy, and label maturity.
The proposed pass criterion is a positive baseline-minus-candidate mean loss improvement of at least 0.02 and a block-bootstrap 95% interval excluding zero.
The 0.02 threshold is an illustrative pre-run decision rule, not an industry standard, observed result, or economically calibrated value.
Model-risk and the business owner must approve the threshold before examining the final test; absent approval, treat it as a candidate rule only.
Guardrails are no material deterioration in high-volatility underprediction, no unresolved point-in-time leakage, and no missed operational security or human-review requirements.
Define high volatility using a threshold fitted on development data only, then report proxy loss and counts in high- and low-volatility groups without retuning on the test.
Report calibration as observed average squared return divided by average predicted variance, with time-block uncertainty and clear treatment of a near-zero denominator.
Also report squared-error loss as a secondary metric, and the proportion of squared-return proxies above several fixed multiples of forecast variance as diagnostic tail indicators.
A daily squared return is a noisy proxy; a high proxy-to-forecast ratio on one day alone is not proof of miscalibration.
The planned protocol uses expanding training windows and successive chronological validation blocks, with each training set ending before its evaluation block.
As an illustrative schedule, reserve the earliest 60% of eligible sessions for initial training, the next 20% for sequential validation, and the final 20% as untouched test.
Choose actual date cutoffs and minimum history before inspecting results; percentages alone are not a substitute for calendar boundaries or a sample-size calculation.
At each origin, include only labels matured by that origin; if targets span multiple sessions, purge overlapping target intervals and embargo as appropriate.
For the one-session target, enforce that the last training label has matured before the first validation forecast cutoff; compare timestamp intervals rather than assuming a fixed row gap.
Tune decay and any floor solely in development folds, freeze them, then run the held-out sequence exactly once for the prespecified comparison.
Use paired time-block bootstrap of session-level loss differences to preserve some temporal dependence, with block length set on development data and sensitivity to longer blocks.
Nonstationarity, limited crisis episodes, proxy noise, and block-length choice constrain interval interpretation; no bootstrap guarantees future-regime performance.
Report per-regime and per-calendar-segment counts and losses, missing-data exclusions, correction vintages, and the share of days on fallback.
Do not treat overlapping multi-session forecasts as independent if the horizon changes.
Sample size, power, empirical confidence interval, final decision, run artifact, and observed model performance are unknown because no evaluation has been executed.
A shadow workflow can measure alert load, data timeliness, and analyst review without changing decisions; a prospective controlled workflow study is needed for business impact.
No randomized study is specified or claimed; any future allocation of reviews must address interference, capacity, human behavior, and appropriate unit of assignment.

## 10. Action Taken

No action taken as of 2026-11-07: this is a proposed exercise with no organization, data license, approved model, or implementation record.
The recommendation is to retain the baseline while securing point-in-time history, agreeing the evaluation rule, and running a shadow-only candidate.
A risk manager and independent model-risk reviewer would approve any change in the review interface and its escalation rules.
The quant team would hand over model configuration, replay artifacts, and runbook to data engineering and risk operations.
There is no effective deployment date, region, user population, approved artifact, or rollout record.
Promotion would stop for leakage, unmet guardrails, inaccessible vintage lineage, or operational failure; rollback would disable the candidate and preserve audit history.
Automated orders, position sizing, and assertions of investment performance are excluded.

## 11. Outcome & Results

**Source declaration:** All numerical examples are constructed teaching values, not measurements of any named organization or evidence of realized impact.

| Metric | Baseline | Candidate | Change / uncertainty | Evidence and limitation |
|---|---|---|---|---|
| Mean final-test quasi-likelihood loss | Not measured | Not measured | Unknown; no interval | No evaluation run exists |
| Review quality and analyst effort | Not measured | Not measured | Unknown | Requires prospective workflow measurement |
| Forecast availability and data quality | Not measured | Not measured | Unknown | Requires service instrumentation and eligible-session denominator |
| Trading performance or savings | Not applicable to decision scope | Not claimed | No inference | This case does not recommend or evaluate trades |

The risk manager would commission a prospective measurement after the offline gate, with a predefined window, denominator, and escalation-load guardrail.
The illustrative worked calculation in Section 6 is not an outcome; CLM-01 is the only mathematical claim supported by ART-01.
CLM-01: given the stated inputs and assumptions, the EWMA recurrence produces approximately `0.00011753` variance and 1.0841% daily volatility; it does not establish superior future performance.

## 12. Failure Modes & What Went Wrong

No actual incident or rejected model is documented; the following are prospective failure hypotheses.

| ID | Trigger / symptom | Detection and containment | Owner / prevention |
|---|---|---|---|
| F-01 | Revised close leaks into an older snapshot | Compare feature availability against cutoff; invalidate affected evaluation | Data engineer / immutable vintages |
| F-02 | Split adjustment missing or applied twice | Reconcile actions and adjusted/unadjusted prices; quarantine return | Data engineer / versioned adjustment ledger |
| F-03 | Missing session becomes zero return | Calendar and endpoint checks; defer forecast | Data engineer / no forward fill |
| F-04 | Variance displayed as volatility | Unit and dimensional tests; block dashboard release | Quant / explicit schema units |
| F-05 | Annualization ignores covariance or varying future risk | Display assumptions; prohibit interpretation as unconditional annual prediction | Model risk / interpretation review |
| F-06 | Shock persists after regime shift | Monitor paired loss, calibration, and recent data; route to manual review | Quant / predeclared refit and reassessment |
| F-07 | EWMA state is stale or reset after restart | Check state session and lineage; replay from last valid checkpoint | Operations / restore rehearsal |
| F-08 | Tail events removed as outliers | Review exclusions against raw ledger and event context | Model risk / explicit exclusion approval |
| F-09 | Tuning uses final test labels | Audit run lineage; invalidate test and reserve a new future holdout | Independent reviewer / access separation |
| F-10 | Candidate loss improves but review burden rises | Track flags, review capacity, and missed escalations; stop workflow rollout | Risk manager / business guardrails |

A regime alert is a reason to investigate, not proof that a new model has become better.
If valid data are absent, retaining a stale numeric forecast without a clear warning is not a safe fallback.

## 13. Transferable Lesson

A risk forecast is a contract among the information cutoff, target horizon, unit, proxy, decision, and failure response.
The design transfers to other time-series problems only if the target and availability clocks are redefined for that domain.
In demand forecasting, for example, returns, market calendars, and corporate actions would be replaced by sales availability and stockout-aware demand labels.
The transferable mechanism is immutable as-of replay and honest out-of-time evaluation, not a universal 0.94 decay or 252-session scaling rule.
A more complex model is justified only when incremental benefits survive paired evaluation, implementation cost, and operational safeguards.
The counterexample is a stable process with scarce data: a transparent constant or historical baseline can be safer than a frequently retuned model.

## 14. Connections & Cross-References

Related case: [Retail CPG Demand Forecasting](01_Retail_CPG_Demand_Forecasting_At_Scale.md) illustrates a different target and decision setting.
Related case: [Energy Load Forecasting](03_Energy_Load_Forecasting.md) illustrates temporal forecasting under a different calendar and physical outcome.
The original directory title is `03_Financial_Time_Series_Volatility_Forecasting.md`; the inaccessible source is not evidence for the constructed case.
Source references and exact scope are in Section 5e; no source is cited as proof of this exercise's results.

### Interview pack: conceptual questions and answers

**Q1. What is the prediction target?**
Next-session conditional variance of decimal close-to-close log return, assessed with the next session's squared return as a noisy proxy under the zero-mean working assumption.
**Q2. Why is volatility not variance?**
Volatility is the square root of variance; a variance of `0.0004` corresponds to 2% daily volatility, not 0.04%.
**Q3. Why compare with historical variance?**
It is transparent and operationally feasible, so a candidate should beat it on matched, out-of-time rows before added complexity is justified.
**Q4. Why is a single squared return noisy?**
It is one realization of the return distribution and can be high or low even if the conditional variance forecast is correct.
**Q5. How would a corrected close cause leakage?**
Using a revised value published tomorrow in yesterday's features gives the historical model information it did not have at the decision time.
**Q6. What is the meaning of a 0.94 decay?**
It assigns 94% weight to the preceding variance estimate and 6% to the newest squared return; it is a proposed setting, not a universal optimum.
**Q7. How does a shock affect EWMA?**
A squared return above prior variance raises the next estimate, with a larger one-step response at lower decay.
**Q8. Why not annualize by multiplying volatility by 252?**
Under zero serial covariance and constant daily variance, variances add, so volatility scales with the square root of 252.
**Q9. When does square-root scaling fail?**
It is not exact when return covariances are nonzero, future variances differ, the target period differs, or calendar and session definitions are inconsistent.
**Q10. What is the strongest simple alternative?**
The existing trailing-20-session mean of squared returns, computed with the same as-of data and eligibility filters.
**Q11. How is decay selected without leakage?**
Use expanding development folds with matured labels, choose the parameter once, and freeze it before the untouched final test.
**Q12. What does a better mean loss not show?**
It does not prove better tail protection, realized business benefit, causal impact, or trading profitability.
**Q13. What is the response to a suspected regime shift?**
Flag the uncertainty, inspect current data quality and paired diagnostic losses, retain human oversight, and follow a preapproved reassessment process.
**Q14. Why not remove extreme returns automatically?**
True extreme observations contain information about risk; exclusion can make a forecast look accurate by deleting the cases it must address.
**Q15. What if a cutoff is missed?**
Defer the candidate, document why, and use a valid independently available baseline or manual review; do not fabricate a forecast.
**Q16. What if every model is worse than baseline?**
Keep the baseline and investigate model assumptions or data availability; document the failed selection rather than promoting a complex model.
**Q17. What if labels have later corrections?**
Freeze a label-vintage rule, ensure maturity before training, and separately assess sensitivity to first-published versus later corrected labels.
**Q18. What is the operational unit of reliability?**
A correctly delivered forecast or explicit deferral for an eligible target session, measured at the review interface by the deadline.
**Q19. Is this investment advice?**
No: the example concerns a hypothetical risk-review system and does not propose a transaction, position, or investment strategy.
**Q20. What would falsify the model recommendation?**
A valid chronological holdout showing insufficient improvement, leakage, unacceptable tail underprediction, or operational safeguards that cannot be met.

### Numerical and production decision laboratories

The following independent, constructed one-step scenarios exercise the same recurrence with different starting variances, observed returns, decay settings, and next-session proxies.
Every `v` and `y` is in squared decimal-return units, and every percentage is a daily volatility or daily return, not an annualized investment outcome.
Each line is a separate teaching observation; none is a measurement, simulation run, recommendation, or performance claim.
The historical baseline for each scenario is deliberately fixed at the stated previous variance to make the arithmetic comparison transparent; it is not presented as a measured 20-session history.
Loss differences are single-realization diagnostics and cannot establish model superiority.

**Laboratory 001:** Prior variance `0.00003710`, latest return `-0.61%`, decay `0.94`, and next-session proxy `0.00028561`.
Latest squared return `0.00003721` is `1.003` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00003487`, and the shock contribution is `0.00000223`.
Candidate variance is `0.00003711` versus illustrative baseline `0.00003710`; the update raises the estimate.
Daily volatility is `0.6092%` for the candidate and `0.6091%` for the baseline.
At the constructed proxy, candidate loss is `-2.5047` and baseline loss is `-2.5035`; their difference is `+0.0012` baseline minus candidate.
The proxy is `7.697` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-0.61%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 002:** Prior variance `0.00003920`, latest return `+0.82%`, decay `0.97`, and next-session proxy `0.00082944`.
Latest squared return `0.00006724` is `1.715` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00003802`, and the shock contribution is `0.00000202`.
Candidate variance is `0.00004004` versus illustrative baseline `0.00003920`; the update raises the estimate.
Daily volatility is `0.6328%` for the candidate and `0.6261%` for the baseline.
At the constructed proxy, candidate loss is `10.5891` and baseline loss is `11.0123`; their difference is `+0.4233` baseline minus candidate.
The proxy is `20.715` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+0.82%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 003:** Prior variance `0.00004130`, latest return `-1.03%`, decay `0.90`, and next-session proxy `0.00165649`.
Latest squared return `0.00010609` is `2.569` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00003717`, and the shock contribution is `0.00001061`.
Candidate variance is `0.00004778` versus illustrative baseline `0.00004130`; the update raises the estimate.
Daily volatility is `0.6912%` for the candidate and `0.6427%` for the baseline.
At the constructed proxy, candidate loss is `24.7209` and baseline loss is `30.0141`; their difference is `+5.2932` baseline minus candidate.
The proxy is `34.670` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.03%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 004:** Prior variance `0.00004340`, latest return `+1.24%`, decay `0.94`, and next-session proxy `0.00018225`.
Latest squared return `0.00015376` is `3.543` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00004080`, and the shock contribution is `0.00000923`.
Candidate variance is `0.00005002` versus illustrative baseline `0.00004340`; the update raises the estimate.
Daily volatility is `0.7073%` for the candidate and `0.6588%` for the baseline.
At the constructed proxy, candidate loss is `-6.2596` and baseline loss is `-5.8457`; their difference is `+0.4139` baseline minus candidate.
The proxy is `3.643` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.24%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 005:** Prior variance `0.00004550`, latest return `-1.45%`, decay `0.97`, and next-session proxy `0.00064516`.
Latest squared return `0.00021025` is `4.621` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00004413`, and the shock contribution is `0.00000631`.
Candidate variance is `0.00005044` versus illustrative baseline `0.00004550`; the update raises the estimate.
Daily volatility is `0.7102%` for the candidate and `0.6745%` for the baseline.
At the constructed proxy, candidate loss is `2.8953` and baseline loss is `4.1815`; their difference is `+1.2862` baseline minus candidate.
The proxy is `12.790` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.45%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 006:** Prior variance `0.00004760`, latest return `+1.66%`, decay `0.90`, and next-session proxy `0.00139129`.
Latest squared return `0.00027556` is `5.789` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00004284`, and the shock contribution is `0.00002756`.
Candidate variance is `0.00007040` versus illustrative baseline `0.00004760`; the update raises the estimate.
Daily volatility is `0.8390%` for the candidate and `0.6899%` for the baseline.
At the constructed proxy, candidate loss is `10.2024` and baseline loss is `19.2761`; their difference is `+9.0737` baseline minus candidate.
The proxy is `19.764` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.66%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 007:** Prior variance `0.00004970`, latest return `-1.87%`, decay `0.94`, and next-session proxy `0.00010201`.
Latest squared return `0.00034969` is `7.036` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00004672`, and the shock contribution is `0.00002098`.
Candidate variance is `0.00006770` versus illustrative baseline `0.00004970`; the update raises the estimate.
Daily volatility is `0.8228%` for the candidate and `0.7050%` for the baseline.
At the constructed proxy, candidate loss is `-8.0936` and baseline loss is `-7.8570`; their difference is `+0.2366` baseline minus candidate.
The proxy is `1.507` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-1.87%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 008:** Prior variance `0.00005180`, latest return `+2.08%`, decay `0.97`, and next-session proxy `0.00048400`.
Latest squared return `0.00043264` is `8.352` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00005025`, and the shock contribution is `0.00001298`.
Candidate variance is `0.00006323` versus illustrative baseline `0.00005180`; the update raises the estimate.
Daily volatility is `0.7951%` for the candidate and `0.7197%` for the baseline.
At the constructed proxy, candidate loss is `-2.0136` and baseline loss is `-0.5245`; their difference is `+1.4891` baseline minus candidate.
The proxy is `7.655` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+2.08%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 009:** Prior variance `0.00005390`, latest return `-2.29%`, decay `0.90`, and next-session proxy `0.00114921`.
Latest squared return `0.00052441` is `9.729` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00004851`, and the shock contribution is `0.00005244`.
Candidate variance is `0.00010095` versus illustrative baseline `0.00005390`; the update raises the estimate.
Daily volatility is `1.0047%` for the candidate and `0.7342%` for the baseline.
At the constructed proxy, candidate loss is `2.1830` and baseline loss is `11.4928`; their difference is `+9.3098` baseline minus candidate.
The proxy is `11.384` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-2.29%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 010:** Prior variance `0.00005600`, latest return `+2.50%`, decay `0.94`, and next-session proxy `0.00004489`.
Latest squared return `0.00062500` is `11.161` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00005264`, and the shock contribution is `0.00003750`.
Candidate variance is `0.00009014` versus illustrative baseline `0.00005600`; the update raises the estimate.
Daily volatility is `0.9494%` for the candidate and `0.7483%` for the baseline.
At the constructed proxy, candidate loss is `-8.8161` and baseline loss is `-8.9886`; their difference is `-0.1724` baseline minus candidate.
The proxy is `0.498` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+2.50%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 011:** Prior variance `0.00005810`, latest return `-2.71%`, decay `0.97`, and next-session proxy `0.00034596`.
Latest squared return `0.00073441` is `12.640` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00005636`, and the shock contribution is `0.00002203`.
Candidate variance is `0.00007839` versus illustrative baseline `0.00005810`; the update raises the estimate.
Daily volatility is `0.8854%` for the candidate and `0.7622%` for the baseline.
At the constructed proxy, candidate loss is `-5.0405` and baseline loss is `-3.7988`; their difference is `+1.2417` baseline minus candidate.
The proxy is `4.413` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-2.71%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 012:** Prior variance `0.00006020`, latest return `+2.92%`, decay `0.90`, and next-session proxy `0.00093025`.
Latest squared return `0.00085264` is `14.163` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00005418`, and the shock contribution is `0.00008526`.
Candidate variance is `0.00013944` versus illustrative baseline `0.00006020`; the update raises the estimate.
Daily volatility is `1.1809%` for the candidate and `0.7759%` for the baseline.
At the constructed proxy, candidate loss is `-2.2067` and baseline loss is `5.7348`; their difference is `+7.9415` baseline minus candidate.
The proxy is `6.671` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+2.92%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 013:** Prior variance `0.00006230`, latest return `-0.40%`, decay `0.94`, and next-session proxy `0.00179776`.
Latest squared return `0.00001600` is `0.257` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00005856`, and the shock contribution is `0.00000096`.
Candidate variance is `0.00005952` versus illustrative baseline `0.00006230`; the update reduces the estimate.
Daily volatility is `0.7715%` for the candidate and `0.7893%` for the baseline.
At the constructed proxy, candidate loss is `20.4741` and baseline loss is `19.1730`; their difference is `-1.3012` baseline minus candidate.
The proxy is `30.203` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-0.40%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 014:** Prior variance `0.00006440`, latest return `+0.61%`, decay `0.97`, and next-session proxy `0.00023104`.
Latest squared return `0.00003721` is `0.578` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00006247`, and the shock contribution is `0.00000112`.
Candidate variance is `0.00006358` versus illustrative baseline `0.00006440`; the update reduces the estimate.
Daily volatility is `0.7974%` for the candidate and `0.8025%` for the baseline.
At the constructed proxy, candidate loss is `-6.0295` and baseline loss is `-6.0628`; their difference is `-0.0333` baseline minus candidate.
The proxy is `3.634` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+0.61%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 015:** Prior variance `0.00006650`, latest return `-0.82%`, decay `0.90`, and next-session proxy `0.00073441`.
Latest squared return `0.00006724` is `1.011` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00005985`, and the shock contribution is `0.00000672`.
Candidate variance is `0.00006657` versus illustrative baseline `0.00006650`; the update raises the estimate.
Daily volatility is `0.8159%` for the candidate and `0.8155%` for the baseline.
At the constructed proxy, candidate loss is `1.4143` and baseline loss is `1.4255`; their difference is `+0.0112` baseline minus candidate.
The proxy is `11.031` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-0.82%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 016:** Prior variance `0.00006860`, latest return `+1.03%`, decay `0.94`, and next-session proxy `0.00152100`.
Latest squared return `0.00010609` is `1.547` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00006448`, and the shock contribution is `0.00000637`.
Candidate variance is `0.00007085` versus illustrative baseline `0.00006860`; the update raises the estimate.
Daily volatility is `0.8417%` for the candidate and `0.8283%` for the baseline.
At the constructed proxy, candidate loss is `11.9131` and baseline loss is `12.5848`; their difference is `+0.6717` baseline minus candidate.
The proxy is `21.468` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.03%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 017:** Prior variance `0.00007070`, latest return `-1.24%`, decay `0.97`, and next-session proxy `0.00013924`.
Latest squared return `0.00015376` is `2.175` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00006858`, and the shock contribution is `0.00000461`.
Candidate variance is `0.00007319` versus illustrative baseline `0.00007070`; the update raises the estimate.
Daily volatility is `0.8555%` for the candidate and `0.8408%` for the baseline.
At the constructed proxy, candidate loss is `-7.6200` and baseline loss is `-7.5876`; their difference is `+0.0324` baseline minus candidate.
The proxy is `1.902` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-1.24%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 018:** Prior variance `0.00007280`, latest return `+1.45%`, decay `0.90`, and next-session proxy `0.00056169`.
Latest squared return `0.00021025` is `2.888` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00006552`, and the shock contribution is `0.00002102`.
Candidate variance is `0.00008654` versus illustrative baseline `0.00007280`; the update raises the estimate.
Daily volatility is `0.9303%` for the candidate and `0.8532%` for the baseline.
At the constructed proxy, candidate loss is `-2.8647` and baseline loss is `-1.8123`; their difference is `+1.0524` baseline minus candidate.
The proxy is `6.490` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.45%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 019:** Prior variance `0.00007490`, latest return `-1.66%`, decay `0.94`, and next-session proxy `0.00126736`.
Latest squared return `0.00027556` is `3.679` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00007041`, and the shock contribution is `0.00001653`.
Candidate variance is `0.00008694` versus illustrative baseline `0.00007490`; the update raises the estimate.
Daily volatility is `0.9324%` for the candidate and `0.8654%` for the baseline.
At the constructed proxy, candidate loss is `5.2272` and baseline loss is `7.4213`; their difference is `+2.1942` baseline minus candidate.
The proxy is `14.577` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.66%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 020:** Prior variance `0.00007700`, latest return `+1.87%`, decay `0.97`, and next-session proxy `0.00007056`.
Latest squared return `0.00034969` is `4.541` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00007469`, and the shock contribution is `0.00001049`.
Candidate variance is `0.00008518` versus illustrative baseline `0.00007700`; the update raises the estimate.
Daily volatility is `0.9229%` for the candidate and `0.8775%` for the baseline.
At the constructed proxy, candidate loss is `-8.5424` and baseline loss is `-8.5553`; their difference is `-0.0130` baseline minus candidate.
The proxy is `0.828` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+1.87%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 021:** Prior variance `0.00007910`, latest return `-2.08%`, decay `0.90`, and next-session proxy `0.00041209`.
Latest squared return `0.00043264` is `5.470` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00007119`, and the shock contribution is `0.00004326`.
Candidate variance is `0.00011445` versus illustrative baseline `0.00007910`; the update raises the estimate.
Daily volatility is `1.0698%` for the candidate and `0.8894%` for the baseline.
At the constructed proxy, candidate loss is `-5.4749` and baseline loss is `-4.2351`; their difference is `+1.2398` baseline minus candidate.
The proxy is `3.600` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-2.08%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 022:** Prior variance `0.00008120`, latest return `+2.29%`, decay `0.94`, and next-session proxy `0.00103684`.
Latest squared return `0.00052441` is `6.458` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00007633`, and the shock contribution is `0.00003146`.
Candidate variance is `0.00010779` versus illustrative baseline `0.00008120`; the update raises the estimate.
Daily volatility is `1.0382%` for the candidate and `0.9011%` for the baseline.
At the constructed proxy, candidate loss is `0.4835` and baseline loss is `3.3504`; their difference is `+2.8668` baseline minus candidate.
The proxy is `9.619` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+2.29%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 023:** Prior variance `0.00008330`, latest return `-2.50%`, decay `0.97`, and next-session proxy `0.00002500`.
Latest squared return `0.00062500` is `7.503` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00008080`, and the shock contribution is `0.00001875`.
Candidate variance is `0.00009955` versus illustrative baseline `0.00008330`; the update raises the estimate.
Daily volatility is `0.9978%` for the candidate and `0.9127%` for the baseline.
At the constructed proxy, candidate loss is `-8.9637` and baseline loss is `-9.0929`; their difference is `-0.1292` baseline minus candidate.
The proxy is `0.251` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-2.50%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 024:** Prior variance `0.00008540`, latest return `+2.71%`, decay `0.90`, and next-session proxy `0.00028561`.
Latest squared return `0.00073441` is `8.600` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00007686`, and the shock contribution is `0.00007344`.
Candidate variance is `0.00015030` versus illustrative baseline `0.00008540`; the update raises the estimate.
Daily volatility is `1.2260%` for the candidate and `0.9241%` for the baseline.
At the constructed proxy, candidate loss is `-6.9026` and baseline loss is `-6.0238`; their difference is `+0.8788` baseline minus candidate.
The proxy is `1.900` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+2.71%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 025:** Prior variance `0.00008750`, latest return `-2.92%`, decay `0.94`, and next-session proxy `0.00082944`.
Latest squared return `0.00085264` is `9.744` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00008225`, and the shock contribution is `0.00005116`.
Candidate variance is `0.00013341` versus illustrative baseline `0.00008750`; the update raises the estimate.
Daily volatility is `1.1550%` for the candidate and `0.9354%` for the baseline.
At the constructed proxy, candidate loss is `-2.7048` and baseline loss is `0.1354`; their difference is `+2.8402` baseline minus candidate.
The proxy is `6.217` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-2.92%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 026:** Prior variance `0.00008960`, latest return `+0.40%`, decay `0.97`, and next-session proxy `0.00165649`.
Latest squared return `0.00001600` is `0.179` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00008691`, and the shock contribution is `0.00000048`.
Candidate variance is `0.00008739` versus illustrative baseline `0.00008960`; the update reduces the estimate.
Daily volatility is `0.9348%` for the candidate and `0.9466%` for the baseline.
At the constructed proxy, candidate loss is `9.6096` and baseline loss is `9.1675`; their difference is `-0.4421` baseline minus candidate.
The proxy is `18.955` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+0.40%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 027:** Prior variance `0.00009170`, latest return `-0.61%`, decay `0.90`, and next-session proxy `0.00018225`.
Latest squared return `0.00003721` is `0.406` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00008253`, and the shock contribution is `0.00000372`.
Candidate variance is `0.00008625` versus illustrative baseline `0.00009170`; the update reduces the estimate.
Daily volatility is `0.9287%` for the candidate and `0.9576%` for the baseline.
At the constructed proxy, candidate loss is `-7.2452` and baseline loss is `-7.3095`; their difference is `-0.0643` baseline minus candidate.
The proxy is `2.113` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-0.61%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 028:** Prior variance `0.00009380`, latest return `+0.82%`, decay `0.94`, and next-session proxy `0.00064516`.
Latest squared return `0.00006724` is `0.717` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00008817`, and the shock contribution is `0.00000403`.
Candidate variance is `0.00009221` versus illustrative baseline `0.00009380`; the update reduces the estimate.
Daily volatility is `0.9602%` for the candidate and `0.9685%` for the baseline.
At the constructed proxy, candidate loss is `-2.2946` and baseline loss is `-2.3963`; their difference is `-0.1017` baseline minus candidate.
The proxy is `6.997` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+0.82%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 029:** Prior variance `0.00009590`, latest return `-1.03%`, decay `0.97`, and next-session proxy `0.00139129`.
Latest squared return `0.00010609` is `1.106` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00009302`, and the shock contribution is `0.00000318`.
Candidate variance is `0.00009621` versus illustrative baseline `0.00009590`; the update raises the estimate.
Daily volatility is `0.9808%` for the candidate and `0.9793%` for the baseline.
At the constructed proxy, candidate loss is `5.2126` and baseline loss is `5.2555`; their difference is `+0.0429` baseline minus candidate.
The proxy is `14.462` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.03%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 030:** Prior variance `0.00009800`, latest return `+1.24%`, decay `0.90`, and next-session proxy `0.00010201`.
Latest squared return `0.00015376` is `1.569` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00008820`, and the shock contribution is `0.00001538`.
Candidate variance is `0.00010358` versus illustrative baseline `0.00009800`; the update raises the estimate.
Daily volatility is `1.0177%` for the candidate and `0.9899%` for the baseline.
At the constructed proxy, candidate loss is `-8.1903` and baseline loss is `-8.1896`; their difference is `+0.0007` baseline minus candidate.
The proxy is `0.985` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+1.24%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 031:** Prior variance `0.00010010`, latest return `-1.45%`, decay `0.94`, and next-session proxy `0.00048400`.
Latest squared return `0.00021025` is `2.100` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00009409`, and the shock contribution is `0.00001262`.
Candidate variance is `0.00010671` versus illustrative baseline `0.00010010`; the update raises the estimate.
Daily volatility is `1.0330%` for the candidate and `1.0005%` for the baseline.
At the constructed proxy, candidate loss is `-4.6097` and baseline loss is `-4.3742`; their difference is `+0.2355` baseline minus candidate.
The proxy is `4.536` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.45%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 032:** Prior variance `0.00010220`, latest return `+1.66%`, decay `0.97`, and next-session proxy `0.00114921`.
Latest squared return `0.00027556` is `2.696` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00009913`, and the shock contribution is `0.00000827`.
Candidate variance is `0.00010740` versus illustrative baseline `0.00010220`; the update raises the estimate.
Daily volatility is `1.0363%` for the candidate and `1.0109%` for the baseline.
At the constructed proxy, candidate loss is `1.5613` and baseline loss is `2.0561`; their difference is `+0.4949` baseline minus candidate.
The proxy is `10.700` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.66%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 033:** Prior variance `0.00010430`, latest return `-1.87%`, decay `0.90`, and next-session proxy `0.00004489`.
Latest squared return `0.00034969` is `3.353` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00009387`, and the shock contribution is `0.00003497`.
Candidate variance is `0.00012884` versus illustrative baseline `0.00010430`; the update raises the estimate.
Daily volatility is `1.1351%` for the candidate and `1.0213%` for the baseline.
At the constructed proxy, candidate loss is `-8.6085` and baseline loss is `-8.7378`; their difference is `-0.1293` baseline minus candidate.
The proxy is `0.348` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-1.87%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 034:** Prior variance `0.00010640`, latest return `+2.08%`, decay `0.94`, and next-session proxy `0.00034596`.
Latest squared return `0.00043264` is `4.066` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00010002`, and the shock contribution is `0.00002596`.
Candidate variance is `0.00012597` versus illustrative baseline `0.00010640`; the update raises the estimate.
Daily volatility is `1.1224%` for the candidate and `1.0315%` for the baseline.
At the constructed proxy, candidate loss is `-6.2332` and baseline loss is `-5.8968`; their difference is `+0.3364` baseline minus candidate.
The proxy is `2.746` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+2.08%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 035:** Prior variance `0.00010850`, latest return `-2.29%`, decay `0.97`, and next-session proxy `0.00093025`.
Latest squared return `0.00052441` is `4.833` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00010524`, and the shock contribution is `0.00001573`.
Candidate variance is `0.00012098` versus illustrative baseline `0.00010850`; the update raises the estimate.
Daily volatility is `1.0999%` for the candidate and `1.0416%` for the baseline.
At the constructed proxy, candidate loss is `-1.3304` and baseline loss is `-0.5550`; their difference is `+0.7754` baseline minus candidate.
The proxy is `7.689` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-2.29%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 036:** Prior variance `0.00011060`, latest return `+2.50%`, decay `0.90`, and next-session proxy `0.00179776`.
Latest squared return `0.00062500` is `5.651` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00009954`, and the shock contribution is `0.00006250`.
Candidate variance is `0.00016204` versus illustrative baseline `0.00011060`; the update raises the estimate.
Daily volatility is `1.2729%` for the candidate and `1.0517%` for the baseline.
At the constructed proxy, candidate loss is `2.3669` and baseline loss is `7.1450`; their difference is `+4.7781` baseline minus candidate.
The proxy is `11.095` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+2.50%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 037:** Prior variance `0.00011270`, latest return `-2.71%`, decay `0.94`, and next-session proxy `0.00023104`.
Latest squared return `0.00073441` is `6.517` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00010594`, and the shock contribution is `0.00004406`.
Candidate variance is `0.00015000` versus illustrative baseline `0.00011270`; the update raises the estimate.
Daily volatility is `1.2248%` for the candidate and `1.0616%` for the baseline.
At the constructed proxy, candidate loss is `-7.2646` and baseline loss is `-7.0407`; their difference is `+0.2239` baseline minus candidate.
The proxy is `1.540` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-2.71%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 038:** Prior variance `0.00011480`, latest return `+2.92%`, decay `0.97`, and next-session proxy `0.00073441`.
Latest squared return `0.00085264` is `7.427` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00011136`, and the shock contribution is `0.00002558`.
Candidate variance is `0.00013694` versus illustrative baseline `0.00011480`; the update raises the estimate.
Daily volatility is `1.1702%` for the candidate and `1.0714%` for the baseline.
At the constructed proxy, candidate loss is `-3.5328` and baseline loss is `-2.6750`; their difference is `+0.8578` baseline minus candidate.
The proxy is `5.363` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+2.92%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 039:** Prior variance `0.00011690`, latest return `-0.40%`, decay `0.90`, and next-session proxy `0.00152100`.
Latest squared return `0.00001600` is `0.137` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00010521`, and the shock contribution is `0.00000160`.
Candidate variance is `0.00010681` versus illustrative baseline `0.00011690`; the update reduces the estimate.
Daily volatility is `1.0335%` for the candidate and `1.0812%` for the baseline.
At the constructed proxy, candidate loss is `5.0958` and baseline loss is `3.9569`; their difference is `-1.1389` baseline minus candidate.
The proxy is `14.240` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-0.40%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 040:** Prior variance `0.00011900`, latest return `+0.61%`, decay `0.94`, and next-session proxy `0.00013924`.
Latest squared return `0.00003721` is `0.313` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00011186`, and the shock contribution is `0.00000223`.
Candidate variance is `0.00011409` versus illustrative baseline `0.00011900`; the update reduces the estimate.
Daily volatility is `1.0681%` for the candidate and `1.0909%` for the baseline.
At the constructed proxy, candidate loss is `-7.8581` and baseline loss is `-7.8663`; their difference is `-0.0082` baseline minus candidate.
The proxy is `1.220` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+0.61%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 041:** Prior variance `0.00012110`, latest return `-0.82%`, decay `0.97`, and next-session proxy `0.00056169`.
Latest squared return `0.00006724` is `0.555` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00011747`, and the shock contribution is `0.00000202`.
Candidate variance is `0.00011948` versus illustrative baseline `0.00012110`; the update reduces the estimate.
Daily volatility is `1.0931%` for the candidate and `1.1005%` for the baseline.
At the constructed proxy, candidate loss is `-4.3314` and baseline loss is `-4.3807`; their difference is `-0.0493` baseline minus candidate.
The proxy is `4.701` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-0.82%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 042:** Prior variance `0.00012320`, latest return `+1.03%`, decay `0.90`, and next-session proxy `0.00126736`.
Latest squared return `0.00010609` is `0.861` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00011088`, and the shock contribution is `0.00001061`.
Candidate variance is `0.00012149` versus illustrative baseline `0.00012320`; the update reduces the estimate.
Daily volatility is `1.1022%` for the candidate and `1.1100%` for the baseline.
At the constructed proxy, candidate loss is `1.4162` and baseline loss is `1.2853`; their difference is `-0.1309` baseline minus candidate.
The proxy is `10.432` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.03%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 043:** Prior variance `0.00012530`, latest return `-1.24%`, decay `0.94`, and next-session proxy `0.00007056`.
Latest squared return `0.00015376` is `1.227` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00011778`, and the shock contribution is `0.00000923`.
Candidate variance is `0.00012701` versus illustrative baseline `0.00012530`; the update raises the estimate.
Daily volatility is `1.1270%` for the candidate and `1.1194%` for the baseline.
At the constructed proxy, candidate loss is `-8.4157` and baseline loss is `-8.4217`; their difference is `-0.0060` baseline minus candidate.
The proxy is `0.556` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-1.24%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 044:** Prior variance `0.00012740`, latest return `+1.45%`, decay `0.97`, and next-session proxy `0.00041209`.
Latest squared return `0.00021025` is `1.650` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00012358`, and the shock contribution is `0.00000631`.
Candidate variance is `0.00012989` versus illustrative baseline `0.00012740`; the update raises the estimate.
Daily volatility is `1.1397%` for the candidate and `1.1287%` for the baseline.
At the constructed proxy, candidate loss is `-5.7761` and baseline loss is `-5.7336`; their difference is `+0.0426` baseline minus candidate.
The proxy is `3.173` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.45%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 045:** Prior variance `0.00012950`, latest return `-1.66%`, decay `0.90`, and next-session proxy `0.00103684`.
Latest squared return `0.00027556` is `2.128` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00011655`, and the shock contribution is `0.00002756`.
Candidate variance is `0.00014411` versus illustrative baseline `0.00012950`; the update raises the estimate.
Daily volatility is `1.2004%` for the candidate and `1.1380%` for the baseline.
At the constructed proxy, candidate loss is `-1.6500` and baseline loss is `-0.9453`; their difference is `+0.7046` baseline minus candidate.
The proxy is `7.195` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.66%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 046:** Prior variance `0.00013160`, latest return `+1.87%`, decay `0.94`, and next-session proxy `0.00002500`.
Latest squared return `0.00034969` is `2.657` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00012370`, and the shock contribution is `0.00002098`.
Candidate variance is `0.00014469` versus illustrative baseline `0.00013160`; the update raises the estimate.
Daily volatility is `1.2029%` for the candidate and `1.1472%` for the baseline.
At the constructed proxy, candidate loss is `-8.6682` and baseline loss is `-8.7458`; their difference is `-0.0776` baseline minus candidate.
The proxy is `0.173` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+1.87%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 047:** Prior variance `0.00013370`, latest return `-2.08%`, decay `0.97`, and next-session proxy `0.00028561`.
Latest squared return `0.00043264` is `3.236` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00012969`, and the shock contribution is `0.00001298`.
Candidate variance is `0.00014267` versus illustrative baseline `0.00013370`; the update raises the estimate.
Daily volatility is `1.1944%` for the candidate and `1.1563%` for the baseline.
At the constructed proxy, candidate loss is `-6.8531` and baseline loss is `-6.7837`; their difference is `+0.0694` baseline minus candidate.
The proxy is `2.002` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-2.08%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 048:** Prior variance `0.00013580`, latest return `+2.29%`, decay `0.90`, and next-session proxy `0.00082944`.
Latest squared return `0.00052441` is `3.862` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00012222`, and the shock contribution is `0.00005244`.
Candidate variance is `0.00017466` versus illustrative baseline `0.00013580`; the update raises the estimate.
Daily volatility is `1.3216%` for the candidate and `1.1653%` for the baseline.
At the constructed proxy, candidate loss is `-3.9038` and baseline loss is `-2.7965`; their difference is `+1.1073` baseline minus candidate.
The proxy is `4.749` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+2.29%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 049:** Prior variance `0.00013790`, latest return `-2.50%`, decay `0.94`, and next-session proxy `0.00165649`.
Latest squared return `0.00062500` is `4.532` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00012963`, and the shock contribution is `0.00003750`.
Candidate variance is `0.00016713` versus illustrative baseline `0.00013790`; the update raises the estimate.
Daily volatility is `1.2928%` for the candidate and `1.1743%` for the baseline.
At the constructed proxy, candidate loss is `1.2149` and baseline loss is `3.1233`; their difference is `+1.9084` baseline minus candidate.
The proxy is `9.912` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-2.50%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 050:** Prior variance `0.00014000`, latest return `+2.71%`, decay `0.97`, and next-session proxy `0.00018225`.
Latest squared return `0.00073441` is `5.246` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00013580`, and the shock contribution is `0.00002203`.
Candidate variance is `0.00015783` versus illustrative baseline `0.00014000`; the update raises the estimate.
Daily volatility is `1.2563%` for the candidate and `1.1832%` for the baseline.
At the constructed proxy, candidate loss is `-7.5993` and baseline loss is `-7.5721`; their difference is `+0.0272` baseline minus candidate.
The proxy is `1.155` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+2.71%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 051:** Prior variance `0.00014210`, latest return `-2.92%`, decay `0.90`, and next-session proxy `0.00064516`.
Latest squared return `0.00085264` is `6.000` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00012789`, and the shock contribution is `0.00008526`.
Candidate variance is `0.00021315` versus illustrative baseline `0.00014210`; the update raises the estimate.
Daily volatility is `1.4600%` for the candidate and `1.1921%` for the baseline.
At the constructed proxy, candidate loss is `-5.4268` and baseline loss is `-4.3188`; their difference is `+1.1080` baseline minus candidate.
The proxy is `3.027` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-2.92%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 052:** Prior variance `0.00014420`, latest return `+0.40%`, decay `0.94`, and next-session proxy `0.00139129`.
Latest squared return `0.00001600` is `0.111` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00013555`, and the shock contribution is `0.00000096`.
Candidate variance is `0.00013651` versus illustrative baseline `0.00014420`; the update reduces the estimate.
Daily volatility is `1.1684%` for the candidate and `1.2008%` for the baseline.
At the constructed proxy, candidate loss is `1.2929` and baseline loss is `0.8040`; their difference is `-0.4888` baseline minus candidate.
The proxy is `10.192` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+0.40%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 053:** Prior variance `0.00014630`, latest return `-0.61%`, decay `0.97`, and next-session proxy `0.00010201`.
Latest squared return `0.00003721` is `0.254` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00014191`, and the shock contribution is `0.00000112`.
Candidate variance is `0.00014303` versus illustrative baseline `0.00014630`; the update reduces the estimate.
Daily volatility is `1.1959%` for the candidate and `1.2095%` for the baseline.
At the constructed proxy, candidate loss is `-8.1393` and baseline loss is `-8.1326`; their difference is `+0.0067` baseline minus candidate.
The proxy is `0.713` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-0.61%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 054:** Prior variance `0.00014840`, latest return `+0.82%`, decay `0.90`, and next-session proxy `0.00048400`.
Latest squared return `0.00006724` is `0.453` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00013356`, and the shock contribution is `0.00000672`.
Candidate variance is `0.00014028` versus illustrative baseline `0.00014840`; the update reduces the estimate.
Daily volatility is `1.1844%` for the candidate and `1.2182%` for the baseline.
At the constructed proxy, candidate loss is `-5.4217` and baseline loss is `-5.5541`; their difference is `-0.1324` baseline minus candidate.
The proxy is `3.450` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+0.82%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 055:** Prior variance `0.00015050`, latest return `-1.03%`, decay `0.94`, and next-session proxy `0.00114921`.
Latest squared return `0.00010609` is `0.705` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00014147`, and the shock contribution is `0.00000637`.
Candidate variance is `0.00014784` versus illustrative baseline `0.00015050`; the update reduces the estimate.
Daily volatility is `1.2159%` for the candidate and `1.2268%` for the baseline.
At the constructed proxy, candidate loss is `-1.0458` and baseline loss is `-1.1656`; their difference is `-0.1198` baseline minus candidate.
The proxy is `7.774` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.03%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 056:** Prior variance `0.00015260`, latest return `+1.24%`, decay `0.97`, and next-session proxy `0.00004489`.
Latest squared return `0.00015376` is `1.008` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00014802`, and the shock contribution is `0.00000461`.
Candidate variance is `0.00015263` versus illustrative baseline `0.00015260`; the update raises the estimate.
Daily volatility is `1.2355%` for the candidate and `1.2353%` for the baseline.
At the constructed proxy, candidate loss is `-8.4934` and baseline loss is `-8.4935`; their difference is `-0.0002` baseline minus candidate.
The proxy is `0.294` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+1.24%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 057:** Prior variance `0.00015470`, latest return `-1.45%`, decay `0.90`, and next-session proxy `0.00034596`.
Latest squared return `0.00021025` is `1.359` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00013923`, and the shock contribution is `0.00002102`.
Candidate variance is `0.00016025` versus illustrative baseline `0.00015470`; the update raises the estimate.
Daily volatility is `1.2659%` for the candidate and `1.2438%` for the baseline.
At the constructed proxy, candidate loss is `-6.5799` and baseline loss is `-6.5377`; their difference is `+0.0422` baseline minus candidate.
The proxy is `2.159` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-1.45%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 058:** Prior variance `0.00015680`, latest return `+1.66%`, decay `0.94`, and next-session proxy `0.00093025`.
Latest squared return `0.00027556` is `1.757` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00014739`, and the shock contribution is `0.00001653`.
Candidate variance is `0.00016393` versus illustrative baseline `0.00015680`; the update raises the estimate.
Daily volatility is `1.2803%` for the candidate and `1.2522%` for the baseline.
At the constructed proxy, candidate loss is `-3.0413` and baseline loss is `-2.8278`; their difference is `+0.2134` baseline minus candidate.
The proxy is `5.675` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.66%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 059:** Prior variance `0.00015890`, latest return `-1.87%`, decay `0.97`, and next-session proxy `0.00179776`.
Latest squared return `0.00034969` is `2.201` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00015413`, and the shock contribution is `0.00001049`.
Candidate variance is `0.00016462` versus illustrative baseline `0.00015890`; the update raises the estimate.
Daily volatility is `1.2831%` for the candidate and `1.2606%` for the baseline.
At the constructed proxy, candidate loss is `2.2086` and baseline loss is `2.5665`; their difference is `+0.3580` baseline minus candidate.
The proxy is `10.920` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.87%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 060:** Prior variance `0.00016100`, latest return `+2.08%`, decay `0.90`, and next-session proxy `0.00023104`.
Latest squared return `0.00043264` is `2.687` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00014490`, and the shock contribution is `0.00004326`.
Candidate variance is `0.00018816` versus illustrative baseline `0.00016100`; the update raises the estimate.
Daily volatility is `1.3717%` for the candidate and `1.2689%` for the baseline.
At the constructed proxy, candidate loss is `-7.3503` and baseline loss is `-7.2991`; their difference is `+0.0513` baseline minus candidate.
The proxy is `1.228` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+2.08%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 061:** Prior variance `0.00016310`, latest return `-2.29%`, decay `0.94`, and next-session proxy `0.00073441`.
Latest squared return `0.00052441` is `3.215` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00015331`, and the shock contribution is `0.00003146`.
Candidate variance is `0.00018478` versus illustrative baseline `0.00016310`; the update raises the estimate.
Daily volatility is `1.3593%` for the candidate and `1.2771%` for the baseline.
At the constructed proxy, candidate loss is `-4.6218` and baseline loss is `-4.2183`; their difference is `+0.4035` baseline minus candidate.
The proxy is `3.975` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-2.29%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 062:** Prior variance `0.00016520`, latest return `+2.50%`, decay `0.97`, and next-session proxy `0.00152100`.
Latest squared return `0.00062500` is `3.783` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00016024`, and the shock contribution is `0.00001875`.
Candidate variance is `0.00017899` versus illustrative baseline `0.00016520`; the update raises the estimate.
Daily volatility is `1.3379%` for the candidate and `1.2853%` for the baseline.
At the constructed proxy, candidate loss is `-0.1307` and baseline loss is `0.4987`; their difference is `+0.6293` baseline minus candidate.
The proxy is `8.497` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+2.50%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 063:** Prior variance `0.00016730`, latest return `-2.71%`, decay `0.90`, and next-session proxy `0.00013924`.
Latest squared return `0.00073441` is `4.390` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00015057`, and the shock contribution is `0.00007344`.
Candidate variance is `0.00022401` versus illustrative baseline `0.00016730`; the update raises the estimate.
Daily volatility is `1.4967%` for the candidate and `1.2934%` for the baseline.
At the constructed proxy, candidate loss is `-7.7822` and baseline loss is `-7.8634`; their difference is `-0.0812` baseline minus candidate.
The proxy is `0.622` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-2.71%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 064:** Prior variance `0.00016940`, latest return `+2.92%`, decay `0.94`, and next-session proxy `0.00056169`.
Latest squared return `0.00085264` is `5.033` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00015924`, and the shock contribution is `0.00005116`.
Candidate variance is `0.00021039` versus illustrative baseline `0.00016940`; the update raises the estimate.
Daily volatility is `1.4505%` for the candidate and `1.3015%` for the baseline.
At the constructed proxy, candidate loss is `-5.7968` and baseline loss is `-5.3675`; their difference is `+0.4293` baseline minus candidate.
The proxy is `2.670` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+2.92%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 065:** Prior variance `0.00017150`, latest return `-0.40%`, decay `0.97`, and next-session proxy `0.00126736`.
Latest squared return `0.00001600` is `0.093` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00016636`, and the shock contribution is `0.00000048`.
Candidate variance is `0.00016683` versus illustrative baseline `0.00017150`; the update reduces the estimate.
Daily volatility is `1.2916%` for the candidate and `1.3096%` for the baseline.
At the constructed proxy, candidate loss is `-1.1020` and baseline loss is `-1.2811`; their difference is `-0.1791` baseline minus candidate.
The proxy is `7.596` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-0.40%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 066:** Prior variance `0.00017360`, latest return `+0.61%`, decay `0.90`, and next-session proxy `0.00007056`.
Latest squared return `0.00003721` is `0.214` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00015624`, and the shock contribution is `0.00000372`.
Candidate variance is `0.00015996` versus illustrative baseline `0.00017360`; the update reduces the estimate.
Daily volatility is `1.2648%` for the candidate and `1.3176%` for the baseline.
At the constructed proxy, candidate loss is `-8.2995` and baseline loss is `-8.2523`; their difference is `+0.0472` baseline minus candidate.
The proxy is `0.441` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+0.61%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 067:** Prior variance `0.00017570`, latest return `-0.82%`, decay `0.94`, and next-session proxy `0.00041209`.
Latest squared return `0.00006724` is `0.383` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00016516`, and the shock contribution is `0.00000403`.
Candidate variance is `0.00016919` versus illustrative baseline `0.00017570`; the update reduces the estimate.
Daily volatility is `1.3007%` for the candidate and `1.3255%` for the baseline.
At the constructed proxy, candidate loss is `-6.2488` and baseline loss is `-6.3013`; their difference is `-0.0525` baseline minus candidate.
The proxy is `2.436` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-0.82%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 068:** Prior variance `0.00017780`, latest return `+1.03%`, decay `0.97`, and next-session proxy `0.00103684`.
Latest squared return `0.00010609` is `0.597` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00017247`, and the shock contribution is `0.00000318`.
Candidate variance is `0.00017565` versus illustrative baseline `0.00017780`; the update reduces the estimate.
Daily volatility is `1.3253%` for the candidate and `1.3334%` for the baseline.
At the constructed proxy, candidate loss is `-2.7441` and baseline loss is `-2.8034`; their difference is `-0.0592` baseline minus candidate.
The proxy is `5.903` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.03%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 069:** Prior variance `0.00017990`, latest return `-1.24%`, decay `0.90`, and next-session proxy `0.00002500`.
Latest squared return `0.00015376` is `0.855` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00016191`, and the shock contribution is `0.00001538`.
Candidate variance is `0.00017729` versus illustrative baseline `0.00017990`; the update reduces the estimate.
Daily volatility is `1.3315%` for the candidate and `1.3413%` for the baseline.
At the constructed proxy, candidate loss is `-8.4967` and baseline loss is `-8.4841`; their difference is `+0.0126` baseline minus candidate.
The proxy is `0.141` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-1.24%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 070:** Prior variance `0.00018200`, latest return `+1.45%`, decay `0.94`, and next-session proxy `0.00028561`.
Latest squared return `0.00021025` is `1.155` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00017108`, and the shock contribution is `0.00001262`.
Candidate variance is `0.00018369` versus illustrative baseline `0.00018200`; the update raises the estimate.
Daily volatility is `1.3553%` for the candidate and `1.3491%` for the baseline.
At the constructed proxy, candidate loss is `-7.0474` and baseline loss is `-7.0422`; their difference is `+0.0052` baseline minus candidate.
The proxy is `1.555` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+1.45%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 071:** Prior variance `0.00018410`, latest return `-1.66%`, decay `0.97`, and next-session proxy `0.00082944`.
Latest squared return `0.00027556` is `1.497` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00017858`, and the shock contribution is `0.00000827`.
Candidate variance is `0.00018684` versus illustrative baseline `0.00018410`; the update raises the estimate.
Daily volatility is `1.3669%` for the candidate and `1.3568%` for the baseline.
At the constructed proxy, candidate loss is `-4.1460` and baseline loss is `-4.0947`; their difference is `+0.0514` baseline minus candidate.
The proxy is `4.439` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.66%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 072:** Prior variance `0.00018620`, latest return `+1.87%`, decay `0.90`, and next-session proxy `0.00165649`.
Latest squared return `0.00034969` is `1.878` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00016758`, and the shock contribution is `0.00003497`.
Candidate variance is `0.00020255` versus illustrative baseline `0.00018620`; the update raises the estimate.
Daily volatility is `1.4232%` for the candidate and `1.3646%` for the baseline.
At the constructed proxy, candidate loss is `-0.3263` and baseline loss is `0.3076`; their difference is `+0.6339` baseline minus candidate.
The proxy is `8.178` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.87%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 073:** Prior variance `0.00018830`, latest return `-2.08%`, decay `0.94`, and next-session proxy `0.00018225`.
Latest squared return `0.00043264` is `2.298` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00017700`, and the shock contribution is `0.00002596`.
Candidate variance is `0.00020296` versus illustrative baseline `0.00018830`; the update raises the estimate.
Daily volatility is `1.4246%` for the candidate and `1.3722%` for the baseline.
At the constructed proxy, candidate loss is `-7.6045` and baseline loss is `-7.6096`; their difference is `-0.0051` baseline minus candidate.
The proxy is `0.898` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-2.08%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 074:** Prior variance `0.00019040`, latest return `+2.29%`, decay `0.97`, and next-session proxy `0.00064516`.
Latest squared return `0.00052441` is `2.754` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00018469`, and the shock contribution is `0.00001573`.
Candidate variance is `0.00020042` versus illustrative baseline `0.00019040`; the update raises the estimate.
Daily volatility is `1.4157%` for the candidate and `1.3799%` for the baseline.
At the constructed proxy, candidate loss is `-5.2961` and baseline loss is `-5.1779`; their difference is `+0.1181` baseline minus candidate.
The proxy is `3.219` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+2.29%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 075:** Prior variance `0.00019250`, latest return `-2.50%`, decay `0.90`, and next-session proxy `0.00139129`.
Latest squared return `0.00062500` is `3.247` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00017325`, and the shock contribution is `0.00006250`.
Candidate variance is `0.00023575` versus illustrative baseline `0.00019250`; the update raises the estimate.
Daily volatility is `1.5354%` for the candidate and `1.3874%` for the baseline.
At the constructed proxy, candidate loss is `-2.4512` and baseline loss is `-1.3279`; their difference is `+1.1233` baseline minus candidate.
The proxy is `5.902` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-2.50%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 076:** Prior variance `0.00019460`, latest return `+2.71%`, decay `0.94`, and next-session proxy `0.00010201`.
Latest squared return `0.00073441` is `3.774` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00018292`, and the shock contribution is `0.00004406`.
Candidate variance is `0.00022699` versus illustrative baseline `0.00019460`; the update raises the estimate.
Daily volatility is `1.5066%` for the candidate and `1.3950%` for the baseline.
At the constructed proxy, candidate loss is `-7.9412` and baseline loss is `-8.0204`; their difference is `-0.0792` baseline minus candidate.
The proxy is `0.449` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+2.71%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 077:** Prior variance `0.00019670`, latest return `-2.92%`, decay `0.97`, and next-session proxy `0.00048400`.
Latest squared return `0.00085264` is `4.335` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00019080`, and the shock contribution is `0.00002558`.
Candidate variance is `0.00021638` versus illustrative baseline `0.00019670`; the update raises the estimate.
Daily volatility is `1.4710%` for the candidate and `1.4025%` for the baseline.
At the constructed proxy, candidate loss is `-6.2017` and baseline loss is `-6.0732`; their difference is `+0.1284` baseline minus candidate.
The proxy is `2.237` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-2.92%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 078:** Prior variance `0.00019880`, latest return `+0.40%`, decay `0.90`, and next-session proxy `0.00114921`.
Latest squared return `0.00001600` is `0.080` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00017892`, and the shock contribution is `0.00000160`.
Candidate variance is `0.00018052` versus illustrative baseline `0.00019880`; the update reduces the estimate.
Daily volatility is `1.3436%` for the candidate and `1.4100%` for the baseline.
At the constructed proxy, candidate loss is `-2.2536` and baseline loss is `-2.7425`; their difference is `-0.4889` baseline minus candidate.
The proxy is `6.366` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+0.40%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 079:** Prior variance `0.00020090`, latest return `-0.61%`, decay `0.94`, and next-session proxy `0.00004489`.
Latest squared return `0.00003721` is `0.185` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00018885`, and the shock contribution is `0.00000223`.
Candidate variance is `0.00019108` versus illustrative baseline `0.00020090`; the update reduces the estimate.
Daily volatility is `1.3823%` for the candidate and `1.4174%` for the baseline.
At the constructed proxy, candidate loss is `-8.3279` and baseline loss is `-8.2893`; their difference is `+0.0386` baseline minus candidate.
The proxy is `0.235` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-0.61%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 080:** Prior variance `0.00020300`, latest return `+0.82%`, decay `0.97`, and next-session proxy `0.00034596`.
Latest squared return `0.00006724` is `0.331` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00019691`, and the shock contribution is `0.00000202`.
Candidate variance is `0.00019893` versus illustrative baseline `0.00020300`; the update reduces the estimate.
Daily volatility is `1.4104%` for the candidate and `1.4248%` for the baseline.
At the constructed proxy, candidate loss is `-6.7834` and baseline loss is `-6.7981`; their difference is `-0.0146` baseline minus candidate.
The proxy is `1.739` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+0.82%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.

**Laboratory 081:** Prior variance `0.00020510`, latest return `-1.03%`, decay `0.90`, and next-session proxy `0.00093025`.
Latest squared return `0.00010609` is `0.517` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00018459`, and the shock contribution is `0.00001061`.
Candidate variance is `0.00019520` versus illustrative baseline `0.00020510`; the update reduces the estimate.
Daily volatility is `1.3971%` for the candidate and `1.4321%` for the baseline.
At the constructed proxy, candidate loss is `-3.7758` and baseline loss is `-3.9564`; their difference is `-0.1806` baseline minus candidate.
The proxy is `4.766` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.03%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 082:** Prior variance `0.00020720`, latest return `+1.24%`, decay `0.94`, and next-session proxy `0.00179776`.
Latest squared return `0.00015376` is `0.742` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00019477`, and the shock contribution is `0.00000923`.
Candidate variance is `0.00020399` versus illustrative baseline `0.00020720`; the update reduces the estimate.
Daily volatility is `1.4283%` for the candidate and `1.4394%` for the baseline.
At the constructed proxy, candidate loss is `0.3154` and baseline loss is `0.1946`; their difference is `-0.1208` baseline minus candidate.
The proxy is `8.813` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.24%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 083:** Prior variance `0.00020930`, latest return `-1.45%`, decay `0.97`, and next-session proxy `0.00023104`.
Latest squared return `0.00021025` is `1.005` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00020302`, and the shock contribution is `0.00000631`.
Candidate variance is `0.00020933` versus illustrative baseline `0.00020930`; the update raises the estimate.
Daily volatility is `1.4468%` for the candidate and `1.4467%` for the baseline.
At the constructed proxy, candidate loss is `-7.3679` and baseline loss is `-7.3679`; their difference is `+0.0000` baseline minus candidate.
The proxy is `1.104` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-1.45%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 084:** Prior variance `0.00021140`, latest return `+1.66%`, decay `0.90`, and next-session proxy `0.00073441`.
Latest squared return `0.00027556` is `1.304` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00019026`, and the shock contribution is `0.00002756`.
Candidate variance is `0.00021782` versus illustrative baseline `0.00021140`; the update raises the estimate.
Daily volatility is `1.4759%` for the candidate and `1.4540%` for the baseline.
At the constructed proxy, candidate loss is `-5.0602` and baseline loss is `-4.9877`; their difference is `+0.0724` baseline minus candidate.
The proxy is `3.372` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.66%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 085:** Prior variance `0.00021350`, latest return `-1.87%`, decay `0.94`, and next-session proxy `0.00152100`.
Latest squared return `0.00034969` is `1.638` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00020069`, and the shock contribution is `0.00002098`.
Candidate variance is `0.00022167` versus illustrative baseline `0.00021350`; the update raises the estimate.
Daily volatility is `1.4889%` for the candidate and `1.4612%` for the baseline.
At the constructed proxy, candidate loss is `-1.5528` and baseline loss is `-1.3278`; their difference is `+0.2251` baseline minus candidate.
The proxy is `6.862` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.87%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 086:** Prior variance `0.00021560`, latest return `+2.08%`, decay `0.97`, and next-session proxy `0.00013924`.
Latest squared return `0.00043264` is `2.007` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00020913`, and the shock contribution is `0.00001298`.
Candidate variance is `0.00022211` versus illustrative baseline `0.00021560`; the update raises the estimate.
Daily volatility is `1.4903%` for the candidate and `1.4683%` for the baseline.
At the constructed proxy, candidate loss is `-7.7854` and baseline loss is `-7.7963`; their difference is `-0.0108` baseline minus candidate.
The proxy is `0.627` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+2.08%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 087:** Prior variance `0.00021770`, latest return `-2.29%`, decay `0.90`, and next-session proxy `0.00056169`.
Latest squared return `0.00052441` is `2.409` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00019593`, and the shock contribution is `0.00005244`.
Candidate variance is `0.00024837` versus illustrative baseline `0.00021770`; the update raises the estimate.
Daily volatility is `1.5760%` for the candidate and `1.4755%` for the baseline.
At the constructed proxy, candidate loss is `-6.0391` and baseline loss is `-5.8523`; their difference is `+0.1868` baseline minus candidate.
The proxy is `2.261` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-2.29%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 088:** Prior variance `0.00021980`, latest return `+2.50%`, decay `0.94`, and next-session proxy `0.00126736`.
Latest squared return `0.00062500` is `2.843` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00020661`, and the shock contribution is `0.00003750`.
Candidate variance is `0.00024411` versus illustrative baseline `0.00021980`; the update raises the estimate.
Daily volatility is `1.5624%` for the candidate and `1.4826%` for the baseline.
At the constructed proxy, candidate loss is `-3.1262` and baseline loss is `-2.6568`; their difference is `+0.4693` baseline minus candidate.
The proxy is `5.192` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+2.50%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 089:** Prior variance `0.00022190`, latest return `-2.71%`, decay `0.97`, and next-session proxy `0.00007056`.
Latest squared return `0.00073441` is `3.310` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00021524`, and the shock contribution is `0.00002203`.
Candidate variance is `0.00023728` versus illustrative baseline `0.00022190`; the update raises the estimate.
Daily volatility is `1.5404%` for the candidate and `1.4896%` for the baseline.
At the constructed proxy, candidate loss is `-8.0489` and baseline loss is `-8.0953`; their difference is `-0.0464` baseline minus candidate.
The proxy is `0.297` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-2.71%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 090:** Prior variance `0.00022400`, latest return `+2.92%`, decay `0.90`, and next-session proxy `0.00041209`.
Latest squared return `0.00085264` is `3.806` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00020160`, and the shock contribution is `0.00008526`.
Candidate variance is `0.00028686` versus illustrative baseline `0.00022400`; the update raises the estimate.
Daily volatility is `1.6937%` for the candidate and `1.4967%` for the baseline.
At the constructed proxy, candidate loss is `-6.7200` and baseline loss is `-6.5642`; their difference is `+0.1558` baseline minus candidate.
The proxy is `1.437` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+2.92%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 091:** Prior variance `0.00022610`, latest return `-0.40%`, decay `0.94`, and next-session proxy `0.00103684`.
Latest squared return `0.00001600` is `0.071` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00021253`, and the shock contribution is `0.00000096`.
Candidate variance is `0.00021349` versus illustrative baseline `0.00022610`; the update reduces the estimate.
Daily volatility is `1.4611%` for the candidate and `1.5037%` for the baseline.
At the constructed proxy, candidate loss is `-3.5954` and baseline loss is `-3.8088`; their difference is `-0.2134` baseline minus candidate.
The proxy is `4.857` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-0.40%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 092:** Prior variance `0.00022820`, latest return `+0.61%`, decay `0.97`, and next-session proxy `0.00002500`.
Latest squared return `0.00003721` is `0.163` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00022135`, and the shock contribution is `0.00000112`.
Candidate variance is `0.00022247` versus illustrative baseline `0.00022820`; the update reduces the estimate.
Daily volatility is `1.4915%` for the candidate and `1.5106%` for the baseline.
At the constructed proxy, candidate loss is `-8.2983` and baseline loss is `-8.2757`; their difference is `+0.0226` baseline minus candidate.
The proxy is `0.112` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+0.61%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 093:** Prior variance `0.00023030`, latest return `-0.82%`, decay `0.90`, and next-session proxy `0.00028561`.
Latest squared return `0.00006724` is `0.292` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00020727`, and the shock contribution is `0.00000672`.
Candidate variance is `0.00021399` versus illustrative baseline `0.00023030`; the update reduces the estimate.
Daily volatility is `1.4629%` for the candidate and `1.5176%` for the baseline.
At the constructed proxy, candidate loss is `-7.1149` and baseline loss is `-7.1360`; their difference is `-0.0211` baseline minus candidate.
The proxy is `1.335` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-0.82%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 094:** Prior variance `0.00023240`, latest return `+1.03%`, decay `0.94`, and next-session proxy `0.00082944`.
Latest squared return `0.00010609` is `0.456` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00021846`, and the shock contribution is `0.00000637`.
Candidate variance is `0.00022482` versus illustrative baseline `0.00023240`; the update reduces the estimate.
Daily volatility is `1.4994%` for the candidate and `1.5245%` for the baseline.
At the constructed proxy, candidate loss is `-4.7109` and baseline loss is `-4.7980`; their difference is `-0.0872` baseline minus candidate.
The proxy is `3.689` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.03%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 095:** Prior variance `0.00023450`, latest return `-1.24%`, decay `0.97`, and next-session proxy `0.00165649`.
Latest squared return `0.00015376` is `0.656` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00022746`, and the shock contribution is `0.00000461`.
Candidate variance is `0.00023208` versus illustrative baseline `0.00023450`; the update reduces the estimate.
Daily volatility is `1.5234%` for the candidate and `1.5313%` for the baseline.
At the constructed proxy, candidate loss is `-1.2308` and baseline loss is `-1.2941`; their difference is `-0.0633` baseline minus candidate.
The proxy is `7.138` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-1.24%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 096:** Prior variance `0.00023660`, latest return `+1.45%`, decay `0.90`, and next-session proxy `0.00018225`.
Latest squared return `0.00021025` is `0.889` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00021294`, and the shock contribution is `0.00002102`.
Candidate variance is `0.00023396` versus illustrative baseline `0.00023660`; the update reduces the estimate.
Daily volatility is `1.5296%` for the candidate and `1.5382%` for the baseline.
At the constructed proxy, candidate loss is `-7.5814` and baseline loss is `-7.5789`; their difference is `+0.0025` baseline minus candidate.
The proxy is `0.779` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+1.45%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 097:** Prior variance `0.00023870`, latest return `-1.66%`, decay `0.94`, and next-session proxy `0.00064516`.
Latest squared return `0.00027556` is `1.154` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00022438`, and the shock contribution is `0.00001653`.
Candidate variance is `0.00024091` versus illustrative baseline `0.00023870`; the update raises the estimate.
Daily volatility is `1.5521%` for the candidate and `1.5450%` for the baseline.
At the constructed proxy, candidate loss is `-5.6531` and baseline loss is `-5.6375`; their difference is `+0.0156` baseline minus candidate.
The proxy is `2.678` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-1.66%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 098:** Prior variance `0.00024080`, latest return `+1.87%`, decay `0.97`, and next-session proxy `0.00139129`.
Latest squared return `0.00034969` is `1.452` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00023358`, and the shock contribution is `0.00001049`.
Candidate variance is `0.00024407` versus illustrative baseline `0.00024080`; the update raises the estimate.
Daily volatility is `1.5623%` for the candidate and `1.5518%` for the baseline.
At the constructed proxy, candidate loss is `-2.6176` and baseline loss is `-2.5538`; their difference is `+0.0639` baseline minus candidate.
The proxy is `5.700` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+1.87%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 099:** Prior variance `0.00024290`, latest return `-2.08%`, decay `0.90`, and next-session proxy `0.00010201`.
Latest squared return `0.00043264` is `1.781` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00021861`, and the shock contribution is `0.00004326`.
Candidate variance is `0.00026187` versus illustrative baseline `0.00024290`; the update raises the estimate.
Daily volatility is `1.6183%` for the candidate and `1.5585%` for the baseline.
At the constructed proxy, candidate loss is `-7.8581` and baseline loss is `-7.9029`; their difference is `-0.0448` baseline minus candidate.
The proxy is `0.390` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-2.08%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 100:** Prior variance `0.00024500`, latest return `+2.29%`, decay `0.94`, and next-session proxy `0.00048400`.
Latest squared return `0.00052441` is `2.140` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00023030`, and the shock contribution is `0.00003146`.
Candidate variance is `0.00026176` versus illustrative baseline `0.00024500`; the update raises the estimate.
Daily volatility is `1.6179%` for the candidate and `1.5652%` for the baseline.
At the constructed proxy, candidate loss is `-6.3991` and baseline loss is `-6.3387`; their difference is `+0.0603` baseline minus candidate.
The proxy is `1.849` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+2.29%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 101:** Prior variance `0.00024710`, latest return `-2.50%`, decay `0.97`, and next-session proxy `0.00114921`.
Latest squared return `0.00062500` is `2.529` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00023969`, and the shock contribution is `0.00001875`.
Candidate variance is `0.00025844` versus illustrative baseline `0.00024710`; the update raises the estimate.
Daily volatility is `1.6076%` for the candidate and `1.5719%` for the baseline.
At the constructed proxy, candidate loss is `-3.8141` and baseline loss is `-3.6549`; their difference is `+0.1592` baseline minus candidate.
The proxy is `4.447` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-2.50%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 102:** Prior variance `0.00024920`, latest return `+2.71%`, decay `0.90`, and next-session proxy `0.00004489`.
Latest squared return `0.00073441` is `2.947` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00022428`, and the shock contribution is `0.00007344`.
Candidate variance is `0.00029772` versus illustrative baseline `0.00024920`; the update raises the estimate.
Daily volatility is `1.7255%` for the candidate and `1.5786%` for the baseline.
At the constructed proxy, candidate loss is `-7.9686` and baseline loss is `-8.1171`; their difference is `-0.1485` baseline minus candidate.
The proxy is `0.151` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `+2.71%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 103:** Prior variance `0.00025130`, latest return `-2.92%`, decay `0.94`, and next-session proxy `0.00034596`.
Latest squared return `0.00085264` is `3.393` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00023622`, and the shock contribution is `0.00005116`.
Candidate variance is `0.00028738` versus illustrative baseline `0.00025130`; the update raises the estimate.
Daily volatility is `1.6952%` for the candidate and `1.5852%` for the baseline.
At the constructed proxy, candidate loss is `-6.9509` and baseline loss is `-6.9122`; their difference is `+0.0387` baseline minus candidate.
The proxy is `1.204` times the candidate variance: a single realization cannot establish calibration.
For this case the data owner must confirm the `-2.92%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 104:** Prior variance `0.00025340`, latest return `+0.40%`, decay `0.97`, and next-session proxy `0.00093025`.
Latest squared return `0.00001600` is `0.063` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00024580`, and the shock contribution is `0.00000048`.
Candidate variance is `0.00024628` versus illustrative baseline `0.00025340`; the update reduces the estimate.
Daily volatility is `1.5693%` for the candidate and `1.5919%` for the baseline.
At the constructed proxy, candidate loss is `-4.5318` and baseline loss is `-4.6095`; their difference is `-0.0777` baseline minus candidate.
The proxy is `3.777` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `+0.40%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.
**Laboratory 105:** Prior variance `0.00025550`, latest return `-0.61%`, decay `0.90`, and next-session proxy `0.00179776`.
Latest squared return `0.00003721` is `0.146` times the starting variance; the sign of the return does not affect this update.
The retained contribution is `0.00022995`, and the shock contribution is `0.00000372`.
Candidate variance is `0.00023367` versus illustrative baseline `0.00025550`; the update reduces the estimate.
Daily volatility is `1.5286%` for the candidate and `1.5984%` for the baseline.
At the constructed proxy, candidate loss is `-0.6680` and baseline loss is `-1.2360`; their difference is `-0.5680` baseline minus candidate.
The proxy is `7.694` times the candidate variance: a large realization warrants tail-context review, not an automatic model rejection.
For this case the data owner must confirm the `-0.61%` return and previous variance were available by cutoff; the later proxy cannot be used as an input.

## 15. Quick Reference Card

| Field | Case-specific answer |
|---|---|
| Case | Hypothetical next-session financial volatility forecast for risk review |
| Decision owner | Market-risk manager, with independent model-risk oversight |
| Input | Point-in-time adjusted closes, session calendar, and previous validated variance state |
| Output | Next-session variance and its square-root daily volatility, with baseline and quality status |
| Baseline | Mean squared return across the last 20 eligible sessions |
| Primary method | Proposed EWMA variance recurrence with development-only decay selection |
| Key constraint | No information published or ingested after the decision cutoff may enter the forecast |
| Primary metric | Mean Gaussian quasi-likelihood loss on matched, matured holdout sessions |
| Main guardrail | No unacceptable tail underprediction or unresolved lineage and operating failures |
| Result | Not measured; constructed arithmetic only, with no claimed trading outcome |
| Main failure mode | Retrospective data corrections leak future information into historical features |
| Transferable lesson | Match the data vintage, target horizon, variance units, and decision safeguards before judging forecast sophistication |
