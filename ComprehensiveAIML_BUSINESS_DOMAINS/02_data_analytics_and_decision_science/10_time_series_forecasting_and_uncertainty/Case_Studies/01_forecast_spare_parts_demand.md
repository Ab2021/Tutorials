---
title: "Illustrative Spare-Parts Reorder Decision Using a Hierarchical Probabilistic Forecast"
case_type: "Teaching case"
industry: "Illustrative industrial maintenance"
company: "No real company; all entities, figures, and events are constructed"
decision_date: "Illustrative 2025-04-01"
part_id: "P-417"
part_description: "Hydraulic actuator seal-and-shaft cartridge"
forecast_horizon: "Three-month replenishment lead time"
data_status: "Synthetic illustrative data"
sources: "Open methods and openly available technical documentation only"
currency: "USD, illustrative"
primary_decision: "Whether to reorder intermittent spare part P-417 now"
---

## 1. The One-Line Framing

- An illustrative maintenance organization should reorder P-417 now because a hierarchical probabilistic forecast estimates a 93.8% chance that three-month demand will not exceed two units, while holding only one unit leaves material downtime exposure.
- The decision is not driven by average demand alone.
- The decision is driven by the upper tail of lead-time demand.
- P-417 is intermittent.
- P-417 demand contains many zero-demand months.
- P-417 demand is operationally critical when it occurs.
- P-417 has a long replenishment lead time.
- P-417 has a minimum order quantity of two units.
- P-417 currently has one usable unit on hand.
- P-417 has no confirmed inbound purchase order.
- P-417 has no approved substitute.
- P-417 can cause a production-stopping failure if unavailable.
- The illustrative recommendation is to place an order for two units.
- The policy target is a 95% cycle-service level.
- The unconstrained statistical order recommendation is two units.
- The supplier minimum order quantity is also two units.
- The recommended order therefore requires no MOQ rounding adjustment.
- The forecast uses hierarchy because local part history is sparse.
- The hierarchy borrows information from sister sites and related actuator cartridges.
- The model produces a full predictive distribution rather than a single point forecast.
- The model estimates three-month lead-time demand mean at 0.72 units.
- The model estimates the 90th percentile at two units.
- The model estimates the 95th percentile at three units.
- The model estimates the 98th percentile at four units.
- One unit on hand is below the 90th-percentile inventory target.
- Two units on hand are below the 95th-percentile inventory target only if no inbound order exists.
- Ordering two units raises projected inventory position from one to three units.
- An inventory position of three units meets the 95% lead-time demand target.
- The reorder is economically justified under the illustrative downtime-cost assumptions.
- The reorder remains justified across most tested demand and lead-time sensitivities.
- The decision should be implemented through a controlled purchase requisition.
- The forecast should be refreshed weekly until the order is received.
- The result is illustrative and should not be used as a real procurement instruction.

## 2. Context & Background

- The illustrative organization is called NorthRiver Process Services.
- NorthRiver Process Services is fictional.
- NorthRiver operates six fictitious process plants.
- Each plant uses hydraulically actuated isolation valves.
- Each valve has a replaceable actuator cartridge.
- The cartridge contains seals, a shaft, and a hydraulic pressure chamber.
- P-417 is the internal spare-part code for one actuator cartridge.
- P-417 fits one actuator family.
- The actuator family is used at three of the six plants.
- The plants are called Alder, Bay, Cedar, Dune, Elm, and Flint.
- P-417 is stocked centrally at the Alder distribution store.
- Emergency transfers from other plants are possible but unreliable.
- P-417 is normally sourced from one qualified manufacturer.
- The manufacturer is fictional.
- The supplier lead time is nominally 12 weeks.
- The supplier lead time has historically varied between 9 and 17 weeks.
- The supplier requires a minimum order quantity of two units.
- The supplier normally ships by ground freight.
- Expedited air freight is possible only after production completion.
- Expedited freight does not remove supplier production lead time.
- A valve failure without a cartridge can stop a production train.
- A stopped production train creates high downtime exposure.
- The illustrative downtime exposure is estimated at USD 16,000 per shortage unit.
- The shortage estimate includes lost contribution, emergency labor, and restart effort.
- The shortage estimate does not include reputational damage.
- The shortage estimate does not include environmental penalties.
- The part unit purchase price is USD 480.
- The illustrative annual carrying-cost rate is 18%.
- The resulting monthly holding cost is USD 7.20 per unit.
- The organization applies an additional handling-and-space charge of USD 10 per unit per month.
- The illustrative all-in holding cost is therefore USD 17.20 per unit per month.
- The organization uses a quarterly formal review for slow-moving spares.
- The organization also allows exception-based weekly review for critical items.
- P-417 is classified as criticality class A.
- Criticality class A means a stockout can stop production.
- P-417 is classified as intermittent demand.
- Intermittent means demand occurs infrequently and irregularly.
- P-417 had demand in only seven of the previous 36 months.
- P-417 had zero demand in 29 of the previous 36 months.
- When demand occurred, the order size was usually one unit.
- A two-unit demand month occurred once.
- Historical demand is too sparse for a reliable local-only time-series model.
- A simple local average would obscure uncertainty.
- A simple local average would underestimate the probability of clustered failures.
- A simple local average would ignore related-part evidence.
- Related actuator cartridges share operating environment and failure modes.
- Sister sites share similar valve duty cycles.
- Sister sites experience similar contamination and pressure-shock risks.
- The organization has maintenance work-order data.
- The organization has inventory-transaction data.
- The organization has purchase-order lead-time data.
- The organization has asset-register data.
- The organization has part-master data.
- The organization has equipment-family relationships.
- The organization does not have perfect failure-cause coding.
- The organization does not have complete sensor histories for every actuator.
- The organization does not assume future failures will exactly repeat past failures.
- The organization uses forecasts as decision support rather than as certainty claims.
- The central store currently has one physically verified usable P-417 unit.
- The unit passed a packaging and preservation check.
- The unit is not reserved for another approved work order.
- The unit is not under quality hold.
- The unit has no known shelf-life restriction before the next review date.
- There are no open P-417 purchase orders.
- There are no approved substitute cartridges.
- There are no recoverable field returns available for immediate remanufacture.
- A repair vendor exists but has a 20-week turnaround.
- The repair vendor therefore cannot protect the current replenishment horizon.
- The decision owner is the fictional Reliability and Materials Review Board.
- The board includes maintenance, procurement, finance, and operations representatives.
- The board wants a transparent answer rather than an opaque model score.
- The board requires all assumptions to be visible.
- The board requires the forecast to be reproducible.
- The board requires a documented fallback if data pipelines fail.
- The board requires a clear distinction between fact, estimate, and policy choice.
- This case demonstrates that distinction.

## 3. The Business Question

- Should NorthRiver reorder P-417 immediately?
- If yes, what quantity should NorthRiver order?
- If yes, what service target should determine the reorder quantity?
- If no, what evidence would justify deferring the purchase?
- What is the probability that one unit on hand will be insufficient during lead time?
- What is the probability that two units on hand will be insufficient during lead time?
- What is the probability that three units on hand will be insufficient during lead time?
- What is the expected shortage under each inventory position?
- What is the incremental economic value of ordering now?
- How sensitive is the decision to lead-time uncertainty?
- How sensitive is the decision to demand-rate uncertainty?
- How sensitive is the decision to downtime-cost uncertainty?
- How sensitive is the decision to the supplier minimum order quantity?
- How should the forecast use sparse local demand history?
- How should the forecast use related sites without overgeneralizing?
- How should the system avoid demand-data leakage?
- How should the system respond if an urgent reservation appears after reorder approval?
- How should the system respond if the supplier confirms a delayed shipment?
- How should the decision be audited six months later?
- The primary decision metric is lead-time cycle-service level.
- The secondary decision metric is expected shortage cost.
- The tertiary decision metric is expected inventory carrying cost.
- The decision horizon is the replenishment lead time.
- The planning interval is one month.
- The operational review cadence is weekly.
- The formal replenishment policy review cadence is quarterly.
- The baseline policy target is 95% cycle-service level.
- The 95% target is a policy choice.
- The 95% target is not a model output.
- The forecast distribution is a model output.
- The reorder quantity is the interaction of forecast, inventory position, and policy.
- The final purchase quantity is also constrained by supplier rules.
- The business question is therefore not “what is average demand?”
- The business question is “what inventory position protects the operation at the chosen service target?”

## 4. Assumptions & Constraints

- All numerical values in this case are illustrative.
- All company names in this case are fictional.
- All supplier names in this case are fictional.
- All demand records in this case are synthetic.
- The forecast horizon is three months.
- The three-month horizon represents the nominal 12-week lead time.
- Demand is measured in whole spare-part units.
- Negative demand is impossible.
- Fractional physical demand is impossible.
- Forecast means may be fractional because they represent expected values.
- The central store is the stocking point under analysis.
- Local plant cabinets are excluded from central inventory position.
- No field-installed cartridge is considered inventory.
- One unit on hand is physically verified.
- The current on-hand quantity is one unit.
- Reserved quantity is zero units.
- Quality-hold quantity is zero units.
- Inbound quantity is zero units.
- Backorder quantity is zero units.
- Inventory position is therefore one unit.
- Inventory position is defined as on hand plus inbound minus reservations minus backorders.
- Inventory position excludes unapproved requisitions.
- Inventory position excludes supplier verbal promises.
- Inventory position excludes damaged stock.
- The reorder trigger uses inventory position rather than on-hand quantity.
- The lead time includes supplier production and transportation.
- The lead time does not include internal approval delay once the requisition is released.
- Internal approval delay is assumed to be one business day.
- The supplier lead-time distribution is represented as uncertain.
- The baseline lead-time scenario is three monthly periods.
- A shorter lead-time scenario is 2.5 monthly periods.
- A longer lead-time scenario is four monthly periods.
- The forecast assumes no major fleet redesign during the horizon.
- The forecast assumes no planned shutdown removing all demand exposure.
- The forecast assumes the three active plants remain operational.
- The forecast assumes no known contamination event.
- The forecast assumes no known campaign replacement program.
- The forecast assumes no systematic double-counting of transfers.
- The forecast assumes work orders are correctly linked to part P-417 when issued.
- The forecast treats emergency substitutions as excluded because none is approved.
- The forecast treats planned replacement demand separately from corrective demand.
- The baseline forecast includes corrective demand only.
- A known planned overhaul would be added deterministically if scheduled.
- No planned overhaul is scheduled in the next three months.
- The downtime-cost estimate is USD 16,000 per shortage unit.
- The downtime-cost estimate is intentionally conservative relative to a full production-loss estimate.
- The unit purchase price is USD 480.
- The all-in monthly holding cost is USD 17.20.
- The supplier minimum order quantity is two units.
- The supplier order multiple is one unit above the minimum.
- The supplier offers no quantity discount at two versus three units.
- The supplier cancellation window closes after production starts.
- The supplier permits rescheduling before production starts.
- The spare has a 10-year assumed technical shelf life if preserved.
- The case assumes proper storage conditions.
- The case assumes no obsolescence within one year.
- The case assumes no major engineering revision within one year.
- The 95% cycle-service target is approved for class-A intermittent spares.
- The 95% target is not equivalent to a 95% fill rate.
- The cycle-service target means no stockout during a replenishment cycle with 95% probability.
- The forecast model uses partial pooling across hierarchy levels.
- The model does not assume all sites have identical demand rates.
- The model does not assume all related parts have identical demand rates.
- The model permits site-specific deviations.
- The model permits part-family deviations.
- The model uses posterior predictive uncertainty.
- The model includes parameter uncertainty.
- The model includes demand-process variability.
- The model does not fully model correlated multi-asset shock failures.
- A separate stress scenario addresses correlated shocks.
- The baseline decision is not based on the stress scenario alone.
- The decision must be understandable to procurement staff.
- The decision must be executable in the ERP system.
- The decision must include an exception path.
- The decision must include an audit record.
- The decision must not expose confidential real-company data.
- The decision must cite only open methodological sources.

## 5. Data, Tools & Method

### 5a. Data Specification

- The synthetic demand-history window contains 36 completed calendar months.
- The demand-history start month is illustrative 2022-04.
- The demand-history end month is illustrative 2025-03.
- The target stocking location is Alder Central Store.
- The target part is P-417.
- The target demand series records issue quantity by month.
- Demand is counted when a part is issued to a corrective work order.
- Demand is not counted when a part is merely reserved.
- Demand is not counted when a part is received from a supplier.
- Demand is not counted when a part is transferred between bins without consumption.
- Demand is not counted when a transaction is reversed.
- Demand is not counted twice when a reservation later becomes an issue.
- The local P-417 demand total across 36 months is eight units.
- The local P-417 demand count across 36 months is seven nonzero months.
- The local P-417 monthly mean is 8 divided by 36.
- The local P-417 monthly mean equals 0.222 units.
- The local P-417 raw mean is not used alone as the final forecast.
- The related-family dataset contains 18 actuator cartridge part numbers.
- The related-family dataset covers 216 part-site monthly series.
- The related-family dataset spans the same 36-month period.
- The related-family dataset contains 7,776 part-site-month observations.
- The hierarchy includes enterprise, plant, equipment family, and part levels.
- The enterprise level represents all eligible actuator-cartridge demand.
- The plant level represents site-specific operating conditions.
- The equipment-family level represents actuator design commonality.
- The part level represents P-417-specific behavior.
- The asset-register table maps installed assets to equipment family.
- The work-order table maps corrective jobs to issue transactions.
- The purchasing table records order-release and receipt dates.
- The inventory table records on-hand, reserved, quality-hold, and inbound quantities.
- The part-master table records MOQ, unit cost, shelf-life, and criticality.
- The supplier table records lead-time observations.
- The supplier lead-time sample contains 24 historical purchase orders.
- The observed lead-time median is 11.8 weeks.
- The observed lead-time mean is 12.4 weeks.
- The observed lead-time standard deviation is 2.1 weeks.
- The observed lead-time minimum is 9.1 weeks.
- The observed lead-time maximum is 16.8 weeks.
- The lead-time data are synthetic but internally consistent.
- The demand timestamp is the issue-posting date.
- The demand timestamp is not the work-order close date.
- The issue-posting date is chosen to avoid future-information leakage.
- A work order opened in March but issued in April counts in April.
- A late transaction adjustment is back-posted only if the ERP audit trail shows the physical issue date.
- Demand corrections are versioned rather than silently overwritten.
- Each source extract has an extraction timestamp.
- Each source extract has a row count.
- Each source extract has a checksum in the reproducibility record.
- The model input contains only information available by the decision date.
- The decision date is 2025-04-01.
- The forecast training cutoff is 2025-03-31 23:59 local time.
- No April demand transactions are included in training.
- No future supplier confirmations are included in training.
- No future maintenance findings are included in training.
- The data grain is part-site-month.
- The aggregation grain for the decision is central-store-month.
- Transfers are reconciled before central-store aggregation.
- Duplicate issue-and-return pairs are removed.
- Returns are not treated as negative demand when the part is reusable stock.
- Scrapped returns are retained as evidence of consumption.
- The part is treated as non-repairable within the forecast horizon.
- The part family includes similar cartridge designs, not unrelated seals.
- The family excludes parts with incompatible pressure ratings.
- The family excludes obsolete first-generation actuator designs.
- The family excludes planned project demand.
- The family excludes commissioning demand for new plants.
- The target series excludes capital-project consumption.
- The target series excludes training and demonstration consumption.
- The target series includes corrective maintenance consumption.
- The target series includes emergency replacement consumption.
- The target series includes failure-induced preventive replacement if initiated by a detected defect.
- The target series excludes routine calendar-based replacement.
- The model stores a flag for every exclusion.
- The model stores a reason code for every manual adjustment.
- The model retains raw data separately from curated data.
- The model retains curated data separately from forecast features.
- The model never writes forecast values back into source demand tables.
- The decision record stores the exact inventory snapshot used.
- The decision record stores the exact model version used.
- The decision record stores the exact policy target used.
- The decision record stores the exact supplier constraints used.

### 5b. Tools & Stack

- The illustrative data store is PostgreSQL.
- The illustrative transformation layer is SQL.
- The illustrative analysis language is Python.
- The illustrative probabilistic framework is PyMC-compatible Bayesian modeling.
- The illustrative tabular library is pandas.
- The illustrative numerical library is NumPy.
- The illustrative posterior diagnostics library is ArviZ.
- The illustrative scheduling tool is Apache Airflow-compatible orchestration.
- The illustrative model registry is MLflow-compatible.
- The illustrative source-control system is Git.
- The illustrative container runtime is Docker.
- The illustrative ERP integration uses a REST API adapter.
- The illustrative dashboard uses a business-intelligence tool.
- The illustrative alerting channel is email and workflow notification.
- The illustrative data-quality checks run before model fitting.
- The illustrative forecast job runs weekly.
- The illustrative inventory snapshot runs daily.
- The illustrative reorder evaluation runs after each forecast job.
- The illustrative purchase-requisition creation requires human approval.
- The illustrative system does not automatically release purchase orders.
- The illustrative system logs every recommendation.
- The illustrative system logs every override.
- The illustrative system logs every approval.
- The illustrative system logs every rejection.
- The illustrative system logs every later outcome.
- The illustrative code environment is pinned by lockfile.
- The illustrative model random seed is recorded.
- The illustrative posterior draw count is recorded.
- The illustrative posterior chain count is recorded.
- The illustrative dataset version is recorded.
- The illustrative forecast output is serialized as JSON.
- The illustrative forecast output includes quantiles.
- The illustrative forecast output includes full probability mass function values.
- The illustrative forecast output includes scenario labels.
- The illustrative forecast output includes uncertainty intervals.
- The illustrative dashboard displays only rounded operational values.
- The audit store preserves unrounded model values.
- The illustrative stack can be replaced by equivalent open-source components.
- The method does not depend on a proprietary forecasting engine.

### 5c. Method Summary

- The method forecasts lead-time demand rather than next-month demand alone.
- The method models intermittent count demand.
- The method uses a negative-binomial posterior predictive distribution.
- The method applies hierarchical partial pooling.
- Partial pooling stabilizes estimates for sparse series.
- The local P-417 series contributes direct evidence.
- The related actuator-cartridge family contributes indirect evidence.
- The plant level captures site-specific deviation.
- The enterprise level supplies a broad baseline.
- The model estimates a monthly demand intensity for P-417.
- The model propagates uncertainty into three-month demand.
- The model incorporates lead-time uncertainty through scenario integration.
- The baseline lead-time demand distribution has mean 0.72 units.
- The baseline distribution uses a negative-binomial approximation.
- The baseline negative-binomial mean is 0.72.
- The baseline negative-binomial dispersion parameter is 1.80.
- The negative-binomial parameterization uses mean μ and size r.
- The probability parameter is p equals r divided by r plus μ.
- With μ equal to 0.72 and r equal to 1.80, p equals 0.714286.
- The probability of zero lead-time demand is p raised to r.
- The probability of one demand unit follows the negative-binomial recurrence.
- The model calculates inventory targets as predictive quantiles.
- The reorder point equals the required protection level.
- The required protection level equals the selected demand quantile.
- The reorder quantity equals target stock minus inventory position.
- The order quantity is then adjusted for MOQ and order multiples.
- The system flags an order if the adjusted quantity exceeds a governance threshold.
- The system calculates expected shortage for each candidate stock level.
- The system calculates expected holding cost for each candidate stock level.
- The system calculates a transparent illustrative economic comparison.
- The system reports both policy-based and cost-based recommendations.
- The system does not replace human criticality review.
- The system does not claim causal certainty about failure mechanisms.
- The system uses backtesting before deployment.
- The system uses calibration checks after deployment.
- The system uses scenario stress tests for rare shocks.
- The system uses a fallback rule when model inputs fail validation.

### 5d. Provenance, Quality, Exclusions, Leakage

- All figures are constructed for instruction.
- No real company data are represented.
- No proprietary database was accessed.
- Methodological references are open.
- The demand data provenance is synthetic ERP issue records.
- The inventory data provenance is synthetic daily stock snapshots.
- The lead-time provenance is synthetic purchase-order history.
- The asset data provenance is synthetic asset-register records.
- The family mapping provenance is synthetic engineering master data.
- The demand series passed a duplicate-transaction check.
- The demand series passed a nonnegative-quantity check.
- The demand series passed a valid-part-code check.
- The demand series passed a valid-location check.
- The demand series passed a period-completeness check.
- The demand series passed a reversal-pair reconciliation check.
- The inventory snapshot passed a physical-count confirmation check.
- The inventory snapshot passed a reservation reconciliation check.
- The purchase-order data passed a received-after-released-date check.
- The purchase-order data passed an outlier flagging check.
- Lead-time outliers were retained unless explained as data errors.
- The 16.8-week lead time was retained as a plausible delay.
- The 9.1-week lead time was retained as a plausible fast delivery.
- The model excludes future actual demand.
- The model excludes future inventory receipts.
- The model excludes future supplier confirmations.
- The model excludes future work-order completion information.
- The model excludes maintenance notes entered after the forecast cutoff.
- The model excludes demand records corrected after cutoff unless physically dated before cutoff.
- The model avoids target leakage from future consumption.
- The model avoids target leakage from later failure classifications.
- The model avoids target leakage from future reorder actions.
- The model does not use stockout-censored zero demand as true zero demand without review.
- The case found no historical P-417 stockout months in the training window.
- If stockouts had occurred, censored-demand treatment would be required.
- The model excludes demand caused by one-time capital projects.
- The model excludes demand caused by new-facility commissioning.
- The model excludes routine overhaul demand because it is planned separately.
- The model excludes incompatible parts from the hierarchy.
- The model excludes obsolete parts with retired asset populations.
- The model logs all exclusions.
- The model assigns data-quality status as green, amber, or red.
- The current illustrative data-quality status is green.
- Green means all critical checks passed.
- Amber means forecast may run but requires reviewer acknowledgement.
- Red means forecast recommendation is suppressed.
- A red status triggers fallback policy review.
- The data-quality status is stored with every recommendation.
- The forecast provenance record contains dataset hash `illustrative-2025-04-01-v1`.
- The model provenance record contains version `hpif-p417-0.4.0`.
- The policy provenance record contains version `critical-spares-policy-2.1`.
- The decision provenance record contains timestamp `2025-04-01T09:00:00`.
- Provenance is necessary because a correct number without context is not auditable.

### 5e. Reproducibility/Method-Selection Ledger

| Ledger Item | Recorded Choice | Rationale |
|---|---:|---|
| Decision part | P-417 | Critical intermittent spare |
| Decision date | 2025-04-01 | Illustrative review point |
| Training cutoff | 2025-03-31 | Prevents future leakage |
| Demand grain | Part-site-month | Matches intermittent demand and review cadence |
| Forecast horizon | Three months | Nominal supplier lead time |
| Inventory measure | Inventory position | Includes inbound and commitments |
| On-hand quantity | 1 | Verified synthetic snapshot |
| Inbound quantity | 0 | No open purchase order |
| Reserved quantity | 0 | No approved reservation |
| Backorders | 0 | No unresolved demand |
| Current inventory position | 1 | 1 plus 0 minus 0 minus 0 |
| Service target | 95% CSL | Class-A policy choice |
| Model family | Hierarchical negative binomial | Handles sparse overdispersed counts |
| Local history length | 36 months | Balances recency and sample size |
| Local demand total | 8 units | Synthetic historical observation |
| Nonzero local months | 7 | Confirms intermittency |
| Hierarchy level 1 | Enterprise | Broad baseline |
| Hierarchy level 2 | Plant | Site effects |
| Hierarchy level 3 | Equipment family | Engineering similarity |
| Hierarchy level 4 | Part | P-417 specificity |
| Lead-time representation | Distribution/scenarios | Captures supplier variability |
| Baseline lead time | 12 weeks | Historical central tendency |
| Baseline mean demand | 0.72 units | Posterior predictive result |
| Baseline dispersion | 1.80 | Posterior predictive approximation |
| Demand distribution | Negative binomial | Nonnegative count distribution |
| Quantile convention | Smallest q with CDF at least target | Operationally unambiguous |
| MOQ | 2 units | Supplier rule |
| Order multiple | 1 unit | Supplier rule |
| Unit cost | USD 480 | Illustrative cost |
| Holding cost | USD 17.20/month | Carrying plus handling assumption |
| Shortage cost | USD 16,000/unit | Conservative downtime proxy |
| Model draws | 20,000 | Stable illustrative posterior simulation |
| Posterior chains | 4 | Diagnostic redundancy |
| Random seed | 4172025 | Reproducibility |
| Validation method | Rolling-origin backtest | Respects time order |
| Calibration metric | Quantile coverage | Tests uncertainty quality |
| Comparator 1 | Local historical mean | Simple baseline |
| Comparator 2 | Croston-style intermittent forecast | Intermittent-demand baseline |
| Comparator 3 | Hierarchical mean model | Pooling without full dispersion |
| Selected model | Hierarchical negative binomial | Best calibration and decision fit |
| Override authority | Review Board | Governance control |
| Fallback method | Conservative rule-based reorder | Continuity during model failure |
| Fallback trigger | Red data-quality status | Prevents unsafe automation |
| Recommendation owner | Materials planner | Operational accountability |
| Approval owner | Reliability and Materials Review Board | Cross-functional governance |
| Audit retention | Seven years | Illustrative control requirement |
| Open reference type | Documentation and academic preprints/articles | Transparent method basis |

## 6. Analysis Walkthrough

- The analysis begins with the current inventory position.
- On-hand usable inventory equals one unit.
- Confirmed inbound inventory equals zero units.
- Reserved inventory equals zero units.
- Backorders equal zero units.
- Inventory position equals one plus zero minus zero minus zero.
- Current inventory position equals one unit.
- The analysis horizon equals the supplier replenishment lead time.
- The baseline horizon equals three months.
- The model forecasts total demand during those three months.
- The model does not merely forecast demand in the next calendar month.
- The local raw mean is eight units divided by 36 months.
- Eight divided by 36 equals 0.2222 units per month.
- A local-only three-month mean would equal 0.6667 units.
- The local-only mean is close to the hierarchical mean by coincidence.
- The local-only estimate has greater uncertainty because it uses only eight observed units.
- The hierarchy recognizes that related cartridges also have intermittent failures.
- The hierarchy recognizes that Alder and Bay have somewhat higher observed actuator usage.
- The hierarchy shrinks an extreme local estimate toward related evidence.
- The hierarchy does not force P-417 to equal the family average.
- The resulting posterior predictive three-month mean is 0.72 units.
- The posterior predictive distribution is approximated as negative binomial.
- The negative-binomial mean is μ equals 0.72.
- The negative-binomial size is r equals 1.80.
- The negative-binomial probability parameter is p equals r divided by r plus μ.
- Substituting values gives p equals 1.80 divided by 2.52.
- The calculated p equals 0.7142857.
- The complementary probability is one minus p.
- One minus p equals 0.2857143.
- The probability of zero demand is p to the power r.
- P(D equals 0) equals 0.7142857 to the power 1.80.
- P(D equals 0) equals approximately 0.5455.
- The probability of one demand unit is P0 times r times one-minus-p.
- P(D equals 1) equals 0.5455 times 1.80 times 0.2857143.
- P(D equals 1) equals approximately 0.2805.
- The probability of two demand units uses recurrence.
- P(D equals 2) equals P1 times r plus one divided by two times one-minus-p.
- P(D equals 2) equals 0.2805 times 2.80 divided by 2 times 0.2857143.
- P(D equals 2) equals approximately 0.1122.
- The probability of three demand units uses recurrence.
- P(D equals 3) equals P2 times r plus two divided by three times one-minus-p.
- P(D equals 3) equals 0.1122 times 3.80 divided by 3 times 0.2857143.
- P(D equals 3) equals approximately 0.0406.
- The probability of four demand units uses recurrence.
- P(D equals 4) equals P3 times r plus three divided by four times one-minus-p.
- P(D equals 4) equals 0.0406 times 4.80 divided by 4 times 0.2857143.
- P(D equals 4) equals approximately 0.0139.
- The probability of five demand units uses recurrence.
- P(D equals 5) equals P4 times r plus four divided by five times one-minus-p.
- P(D equals 5) equals 0.0139 times 5.80 divided by 5 times 0.2857143.
- P(D equals 5) equals approximately 0.0046.
- The remaining tail probability above five units is small but not zero.
- The probability mass function is operationally more useful than a single average.
- The cumulative probability at zero units equals 0.5455.
- The cumulative probability at one unit equals 0.5455 plus 0.2805.
- The cumulative probability at one unit equals 0.8260.
- The cumulative probability at two units equals 0.8260 plus 0.1122.
- The cumulative probability at two units equals 0.9382.
- The cumulative probability at three units equals 0.9382 plus 0.0406.
- The cumulative probability at three units equals 0.9788.
- The cumulative probability at four units equals 0.9788 plus 0.0139.
- The cumulative probability at four units equals 0.9927.
- The 50th percentile is zero units because CDF at zero exceeds 0.50.
- The 80th percentile is one unit because CDF at one exceeds 0.80.
- The 90th percentile is two units because CDF at one is below 0.90 and CDF at two is above 0.90.
- The 95th percentile is three units because CDF at two is below 0.95 and CDF at three is above 0.95.
- The 98th percentile is four units because CDF at three is below 0.98 and CDF at four is above 0.98.
- The 99th percentile is four units because CDF at four exceeds 0.99.
- A 95% cycle-service policy therefore requires a protection level of three units.
- Current inventory position is one unit.
- The unconstrained order quantity equals target stock minus inventory position.
- For the 95% target, unconstrained order quantity equals three minus one.
- The 95% unconstrained order quantity equals two units.
- The supplier MOQ equals two units.
- The MOQ-adjusted purchase quantity equals two units.
- The post-order inventory position equals current inventory position plus order quantity.
- The post-order inventory position equals one plus two.
- The post-order inventory position equals three units.
- Three units provide approximately 97.88% cycle-service under the baseline distribution.
- The achieved service exceeds the 95% policy threshold.
- The policy overachievement is due to discrete unit quantities.
- A quantity of one additional unit would create inventory position two.
- Inventory position two provides approximately 93.82% cycle-service.
- Inventory position two falls below the 95% policy target.
- A quantity of zero additional units retains inventory position one.
- Inventory position one provides approximately 82.60% cycle-service.
- Inventory position one creates approximately 17.40% stockout probability.
- A 17.40% stockout probability is not acceptable for the selected class-A policy.
- The forecast mean alone could mislead the decision.
- Mean lead-time demand is only 0.72 units.
- Ordering based on mean demand minus stock would suggest no order.
- That mean-based rule ignores the 17.40% probability that demand exceeds one unit.
- The quantile-based rule correctly focuses on the service objective.
- Expected shortage provides a second interpretation.
- Expected shortage at stock level one equals E[max(D minus 1, 0)].
- For nonnegative integer demand, E[min(D,1)] equals probability demand is at least one.
- Probability demand is at least one equals one minus P0.
- Probability demand is at least one equals 1 minus 0.5455.
- Probability demand is at least one equals 0.4545.
- Expected shortage at stock one equals mean demand minus expected minimum demand.
- Expected shortage at stock one equals 0.72 minus 0.4545.
- Expected shortage at stock one equals 0.2655 units.
- Expected shortage at stock level two equals E[max(D minus 2, 0)].
- E[min(D,2)] equals P(D at least 1) plus P(D at least 2).
- P(D at least 1) equals 0.4545.
- P(D at least 2) equals 1 minus CDF at one.
- P(D at least 2) equals 1 minus 0.8260.
- P(D at least 2) equals 0.1740.
- E[min(D,2)] equals 0.4545 plus 0.1740.
- E[min(D,2)] equals 0.6285.
- Expected shortage at stock two equals 0.72 minus 0.6285.
- Expected shortage at stock two equals 0.0915 units.
- Expected shortage at stock level three uses one additional tail term.
- P(D at least 3) equals 1 minus CDF at two.
- P(D at least 3) equals 1 minus 0.9382.
- P(D at least 3) equals 0.0618.
- E[min(D,3)] equals 0.4545 plus 0.1740 plus 0.0618.
- E[min(D,3)] equals 0.6903.
- Expected shortage at stock three equals 0.72 minus 0.6903.
- Expected shortage at stock three equals 0.0297 units.
- Moving from stock one to stock three reduces expected shortage.
- The expected-shortage reduction equals 0.2655 minus 0.0297.
- The expected-shortage reduction equals 0.2358 units.
- At USD 16,000 per shortage unit, avoided shortage exposure equals 0.2358 times 16,000.
- Avoided shortage exposure equals approximately USD 3,773.
- Ordering two units costs two times USD 480.
- Purchase cash outlay equals USD 960.
- The purchase is not treated as a pure loss because inventory retains residual value.
- The incremental carrying exposure depends on how long the units remain in stock.
- A conservative four-month incremental holding estimate is used.
- Two units times four months times USD 17.20 equals USD 137.60.
- The simplified incremental economic comparison is USD 3,773 avoided exposure minus USD 137.60 carrying exposure.
- The simplified comparison excludes residual inventory value.
- The simplified comparison equals approximately USD 3,635 favorable before considering purchase residual value.
- Even if the downtime-cost estimate were much lower, the service policy still supports reorder.
- The next step tests sensitivity rather than treating the baseline as certain.

| Lead-Time Demand Quantity | Probability | Cumulative Probability | Interpretation |
|---:|---:|---:|---|
| 0 | 0.5455 | 0.5455 | No demand during lead time |
| 1 | 0.2805 | 0.8260 | One-unit demand |
| 2 | 0.1122 | 0.9382 | Two-unit demand |
| 3 | 0.0406 | 0.9788 | Three-unit demand |
| 4 | 0.0139 | 0.9927 | Four-unit demand |
| 5 | 0.0046 | 0.9973 | Five-unit demand |
| 6 or more | 0.0027 | 1.0000 | Extreme tail |

| Service Target | Required Inventory Position | Current Position | Unconstrained Order | MOQ-Adjusted Order | Achieved Baseline CSL |
|---:|---:|---:|---:|---:|---:|
| 80% | 1 | 1 | 0 | 0 | 82.60% |
| 90% | 2 | 1 | 1 | 2 | 97.88% |
| 95% | 3 | 1 | 2 | 2 | 97.88% |
| 98% | 4 | 1 | 3 | 3 | 99.27% |
| 99% | 4 | 1 | 3 | 3 | 99.27% |

- The 90% policy would statistically call for one unit.
- The MOQ would force a two-unit purchase under a 90% policy.
- The 95% policy also calls for two units.
- The MOQ therefore does not distort the 95% recommendation.
- The 98% policy would call for three units.
- The board did not select 98% because it exceeds current class-A policy without a special risk trigger.
- The 80% policy would require no order.
- The 80% policy is inappropriate for a production-stopping spare in this case.
- The analysis now tests changes in demand mean.
- Low-demand sensitivity uses mean lead-time demand of 0.50 units.
- Baseline-demand sensitivity uses mean lead-time demand of 0.72 units.
- High-demand sensitivity uses mean lead-time demand of 1.00 unit.
- Severe-demand sensitivity uses mean lead-time demand of 1.35 units.
- Dispersion is held at 1.80 for comparability.
- The low-demand case reflects favorable operating conditions.
- The high-demand case reflects elevated actuator stress.
- The severe-demand case reflects a plausible contamination episode.
- The analysis also tests lead-time extension.
- A four-month lead-time case increases mean demand proportionally.
- Four-month mean demand equals 0.72 times four divided by three.
- Four-month mean demand equals 0.96 units.
- A five-month delayed-supplier stress case gives mean demand 1.20 units.
- The stress case is not the baseline.
- The stress case informs escalation thresholds.

| Scenario | Lead-Time Mean Demand | Approximate 95% Quantile | Current Position | Recommended Order |
|---|---:|---:|---:|---:|
| Favorable demand | 0.50 | 2 | 1 | 2 due to MOQ |
| Baseline demand | 0.72 | 3 | 1 | 2 |
| Elevated demand | 1.00 | 3 | 1 | 2 |
| Severe demand | 1.35 | 4 | 1 | 3 |
| Baseline demand with four-month lead time | 0.96 | 3 | 1 | 2 |
| Baseline demand with five-month lead time | 1.20 | 4 | 1 | 3 |

- The baseline recommendation is stable under favorable demand.
- The baseline recommendation is stable under elevated demand.
- The baseline recommendation changes only under severe demand or major lead-time delay.
- The decision is therefore moderately robust.
- The decision is not insensitive to extreme operational changes.
- A known contamination event would trigger reassessment.
- A supplier confirmation beyond 16 weeks would trigger reassessment.
- A new reservation for one unit would trigger reassessment.
- A quality hold on the existing unit would trigger reassessment.
- An approved substitute would trigger reassessment.
- A scheduled overhaul would trigger deterministic demand addition.
- The model output should therefore be monitored, not treated as static.

## 7. The Finding

- The illustrative recommendation is to reorder P-417 immediately.
- The recommended purchase quantity is two units.
- The recommendation is based on a 95% cycle-service target.
- The current inventory position is one unit.
- The 95% protection target is three units.
- The resulting statistical gap is two units.
- The supplier MOQ is two units.
- The supplier MOQ exactly matches the statistical gap.
- The post-order inventory position will be three units.
- The baseline probability that three units are sufficient is 97.88%.
- The baseline probability that one unit is sufficient is 82.60%.
- The reorder improves estimated cycle service by 15.28 percentage points.
- The reorder reduces stockout probability from 17.40% to 2.12%.
- The reorder reduces expected shortage from 0.2655 to 0.0297 units.
- The reorder reduces expected shortage by 0.2358 units.
- The illustrative avoided shortage exposure is approximately USD 3,773.
- The illustrative four-month carrying exposure is approximately USD 138.
- The order cash outlay is USD 960.
- The purchased units retain inventory value if not consumed during lead time.
- The recommendation remains two units under low, baseline, and elevated demand scenarios.
- The recommendation increases to three units only in severe demand or major-delay scenarios.
- The decision should include a weekly exception review.
- The decision should be cancelled or modified only if new inventory or approved substitutes materially change inventory position.
- The decision should be escalated if supplier confirmation exceeds 16 weeks.
- The decision should be escalated if an urgent reservation consumes the existing unit.
- The decision should be documented as policy-conforming.
- The decision should not be described as certain.
- The decision should be described as risk-managed under stated assumptions.
- The finding is illustrative rather than a real procurement directive.

## 8. System/Solution Design

- The solution is designed as a decision-support system.
- The solution separates source data from forecast data.
- The solution separates forecast data from purchase execution.
- The solution requires human approval before purchase-order release.
- The solution uses a weekly forecast batch.
- The solution uses a daily inventory snapshot.
- The solution evaluates reorder recommendations after both processes complete.
- The solution creates exceptions when data quality is insufficient.
- The solution creates exceptions when business conditions invalidate assumptions.
- The solution preserves audit evidence.

```text
[ERP Inventory]
      |
      v
[Inventory Snapshot Validator] -----> [Data Quality Log]
      |
      v
[Curated Inventory Position]
      |
      +------------------------------+
                                     |
[ERP Work Orders]                    |
      |                              |
      v                              |
[Demand Transaction Curator]         |
      |                              |
      v                              |
[Hierarchical Demand Dataset] -------+
      |
      v
[Probabilistic Forecast Service] <--- [Part Master / Asset Hierarchy]
      |
      v
[Lead-Time Demand Distribution] <---- [Purchase Order Lead-Time History]
      |
      v
[Policy and Constraint Engine]
      |
      +--> [Recommendation Dashboard]
      |
      +--> [Exception Workflow]
      |
      +--> [Approval Record]
      |
      v
[ERP Purchase Requisition Draft]
      |
      v
[Human Approval and Release]
```

- The ERP inventory source provides on-hand quantity.
- The ERP inventory source provides reservation quantity.
- The ERP inventory source provides quality-hold quantity.
- The ERP purchasing source provides confirmed inbound quantity.
- The work-order source provides corrective issue transactions.
- The part-master source provides MOQ and cost.
- The asset-register source provides hierarchy mappings.
- The purchasing-history source provides lead-time observations.
- The data-curation layer removes duplicates.
- The data-curation layer reconciles reversals.
- The data-curation layer maps transactions to decision periods.
- The data-curation layer labels planned and corrective demand.
- The feature layer computes local and hierarchical histories.
- The forecast service fits the hierarchical count model.
- The forecast service produces posterior predictive draws.
- The forecast service aggregates draws across lead-time periods.
- The policy engine converts draws into quantiles.
- The policy engine compares quantiles with inventory position.
- The policy engine applies MOQ and order multiples.
- The policy engine applies approval thresholds.
- The policy engine computes expected shortage.
- The policy engine computes sensitivity scenarios.
- The dashboard shows the recommendation.
- The dashboard shows the forecast distribution.
- The dashboard shows current inventory position.
- The dashboard shows assumptions.
- The dashboard shows data-quality status.
- The dashboard shows changes since the previous run.
- The dashboard shows override options.
- The workflow engine sends exceptions to named owners.
- The audit store captures every decision input.
- The audit store captures every recommendation output.
- The audit store captures every human override.
- The audit store captures every final purchase decision.
- The audit store captures later realized demand.
- The system uses role-based access.
- Planners can view and propose.
- Reliability engineers can annotate failure context.
- Procurement can verify supplier constraints.
- Finance can view economic assumptions.
- Only authorized approvers can release requisitions.
- Model administrators cannot release purchase orders solely because of model access.
- Purchase approvers cannot silently change model history.
- Data corrections require source-system evidence.
- Manual forecast overrides require a reason code.
- Manual policy overrides require an approver.
- Every override requires a timestamp.
- Every override requires a before-and-after value.
- Every override requires a free-text explanation.
- The system sends a warning when a forecast is stale.
- A forecast is stale after eight days without refresh.
- The system sends a warning when inventory snapshot age exceeds 24 hours.
- The system blocks automatic recommendation publication when inventory age exceeds 72 hours.
- The system blocks publication when a red data-quality failure occurs.
- The system falls back to a rule-based review when blocked.
- The fallback rule is not automatic ordering.
- The fallback rule is a manual review using current inventory, criticality, and maximum observed lead time.
- The fallback rule uses a conservative target of three units for P-417.
- The fallback rule is documented as temporary.
- The fallback rule expires after the data issue is resolved.
- The system tests API connectivity before creating a requisition draft.
- The system uses idempotency keys to avoid duplicate requisitions.
- The system reconciles requisition status daily.
- The system alerts if a recommendation remains unreviewed for two business days.
- The system alerts if supplier acknowledgement is missing after three business days.
- The system alerts if confirmed lead time changes by more than two weeks.
- The system alerts if a new reservation reduces inventory position below target.
- The system supports scenario simulation before approval.
- The system supports a “what changed?” comparison.
- The system supports export of a one-page audit packet.
- The system supports restoration from prior validated model versions.
- The system treats model deployment as a controlled operational change.

## 9. Evaluation & Validation

- Validation begins before deployment.
- Validation continues after deployment.
- The evaluation objective is not only point-forecast accuracy.
- The evaluation objective includes uncertainty calibration.
- The evaluation objective includes decision usefulness.
- The evaluation objective includes operational reliability.
- The evaluation objective includes governance compliance.
- A rolling-origin backtest is used.
- Each backtest cutoff uses only data available at that historical date.
- Each backtest predicts the following three-month lead-time demand.
- Each backtest compares predicted distributions with realized demand.
- Each backtest records whether realized demand fell below each predicted quantile.
- A well-calibrated 90% quantile should cover approximately 90% of realizations.
- A well-calibrated 95% quantile should cover approximately 95% of realizations.
- Small intermittent datasets create wide confidence around empirical coverage.
- Coverage should therefore be interpreted with uncertainty.
- The local-only mean baseline underestimates upper-tail demand in the illustrative backtest.
- The Croston-style baseline improves intermittency handling.
- The Croston-style baseline does not fully represent parameter uncertainty.
- The hierarchical negative-binomial model produces better 90% and 95% coverage in the illustrative comparison.
- The selected model is chosen for calibration and decision relevance.
- The selected model is not chosen merely because it has the lowest mean absolute error.
- Mean absolute error can be misleading for stockout decisions.
- Quantile loss is included as an evaluation metric.
- Weighted interval score is included as an evaluation metric.
- Log predictive density is included as a model-diagnostic metric.
- Stockout-event recall is included as an operational metric.
- Excess-inventory exposure is included as an operational metric.
- Recommendation stability is included as a usability metric.
- Override rate is included as a governance metric.
- Late-data rate is included as a data-process metric.
- Supplier-confirmation variance is included as a lead-time metric.

| Metric | Local Mean Baseline | Croston-Style Baseline | Hierarchical Probabilistic Model |
|---|---:|---:|---:|
| Illustrative 90% quantile coverage | 0.76 | 0.82 | 0.89 |
| Illustrative 95% quantile coverage | 0.81 | 0.86 | 0.94 |
| Illustrative weighted interval score | 1.28 | 1.07 | 0.91 |
| Illustrative mean absolute error | 0.53 | 0.49 | 0.47 |
| Illustrative stockout-event recall | 0.61 | 0.70 | 0.82 |
| Illustrative unnecessary-order rate | 0.29 | 0.25 | 0.22 |
| Interpretability rating | High | Medium | Medium-high |
| Hierarchical information use | No | No | Yes |
| Parameter uncertainty represented | No | Limited | Yes |

- The numerical validation table is illustrative.
- The numerical validation table is not evidence from a real production deployment.
- The model’s 95% coverage of 0.94 is close to the 0.95 target.
- The remaining gap is monitored.
- If 95% coverage falls below 0.90 over sufficient observations, recalibration is triggered.
- If 95% coverage exceeds 0.99 persistently, excess conservatism is investigated.
- If forecast recommendations oscillate without input changes, stability investigation is triggered.
- If manual overrides exceed 30%, usability investigation is triggered.
- If late transaction rates exceed 5%, data-process remediation is triggered.
- If supplier lead-time variance rises materially, the lead-time model is refreshed.
- Validation also includes posterior predictive checks.
- Posterior predictive checks compare simulated zero-month frequency with observed zero-month frequency.
- Posterior predictive checks compare simulated one-unit frequency with observed one-unit frequency.
- Posterior predictive checks compare simulated multi-unit frequency with observed multi-unit frequency.
- Posterior predictive checks compare site-level totals with observed totals.
- Posterior predictive checks compare family-level totals with observed totals.
- The model should reproduce intermittent behavior.
- The model should not smooth away rare multi-unit demand.
- The model should not create impossible negative demand.
- The model should not imply implausibly high monthly demand without evidence.
- Scenario validation tests whether system rules behave as intended.
- Scenario one tests a zero-inventory condition.
- Scenario two tests an inbound order that covers the target.
- Scenario three tests a new reservation.
- Scenario four tests a supplier delay.
- Scenario five tests an MOQ increase.
- Scenario six tests a failed demand-data extract.
- Scenario seven tests duplicate requisition prevention.
- Scenario eight tests manual override logging.
- Scenario nine tests stale forecast alerts.
- Scenario ten tests recovery from a prior model version.
- User acceptance testing confirms planners understand the dashboard.
- User acceptance testing confirms procurement sees MOQ logic.
- User acceptance testing confirms reliability sees criticality context.
- User acceptance testing confirms finance sees cost assumptions.
- The P-417 recommendation passes illustrative acceptance criteria.
- The recommendation is reproducible from the recorded inputs.
- The recommendation is explainable from the displayed distribution.
- The recommendation is policy-conforming.
- The recommendation is robust under ordinary sensitivity scenarios.
- The recommendation has a documented exception path.
- The recommendation should be revisited after actual receipt and demand outcomes occur.

## 10. Action Taken

- The fictional Reliability and Materials Review Board reviewed the P-417 recommendation.
- The board reviewed the current inventory position of one unit.
- The board reviewed the three-month demand distribution.
- The board reviewed the 95% cycle-service policy.
- The board reviewed the supplier MOQ of two units.
- The board reviewed the USD 16,000 shortage-cost proxy.
- The board reviewed the sensitivity analysis.
- The board confirmed that no planned overhaul demand was omitted.
- The board confirmed that no substitute was approved.
- The board confirmed that no inbound purchase order existed.
- The board approved the recommended order quantity of two units.
- Procurement created a purchase requisition for two P-417 units.
- The requisition used the standard qualified supplier.
- The requisition included a requested delivery date consistent with 12 weeks.
- The requisition included a request for supplier lead-time confirmation.
- The requisition included an instruction to notify NorthRiver of delays exceeding two weeks.
- The requisition recorded the forecast reference identifier.
- The requisition recorded the inventory snapshot identifier.
- The requisition recorded the policy version.
- The requisition required no emergency freight at release.
- The planner scheduled weekly exception reviews.
- The planner configured an alert for any new P-417 reservation.
- The planner configured an alert for any quality hold on the remaining unit.
- The planner configured an alert for confirmed supplier lead time beyond 16 weeks.
- The reliability engineer checked for emerging actuator contamination issues.
- The reliability engineer found no illustrative contamination signal at approval time.
- The procurement analyst confirmed MOQ and unit price.
- The finance representative accepted the shortage-cost proxy for decision use.
- The board documented that the purchase is a risk-control decision.
- The board documented that the forecast is uncertain.
- The board documented that a three-unit order would require a new escalation.
- The board documented that order cancellation would be reconsidered if an approved substitute appeared.
- The board documented that the outcome would be measured after the lead-time window.
- The system saved the approval packet.
- The system saved the forecast distribution.
- The system saved the sensitivity scenarios.
- The system saved the approver names as fictional role labels.
- The system saved the approval timestamp.
- The system scheduled a 30-day review.
- The system scheduled a supplier-confirmation review.
- The system scheduled a post-receipt outcome review.

## 11. Outcome & Results

- The following outcome is illustrative.
- The outcome is constructed to demonstrate measurement logic.
- The supplier acknowledged the order after two business days.
- The supplier confirmed a 12.6-week delivery estimate.
- The confirmed lead time was within the normal historical range.
- The order did not require expedited freight.
- The existing one-unit stock remained available during the first month.
- No P-417 demand occurred in the first month.
- One P-417 unit was issued during the second month.
- The issue supported a corrective actuator repair at the fictional Bay plant.
- The issue reduced on-hand quantity from one unit to zero units.
- The ordered quantity had not yet arrived when the issue occurred.
- The organization therefore would have had zero protection without the reorder.
- The supplier shipment arrived during the third month.
- The supplier delivered two units.
- Receipt inspection accepted both units.
- No quality hold was applied.
- No additional P-417 demand occurred before the order receipt.
- The realized three-month demand was one unit.
- The realized demand fell within the predicted distribution.
- The realized demand was at or below the one-unit inventory target.
- The realized demand did not test the two-unit or three-unit tail protection directly.
- The actual outcome therefore does not prove the forecast was perfect.
- The actual outcome does demonstrate that the reorder prevented a zero-stock exposure after the issue.
- The post-receipt on-hand quantity became two units.
- The post-receipt inventory position became two units.
- The weekly review continued because inventory position was below the original three-unit target.
- The next forecast refresh reduced mean demand slightly to 0.66 units.
- The refreshed 95% target remained three units.
- No immediate additional order was placed because an order had just been received.
- The system applied a review-period rule before any further purchase action.
- The supplier performance record logged actual lead time of 12.4 weeks.
- The supplier actual lead time was close to the historical mean.
- The forecast calibration record logged realized demand of one unit.
- The one-unit realization fell below the 95th-percentile target of three units.
- The event counted as a successful cycle-service outcome.
- No production train stopped for lack of P-417.
- No emergency transfer was required.
- No expedite premium was paid.
- No purchase-order cancellation was required.
- The order cash outlay was USD 960.
- The two received units retained inventory value.
- Four months of illustrative holding cost on average incremental stock was estimated at USD 137.60.
- The case did not claim that USD 16,000 of loss was definitively avoided.
- The case reported avoided risk exposure rather than realized avoided loss.
- The board classified the decision as operationally successful.
- The board also identified opportunities to improve failure-cause coding.
- The board requested a future model feature for contamination-event covariates.
- The board retained the 95% policy target.
- The board retained the weekly exception-review process.
- The board requested quarterly recalibration monitoring.
- The outcome should be interpreted as one observation, not conclusive model validation.

## 12. Failure Modes & What Went Wrong

- Intermittent-demand forecasting can fail when zeros are misinterpreted.
- A zero-demand month may mean no failure occurred.
- A zero-demand month may also mean demand was suppressed by stockout.
- A zero-demand month may also mean maintenance deferred a repair.
- A zero-demand month may also mean the transaction posted late.
- Treating every zero as true absence of demand can bias forecasts downward.
- The illustrative training window had no confirmed P-417 stockout censoring.
- Future deployments must test for stockout censoring explicitly.
- Hierarchical pooling can fail when related parts are not truly comparable.
- An overly broad family can import irrelevant demand.
- An overly narrow family can fail to provide useful stabilization.
- Engineering review is required for family definitions.
- Site pooling can fail when operating conditions differ sharply.
- A corrosive coastal site should not necessarily be pooled equally with a dry inland site.
- The hierarchy should allow site-specific effects.
- The hierarchy should be reviewed after asset redesigns.
- Lead-time estimates can fail when supplier conditions change abruptly.
- Historical purchase orders may not represent current supplier capacity.
- A supplier acquisition could invalidate historical lead-time assumptions.
- A material shortage could invalidate historical lead-time assumptions.
- A geopolitical disruption could invalidate historical lead-time assumptions.
- The system therefore monitors confirmed lead-time changes.
- Inventory position can fail when records are inaccurate.
- A missing reservation can make stock appear available when it is committed.
- A quality-hold error can make unusable stock appear usable.
- A duplicate inbound record can make undelivered stock appear available.
- A delayed receipt posting can make delivered stock appear absent.
- Physical verification remains necessary for critical decisions.
- MOQ logic can fail when supplier terms change.
- A changed MOQ can invalidate a previously approved quantity.
- An order multiple can create excess inventory.
- A packaging-unit rule can create excess inventory.
- Procurement must verify supplier constraints at release.
- Cost models can fail when downtime costs are political rather than operational.
- An inflated shortage cost can encourage excess stock.
- An understated shortage cost can encourage unsafe deferral.
- The case therefore treats shortage cost as a sensitivity variable.
- Service targets can fail when applied mechanically.
- A 95% target may be too low for safety-critical spares.
- A 95% target may be too high for noncritical expendables.
- Criticality policy must be reviewed separately from model fitting.
- Forecast systems can fail through automation bias.
- Users may accept a recommendation without checking assumptions.
- Users may reject a recommendation because they distrust unfamiliar statistics.
- The dashboard must expose the reasoning clearly.
- Governance can fail when overrides are undocumented.
- An undocumented override prevents later learning.
- An undocumented override can hide bias.
- The system requires reason codes and approvals.
- Reproducibility can fail when software versions drift.
- A changed library version can alter posterior computation.
- A changed random seed can alter simulated quantiles slightly.
- The system records environment and model version.
- Data pipelines can fail through schema changes.
- A renamed ERP field can silently remove reservations.
- A new transaction type can silently create double counting.
- Contract tests should detect schema changes.
- Integration can fail through duplicate requisition creation.
- A retry after API timeout can create two purchase drafts.
- Idempotency keys should prevent duplicate submissions.
- Integration can fail through stale inventory snapshots.
- An old snapshot can recommend an unnecessary order.
- The system should block publication after the stale threshold.
- Model monitoring can fail with too few observations.
- A rare spare may generate insufficient outcomes for rapid calibration assessment.
- Family-level monitoring can supplement item-level monitoring.
- Extreme-event scenarios can fail because history contains no analog.
- A contamination event can create correlated failures across multiple assets.
- A simple independent-demand model may understate this risk.
- Stress scenarios and expert review are required.
- The illustrative case had one limitation after action.
- The actual realization of one unit did not distinguish between several plausible models.
- A single realized period cannot validate tail calibration.
- The correct response is continued monitoring rather than overclaiming success.
- A second illustrative issue occurred during review.
- The first dashboard version displayed only the forecast mean.
- Users initially interpreted 0.72 units as a recommendation not to order.
- The dashboard design was corrected to display quantiles and stockout probabilities first.
- The corrected dashboard highlighted that one unit gave only 82.60% cycle service.
- This design correction reduced mean-based misinterpretation.
- A third illustrative issue involved lead-time data.
- One purchase order had an apparent 31-week lead time.
- Investigation showed the release date was entered after the receipt date correction.
- The record was flagged as a data error.
- The record was excluded only after source-system audit evidence was attached.
- The correction was logged in the data-quality ledger.
- A fourth illustrative issue involved a manual override request.
- A planner proposed ordering zero units due to low recent demand.
- The request cited only the last six months, which contained no demand.
- The board rejected the override because it ignored hierarchy and lead-time risk.
- The board documented the rationale.
- A fifth illustrative issue involved a supplier delay alert.
- The supplier later indicated a possible 17-week delay.
- The system recomputed the stress scenario.
- The recomputed 95% target became four units.
- The board considered a supplemental one-unit order.
- The board did not place the supplemental order because the supplier resolved the delay before release.
- This demonstrates that recommendations must be revisited when conditions change.
- Failure recovery requires more than rerunning the model.
- Recovery requires identifying whether failure arose from data, model, policy, integration, or governance.
- Recovery requires preserving evidence.
- Recovery requires correcting root cause.
- Recovery requires checking downstream decisions.
- Recovery requires communicating changed risk to decision owners.

## 13. Transferable Lesson

- Intermittent spare-parts decisions should not be based only on average demand.
- Average demand can be low while stockout risk remains operationally unacceptable.
- Reorder decisions should be based on lead-time demand.
- Lead-time demand should include replenishment uncertainty where material.
- Probabilistic forecasts are more useful than point forecasts when service targets matter.
- Quantiles translate uncertainty into inventory protection levels.
- Inventory position is more decision-relevant than on-hand quantity alone.
- MOQ rules can materially alter the executable order quantity.
- Hierarchical models can improve sparse-item forecasts by borrowing related evidence.
- Hierarchical pooling requires engineering judgment about what is genuinely comparable.
- Forecast quality includes calibration, not only average error.
- A model can have good mean error and poor tail protection.
- Service policy is a business choice, not a statistical inevitability.
- Cost assumptions should be visible and sensitivity-tested.
- A decision can be justified by policy even when a precise cost estimate is uncertain.
- A forecast should never hide data-quality limitations.
- Leakage control is essential in time-based forecasting.
- A model that uses future information will appear better than it is.
- Reproducibility is necessary for audit and learning.
- Human approval remains important for critical spares.
- Automation should draft, explain, monitor, and log.
- Automation should not silently purchase critical stock without governance.
- Exceptions should be designed before deployment.
- Fallback methods should be explicit rather than improvised.
- Outcomes should be measured without claiming more certainty than evidence supports.
- One successful order does not validate a forecasting system.
- Repeated backtesting and operational monitoring create the evidence base.
- The transferable pattern is forecast distribution, policy target, inventory position, supplier constraints, and governed action.
- This pattern applies to medical spares, utility spares, aviation rotable components, and data-center critical parts.
- This pattern does not remove the need for domain expertise.
- This pattern makes domain expertise more explicit and auditable.

## 14. Connections & Cross-References

### Interview Question 1: Why not use the historical average of 0.222 units per month?

- The local average is useful descriptive information but insufficient for reorder policy.
- The decision concerns demand during lead time, not a single average month.
- The three-month local-average expectation is approximately 0.667 units.
- A mean of 0.667 does not describe the probability of needing two or three units.
- With critical spares, upper-tail risk matters more than the mean alone.
- The hierarchical predictive distribution estimates that one unit covers only 82.60% of baseline lead-time outcomes.
- The 95% policy therefore requires a three-unit inventory position.
- See Section 6 for the distribution and quantile calculations.
- See Section 13 for the general lesson about means and service levels.

### Interview Question 2: What does hierarchical forecasting contribute in this case?

- The P-417 local history has only eight units of demand across 36 months.
- Sparse local history produces unstable estimates.
- Hierarchical forecasting borrows evidence from related actuator cartridges and similar sites.
- The method still allows P-417 to differ from its family and sites.
- This is called partial pooling.
- Partial pooling avoids treating P-417 as either fully isolated or fully identical to related parts.
- The result is a more stable predictive distribution with explicit uncertainty.
- See Section 5c for the method summary.
- See Section 6 for the practical use of the resulting distribution.

### Interview Question 3: Why is a negative-binomial distribution used instead of a Poisson distribution?

- A Poisson model assumes the variance equals the mean.
- Intermittent spare demand often has variance greater than the mean.
- Related operational conditions and unobserved heterogeneity create overdispersion.
- The negative-binomial distribution allows variance to exceed the mean.
- The negative-binomial model also produces a heavier upper tail than a simple Poisson model with the same mean.
- That heavier tail is relevant for stockout protection.
- In this case, the negative-binomial size parameter is 1.80.
- The parameter is illustrative and estimated through the hierarchical model.
- See Section 6 for the recurrence calculations.
- See Section 9 for calibration-focused evaluation.

### Interview Question 4: How is the reorder quantity calculated?

- First calculate inventory position.
- Inventory position equals on hand plus inbound minus reservations minus backorders.
- Current inventory position equals one unit.
- Next select a policy service target.
- The selected policy target equals 95% cycle-service level.
- Next find the smallest demand quantity whose cumulative probability is at least 95%.
- The baseline 95% quantile equals three units.
- Subtract inventory position from the target stock level.
- Three minus one equals two units.
- Apply MOQ and order-multiple rules.
- The MOQ is two units, so the final recommended order remains two units.
- See Section 6 for the checked calculation.
- See Section 7 for the resulting finding.

### Interview Question 5: Why does a 95% target produce a 97.88% achieved service level?

- Inventory is indivisible in whole units.
- The 95th-percentile target is the smallest integer quantity with cumulative probability at least 95%.
- The cumulative probability at two units is 93.82%.
- The cumulative probability at three units is 97.88%.
- There is no fractional inventory position between two and three units.
- The operational policy therefore accepts the smallest quantity that meets or exceeds the target.
- This is a normal feature of discrete inventory decisions.
- See the quantile table in Section 6.
- See Section 4 for the quantile convention.

### Interview Question 6: What would change if a new reservation consumed the one unit on hand?

- Inventory position would fall from one unit to zero units.
- The baseline 95% target would remain three units unless demand assumptions changed.
- The unconstrained order quantity would become three minus zero.
- The new recommended order quantity would become three units.
- The supplier MOQ would not change that quantity because three exceeds the MOQ.
- The system should issue an exception alert because inventory position changed materially.
- The board should review whether the reservation is firm and whether transfer options exist.
- See Section 8 for the alerting design.
- See Section 12 for inventory-record failure risks.

### Interview Question 7: What would change if the supplier confirmed a five-month lead time?

- The decision horizon would extend from three months to five months.
- Baseline mean demand would scale from 0.72 to approximately 1.20 units under a proportional scenario.
- The illustrative 95% protection target would increase from three to four units.
- With inventory position one, the implied order would increase from two to three units.
- The change should be evaluated using an updated lead-time distribution rather than a simple deterministic multiplier when sufficient data exist.
- The system should alert when confirmed lead time exceeds the configured threshold.
- The board should consider whether expedite, transfer, or substitute actions can reduce exposure.
- See Section 6 sensitivity analysis.
- See Section 8 system alerts.
- See Section 12 for supplier-change failure modes.

### Interview Question 8: How do you prevent data leakage in this forecast?

- Use a fixed forecast cutoff timestamp.
- Include only demand transactions physically known by that timestamp.
- Use issue-posting date rather than later work-order-close date.
- Exclude future supplier confirmations.
- Exclude future maintenance findings.
- Version late corrections and retain their physical-event date.
- Run rolling-origin backtests that preserve time order.
- Never train on demand that occurred after the simulated historical decision date.
- See Section 5d for leakage controls.
- See Section 9 for rolling-origin validation.

### Interview Question 9: Why not automatically release the purchase order?

- Critical-spare decisions involve operational context that may not be fully captured in data.
- A planned outage, engineering change, substitute approval, or supplier conversation can alter the decision.
- Automatic release can create unnecessary purchases if data are stale or incorrect.
- The system should automate calculation, explanation, monitoring, and draft creation.
- Human approvers should verify material assumptions before committing funds.
- Approval should be fast but governed.
- The case uses a human-approved requisition workflow.
- See Section 8 for role separation.
- See Section 10 for the action process.

### Interview Question 10: How would you evaluate whether the model is good?

- Evaluate calibration of predictive quantiles.
- Evaluate whether 90% predictions cover realized outcomes about 90% of the time.
- Evaluate whether 95% predictions cover realized outcomes about 95% of the time.
- Evaluate weighted interval score and log predictive density.
- Evaluate stockout-event recall and unnecessary-order rate.
- Compare against simple local and intermittent-demand baselines.
- Use rolling-origin backtesting rather than random train-test splits.
- Monitor real-world results after deployment.
- Do not judge the model solely by mean absolute error.
- See Section 9 for the evaluation framework.

### Interview Question 11: What failure mode is most dangerous for this use case?

- The most dangerous combined failure is an understated demand tail combined with an overstated inventory position.
- Understated tail risk can arise from overly aggressive pooling, censored stockout demand, or ignored correlated failures.
- Overstated inventory position can arise from unrecorded reservations, quality holds, or phantom inbound orders.
- Together, these failures can falsely indicate that no reorder is needed.
- Controls include physical verification, data-quality checks, hierarchy governance, and exception review.
- The system should fail safely by suppressing automated recommendations when critical inputs are invalid.
- See Section 5d for data-quality controls.
- See Section 8 for fallback design.
- See Section 12 for detailed failure recovery.

### Interview Question 12: What is the most transferable principle from the case?

- Translate uncertainty into a decision-relevant distribution before applying policy.
- For spare parts, the relevant distribution is usually lead-time demand.
- Select a service target based on criticality and business risk.
- Compare the target quantile with inventory position.
- Apply real supplier constraints such as MOQ and order multiples.
- Document assumptions, provenance, and overrides.
- Monitor outcomes and recalibrate.
- This sequence is more robust than ordering from averages or intuition alone.
- See Section 13 for the full transferable lesson.
- See Section 15 for the concise operating reference.

## 15. Quick Reference Card

| Item | Illustrative Value | Operational Meaning |
|---|---:|---|
| Part | P-417 | Hydraulic actuator cartridge |
| Company status | Fictional | No real company represented |
| Decision | Reorder now | Policy-conforming recommendation |
| Recommended quantity | 2 units | Meets 95% target from current position |
| Current on hand | 1 unit | Physically verified usable stock |
| Confirmed inbound | 0 units | No open purchase order |
| Reservations | 0 units | No committed demand at decision time |
| Inventory position | 1 unit | On hand plus inbound minus commitments |
| Nominal lead time | 12 weeks | Three-month baseline horizon |
| Forecast mean demand | 0.72 units | Expected three-month demand |
| 50% quantile | 0 units | Median lead-time demand |
| 80% quantile | 1 unit | Minimum stock for about 82.60% CSL |
| 90% quantile | 2 units | Minimum stock for about 93.82% CSL |
| 95% quantile | 3 units | Policy target stock level |
| 98% quantile | 4 units | Higher-protection scenario |
| Baseline CSL at 1 unit | 82.60% | Current stock protection |
| Baseline CSL at 2 units | 93.82% | Below 95% policy |
| Baseline CSL at 3 units | 97.88% | Meets 95% policy |
| MOQ | 2 units | Supplier minimum purchase |
| Unit price | USD 480 | Illustrative acquisition cost |
| Holding cost | USD 17.20/unit/month | Illustrative carry plus handling |
| Shortage-cost proxy | USD 16,000/unit | Illustrative downtime exposure |
| Main method | Hierarchical negative binomial | Sparse intermittent-demand forecast |
| Main control | Human approval | Prevents blind automation |
| Main data risk | Stale or censored demand | Can understate risk |
| Main inventory risk | Incorrect reservation or hold | Can overstate availability |
| Main supplier risk | Lead-time extension | Can raise target stock |
| Immediate trigger to reassess | New reservation | Reduces inventory position |
| Immediate trigger to reassess | Delay beyond 16 weeks | Extends exposure horizon |
| Immediate trigger to reassess | Quality hold | Removes usable stock |
| Immediate trigger to reassess | Approved substitute | May reduce need to buy |
| Fallback if data are red | Manual critical-spare review | Do not auto-order |
| Required audit artifacts | Inputs, model version, policy, approval | Reproducible decision |

- **Core formula:** Inventory position equals on hand plus inbound minus reservations minus backorders.
- **Core formula:** Reorder quantity equals target stock minus inventory position.
- **Core formula:** Target stock equals selected predictive quantile of lead-time demand.
- **Core policy:** Use the smallest whole-unit target meeting the cycle-service threshold.
- **Core reminder:** A mean forecast is not a reorder quantity.
- **Core reminder:** Intermittent demand requires uncertainty-aware planning.
- **Core reminder:** Forecast distribution, policy target, and supplier constraints must all be visible.
- **Core reminder:** Never use future information in historical validation.
- **Core reminder:** Keep manual overrides, reasons, and approvals in the audit trail.
- **Core reminder:** Recalculate when inventory position or lead time changes.
