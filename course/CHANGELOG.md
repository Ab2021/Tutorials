# Changelog

All substantive changes to the curriculum are recorded here.

## 2026.15.0 — 3 November 2026

### Added — Module 10 written in full
- Module 10 (`02_methods/10_inventory_optimization_multi_echelon/`) is now written in full: ten
  topic files, 13,153 lines, each with all 17 required sections and every relative link resolving.
- `01_cost-service-frontier.md` — the trade-off curve, the marginal cost of the next point of
  service, and why service targets are budget decisions rather than inventory settings.
- `02_eoq-limits.md` — the EOQ derivation and the eight assumptions it rests on, priced one at a
  time against the cost of the assumption failing.
- `03_newsvendor.md` — the critical fractile, the underage-versus-overage ratio, and the
  pull-to-centre behaviour that makes practitioners order away from the computed quantity.
- `04_safety-stock.md` — the protection period as the fundamental quantity, the omitted
  lead-time-variance term, and the honest estimation of sigma under censored history.
- `05_service-metrics.md` — cycle service level, unit and line fill rate, ready rate and OTIF as
  measures that are related but not interchangeable, and the normal loss function that converts
  between them.
- `06_policy-families.md` — (s,Q), (s,S), (R,S) and (R,s,S) as two decisions on two dimensions,
  the min-max/order-up-to confusion, and how to identify the family a system actually runs from
  its configuration rather than its label.
- `07_lead-time-demand.md` — lead-time demand as a random sum, the variance identity, the limit of
  the normal approximation at low mean demand, dispersion testing, and the censoring bias.
- `08_multi-echelon.md` — installation versus echelon stock, what Clark and Scarf established,
  MEIO as practised, DRP deployment, shortage allocation, and buffer placement across a network.
- `09_pooling-postponement.md` — the square-root law and the correlation condition that makes it
  fail, component commonality, and the three forms of postponement with the conditions each needs.
- `10_segmentation-shelf-life.md` — the ABC×XYZ matrix, demand-pattern classification, DDMRP zones
  assessed against the evidence actually available, and perishability as a third policy dimension.

### Changed
- Module 10 status table reports 10 `complete` and the module total of 13,153 lines.
- `modules/02_methods/_index.md` reports four modules complete in Part II; `VALIDATION_STATUS.md`
  now records 102 of 476 topic units and 132,643 lines.
- The module-close link sweep converted eight stale `(planned)` markers across seven files to live
  links. Two were plain-text file lists inside `text` fences; those were rewritten as bulleted link
  lists, and the "not yet written" heading in `03_newsvendor.md` was replaced with a section that
  describes what each sibling now covers.

### Notes
- **Module 06 remains deferred at the user's request.** A verification pass confirmed that no file
  in Module 10 references `06_machine_learning_foundations`, in any form.
- Two writer sessions were interrupted by the API session usage limit. `05_service-metrics.md` was
  cut during a compression pass at 1,338 lines and `06_policy-families.md` was cut after section
  14. Both were verified on disk rather than assumed: section 05 was structurally complete and was
  accepted at 1,338 lines, inside the module's stated 1,200–1,800 target; sections 15 to 17 of
  topic 06 were written during the close and checked against the file's own body.
- `04_safety-stock.md` reports a single `!` character. It is the `!=` operator inside a formula,
  not prose punctuation.
- Source discipline in this module: no named inventor is attributed to the safety-stock formula;
  the ERP omission rate for the lead-time-variance term is stated as field inference rather than a
  measured rate; DDMRP performance figures are graded D and no adoption percentage is reproduced;
  and the service-compounding rule is presented as arithmetic valid only for independent local
  buffering rather than as a general result.

## 2026.14.0 — 3 November 2026

### Added — Module 09 written in full
- Module 09 (`02_methods/09_demand_sensing_promotions_npi/`) is now written in full: ten topic
  files, 12,982 lines, each with all 17 required sections and every relative link resolving.
- `01_demand-sensing-signals.md` — the signal-and-lead-time layer above forecasting: which signals
  arrive before the decision gate, which are lagging, and how upstream order data is distorted.
- `02_sales-decomposition.md` — baseline, seasonality, promotion, distribution and price defined
  before estimation, the identification problem, and an auditable component table.
- `03_promotion-mechanics.md` — feature, display, temporary price reduction, coupon and trade
  deal, the data each generates, and the funded-versus-shelf-price gap.
- `04_baseline-incrementality.md` — the counterfactual ladder from naive before-after through
  control-store difference-in-differences to randomised holdout, and the assumption each buys.
- `05_cannibalization-halo.md` — pull-forward, cannibalization and halo as the three adjustments
  that turn gross lift into incremental margin.
- `06_price-display-effects.md` — reference price, threshold and saturation, and the confounding
  structure that makes observational elasticities unreliable.
- `07_causal-ml-uplift.md` — conditional average treatment effects, the meta-learner family, honest
  splitting, and the Qini evaluation problem.
- `08_npi-forecasting.md` — analogy, attribute, diffusion and distribution-led methods for a
  product with no sales history, and why no diffusion parameter can be estimated from a launch.
- `09_distribution-assortment.md` — numeric and weighted distribution, on-shelf availability,
  censored demand, and the sales decomposition into distribution and rate.
- `10_post-launch-learning.md` — revising a forecast with early observations, promotion and launch
  ROI against adjusted rather than gross lift, and portfolio learning.

### Changed
- Module 09 status table reports 10 `complete` and the module total.
- `modules/02_methods/_index.md` reports three modules complete in Part II; `VALIDATION_STATUS.md`
  now records 92 of 476 topic units and 119,490 lines.
- The module-close link sweep converted 18 stale `(planned)` markers across six files to live
  links. Two of them were plain-text file lists inside `text` fences, where a markdown link cannot
  render; those were rewritten as bulleted link lists so the siblings became reachable.

### Notes
- **Module 06 remains deferred at the user's request.** A verification pass confirmed that no
  Module 09 file contains a link to, or description of, Module 06's contents. Because uplift
  modelling would normally draw its vocabulary from that module, the topic 07 brief required the
  necessary learning concepts to be stated in the file itself.
- Two writers reported exceeding the 5–8 search budget this module, at nine searches each. No
  fabricated citation entered either file, and both disclosed the overrun in their hand-backs.
- The topic 06 writer declined to confirm the widely quoted "average grocery own-price elasticity
  near −2.6" from any source it reached, confined it to a note as an order of magnitude, and
  asserted only the dispersion finding. It named its regulatory instruments but stated that
  consolidated texts were not retrieved.
- The topic 04 writer kept the "72 per cent of trade promotions do not break even" figure only as a
  grade-D prior about a base rate, never as a calculation input, because its provenance could not
  be traced to retrievable originals.
- The topic 05 writer deleted four candidate citations whose records the searches did not
  confirm, and omitted DOIs wherever none was returned rather than construct plausible ones.
- The topic 08 writer cited intention-to-trial conversion studies for the direction of bias only,
  because the adjustment coefficients were not returned, and states in three places that no
  diffusion parameter is estimated from the launch item's own data.
- The topic 09 writer separated a cross-sectional convexity result from a within-brand temporal
  elasticity, so the file does not claim distribution responds uniformly, and graded the
  industry-funded out-of-stock studies C with an explanation of the grading.
- The topic 10 writer deleted its practice-versus-evidence table when its one quantitative
  assertion, that ROI on full-margin lift overstates by "a factor of two to four", could not be
  sourced. The table went rather than the claim being repeated unqualified.

## 2026.13.0 — 3 November 2026

### Added — Module 08 written in full
- Module 08 (`02_methods/08_hierarchical_intermittent_probabilistic_forecasting/`) is now written
  in full: ten topic files, 12,992 lines, each with all 17 required sections and every relative
  link resolving.
- `01_hierarchy-structures.md` — product, location, customer and temporal aggregation, the
  summing matrix, and the paths between levels.
- `02_aggregation-consistency.md` — bottom-up, top-down, middle-out, and optimal reconciliation,
  with the error consequences of each.
- `03_mint-reconciliation.md` — MinT in detail: the generalised least-squares form, covariance
  estimators, shrinkage, and constrained reconciliation.
- `04_temporal-cross-temporal.md` — reconciliation across time buckets as well as entities, and
  where the published gains concentrate.
- `05_intermittent-classification.md` — the Syntetos–Boylan grid, routing series to method
  families, misclassification cost, and why static classification fails over a product lifecycle.
- `06_compound-poisson-hurdle.md` — occurrence and size as separate processes, lead-time
  convolution, and why a point estimator cannot answer a fill-rate question.
- `07_probabilistic-forecasts.md` — the predictive distribution as a commitment, the functional
  each downstream decision reads, and calibration as a tested property.
- `08_conformal-prediction.md` — split and adaptive conformal methods, the exchangeability seam,
  and the documented failure of naive conformal intervals on dependent demand.
- `09_scoring-rules.md` — propriety, pinball loss, CRPS, log score, energy and variogram scores,
  and calibration diagnostics as distinct from scores.
- `10_quantiles-decisions.md` — cycle service level, fill rate, ready rate, the critical fractile,
  risk appetite, and the governance of a service target.

### Changed
- Module 08 status table reports 10 `complete` and the module total.
- `modules/02_methods/_index.md` reports two modules complete in Part II; `VALIDATION_STATUS.md`
  now records 82 of 476 topic units and 106,508 lines.
- The module-close link sweep found that `02_aggregation-consistency.md` named three sibling
  topics by file names that do not exist (`03_minimum-trace-reconciliation.md`,
  `04_intermittent-demand.md`, `05_probabilistic-forecasting.md`). All three were corrected to the
  real file names and converted to live links.

### Notes
- **Module 06 remains deferred at the user's request.** A verification pass confirmed that no
  Module 08 file contains a link to, or description of, Module 06's contents.
- Two writers narrowed claims rather than assert them. The topic 05 writer declined to state the
  exact diagonal-approximation inequality from the Syntetos–Boylan–Croston exchange because the
  sign and constants could not be verified from reachable material, describing it only
  qualitatively. The topic 06 writer dropped the claim that hurdle models systematically mis-fit
  intermittent demand, because at the fitted bucket the hurdle and compound Poisson forms are
  observationally equivalent; it states the equivalence and confines the divergence to
  extrapolation, identification and fractional horizons.
- The topic 07 writer removed the GefCom 2014 "99 quantile levels" and M4 "100,000 series" details
  because the session's searches did not return them, rather than repeating widely circulated
  figures from memory.
- The topic 08 writer stated that a production-scheduling demand forecast is not a high-risk AI
  system under Regulation (EU) 2024/1689 and that "SR 11-7 compliant" has no meaning outside
  banking, claiming only that the documentation discipline transfers.
- The topic 09 writer disclosed that it ran nine searches against a budget of eight. No fabricated
  citation entered the file.
- The topic 04 writer inverted an instruction in its own brief: the brief said temporal
  reconciliation helps the shortest horizon, whereas the literature reports gains largest at the
  coarsest level, with the UK A&E case the documented short-horizon exception. The file follows the
  literature. The brief was wrong.

## 2026.12.0 — 3 November 2026

### Added — Module 07 written in full
- Module 07 (`02_methods/07_time_series_forecasting/`) is now written in full: ten topic files,
  12,942 lines, each with all 17 required sections and every relative link resolving. This is the
  first module written in Part II.
- `01_forecastability-benchmarks.md` — naive, seasonal-naive, drift and mean benchmarks; forecast-
  ability; the M-competition lineage and what it does and does not establish.
- `02_exponential-smoothing.md` — the smoothing family as one idea; damped trend; additive versus
  multiplicative seasonality; the state-space formulation and the ETS taxonomy.
- `03_arima-sarimax.md` — stationarity and differencing, order selection, seasonal structure,
  exogenous regressors, the leakage trap, and residual diagnostics.
- `04_regression-lags-fourier.md` — harmonic seasonality, moving holidays, distributed lags, and a
  catalogue of future-leakage errors with the red flag for each.
- `05_hierarchy-aggregation.md` — where to forecast, top-down versus bottom-up versus middle-out,
  temporary aggregation, and the reconciliation problem previewed without solving it.
- `06_intermittent-demand.md` — Croston, SBA and TSB, the bias-correction history, and classification
  used as a screen rather than a verdict.
- `07_forecast-combination.md` — the combination puzzle, the error-correlation term, and why equal
  weights beat estimated weights on short samples.
- `08_rolling-origin-backtesting.md` — expanding and sliding origins, horizon effects, and the
  selection-versus-evaluation leakage.
- `09_error-metrics-cost.md` — the metric family, asymmetric cost, and the newsvendor critical
  fractile that turns an error distribution into a decision quantile.
- `10_overrides-fva.md` — forecast value added, the three override reasons, and how to measure a
  planner's adjustment against the right baseline.

### Changed
- Module 07 status table reports 10 `complete` and the module total.
- `modules/02_methods/_index.md` reports one module complete in Part II; `VALIDATION_STATUS.md` now
  records 72 of 476 topic units and 93,516 lines.

### Notes
- **Module 06 remains deferred at the user's request** (2 November 2026). Module 07's prerequisite
  list names Module 06, so the module index and every topic file carry an explicit note instead of
  pointing at unwritten material. A verification pass confirmed that no Module 07 file contains a
  link to, or description of, Module 06's contents.
- One writer corrected an error carried in its own brief: the brief described the Syntetos–Boylan
  bias correction as removing "a bias of exactly p/2", which no source the writer could reach
  supported. The file states the bias as being of order 1/(2p) and notes that published
  approximations differ in their constants. The brief's version was not printed.
- Another writer could not confirm the current edition year of ISO 3534-1 and omitted the year
  rather than guessing it.

## 2026.11.0 — 2 November 2026

### Added — Module 05 written in full
- Module 05 (`01_foundations/05_experimentation_pilots_evidence/`) is now written in full: ten
  topic files, 12,955 lines, each with all 17 required sections and every relative link resolving.
- `01_claim-hypothesis-design.md` — claim → question → estimand → design → metric → decision rule,
  with the threshold arithmetic that turns an effect into a decision.
- `02_randomisation-units.md` — assignment unit versus analysis unit versus decision unit, and the
  contamination that follows a mismatched choice.
- `03_ab-cluster-designs.md` — two-arm, multi-arm, factorial, and cluster designs, the analysis each
  demands, and the operational constraints that rule a design out.
- `04_geo-experiments.md` — matched-market and donor-pool designs, placebo tests, and why geo power
  is usually terrible.
- `05_switchback-pilots.md` — time-based alternation, carryover and washout, the pilot effect, and a
  schedule construction the reader can redraw.
- `06_power-mde.md` — the four-quantity arithmetic, design effects, effective sample size, and how to
  say honestly that a pilot was not powered for the question asked of it.
- `07_interference-spillover.md` — SUTVA and its two failures, concrete supply-chain spillover
  channels, the sign of the bias each induces, and partial-interference designs.
- `08_sequential-testing.md` — alpha spending, always-valid inference, and the multiple-looks
  arithmetic behind early stopping.
- `09_mmm-incrementality.md` — what a marketing-mix model can and cannot identify, and the
  experiment-calibrated architecture that is the honest version.
- `10_vendor-evidence-appraisal.md` — the six appraisal questions, survivorship and regression to the
  mean in published results, and a scoring sheet usable in a procurement meeting.

### Changed
- Module 05 status table reports 10 `complete` and the module total.
- `modules/01_foundations/_index.md` reports six modules complete; `VALIDATION_STATUS.md` now records
  62 of 476 topic units and 80,574 lines.
- Forward references that were written as "(planned)" because their targets did not yet exist have
  been converted to live links now that every Module 05 topic is on disk.
- One brief error was corrected in production rather than propagated: the Part I index sits at
  `../_index.md` from a module folder, not `../../_index.md`, which resolves to a nonexistent
  `modules/_index.md`. Two writers caught it independently and verified the resolving path.

### Notes
- **Module 06 remains deferred at the user's request** (2 November 2026). Module 07 is the next
  module in production; its index lists Module 06 as a prerequisite, which will need a stated
  substitute when that module opens.

## 2026.10.0 — 2 November 2026

### Added — Module 04 written in full
- Module 04 (`01_foundations/04_statistics_causal_inference/`) is now written in full: ten topic
  files, 12,975 lines, each with all 17 required sections and every relative link resolving.
- `01_estimands-question-types.md` — descriptive, predictive, and causal questions; the estimand as
  population, unit, outcome, time, intervention, and comparison; the target-trial framing.
- `02_probability-bootstrap.md` — intervals that survive skewed and dependent data; the bootstrap's
  three conditions; block and cluster resampling; prediction versus confidence intervals.
- `03_hypothesis-tests.md` — p-values, effect sizes, multiplicity; family-wise error versus false
  discovery rate; Benjamini–Hochberg by hand; equivalence testing.
- `04_regression-time-series.md` — harmonic seasonality, moving holidays, distributed lags, trend
  specification, and diagnostics before interpretation.
- `05_autocorrelation-robust-errors.md` — serial correlation, heteroskedasticity, clustering, HAC
  errors, and the design-effect arithmetic behind the effective sample size.
- `06_dags.md` — backdoor adjustment, collider bias, mediation, over-control, and two worked
  adjustment-set derivations.
- `07_experiments-quasi.md` — cluster, stepped-wedge, and switchback designs; the quasi-experimental
  family and the assumption each buys; a constraint-to-design decision table.
- `08_difference-in-differences.md` — the 2x2 arithmetic, event studies, staggered adoption and the
  Goodman-Bacon weighting problem, with the corrected estimators named.
- `09_elasticity-endogeneity.md` — simultaneity, the identification problem, instruments and the
  local average treatment effect, and a worked case where the naive estimate has the wrong sign.
- `10_sensitivity-omitted-bias.md` — the omitted-variable bias formula, robustness values, E-values,
  coefficient stability, and the module's closing argument about assumption chains.

### Changed
- Module 04 status table reports 10 `complete` and the module total.
- `modules/01_foundations/_index.md` reports five modules complete; `VALIDATION_STATUS.md` now
  records 52 of 476 topic units and 67,619 lines.

### Notes
- One agent corrected three citation attributions that were wrong in its brief rather than
  propagating them: the Table 2 fallacy paper (Westreich and Greenland, not Hernán and Robins), the
  DAGitty methods paper's venue, and the reference for the definition of a confounder
  (VanderWeele and Shpitser). Each correction is recorded in the file's own source note.
- **Module 06 is deferred at the user's request** (2 November 2026) and will not be written in this
  pass. Its index and survey file stand; its topic rows remain `planned` so the gap is visible
  rather than silently absent.

## 2026.9.0 — 2 November 2026

### Added — Module 03 written in full
- Module 03 (`01_foundations/03_data_engineering_interoperability/`) is now written in full:
  eleven topic files, 14,231 lines, each with all 17 required sections and every relative link
  resolving.
- `01_layered-architecture.md` — raw, staging, conformed, and mart layers; contracts and owners at
  each boundary; the medallion pattern and its critiques.
- `02_batch-microbatch-streaming.md` — the latency ladder, freshness as a budget, watermarks and
  late data, exactly-once versus effectively-once, and which decisions justify which latency.
- `03_ingestion-patterns.md` — full, incremental, CDC, and file-drop ingestion; high-water marks,
  tombstones, schema drift, idempotent loads, and replay.
- `04_sql-supply-chain-patterns.md` — as-of joins, bitemporal questions, snapshot grain, allocation
  by window function, SCD traps, and gap-and-island stockout spells.
- `05_python-structure-tests.md` — layout, types, dependency pinning, and a test strategy whose
  unit tests encode supply chain invariants.
- `06_dbt-models-tests.md` — the model DAG, materialisations, incremental models, and tests that
  encode invariants rather than syntax.
- `07_orchestration.md` — dependencies, backfills, retries, DST hazards, and the planning cutoff as
  a hard deadline.
- `08_edi-x12-edifact.md` — the transaction set and its EDIFACT analogues, envelope nesting, the
  acknowledgement chain, and partner implementation-guide variance.
- `09_gs1-epcis.md` — the identifier stack, the four event types plus association events,
  aggregation as the case-to-pallet link, and the regulatory hooks.
- `10_apis-contracts-mdm.md` — APIs as interfaces, contract clauses, versioning and deprecation,
  and entity resolution with its false-merge and false-split costs.
- `11_quality-quarantine-lineage.md` — test severity, quarantine, replay testing, column-level
  lineage, and the module's closing reproducibility argument.

### Changed
- Module 03 status table reports 11 `complete` and the module total.
- `modules/01_foundations/_index.md` reports four modules complete; `VALIDATION_STATUS.md` now
  records 42 of 476 topic units and 54,644 lines.

### Notes
- Cross-topic forward links were written as plain text marked "planned" while the target did not
  yet exist, so no file ever shipped with a dead link; all eleven files now resolve every relative
  link.
- Two claim clusters were reduced to graded vendor entries rather than repeated as fact: the EDI
  market figures (78% of B2B transactions, 20 billion transactions, USD 6.4 trillion) which trace
  to a single vendor blog, and the EPCIS adoption percentages, which measure general GS1
  standards use rather than EPCIS. Both are marked "do not cite as a measurement".
- The Arden Cold Chain, Calder Health, Calder Foods, and Kestrel Health cases, and every cost and
  incident figure attached to them, are constructed teaching examples and are labelled as such in
  the files.

## 2026.8.0 — 2 November 2026

### Added — Module 01A written in full
- Module 01A (`01_foundations/01A_foundations_units_time_assumptions/`) is now written in full:
  eleven topic files, 14,344 lines, each with all 17 required sections.
- `01_unit-of-measure-hierarchy.md` — each, inner pack, case, layer, and pallet; which conversions
  are reversible and which destroy information.
- `02_container-loading.md` — weight, cube, and floor-space constraints and how to identify which
  one binds; stowage factor and broken stowage.
- `03_case-pack-moq-rounding.md` — case pack, order multiple, and MOQ; why a fractional plan is not
  executable and how rounding compounds across levels.
- `04_calendars-weeks-time-zones.md` — fiscal calendars, the NRF 4-5-4 retail calendar, ISO 8601
  week numbering, IANA time zones and DST edge cases, cutoffs, and planning buckets.
- `05_lead-time-definitions.md` — nominal, actual, promised, and replenishment lead time; the
  distribution behind the average and the cost of inflated planned lead times.
- `06_currency-and-fx.md` — transaction, functional, and presentation currency; IAS 21 closing
  versus average rates; transaction, translation, and economic exposure; constant-currency
  reporting.
- `07_hierarchy-design.md` — product, location, customer, and time hierarchies; the fan trap and
  the chasm trap; additive, semi-additive, and non-additive measures; reconciliation tests.
- `08_measurement-error-bias.md` — accuracy, precision, trueness, and resolution as distinct
  quantities; cycle-count sampling and record-accuracy protocols; errors-in-variables and
  attenuation bias.
- `09_forecast-error-metrics.md` — MAE, RMSE, MAPE, sMAPE, MASE, bias, and tracking signal; what
  each one weights and how the choice changes the model selected.
- `10_intermittent-metrics.md` — demand classification by inter-demand interval and squared
  coefficient of variation; why percentage errors explode near zero; the metrics that survive.
- `11_demand-lead-time-distributions.md` — normal, gamma, lognormal, Poisson, and compound
  distributions; lead-time demand when both demand and lead time vary.

### Changed
- Module 01A status table reports 11 `complete` and the module total.
- `modules/01_foundations/_index.md` reports three modules complete; `VALIDATION_STATUS.md` now
  records 31 of 476 topic units and 40,413 lines.

### Notes
- Two files were repaired during production: a `../../` cross-module link in
  `08_measurement-error-bias.md` that resolved to a non-existent path was corrected to `../`.
- Several candidate figures were softened rather than repeated after the underlying source could
  not be retrieved first hand: the St. John (1985) 41% inflated-lead-time cost, the Nielsen et al.
  (2017) 35% variance share, the ADI/CV² cut-offs attributed to Syntetos, Boylan and Croston
  (2005), and the Hyndman and Koehler (2006) M3 rescoring pair counts. Each is either presented as
  a teaching example or described only by direction, with a §16 note saying so.
- The synthetic cases in topics 08, 09, and 11 (Ravenhill Distribution, the eight-error
  scoreboard, and the four-family quantile table) are labelled as constructed teaching examples,
  not published results.

## 2026.7.0 — 27 October 2026

### Added — Module 02 written in full
- Module 02 (`01_foundations/02_supply_chain_data_landscape/`) is now written in full: ten topic
  files, 12,900 lines, each with all 17 required sections.
- `01_systems-of-record.md` — what ERP, WMS, TMS, OMS, POS, PLM, and CRM each record and, more
  importantly, what each cannot answer.
- `02_master-data.md` — item, location, supplier, customer, and BOM masters; the identifier
  stack that joins them; the arithmetic of why duplicate item masters do not double statistical
  safety stock.
- `03_transaction-data.md` — orders, shipments, receipts, invoices, and inventory snapshots at
  their correct grain; the header–line–allocation hierarchy.
- `04_reference-and-external.md` — calendar, weather, macroeconomic, trade, and geospatial
  sources, with their known defects.
- `05_telemetry-iot-vision.md` — sensor, telematics, RFID, and image data as evidence, including
  the difference between a sensor reading and a decision-grade measurement.
- `06_data-movement.md` — EDI, API, flat file, and event transport, and how each constrains
  latency, completeness, and schema stability.
- `07_access-and-licensing.md` — access pathways, commercial licences, privacy, and contractual
  limits on use.
- `08_data-quality-dimensions.md` — completeness, validity, consistency, timeliness, uniqueness,
  and accuracy against ISO/IEC 25012 and ISO 8000 framing.
- `09_master-data-drift.md` — the module's worked case: five drift types, supersession chains,
  bitemporal capture, and a 36-month pack-change reconstruction with the phantom-case
  arithmetic shown.
- `10_source-selection-trust.md` — the hierarchy of trust, per-field source of record, and the
  source-selection record that closes the module.

### Changed
- Module 02 status table reports 10 `complete` and the module total.
- `modules/01_foundations/_index.md` and `VALIDATION_STATUS.md` updated to 20 of 476 topic units.

### Notes
- Several candidate statistics were removed rather than repeated after the underlying study
  could not be retrieved: the "$3.1 trillion cost of bad data" figure, the "15–25% of revenue"
  claim, the "80% of analyst time spent cleaning" claim, and the "3% of data meets quality
  standards" vendor survey are all present only as graded E or D entries marked "do not use".
- The NRF retail calendar is recorded as 4-5-4 throughout; the widespread "4-4-5" usage is
  described as a variant, not as the standard.

## 2026.6.0 — 27 October 2026

### Added — expanded topic files, first module
- Module 01 (`01_foundations/01_supply_chain_foundations/`) is now written in full: ten topic
  files, 13,169 lines, each with all 17 required sections.
- `01_three-flows-one-network.md` — physical, information, and financial flows; push–pull
  boundaries; bullwhip mechanics; cash conversion cycle arithmetic; GS1/EPCIS/Incoterms/IFRS 15
  interfaces.
- `02_decision-stack.md` — strategic, tactical, operational, and execution layers; time fences;
  decision rights; decision latency; agentic planning and decision governance.
- `03_network-entities.md` — party, location, item, and logistics-unit identity; unit-of-measure
  conversion; BOM structures; network topology; multi-tier visibility.
- `04_planning-processes.md` — S&OP, IBP, S&OE; the planning calendar; consensus forecasting and
  forecast value added; maturity models and their misuse.
- `05_inventory-and-service.md` — the inventory taxonomy; Little's law; CSL versus fill rate
  divergence; intermittent-demand distributions; obsolete and at-risk stock.
- `06_cost-and-working-capital.md` — the full cost taxonomy; holding-cost composition; DIO/DSO/DPO
  with correct denominators; supply chain finance and term stretching; total landed cost.
- `07_manufacturing-basics.md` — manufacturing archetypes; MPS/MRP/capacity; BOM accuracy and
  error propagation; routing and queue time; OEE and its misuse; ISA-95 data ownership.
- `08_transport-and-incoterms.md` — mode economics; the eleven Incoterms 2020 rules; landed-cost
  composition; duty and CBAM; carrier liability regimes.
- `09_warehousing-and-sourcing.md` — warehouse process flow and labour arithmetic; automation
  payback; the Kraljic portfolio and its critiques; category strategy.
- `10_archetypes-and-metric-conflicts.md` — functional/innovative and efficient/responsive
  alignment; Goodhart-style definition gaming; paired metrics; cost-of-compromise arithmetic.

### Changed
- The module index status table now reports 10 `complete`, 0 `planned`, and the module total.
- `VALIDATION_STATUS.md` records topic-level production as in progress: 10 of 476 units written.

### Notes
- Style follows the clinical-foundations exemplar: theory-dense, one short code or calculation
  block per file, evidence graded A–E. Several subagent drafts corrected claims that could not be
  sourced at first hand (Walmart OTIF targets, Incoterms revision status, OEE benchmarks,
  forecast-averaging claims) rather than repeating them.

## 2026.5.0 — 27 October 2026

### Changed — structural reorganization
- The 48 module folders under `modules/` are now grouped into **seven part folders**:
  `01_foundations` (7), `02_methods` (16), `03_commerce` (4), `04_value_governance` (4),
  `05_labs_electives` (2), `06_domain_electives` (10), `07_engineering_electives` (5).
- Module numbers are unchanged. The parts are a navigation layer, not a renumbering: every
  reference to Modules 00–45, every prerequisite, and every week mapping still resolves.
- All 48 module `_index.md` files updated: the survey-file link moved from `../../course/` to
  `../../../course/`, and the `Folder:` metadata now records the full part path.
- All 48 module `_index.md` version stamps moved to 2026.5.

### Added
- Seven part-level `_index.md` files. Each states the part's purpose, lists its modules with topic
  counts, draws the intra-part dependency sequence, and records what the part deliberately excludes.
- `README.md` at the package root: a single entry point with the directory tree, a "start here"
  table, the seven parts, the core term sequence, package counts, and the evidence conventions.

### Fixed
- `modules/README.md` linked to `../MODULE_EXPANSION_SPEC.md`, which does not exist at that path;
  it now points to `../course/MODULE_EXPANSION_SPEC.md`.
- `MODULE_EXPANSION_SPEC.md` folder-layout tree, agent write-scope rule, and verification commands
  updated for the part-folder depth (`modules/*/*/[0-9][0-9]_*.md`).
- `README.md` and `VALIDATION_STATUS.md` no longer describe 48 flat folders.

## 2026.4.0 — 27 October 2026

### Added
- Module 41 — Maritime, rail, and air cargo operations engineering: berth and quay crane capacity, yard dwell and rehandles, gate and hinterland evacuation, rail paths and shunting, ULD build-up and cutoffs, disruption recovery, terminal simulation, dangerous goods and ISPS, automation payback.
- Module 42 — Treasury, credit, and financial supply chain modelling: cash conversion cycle, driver-based cash forecasting, PD/LGD/EAD expected loss, trade finance instruments, reverse factoring and dynamic discounting, FX and interest-rate hedging, NPV and cost of capital, credit policy, KYC/AML/sanctions, supply chain finance disclosure.
- Module 43 — Security engineering for supply chains: STRIDE threat modelling, attack trees and ALE, OT and the Purdue model, zero trust and segmentation, third-party risk, SBOM and software provenance, cargo crime and C-TPAT/AEO, payment fraud, incident response and continuity, standards mapping.
- Module 44 — MLOps and production analytics engineering: lifecycle framing, versioning and reproducibility, experiment tracking, point-in-time features and leakage, orchestration, serving patterns, CI/CD/CT, drift and delayed-label monitoring, retraining and champion–challenger, model governance and platform economics.
- Module 45 — Plant layout, facility design, and detailed engineering interfaces: facility types and flow, systematic layout planning, storage and dock capacity, handling equipment and automation, P&ID and isometrics, MTO and revision control, safety and fire, layout simulation, design for supply chain.
- Five matching `modules/41_*`–`modules/45_*` index folders with 50 topic units.

### Changed
- Program package is now 48 module files and 48 module index folders. Modules 00–30 keep their numbering; 31–40 are post-core domain electives and 41–45 are cross-cutting engineering electives.
- `README.md`, `COURSE_OUTLINE.md`, `COMPETENCY_MAP.md`, `VALIDATION_STATUS.md`, `_course_spec.md`, `MODULE_EXPANSION_SPEC.md`, `INSTRUCTOR_KIT.md`, `GRADING_CALIBRATION.md`, `modules/README.md`, and `TOPIC_COVERAGE_MAP.md` updated for the new module set.
- `TOPIC_COVERAGE_MAP.md` now maps 93 topics; the former engineering gaps (maritime/rail/air operations, treasury and credit, security engineering, standalone MLOps, plant layout and detailed engineering) are closed by dedicated modules.

### Fixed
- Module 40's closing footer now links forward to Module 41 instead of claiming to end the elective set.

## 2026.3.0 — 27 October 2026

### Added
- Module 31 — Aftermarket, spare parts, and service supply chains: installed base, failure demand, criticality, METRIC sparing, repair loop, field service, availability contracts.
- Module 32 — Food, agriculture, and perishables: harvest and yield, perishable forecasting, FEFO, newsvendor with salvage, cold chain, HACCP/FSMA 204, food waste.
- Module 33 — Healthcare provider and life sciences: item master and UDI, par levels, OR case carts, consignment, shortages and recalls, clinical trial supply.
- Module 34 — Energy, utilities, and capital projects: critical spares, outage kitting, long-lead equipment, bulk logistics, earned value, commodity exposure, storm response.
- Module 35 — Public sector, defence, and humanitarian logistics: public procurement, whole-life cost, PBL, stockpiles, pipeline, last mile, cash versus in-kind, donor reporting.
- Module 36 — Global freight, ports, and trade policy: Incoterms, landed cost, classification and origin, port congestion, demurrage, freight procurement, chokepoints, CBAM and sanctions.
- Module 37 — Circular economy and reverse networks: disposition hierarchy, grading, reverse network design, refurbishment, fraud, packaging reuse, battery second life, circular metrics.
- Module 38 — Process industries and commodities: continuous and batch planning, yield and mass balance, blending LP, campaigns, bulk storage, hedging, laytime, debottlenecking, process safety.
- Module 39 — Mining, metals, and forestry: reserves and grade, cut-off, blending, extraction, pit-to-port, offtake, harvest scheduling, social licence and closure.
- Module 40 — Telecom, technology hardware, and aerospace: configuration and BOM, allocation, EMS, NPI, network rollout, channel sell-through, AOG pools, obsolescence, export control.
- Ten matching `modules/31_*`–`modules/40_*` index folders with 96 topic units.

### Changed
- Program package is now 43 module files and 43 module index folders. Modules 00–30 keep their numbering; 31–40 are post-core domain electives.
- `README.md`, `COURSE_OUTLINE.md`, `COMPETENCY_MAP.md`, `VALIDATION_STATUS.md`, `_course_spec.md`, `MODULE_EXPANSION_SPEC.md`, `modules/README.md`, and `TOPIC_COVERAGE_MAP.md` updated for the new module set.
- `TOPIC_COVERAGE_MAP.md` now maps 88 topics; the former business-domain gaps (energy, healthcare, agriculture, public sector, freight, reverse flows, service, process industries, mining, technology) are closed by dedicated modules. Remaining gaps are explicitly out-of-scope boundaries.

### Fixed
- The coverage audit no longer claims that business-domain depth is complete while offering only a "build a primer" method for half the economy.

## 2026.2.1 — 27 October 2026

### Added
- `TOPIC_COVERAGE_MAP.md`: 78 canonical topics mapped to modules and depth, with 10 known gaps stated explicitly.
- Module 15 §4.16: S&OP, IBP, and S&OE cycle, a worked demand–supply reconciliation, maturity model, and S&OP-specific failure modes.
- Module 10 §4.15: distribution requirements planning, deployment, and shortage-allocation rules with a worked example.
- Module 25 §4.5: supply chain finance instruments, including reverse factoring, dynamic discounting, and inventory finance.
- Module 27 §5.4–5.5, §6.6–6.8: licence and provenance audit, benchmark reproduction checklist, tool selection by constraint, build versus buy, and TCO.
- Module 28 §4.5–4.6, §7.7: evidence audit, counter-case discipline, and a pharmaceutical cold-chain deep dive.
- Module 29 §10A–10D: build-a-primer method, readiness diagnostic, thirty-minute sector interview, and primer quality checklist.
- Module 30 §3B–3D, §13B–13C, §14.2–14.4: data contracts, cost guardrails, local-only path, architecture review, demo script, lab map, and portfolio checklist.

### Changed
- Module 29 expanded to seven applied labs and 18 interview questions; module 30 expanded to 550 lines with the integration and review material.
- Module 25 and 26 lab numbering corrected from inherited 23.x and 24.x to 25.x and 26.x.
- `README.md`, `COURSE_OUTLINE.md`, and `VALIDATION_STATUS.md` now reference the coverage map and state the module-index status.

### Fixed
- Coverage gaps identified by audit: S&OP had only passing mentions; DRP and supply chain finance were not named topics; responsible-AI language was implicit rather than explicit.

## 2026.2.0 — 27 October 2026

### Added
- Module 21 — E-commerce supply chain analytics.
- Module 22 — Quick commerce and on-demand delivery.
- Module 23 — Omnichannel fulfillment and returns.
- Module 24 — Recommendation systems and personalization.
- Module 28 — Domain case library and sector playbooks.
- Domain-project assessment replacing the capstone: 13-artifact dossier and individual defense.

### Changed
- Module numbering after the removals was made contiguous: product/economics moved to 25, quality/model risk to 26, datasets to 27, domain cases to 28, electives to 29, labs to 30.
- `COURSE_OUTLINE.md`, `README.md`, `COMPETENCY_MAP.md`, `INSTRUCTOR_KIT.md`, `VALIDATION_STATUS.md`, and `GRADING_CALIBRATION.md` were rewritten around the new domain map.
- Term 3 is now "AI, Channels, and Domain Analytics"; assessment is labs 25%, case critiques 20%, midterm forecast dossier 20%, track playbook 20%, domain project 15%.
- Program outcomes now include e-commerce/quick-commerce/omnichannel fulfillment, recommendation systems, domain cases, and sector playbooks.

### Removed
- Module 21 — Deployment and MLOps (monitoring, drift, and incident content retained in Modules 19, 26, and 30).
- Module 22 — Privacy, security, ethics, and responsible AI (retained in Modules 26 and 29).
- Module 25 — Capstone and hackathons (replaced by the domain project).
- Module 26 — Career, portfolio, and interviews (interview questions retained in every module).
- Module 28 — Sources and further reading (source ladders retained in Modules 27 and 28).

### Known issues
- Topic-level files under `modules/` are indexed but not yet written.
- External review and learner pilot are pending.

## 2026.1.0 — 27 October 2026

### Added
- Full program architecture: 40 weeks, three terms, three tracks, 33 module files.
- Module 00 orientation and 2026 industry map.
- Modules 01–28 covering supply chain foundations, data, statistics, machine learning, forecasting, inventory, network design, transportation, warehousing, procurement, resilience, manufacturing, pricing, assortment, AI, control towers, simulation, e-commerce, quick commerce, omnichannel, recommendations, product economics, model risk, datasets, and domain cases.
- Advanced electives 29–30 for industry primers and production labs.
- `_research/industry_brief.md` with dated 2025–2026 sources.
- `COMPETENCY_MAP.md`, `VALIDATION_STATUS.md`, `MODULE_EXPANSION_SPEC.md`, and `_course_spec.md`.

### Design decisions
- Forecasting, inventory, and optimization are treated as one decision system rather than separate specialties.
- Every technical module requires a baseline comparison and a cost or service interpretation.
- Regulation and standards are integrated into domain modules, not isolated in an appendix.
- Vendor and analyst claims are labelled and dated.
- The capstone requires a decision, a baseline, temporal validation, an impact estimate, and a defense.
- The 2026.2.0 domain project applies the same standard to an e-commerce, quick-commerce, omnichannel, or sector case.

### Known issues
- External review and learner pilot are pending.
- Some regulatory dates and market figures are dated snapshots and require re-verification.
- Module-level line counts vary; the expanded topic-level edition is planned but not produced.
- Labs reference public datasets whose licenses and availability may change.

## Planned

- 2026.1.1: link and consistency pass; lab starter notebooks.
- 2026.2.0: expanded topic files under `modules/` per `MODULE_EXPANSION_SPEC.md`.
- 2026.2.1: instructor kit with anchor solutions and grading calibration.
- 2027.1.0: regulatory refresh and cohort feedback.
