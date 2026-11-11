### 3.6 Measurement definitions that decide the estimate

Three measurement choices determine the number before any statistics run, and each has a defensible alternative that produces a different answer.

**Outcome definition.** "Purchase" can mean order placed, order paid, order shipped, or order retained past the return window. Each definition includes a different set of customers and each is measured at a different lag. For a 14-day window, order-placed counts are available within hours and retain-past-return counts are available in 30–45 days. Reading the test at the earlier definition is legitimate for an interim decision; reading the early definition and calling it the final one systematically overstates effects when returns are elevated in the treated arm, which is exactly what a discount offer does. State the definition and the lag together.

**Attribution window inside the outcome window.** If the outcome is "purchase attributable to the campaign" and attribution uses a 7-day click window while the experiment window is 14 days, then days 8–14 of assignment contribute purchases that only count if the customer clicks again. The estimand drifts toward "purchase within 7 days of a click", which is not what the decision needs. The cleanest practice is to make the primary outcome an unconditional purchase flag over the full window, with attribution used only for descriptive decomposition.

**Denominator population.** The assignment ledger's row count, the consent-eligible count, and the deliverable count differ by 5–20% in normal operations. The primary denominator is the assignment ledger, because that is the population randomization defined. Reporting on the deliverable subset is a different estimand and must be labelled. The failure to distinguish these produces the classic discrepancy where two teams report the "same test" with effects that differ by a factor of two, and both are arithmetically correct.

**Event time versus processing time.** A purchase that happens on day 13 but lands in the warehouse on day 22 belongs to the cohort by event time. Cohort assignment by ingest time creates a bias correlated with arm whenever the treated arm generates a different mix of channels — app purchases, for instance, sync later than web purchases. Store both timestamps, filter on event time, and cut the cohort on a maturity rule derived from the ingest-lag distribution rather than from the query date.

**Exposure definitions are not interchangeable.** An impression (the ad was served), a viewable impression (at least 50% of pixels for at least one second under the Media Rating Council convention), a click, and a session are four different variables with four different relationships to assignment. Any of them used as a filter creates the collider problem in Section 6.2; any of them used as an outcome measures a different behaviour. The safe uses are: assignment as the analysis unit, delivery as a compliance diagnostic, and the purchase flag as the outcome.

### 3.7 The vocabulary of "lift" in stakeholder conversation

**Absolute lift** is the difference in rates, in percentage points. It is the quantity that multiplies by population size to produce incremental units, so it is the only lift that connects to money.

**Relative lift** divides the absolute lift by the baseline. It is scale-free, which makes it comparable across campaigns and useless for a budget decision on its own: 24% relative on a 4.3% baseline is 1.03 points, and 24% relative on a 0.4% baseline is 0.10 points.

**Incremental units** is absolute lift times the population to which it applies. This is where transport assumptions enter: the tested population was filtered, the deployment population is not, and the marginal unit has a different baseline.

**Return on ad spend (ROAS)** divides incremental revenue by spend. It inherits every assumption of incremental units plus the revenue definition. A ROAS computed from a point estimate with no interval communicates more certainty than the design supports, and a ROAS computed from attributed revenue is not an incremental quantity at all.

**Statistical significance** is a statement about a null hypothesis, not about profitability. A significant 1.03-point effect at a 1.39-point break-even is a statistically significant loss.

**Takeaway:** Two of these six terms are decision inputs (absolute lift, incremental units), two are diagnostics (relative lift, significance), and two require explicit assumptions that are usually left unstated (ROAS, transport).
