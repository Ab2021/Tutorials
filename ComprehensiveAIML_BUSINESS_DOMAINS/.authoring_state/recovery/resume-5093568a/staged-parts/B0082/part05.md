## 9. Evaluation & Validation

### 9a. What has to be evaluated, and against what comparator

Three distinct questions need three distinct evaluation designs, and conflating them is the most common way an incident review produces an unfalsifiable conclusion.

| Question | Numerator | Denominator | Population | Window | Comparator | Guardrail | Status |
|---|---|---|---|---|---|---|---|
| Did the feed break? | orders with distance_km outside [0,100] or unit flag mismatched | all scored orders | region/channel slice | 09:05–12:00 | same slice, prior 7 days | zero contract violations | executed on constructed data |
| Did decisions degrade? | contacts landing on orders that later arrive late | contacts placed | contacted orders with matured labels | 08:00–12:00, aged 24 h | same window, reference model on clean feed | no fall in clean-cohort precision | designed, not executed |
| Did the intervention help? | late rate among contacted orders | contacted orders in treated arm | eligible orders, same score band | 4 weeks | uncontacted arm in the same band | no change in complaint rate | designed, not executed |

Each row's status is stated honestly: the first is an arithmetic check on constructed data, and the other two are designs that a real deployment would have to run. Nothing in Sections 10 and 11 reports them as executed.

### 9b. The decision-degradation test, sized

The degradation test compares queue precision under the corrupted feed against queue precision under the clean feed, on the same clock window, using matured labels. Its sample size is not a matter of taste; it is determined by the effect size and the two-proportion variance.

```text
FORMULA [F-15]: n per arm = 7.849 * 2 * p_bar * (1 - p_bar) / (p1 - p2)^2
WHERE: p1 = 0.375 clean-feed queue precision; p2 = 0.23 corrupted-feed queue precision;
       p_bar = 0.3025; 7.849 = (z_{0.975} + z_{0.80})^2 for alpha 0.05 two-sided and
       power 0.80; n in contacts per arm.
CONSTRAINT: two independent arms with equal allocation; the test is on matured labels only,
       so both arms must be aged to the same cut-off; if the arms are unequal the variance
       term changes and 7.849 must be recomputed rather than reused.
NUMERICAL EXAMPLE: p1 - p2 = 0.145; p_bar = 0.3025; p_bar*(1-p_bar) = 0.211044
       n = 7.849 * 2 * 0.211044 / 0.021025 = 3.3129 / 0.021025 = 157.57
       so 158 contacts per arm, 316 in total
       at 30 contacts per hour that is 10.5 hours of queue time
INTERPRETATION: the decision-degradation question is answerable inside one shift, which is
       why it is worth pre-registering. The 158-contact figure is small because the effect
       is large — a 39 percent relative drop — and large effects are cheap to measure.
```

The same test on **observed** rather than matured precision needs 222 contacts per arm instead of 158, because the maturity process compresses both arms toward each other. `0.238231` against `0.134716` is a difference of 0.103515 rather than 0.145, and the required sample rises by a factor of 1.41. That is a concrete cost of skipping the maturity correction: measuring the wrong quantity costs 41 percent more contacts to reach the same conclusion, and at 30 contacts per hour that is 4.3 extra hours of exposure.

### 9c. The intervention-effect test, and why it is the hard one

Whether a contact actually prevents lateness cannot be answered by comparing contacted orders to all uncontacted orders, because the contact decision is made on the score. The only defensible design is a randomised holdout within the score band being served.

```text
FORMULA [F-16]: n per arm for an intervention effect of e = 0.18 on a base of p1 = 0.375
WHERE: treated arm late rate p2 = p1 * (1 - e) = 0.3075; p_bar = 0.341250;
       alpha 0.05 two-sided, power 0.80.
CONSTRAINT: requires randomisation within the band, not across bands, because the rescue
       rate plausibly varies with rank. Requires the holdout to be unobservable to agents,
       or they will compensate by calling holdout orders informally.
NUMERICAL EXAMPLE: p1 - p2 = 0.0675; p_bar*(1-p_bar) = 0.224760
       n = 7.849 * 2 * 0.224760 / 0.00455625 = 3.5283 / 0.00455625 = 774.4
       so 775 contacts per arm, 1,550 in total
       at a 90/10 split: treated arm fills in 775/27 = 28.7 hours,
       control arm needs 775/3 = 258.3 hours of clock time to accumulate
INTERPRETATION: the intervention question needs about 11 days of clock time at a 10 percent
       holdout, against 10.5 hours for the degradation question. That asymmetry is the reason
       teams measure degradation and assert intervention effects.
```

The holdout is expensive but it is the only design that produces the number the value model needs. Without it, `e = 0.18` stays an assumption, and every currency figure in Section 6 inherits that assumption's error. Section 13 treats "we priced the decision on an unmeasured rescue rate" as a transferable lesson rather than a footnote.

### 9d. Guardrails

A degradation test that only checks the headline number can pass while the system gets worse in a way nobody looked at. Five guardrails are specified with thresholds, each computed from the same parameter set rather than chosen for roundness.

| Guardrail | Threshold | Derivation | Breach response |
|---|---|---|---|
| Clean-cohort queue precision | not below 0.360 | 4% relative fall from 0.375, inside the 0.0161 standard error at n=900 | investigate before scaling containment |
| Cohort slot share | not above 0.60 | share at which harm 39.45/hour equals one false-page budget per shift | page the cohort owner |
| Feature contract violations | exactly zero | contract is binary; any violation is a defect | block the producer version |
| Score distribution drift, clean cohort | PSI below 0.0313 | half the 0.0614 15-minute critical value, as an early watch level | review within the hour |
| Contacted-order complaint rate | not above 1.02x baseline | a 2% increase is the smallest worth investigating against a 3% sampling error | pause queue expansion |

The first guardrail is worth examining because its derivation is easy to get wrong. At 900 contacts the standard error of a 0.375 proportion is `sqrt(0.375*0.625/900) = 0.016136`, so a 4 percent relative fall is `0.375 * 0.04 = 0.015`, or just under one standard error. That is deliberately a watch level rather than a paging threshold: it will fire on ordinary variation about a third of the time by design, and its purpose is to make an investigating human look at a slice, not to wake anyone.

### 9e. Uncertainty, and what is not asserted

Every number in this case carries an uncertainty that is stated or explicitly not computed, and the distinction is maintained.

Computed uncertainty:

- Sampling uncertainty on queue precision: at 900 contacts, a 95 percent interval on 0.375 is `0.375 ± 1.96 * 0.016136`, or [0.3434, 0.4066]. At 225 contacts it is [0.3117, 0.4383].
- Sampling uncertainty on observed precision: `0.23 ± 1.96 * 0.014029` at 900 contacts is [0.2025, 0.2575].
- The difference of two independent proportions at 900 each has standard error `sqrt(0.375*0.625/900 + 0.23*0.77/900) = 0.021379`, so the estimated 0.145 gap is 6.8 standard errors from zero.

Not computed, and why:

- No confidence interval is attached to the 223.16 harm figure, because it is a linear combination of three assumed parameters (`e`, `L`, `c`) rather than a measured quantity. An interval on it would express the uncertainty of the assumptions, and a point estimate with a fabricated interval is worse than a point estimate labelled as conditional.
- No interval is attached to the PSI statistic itself. Its null distribution was simulated, but a confidence interval for a non-central PSI requires a non-centrality model that this case does not have.
- No significance is claimed for the comparison between the two-window and single-window alert policies, because both are evaluated against the same realised incident and neither has replicates.

The one sensitivity that is fully table-driven, because it is the assumption most likely to be challenged:

| Rescue rate e | e*L | Break-even precision | Reference value/hour | Live value/hour | Harm/hour |
|---:|---:|---:|---:|---:|---:|
| 0.060 | 10.80 | 0.537037 | -52.50 | -99.48 | 46.98 |
| 0.086 | 15.47 | 0.375000 | 0.00 | -67.28 | 67.28 |
| 0.090 | 16.20 | 0.358025 | 8.25 | -62.22 | 70.47 |
| 0.120 | 21.60 | 0.268519 | 69.00 | -24.96 | 93.96 |
| 0.180 | 32.40 | 0.179012 | 190.50 | 49.56 | 140.94 |
| 0.240 | 43.20 | 0.134259 | 312.00 | 124.08 | 187.92 |
| 0.360 | 64.80 | 0.089506 | 555.00 | 273.12 | 281.88 |

Read the top rows first, because they overturn the incident conclusion rather than refining it. At `e = 0.12` the reference queue earns `30 * (0.375*21.60 - 5.80) = 30 * (8.10 - 5.80) = 69.00` per hour, while the live queue at that rescue rate loses `30 * (0.23*21.60 - 5.80) = 30 * -0.832 = -24.96` per hour. At `e = 0.06` even the reference queue is value-destroying at `30 * (4.05 - 5.80) = -52.50` per hour.

The exact zero-value rescue rate for the reference queue is `e = c / (P_0 * L) = 5.80 / (0.375 * 180) = 5.80 / 67.50 = 0.085926`. Below 8.6 percent, the entire proactive queue destroys value even when it is working perfectly and no incident has happened.

That is the sensitivity that should govern the runbook. At `e = 0.18` the incident's harm is 140.94 per hour and containment is unambiguously correct. At `e = 0.09` the harm is 70.47 per hour while the reference queue earns only 8.25 per hour, so the 95-minute incident consumes `70.47 / 8.25 = 8.5` hours of programme value — and the correct escalation is to the programme owner, not to the on-call lead. The incident does not change the rescue rate; it reveals how thin the margin always was.
