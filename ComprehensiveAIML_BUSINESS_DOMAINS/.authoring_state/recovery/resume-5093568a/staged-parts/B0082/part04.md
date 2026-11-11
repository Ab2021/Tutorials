## 7. The Finding

Four claims, each with its support and its scope.

**CLM-01.** The observed input shift is not sampling noise. At 900 orders the statistic is 0.154833, or 10.09 times the 0.999 critical value of its null distribution, and a seeded 3,000-trial simulation confirms the calibration to within 7 percent. The shift is real. This is an arithmetic conclusion from Section 6.2, and it says nothing about decisions.

**CLM-02.** The shift is not explainable by the campaign geography alone. A campaign that genuinely moved 22 percent of orders into longer-distance bands would produce a pooled PSI of 0.000958 against the reference — under one unit of `N * PSI` at 900 orders, which is indistinguishable from no shift. The observed 0.154833 is 161 times that. The gap is the evidence that the population's inputs changed by more than its composition.

```text
FORMULA [F-13]: PSI_campaign = PSI( p , w*shard + (1-w)*p )
WHERE: p the reference distribution; shard the campaign cohort's own clean distance
       distribution (0.44, 0.31, 0.25); w = 0.22 the cohort share.
CONSTRAINT: this is the counterfactual for "the campaign happened and the features are
       honest". It is a mixture calculation, not a measurement.
NUMERICAL EXAMPLE: mixture q = (0.4868, 0.3022, 0.2110)
       0.22*0.44 + 0.78*0.50 = 0.4868;  0.22*0.31 + 0.78*0.30 = 0.3022
       0.22*0.25 + 0.78*0.20 = 0.2110
       PSI = 0.000958, and 900 * 0.000958 = 0.862, far below the 13.8155 critical value
INTERPRETATION: a genuine campaign of this size is statistically invisible in this feature.
       The alarm cannot be a campaign alarm. Either the cohort is far larger than 22 percent,
       or the feature is wrong, and only the second survives the unit-range check.
```

**CLM-03.** The decision cost of the incident is bounded by queue saturation, not by the size of the corrupted share. At 30 contacts per hour and 900 orders per hour, any corruption above 3.33 percent fills the queue and drives realised precision to the cohort's own 0.23. The harm is 140.94 currency units per hour, and it is the same for a 4 percent corruption as for a 22 percent one. This is arithmetic from F-03 and F-04 and it is the most operationally useful single number in the case.

**CLM-04.** The dashboard precision of 0.238 is not evidence of a quality drop and neither is it evidence of stability. It is the two-hour-maturity shadow of a true 0.375, and the correct correction returns exactly 0.375. No conclusion about model quality can be drawn from it in either direction until labels mature or the inverse map is applied. Section 6.5 supplies the map and its validity condition.

Two things this case explicitly does **not** establish.

It does not establish that v43 is a worse model than v42, or than any alternative. The only quality comparison in the file is between a queue precision under corruption and a queue precision without it, which is a statement about the data feed.
It does not establish the business outcome. The 223.16 currency-unit figure is the harm the queue would have accrued over 95 minutes under the stated value model. Whether service credits were actually paid, and whether customers churned, was never observed and is not claimed.

## 8. System/Solution Design

### 8a. The scoring and monitoring paths as they were

The pre-incident architecture is drawn as three paths, because the failure lives in the interaction between them rather than inside any one.

```text
order accepted (event time)
   |
   +--> feature fetch: distance_km, courier_load_ratio, promised_window_min
   |        |  online store read, 5 min freshness contract
   |        +--> contract validation: type, unit name, range 0..100
   |
   +--> scoring: v43 artifact, one score per order
   |        |
   |        +--> eligibility filter (unserviceable, already delivered, no consent)
   |        +--> rank and take top 30 of the hour
   |        +--> queue write: order_id, score, slot, artifact version
   |
   +--> async: drift job (15 min + 1 hour windows, per feature)
            +--> PSI vs stored reference snapshot
            +--> alert router (two consecutive windows above 0.2)
```

Three architectural properties determined the incident's shape.

The validation step checked type and broad range but not unit semantics. A value of 34,200 is a valid float in the range 0 to 100,000 and was rejected only because the range ceiling was 100 — a guard that catches the metres error only when the distance is short. At 24.6 kilometres the corrupted value is 24,600, which the validator rejected; at 0.0246 kilometres, the metres value 24.6 passed and looked entirely plausible. That is why the corruption was partial rather than total and why its share settled at 22 percent instead of 100.

The feature read happens before the model runs, which is what makes model rollback structurally powerless against a data defect.
The drift job compared against a stored snapshot rather than a rolling recomputation, so any backfill that regenerated the snapshot would silently rebase every subsequent PSI. Section 12 treats that as a distinct failure mechanism rather than a footnote.

### 8b. The design that would have shortened exposure

Four changes, ordered by how much exposure each removes per unit of engineering effort.

**Change 1: replace the fixed threshold with a size-indexed critical value.** The alert should compute `chi-square_{0.999}(k-1) / N_window` for every window and compare the observed PSI against that, where `N_window` is the actual order count in the window. A low-traffic window gets a high threshold, a high-traffic window a low one, and the false-page rate stays near the stated 0.001 regardless of volume. The comparison is one division and removes the entire class of "threshold copied from a blog post" errors. This change alone does not shorten this incident, because the statistic was already past its threshold; it prevents the far more common opposite failure, where a high-volume system alerts constantly at 0.2 while a low-volume one never alerts at all.

**Change 2: alert on the slot share, not on the input distribution.** The queue's own composition is a direct measurement of the thing that costs money, and it needs no reference distribution.

```text
FORMULA [F-14]: alert if (orders from any single cohort in the top K) / K > gamma
WHERE: cohort a categorical key known before scoring (campaign flag, region, channel);
       K slots per hour; gamma a share threshold set from the saturation analysis.
CONSTRAINT: the cohort must be defined by a field that is not itself corrupted. A cohort
       defined by distance would inherit the defect and agree with the broken ranking.
NUMERICAL EXAMPLE: at 22 percent corruption the cohort fills 30 of 30 slots, share 1.00.
       At 2 percent corruption the cohort supplies 18 of 30, share 0.60.
       At 1 percent corruption, 9 of 30, share 0.30.
INTERPRETATION: set gamma at 0.60 and the monitor fires at a 2 percent corruption, where the
       input-distribution test needs ten hours of traffic to see anything. This is the change
       that actually shortens exposure, because it measures the decision rather than the input.
```

The threshold `gamma = 0.60` is derived, not chosen for comfort: at that share the queue's precision is 0.334, harm is 39.45 per hour, and the expected harm during a one-window detection delay exceeds the false-page budget. Below it, harm is small enough that a slower response is affordable.

**Change 3: make the cohort key a first-class monitoring dimension.** Every score record should carry the campaign flag, the producer version, and the artifact version, so that precision, slot share, and score distribution can be sliced by all three without a join. In this incident the campaign flag existed at scoring time and was simply not logged. The cost of logging it is one column; the cost of not logging it was a twenty-five-minute diagnosis under uncertainty.

**Change 4: put the unit in the field name and the unit in the assertion.** `distance_km` rather than `distance`, plus an assertion `0 <= distance_km <= 100` enforced at the producer with a dead-letter queue rather than at the consumer with a log line. A producer that rejects its own bad rows stops the corruption at source; a consumer that logs and continues converts a producer bug into a model-quality incident, which is strictly worse because it is harder to diagnose and it corrupts the training history.

### 8c. The containment mechanism

Containment here means routing the affected cohort to a fallback ranker for the duration. The mechanism is a routing rule in the scoring service keyed on a field that is not corrupted.

```text
score request
   |
   +--> read campaign_flag, feature_source_version, distance_km
   |
   +--> if feature_source_version is in BLOCKLIST and campaign_flag is set:
   |        route to fallback ranker  (courier_load_ratio, promised_window_min, hour)
   |        write score with artifact = "fallback-v1", cohort = "campaign"
   |
   +--> else: score with v43 as usual
   |
   +--> eligibility and rank operate on the blended score
```

The properties that make this the right containment rather than a hack.

It is reversible in one configuration change, and it is scoped to a cohort rather than to a version, so it does not require a deploy.
It uses the producer version as the blocklist key, which means it can be armed before the producer is fixed and disarmed after, without a second diagnosis.
It writes an explicit `artifact` value, so post-hoc analysis can separate fallback-scored orders from model-scored ones and no evaluation is silently pooled across two different decision rules.
It does not attempt to repair the corrupted value. Repairing would require a unit guess per row, and a wrong guess in the repair path is indistinguishable from a wrong guess in the producer.

Fallback quality is the weak point and should be stated as a number rather than a hope. The fallback uses courier load ratio and promised window, both clean. Its precision on the campaign cohort is assumed at 0.30 against the cohort's own 0.23 and the clean population's 0.375. That assumption yields containment precision `P_c = 0.3485` and recovery of 79.1 percent of the gap in Section 6.7. If the fallback were no better than the corrupted ranking, containment would recover nothing and the decision would fall back to a queue pause.

### 8d. Alert contract: what the page should carry

An alert that arrives without the numbers needed to triage costs more than it saves. The contract below is derived from the diagnosis path actually walked in this case, and each field exists because its absence caused a step in Section 6 to be taken under uncertainty.

| Field | Example value | Why it is on the page |
|---|---|---|
| Statistic and threshold | PSI 0.1548 vs critical 0.0614 | lets the on-call see margin, not just breach |
| Window count and span | 225 orders, 09:15–09:30 | sample size drives the critical value |
| Bin contributions | 18.3%, 11.0%, 70.7% | localises the shift to the tail bin |
| Reference snapshot hash | `ref-2026-10-03-7d` | detects a rebased baseline |
| Producer versions in window | v117 (94%), v118 (6%) | correlates shift with a release |
| Cohort slot share | 1.00 for campaign cohort | the decision-relevant signal |
| Queue precision, 2-hour maturity | 0.1347 (true 0.230 inverse) | prevents misreading the dashboard |
| Last 24-hour page count | 1 | gives base rate for triage urgency |

The sixth field is the one that would have saved the most time. The cohort slot share was computable in the same query as the PSI and would have said, unambiguously, that the campaign cohort held every slot — collapsing the "campaign mix or unit bug" debate into a single follow-up question about whether the cohort's geography had really changed.

### 8e. Operational readiness and rollback

| Capability | Before incident | Target | Gap |
|---|---|---|---|
| Cohort share monitor | none | per-15-minute, per-cohort | build |
| Size-indexed PSI threshold | fixed 0.2 | computed per window | configuration |
| Producer version on score record | absent | required field | schema change |
| Unit assertion at producer | log only | reject to dead-letter | producer change |
| Fallback ranker | none resident | resident, cohort-routed | build |
| Maturity-corrected precision | absent | inverse map in dashboard | dashboard change |
| Containment approval | 15 min | pre-authorized for data defects | policy change |

The last row is the cheapest and the most valuable. In this incident containment would have completed at 09:30 rather than 10:06 had the data-defect route been pre-authorized, removing 35.23 currency units of harm and, more importantly, removing the fifteen-minute argument about authority during the incident itself. Pre-authorising a narrow, reversible, cohort-scoped configuration change is a policy decision that costs a paragraph of documentation.
