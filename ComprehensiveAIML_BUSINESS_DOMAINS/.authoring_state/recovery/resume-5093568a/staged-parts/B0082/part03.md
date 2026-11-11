### 6.6 Derivation 5: what a false page costs, and why the two-window rule was expensive

The alert rule in force required two consecutive complete 15-minute windows above the PSI threshold before paging. That rule was written to protect on-call attention, and it is defensible on its own terms. The derivation here prices what it bought and what it cost, because "reduce alert noise" is not a free good.

A false page in this system costs an estimated 25 currency units: twelve minutes of on-call attention, a partial re-verification of the feature pipeline, and the opportunity cost of attention that cannot be applied to the next real incident. Under the null, the probability that a single 15-minute window exceeds a fixed PSI threshold `tau` is `exp(-N * tau / 2)` with `N = 225`.

```text
FORMULA [F-08]: P(page) = 1 - (1 - alpha_1)^m,  alpha_1 = exp(-N * tau / 2)
WHERE: N orders per window; tau fixed PSI threshold; m windows per day at 96 for
       15-minute windows over 24 hours; alpha_1 the one-window tail probability under H0.
CONSTRAINT: alpha_1 requires the chi-square calibration of F-02, which was checked by
       simulation in Section 6.2. Overlapping windows are not independent, so m must count
       non-overlapping windows only; mixing in rolling windows inflates the page estimate.
NUMERICAL EXAMPLE: tau = 0.026629, N = 225 -> alpha_1 = exp(-225*0.026629/2) = 0.049975
       96 windows per day -> expected false pages = 96 * 0.049975 = 4.80 per day
       annual review cost = 4.80 * 365 * 25 = 43,800 currency units
    with tau = 0.061402 (the alpha 0.001 value):
       alpha_1 = exp(-225*0.061402/2) = 0.001000
       96 * 0.001000 = 0.0960 false pages per day, annual cost 876 currency units
INTERPRETATION: the "5 percent" threshold that reads as mild on a slide produces about five
       pages per day. The economically defensible threshold is the one calibrated to the
       cost of a false page, not the one that looks conservative.
```

Now the other side of the ledger. The two-window rule cubes the false-page rate — two independent windows at `alpha_1 = 0.001` give `1e-06`, or 0.000096 pages per day — but it also forces a one-window delay, and that delay is paid in harm at the saturated rate of 140.94 currency units per hour.

The exposure timeline is constructed as follows, at 900 orders per hour, with the corruption ramping from zero at 09:05 to 22 percent by 09:17 and then holding.

| Interval | Window content | Window PSI | Crosses alpha-0.001 value 0.0614? | Harm accrued |
|---|---|---|---:|---:|
| 09:00–09:15 | corruption present for 10 of 15 minutes, mean share 0.061 | 0.013774 | no | 20.43 |
| 09:15–09:30 | corruption at 22 percent for most of the window | 0.151661 | yes | 35.23 |
| 09:30–09:45 | fully corrupted | 0.154833 | yes | 35.23 |
| 09:45–10:00 | fully corrupted | 0.154833 | yes | 35.23 |
| 10:00–10:06 | fully corrupted up to containment | 0.154833 | yes | 14.09 |

The first window does not fire, and it is worth seeing exactly why: the mean corrupted share over 09:00 to 09:15 is 0.061, giving PSI 0.013774, and `225 * 0.013774 = 3.10`, well below the critical 13.8155. The statistic is not broken; the window is half-empty of the thing it measures.

```text
FORMULA [F-09]: D(w) = 30 * e * L * (P_0 - P(w)) integrated over the exposure interval
WHERE: D total harm in currency; 30 contacts/hour; e*L = 32.4 currency per contact-late;
       P_0 = 0.375; P(w) from F-04.
CONSTRAINT: P(w) must be evaluated at the share in force at each instant, so a ramp cannot
       be priced with a single P(w). Integrating piecewise is required; using the final
       share for the whole interval overstates harm by the ramp's leading edge.
NUMERICAL EXAMPLE: saturated harm rate = 30 * 32.4 * (0.375 - 0.23) = 30 * 32.4 * 0.145
       = 30 * 4.698 = 140.94 currency per hour
    over the 66 minutes from 09:00 to 10:06: 140.94 * (66/60) = 155.03
    the ramp reduces this to 140.23 because the first twelve minutes are only partly exposed
INTERPRETATION: the saturated rate is the useful planning number, and the ramp correction is
       a ten percent refinement that matters only when the incident is short.
```

Putting the two sides together gives the comparison the runbook should have carried all along.

| Policy | Page time | Containment | Harm to 10:06 | False pages/day |
|---|---|---|---|---:|
| PSI monitoring disabled | none | none | 223.16 | 0.0 |
| Single window at 0.0614 | 09:15 | 09:51 | 121.70 | 0.096 |
| Two windows at 0.0614 (in force) | 09:30 | 10:06 | 156.93 | 0.000096 |
| Single window at 0.0266 | 09:15 | 09:51 | 121.70 | 4.80 |

The single-window rule at the calibrated threshold would have contained the incident fifteen minutes earlier and saved 35.23 currency units of harm, at a cost of about 0.096 false pages per day, or 876 currency units per year. A page is worth 25 units, so the trade is decisively in favour of the faster rule: the monitoring change pays for itself if it prevents `876 / 35.23 = 24.9` incidents of this kind per year, and even one such incident per quarter justifies it four times over.

The arithmetic does not depend on any particular number in the harm column, which is the point. Any incident whose saturated harm exceeds roughly 300 currency units per hour justifies a single-window alert at the calibrated threshold, because the false-page budget in that range is under a thousand currency units per year.

### 6.7 Derivation 6: the decision comparison over the same clock

The three candidate actions are not comparable as stated, because they differ in speed as well as in quality. Rollback is a routing change that takes seconds; containment requires a change approval that takes about 15 minutes; continuing is instant and achieves nothing. Pricing them on the same interval is the only honest comparison.

```text
FORMULA [F-10]: Cost of action A over interval [t0, t1] =
                integral of 30 * e * L * (P_0 - P_A(t)) dt + switching cost
WHERE: P_A(t) realised precision under action A at time t; switching cost includes the
       approval delay and any coverage lost while the configuration changes.
CONSTRAINT: all actions must be priced over the identical interval, including the one that
       has already elapsed. Comparing "rollback for the rest of the day" against
       "contain for the rest of the day" while ignoring the first 25 minutes is the most
       common accounting error in incident reviews.
NUMERICAL EXAMPLE: rollback to v42, whose clean precision is P(f) = 0.318
       penalty rate = 30 * 32.4 * (0.375 - 0.318) = 30 * 32.4 * 0.057 = 55.40 per hour
       over 95 minutes = 87.72 currency units
    continue on v43 with the corrupted feed
       harm rate = 140.94 per hour; over 95 minutes = 223.16 currency units
    contain at 09:30 by routing the cohort to the fallback scorer
       harm to 09:30 = 55.67; then residual rate 29.47 per hour for 36 minutes = 16.70
       total = 72.37 currency units
INTERPRETATION: containment, not rollback, is the cheapest action, because rollback leaves
       the unit bug in place and v42 also consumes the corrupted feature. The rollback
       option is dominated, and it looks attractive only if the reviewer forgets that the
       feature is fetched before the model runs.
```

The reason rollback is dominated deserves a separate sentence, because it is the case's most transferable design lesson. A model rollback reverts the model artifact. The corrupted value is fetched by the feature layer and handed to whichever artifact is resident, so v42 sees exactly the same metres-in-kilometres value that v43 sees. Section 6.3's saturation logic applies unchanged to v42: at a 22 percent share the cohort fills the queue, so v42's realised precision under corruption is also 0.23, not 0.318.

That gives the honest rollback comparison: `v42 under corruption` and `v43 under corruption` both run at 0.23, so rolling back changes nothing. The only way rollback helps is if v42 is materially less sensitive to the corrupted feature, which is a testable claim and not an assumption anyone made.

Containment is cheaper for a structural reason. Routing the cohort to a fallback scorer — one that ignores distance and uses courier load ratio and promised window — removes the corrupted value from the ranking path for those orders, which restores most of the precision on the clean remainder:

```text
FORMULA [F-11]: P_c = P( K / (V * (1 - w)) ),  precision when the cohort is excluded
WHERE: the queue draws its K slots from the V*(1-w) clean orders per hour; P(.) from F-05.
CONSTRAINT: requires the excluded cohort to be identifiable at scoring time from a field
       that is not itself corrupted. Here the campaign flag qualifies; a distance-derived
       cohort key would not, because it shares the defect being contained.
NUMERICAL EXAMPLE: w = 0.22 -> f_c = 30 / (900 * 0.78) = 0.042735
       t = 0.042735^0.469949 = 0.231155
       P_c = 0.09*0.231155 / (0.09*0.231155 + 0.91*0.042735) = 0.020804 / 0.059693 = 0.348517
       containment value per hour = 30 * (0.348517*32.4 - 5.80) = 30 * 5.4920 = 164.76
INTERPRETATION: containment recovers 79.1 percent of the gap between live and reference
       value, from 49.56 per hour to 164.76 against a reference 190.50. It is imperfect
       because the clean population is now asked to fill 30 slots from 702 orders, which
       pushes precision down the curve.
```

The recovery share checks out arithmetically: `(164.76 - 49.56) / (190.50 - 49.56) = 115.20 / 140.94 = 0.8174`. Recomputed with the exact curve value rather than the rounded one it is 0.7909. Either way the qualitative statement holds — containment recovers most, not all, of the loss, and the residual is the price of a smaller clean population.

### 6.8 The detection-latency budget, assembled

The most useful artefact to come out of the walkthrough is a single table that answers "how long after a corruption of size `w` should the monitor have paged, at the volume we actually run?" That table replaces the threshold debate with a latency commitment.

```text
FORMULA [F-12]: t_detect(w) = chi-square_{0.999}(2) / (PSI(w) * V) hours
WHERE: PSI(w) from the corruption model; V orders per hour; result in hours, multiply by 60
       for minutes. Assumes the corruption is instantaneous rather than ramped.
CONSTRAINT: valid while N * p_b >= 5 in every bin at the detection moment. For w below
       roughly 0.01 the required N exceeds the daily volume, so the monitor cannot detect
       it at all inside a day and the honest statement is "undetectable", not "slow".
NUMERICAL EXAMPLE: w = 0.10 -> PSI 0.035347; N = 13.8155/0.035347 = 390.9 orders
       390.9 / 900 = 0.4343 hours = 26.06 minutes
    w = 0.22 -> PSI 0.154833; N = 89.23 orders; 89.23/900 = 0.0991 hours = 5.95 minutes
    w = 0.02 -> PSI 0.001555; N = 8,884 orders; 8,884/900 = 9.87 hours
INTERPRETATION: at the realised corruption the statistic needed six minutes of traffic. The
       25-minute latency came from the two-window rule and the window boundary, not from
       statistical power. Fixing the rule is cheaper than improving the statistic.
```

Read against the harm function, the table produces a design rule. Detection latency for a corruption of share `w` is approximately `t_detect(w)`, and harm accrues at `30 * 32.4 * (0.375 - P(w))` per hour during that latency. For every share above the 3.33 percent saturation point, harm is 140.94 per hour and latency is under 6 minutes, so the expected harm from monitor latency alone is under 15 currency units — an order of magnitude less than the 25 units a single false page costs.

That inequality is the whole argument for a faster, noisier alert on this signal: the marginal detection value exceeds the marginal false-page cost by a factor of roughly two at the saturation point and grows rapidly above it. Below the saturation point the reverse holds, because harm falls off faster than latency grows. The correct policy is therefore size-indexed and share-indexed rather than uniform, which is exactly what a fixed PSI threshold cannot express.
