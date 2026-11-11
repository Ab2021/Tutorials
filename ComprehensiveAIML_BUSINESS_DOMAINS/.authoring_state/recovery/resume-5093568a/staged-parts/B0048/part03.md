## 5. Worked Numerical Example

This section carries four exercises on the same cohort, each answering a different question a
programme asks in sequence: what does the base rate do to precision (5.1), what does the currency
cutoff say to do (5.2), is the probability scale trustworthy (5.3), and what does the review team
actually have room for (5.4). A fifth exercise (5.5) estimates what the review queue never sees.
Every number is ILLUSTRATIVE and is re-derived inside the text so the reader can check the
arithmetic without the tool that produced the tables.

Inputs held fixed throughout, restated so they are not carried in the head:

```text
cohort N         = 4,000,000 events per 30-day cycle
prevalence pi    = 0.0012  -> positives 4,800, negatives 3,995,200
loss given fraud = 340.00 cur per confirmed fraud
preventable r    = 0.55    -> V = 187.00 cur per caught fraud
review cost      = 0.200 h x 41.00 cur/h + 1.30 cur = 8.20 + 1.30 = 9.50 cur
capacity         = 30 investigators x 160 h = 4,800 h -> 24,000 reviews
```

### 5.1 Base-rate exercise

Score the cohort into twelve equal alert blocks of 4,000, ordered by score, and count the confirmed
frauds in each block after the 90-day maturity horizon. The counts are the ILLUSTRATIVE input vector
`620, 505, 430, 372, 315, 268, 226, 188, 152, 121, 95, 72`, which sums to 3,364 — that is, 70.08
per cent of the 4,800 positives appear somewhere in the 48,000 highest-scoring events. The
remaining 1,436 positives are dispersed through the 3,952,000 events below block 12 and are not
observable without audit.

Step 1, block precision. For block j, precision_j = tp_j/4000. Block 1 gives 620/4,000 = 0.15500;
block 6 gives 268/4,000 = 0.06700; block 12 gives 72/4,000 = 0.01800. The vector falls by a factor
of 8.6 from first to last block, which is the ranking signal expressed as a rate.

Step 2, cumulative precision and cumulative net value. At capacity K = 24,000 (blocks 1–6) the
cumulative true positives are 620+505+430+372+315+268 = 2,510, so cumulative precision is
2,510/24,000 = 0.104583 and cumulative net value is 2,510 x 187.00 − 24,000 x 9.50 =
469,370 − 228,000 = 241,370 cur.

Step 3, the marginal block. Block 7 adds 226 positives for 4,000 reviews: net 226 x 187.00 −
38,000 = 42,262 − 38,000 = 4,262 cur, or 1.0655 cur per alert. Block 8 adds 188 x 187.00 − 38,000 =
35,156 − 38,000 = −2,844 cur. So the *economic* optimum is at the end of block 7, K* = 28,000, with
2,736 positives and net value 2,736 x 187.00 − 28,000 x 9.50 = 511,632 − 266,000 = 245,632 cur.

```text
FORMULA: marginal block value = tp_j*V - 4000*c
WHERE: tp_j positives in block j (events); V cur per positive; c cur per review
CONSTRAINT: blocks of equal size; block membership determined by rank, not by outcome
NUMERICAL EXAMPLE: block 7: 226*187.00 - 4000*9.50 = 42262 - 38000 = +4262 cur
                   block 8: 188*187.00 - 4000*9.50 = 35156 - 38000 = -2844 cur
INTERPRETATION: the sign change between blocks 7 and 8 locates the cutoff at p ~ 0.05
```

Independent re-check of the sign change: the cutoff from G-1 is 9.50/187.00 = 0.050802. Block 7's
precision is 0.05650, above it; block 8's is 0.04700, below it. The block arithmetic and the
formula agree, which is the only reason to trust either. The cutoff falls inside block 7, so a
finer tier within that block would capture a little more value; with only block-level data the
programme stops at 28,000 and accepts the small loss from the granularity.

Sensitivity to the loss assumption, holding everything else fixed: at L = 250 the cutoff rises to
9.50/137.50 = 0.069091 and blocks beyond the fifth (precision 0.07875) are uneconomic; at L = 480
the cutoff falls to 9.50/264.00 = 0.035985 and block 9 (0.03800) is still economic. The optimum
moves by two blocks — 8,000 reviews, a third of the team's month — for a factor-of-two change in a
loss input that is often estimated from a single historical quarter. That elasticity is the case
for measuring L properly rather than debating thresholds.

### 5.2 Cost-sensitive allocation exercise

The capacity is 24,000 but the economic optimum is 28,000. That 4,000-alert gap is a capacity
problem with a price, and the price is computable. Value at 24,000 is 241,370 cur; value at 28,000
is 245,632 cur; the gap is 4,262 cur per cycle, or 0.88 per cent of the value at optimum. That is
small enough that buying 1,000 extra investigator-hours (4,000 reviews at 0.20 h = 800 h, plus
queue slack) to close it is not justified at a loaded cost of 41 cur/h: 800 h costs 32,800 cur to
recover 4,262 cur.

The reverse question — what if capacity falls — is sharper. Cut the team from 30 to 25
investigators and capacity becomes 4,000 h / 0.200 = 20,000 reviews. Value falls to the block-5
cumulative: 2,242 positives, 2,242 x 187.00 − 20,000 x 9.50 = 419,254 − 190,000 = 229,254 cur. The
loss is 241,370 − 229,254 = 12,116 cur per cycle, so each lost investigator is worth about
2,423 cur per cycle at the margin — considerably more than the loaded cost of an investigator
(160 h x 41 = 6,560 cur), which means the team is *under*-staffed at this loss level and the
programme should be arguing for more capacity, not defending the current one. The break-even
condition is that an extra investigator must generate at least 6,560 cur, i.e. at least
6,560/187.00 = 35.08 additional caught frauds, i.e. at least 35.08/0.0565 = 621 reviews' worth of
block-7-grade alerts. Blocks 7 and 8 supply 4,000 and 4,000 more alerts at precisions 0.0565 and
0.0470, so the 31st and 32nd investigators clear that bar and the 33rd (block 9, precision 0.0380,
which yields 4,000 x 0.0380 x 187.00 − 4,000 x 9.50 = −9,576 cur) does not. The optimal headcount
at these inputs is 32.

```text
FORMULA: investigator break-even = (loaded cost per investigator) / (V * precision at margin)
WHERE: cost 6,560 cur per 160-h investigator; precision at margin is the block precision
CONSTRAINT: marginal block fully absorbed; no queueing delay; reviews independent
NUMERICAL EXAMPLE: 6560/(187.00*0.0565) = 6560/10.5655 = 620.9 reviews -> 0.31 investigators
                   i.e. one investigator (8,000 reviews/cycle capability at 40 reviews each)
INTERPRETATION: blocks 7-8 justify two more investigators; block 9 justifies none
```

The unit economics in per-caught terms make the same point in a form finance accepts: at 24,000
reviews the programme spends 24,000 x 9.50 / 2,510 = 228,000/2,510 = 90.84 cur per caught fraud
against 187.00 cur of prevented exposure, a coverage ratio of 2.06. At 28,000 the cost per caught
rises to 266,000/2,736 = 97.22 cur, coverage 1.92. At 32,000 it rises to 304,000/2,924 = 103.97 cur,
coverage 1.80, still above one — which is why the capacity question and the cutoff question have
different answers at the same loss level.

### 5.3 Calibrated-bin exercise

Take 250,000 scored events with a mean predicted probability of 0.0012 and eight populated
reliability bins with (mean predicted per 1,000, observed per 1,000) of (3.75, 4), (8.75, 9),
(14.6875, 15), (22.5, 24), (32.8125, 31), (49.375, 50), (76.5625, 78), (91.5625, 89). Bin widths are
chosen so each bin holds 31,250 events; with a mean rate of 0.0012 that is about 37.5 expected
positives per bin, the minimum for a bin to carry any information.

Expected positives total 3.75+8.75+14.6875+22.5+32.8125+49.375+76.5625+91.5625 = 300.0 per
1,000 x 31.25, i.e. 300 positives in 250,000 events. Observed positives total 4+9+15+24+31+50+78+89
= 300. The marginal totals agree exactly, which is what a calibrated model must produce on its own
calibration sample.

Per-bin ratios (observed/expected): 1.0667, 1.0286, 1.0213, 1.0667, 0.9448, 1.0127, 1.0188, 0.9720.
Mean absolute deviation is 0.0625+0.25+0.3125+1.5+1.8125+0.625+1.4375+2.5625 = 8.75 events, so the
expected calibration error is 8.75/250,000 = 0.00003500, which is 2.92 per cent of the base rate.
A Pearson-type statistic gives sum (o-e)^2/e = 0.3372 on 7 degrees of freedom, far inside its
expectation of 7 — the bins are *too* close, which itself is a smell: at 37.5 expected positives a
chi-square of 0.34 has probability around 0.006 and suggests the bins were fitted rather than
evaluated on held-out data.

The single most useful diagnostic is the reliability slope, obtained by regressing observed on
expected across bins: slope 0.98474, intercept 0.57212 per 1,000. A slope below one means the model
is over-dispersed — it spreads predictions wider than reality — and a positive intercept means it
is slightly over-predicting in the lowest bin where 3.75 was predicted against 4 observed.

```text
FORMULA: reliability slope = cov(observed, expected)/var(expected) over populated bins
WHERE: observed and expected in events per 1,000 per bin; unweighted least squares across bins
CONSTRAINT: >=8 populated bins; bin edges fixed before evaluation; no refitting on the same data
NUMERICAL EXAMPLE: mean expected 37.5, mean observed 37.5; cov term sums to 3,187.5,
                   var term 3,237.0 -> slope 3187.5/3237.0 = 0.98474
INTERPRETATION: predictions are slightly too spread; multiply deviations from the mean by 0.985
```

Translating calibration into money: at the operating point the *rate* form and the *count* form of
precision agree to six decimals (0.104583 both ways, section 4.1 B-2), which means the bin-level
calibration error of 2.92 per cent does not move the cutoff materially. At precision 0.104583 the
cutoff in probability terms is 0.050802, and a 2.92 per cent relative error in the probability
scale moves the effective cutoff to somewhere between 0.0493 and 0.0523 — worth at most a few
hundred reviews out of 24,000. Calibration errors bite when they are *local*, at the cutoff,
rather than global; a model can pass a global ECE test and still be 40 per cent off in the one bin
where the decision is made. That is why the operating-bin check in section 9 exists.

### 5.4 Capacity and time-scale exercise

Capacity is 24,000 reviews per 30-day cycle, which is 24,000/30 = 800 reviews per day and
24,000/4.345 = 5,523 per average week (using 4.345 weeks per month). But investigators do not work
weekends, and fraud does not respect the calendar: the weekly capacity over five working days is
5,523/5 = 1,105 per working day, while Friday-to-Monday volume typically carries 45 per cent of the
week's transactions, i.e. 0.45 x 4,000,000/4.345 = 414,000 transactions in four days against a
five-day alert budget. The queue, not the month, is the binding constraint.

Convert the monthly plan into a shift plan. Thirty investigators x 160 productive hours = 4,800
hours, spread over 21 working days gives 228.57 investigator-hours per working day, which is 7.62
hours per investigator per day. That is a 7.6-hour workload inside an 8-hour shift: 4.7 per cent
slack, consumed entirely by training, huddles, and queue recovery. There is no reserve.
There is no reserve. Any 8 per cent surge in alert volume produces 24,000 x 1.08 = 25,920 alerts
against 24,000 of capacity: 1,920 alerts, 384 investor-hours, 8.0 per cent of the month.

The surge has two handling policies and they cost differently. Policy A, first-in-first-out, keeps
the alert mix of the whole month (precision 0.104583) but reviews the *oldest* 24,000 alerts, which
are disproportionately from the start of the month and therefore older than the ideal. Value
forgone by deferring 1,920 alerts at the average precision is 1,920 x 0.104583 x 187.00 = 37,550 cur,
against cost saved of 1,920 x 9.50 = 18,240 cur, a net loss of 19,310 cur. Policy B, score-ordered
deferral, drops the 1,920 lowest-scoring alerts, whose precision is the tail precision of about
0.018, so value forgone is 1,920 x 0.018 x 187.00 = 6,463 cur against 18,240 cur saved, a net gain
of 11,777 cur. The difference between the policies on the same surge is 31,087 cur, which is more
than the monthly contribution of three investigators. Deferral order is a policy decision with a
price and it is almost always left to the queue software's default.

```text
FORMULA: surge cost = deferred_alerts * (precision_deferred * V - c)
WHERE: deferred_alerts is alerts beyond capacity; precision_deferred is the precision of the
       specific alerts chosen for deferral, not the average
CONSTRAINT: deferred alerts are worked within the next cycle or written off, not silently aged
NUMERICAL EXAMPLE: score-ordered deferral: 1920*(0.018*187.00 - 9.50) = 1920*(3.366-9.50)
                   = 1920*(-6.134) = -11777 cur  (a gain, since cost is avoided)
                   FIFO deferral: 1920*(0.104583*187.00 - 9.50) = 1920*(19.557-9.50)
                   = 1920*(10.057) = +19310 cur  (a loss, value dropped exceeds cost saved)
INTERPRETATION: dropping the tail saves money; dropping the head destroys it
```

Note the sign convention carefully: for a deferred alert the programme avoids the review cost c
and forgoes the expected benefit precision x V, so the net change is c − precision x V. When
precision is high (FIFO defers good alerts) the net is negative, i.e. a loss of 19,310 cur; when
precision is low (score-ordered defers junk) the net is positive, i.e. a gain of 11,777 cur.

The staffing arithmetic also yields the value of a triage stage, which is the only structural fix
for a team with no slack. With a 4-minute triage at 41 cur/h = 2.7333 cur per alert passing a
fraction f = 0.4 to full review, effective hours per alert are 0.0667 + 0.4 x 0.200 = 0.14667, so
capacity rises from 24,000 to 4,800/0.14667 = 32,727 reviews. True positives at that capacity:
the cumulative count at 32,000 is 2,924, and the remaining 727 alerts sit at block-9 precision
0.0380, giving 2,924 + 727 x 0.0380 = 2,924 + 27.63 = 2,951.6. Net value is
2,951.6 x 187.00 − 32,727 x 6.5333 = 551,949 − 213,816 = 338,133 cur, a 40 per cent improvement on
241,370 cur. The break-even pass rate computed in 4.4 Limit 8 is f = 0.7026; the observed loop
behaviour (f = 0.7 gives net 241,725 cur) confirms that triage stops paying around there.

### 5.5 Audit exercise: what the queue never sees

The reviewed 24,000 events yield 2,510 confirmed frauds. The programme's exposure is the
3,976,000 events it did not review. Sample 40,000 of them uniformly at random, wait out the
maturity horizon, and find 19 confirmed frauds. Inclusion probability is q = 40,000/3,976,000 =
0.010060, so the estimated unreviewed fraud count is 19/0.010060 = 1,888.7 events.

The denominator matters and is easy to get wrong. The audit samples from unreviewed events,
which number 4,000,000 − 24,000 = 3,976,000, not from the whole cohort. Using 40,000/3,976,000
gives q = 0.0100604 and 19/q = 1,888.6. Using the cohort denominator 40,000/4,000,000 = 0.0100000
gives 1,900.0. The two differ by 11 events, which is smaller than the standard error of about 433
and therefore immaterial to the estimate — but the *principle* is not immaterial, because a
programme that audits from the wrong frame can be off by much more than 11 when the reviewed
fraction is large.

```text
FORMULA: T_hat = observed_audit_positives / q,  q = n_audit / n_unreviewed
WHERE: n_audit sampled events; n_unreviewed = N - K; observations matured
CONSTRAINT: uniform or known-probability sampling; q>0 in every stratum; labels final
NUMERICAL EXAMPLE: q = 40000/3976000 = 0.0100604; T_hat = 19/0.0100604 = 1888.6
                   se = sqrt(19*(1-q))/q = sqrt(18.809)/0.0100604 = 4.337/0.0100604 = 431.1
                   95 per cent interval: 1888.6 +/- 1.96*431.1 = 1043.6 to 2733.6
INTERPRETATION: total fraud is 2510 + 1889 = 4399, so recall is 2510/4399 = 0.5706
```

An alternative estimator that uses the reviewed count as known: the programme knows TP = 2,510
exactly and estimates the remainder, so total = 4,399 with the whole uncertainty carried by the
audit term. Recall is 0.5706 with a 95 per cent interval from 2510/(2510+2733.6) = 0.4787 to
2510/(2510+1043.6) = 0.7063. The interval is 23 points wide on a 40,000-event audit — that is the
price of estimating a rare event without stratifying. Stratifying the audit by score decile would
cut the interval by roughly the ratio of between-stratum to total variance, which for a score-
ordered population is often a factor of three or four; the practical recommendation is to sample
proportionally within decile rather than uniformly, and to oversample the top deciles where the
fraud density and hence the information per event are higher.

Independent check of the interval by a different route: a Poisson approximation to 19 observed
events gives a 95 per cent interval on the count of about 11.4 to 29.7 (using the Garwood form),
which scaled by 1/q gives 1,133 to 2,952 — wider than the binomial-based interval but consistent
with it, as it should be since the counts are small and the Poisson approximation is conservative.
Reporting only the point estimate 1,889 would understate the uncertainty by a factor of six in
width terms.

**Takeaway from section 5.** The same cohort produces five different "correct" cutoffs depending on
which question is asked: 28,000 by unconstrained value, 24,000 by hard capacity, 32,727 by triage
economics, 32 investigators by staffing break-even, and 0.050802 by probability threshold. Only the
last is a number a model can output; the rest are numbers an organisation must decide.
