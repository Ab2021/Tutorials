## 5. Worked Numerical Example

### 5.1 The scenario, stated as a contract

A checkout team proposes replacing a three-step address form with a single-page form. The decision owner is the checkout product lead; the deadline is the next release train, 22 calendar days out. The pre-registration, committed before any exposure, contains the following.

| Field | Value | Why this value |
|---|---|---|
| Decision | Ship the single-page form to 100% of mobile web sessions, or keep the current form | Binary, reversible, deadline-bound |
| Estimand | Difference in session-level purchase rate, new minus current, over the 14 days after first exposure | The metric the release decision is actually about |
| Randomization unit | Session | The change is rendered per session; a session-level split is technically feasible and is what the team already operates |
| Analysis unit | Session | Matches the randomization unit, so the design effect is 1 |
| Population | Eligible mobile web sessions with a non-empty cart | Pre-treatment filter; excludes sessions that cannot check out |
| Primary metric | Purchases per eligible session, deduplicated on order ID | Prevents double counting a retried confirmation |
| Baseline | $p_C=0.120$ from the trailing 28 days, same population | Historical, same window length |
| Effect of interest | $\delta=0.012$ absolute (10% relative) | Smallest lift whose monthly margin exceeds build and run cost |
| $\alpha$ | 0.05 two-sided | Team convention; guardrail tests get their own budget |
| Power | 0.80 | Team convention |
| Guardrails | Crash rate per session, refund rate per purchase, p95 time-to-confirmation | Each with a tolerance in Section 5.6 and Section 8 |
| Stopping rule | Fixed horizon at 13,300 sessions per arm, no interim analysis | Chosen to keep the analysis simple; Section 5.5 quantifies what peeking would have cost |

Two design decisions deserve explicit defence. Analysing at session level matches randomization, so no design effect is applied. But the estimand is then a *session-weighted* effect, and heavy users contribute more sessions than light users; a user-weighted effect could differ. That limitation is recorded in Section 8 rather than repaired by post-hoc reweighting, because reweighting after seeing the result would introduce its own selection.

### 5.2 Sample size, computed twice and explained

Using [E-02] with the *planning* variance $\sigma^2=p_C(1-p_C)=0.12\times0.88=0.1056$ in both arms:

- variance sum: $0.1056+0.1056=0.2112$
- quantile sum: $z_{0.975}+z_{0.80}=1.95996+0.84162=2.80158$
- squared: $2.80158^2=7.84887$
- numerator: $7.84887\times0.2112=1.657681$
- divide by $\delta^2=0.000144$: $1.657681/0.000144=11511.68$

So the design needs **11,512 sessions per arm**, or 23,024 total. At forecast traffic of 1,900 eligible sessions per day, a 14-day window delivers $1900\times14=26{,}600$, i.e. 13,300 per arm — 15.5% headroom over the requirement, which is prudent because $p_C$ can drift between the planning period and the run.

Now recompute with the *realized* variance. The treated arm lands at 0.132030, so its variance is $0.132030\times0.867970=0.114596$; the variance sum becomes $0.1056+0.114596=0.220196$. Substituting gives $7.84887\times0.220196/0.000144=12001.1$, so 12,001 per arm would have delivered 80% power against the effect actually seen. The difference between 11,512 and 12,001 is not an error; it is the difference between planning with the control variance you have and computing with the treated variance you only learn later. Pre-launch sizing must use the control variance, and the 13,300 actually collected happened to cover both.

Three independent checks:

1. **Dimension check.** $\delta^2$ has units of rate$^2$; the numerator $7.84887\times$variance-sum also has rate$^2$; the quotient is a count. Correct.
2. **Order-of-magnitude check.** At $n=13{,}300$, $\mathrm{SE}=\sqrt{0.220196/13300}=0.004069$, and $2.80158\times0.004069=0.011399$, close to the 0.012 target. Correct.
3. **Reverse check.** Substituting 11,512 into the MDE expression with the planning variance returns $2.80158\times\sqrt{0.2112/11512}=2.80158\times0.0042827=0.011998$. Correct to four significant figures.

### 5.3 Assignment audit before any outcome is read

The split is 50/50 on a hashed session identifier with a salt committed to the repository. Two checks run on day one, before any purchase data is joined.

**Count check.** Realized assignments: 13,306 treatment, 13,294 control, total 26,600. Under the design each arm has expected count 13,300. The chi-square statistic is
$$\chi^2=\frac{(13306-13300)^2}{13300}+\frac{(13294-13300)^2}{13300}=\frac{36}{13300}+\frac{36}{13300}=0.005414.$$
With one degree of freedom and a null $\chi^2_1$ distribution, $p\approx0.942$. No mismatch. The statistic is tiny because each deviation is 6 units against an expected 13,300, a relative deviation of 0.045%.

**Sensitivity of the check.** The check catches material breakage but not microscopic drift. At 26,600 units, the 0.05 critical value 3.841 corresponds to a split near 13,351/13,249 — about 50.19/49.81. So the test flags departures of roughly 0.2 percentage points or more at this sample size. For calibration:

| Split | $\chi^2$ | $p$ | Verdict |
|---|---:|---:|---|
| 13,306 / 13,294 | 0.0054 | 0.942 | clean |
| 13,120 / 12,880 | 2.215 | 0.137 | inside noise |
| 13,234 / 12,766 | 8.424 | 0.0037 | suspicious; audit |
| 13,300 / 12,700 | 13.846 | 0.0002 | broken; void the test |

The last row is only a 2.3% imbalance in raw counts. That is why the check must be automated with a threshold: human eyes cannot distinguish 0.2% from 2% on a dashboard, and the chi-square statistic separates them by a factor of 60.

**Exposure check.** The treatment-rendered log records 13,241 of the 13,306 assigned sessions as having rendered the single-page form, a 99.51% render rate. The 65 missing sessions remain in the ITT denominator. Dropping them would compare the new form only among sessions where it loaded against a control that includes sessions where it would have failed to load, which is a different and more flattering estimand than the one registered.

### 5.4 The primary readout, arithmetic shown

Fourteen days after the last session entered, the frozen analysis script produces:

| Quantity | Treatment | Control |
|---|---:|---:|
| Eligible sessions | 13,300 | 13,300 |
| Purchases (deduplicated on order ID) | 1,756 | 1,596 |
| Purchase rate | 0.1320301 | 0.1200000 |
| Variance $p(1-p)$ | 0.114596 | 0.105600 |

**Step 1 — the difference.** $\hat\tau=0.1320301-0.1200000=0.0120301$, i.e. +1.203 percentage points.

**Step 2 — the pooled null rate.** $\bar p=3352/26600=0.1260150$.

**Step 3 — the null standard error from [E-03].** First the product $\bar p(1-\bar p)=0.1260150\times0.8739850=0.110136$. Then the reciprocal-sum $1/13300+1/13300=2/13300=1.503759\times10^{-4}$. Multiply: $0.110136\times1.503759\times10^{-4}=1.65623\times10^{-5}$. Square root: $\mathrm{SE}_0=0.0040697$.

**Step 4 — the test statistic.** $z=0.0120301/0.0040697=2.9560$.

**Step 5 — the two-sided p-value.** $2(1-\Phi(2.9560))=2\times0.001558=0.00312$.

**Step 6 — the confidence interval, unpooled.** The treated variance term is $0.114596/13300=8.6162\times10^{-6}$; the control term is $0.1056/13300=7.9398\times10^{-6}$; the sum is $1.65560\times10^{-5}$; the square root is $\mathrm{SE}_1=0.0040689$. Margin: $1.95996\times0.0040689=0.0079745$. Interval: $[0.0120301-0.0079745,\ 0.0120301+0.0079745]=[0.0040556,\ 0.0200046]$.

**Step 7 — the relative interval.** Dividing the absolute bounds by $p_C=0.12$: $[0.033797,\ 0.166705]$, i.e. **+3.4% to +16.7% relative**.

**Step 8 — realized power.** Under the alternative with the achieved variance sum, $z_{\text{alt}}=\delta/\mathrm{SE}_0=2.9560$, so power $=\Phi(2.9560-1.95996)+\Phi(-2.9560-1.95996)=\Phi(0.9960)+6\times10^{-8}=0.8404+0.0000=0.840$. Realized power is 84%, marginally above target because traffic overshot by 15.5%.

**Step 9 — decision.** The interval lies entirely above zero, so the effect is not neutral, and its lower bound of +0.41pp clears the pre-registered economics (Section 5.9). The readout state is "decisive win on primary, pending guardrails."

**What a different interval would have meant.** Had the interval been $[-0.002,\ 0.016]$, the correct readout would have been state 3 — undersized inconclusive — not "no effect." The distinction matters for the next action: state 2 means stop; state 3 means the instrument was inadequate and the design, not the product, needs attention.

### 5.5 What peeking would have done

Traffic arrived evenly, so the planned 26,600 total corresponds to information fractions 0.25, 0.50, 0.75, and 1.00 at roughly days 3.5, 7, 10.5, and 14. Suppose the team had looked at each and stopped at the first $p<0.05$.

**At n = 6,650 per arm** (a quarter of the traffic) the same rates give
$$\mathrm{SE}=\sqrt{0.110136\times\tfrac{2}{6650}}=\sqrt{3.31243\times10^{-5}}=0.0057554,$$
so $z=0.0120301/0.0057554=2.0903$ and $p=0.0366$. The test "passes" at the first look, and the team ships at day 3.5 with an estimate whose 95% interval is $[0.00075,\ 0.0233]$ — a range from "barely worth the build" to "nearly twice the planning assumption."

**At n = 9,500 per arm** the same exercise gives $\mathrm{SE}=\sqrt{0.110136\times2/9500}=0.0048146$, $z=2.4987$, $p=0.0125$.

**At n = 13,300** it gives $z=2.9560$, $p=0.00312$.

Both early looks would have crossed a naive 0.05 line. That is the mechanical reason peeking inflates error: the null path wanders, and an early excursion is common. The simulation in Section 4.3.7 quantifies the cost — five naive looks at $c=1.96$ produce a family-wise error rate of 0.143 instead of 0.050, a 2.9× inflation. Against that, the team would have saved 10.5 days. Sometimes that trade is correct, but it must be priced in advance: a five-look plan with an honest constant boundary near $c=2.41$ would *not* have rejected at $n=6{,}650$ ($z=2.09<2.41$), which is precisely the protection being purchased.

The secondary lesson is about the interval, not the test. At day 3.5 the interval half-width was 0.0113, wider than the entire planning effect $\delta=0.012$. Any decision taken at that point is a decision taken with an instrument that cannot yet resolve the question it was built for.

### 5.6 The guardrail that nearly inverted the decision

The crash guardrail counts sessions where the checkout page raised an unhandled error. Counts: 56 in treatment (rate 0.0042105), 41 in control (rate 0.0030827). The absolute difference is $+0.0011278$, i.e. +1.13 crashes per thousand sessions, a **+36.6% relative increase**.

**Pooled null standard error.** $\bar p_g=97/26600=0.0036466$; $\bar p_g(1-\bar p_g)=0.0036333$; times $2/13300=1.503759\times10^{-4}$ gives $5.4635\times10^{-7}$; square root $0.0007392$.

**Test.** $z_g=0.0011278/0.0007392=1.526$; two-sided $p=0.127$; one-sided $p=0.0635$.

**Unpooled interval.** The arm variances are $0.0041932$ and $0.0030732$; over 13,300 each they sum to $5.4632\times10^{-7}$; $\mathrm{SE}_1=0.0007391$. Two-sided 95% interval: $[0.0011278-0.0014487,\ 0.0011278+0.0014487]=[-0.000321,\ 0.002577]$, which crosses zero.

**One-sided upper bound.** $0.0011278+1.644854\times0.0007391=0.0011278+0.0012158=0.0023436$. Relative to the control rate: $0.0023436/0.0030827=0.7602$, i.e. **up to +76.0%**.

**The tolerance decided the outcome.** The guardrail was registered with the rule: "breach if the one-sided 95% upper bound on the relative increase exceeds 50%." The bound is 76%, so the guardrail **breached**, even though the point estimate was not statistically distinguishable from zero. Under the pre-registration the experiment therefore could not be shipped to 100% without an engineering fix.

Three counterfactual guardrail definitions, same data:

| Rule | Result | Consequence |
|---|---|---|
| "$p<0.05$ two-sided on crash rate" | Fails to reject ($p=0.127$) → no harm found | Ships a change that may raise crashes up to 76% |
| "Any numeric increase in crash count" | 56 > 41 → breach | Stops on noise; this guardrail would fire on roughly half of all harmless changes |
| "One-sided 95% upper bound of relative increase > 50%" | 76% > 50% → breach | Correctly converts an imprecise harm estimate into a bounded decision |

The middle rule is the one teams adopt informally, and it has a 50% false-alarm rate by construction — a guardrail that fires on noise teaches everyone to ignore guardrails.

**Power of that guardrail, stated honestly.** At 13,300 per arm and $p\approx0.0031$, the 80%-power MDE is
$$\mathrm{MDE}_g=2.80158\times\sqrt{2\times0.0030827\times0.9969173/13300}=2.80158\times0.0006802=0.001906,$$
which is a **61.8% relative increase**. The observed +36.6% was, a priori, undetectable. So the guardrail's veto came entirely from the bound-based tolerance, and the team must accept that such a rule fires often on imprecision. A powered guardrail against a 25% relative increase would need
$$n=\frac{7.84887\times2\times0.0030827\times0.9969173}{(0.25\times0.0030827)^2}=\frac{0.048242}{5.9394\times10^{-7}}=81{,}224$$
per arm — 6.1× the traffic, or 86 days at 1,900 per day. Guardrail power and calendar time are in direct conflict, and the conflict must be named in the pre-registration rather than discovered at readout.

### 5.7 The novelty check as an interaction test

Split the 14 days into the first 3 days (2,850 sessions per arm) and the remaining 11 days (10,450 per arm). Observed counts:

| Window | Treatment | Control | Treatment rate | Control rate | Difference |
|---|---:|---:|---:|---:|---:|
| Days 1–3 | 405 | 353 | 0.142105 | 0.123860 | +0.018246 |
| Days 4–14 | 1,351 | 1,243 | 0.129282 | 0.118947 | +0.010335 |

The early effect is 1.82pp and the late effect is 1.03pp, a visible decline. The question is whether the decline is distinguishable from sampling noise. Compute the difference-of-differences from [E-01] applied twice:
$$\hat\Delta=0.018246-0.010335=0.007911.$$
$$\mathrm{SE}(\hat\Delta)=\sqrt{\frac{0.1421\times0.8579}{2850}+\frac{0.1239\times0.8761}{2850}+\frac{0.1293\times0.8707}{10450}+\frac{0.1189\times0.8811}{10450}}.$$
The four terms are $4.2766\times10^{-5}$, $3.8075\times10^{-5}$, $1.0771\times10^{-5}$, and $1.0029\times10^{-5}$; their sum is $1.01641\times10^{-4}$; the square root is $0.010082$.

$$z=0.007911/0.010082=0.7846,\quad p=0.433.$$

The interaction is not significant. Worse, the design has poor power to detect one: the 80%-power MDE for the interaction is $2.80158\times0.010082=0.02825$, nearly eight times the observed decline. So the honest statement is: "the effect is smaller in the later window, but this design cannot distinguish a genuine 0.8pp novelty decay from zero; a dedicated duration test with a longer horizon and a pre-declared interaction hypothesis is required."

This is the cleanest available example of a claim that looks like a finding and is not one. The decline is real in these numbers and consistent with novelty, but also with a slow rollout of the winning variant, with weekday composition differences between days 1–3 and 4–14, and with a mid-test marketing change the team forgot to log. Only one of those four is a novelty effect, and the data cannot separate them.

### 5.8 The revenue metric the team also wanted

The stakeholder asked for revenue per eligible session as a secondary metric. Measured: 59,200 currency units in control over 13,300 sessions, i.e. **4.4511 per session**; 60,600 in treatment, i.e. **4.5564 per session**. Difference: **+0.1053 per session**, or +2.37% relative.

This metric is far noisier per unit than the rate. Revenue per session is heavy-tailed — most sessions spend nothing, a few spend a lot. Taking an illustrative per-session standard deviation of 18 units, the standard error of the difference at 13,300 per arm is
$$\mathrm{SE}=\sqrt{\frac{18^2}{13300}+\frac{18^2}{13300}}=\sqrt{4.8722\times10^{-2}}=0.2207,$$
so $z=0.1053/0.2207=0.477$ and $p=0.634$. The 80%-power MDE is $2.80158\times0.2207=0.6184$ per session, which is **13.9% of the control mean**. To detect a 10% lift — $\delta=0.4451$ — the requirement is
$$n=\frac{7.84887\times2\times324}{0.4451^2}=\frac{5086.07}{0.198114}=25{,}672$$
per arm, essentially double the primary metric's requirement. Even a generous CUPED adjustment at $\rho=0.55$ brings it to $25{,}672\times0.6975=17{,}906$ per arm, still 35% above what was collected.

The general lesson: **do not add a noisy secondary metric to a test sized for a rate metric and then interpret its sign.** Either pre-register it as a directional diagnostic with no decision authority, or run a separate, longer test. The reported "revenue also trended up (+2.4%)" from a design with 52% of the required sample is the multiplicity problem from Section 4.3.8 wearing a business costume — and here the direction happens to agree with the primary metric, which makes it feel like corroboration while adding no evidence.

### 5.9 The final decision record

| Item | Recorded value |
|---|---|
| Readout state | Decisive win on primary; guardrail breach on crash bound |
| Primary estimate | +1.203pp, 95% CI [0.406pp, 2.000pp], $z=2.956$, $p=0.0031$ |
| Relative estimate | +10.03%, 95% CI [+3.38%, +16.67%] |
| Guardrail | Crash bound +76% against a 50% tolerance → breach |
| Power realized | 0.840 on primary; 0.618-relative MDE on the guardrail (underpowered by design) |
| Interaction check | Duration decline $p=0.433$; not interpretable as novelty |
| Decision | Ship at 25% with a kill switch; harden the validation path; re-read the crash rate after 7 days |
| Follow-up | Duration test at 50/50 for 28 days to separate novelty from durable effect |
| Recorded limitations | Session-weighted estimand; 14-day horizon; guardrail tolerance is bound-based and will fire on imprecision; revenue metric underpowered |

The decision was neither "ship" nor "kill." It was a conditional shipment with a named repair, a re-read date, and an acknowledged power limitation. The value of the record is that a reader six months later can reconstruct *why* 25% rather than 100%, and what would have to change for the rest.

> **Takeaway:** in a real readout the primary test and the guardrail test can point in opposite directions, and the pre-registered tolerance — not the p-value — is what resolves the conflict.

## 6. Visual / Geometric Intuition

### 6.1 Two sampling distributions and the two error regions

Picture two bell curves on one axis, both with standard error $\mathrm{SE}$: one centred at 0 (the null) and one centred at the true effect $\delta$. The decision rule draws vertical critical lines at $\pm z_{1-\alpha/2}\mathrm{SE}$ around zero.

```text
                      decision axis (effect estimate)
        |                        |                        |
   -z*SE                       0.0                     +z*SE
        \______________________/  \_____________________/
         reject (negative)          reject (positive)
              ^                          ^
              |  alpha/2 = 0.025         |  alpha/2 = 0.025

   null curve:            /\                  <- area outside lines = alpha = 0.05
                         /  \
                        /    \
   alt curve:                    /\           <- shifted right by delta
                                /  \
                               /    \
                      area right of +z*SE = power = 1-beta
```

Three geometric facts follow, and each is a common interview answer.

**Fact 1 — the two areas trade off along one axis.** Shrinking $\alpha$ moves the critical lines outward, which moves the alternative curve's mass inside them, lowering power. No setting improves both without changing $\mathrm{SE}$, which means changing the sample size or reducing variance.

**Fact 2 — power is the area beyond the line, not the distance between centres.** With $\delta/\mathrm{SE}=2.80$ and $z_{1-\alpha/2}=1.96$, power is $\Phi(2.80-1.96)=\Phi(0.84)=0.80$. The quantity that matters is $\delta/\mathrm{SE}-z$, so an experiment with $\delta/\mathrm{SE}=2.0$ is badly underpowered (power 0.52), not "almost enough."

**Fact 3 — the interval is the same picture viewed backwards.** The confidence interval is the set of null values that would not be rejected. A 95% interval excluding zero is exactly a test rejecting at 0.05. This is why reporting a p-value from a pooled-variance test and an interval from an unpooled one is an inconsistency: they describe different curves.

### 6.2 Why the sample-size curve is a square-root curve

Power against $n$ follows $\Phi(\delta\sqrt n/\sigma\sqrt2-z_{1-\alpha/2})$. Since the argument grows with $\sqrt n$, traffic and power are not linearly related. At 50% power, $z_{1-\beta}=0$ and the requirement is $n_{50}=z_\alpha^2\cdot2\sigma^2/\delta^2$; at 80% power it is $n_{80}=(z_\alpha+0.8416)^2\cdot2\sigma^2/\delta^2$. The ratio is
$$\left(\frac{2.80158}{1.95996}\right)^2=2.043.$$
Buying the second 30 points of power (from 0.50 to 0.80) costs about twice the traffic of the first 50 points. Extending the same logic to 95% power, $(1.95996+1.64485)^2/(1.95996)^2=3.39$ — 3.4× the 50%-power traffic. This shape is why 80% is a convention: it sits near the knee.

### 6.3 The interval-width funnel

Interval width shrinks as $1/\sqrt n$. In the worked setting, using $\bar p(1-\bar p)=0.110136$ throughout:

| Sessions per arm | $\mathrm{SE}$ | 95% half-width | 80%-power MDE |
|---:|---:|---:|---:|
| 6,650 | 0.005755 | 0.011280 | 0.016124 |
| 9,500 | 0.004815 | 0.009437 | 0.013491 |
| 13,300 | 0.004070 | 0.007977 | 0.011403 |
| 20,000 | 0.003319 | 0.006505 | 0.009298 |

Going from 13,300 to 20,000 per arm is 50% more traffic for 18% narrower intervals. A team that "adds a week" to rescue a marginal result usually buys much less precision than it expects, which is why the pre-registered extension rule must be a doubling of the target rather than a vague top-up.

### 6.4 The SRM picture: what a real imbalance looks like

The $\chi^2_1$ distribution is heavily right-skewed: 95% of its mass sits below 3.841, and the density falls off exponentially beyond that. The Section 5.3 cases place four splits on that axis:

```text
 chi-square_1 density
   |
   |###                               3.841 = 0.05 critical value
   |####                             |
   |#####                            |
   |######                           |
   |#######                          |
   |########                         |
   |#########                        |
   |##########                       |
   +--+--+--+--+--+--+--+--+--+--+--+--+------> statistic
      0.005 2.215    8.424      13.846
        ^     ^         ^          ^
      clean  noise   audit      broken
```

The visual point is that 0.0% and 2.3% raw imbalance are near neighbours in count space but separated by a factor of 2,500 in statistic space, because the statistic is quadratic in the deviation while the tail probability is exponential in the statistic. Human scanning cannot make that discrimination; a threshold can.

### 6.5 The peeking picture: a random walk with a boundary

Under the null, standardized partial sums over information fractions behave like a random walk with independent increments. A fixed-horizon test inspects the walk's position at $t=1$. A peeking process asks whether the walk *ever* crossed a boundary, which is a first-passage problem, and first-passage probabilities greatly exceed endpoint probabilities.

```text
   Z
 3 |                    *  <- endpoint check only: |Z| = 1.2, no rejection
   |            *   *
 2 |- - - - - - - - - - - - -  boundary +/- 1.96
   |     *  /  \  *
 1 |    /  *    \/  *
   |  /              \
 0 *------------------*--------> information fraction t
   0    .25   .5   .75   1.0
        ^
        | touch of about 2.1 at t = 0.4, then return
        naive peeking rejects; fixed-horizon test does not
```

The simulation quantified this: five looks at 1.96 yield a family-wise error of 0.143; ten looks yield 0.195. To restore 0.05 the boundary must move to about 2.41 for five equal looks. The path drawn above is exactly the pathology — an excursion that a fixed-horizon test would never see becomes a press release if someone was watching.

### 6.6 Guardrail geometry: the tolerance is a line on the interval scale

The crash guardrail is clearest as an interval on a relative scale with a vertical tolerance line.

```text
 relative change in crash rate
   -50%        0%       +50%       +100%
     |          |         |          |
     |          |=========|          |   95% interval: [-10.4%, +76.0%]
     |          |    ^    |          |
     |          |   +36.6%|          |   point estimate
     |          |         |          |
     |          |   tolerance 50%   |
```

The point estimate (+36.6%) sits inside the tolerance; the interval's upper bound (+76.0%) does not. This is the whole argument for bound-based guardrails: the decision is about the true harm, and an interval reaching 76% says the true harm could be 76%. A point-estimate rule would have shipped.

### 6.7 The design-effect picture: variance moves to the cluster level

With $m$ observations per cluster, the variance of a cluster mean is $\sigma^2[1+(m-1)\rho]/m$. As $m$ grows, the $1/m$ factor stops helping once $(m-1)\rho$ dominates the 1, so the curve flattens at a floor set by $\rho$.

```text
 Var of cluster mean
   |
 s2|*
   | *
   |  *
   |   *____
   |        *******________________  <- floor: rho * s2
   |
   +--------------------------------> m (observations per cluster)
       1   5   10   20   50  100
```

At $\rho=0.02$ the design effect reaches 1.98 at $m=50$ and 16.98 at $m=800$. Adding a ninth hundred sessions to a user cluster is nearly worthless; adding users is what buys precision. That single picture explains a large family of underpowered marketplace experiments.

> **Takeaway:** three pictures — two shifted curves, a first-passage walk, and an interval against a tolerance line — explain power, peeking, and guardrail breach without any probability algebra.

## 7. Comparison / Selection Table

### 7.1 Choosing the randomization unit

| Unit | Use when | Precision cost | Failure it invites | Fallback if wrong |
|---|---|---|---|---|
| Session | Change renders per session; user-level mixing acceptable; high traffic | None (design effect 1) | Estimand is session-weighted; repeat users counted many times | Re-analyse with user-clustered SEs and report the estimand change |
| User | Change is durable (onboarding, pricing, recommendations); carryover likely | Low unless outcomes are per-session | Mixed experience across devices or when logged out | Device-level split with a merge caveat |
| Device | Anonymous traffic dominates; identity unreliable | Low | Cross-device users get both arms, undermining exchangeability | Split users for logged-in traffic, devices for the rest, report separately |
| Store / region | Operational change where units interfere (staffing, delivery supply) | High: design effect from [E-06] | Few clusters make precision cluster-count-bound | Switchback in time, or matched pairs with a paired analysis |
| Time-based switchback | Supply-side or marketplace change with strong interference | Moderate; needs many periods | Period effects confound treatment if the schedule is not randomized | Randomize the schedule and include period fixed effects |

Selection rule: choose the coarsest unit at which the treatment is well defined and interference is absent, then compute the design effect. If the resulting sample size is unaffordable, do not quietly revert to a finer unit — that reintroduces interference and produces a confidently wrong answer.

### 7.2 Choosing the inference approach for the primary metric

| Situation | Method | Why | Disqualifier |
|---|---|---|---|
| Binary rate, large counts | Two-proportion z with pooled null SE [E-03] | Exact null variance, closed form | Any expected cell under about 5 |
| Binary rate, small counts | Fisher exact or score/Wilson interval | Wald degenerates; score intervals retain coverage | Very large data (compute only) |
| Continuous, roughly symmetric | Welch t / difference in means [E-01] | Robust to unequal variances | Heavy tails with small n; use bootstrap |
| Continuous, heavy-tailed | Bootstrap or pre-declared winsorized mean | Mean is unstable; tail dominates | Winsorization chosen after seeing data |
| Ratio of random sums | Delta method [E-04] or bootstrap | Denominator randomness must be priced | Denominator materially moved by treatment |
| Clustered assignment | Cluster-robust SEs or cluster-level aggregation [E-06] | Unit-level SEs are far too small | Fewer than ~40 clusters; use randomization inference |
| Sequential design | Alpha-spending boundary [E-09] | Controls first-passage error | Boundary chosen post hoc |
| Many metrics | Bonferroni or BH [E-10], or a pre-declared gatekeeping order | Controls FWER or FDR honestly | Family declared after seeing results |

### 7.3 Choosing how to handle non-compliance

| Goal | Analysis | Assumption | Report alongside |
|---|---|---|---|
| Effect of *offering* the change (rollout decision) | ITT: compare assigned arms | Randomization only | Compliance rate, so a reader can scale |
| Effect of *using* the change for compliers | IV/LATE with assignment as instrument | Exclusion restriction, monotonicity, relevance | First-stage strength, complier share |
| Effect if everyone complied | Scale ITT by compliance | Homogeneous effect among compliers, or an explicit model | A statement that this is an extrapolation |
| Diagnostic only | As-treated or per-protocol | None — not causal | An explicit "not causal" label |

The rule is boring and correct: **report ITT as the headline, IV as a secondary only if the first stage is strong, and never present per-protocol as the effect of the feature.** Per-protocol comparisons reintroduce exactly the selection randomization removed.

### 7.4 Choosing the guardrail set and its tolerance

| Guardrail type | Typical metric | Tolerance style | Why that style |
|---|---|---|---|
| Reliability | Crash/error rate per session | Bound-based: upper 95% relative increase ≤ T | Rare events make point estimates unstable; the bound captures plausible harm |
| Economics | Refund rate, return rate, contribution margin per order | Bound on margin; point estimate on rate | Margin has a hard business floor |
| Experience | p95 latency, time-to-interactive | Absolute threshold, not relative | Perceived thresholds are absolute (e.g. p95 ≤ 2.5 s) |
| Trust | Support contacts per 1,000 orders | Bound-based | Rare, lagged, and expensive to measure well |
| Long-run health | 28-day retention, repeat purchase | Directional only, no gate | Cannot be measured inside the test window; needs a holdback |

The governing principle: **a guardrail without a tolerance is a metric, and a tolerance without a bound is a point-estimate rule.** Bound-based tolerances are usually right for rare harms, because the decision concerns plausible worst case rather than observed average.

### 7.5 Choosing between experiment, quasi-experiment, and decision without measurement

| Option | Cost | Identification strength | When it is the right choice |
|---|---|---|---|
| Randomized experiment | Traffic + engineering + calendar | Strongest under stated assumptions | Change reversible, unit-level assignment feasible, harm bounded |
| Switchback / cluster-randomized | Same plus design-effect multiplication | Strong if the schedule is randomized | Interference is structural (marketplace, shared supply) |
| Difference-in-differences | Historical data only | Requires parallel trends; vulnerable to concurrent shocks | Rollout already happened at a level you cannot randomize |
| Regression discontinuity | Needs a threshold rule | Strong locally at the cutoff | Eligibility is rule-based (score cutoffs, size thresholds) |
| Synthetic control | Aggregate data | Requires a credible donor pool | One or few treated units, long pre-period |
| Ship and monitor | Nearly free | Weak; confounded by seasonality and concurrent change | Reversal is cheap, harm bounded, effect large in aggregate |
| Decide without measurement | Free | None | Decision dominated by constraint (legal, contractual, irreversible harm) |

The honest entry is "ship and monitor." It is legitimate when the change is cheap to reverse and the expected effect exceeds plausible drift. It becomes malpractice when used to avoid the cost of a test for a change that is expensive to unwind — and the tell is usually that nobody wrote down the reversal cost.

### 7.6 Choosing the primary metric form

| Form | Definition | Strength | Weakness |
|---|---|---|---|
| Rate (conversions / eligible) | Binary per unit | Simple variance, closed form | Insensitive to magnitude per converter |
| Mean per unit (revenue / session) | Continuous | Captures magnitude | Heavy tails inflate variance drastically |
| Ratio of sums (revenue / sessions when sessions vary) | Ratio metric | Matches business reporting | Needs [E-04]; denominator can move |
| Quantile (p95 latency) | Order statistic | Matches perception | Slower convergence; needs larger n or bootstrapping |
| Trimmed / winsorized mean | Tail-limited mean | Robust to outliers | Depends on the trimming rule; must be pre-declared |
| Composite / guardrail index | Weighted combination | Single decision number | Weights become the argument; hides component regressions |

### 7.7 The selection procedure in words

1. Is the change reversible and is unit-level assignment feasible? If no, go to Section 7.5 and pick a quasi-experiment; do not run a broken experiment.
2. Do units interfere through a shared resource? If yes, choose a cluster or switchback design and apply [E-06]. If no, choose the coarsest unit at which the treatment is well defined.
3. Is the outcome binary with adequate counts? Use [E-03]. Ratio of random sums? Use [E-04]. Heavy-tailed? Pre-declare winsorization or plan a bootstrap.
4. Compute $n$ from [E-02] with the *control* variance, add headroom for rate drift, convert to calendar days at forecast traffic.
5. If the calendar exceeds the decision deadline, raise $\delta$ — accept that smaller effects will not be resolved — or pick a cheaper design from Section 7.5. Never silently coarsen the unit.
6. Define guardrails with bound-based tolerances, compute their MDE, and state whether they are powered or merely directional.
7. Freeze the analysis script, hash it, and name the person who runs it.

> **Takeaway:** unit choice, inference method, non-compliance handling, guardrail tolerance, and design type are five separable decisions, and each has a disqualifier that must be checked before the others are optimized.
