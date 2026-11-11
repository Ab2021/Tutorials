## 10. Key Theorems / Results Reference

Each entry states the result, the conditions under which it holds, and the operational consequence. Proofs are omitted where the derivation is in Section 4; what matters here is the *scope* of each result, because scope errors are how correct theorems produce wrong policies.

### 10.1 Bayes posterior for precision

**Statement.** For a cohort with prevalence pi, detection rate TPR, and false-positive rate FPR, precision PPV = pi·TPR / (pi·TPR + (1−pi)·FPR).

**Conditions.** TPR and FPR measured on the same cohort, same window, same eligibility rules. Event independence is *not* required for the identity (it is an average), but it is required for the confidence interval around it.

**Consequence.** Precision is jointly determined by the model and the population. A model cannot be certified as "24% precise" without naming a prevalence. Corollary: any precision figure quoted cross-channel requires the prevalence to be quoted with it.

### 10.2 Neyman-Pearson and the shape of the optimal frontier

**Statement.** For a fixed FPR budget, the most powerful test is a likelihood-ratio threshold: alert when P(x|fraud)/P(x|legitimate) exceeds a constant. The ROC frontier attained by such tests is concave, and any point inside the concave hull is dominated.

**Conditions.** Both class-conditional densities known (or well estimated); a single FPR constraint.

**Consequence.** If your measured ROC has points inside its own convex hull, those operating points are unattainable-or-dominated and should not be chosen. In practice, hull violations are a sign of score quantisation or of two populations mixed together — often the finding that reveals a segment boundary.

### 10.3 Bayes-optimal threshold equals the cost ratio

**Statement.** Under a binary action with loss λ_FN per missed fraud and λ_FP per false alert, the expected-loss-minimising rule alerts when p ≥ λ_FP/(λ_FP + λ_FN).

**Conditions.** Losses constant across events; actions binary; probabilities calibrated.

**Consequence.** With λ_FN = rho·L = 198.40 and λ_FP = c_a + c_f = 16.10, the threshold is 16.10/214.50 = 0.0751 — *not* the 0.0803 derived in Section 4.1. The difference is structural: the loss-ratio form charges review cost to the false-alert branch, while the value form charges it to every alert. Both are correct under their own accounting; mixing them produces a threshold that is right for neither.

```text
  loss-ratio form:     0.0751  (review cost inside the FP branch)
  value form:          0.0803  (review cost charged to all alerts)
  difference:          0.0052 probability = 1.04 currency per alert at this margin
```

This is the most common silent error in threshold derivation, and the fix is to write down which branch pays c_a before dividing.

### 10.4 Proper scoring rules and the Murphy decomposition

**Statement.** Brier score decomposes exactly as Brier = reliability − resolution + uncertainty, where reliability = Σ (n_b/N)(p_b − o_b)², resolution = Σ (n_b/N)(o_b − ō)², uncertainty = ō(1 − ō). Both Brier and log loss are strictly proper: expected loss is minimised by reporting the true probability.

**Conditions.** Decomposition uses discrete bins; the identity is exact for the binned estimate and approximate for the continuous score. Resolution and reliability trade off: a model can improve reliability by predicting closer to the base rate, which destroys resolution.

**Consequence.** Calibration work that raises reliability while lowering resolution is a *net loss* in Brier terms. The illustrative audit shows resolution of 0.0000988 against reliability of 0.0000033 — the model is already nearly perfectly reliable and its entire skill (2.66%) comes from the small resolution term. Flattening predictions "to improve calibration" would remove the only skill present.

```text
  Brier          = 0.0000033 - 0.0000988 + 0.0035870 = 0.0034915
  skill          = (resolution - reliability)/uncertainty = 2.66%
  flatten to ō   -> reliability 0, resolution 0, skill 0   (a strictly worse model)
```

### 10.5 Monotone invariance of ranking metrics

**Statement.** For any strictly increasing transform g, AUC(g(s)) = AUC(s) and PR-AUC(g(s)) = PR-AUC(s). Precision at a fixed *quantile* is also invariant; precision at a fixed *threshold* is not.

**Conditions.** g strictly increasing and independent of the label. A transform fitted on labels (isotonic, Platt) is not label-independent and can change ranking on held-out data through overfitting.

**Consequence.** AUC cannot detect calibration breakage; calibration cannot detect ranking breakage. A monitoring suite needs both, and it needs precision at capacity — which is invariant to monotone transforms only when capacity is expressed as a count, not a threshold.

### 10.6 Logit additivity and prior correction

**Statement.** If class-conditional densities belong to an exponential family with equal dispersion, the posterior logit is an additive function of features plus a prior term. Changing prevalence from pi₁ to pi₂ shifts every logit by a constant ln(odds(pi₂)/odds(pi₁)), leaving the ranking and every likelihood ratio unchanged.

**Conditions.** Equal-dispersion exponential family (logistic regression, naive Bayes with shared variance). Violated by models with prevalence-dependent structure (tree ensembles fit to imbalanced data, reweighted training).

**Consequence.** Prior correction is legitimate for logistic models and *not* automatically for gradient-boosted trees trained with class weights. Applying a constant logit shift to a GBM can leave it mis-calibrated because the weighting changed the fitted function, not just the intercept.

```text
  pi 0.004 -> 0.002:  shift = -6.2146 - (-5.5175) = -0.6971
  score 0.0803 -> sigmoid(-3.1358) = 0.0417
```

### 10.7 Value of information: why ranking beats calibration here

**Statement.** The value of improving a score's ordering is bounded below by zero and above by the oracle gap at the operating capacity; the value of correcting a monotone calibration error at a fixed capacity is zero, because the threshold absorbs it.

**Conditions.** Capacity fixed in alerts; threshold free to move; costs unchanged.

**Consequence.** At 8,800 alerts the oracle gap is 1,180,544 per week while the threshold slack is 0.14 per alert (1,232 per week total). Ranking improvement therefore dominates calibration improvement by roughly a factor of 958 on this programme. The theorem is not deep; the failure to apply it is common, because calibration projects are easier to scope and easier to declare finished.

```text
  oracle gap            = 1,462,320 - 281,776 = 1,180,544
  threshold slack value = 0.14 * 8,800        = 1,232
  ratio                 = 958
```

### 10.8 Precision-recall dominance and the iso-lift curve

**Statement.** On a precision-recall plane, the line precision = lift × prevalence is an iso-lift contour. A model's PR curve crosses each contour at the alert rate where it achieves that lift. Comparing two models by maximum lift is equivalent to comparing them at possibly different alert rates, which is meaningless under a capacity constraint.

**Conditions.** Single prevalence; the same recall definition.

**Consequence.** Compare models by precision at the *same* alert count, or by the area between their curves over the capacity-relevant range only. The illustrative model achieves lift 60.0 at 8,800 alerts; a competitor achieving lift 80 at 2,000 alerts is not comparable and may well be worse at 8,800.

### 10.9 Knapsack optimality and its failure

**Statement.** Maximising total value subject to a single capacity constraint is a 0/1 knapsack. The fractional relaxation is solved exactly by value-per-unit-capacity sorting; the integer version is NP-hard and the greedy solution is within a factor of 2 of optimal in the worst case, usually much closer.

**Conditions.** Values and sizes independent across items; single constraint. With deadlines (time-critical cases) the problem becomes scheduling, and greedy value-density can be arbitrarily bad.

**Consequence.** Use value-per-minute sorting for a static queue, and carve out a reserved lane for time-critical alerts whose option value is not captured by static EV. Section 4.4's limit 5 gives the numbers (6.40 versus 1.33 per minute) and the exception.

### 10.10 Summary table of results and their use

| Result | Gives you | Requires | Typical misuse |
|---|---|---|---|
| Bayes PPV | Precision from prevalence | Same-cohort TPR/FPR | Quoting precision without prevalence |
| Neyman-Pearson | Shape of the attainable frontier | Known densities | Choosing a hull-dominated point |
| Cost-ratio threshold | Optimal cut under binary loss | Constant losses | Mixing the two cost accountings |
| Brier decomposition | Where probability skill lives | Binned scores | "Improving calibration" while destroying resolution |
| Monotone invariance | Which metrics can detect what | Strictly increasing transform | Using AUC to certify calibration |
| Prior correction | Portability across prevalence | Exponential family | Applying it to a reweighted GBM |
| Value of information | Where modelling effort pays | Fixed capacity | Spending on calibration at a floating threshold |
| Iso-lift contour | Comparable model comparison | Fixed alert rate | Comparing maximum lift |
| Knapsack | Capacity allocation with variable size | Independent items | Ignoring deadlines |
