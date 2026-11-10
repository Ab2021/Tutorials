---
title: "Linear trees and boosting"
topic: "Linear models, decision trees, and gradient boosting as one additive family"
collection: "supervised-ml-and-feature-engineering"
level: "core"
evidence: "DERIVED"
created: "2026-11-10"
last_reviewed: null
content_status: "draft"
version: "2.0"
notation_style: "plain-text"
source_status: "OPEN"
prerequisites: ["Probability and conditional rates", "Matrix algebra and the normal equations", "The prediction contract from 02_Problem_Formulation_and_Losses.md"]
pattern_tags: ["regularisation", "axis-aligned-partition", "second-order-boosting", "shrinkage", "monotone-constraints", "model-selection"]
review_trigger: "A change to the loss, the feature encoding, the deployed decision threshold, or the label horizon."
---

# Linear trees and boosting

> **Evidence:** every formula here is DERIVED and every arithmetic value is VERIFIED by hand recomputation shown in place. The workload sizes and the ten-row dataset are ILLUSTRATIVE constructions, not measurements of any organisation. No benchmark claim, leaderboard position, or production result is asserted. Method attributions are to named published results in Section 10; the source bodies were not re-read for this revision, so the coverage field stays OPEN.

## 1. File Header

### 1.1 What decision this sheet changes

The decision in view is narrow and recurring: given a tabular prediction problem, a loss, a fixed serving budget, and a requirement to explain the output, which model family do you fit, with which regulariser setting, and what evidence would make you abandon it. That decision is made dozens of times inside a single project — one per candidate feature set, one per candidate loss, one per retrain — and each choice silently fixes the shape of the function the system can represent. A team that picks gradient-boosted trees because they won a public benchmark on another problem has made an unexamined assumption about *this* problem's geometry, and that assumption is rarely written down anywhere.

This sheet exists so the choice can be argued from properties rather than from habit. It treats linear models, single trees, random forests, and boosted ensembles as four points on one additive scale, then shows where each point breaks.

### 1.2 Why the title says "linear trees"

The phrase is deliberately load-bearing because two distinct objects hide behind it, and confusing them causes real modelling errors.

The first object is a **linear model and a tree model compared as two answers to the same question**. The linear model assumes the response is a weighted sum of inputs; the tree assumes it is a piecewise-constant function of axis-aligned boxes. These are different inductive biases with different failure geometry, and most practical selection questions reduce to which geometry the data actually has.

The second object is a **tree with linear leaves**, sometimes called a model tree or a piecewise-linear tree. Here each leaf fits a small linear model instead of a constant. That hybrid is not a curiosity: it lets a tree represent a global linear trend without spending splits on it, which changes the sample-efficiency and extrapolation behaviour described in Sections 4.3, 6.4, and 8.

Both objects appear below. Section 3 names them separately so that a reader who asks "does this sheet mean linear trees or linear-versus-trees" gets an unambiguous answer.

### 1.3 The four commitments a model choice actually makes

Any tabular model commits to four things at once, and a selection argument that ignores one of them will be overturned later.

| Commitment | What it fixes | Where it is settled in this sheet |
|---|---|---|
| Function shape | Which target functions are cheap to represent | Sections 2.2 and 6 |
| Loss and link | What "wrong" means and how gradients behave | Sections 4.2 and 4.3 |
| Capacity control | How much memorisation is permitted | Sections 4.1, 4.4, and 9 |
| Explanatory contract | What an attribution means to a human | Sections 3.6 and 7 |

A model that wins on shape but loses on explanatory contract is not a win; it is a trade that must be named out loud.

### 1.4 Audience and assumed starting point

The reader is assumed to know how to fit a logistic regression and to have seen a decision tree drawn on a slide. Assumed missing is the second-order view of boosting — that each round solves a tiny regularised quadratic — and the habit of computing split gain by hand to check that a library did what the configuration said. Section 4 builds both. Section 5 executes them on ten rows so that every number can be reproduced with a calculator.

### 1.5 Scope boundary

| In scope | Out of scope | Where the excluded topic lives |
|---|---|---|
| Ridge, lasso, elastic net closed forms and degrees of freedom | Feature construction, encoding, and point-in-time correctness | [04 Feature Engineering Patterns](04_Feature_Engineering_Patterns.md) |
| CART splitting criteria and greedy search | Split protocol, leakage auditing, metric choice | [01 Model Validation and Error Analysis](01_Model_Validation_and_Error_Analysis.md) |
| Second-order gradient boosting with leaf weights, lambda, gamma, shrinkage | Deriving the loss from the decision cost | [02 Problem Formulation and Losses](02_Problem_Formulation_and_Losses.md) |
| Random forests as variance reduction | Class weighting, probability calibration, threshold choice | [05 Imbalance and Calibration](05_Imbalance_and_Calibration.md) |
| Monotone constraints and linear leaves | Neural architectures, sequence models, embeddings | Module 04 deep learning material |
| Cost of training and serving these families | Cluster provisioning and infrastructure economics | Module 06 platform material |

### 1.6 Evidence and notation policy

Display mathematics appears in fenced `text` blocks with identifiers of the form `LTB-NN`. Symbols are redefined inside each block so a copied block stays readable. Counts are integer record counts; rates are dimensionless in `[0,1]`; currency is the unnamed unit `CU`. A block tagged ILLUSTRATIVE carries constructed inputs even when its arithmetic is exact; a block tagged VERIFIED has been recomputed at least twice by different routes, and the second route is printed.

The distinction is not pedantry. "Log-loss 0.31 on 10 rows" is arithmetically checkable and operationally meaningless. A reader who repeats such a figure as a benchmark has converted a teaching device into a fabricated result, which is the single most common way model-selection material misleads its audience.

### 1.7 The minimum comparison record

Before trusting any statement of the form "boosting beat the linear model", seven fields must be present. This sheet uses exactly these fields in Section 5 and Section 7.

1. The loss, stated with its link and its units.
2. The evaluation split rule, and what dependence it forces the model to survive.
3. The number of rows used to fit, and the number of distinct entities behind them.
4. The capacity control actually applied — lambda, gamma, depth, rounds, shrinkage — not the grid that was searched.
5. The comparison baseline and its score on identical rows.
6. The serving constraint: latency budget, model size ceiling, or interpretability requirement.
7. The uncertainty of the difference, or an explicit statement that it was not measured.

**Takeaway:** a family comparison is a five-tuple of loss, split, capacity, baseline, and constraint; drop any one and the comparison reverts to folklore.

## 2. Concept Map / Mental Model

### 2.1 One additive scale, four operating points

All four families can be written as a sum of simple functions of the input. What differs is the *dictionary* of simple functions and the *procedure* that selects them.

```text
f(x) = sum over m of  nu_m * b_m(x)          LTB-01

  family              b_m(x)                        selection procedure
  ------------------  ----------------------------  ---------------------------
  linear / ridge      w_j * x_j  (one per feature)  closed form or convex solver
  single tree         constant on an axis box        greedy recursive partition
  random forest       constant on an axis box        greedy partition on bootstraps
  boosted ensemble    constant (or linear) on box    greedy partition on residuals

  nu_m  = contribution weight of term m, dimensionless
  b_m   = basis function, maps R^d -> R
```

Reading the scale this way makes two facts immediate. First, a linear model is a single-term model whose basis functions are the raw coordinate projections — it cannot represent an interaction unless one is written into `x`. Second, a forest and a boosted ensemble use the *same* basis functions and differ only in how the terms are selected and weighted, which is why their error profiles diverge so sharply despite identical geometry.

### 2.2 Geometry: hyperplanes against boxes

A linear model divides the space with one hyperplane per decision boundary direction: the set `{x : w'x = c}` is a flat surface of dimension `d-1`. It can only express a target whose level sets are parallel slabs, and it extends that assumption to infinity in every direction.

A tree divides space with axis-aligned cuts, recursively. The regions are boxes, the level sets are unions of boxes, and outside the observed range the prediction is constant because no cut exists there.

The practical consequence is the single most useful diagnostic in this sheet: **if the target has a diagonal ridge or a multiplicative structure, trees need many cuts to imitate it; if the target is a smooth monotone ramp extending beyond the data, linear models reproduce it for free while trees flatten.** Section 6 quantifies the first half of that sentence with a staircase bound.

### 2.3 Three separate capacity knobs, often conflated

Capacity control in tree ensembles is not one dial. There are three, they act on different mechanisms, and turning the wrong one is why "we regularised it and it still overfits" happens.

```text
knob                 mechanism                        symptom when too loose
-------------------  -------------------------------  ----------------------------
depth / min_child    limits interaction order and      memorises interactions among
  (+ gamma)            per-leaf sample size              a handful of rows
lambda / min_gain    shrinks leaf magnitude and        leaf values chase noise,
                       refuses unprofitable splits       especially in small leaves
rounds x shrinkage   limits total additive movement    model fits the training
                       from the initial intercept        labels more each round
```

Depth controls how many rows can be isolated together; lambda controls how loudly each isolated group speaks; rounds control how far the whole function travels. A model can be shallow, heavily shrunk, and still overfit if it runs for three thousand rounds on six hundred rows, because shrinkage only delays movement, it does not bound it.

### 2.4 Boosting as functional gradient descent

The mental model that makes every boosting hyperparameter interpretable: the model is a point in function space, the loss is a functional, and each tree is one step.

```text
   initialise F_0(x) = argmin_c sum_i L(y_i, c)        (a constant, usually log-odds)

   for m = 1..M:
       g_i = dL(y_i, F)/dF      evaluated at F = F_{m-1}(x_i)
       h_i = d2L(y_i, F)/dF2    evaluated at F = F_{m-1}(x_i)
       fit a small tree to the pair (g, h)  ->  leaves R_1..R_J
       for each leaf j:  w_j = - sum_{i in R_j} g_i / ( sum_{i in R_j} h_i + lambda )
       F_m(x) = F_{m-1}(x) + nu * w_{j(x)}

   g, h are per-row first and second derivatives of the loss w.r.t. the score
   nu is the shrinkage (learning rate), dimensionless, typically 0.01 to 0.3
```

Every design choice in a real library is a variation on a line of that loop. Subsampling changes which rows contribute to `g` and `h`. Column sampling changes which features the tree may use. Gamma adds a constant to the split criterion, so a split must earn more than `gamma` before it exists. Monotone constraints restrict the set of admissible cuts. Nothing here is mysterious once the loop is visible.

### 2.5 Where each family's error comes from

| Family | Dominant error at small `n` | Dominant error at large `n`, bad features |
|---|---|---|
| Ridge | Bias from the linear restriction | Bias from misspecified interactions |
| Lasso | Selection instability | Drops one of two correlated useful features |
| Single tree | Variance: a few rows move a whole box | Greedy myopia, unstable structure |
| Random forest | Underfits because trees are shallow and averaged | Cannot extrapolate, large model size |
| Boosting | Overfits small leaves, especially with lambda near zero | Chases label noise in high-cardinality columns |
| Linear-leaf tree | Split instability plus coefficient noise in thin leaves | Overfits interactions inside leaves |

### 2.6 Diagnosis map: symptom to knob

This table is the operational core of the sheet. Enter with a symptom, leave with one experiment and one thing to check first.

| Symptom | First check | Likely knob | One experiment |
|---|---|---|---|
| Train loss 0.09, validation loss 0.31 | Are validation rows independent of training entities | Rounds, then depth | Halve rounds, watch validation, not train |
| Predictions identical for every row | Are all split gains below gamma | Gamma, lambda | Set gamma to zero and print the top gain |
| Validation AUC fine, calibration terrible | Was early stopping done on the reported split | Rounds and the monitoring set | Re-fit with a third split for stopping |
| Linear model collapses, tree holds | Feature units or scale changed | Standardisation | Compare coefficient magnitudes across folds |
| Deep model barely beats depth 3 | Features already encode the interactions | Depth | Try depth 2 with more rounds |
| Two correlated features swap importance | Importance is split-count based | Importance type | Switch to permutation importance |
| Predictions flat outside training range | Extrapolation demand is real | Family | Test a linear term or linear leaves |

**Takeaway:** name the symptom, then the knob, then the experiment; "tune the hyperparameters" is not a diagnosis.

## 3. Foundational Definitions

### 3.1 Linear model, ridge, lasso, elastic net

A linear model predicts `eta = b_0 + sum_j b_j x_j` and passes `eta` through a link. For regression the link is identity; for binary outcomes it is the logistic function. The coefficients are estimated by minimising a loss plus a penalty.

Ridge adds `lambda * sum_j b_j^2`. Its effect is continuous shrinkage: no coefficient reaches exactly zero, so it never performs selection, but it stabilises correlated blocks by spreading weight among them. Lasso adds `lambda * sum_j |b_j|`, whose subgradient at zero produces exact zeros, so it selects. Elastic net mixes the two with a ratio parameter and is the default when features arrive in correlated groups and you want the group kept or dropped together.

The units matter: `b_j` carries the unit of the response divided by the unit of `x_j`, so a penalty on `b_j` is only meaningful after every feature is on a comparable scale. This is why an unscaled ridge fit is not interpretable as "regularised" — the penalty is punishing whichever feature happens to have the smallest numerical range.

### 3.2 Split gain, leaf weight, gamma, lambda

Four quantities appear in every boosting configuration and are routinely confused.

- **Leaf weight** `w_j` is the value added to the score for rows landing in leaf `j`. It has the unit of the score (log-odds for classification). It is negative when the leaf's rows are worse than the current model believes.
- **Split gain** is the reduction in the regularised objective obtained by adding that split, computed from the same `G` and `H` sums. It is dimensionless because the objective is a sum of losses, and it is compared against `gamma`.
- **Gamma** is the minimum gain a split must achieve to be admitted. It acts per split, not per tree, so a tree can have a profitable root split and unprofitable children, in which case the children are pruned away and the root becomes a leaf.
- **Lambda** is the L2 penalty on leaf weights. It appears in the denominator of every weight, so it damps small leaves more than large ones, which is precisely the intended behaviour: a leaf with four rows and `H = 1.0` is a noisy estimate, and lambda turns its shout into a murmur.

### 3.3 Shrinkage and rounds are one parameter, not two

`nu` (shrinkage) and `M` (rounds) are not independent. If the target function needs total movement `D` from the intercept, then after `M` rounds with per-round movement `nu * w`, the achieved movement is approximately `D * (1 - (1-nu)^M)` when successive tree contributions are of similar size.

```text
fraction of asymptotic movement reached after M rounds:  R(M) = 1 - (1 - nu)^M

  nu = 0.05  ->  R(50)  = 1 - 0.95^50  = 0.9231   R(100) = 0.9941
  nu = 0.10  ->  R(50)  = 1 - 0.90^50  = 0.9948   R(100) = 1.0000
  nu = 0.30  ->  R(50)  = 1 - 0.70^50  = 1.0000   R(100) = 1.0000

  half of the asymptotic movement is reached at  M = ln(0.5)/ln(1-nu)
  nu = 0.05 -> M = 13.51   nu = 0.10 -> M = 6.58   nu = 0.30 -> M = 1.94
```

The identity is exact for identical contributions and approximate otherwise, because once the model approaches the optimum the per-round `w` shrinks as the residuals shrink. That interaction is why `nu = 0.05` with 300 rounds and `nu = 0.3` with 50 rounds are not equivalent: they reach similar totals but the small-`nu` run has finer resolution near the optimum, trading compute for accuracy.

### 3.4 Row and column subsampling

Row subsampling fits each tree on a fraction `s_r` of rows; column subsampling restricts the features considered per split to a fraction `s_c`. They act on different variances. Row sampling decorrelates trees, which is the mechanism that makes averaging work (§10.2). Column sampling prevents one strong feature from dominating every tree's root split, which reduces the effective depth needed to reach a given accuracy and adds a randomisation that often improves generalisation slightly at the cost of more rounds.

Both are *regularisers that cost compute*: shrinking `s_r` reduces per-tree work but increases the number of rounds needed for the same total movement, so wall-clock time frequently rises even as per-tree time falls.

### 3.5 Monotone constraints

A monotone constraint declares that the model's output must be non-decreasing in a named feature. It is enforced during tree construction: when evaluating a candidate split on constrained feature `k`, the candidate's left leaf weight must not exceed the right leaf weight for an increasing constraint. If the unconstrained optimum violates that, the library solves a constrained version of the quadratic (typically an isotonic projection of the leaf values).

The constraint is a *statement about the world*, not a regulariser in the statistical sense. When it is true, it buys sample efficiency and protects the model outside the observed range. When it is false, the fit degrades and, worse, the degradation is invisible in aggregate metrics because a monotone model still produces plausible outputs.

### 3.6 Three things called "feature importance"

These are not interchangeable and are routinely reported interchangeably.

| Variant | Definition | Fails when |
|---|---|---|
| Gain importance | Mean split gain contributed by splits on that feature | Biased toward high-cardinality features with many candidate cut points |
| Split-count importance | Number of splits using the feature | Same bias, worse; ignores gain entirely |
| Permutation importance | Loss increase after shuffling that column on held-out rows | Correlated features split the credit; needs repeated shuffles for stability |

A cardinality-rich identifier-like column can top gain and split-count importance while carrying no transferable signal, because it offers many thresholds each of which looks mildly profitable on training rows. The discriminating check is permutation importance computed on a held-out split with the shuffle repeated enough times to estimate a standard deviation.

### 3.7 Bins, histograms, and the discretisation contract

Histogram-based boosting replaces each feature with an integer bin index, typically 255 bins or fewer. The contract is that every threshold candidate is a bin boundary, so the model can only cut where a boundary falls. With 255 bins the resolution loss is small relative to typical label noise, and the gain is large: the solver scans `d * bins` per node instead of sorting `n` values per feature per node.

The contract has two edge cases worth remembering. Ties heavier than the bin budget force quantile boundaries to collide, producing empty or degenerate bins (Section 8). And a binned model cannot represent a threshold finer than its bin width, which matters when the business rule is exactly at a specific value such as a policy limit.

### 3.8 Linear leaves and model trees

A model tree replaces the constant `w_j` with a linear function fitted to the rows in leaf `j`. The gain is representational: a global linear trend is learned once rather than rebuilt from dozens of splits. The cost is variance: each leaf now estimates as many parameters as it uses features, so thin leaves produce wild coefficients.

The safe configuration bounds both ends: minimum leaf size large enough for the coefficient count, and a ridge penalty on the leaf fit. Section 4.3 works through a three-point leaf where a linear fit reduces squared error by a factor of about 148 relative to a constant, and immediately shows why three points and two parameters is the boundary case rather than a recommendation.

### 3.9 Terminology that collides

- "Weight decay" in a neural network is ridge under another name when the loss is a sum rather than a mean; the effective lambda differs by a factor of `n`.
- "GBM", "GBDT", "GBRT", and "gradient boosting" name the same object with different loss defaults.
- "Boosting" is sometimes used for AdaBoost's exponential reweighting, which is a special case of the gradient view only when the loss is exponential.
- "Pruning" in CART post-pruning is a different operation from gamma's per-split admission test, though both remove splits.
- "Feature importance" without a qualifier is the most common source of contradictory model reports in a review.

### 3.10 The initial score is a parameter

Every ensemble starts from a constant. For squared loss that constant is the mean; for logistic loss it is the log-odds of the base rate. Getting it wrong is not fatal because boosting will correct it, but it changes how many rounds are needed and it changes the meaning of the first few trees. In a heavily imbalanced problem the log-odds start can be near `-4` for a 2% base rate, which means the first tree is fitting residuals around a very confident prior — a detail that explains why early rounds look uninformative in partial-dependence plots.
