> Module 04 — Statistics and Causal Inference
> Topic 06 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 04.01, 04.03

# DAGs: Confounding, Colliders, Mediators, Bad Controls

---

## 1. Why this topic exists

Every causal analysis contains a covariate list, and the covariate list is a causal graph that
nobody drew. When an analyst writes `units ~ promo + competitor_price + in_stock_rate + store_size`,
they have asserted, without saying so, that those four variables between them block every
non-causal path from promotion to units and disturb no causal one. That assertion is either true
or false. The regression cannot tell them which. It will return a coefficient, a standard error,
and a p-value in either case, and all three will be computed correctly.

Two analysts with the same warehouse, the same estimand, and different covariate lists will
produce different numbers, and neither will be lying. This is not a software problem and not a
statistical one in the narrow sense. It is a problem of unstated structure, and the directed
acyclic graph (DAG) is the notation that makes the structure stateable.

The failure has three shapes, and all three are routine in supply chain work.

1. **Something is left out.** A competitor cut its price in the same weeks as the promotion, and
   category demand rose for reasons that had nothing to do with the promotion. Omitting the
   competitor's action leaves a back-door path open and the estimate absorbs the competitor's
   effect. The coefficient is biased upward, sometimes by more than the effect being measured.
2. **Something is put in that should not be.** The analyst drops stock-out weeks "to avoid
   contaminating the sales data", or controls for distribution-centre (DC) utilisation "because
   capacity affects service". Both are conditioning on a collider. Both create an association
   between promotion and sales that does not exist in the population, and in the stock-out case
   the association runs against the promotion: promoted weeks look worse than they were.
3. **Something is put in that changes the question.** The analyst controls for on-shelf
   availability. Availability is partly caused by the promotion, because promoted units sell
   through and deplete the shelf. Adjusting for it estimates the effect of the promotion holding
   availability fixed, which is a direct effect, not the total effect the business asked for. The
   number is not wrong. It answers a different question, and nobody wrote down that the question
   had changed.

The DAG is not a diagram of findings. This distinction carries most of the topic. A path diagram
from a fitted structural equation model has coefficients on its arrows and a chi-square goodness
of fit; it is a picture of a model that has been estimated, and its arrows are claims the data
were allowed to support or reject. A DAG has no coefficients, no fit statistic, and no
likelihood. Its arrows are assumptions brought to the data, and the data cannot confirm them.
What the data can do is test the conditional independencies the graph implies, and even that only
under an assumption (faithfulness) that is itself untestable.

The notation has a provenance worth knowing because it explains why the vocabulary is
epidemiological rather than econometric. Sewall Wright introduced path analysis in the 1920s; the
modern graphical treatment — d-separation, the back-door criterion, the do-operator — was
developed by Judea Pearl and collaborators and consolidated in the 1990s. Greenland, Pearl and
Robins connected it to epidemiological confounding in 1999, and the epidemiological literature
then did the hard work of turning it into practice: Schisterman, Cole and Platt on over-adjustment
(2009), Elwert and Winship on endogenous selection bias (2014), Westreich and Greenland on
covariate coefficients (2013), Cinelli, Forney and Pearl on good and bad controls (2022).
Operations management came later; the DAG is still rare in a supply chain dashboard.

This topic sits where it does because of what the neighbouring topics do and do not supply.
Topic 04.01 fixed the estimand; a DAG is how you state which comparisons can identify it. Topic
04.03 gave you tests; a DAG tells you which conditional independencies are worth testing and,
more importantly, that most of the graph is not testable at all. Topic 04.05 will fix standard
errors; robust errors repair the variance of an estimate whose point value may be biased by an
adjustment decision made in the first ten minutes of the project. Topic 04.07 removes the need for
most of this machinery by randomising. Topic 04.09 is the canonical supply chain endogeneity case
— price and quantity determined together — and is unreadable without the back-door idea. Topic
04.10 asks how strong an unmeasured confounder would have to be to overturn the result, and that
question cannot even be posed until the graph says which confounders are unmeasured.

Supply chain is unusually exposed to all of this, for a structural reason. The variables an
analyst most wants to control for are the operational responses to the treatment. Availability,
utilisation, lead time, safety stock, fill rate, on-time-in-full (OTIF) performance, expedite
spend: every one of them is plausibly caused by the intervention and plausibly causes the outcome.
They look like the best covariates in the dataset. They are the most dangerous, because each is a
mediator, a collider, or both, and the dataset cannot tell you which. Only a stated causal story
can, and the story has to be written before the regression, not after it.

## 2. Learning outcomes

By the end of this topic you can:

1. **Draw** a DAG for a named supply chain question, with nodes in temporal order, treatment and
   outcome marked, and unmeasured variables drawn as latent rather than omitted.
2. **Enumerate** every path between two nodes in a graph of up to ten nodes, and **classify** each
   as causal, back-door, or closed.
3. **Apply** the back-door criterion by hand to derive a minimal sufficient adjustment set, and
   **explain** why a larger set that also blocks the back doors is not the same set.
4. **Identify** confounders, colliders, mediators, and descendants of treatment in a candidate
   covariate list, and **state** the direction of the bias each mistake induces where the sign is
   determinable.
5. **Distinguish** a DAG from a fitted path diagram, and **explain** why the first is an
   assumption and the second is a result.
6. **Recognise** when a graph implies that the effect is not identifiable from the available data,
   and **name** the design or assumption that would be needed instead.
7. **Write** an adjustment declaration — the graph, the estimand, the adjustment set, and the
   variables deliberately excluded — and **place** it in a project appendix where a reviewer can
   find it.

Each outcome maps to an exercise in section 12 and a self-check item in section 13.

## 3. The core mental model

Hold one picture in your head: a set of nodes joined by arrows, a treatment, an outcome, and a
single question asked of every path between them — *is this path open, and if I condition on a
set of variables, does it stay open?*

Three structures exhaust the local vocabulary. A **chain** is `A -> B -> C`, and it transmits
association. A **fork** is `A <- B -> C`, and it also transmits association, because `B` is a
common cause. A **collider** is `A -> B <- C`, and it blocks association, because `B` is a common
effect. Conditioning on the middle node has opposite effects in the three cases: it blocks the
chain, blocks the fork, and opens the collider. That single asymmetry is the whole of d-separation,
and it is the reason that "control for everything you have" is not a conservative strategy but an
arbitrary one.

### 3.1 Figure 1 — confounding

```text
        (C) competitor price move
          |                    |
          |                    |
          v                    v
        (P) promotion ----> (U) units sold

    causal path:      P -> U
    back-door path:   P <- C -> U        (open, unconditioned)
    induced bias:     confounding (omitted-variable bias)
```

The path `P <- C -> U` starts with an arrow into the treatment. It is not a causal path — nothing
travels from promotion back to the competitor and forward to sales — but it is a path, and
association flows along it. Until `C` is conditioned on, the regression of `U` on `P` estimates the
effect of promotion plus the effect of the competitor's price move, mixed in the proportion in
which the two co-vary.

Adjusting for `C` blocks it. That is the entire justification for including competitor price in
the model, and it is a justification that can be written down in one line. Note what the line
does not say: it does not say that `C` predicts `U` well, or that `C` is correlated with `U`. Many
variables predict the outcome and are not confounders; including them buys precision at best and
buys bias at worst.

### 3.2 Figure 2 — a collider

```text
        (P) promotion ----------> (U) units sold
          |                          |
          |                          |
          v                          v
        (Z) stock-out <-------------+

    causal path:      P -> U
    collider at Z:    P -> Z <- U
    closed when Z is not conditioned on
    opened by conditioning on Z (for example, dropping stock-out weeks)
    induced bias:     collider stratification bias (endogenous selection bias)
```

Here the analyst who "cleans" the data by removing stock-out weeks has conditioned on `Z`. Among
stock-out weeks, `Z` is fixed at one, so a high promotion depth must be offset by lower-than-
expected sales for the arithmetic of the stock-out to hold. Promotion and sales become negatively
associated within the selected sample, and the estimated promotion effect falls — or turns
negative — even if the promotion was effective in the full population.

> **Note:** The collider is the structure most often mistaken for a confounder, because it feels
> irresponsible to leave a variable that is obviously related to the outcome out of the model.
> The feeling is not evidence. In Figure 2, `Z` is related to both `P` and `U` and must still be
> left alone.

### 3.3 The three positions, side by side

| Structure | Graph | Association flows? | Condition on the middle node? |
|---|---|---|---|
| Confounder (fork) | `X <- C -> Y` | Yes, when unconditioned | Yes, it blocks the path |
| Mediator (chain) | `X -> M -> Y` | Yes, it is the causal path | Only for a direct effect, never for a total effect |
| Collider | `X -> Z <- Y` | No, when unconditioned | No, it opens a path that was closed |

The table is worth memorising as a single sentence: **adjust for common causes, do not adjust for
common effects, and adjust for consequences of the treatment only when you mean to change the
question.** Almost every adjustment error in applied work is a violation of one of those three
clauses.

### 3.4 Four properties that follow

**Property 1 — The graph states assumptions, not findings.** An arrow from `P` to `U` is a claim
that changing `P` would change `U`, defended by mechanism, prior evidence, or domain theory. No
regression can establish it, because the data are consistent with any graph that implies the same
conditional independencies. A DAG with no arrow between two variables is asserting the absence of
a direct causal effect, which is a stronger claim than most analysts realise they are making.

**Property 2 — d-separation is the only thing the graph predicts, and it predicts independencies,
not coefficients.** A graph implies that certain pairs of variables are independent given certain
others, which is testable in principle. The size of an effect, the sign of a bias, and the fit of
a model are not in the graph. Adjustment is also a graph operation: conditioning means removing the
edges into a node and asking what paths remain.

**Property 3 — A variable's role is relative to the question.** Store size is a confounder for
the effect of promotion on sales; it is a neutral precision variable for the effect of a
distribution-centre reorganisation on pick rate; it is a collider if both promotion assignment and
sales affect which stores remain in the panel. There is no such thing as "a confounder" in the
abstract, only a confounder *for a given treatment, outcome, and population*.

**Property 4 — Omitting a node is not neutrality.** A graph drawn with only the variables in the
warehouse is a graph that asserts there are no unmeasured common causes among them. That is
almost always false. Draw the variables you believe exist and cannot measure, as latent nodes,
and let the graph tell you what you cannot identify. A DAG that hides its unmeasured confounders
is worse than no DAG, because it looks complete.

### 3.5 Why the acyclicity matters, and where supply chains violate it

A DAG forbids cycles: you cannot have `A -> B -> A`. The restriction is what makes the
path-counting and d-separation algorithms work, and it is why the object is called acyclic rather
than merely directed.

Supply chains are full of feedback. The bullwhip effect is a loop: orders amplify demand, which
triggers capacity responses, which change lead times, which change ordering behaviour. Inventory
policies create loops between stock levels and order rates. Promotions create loops between price,
sell-through, and replenishment. None of these can be drawn as a DAG without modification.

The standard resolution is to **unroll time**: draw `P_t -> U_t`, `U_t -> A_{t+1}`, `A_{t+1} ->
U_{t+1}`, and so on. The graph is acyclic because each node is indexed by time, and it represents
the feedback faithfully. You lose compactness and gain the ability to see that a variable measured
at `t+1` cannot confound a treatment at `t`. Section 5.12 treats the consequences.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding system |
|---|---|---|---|
| Directed acyclic graph (DAG) | A set of nodes joined by directed edges with no directed cycle | Called a "flowchart" or a "model diagram" | Graph object; adjacency matrix or edge list |
| Node (vertex) | A variable, attribute, or event represented in the graph | Used for a group of variables collapsed without justification | Node label |
| Edge (arc) | A directed arrow from one node to another, asserting a direct causal effect | Read as correlation or as data flow | Ordered pair of node labels |
| Directed path | A sequence of nodes connected by edges all pointing the same way | Confused with any connection | Ordered node list |
| Causal path | A directed path from treatment to outcome | Confused with any open path | Ordered node list |
| Non-causal path | Any path from treatment to outcome containing at least one arrow pointing backwards | Assumed absent if no arrow is drawn | Ordered node list |
| Back-door path | A path from treatment to outcome whose first edge points into the treatment | Confused with any confounding variable | Ordered node list |
| Front-door path | A path from treatment to outcome that leaves the treatment forward and enters the outcome forward, through a mediator | Rarely recognised as a separate case | Ordered node list |
| Collider | A node on a path with two arrows pointing into it | Treated as a confounder because it correlates with both ends | Node label |
| Chain | `A -> B -> C`; `B` transmits association | Confused with a fork | Three-node motif |
| Fork | `A <- B -> C`; `B` is a common cause | Confused with a chain | Three-node motif |
| d-separation | A criterion for when a set of nodes blocks every path between two nodes | Treated as a property of the variables rather than of the graph and the conditioning set | Boolean, given a graph and a set |
| Conditioning (adjustment) | Holding a variable fixed in the analysis, by stratification, regression, matching, or restriction | Equated with "adding to the regression" | Set of node labels |
| Adjustment set | A set of variables conditioned on to identify a causal effect | Any covariate list | Set of node labels |
| Minimal sufficient adjustment set | An adjustment set that identifies the effect and has no proper subset that also does | Reported as "the" adjustment set when several exist | Set of node labels |
| Back-door criterion | The condition under which an adjustment set identifies the total effect of treatment on outcome | Applied without checking the descendant clause | Boolean criterion |
| Front-door criterion | A condition under which a mediator identifies the effect despite unmeasured confounding of treatment and outcome | Confused with mediation analysis | Boolean criterion |
| Confounder | A variable that is a common cause of treatment and outcome, or more precisely a member of a minimal sufficient adjustment set | Defined as "correlated with both" | Node label |
| Collider variable | A variable caused by two or more other variables on a path | Called a confounder when one cause is the treatment | Node label |
| Mediator | A variable on a directed path from treatment to outcome | Controlled for when a total effect is wanted | Node label |
| Descendant | A node reachable from another by a directed path | Confused with a direct child | Node label |
| Overadjustment (over-control) | Adjustment for a mediator or a descendant of a mediator when a total effect is wanted | Called "being thorough" | Analysis decision |
| Bad control | A variable whose inclusion increases bias in the target estimand | Assumed to be any weak predictor | Node label |
| M-bias | Bias from conditioning on a pre-treatment collider of two unmeasured causes of treatment and outcome | Treated as a reason to avoid all adjustment | Bias mechanism |
| Collider stratification bias | Bias induced by conditioning on a collider | Called "selection bias" only | Bias mechanism |
| Endogenous selection bias | Selection into the sample or the analysis driven by treatment and outcome | Confused with non-response | Bias mechanism |
| Selection bias | Any bias from the process that determines which units are observed | Used as a catch-all | Bias mechanism |
| Unmeasured (latent) confounding | Confounding by a variable not in the data | Assumed away when the covariate list looks long | Node label, drawn dashed |
| Latent variable | A variable believed to exist but not measured | Omitted from the graph entirely | Node label |
| Causal Markov condition | Every variable is independent of its non-descendants given its parents | Rarely stated and often assumed | Assumption |
| Faithfulness | Every conditional independence in the distribution is implied by the graph | Assumed silently | Assumption |
| Positivity | Every covariate pattern has a nonzero probability of each treatment condition | Checked only in matching studies | Assumption |
| Consistency | The observed outcome equals the potential outcome under the treatment received | Broken by vague treatment definitions | Assumption |
| Exchangeability | Treated and untreated units would have had the same outcome absent treatment, within strata | Asserted rather than argued | Assumption |
| Bias amplification | Bias that grows when a strong instrument-like covariate is added to a model with unmeasured confounding | Mistaken for precision improvement | Bias mechanism |
| Table 2 fallacy | Interpreting every coefficient in one adjusted model as a causal effect | Very common in applied papers and dashboards | Reporting error |
| Adjustment declaration | A written statement of the graph, the estimand, the set adjusted, and the set deliberately excluded | Replaced by a sentence in the limitations | Document |
| Time ordering | The assumption that every edge points forward in time | Violated by "baseline" variables measured after treatment | Ordered node list |

Four of these deserve expansion.

**Confounder, as a formal matter.** The intuitive definition — correlated with treatment and with
outcome — is wrong in both directions. A variable can be correlated with both and not be a
confounder (a collider is), and a variable can be a confounder without being marginally correlated
with the treatment (a suppressor). VanderWeele and Shpitser (2013) review six candidate
definitions and settle on one: a confounder is a pre-exposure covariate that is a member of some
minimally sufficient adjustment set. That definition is not circular once the graph is given, and
it has the property that adjusting for all confounders suffices to control confounding while each
individual confounder contributes to bias reduction in some setting. For practice, the graph is
prior and the definition follows from it.

**Over-adjustment, as three distinct errors.** Schisterman, Cole and Platt (2009) separate what
the literature had been calling over-adjustment into cases. Adjusting for an intermediate variable
on the causal path biases the total-effect estimate, usually toward the null. Adjusting for a
descendant of the exposure that is off the causal path is bias-neutral in expectation but costs
precision. Adjusting for a descendant of the outcome can induce collider-stratification bias.
Only the first is over-adjustment in their strict sense; the others are unnecessary adjustment of
different kinds. The distinction matters because the remedies differ: the first requires
re-specifying the estimand, the second requires deleting a column, the third requires thinking
about selection.

## 5. The mechanics, step by step

This section turns a business question into an adjustment declaration. The sequence is fixed and
short: fix the estimand, list the nodes in time order, draw edges from mechanism, enumerate the
paths, apply the criterion, screen the candidate set, handle the unmeasured, write it down,
pre-specify the check. Nine steps, perhaps forty minutes of thinking, and all of it before the
first regression.

### 5.1 Step 1 — Fix the estimand before drawing anything

A DAG is a graph *for a question*. The same warehouse supports a dozen graphs, and they will
differ in which node is the treatment. Before drawing, write the estimand in the six slots from
Topic 04.01: population, unit, outcome, time, intervention, comparison. In particular, decide
whether you want the **total effect** or a **direct effect**.

The total effect of promotion on sales includes every path by which the promotion changes sales,
including the path through availability. The direct effect holds availability fixed. The first is
the right estimand for "should we run the promotion as we run it"; the second is the right estimand
for "what would the promotion do if we pre-built stock". They are different numbers, and the
choice between them determines whether availability belongs in the adjustment set or must be
excluded from it.

> **Note:** The single most common DAG failure in practice is not a wrong arrow. It is drawing the
> graph before deciding which effect is wanted, then adjusting for whatever the business finds
> interesting. The graph is downstream of the estimand, always.

### 5.2 Step 2 — List the nodes and put them in time order

Write down every variable you believe is relevant, measured or not. Then sort them by the time at
which they are determined, earliest first. The sort is not decoration: it constrains which edges
are even possible. A variable determined after the outcome cannot cause the outcome. A variable
determined after the treatment cannot be a confounder for it.

A practical template for a supply chain question has five layers.

| Layer | Typical nodes | Why the layer matters |
|---|---|---|
| Structural, slow-moving | Store format, DC location, supplier country, category | Determined years before; almost always safe confounders or precision variables |
| Seasonal and external | Weather, competitor action, macro demand, holiday calendar | Determined before the treatment period but often unmeasured |
| Assignment | The treatment itself; the manager's selection rule | The node whose causes are the confounders |
| Operational response | Availability, utilisation, lead time, expedite spend, OTIF | Determined after treatment; mediators and colliders live here |
| Outcome and descendants | Sales, margin, basket size, returns, loyalty | Never adjust for descendants of the outcome |

The fourth layer is where supply chain analyses go wrong. Utilisation responds to demand,
availability to sell-through, OTIF to allocation. All three belong to the right of the treatment,
and all three are routinely placed on the left.

### 5.3 Step 3 — Draw edges from mechanism, not from correlation

An edge means "an intervention on the parent would change the child, with everything else held
fixed". That is a mechanistic claim, and the justification for it is domain knowledge, prior
studies, or a physical or contractual mechanism. It is not a correlation coefficient, a feature
importance score, or a line in a data dictionary.

Three rules make the drawing faster and more honest.

1. **Draw the edge if you can describe the mechanism in one sentence.** "A deeper discount pulls
   forward demand, so units sold in the window rise" is an edge. "Promotion and sales are
   correlated" is not.
2. **Draw the edge even when the mechanism is uncertain, and mark it.** A dashed edge is a
   statement of uncertainty; an absent edge is a statement of certainty that you probably cannot
   defend. Section 13's rubrics reward graphs that show contested edges.
3. **Draw the unmeasured nodes.** Competitor intent, unobserved demand shocks, supplier financial
   distress, true shelf condition. If they are not on the page, the graph asserts they do not
   exist.

### 5.4 Step 4 — Enumerate the paths between treatment and outcome

A path is any sequence of adjacent nodes, ignoring direction. Two nodes with ten nodes between
them may have dozens of paths, and every one must be classified.

The classification is mechanical:

- A **causal path** is directed from treatment to outcome. It carries the effect you want.
- A **back-door path** begins with an edge into the treatment. It carries confounding.
- A path that begins with an edge out of the treatment but reaches the outcome through a
  non-directed route is a **partially directed path**; it is handled by the same blocking rules
  and is where most errors occur.

For realistic graph sizes, enumerate by hand first. Software is faster, but the hand
enumeration is what catches the path you forgot to draw.

### 5.5 Step 5 — Apply the back-door criterion

A set `Z` satisfies the **back-door criterion** relative to an ordered pair `(X, Y)` if:

1. No node in `Z` is a descendant of `X`.
2. `Z` blocks every path between `X` and `Y` that contains an arrow into `X`.

If such a `Z` exists, the total effect is identified by the adjustment formula:

`P(y | do(x)) = Σ_z P(y | x, z) · P(z)`

and in a linear model the corresponding regression coefficient on `X` in a model that includes `Z`
estimates the total effect, provided the model is correctly specified in functional form.

Three practical notes.

**The descendant clause is not a formality.** It rules out mediators and every other consequence
of treatment, because conditioning on them changes the estimand or opens a collider. A set that
blocks all back doors but includes a mediator fails the criterion.

**Minimality is separate.** Many sets satisfy the criterion; the minimal sufficient sets have no
redundant member. They are not unique — either of two variables may block the same path — so
reporting "the" adjustment set hides a choice.

**Identification is not precision.** A set of weak predictors blocks the back doors and gives a
noisy estimate. Adding neutral variables that predict the outcome is legitimate, provided they
are not colliders or descendants of treatment.

### 5.6 Step 6 — Screen the candidate set for colliders and descendants

Given the business's proposed covariate list, classify every variable against the graph.

| Role | Test on the graph | Correct action for a total effect |
|---|---|---|
| Confounder | Ancestor of treatment and of outcome, on an open back-door path | Include |
| Proxy for a confounder | Descendant of an unmeasured confounder, affects outcome only through it | Include with a caveat; it blocks the path imperfectly |
| Neutral precision variable | Cause of outcome only, not of treatment, not a descendant of either | Include if it predicts the outcome; harmless |
| Mediator | On a directed path from treatment to outcome | Exclude for a total effect; include only for a direct effect |
| Collider | Common effect of treatment and outcome, or of two causes one of which is associated with each | Exclude; conditioning opens a path |
| Descendant of treatment | Reachable from treatment | Exclude unless it is the mediator you mean to hold fixed |
| Descendant of outcome | Reachable from outcome | Exclude; conditioning induces selection bias |
| Selection variable | Determines inclusion in the sample or the analysis | Cannot be excluded if the sample is already selected; the estimand must be restated for the selected population |

The last row deserves a note. If the data contain only stores that stayed open, or suppliers that
survived, or lanes with complete records, then the sample is already conditioned on a variable
that may be a collider. Deleting a column cannot undo it. The honest response is to restate the
estimand for the selected population, and to say so, rather than to report a population effect
that the sampling cannot support.

### 5.7 Step 7 — Handle what you cannot measure

Redraw the graph with the unmeasured variables as latent nodes. Then ask the only question that
matters: **does every back-door path pass through a latent node?**

If yes, no adjustment set of observed variables identifies the effect. There are six responses,
and they should be considered in this order.

1. **Find a proxy** — a measured descendant of the latent confounder; the residual
   confounding is a quantity to discuss, not to assume away.
2. **Find an instrument** — shifts treatment and affects the outcome only through it.
3. **Use a front-door path** — a mediator satisfying the criterion stated below.
4. **Change the design** — difference-in-differences, regression discontinuity, a pilot.
5. **Bound the effect** — report the range consistent with the assumed confounding.
6. **Decline the causal claim** — report the association and say what identification
   would require.

The **front-door criterion** applies when a set `M` satisfies three conditions: `M` intercepts
every directed path from `X` to `Y`; there is no unblocked back-door path from `X` to `M`; and
every back-door path from `M` to `Y` is blocked by `X`. Then

`P(y | do(x)) = Σ_m P(m | x) · Σ_x' P(y | m, x') · P(x')`

The formula is worth reading once for what it does: it uses the treatment as its own control for
the mediator-outcome relationship, which is why it survives confounding of `X` and `Y`. Section
13.6 works an example.

### 5.8 Step 8 — Write the adjustment declaration

The declaration is one page and it is the deliverable of this topic. It has six parts.

1. **The estimand**, in six slots.
2. **The graph**, as a figure plus an edge list, including latent nodes.
3. **The adjustment set**, with the back-door paths it blocks enumerated.
4. **The variables deliberately excluded**, each with a one-line reason: mediator, collider,
   descendant, unavailable.
5. **The identification argument**, in one paragraph: why the remaining paths are closed, and what
   would have to be true for that to fail.
6. **The check that would embarrass the graph**: a negative-control outcome, a placebo period, a
   pre-trend test, or a sensitivity analysis.

Part four is the part reviewers value most and analysts write least. A reader can reconstruct a
covariate list from a regression table; they cannot reconstruct the reasoning that kept three
plausible variables out. That reasoning is the analysis.

### 5.9 Step 9 — Pre-specify the check that would embarrass the graph

Every graph implies testable independencies. Pick the one that would do the most damage if it
failed, and test it before looking at the effect estimate.

Four stand out: a negative-control outcome that the treatment cannot affect; a placebo
period before assignment; a pre-trend test in a difference-in-differences design; and a
re-estimate on the sample the collider would have selected. None confirms the graph. Each
can refute it, and a refuted graph is a cheap discovery.

### 5.10 The four adjustment mistakes in one table

| Mistake | Graph shape | What the analysis does | Usual direction of bias |
|---|---|---|---|
| Omit a confounder | `X <- C -> Y` | Leaves the back-door path open | Same sign as the product of `C`'s two effects; not determinable from the graph alone |
| Include a collider | `X -> Z <- Y` | Opens a closed path | Not determinable without signs and magnitudes; often toward the null for the effect of interest |
| Include a mediator | `X -> M -> Y` | Changes the estimand from total to direct | Removes the indirect component; direction depends on its sign |
| Include a descendant of treatment | `X -> D -> ...` | Partially blocks the causal path and may open a collider | Attenuates the total effect and adds an unpredictable collider component |

Two of these are correctable by deleting a column from the regression. The first is not: an
omitted confounder requires new data, a proxy, a design change, or a sensitivity analysis. That
asymmetry is why the graph is worth drawing before the data are assembled, when the missing
variable can still be measured.

### 5.11 What a DAG cannot encode

The list is long enough that it should be taught explicitly, because the limits are where the
method gets oversold.

| Cannot encode | Consequence | What to use instead |
|---|---|---|
| Feedback loops | Bullwhip and inventory dynamics cannot be drawn | Time-unrolled graphs; system dynamics; g-methods |
| Effect modification | One edge cannot say the effect differs by store type | Separate graphs by stratum, or edge labels with a modifier |
| Dose and duration | `X` is a node, not a function of dose | Potential outcomes indexed by dose; dose-response models |
| Functional form | The graph says nothing about linearity or interactions | Specify and test the model; the graph is not a regression |
| Magnitude of bias | An open path does not say how much bias it carries | Sensitivity analysis (Topic 04.10) |
| Interference and spillover | The stable unit treatment value assumption (SUTVA) is built in | Causal graphs with network structure; exposure mapping |
| Measurement error | Standard d-separation assumes variables measured without error | Latent-variable graphs with measurement models |
| Sampling mechanism | Selection can be drawn as a node only if the mechanism is known | Selection diagrams; explicit weighting |

The functional-form row is the one most often forgotten. A DAG licenses a set of variables, not a
model. Having established that `{C, H}` blocks the back doors, you still have to decide whether
the effect is linear, whether it interacts with store format, and whether the outcome needs a
transformation. Those are modelling decisions, and the graph does not make them for you.

### 5.12 Time ordering and the "baseline" trap

Every edge must point forward in time. This sounds trivial and produces a large share of real
errors, because the word "baseline" is used loosely to mean "measured before the analysis" rather
than "measured before the treatment".

| Variable called "baseline" | Actually measured | Status | Consequence of adjusting |
|---|---|---|---|
| Pre-period average sales | Before assignment | Confounder or precision variable | Legitimate |
| Store size from the master data | Before assignment | Confounder | Legitimate |
| In-stock rate during the promotion | After assignment | Mediator and possibly collider | Changes the estimand; may induce bias |
| Average price paid during the promotion | After assignment | Mediator or descendant | Removes the price effect you are measuring |
| Whether the store remained in the panel | After assignment | Selection variable | Conditions on a collider |
| Basket size during the promotion | After the outcome | Descendant of the outcome | Induces selection bias |

The remedy is procedural rather than statistical: timestamp every covariate with the date at which
it is determined, and require that date to precede the treatment date. A covariate whose timestamp
is after assignment is not a covariate. It is an outcome, a mediator, or a selection variable, and
it belongs in the graph on the right-hand side where you can see it.

> **Note:** When treatment and confounders are measured repeatedly, a confounder at `t+1` that is
> affected by treatment at `t` is a mediator, not a confounder, and standard adjustment fails.
> That setting needs marginal structural models or g-computation, which are outside this scope.

## 6. Worked example

### 6.1 Setting

A grocery retailer runs a 15% price discount for eight weeks in 60 of its 240 stores. The other
180 stores hold price. The commercial team asks whether the promotion increased units sold, and
the analyst receives a warehouse table with eight candidate covariates that "might affect sales".
All figures below are constructed from a stated linear model so that the arithmetic can be
checked; they are not a claim about any real retailer.

The population model, with coefficients in thousand units per store-week per 10 percentage points
of discount, is:

```text
P = 0.5 C + 0.4 H + e_P            Var(e_P) = 1.00, Var(P) = 1.41
A = -0.6 P + 0.8 S + 0.5 D + e_A  Var(e_A) = 0.40, Var(A) = 1.80
U = 0.8 P + 0.9 A + 1.2 C + 0.9 H - 1.0 S + 0.3 W + e_U
```

where `P` is promotion depth, `U` is units sold, `C` is the competitor's price move, `H` is the
store's pre-period baseline volume, `A` is on-shelf availability, `S` is supplier disruption, `D`
is DC replenishment capacity, and `W` is unseasonal heat. All exogenous terms are independent.

Two features carry the lesson. The direct price response is large: `0.8`, so a 10 point discount
raises sales by 800 units per store-week before any supply response. The availability response is
also large and negative: `-0.6`, so the same discount lowers availability by 0.6 index points, and
each point of lost availability costs 900 units. The **total effect** is therefore
`0.8 + (0.9 × -0.6) = 0.26` thousand units per store-week per 10 points. The promotion works, and
two-thirds of what it wins on price it loses on the shelf.

### 6.2 The graph

```text
   (C) competitor price ----> (P) promotion ----> (U) units sold
                                  |    \             ^
                                  |     \            |
                                  v      +-----------+
                             (A) availability
                                  ^
                                  |
                    (D) DC capacity
                    (S) supplier disruption ----------> (U)
                    (H) store baseline ----> (P) and (H) ----> (U)
                    (W) unseasonal heat --------------------> (U)

   causal paths:   P -> U ;  P -> A -> U
   back-door:      P <- C -> U ;  P <- H -> U
   collider at A:  P -> A <- S -> U   (opened if A is conditioned on)
```

### 6.3 The covariate audit

The analyst's eight candidates, classified against the graph before any model is fitted.

| Warehouse column | Graph role | Adjust for total effect? | Reason |
|---|---|---|---|
| `competitor_price` | Confounder `C` | Yes | Blocks `P <- C -> U` |
| `store_baseline` | Confounder `H` | Yes | Blocks `P <- H -> U`; also predicts the outcome |
| `on_shelf_availability` | Mediator on `P -> A -> U`; collider with `S` | No | Blocks the indirect path and opens `P -> A <- S -> U` |
| `stockout_flag` | Descendant of `U`, collider of `P` and `U` | No | Conditioning selects on the outcome |
| `dc_capacity` | Cause of `A` only, not of `P` | No | Neutral for the total effect; adds nothing |
| `unseasonal_heat` | Cause of `U` only | Optional | Neutral precision variable; legitimate if measured before the window |
| `gross_margin` | Descendant of `P` and `U` | Never | Changes the estimand and induces selection bias |
| `basket_size` | Descendant of `U` | Never | Conditioning on a consequence of the outcome |

Six of the eight are defensible only if the analyst can say which of the four roles applies. Two
are never defensible. The audit takes fifteen minutes and prevents the most expensive error in
the project.

### 6.4 Five specifications

Estimates are population values from the model above, so they are exact rather than sampled. A
real dataset of 240 stores would add sampling error of roughly ±0.1 to each.

| # | Specification | Estimate | What it actually estimates |
|---|---|---:|---|
| S1 | `U ~ P` | 0.94 | Total effect plus confounding by `C` and `H` |
| S2 | `U ~ P + C + H` | 0.26 | **Total effect — the target** |
| S3 | `U ~ P + C + H + A` | 0.43 | Direct effect plus collider bias through `A` |
| S4 | `U ~ P + C + H + A + S` | 0.80 | Direct effect: the promotion with availability held fixed |
| S5 | `U ~ P + C + H + W` | 0.26 | Total effect, slightly more precise |
| S6 | `U ~ P + C + H + G` | not reported | Not interpretable; `G` is a descendant of both |

The naive estimate overstates the total effect by a factor of 3.6. The most common "careful"
specification, S3, overstates it by a factor of 1.7 while appearing to be the most rigorous,
because it controls for more variables. Neither error is visible in the regression output.

The decomposition of S3 is instructive and is exact in this linear model:

```text
S3 = direct effect + collider bias
   = 0.80 + (-1.0 x +0.372)
   = 0.80 - 0.372
   = 0.428

   where +0.372 is the coefficient of P in the auxiliary regression of S on P and A:
   conditioning on A makes promotion and disruption positively associated,
   and disruption lowers sales, so the induced bias is negative.
```

Two biases with opposite signs partly cancel. Adjusting for `A` blocks the negative indirect path,
pushing the estimate up from 0.26 to 0.80; conditioning on the collider pushes it back down by
0.37. The net 0.43 is closer to the truth than 0.80 or 0.94 — which is what makes the mistake so
hard to catch: the answer looks reasonable.

> **Note:** The two components happen to be additive here because the model is linear with no
> interaction. In a nonlinear or interacting system they are not, and the net direction of
> adjusting for a mediator-collider cannot be signed from the graph alone. Do not carry the
> arithmetic of this example into a system that violates its assumptions.

### 6.5 What the numbers mean for the decision

With a 15 point discount, `P = 1.5`:

| Quantity | Calculation | Result |
|---|---|---:|
| Naive estimate | 0.94 × 1.5 | 1,410 units per store |
| Total effect | 0.26 × 1.5 | 390 units per store |
| Availability-adjusted | 0.43 × 1.5 | 640 units per store |
| Direct effect at fixed availability | 0.80 × 1.5 | 1,200 units per store |
| Units gained on price | 1,200 | +1,200 |
| Units lost to stock-out | 0.9 × 0.6 × 1.5 × 1,000 | −810 |
| Net | 1,200 − 810 | +390 |

The decision-relevant reading is not "the promotion produced 390 units". It is that the price
response is 1,200 units and the supply system gives back 810 of them. If the retailer can pre-build
or replenish through the promotion, the direct effect is the relevant number and the promotion is
worth roughly three times what the total effect suggests. If it cannot, the total effect is the
relevant number and the promotion is marginal. The two estimates are not in conflict; they answer
the two questions the business might be asking, and the DAG is what tells you which is which.

The naive estimate supports a rollout; the total effect probably does not clear the discount's
margin cost. One number is 3.6 times the other, and both come from the same table.

## 7. Code, query, or calculation

The artefact is the graph plus the adjustment declaration, in a form that can be pasted into a
project appendix and read by a reviewer who does not use the analysis software.

```text
# graph: promotion -> units sold            (supply chain, store-week)
nodes
  P  promotion depth             treatment, 10pp units
  U  units sold per store-week   outcome
  C  competitor price move       observed confounder
  H  store baseline volume       observed confounder
  A  on-shelf availability       mediator; collider given S
  S  supplier disruption         UNOBSERVED (latent)
  D  DC replenishment capacity   observed, cause of A only
  W  unseasonal heat             observed, cause of U only

edges
  C -> P    C -> U
  H -> P    H -> U
  P -> U    P -> A    A -> U
  S -> A    S -> U
  D -> A    W -> U

adjustment  (estimand: total effect of P on U)
  open back-door paths:  P <- C -> U ; P <- H -> U
  blocked by:            { C, H }
  minimal sufficient set { C, H }
  excluded:  A  mediator and collider
             D  neutral for this estimand
             any descendant of P or U
```

Three details in the block are doing work. `S` is drawn as a node even though it is unobserved,
which is what makes the collider at `A` visible. `D` is included to show that a variable can be a
cause of a mediator and still be irrelevant to the total effect. The excluded list is part of the
declaration, not commentary.

## 8. Data, standards, and regulatory context

### 8.1 What governs causal reporting

No standard requires a DAG in commercial supply chain analytics. Several require something close
enough that the absence is a choice rather than an oversight.

| Instrument | Scope | What it asks for | Grade |
|---|---|---|---|
| ICH E9(R1) addendum | Clinical trials, adopted 2019 | An estimand stated before analysis, with sensitivity analyses aligned to it | B |
| Federal Reserve SR 11-7 | US bank model risk, 2011 | Conceptual soundness of a model, documented and independently reviewed | B |
| STROBE-style reporting conventions | Observational epidemiology | Confounders named and the adjustment justified | B |
| DAG reporting guidance (Feeney et al., 2025) | Clinical research methods | Consensus graph pre-registered, reported with the adjustment set and the reasoning | B |
| DAG use review (Tennant et al., 2021) | Applied health research | Graph available, adjustment set stated, estimand stated | A |

The middle rows are the useful analogy. SR 11-7's "conceptual soundness" is, in plain language, the
requirement that someone can explain why the model's structure is the right structure. A DAG is
the cheapest artefact that satisfies that requirement for a causal model, and it is far more
legible to a reviewer than a paragraph of prose.

### 8.2 Tools and file formats

| Tool | What it does | Output | Note |
|---|---|---|---|
| DAGitty (browser) | Draws graphs, finds minimal adjustment sets, identifies biasing paths | Graph, adjustment sets, testable implications | Free and open source; the reference implementation |
| `dagitty` R package | Same algorithms programmatically | Adjustment sets as R objects | Textor et al., 2016 |
| DoWhy / PyWhy | Graph-based identification and estimation | Estimand, estimand, refutation tests | Python; useful for pipelines |
| `networkx` / `graphviz` | Graph construction and rendering | Adjacency structures, images | No causal semantics; you supply them |

The formats matter for reproducibility. A DAG exported as an image cannot be checked. Export the
edge list, keep it in version control with the analysis code, and render the figure from the edge
list so the two cannot diverge.

### 8.3 The measurement consequence

A DAG is a measurement plan as well as an analysis plan. Every node that matters and is not
measured is a data request, and the graph is the argument for making it. This is the practical
value of drawing the graph early: it converts "we should have more data" into "we need the
competitor's price index for these 240 stores in these eight weeks", which is a request someone
can act on.

There is a countervailing pressure worth naming. Data-minimisation principles in privacy
regulation discourage collecting variables that are not needed. A minimal sufficient adjustment
set is a defensible answer to "why are you collecting this", and a covariate list assembled by
habit is not. The graph serves both goals at once: it justifies the variables that are in it and
the variables that are deliberately out.

## 9. Industry practice and case evidence

### 9.1 How often graphs are actually reported

Tennant and colleagues reviewed 234 applied health research articles that used DAGs. The
proportions are the best available calibration of practice, and they are not flattering.

| Practice | Share of reviewed articles | Grade |
|---|---:|---|
| Made the DAG available to the reader | 62% | A |
| Reported the adjustment set implied by the DAG | 49% | A |
| Stated the estimand | 21% | A |
| Reported results from the DAG-implied set without modification | Not reported in the summary consulted | A |

Half the articles that used a graph did not report the set it implied, and four in five did not
state what they were estimating. The graph was a thinking tool, discarded at reporting time —
precisely when it would have been most useful to a reader.

### 9.2 The catalogue of good and bad controls

Cinelli, Forney and Pearl (2022) is the most useful single teaching document for this topic. It
works through eighteen canonical models with linear algebraic derivations in an appendix, and the
taxonomy is worth memorising.

| Model group | Structure | Verdict |
|---|---|---|
| 1–6 | Block a back-door path, including via a mediator's confounder | Good controls |
| 7 | M-bias: pre-treatment collider | Bad, and the variation is undecidable |
| 8, 13 | Cause of outcome only, uncorrelated with treatment | Neutral, possibly good for precision |
| 9 | Cause of outcome only, correlated with treatment | Neutral, possibly bad |
| 10 | Bias amplification; also an instrument | Bad |
| 11–12 | Mediators and descendants of mediators | Over-control |
| 14–15 | Post-treatment, not inherently bad; 15 helps under selection | Context-dependent |
| 16–17 | Selection and collider stratification | Bad |
| 18 | Descendant of the outcome | Bad |

The pedagogical value is the middle rows. They show that "pre-treatment" is not a licence to
adjust and "post-treatment" is not an automatic prohibition; the structure of the paths matters,
not the label on the variable.

### 9.3 Collider bias in the field

The canonical demonstration is the birth-weight paradox. Hernández-Díaz, Schisterman and Hernán
(2006) used 1991 United States linked birth and infant death records, approximately three million
births, and reproduced a puzzle: infants of smoking mothers had higher low-birth-weight rates and
higher overall mortality, yet among low-birth-weight infants, mortality was lower for the infants
of smokers (relative risk 0.79). Birth weight is a collider — smoking reduces it, and unmeasured
causes of low birth weight also raise mortality — so conditioning on it produces a spurious
protective association.

Schisterman, Cole and Platt (2009) give the arithmetic. Using United States linked birth and
infant death data for 1999–2001, with 10,035,444 births after exclusions, the crude relative risk
of neonatal mortality for maternal smoking was 2.49 (95% confidence interval 2.41 to 2.56).
Adjusting for birth weight moved it to 2.03 (1.97 to 2.09), an 18% attenuation. The adjusted number
is not a better total effect; it estimates a different quantity, and is biased for both the total
and the direct effect when an unmeasured common cause exists.

The supply chain analogues are direct and mostly unrecognised.

| Analytic habit | Collider being conditioned on | Likely consequence |
|---|---|---|
| Drop stock-out weeks "to clean the sales data" | Stock-out, caused by promotion and by sales | Promotion effect biased downward |
| Control for DC utilisation | Utilisation, caused by volume and by capacity | Spurious association between demand and capacity constraints |
| Analyse only suppliers that survived the year | Survival, caused by performance and by spend | Cost or quality effects biased |
| Restrict to stores that stayed open | Closure, caused by performance and by format | Format effects biased |
| Control for OTIF when estimating a routing effect | OTIF, caused by routing and by order mix | Routing effect biased, direction unpredictable |

The fifth row catches experienced analysts: OTIF is the outcome everyone trusts and the covariate
everyone reaches for, and if routing and order mix both affect it, controlling for it in a model of
routing is a mistake.

### 9.4 The M-bias argument, and what to do about it

The disagreement is genuine and worth understanding rather than resolving by slogan.

| Position | Claim | Implication for practice |
|---|---|---|
| Pearl | `X <- U1 -> Z <- U2 -> Y` is a real structure; conditioning on `Z` opens a path | Prefer the graph; do not adjust mechanically |
| Rubin | Adjust for all pre-treatment covariates; exact M-structures are rare | Adjust by default; do not curate |
| Ding and Miratrix (2015) | M-bias is typically a higher-order term, small relative to confounding; mild departures from a pure M make adjustment preferable | Adjust, but do not add known post-treatment colliders |
| Liu et al. (2012) | Bias small across 178 simulated scenarios unless associations are very large | Same |

For supply chain work, three rules cover almost every case.

1. **Never adjust for a post-treatment variable unless the estimand requires it.** This is not the
   M-bias case, and there is no controversy about it.
2. **Do not add a pre-treatment variable solely because it looks like a collider.** If it is a
   confounder, omitting it is worse; if it is a pure M-collider, the bias is usually small.
3. **If a pre-treatment variable is caused by two independent factors, one of which affects the
   outcome, flag it.** That is the exact M-structure, the one case that is undecidable without
   magnitudes. Report both estimates.

Most adjustment decisions are easy once the graph is drawn. A handful are not, and the honest
response to those is two numbers and a sentence.

### 9.5 What platforms and vendors do

Commercial causal platforms have converged on graph-based identification, which is a genuine
advance and also a source of overclaiming.

| Source | What it does | Grade |
|---|---|---|
| Amazon Science (2025), end-to-end causal attribution for supply chain operations | Learns a graph from observational key performance indicator data, compares it with a business-defined baseline graph, estimates with a structural causal model, and runs refutation tests | D |
| DoWhy documentation, online shop attribution example | A runnable graph of shopping events, spend, price, units, revenue, and profit, with counterfactual queries | C |
| Commercial "causal AI" platforms generally | Emphasise graph discovery and automated adjustment | D |

The Amazon framework keeps a business-defined baseline graph alongside the learned one, which is
the right instinct: an algorithm can propose edges, but only a human can say whether one is a
mechanism or a sampling artifact. A learned graph that contradicts domain knowledge is evidence
about the data, not about the world.

> **Note:** Grades D and C here reflect provenance, not quality. Vendor descriptions of graph
> discovery omit the assumptions that make discovery work — causal sufficiency, faithfulness, and
> often linearity — and those assumptions are exactly what a practitioner needs to check.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Covariate list from availability | The list matches the warehouse schema exactly | Adjustment is whatever was joinable | Confounders omitted, colliders included | Draw the graph first; issue data requests from it |
| Adjusting for a mediator | "We controlled for availability / utilisation / lead time" | Blocks the indirect path | Total effect understated, sometimes sign-flipped | State total versus direct before choosing covariates |
| Adjusting for a collider | "We excluded stock-out weeks" | Opens a closed path | Spurious association, often against the treatment | Mark colliders on the graph and exclude them |
| Graph drawn after the model | No graph in the analysis plan; one appears in the appendix | Edges chosen to match the specification | The graph cannot challenge anything | Pre-register the graph with the estimand |
| Latent variables omitted | Every node in the graph is a column in the data | The graph asserts causal sufficiency | Unmeasured confounding invisible; overconfidence | Draw unmeasured nodes dashed |
| Descendant of outcome included | Basket size, margin, or returns in the covariate list | Conditioning on a consequence of the outcome | Selection bias of unpredictable sign | Timestamp every covariate; exclude post-outcome variables |
| Table 2 fallacy | Every coefficient in the model is interpreted | Covariate coefficients are direct effects in a model built for a total effect | False causal claims about secondary variables | Report only the exposure coefficient; footnote the rest |
| "Baseline" measured after treatment | Covariate date not checked against assignment date | Post-treatment variable treated as pre-treatment | Mediator or collider bias, invisible | Require a timestamp per covariate |
| Feedback drawn as a DAG | A cycle in the graph, or a loop silently broken | Acyclic algorithms applied to a dynamic system | Wrong adjustment set for time-varying confounding | Unroll time; use g-methods |
| Learned graph treated as evidence | An algorithm proposes edges and they are accepted | Discovery assumes sufficiency and faithfulness | Confident structure with no mechanism | Compare with a domain graph; keep both |
| Faithfulness assumed silently | A path is expected to show up as a correlation | Cancellation can hide a real path | Graph appears falsified when it is not | Test implications, do not expect every path to show |
| Over-adjustment for precision | Dozens of pre-treatment covariates, no graph | Variance reduction traded for bias amplification | Precise but biased estimate | Justify each covariate by role, not by predictive power |

Two deserve expansion.

**Adjusting for a mediator** survives review because it looks like thoroughness and produces
a plausible number. The tell is the sentence "holding availability constant, the promotion
raised sales by X". That is a direct effect, and it may be what is wanted — but only if
someone asked for it. The failure is the unannounced change of estimand.

**The Table 2 fallacy** is the reporting counterpart. Westreich and Greenland (2013) show
that in one adjusted model the exposure coefficient estimates a total effect while each
covariate coefficient estimates a controlled direct effect, and can be confounded even when
the exposure estimate is not. A dashboard of "drivers" from one regression commits this
error at scale.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| Causal Markov condition | Every variable is independent of its non-descendants given its parents | Selection into the sample; unmeasured common causes | Not directly testable; implied independencies can be checked |
| Faithfulness | Observed independencies correspond to missing paths | Cancelling paths; small samples | Test the implied independencies; treat failure as ambiguous, not refuting |
| Correct node set | Each node is one well-defined construct | Collapsing distinct variables; measurement error | Audit definitions; split nodes where mechanisms differ |
| Correct edge set | Every direct causal effect is drawn, and no others | Omitted mechanisms; reversed arrows | Domain review; compare with a second analyst's graph |
| Acyclicity | Path algorithms require no cycles | Feedback, bullwhip, inventory loops | Unroll time; check for cycles before analysis |
| Time ordering is known and respected | Edges must point forward | Covariates measured after treatment | Timestamp covariates; compare with assignment date |
| No interference (SUTVA) | Standard adjustment assumes units do not affect each other | Shared demand, shared supply, allocation | Spillover tests; network exposure mapping |
| Positivity | Every covariate pattern has a chance of each treatment | Deterministic assignment rules | Overlap diagnostics |
| Consistency | The treatment is one well-defined thing | Several versions of "the promotion" | Audit treatment definitions across units |
| The graph is stable over the window | One graph applies to the whole period | Process change, new system, policy shift | Split the window; re-estimate |

Two limits are worth stating plainly.

**The graph is not testable as a whole.** The data can reject it by showing a dependence it
says should be absent, but acceptance is not confirmation: many graphs imply the same
independencies. Two analysts can hold different graphs, fit the same data, and both pass
every test. The graph is an argument, settled by evidence and judgement together.

**The graph does not produce the number.** It says which variables to condition on. It does
not supply the functional form, the distribution, the standard error, or the magnitude of
any remaining bias.

## 12. Exercises

### Exercise 1 — Draw the graph (application, 35 minutes)

Choose a causal question from your own work with a named treatment and a named outcome. Draw the
DAG from memory, in temporal layers, with at least eight nodes.

**Deliverable:** a hand-drawn or text graph, an edge list, and one mechanism sentence per edge.
**Acceptance criteria:** every edge has a mechanism; at least one unmeasured variable drawn latent;
treatment and outcome marked; no node is a vague category such as "demand" left undefined.
**Time:** 35 minutes.

### Exercise 2 — Classify the covariates (analysis, 40 minutes)

Using the eight-column covariate list in section 6.3 and the graph in section 6.2, assign each
column one of: confounder, proxy, neutral, mediator, collider, descendant of treatment, descendant
of outcome, selection variable. State the action for a total effect and the bias if it is wrong.

**Deliverable:** an eight-row table with role, action, and bias direction or "not determinable".
**Acceptance criteria:** every role justified by a path, not by correlation with the outcome; at
least one variable marked "not determinable" with the reason; both "never" columns identified.
**Time:** 40 minutes.

### Exercise 3 — Derive the adjustment set (calculation, 50 minutes)

A procurement team asks whether supplier consolidation reduces cost per unit. Volume `V` and
complexity `K` cause both consolidation `X` and cost `Y`; capability `L` causes both; compliance
`M` is caused by `X` and causes `Y`; status `R` is caused by `X` and `L` and causes `Y`; congestion
`W` causes `Y` only. The sample is restricted to categories with three or more suppliers.

**Deliverable:** the graph, every back-door path, the minimal sufficient set, and each exclusion.
**Acceptance criteria:** all three back-door paths listed; the descendant clause applied to `M`
and `R`; minimality argued, not asserted; the selection restriction discussed.
**Time:** 50 minutes.

### Exercise 4 — Find the collider (critique, 35 minutes)

An analyst estimating the effect of a warehouse automation programme on pick rate excludes all
weeks in which the DC was above 90% utilisation, "because those weeks are not representative".

**Deliverable:** the graph, a paragraph on the induced association, and the corrected analysis.
**Acceptance criteria:** the collider identified with both its causes; the direction of the induced
association reasoned; the corrected analysis names the estimand the restriction implies.
**Time:** 35 minutes.

### Exercise 5 — Total versus direct effect (calculation, 50 minutes)

Using section 6.1, compute the total, direct, and indirect effects. Then compute what adjusting for
availability returns and decompose the gap into the mediator and collider components.

**Deliverable:** the three effects, the adjusted estimate, and the decomposition, each with units.
**Acceptance criteria:** arithmetic shown; the signs of both components explained; a sentence
naming the estimand the business should want and why.
**Time:** 50 minutes.

### Exercise 6 — When nothing is identifiable (design, 50 minutes)

A retailer wants the effect of a supplier development programme on defect rate. The programme was
targeted at suppliers the buyer already believed promising, and no measure of that belief exists.
A plausible mediator is audit frequency.

**Deliverable:** the graph with the latent node, a statement that no adjustment set exists, and an
assessment of the front-door criterion through audit frequency.
**Acceptance criteria:** unmeasured confounding shown on every back-door path; the three front-door
conditions checked one by one; a fallback design named if the criterion fails.
**Time:** 50 minutes.

### Exercise 7 — Critique a causal claim (critique, 45 minutes)

Take a vendor case study, a published paper, or an internal deck reporting a causal effect. Extract
its covariate list and reconstruct the graph it implies.

**Deliverable:** the reconstructed graph, variables that should have been in and were not, ones
that should not have been and were, and a corrected claim.
**Acceptance criteria:** every judgement tied to a path; "wrong" distinguished from "answers a
different question"; at least one claim the original gets right.
**Time:** 45 minutes.

### Exercise 8 — Write the adjustment declaration (design, 50 minutes)

Write the six-part declaration from section 5.8 for the question in Exercise 1.

**Deliverable:** one page with estimand, graph, adjustment set, exclusions with reasons,
identification argument, and pre-specified check.
**Acceptance criteria:** a reason for every variable a reviewer would expect to see; the
identification argument names the assumption most likely to fail; the check runs first.
**Time:** 50 minutes.

## 13. Self-check answers

### 13.1 Exercise 1 rubric

A strong answer has nodes that are single constructs, edges with mechanisms, and at least one
latent node. It places the treatment and outcome in a temporal order and shows an unmeasured
variable on a back-door path. A weak answer has a node called "demand" or "operations" that
aggregates three mechanisms, edges justified by correlation, and no latent variables — which is a
graph asserting that nothing important is unmeasured.

### 13.2 Exercise 2 answers

| Column | Role | Action | Bias if wrong |
|---|---|---|---|
| `competitor_price` | Confounder | Include | Omitted: estimate absorbs competitor effect, upward here |
| `store_baseline` | Confounder and precision variable | Include | Omitted: selection on strong stores inflates the estimate |
| `on_shelf_availability` | Mediator and collider | Exclude for total effect | Included: blocks −0.54 indirect path, opens collider path −0.37 |
| `stockout_flag` | Descendant of outcome and collider | Exclude | Included: spurious negative association between promotion and sales |
| `dc_capacity` | Cause of mediator only | Exclude | Neutral; no bias, no gain for the total effect |
| `unseasonal_heat` | Cause of outcome only | Optional | Neutral; improves precision if measured pre-treatment |
| `gross_margin` | Descendant of treatment and outcome | Never include | Changes the estimand; induces selection bias |
| `basket_size` | Descendant of outcome | Never include | Selection bias of unpredictable sign |

### 13.3 Exercise 3 — derivation of the adjustment set

Draw the graph and enumerate every path from `X` to `Y`.

| # | Path | Type | Open? |
|---:|---|---|---|
| 1 | `X <- V -> Y` | Back-door | Yes |
| 2 | `X <- K -> Y` | Back-door | Yes |
| 3 | `X <- L -> Y` | Back-door | Yes |
| 4 | `X -> M -> Y` | Causal | Yes; this is the effect |
| 5 | `X -> R <- L -> Y` | Collider at `R` | No, unless `R` is conditioned on |

The back-door criterion requires a set that blocks paths 1–3 and contains no descendant of `X`.
`{V, K, L}` does this, and each member is necessary: `V` blocks path 1, `K` blocks path 2, `L`
blocks path 3, and no member blocks another's path. It is therefore a minimal sufficient
adjustment set.

Three exclusions carry the lesson. `M` is a descendant of `X` on the causal path, so adjusting for
it violates clause 1 and changes the estimand from total to direct. `R` is also a descendant of
`X`, and it is a collider on path 5, so adjusting for it both violates clause 1 and opens a path
that was closed — two independent reasons. `W` is neither a confounder nor a descendant; including
it is harmless but not minimal, and it should be reported as a precision choice rather than as
part of the adjustment set.

The sample restriction is separate. If inclusion requires three or more suppliers, and both
consolidation and cost influence how many suppliers a category has, the sample is conditioned on a
variable that may be a collider. The estimand is then the effect in categories with three or more
suppliers, which is a legitimate but narrower question, and the write-up must say so.

### 13.4 Exercise 4 answer

Utilisation is caused by volume and by capacity. Conditioning on high utilisation — by restricting
the sample to weeks above 90% — opens the path `automation -> utilisation <- volume -> pick rate`,
where volume is any demand-side driver of pick rate. Among high-utilisation weeks, automation and
volume become negatively associated, so the automation effect is biased downward. The correction
is either to include all weeks and control the confounders of the automation decision, or to
restate the estimand for high-utilisation weeks and say so.

### 13.5 Exercise 5 arithmetic

```text
total effect       = 0.8 + (0.9 x -0.6)          = 0.26
direct effect      = 0.8
indirect effect    = 0.9 x -0.6                  = -0.54
adjusted for A     = 0.8 + collider bias
collider bias      = -1.0 x (+0.372)             = -0.372
adjusted for A     = 0.428

mediator component = 0.8 - 0.26                  = +0.54
collider component = 0.428 - 0.8                 = -0.372
net                = 0.26 + 0.54 - 0.372          = 0.428
```

The business should want the total effect, 0.26, because the decision is whether to run the
promotion through the supply system as it actually operates. The direct effect, 0.8, is the right
number only if the retailer commits to pre-building or expediting, which is a different decision
with its own cost.

### 13.6 Exercise 6 — the front-door derivation

The graph has `U -> X`, `U -> Y`, `X -> M -> Y`, with `U` latent. Every back-door path from `X` to
`Y` passes through `U`, so no observed adjustment set identifies the effect. Check the front-door
criterion through `M`.

1. `M` intercepts every directed path from `X` to `Y`: yes, `X -> M -> Y` is the only one.
2. No unblocked back-door path from `X` to `M`: the only path into `X` is from `U`, and `U` reaches
   `M` only through `X`, so no such path exists.
3. Every back-door path from `M` to `Y` is blocked by `X`: `M <- X <- U -> Y` is blocked by `X`.

The criterion holds. With binary variables and the conditional probabilities below, the front-door
formula gives the effect.

```text
P(M=1|X=1)=0.80   P(M=1|X=0)=0.30   P(X=1)=0.50
P(Y=1|M=1,X=1)=0.10   P(Y=1|M=1,X=0)=0.15
P(Y=1|M=0,X=1)=0.50   P(Y=1|M=0,X=0)=0.55

do(X=1): 0.80 x (0.5x0.10 + 0.5x0.15) + 0.20 x (0.5x0.50 + 0.5x0.55)
       = 0.80 x 0.125 + 0.20 x 0.525 = 0.205
do(X=0): 0.30 x 0.125 + 0.70 x 0.525 = 0.405
front-door effect = 0.205 - 0.405 = -0.200

naive contrast    = 0.180 - 0.430 = -0.250
```

The naive contrast overstates the benefit by 5 percentage points, because the unmeasured factor
raises both programme participation and defect risk. The front-door estimate is not free: it
assumes no direct effect of `X` on `Y` except through `M`, an assumption as untestable as the
confounding it replaces. What changes is that it is stated.

### 13.7 Exercise 7 rubric

A strong critique reconstructs the graph the analysis implies and finds at least one bad control
and one omitted confounder. It distinguishes a variable that is wrong from one that answers a
different estimand, and it acknowledges what the original got right. A weak critique lists
"omitted variable bias" without naming a variable, or attacks the sample size.

### 13.8 Exercise 8 rubric

A strong declaration is one page, has an edge list as well as a figure, and lists exclusions with
reasons. Its identification argument names the assumption most likely to fail and the check that
would detect it. A weak declaration is a paragraph in the limitations saying "residual confounding
cannot be excluded", which is true of every observational study and therefore informative about
none.

## 14. Interview questions

**1. "What is a DAG, and why would you draw one?"** Strong: a directed acyclic graph whose arrows
are causal assumptions, drawn to choose variables and expose non-identifiability; a statement of
assumptions, not a diagram of results. Weak: "a flowchart of the process".

**2. "What is the difference between a confounder and a collider?"** Strong: a common cause, whose
adjustment blocks a path, versus a common effect, whose adjustment opens one; one variable can be
either. Weak: "a confounder affects both variables", which describes a collider too.

**3. "Why not control for everything you have?"** Strong: adjustment is not monotone in
information; mediators change the estimand, colliders and descendants of the outcome add bias.
Weak: "multicollinearity", a precision problem rather than the main issue.

**4. "Your model controls for on-shelf availability. What have you estimated?"** Strong: a direct
effect holding availability fixed, because it is a mediator; and collider bias if demand also
causes it. Weak: "a better estimate because availability affects sales".

**5. "What is the back-door criterion?"** Strong: a set blocking every path with an arrow into the
treatment, containing no descendant of it; then the adjustment formula identifies the total effect.
Weak: "control for confounders", which is the thing being defined.

**6. "When can you not identify an effect, even with a lot of data?"** Strong: when every back-door
path runs through an unmeasured variable; the options are a proxy, an instrument, a front-door
path, a design change, bounds, or declining the claim. Weak: "when the data are observational".

**7. "What is M-bias, and would you adjust for the variable?"** Strong: a pre-treatment collider of
two unmeasured causes; the bias is usually small, the debate is real, and the decision turns on
whether the variable might also confound. Weak: "always adjust for pre-treatment variables".

**8. "What is the Table 2 fallacy?"** Strong: in one adjusted model the exposure coefficient is a
total effect while covariate coefficients are controlled direct effects and may be confounded even
when the exposure is not. Weak: "putting too many variables in Table 2".

**9. "How would you handle the bullwhip in a DAG?"** Strong: unroll time so each node is indexed by
period; the graph stays acyclic, and time-varying confounding then needs g-methods. Weak: "DAGs
cannot handle feedback", true only of the compact form.

**10. "A vendor says their platform discovers the causal graph. What do you ask?"** Strong: which
assumptions discovery uses, sufficiency, faithfulness, often linearity, and whether the learned
graph was compared with a domain graph. Weak: "how accurate is it".

## 15. Key terms

- **DAG** — a directed acyclic graph whose edges are causal assumptions.
- **Node** — a variable or event in the graph.
- **Edge** — a directed arrow asserting a direct causal effect.
- **Path** — any sequence of adjacent nodes, ignoring direction.
- **Causal path** — a directed path from treatment to outcome.
- **Back-door path** — a path whose first edge points into the treatment.
- **Collider** — a node with two arrows pointing into it on a path.
- **d-separation** — the criterion for a set of nodes blocking every path between two nodes.
- **Adjustment set** — the variables conditioned on to identify an effect.
- **Minimal sufficient adjustment set** — a sufficient set with no redundant member.
- **Back-door criterion** — blocks all paths into the treatment, contains no descendant of it.
- **Front-door criterion** — identifies an effect through a mediator despite unmeasured confounding.
- **Confounder** — a member of a minimal sufficient adjustment set; a common cause in the simple case.
- **Mediator** — a variable on the directed path from treatment to outcome.
- **Over-adjustment** — conditioning on a mediator or its descendant when a total effect is wanted.
- **Bad control** — a variable whose inclusion increases bias in the target estimand.
- **M-bias** — bias from conditioning on a pre-treatment collider of two unmeasured causes.
- **Collider stratification bias** — the bias induced by conditioning on a collider.
- **Endogenous selection bias** — selection into the sample driven by treatment and outcome.
- **Identifiability** — whether the estimand is computable from the observed distribution.
- **Causal sufficiency** — no unmeasured common cause of any two observed variables.
- **Faithfulness** — every independence in the data corresponds to a missing path.
- **Table 2 fallacy** — reading every coefficient in one model as a causal effect.
- **Adjustment declaration** — the written graph, set, exclusions, and identification argument.

## 16. Sources

**Method and theory — Grade A**

- Pearl, J. and Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic
  Books. The back-door idea and the do-operator for a general reader.
- Pearl, J., Glymour, M. and Jewell, N. P. (2016). *Causal Inference in Statistics: A Primer*.
  Wiley. Definition 2.4.1 (d-separation) and Definition 3.3.1 (back-door criterion).
- Cinelli, C., Forney, A. and Pearl, J. (2022). "A Crash Course in Good and Bad Controls."
  *Sociological Methods & Research*, 53(3), 1071–1104. DOI: 10.1177/00491241221099552. Eighteen
  canonical models with algebraic derivations, matching the numbering in section 9.2.
- VanderWeele, T. J. and Shpitser, I. (2013). "On the Definition of a Confounder." *Annals of
  Statistics*, 41(1), 196–220. DOI: 10.1214/12-AOS1058.
- Schisterman, E. F., Cole, S. R. and Platt, R. W. (2009). "Overadjustment Bias and Unnecessary
  Adjustment in Epidemiologic Studies." *Epidemiology*, 20(4), 488–495.
  DOI: 10.1097/EDE.0b013e3181a819a1.
- Westreich, D. and Greenland, S. (2013). "The Table 2 Fallacy: Presenting and Interpreting
  Confounder and Modifier Coefficients." *American Journal of Epidemiology*, 177(4), 292–298.
  DOI: 10.1093/aje/kws412.
- Elwert, F. and Winship, C. (2014). "Endogenous Selection Bias: The Problem of Conditioning on a
  Collider Variable." *Annual Review of Sociology*, 40, 31–53.
  DOI: 10.1146/annurev-soc-071913-043455.
- Hernández-Díaz, S., Schisterman, E. F. and Hernán, M. A. (2006). "The Birth Weight 'Paradox'
  Uncovered?" *American Journal of Epidemiology*, 164(11), 1115–1120. DOI: 10.1093/aje/kwj275.
- Ding, P. and Miratrix, L. W. (2015). "To Adjust or Not to Adjust? Sensitivity Analysis of M-Bias
  and Butterfly-Bias." *Journal of Causal Inference*, 3(1), 41–57. DOI: 10.1515/jci-2013-0021.
- Liu, W., Brookhart, M. A., Schneeweiss, S., Mi, X. and Setoguchi, S. (2012). "Implications of
  M-Bias for Healthcare Studies." *American Journal of Epidemiology*, 176(10), 938–948.

**Reviews and reporting guidance — Grade A/B**

- Tennant, P. W. G. et al. (2021). "Use of Directed Acyclic Graphs (DAGs) to Identify Confounders
  in Applied Health Research: Review and Recommendations." *International Journal of Epidemiology*,
  50(2), 620–632. Review of 234 articles; reporting checklist.
- Feeney, T., Hartwig, F. P. and Davies, N. M. (2025). "How to Use Directed Acyclic Graphs: Guide
  for Clinical Researchers." *BMJ*, 388, e078226. DOI: 10.1136/bmj-2023-078226. Seven-step
  process and pre-registration guidance.

**Tools — Grade B**

- Textor, J., van der Zander, B., Gilthorpe, M. K., Liskiewicz, M. and Ellison, G. T. H. (2016).
  "Robust Causal Inference Using Directed Acyclic Graphs: The R Package 'dagitty'."
  *International Journal of Epidemiology*, 45(6), 1887–1894.

**Operations management — Grade A**

- Xia, Y., Reza, S., Lim, N. and Ho, T.-H. (2017). "OM Forum — Causal Inference Models in
  Operations Management." *Manufacturing & Service Operations Management*, 19(4), 509–525.
  DOI: 10.1287/msom.2017.0659.

**Industry, vendor, and documentation — Grade C/D**

- Amazon Science (2025). "An End-to-End Causal Modeling Framework for Advanced Attribution in
  Supply Chain Operations." Applied pipeline with a business-defined baseline graph. Grade D.
- DoWhy documentation, "Causal Attributions and Root-Cause Analysis in an Online Shop." Runnable
  graph-based attribution example. Grade C.

**Press and anecdote — Grade E**

- No press item or anecdote supports a quantitative claim in this topic. Every number in sections
  6, 9.3, and 13 is either taken from a cited study or constructed arithmetic, labelled as such.

**Note on citations.** Two citations commonly attributed in this area are corrected here.
Westreich and Greenland (2013) is the Table 2 fallacy paper, not a paper on the definition of a
confounder; the definition paper is VanderWeele and Shpitser (2013). The DAGitty methods paper is
in the *International* Journal of Epidemiology (2016), not the *American* one; the earlier 2011
item is a letter in *Epidemiology*. Volume and page numbers are given only where the sources
consulted confirmed them.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Prerequisite, written: [`01_estimands-question-types.md`](01_estimands-question-types.md)
- Written sibling: [`02_probability-bootstrap.md`](02_probability-bootstrap.md)
- Written sibling: [`03_hypothesis-tests.md`](03_hypothesis-tests.md)
- Written sibling: [`04_regression-time-series.md`](04_regression-time-series.md)
- `05_autocorrelation-robust-errors.md` (planned) — variance, not identification
- `07_experiments-quasi.md` (planned) — design as the alternative to adjustment
- `08_difference-in-differences.md` (planned) — a design that does not need the graph's back doors closed by covariates
- `09_elasticity-endogeneity.md` (planned) — the canonical simultaneity case
- `10_sensitivity-omitted-bias.md` (planned) — how much latent confounding would overturn the estimate
- Survey file: [`../../../course/04_statistics_causal_inference.md`](../../../course/04_statistics_causal_inference.md)
- Planning context for the operational nodes: [`../01_supply_chain_foundations/04_planning-processes.md`](../01_supply_chain_foundations/04_planning-processes.md)
- Return to the foundations map: [`../_index.md`](../_index.md)
