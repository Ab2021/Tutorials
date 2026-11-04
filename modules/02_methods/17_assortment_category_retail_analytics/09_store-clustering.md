# Store Clustering and Localisation

> Module 17 — Assortment, Category, and Retail Analytics  
> Topic 09 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Module 04, Module 16, and topics 01 and 03  
> Version 2026.5

## 1. Why this topic exists

A retailer rarely has the resources to build a separate range and planogram for every store.
A single national range is also rarely right for every store.
A neighborhood store serving frequent small trips may need a different mix from a commuter store with the same floor area.
A store cluster is a practical compromise: use shared decisions where needs are similar and local exceptions where they are not.

The category manager must decide which need states receive scarce shelf space in each store.
The space planner must fit that choice into a particular fixture.
The supply planner must replenish the chosen items at acceptable availability and cost.
The field team must execute resets without an unmanageable number of variants.
Finance must decide whether the expected incremental contribution pays for this additional complexity.

Grouping stores by postal code or income alone does not answer those questions.
Those variables may describe the surrounding population but not what customers actually buy in the category.
Conversely, sales alone reflect the range and stock that the retailer chose to offer.
A store cannot sell a product that it did not carry, and observed low sales during an outage are not proof of low demand.
The planning problem is to extract useful demand signals while recognising where historic decisions censor them.

A cluster is not a permanent natural type of store.
It is a versioned decision aid for a defined category, store population, season, and planning cycle.
The same store may need different groups for coffee, baby care, and pet food.
The assortment assigned to a cluster remains subject to store-specific space, regulation, supply, and shopper-need constraints.

The expensive error is not simply choosing an imperfect clustering algorithm.
It is producing an attractive segmentation that cannot be executed or whose differences are caused by prior assortment policy.
A second error is treating improved fit to historical sales as evidence of incremental profit.
Both require a comparison against a feasible current policy and a measured test of the proposed change.

This lesson uses an illustrative chilled-meal example.
Its figures are invented for teaching, in United States dollars (USD), and do not describe a real retailer.
The example separates an initial demand description, a forecast of benefit, a proposed plan, a target, and an approved commitment.
None of those items is a substitute for the others.

## 2. Learning outcomes

After studying this topic you can:

1. **Define** a store-category planning population, demand window, and feasible decision; exercise 1 assesses the definition.
2. **Calculate** a demand-mix feature and explain sales censoring; exercises 2 and 3 assess the calculation and critique.
3. **Choose** and defend a clustering approach, distance, and candidate number of clusters; exercise 4 assesses the choice.
4. **Measure** assignment stability and distinguish it from causal impact; exercise 5 assesses the calculation.
5. **Translate** cluster profiles into an executable assortment while respecting shelf, supply, and coverage constraints; exercise 6 assesses the translation.
6. **Compute** a pilot difference-in-differences estimate and net contribution; exercise 7 assesses the economics.
7. **Design** a credible pilot, rollout controls, and exception process; exercise 8 assesses the design.
8. **Critique** an equity, privacy, and new-store deployment scenario; exercise 9 assesses governance.

A defensible recommendation may retain the current range, run a limited experiment, or delay clustering until availability data improve.
A cluster label by itself is not a commercial outcome.
Every outcome above is assessed in the exercises and explained in the answer section.

## 3. The core mental model

```text
Category decision + planning horizon + constraints
                      |
                      v
Store-week sales + availability + eligible range + context
                      |
                      v
Comparable demand features and quality / uncertainty flags
                      |
          +-----------+------------+
          v                        v
   Candidate clusters       Operational feasibility
   and stable profiles      space, supply, rules, cost
          |                        |
          +-----------+------------+
                      v
       Cluster-level range + store-level exceptions
                      |
                      v
      Approved plan -> execution and availability
                      |
                      v
    Randomized or defensible comparison with current policy
                      |
                      v
 Incremental category contribution - implementation cost
                      |
                      v
         Scale, change, retain current policy, or stop
                      |
                      +----> monitored next planning cycle
```

The diagram starts with a decision, not an algorithm.
The decision fixes the category, store population, planning cadence, performance objective, and binding limits.
Changing the decision can change the appropriate features and cluster count.
A group useful for shelf allocation may not be useful for delivery routing.

Sales data enter as observations of a system, not as pure expressions of preference.
A shopper can buy an item only if it is ranged, delivered, visible, in stock, and appropriately priced.
Treat missing or unreliable availability as uncertainty, not as a convenient zero.
If multiple stores had different ranges, their sales shares may reflect different opportunities to purchase.

Features compress the evidence to a store-category profile.
For example, the share of comparable category units in ready-to-eat meals describes a demand mix, not store size.
Units per trading day describe volume, not mix.
A two-feature model can use both, but feature scales and weights determine which differences count.

Clustering groups similar feature vectors under a chosen distance and a chosen number of groups.
It does not establish that changing a store's assortment will change its outcome.
Profiles explain what is different about groups; they are not permission to assume every individual store behaves like its centroid.
A store close to a boundary needs more scrutiny than a store clearly embedded in a group.

A cluster-level proposition is translated into an executable store plan.
It needs approved item eligibility, minimum need-state coverage, physical capacity, supplier service, and a replenishment design.
Some stores will require explicit exceptions.
An exception is not a failure if it is documented, owned, reviewed, and included in economics.

The evaluation compares the proposed policy with the policy that would otherwise have run.
Randomization provides the clearest way to address unobserved differences when it is feasible.
Sales and gross margin can shift between items without increasing category or retailer contribution.
A result is decision-useful only when its outcome scope and costs match the decision.

Three feedback loops matter.
First, poor availability can distort the next cluster refresh if not corrected or flagged.
Second, local changes can alter actual demand patterns, so old profiles may become stale.
Third, frequent group changes create resets and disrupt customers, even if each change improves an algorithmic score slightly.
Control refresh cadence and require a business reason for reassignment.

## 4. Definitions and vocabulary

| Term | Definition and unit or coding | Common misuse |
|---|---|---|
| Store | A trading location with a persistent identifier and effective dates | Treating a moved or rebuilt location as necessarily unchanged |
| Category | A defined commercial grouping of items, versioned over time | Comparing shares before aligning category definitions |
| Store-category | One store and one category for a defined period | Assuming one group per store is optimal for every category |
| Stock-keeping unit (SKU) | Retailer-defined sellable item or variant; identifier | Equating a pack-size change with a continuous item without mapping |
| Assortment | Items authorised for a location and time window | Assuming all authorised items were on shelf |
| Need state | Shopper problem or occasion that an item can serve | Inferring that all products within a sales segment are substitutes |
| Demand mix | Relative demand across defined segments; dimensionless share | Treating a share as total demand or contribution |
| Observed sales | Completed transactions, in units or USD per period | Calling censored sales unconstrained demand |
| Lost sales | Unobserved purchases not made because supply or choice was unavailable | Adding an unsupported estimate as if it were measured |
| Availability | Item can be purchased during a specified opportunity; percentage | Equating a positive end-of-day inventory snapshot with all-day shelf availability |
| Eligible exposure | Time or store-days when an item could have been sold under a defined rule | Including pre-launch days in an item's sales rate denominator |
| Trading day | A day when the store was open for business | Including closure days as zero-demand observations |
| Store-week | One store in one calendar or fiscal week | Summing different week definitions without alignment |
| Baseline | The pre-intervention reference period or comparator policy, as specified | Treating any historic value as a counterfactual |
| Feature | A measured or derived input to clustering, with a unit and time window | Including target-period outcomes that would not be known at deployment |
| Feature vector | Ordered feature values for one store-category | Using a vector whose dimensions change silently between runs |
| Normalisation | Transformation to put quantities on comparable scales | Assuming all normalisation schemes encode the same business priorities |
| Standardisation | Subtract a fitted location and divide by a fitted scale | Refitting scaling independently on future stores and moving the coordinate system |
| Distance | A chosen measure of dissimilarity between feature vectors | Calling a numerical distance meaningful without checking its scale and dimensions |
| Cluster | A set of stores grouped under a particular method and configuration | Treating a cluster as an intrinsic permanent identity |
| Centroid | Coordinate-wise mean of member feature vectors | Assuming the mean is an actual store or always a feasible range |
| Medoid | Observed member minimising chosen within-group dissimilarity | Calling any representative store a mathematically selected medoid |
| Assignment | Mapping from an eligible store to a particular cluster version | Comparing raw label numbers across independently fitted runs |
| Boundary store | Store whose nearest and second-nearest groups have similar distances | Assuming the assigned group is certain because it has a label |
| Stability | Reproducibility of assignments or profiles under specified perturbations | Equating stability with accuracy or causal benefit |
| Adjusted Rand index (ARI) | Label-invariant agreement between two partitions, adjusted for chance | Using it as a direct estimate of commercial value |
| Silhouette | Relative separation of each observation from own and nearest other group, usually from -1 to 1 | Selecting the only feasible business policy by maximising this score |
| Holdout | Stores or time periods excluded from model fitting and selection | Calling the test untouched after repeatedly tuning against it |
| Minimum cluster size | Policy threshold for stores per group; store count | Treating a small cluster as automatically statistically invalid or commercially feasible |
| Exception | Approved deviation from a cluster's default range or planogram | Applying an undocumented override and retaining misleading compliance statistics |
| Planogram | Versioned visual and data specification for item placement | Assuming issuing the file means the reset occurred |
| Facing | One item-width presentation on the shelf; count | Assuming all items and facings occupy the same physical width |
| Contribution | Revenue less defined variable costs; USD per specified period | Comparing margins that include different cost categories |
| Incrementality | Difference between an outcome under the change and its counterfactual | Counting sales transferred from another item as entirely new |
| Difference-in-differences | Difference between treated and control changes across time; outcome units | Assuming parallel trends are proved by one pre-period observation |
| Cannibalisation | Sales or contribution transferred from existing items to another item | Counting every switched unit as a category gain |
| Target | Desired business outcome with a horizon and owner | Presenting a target as a forecast or contractual commitment |
| Forecast | Conditional estimate of a future outcome with assumptions | Treating the point estimate as an observed result |
| Commitment | Approved obligation with owner, scope, resources, and dates | Calling an unapproved proposal a committed reset |

The terms apply at a stated grain and date.
If the category hierarchy changes, version the mapping before computing historical features.
If a store changes format, record whether it is the same analytic unit for the question at hand.

## 5. The mechanics, step by step

### Step 1 — Define the decision and population

**Input:** a business question, trading estate, category hierarchy, and planning calendar.
**Output:** a written decision brief with a named owner and counterfactual policy.

Specify what the group assignment may change.
Examples include assortment breadth by need state, a regional product slot, or allocation of facings.
Do not let a broad goal such as "understand customers" stand in for an executable choice.

Define the unit as store-category, not necessarily store.
A store can belong to one group for chilled meals and another for coffee.
Fix the geography, format, ownership, channel, and time horizon.
For an omnichannel store, decide whether pickup orders are attributed to the fulfilment store, the customer catchment, or a separately modelled channel.
The right attribution depends on which location holds stock and which decision the group is intended to change.

Choose the comparison policy before examining the group profiles.
It might be the existing national planogram, an existing format plan, or the current store-specific policy.
The proposed policy must beat an achievable alternative, not an imaginary unconstrained optimum.
Record whether current stores already have exceptions and include their costs.

Set the objective at a relevant scope.
Category contribution is often more informative than unit sales because a mix shift can change margin.
If the category drives store traffic, collect broader basket or trip outcomes as guardrails where feasible.
Include implementation and recurrent complexity costs.
Define any non-negotiable requirements before ranking candidate solutions.

**Branch:** if a regulation or contract mandates a particular item or range, record the scope and effective dates as an eligibility constraint.
If no reliable authorisation source exists, resolve it before issuing the plan.
Do not claim a contract applies merely because a vendor asks for a product.

### Step 2 — Map decisions to shopper needs

**Input:** category role, shopper missions, existing assortment, and item attributes.
**Output:** a mutually interpretable need-state map and a list of protected coverage rules.

Ask which meaningful choice can vary across stores.
A chilled-meal range might distinguish ready-to-eat and cook-at-home meals, while also protecting a suitable dietary option.
Those dimensions can overlap; define the denominator and mapping so that any calculated shares remain interpretable.

Map every included SKU to the feature categories used in the analysis.
Where an item has multiple relevant attributes, keep a separate field for each attribute rather than double-counting it in a mutually exclusive segmentation.
Date the mapping when packaging, recipe, or category ownership changes.
Review unknown or ambiguous items with the category manager.

Identify customer needs that must remain represented even when their observed volume is small.
A low-volume product can be the only suitable item for a particular need.
A minimum-coverage rule is a business policy, not a conclusion established by cluster geometry.
Record the evidence and owner behind each rule.

**Branch:** if the business can change only facings, avoid presenting a demand cluster as authority to delist SKUs.
If it can change assortment, evaluate substituted products and available shelf width together.

### Step 3 — Assemble a trustworthy panel

**Input:** item-store transactions, item-store authorisations, inventory or shelf-availability signals, store calendar, and item hierarchy.
**Output:** a versioned store-item-week panel with quality flags.

Choose a history window that represents the intended selling season.
A 26-week window can provide more observations than a 4-week window, but it may blur a seasonal mission or a recent format change.
Compare candidate windows rather than assuming a longer window is always better.
Use fiscal and calendar week definitions consistently across all sources.

Deduplicate transactions and handle returns under a stated convention.
For example, use net unit sales only if returns can be attributed to the original sales period; otherwise retain sales and returns separately.
Distinguish negative corrections from genuine negative demand.
Reconcile category totals to a trusted finance or point-of-sale control report.

Join stores and SKUs using effective dates.
Do not attach today's hierarchy to all past weeks without considering reclassifications.
Record store openings, closures, refurbishments, and major fixture changes.
Separate a week with no trading from a trading week with zero observed sales.

Construct an item-store eligibility flag and an availability signal.
An item may be authorised but not delivered, delivered but not on shelf, or available for only part of a day.
A single inventory snapshot is a proxy, not continuous shelf exposure.
Record the measurement source and known timing gaps.

Retain missingness as a flag.
Do not silently change missing stock records to zero inventory or to full availability.
For each store, report coverage of sales, inventory, assortment, and hours data.
Set minimum quality thresholds before deciding which stores qualify for model fitting.

**Branch:** if a store has too little comparable history, hold it out for a new-store rule rather than inventing a complete panel.
If a whole region lacks stock data, flag regional bias and do not claim equal confidence across regions.

### Step 4 — Separate observed sales from demand opportunity

**Input:** reconciled panel, eligibility and availability measures, price, and promotion history.
**Output:** comparable demand features plus a documented uncertainty flag.

A simple rate divides observed units by eligible in-stock trading days.
That rate can be useful when demand within the day is reasonably comparable and shelf availability is measured reliably.
It is not automatically an unbiased estimate of unconstrained demand.
Items may stock out precisely on unusually busy days, and availability may change during a day.
An end-of-day snapshot misses those within-day dynamics.

Consider an item with 60 observed units over 10 eligible days, but only 6 known in-stock days.
The observed rate over the 6 in-stock days is 10 units per in-stock day.
It is not established that the 4 other days would also have sold 10 units each.
Report the 6-day exposure, the uncertainty, and any assumption used to extrapolate.

A product absent from a range has no direct sales history at that store.
Use comparable stores, online search or order signals where appropriate, customer research, or a deliberate test to inform the possibility of demand.
Such evidence remains indirect and depends on comparability.
Never encode an unoffered item's sales as definitive evidence of no interest.

Price and promotion change what customers choose.
Either restrict the feature window to comparable conditions or model and report the relevant differences.
Simply deleting all promoted weeks can create another bias when promotion intensity differs by store.
When adjustment is too uncertain, retain sensitivity scenarios rather than presenting a false precise correction.

**Branch:** if inventory is too unreliable to distinguish low demand from missing supply, make that limitation visible and avoid a definitive local deletion.
A measurement improvement or pilot can be more valuable than a new clustering run.

### Step 5 — Engineer demand-led features

**Input:** comparable store-category observations and the defined planning decisions.
**Output:** a compact, documented feature matrix and data dictionary.

Use features that can influence an actual range or space choice.
Possible features include need-state unit shares, sales per open day, repeat purchase or trip signals when lawfully available, and seasonal demand patterns.
A share needs a clear numerator and denominator.
Use the same eligible category definition in both.

A large store's total category units can dominate a raw-distance model.
Shares help describe mix independent of size, but they discard absolute demand.
Keep a separate volume or capacity measure only if it informs the intended decision.
For example, volume may influence the depth of stock rather than which need states are represented.

Shares sum to one, so they are not independent dimensions.
An increase in one share necessarily changes the others even if their absolute rates do not change.
A model with all shares and other overlapping features can double-weight the same commercial signal.
For suitable strictly positive compositions, log-ratio approaches may be considered; zero handling and interpretability require explicit decisions.
For a simple business use case, a small number of interpretable shares with sensitivity checks may be sufficient.

Avoid raw SKU-level columns when items were not offered consistently.
That matrix can cluster stores by past planogram assignment rather than by underlying shopper need.
Aggregate to comparable need states where justified and retain exposure flags.
An individual item can still need a separate review when its unique customer role matters.

Demographics and catchment data may supplement, not replace, observed demand.
They may help with cold starts or reveal geographic coverage gaps.
They can also encode historic exclusion or lead to inappropriate treatment of protected groups.
Use only lawful, necessary, reviewed variables and assess the consequences for access.
Do not infer an individual shopper's identity or preference from a store cluster.

Keep a feature dictionary with formula, unit, numerator, denominator, source, window, missingness, owner, and known bias.
Make a data-cut date explicit.
Exclude features that depend on outcomes observed only after the intended decision date.

### Step 6 — Transform, weight, and inspect features

**Input:** feature matrix, data-quality rules, and business priorities.
**Output:** a reproducible transformation fitted on training data and a distance definition.

A distance compares values only after deciding what a unit of difference means.
A 10-percentage-point difference in need-state share is not automatically equivalent to a 10-unit difference in weekly sales.
An unscaled Euclidean distance will usually give greater influence to numerically larger features.

Standardisation subtracts a training-set mean and divides by a training-set standard deviation.
This can make variables numerically comparable, but it assigns influence relative to observed variation, not necessarily relative to economic importance.
A rare but important need-state feature may be distorted by extreme scaling.
A near-zero scale needs an explicit policy, such as dropping a constant feature or applying an approved alternative.

Robust scaling can reduce sensitivity to extreme observations.
Its median and interquartile range must still be fitted on the training population.
If the interquartile range is zero, do not divide by it.
Record the chosen treatment and examine which stores are affected.

Weights can encode a category team's priorities, but they are modelling choices.
For Euclidean distance, multiplying a feature by a weight changes its squared contribution to the distance.
Document the weight, reason, and sensitivity to plausible alternatives.
A demand feature should not be given effectively zero influence simply because a demographic variable is easier to source.

Inspect distributions, missingness, correlations, and outliers before fitting.
Investigate an outlier as a possible data error, real niche, or atypical format.
Do not cap a legitimate high-demand store solely because it complicates a neat picture.
Separate incompatible formats when a shared plan is impossible, or include format as a hard feasibility rule.

Fit all transformations using training data only.
Apply the frozen parameters to validation and deployment stores.
Refitting the scaler on each new batch changes the distance geometry and can change assignments without any actual change in a store.

### Step 7 — Fit candidate groupings

**Input:** transformed training features, candidate methods, candidate group counts, and operational bounds.
**Output:** candidate partitions, profiles, and diagnostics.

For compact continuous numerical features, k-means is a common baseline.
It minimises within-cluster squared Euclidean distances to centroids for a fixed number of groups.
Its iterative solution depends on initialization and may find a local rather than global minimum.
Use several seeded initializations and retain the configuration and software version.
K-means does not itself enforce shelf constraints, minimum group size, or a suitable balance of stores.

A medoid-based method represents each group by an actual observed store.
It can support alternative distances and improve interpretability, though computational cost can increase with estate size.
A hierarchical method can reveal nested structure and supports a dendrogram, but a cut still requires a business choice.
A rule-based segmentation may be easier to govern where the decision boundary is already known and defensible.
Compare against a simple rule or current format group, not just against other sophisticated algorithms.

Candidate group counts should reflect commercial capacity.
If a team can maintain at most four planograms for a category, a solution requiring twelve independent variants is not automatically attractive.
Conversely, imposing one group for convenience does not prove that local demand differences are immaterial.
Set minimum group size as an economic or operational condition, not as a universal statistical law.

Inspect within-group dispersion, separation, group sizes, and cluster profiles.
Silhouette can identify overlap, but its preference depends on the distance and can penalise useful non-spherical structure.
The elbow in within-cluster sum of squares is subjective and always improves or stays equal as group count increases.
Neither score estimates incremental category contribution.

For each candidate, ask whether profiles suggest different feasible choices.
If two clusters receive identical ranges under all constraints, their extra label may not justify another planogram.
If a small group requires a unique, important need-state option, an exception may serve it better than a new estate-wide group.
Do not merge a vulnerable group simply to improve average profit without examining access consequences.

**Branch:** when a candidate violates a hard constraint, reject it or modify the formulation and refit.
Do not silently reassign stores after fitting and still report the original cluster diagnostics as if unchanged.

### Step 8 — Test stability and out-of-sample usefulness

**Input:** candidate partitions, saved transformations, and repeated time windows or store samples.
**Output:** stability report, uncertainty flags, and an approved assignment rule if warranted.

A useful group should persist enough to support the reset cycle.
Compare assignments across adjacent, comparable windows, accounting for actual seasonality and demand changes.
A changed assignment may be a real signal, not necessarily a model defect.
Report both the change rate and the commercial implications of resetting affected stores.

Cluster label names are arbitrary.
Group 1 in one run can represent group 2 in the next.
Use a label-invariant partition measure such as adjusted Rand index, or align labels by a documented matching rule before reporting transition counts.
Restrict comparisons to the same store universe and report exclusions.

For resampling, refit on perturbed training samples and assign a common set of evaluation stores to each fitted model.
Compare partitions on that common set.
If bootstrap samples omit or repeat stores, do not calculate agreement as if their member lists were identical without an explicit common assignment step.
Use repeated seeds separately from resampling to distinguish algorithm initialization instability from data instability.

Hold out later periods where possible to respect the decision timeline.
Check whether a training assignment predicts a meaningful future mix or helps choose a feasible plan without using future information.
A stable historical grouping may still be useless if future profiles converge or if the same choice is best for every group.
A high agreement score is not proof of causal impact.

For a distance-based rule, inspect the distance to the nearest and second-nearest centroid.
A small gap indicates boundary sensitivity, not a calibrated probability of membership.
Define a review band before deployment; its width should be justified by repeatability and reset economics.
New stores and missing-data stores need a separate assignment or temporary default.

### Step 9 — Turn profiles into feasible plans

**Input:** stable profiles, current planograms, product attributes, and constraints.
**Output:** one proposed cluster-level range per variant and a store-level execution schedule.

For every group, identify which need states are over- or under-represented relative to the current policy.
Translate that difference into an explicit decision: which item, facing, or space allocation changes and what is displaced.
A cluster name such as "convenience" is not a plan.
An individual SKU recommendation also needs evidence that it can serve the inferred need.

Check available linear shelf width, fixture height and depth, display rules, item dimensions, and replenishment intervals.
Calculate whether the proposed facings physically fit and hold enough stock between deliveries.
Review shelf access and category adjacency with the space planner and store staff.
A store may share demand with a cluster yet need a different planogram size.
Separate demand clusters from format-specific execution variants where necessary.

Validate supplier capacity, minimum order quantities, case packs, lead times, and delivery days.
A locally attractive item that cannot be supplied consistently can reduce both sales and shopper trust.
Check whether new regional products create excessive distribution or handling cost.
Do not assume all stores can receive every proposed item through their current network.

Record any mandatory need-state representation, price architecture, safety, food-handling, or contract rules.
A low-volume required option may be protected even if the simple clustering objective would remove it.
Obtain the appropriate business or legal review for actual obligations.
The group cannot override a binding rule.

Define a limited exception process.
Each request should state the store, category, requested change, reason, evidence, cost, owner, approval date, expiry date, and review trigger.
Record whether the exception changes the default cluster, planogram size, or a single item.
An exception must be visible in the delivered range and in evaluation data.

### Step 10 — Estimate value and uncertainty

**Input:** proposed versus current plans, forecast response, costs, constraints, and planning horizon.
**Output:** a decision-ready forecast with scenarios and an explicit approval threshold.

Forecast category-level incremental contribution, not just sales of newly listed items.
For each change, identify expected substitution, lost sales from delisted items, margin, and relevant changes in waste or handling.
Where a credible response estimate is unavailable, show a range and label it as an assumption.
Do not add unmeasured halo effects as a confirmed benefit.

Use a consistent time horizon.
Multiply a per-store-week effect by the number of affected stores and weeks only when that effect is expected to apply to those stores and weeks.
Separate one-time reset costs from recurring labour, logistics, licensing, and forecast-maintenance costs.
Do not charge the same cost twice through both contribution and a separate cost line.

Include adverse scenarios: lower response, delayed implementation, more stockouts, higher waste, and longer resets.
Calculate break-even effect as incremental cost divided by exposed store-weeks when units and scope match.
A positive point forecast can still be unattractive if downside risk is large or the effect is highly uncertain.
A target is not evidence that the forecast will occur.

**Branch:** if the projected value is below implementation cost under plausible scenarios, simplify the policy, improve the evidence, or retain the current plan.
Do not select a larger number of clusters merely because it fits past data better.

### Step 11 — Design a comparative pilot

**Input:** candidate plan, eligible stores, operational calendar, and outcome definitions.
**Output:** a pre-specified protocol, randomization or comparison design, and decision rule.

When feasible, randomize eligible stores to proposed and current policies.
Block or stratify before randomization by relevant format, baseline performance, or distribution route where it improves comparability and operational feasibility.
Keep all stores in the assigned analysis group even if some resets fail, for an intention-to-treat estimate of the policy as deployed.
Separately report execution compliance; it does not replace the assigned-group comparison.

Define a pre-period and a post-period in comparable weeks.
Measure category contribution at store-week level under the same accounting rules for both arms.
Record individual-item and adjacent-category changes to understand substitution and potential spillovers.
Track availability, waste, customer access, and reset compliance as guardrails.

A difference-in-differences calculation compares each arm's change from baseline.
Randomization supports comparability in expectation; chance imbalance remains possible in a finite sample.
In a non-randomized comparison, a parallel-trends assumption is required for the simple estimator to identify an effect.
A few pre-period observations can be suggestive, not proof, of that assumption.
External events, concurrent promotions, and different execution dates can invalidate the interpretation.

Set the pilot duration and sample using plausible effect variability, desired precision, operational constraints, and an agreed decision threshold.
Account for repeated observations within stores and any randomization at a higher level such as a distribution route.
A confidence interval or other uncertainty interval should match the randomization and analysis unit.
A small pilot may be suitable for feasibility but too imprecise for an investment decision.

Define the action rule before observing outcomes.
For example, require positive expected net contribution, acceptable downside, minimum availability, and protected-need coverage before scaling.
Define who can stop a pilot for safety, legal, or severe service failures.
Keep forecasts, targets, observed estimates, and approval decisions in separate fields.

### Step 12 — Deploy, monitor, and retire

**Input:** approved protocol, store assignments, execution files, and monitoring owners.
**Output:** versioned deployment records and a controlled refresh or retirement decision.

Freeze a model version with feature definitions, transformations, group profiles, assignment logic, and the training data cut.
Export assignments with effective dates and the associated planogram version.
Keep the previous approved version for reproducibility and rollback.
An audit should be able to connect a store's shelf instructions to the decision and data available at approval.

Train field staff on the change and its reason.
Confirm item authorisation and supply before requesting a reset.
Record actual reset dates and verification, not just scheduled dates.
A store cannot fairly be evaluated as fully treated before the plan is actually executable.
The main assigned-group estimate should nevertheless preserve the original randomization.

Monitor both outcomes and data quality.
Look for abnormal stockouts, missing shelf records, unexpected returns, shifts in mix, item delistings, and changes in category contribution.
Track exception counts, reasons, and expiry dates.
A high exception rate can indicate a missing constraint or a poor cluster design rather than uncooperative stores.

Set a refresh cadence tied to the category reset cycle.
Avoid forcing a store to change planograms for a minor centroid movement.
A hysteresis rule can require a store to remain clearly closer to a new group across consecutive reviews before reassignment.
Specify the review band, number of reviews, emergency exceptions, and owner before applying the rule.
The rule trades responsiveness for lower churn; monitor both.

A significant format change, new competitor, supply interruption, or change in category role can justify an earlier review.
Retire the model if inputs become unavailable, mapping changes invalidate comparisons, or measured net benefits do not justify complexity.
Retaining the current plan is a legitimate decision when the evidence is weak.

## 6. Worked example

### Case and decision

A fictional retailer operates 40 comparable supermarkets selling chilled meals.
The category team currently uses one national range, with store-specific fixture sizes already handled in the existing planogram.
Managers suspect that stores differ in their mix of ready-to-eat meals and cook-at-home meals.
They propose a demand-led grouping to decide how many of the limited optional slots should serve each need state.

The decision is not to remove every cook-at-home option from a ready-to-eat-oriented store.
The plan must retain at least one suitable option for each protected need state and comply with local product eligibility rules.
The candidate changes a few optional slots, not the entire category.
The baseline is the current national range with its existing size adaptations.

For illustration, six stores from the training population have comparable 13-week unit-sales records.
Assume all included units are mapped to exactly one of the two meal need states.
Assume the products were consistently ranged and sufficiently available over the window.
Those assumptions are deliberately strong and must be checked in a real project.
Units here are packs sold, not individual meals or USD.

| Store | Ready-to-eat units, 13 weeks | Total category units, 13 weeks | Ready-to-eat share |
|---|---:|---:|---:|
| A | 200 packs | 1,000 packs | 20% |
| B | 240 packs | 1,000 packs | 24% |
| C | 260 packs | 1,000 packs | 26% |
| D | 700 packs | 1,000 packs | 70% |
| E | 740 packs | 1,000 packs | 74% |
| F | 760 packs | 1,000 packs | 76% |

For store B, 240 packs divided by 1,000 packs equals 0.24, or 24%.
The remaining 76% of comparable category units are cook-at-home packs under the exhaustive two-state mapping.
The share says nothing directly about total profit or unmet demand.
The identical denominators keep the hand calculation easy; real stores generally differ in volume.

### Candidate groups and boundaries

Suppose the team proposes two groups and, for this teaching illustration, uses one-dimensional Euclidean distance on the ready-to-eat percentage.
The resulting proposed memberships are A–C and D–F.
Their centroids are the unweighted means of the store-level percentages.

```text
Lower-share centroid = (20 + 24 + 26) / 3 = 23.333 percentage points
Higher-share centroid = (70 + 74 + 76) / 3 = 73.333 percentage points
Midpoint = (23.333 + 73.333) / 2 = 48.333 percentage points
```

The midpoint is the equal-distance boundary for these two centroids in this one-feature space.
A store with 46% ready-to-eat share is 22.667 percentage points from the lower centroid.
It is 27.333 percentage points from the higher centroid.
The nearest-centroid rule assigns it to the lower-share group.
Its distance gap is only 4.6667 percentage points, so a review band of 5 percentage points flags it for human review.
This band is an illustrative policy choice, not a probability or a universal threshold.

A store exactly at the midpoint has equal distances.
The assignment rule must specify a deterministic tie break, while still flagging the case for review.
This example chooses the lower-share group on an exact tie.
A decision maker can instead hold the prior approved plan while the tie is reviewed.

The training groups contain three stores each, far too few to establish a robust production grouping.
The numbers illustrate mechanics only.
The full 40-store estate would need sufficient, reliable observations, comparison with current plans, and stability analysis.
No silhouette, adjusted Rand index, or statistical confidence interval is inferred from the six rows.

### Translate the group into choices

Suppose each store has eight optional chilled-meal slots after mandatory core items are placed.
The illustrative cluster proposition is:

| Policy | Lower-share group | Higher-share group | Common safeguards |
|---|---|---|---|
| Current national policy | 4 ready-to-eat and 4 cook-at-home optional slots | 4 ready-to-eat and 4 cook-at-home optional slots | Core products and store fixture rules remain |
| Proposed policy | 2 ready-to-eat and 6 cook-at-home optional slots | 6 ready-to-eat and 2 cook-at-home optional slots | Retain protected need-state coverage and verify shelf fit |

An optional slot is a planning count, not necessarily an equal width of physical shelf.
Each candidate plan must be validated against actual item widths, shelf capacity, and replenishment.
The chosen products must be eligible and supplied to the relevant stores.
The group label does not establish which brand or dietary option should be removed.

Assume the space team confirms that the chosen items fit each relevant fixture and that supply can support the pilot.
A specific store that lacks suitable chilled depth keeps an approved format exception.
A store close to the assignment boundary holds its existing mix until the category owner reviews it.
Those checks are assumptions for the example, not observed facts.

### Pilot comparison

The retailer randomizes 40 eligible stores: 20 to the proposed group-specific policy and 20 to the current policy.
Randomization is blocked on format and baseline category contribution.
Both arms are observed for a comparable pre-period and a 12-week post-period.
The outcome is average category contribution per store per week, in USD, after product costs and normal variable selling costs but before incremental reset and pilot-specific costs.
Assume the outcome uses the same category and accounting definitions in both arms.

| Assignment | Pre-period contribution | Post-period contribution | Change |
|---|---:|---:|---:|
| Proposed policy, 20 stores | USD 110/store-week | USD 128/store-week | +USD 18/store-week |
| Current policy, 20 stores | USD 110/store-week | USD 119/store-week | +USD 9/store-week |

The difference-in-differences estimate is USD 18 minus USD 9, or USD 9 per store-week.
The control group's USD 9 increase is not attributed to the new policy.
The treatment group's full USD 18 increase is not attributed to the policy either.
The estimate compares changes, subject to the design and measurement assumptions.

The pilot covers 20 assigned treatment stores for 12 weeks, or 240 treatment store-weeks.
At the average estimated effect, the illustrative incremental category contribution is:

```text
USD 9/store-week × 20 stores × 12 weeks = USD 2,160
```

Assume one-time reset costs of USD 800 and additional pilot-specific handling costs of USD 400.
These costs are not already subtracted from the contribution outcome.
The illustrative pilot net is:

```text
USD 2,160 - USD 800 - USD 400 = USD 960
```

Break-even incremental contribution is USD 1,200 divided by 240 store-weeks, or USD 5 per store-week.
The USD 9 estimate exceeds this point threshold by USD 4 per store-week.
This arithmetic is not an uncertainty interval and does not establish that the true effect exceeds break-even.
The individual store-week data and allocation details are needed to estimate uncertainty correctly.

The USD 960 figure is an illustrative observed-pilot calculation, conditional on the table and costs.
A future rollout forecast requires separate assumptions about other stores, future duration, reset cost, service, and persistence.
If a decision target is USD 6 per store-week after specified costs, label it a target, not the observed USD 9 gross effect.
An approved commitment would additionally identify the budget, stores, dates, accountable owner, and stop conditions.
This example contains no such approval.

### Interpret the result

The estimate concerns the proposed policy package: cluster assignment, optional slot changes, implementation, and actual compliance.
It does not identify the effect of the clustering algorithm independently of the assortment changes.
It also does not identify the isolated effect of a particular product or facing.
Store-level randomization and consistent measurement protect the comparison in expectation, but a small sample can produce an imprecise result.

Before any scale decision, inspect availability, execution, protected need-state coverage, and customer complaints.
Check for cross-store spillovers if customers shop in both treatment and control stores.
Check whether a regional promotion or supply disruption affected one arm differently by chance.
Use the original assignment for the primary analysis and report actual compliance separately.
A positive category result can coexist with an unacceptable access or waste outcome.

A reasonable decision is to review store-level uncertainty and guardrails before scaling.
If evidence is too imprecise, extend or redesign the pilot rather than treating USD 960 as a guaranteed return.
If constraints cannot be maintained in some stores, retain their current plans or approve documented alternatives.

## 7. Code, query, or calculation

The following standard-library Python 3 program reproduces the six-store feature calculation, the proposed two-group centroids, a new-store assignment, and the pilot arithmetic.
It uses the illustrative group memberships from section 6; it does **not** fit k-means or claim to choose the best number of clusters.
It uses only information available before assigning the hypothetical new store.
The program is self-contained and requires no external file or package.

The feature is a percentage on a 0–100 scale, and distance is in percentage points.
Each store has equal weight in a centroid, so this is a mean of store shares rather than a pooled unit share.
In this particular dataset all denominators are equal; for unequal denominators the two definitions differ.
The code applies a lower-group tie break and flags a distance gap of at most 5 percentage points for review.
The financial result excludes any unlisted cost and has no statistical interval.

```python
from statistics import mean

# Six illustrative stores; sales are packs in a 13-week window.
rows = [
    ("A", 200, 1000, "lower"),
    ("B", 240, 1000, "lower"),
    ("C", 260, 1000, "lower"),
    ("D", 700, 1000, "higher"),
    ("E", 740, 1000, "higher"),
    ("F", 760, 1000, "higher"),
]

# Authorised, in-stock exposure is assumed comparable in this example.
shares = {}
labels = {}
for store, ready_units, category_units, group in rows:
    if category_units <= 0 or not 0 <= ready_units <= category_units:
        raise ValueError(f"Invalid sales inputs for store {store}")
    shares[store] = 100.0 * ready_units / category_units
    labels[store] = group

# These assignments are inputs to the demonstration, not fitted labels.
centroids = {
    group: mean(shares[s] for s in shares if labels[s] == group)
    for group in ("lower", "higher")
}

print("Training shares (%):")
for store in sorted(shares):
    print(f"  {store}: {shares[store]:.1f} -> {labels[store]}")
print(f"Lower centroid: {centroids['lower']:.3f}%")
print(f"Higher centroid: {centroids['higher']:.3f}%")

# Assign one new store using the saved training centroids.
new_share = 46.0  # percentage of comparable category packs
ranked = sorted(
    ((abs(new_share - value), group) for group, value in centroids.items()),
    key=lambda item: (item[0], item[1] != "lower"),
)
nearest_distance, assigned = ranked[0]
second_distance, _ = ranked[1]
gap = second_distance - nearest_distance
review = gap <= 5.0  # percentage points; an illustrative policy threshold
print(f"New store: {assigned}; distance gap {gap:.3f} pp; review {review}")

# Difference-in-differences: USD per store-week.
treatment_pre, treatment_post = 110, 128
control_pre, control_post = 110, 119
effect = (treatment_post - treatment_pre) - (control_post - control_pre)
treatment_stores, post_weeks = 20, 12
reset_cost, handling_cost = 800, 400  # USD; not in contribution above
store_weeks = treatment_stores * post_weeks
gross = effect * store_weeks
net = gross - reset_cost - handling_cost
breakeven = (reset_cost + handling_cost) / store_weeks
print(f"DiD: USD {effect}/store-week")
print(f"Gross incremental contribution: USD {gross:,}")
print(f"Pilot net: USD {net:,}")
print(f"Break-even effect: USD {breakeven:.2f}/store-week")
```

Expected output:

```text
Training shares (%):
  A: 20.0 -> lower
  B: 24.0 -> lower
  C: 26.0 -> lower
  D: 70.0 -> higher
  E: 74.0 -> higher
  F: 76.0 -> higher
Lower centroid: 23.333%
Higher centroid: 73.333%
New store: lower; distance gap 4.667 pp; review True
DiD: USD 9/store-week
Gross incremental contribution: USD 2,160
Pilot net: USD 960
Break-even effect: USD 5.00/store-week
```

To adapt this artifact, replace the hard-coded memberships with an approved fitting procedure and keep training, validation, and deployment periods separate.
Persist the training transformation and centroid ordering alongside the versioned model.
Use a reproducible tie rule and do not interpret a distance gap as a calibrated probability.
Do not estimate confidence from the six training points or from the four aggregate pilot means alone.

A production system also needs actual range eligibility, availability, fixture, and exception data.
The program intentionally does not simulate those inputs or assert a real return.
Checking unit and denominator validity does not repair biased or censored sales data.
If the new store has no comparable category history, route it to a cold-start policy rather than inventing a share.

## 8. Data, standards, and regulatory context

### Minimum data contract

| Dataset | Grain and minimum fields | Decision use | Critical control |
|---|---|---|---|
| Sales | Store, SKU, transaction or day, units, net revenue, date | Demand mix and contribution | Reconcile units and revenue to source totals |
| Range eligibility | Store, SKU, effective start and end, status | Distinguish unoffered from no sales | Date-effective join; preserve unknown status |
| Availability | Store, SKU, time or day, source, in-stock indicator | Flag censored observations | Check timing, gaps, and shelf-versus-system discrepancy |
| Store calendar | Store, day, trading hours, closure and remodel flags | Correct exposure and eligible population | Distinguish missing data from closed days |
| Product hierarchy | SKU, category, need state, dates, attributes | Comparable numerator and denominator | Version reclassifications and attribute changes |
| Price and promotion | Store, SKU, dates, price, promotion type | Context and comparability | Align effective dates to sales |
| Fixture | Store, fixture, usable width, height, depth, effective dates | Plan feasibility | Validate as-built dimensions |
| Supply | SKU, node, lead time, case pack, service and restrictions | Replenishment feasibility | Confirm current route and capacity |
| Costs | Store, SKU or category, period, cost definition | Incremental contribution and reset case | Avoid double counting; agree finance ownership |
| Execution | Store, plan version, planned and actual reset dates, status | Pilot treatment and compliance | Preserve assigned and actual policy separately |
| Exceptions | Store, category, reason, approval, expiry | Local overrides and governance | Reconcile approved exception to executed range |
| Assignment | Store, category, model version, label, effective dates | Reproducible deployment | Freeze feature definitions and tie rule |

The store-week panel should have one unambiguous fiscal-week definition.
A week spanning a store opening or closure may require partial-week exposure.
An item available only online should not automatically be interpreted as shelf availability in the physical store.
When returns occur after the original sale, define whether the outcome is booked by return date or restated to sale date.
Use the same rule in both pilot arms.

### Identifiers and interoperability

The Global Trade Item Number (GTIN) is a GS1 identifier for trade items where it is used.
A retailer may also have an internal SKU for inventory and assortment management.
A change in pack configuration can require a different trade-item identity; maintain a date-effective crosswalk rather than assuming every product name uniquely identifies a sellable unit.
GS1 guidance describes identifier allocation; a GTIN by itself does not prove that the item is ranged or available at a particular store.

A Global Location Number (GLN) can identify a party or location in the GS1 system where adopted.
An internal store identifier may still be necessary for historical continuity and joins to trading calendars.
Neither identifier provides shelf dimensions, operating hours, or assortment policy.
Store moves, re-openings, and format conversions require an explicit continuity decision.

GS1 standards are industry interoperability conventions unless made mandatory by an applicable contract, programme, or legal requirement.
The clustering method does not itself impose a GS1 mandate.
Verify the relevant current allocation rules and trading-partner requirements for the implementation.
Do not infer a legal compliance claim from an identifier format alone.

### Privacy and responsible use

An analysis using only aggregated store sales may not require individual-level records.
If loyalty or transaction data can identify a person directly or indirectly, use the applicable privacy framework and internal governance.
The EU General Data Protection Regulation (GDPR) defines principles including data minimisation and purpose limitation; applicability depends on the facts and jurisdiction.
An aggregated output is not automatically anonymous if small cells or linkage can reveal individuals.
Assess re-identification risk, access controls, retention, and disclosure before publishing profiles.

Use the minimum data necessary for the category decision.
Consider whether a store-level demand feature can answer the question without carrying a person-level identifier into the clustering table.
Record the lawful basis, permitted purpose, retention period, and access owner when personal data are involved.
Consult the relevant privacy and legal teams rather than assuming the same rule applies in every country.

Demographic variables can be correlated with protected characteristics and past unequal access.
Even a model that excludes explicit protected attributes can produce uneven availability of essential or culturally important products.
Test coverage and service outcomes across relevant geographies and groups where lawful and appropriately governed.
A review should examine the proposed decision and its effects, not merely whether the algorithm contains a particular column.
Fairness requirements and protected categories vary by jurisdiction; obtain local advice.

### Lineage and timing

Save the source extraction time, fiscal calendar, category mapping version, training window, and feature code version.
Record both the effective date of an assignment and the date it was generated.
A historical backtest must use only information available at its simulated decision date.
A later corrected inventory record may be useful for audit but must not silently enter a historical prediction as if it were known then.

Keep raw, cleaned, feature, assignment, and execution layers distinguishable.
Store corrections as versioned updates rather than overwriting the evidence used at approval.
Document who approved exclusions and how unknown values are treated.
Retention and access decisions should follow the organisation's applicable records and privacy rules.

## 9. Industry practice and case evidence

### Public case: East of England Co-op

RELEX Solutions publishes a case study naming East of England Co-op and describing localised planograms and retail planning.
The supplier reports changes in availability, sales, and inventory in connection with the project.
The report is a vendor case study, not a randomized independent evaluation.
Consult the linked source for its reported periods, scope, and metric definitions before comparing figures to another retailer.

The source is useful because it describes an operational implementation rather than merely a clustering score.
The named retailer, planning process, and claimed outcomes are public case evidence.
The supplier's reported association does not isolate the causal effect of store clustering from forecasting, replenishment, process, or other concurrent changes.
It should not be treated as proof that another retailer will obtain the same return.

[INFERENCE] The case illustrates why a local planogram requires integration with inventory and store execution.
It does not establish that a specific demand-feature set, cluster count, or review cadence is optimal.
An analyst should seek implementation dates, comparison groups, cost definitions, and uncertainty before using any reported result as an investment benchmark.

### Public case: PepsiCo category study

PepsiCo's public case study on clustering assortment to meet localised shopper demand describes a small-format salty-snacks assortment problem.
The document discusses limited space and the limitations of a uniform assortment.
It is a supplier-authored case and reflects the supplier's perspective and incentives.
Its reported findings should not be confused with an independent trial of the hypothetical chilled-meal policy in this lesson.

[INFERENCE] The transferable question is whether grouping stores by relevant shopper demand can change a feasible range decision.
That is a hypothesis to test for each category, not a guarantee that a similar cluster structure exists elsewhere.
A supplier's item recommendations also require retailer-side checks for category incrementality, competing items, need-state coverage, and network cost.

### What a credible implementation report should disclose

A practical report should name the current policy and the exact change.
It should give the number and type of stores, category, calendar, and assignment unit.
It should distinguish a plan issued from a plan executed.
It should report treatment and control definitions and the handling of noncompliance.
It should name the financial outcome and costs included.
It should show uncertainty and material guardrail outcomes, not just a headline percentage.

A before-and-after sales increase can reflect seasonality, inflation, a promotion, a competitor closure, or changes in availability.
A named case can be useful evidence of feasibility without proving the source of the reported improvement.
A vendor claim is stronger when it discloses its method and underlying comparison, but that does not turn it into independent peer review.

### Typical operating model

Category management defines the shopper proposition and protected needs.
Analytics prepares the comparable demand data and assesses candidate groups.
Space planning turns the proposition into fixture-specific plans.
Supply chain validates sourcing and replenishment.
Store operations records execution and exceptions.
Finance owns the contribution and cost reconciliation.
Privacy and legal teams review applicable data and obligations.

These responsibilities are a recommended governance pattern, not a claim that every retailer uses the same organisation.
Agree the accountable decision owner and the escalation route in the project charter.
A clustering report without a named plan owner is unlikely to change the shelf.
A plan without a measurement owner cannot establish whether the change helped.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Cluster on censored sales | Low share coincides with stockouts | Missing supply suppresses observed units | Wrong local range | Use exposure flags, sensitivity bounds, and availability improvement |
| Infer zero demand from no listing | Feature matrix has zeros before authorisation | Unoffered item has no opportunity to sell | Premature rejection of unmet need | Mark not-ranged separately; use comparable evidence or a test |
| Group by old planogram | Clusters mirror prior assortment codes | Offered products determine observed mix | Reinforces prior decisions | Use comparable need states and account for exposure |
| Let volume dominate | Largest stores form a cluster regardless of mix | Unscaled units overwhelm shares | Group reflects size, not demand mix | Separate volume from mix; scale and test weights |
| Overweight demographics | Similar-looking catchments drive all assignments | Proxy overwhelms observed category demand | Poor fit and potential access harms | Lead with relevant demand; review lawful use and outcomes |
| Leakage from future | Backtest score is unusually good | Features include post-decision outcomes | Inflated expected performance | Freeze as-of dates and use forward validation |
| Compare raw cluster labels | Most stores appear to change groups after refit | Labels permute without profile changes | False instability alarm | Align labels or use label-invariant agreement |
| Treat stability as value | High ARI cited as proof of gain | Reproducibility is not causal impact | Expensive rollout without benefit | Test the changed policy against a counterfactual |
| Optimise silhouette alone | Chosen group count exceeds planning capacity | Statistical separation ignores cost and constraints | Unmanageable variants | Compare feasible policies and incremental economics |
| Ignore store boundaries | Frequent near-tie reassignments | Small noise crosses a hard threshold | Reset churn and customer disruption | Use review bands, hysteresis, and cadence |
| Exclude low-history stores silently | New stores absent from assignment export | No cold-start policy | Stores receive no usable plan | Assign a documented provisional default and review date |
| Treat every exception as error | High override rate is suppressed | Model misses fixture or supply conditions | Inaccurate compliance and poor service | Capture reasons and improve constraints |
| Ignore shelf fit | Plan includes more width than fixture permits | A slot count is mistaken for physical space | Partial or failed execution | Validate item dimensions, facings, and as-built fixtures |
| Ignore supply feasibility | Recommended local item is often unavailable | Capacity, route, or case pack not checked | Lost sales and poor trust | Validate supply and monitor availability before reset |
| Confuse own sales with incrementality | New item's sales rise but category contribution does not | Cannibalisation or margin dilution | Overstated value | Measure total category and relevant adjacent outcomes |
| Compare unadjusted pilot arms | One arm has higher baseline performance | Baseline differences and time effects remain | Misleading result | Randomize or justify comparison and analyse change |
| Ignore cluster-level dependence | Confidence interval assumes each store-week independent | Repeated weeks within a store are correlated | Overconfident decision | Match uncertainty estimation to randomization unit |
| Hide failed execution | Plan issued but shelf checks are absent | Intended treatment differs from observed shelf | Effect cannot be interpreted | Record assignment, actual reset, and compliance separately |
| Double-count benefits | Both item uplift and category uplift are added | Item sales are already inside category outcome | Inflated business case | Reconcile outcome hierarchy and count each benefit once |
| Double-count costs | Normal handling is in contribution and subtracted again | Finance definitions differ | Understated net | Agree a cost bridge and accounting boundary |
| Disregard rare needs | Protected option removed in low-volume locations | Average sales overwhelms access requirement | Customer harm and potential compliance exposure | Set explicit coverage rules and review affected stores |
| Change hierarchy midstream | Segment shares jump after recoding | SKU mapping changed, not customer demand | False drift and reassignment | Version mapping and recompute comparable history |
| Report a target as result | Presentation says expected gain is achieved | Forecast and measured outcome are merged | False accountability and premature scaling | Separate target, forecast, estimate, and commitment |
| Refit scaling on deployment batch | Same store changes label without new sales | Coordinate system shifts | Irreproducible assignments | Persist training transformations and model version |

A red flag is an investigation trigger, not automatic proof of a specific cause.
For each flag, assign an owner and an action date.
The appropriate mitigation depends on the business decision and quality of the available evidence.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or bound it |
|---|---|---|---|
| Comparable category definition | Shares need consistent numerator and denominator | SKU reclassification or recipe change | Audit date-effective mapping and bridge old to new hierarchy |
| Availability is sufficiently observed | Sales should reflect opportunities to buy | Missing shelf stock or stale inventory snapshots | Compare signals with audits and test high- versus low-coverage stores |
| Ranges are comparable for the selected feature | Different offers can distort mix | A need state is absent from one store's range | Report eligibility by store and test a comparable subset |
| Training period represents the decision season | Profiles should inform future plans | Holidays, weather, or trend change | Compare matched seasonal windows and forward holdouts |
| Group structure is useful for the category | Similar profiles must support similar decisions | Stores with similar mix need different products or constraints | Compare actual candidate plans and exception rates |
| Selected distance reflects business priorities | Closeness drives assignment | Scale or weights overemphasise irrelevant variables | Vary scaling and weights; inspect changed stores |
| Group assignments are stable enough | Resets have cost and customer impact | Noisy small samples or structural demand shift | Compare common stores over time and resamples; price churn |
| New stores can receive a safe provisional plan | Assignment is required before history exists | No similar format or catchment and unusual local demand | Use a conservative default, monitor, and schedule early review |
| Fixture and supply records are current | Proposed plan must be executable | Remodel, wrong dimensions, or route change | Validate store survey and supply commitments before reset |
| Pilot outcome is measured consistently | Differences need a common financial definition | Cost allocation or category membership changes | Reconcile finance data and freeze definitions across arms |
| Randomization remains meaningful | Assigned groups should be comparable in expectation | Allocation is overridden or treatment contaminates controls | Audit randomization, preserve assignments, and report spillovers |
| No material uncontrolled concurrent change | Simple effect attribution needs comparable exposure | Promotion, pricing, or competitor changes differ across arms | Log concurrent events and examine sensitivity |
| Non-randomized trends would have been parallel | Difference-in-differences needs a counterfactual trend | Differential pre-trends or shocks | Inspect multiple pre-periods and justify a credible comparator |
| Effect can be extrapolated | A pilot forecast is used for a wider decision | New regions, season, or supply conditions differ | Segment the forecast and test in additional conditions |
| Costs cover the relevant horizon | Net value requires all incremental costs | Omitted recurring data, labor, waste, or licensing costs | Reconcile one-time and recurring costs with owners |
| No unacceptable access harm | Average gain is not the only objective | Small group loses its sole suitable product | Audit coverage by store and affected need, with governed review |

An assumption may be plausible without being verified.
A sensitivity range is preferable to a precise unsupported correction.
Where a binding constraint cannot be verified, the appropriate action may be to hold the affected plan.

## 12. Exercises

### Exercise 1 — Frame the decision

**Time:** 15 minutes.
A retailer has 120 stores across two fixture formats and wants to localise breakfast cereal.
It can change four optional SKU slots per store at the next semiannual reset.
The current policy has one core range and two format-specific layouts.

**Deliverable:** a one-page decision brief.
State the unit, eligible population, comparison policy, objective, horizon, and three constraints.
Identify the owner of the proposed plan and the owner of the measurement.

**Acceptance criteria:** the brief distinguishes demand grouping from fixture execution.
It compares the proposed plan with the current two-layout policy, not with an infeasible perfect plan.
It includes a protected need-state rule, a supply or physical constraint, and reset cost.

### Exercise 2 — Calculate comparable shares

**Time:** 10 minutes.
Store G sells 378 ready-to-eat packs out of 1,200 comparable category packs in 13 weeks.
Store H sells 552 ready-to-eat packs out of 1,600 comparable category packs in the same period.
Assume identical eligibility and reliable availability.

**Deliverable:** each store's ready-to-eat share, the difference in percentage points, and the unweighted mean of the two store shares.
State why that mean is not necessarily the pooled share.

**Acceptance criteria:** percentages and percentage points are labelled correctly.
The mean is based on stores, not units.
The answer identifies the role of unequal denominators.

### Exercise 3 — Critique the demand signal

**Time:** 15 minutes.
A ready-to-eat item was ranged for 14 trading days in a store.
It sold 72 packs on six confirmed in-stock days; availability is missing on the other eight days.
An analyst replaces the eight missing days with zero sales and proposes removing the item.

**Deliverable:** a short critique and two immediate next actions.
Compute the observed in-stock-day rate while clearly stating its limitation.

**Acceptance criteria:** the answer does not infer demand on missing days.
It distinguishes a day-level rate from an unbiased lost-sales estimate.
At least one action improves or checks availability evidence.

### Exercise 4 — Choose an implementable grouping

**Time:** 20 minutes.
A four-group candidate has a silhouette of 0.52 and needs four planograms.
A two-group candidate has a silhouette of 0.41 and needs two planograms.
The planning team can maintain two variants this year, and both candidates satisfy basic quality checks.
A third option is the current single national plan.

**Deliverable:** a decision recommendation and a list of evidence still needed.
Explain whether an exception could serve an important niche without a third standing planogram.

**Acceptance criteria:** the recommendation does not treat silhouette as financial return.
It compares feasible choices and includes reset and recurring complexity costs.
It retains essential need-state coverage and does not promise the two-group option is profitable.

### Exercise 5 — Assess stability

**Time:** 15 minutes.
After aligning group labels across two comparable windows, 18 of 24 common stores retain the same assignment.
Four stores are close to a boundary, and two stores have undergone a major remodel.
The model owner proposes refreshing every month.

**Deliverable:** the observed retention and change percentages, plus a response to the monthly refresh proposal.
Name one label-invariant agreement metric and one reason the raw percentage is incomplete.

**Acceptance criteria:** the denominator is 24 common stores.
The response distinguishes real structural change from noise.
It proposes a way to control unnecessary resets without indefinitely ignoring a real shift.

### Exercise 6 — Build an executable range

**Time:** 25 minutes.
A proposed cluster range uses six optional items, each with a single facing.
Their package widths are 11, 12, 10, 14, 9, and 13 centimeters (cm).
The usable shelf width is 72 cm.
One item is the only suitable option for a protected need state and cannot be removed under the retailer's policy.

**Deliverable:** calculate the occupied width and slack, then propose a feasible next step.
List two operational checks beyond shelf width and explain how an exception would be recorded.

**Acceptance criteria:** the answer recognises that the widths alone do not establish depth, height, supply, or accessibility.
It preserves the protected option and does not assume all products have interchangeable shopper roles.
The exception has an owner, reason, and review date.

### Exercise 7 — Calculate pilot economics

**Time:** 20 minutes.
A randomized pilot has 15 treatment stores observed for 10 post-period weeks.
Average category contribution is USD 200 to USD 221 per store-week in treatment and USD 190 to USD 198 per store-week in control.
Additional reset and handling costs total USD 1,350 and are not included in contribution.

**Deliverable:** calculate the difference-in-differences estimate, gross incremental contribution, net value, and break-even effect.
List two reasons the point estimate alone is insufficient for rollout.

**Acceptance criteria:** the control trend is subtracted before multiplying by treatment store-weeks.
Costs are subtracted exactly once.
The answer names uncertainty and at least one execution or scope issue.

### Exercise 8 — Design and monitor a pilot

**Time:** 30 minutes.
A retailer proposes cluster-specific ranges in 60 stores with different baseline performance.
Stores share several distribution routes, and local promotion calendars vary.
The retailer can randomize stores but must keep supply operational.

**Deliverable:** a one-page protocol with assignment, analysis unit, outcome, guardrails, duration rationale, compliance measurement, and stop or scale rule.
Include the treatment of exceptions and a process for a failed reset.

**Acceptance criteria:** the design preserves assigned treatment for the primary comparison.
It addresses baseline differences, route or promotion effects, and correlated repeated observations.
The rule distinguishes a forecast from an observed estimate and an approved rollout.

### Exercise 9 — Cold start, privacy, and access

**Time:** 25 minutes.
A new store has no sales history.
A proposed model uses identifiable loyalty transactions from nearby stores and demographic characteristics to assign it immediately.
The proposed range would omit the only product serving a small dietary need.

**Deliverable:** propose a provisional assignment process, a minimum-data approach, an access safeguard, and a review trigger.
State what requires legal or privacy review rather than making a universal legal claim.

**Acceptance criteria:** the answer does not pretend the new store has observed demand.
It considers whether aggregated evidence is sufficient and retains the protected need pending review.
It assigns an owner and date for reassessment.

## 13. Self-check answers

### Answer 1 — Decision framing rubric

A strong brief defines store-category as the analytic unit and both fixture formats as the planning population.
The decision concerns four optional cereal slots, subject to the core and format-specific execution rules.
The current core range and two layouts are the counterfactual policy.
A suitable objective is incremental category contribution net of reset and recurring implementation cost over a defined horizon.
The protected-need rule, usable shelf capacity, and supply eligibility are distinct constraints.
Category management can own the proposition while analytics and finance own measurement and reconciliation.
A weak brief gives an algorithm but does not say what a store will do differently.

### Answer 2 — Shares

Store G has 378 / 1,200 = 0.315, or 31.5%.
Store H has 552 / 1,600 = 0.345, or 34.5%.
H exceeds G by 3.0 percentage points.
The unweighted store mean is (31.5% + 34.5%) / 2 = 33.0%.
The pooled unit share is (378 + 552) / (1,200 + 1,600) = 930 / 2,800 = 33.214% approximately.
The pooled share weights the larger-denominator store more heavily.
Neither number is inherently correct without a stated business purpose.

### Answer 3 — Censoring

The observed rate on confirmed in-stock days is 72 / 6 = 12 packs per in-stock day.
The eight missing days are not confirmed out-of-stock days and are not evidence of zero demand.
Even if they were out of stock, 12 packs per day extrapolated to them would be an assumption rather than an observed lost-sales count.
Inspect inventory and shelf-audit records for those days and investigate the missing signal.
Compare with a suitable high-availability period or a controlled test before making a deletion decision.
Report exposure and uncertainty alongside the rate.

### Answer 4 — Group selection rubric

The four-group candidate violates the stated two-variant operating limit unless capacity is deliberately changed.
A higher silhouette does not convert an infeasible plan into a viable one.
Compare the two-group policy and current policy using forecast contribution, reset cost, operating cost, and need-state coverage.
Use a controlled pilot if the benefit is uncertain and worth testing.
A documented exception may serve a specific niche if the supply, fixture, and governance costs are manageable.
If neither new policy has a credible net benefit, keep the current policy.

### Answer 5 — Stability

Retention is 18 / 24 = 75% of common stores.
Change is 6 / 24 = 25% of common stores.
Adjusted Rand index is one label-invariant partition agreement measure.
The retention rate depends on label alignment and does not say whether the changed stores are economically important.
The two remodels may reflect a genuine structural change rather than noise.
Review the four boundary stores and consider a pre-specified hysteresis band and scheduled refresh.
Allow an earlier review for the remodels instead of blindly holding every assignment fixed.

### Answer 6 — Shelf feasibility

The widths sum to 11 + 12 + 10 + 14 + 9 + 13 = 69 cm.
The slack is 72 - 69 = 3 cm before any required dividers or clearances.
The six facings therefore fit the stated linear-width arithmetic only if the 72 cm is genuinely usable and no further clearance is required.
Check package height and depth, capacity between deliveries, and product supply.
Preserve the protected option and seek a space-planner review if clearance is inadequate.
An exception record should identify the store, SKU change, reason, approver, effective date, expiry or review date, and expected cost.

### Answer 7 — Pilot economics

Treatment change is USD 221 - USD 200 = USD 21 per store-week.
Control change is USD 198 - USD 190 = USD 8 per store-week.
The difference-in-differences estimate is USD 21 - USD 8 = USD 13 per store-week.
There are 15 × 10 = 150 treatment store-weeks.
Gross incremental contribution is USD 13 × 150 = USD 1,950.
Net value is USD 1,950 - USD 1,350 = USD 600.
Break-even is USD 1,350 / 150 = USD 9 per store-week.
The point estimate lacks an uncertainty interval and may not generalise to other stores or seasons.
Check compliance, availability, spillovers, and whether the measured scope includes all material costs and impacts.

### Answer 8 — Pilot design rubric

A strong protocol randomly assigns stores, potentially within baseline-performance and format blocks, subject to route feasibility.
It specifies the randomization unit and uses an uncertainty method appropriate to that unit and repeated store observations.
The primary outcome is consistently defined category contribution, with an explicitly stated post-period and pre-period.
The duration should reflect realistic reset timing and the precision required for the decision.
Measure availability, need-state coverage, waste, and actual execution as guardrails.
Retain assigned treatment for the primary intention-to-treat comparison and analyse compliance separately.
Log promotions, route disruptions, exceptions, and the date of each failed or completed reset.
Set stop and scale conditions before examining results and identify the person authorised to decide.

### Answer 9 — Cold-start rubric

A strong answer assigns a conservative, feasible provisional plan based on format and appropriate aggregated evidence rather than fictional store sales.
It preserves the only protected dietary option under the stated business rule.
It asks whether aggregated features suffice before requesting identifiable loyalty data.
It requires applicable privacy and legal review of purpose, access, retention, and the use of demographic variables.
It names a category owner and a review date after sufficient comparable trading and availability history accumulate.
It also checks whether the initial plan fits the fixture and can be supplied.

## 14. Interview questions

### 1. Why cluster stores at all?

A strong answer describes the trade-off between a uniform range and fully bespoke plans, including complexity and reset cost.
It names a decision that a cluster will change.
A weak answer treats clustering as an end in itself.

### 2. Why is sales mix not identical to demand mix?

A strong answer explains range eligibility, stockouts, pricing, promotion, and substitution.
It identifies missing exposure and proposes a test or bounded sensitivity analysis.
A weak answer assumes a zero sale always proves a zero preference.

### 3. How would you select features?

A strong answer starts with category decisions, comparable need-state shares, volume where relevant, and data timing.
It discusses overlapping shares, scale, and the consequences of demographic proxies.
A weak answer includes every available field without an interpretation or quality check.

### 4. How would you choose the number of clusters?

A strong answer compares fit, stability, group size, actionable differences, constraints, and expected net economics against a simple baseline.
It recognises that a higher statistical score is not necessarily a better plan.
A weak answer chooses the maximum silhouette without checking operating capacity.

### 5. What does an unstable assignment mean?

A strong answer considers sampling noise, boundary proximity, feature drift, genuine structural change, and arbitrary label permutation.
It discusses cadence and the cost of resetting stores.
A weak answer declares the model broken solely because label numbers changed.

### 6. How would you assign a new store?

A strong answer uses a documented provisional plan based on available lawful comparable evidence, preserves critical needs, and sets a review date.
It distinguishes an estimated proxy from observed demand.
A weak answer invents a sales history or leaves the store without a range.

### 7. What makes a cluster plan feasible?

A strong answer includes physical shelf fit, coverage, supply, replenishment, local obligations, staffing, and exceptions.
It separates demand grouping from fixture-specific execution.
A weak answer assumes an equal count of slots means equal shelf width.

### 8. How would you measure incremental benefit?

A strong answer states the counterfactual, randomization where feasible, outcome scope, category cannibalisation, implementation costs, uncertainty, and guardrails.
It preserves original assignment in the primary analysis.
A weak answer counts all sales of newly listed items as incremental.

### 9. What can difference-in-differences tell you?

A strong answer calculates the treated change minus control change and explains identification assumptions and concurrent shocks.
It distinguishes randomised assignment from an observational parallel-trends argument.
A weak answer subtracts only the post-period levels and ignores baseline differences.

### 10. How do you govern the model after rollout?

A strong answer versions data and transformations, records assignment and execution, monitors drift and exceptions, and defines rollback and retirement.
It ties refresh frequency to the operational reset cycle.
A weak answer assumes a one-time clustering exercise requires no ongoing ownership.

## 15. Key terms

- [Adjusted Rand index](#4-definitions-and-vocabulary) — label-invariant partition agreement adjusted for chance.
- [Assignment](#step-8--test-stability-and-out-of-sample-usefulness) — versioned mapping of a store-category to a group.
- [Availability](#step-4--separate-observed-sales-from-demand-opportunity) — opportunity to buy an item at the relevant time.
- [Baseline](#step-1--define-the-decision-and-population) — stated reference period or feasible comparison policy.
- [Boundary store](#step-8--test-stability-and-out-of-sample-usefulness) — store with similar distance to competing group profiles.
- [Cannibalisation](#step-10--estimate-value-and-uncertainty) — transfer of sales or contribution from existing items.
- [Category](#4-definitions-and-vocabulary) — versioned commercial grouping of products.
- [Centroid](#candidate-groups-and-boundaries) — mean feature vector of a group's members.
- [Cluster](#3-the-core-mental-model) — stores grouped under a specified method and decision context.
- [Cold start](#answer-9--cold-start-rubric) — assignment before a store has adequate comparable history.
- [Commitment](#pilot-comparison) — approved obligation with owner, scope, resources, and timing.
- [Contribution](#step-10--estimate-value-and-uncertainty) — revenue less specified variable costs.
- [Counterfactual](#step-11--design-a-comparative-pilot) — outcome or policy that would have occurred without the proposed change.
- [Demand mix](#step-5--engineer-demand-led-features) — relative demand among explicitly defined segments.
- [Difference-in-differences](#pilot-comparison) — treatment change minus control change.
- [Distance](#step-6--transform-weight-and-inspect-features) — chosen dissimilarity between feature vectors.
- [Eligible exposure](#step-4--separate-observed-sales-from-demand-opportunity) — time or days in which a product could be sold under a defined rule.
- [Exception](#step-9--turn-profiles-into-feasible-plans) — approved and reviewed departure from the default plan.
- [Facing](#step-9--turn-profiles-into-feasible-plans) — one visible product-width presentation on a shelf.
- [Feature](#step-5--engineer-demand-led-features) — measured input to grouping with defined grain and timing.
- [Forecast](#step-10--estimate-value-and-uncertainty) — conditional estimate of future performance.
- [Hysteresis](#step-12--deploy-monitor-and-retire) — a rule that avoids reassignment on small or transient changes.
- [Incrementality](#step-11--design-a-comparative-pilot) — change relative to a credible counterfactual.
- [Medoid](#step-7--fit-candidate-groupings) — observed member minimising within-group dissimilarity.
- [Need state](#step-2--map-decisions-to-shopper-needs) — customer problem or occasion the assortment should serve.
- [Planogram](#step-9--turn-profiles-into-feasible-plans) — versioned specification of product placement.
- [Silhouette](#step-7--fit-candidate-groupings) — distance-based indicator of within-group cohesion and separation.
- [Stability](#step-8--test-stability-and-out-of-sample-usefulness) — reproducibility under a specified perturbation.
- [Stock-keeping unit](#4-definitions-and-vocabulary) — retailer-defined sellable item or variant.
- [Target](#pilot-comparison) — desired outcome, not an observed effect or approved commitment.

## 16. Sources

1. Rousseeuw, Peter J. (1987), "Silhouettes: A Graphical Aid to the Interpretation and Validation of Cluster Analysis," *Journal of Computational and Applied Mathematics*, 20, 53–65. [DOI: 10.1016/0377-0427(87)90125-7](https://doi.org/10.1016/0377-0427(87)90125-7). Method reference for silhouette; the measure is not a commercial objective.
2. Hubert, Lawrence, and Phipps Arabie (1985), "Comparing Partitions," *Journal of Classification*, 2, 193–218. [DOI: 10.1007/BF01908075](https://doi.org/10.1007/BF01908075). Source for adjusted partition comparison.
3. scikit-learn, [Clustering user guide](https://scikit-learn.org/stable/modules/clustering.html). Official documentation for clustering algorithms, evaluation, and implementation details; consult the version used in a project.
4. scikit-learn, [Common pitfalls and recommended practices](https://scikit-learn.org/stable/common_pitfalls.html). Official documentation on preprocessing leakage and consistent transformations.
5. GS1, [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/). Official trade-item identifier allocation guidance; check the current rule for the relevant product change.
6. GS1, [Global Location Number](https://www.gs1.org/standards/id-keys/gln). Official description of location and party identification; identifiers do not supply physical fixture attributes.
7. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Primary legal text, including Article 5 principles; applicability depends on circumstances and jurisdiction.
8. RELEX Solutions, [East of England Co-op case study](https://www.relexsolutions.com/resources/east-of-england-co-op/). Named public vendor case concerning localised planograms; reported outcomes are vendor claims, not independent causal estimates.
9. PepsiCo, [Clustering Assortment to Meet Localized Shopper Demand](https://www.pepviz.com/prod/s3fs-public/2021-07/pepviz-clustering-assortment-to-meeet-localized-shopper-demand.pdf). Supplier case study of small-format salty-snack assortment; interpret in light of source incentives and disclosed methods.
10. Angrist, Joshua D., and Jörn-Steffen Pischke (2009), *Mostly Harmless Econometrics: An Empiricist's Companion*, Princeton University Press. Reference for causal comparison and difference-in-differences assumptions.

Sources support definitions, methods, and named cases; the numerical example is fictional.
Legal and regulatory references are educational and were framed as of November 2026; seek current local advice for an actual deployment.

## 17. Links

- [Module 17 topic index](_index.md).
- [Module 17 survey](../../../course/17_assortment_category_retail_analytics.md).
- [Assortment decision stack](01_assortment-decision-stack.md).
- [Category roles and decision trees](03_category-roles.md).
- [Space elasticity and planograms](06_space-elasticity.md).
- [Simulation and optimization production labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).
