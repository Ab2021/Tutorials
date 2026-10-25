# Trial Designs

> Module 05 — Study Design and Evidence Generation
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01
> Version 2026.3

## 1. Why this topic exists

The trial design is the structure that turns a question into an identifiable effect. It decides what can be estimated, what cannot, and what assumptions must be defended. A drug, device, or model can be evaluated in a parallel-group trial, a crossover, a factorial, a cluster trial, a stepped-wedge, a platform, a basket, an umbrella, an adaptive trial, a decentralised trial, or a single-arm study with an external control. These are not variations on one template. They answer different questions, carry different biases, and require different arithmetic.

Industry teams often inherit a design choice made for commercial or logistical reasons and then ask the statistician to "make it work." The request is usually impossible in the way it is framed. A design chosen because it is cheap cannot be repaired by analysis if it cannot identify the estimand. A design chosen because a competitor used it may not fit the product's claim. The professional skill is to select the design from the question and the constraints, and to explain the trade-offs in language a programme team can act on.

The commercial stakes are direct. A parallel-group superiority trial is the default for a reason: it is robust to many assumptions and familiar to regulators. A crossover trial can be far smaller but only for stable, reversible conditions with no carryover. A cluster trial is necessary when the intervention is delivered at the group level, but it can require many more participants than an individually randomised trial. A platform trial can answer several questions at once, but it needs a governance structure and a shared control that regulators will accept. Choosing the wrong structure costs either precision or credibility, and sometimes both.

This topic gives you a design-selection procedure, the arithmetic for the common structures, and a worked adaptive trial carried from estimand to decision. It also gives you the failure modes that appear in protocol review: the crossover with a carryover problem, the cluster trial powered as if individuals were independent, the platform with an unadjusted shared control, the adaptive design with an interim that nobody simulated.

### What a design must deliver

A trial design has five jobs:

1. **Identify the estimand.** The structure must make the contrast of interest estimable, under stated assumptions.
2. **Control the plausible alternatives.** Randomisation, blinding, and concurrent controls address confounding, ascertainment, and secular trends.
3. **Deliver enough information.** The sample size and follow-up must produce a precise enough estimate for the decision.
4. **Be executable.** Recruitment, retention, delivery, and data capture must be feasible in the intended settings.
5. **Be defensible.** The design must survive review by a regulator, an ethics committee, a payer, and a clinician.

Most design debates are really debates about which of these five jobs is being sacrificed. A team that proposes a single-arm study is sacrificing job 2 for job 4. A team that proposes a large pragmatic trial is sacrificing job 4 for job 3 and job 5. Making the sacrifice explicit is the first step toward a design that the organisation can defend.

### Why design selection is not a menu

A menu implies that any design can answer any question if the sample is large enough. That is false. A crossover design cannot estimate a cure effect because the first period changes the second. A cluster design cannot estimate an individual-level effect without assumptions about interference. A platform trial cannot compare arms that were never randomised concurrently without a bridge. A single-arm trial cannot estimate a comparative effect at all; it can only compare the observed outcome with an external benchmark.

The design is therefore a claim about the world: that the condition is stable, that the intervention is reversible, that clusters are exchangeable, that the external control is comparable. If the claim is false, the design fails regardless of the analysis. The protocol must state the claim and the evidence for it.

### The cost of redesign

Redesign is expensive in a way that is easy to underestimate. A trial that stops for futility after two years has consumed the site relationships, the recruitment infrastructure, and the goodwill of investigators. A trial that fails because the design could not identify the effect has consumed all of that and produced no usable evidence. A trial that succeeds but answers the wrong question is the most expensive outcome, because it produces a result that the organisation must then explain away.

The cheapest design decision is the one made before the protocol is written. This topic is therefore placed early in the module, before sample size, estimands, and protocol writing, so that the design vocabulary is available when those topics are studied.

### What trial designs are not

Four misconceptions cause most of the avoidable design errors.

**A design is not a statistical test.** The choice between a t-test and a mixed model is an analysis decision. The choice between a parallel and a cluster trial is a design decision. Confusing the two leads teams to believe that a different analysis can rescue a design that cannot identify the effect.

**A design is not a sample size.** A large trial with the wrong comparator is still the wrong trial. Size buys precision, not validity. A small trial with a credible design can be more informative than a large trial with a biased one.

**A design is not a regulatory formality.** The protocol must satisfy the regulator, but the design exists to answer the question. A design that is optimised for the checklist and not for the question will produce a result that the organisation cannot use.

**A design is not fixed once the protocol is signed.** Designs can be amended, but amendments after unblinding are not design changes; they are analysis changes. The distinction matters for the integrity of the trial and for the credibility of the result.

### The design review conversation

A design review is a conversation with four participants: the clinician, the statistician, the operations lead, and the commercial lead. Each brings a different constraint.

- The clinician asks whether the design is ethical and whether it will answer a question that matters in practice.
- The statistician asks whether the design can identify the estimand and whether the operating characteristics are acceptable.
- The operations lead asks whether the design can be delivered on time and on budget.
- The commercial lead asks whether the result will support the claim and the price.

A good design satisfies all four, or states explicitly which one is being traded away. A design that satisfies only the statistician may be undeliverable. A design that satisfies only the commercial lead may be uninterpretable. The review should end with a written record of the trade-offs, not only a decision.

### Why the design vocabulary matters to a data scientist

A data scientist is often the person who sees the data first. That position creates a responsibility: to notice when the data cannot answer the question, and to say so before the analysis begins. A model can be fitted to any dataset. It cannot create a counterfactual that the design did not provide.

Knowing the design vocabulary lets you ask the right questions at the right time: What is the unit of allocation? What is the control? Is the analysis population prespecified? Was the interim rule fixed before unblinding? These questions are more valuable than a sophisticated model applied to a flawed design.

## 2. Learning outcomes

By the end of this topic you can:

1. **Select** a trial structure (parallel, crossover, factorial, cluster, stepped-wedge, platform, basket, umbrella, adaptive, single-arm) from the estimand and the delivery level, and justify the choice against at least two alternatives. (Exercise 1, self-check 1)
2. **Specify** allocation, stratification, blinding, and the analysis population for a given design, and explain how each choice affects bias. (Exercise 2, self-check 3)
3. **Compute** the sample size for a parallel trial, a crossover trial, and a cluster trial, including the design effect and the inflation for unequal cluster sizes. (Exercises 3 and 4, self-check 4)
4. **Design** a prespecified interim analysis for futility or efficacy, and explain why the operating characteristics must be simulated rather than assumed. (Exercise 5, self-check 5)
5. **Distinguish** a platform trial from a set of separate trials, and identify the multiplicity and shared-control problems that a platform creates. (Exercise 6, interview 6)
6. **Evaluate** a decentralised trial plan for data integrity, endpoint ascertainment, and regulatory acceptability. (Exercise 7, interview 8)
7. **Critique** a published trial design and propose a defensible alternative with the associated cost. (Exercise 8, interview 9)

The outcomes are cumulative. Outcome 4 assumes outcome 1, and outcome 7 assumes all of the preceding outcomes.

## 3. The core mental model

A trial design is a set of nested choices. Each choice constrains the next. The model below shows the order in which the choices should be made and the feedback loop when a constraint makes an earlier choice infeasible.

```text
                    ESTIMAND
        population | treatment | variable |
        intercurrent events | summary
                       |
                       v
              UNIT OF ALLOCATION
        individual | cluster | period | sequence
                       |
                       v
              COMPARISON STRUCTURE
   parallel | crossover | factorial | cluster |
   stepped-wedge | platform | basket | umbrella
                       |
                       v
              CONTROL AND BLINDING
   placebo | active | usual care | external |
   open | single | double | assessor
                       |
                       v
              ALLOCATION AND STRATIFICATION
   1:1 | 2:1 | adaptive | minimisation | strata
                       |
                       v
              FOLLOW-UP AND ASSESSMENT
   visits | windows | remote | registry linkage
                       |
                       v
              ANALYSIS POPULATION AND ESTIMATOR
   ITT | per-protocol | mITT | principal stratum
                       |
                       v
              INTERIM AND ADAPTATION (optional)
   futility | efficacy | SSR | enrichment | arm drop
                       |
                       v
              SIZE AND OPERATING CHARACTERISTICS
   analytical | simulation | sensitivity table
                       |
                       +---- if infeasible ----> revise structure
```

The vertical order matters. The unit of allocation determines whether the design is a cluster trial. The comparison structure determines which biases are controlled by design and which must be handled by analysis. The control and blinding determine what the comparator actually is. The allocation ratio determines the sample size and the precision of the comparison. The follow-up determines the horizon and the missing-data problem. The analysis population must match the estimand, not the other way around.

### The unit-of-allocation decision

The most consequential choice is the unit of allocation. If the intervention is delivered to an individual, individual randomisation is usually possible. If the intervention is delivered to a ward, clinic, hospital, or region, individual randomisation may be impossible without contamination: a clinician who has learned the new workflow cannot unlearn it for control patients.

| Delivery level | Unit of allocation | Design consequence |
|---|---|---|
| Individual patient | Patient | Parallel, crossover, factorial |
| Clinician or care team | Clinician or team | Cluster, with clustering by team |
| Ward or clinic | Ward or clinic | Cluster or stepped-wedge |
| Hospital | Hospital | Cluster, stepped-wedge, or matched pair |
| Region or policy | Region | Stepped-wedge or difference-in-differences |
| Sequence of treatments within patient | Period | Crossover or N-of-1 |

The unit of allocation is not always the unit of analysis. In a cluster trial, the unit of analysis may be the individual, but the variance must account for the cluster. In a crossover trial, the unit of analysis is the participant-period, and the variance must account for within-person correlation.

### The comparison-structure decision

Each structure answers a different question.

**Parallel groups.** Participants are randomised to one strategy for the duration. The structure estimates a between-group contrast. It is robust to carryover and period effects because there are no periods. It is the default for confirmatory trials.

**Crossover.** Each participant receives both strategies in a randomised order. The structure estimates a within-person contrast, which removes between-person variability and can be much more efficient. It requires a stable condition, a reversible treatment, and no carryover. It cannot estimate a cure or a durable modification of disease.

**Factorial.** Participants are randomised on two or more factors simultaneously. The structure estimates main effects and interactions. It is efficient when the factors are independent and the interaction is not the question. It becomes inefficient when the interaction is large, because the main-effect estimate is then an average over incompatible subgroups.

**Cluster.** Groups are randomised. The structure estimates a group-level effect or an individual-level effect under assumptions about interference. It is necessary for delivery-level interventions and for contamination control. It is less efficient than individual randomisation when the intracluster correlation is positive.

**Stepped-wedge.** Clusters cross from control to intervention in a randomised sequence. The structure is useful when the intervention cannot be withheld and when roll-out is inevitable. It confounds time with treatment unless the analysis accounts for secular trends, and it is sensitive to missing data at the cluster level.

**Platform.** A master protocol with a shared control and multiple experimental arms that can enter and leave over time. The structure is efficient for a disease with many candidate treatments and a stable infrastructure. It requires a governance body, a shared endpoint, and a prespecified strategy for non-concurrent comparisons.

**Basket.** One treatment is tested across multiple diseases or subtypes in a master protocol. The structure tests whether the treatment works in any of several populations. It requires a prespecified decision rule for pooling and for declaring a subtype effect.

**Umbrella.** Multiple treatments are tested in one disease, often with biomarker-defined subgroups. The structure is a platform restricted to one disease. It shares the platform's governance and multiplicity problems.

**Adaptive.** The design allows prespecified changes based on interim data: stopping for efficacy or futility, re-estimating the sample size, dropping arms, or enriching the population. The structure can be more efficient and more ethical, but it requires simulation and strict data access control.

**Single-arm with external control.** One treatment is given to all participants, and the comparison is made with data from outside the trial. The structure is used when randomisation is unethical or infeasible, particularly in rare diseases or device studies. It is the weakest comparative structure because it cannot control unmeasured confounding or secular trends by design.

### The bias-control map

Different structures control different biases by design. The table is a reminder that analysis cannot fully compensate for a structural weakness.

| Structure | Confounding | Selection | Ascertainment | Secular trend | Carryover |
|---|---|---|---|---|---|
| Parallel randomised | Design | Design | Blinding | Concurrent control | Not applicable |
| Crossover | Design | Design | Blinding | Within-person | Design and washout |
| Factorial | Design | Design | Blinding | Concurrent control | Not applicable |
| Cluster randomised | Design at cluster level | Design at cluster level | Blinding | Concurrent control | Contamination possible |
| Stepped-wedge | Design plus time adjustment | Design | Blinding | Analysis and design | Contamination possible |
| Platform | Design for concurrent arms | Design | Blinding | Concurrent control for concurrent arms | Not applicable |
| Single-arm external | Analysis only | Analysis only | Analysis only | Analysis only | Not applicable |

The single-arm row is the important one. Every bias is addressed by analysis, which means every bias depends on an unverifiable assumption. That does not make single-arm studies worthless; it makes them weak evidence for a comparative claim.

### The efficiency frontier

Designs differ in efficiency. For the same power, a crossover trial usually needs fewer participants than a parallel trial; a cluster trial usually needs more; a factorial trial can answer two questions for less than two trials; a platform can share infrastructure and control participants. The efficiency gain is always paid for with an assumption. The crossover pays with the no-carryover assumption. The cluster pays with the exchangeability of clusters. The platform pays with the comparability of arms over time. The factorial pays with the no-interaction assumption.

The design review should state the assumption and the price. "This design saves 40 percent of the sample but requires that the treatment have no carryover beyond a two-week washout; the pilot suggests the effect resolves within five days" is a defensible statement. "This design is more efficient" is not.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Parallel-group trial | Participants are randomised to one of two or more strategies for the duration of follow-up | Used as a synonym for "randomised trial" | Design category |
| Crossover trial | Each participant receives two or more strategies in a randomised sequence | Used when the condition is not stable | Design category |
| Carryover | The effect of one period's treatment persists into the next period | Ignored when the treatment is long-acting | Effect, outcome units |
| Period effect | A change in outcome over time that is unrelated to treatment | Confused with carryover | Effect, outcome units |
| Washout | A treatment-free interval between periods | Omitted to reduce trial duration | Duration |
| Factorial design | Randomisation on two or more factors simultaneously | Used when a large interaction is likely | Design category |
| Cluster randomised trial | Randomisation of groups rather than individuals | Analysed as if individuals were independent | Design category |
| Intracluster correlation coefficient (ICC) | The proportion of total variance attributable to between-cluster variation | Assumed to be zero without evidence | Proportion, 0 to 1 |
| Design effect | The factor by which the sample size must be multiplied to account for clustering: 1 + (m - 1) * ICC | Applied without checking cluster size variation | Multiplier |
| Stepped-wedge | Clusters switch from control to intervention in a randomised sequence | Treated as a substitute for randomisation without time adjustment | Design category |
| Platform trial | A master protocol with a shared control and multiple experimental arms | Treated as several independent trials | Design category |
| Basket trial | One treatment tested across multiple diseases or subtypes | Pooled without prespecified decision rules | Design category |
| Umbrella trial | Multiple treatments tested in one disease | Confused with a platform trial | Design category |
| Master protocol | A single protocol governing multiple substudies or arms | Treated as a single trial with one analysis | Document |
| Adaptive design | A design with prespecified modifications based on interim data | Any unplanned change after looking at data | Design category |
| Futility | Stopping a trial when the interim data suggest the target effect is unlikely | Confused with harm | Decision rule |
| Sample-size re-estimation (SSR) | Adjusting the sample size based on interim information | Unblinded re-estimation without alpha control | Procedure |
| Seamless design | A Phase 2 and Phase 3 trial combined in one protocol | Two separate trials with a shared protocol number | Design category |
| Enrichment | Selecting participants more likely to respond | Post hoc subgroup selection | Design feature |
| Allocation ratio | The ratio of participants assigned to each arm | Chosen for convenience without power implications | Ratio |
| Stratification | Randomisation within defined subgroups to balance prognostic factors | Confused with subgroup analysis | Procedure |
| Minimisation | Adaptive allocation balancing prognostic factors across arms | Described as randomisation without qualification | Procedure |
| Blinding | Concealment of allocation from participants, clinicians, assessors, or analysts | Used as a synonym for allocation concealment | Procedure |
| Open-label | No blinding | Assumed to be biased in all cases | Procedure |
| Intention-to-treat (ITT) | Analysis by assigned group regardless of treatment received | Applied to a subset with complete data | Principle |
| Modified ITT (mITT) | An ITT-like population with prespecified exclusions | Defined after seeing the data | Population |
| Per-protocol | Analysis restricted to adherent participants | Presented as the primary analysis when ITT is null | Population |
| Estimand | The precise quantity to be estimated | Confused with the endpoint | Structured definition |
| Intercurrent event | An event after treatment initiation that affects interpretation | Treated as missing data | Event category |
| Contamination | Control participants receive the intervention or its effects | Ignored in cluster trials | Proportion |
| Co-intervention | A treatment given alongside the assigned strategy | Not recorded | Variable |
| Decentralised trial | A trial with remote visits, local providers, or digital data capture | Assumed to be less rigorous | Design feature |
| Digital endpoint | An endpoint measured by a sensor or app | Assumed to be valid without verification | Variable |
| Data monitoring committee (DMC) | An independent group reviewing interim data | Confused with the trial steering committee | Governance body |

Two distinctions recur in design review.

**Allocation concealment versus blinding.** Allocation concealment prevents the person enrolling a participant from knowing the next assignment. Blinding prevents participants, clinicians, or assessors from knowing the assignment after enrolment. A trial can have concealed allocation and no blinding (an open-label trial), and it can have blinding but poor concealment (for example, if the randomisation list is visible). Both matter, and they address different biases.

**Futility versus harm.** Futility means the trial is unlikely to demonstrate the target benefit. Harm means the intervention is causing injury. A futility boundary can stop a trial that is safe but ineffective. A harm signal requires a separate monitoring rule and often a different governance response. Conflating the two can cause a trial to continue after a safety signal because the efficacy boundary has not been crossed.

## 5. The mechanics, step by step

The procedure has fourteen steps. Steps 1 to 3 fix the question and the unit of allocation. Steps 4 to 7 specify the operational structure. Steps 8 to 10 handle adaptation, clustering, and decentralisation. Steps 11 to 14 size, simulate, and govern the design.

### Step 1: Restate the estimand and the decision

**Input:** the claim translation from Topic 01, including the estimand and the decision rule.

**Output:** a one-paragraph design brief stating the population, treatment conditions, variable, intercurrent-event strategy, summary measure, decision, and acceptable uncertainty.

**Rationale:** The design brief is the reference document for every later choice. Without it, design debates become preferences. With it, each choice can be tested against the question.

The brief should state the decision threshold explicitly. "We need a 95 percent confidence interval excluding a 20 percent relative reduction" is a design constraint. "We need to show a benefit" is not.

### Step 2: Choose the unit of allocation

**Input:** the delivery level of the intervention and the risk of contamination.

**Output:** the unit of allocation (patient, clinician, ward, hospital, region, period) and a justification.

**Rationale:** The unit of allocation determines the design family and the variance structure. It is the first irreversible choice in most protocols.

Use the following decision logic.

| Condition | Unit | Consequence |
|---|---|---|
| Intervention delivered to an individual and no contamination | Patient | Individual randomisation |
| Intervention delivered by a clinician who treats both arms | Clinician or team | Cluster or cluster-crossover |
| Intervention changes ward or clinic workflow | Ward or clinic | Cluster or stepped-wedge |
| Intervention is a policy or facility change | Hospital or region | Cluster, stepped-wedge, or difference-in-differences |
| Intervention is a sequence of short-acting treatments | Period within patient | Crossover or N-of-1 |
| Intervention cannot be withheld and roll-out is fixed | Cluster by time | Stepped-wedge |

If the unit is a cluster, the design must specify the cluster size, the number of clusters, and the ICC assumption. If the unit is a period, the design must specify the washout and the carryover assumption.

### Step 3: Choose the comparison structure

**Input:** the unit of allocation, the condition, and the question.

**Output:** a structure from the list in Section 3, with at least two rejected alternatives and the reason for rejection.

**Rationale:** The structure determines which biases are controlled by design and which must be handled by analysis. The rejection reasons are part of the evidence that the choice was reasoned.

The branch table below is the core of this step.

| If the question is... | And the condition is... | Then the structure is... | Because... |
|---|---|---|---|
| Superiority of one strategy | Any | Parallel group | Robust and familiar |
| Superiority, stable reversible condition | Chronic, stable, short-acting treatment | Crossover | Efficient within-person contrast |
| Two independent questions | No expected interaction | Factorial | Two answers for one trial |
| Delivery-level intervention | Ward or clinic | Cluster | Avoids contamination |
| Roll-out cannot be withheld | Facility-level | Stepped-wedge | Ethical and feasible |
| Many candidate treatments | High-event disease with infrastructure | Platform | Shared control and infrastructure |
| One treatment, several diseases | Biomarker-defined subtypes | Basket | Tests a shared mechanism |
| Several treatments, one disease | Biomarker-defined subgroups | Umbrella | Tests multiple options |
| Rare disease, no feasible control | Well-characterised natural history | Single-arm external control | Feasibility, with assumptions |
| Early signal, dose finding | Any | Adaptive or seamless | Efficiency and ethical stopping |

The table is not exhaustive. A trial can combine structures: a cluster randomised platform, a crossover with an adaptive dose, a factorial embedded in a platform. The combination must be justified, and the analysis must account for all the structure's features.

### Step 4: Choose the control

**Input:** the standard of care, the ethics of the comparison, and the claim.

**Output:** a named control (placebo, active comparator, usual care, no treatment, external control) with a justification.

**Rationale:** The control defines the counterfactual. "Usual care" is not a control until it is described, because usual care varies by site and era.

The choice is constrained by ethics and by assay sensitivity. A placebo control is acceptable when there is no proven effective treatment, or when withholding treatment for the trial duration poses minimal risk. An active control is required when a proven treatment exists and withholding it would be unethical. A usual-care control is appropriate for a pragmatic question but requires a detailed description of the care pathway. An external control is appropriate only when a concurrent control is infeasible and the external data are comparable.

### Step 5: Choose the allocation ratio and stratification

**Input:** the control, the expected effect, and the prognostic factors.

**Output:** an allocation ratio, a stratification scheme, and a randomisation method.

**Rationale:** The allocation ratio affects power and the acceptability of the trial to participants. Equal allocation is most efficient for a two-arm superiority trial. Unequal allocation, such as 2:1, can be more acceptable when the intervention is promising, but it costs power.

Stratification balances prognostic factors across arms. The factors should be few, prespecified, and strongly prognostic. Over-stratification creates sparse strata and can complicate the analysis. Minimisation can balance more factors but must be described accurately and must include a random component to preserve allocation concealment.

### Step 6: Choose the blinding strategy

**Input:** the intervention, the control, and the outcome.

**Output:** a blinding statement covering participants, clinicians, outcome assessors, and analysts.

**Rationale:** Blinding reduces performance and detection bias. It is not always possible, and it is not always necessary. An objective outcome such as all-cause mortality is less susceptible to detection bias than a subjective outcome such as pain score.

The blinding statement should be specific. "Double-blind" is not sufficient. State who is blinded, how the blinding is maintained, what happens when unblinding is necessary, and how the analysis protects the blind.

### Step 7: Choose follow-up, assessment, and the analysis population

**Input:** the estimand, the outcome, and the expected course of the condition.

**Output:** a visit schedule with windows, a definition of the primary time point, and the analysis population.

**Rationale:** The follow-up must cover the period over which the effect is expected and the decision is made. Assessment windows must be wide enough to be feasible and narrow enough to preserve the time point.

The analysis population must match the estimand. A treatment-policy estimand requires ITT. A hypothetical estimand may require per-protocol or principal-stratum analysis. A safety estimand may require the as-treated population. The protocol should define each population precisely and state which one is primary.

### Step 8: Decide whether to adapt

**Input:** the expected effect, the uncertainty, the ethics of continuing, and the cost of the trial.

**Output:** either a fixed design or a prespecified adaptation plan with operating characteristics.

**Rationale:** Adaptation can reduce exposure to ineffective treatments and improve efficiency. It also adds complexity, requires unblinded access, and can bias the trial if not properly controlled.

The main adaptation types are:

| Adaptation | Trigger | Effect on design | Alpha control |
|---|---|---|---|
| Futility stopping | Interim effect below a prespecified boundary | Reduces sample size and exposure | Usually conservative; no inflation |
| Efficacy stopping | Interim effect above a prespecified boundary | Reduces sample size | Alpha spending function required |
| Sample-size re-estimation | Blinded or unblinded interim variance or event rate | Changes final size | Blinded: usually no adjustment; unblinded: requires methods |
| Arm dropping | Interim data on multiple arms | Reduces sample size | Multiplicity control required |
| Enrichment | Interim signal in a subgroup | Restricts population | Prespecification and alpha control |
| Seamless Phase 2/3 | End of Phase 2 | Combines two trials | Careful control of type I error |

A fixed design is not automatically less efficient. An adaptive design that is poorly specified can be less powerful and harder to defend than a fixed design. The test is whether the adaptation improves the expected decision, not whether it is fashionable.

### Step 9: If cluster, specify the clustering

**Input:** the unit of allocation and the expected cluster sizes.

**Output:** the number of clusters, the expected cluster size, the ICC assumption, the cluster-size variation assumption, and the analysis method.

**Rationale:** Clustering reduces the effective sample size. The design effect is 1 + (m - 1) * ICC, where m is the average cluster size. If clusters vary in size, the design effect is larger than the simple formula suggests.

The number of clusters is often the binding constraint. With a fixed number of clusters, increasing the cluster size has diminishing returns, because the ICC determines how much independent information each additional participant provides. A trial with too few clusters cannot be rescued by enrolling more participants per cluster.

A useful feasibility check for a cluster trial with a fixed number of clusters k and an individually randomised sample size n per arm is whether k is greater than n times the ICC. If it is not, the cluster design cannot achieve the target power even with an infinite cluster size. The check is approximate, but it is a fast way to discover that the available cluster count is inadequate before the protocol is written.

Cluster size variation is the second constraint. If some clusters are much larger than others, the effective information is lower than the average cluster size suggests. The protocol should report the expected distribution of cluster sizes, not only the mean, and the sample size should use the modified design effect when the coefficient of variation is above about 0.23.

A third constraint is the analysis method. A cluster trial can be analysed with a cluster-level summary, a mixed-effects model, or a generalised estimating equation. Each method makes different assumptions about the correlation structure. The protocol should name the primary method and prespecify a sensitivity analysis with an alternative method.

### Step 10: If decentralised, specify the elements and the safeguards

**Input:** the geographical spread, the intervention, and the endpoints.

**Output:** a list of decentralised elements (remote consent, local labs, home visits, digital endpoints, direct-to-participant shipping) and the safeguards for each.

**Rationale:** Decentralised elements can improve access and retention. They can also introduce measurement variability, data provenance problems, and regulatory uncertainty.

For each element, state the source of the data, the validation method, the plan for missing data, and the regulatory basis. A home-collected sample analysed at a local laboratory is not the same measurement as a sample analysed at a central laboratory. The protocol must say which is used for the primary endpoint and why.

### Step 11: Size the design

**Input:** the effect, the variance or event rate, alpha, power, allocation ratio, dropout, and clustering.

**Output:** a sample size with an arithmetic trail and a sensitivity table.

**Rationale:** The sample size is the translation of the design into a budget. It must reflect the design's variance structure, not an idealized individual randomised trial.

The formulas for the common structures are in Section 6 and Section 7. The key point for this step is that the sample size must be computed for the actual design, including the allocation ratio, the clustering, the crossover correlation, and the expected dropout.

The general form for a two-arm comparison of proportions with allocation ratio r (arm 2 divided by arm 1) is:

```text
n1 = (z_alpha/2 + z_beta)^2 * [p1(1-p1) + p2(1-p2)/r] / (p1 - p2)^2
n2 = r * n1
```

For a continuous outcome with a common standard deviation, the corresponding form is:

```text
n1 = (z_alpha/2 + z_beta)^2 * sigma^2 * (1 + 1/r) / delta^2
```

For a time-to-event outcome, the number of events rather than participants drives power. The Schoenfeld formula gives the required number of events:

```text
d = (z_alpha/2 + z_beta)^2 / (log(HR))^2 / (p1 * p2)
```

where p1 and p2 are the allocation proportions. The number of participants is the number of events divided by the expected event probability over the follow-up period.

For a crossover trial with a continuous outcome, the paired analysis uses the standard deviation of the within-person difference:

```text
n_pairs_per_sequence = (z_alpha/2 + z_beta)^2 * sigma_within^2 / delta^2
```

For a cluster trial, the individually randomised sample size is multiplied by the design effect:

```text
DE = 1 + (m - 1) * ICC
```

and, when cluster sizes vary, by the modified design effect:

```text
DE = 1 + ((1 + CV^2) * m - 1) * ICC
```

These formulas are approximations. They assume independent observations, a known variance, and a normal approximation. The protocol should state the approximation and, for small samples or extreme proportions, use an exact or simulation-based method.

### Step 12: Simulate the operating characteristics

**Input:** the design, the adaptation rules, and the assumed parameters.

**Output:** a simulation report with type I error, power, expected sample size, and the probability of each decision, under a range of scenarios.

**Rationale:** Analytical formulas are unavailable or unreliable for many adaptive designs. Simulation is the only way to know how the design behaves.

The simulation report should include the scenarios, the number of replicates, the random seed, the code, and the assumptions. A simulation with 1,000 replicates has substantial Monte Carlo error; 10,000 or more is preferable for a confirmatory design.

### Step 13: Specify implementation and fidelity

**Input:** the intervention, the sites, and the training plan.

**Output:** a fidelity plan with the components that must be delivered, the monitoring method, and the definition of a protocol deviation.

**Rationale:** A design assumes that the intervention is delivered as described. If delivery varies across sites, the estimated effect is an average over different interventions, and the claim becomes ambiguous.

Fidelity is particularly important for behavioural, digital, and workflow interventions. The protocol should define the minimum acceptable delivery, the method for measuring it, and the analysis that handles deviations.

A fidelity plan has three components. The first is the intervention specification: the components that must be delivered, the frequency, and the expected dose. The second is the monitoring method: site self-report, direct observation, system logs, or a combination. The third is the response to deviations: retraining, site-level corrective action, or exclusion from the per-protocol population. The three components should be written before the trial starts, because a fidelity problem discovered at the end of the trial cannot be fixed.

For digital interventions, the system log is often the best fidelity measure. It records whether the alert fired, whether the clinician opened it, and how long the response took. The protocol should state which log fields are used, how missing logs are handled, and whether the log is a source document. A log that is incomplete or manipulable is not a fidelity measure.

### Step 14: Establish governance

**Input:** the design, the risks, and the regulatory context.

**Output:** a DMC charter, a registration entry, a statistical analysis plan (SAP), and a publication plan.

**Rationale:** Governance protects the trial from bias and from pressure to change the analysis after results. The DMC reviews interim data and recommends continuation or stopping. The SAP prespecifies the analysis. The registration makes the trial discoverable and reduces selective reporting.

The DMC charter should state who sees what, when, and how decisions are made. The SAP should be finalized before unblinding. The registration should include the primary endpoint and the analysis plan.

The DMC's independence is a design feature, not an administrative detail. If the sponsor's team can see unblinded comparative results, the design's protection against bias is weakened, even if no one acts on the information. The protocol should state the firewall: who prepares the unblinded output, who receives it, and how the blinded team is kept separate. For adaptive trials, the firewall is even more important, because the adaptation rule depends on unblinded data.

A second governance feature is the change-control procedure. Any change to the design after the protocol is signed should be documented, justified, and, where required, approved by the regulator and ethics committee before implementation. A change made after unblinding is not a design change; it is an analysis change, and it must be reported as such.

### Common design combinations

Real trials often combine structures. The combination must be described precisely, because each component adds an assumption and a reporting requirement.

| Combination | Why it is used | Added assumption | Reporting implication |
|---|---|---|---|
| Cluster randomised platform | Deliver a ward-level intervention while testing several components | Clusters are exchangeable across arms and periods | CONSORT cluster extension plus platform reporting |
| Crossover with adaptive dose | Find a dose within participants | No carryover and a stable dose-response | Crossover reporting plus dose-selection detail |
| Factorial within a platform | Test two interventions in one disease | No material interaction between the factors | Factorial reporting and interaction analysis |
| Stepped-wedge with a concurrent control | Roll out a policy while retaining a randomised comparison | Time trends are correctly modelled | Stepped-wedge extension and time-adjustment detail |
| Basket with a Bayesian hierarchical model | Borrow information across subtypes | Subtypes are exchangeable to a specified degree | Prior specification and borrowing diagnostics |
| Decentralised cluster trial | Deliver a digital intervention across clinics | Remote measurement is valid across sites | DCT guidance plus cluster reporting |
| Seamless Phase 2/3 with enrichment | Move from a broad population to a responsive subgroup | The subgroup effect is stable across stages | Estimand stability and alpha control |

The combination table is also a warning. Each added component increases the protocol's complexity and the number of ways it can fail. A design that combines a platform, a cluster structure, and an adaptive enrichment needs a simulation that reflects all three, not three separate simulations stitched together. The operating characteristics of the combination are not the product of the operating characteristics of the parts.

A practical rule is to combine structures only when each component is necessary for the question. A platform is necessary when several treatments must be tested against one control. A cluster is necessary when the intervention is delivered at the group level. An adaptive enrichment is necessary when the population is heterogeneous and the effect is expected in a subgroup. If a component can be removed without changing the question, it should be removed.

### Edge cases

**Non-inferiority margins.** A non-inferiority trial needs an active control, a margin justified from historical evidence, and assay sensitivity. The margin is not a design convenience; it is a claim about the largest acceptable loss. A trial with a margin larger than the control's historical effect cannot distinguish a non-inferior product from an ineffective one.

**Multi-arm multiplicity.** A trial with several experimental arms and a shared control has correlated comparisons and an inflated family-wise error rate. The design must prespecify the testing strategy: a fixed sequence, a gatekeeping procedure, or a multiplicity adjustment. The sample size must reflect the strategy. A trial powered for one comparison and analysed for five is not powered for any of them.

**Cluster size variation.** The design effect 1 + (m - 1) * ICC assumes equal cluster sizes. When cluster sizes vary, the effective sample size is lower. A common adjustment uses the coefficient of variation of cluster size: the design effect becomes 1 + ((CV^2 + 1) * m - 1) * ICC. A CV above 0.23 can materially change the sample size.

**Adaptive interim timing.** The information fraction at the interim determines the value of the adaptation. An interim at 25 percent of the events has unstable estimates and a high probability of stopping on noise. An interim at 75 percent has little opportunity to save resources. The protocol should justify the timing and report the operating characteristics at each planned look.

**External control drift.** An external control is only valid if the population, era, and care pathway are comparable. Drift in standard of care, diagnostic criteria, or supportive care can produce a biased comparison that no adjustment can fix. The design should include a concurrent control where possible, and a sensitivity analysis that quantifies the effect of plausible drift.

**Missing endpoint data.** Missing data can bias the estimate if the missingness is related to the outcome. The design should minimize missingness through follow-up procedures, and the analysis should prespecify the primary handling method and sensitivity analyses. A trial with 30 percent missing primary outcomes is unlikely to support a confirmatory claim, regardless of the analysis method.

The design determines how much missingness is tolerable. A trial with a short follow-up, a simple endpoint, and a stable population can tolerate more missingness than a trial with a long follow-up, a composite endpoint, and a sick population. The protocol should state the expected missingness, the reason for the expectation, and the maximum acceptable missingness before the trial is considered uninformative.

### Choosing between a fixed and an adaptive design

The choice is not about sophistication. It is about whether the adaptation changes a decision that the trial would otherwise make too late or at too high a cost.

| Condition | Fixed design | Adaptive design |
|---|---|---|
| Effect is well established and the main risk is precision | Preferred | Adds complexity without much benefit |
| High uncertainty about the event rate | Acceptable | Blinded sample-size review can help |
| Intervention may be ineffective and exposure is burdensome | Acceptable | Futility stopping is ethical and efficient |
| Several doses or arms with uncertain benefit | Inefficient | Arm dropping or seamless design is efficient |
| Regulatory pathway requires a single confirmatory trial | Preferred | Acceptable if operating characteristics are documented |
| Infrastructure cannot support rapid interim data cleaning | Necessary | Not feasible |
| Decision timeline is short | Preferred | Interim can shorten the timeline |

The table is a heuristic. The real test is whether the adaptive design's expected decision value, including the cost of complexity and the risk of bias, exceeds the fixed design's. A trial that saves 10 percent of its sample but adds six months of simulation and review may not be worth adapting.

### Decision table: design selection under constraints

| Constraint | Preferred design | Acceptable alternative | Usually unacceptable |
|---|---|---|---|
| Individual delivery, no contamination | Parallel group | Factorial if two questions | Cluster (inefficient) |
| Ward-level delivery | Cluster randomised | Stepped-wedge if roll-out fixed | Individually randomised (contamination) |
| Stable reversible condition | Crossover | N-of-1 series | Parallel (wasteful) |
| Rare disease, no concurrent control feasible | Single-arm with external control | Bayesian borrowing with pre-specification | Uncontrolled case series with a historical benchmark |
| Many treatments, one disease | Platform | Umbrella | Separate small trials |
| One treatment, several diseases | Basket | Platform with disease-specific arms | Pooled single-arm series |
| Urgent public health emergency | Platform with pragmatic endpoints | Large simple trial | Small explanatory trial |
| Device with learning curve | Randomised with a run-in | Stepped-wedge | Single-arm before-after |

The "usually unacceptable" column is not absolute. There are contexts in which an apparently weak design is the only feasible option. In those contexts, the design must be accompanied by a stronger analysis plan and a more cautious claim.

### Signals that the design needs specialist review

Escalate the design to a statistician and, where relevant, to the regulator before committing when any of the following is true:

- The unit of allocation is not the patient.
- The trial has more than two arms or more than one primary endpoint.
- The protocol includes an interim analysis or an adaptation.
- The design uses an external control, historical data, or Bayesian borrowing.
- The primary endpoint is a composite, a surrogate, or a digital measure.
- The expected effect is small relative to the variance.
- The population is rare or difficult to recruit.
- The intervention is delivered by a clinician or a care team.
- The trial uses decentralised elements for the primary endpoint.
- The analysis population is not the full randomised set.
- The comparator is expected to change during the trial.
- The trial is intended to support a label change or a coverage decision.

The list is not exhaustive, but it covers the situations in which an apparently reasonable design most often fails review.

## 6. Worked example

### The setting

A mid-sized biotech is developing an oral JAK inhibitor for moderate ulcerative colitis. The Phase 2 dose-ranging study suggested a clinical remission rate of about 42 percent at week 12 for the selected dose, against 30 percent for placebo. The programme team wants to move directly to a single confirmatory trial rather than run separate Phase 2b and Phase 3 studies. The board has approved a budget for a seamless Phase 2/3 design with an interim analysis, on the condition that the operating characteristics are documented.

### Step 1: Estimand and decision

The estimand is:

> In adults with moderately to severely active ulcerative colitis who have failed at least one conventional therapy, the difference in the proportion achieving clinical remission at week 12 between the selected dose of the investigational drug and placebo, regardless of whether participants discontinued study treatment or used rescue therapy.

The intercurrent-event strategy is treatment policy for discontinuation and rescue therapy. The summary measure is the risk difference, with the risk ratio as a secondary summary.

The decision is a go/no-go decision for Phase 3 investment. The company will proceed if the trial demonstrates a risk difference of at least 8 percentage points with a two-sided 95 percent confidence interval excluding zero, and if the safety profile is acceptable. The 8-point threshold is the minimum commercially meaningful effect, not the target effect.

### Step 2: Unit of allocation

The intervention is an oral drug delivered to individual patients. There is no meaningful risk of contamination between participants, because the drug is not available outside the trial and the care pathway is not altered by assignment. The unit of allocation is the patient.

### Step 3: Comparison structure

A parallel-group, double-blind, placebo-controlled design is selected. Alternatives considered:

- **Crossover.** Rejected because ulcerative colitis activity fluctuates and the drug has a biological effect that may persist beyond a washout; a crossover would require a long washout and would not estimate the week-12 remission effect cleanly.
- **Factorial with a second agent.** Rejected because there is no second agent ready and the interaction would complicate the claim.
- **Platform.** Rejected because the company has one asset and no infrastructure for adding arms.
- **Single-arm external control.** Rejected because placebo response in ulcerative colitis is variable and an external benchmark would not be credible.

### Step 4: Control

Placebo is the control. The standard of care includes mesalamine and, in some participants, corticosteroids; background therapy is allowed and must be stable at randomisation. Withholding the investigational drug is acceptable because participants continue standard care, and the trial duration is 12 weeks.

### Step 5: Allocation and stratification

Allocation is 1:1, stratified by prior biologic exposure (yes or no) and by baseline disease severity (moderate versus severe). The randomisation is centrally administered with concealed allocation. A 2:1 allocation was considered and rejected because it would reduce power by about 11 percent for the same total sample and the drug is not yet proven.

### Step 6: Blinding

The trial is double-blind: participants, investigators, site staff, and the sponsor's clinical team are blinded. The drug and placebo are identical in appearance and packaging. An independent unblinded statistician prepares the interim analysis. The DMC is the only body with access to unblinded comparative results.

### Step 7: Follow-up and analysis population

The primary endpoint is clinical remission at week 12, defined by a total Mayo score of 2 or less with no subscore greater than 1. Follow-up visits are at weeks 2, 4, 8, and 12, with a window of plus or minus 5 days. The primary analysis population is ITT: all randomised participants, with participants who discontinued treatment or used rescue therapy retained in their assigned group. Missing remission status is handled by multiple imputation under a missing-at-random assumption, with a non-responder imputation sensitivity analysis.

### Step 8: Adaptation

The design has one interim analysis at 50 percent of the planned evaluable participants. The interim has two components:

1. **Blinded sample-size review.** The pooled remission rate is used to re-estimate the sample size, with no unblinding of the treatment effect. The re-estimation is capped at 1.3 times the original size.
2. **Unblinded futility review.** The DMC reviews the observed risk difference. If the difference is below 4 percentage points, the DMC may recommend stopping for futility. There is no early efficacy stopping, because the company wants a full data set for the Phase 3 decision and because an early efficacy claim would require a more complex alpha-spending plan.

The futility rule is binding only in the sense that the DMC must consider it; the final recommendation is the DMC's. The protocol states the boundary and the rationale.

### Step 9: Clustering

Not applicable. The unit of allocation is the patient.

### Step 10: Decentralisation

The trial uses a hybrid model: site visits at screening, week 4, and week 12; remote visits at weeks 2 and 8; and electronic patient-reported outcomes weekly. The primary endpoint is assessed at the site to avoid measurement variability. Remote visits are used for safety and adherence, not for the primary endpoint. The protocol states that site and remote data are both source documents and that the site retains responsibility for the endpoint assessment.

### Step 11: Sample size arithmetic

Assumptions:

- Placebo remission rate: 30 percent.
- Target remission rate: 42 percent.
- Absolute difference: 12 percentage points.
- Relative difference: 40 percent.
- Alpha: 0.05, two-sided.
- Power: 90 percent.
- Allocation: 1:1.
- Dropout or missing primary endpoint: 12 percent.

The per-arm evaluable sample size is:

```text
n = (z_alpha/2 + z_beta)^2 * [p0(1-p0) + p1(1-p1)] / (p1 - p0)^2
  = (1.959964 + 1.281552)^2 * [0.30*0.70 + 0.42*0.58] / (0.12)^2
  = 10.507 * [0.21 + 0.2436] / 0.0144
  = 10.507 * 0.4536 / 0.0144
  = 331.0
```

Rounding up gives 331 evaluable participants per arm, or 662 evaluable in total. Inflating for 12 percent missing gives 331 / 0.88 = 376.1, so 377 randomised per arm and 754 in total.

The interim at 50 percent of evaluable participants occurs at 166 evaluable per arm, or 189 randomised per arm (378 total randomised at the interim, allowing for the same dropout rate). The final analysis includes the interim participants plus the additional participants.

### Step 12: Operating characteristics

The design was simulated with 30,000 replicates under the null and alternative hypotheses. The futility rule was an observed risk difference below 4 percentage points at the interim. The final test was a two-sided z-test at alpha 0.05 on the full evaluable sample.

| Scenario | True placebo rate | True drug rate | Type I error or power | Probability of futility stop | Expected randomised N |
|---|---:|---:|---:|---:|---:|
| Null | 30% | 30% | 0.023 | 0.786 | 403 |
| Target | 30% | 42% | 0.870 | 0.062 | 644 |
| Smaller effect | 30% | 38% | 0.552 | 0.215 | 593 |
| Aggressive futility | 30% | 42% | 0.834 | 0.117 | 625 |
| Lenient futility | 30% | 42% | 0.886 | 0.031 | 654 |

The nominal power was 90 percent. The realised power is 87.0 percent because the futility rule stops 6.2 percent of trials that would have been successful. The expected sample size under the target effect is 644, compared with 754 for the fixed design, a saving of about 15 percent. Under the null, the expected sample size is 403, a saving of about 47 percent.

The type I error is 2.3 percent, below the nominal 5 percent. This is a consequence of futility stopping: trials with no effect are often stopped early, which reduces the chance of a false positive at the final analysis. The design is therefore conservative on type I error and slightly underpowered relative to the nominal calculation. Both are acceptable if stated in the protocol.

### Step 13: Implementation and fidelity

The drug is delivered orally, so fidelity is primarily adherence. Adherence is measured by pill count and by a participant diary. Sites are trained on the remission definition and on the handling of rescue therapy. The protocol defines a major deviation as any participant who receives prohibited concomitant therapy for more than 14 days; these participants remain in the ITT population but are flagged for the per-protocol sensitivity analysis.

### Step 14: Governance

An independent DMC reviews the interim data. The SAP is finalized before the interim analysis and specifies the primary estimand, the imputation model, the futility boundary, and the sensitivity analyses. The trial is registered before first enrolment, with the primary endpoint and the analysis plan. The company commits to publish the results regardless of direction.

### Two diagnostics that reveal a flawed design

**Diagnostic 1: the placebo-rate check.** The sample size assumes a 30 percent placebo remission rate. If the true rate is 24 percent, the absolute difference of 12 points corresponds to a drug rate of 36 percent, and the per-arm sample size becomes:

```text
n = 10.507 * [0.24*0.76 + 0.36*0.64] / (0.12)^2
  = 10.507 * [0.1824 + 0.2304] / 0.0144
  = 10.507 * 0.4128 / 0.0144
  = 301.2
```

The sample size falls to 302 per arm, which is smaller, but the power for the same absolute difference is actually higher because the variance is lower. The diagnostic is not the sample size; it is whether the assumed placebo rate is credible. A blinded sample-size review at the interim can detect a mismatch in the pooled rate, but it cannot tell whether the mismatch is in the placebo arm or the drug arm. The design should include a plan for acting on the review.

**Diagnostic 2: the futility-boundary calibration.** The futility boundary of 4 percentage points was chosen to avoid stopping a trial with a real but modest effect. The simulation shows that it stops 6.2 percent of trials with the target effect and 21.3 percent of trials with a smaller effect. If the company's decision rule requires an 8-point difference, a trial with a true 8-point effect has a substantial chance of being stopped for futility even though it could have met the decision rule. The boundary should be calibrated to the decision rule, not to the target effect.

### Vary one assumption: the effect size

If the true drug remission rate is 38 percent rather than 42 percent, the absolute difference is 8 points, which is exactly the decision threshold. The fixed-design power at 331 evaluable per arm is about 59 percent, and the adaptive design's power is 55 percent after futility. The trial would be underpowered for the decision it is meant to inform. The programme team must either increase the sample size, accept a higher risk of a false negative, or redefine the decision rule.

### Vary one assumption: the dropout rate

If dropout is 20 percent rather than 12 percent, the randomised sample size becomes 331 / 0.80 = 414 per arm, or 828 in total. The evaluable sample is unchanged, so the power is unchanged if the dropout is non-informative. If the dropout is informative, the ITT estimate may be biased toward the null, and the non-responder imputation sensitivity analysis becomes decisive. The design should budget for retention, not only for enrolment.

### Vary one assumption: the interim timing

If the interim occurs at 30 percent of the evaluable sample rather than 50 percent, the futility estimate is based on 100 evaluable per arm. The simulation shows a higher probability of stopping a truly effective trial (because the estimate is noisier) and a smaller expected sample-size saving. If the interim occurs at 70 percent, the saving is smaller but the futility decision is more reliable. The choice is a trade-off between efficiency and decision quality, and it should be stated in the protocol.

### The decision

The company accepts the design with the 50 percent interim, the 4-point futility boundary, and the 8-point decision rule. The trial is registered, the SAP is locked, and the DMC is constituted. The board's condition is met: the operating characteristics are documented, and the design has a prespecified rule for stopping.

### Fixed versus adaptive at a glance

| Property | Fixed design | Adaptive design in this example |
|---|---|---|
| Evaluable sample per arm | 331 | 331 maximum, 166 at interim |
| Nominal power | 90% | 90% before futility; 87.2% realised |
| Type I error | 5% | 2.3% realised |
| Expected sample under H1 | 662 | 644 |
| Expected sample under H0 | 662 | 403 |
| Governance | Standard DMC | DMC plus unblinded statistician plus simulation report |
| Regulatory interaction | Standard | Recommended before commitment |
| Complexity risk | Low | Moderate; depends on data cleaning speed |

The table makes the trade-off concrete. The adaptive design saves about 20 participants under the target effect and about 259 under the null. The saving under the null is the ethical and financial justification. The cost is a more complex design, a small power loss, and a dependence on timely interim data.

### What the interim does not tell you

The interim analysis estimates the effect at 50 percent of the information. That estimate is noisy. A trial that looks promising at the interim can fail at the final analysis, and a trial that looks modest at the interim can succeed. The futility rule is a decision aid, not a prediction. The protocol should state that the interim result will not be used to revise the target effect or the decision rule.

This is a discipline issue as much as a statistical one. Programme teams often want to reinterpret the interim result, especially when it is close to the boundary. The SAP should lock the decision rule and the DMC charter should state that the rule will be applied as written. A decision rule that is renegotiated at the interim is not a decision rule.

### What the example shows

The example shows that an adaptive design is not automatically better than a fixed design. It is better only when the adaptation is calibrated to the decision and the operating characteristics are understood. The 15 percent expected sample-size saving is real, but it comes with a 2.8 percentage point loss of power and a more complex governance structure. Those trade-offs belong in the protocol, not in a footnote.

## 7. Code, query, or calculation

The first block implements the sample-size arithmetic for a parallel trial with unequal allocation and the design-effect calculation for a cluster trial. The second block simulates the two-stage adaptive design used in the worked example. Both use only the Python standard library.

```python
import math


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's approximation)."""
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285104469687e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def n_parallel(p0, p1, alpha=0.05, power=0.90, ratio=1.0):
    """Evaluable sample size for arm 1; arm 2 is ratio times larger."""
    za = norm_ppf(1 - alpha / 2)
    zb = norm_ppf(power)
    return (za + zb) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1) / ratio) / (p1 - p0) ** 2


def cluster_design_effect(m, icc, cv=0.0):
    """Design effect with optional coefficient of variation of cluster size."""
    return 1 + ((1 + cv ** 2) * m - 1) * icc


if __name__ == "__main__":
    print("parallel, 1:1, 30% vs 42%, 90% power:")
    n = n_parallel(0.30, 0.42, power=0.90)
    print(f"  evaluable per arm = {math.ceil(n)}")
    print(f"  randomised per arm at 12% dropout = {math.ceil(math.ceil(n)/0.88)}")
    n21 = n_parallel(0.30, 0.42, power=0.90, ratio=2.0)
    print(f"  2:1 allocation, arm 1 = {math.ceil(n21)}, arm 2 = {math.ceil(2*n21)}")
    print("cluster design effects, m=250:")
    for icc in (0.01, 0.03, 0.05):
        print(f"  ICC {icc}: DE={cluster_design_effect(250, icc):.2f}")
    print("cluster size variation, ICC=0.03, m=250:")
    for cv in (0.0, 0.25, 0.50):
        print(f"  CV {cv}: DE={cluster_design_effect(250, 0.03, cv):.2f}")
```

Expected output:

```text
parallel, 1:1, 30% vs 42%, 90% power:
  evaluable per arm = 331
  randomised per arm at 12% dropout = 377
  2:1 allocation, arm 1 = 243, arm 2 = 485
cluster design effects, m=250:
  ICC 0.01: DE=3.49
  ICC 0.03: DE=8.47
  ICC 0.05: DE=13.45
cluster size variation, ICC=0.03, m=250:
  CV 0.0: DE=8.47
  CV 0.25: DE=8.94
  CV 0.5: DE=10.34
```

The 2:1 result deserves a comment. With 2:1 allocation, the smaller arm is 243 evaluable and the larger arm is 485, for a total of 728, compared with 662 for equal allocation. The unequal design costs about 10 percent more participants for the same power. The benefit is that more participants receive the investigational drug, which may be an ethical or recruitment advantage. That trade-off should be explicit.

The cluster results show why the ICC must be estimated, not assumed. At an ICC of 0.05, the design effect is 13.45; a trial that would need 331 participants per arm under individual randomisation would need about 4,452 per arm under clustering with 250 per cluster, or 18 clusters per arm. If only 12 clusters per arm are available, the trial cannot achieve the target power by increasing the cluster size alone.

The second block simulates the adaptive design.

```python
import math
import random


def simulate_two_stage(p0, p1, n_stage1, n_final, futility_diff,
                       reps=30000, seed=2026):
    """Two-stage parallel design with futility stopping only."""
    random.seed(seed)
    rejected = 0
    stopped = 0
    total_n = 0
    for _ in range(reps):
        x0 = sum(random.random() < p0 for _ in range(n_stage1))
        x1 = sum(random.random() < p1 for _ in range(n_stage1))
        if (x1 - x0) / n_stage1 < futility_diff:
            stopped += 1
            total_n += 2 * n_stage1
            continue
        x0 += sum(random.random() < p0 for _ in range(n_final - n_stage1))
        x1 += sum(random.random() < p1 for _ in range(n_final - n_stage1))
        pH0, pH1 = x0 / n_final, x1 / n_final
        se = math.sqrt(pH0 * (1 - pH0) / n_final + pH1 * (1 - pH1) / n_final)
        if se > 0 and (pH1 - pH0) / se > 1.959964:
            rejected += 1
        total_n += 2 * n_final
    return {
        "rejection_rate": round(rejected / reps, 4),
        "futility_rate": round(stopped / reps, 4),
        "expected_n": round(total_n / reps, 1),
    }


if __name__ == "__main__":
    scenarios = {
        "null 30 vs 30": (0.30, 0.30),
        "target 30 vs 42": (0.30, 0.42),
        "small 30 vs 38": (0.30, 0.38),
    }
    for name, (p0, p1) in scenarios.items():
        result = simulate_two_stage(p0, p1, 166, 332, 0.04)
        print(name, result)
    print("aggressive futility 30 vs 42",
          simulate_two_stage(0.30, 0.42, 166, 332, 0.06))
    print("lenient futility 30 vs 42",
          simulate_two_stage(0.30, 0.42, 166, 332, 0.02))
```

Expected output (Monte Carlo values may differ slightly with a different seed):

```text
null 30 vs 30 {'rejection_rate': 0.023, 'futility_rate': 0.7855, 'expected_n': 403.2}
target 30 vs 42 {'rejection_rate': 0.8696, 'futility_rate': 0.0615, 'expected_n': 643.6}
small 30 vs 38 {'rejection_rate': 0.552, 'futility_rate': 0.2153, 'expected_n': 592.5}
aggressive futility 30 vs 42 {'rejection_rate': 0.8338, 'futility_rate': 0.1168, 'expected_n': 625.2}
lenient futility 30 vs 42 {'rejection_rate': 0.8856, 'futility_rate': 0.0311, 'expected_n': 653.7}
```

The simulation is deliberately simple. It assumes no dropout, no stratification, and a immediate endpoint. A real simulation report would include those features, plus scenarios for deviations from the assumptions. The point of the block is to show that the operating characteristics of an adaptive design cannot be read off a formula: they depend on the joint behaviour of the interim rule and the final test.

The third block computes the sample size for a crossover trial and a cluster trial from the same effect, to show the efficiency comparison.

```python
import math


def n_crossover(sd_within, delta, alpha=0.05, power=0.90):
    """Pairs per sequence for a two-period crossover, paired analysis."""
    za = 1.959964
    zb = 1.281552
    return (za + zb) ** 2 * (sd_within ** 2) / (delta ** 2)


def n_cluster(n_individual_per_arm, m, icc, cv=0.0):
    """Individuals per arm and clusters per arm for a cluster trial."""
    de = 1 + ((1 + cv ** 2) * m - 1) * icc
    n_cluster_total = n_individual_per_arm * de
    return math.ceil(n_cluster_total), math.ceil(n_cluster_total / m)


if __name__ == "__main__":
    print("crossover, within-person SD 1.0, delta 0.5:",
          math.ceil(n_crossover(1.0, 0.5)), "pairs per sequence")
    print("cluster, 332 per arm, m=250, ICC=0.03:",
          n_cluster(332, 250, 0.03))
    print("cluster, 332 per arm, m=250, ICC=0.03, CV=0.5:",
          n_cluster(332, 250, 0.03, 0.5))
```

Expected output:

```text
crossover, within-person SD 1.0, delta 0.5: 43 pairs per sequence
cluster, 332 per arm, m=250, ICC=0.03: (2813, 12)
cluster, 332 per arm, m=250, ICC=0.03, CV=0.5: (3435, 14)
```

The crossover calculation uses the standard deviation of the within-person difference, not the between-person standard deviation. If the within-person correlation is 0.5 and the between-person standard deviation is 1.0, the within-person difference standard deviation is the square root of 2 * 1.0^2 * (1 - 0.5) = 1.0. The result, 43 pairs per sequence, is far smaller than the parallel design's 331 per arm. That efficiency is the reason crossover designs are attractive for stable, reversible conditions. It is also why they are dangerous when the no-carryover assumption is false.

## 8. Data, standards, and regulatory context

### Binding requirements

**ICH E6(R3).** The Good Clinical Practice guideline adopted at Step 4 on 6 January 2025 sets the requirements for trial conduct, data governance, and protocol content. Its Annex 1 covers investigator and sponsor responsibilities, and its protocol appendix specifies the content that a protocol must contain. In the EU, the EMA Step 5 version took effect on 23 July 2025. In the US, the FDA issued the guideline in September 2025. E6(R3) is binding to the extent a jurisdiction adopts it. It does not prescribe a design, but it does require that the design be described and justified.

**ICH E9.** The Statistical Principles for Clinical Trials (E9, 1998) remains the foundational statistical guideline. It covers superiority, equivalence, and non-inferiority trials, and it requires prespecification of the primary analysis. It is guidance, not law, but it is the reference standard for statistical review.

**ICH E10.** The Choice of Control Group guideline (E10, 2000) sets out the five control types: placebo concurrent, no-treatment concurrent, dose-response concurrent, active concurrent, and external. It introduces assay sensitivity and the conditions under which an external control is acceptable. It is guidance, but a trial that ignores it will struggle in review.

**EU Clinical Trials Regulation.** Regulation (EU) No 536/2014 governs the conduct of clinical trials in the EU and requires a single application through the Clinical Trials Information System. It is binding.

**US device regulations.** For device trials, the Investigational Device Exemption (IDE) regulation at 21 CFR Part 812 governs the conduct of studies, and the premarket approval regulation at 21 CFR Part 814 governs the evidence standard. These are binding.

### Formal guidance

**FDA adaptive designs.** The final guidance "Adaptive Designs for Clinical Trials of Drugs and Biologics" (December 2019) remains the current statement. It requires prespecification of the adaptation rule, a data access plan, simulation reports, and appropriate interaction with the FDA. It covers Bayesian adaptive designs and complex trials that rely on computer simulation. The guidance was not revised in 2025 in the sources checked; teams should verify the current version before relying on the date.

**FDA master protocols.** The FDA has issued guidance on master protocols for drug and biological product development. A draft was issued in December 2023, and the FDA's guidance page has listed later versions. The guidance addresses platform, basket, and umbrella trials, and it emphasizes prespecification, governance, and multiplicity control. Because the document has been in flux, the protocol should cite the version it relies on and verify the current status at submission.

**FDA decentralised trials.** The final guidance "Conducting Clinical Trials With Decentralized Elements" was issued in September 2024 (Federal Register notice 18 September 2024, docket FDA-2022-D-2870). It states that the regulatory requirements are the same as for traditional trials, and it covers remote visits, local health care providers, digital health technologies, informed consent, and investigational product shipping. It is guidance, not regulation, but it defines the FDA's expectations for decentralised elements.

**FDA externally controlled trials.** The draft guidance "Considerations for the Design and Conduct of Externally Controlled Trials for Drug and Biological Products" (February 2023) applies to drugs and biologics, not devices. It covers historical and concurrent external controls, confounding, and sensitivity analyses. It remains a draft.

**FDA real-world evidence for devices.** The final guidance "Use of Real-World Evidence to Support Regulatory Decision-Making for Medical Devices" (December 2025, docket FDA-2023-D-4395) supersedes the 2017 version. It describes relevance and reliability assessments for real-world data and how they fit into benefit-risk review. It is relevant to device trials that use registry or EHR data as part of the evidence package.

**ICH E9(R1).** The estimand addendum (Step 4, 20 November 2019; EMA Step 5 effective 30 July 2020; FDA guidance May 2021) applies to all confirmatory trials. It requires the estimand to be defined before the design and analysis. For adaptive trials, the estimand must remain the same across adaptations; an adaptation that changes the estimand changes the question.

### Reporting conventions

**CONSORT 2025.** The updated statement (BMJ 2025;389:e081123) supersedes CONSORT 2010. It has a 30-item checklist with a new open-science section, and it integrates the Harms, Outcomes, Non-pharmacological Treatment, and TIDieR extensions. It is a reporting convention, enforced by journals, not a design requirement.

**CONSORT extensions.** The cluster randomised trial extension (Campbell, Elbourne, Altman, BMJ 2004) requires reporting of the rationale for clustering, the ICC, and the flow of clusters and individuals. The stepped-wedge extension and the pilot and feasibility extensions impose analogous requirements. These are conventions, but they are the minimum for a readable trial report.

**SPIRIT 2025.** The protocol guideline (BMJ 2025;389:e081477) has a 34-item checklist, including items on patient and public involvement, trial monitoring, and open science. It supersedes SPIRIT 2013.

### What the standards do not settle

| Question | Standard | What it leaves open |
|---|---|---|
| Which design? | ICH E9, E10 | The choice among feasible designs |
| How many participants? | ICH E9 | The assumed effect, variance, and dropout |
| How to handle intercurrent events? | ICH E9(R1) | The strategy that matches the claim |
| When to stop for futility? | FDA adaptive guidance | The boundary and its calibration |
| How to analyse a cluster trial? | CONSORT extension | The ICC and the analysis model |
| How to use a decentralised element? | FDA DCT guidance | The endpoint and the validation method |
| How to report? | CONSORT 2025 | The interpretation and the claim |

The pattern is consistent: the standards constrain the process and the reporting, not the scientific judgement. The judgement must be justified in the protocol.

A final note on version control. Several of the instruments cited in this section changed in 2024 to 2026. CONSORT and SPIRIT were updated in 2025, ICH E6(R3) was adopted in January 2025, and the FDA device real-world evidence guidance was reissued in December 2025. A protocol that cites an older version may still be acceptable if the version was current at the time of design, but the submission should state which version it followed and why. A protocol that cites a document without a version or date is not checkable.

## 9. Industry practice and case evidence

### RECOVERY: a platform that changed practice

The RECOVERY trial was a large, pragmatic, randomised platform in the United Kingdom. It used a simple design, broad eligibility, routine data collection, and multiple treatment arms. The dexamethasone comparison randomised 2,104 participants to dexamethasone and 4,321 to usual care. Overall 28-day mortality was 22.9 percent versus 25.7 percent (age-adjusted rate ratio 0.83, 95 percent confidence interval 0.75 to 0.93). Among participants receiving invasive mechanical ventilation, mortality was 29.3 percent versus 41.4 percent (rate ratio 0.64, 95 percent confidence interval 0.51 to 0.81). Among those receiving no respiratory support, there was no benefit (17.8 percent versus 14.0 percent; rate ratio 1.19, 95 percent confidence interval 0.92 to 1.55).

The documented facts are the design features and the subgroup results. The inference is that the platform structure allowed multiple questions to be answered with shared infrastructure and a common control, and that the broad population made the heterogeneity visible. The design did not require a new protocol for each arm; it required a master protocol, a governance structure, and a commitment to publish.

**Lesson.** A platform is a governance model as much as a statistical design. The statistical efficiency comes from the shared control and the common infrastructure; the credibility comes from the prespecified comparisons and the transparent reporting.

### REMAP-CAP: a Bayesian platform in critical care

REMAP-CAP is a randomised, adaptive platform trial in critically ill patients with community-acquired pneumonia and other severe infections. It uses Bayesian analysis, multiple domains, and adaptive randomisation. The platform has reported results for several interventions, including corticosteroids and IL-6 receptor antagonists, and it has been used to inform guidelines during the COVID-19 pandemic.

The documented facts are the platform structure, the Bayesian framework, and the published domain results. The inference is that the platform's efficiency depends on a stable infrastructure, a common endpoint, and a statistical model that can handle multiple domains and non-concurrent data. The Bayesian approach allows borrowing across domains, but it also requires a prior specification that reviewers will scrutinise.

**Lesson.** A platform's statistical model is part of its design. A Bayesian platform must prespecify the priors, the borrowing rules, and the decision thresholds. A platform that changes its model after seeing data is not adaptive; it is unplanned.

### Apple Heart Study: a decentralised, siteless design

The Apple Heart Study enrolled 419,297 participants over eight months in a siteless, app-based study. Over a median 117 days, 2,161 participants (0.52 percent) received an irregular pulse notification. Among 450 notified participants who returned analyzable ECG patches, atrial fibrillation was present in 34 percent. The positive predictive value for notification concordance with simultaneous patch-detected atrial fibrillation was 0.84.

The documented facts are the enrolment scale, the notification rate, and the confirmation rate. The inference is that the decentralised design was feasible at scale, but the lack of a randomised comparison and the incomplete patch return limited the claim to detection performance. The study did not measure stroke, mortality, or the downstream care pathway.

**Lesson.** Decentralisation changes the logistics, not the logic. A siteless design can enrol hundreds of thousands of participants, but it still needs a reference standard, a comparison, and a prespecified endpoint if the claim is comparative.

### IDx-DR: a narrow device claim

The FDA authorised IDx-DR through the De Novo pathway in 2018 (DEN180001) for autonomous detection of diabetic retinopathy in adults with diabetes. The authorisation was based on a prospective clinical study with a defined intended use, an image-acquisition protocol, and a reference standard.

The documented fact is the authorisation and the intended-use restriction. The inference is that the design was a single-arm or non-randomised diagnostic accuracy study, not a comparative outcome trial, because the claim was detection rather than impact. The narrow claim made the evidence package feasible.

**Lesson.** The design must match the claim class. A diagnostic claim can be supported by an accuracy study; an impact claim cannot.

### Cluster trials in practice

Cluster randomised trials are common in primary care, public health, and health services research. The CONSORT extension for cluster trials was published in 2004, and reviews have repeatedly found that a minority of cluster trials account for clustering in their sample-size calculations. The methodological literature (for example, Rutterford, Copas, and Eldridge, 2015) documents the consequences: underpowered trials, inflated type I error, and misleading conclusions.

The documented fact is the prevalence of the methodological problem. The inference is that cluster designs are often chosen for feasibility without a corresponding investment in the statistical design. The mitigation is to involve a statistician at the design stage, to estimate the ICC from prior data, and to report the ICC in the results.

### What companies actually do

The following practices are inferred from public protocols, guidance, and published trials.

1. **Design selection is often driven by feasibility.** The choice between parallel and cluster is frequently made by operations, not statistics, because the intervention cannot be delivered any other way. The statistical team is then asked to "adjust" the sample size.
2. **Adaptive designs are used selectively.** Seamless Phase 2/3 and futility designs are common in oncology and rare disease, where the cost of a full Phase 3 is high and the interim data are informative. They are less common in primary care, where the infrastructure is simpler and the effect sizes are smaller.
3. **Platforms require a consortium.** Most public platforms are funded by governments or charities, because the governance and infrastructure costs are difficult to justify for a single asset.
4. **Decentralised elements are adopted incrementally.** Companies often start with remote visits for safety follow-up and ePRO, then extend to remote consent and local labs. The primary endpoint usually remains site-assessed until the digital measure is validated.
5. **Regulator interaction is design-specific.** Sponsors use Type C meetings, special protocol assessments, and scientific advice to test adaptive and platform designs. The interaction is not a guarantee, but it reduces the risk of a design that the regulator will reject.

### Reading the case evidence critically

The four cases in this section are often cited as evidence that a particular design is superior. That reading is too simple. Each case is a demonstration of a design working under specific conditions.

**RECOVERY** worked because the disease was common, the endpoint was short-term mortality, the intervention was inexpensive, and the health system had a unified data infrastructure. A platform for a rare disease with a complex endpoint would not have achieved the same scale. The design was necessary but not sufficient; the context made it powerful.

**REMAP-CAP** worked because the critical care community had a research network, a common data platform, and a willingness to accept Bayesian methods. The platform's efficiency came from the network, not only from the statistics. A company trying to replicate the design without the network would find the governance costs prohibitive.

**The Apple Heart Study** worked because the product was already in consumers' hands and the study could recruit at near-zero marginal cost per participant. The design's limitation was not the decentralisation; it was the absence of a randomised comparison and the incomplete reference standard. A decentralised trial with a randomised design and a complete endpoint would have been more informative and much more expensive.

**IDx-DR** worked because the claim was narrow and the reference standard was clear. The design was feasible because the device output could be compared with a graded reference standard in a prospective study. A broader claim about visual outcomes would have required a different design and a much larger sample.

The inference is not that these designs are universally superior. It is that design and context must match. A team should ask which features of the context made the design work, and whether those features are present in its own setting.

### The evidence on design quality

The methodological literature consistently finds gaps between recommended and actual practice. Reviews of cluster trials have found that a minority account for clustering in sample-size calculations. Reviews of crossover trials have found that carryover and period effects are often not tested. Reviews of adaptive trials have found that simulation reports are sometimes incomplete.

These findings are documented in the methodological literature, not in a single systematic review cited here. The practical implication is that a design review should not assume that a published trial used its design correctly. The protocol and the statistical analysis plan are the primary evidence of design quality, and they should be read before the results are accepted.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Crossover with carryover | No washout or a long-acting drug | Treatment effect persists into the next period | Biased within-person contrast | Use a parallel design or a longer washout |
| Cluster trial powered as individual | Sample size equals an individual trial | ICC ignored | Severe underpowering; inflated type I error | Design effect and ICC sensitivity |
| Too few clusters | Cluster count fixed by site availability | Cluster size cannot compensate for cluster count | Underpowered; unstable variance | Feasibility check; more clusters or a different design |
| Unadjusted shared control | Platform arms added over time | Non-concurrent controls differ from concurrent controls | Biased comparisons | Concurrent randomisation; bridge or model adjustment |
| Multiplicity uncontrolled | Several arms or endpoints, no testing strategy | Family-wise error inflation | False-positive claim | Prespecified hierarchy or alpha allocation |
| Adaptive rule not prespecified | "We will look at the interim and decide" | Post hoc adaptation | Type I error inflation; uninterpretable result | Protocol amendment before unblinding; simulation |
| Interim too early | Futility at 20% information | Unstable estimate | Stopping on noise | Information-based timing; calibration |
| External control drift | Historical control from a different era | Standard of care changed | Biased effect | Concurrent control; drift sensitivity analysis |
| Decentralised endpoint unvalidated | Home device used for primary endpoint | Measurement variability | Biased or noisy estimate | Validation substudy; site assessment for primary |
| Blinding broken | Side effects reveal assignment | Performance and detection bias | Inflated effect | Sham control; blinded assessors |
| ITT population undefined | "All evaluable patients" | Post hoc exclusions | Selection bias | Prespecify ITT and per-protocol |
| Over-stratification | Many strata with few participants | Sparse strata | Imbalanced arms; unstable estimates | Limit strata; use minimisation |
| Fidelity not measured | No delivery monitoring | Intervention varies by site | Ambiguous estimate | Fidelity plan and monitoring |
| No DMC or charter | Interim data seen by sponsor | Pressure to change design | Loss of credibility | Independent DMC with charter |

### Red flags in a design review

- The design is justified by "what the sites can do" without a statement of the estimand.
- The sample size is borrowed from a similar trial without checking the event rate.
- The protocol says "double-blind" but the placebo is not identical.
- The cluster trial reports an ICC of zero without a source.
- The platform has no rule for adding or dropping arms.
- The adaptive design has no simulation report.
- The crossover has no washout or a washout shorter than the drug's half-life.
- The single-arm study has a historical control but no drift analysis.
- The decentralised plan uses a home device for the primary endpoint without a validation study.
- The analysis population is defined after the results are known.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The estimand is stable across the trial | Adaptations and analyses must estimate the same quantity | A protocol change that redefines the endpoint | Estimand statement in the SAP; change control |
| The control is stable | A moving comparator invalidates the contrast | Guideline change, new product | Guideline surveillance; concurrent control |
| Participants are exchangeable across arms | Randomisation balances confounders | Differential recruitment or contamination | Baseline table; allocation concealment audit |
| The condition is stable for a crossover | Within-person contrast requires stable disease | Fluctuating disease, period effects | Run-in; period-effect test |
| No carryover | The second period must start clean | Long half-life, irreversible effect | Washout; carryover test |
| Clusters are exchangeable | Cluster randomisation requires comparable clusters | Clustering by geography or case mix | Baseline cluster characteristics; matching |
| The ICC is correctly specified | Sample size depends on it | ICC uncertainty, cluster size variation | Sensitivity analysis over a range |
| The event rate is correctly specified | Power depends on it | Population shift, improved care | Blinded sample-size review |
| Participants adhere sufficiently | ITT effect depends on delivery | Side effects, complex regimen | Pilot adherence; monitoring |
| Missing data are limited | Complete-case analysis is biased otherwise | Differential dropout | Missing-data sensitivity analyses |
| The digital endpoint measures the construct | Decentralised trials depend on it | Device error, user error | Validation substudy |
| The external control is comparable | Single-arm designs depend on it | Era drift, population differences | Overlap assessment; sensitivity analyses |
| The interim data are clean and timely | Adaptive decisions depend on them | Data entry lag, query backlog | Data cleaning plan; interim dry run |
| The comparator is delivered as described | The contrast depends on it | Training gaps, supply shortages | Fidelity monitoring; site audits |

### Limits of this topic

This topic covers design selection and the arithmetic of common structures. It does not cover the full theory of adaptive designs, the details of Bayesian borrowing, or the analysis of platform trials. Those are specialised topics with their own literature. It also does not cover the operational details of trial conduct, which are governed by ICH E6(R3) and site standard operating procedures.

The arithmetic in Sections 6 and 7 uses normal approximations. For small samples, extreme proportions, or survival outcomes with few events, exact or simulation-based methods are preferable. The worked example is designed to be reproducible by hand, not to be copied into a protocol without a statistician's review.

The case evidence is drawn from public sources and is not a systematic review. The inferences about company practice are labelled as inferences. The regulatory statements are educational summaries, not legal advice. A submission team should verify the current text and date of every guidance it relies on, particularly in areas where documents have changed in 2025 and 2026.

### A note on design and analysis

A design is only as good as the analysis that follows it. The protocol should specify the primary analysis, the analysis population, the handling of missing data, and the sensitivity analyses. The SAP should be written before unblinding and should be consistent with the protocol. A design that is sound but analysed with an unreported switch of populations is not a sound design in practice.

The opposite is also true. A sophisticated analysis cannot repair a design that cannot identify the estimand. The design and the analysis are a pair; the protocol should describe both, and the review should test both. A team that spends months on the design and days on the SAP has misallocated its effort.

## 12. Exercises

### Exercise 1 — Select a design (application, 35 minutes)

**Deliverable:** a design recommendation for three scenarios: (a) a ward-level infection-control bundle, (b) a short-acting inhaled bronchodilator for stable asthma, (c) a rare metabolic disease with no approved treatment. For each, name the unit of allocation, the structure, the control, and one rejected alternative.

**Acceptance criteria:** each recommendation follows from the delivery level and the condition; the rejected alternative has a specific reason; the control is ethically defensible.

**Time:** 35 minutes.

### Exercise 2 — Specify allocation and blinding (application, 25 minutes)

**Deliverable:** an allocation, stratification, and blinding plan for a double-blind trial of an injectable biologic with a visible injection-site reaction.

**Acceptance criteria:** allocation concealment described; stratification factors justified; blinding plan addresses the visible reaction; unblinding procedure stated.

**Time:** 25 minutes.

### Exercise 3 — Parallel and crossover arithmetic (calculation, 40 minutes)

**Deliverable:** sample sizes for (a) a parallel trial with 25 percent versus 35 percent event rates, 90 percent power, 10 percent dropout; (b) a crossover trial with a within-person SD of 1.5, a target difference of 0.6, 90 percent power. Show the formulas and the assumptions.

**Acceptance criteria:** formulas correct; numbers correct to the nearest participant; the crossover calculation uses the within-person SD; the assumptions are stated.

**Time:** 40 minutes.

### Exercise 4 — Cluster design (calculation, 40 minutes)

**Deliverable:** the number of clusters per arm for a trial with 300 individuals per arm, an average cluster size of 120, an ICC of 0.04, and a cluster-size CV of 0.4. Compare the result with the equal-cluster-size formula.

**Acceptance criteria:** design effect calculated with and without the CV; cluster counts rounded up; the difference explained; the feasibility of the cluster count discussed.

**Time:** 40 minutes.

### Exercise 5 — Adaptive interim (design, 45 minutes)

**Deliverable:** a two-stage futility design for a trial with a 20 percent control event rate and a target 30 percent event rate, including the interim timing, the futility boundary, and a simulation plan.

**Acceptance criteria:** interim timing justified by information fraction; futility boundary calibrated to the decision rule; simulation scenarios listed; alpha control explained.

**Time:** 45 minutes.

### Exercise 6 — Critique a platform (critique, 40 minutes)

**Deliverable:** a one-page critique of a published platform trial, identifying the shared-control strategy, the multiplicity approach, and one threat to the validity of a non-concurrent comparison.

**Acceptance criteria:** specific to the trial; distinguishes concurrent from non-concurrent comparisons; proposes a mitigation.

**Time:** 40 minutes.

### Exercise 7 — Decentralised trial plan (design, 35 minutes)

**Deliverable:** a decentralised-element plan for a trial of a digital hypertension intervention, covering consent, blood pressure measurement, data flow, and safety monitoring.

**Acceptance criteria:** each element has a validation and a regulatory basis; the primary endpoint measurement is justified; missing-data handling is stated.

**Time:** 35 minutes.

### Exercise 8 — Redesign a flawed trial (critique and design, 60 minutes)

**Deliverable:** a two-page redesign of a published trial whose design cannot support its claim, with the new design, the sample-size approach, and the cost implication.

**Acceptance criteria:** the flaw is diagnosed correctly; the new design addresses it; the sample-size approach is explicit; the cost implication is quantified approximately.

**Time:** 60 minutes.

## 13. Self-check answers

**Answer 1 (design selection).** (a) A ward-level infection-control bundle should use a cluster randomised design, because the intervention is delivered to the ward and contamination is likely. The control is usual infection-control practice. A parallel individual design is rejected because it cannot prevent contamination. (b) A short-acting inhaled bronchodilator for stable asthma can use a crossover design, because the condition is stable, the effect is reversible, and a washout is feasible. The control is placebo. A parallel design is acceptable but less efficient. (c) A rare metabolic disease with no approved treatment may use a single-arm design with an external control if randomisation is infeasible, but a randomised trial with a small sample or a Bayesian design with borrowing may be preferable. The control is the best available supportive care or a natural-history cohort.

**Answer 2 (allocation and blinding).** Allocation should be central and concealed, with stratification by baseline severity and prior biologic use. The visible injection-site reaction threatens blinding; the plan should use a sham injection or a masked assessor for the primary endpoint, and should state that participants and investigators may become unblinded but the endpoint adjudication remains blinded. The unblinding procedure should be available for safety emergencies and should record who unblinded whom.

**Answer 3 (parallel and crossover arithmetic).** (a) p0 = 0.25, p1 = 0.35. n = (1.959964 + 1.281552)^2 * [0.25*0.75 + 0.35*0.65] / (0.10)^2 = 10.507 * [0.1875 + 0.2275] / 0.01 = 10.507 * 0.415 / 0.01 = 436.0, so 437 per arm. With 10 percent dropout, 437 / 0.90 = 485.6, so 486 per arm and 972 total. (b) n = (1.959964 + 1.281552)^2 * (1.5^2) / (0.6^2) = 10.507 * 2.25 / 0.36 = 65.7, so 66 pairs per sequence, or 132 participants in a two-period crossover.

**Answer 4 (cluster).** Equal cluster sizes: DE = 1 + (120 - 1) * 0.04 = 1 + 4.76 = 5.76. Individuals per arm = 300 * 5.76 = 1,728; clusters per arm = 1,728 / 120 = 14.4, so 15 clusters per arm. With CV = 0.4: DE = 1 + ((1 + 0.16) * 120 - 1) * 0.04 = 1 + (139.2 - 1) * 0.04 = 1 + 5.528 = 6.528. Individuals per arm = 300 * 6.528 = 1,958.4; clusters per arm = 1,958.4 / 120 = 16.3, so 17 clusters per arm. The variation adds about 13 percent to the cluster count. If only 15 clusters per arm are available, the trial is underpowered.

**Answer 5 (adaptive interim).** A reasonable design has an interim at 50 percent of the planned evaluable sample, a futility boundary based on an observed difference below 5 percentage points, and a final two-sided test at alpha 0.05. The simulation should cover the null (20 versus 20), the target (20 versus 30), a smaller effect (20 versus 27), and a higher control rate (25 versus 35). The simulation should report type I error, power, the probability of futility stopping, and the expected sample size. Alpha is controlled because there is no early efficacy stopping; futility stopping can only reduce the type I error.

**Answer 6 (platform critique rubric).** A strong critique identifies the shared control, states whether the comparisons are concurrent or non-concurrent, describes the multiplicity strategy, and names a specific threat such as changes in standard of care over time, differential endpoint ascertainment, or arm-entry criteria that differ across periods. It proposes a mitigation such as concurrent randomisation within a period, a bridge study, or a sensitivity analysis. A weak critique summarises the platform's size and calls it efficient.

**Answer 7 (decentralised plan rubric).** A strong plan specifies remote consent with identity verification, a validated home blood pressure device with a calibration schedule, a data flow with source documentation, and a safety monitoring plan with local escalation. It states that the primary endpoint is measured with the validated device and that site measurement is used for a validation substudy. It also states how missing data will be handled. A weak plan lists remote visits without addressing measurement validity.

**Answer 8 (redesign rubric).** A strong redesign names the original claim, diagnoses the design flaw (for example, a single-arm study supporting a comparative claim, or a cluster trial powered as individual), proposes a design that can identify the estimand, states the sample-size approach and assumptions, and quantifies the cost implication. A weak redesign proposes "a randomised trial" without specifying the structure, the control, or the sample size.

## 14. Interview questions

**Question 1. When would you choose a crossover trial over a parallel trial?**

A strong answer: when the condition is stable, the treatment effect is reversible, carryover can be excluded by a washout, and the within-person contrast is the question. It names the efficiency gain and the assumptions. A weak answer says crossover is "more efficient" without the conditions.

**Question 2. What is the design effect, and why does it matter?**

A strong answer: it is the factor by which the sample size must be multiplied to account for clustering, equal to 1 + (m - 1) * ICC; it matters because ignoring it underpowers the trial and inflates the type I error. A weak answer defines it without explaining the consequence.

**Question 3. Why can a cluster trial not be fixed by enrolling more participants per cluster?**

A strong answer: because the ICC determines how much independent information each additional participant provides; with a fixed number of clusters, the variance of the cluster-level effect is bounded, and adding participants within clusters yields diminishing returns. A weak answer says "clustering reduces power" without the mechanism.

**Question 4. What are the operating characteristics of an adaptive design, and how do you obtain them?**

A strong answer: type I error, power, expected sample size, and the probability of each decision, obtained by simulation under a range of scenarios, with the code and seed reported. A weak answer says "you simulate them" without naming the quantities.

**Question 5. What is the difference between a platform trial and a set of separate trials?**

A strong answer: a platform uses a master protocol, a shared control, common infrastructure, and prespecified rules for adding and dropping arms; separate trials have independent protocols and controls. The platform gains efficiency but creates multiplicity and non-concurrency problems. A weak answer says a platform is "a trial with many arms."

**Question 6. How do you control multiplicity in a multi-arm trial?**

A strong answer: prespecify a testing hierarchy, a gatekeeping procedure, or an alpha allocation across arms and endpoints; power the trial for the strategy; report all arms. A weak answer says "Bonferroni" without the hierarchy or the power implications.

**Question 7. When is a single-arm trial with an external control acceptable?**

A strong answer: when randomisation is infeasible or unethical, the natural history is well characterised, the endpoint is objective, the external data are contemporaneous and comparable, and the analysis includes confounding adjustment and sensitivity analyses. A weak answer says "in rare diseases" without the conditions.

**Question 8. What are the main risks of a decentralised trial?**

A strong answer: measurement variability from local or home devices, data provenance and integrity, informed-consent challenges, shipping and storage of investigational products, and regulatory uncertainty about remote assessments. A weak answer says "less control" without specifics.

**Question 9. How would you critique a trial that reports a significant result in a subgroup?**

A strong answer: check whether the subgroup was prespecified, whether the interaction test is significant, whether the trial was powered for the subgroup, and whether the result is consistent with the overall effect and biological plausibility. A weak answer says "subgroup analyses are exploratory" without the checks.

**Question 10. What would make you reject a proposed design?**

A strong answer: the design cannot identify the estimand, the control is not credible, the sample size is not linked to the decision, the adaptation is not prespecified, or the analysis population is mismatched. A weak answer names a design type it dislikes.

## 15. Key terms

- **Adaptive design:** a design with prespecified modifications based on interim data.
- **Allocation concealment:** concealment of the next assignment from the person enrolling participants.
- **Allocation ratio:** the ratio of participants assigned to each arm.
- **Assay sensitivity:** the ability of a trial to distinguish an effective treatment from an ineffective one.
- **Basket trial:** one treatment tested across multiple diseases or subtypes.
- **Blinding:** concealment of assignment from participants, clinicians, assessors, or analysts.
- **Carryover:** persistence of a treatment effect into a subsequent period.
- **Cluster randomised trial:** randomisation of groups rather than individuals.
- **Contamination:** control participants receiving the intervention or its effects.
- **Crossover trial:** each participant receives multiple strategies in a randomised sequence.
- **Design effect:** the sample-size multiplier for clustering, 1 + (m - 1) * ICC.
- **Digital endpoint:** an endpoint measured by a sensor or app.
- **Enrichment:** selecting participants more likely to respond.
- **Estimand:** the precise quantity to be estimated.
- **Factorial design:** randomisation on two or more factors simultaneously.
- **Futility:** stopping when the interim data suggest the target effect is unlikely.
- **Intention-to-treat (ITT):** analysis by assigned group regardless of treatment received.
- **Intracluster correlation coefficient (ICC):** the proportion of variance attributable to between-cluster variation.
- **Master protocol:** a single protocol governing multiple substudies or arms.
- **Minimisation:** adaptive allocation balancing prognostic factors.
- **Modified ITT (mITT):** an ITT-like population with prespecified exclusions.
- **Parallel-group trial:** participants randomised to one strategy for the duration.
- **Per-protocol:** analysis restricted to adherent participants.
- **Platform trial:** a master protocol with a shared control and multiple experimental arms.
- **Sample-size re-estimation (SSR):** adjusting the sample size based on interim information.
- **Seamless design:** a Phase 2 and Phase 3 trial combined in one protocol.
- **Stepped-wedge:** clusters switch from control to intervention in a randomised sequence.
- **Stratification:** randomisation within defined subgroups.
- **Umbrella trial:** multiple treatments tested in one disease.
- **Washout:** a treatment-free interval between periods.

## 16. Sources

All URLs accessed 25 October 2026.

### Guidelines and regulations

- ICH, E6(R3) Guideline for Good Clinical Practice, Step 4, 6 January 2025. https://database.ich.org/sites/default/files/ICH_E6%28R3%29_Step4_FinalGuideline_2025_0106.pdf
- ICH, E9 Statistical Principles for Clinical Trials, Step 4, 1998. https://database.ich.org/sites/default/files/E9_Guideline.pdf
- ICH, E9(R1) Addendum on Estimands and Sensitivity Analysis, Step 4, 20 November 2019. https://database.ich.org/sites/default/files/E9-R1_Step4_Guideline_2019_1203.pdf
- ICH, E10 Choice of Control Group and Related Issues in Clinical Trials, Step 4, 20 July 2000. https://database.ich.org/sites/default/files/E10_Guideline.pdf
- European Union, Regulation (EU) No 536/2014 on clinical trials. https://eur-lex.europa.eu/eli/reg/2014/536/oj
- US Code of Federal Regulations, 21 CFR Part 812, Investigational Device Exemptions. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-812
- US Code of Federal Regulations, 21 CFR Part 814, Premarket Approval. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-814

### FDA guidance

- FDA, "Adaptive Designs for Clinical Trials of Drugs and Biologics," final guidance, December 2019. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/adaptive-design-clinical-trials-drugs-and-biologics-guidance-industry
- FDA, "Master Protocols for Drug and Biological Product Development," guidance (draft December 2023; check current version). https://www.fda.gov/regulatory-information/search-fda-guidance-documents/master-protocols-drug-and-biological-product-development
- FDA, "Conducting Clinical Trials With Decentralized Elements," final guidance, September 2024. https://www.fda.gov/media/167696/download
- FDA, "Considerations for the Design and Conduct of Externally Controlled Trials for Drug and Biological Products," draft guidance, February 2023. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-design-and-conduct-externally-controlled-trials-drug-and-biological-products
- FDA, "Use of Real-World Evidence to Support Regulatory Decision-Making for Medical Devices," final guidance, December 2025. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/use-real-world-evidence-support-regulatory-decision-making-medical-devices

### Reporting guidelines

- Hopewell S, Chan A-W, Collins GS, et al. CONSORT 2025 statement. BMJ 2025;389:e081123. https://doi.org/10.1136/bmj-2024-081123
- Chan A-W, Boutron I, Hopewell S, et al. SPIRIT 2025 statement. BMJ 2025;389:e081477. https://doi.org/10.1136/bmj-2024-081477
- Campbell MK, Elbourne DR, Altman DG. CONSORT statement: extension to cluster randomised trials. BMJ 2004;328:702-708. https://doi.org/10.1136/bmj.328.7441.702

### Methods

- Rutterford C, Copas A, Eldridge S. Methods for sample size determination in cluster randomized trials. Int J Epidemiol 2015;44:1051-1067. https://doi.org/10.1093/ije/dyv113
- Hemming K, Girling AJ, Sitch AJ, et al. Sample size calculations for cluster randomised controlled trials with a fixed number of clusters. BMC Med Res Methodol 2011;11:102. https://doi.org/10.1186/1471-2288-11-102
- Loudon K, Treweek S, Sullivan F, et al. The PRECIS-2 tool: designing trials that are fit for purpose. BMJ 2015;350:h2147. https://doi.org/10.1136/bmj.h2147
- Hernán MA, Robins JM. Using Big Data to Emulate a Target Trial When a Randomized Trial Is Not Available. Am J Epidemiol 2016;183:758-764. https://doi.org/10.1093/aje/kwv254

### Case evidence

- RECOVERY Collaborative Group. Dexamethasone in Hospitalized Patients with Covid-19. N Engl J Med 2021;384:693-704. https://doi.org/10.1056/NEJMoa2021436
- Perez MV, Mahaffey KW, Hedlin H, et al. Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation. N Engl J Med 2019;381:1909-1917. https://doi.org/10.1056/NEJMoa1901183
- FDA, De Novo classification decision for IDx-DR, DEN180001, 2018. https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
- REMAP-CAP Investigators. Interleukin-6 Receptor Antagonists in Critically Ill Patients with Covid-19. N Engl J Med 2021;384:1491-1502. https://doi.org/10.1056/NEJMoa2100433

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`01_from-claim-to-evidence.md`](01_from-claim-to-evidence.md)
- Next topic: [`03_real-world-evidence.md`](03_real-world-evidence.md)
- Related topic: [`06_sample-size-and-power.md`](06_sample-size-and-power.md)
- Module survey file: [`../../05_study_design_evidence.md`](../../05_study_design_evidence.md)
- Labs: [`../../labs/`](../../labs/)
- Cohort pipeline lab: [`../../labs/03_cohort_pipeline.ipynb`](../../labs/03_cohort_pipeline.ipynb)
