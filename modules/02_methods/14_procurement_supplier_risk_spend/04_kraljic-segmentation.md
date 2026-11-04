# Kraljic Segmentation and Sourcing Strategy

> Module 14 — Procurement, Supplier Risk, and Spend Analytics
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 14 topics 01–03 and basic arithmetic
> Version 2026.5

## 1. Why this topic exists

A procurement director cannot manage every category
as if it were a high-stakes partnership, nor conduct
an auction for every purchase. Buyers face limited
time, operational teams face different disruption
consequences, and suppliers operate in markets with
different capacities and switching barriers. A
common decision frame helps the team allocate
attention without pretending that one negotiation
technique fits every purchase.

Consider a low-cost sterilization filter required to
release a medicine. Its annual invoice value may be
small, but a failed filter can stop production.
Conversely, corrugated cartons may account for a
large budget while being available from several
qualified suppliers. Sorting only by spend directs
leadership attention toward the cartons and can miss
the filter's supply constraint.

Kraljic segmentation places a defined purchasing
category on two dimensions: business impact and
supply risk. The resulting four segments suggest
distinct starting strategies: leverage, strategic,
bottleneck, and non-critical. The matrix is a
discussion and resource-allocation device, not a
mathematical proof of the best award. A category is
not a supplier, and a supplier's overall
relationship can span several categories.

This lesson shows how to define the decision, build
scores from evidence, expose uncertain ratings,
challenge segmentation bias, and govern actions over
time. A hypothetical manufacturer supplies the
worked example. Figures are illustrative, in US
dollars (USD), and relate to a 2025 annual baseline
unless otherwise stated. The proposed actions are
plans for analysis, not supplier commitments or
realized savings.

## 2. Learning outcomes

After this lesson you can identify the four
quadrants and explain why high spend and supply risk
are separate dimensions; exercise 1 tests recall.
You can define a category and decision boundary, map
evidence to transparent ordinal scores, and classify
categories using an explicit threshold; exercises 2
and 3 test construction and arithmetic.

You can compare sourcing actions using total
economics, constraints and implementation risk
rather than treating a quadrant as an award
instruction; exercise 4 tests the comparison. You
can quantify threshold sensitivity, missing
information and alternative weight choices; exercise
5 tests robustness. You can detect framing,
incumbent, geography and survivorship biases and
specify counter-evidence; exercise 6 tests critique.

You can design decision rights, a review cadence and
a traceable override; exercise 7 tests governance.
You can reproduce a small calculation, distinguish
scenario assumptions from commitments, and explain
the limitations of an ordinal index; exercise 8
tests execution.

## 3. The core mental model

```text
                         SUPPLY RISK
                     low                 high
BUSINESS     high   LEVERAGE             STRATEGIC
IMPACT              compete carefully    secure and collaborate
             low    NON-CRITICAL         BOTTLENECK
                    simplify demand      assure continuity

scope -> evidence -> scores -> challenge -> action -> review
                        |          |           |
                  uncertainty   bias check   owner + measures
```

The horizontal dimension concerns the difficulty and
consequences of obtaining an acceptable supply,
including concentration, qualification, capacity and
recovery. The vertical dimension concerns the
importance of the category to the buyer, which can
include expenditure and operational contribution.
These dimensions are related in practice but must be
scored separately. If loss consequence appears on
both axes, record the overlap and avoid treating the
combined score as independent evidence.

Leverage categories usually invite competition,
specification discipline, consolidation where
justified, and negotiation. Strategic categories
warrant senior cross-functional attention,
continuity, relationship management and carefully
designed collaboration. Bottleneck categories
emphasize availability and resilience even where the
annual bill is modest. Non-critical categories favor
efficient ordering and demand controls rather than
expensive negotiation events.

The arrows are not automatic. A strategic category
may need competition, a leverage category may
require a reserve, and a bottleneck category may
become leverage after qualification. The label is a
hypothesis about action; the case for a specific
action rests on cost, capacity, market evidence,
regulation, risk appetite and stakeholder judgment.
A score is a point estimate, not a probability of
disruption.

## 4. Definitions and vocabulary

| Term | Meaning and common misuse | Unit or convention |
|---|---|---|

| Category | Decision-relevant set of substitutable goods or services; not automatically a ledger account | Defined scope and geography |
| Business impact | Importance to the buyer across chosen criteria; not simply unit price | Ordinal score |
| Supply risk | Difficulty of securing acceptable supply; not supplier financial risk alone | Ordinal score |
| Leverage | High impact, low supply risk; not permission to ignore supplier viability | Quadrant |
| Strategic | High impact, high supply risk; not necessarily a sole source | Quadrant |
| Bottleneck | Low impact, high supply risk; not an unimportant item | Quadrant |
| Non-critical | Low impact, low supply risk; not zero risk | Quadrant |
| Qualified source | Supplier approved for the defined specification, site and period; not merely a name on a list | Supplier-site-scope |
| Concentration | Share held by one supplier or group; not proof of market power | Percent of scoped spend or units |
| Switching time | Elapsed time to qualify and activate an alternative; not just contracting time | Months |
| Threshold | Declared boundary separating high and low; not a universal law | Score points |
| Weight | Relative contribution of a criterion; not empirical causality | Shares summing to one |
| Override | Documented departure from mechanical classification; not deletion of evidence | Reason, owner, expiry |
| Sensitivity | Change in result when inputs or assumptions change; not validation | Points or scenario |
| Confidence | Quality and completeness of supporting evidence; not the score itself | Low, medium or high |
| Total cost of ownership | Life-cycle cost of an option; not just purchase price | USD per defined horizon |
| Exposure | Potential scale of adverse consequence; not automatically expected loss | USD, time or service |
| Expected loss | Probability multiplied by consequence for a defined event and horizon; not an ordinal index | USD per horizon |
| Supplier parent | Group used for concentration analysis; not necessarily the contracting entity | Effective-dated identifier |
| Segmentation bias | Systematic distortion in scope, evidence or scoring; not merely disagreement | Narrative and test |
| Risk appetite | Decision-maker tolerance for specified exposures; not a substitute for evidence | Approved policy |
| Trigger | Event requiring review before routine cadence; not an automatic award | Observable condition |

## 5. The mechanics, step by step

### 1. State the decision and horizon

Write the decision, the accountable executive, the
relevant sites, and the date by which an action is
needed. A sourcing decision for the next contract
differs from a resilience investment over several
years. Do not blend a historical spend observation,
a forecast, an approved budget and a contractual
commitment into one number.

Input: business request and current contracts.
Output: a one-page decision charter. Branch: if the
decision concerns an individual supplier rather than
a market category, use a supplier scorecard as well
as segmentation.

### 2. Define a coherent category

Group items that share a supply market,
substitutable specifications and a sourcing
decision. Keep regulated variants separate where
qualification requirements differ materially. A
category may be split by geography when logistics,
law or supplier access differ; retain a roll-up to
reveal enterprise exposure.

Input: invoice lines, item descriptions,
specifications and site requirements. Output:
versioned item-to-category mapping with inclusions
and exclusions. Branch: if stakeholders cannot agree
on substitution, create provisional subcategories
and resolve with engineering.

### 3. Reconcile the baseline

Use a documented period and currency, signed
credits, supplier legal entities and parent
mappings. Reconcile the selected population to the
spend cube and explain exclusions. Spend is one
input, not a direct measure of value or risk. Avoid
adding purchase order values and invoices for the
same transaction.

Input: reconciled cube, demand plan and category
map. Output: annual scoped spend, units and
coverage. Branch: if classification is incomplete,
quantify unknowns and restrict confidence rather
than distribute them silently.

### 4. Map the supply market

List suppliers actually qualified for the exact
product, site and country. Distinguish firms that
could potentially qualify from sources ready to
ship. Identify common ownership, single
manufacturing sites, sub-tier dependencies,
bottleneck equipment, capacity and transport routes.

Input: supplier master, quality approvals, contracts
and market research. Output: source map and dated
evidence. Branch: when only a distributor is
visible, mark the upstream manufacturer unknown and
commission a trace rather than infer independence.

### 5. Define impact criteria

Select criteria that represent the business decision
without counting the same consequence twice. This
example uses annual spend and operational
criticality at equal weight. Criticality reflects
effect on production or service if the input is
unavailable; it is not the same as frequency of past
shortages.

Input: finance baseline and operations assessment.
Output: scored definitions and supporting data.
Branch: if safety or patient harm cannot be
represented acceptably by an average, use a
separately governed mandatory escalation rule.

### 6. Define supply-risk criteria

Use qualified-source concentration and switching
difficulty as practical indicators. Define each
rating before seeing the desired quadrant. A high
risk score does not mean a known probability of
failure, and a supplier’s strong delivery history
does not establish the resilience of a common
upstream plant.

Input: qualification register, contracts, recovery
evidence and market intelligence. Output: rating and
confidence for each criterion. Branch: a source
available only after a lengthy validation is not
counted as immediately qualified.

### 7. Calibrate scores

Agree anchors with procurement, operations, finance
and quality. Assign scores independently, then
compare disagreements against evidence. Preserve the
initial rating, dissent and final rationale; the
meeting should not replace data with the senior
person’s preference.

Input: criteria, anchors and evidence. Output:
criterion ratings on the same one-to-five ordinal
scale. Branch: if evidence falls between anchors,
select the conservative supported rating and show
the alternative in sensitivity analysis.

### 8. Compute and plot

Multiply each criterion score by its declared weight
and add within its axis. Apply the stated boundary
consistently, including equality. Show both
component ratings alongside the plotted point so
that users can see different routes to the same
composite.

Input: four scores and two pairs of weights. Output:
two axis scores and preliminary quadrant. Branch: if
either score is near its boundary, present plausible
alternatives rather than declaring the label stable.

### 9. Challenge the result

Ask who selected the boundaries, which sources are
invisible, whose operational loss is excluded, and
whether incumbents received better evidence than
challengers. Test subcategories separately and
examine whether the result changes when missing data
is represented adversely.

Input: provisional matrix and evidence register.
Output: challenge log, confidence and change
requests. Branch: if one site faces a
safety-critical constraint hidden by aggregation,
manage that subcategory separately.

### 10. Select actions

Generate feasible options, then compare cost,
service, resilience, implementation time and
contractual restrictions. Assign actions suited to
the category, not just a generic quadrant label.
Separate no-regret controls from investments
requiring approval.

Input: segment, constraints, demand and supplier
capabilities. Output: dated category plan with
owner, measures, dependencies and decision gate.
Branch: if the apparent saving assumes unqualified
capacity, price qualification and defer any award
until approval.

### 11. Approve and communicate

A cross-functional forum reviews ratings and
recommendations. The category owner proposes,
operations and quality validate consequences and
qualification, finance checks economics, and the
designated executive approves material commitments.
Legal and compliance review applies where needed.

Input: decision pack, disagreements and options.
Output: signed decision and exception register.
Branch: if policy prohibits an award, the
segmentation result does not override policy.

### 12. Monitor and refresh

Review material categories at least annually as an
illustrative governance choice and sooner on a
trigger. Monitor actions and outcomes separately
from the segment; a move to low risk can result from
qualified alternatives, but moving the dot is not
itself value.

Input: scorecard, incidents, qualification and
market changes. Output: dated refresh and closed or
renewed exceptions. Branch: a supplier acquisition,
quality withdrawal or plant outage triggers an
immediate reassessment.

### Score anchors for the worked example

These anchors are local training choices, not
Kraljic requirements. Annual spend is scoped 2025
USD: 1 is below $0.5 million; 2 is $0.5 million to
below $2 million; 3 is $2 million to below $5
million; 4 is $5 million to below $10 million; 5 is
at least $10 million. Boundaries are inclusive at
the lower end.

Operational criticality uses 1 for negligible
interruption with readily available substitute, 2
for manageable local disruption, 3 for material
service degradation, 4 for major production
interruption, and 5 for a sustained site-wide or
safety-critical interruption. The assessment is
conditional on loss of the category, not a measured
probability of loss.

Qualified-source concentration uses the largest
parent share of annual category units: 1 below 30%,
2 from 30% to below 50%, 3 from 50% to below 70%, 4
from 70% to below 90%, and 5 at least 90%. A group
with multiple subsidiaries is one parent for this
criterion. Switching difficulty is 1 for less than
one month, 2 for one to under three months, 3 for
three to under six months, 4 for six to under twelve
months, and 5 for twelve months or more. Use a
credible end-to-end qualification estimate, not the
contract negotiation duration alone.

### Category strategy: choice and trade-offs

### Leverage

Benchmark specification and cost drivers; run a
qualified competitive event when credible suppliers
have capacity. Pool demand only if freight, local
requirements and capacity remain feasible. Negotiate
price, indexation, service and termination together.

A 3% price reduction on $8 million annual
addressable spend is $240,000 per year before
implementation cost. If qualification costs $90,000
once and the discount holds one year, the first-year
net is $150,000, not $240,000 realized savings.
Track actual invoiced quantities and price
compliance.

### Strategic

Develop a joint operating plan with transparent
capacity, recovery, quality and financial review.
Negotiate continuity obligations and data rights
without assuming that an incumbent partnership is
the only choice. Evaluate a second source, redesign
or inventory as separately costed options.

A planned $400,000 second-source qualification is
not automatically justified by a high score. If a
defensible annual disruption probability falls from
8% to 3% and the modeled loss is $6 million per
event, modeled expected annual loss falls $300,000.
Compare timing and recurring costs over an approved
horizon.

### Bottleneck

Protect availability through validated buffer stock,
capacity reservation, repair capability or a
qualified substitute. Check shelf life, minimum
order size, storage controls and recovery lead time.
Avoid spending more on an elaborate tender than the
potential benefit.

A buffer of 1,200 units at $40 per unit ties up
$48,000 in inventory. At a 20% annual carrying rate
the modeled carrying cost is $9,600 per year,
excluding expiry, warehouse setup and obsolescence.
It is a cost estimate, not the value of avoided
downtime.

### Non-critical

Simplify catalog, approval and invoicing; automate
low-value transactions where controls permit. Reduce
unnecessary variants and set clear buying channels.
Aggregate only where the market and service
requirements support it.

If 2,400 orders per year cost $18 each to process
and a catalog reduces them to 800 at the same unit
processing cost, modeled process cost falls by
$28,800 per year. This assumes the displaced work
and transaction cost are actually avoidable;
validate with process owners.

### Comparing possible sourcing actions

A quadrant supplies a starting question, not the
answer. Before seeking approval, put feasible
alternatives on the same planning horizon and make
their assumptions visible. A useful decision pack
contains a status quo, at least one feasible change,
and a reason why other attractive options cannot
currently be implemented.

For a leverage category, the status quo may be a
renewal with an indexed price. Compare that with a
competitive event and with a specification change.
Include the cost of preparing bids, engineering
review, freight differences, transition inventory,
contract termination and any minimum volume. A
nominal price reduction achieved by moving volume
to a distant plant can disappear after transport
and inventory costs.

For a strategic category, compare a relationship
plan, a qualified second source, an alternative
design and a resilience inventory policy. Each
option reduces a different exposure. Inventory can
cover a short outage but not necessarily a long
plant closure. A second source may share the same
sub-tier material. A redesign can remove a
dependency but consume engineering capacity and
require customer approval.

For bottleneck purchases, establish a recovery
objective in days and compare it with the time
needed to detect, replace and validate supply.
An emergency stock target should reflect demand
through that interval, variability, usable shelf
life and replenishment reliability. Stock held
centrally may not be usable at every plant if
transport, storage or product qualification differ.

For non-critical purchases, compare the number of
requests, approvals, orders and invoices before and
after a proposed catalog. A lower number of
transactions is useful only if the organization
actually avoids work or can redirect capacity to
work of greater value. Prevent catalog convenience
from becoming uncontrolled demand or a way to
bypass budget authority.

### Distinguishing the market from the incumbent

The category matrix describes the buyer's position
in a supply market; it is not a report card for
the current supplier. A high-performing incumbent
can operate in a high-risk category, for example
when it is the only qualified manufacturer. A weak
incumbent can operate in a low-risk category if
several genuinely viable replacements exist.

Ask procurement to separate three populations:
suppliers that exist in the market, suppliers that
could meet the specification after work, and
suppliers currently approved and capable of
delivery. Record capacity and approval status at
the product-site level. A prospect's sales claim
is a lead, not demonstrated available capacity.

Map how attractive the buyer is to each supplier.
A buyer that represents a small share of supplier
revenue may have limited priority during a
shortage, even if its own category spend is high.
A supplier that relies heavily on the buyer may
offer commitment but also face financial stress
if the buyer rapidly withdraws volume. These are
relationship questions to investigate, not
mechanical quadrant assignments.

### Bias review before approval

Start by asking whether the category boundary was
selected because it reflects a real market or
because it produces a preferred score. A broad
enterprise category can hide a single site that
has no substitute. An excessively narrow category
can hide shared leverage or common exposure.
Compare both views when the sourcing decision and
operational risk occur at different grains.

Check survivorship bias in historical incidents.
If the organization has always purchased extra
inventory to protect a fragile component, an
absence of shutdowns may reflect that protection
rather than intrinsically reliable supply.
Removing the buffer based only on past service
could increase exposure. Record controls already
in place when interpreting historical outcomes.

Check incumbent information advantage. A current
supplier has years of performance records while a
potential entrant may have only a proposal.
Compare evidence types fairly and require
qualification before treating the entrant as
available. Avoid both assuming an unknown
challenger is safe and assuming it cannot qualify.

Check geography and reporting bias. A plant that
reports every near miss can appear riskier than a
plant that logs only shutdowns. Compare reporting
definitions and ask for missing observations
before ranking the locations. Request local input
from the people who manage actual workarounds.
Their evidence is especially important where
central systems lack the cause of a delay.

Finally, retain minority views. Record the scorer,
date, supporting evidence and dissent before
approving the result. An override should state the
decision, business reason, approving authority,
expiration and conditions for reconsideration.
An override does not erase the original score or
authorize an exception to law, quality policy or
contractual obligations.

### Governance measures and escalation

Track evidence completeness separately from
business outcomes. Useful process measures include
the share of material categories reviewed on time,
the share with current qualification evidence,
and the number of expired overrides. State each
measure's denominator, review date and owner. A
complete register is necessary for governance but
does not prove that supply has become resilient.

Outcome measures can include actual recovery time
after an interruption, the share of demand covered
by independently qualified capacity, and realized
cost against an agreed comparable baseline. Each
requires a clear scope and an explanation of
external changes. Avoid claiming that a lower
composite score caused an outcome without evidence.

Escalate an ownership change, factory closure,
loss of certification, material quality incident,
supplier insolvency signal or significant change
in demand. The category owner collects facts and
proposes a revised score; operations and quality
assess continuity and qualification; finance
reviews economics; the authorized executive
approves material commitments. A review trigger
initiates assessment, not automatic termination.

## 6. Worked example

A hypothetical medical-device manufacturer is
planning sourcing for four categories across its US
plants. The data below are a 2025 historical
baseline, not a forecast of 2026 demand. Spend
excludes recoverable tax. Parent shares use units;
switching time includes qualification and site
release. Each number is illustrative. The scoring
anchors are those in section 5.

| Category | Spend USD million/year | Criticality | Largest parent share | Switch months | Spend score | Criticality score | Concentration score | Switch score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cartons | 8.4 | 3 | 42% | 2 | 4 | 3 | 2 | 2 |
| Control chips | 12.6 | 5 | 93% | 14 | 5 | 5 | 5 | 5 |
| Sterile filters | 0.8 | 3 | 81% | 8 | 2 | 3 | 4 | 4 |
| Office supplies | 0.35 | 1 | 25% | 0.5 | 1 | 1 | 1 | 1 |

For this example, impact equals 0.5 times spend
score plus 0.5 times criticality score. Supply risk
equals 0.5 times concentration score plus 0.5 times
switching score. Scores range from 1 to 5. Define
high as at least 3.0 and low as below 3.0 on each
axis. The equal weights and cutoff are governance
choices; they are not properties of the original
Kraljic framework.

| Category | Impact calculation | Risk calculation | Segment |
|---|---|---|---|
| Cartons | 0.5 × 4 + 0.5 × 3 = 3.5 | 0.5 × 2 + 0.5 × 2 = 2.0 | Leverage |
| Control chips | 0.5 × 5 + 0.5 × 5 = 5.0 | 0.5 × 5 + 0.5 × 5 = 5.0 | Strategic |
| Sterile filters | 0.5 × 2 + 0.5 × 3 = 2.5 | 0.5 × 4 + 0.5 × 4 = 4.0 | Bottleneck |
| Office supplies | 0.5 × 1 + 0.5 × 1 = 1.0 | 0.5 × 1 + 0.5 × 1 = 1.0 | Non-critical |

The total scoped annual spend is $8.4 million +
$12.6 million + $0.8 million + $0.35 million =
$22.15 million. Control chips represent $12.6
million / $22.15 million = 56.9% of this
four-category baseline, rounded. This is not the
share of all company spend. The filter's low annual
spend does not make its supply risk low.

For cartons, suppose 70% of the $8.4 million
baseline is genuinely addressable and a competitive
event produces a forecast 4% price reduction on that
portion, with $80,000 one-time qualification and
transition cost. The projected gross annual effect
is $8.4 million × 0.70 × 0.04 = $235,200. The
projected first-year net is $155,200 before volume,
freight, inflation, tax and other changes. It
becomes realized only when qualified awards are
implemented and actual comparable transactions
confirm the effect.

For chips, suppose a qualified alternative would
reduce a modeled 10% annual probability of a
six-week outage to 4%, and the scenario's net loss
per outage is $5 million. The reduction in modeled
expected annual loss is (0.10 − 0.04) × $5 million =
$300,000. A $650,000 qualification project with
$30,000 annual maintenance is not obviously
justified on year-one expected value: $300,000 −
$650,000 − $30,000 = −$380,000. Over multiple years,
timing, discounting, correlated failure and risk
appetite matter. The assumed probabilities are
scenario inputs, not empirical estimates, and the
model excludes partial outages and other benefits.

For filters, eight months of switching means a
buffer must be matched to replenishment and shelf
life rather than chosen from the score alone. For
office supplies, a controlled catalog can remove
administrative touches, but a stated processing-cost
rate is not automatically a cash saving. The four
categories thus produce different actions even
though all appear in the same spend cube.

### Boundary sensitivity and disagreement

The filter impact score is 2.5, below the 3.0
boundary. If operations supports a criticality
rating of 4 rather than 3, the impact becomes (2 +
4) / 2 = 3.0 and the category becomes strategic
under the stated inclusive cutoff. This is a
decision-relevant ambiguity, not evidence that
either label is objectively correct. Record both
ratings, the plant-specific consequence and the
owner who resolves the disagreement. A separate
safety escalation can require continuity action
regardless of label.

Suppose the carton team's switching estimate
increases from two to seven months after quality
review. Switching score rises from 2 to 4, risk
becomes (2 + 4) / 2 = 3.0, and the category changes
from leverage to strategic. A price-only tender
would then be inappropriate without qualification
and continuity controls. The result is sensitive to
evidence, not merely to the arithmetic.

## 7. Code, query, or calculation

The following Python 3 program is standalone and
uses only the standard library. It reproduces the
four scores and two scenario calculations. Its
inputs are already calibrated ordinal ratings; it
does not independently verify the underlying claims.
The threshold is inclusive and labels are based on
unrounded values.

```python
# Python 3; USD and one-year scenarios.
rows = [
    ("Cartons", 8.4, 4, 3, 2, 2),
    ("Control chips", 12.6, 5, 5, 5, 5),
    ("Sterile filters", 0.8, 2, 3, 4, 4),
    ("Office supplies", 0.35, 1, 1, 1, 1),
]
threshold = 3.0

def classify(impact, risk):
    if impact >= threshold:
        return "Strategic" if risk >= threshold else "Leverage"
    return "Bottleneck" if risk >= threshold else "Non-critical"

for name, spend, spend_score, criticality, concentration, switch in rows:
    impact = 0.5 * spend_score + 0.5 * criticality
    risk = 0.5 * concentration + 0.5 * switch
    print(f"{name}: impact={impact:.1f}, risk={risk:.1f}, "
          f"segment={classify(impact, risk)}")

total = sum(row[1] for row in rows)
print(f"Total: ${total:.2f} million/year")
gross = 8_400_000 * 0.70 * 0.04
net = gross - 80_000
print(f"Carton gross: ${gross:,.0f}; first-year net: ${net:,.0f}")
loss_reduction = (0.10 - 0.04) * 5_000_000
print(f"Chip modeled annual loss reduction: ${loss_reduction:,.0f}")
```

Expected output:

```text
Cartons: impact=3.5, risk=2.0, segment=Leverage
Control chips: impact=5.0, risk=5.0, segment=Strategic
Sterile filters: impact=2.5, risk=4.0, segment=Bottleneck
Office supplies: impact=1.0, risk=1.0, segment=Non-critical
Total: $22.15 million/year
Carton gross: $235,200; first-year net: $155,200
Chip modeled annual loss reduction: $300,000
```

The script uses floating-point arithmetic for
display and simple scenario calculations. For
financial settlement use the organization's approved
decimal and rounding policies. It assumes that
component scores are valid, categories do not
overlap, and historical spend is appropriate for the
scenario. Changing the threshold or one score
changes classification; it does not recalculate the
underlying commercial value.

## 8. Data, standards, and regulatory context

Build a traceable evidence register with category
identifier, item scope, site, legal entity, source
system, observation date, unit, amount, parent
mapping, score, confidence, reviewer and next
review. Source the spend baseline from a reconciled
cube. Preserve excluded transactions and unresolved
classifications rather than assigning them to a
convenient category.

Supplier qualification records should identify the
approved supplier, manufacturing site, product
revision, approved sites and effective dates. A
parent-level aggregation answers corporate
concentration questions; the legal entity and site
remain necessary for contracting and physical
disruption. Two distributors may rely on the same
factory, and two factories may rely on a single
sub-tier component.

UNSPSC is a product and services classification
system; GS1 Global Product Classification (GPC) is
another classification approach. Neither determines
the business's sourcing category or the Kraljic
quadrant. An enterprise may map either taxonomy into
its own decision-relevant category, with a versioned
crosswalk and exceptions. A product classification
does not establish supplier qualification,
substitutability or regulatory clearance.

ISO 31000:2018 provides general risk-management
guidelines, not a prescribed Kraljic scoring formula
or a certifiable management-system requirement. ISO
20400:2017 gives sustainable procurement guidance;
teams should consider relevant sustainability issues
in sourcing decisions without asserting that a
particular matrix satisfies every requirement. The
cited standards are guidance; any binding
requirement comes from applicable law, contract or
another explicitly applicable instrument.

In regulated medical-device contexts, US Food and
Drug Administration (FDA) Quality Management System
Regulation requirements must be checked against the
current rule and product scope. The FDA states that
its rule incorporating ISO 13485:2016 by reference
became effective on February 2, 2026. Supplier
controls and changes require the organization's
quality and legal assessment; this educational
matrix cannot authorize a supplier change. Public
procurement, competition and data-protection
obligations vary by jurisdiction. Do not share
competitors' confidential pricing or assume
collaborative sourcing exempts a team from
applicable competition law.

Keep source facts separate from analyst judgments.
Evidence may be confidential: restrict access to
supplier financial data and negotiation records,
establish retention rules, and disclose only what is
necessary for the decision. A low-confidence score
is not a low-risk score. Version the taxonomy,
thresholds, weights and effective dates so historic
decisions can be reconstructed.

### A practical evidence register

At category creation, capture the decision grain:
the item family, required specifications, receiving
locations, currencies and dates. Keep the source
identifiers that connect the category to invoices,
contracts and approved supplier records. Record
who created the mapping and who can change it.
That lineage allows a reviewer to reproduce a score
without depending on a spreadsheet author's memory.

For each criterion, store the measured observation
and the assigned rating as separate fields. A
record might contain a switching estimate of eight
months, a rating of 4, the engineering assessment
date, and the name of the approving reviewer. The
rating can then be recalculated when the boundary
changes without rewriting the original observation.

For every source, retain the date it was valid as
well as the date it entered the analytical system.
A new supplier parent may be reported in February
even though the acquisition closed in January.
The distinction matters when reconstructing a
decision taken between those dates. A historical
matrix should not silently adopt today's ownership
structure and appear to have known it in advance.

Unknown observations deserve an explicit status.
For example, an absent factory address should not
be converted into a unique factory identifier for
each distributor. Such an imputation would create
apparent diversification without supporting
evidence. Request the information, record the
response deadline, and show a scenario in which
the sources share a facility.

### Scoring and confidence are different fields

An analyst can have high confidence that a
category has a high concentration score because
the qualified-source register and unit volumes
have been independently checked. The same analyst
can have low confidence in a high switching score
if the only estimate is an old email. Both may
produce the same plotted point, but the next
action differs: the first needs a resilience
decision and the second also needs better evidence.

One practical confidence scale is high when the
observation is current, independently supported
and applicable to the exact scope; medium when
one material qualification remains open; and low
when the estimate relies on an unverified claim
or an outdated source. These are local working
definitions, not a statistical confidence interval.
Require a named evidence owner for every low-
confidence rating used in a material decision.

If a rating is uncertain over an interval, compute
the possible combinations rather than replacing
the uncertainty with a decimal. A switching score
between 2 and 4, combined with concentration 3 at
equal weight, produces a risk interval from
(3 + 2) / 2 = 2.5 to (3 + 4) / 2 = 3.5. With a
3.0 cutoff the category can fall on either side.
The right response is to examine the underlying
switching evidence and prepare a conditional plan.

If both uncertainty and operational consequence
are material, the decision authority can require
an interim control while the evidence is gathered.
The interim control should have a cost, an owner,
an expiry date and an explicit release condition.
This does not imply that every unknown must be
scored as the maximum; it prevents uncertainty
from being mistaken for evidence of safety.

### Comparing weights without overclaiming

Equal weights are transparent but not necessarily
correct. Consider a category with a spend rating
of 4 and a criticality rating of 2. Equal weights
produce an impact of 3.0. If the team uses weights
of 0.25 for spend and 0.75 for criticality, the
impact becomes 0.25 × 4 + 0.75 × 2 = 2.5. If it
uses 0.75 for spend and 0.25 for criticality, the
impact becomes 3.5. All three calculations are
arithmetically valid; they express different
managerial priorities.

The team should set a defensible range of weights
before choosing an action and should explain
whether the recommendation changes across that
range. Do not search for weights that produce a
preferred label. When safety or regulatory impact
must not be traded against spend, apply a separate
mandatory gate instead of relying on a weighted
average to guarantee the right outcome.

Ordinal ratings are ordered categories, not
necessarily equally spaced measurements. The
distance between ratings 1 and 2 need not equal
the distance between 4 and 5. A weighted average
is therefore a practical prioritization convention.
It should be supplemented by the underlying
observations, a sensitivity analysis and informed
judgment, not interpreted as a physical quantity.

### Controlling the classification lifecycle

At initial publication, freeze the scoring version,
category map, evidence snapshot and calculated
quadrants. Make the published version read-only
for ordinary users. New evidence can create a new
version while preserving the previous decision's
inputs. This matters when management later asks
why a particular award was made.

An approval record should distinguish the analyst
who calculated a score from the person authorized
to accept a strategy. Quality, engineering and
operations own the technical facts within their
remits. Procurement owns the category proposal;
finance checks baseline and economics; the
designated executive approves commitments. These
roles can be combined in a small organization, but
the person taking risk must be identifiable.

At each refresh, compare changes in underlying
data with changes in the method. A category may
move because an alternative supplier was approved,
because its parent share changed, or because the
scoring boundary was revised. These are different
causes. A lower rating caused only by a new
threshold should not be reported as a resilience
improvement.

Close the loop by connecting approved actions to
actual procurement events. A second source is not
operationally available merely because an action
plan says it will be qualified. Record completed
validation, contracted capacity, tested logistics
and any remaining dependency. Where the source has
not been activated, show the original exposure
alongside the planned future state.

## 9. Industry practice and case evidence

Kraljic's 1983 Harvard Business Review article
framed purchasing as supply management and proposed
a portfolio approach to purchasing categories. It is
a conceptual management framework, not a validation
of any particular numerical weighting. The
four-quadrant operationalization in this lesson is a
teaching convention built around business impact and
supply risk. The article is evidence for the
historical framework, not for the hypothetical
case's savings.

In 2011, the Tōhoku earthquake and tsunami disrupted
Japanese manufacturing supply chains. Toyota
publicly described the disruption and its production
response in its 2011 annual report. The event
illustrates that physical dependencies and recovery
time matter to sourcing, but a company-wide event
does not prove that a particular Kraljic score
predicted the loss. [Inference] Mapping shared
upstream sites could help a buyer identify exposure
that separate supplier names conceal.

In 2021, the semiconductor shortage affected
automobile production. Ford's 2021 annual report
discussed semiconductor supply constraints and their
production effects. This documents a real
demand-and-supply disruption, not the quantitative
probabilities in the worked example. [Inference] For
a buyer dependent on a specialized chip, capacity,
design qualification and common upstream fabrication
are relevant evidence when assessing switching
difficulty.

A 2023 U.S. Government Accountability Office report
on semiconductor supply chains describes
vulnerabilities and federal efforts to increase
domestic semiconductor manufacturing. It provides
public-sector context, not an estimate of any one
buyer's risk. Use named cases to generate questions,
then gather current, category-specific evidence
before making an award. Historical disruptions
differ in products, regions, contracts, and recovery
conditions.

Published examples may overrepresent visible
failures and surviving firms. Absence of a reported
incident is not evidence of resilience. A supplier's
private near misses, inventory buffers and upstream
constraints may not appear in public reporting.
Compare multiple sources, record dates, and
explicitly distinguish what each source says from
the buyer's inference.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|

| Spend-only segmentation | Low-spend plant consumables omitted | Value is equated with invoices | Production exposure hidden | Add operational criticality and escalation |
| Supplier-count illusion | Many names but one factory | Legal entities mask a common node | False diversification | Map parent and production site |
| Stale qualification | Approved list has old revisions | Availability assumed from historical approval | Switch fails in an outage | Revalidate site and specification |
| Boundary gaming | Scores repeatedly set at 2.9 | Rating adjusted to avoid scrutiny | Weak oversight | Retain evidence and independent challenge |
| Incumbent bias | Only incumbent data are detailed | Challengers lack equal evidence | Competition suppressed | Use common evidence requests |
| False precision | Scores reported to two decimals | Ordinal ranks treated as measurements | Overconfident decisions | Show components and sensitivity |
| Double counting | Downtime appears on both axes | Same consequence counted twice | Inflated importance | Document constructs and overlap |
| Averaging away hazards | Enterprise score masks one plant | Aggregate dilutes local criticality | Unsafe continuity plan | Split material site-level exposure |
| Unpriced transition | Auction saving ignores validation | Switching costs omitted | Negative net benefit | Model life-cycle cost and timing |
| Missing sub-tier | Distributor cannot name origin | Upstream dependency invisible | Common-cause outage | Request provenance and contingency |
| Unqualified capacity | Bid is treated as available supply | Approval or ramp-up omitted | Award cannot be fulfilled | Gate award on qualification |
| Static matrix | No refresh after acquisition | Market structure changed | Obsolete strategy | Use event-based review |
| Biased missingness | Unknown supplier data scored low | Absence treated as safety | Systematic underestimation | Mark unknown and bound outcomes |
| Unowned action | Quadrant has no named owner | Analysis stops at plot | No risk reduction | Assign action, deadline and metric |
| Forced standardization | Sites lose required variants | Volume pooling ignores specification | Quality or service failure | Engineering approves equivalence |
| Misused probability | Risk score described as 80% | Ordinal scale confused with probability | Invalid expected loss | Estimate event probabilities separately |
| Supplier distress | Aggressive price cut erodes viability | Buyer power ignores economics | Delivery or quality deterioration | Review sustainability and capacity |
| Confidentiality breach | Supplier quotes circulated broadly | Access controls absent | Legal and commercial harm | Restrict access and involve counsel |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|

| Category scope is coherent | One strategy addresses a common market | Products have incompatible specifications | Split by qualification and market |
| Spend is reconciled | Impact baseline represents scoped purchases | Credits or duplicate invoices omitted | Bridge to cube and review exceptions |
| Scores are comparable | A common scale supports review | Different assessors use different anchors | Calibrate with examples and audit evidence |
| Weights reflect priorities | Composite expresses chosen trade-offs | Safety consequence is averaged away | Use explicit escalation and sensitivity |
| Parent mapping is current | Concentration has the correct denominator | Merger or sale changes ownership | Effective-date and event-trigger refresh |
| Switching time is credible | Risk rating reflects actual substitution | Validation or equipment lead time omitted | Validate with engineering and quality |
| Historical spend informs future scope | Scenario has a usable baseline | Volume or specification changes materially | Use demand scenarios, not history alone |
| Scenario events are defined | Expected loss arithmetic has meaning | Losses overlap or probabilities are inconsistent | Specify event, horizon and dependence |
| Supplier capacity is usable | Competitive option is feasible | Capacity committed elsewhere | Obtain dated evidence and conditional offers |
| Risk is not a calibrated probability | Avoid false expected-loss inference | Stakeholders interpret scores as chances | Label ordinal scale and estimate probabilities separately |
| Data access is lawful | Analysts can review source evidence | Contracts or privacy rules limit disclosure | Use approved access and legal review |
| The threshold is a choice | Quadrants are consistently assigned | Users mistake it for a standard | Show alternative thresholds and decision impact |

## 12. Exercises

### 1. Recall and explain

A plant buys expensive cartons from many approved
mills and inexpensive sterile filters from one
approved source. Identify the likely segment of each
and explain why spend and supply risk differ.

Deliverable: four sentences and a two-axis sketch.
Acceptance: carton leverage and filter bottleneck
are conditional on low versus high impact and risk,
not asserted from price alone. Time: 10 minutes.

### 2. Define a category

A company buys standard and custom printed labels at
three plants. One plant has a validated safety label
and the others use substitutable stationery. Define
sourcing boundaries and identify evidence needed
before pooling.

Deliverable: a charter with scope, exclusions, grain
and owner. Acceptance: separate the
safety-constrained scope or justify pooling with
documented equivalence; preserve a roll-up. Time: 20
minutes.

### 3. Calculate a segment

A category has spend score 4, criticality 2,
concentration 3 and switching 4. Use equal weights
and high at or above 3.0.

Deliverable: both calculations and quadrant.
Acceptance: classify equality correctly and show
inputs, boundary and interpretation. Time: 10
minutes.

### 4. Evaluate a proposal

On a $6 million annual baseline, 75% is addressable.
A 5% price reduction is forecast; first-year
transition costs are $120,000. Calculate the modeled
gross and first-year net, and name two omitted
factors.

Deliverable: arithmetic and a decision caveat.
Acceptance: distinguish forecast from realized
benefit and account for one-time cost. Time: 15
minutes.

### 5. Test sensitivity

A filter category has spend score 2, criticality 3,
concentration 4 and switch 4. Recompute the segment
when criticality rises to 4, using the
worked-example rules.

Deliverable: two points, labels and an explanation.
Acceptance: identify the inclusive boundary and
describe the evidence needed to resolve the
difference. Time: 15 minutes.

### 6. Challenge bias

A buyer calls a category competitive because four
distributor names appear in the supplier master. Two
share a parent, and all four buy from the same
factory. A new supplier lacks a completed
questionnaire.

Deliverable: a bias and evidence log with three
tests. Acceptance: distinguish legal names, parents
and manufacturing sites; do not treat missing
questionnaire data as low risk. Time: 20 minutes.

### 7. Design governance

A strategic sourcing committee wants a quarterly
dashboard for a regulated category with annual
contracts. Define roles, overrides, triggers, and
measures without claiming a change in quadrant is a
benefit.

Deliverable: one-page governance design. Acceptance:
assign decision rights, dated evidence, exception
expiry, event triggers, quality gate and outcome
measures. Time: 25 minutes.

### 8. Reproduce and adapt

Run or transcribe the calculation in section 7. Add
a category with scores 3, 3, 2, 4 in the same input
order.

Deliverable: new axis scores, quadrant, and one
limitation of the program. Acceptance: show both
values equal 3.0 and classify strategic using
unrounded inclusive thresholds. Time: 15 minutes.

## 13. Self-check answers

### 1. Conditional segmentation

Cartons are leverage if impact is high and risk is
low; filters are bottleneck if impact is low and
risk is high. Low spend does not establish low
criticality, and several names do not prove
independent qualified capacity.

### 2. Category design rubric

A strong charter defines plant, product revision,
supplier qualification and decision horizon. It
separates the validated safety label unless quality
approves equivalence. It retains a parent roll-up
and identifies operations, quality and procurement
owners. A weak answer pools solely to increase
spend.

### 3. Segment arithmetic

Impact is (4 + 2) / 2 = 3.0. Risk is (3 + 4) / 2 =
3.5. Both meet or exceed 3.0, so the segment is
strategic. The answer is conditional on the
specified scales and threshold.

### 4. Proposal economics

Addressable spend is $6,000,000 × 0.75 = $4,500,000
per year. Forecast gross effect is $4,500,000 × 0.05
= $225,000 per year. First-year net is $225,000 −
$120,000 = $105,000. Freight, demand changes,
qualification delays and supplier viability are
examples of omitted factors.

### 5. Boundary test

Initially impact is (2 + 3) / 2 = 2.5 and risk is (4
+ 4) / 2 = 4.0, yielding bottleneck. With
criticality 4, impact becomes 3.0 and risk remains
4.0, yielding strategic. Check plant consequence,
recovery time and source evidence before choosing a
rating.

### 6. Bias critique rubric

Count three parent groups if only two of the four
distributors share one parent, but recognize one
common factory for the physical dependency. Verify
ownership dates, factory and sub-tier provenance,
and qualification of alternatives. Score the
unanswered questionnaire as unknown, obtain
comparable evidence and test a conservative
scenario.

### 7. Governance rubric

A strong design names an accountable executive, a
category owner, quality and operations reviewers,
and finance support. It keeps original scores and
signed overrides with expiry, triggers review on
acquisition, outage or qualification loss, and
measures continuity and realized economics. A
dashboard alone is not an operating control.

### 8. Code adaptation

Impact is (3 + 3) / 2 = 3.0 and risk is (2 + 4) / 2
= 3.0; the category is strategic. The program does
not validate ratings, estimate uncertainty, perform
qualification or recommend an award. An input row
requires an explicit name and spend value in
addition to the four scores.

## 14. Interview questions

### What does each axis mean?

A strong answer separates business importance from
difficulty of obtaining supply and states how each
is measured. A weak answer equates both with annual
spend.

### How do you choose the category grain?

A strong answer uses the sourcing decision,
substitution, geography and qualification, with a
roll-up. A weak answer blindly copies the
general-ledger account.

### Why can a small purchase be a bottleneck?

A strong answer connects limited qualified supply
and switching time to continuity, while checking
operational importance separately. A weak answer
says low spend means low risk.

### What can invalidate a leverage strategy?

A strong answer tests capacity, common ownership,
switching cost and quality approval. A weak answer
starts an auction without checking feasible bidders.

### How should a strategic supplier be managed?

A strong answer discusses continuity, incentives,
data, dependence and alternatives. A weak answer
assumes an exclusive partnership is always optimal.

### What does a score of 4 mean?

A strong answer describes an ordinal anchor and
evidence, not an 80% failure probability. A weak
answer computes expected loss directly from the
score.

### How do you handle a disputed score?

A strong answer records independent evidence, tests
plausible ratings and escalates decision-relevant
differences. A weak answer lets the most senior
attendee choose silently.

### How can a matrix hide common-cause risk?

A strong answer traces parent, site and sub-tier
dependencies across categories. A weak answer counts
supplier IDs and declares diversification.

### How do you prove savings?

A strong answer distinguishes addressable forecast,
implementation costs and actual comparable
transactions. A weak answer counts the negotiated
percentage as realized savings.

### When should the matrix be refreshed?

A strong answer combines a routine cadence with
event triggers and named owners. A weak answer
treats the original plot as permanent.

## 15. Key terms

- **Category:** Decision-relevant set of substitutable goods or services; not automatically a ledger account. See [definitions](#4-definitions-and-vocabulary).
- **Business impact:** Importance to the buyer across chosen criteria; not simply unit price. See [definitions](#4-definitions-and-vocabulary).
- **Supply risk:** Difficulty of securing acceptable supply; not supplier financial risk alone. See [definitions](#4-definitions-and-vocabulary).
- **Leverage:** High impact, low supply risk; not permission to ignore supplier viability. See [definitions](#4-definitions-and-vocabulary).
- **Strategic:** High impact, high supply risk; not necessarily a sole source. See [definitions](#4-definitions-and-vocabulary).
- **Bottleneck:** Low impact, high supply risk; not an unimportant item. See [definitions](#4-definitions-and-vocabulary).
- **Non-critical:** Low impact, low supply risk; not zero risk. See [definitions](#4-definitions-and-vocabulary).
- **Qualified source:** Supplier approved for the defined specification, site and period; not merely a name on a list. See [definitions](#4-definitions-and-vocabulary).
- **Concentration:** Share held by one supplier or group; not proof of market power. See [definitions](#4-definitions-and-vocabulary).
- **Switching time:** Elapsed time to qualify and activate an alternative; not just contracting time. See [definitions](#4-definitions-and-vocabulary).
- **Threshold:** Declared boundary separating high and low; not a universal law. See [definitions](#4-definitions-and-vocabulary).
- **Weight:** Relative contribution of a criterion; not empirical causality. See [definitions](#4-definitions-and-vocabulary).
- **Override:** Documented departure from mechanical classification; not deletion of evidence. See [definitions](#4-definitions-and-vocabulary).
- **Sensitivity:** Change in result when inputs or assumptions change; not validation. See [definitions](#4-definitions-and-vocabulary).
- **Confidence:** Quality and completeness of supporting evidence; not the score itself. See [definitions](#4-definitions-and-vocabulary).
- **Total cost of ownership:** Life-cycle cost of an option; not just purchase price. See [definitions](#4-definitions-and-vocabulary).
- **Exposure:** Potential scale of adverse consequence; not automatically expected loss. See [definitions](#4-definitions-and-vocabulary).
- **Expected loss:** Probability multiplied by consequence for a defined event and horizon; not an ordinal index. See [definitions](#4-definitions-and-vocabulary).
- **Supplier parent:** Group used for concentration analysis; not necessarily the contracting entity. See [definitions](#4-definitions-and-vocabulary).
- **Segmentation bias:** Systematic distortion in scope, evidence or scoring; not merely disagreement. See [definitions](#4-definitions-and-vocabulary).
- **Risk appetite:** Decision-maker tolerance for specified exposures; not a substitute for evidence. See [definitions](#4-definitions-and-vocabulary).
- **Trigger:** Event requiring review before routine cadence; not an automatic award. See [definitions](#4-definitions-and-vocabulary).

## 16. Sources

Kraljic, Peter. “Purchasing Must Become Supply
Management.” Harvard Business Review, September
1983. Foundational conceptual source; it does not
validate the lesson's invented weights or numbers.
[Harvard Business
Review](https://hbr.org/1983/09/purchasing-must-become-supply-management).

ISO, ISO 31000:2018, Risk management — Guidelines.
General risk-management guidance, not a mandatory
matrix. [ISO
31000](https://www.iso.org/standard/65694.html).

ISO, ISO 20400:2017, Sustainable procurement —
Guidance. [ISO
20400](https://www.iso.org/standard/63026.html).

UNSPSC, United Nations Standard Products and
Services Code. Product and service taxonomy
information. [UNSPSC](https://www.unspsc.org/).

GS1, Global Product Classification. Product
classification information. [GS1
GPC](https://www.gs1.org/standards/gpc).

U.S. Food and Drug Administration, Quality
Management System Regulation resources. Consult
current requirements and effective dates for
applicable devices. [FDA
QMSR](https://www.fda.gov/medical-devices/quality-and-compliance-medical-devices/quality-management-system-regulation-qmsr).

Toyota Motor Corporation, Annual Report 2011. Public
company account of the earthquake and production
disruption. [Toyota annual
reports](https://global.toyota/en/ir/library/annual/).

Ford Motor Company, 2021 Annual Report. Public
company discussion of semiconductor supply
constraints. [Ford annual
reports](https://shareholder.ford.com/investors/financials/annual-reports/default.aspx).

U.S. Government Accountability Office, Semiconductor
Supply Chain: Information on Federal Efforts to
Improve Domestic Manufacturing, GAO-23-106394, 2023.
Public-sector context; consult the report for its
scope. [GAO
search](https://www.gao.gov/search?search_api_fulltext=semiconductor%20supply%20chain).

## 17. Links

- [Module 14 index](_index.md)
- [Source-to-pay](01_source-to-pay.md)
- [Spend cube construction](02_spend-cube.md)
- [Module 14 survey](../../../course/14_procurement_supplier_risk_spend.md)
- [Module expansion specification](../../../course/MODULE_EXPANSION_SPEC.md)

