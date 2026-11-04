# Spend Cube Construction

> Module 14 — Procurement, Supplier Risk, and Spend Analytics
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 14 topic 01 and basic accounting
> Version 2026.5

## 1. Why this topic exists

A category manager is preparing a packaging tender.
The enterprise resource planning (ERP) system shows
dozens of vendor names, while finance reports one
expense total and accounts payable reports a
different invoice total. A plant manager insists
that a major supplier is absent. Negotiating before
those differences are understood can create a false
baseline, overlook a contract, or double-count the
same purchase.

A spend cube is a controlled analytical view of
purchasing expenditure. Its central dimensions are
supplier, category, business unit, and time. Its
measure is a defined monetary amount at a declared
transaction grain. It is not merely a dashboard with
filters: the useful product is an explainable chain
from source accounting records to a decision-ready
population.

The chief financial officer needs confidence that
the scope reconciles to the general ledger (GL).
Procurement needs to see the commercial supplier and
the purchased product or service. Business units
need to recognize their own activity. Those needs
differ: a GL account represents accounting
treatment, a vendor number represents a payee in one
system, and a supplier parent represents a
relationship for a particular purpose. None alone
answers every question.

For example, a single corporate group can sell both
corrugated packaging and maintenance services
through several legal entities. Grouping the
entities reveals possible bargaining leverage;
assigning all of their spend to one category
destroys the distinction between different markets.
Conversely, splitting each entity by spelling hides
common ownership. A reliable cube keeps legal
entity, commercial parent, and category as separate
dimensions.

The business decision determines the boundary. A
sourcing team may want invoiced third-party
operating spend excluding recoverable tax. Finance
may report expense including payroll and
depreciation, while a capital-investment team needs
fixed-asset purchases. All three can be legitimate,
but they cannot be compared without a bridge.

This lesson builds an invoice-based cube for
sourcing, then reconciles it to a GL control total.
It covers invoice and accounting timing, reversals,
currency, supplier resolution, taxonomy, allocation,
hierarchy changes, and release controls. It does not
imply that historical spend is addressable,
contract-compliant, or an achievable saving. These
are separate investigations.

The case and numerical examples are hypothetical.
Amounts are in US dollars (USD), and the accounting
period is calendar year 2025 unless stated
otherwise. Examples are educational, not accounting,
legal, tax, or procurement advice.

## 2. Learning outcomes

After this lesson you can:

1. Define a sourcing decision, population, grain, sign, period, currency, and exclusions in a cube charter; exercise 1 tests the charter.
2. Construct a signed bridge from GL expense to the invoice-based cube without treating legitimate timing differences as data defects; exercise 2 tests the arithmetic.
3. Resolve vendor records into legal entities and commercial parents while preserving source identifiers and effective dates; exercise 3 tests the hierarchy.
4. Assign a category at line level using evidence and retain an explicit unknown class; exercise 4 tests classification and coverage.
5. Calculate concentration, category mix, business-unit shares, and a bounded opportunity without claiming those numbers are realized savings; exercises 5 and 6 test interpretation.
6. Design controls for duplicates, credits, exchange rates, and allocation; exercise 7 tests release decisions.
7. Run and interpret a small reproducible calculation and explain its assumptions; exercise 8 tests code and reconciliation.
8. Specify a governed publication process with ownership, exception handling, and versioned restatement; exercise 9 tests the operating design.

## 3. The core mental model

A cube is a set of additive facts with descriptive
dimensions, surrounded by controls. The word cube
describes multidimensional analysis; it does not
require a particular database product. The same
facts can be stored in a relational table, a
columnar warehouse, or a spreadsheet for a small
controlled example.

```text
GL balances and journals ----------------------> control total
                                                       |
Invoice lines --> normalize --> deduplicate --> classify --> cube
       |              |             |              |          |
       |         FX and sign   source identity   category     +--> supplier
       |                                                  +--> business unit
       +--> source evidence                               +--> period
                    |                                     +--> drill-through
Supplier master --> legal entity --> parent mapping        |
                                                       bridge
                                              GL +/- scope/timing
                                                  = cube total
```

Start with a fact table: one row for each source
invoice line, or for each allocated portion if the
business unit split creates multiple facts. The fact
key includes the source system, invoice identifier,
line identifier, and allocation sequence. A credit
note is a separate signed fact linked to its
original where possible. The fact is not the
purchase order (PO), receipt, payment, or GL journal
as well: joining all four as if they were
independent expenditures multiplies the amount.

Dimensions describe the fact. Supplier dimension
answers who supplied or billed. Category dimension
answers what was bought. Business-unit dimension
answers who consumes or owns the charge. Time
dimension answers when the amount belongs under the
selected date convention. Geography, contract,
payment terms, and source system can extend the view
without changing the basic principle.

A measure is additive only under a stated rule. Net
invoice amount in a single reporting currency can be
summed over supplier, category, business unit, and
time, provided each fact appears exactly once and
allocation weights sum to one. Distinct supplier
count is not additive across categories because one
supplier can appear in several categories.
Percentages and average prices also require
recomputation from their numerators and
denominators.

The reconciliation is a bridge, not a demand that
unrelated measures match by coincidence. Starting
with GL expense, remove out-of-scope accounting
items, add invoice amounts not yet expensed, and
adjust any other explicitly identified difference.
Each bridge item has a sign, owner, evidence, and
mutually exclusive population. The remaining
unexplained difference is a control failure or a
documented exception, not a plug.

Drill-down is a business capability. A user should
be able to move from a parent-supplier total to
legal entity, source vendor, document, and line,
subject to permissions. The reverse roll-up must
reproduce the published total. A figure that cannot
be traced to evidence is an opinion, even when it is
displayed to two decimal places.

Finally, the cube is a versioned snapshot. Late
invoices, corrected mappings, and ownership changes
can change history. Publish the refresh date and
rule version, distinguish a restatement from a newly
closed period, and keep enough lineage to reproduce
the answer used in the sourcing decision.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Spend cube | Multidimensional view of a defined purchasing measure, usually currency per period | Treating all company cost as sourcing spend |
| Fact grain | Smallest represented event, here an invoice line or allocated part | Mixing document totals with line totals |
| Dimension | Attribute used to slice a fact, such as supplier or business unit | Treating a dimension as a monetary measure |
| Measure | Numeric quantity with defined units, sign, and aggregation | Summing percentages across groups |
| GL | General ledger, the accounting system of record for posted balances | Assuming every GL expense has an invoice |
| Accounts payable (AP) | Payables process and subledger for supplier obligations | Equating invoice totals to cash paid |
| Invoice | Supplier billing document with dated, priced lines | Counting the invoice and PO as two purchases |
| Credit note | Document reducing a previous charge, represented with a negative sign | Dropping credits to make spend positive |
| Debit | Accounting entry whose balance effect depends on account type | Assuming every debit is procurement spend |
| Accrual | Accounting estimate for cost recognized before an invoice | Adding both accrual and subsequent invoice without a bridge |
| Prepayment | Payment or invoice recognized as an asset before expense | Calling cash payment current-period consumption |
| Capital expenditure | Spending recorded as an asset under the entity's accounting policy | Assuming it is always excluded from all procurement views |
| Recoverable tax | Tax expected to be reclaimed under applicable rules | Treating tax status as identical in all jurisdictions |
| Net amount | Amount after defined credits and exclusions, expressed in currency | Assuming net automatically means tax-exclusive |
| Source key | System-scoped document and line identifier | Using an invoice number alone across all systems |
| Duplicate | Two records representing the same underlying economic event in the same measure | Deleting a legitimate installment or repeated line |
| Legal entity | Registered party that enters an obligation | Equating it automatically to a group parent |
| Vendor record | System-specific payable-party master record | Assuming one vendor number means one global company |
| Parent | Controlling or commercially relevant grouping at a stated date | Treating shared names as proof of ownership |
| Taxonomy | Controlled hierarchy for classifying goods and services | Assuming a supplier has exactly one category |
| UNSPSC | United Nations Standard Products and Services Code classification | Assuming a code itself proves the purchased item |
| GPC | GS1 Global Product Classification for classifying products | Treating it as a universal procurement services taxonomy |
| Classification confidence | Evidence-based measure or label for assignment certainty | Interpreting an uncalibrated model score as probability |
| Unknown | Explicit bucket for missing or unresolved dimension values | Hiding incomplete data in a plausible category |
| Business unit (BU) | Organizational owner of a charge under a defined hierarchy | Confusing consuming unit with paying legal entity |
| Allocation | Distribution of a source amount using weights summing to one | Joining a cost center to multiple BUs without weights |
| Exchange rate | Reporting-currency units per local-currency unit for a date and rate type | Multiplying by a rate quoted in the reverse direction |
| Coverage | Eligible amount or count with a valid attribute divided by its defined population | Using row coverage as if it were spend coverage |
| Reconciliation bridge | Signed, evidence-backed explanation of differences between totals | Forcing agreement with an unexplained adjustment |
| Concentration | A named supplier or set of suppliers' share of a defined denominator | Calling share alone a complete risk assessment |
| Addressable spend | Portion plausibly subject to a defined sourcing action | Calling all historical spend negotiable |
| Baseline | Documented comparison amount and conditions for a sourcing decision | Treating it as a budget or contract commitment |
| Restatement | Reissued historical view after a controlled change | Silently rewriting a previously published result |
| Lineage | Trace from reported value to transformations and source records | Keeping only an aggregate dashboard export |

## 5. The mechanics, step by step

### 5.1 Commission the decision and charter

Input: decision request, reporting period, finance
policy, and source inventory. Output: a signed cube
charter and acceptance criteria. Ask what action the
owner expects to take, such as selecting packaging
categories for a competitive event. A general
request for every supplier fact should not be
allowed to leave the measure undefined.

Write the population as a predicate. For this case
it is third-party posted invoice lines with a 2025
posting date, excluding recoverable tax, payroll,
and depreciation, including credits and capital
invoices. Including capital invoices is a choice for
this sourcing view, not an accounting claim that
they are expense. A different capital-expenditure
decision might use a separate measure and a
different scope.

Specify the reporting currency, conversion date,
rate source, and sign convention. Record whether the
business unit reflects the original posting or a
current hierarchy. Specify whether the supplier
parent is as-of invoice date or as-of reporting
date. Define an owner for each judgment and a
release tolerance approved by finance. A tolerance
is a decision rule, not permission to conceal a
known difference.

Branch: if the decision is cash management, use
payment facts and cash dates, not the invoice-based
sourcing measure. If the decision is financial
statement expense, use the GL basis and accounting
policy, not a cube designed around invoiced
purchases. Publish different measures separately
rather than forcing them into one label.

### 5.2 Inventory systems and extract controls

Input: ERP and AP source inventories, chart of
accounts, mapping tables, and source control totals.
Output: immutable extracts with source system,
extraction time, record counts, signed amounts, and
period parameters. Include all systems within the
charter, including acquired business units if in
scope.

For each system record the document statuses that
represent posted economic events. Identify
cancellations, reversal documents, tax lines,
freight lines, intercompany accounts, and document
currency. Capture both posting and invoice dates.
Extract identifiers as text where leading zeros
matter. Preserve original amounts and currencies
rather than overwriting them during conversion.

Obtain GL totals at a compatible level: entity,
account, accounting period, ledger, and currency.
Record whether the GL has already been closed or can
receive late journals. Obtain a trial balance or an
equivalent finance-approved control extract. The
invoice extract and GL extract need consistent
cutoff timestamps, or a separately documented cutoff
bridge.

Branch: where AP lines do not carry business unit or
category evidence, recover it from coding
distributions, PO lines, or a controlled allocation.
If no defensible source exists, retain unknown.
Never invent missing detail merely to complete a
chart.

### 5.3 Normalize grain, sign, and status

Input: raw document lines and posting rules. Output:
a normalized fact staging table. Declare a
uniqueness key such as source system plus company
code plus fiscal year plus document number plus line
number. Invoice numbers alone are not globally
unique, and a single invoice can legitimately have
many lines.

Normalize sign so charges increase spend and credits
reduce it. Accounting systems may store debits and
credits in a separate indicator rather than a signed
value. Test the transformation using a known charge,
a known credit, and a cancellation. Preserve both
the original sign and the normalized sign with a
transformation version.

Identify a duplicate by business evidence, not only
equal amounts. Compare supplier legal entity,
document number, currency, invoice date, tax
identity, and document type, then inspect candidate
pairs. A repeated amount can be a valid monthly
subscription, a partial shipment, or a second legal
entity. A source-system replay with the same
immutable source key is normally an ingestion
duplicate; quarantine it and retain the audit
record.

Reversals require temporal care. If the original and
reversal fall in the period, their net effect may be
zero. If the reversal occurs in the next period,
remove neither from the earlier published
posting-period view without an explicit restatement
policy. A reissued invoice is a new fact linked to
the canceled document, not a reason to drop both.

### 5.4 Define the monetary basis

Input: normalized facts, tax treatment, rate table,
and accounting policy. Output: a signed,
consistently converted measure and excluded-amount
ledger. Separate gross amount, tax, freight,
discounts, and credit where the source supports it.
State whether freight is included in item category
or assigned to logistics; either choice requires one
and only one allocation.

For each local-currency line, multiply the eligible
signed amount by the approved
reporting-currency-per-local-currency rate. Use the
date and rate type in the charter. Do not use the
current spot rate for some records and historical
rates for others. If the required rate is missing,
quarantine the fact and report its local-currency
exposure; do not substitute one without an approved
rule.

Rounding can create small differences. Prefer
retaining sufficient precision through aggregation
and rounding for display. If source postings are
already rounded per line, reconcile against those
posted values. Document whether the rate itself or
converted amount is rounded and whether finance uses
a different translation method.

A rebate may be a supplier credit, a year-end
journal, or an off-invoice settlement. Include it
when the charter and evidence allow attribution. If
it is only in the GL, show it explicitly in the
bridge rather than allocating it arbitrarily to
invoice lines. Amounts with unknown tax status need
an exception, not an assumption that every tax
charge is recoverable.

### 5.5 Build the supplier identity chain

Input: vendor master, legal identifiers, source
documents, and a dated ownership reference. Output:
a crosswalk from source vendor record to legal
entity and, separately, to a commercial parent.
Retain source system and company code with the
vendor identifier. Two systems may both have vendor
1000 but refer to entirely different companies.

Start with authoritative unique identifiers where
available, such as a verified legal registration or
tax identifier. Normalize names and addresses for
candidate generation, not as conclusive proof.
Compare country, registration number, bank details
under restricted access, contracts, and invoices.
Bank-account similarity can be useful evidence but
is sensitive and does not by itself prove common
ownership.

A supplier may have multiple legal entities, and a
legal entity may have multiple source records.
Preserve the many-to-one links explicitly. A
marketplace or payment processor may be the payee
while another firm supplies the goods. Carry payee,
contracting party, and ultimate supplier as distinct
roles where the decision requires them. If the
underlying seller is unknown, do not attribute it to
a fabricated supplier.

Parent hierarchies change with acquisition,
divestiture, and restructuring. A historical
as-of-transaction view reflects ownership at
purchase date. A current-parent view estimates the
current negotiating relationship. Both can be
useful, but mixing them inside one table makes time
comparisons misleading. Store effective start and
end dates, source, reviewer, and mapping version;
prevent overlapping valid parent assignments for a
single role.

Branch: if evidence is ambiguous, keep a unique
unresolved entity and a review task. Do not merge
two firms because their names resemble each other.
An unresolved high-value supplier should be
prioritized over a low-value spelling issue, but
neither should disappear from the cube.

### 5.6 Classify the purchase, not merely the payee

Input: invoice line text, item master, PO, contract,
account, and taxonomy version. Output: category
assignment with evidence, confidence, and method.
Define category levels and mutually exclusive leaf
categories. The hierarchy should support the
decisions procurement makes while allowing a mapping
to external codes where appropriate.

Use the strongest available evidence first: a
verified item or service code, contract line, or PO
line; then a reviewed supplier-and-context rule;
then a descriptive-text classifier; finally an
explicit unknown. A supplier default is dangerous
when the supplier sells across categories. A GL
account may help separate expense types but often
mixes categories or applies accounting treatment
rather than market classification.

Create rules with effective dates and precedence. If
a reviewed contract identifies a line as corrugated
boxes, that evidence should normally outrank a broad
vendor default. When evidence conflicts, route the
line to review and retain both the input evidence
and the chosen reason. Rules must not overlap
silently; test known examples, boundary cases, and
nulls.

Measure coverage by absolute amount as well as line
count. Signed net coverage can become unstable when
credits offset charges. Show the unclassified
absolute amount, the classification method, and the
distribution of confidence. Review high-value and
high-risk unknowns first. Independently sample
auto-classified lines and measure accuracy by value
and category, not only by count.

A category hierarchy can change. Store the taxonomy
version and crosswalk old to new categories where
possible. If one old category splits into two, the
mapping needs line-level evidence; do not invent a
precise historical split. Publish a comparable
restated series only when the data support it.

### 5.7 Map business units and allocate shared costs

Input: cost-center coding distributions,
organization hierarchy, and allocation basis.
Output: one or more BU facts per source line with
weights and lineage. Distinguish the paying legal
entity from the consuming unit and state which one
appears in the cube. A central purchasing team may
pay an invoice used by multiple plants.

When source accounting already contains a defensible
distribution, use those amounts rather than
duplicating the entire invoice for each BU.
Otherwise, obtain an approved allocation driver such
as units consumed, headcount, or floor area,
selected for the cost being allocated.
Effective-date the driver and record the approval. A
software license allocated by headcount is an
estimate, not a measured count of users.

Require each source line's weights to sum to one
within a documented numeric tolerance. Preserve a
residual as unknown if actual allocation is
incomplete; never normalize away a known missing
share without permission. Check that the sum of
allocated parts equals the original signed amount,
including for credits. If the allocation changes
after a reorganization, retain both the historical
and current-hierarchy views as explicitly different
versions.

### 5.8 Bridge to the GL

Input: independently controlled GL total, in-scope
invoice total, and identified reconciling
populations. Output: signed bridge, supporting
detail, and an unexplained residual. Agree the
direction before calculating: in this lesson the
bridge starts with GL expense and reaches invoice
spend. Therefore remove GL-only amounts and add
invoice-only amounts.

First align entity, period, ledger, currency, and
cutoff. Then identify mutually exclusive
differences: payroll and depreciation within expense
but outside sourcing scope; year-end accruals
included in GL but not yet invoiced; capital
invoices included in the cube but posted to an asset
rather than expense; and other approved differences
if present. Each item should be traceable to
document or journal populations.

The residual equals cube total minus the GL starting
total minus the sum of signed bridge adjustments. A
zero residual is necessary for this example but is
not proof that classification or supplier mapping is
correct. Two offsetting mistakes can still net to
zero. Reconcile by entity, month, account family,
and source system to expose offsetting errors, and
examine both positive and negative gross
differences.

Avoid double counting an accrual and the
corresponding invoice. If an accrual was reversed
and replaced by an invoice in the same period, the
GL expense may already equal the invoice amount.
Only the unmatched net accrual at the chosen cutoff
belongs in the bridge. Track matched documents and
reversal dates to substantiate that conclusion.

Investigate residuals rather than adding an
unexplained balancing line. Common causes include
missing systems, document-status filters, rate
differences, wrong period, duplicate facts, tax
treatment, and journals outside AP. Assign an owner
and due date for each open item. If the tolerance is
exceeded, withhold publication or publish only under
an explicitly approved exception.

### 5.9 Publish decision views

Input: reconciled and dimensioned facts, exception
log, and approved access policy. Output: versioned
cube, control sheet, and decision-ready views.
Present total, supplier parent, legal entity,
category, BU, month, and unknown buckets with
visible filter context. Every chart should state the
currency, period, measure, and hierarchy version.

A concentration view divides a supplier parent's
signed net spend by the relevant signed net total,
if that denominator is positive and meaningful. Also
display gross charges and credits when refunds are
material. A small or negative net denominator can
make shares exceed 100 percent or turn negative; use
a separately defined positive-charge view for a
different question and label it.

To identify a sourcing opportunity, intersect
category, supplier, and BU. Quantify fragmented
spend and identify compatible specifications,
contract expiry, demand timing, and sourcing
constraints. A concentration pattern is a lead for
investigation, not a contract renegotiation
instruction. A high share can indicate dependence as
well as negotiating leverage.

Calculate an illustrative opportunity as addressable
baseline times an explicitly hypothetical
improvement rate, then deduct implementation costs
only if modeled consistently. The baseline must
exclude non-addressable contracts, unique
specifications, credits that do not recur, and other
unsupported items. The resulting scenario is neither
a budget reduction nor realized savings.

### 5.10 Operate the cube over time

Input: scheduled source refresh, mapping changes,
and finance controls. Output: release with owner
sign-off and reproducible version. Assign data
engineering to extraction and uniqueness, finance to
reconciliation, master-data stewards to identity,
category managers to taxonomy and decision context,
and privacy/security owners to access.

Keep raw extracts immutable under retention policy.
Version transformations, category mappings, supplier
hierarchy, exchange rates, and reporting cutoff.
Publish a change log that explains material shifts
from source data separately from shifts caused by
remapping. Report the old and new totals by affected
dimension for a restatement.

Monitor missing systems, new vendor records,
unmapped categories, source-to-cube counts,
duplicate candidates, and bridge residuals. A stable
total can conceal deteriorating classification
accuracy, and a small unknown share can conceal a
strategically critical supplier. Choose review
thresholds based on materiality and risk, not a
universal percentage.

A user should be able to request a trace from a
chart cell to its contributing records and
controlled source evidence. Access to
employee-linked invoices, individual names, and bank
information must follow the organization's policy
and applicable law. The analytical supplier identity
can often be displayed without exposing bank account
details.

## 6. Worked example

### 6.1 Charter and data

A fictional manufacturer, Northstar Components,
wants to evaluate its 2025 packaging supplier
strategy. Finance supplies a 2025 expense control of
USD 1,200,000. Procurement builds an invoice-line
cube using posted 2025 invoices, net of recoverable
tax and including capital invoices. All example
amounts are already translated to USD under one
consistent rate policy.

Finance identifies USD 150,000 of payroll and
depreciation in the expense total, and a net USD
50,000 accrual for uninvoiced supplier services at
year-end. Those items appear in the GL expense
control but not in the invoice cube. Invoices for
USD 100,000 of machinery were posted to a capital
asset, so they appear in the cube but not in the
expense control. The identified populations are
mutually exclusive.

| Bridge item | Signed USD | Reason |
|---|---:|---|
| GL 2025 expense control | 1,200,000 | Starting point |
| Remove payroll and depreciation | -150,000 | Out of invoice sourcing scope |
| Remove unmatched net accrual | -50,000 | In GL expense, not yet invoiced |
| Add capital invoices | 100,000 | In cube, not in expense control |
| Expected 2025 invoice cube | 1,100,000 | Reconciled population |

Arithmetic: USD 1,200,000 - USD 150,000 - USD 50,000
+ USD 100,000 = USD 1,100,000. If the extracted cube
instead totaled USD 1,090,000, the residual would be
USD -10,000; it would not be an approved correction
merely because adding USD 10,000 makes the bridge
balance.

### 6.2 Supplier and category mapping

The AP extract contains two source vendor numbers
for the same packaging group. P-101 identifies Atlas
Packaging East LLC and P-102 identifies Atlas
Packaging West LLC. Verified registrations establish
two distinct legal entities, and the ownership
record establishes their common parent Atlas Group
during the relevant period. The cube preserves both
legal entities and rolls them up to the parent for
group-level analysis.

Beacon Boxes Ltd and Cedar Services Inc are separate
legal entities with distinct parents. A fourth
source record lacks sufficient identity evidence and
remains `Unknown supplier`. It is still included in
the total and assigned to machinery on the basis of
a verified item description. Unknown supplier is not
evidence that the purchase was unauthorized.

| ID | Legal supplier or status | Parent | Category | BU | Signed USD |
|---|---|---|---|---|---:|
| 1 | Atlas East | Atlas Group | Packaging | North | 260,000 |
| 2 | Atlas West | Atlas Group | Packaging | South | 180,000 |
| 3 | Beacon Boxes | Beacon | Packaging | North | 140,000 |
| 4 | Beacon Boxes | Beacon | Packaging | South | 90,000 |
| 5 | Cedar Services | Cedar | Maintenance | North | 120,000 |
| 6 | Cedar Services | Cedar | Maintenance | South | 110,000 |
| 7 | Unknown supplier | Unknown | Machinery | South | 100,000 |
| 8 | Atlas East credit | Atlas Group | Packaging | North | -20,000 |
| 9 | Beacon Boxes | Beacon | Packaging | South | 120,000 |

These nine rows are simplified, already aggregated
representations of individual source line
populations. In a production cube each would be
drillable to individual documents and allocations.
Row 8 is a credit; suppressing it would overstate
both the total and Atlas packaging spend by USD
20,000. Row 7 is the capital invoice in the bridge.
There are no other adjustments in this hypothetical
case.

### 6.3 Roll up and check

Atlas net spend is USD 260,000 + USD 180,000 - USD
20,000 = USD 420,000. Beacon net spend is USD
140,000 + USD 90,000 + USD 120,000 = USD 350,000.
Cedar net spend is USD 120,000 + USD 110,000 = USD
230,000. Unknown supplier spend is USD 100,000. The
supplier roll-up is USD 420,000 + USD 350,000 + USD
230,000 + USD 100,000 = USD 1,100,000.

Packaging net spend is USD 420,000 + USD 350,000 =
USD 770,000. Maintenance is USD 230,000 and
machinery is USD 100,000. Category total is
therefore USD 770,000 + USD 230,000 + USD 100,000 =
USD 1,100,000. The same total under different slices
is an important additive check, although it does not
prove the classifications are correct.

North BU spend is USD 260,000 + USD 140,000 + USD
120,000 - USD 20,000 = USD 500,000. South BU spend
is USD 180,000 + USD 90,000 + USD 110,000 + USD
100,000 + USD 120,000 = USD 600,000. North and South
sum to USD 1,100,000. If a shared invoice were
allocated to both units, the allocated pieces rather
than the full invoice would appear in each total.

### 6.4 Interpret the decision

Atlas represents USD 420,000 / USD 1,100,000 =
38.18% of the entire cube and USD 420,000 / USD
770,000 = 54.55% of packaging spend. Beacon
represents USD 350,000 / USD 770,000 = 45.45% of
packaging spend. These two shares use different
denominators from a share of all enterprise spend;
label each chart accordingly.

Packaging accounts for USD 770,000 / USD 1,100,000 =
70.00% of the cube. Unknown-supplier spend is USD
100,000 / USD 1,100,000 = 9.09% of signed net spend.
For a non-canceling coverage measure, absolute
known-supplier amount is USD 1,020,000 and total
absolute amount is USD 1,140,000, so absolute
supplier coverage is 89.47%. The difference between
9.09% net unknown and 10.53% absolute unknown is
driven by the USD 20,000 credit in the known
population.

The packaging group appears fragmented across two
suppliers and two BUs. That suggests an opportunity
to examine specifications, contracts, plant
requirements, and capacity. It does not establish
that the items are interchangeable or that Atlas and
Beacon are qualified alternatives. It also does not
establish that Atlas can handle an additional plant
or that Beacon's existing prices are inferior.

Suppose procurement verifies that USD 500,000 of the
USD 770,000 packaging baseline is contestable after
removing committed or incompatible items. At an
illustrative 3% reduction, the gross scenario is USD
500,000 × 0.03 = USD 15,000 per comparable year. If
documented one-time transition cost is USD 8,000 and
there are no other modeled costs, first-year net
scenario is USD 7,000. This is a scenario, not a
forecast, commitment, or realized saving; the
assumed reduction requires market evidence and the
calculation excludes any unmodeled service or
quality effects.

The next action is to retrieve the contracts and
line specifications, validate the USD 500,000
eligibility, investigate the USD 100,000 unknown
supplier, and request a supplier-capacity
assessment. A manager should not divide USD 15,000
by the entire USD 1,100,000 and describe it as a
verified enterprise-wide saving rate.

## 7. Code, query, or calculation

The following Python 3 example uses only the
standard library. It reproduces the worked example
at the aggregate-row level. The input rows are
presumed already normalized, deduplicated,
classified, converted to USD, and in scope. It does
not perform supplier matching, currency conversion,
or GL extraction. In production those upstream
processes need their own controls.

Amounts are integer USD in this small case, avoiding
binary floating-point rounding in monetary sums.
Percentages are formatted for display only. Real
implementations should preserve source precision and
use decimal arithmetic or integer minor units with
an explicit currency scale.

```python
from collections import defaultdict

# parent, category, business unit, signed USD
rows = [
    ("Atlas Group", "Packaging", "North", 260_000),
    ("Atlas Group", "Packaging", "South", 180_000),
    ("Beacon", "Packaging", "North", 140_000),
    ("Beacon", "Packaging", "South", 90_000),
    ("Cedar", "Maintenance", "North", 120_000),
    ("Cedar", "Maintenance", "South", 110_000),
    ("Unknown", "Machinery", "South", 100_000),
    ("Atlas Group", "Packaging", "North", -20_000),
    ("Beacon", "Packaging", "South", 120_000),
]

def aggregate(index):
    totals = defaultdict(int)
    for row in rows:
        totals[row[index]] += row[3]
    return dict(totals)

supplier = aggregate(0)
category = aggregate(1)
unit = aggregate(2)
cube = sum(row[3] for row in rows)

gl_expense = 1_200_000
payroll_depreciation = 150_000
unmatched_accrual = 50_000
capital_invoices = 100_000
bridge = (gl_expense - payroll_depreciation
          - unmatched_accrual + capital_invoices)
residual = cube - bridge

assert sum(supplier.values()) == cube
assert sum(category.values()) == cube
assert sum(unit.values()) == cube
assert residual == 0

absolute_total = sum(abs(row[3]) for row in rows)
known_absolute = sum(abs(row[3]) for row in rows
                     if row[0] != "Unknown")
print(f"Cube USD: {cube:,}")
print(f"Bridge USD: {bridge:,}; residual USD: {residual:,}")
print(f"Suppliers USD: {supplier}")
print(f"Categories USD: {category}")
print(f"Business units USD: {unit}")
print(f"Atlas packaging share: "
      f"{100 * supplier['Atlas Group'] / category['Packaging']:.2f}%")
print(f"Absolute supplier coverage: "
      f"{100 * known_absolute / absolute_total:.2f}%")
```

Expected output:

```text
Cube USD: 1,100,000
Bridge USD: 1,100,000; residual USD: 0
Suppliers USD: {'Atlas Group': 420000, 'Beacon': 350000, 'Cedar': 230000, 'Unknown': 100000}
Categories USD: {'Packaging': 770000, 'Maintenance': 230000, 'Machinery': 100000}
Business units USD: {'North': 500000, 'South': 600000}
Atlas packaging share: 54.55%
Absolute supplier coverage: 89.47%
```

The assertion checks consistency of the worked data,
not independent correctness. The same rows generate
every dimensional subtotal, so agreement between
them cannot detect a category assigned to the wrong
row. The bridge becomes independent only if finance
supplies the GL control and reconciling items from
controlled sources rather than deriving them from
the cube to force agreement.

The absolute coverage calculation treats the credit
as an absolute USD 20,000 contribution. This avoids
net cancellation but is not a measure of invoice
count or classification accuracy. In a real extract
calculate it from source lines before grouping,
because taking the absolute value of a netted group
loses the gross credit information.

## 8. Data, standards, and regulatory context

### 8.1 Source records and identifiers

The minimum record set is invoice and credit lines,
posting and document dates, source-system keys,
supplier master, business-unit coding, GL account
and balance, and a rate table for foreign-currency
records. Optional but useful sources include PO
lines, receipts, contracts, item master, and
supplier ownership data. They are evidence for
interpretation, not additional spend measures to sum
into the invoice cube.

A stable key protects lineage. Preserve company
code, fiscal year, document number, and line number
where relevant; also preserve the source system
because identifiers are rarely enterprise-global.
Check the source system's actual key design rather
than assuming that this illustrative tuple is
universally sufficient. Retain source document type,
tax basis, and reversal references.

GS1's Global Location Number (GLN) can identify
parties or locations within the GS1 system when
assigned and used correctly. It is useful for
distinguishing locations, but it does not replace an
organization's evidence of legal ownership or prove
that two invoice records are the same legal
supplier. See the GS1 identification-key reference
in section 16.

The Legal Entity Identifier (LEI), administered
through the Global LEI System, is a standardized
identifier for legal entities participating in
financial transactions. Where a supplier has a valid
LEI, it can improve legal-entity linkage. Not every
supplier has one, and an LEI alone does not answer
which commercial relationship is appropriate for a
sourcing decision. Parent relationship data also
need dated interpretation and coverage checks.

### 8.2 Category standards

UNSPSC is a hierarchical classification of products
and services. Its codes can support common reporting
across systems, but code assignment still depends on
evidence about the actual purchase. Version and
licensing conditions matter; an organization should
use the relevant official release and validate its
rights and implementation. A generic description or
a supplier name does not become precise merely
because a code has been filled in.

GS1 GPC groups products for consistent product
classification and data exchange. Its scope and
structure differ from a procurement service
taxonomy. A company can maintain a business-oriented
internal hierarchy and a documented crosswalk to
external codes. Crosswalks can be one-to-many,
version-sensitive, and incomplete; retain unknown or
ambiguous results rather than forcing a one-to-one
mapping.

Neither UNSPSC nor GPC dictates whether a particular
expense belongs in an organization's sourcing
baseline. Scope, expense versus asset treatment,
credits, and tax treatment are separate policy
questions. Similarly, a category code does not
certify supplier capability, safety, or contract
compliance.

### 8.3 Accounting and currency

International Accounting Standard (IAS) 2 addresses
inventories, IAS 16 addresses property, plant and
equipment, and IAS 21 addresses the effects of
changes in foreign exchange rates where
International Financial Reporting Standards apply.
Their official descriptions identify relevant
accounting topics; whether and how a specific
organization applies them depends on its reporting
framework, facts, and professional judgment. The
sourcing cube is not itself a financial statement
and should not replace accounting policy.

The bridge in this lesson adds capital invoices
because the illustrative sourcing scope includes
them. That does not change how finance records those
invoices. It also removes an unmatched expense
accrual because the cube is invoice-based; a
consumption or expense cube could make a different
choice. Avoid assuming the treatment of a real
transaction without the accounting evidence.

Exchange-rate source, quotation direction, date,
rate type, and rounding policy are part of the data
contract. Keep local amount and currency alongside
reporting amount. An analysis using transaction-date
rates answers a different question from one using
constant-currency rates. A constant-currency
comparison can help isolate operational changes but
must be labeled and must not be mistaken for posted
accounting results.

### 8.4 Tax, privacy, and retention

Recoverability of value-added tax or other indirect
tax depends on the transaction and applicable
jurisdiction. Tax codes and a tax owner's validated
treatment are required; this lesson makes no blanket
claim that all input tax is recoverable. Gross and
net measures should remain separately auditable when
source data permit.

Supplier records may contain personal data,
including sole-trader names, email addresses, and
bank details. For entities subject to the EU General
Data Protection Regulation (GDPR), principles in
Article 5 include data minimization and storage
limitation. Applicability and lawful processing
depend on the facts; consult qualified privacy
counsel. An analytical cube usually does not need
broad exposure of bank account numbers.

Define retention and access with the records and
privacy owners, including audit evidence and
deletion schedules. Restrict source-document
drill-through and sensitive fields separately from
aggregate category totals. Log access where policy
requires it. Masking a dashboard is insufficient if
an unrestricted underlying extract remains
available.

### 8.5 Data-quality evidence

Track completeness by source, entity, and period;
uniqueness by source key; validity against
controlled reference tables; and timeliness relative
to the agreed close. Track category and supplier
accuracy through independently reviewed samples. For
value-weighted accuracy, select and evaluate samples
with a stated method and avoid representing a
convenience sample as a statistical confidence
interval.

Keep the original source value, mapped value, rule
version, effective date, and reviewer for each
override. An approved exception is an explicit risk
acceptance with owner and expiration; it is not an
invisible default. Published figures should carry
the source cutoff and the date at which mapping and
hierarchy data were valid.

## 9. Industry practice and case evidence

### 9.1 Public guidance versus hypothetical example

The worked manufacturer is fictional. No company
outcome, supplier saving, or implementation claim
should be inferred from it. The following public
references establish practices and constraints, not
a measured benefit from this particular cube.

The Organisation for Economic Co-operation and
Development (OECD) describes public procurement as a
substantial government economic activity and
publishes procurement indicators and policy
analysis. Its material illustrates why consistent
procurement data and transparent definitions matter
for public decisions. A government purchasing
population and a private manufacturer's invoice
scope differ, so a public-sector statistic cannot be
used as the manufacturer's benchmark without
additional evidence.

The UK Government's guidance on the Commercial
Continuous Improvement Assessment Framework
describes expectations for commercial capability and
data use. It is public-sector guidance, not proof
that a specific private company achieved savings by
using a spend cube. The relevant lesson is to
connect data quality and commercial decisions, while
retaining the organization's own controls and
context.

The U.S. Government Accountability Office (GAO) has
discussed federal strategic sourcing and
opportunities to coordinate purchasing across
agencies. Federal guidance and studies distinguish
agency structures and purchasing authorities; the
mere existence of fragmented spend does not
establish that aggregation is feasible or beneficial
for any particular category. Check the individual
report's date, population, and recommendations
before using a number from it.

### 9.2 How practitioners use the cube

A buyer often starts with a supplier-parent ranking,
then drills into category and contract. The ranking
is useful for identifying account-management
relationships and possible concentration, but legal
entities and contract terms remain necessary for
execution. A parent may not have a single globally
negotiable contract, and an acquisition may make
historical parent attribution sensitive to the as-of
rule.

A category manager starts with category, then
examines suppliers, locations, demand
specifications, and renewal dates. The category
hierarchy should reflect actual substitution
markets. Treating all information-technology
expenditure as one category may hide distinct
software, hardware, cloud, and support markets. A
more detailed hierarchy is useful only if evidence
and stewardship are sufficient to maintain it.

A finance controller starts with the GL bridge and a
cutoff. A controller may accept a documented accrual
difference while rejecting an unsupported difference
from a missing AP system. Procurement and finance
should agree on the measure before comparing
results. The question is not whose number is right
in isolation, but which number answers the decision
and how the other is reconciled.

A supply-risk analyst starts with supplier group and
location, then adds material, site, and sub-tier
information. Spend share alone is a weak proxy for
operational criticality: a low-cost sole-source
component can stop a plant, while a large spend on
readily substitutable items can be less critical.
The cube is an input to risk analysis, not a
complete dependency map.

### 9.3 Decision patterns and counterexamples

When several BUs buy genuinely comparable items from
the same or competing suppliers, a coordinated event
may create a more informed negotiation. Validate
delivery terms, volumes, quality, switching
constraints, and existing commitments first. If
product specifications differ, a single price
comparison can create apparent variance with no
actionable saving.

When a supplier dominates a category, combining
spend may reveal leverage but also reveal
dependency. A buyer should not assume that asking
for a price cut will leave capacity, service, and
resilience unchanged. An alternative-supplier
qualification may have a measurable cost and lead
time that belong in the business case.

When one supplier appears unusually expensive,
examine unit of measure, service bundle, freight,
taxes, timing, and purchased mix before attributing
the gap to price. Total spend divided by quantity is
interpretable only when quantities are comparable
and the numerator covers the same scope. A
price-volume-mix analysis requires richer line
detail than a supplier total.

When unknown spend is concentrated in a few large
records, targeted stewardship can materially improve
visibility. But focusing only on value can miss
low-spend regulated or critical purchases.
Prioritize using both monetary materiality and
business risk, and record why a review queue was
ordered as it was.

### 9.4 Evidence discipline

A documented company announcement is evidence that
the company made a claim, not necessarily
independent verification of a causal outcome. A
benchmark from a vendor can motivate a question but
should not be applied as a saving rate without
understanding population, baseline, and selection
bias. A pilot result needs a comparable
counterfactual and treatment of implementation cost
before it supports a causal claim.

For each case brought to a sourcing committee,
separate observed data, mapping assumptions,
inference, proposed action, and expected outcome. In
the Northstar case the observed hypothetical rows
establish USD 770,000 of packaging spend. The
inference is that supplier and BU fragmentation may
warrant investigation. The proposed action is to
validate specifications and contracts. The USD
15,000 gross opportunity is a conditional scenario,
not an observed outcome.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Mixed grains | Totals rise after PO joins | One invoice matches multiple PO or receipt rows | Spend is duplicated | Aggregate or allocate before joining; test keys |
| Incomplete systems | One entity has implausibly low spend | Missing AP feed or company code | False concentration and missed spend | Reconcile source inventory and control totals |
| Wrong cutoff | Late invoices move unexpectedly | GL and AP extracted at different timestamps | Apparent unexplained residual | Align cutoff or show timing bridge |
| Missing credits | Net spend exceeds supplier statements | Credit types filtered out | Baseline and savings overstated | Include signed credits and trace originals |
| Double-counted accrual | Expense plus invoices both included | Accrual not reversed or matched | Duplicate period spend | Bridge unmatched net accrual only |
| Parent overmerge | Different firms share one name | Fuzzy matching treated as proof | Wrong leverage and risk view | Verify legal identity and ownership |
| Parent oversplit | Many spellings rank separately | No source-to-entity crosswalk | Fragmentation hidden | Steward validated master mappings |
| Supplier category default | One supplier appears only in one category | Vendor name substitutes for line evidence | Category mix is misleading | Use line, item, PO, and contract evidence |
| Missing tax distinction | Gross equals net everywhere | Recoverable and nonrecoverable tax combined | Inconsistent baseline | Use validated tax codes and show basis |
| FX inversion | One currency creates extreme totals | Rate direction reversed | Material monetary distortion | Store quote direction and test known conversion |
| Unweighted BU join | BU totals exceed source amount | Shared costs copied to each BU | Double counting by organization | Allocate with weights summing to one |
| Hidden unknowns | Coverage shows 100% suddenly | Missing values assigned to default class | False confidence | Publish unknown and audit rule changes |
| Hierarchy drift | Past supplier rank changes | Current parent applied without labeling | Misleading trend | Version as-of and current views |
| Scope plug | A large unexplained bridge item | Residual is relabeled adjustment | False reconciliation | Require evidence, owner, and independent review |
| Nonadditive metric | Summed supplier counts exceed total | Same supplier exists in multiple cells | Invalid dashboard totals | Recompute distinct counts at each grain |
| Negative denominator | Shares exceed 100% | Credits outweigh charges in a slice | Misleading concentration | Show signed and gross views with denominator |
| Rebate omission | GL differs at year end | Rebate recorded outside AP | Net spend is overstated | Trace settlement or bridge separately |
| Unsupported opportunity | Forecast saving equals spend times generic rate | Addressability and costs ignored | Unfunded business case | Validate scope, assumptions, and costs |
| Sensitive data leak | Analysts can export bank details | Source access exceeds analytical need | Privacy and fraud exposure | Minimize fields and apply access controls |
| Silent restatement | Historical dashboard changes without notice | Mapping or source overwritten | Prior decisions cannot be reproduced | Version releases and disclose changes |

Each warning is a prompt for investigation, not
proof of error. For example, a legitimate
acquisition can create a real concentration
increase, and a year-end credit can create a
negative supplier period. The control must
distinguish these from data defects using source
evidence.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| All in-scope AP systems are extracted | Establishes population completeness | Acquisition or manual ledger omitted | Compare inventory, entity totals, and GL bridge |
| Posting date defines the period | Makes cutoff reproducible | Decision needs consumption or payment date | Compare dates and build a separate measure |
| Invoice lines represent sourcing activity | Provides a consistent event grain | Material direct journals or rebates outside AP | Quantify and bridge non-AP items |
| Sign conventions are normalized | Makes credits reduce spend | Source reversal indicator interpreted incorrectly | Trace known invoice, credit, and reversal |
| Reporting currency policy is consistent | Makes amounts additive | Missing or inverted exchange rate | Recalculate sampled conversions and rate coverage |
| Supplier parent mapping is evidenced | Makes group analysis meaningful | Similar names or acquisition-date ambiguity | Review legal identifiers and effective dates |
| Category represents the purchased item | Supports sourcing-market analysis | Vendor-only default or vague description | Sample line evidence and quantify unknowns |
| BU allocations are conserved | Prevents inflated organizational totals | Many-to-many joins or missing weights | Test weights and signed amount per line |
| GL and cube controls are independent | Makes reconciliation informative | Bridge derived solely from cube | Inspect origin and finance approval of totals |
| Bridge populations do not overlap | Prevents double subtraction or addition | Accrual also counted as payroll adjustment | Match supporting records and test exclusivity |
| Net shares have a meaningful denominator | Makes percentages interpretable | Large credits or near-zero net spend | Show gross charges and credits separately |
| Historical spend informs a future decision | Supplies a starting baseline | Volume, price, or specifications change | Refresh demand, contracts, and market evidence |
| The credit is properly assigned | Preserves category and supplier totals | Credit refers to multiple original purchases | Trace original and allocate with evidence |
| Published hierarchy version is stable | Allows comparison and reproducibility | Parent or taxonomy overwritten | Retain dated mapping snapshots and restate |

A complete reconciliation does not validate every
dimension. An accurate supplier master does not
establish contract compliance. A well-classified
historical invoice does not forecast the amount the
business will buy next year. Treat these as separate
claims with separate evidence.

## 12. Exercises

### Exercise 1 — Define the measure

A manager requests a 2025 sourcing baseline but
provides a dashboard with 2025 payments, open POs,
and booked expenses. Some payments settle 2024
invoices. Capital purchases and recoverable tax are
not labeled.

Deliverable: a one-page cube charter with grain,
scope, date, currency, sign, exclusions, owner, and
a reason not to sum the three dashboard measures.
Time: 20 minutes. Acceptance: the charter selects
one event basis, names at least two reconciliation
differences, and identifies unknown tax and capital
treatment as decisions requiring an owner.

### Exercise 2 — Reconcile a bridge

For 2025, GL expense is USD 920,000. It includes USD
120,000 payroll and USD 30,000 net unmatched
accruals. In-scope capital invoices of USD 70,000
are excluded from GL expense. The invoice cube
totals USD 835,000.

Deliverable: signed bridge, expected cube total,
residual, and next action. Time: 15 minutes.
Acceptance: use the GL-to-cube direction,
distinguish expected from observed cube, and do not
plug the residual.

### Exercise 3 — Resolve suppliers

Source A vendor 001 is North Peak LLC, registration
R1. Source B vendor 001 is North Peak Ltd,
registration R2. Source B vendor 019 is North Peak
LLC, registration R1. A verified ownership record
shows R1 and R2 had parent North Peak Holdings from
1 July 2025. The earlier parent is unverified. Each
record has USD 40,000 in the second half of 2025.

Deliverable: source-to-legal and legal-to-parent
mappings with an effective-date rule. Time: 20
minutes. Acceptance: distinguish source keys, retain
two legal entities, calculate the second-half parent
total, and leave unsupported earlier ownership
unresolved.

### Exercise 4 — Classify with evidence

Three signed invoice lines are USD 30,000 for
corrugated cartons with a verified PO item, USD
12,000 for equipment maintenance with a verified
contract line, and USD 8,000 with the description
`miscellaneous`; all have the same supplier. A USD
-2,000 credit is linked to the carton line.

Deliverable: line-level category decisions and
absolute classified-spend coverage. Time: 20
minutes. Acceptance: use the item and contract
evidence, retain the unknown, include the credit,
and state the numerator and denominator.

### Exercise 5 — Interpret concentration

Use the Northstar case in section 6. Calculate
Atlas's share of total net cube spend and of net
packaging spend. Explain why the two answers differ
and give one reason the larger share does not prove
a sourcing saving.

Deliverable: two calculations and a two-sentence
decision interpretation. Time: 10 minutes.
Acceptance: use the right denominators and
distinguish observation from opportunity.

### Exercise 6 — Bound an opportunity

A category has USD 800,000 historical net spend.
Validated contracts and specifications show USD
300,000 is not contestable in the planned event. A
hypothetical 4% reduction applies to the remaining
baseline; one-time implementation costs are USD
12,000.

Deliverable: contestable baseline, gross annual
scenario, first-year net scenario, and two caveats.
Time: 15 minutes. Acceptance: do not apply the rate
to non-contestable spend or call the result realized
savings.

### Exercise 7 — Diagnose controls

A source extract has 500 invoice lines totaling USD
250,000. Joining to a receipt table yields 800 rows
totaling USD 390,000. Twenty invoice lines have two
receipts; the rest of the repeated matches arise
from other lines. A reviewer suggests using
`DISTINCT` on supplier and amount to fix the
difference.

Deliverable: diagnosis, two tests, and a safer
design. Time: 20 minutes. Acceptance: identify
many-to-many or one-to-many multiplication, explain
why equal supplier and amount do not identify a
duplicate, and preserve legitimate invoice lines.

### Exercise 8 — Modify the calculation

Using section 7's rows, add a separate USD -10,000
Beacon packaging credit for South. Assume finance
independently confirms that the source and
corresponding GL expense both change by USD -10,000,
with all bridge adjustments unchanged.

Deliverable: revised cube, bridge, residual, and
Beacon packaging total. Time: 15 minutes.
Acceptance: update both independent control values
as stated, preserve the signed credit, and show
arithmetic.

### Exercise 9 — Design a release process

A global company refreshes monthly. Legal-entity
mappings and categories change between releases, and
finance can post late journals. Some invoices
contain personal supplier details.

Deliverable: a release checklist and responsibility
matrix for engineering, finance, master-data
stewards, procurement, and privacy/security. Time:
30 minutes. Acceptance: include source cutoff,
dimensional and GL controls, dated mappings,
exceptions, restatement, access, and a way to
reproduce a prior report. Explain how unresolved
high-value unknowns are escalated.

## 13. Self-check answers

### Answer 1 — Charter

A defensible choice is posted invoice line, 2025
posting date, signed net-of-recoverable-tax amount
in USD, with credits included and capital inclusion
explicitly decided. Preserve source system and
document-line key. Finance owns the GL bridge,
procurement owns the sourcing interpretation, and
tax owns recoverability rules. A different event
basis is acceptable if its purpose and bridge are
coherent.

Do not sum payments, open POs, and expenses: they
are different lifecycle events and may refer to the
same purchase. Payment timing and unpaid invoices
explain differences from invoice totals. Accruals,
depreciation, payroll, and capital treatment explain
possible differences from expense. The choice of
capital scope must be documented rather than
assumed.

### Answer 2 — Bridge

Expected invoice cube = USD 920,000
- USD 120,000 - USD 30,000 + USD
70,000 = USD 840,000. Observed cube is USD 835,000,
so residual = observed minus expected = USD -5,000.
Investigate missing documents, timing, FX,
exclusions, and independent bridge evidence. Do not
add a USD 5,000 plug.

### Answer 3 — Supplier mapping

Source A/001 and source B/019 map to legal entity
R1. Source B/001 maps to R2, despite sharing the
numeric vendor identifier with source A/001. For
transactions from 1 July 2025 onward, both legal
entities map to North Peak Holdings. The second-half
parent total is three records × USD 40,000 = USD
120,000. Earlier ownership remains unknown until
evidence supports it. The mapping must not collapse
R1 and R2 into a single legal entity.

### Answer 4 — Classification

Assign USD 30,000 of cartons and the linked USD
-2,000 credit to packaging. Assign USD 12,000 of
equipment maintenance to maintenance. Assign the USD
8,000 miscellaneous line to unknown pending review.
Absolute classified amount is USD 30,000 + USD 2,000
+ USD 12,000 = USD 44,000. Total absolute amount is
USD 52,000, so coverage is 84.62%. Signed net spend
is USD 48,000; it is not the denominator used for
this gross-activity coverage measure.

### Answer 5 — Concentration

Atlas share of the cube is USD 420,000 / USD
1,100,000 = 38.18%. Atlas share of packaging is USD
420,000 / USD 770,000 = 54.55%. The category
denominator is smaller because it excludes
maintenance and machinery. The larger category share
shows concentration, not a proven price reduction;
contract terms and substitutability remain
unverified.

### Answer 6 — Opportunity

Contestable baseline = USD 800,000 - USD 300,000 =
USD 500,000. Gross annual scenario = USD 500,000 ×
4% = USD 20,000. First-year net scenario = USD
20,000 - USD 12,000 = USD 8,000, assuming no other
costs and a full comparable year. The rate is
hypothetical and volumes, quality, transition
timing, and supplier response can change the result.
It is not realized savings.

### Answer 7 — Controls

The receipt join repeats invoice facts when one
invoice line matches multiple receipts. `DISTINCT
supplier, amount` can also collapse two valid
same-value invoices and loses the original line
identity. Compare source keys and amounts before and
after the join; count receipt matches per invoice
line and test that line-level allocations sum to the
original. Aggregate receipt evidence to the invoice
grain, use an existence join, or use a validated
allocation appropriate to the business question.
Keep invoice totals independent of raw receipt
multiplicity.

### Answer 8 — Revised calculation

New cube = USD 1,100,000 - USD 10,000 = USD
1,090,000. The revised independent GL expense is USD
1,190,000. Bridge = USD 1,190,000 - USD 150,000 -
USD 50,000 + USD 100,000 = USD 1,090,000. Residual =
USD 0. Beacon packaging is USD 350,000 - USD 10,000
= USD 340,000. The credit belongs to South as
stated, and the parent mapping should remain
unchanged unless evidence says otherwise.

### Answer 9 — Rubric

A strong design assigns engineering responsibility
for immutable extracts, source-key checks, and
allocation conservation. Finance independently
supplies the GL control, signs off bridge items, and
owns late-journal policy. Master-data stewards
approve dated legal-entity and parent mappings;
category owners approve taxonomy and evidence rules.
Procurement evaluates decision relevance rather than
certifying accounting balances. Privacy and security
owners define permitted fields, access, retention,
and incident handling.

The release should record cutoff, period, currency
and rate version, source counts and sums, unknown
amounts, residual and exception owners, hierarchy
versions, and approval. A material late posting or
mapping correction should trigger a governed
restatement with old and new figures available.
High-value unknowns need a named owner, evidence
request, due date, and explicit publication
decision. Merely saying `refresh monthly` does not
make the result reproducible.

## 14. Interview questions

### 14.1 What makes a spend cube different from a supplier report?

A strong answer names a defined fact grain,
dimensions, additive measure, lineage,
reconciliation, and decision. A weak answer
describes only a ranked supplier chart or assumes
the visualization itself guarantees data quality.

### 14.2 Why does invoice spend differ from GL expense?

A strong answer identifies timing, non-invoice
journals, capital assets, payroll, depreciation,
scope, tax, currency, and cutoff as possible
differences, then constructs a signed bridge. A weak
answer declares either source wrong before
identifying the basis.

### 14.3 How do you avoid double-counting invoices and POs?

A strong answer chooses one event grain and joins PO
information as evidence with controlled cardinality.
A weak answer adds both amounts or assumes
`DISTINCT` always repairs a many-to-many join.

### 14.4 How would you group supplier names?

A strong answer separates source vendor, legal
entity, and dated parent; uses validated
identifiers; and sends ambiguous cases to
stewardship. A weak answer merges on name similarity
alone or assumes a single global vendor number.

### 14.5 Should every invoice from a supplier have the same category?

A strong answer says no, because suppliers can sell
multiple products and services, and uses line-level
evidence with a documented fallback. A weak answer
assumes supplier identity proves the nature of every
purchase.

### 14.6 What does a zero bridge residual prove?

A strong answer says totals agree under the stated
scope and adjustments, but offsetting errors and
incorrect dimensions may remain. A weak answer
treats the result as proof of classification
accuracy or complete source coverage.

### 14.7 How would you measure classification coverage?

A strong answer defines population and denominator,
reports unknowns by count and absolute amount, and
tests accuracy independently. A weak answer reports
a single signed-net percentage without considering
credit cancellation.

### 14.8 What does a high supplier share mean?

A strong answer states the denominator and period,
then discusses concentration, commercial leverage,
dependence, and the need for criticality evidence. A
weak answer equates a high share directly with
either guaranteed savings or certain supplier
failure.

### 14.9 How do you handle a merger halfway through the year?

A strong answer distinguishes transaction-date and
current-parent views, preserves legal entities,
effective-dates the ownership mapping, and labels
any restatement. A weak answer overwrites historical
records and reports the change as organic growth.

### 14.10 What is the difference between baseline and savings?

A strong answer limits the baseline to a comparable,
addressable population and separates hypothetical
rate, costs, forecast, negotiated commitment, and
verified realized result. A weak answer multiplies
total spend by an unsupported percentage and books
it as savings.

## 15. Key terms

- [Spend cube](#3-the-core-mental-model): dimensioned, controlled purchasing facts.
- [Fact grain](#4-definitions-and-vocabulary): smallest represented event.
- [Dimension](#3-the-core-mental-model): descriptive slicing attribute.
- [Measure](#3-the-core-mental-model): numeric value with aggregation rules.
- [GL](#8-data-standards-and-regulatory-context): general ledger accounting record.
- [AP](#4-definitions-and-vocabulary): accounts payable subledger and process.
- [Invoice](#5-the-mechanics-step-by-step): supplier billing event.
- [Credit note](#5-the-mechanics-step-by-step): signed reduction to a charge.
- [Accrual](#6-worked-example): expense recognized before invoice.
- [Capital expenditure](#6-worked-example): expenditure recorded as an asset under policy.
- [Tax basis](#8-data-standards-and-regulatory-context): defined gross or net treatment.
- [Source key](#5-the-mechanics-step-by-step): system-scoped line identity.
- [Duplicate](#10-failure-modes-and-red-flags): repeated representation of one event.
- [Legal entity](#5-the-mechanics-step-by-step): contracting or registered party.
- [Supplier parent](#5-the-mechanics-step-by-step): dated ownership or commercial grouping.
- [Taxonomy](#8-data-standards-and-regulatory-context): controlled category hierarchy.
- [UNSPSC](#8-data-standards-and-regulatory-context): product and service coding system.
- [GPC](#8-data-standards-and-regulatory-context): GS1 product classification.
- [Unknown](#5-the-mechanics-step-by-step): explicit unresolved dimensional value.
- [Business unit](#5-the-mechanics-step-by-step): defined organizational owner.
- [Allocation](#5-the-mechanics-step-by-step): conserved split of a fact.
- [Exchange rate](#8-data-standards-and-regulatory-context): currency conversion factor.
- [Coverage](#7-code-query-or-calculation): classified share with stated denominator.
- [Reconciliation](#6-worked-example): evidence-backed bridge between measures.
- [Residual](#6-worked-example): difference remaining after bridge adjustments.
- [Concentration](#6-worked-example): share held by a named supplier or set.
- [Addressable spend](#6-worked-example): baseline eligible for a specific action.
- [Restatement](#5-the-mechanics-step-by-step): controlled revision of a published view.
- [Lineage](#3-the-core-mental-model): trace to sources and transformation rules.
- [Baseline](#6-worked-example): documented comparison amount, not a saving.

## 16. Sources

1. [IFRS Foundation, IAS 2 Inventories](https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/). Official standard overview; applicability depends on the reporting framework.
2. [IFRS Foundation, IAS 16 Property, Plant and Equipment](https://www.ifrs.org/issued-standards/list-of-standards/ias-16-property-plant-and-equipment/). Official overview of the asset-accounting topic.
3. [IFRS Foundation, IAS 21 The Effects of Changes in Foreign Exchange Rates](https://www.ifrs.org/issued-standards/list-of-standards/ias-21-the-effects-of-changes-in-foreign-exchange-rates/). Official overview of foreign-currency accounting.
4. [GS1, Global Location Number](https://www.gs1.org/standards/id-keys/gln). Official description of the identification key and its use.
5. [GS1, Global Product Classification](https://www.gs1.org/standards/gpc). Official classification-system information.
6. [UNSPSC, United Nations Standard Products and Services Code](https://www.unspsc.org/). Official classification and version information.
7. [Global Legal Entity Identifier Foundation, About the LEI](https://www.gleif.org/en/about-lei/introducing-the-legal-entity-identifier-lei). Official description of legal-entity identifiers.
8. [EUR-Lex, Regulation (EU) 2016/679, Article 5](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Primary legal text; applicability and interpretation require professional advice.
9. [OECD, Public procurement](https://www.oecd.org/en/topics/public-procurement.html). Public-sector procurement context and research.
10. [UK Government, Commercial Continuous
Improvement Assessment
Framework](https://www.gov.uk/government/publications/commercial-continuous-improvement-assessment-framework).
Public-sector commercial capability guidance. 11.
[U.S. Government Accountability Office, Strategic
Sourcing](https://www.gao.gov/strategic-sourcing).
Public-sector oversight and reports; consult
individual report dates and scope.

The standards and public guidance above are
background sources, not evidence of the fictional
case's financial results. The worked numbers are
constructed for teaching. Regulatory references
describe subject matter as of the cited source, not
legal advice or a claim that a particular rule
applies to every reader.

## 17. Links

- [Module 14 index](./_index.md).
- [Module 14 survey and lab sections](../../../course/14_procurement_supplier_risk_spend.md).
- [Module 12 lane and tender analytics](../12_transportation_routing_last_mile/03_lane-tender-analytics.md), for a different commercial event grain.
- [Module expansion specification](../../../course/MODULE_EXPANSION_SPEC.md), for the course template and evidence rules.

The module index lists source-to-pay and
classification taxonomy as adjacent topics. Consult
their linked files when available; their current
status is maintained in the index rather than
inferred here.
