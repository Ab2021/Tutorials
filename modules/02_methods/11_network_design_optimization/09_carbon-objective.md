# Carbon in the Objective — emissions factors, scope boundaries, and multi-objective trade-offs

> Module 11 — Network Design and Optimization
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 topics 02, 04, and 07
> Version 2026.5

---

## 1. Why this topic exists

A network model with a carbon term returns a number in tonnes of carbon dioxide equivalent
(tCO2e). That number will be read by four audiences with four different tests: the sustainability
team, which will ask which boundary it sits inside; finance, which will ask what it cost; the
commercial owner, which will ask whether the service promise survived; and, increasingly, an
external reviewer, which will ask for the factor sources and their vintage. A model that cannot
answer all four has produced a figure, not a result.

The failure is almost never in the optimisation. It is in the accounting that surrounds it. Three
composite cases, each assembled from patterns that recur in practice, show the shape.

**The eighteen per cent that was a factor swap.** A distributor replaces a national average road
factor with carrier-reported emissions data. Reported emissions fall 18% and the network does not
change by a single lane. Nothing was abated: the carrier runs newer vehicles and fuller loads than
the national average, and the previous number described the national fleet, not this company's
freight. The arithmetic is correct and the claim it carries is not, and the first reviewer who asks
"what changed physically" will be told "the provenance of a coefficient".

**The rail shift that raised emissions.** A manufacturer moves a long inbound lane from road to
rail, shortening reported tonne-kilometres because the terminal is closer than the road depot, but
adding a 140 km drayage leg at each end and a transhipment. On a tank-to-wheel (TTW) basis the
switch looks a clear win; on a well-to-wheel (WTW) basis the advantage narrows, and once terminal
handling and the extra warehouse energy are counted the sign can reverse. The mode was chosen on a
factor comparison between two boundaries.

**The cap that cost nothing.** A team sets an emissions cap at the previous year's reported
figure and the model reports compliance at zero cost. The figure had been computed on a different
organisational boundary — three sites under operational control then, four now — so the cap was
looser than it appeared by the emissions of an entire distribution centre. The constraint bound
nothing.

Three things follow, and they organise the topic.

First, **the emissions term is the easiest part of the model and the hardest part of the
argument**. Writing `sum over lanes of tkm x factor` takes one line. Defending what counts as a
tonne-kilometre, which factor, which boundary, and which year takes the rest of the work.

Second, **a network model controls a specific slice of a company's inventory, and it is worth
naming that slice precisely**. It controls the transport and distribution it pays for, the number
and location of sites whose energy it consumes, and the handling and storage steps it chooses. It
does not control supplier processes, product design, or the customer's own logistics. A carbon
claim from a network model that does not state its slice is a claim about the company wearing the
costume of a claim about the network.

Third, **cost and carbon are not two objectives of the same kind**. Cost is money the firm pays,
measured in a currency with a price year. Carbon is a physical quantity with an accounting
convention attached, and the convention is contestable in ways that the currency is not. Treating
them as symmetric terms in a weighted sum hides that asymmetry, and the hidden asymmetry is where
the greenwashing risk lives.

**Relationship to earlier modules.** Topic 02 owns the cost decomposition that the emissions
expression mirrors: every lane and site that carries a cost carries an activity, and the emissions
model is the same graph with a different coefficient per arc. Topic 04 owns the facility-location
model into which the emissions term is inserted. Topic 07 owns the trade-off analysis — the sweep,
the threshold, the scenario matrix — of which the abatement curve in section 5 is a variant with
carbon on the horizontal axis. Topic 08 owns uncertainty, which matters because factors, activity
data, and future carbon prices are uncertain in different ways. Module 12 owns transport
operations, where load factors and empty running are decided; Module 36 carries the reporting
regimes and modal-emissions material at depth and is cited rather than reproduced here.

> **Note:** in this edition Module 30 is the simulation and optimisation lab module, not the
> sustainability module. The reporting regimes referenced here live in Module 36 (modal choice and
> emissions; trade policy, sanctions, and carbon) and Module 37 (circular economy and reverse
> networks). Section 17 links to them; section 8 states only what they constrain in a model.

## 2. Learning outcomes

By the end of this topic you can:

1. **Place** a given emissions source into scope 1, scope 2, or a named scope 3 category, **state**
   whether a network decision can move it, and **distinguish** location-based from market-based
   scope 2 accounting for an unchanged physical network.
2. **Build** an emissions expression for a network model from activity data and factors, and
   **justify** every coefficient with a named source and a vintage.
3. **Compute** freight emissions from tonne-kilometres, mode factors, load factors, and empty
   running, **show** how each of the three changes the answer, and **choose** between a
   well-to-wheel and a tank-to-wheel boundary for a mode comparison.
4. **Formulate** the cost-carbon problem three ways — minimise cost subject to an emissions cap,
   minimise emissions subject to a service floor, and minimise a weighted sum — and **explain**
   why they do not agree.
5. **Construct** a marginal abatement cost curve, **order** measures by cost per tonne, and
   **distinguish** a network-level curve from a site-level one.
6. **Derive** the internal carbon price at which two network options rank equally, and **state**
   that the price is a management instrument rather than a measurement.
7. **Identify** rebound and leakage in a reconfiguration, **explain** why offsets do not enter a
   network model as reduction, and **write** the emissions section to the standard in Step 16.

Each outcome maps to an exercise or to section 6, and to a self-check answer in section 13.

## 3. The core mental model

Hold two pictures. The first is a stack of three scopes with a shaded band marking the part a
network model actually controls. The second is a staircase of measures ordered by cost per tonne,
with a horizontal line across it at the internal carbon price.

```text
   SCOPE 1                    SCOPE 2                    SCOPE 3
   fuel burned in             purchased electricity,     the value chain, both directions
   equipment you own          heat, steam
   or control                                            upstream              downstream
                                                          Cat 1 .. Cat 8        Cat 9 .. Cat 15

   +------------------+       +---------------------+    +---------------------------------+
   | plant boilers    |       | DC electricity      |    | Cat 4  transport & distribution |
   | owned/leased     |       | refrigeration       |    |        PAID BY YOU              |
   |   truck fleet    |       | warehouse lighting  |    |   >>> the network model lives   |
   | warehouse LPG    |       | battery charging    |    |   >>> here: lanes, modes, load  |
   +------------------+       +---------------------+    | Cat 9  transport the customer   |
          |                          |                    |        pays for                 |
          |                          |                    | Cat 1  upstream of your tier 1  |
          v                          v                    +---------------------------------+
   moves only if you           moves with the NUMBER               |
   change fleet ownership      and TYPE of sites                   |
   or on-site equipment                                            v
                                                        moves with every lane, mode,
                                                        consolidation, and load decision
```

Two readings of the picture matter.

**The scopes are accounting categories, not physical categories.** The same diesel is scope 1 if
you own the truck and category 4 if a carrier owns it and you pay the invoice. Nothing in the
atmosphere changes with ownership, which is why "we cut our scope 1 emissions" is compatible with
"total emissions rose": the emissions moved to a supplier and reappeared in category 4.

**A network model mostly moves category 4, and partially scope 2.** Lanes, modes, consolidation,
and distance are its variables; the number and type of sites determine site energy, and the
electricity contract determines the market-based number. Scope 1 moves only at the margin. And the
category 4 boundary is set by who pays, not by who drives: under the Greenhouse Gas (GHG) Protocol,
outbound transport the company pays for is upstream category 4, transport the customer pays for is
downstream category 9, and a model can influence both and report only the one it pays for.

```text
  cost per tonne abated, currency per tCO2e
        ^
   360 -|                                                       +-----------+
        |                                                       |  third DC |
   300 -|                                                       |  + rail   |
        |                                                       |  inbound  |
   240 -|                                                       +-----------+
        |                                   +-----------+
   181 -|- - - - - - - - - - - - - - - - - -|- - - - - -|- - - - - - - - - - -  internal price
        |   +-----------+                   | mode shift|                     €181/tCO2e
   120 -|   |   load    |                   |  one lane |
        |   |  factor   |                   +-----------+
    60 -|   |  recovery |         +-----------+
        |   +-----------+         |  route    |
     0 -|  +-----------+          |optimisation|
        +--+-----------+----------+-----------+-------------------------------->
            100   250    400   550   700   850   1,000   1,200   cumulative tCO2e
                                                                  abated per year

  each block's WIDTH is the tonnes abated by that measure; its HEIGHT is the cost per
  tonne; measures are stacked left to right in ascending height. The horizontal line is
  the internal carbon price: every block whose top sits BELOW the line is worth doing
  under that price, and the blocks above it are not. Negative-cost measures sit below the
  axis and should be taken before any of these, which is why a curve that starts at zero
  is already a filtered curve.
```

Five properties carry the rest of the topic.

**Property 1 — Every emissions number is a product, and every argument is about one of the two
factors.** Activity is physical — tonnes moved, kilometres travelled, megawatt-hours consumed — and
the factor converts it to carbon dioxide equivalent (CO2e). Disagreements that look like
disagreements about modelling are almost always about which activity was counted or which factor
was used, and naming which resolves most meetings.

**Property 2 — The boundary is a choice, and the number is meaningful only inside it.** Two
companies with identical operations can report different totals and both be correct, because one
consolidates under operational control and the other under equity share, or because one reports
market-based scope 2 and the other location-based.

**Property 3 — Cost is monotone in activity; emissions are not necessarily monotone in cost.** A
third distribution centre raises fixed cost and lowers transport distance, and the net effect on
emissions depends on the load factors it destroys and the site energy it adds. The trade-off is a
curve, not a slope.

**Property 4 — The frontier is discovered by solving, and abatement is a portfolio with an order.**
A single weighted-sum run picks one point on the cost-emissions curve, determined by a number
nobody chose in a meeting: trace the curve, then choose. Measures differ in cost per tonne by two
or three orders of magnitude, and ranking them shows how much is available cheaply.

**Property 5 — A tonne avoided in one place can be created in another.** Reconfiguration changes
distances, handling counts, load factors, and asset lives, and some of those changes increase
emissions where the model does not look. Rebound and leakage are properties of the
reconfiguration, not externalities of the analysis.

### 3.1 How the Greenhouse Gas Protocol came to dominate

The GHG Protocol is a multi-stakeholder standard from the World Resources Institute and the World
Business Council for Sustainable Development: the Corporate Standard in 2004, the Corporate Value
Chain (Scope 3) Standard in 2011. It dominates for institutional rather than technical reasons. It
arrived first with a complete architecture, giving companies a way to allocate emissions without
double counting inside a corporate boundary. The disclosure infrastructure adopted it: the Carbon
Disclosure Project (CDP) questionnaire, Science Based Targets initiative (SBTi) criteria, and most
lender and investor frameworks ask for GHG Protocol numbers, so a company reporting on another
basis maintains two systems. And it is compatible with the ISO family rather than competing with
it: ISO 14064-1 covers organisational quantification in a compatible structure, ISO 14083 covers
transport chains using the same activity-times-factor logic, and the GLEC Framework from the Global
Logistics Emissions Council aligns with both and supplies logistics-specific defaults.

The practical consequence for a modeller: you will report in GHG Protocol vocabulary whether or not
you intended to, and the fastest way to make a number unusable is a boundary the protocol does not
recognise.

### 3.2 Why scope 2 has two accounting methods

Electricity is the only major energy carrier where a buyer can contract for a specific generation
attribute while receiving physically indistinguishable electrons, so two questions arise. Where
did the electricity come from on average? That is the **location-based method**: grid-average
factors for a geographic area, answering what the grid did to serve this demand. What did this
company contract for? That is the **market-based method**: contractual instruments — renewable
energy certificates (RECs), guarantees of origin (GOs), power purchase agreements (PPAs),
supplier-specific rates — plus a residual mix that prevents the same attribute being claimed twice,
answering what the company paid for.

Because both questions are legitimate, the Scope 2 Guidance requires dual reporting where the
market offers product- or supplier-specific data, with totals labelled by method. For a network
model the consequence is direct: the same site can carry two intensities, and the choice changes
both the level of site emissions and the marginal value of opening a site. Section 6 shows the
choice moving an implied abatement cost from €181 to €79 per tonne.

> **Note:** the Scope 2 Guidance has been under revision, and a 2026 update to the market-based
> method proposes stricter conditions — hourly matching and deliverability — for a contractual
> claim. Check the current edition before fixing an assumption. The structural point in this topic
> survives any revision: a contractual method and a physical method answer different questions.

### 3.3 Freight emissions: whose tonne-kilometre is it

The most persistent argument in logistics carbon accounting is whether a shipment's emissions belong
to the shipper or the carrier. The GLEC Framework settles it arithmetically tidily and politically
unsatisfyingly: ownership of the vehicle is not decisive, and the reporting organisation's position
in the value chain determines the scope category. The carrier reports the diesel in its scope 1; the
shipper reports the same tonne-kilometres in category 4 if it pays for them. The same physical
emissions are counted twice in the economy and once in each inventory.

Two consequences for a network model. **Double counting across firms is by design**, so a supply
chain total cannot be built by adding participants' inventories. And **the incentive to abate sits
with whoever pays**, which is why carrier selection, load consolidation, and contract terms are
carbon levers. A shipper that switches to a carrier with a lower factor has abated nothing
physically unless the fleet changed; it has moved its own reported number.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding case |
|---|---|---|---|
| Greenhouse gas (GHG) | A gas that traps heat in the atmosphere; the seven covered by the Kyoto Protocol are reported together | Used to mean CO2 only | Gas species |
| CO2 equivalent (CO2e) | A common unit combining gases by their global warming potential | Treated as a measured quantity rather than a converted one | Mass of CO2e |
| tCO2e | One metric tonne of CO2e; the standard reporting unit | Written as "tonnes of carbon", which is a different number by a factor of 3.67 | tCO2e |
| Global warming potential (GWP) | The factor converting a gas to CO2e over a stated horizon | Reported without the assessment report used | Dimensionless, AR5 or AR6 |
| Scope 1 | Direct emissions from sources owned or controlled by the organisation | Assumed to cover all fuel the company influences | Category |
| Scope 2 | Indirect emissions from purchased electricity, heat, steam, or cooling | Reported without stating the method | Category |
| Scope 3 | All other indirect emissions in the value chain, split into 15 categories | Used loosely as "supply chain emissions" | Category 1–15 |
| Upstream / downstream | Toward suppliers and raw material, or toward customers and product use | Confused with earlier or later in the reporting year | Direction |
| Category 4 | Upstream transportation and distribution: transport and storage the reporting company pays for | Assumed to cover all outbound freight | Scope 3 category |
| Category 9 | Downstream transportation and distribution: transport the customer pays for | Omitted because the company has no invoice for it | Scope 3 category |
| Organisational boundary | Which entities' emissions are consolidated, under control or equity share | Treated as fixed across years | Boundary choice |
| Operational boundary | Which scopes and categories are included | Confused with the organisational boundary | Boundary choice |
| Operational control | Consolidation basis where the company operates the asset | Confused with financial control | Consolidation approach |
| Equity share | Consolidation basis proportional to ownership | Assumed to equal control in practice | Consolidation approach |
| Location-based scope 2 | Scope 2 computed on grid-average factors for a geographic area | Reported as "the" scope 2 figure | kgCO2e per MWh |
| Market-based scope 2 | Scope 2 computed from contractual instruments and residual mix | Assumed to be self-declared and unaudited | kgCO2e per MWh |
| Residual mix | The grid factor remaining after contractual claims are removed | Ignored, which double counts green attributes | kgCO2e per MWh |
| REC / GO / PPA | Contractual instruments conveying renewable attributes | Treated as physical delivery of electricity | Certificate |
| Emissions factor | The coefficient converting activity to emissions | Quoted without source, vintage, or boundary | kgCO2e per unit of activity |
| Activity data | The measured physical quantity multiplied by a factor | Assumed to be error-free because it is operational data | tkm, kWh, tonne, litre |
| Tonne-kilometre (tkm) | One tonne moved one kilometre | Computed on loaded distance only, omitting empty running | t·km |
| Load factor | Utilised payload as a share of capacity | Confused with the share of kilometres run loaded | Proportion |
| Empty running | Vehicle-kilometres travelled without payload | Excluded from the factor, which understates intensity | Proportion of vehicle-km |
| Vehicle class | The weight and configuration band of a vehicle | Aggregated away by a single average factor | Class code, e.g. rigid or articulated |
| Well-to-wheel (WTW) | Emissions from energy extraction through to use at the wheel | Assumed to equal tank-to-wheel | Boundary |
| Tank-to-wheel (TTW) | Emissions from fuel combusted in the vehicle only | Used for mode comparisons across different energy carriers | Boundary |
| Well-to-tank (WTT) | The upstream part of WTW: extraction, refining, transport, losses | Omitted when comparing electric and liquid fuels | Boundary |
| Base-year recalculation | Restating the base year after structural change, and restating prior years with it | Recomputed silently after an acquisition, or used to make a target look achievable | Policy |
| Emissions intensity | Emissions per unit of output or activity | Compared across different functional units | gCO2e per tkm or per unit |
| GLEC Framework | Global Logistics Emissions Council framework for logistics emissions, aligned with ISO 14083 | Treated as a regulation rather than a voluntary framework | Framework |
| ISO 14083 | International standard for quantifying transport chain GHG emissions | Assumed to cover building energy as well as transport | Standard |
| ISO 14064-1 | International standard for organisational GHG quantification and reporting | Confused with ISO 14083, which is transport-specific | Standard |
| Distance adjustment factor (DAF) | Multiplier correcting great-circle distance to actual travelled distance | Applied inconsistently across modes | Multiplier |
| Transport operation category (TOC) | GLEC classification of a transport activity for factor lookup | Ignored, so factors are matched to the wrong operation | Category code |
| Marginal abatement cost curve (MACC) | Measures ranked by cost per tonne abated, with width equal to tonnes | Read as a forecast of what will be done | Currency per tCO2e |
| Internal carbon price | A price a company applies to its own emissions in decisions | Described as a measurement of external cost | Currency per tCO2e |
| Shadow price | An internal price used for appraisal without a cash flow | Treated as a budget authority | Currency per tCO2e |
| Abatement cost | Cost of a measure divided by the emissions it avoids | Computed without stating the counterfactual | Currency per tCO2e |
| Rebound effect | Emissions reductions offset by increased activity or demand | Confused with leakage | tCO2e |
| Leakage | Emissions displaced outside the boundary rather than avoided | Ignored when the boundary is the firm | tCO2e |
| Carbon offset | A tradable unit representing a reduction or removal elsewhere | Deducted from a scope total, which the standards do not permit | Certificate |
| Gross emissions | Emissions before any offsets or credits | Confused with net emissions | tCO2e |
| Functional unit | The denominator against which emissions are normalised | Chosen after the result is known | Declared unit |
| Allocation | Division of a shared vehicle's or site's emissions among users | Done by mass when volume is the binding constraint | Method |
| Environmentally extended input-output (EEIO) | Spend-based emissions estimation using sector averages | Used for a decision rather than a screening estimate | kgCO2e per currency unit |

Four of these deserve expansion, because they carry the most weight in argument.

**Activity data versus emissions factor.** A model that reports a change in emissions is reporting
a change in one of these two. The discipline is to decompose every reported change into an activity
effect and a factor effect before presenting it. If the factor effect dominates, the change is an
accounting event; if the activity effect dominates, it is an operational one. Section 5, Step 14
gives the decomposition.

**Load factor versus empty running.** Load factor is how full the vehicle is when it is loaded.
Empty running is how often it is not loaded at all. They multiply, and they are often confused
because both are expressed as proportions. A fleet at 80% load factor with 25% empty running has an
effective intensity of `1/(0.80 x 0.75) = 1.67` times the fully loaded, never-empty rate — a
difference large enough to swamp a mode comparison.

**WTW versus TTW.** The two boundaries answer different questions. TTW answers "what comes out of
the tailpipe or the stack". WTW answers "what does delivering this energy cost the atmosphere".
For liquid fuels the WTT share is roughly a fifth; for electricity it can be most of the total.
A mode comparison conducted on TTW systematically favours electrified modes and understates the
advantage of efficient liquid-fuel modes.

**Internal carbon price versus external carbon price.** An external price is a tax, a permit price,
or a border adjustment: a number the firm pays and cannot choose. An internal price is a number the
firm chooses to apply in its own decisions. It is a management instrument with a policy purpose,
and presenting it as a measurement of social cost misrepresents what it is.

## 5. The mechanics, step by step

This is the procedure for putting emissions into a network model in a form that survives review.
Sixteen steps in four phases. Each step states its input and output, why it matters, and the branch
that sends the work back.

### Phase A — Declare the boundary

**Step 1. Fix the organisational boundary and write it down.**

Input: the entity structure, the site list, the consolidation policy. Output: a one-page boundary
note stating which entities are consolidated, on which basis — operational control or equity
share — and which are excluded. Why it matters: the same physical network reports different totals
under the two bases, and a site run under a management contract is inside the boundary under
operational control and outside it under equity share. If the study proposes closing a site, the
note is what stops a deconsolidation being reported as an emissions reduction. Branch: if a
regulatory regime prescribes a basis, that basis governs the reported number and the model must
match it; for internal studies, match the external report and state any deviation.

**Step 2. Fix the operational boundary: which scopes and which categories.**

Input: the boundary note and a value-chain map. Output: a table of every scope and scope 3 category,
marked included or excluded, with the reason. Why it matters: a network model usually reports scope
1, scope 2, and category 4, and excludes categories 1, 5, 9, 10, 11, and 12 because it does not
control them; the exclusion is legitimate, the silence is not. A model reporting "network emissions"
while including only category 4 is reporting freight emissions, and the label should say so. Branch:
if the decision can shift emissions between categories — outsourcing a warehouse converts scope 2
electricity into a landlord's emissions passed through as a service cost — draw the boundary so the
shift is visible.

**Step 3. Decide the functional unit before building anything.**

Input: the decision the model serves. Output: a stated denominator — tCO2e per year, per tonne
delivered, per pallet, or per order. Why it matters: tonnes per year rewards consolidation; grams
per tonne-kilometre rewards load; grams per unit delivered is the only one a customer can see, and a
design that wins on annual tonnes and loses on grams per order is a different recommendation. Report
an absolute total and an intensity together, because totals move with volume and intensities hide
it. Branch: if the decision will be reported externally, use the unit the reporting regime requires
and state the conversion.

**Step 4. Decide how shared activity is allocated.**

Input: the lane structure and consolidation pattern. Output: an allocation rule for consignments
sharing a vehicle or a site. Why it matters: when a truck carries three customers' pallets, the
diesel must be divided. The GHG Protocol's distance-based method allocates by mass for road, air,
and rail and by volume for marine, on the reasoning that mass binds in one case and cube in the
other. Allocating by mass when cube binds understates a light bulky product's share and overstates
a dense one's, which matters most in the mixed-load networks where consolidation savings come
from. Branch: if densities vary widely, run the allocation both ways; if the ranking changes, the
allocation is part of the decision.

### Phase B — Build the activity and the factors

**Step 5. Define the lane, the mass, and the distance.**

Input: shipment records or the modelled flow matrix. Output: a table of origin, destination, mode,
annual mass in tonnes, and distance in kilometres. Why it matters: distance is the least innocent
field in the dataset. Great-circle distance understates road and rail travel; the GLEC Framework
applies distance adjustment factors to correct for it. A model that uses great-circle distances for
one mode and road-network distances for another has built a mode bias into its activity data before
any factor is applied. Branch: use one distance convention across all options and apply the
adjustment per mode; if the carrier reports distance, check whether it means loaded or total
kilometres.

**Step 6. Convert mass and distance into the activity unit, and fix the load basis.**

Input: the lane table plus payload capacity and utilisation data. Output: tonne-kilometres and the
load factor and empty-running assumptions used.

Why it matters: this is where most freight emissions models quietly decide their answer. A
tonne-kilometre computed as mass times distance counts only loaded movement, and a vehicle also
travels empty and partly loaded. The intensity is `vehicle gCO2e per km / (payload capacity x load
factor x (1 - empty share))`, and every term in that denominator is an assumption an average
published factor has already made for you. The UK government conversion factors, for example, are
derived on a 50% laden basis for heavy goods vehicles, with load-sensitivity corrections from about
8% for the lightest rigid vehicles to about 25% for the heaviest articulated ones. Using the average
factor for a fleet running at 70% load understates its intensity by roughly 40% on that term alone.

> **Note:** published factors are averages over a national or industry fleet. Using one is not an
> error; using one without stating that you have adopted its load, empty-running, and vehicle-class
> assumptions is. The three assumptions are the model.

Branch: if carrier data exists, use it and record its basis. If not, use a default and state the
three assumptions you inherited. A spend-based estimate is acceptable for screening and not for a
decision.

**Step 7. Choose the factor, record its provenance and vintage, and never mix vintages silently.**

Input: the activity table. Output: one factor per activity row, with source, year, boundary, and
geography.

Why it matters: the factor's provenance determines the credibility of the whole result, because the
activity data is the company's own and the factor is always someone else's. The families disagree,
sometimes by a factor of two, for legitimate reasons: different fleets, grid mixes, boundaries, and
vintages.

| Source family | Typical use | Strength | Weakness | Grade |
|---|---|---|---|---|
| National government factors | Company reporting, screening | Documented method, annual vintage, free | National fleet average; load and class assumptions inherited | B |
| Industry framework defaults (GLEC, ISO 14083 tables) | Logistics-specific reporting | Mode and operation-category granularity | Defaults, not your operation | B |
| Life-cycle databases | Upstream and material factors | Cradle-to-gate coverage | Boundary conventions vary by database | A |
| Carrier-reported intensity | Contracted freight | Actual fleet and load | Method undisclosed; not comparable across carriers | C |
| Spend-based EEIO factors | Screening when no activity data | Cheap and complete | Sector averages; no operational signal | C |

Branch: if two sources disagree by more than 30% for the same activity, do not average them.
Choose one and report the other as a sensitivity. An average of two incompatible boundaries is a
third boundary nobody documented.

**Step 8. Fix the energy boundary — WTW or TTW — once, for all modes.**

Input: the mode set. Output: a stated boundary and a factor set consistent with it. Why it matters:
a diesel truck against an electric van on TTW ignores the electricity supply chain, where much of
the electric option's emissions sit. The error is not choosing one boundary; it is choosing one for
one mode and another for the other, which happens when factors are assembled from different sources
without checking. Branch: report the primary result on WTW, because it does not advantage a mode by
omission, and the TTW figure alongside if the company's target is defined on TTW. If the two
disagree about the direction of a change, that disagreement is the finding.

### Phase C — Put it in the model

**Step 9. Write the emissions expression as a linear function of the flow variables.**

Input: the facility-location model from topic 04. Output: an emissions expression whose terms are
indexed by the same arcs and sites as the cost expression.

Why it matters: the emissions model is the cost model with a different coefficient. Every arc that
carries a cost per tonne-kilometre carries an intensity in grams per tonne-kilometre; every site
that carries a fixed cost carries an annual energy consumption. If the emissions expression uses a
different network structure from the cost expression, the two objectives are optimising different
problems.

```text
E = sum over arcs (i,j,m) of  flow_ijm x distance_ij x factor_m / 1e6        [tCO2e per year]
  + sum over open sites j of  site_energy_j x grid_factor / 1e3            [tCO2e per year]
  + sum over hubs h of       throughput_h x handling_factor_h / 1e3        [tCO2e per year]
```

Branch: if the model has inventory and time periods, the holding term has an emissions analogue
through warehouse energy per unit held. Include it or state that you have not; a model that counts
transport and omits the energy of the buildings that hold the inventory is counting the smaller
term in many networks.

**Step 10. Choose the formulation and say which one you chose.**

Input: the cost model and the emissions expression. Output: one of three formulations, stated in
words at the top of the results.

| Formulation | Mathematical shape | What it answers | What it hides |
|---|---|---|---|
| Cost with an emissions cap | Minimise cost subject to `E <= cap` | The cheapest network that meets a stated target | Why the cap is at that level; whether cheaper tonnes existed |
| Emissions with a service floor | Minimise `E` subject to service, capacity, and budget constraints | The lowest-emissions network that still works | The cost of the last tonne, which may be enormous |
| Weighted sum | Minimise `cost + p x E` | The network that is optimal at an internal price `p` | That `p` was chosen by someone, and what the alternatives at other prices look like |

Why it matters: the three formulations select different designs from the same feasible set, and each
is legitimate for a different governance question. A cap fits when an external target exists and
the question is how to meet it at least cost; a service floor fits when the emissions commitment is
fixed and the question is whether the network can still function; a price fits when capital
allocation is delegated and needs a consistent rule across projects. Branch: if the cap is
infeasible, do not raise it silently. An infeasible cap is a finding about the network's physical
limits, and the size of the shortfall is the most useful number in the study.

**Step 11. If you use a weighted sum, derive the price rather than asserting it.**

Input: the policy context and the abatement options. Output: an internal carbon price with a stated
derivation and a stated owner.

There are four defensible derivations, and they give different numbers.

1. **Policy-implied.** The price the company expects to pay: a tax, a permit scheme, or a border
   adjustment. The most defensible for compliance-driven investment, and it requires a stated view
   about future policy.
2. **Target-implied.** The price at which the model's own cheapest abatement portfolio delivers the
   committed target. Derived from the target, not from the world, and it should be labelled so.
3. **Curve-implied.** The marginal cost of the last measure needed to hit the target on the curve —
   the target-implied idea computed from the curve rather than the optimiser.
4. **Peer-implied.** A price within the range other companies report; defensible as a starting
   point, indefensible as a justification, because it is a convention and conventions should be
   named.

Branch: whatever the derivation, state who owns the number and when it will be revised. A price
with no owner is either a technical artefact or an unowned policy, and both are worse than a price
with a named owner and a review date.

**Step 12. Trace the frontier with an epsilon-constraint sweep.**

Input: the model and a range of caps. Output: cost as a function of emissions, at each cap, with
the open set recorded. Why it matters: a single weighted-sum run produces one point, and the
frontier is what the committee needs, because it shows the shape of the trade-off — whether the
first tonnes are cheap and the last ruinous, or whether the curve is flat until a cliff. The sweep
is the same machinery as the sensitivity analysis in topic 07, with carbon on the axis instead of
demand.

Procedure: solve with the cap at the current network's emissions, then lower it in steps of, say,
2% until infeasible, recording cost and open set at each step. The steps at which the open set
changes are the network-level abatement opportunities; the slope between steps is the marginal cost
of the tonnes in that interval. Branch: if a step produces infeasibility rather than higher cost,
the network has hit a physical limit at that emissions level. Report the limit and the measures that
would be needed to go below it, even if they lie outside the model's decision space.

### Phase D — Read, test, and report

**Step 13. Build the network abatement curve and compare it with the site curve.**

Input: the frontier, plus discrete measures that the optimiser does not choose. Output: a ranked
list of measures with cost per tonne and tonnes abated.

Why it matters: a site-level curve ranks measures available at one facility — lighting, insulation,
roof solar, chiller upgrade. A network-level curve ranks changes to the shape of the network —
consolidation, mode shift, a site opened or closed — and the two have different cost structures and
owners. A network curve usually starts higher, because a new distribution centre costs millions and
abates hundreds of tonnes, while a site curve has negative-cost measures at the bottom because
retrofits save energy. Presenting a network curve without saying which curve it is invites the
reader to assume the cheap end of the site curve applies. Branch: if the network curve's cheapest
measure costs more than the site curve's most expensive, the network is not where the abatement is,
and the honest recommendation is to say so. Section 6 is exactly this case.

**Step 14. Decompose any claimed change into activity and factor effects.**

Input: last year's and this year's emissions, activity, and factors. Output: a three-part
decomposition — activity, factor, and structure.

```text
change in E = sum (activity_new - activity_old) x factor_old        activity effect
            + sum activity_old x (factor_new - factor_old)          factor effect
            + interaction and structure terms                       residual
```

Why it matters: a reported reduction is only an operational achievement if the activity effect
dominates. In the composite case in section 1, the entire 18% was a factor effect. The
decomposition takes an hour and is the single most effective defence against a greenwashing
accusation, because it separates what the company did from what its suppliers or its government
did.

Branch: if the factor effect is more than a third of the total change, report the change in two
parts and do not aggregate them into one headline. The aggregated number is arithmetically correct
and rhetorically dishonest.

**Step 15. Test rebound, leakage, and boundary shift.**

Input: the recommended design. Output: a short list of ways the reported saving could be smaller or
absent in reality, with an estimate of each. Why it matters: reconfiguration changes behaviour as
well as geometry. Consolidating to fewer, larger sites reduces line-haul distance and can increase
last-mile distance; it also changes the service level, which changes order patterns, which changes
the number of small urgent shipments. Each is a rebound channel, and a saving that is real in the
model and absent in the following year's data is usually one of them. Branch: if the saving depends
on a service level the operations team does not believe it can hold, the saving is not available;
test it with the operations owner before it enters the business case.

**Step 16. Write the emissions section to a fixed standard.**

Input: everything above. Output: a section with the following items, each stated rather than
implied.

| # | Item | What must be stated |
|---:|---|---|
| 1 | Organisational boundary | Consolidation basis, entities included, exclusions |
| 2 | Operational boundary | Scopes and scope 3 categories included, with reasons for exclusions |
| 3 | Scope 2 method | Location-based, market-based, or both, and which one the headline uses |
| 4 | Factor sources and vintage | Named source per activity class, with publication year |
| 5 | Activity data basis | Distance convention, load factor, empty running, vehicle class |
| 6 | Energy boundary | WTW or TTW, applied consistently across modes |
| 7 | Functional unit | The denominator, and at least one absolute total alongside it |
| 8 | Allocation method | Mass, volume, or other, and why |
| 9 | Trade-off treatment | The formulation used, the cap or price, and who chose it |
| 10 | Offsets | Stated separately, with no deduction from the inventory |
| 11 | Rebound and leakage | Named channels, tested or untested |
| 12 | Change decomposition | Activity, factor, and structure effects for any claimed reduction |

Item 12 is the one that separates a study from a claim. A recommendation that reports a lower
number without saying which part of it came from a factor change has not been analysed; it has been
advertised.

### 5.1 Why a cap and a price give different answers

With continuous variables, a cap at emissions level `c` and a price equal to the shadow price of
that cap produce the same design. This equivalence is the reason economists are comfortable with
prices and engineers are comfortable with caps. It fails in three ways in a network model.

1. **Integrality.** Sites are open or closed. The response to a gradually rising price is not a
   gradual reduction; it is a sequence of jumps, with nothing happening across wide price ranges
   and then a whole site changing. A cap, by contrast, can be met by reallocating flows across a
   fixed set of sites over a range of caps. The two instruments therefore do not trace the same
   curve.
2. **Non-convexity of the abatement options.** Measures come in lumps: a rail terminal either
   exists or does not; a fleet is either replaced or not. A price that is too low to justify the
   lump abates nothing, and a small increase above the threshold abates all of it.
3. **The cap is a commitment and the price is a signal.** A cap binds the organisation to an
   outcome and leaves the cost uncertain. A price bounds the cost and leaves the outcome uncertain.
   Which uncertainty the organisation can bear is a governance question, not a modelling one.

> **Note:** the equivalence also fails because a mixed-integer programme has no dual value. Topic 10
> covers the solver behaviour; here, a cap's marginal value must be estimated by re-solving at two
> caps, not read from a dual.

### 5.2 Why a network-level curve differs from a site-level curve

| Property | Site-level curve | Network-level curve |
|---|---|---|
| Decision unit | A measure at one facility | A change to the network's shape |
| Typical measures | Lighting, insulation, chillers, solar | Consolidation, mode shift, site opening or closure |
| Cost structure | Mostly capital, small and divisible | Mostly fixed, lumpy, and large |
| Owner | Facilities or energy manager | Network or operations director |
| Time to effect | Months | Years, with a transition cost |
| Reversibility | High | Low |
| Typical cost per tonne | Often negative to a few hundred | Often several hundred to several thousand |
| Risk of the curve being read as complete | High | High, and worse, because it omits the site measures |

The last row is the practical warning. A network curve drawn alone will show the network as the
expensive route to abatement, which is usually true, and a reader will conclude that nothing should
be done, which is usually false. Draw both, label both, and let the reader see that the cheap
tonnes and the structural tonnes are different objects with different owners.

### 5.3 The activity-data hierarchy, and the rebound checklist

Data quality is a ladder of cost, not of virtue. Level 1 is carrier-reported fuel or energy
allocated to your consignment: the lowest error, and it requires a data-sharing agreement. Level 2
is a carrier-reported intensity in grams per tkm, whose error depends on the carrier's own load
assumptions. Level 3 is modelled activity with your own load data and a default factor, where most
network studies live and adequate for comparing designs. Level 4 replaces your load data with the
factor's defaults; level 5 is spend-based estimation using environmentally extended input-output
(EEIO) factors, which has no operational signal and belongs in screening. State your level; do not
claim level-1 accuracy from level-3 data.

Rebound and leakage are the channels through which a modelled saving disappears, and three
dominate. **Distance rebound:** fewer, larger sites shorten line-haul and lengthen last-mile, so
cost per tonne falls while cost per order rises; measure per order. **Frequency rebound:**
consolidation raises the service bar, so urgent small shipments increase; track the expedited
share. **Load rebound:** cheaper per-unit transport invites lower load factors; monitor load
factor, not only cost. Add modal leakage (rail plus drayage, counted only on the rail leg), boundary
leakage (an outsourced warehouse moving emissions into a supplier's inventory), market leakage
(volume shifting to a competitor), and price leakage (carbon cost passed to customers who leave).
Distance and frequency rebound destroy network business cases because they operate through service
behaviour, which the model does not contain.

## 6. Worked example

Ardenne Home delivers 120,000 tonnes of product a year through two distribution centres and is
considering a third. The instance is small enough to compute by hand and is built so that the
accounting choices, not the arithmetic, decide the recommendation. All figures are illustrative and
internally consistent; costs are in millions of currency units per year at 2026 prices, emissions
in tCO2e per year.

### 6.1 The two options

**Option A — two distribution centres, all road.** Inbound from the plant to centres P and Q;
outbound from each centre to its markets.

| Leg | Mass, t/yr | Distance, km | Activity, M tkm | Road load factor | Empty running |
|---|---:|---:|---:|---:|---:|
| Inbound to P | 60,000 | 400 | 24.0 | 0.70 | 10% |
| Inbound to Q | 60,000 | 500 | 30.0 | 0.70 | 10% |
| Outbound from P | 60,000 | 150 | 9.0 | 0.55 | 12% |
| Outbound from Q | 60,000 | 200 | 12.0 | 0.55 | 12% |
| **Total** | **120,000** | | **75.0** | | |

**Option B — three distribution centres, rail inbound to the third.**

| Leg | Mass, t/yr | Distance, km | Activity, M tkm | Mode | Load factor | Empty running |
|---|---:|---:|---:|---|---:|---:|
| Inbound to P | 40,000 | 400 | 16.0 | road | 0.70 | 10% |
| Inbound to Q | 35,000 | 500 | 17.5 | road | 0.70 | 10% |
| Inbound to R | 45,000 | 700 | 31.5 | rail | average | in factor |
| Outbound from P | 40,000 | 120 | 4.8 | road | 0.55 | 12% |
| Outbound from Q | 35,000 | 150 | 5.25 | road | 0.55 | 12% |
| Outbound from R | 45,000 | 120 | 5.4 | road | 0.55 | 12% |
| **Total** | **120,000** | | **80.45** | | | |

Option B moves 7.3% more tonne-kilometres than A, because the rail terminal is further from the
plant than the road depot. It has a chance of emitting less only because the rail factor is much
lower. This is the "lower factor, longer distance" case in its cleanest form.

### 6.2 Activity to emissions

The road factor is derived from a vehicle intensity of 850 gCO2e per vehicle-kilometre on a
well-to-wheel basis and a 24-tonne payload capacity: 56.217 gCO2e per tkm inbound at 0.70 load and
10% empty running, 73.175 outbound at 0.55 and 12%. Rail is taken at 25.000 gCO2e per tkm at
average load. Site energy is 1,200 MWh per centre per year at 0.25 tCO2e per MWh, or 300 tCO2e per
centre, and the rail terminal adds 3.0 kgCO2e per tonne handled on 45,000 tonnes, or 135 tCO2e.

| Component | Option A, tCO2e | Option B, tCO2e |
|---|---:|---:|
| Inbound road | 3,035.7 | 1,883.3 |
| Inbound rail | — | 787.5 |
| Outbound road | 1,536.7 | 1,130.5 |
| Site energy (scope 2) | 600.0 | 900.0 |
| Terminal handling | — | 135.0 |
| **Total** | **5,172.4** | **4,836.3** |

Option B emits 336.1 tCO2e less, 6.50%. Its advantage on transport alone is 771.1 tCO2e, and the
third site's energy plus the terminal gives back 435.0 tCO2e, or 56% of the transport gain. A
freight-only model would have reported 14.9%; the difference is the scope 2 and handling boundary,
and it is the first number a reviewer should ask about.

### 6.3 Cost and the cost of a tonne

| Component | Option A, m | Option B, m |
|---|---:|---:|
| Road freight at 0.072 per tkm | 5.400 | 3.524 |
| Rail freight at 0.031 per tkm | — | 0.977 |
| Distribution centre fixed and operating cost | 4.200 | 4.800 |
| Rail terminal | — | 0.360 |
| **Total** | **9.600** | **9.661** |

Option B costs 60,900 more per year and saves 336.1 tCO2e. The cost per tonne abated is
`60,900 / 336.1 = 181.2` currency units per tCO2e. Per tonne of product, A costs 80.00 and B costs
80.51, a difference of 0.6%, while the carbon difference is 6.5%. That asymmetry is the normal case
in network design: the carbon term is a small fraction of a large cost, and a decision that is
decisive for carbon is marginal for money.

Because there are only two options in this instance, the cost per tonne abated is also the internal
carbon price at which the two rank equally. At a price `p`, the total including carbon is
`cost + p x emissions`, so the options tie when `p x 336.1 = 60,900`, which gives `p = 181.2`.

At a price of 100, A costs 10,117,240 and B costs 10,144,530, so A wins by 27,290. At 200, A costs
10,634,480 and B costs 10,628,160, so B wins by 6,320. The switch is at 181.2, and the narrowness of
the margin around it is worth noticing: a 1% error in the third centre's operating cost moves the
switch price by about 2 currency units per tonne.

### 6.4 The same project under four accounting choices

Change nothing physical. Change one accounting choice at a time and recompute the saving and the
implied abatement cost.

| Accounting choice | A, tCO2e | B, tCO2e | Saving, tCO2e | Cost per tonne | Reading |
|---|---:|---:|---:|---:|---|
| WTW factors, location-based scope 2 (base) | 5,172.4 | 4,836.3 | 336.1 | 181.2 | The project needs a high internal price |
| TTW factors, location-based scope 2 | 4,365.5 | 4,178.5 | 187.0 | 325.6 | Rail's advantage shrinks when upstream energy is excluded |
| WTW factors, market-based scope 2 at zero | 4,572.4 | 3,801.3 | 771.1 | 79.0 | The project clears a 100 per tonne internal price |
| WTW, empty running excluded from factors | 4,684.4 | 4,512.3 | 172.1 | 353.9 | Ignoring empty running halves the apparent saving |

The table is the argument of this topic in four rows. Nothing about the trucks, the trains, the
warehouses, or the distances changed. The same capital decision is justified at 79 per tonne and
unjustified at 354 per tonne, depending on which boundary the analyst declared. A study that
reports one row and not the others is not lying, and it is not giving the committee what it needs.

> **Note:** the market-based row assumes the company holds contractual instruments covering all
> site electricity, so the market-based site factor is zero. That is a legitimate position and it
> is also a contractual one: it says nothing about the physical grid, and it would be reversed by
> a change in the contract. The location-based row is the physical statement. The Scope 2 Guidance
> asks for both, and this table shows why.

### 6.5 The three formulations on this instance

- **Minimise cost.** A, at 9.600m and 5,172.4 tCO2e; the carbon term is absent.
- **Cost subject to an emissions cap.** A cap at 5,000 tCO2e makes A infeasible and selects B at
  9.661m. A cap at 4,800 tCO2e makes both infeasible: compliance needs a measure outside the model's
  decision space, such as a lower-intensity carrier on the inbound lanes. That 36 tCO2e gap, 0.7% of
  A's total, is a finding, and the right response is to report it rather than relax the cap.
- **Emissions subject to a service floor.** With a cost ceiling at 9.700m, B is selected; at 9.650m,
  A is selected and the emissions minimum rises to 5,172.4. The ceiling is doing the work, and it is
  a management choice wearing a technical label.
- **Weighted sum.** A at 100 per tonne, B at 200 per tonne; the weight is the entire content of the
  difference.

Two options make the formulations look more distinct than they are with dozens of feasible designs.
What survives at scale is structural: a cap fixes the quantity and prices the cost, a price fixes
the cost and leaves the quantity open, and a service floor changes which designs are admissible
before carbon is considered.

### 6.6 What the number does not say

The example is a defensible comparison of two designs on a stated boundary. It does not say that B
is the right decision, for four reasons arithmetic cannot settle.

- **The internal price is not established.** At the company's actual price the ranking may be
  settled; without one, the study has produced a threshold and left the decision to someone else.
- **The rail factor is a default.** It carries an average load, traction mix, and empty-wagon rate;
  a specific service on a specific corridor could be 40% better or worse.
- **The third site's energy is an assumption.** A modern refrigerated facility and an older ambient
  one differ by more than the entire transport saving here.
- **The service effect is absent.** Three sites are closer to the customer and may reduce urgent
  shipments, and they add a handling point and a day of lead time; neither effect is in the model.

A report that states these four limits alongside the 6.5% is one a reviewer can work with. A report
that states the 6.5% alone will be believed, which is the problem.

## 7. Code, query, or calculation

The calculation below reproduces the section 6 arithmetic with a pencil, ending at the switch price.
It is deliberately not library code: the point is that each line is an assumption a reviewer can
challenge.

```text
road inbound  = 850 / (24 x 0.70 x 0.90) = 56.217 g/tkm
road outbound = 850 / (24 x 0.55 x 0.88) = 73.175 g/tkm      rail = 25.000 g/tkm

A transport = (24.0 + 30.0) x 56.217 + (9.0 + 12.0) x 73.175 = 3,035.7 + 1,536.7 = 4,572.4
A total     = 4,572.4 + 2 x 300                                            = 5,172.4

B transport = (16.0 + 17.5) x 56.217 + 31.5 x 25.000
            + (4.8 + 5.25 + 5.4) x 73.175 = 1,883.3 + 787.5 + 1,130.5     = 3,801.3
B total     = 3,801.3 + 3 x 300 + 135                                      = 4,836.3

saving      = 5,172.4 - 4,836.3 = 336.1 tCO2e          (6.50% of A)
cost delta  = 9,660,900 - 9,600,000 = 60,900 currency/yr
abatement   = 60,900 / 336.1 = 181.2 currency/tCO2e
switch price: p* x 336.1 = 60,900  ->  p* = 181.2 currency/tCO2e

at p = 100: A = 9,600,000 + 517,240 = 10,117,240 ; B = 9,660,900 + 483,630 = 10,144,530
            -> A wins by 27,290
at p = 200: A = 10,634,480 ; B = 10,628,160 -> B wins by 6,320
```

> **Note:** the unit slip to watch: 5,172.4 tCO2e at 100 per tonne is 517,240 currency units, not
> 51,724. Getting it wrong makes every project look unattractive and every cap cheap. At a typical
> internal price the carbon term should be a few per cent of freight cost, not a fraction of one.

## 8. Data, standards, and regulatory context

Emissions accounting is governed by a small number of standards and a large number of conventions.
Knowing which is which tells you what a reviewer will accept.

| Source | Governs | What it constrains in a network model | Grade |
|---|---|---|---|
| GHG Protocol Corporate Standard | Organisational inventory, scopes 1 and 2 | Which entities are consolidated and how scope 2 is reported | B |
| GHG Protocol Corporate Value Chain (Scope 3) Standard | Scope 3 categories 1–15 | The category 4 and 9 boundary, and the calculation methods available | B |
| GHG Protocol Scope 2 Guidance | Purchased electricity accounting | Dual reporting; the location-based and market-based methods; residual mix | B |
| ISO 14064-1:2018 | Organisational quantification and reporting | Structure and disclosure requirements, compatible with the GHG Protocol | B |
| ISO 14083:2023 | Transport chain GHG quantification | The activity-times-intensity method, operation categories, and allocation | B |
| GLEC Framework v3.x | Logistics emissions, aligned with ISO 14083 | Default factors, distance adjustment, hub handling conventions | B |
| UK government GHG conversion factors (DESNZ, published annually) | National conversion factors | Vehicle-class and load assumptions embedded in the freight factors | B |
| EU Corporate Sustainability Reporting Directive (CSRD) and ESRS E1 | Sustainability reporting, including climate | Whether the reported figure must be assured, and on what boundary | B |
| EU Carbon Border Adjustment Mechanism (CBAM) | Embedded emissions in covered imports | A price on embedded carbon that can enter a policy scenario as a cost | B |
| SBTi criteria | Target validation, not accounting | Whether a target is expressed as absolute or intensity, and on which scope | C |

**What these constrain, operationally.** They constrain five things: which entities are inside the
boundary, which categories are reported, which method is used for purchased electricity, how
transport activity is converted to emissions and allocated, and what must be disclosed alongside
the number. They do not constrain the model's formulation, the choice of objective, the internal
price, or the design of the network. Those remain management decisions.

**The offset position is not ambiguous, and it is worth stating flatly.** The GHG Protocol requires
gross reporting: scope 1, 2, and 3 totals are reported independent of any purchase, sale, or
transfer of offsets or allowances, and offsets are disclosed separately as optional information. An
offset therefore does not enter a network model as emissions reduction. It can enter a
target-meeting calculation and a cost comparison, but it cannot change the tonnes the network
produces. A model that subtracts offsets from its emissions constraint is not modelling a network;
it is modelling a purchase.

**What is convention rather than requirement.** The choice of a single headline scope 2 method,
reporting category 4 but not category 9, using a default factor rather than a carrier figure,
reporting intensity rather than absolute tonnes, and the internal carbon price itself are all
conventions or management choices: each is defensible, none is mandated, and the professional
obligation is to say which one you used. None of this is legal, tax, or assurance advice.

## 9. Industry practice and case evidence

**Internal carbon pricing is widespread and heterogeneous.** A 2025 analysis by the World Business
Council for Sustainable Development with BCG reports that 1,753 companies in 56 countries used an
internal carbon price in 2024, an increase of 89% over 2021, with a median of about 49 US dollars
per tCO2e, a range of roughly 10 to 130, a quarter of users below 20, and about two thirds using a
shadow price rather than a charged fee. CDP's 2025 technical note reports more than 5,900 companies
disclosing that they use or expect to use one within two years. **C** — company disclosure and
consultancy analysis, not audited.

Two readings matter for a modeller. The median price is far below the implied abatement cost in
section 6, so most network carbon projects do not clear most internal prices. And the distribution
is wide enough that the choice of price is nearly the whole decision, which is why Step 11 insists
on a derivation and an owner.

**The freight abatement evidence is thinner than the general transport evidence.** A 2025 review of
abatement cost studies in transport found ten studies, none focused solely on freight, with
abatement costs spanning roughly minus 2,065 to plus 2,046 US dollars per tCO2e; logistics
optimisation, modal shift to rail and water, and fleet renewal were among the cheaper options and
often negative in cost. **A** The spread is not bad research; abatement cost is specific to the
corridor, the load, and the counterfactual, so importing a published cost per tonne from another
geography imports an assumption the study cannot defend.

**Mode-specific curves exist and are expensive at the margin.** A 2025 analysis of US freight rail
decarbonisation reports overhead catenary electrification at 444 to 710 US dollars per tCO2e, with
battery-electric locomotives more cost-effective over mid-range distances. **A/B** A 2012 CE Delft
study produced marginal abatement cost curves for heavy-duty vehicles for the European Commission.
**C** Both match the pattern in section 6: structural and asset-based measures sit in the hundreds
to thousands per tonne, operational measures far lower.

**The load assumption inside a published factor is large and documented.** The UK government
conversion factor methodology describes freight factors derived on a 50% laden basis, with
load-sensitivity corrections of about 8% for the lightest rigid vehicles rising to about 25% for the
heaviest articulated combinations, and uplifts of roughly 19% and 16% for refrigerated vehicles. It
reports air freight factors of about 2.53, 0.90, and 0.60 kgCO2e per tkm for domestic, short-haul,
and long-haul respectively, before great-circle and radiative-forcing adjustments. **B**, as
described in the 2024 methodology paper; check the annual vintage before quoting any figure. The
point is not the numbers: a single "road freight" average pools vehicle classes whose intensities
differ severalfold and load conditions that differ by a quarter.

**Carrier data is improving and remains uneven.** Industry guidance describes a hierarchy in which
primary carrier data is preferred and framework defaults are the fallback, with the shipper
responsible for aggregating its carriers' activity. **C/D** In practice many shippers receive an
intensity figure without the method behind it, which makes cross-carrier comparison unreliable and a
"carrier switch" saving hard to distinguish from a reporting change.

| Observed practice | What it looks like | Grade | How to read it |
|---|---|---|---|
| Carbon added as a post-hoc multiplier to an existing cost model | One extra column in the results table | C | Cheap, and adequate for ranking only if the multiplier is uniform |
| Emissions cap set at last year's reported figure | A constraint that never binds | C | Usually a boundary artefact, as in section 1 |
| Internal price set at the corporate target's implied cost | A high price that makes every project look worthwhile | C | A policy statement, not a measurement |
| Network abatement curve presented without the site curve | A curve that starts at 200 per tonne | C | Misleading by omission, not by arithmetic |
| Offsets reported separately | A gross figure plus a note | B | The standard position, and the only defensible one |

**The gap between the standards and the models.** The standards are precise about boundaries and
silent about objectives; most practitioner models are the reverse, casual about the boundary and
precise about the objective, because the objective is what the solver needs and the boundary is
what the organisation finds inconvenient. The correction is a one-page boundary note before the
first solve.

## 10. Failure modes and red flags

| Failure | Red flag you can see | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Boundary moved, not network changed | The saving appears in the first year with no lane or site change | Factor provenance or consolidation basis changed | Reported abatement with no physical cause | Decompose the change into activity, factor, and structure effects |
| Single average factor for all road freight | One row in the factor table, no vehicle class | Averages hide class, load, and empty running | Intensity understated or overstated by a quarter or more | Split by vehicle class and load basis; state the inherited assumptions |
| Mixed vintages or boundaries across modes | Road factors from one year, rail from another, or one basis per mode | Factors assembled from different publications | Ranking driven by vintage or boundary, not by operations | One vintage and one boundary per study, or a stated conversion |
| Scope 2 method unstated | The site emissions line has no method label | Location and market methods give different totals | Site opening or closure valued wrongly | Report both; name the headline method |
| Cap set on last year's number | The constraint never binds, at zero cost | Prior figure built on a different boundary | The most important constraint is inert | Rebuild the baseline on the study's boundary |
| Offsets deducted from the inventory | A net figure below the gross sum | A purchase presented as a reduction | Inventory no longer comparable or compliant | Report gross; list offsets separately |
| Emissions counted for transport only | Freight falls, total rises, and nobody notices | Site energy and handling omitted from the model | The dominant term in some networks is missing | Include scope 2 at sites and hub handling |
| Internal price with no owner | "The model used 100 per tonne" and no name | Price inherited from a template | A policy decision hidden as a parameter | Name the derivation, the owner, and the review date |
| Abatement cost computed against no counterfactual | A euro per tonne with no alternative named | Denominator is the measure's own emissions, not the avoided ones | Comparison across measures is meaningless | State the counterfactual design for every measure |
| Freight emissions reported as company emissions | The report says "our emissions fell 12%" | Category 4 presented as the whole inventory | Claim wider than the boundary | Label the number by scope and category |
| Rebound untested | The saving is real in the model and absent in the data | Service level, order frequency, or load factor responds | Business case fails in year two | Test with the operations owner before committing |

**The boundary shift is the one to watch for, because it looks like success.** A team asked to
reduce emissions and given a model will eventually discover that changing a factor source, a
consolidation basis, or a category inclusion moves the number faster than changing the network.
Sometimes that change is correct, because a better factor is a better estimate. The test is whether
the change can be described in physical terms. "We now use the carrier's measured intensity instead
of the national average, and the difference is 18%" is honest. "We reduced emissions 18%" is not,
and the difference is the whole of the professional obligation.

**The transport-only model is the one to watch for in the arithmetic.** In section 6, counting
freight alone gives a 14.9% saving; counting freight, site energy, and terminal handling gives 6.5%.
Both are correct on their own boundary, and a study reporting the first while claiming the second
has made a labelling error, which is what external reviewers are trained to find.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks if it fails |
|---|---|---|
| The organisational boundary is stable across the comparison | Every option must be counted the same way | A site opening or closing changes the boundary as well as the network |
| Factors are internally consistent across modes | Mode comparison is a ratio of intensities | A boundary mismatch reverses the ranking |
| The activity data measures the same thing in every option | Tonne-kilometres must be comparable | Loaded distance in one option and total distance in the other |
| Load factor and empty running are properties of the design | They are the model's main levers after distance and mode | They respond to service pressure, which the model does not contain |
| Site energy scales with site count | Scope 2 needs a driver | A new site may be far more or less efficient than the average of the existing ones |
| The functional unit is constant across options | Intensity comparisons require a fixed denominator | Volume growth or mix change is attributed to the design |
| The internal carbon price represents a durable policy | Otherwise a one-year price drives a ten-year decision | The price is revised and the design is stranded |
| Abatement measures are independent, against the current network as counterfactual | The curve adds them up and each measure is scored against today | Measures interact, and the baseline is itself a design choice that flatters or penalises every measure |
| Reported change is attributable to the decision | The claim is causal | Factors, suppliers, and grids move for reasons unrelated to the network |

Two limits deserve statement. **The emissions model is only as good as its factors, and the factors
are never yours.** Every published factor describes a population, and your operation is one member
of it; the honest position is to state the population and the distance between your operation and
its average, not to claim the factor is measured. **The model cannot see service behaviour.**
Rebound operates through order frequency, urgency, and load consolidation, none of which is a
decision variable in a facility-location model. Topic 08's uncertainty machinery will not find them
either: they are not uncertain parameters but missing structure.

## 12. Exercises

### Exercise 1 — Place the source (recall, 25 minutes)

For each source, name the scope and category, and state whether a network design decision can move
it: (i) diesel burned by a truck the company owns; (ii) electricity at a leased distribution centre
the company operates; (iii) a carrier's fuel on an inbound lane the company pays for; (iv) a
customer's collection fleet; (v) a supplier's process heat upstream of the tier 1 supplier.
**Deliverable:** a five-row table with scope, category, and a yes, no, or marginal verdict.
**Acceptance criteria:** categories numbered and named; the leased-site case resolved explicitly
against the consolidation basis; one sentence on why (iv) is category 9 and (v) is category 1.

### Exercise 2 — Rebuild the factors (calculation, 40 minutes)

Using the vehicle intensity of 850 gCO2e per vehicle-kilometre, compute the intensity in gCO2e per
tkm for a 24-tonne payload at load factors of 0.45, 0.60, and 0.75, with empty running of 0%, 10%,
and 20% respectively.
**Deliverable:** a three-row table with the formula shown for each row.
**Acceptance criteria:** arithmetic correct to three decimal places; the ratio between the highest
and lowest intensity; one sentence on which assumption a published average factor already made.

### Exercise 3 — The four boundaries (calculation, 45 minutes)

Reproduce the four rows of section 6.4 from the lane tables, changing one accounting choice at a
time.
**Deliverable:** the four-row table with total emissions, saving, and implied cost per tonne.
**Acceptance criteria:** each row's change described in one phrase; the TTW factors stated as
assumptions; the market-based row's contractual condition stated.

### Exercise 4 — Find the switch price (calculation, 30 minutes)

Using the section 6 instance, compute the internal carbon price at which A and B rank equally, then
recompute it if the third centre's operating cost is 3% higher than assumed.
**Deliverable:** both prices, with the algebra.
**Acceptance criteria:** the base answer is 181.2; the sensitivity computed from the changed cost
delta, not by scaling the price; a sentence on how sensitive the conclusion is to a cost line item.

### Exercise 5 — Build the curve (design, 60 minutes)

Construct a five-measure marginal abatement cost curve for a distribution network you know or can
describe, with cost per tonne and tonnes abated for each measure, ordered ascending.
**Deliverable:** a table and an ASCII step chart with a horizontal internal price line.
**Acceptance criteria:** five measures with different owners; at least one measure with a negative
cost; each block's width equal to its tonnes; a stated counterfactual for each measure.

### Exercise 6 — Diagnose the claim (critique, 35 minutes)

A company reports a 22% reduction in distribution emissions. Its volume grew 9%, its network is
unchanged, and it switched from a national average factor to carrier-reported intensities in the
reporting year. Write the decomposition and a 150-word assessment.
**Deliverable:** the activity, factor, and structure decomposition, with the assessment.
**Acceptance criteria:** the factor effect identified as the dominant term; the assessment avoids
both accusation and endorsement; one sentence on what evidence would change the conclusion.

### Exercise 7 — Test for leakage (critique, 40 minutes)

A proposal consolidates four distribution centres into two and shifts one inbound lane to rail. List
the rebound and leakage channels, and for each state a measurement that would detect it.
**Deliverable:** a seven-row table of channel, mechanism, and detection measurement.
**Acceptance criteria:** at least one channel operates through service behaviour; at least one
operates outside the company boundary; each measurement available from operational data, not from a
new study.

### Exercise 8 — Write the section (design, 60 minutes)

Write the emissions section for the section 6 study to the twelve-item standard in section 5,
Step 16.
**Deliverable:** a two-page section with the twelve items as headings.
**Acceptance criteria:** all twelve present; the headline number labelled by scope and category;
the internal price derivation and owner named; offsets stated separately; the change decomposition
included.

## 13. Self-check answers

**Exercise 1.** (i) Scope 1; a network decision moves it only through fleet ownership. (ii) Scope 2;
moved by site count and type, and by the electricity contract. (iii) Scope 3 category 4, because the
company pays for the transport; moved by lane, mode, and carrier choice. (iv) Scope 3 category 9,
because the customer pays; influenced but not controlled. (v) Scope 3 category 1, because it is
upstream of the tier 1 supplier and therefore inside purchased goods and services; not moved by the
network model at all. The leased-site case turns on the consolidation basis: under operational
control the company's electricity is scope 2 even though it does not own the building; under equity
share it may not be consolidated.

**Exercise 2.** `850 / (24 x LF x (1 - e))`. At 0.45 and 0%: `850/10.8 = 78.704`. At 0.60 and 10%:
`850/12.96 = 65.586`. At 0.75 and 20%: `850/14.4 = 59.028`. The ratio of the highest to the lowest is
1.33. A published average factor has already chosen a load factor, an empty-running rate, and a
vehicle class; using it means adopting all three without stating them.

**Exercise 3.** Base: 5,172.4 and 4,836.3, saving 336.1, cost per tonne 181.2. TTW: the road
intensity becomes `700/(24 x 0.70 x 0.90) = 46.296` inbound and `700/(24 x 0.55 x 0.88) = 60.262`
outbound, and rail becomes 21.000, giving 4,365.5 and 4,178.5, saving 187.0, and `60,900/187.0 =
325.6`. Market-based at zero: site energy drops out of both, giving 4,572.4 and 3,801.3, saving
771.1, and 79.0. Excluding empty running: intensities rise to 50.595 and 64.394, giving 4,684.4 and
4,512.3, saving 172.1, and 353.9.

**Exercise 4.** The tie is `p x 336.1 = 60,900`, so `p = 181.2`. A 3% rise in the third centre's
operating cost adds `0.03 x 1.60m = 48,000` to B, so the delta becomes 108,900 and the price becomes
`108,900/336.1 = 324.0`. The conclusion is highly sensitive to one line item in one option's fixed
cost, the normal condition of a close comparison and why topic 07's threshold discipline applies
unchanged.

**Exercise 5.** A defensible set, ascending: route optimisation at about 15 per tonne with 90 t
abated; load-factor recovery at about 40 with 140 t; carrier selection on two lanes at about 90 with
60 t; inbound mode shift at about 180 with 340 t; third distribution centre at about 900 with 100 t.
A negative-cost measure usually exists, often a delivery-window change that improves consolidation
and saves money, and it belongs below the axis. Each measure's counterfactual is the current
network, and each owner differs: transport, procurement, network planning.

**Exercise 6.** Volume growth of 9% raises emissions by about 9% before any factor change. A 22%
reduction with volume growth implies a factor and structure effect of roughly 31% in the other
direction. If the factor switch accounts for most of it, the operational achievement is close to
zero and possibly negative. The assessment should say that the reported reduction is real on the
stated basis, that its cause is a change in estimation rather than in operations, and that the
evidence which would change the conclusion is a like-for-like recomputation of the prior year on
the new factor set. That recomputation is the single control that makes year-on-year claims
meaningful.

**Exercise 7.** Distance rebound: measure outbound kilometres per order before and after. Frequency
rebound: measure the expedited shipment share. Load rebound: measure average payload per vehicle.
Modal leakage: count drayage and terminal handling in the rail option. Boundary leakage: compare
total cost and reported emissions of the outsourced service. Market leakage: measure volume in the
consolidated channel. Price leakage: measure volume response to any cost pass-through. Each is
available from transport management, order, and finance data, and none requires a new study.

**Exercise 8.** A strong section states the operational-control boundary and four sites, the
inclusion of scope 1, scope 2, and category 4, the exclusion of categories 1, 9, and 11 with
reasons, the location-based headline with the market-based figure alongside, factor sources with
their year, the 850 g per vehicle-km intensity and its load basis, the WTW boundary, tCO2e per year
and grams per tonne delivered, mass-based allocation, the cap or price and its owner, offsets
separately, rebound channels tested or untested, and the activity-factor decomposition of the 6.5%.
A weak section reports "a 6.5% reduction" and a total.

## 14. Interview questions

**1. "Which scope does a network model actually control?"**
Strong: mostly category 4 for transport it pays for, partially scope 2 through site count and
energy, marginally scope 1 through fleet ownership, with the reported figure saying which.
Weak: "scope 3", with no category and no distinction between paid-for and customer-paid transport.

**2. "Why does the same network produce two different scope 2 numbers?"**
Strong: location-based measures the grid average, market-based measures the contracts, both are
legitimate, and dual reporting exists because they answer different questions.
Weak: "because companies pick the method that flatters them", which confuses a legitimate dual
disclosure with a choice.

**3. "A supplier offers a factor 20% below the national average. Do you use it?"**
Strong: ask for the method, the load and empty-running basis, and the vehicle class; compare on a
like-for-like boundary; if the method is undisclosed, use it as a sensitivity, not the base case.
Weak: "yes, it is more accurate because it is specific", which is not necessarily true.

**4. "What does a marginal abatement cost curve tell you that a total does not?"**
Strong: the order of measures by cost per tonne, the quantity available at each cost, and the
marginal measure a given price or target selects.
Weak: "how much we can reduce", which is a total, not a curve.

**5. "How would you choose an internal carbon price?"**
Strong: name one of the four derivations — policy-implied, target-implied, curve-implied, or
peer-implied — call it a management instrument, and name its owner and review date.
Weak: quoting a survey number as if it were measured.

**6. "Our emissions fell 15% and volume fell 5%. What do you ask?"**
Strong: for the activity-factor decomposition, the boundary, and the factor vintage; a 15% fall on
5% less volume implies a 10% intensity gain that may be entirely a factor change.
Weak: accepting the number and moving on to the target discussion.

**7. "Can offsets be used to meet the cap in the model?"**
Strong: no; gross reporting is required, offsets are disclosed separately, and a purchase does not
change the tonnes the network produces.
Weak: "yes, if they are verified", which conflates a target instrument with an inventory number.

**8. "What is leakage, and where would you look for it in a network redesign?"**
Strong: emissions displaced outside the boundary or into an uncounted leg — drayage, terminal
handling, an outsourced warehouse — tested by whether the total falls, not just the counted part.
Weak: confusing leakage with rebound, or treating it as an accounting technicality.

**9. "A cap makes our network infeasible. What do you report?"**
Strong: the shortfall in tonnes, the measures outside the model's decision space that could close
it, and the fact that this is a physical limit finding rather than an optimisation failure.
Weak: raising the cap until the model solves and reporting the result as compliant.

**10. "What would make you distrust a reported emissions saving?"**
Strong: a saving with no lane or site change, a factor vintage change in the same year, a
transport-only boundary, or an offsets deduction.
Weak: "nothing, if the model is validated", which mistakes arithmetic validation for accounting
validity.

## 15. Key terms

- **GHG** — greenhouse gas; the seven gases reported together under the Kyoto Protocol.
- **CO2e / tCO2e** — carbon dioxide equivalent, and one tonne of it.
- **Scope 1** — direct emissions from sources the organisation owns or controls.
- **Scope 2** — indirect emissions from purchased electricity, heat, steam, or cooling.
- **Scope 3** — all other value-chain emissions, in fifteen categories.
- **Category 4** — upstream transport and distribution the company pays for.
- **Category 9** — downstream transport the customer pays for.
- **Organisational boundary** — which entities are consolidated, on control or equity basis.
- **Operational control** — consolidation basis where the company operates the asset.
- **Location-based scope 2** — grid-average factor for a geographic area.
- **Market-based scope 2** — contractual instruments plus residual mix.
- **Residual mix** — the grid factor remaining after contractual claims are removed.
- **Emissions factor** — coefficient converting activity to emissions, with a source and vintage.
- **Activity data** — the physical quantity multiplied by a factor.
- **Tonne-kilometre (tkm)** — one tonne moved one kilometre.
- **Load factor** — utilised payload as a share of capacity.
- **Empty running** — vehicle-kilometres travelled without payload.
- **WTW / TTW / WTT** — well-to-wheel, tank-to-wheel, and the upstream well-to-tank part.
- **GLEC Framework** — logistics emissions framework aligned with ISO 14083.
- **ISO 14083** — standard for transport chain GHG quantification.
- **ISO 14064-1** — standard for organisational GHG quantification and reporting.
- **DAF** — distance adjustment factor, correcting great-circle distance to travelled distance.
- **TOC / HOC** — GLEC transport and hub operation categories used for factor lookup.
- **MACC** — marginal abatement cost curve; measures ranked by cost per tonne.
- **Internal carbon price** — a price the firm applies to its own emissions in decisions; a shadow
  price when used in appraisal without a cash flow.
- **Abatement cost** — cost of a measure divided by the emissions it avoids, against a stated
  counterfactual.
- **Rebound effect** — reductions offset by increased activity or demand.
- **Leakage** — emissions displaced outside the boundary rather than avoided.
- **Offset** — a tradable unit for a reduction or removal elsewhere; reported separately, never
  deducted.
- **Functional unit** — the declared denominator for an intensity figure.

## 16. Sources

Grading: A peer-reviewed or audited; B regulatory, official, or standard-setting; C company case,
consultancy, or institutional analysis; D vendor or data provider; E press or anecdote.

**Peer-reviewed and technical**

- Ricci, Gonçalves, and D'Agosto (2025). "Transport Sector GHG Mitigation Measures: Abatement Costs
  Application Review." *Future Transportation*. DOI: 10.3390/futuretransp5040195. Ten studies, none
  freight-only; costs from about −2,065 to +2,046 USD per tCO2. **A**
- Martin, Dimanchev, and Neumann (2023). "Carbon abatement costs for renewable fuels in
  hard-to-abate transport sectors." *Advances in Applied Energy*. DOI: 10.1016/j.adapen.2023.100156.
  Renewable fuels at 793–1,598 EUR per tCO2 in 2020, −11–675 by 2050. **A**
- Hernandez and colleagues (2025). "Abatement Cost Curve Analysis of Freight Rail Decarbonization
  Alternatives." OSTI 2569870. Catenary electrification at 444–710 USD per tCO2; battery
  locomotives more cost-effective over mid-range distances. **A/B**

**Standards and official**

- GHG Protocol, *A Corporate Accounting and Reporting Standard* (revised edition, 2004), including
  the requirement to report scope 1, 2, and 3 independent of GHG trades. **B**
- GHG Protocol, *Corporate Value Chain (Scope 3) Accounting and Reporting Standard*, Chapter 4
  (upstream transportation and distribution) and Chapter 11 (reporting). **B**
- GHG Protocol, *Scope 2 Guidance* (2015), dual reporting requirement and the two methods. **B**
- GHG Protocol, "Scope 2 Market-Based Method Update," October 2026. Proposed hourly matching and
  deliverability conditions; consult the current text before relying on a date. **B**
- ISO 14064-1:2018, *Greenhouse gases — Part 1: Specification with guidance at the organization
  level for quantification and reporting of greenhouse gas emissions and removals*. **B**
- ISO 14083:2023, *Greenhouse gases — Quantification and reporting of greenhouse gas emissions
  arising from transport chain operations*. **B**
- Smart Freight Centre, *GLEC Framework* v3 (2023, with later updates), including the statement that
  ownership of the vehicle is not decisive and that value-chain position determines the category.
  **B**
- UK Department for Energy Security and Net Zero (DESNZ), *2024 Government GHG Conversion Factors
  for Company Reporting: Methodology Paper*, Chapter 6 on freight land transport. Figures in
  section 9 are as described there and should be checked against the current vintage. **B**
- Regulation (EU) 2023/956 establishing the carbon border adjustment mechanism. **B**

**Company, consultancy, and institutional**

- WBCSD and BCG (September 2025). "Integrating climate with financials: Internal Carbon Pricing."
  1,753 companies in 56 countries in 2024; median about 49 USD per tCO2e. **C**
- CDP (2025). *Carbon Pricing Technical Note*, and *Corporate Health Check 2025*. Disclosure counts
  and the share of companies using a shadow price. **C**
- CE Delft (2012). *Marginal abatement cost curves for Heavy Duty Vehicles*, for the European
  Commission. **C**
- Science Based Targets initiative, criteria and validation guidance. Target-setting rules, not an
  accounting standard. **C**

**Vendor and data provider**

- Climatiq, GLEC default factor datasets, with life-cycle boundary tags. Machine-readable and
  useful; the underlying values are the framework's. **D**
- PITT OHIO, "Scope 3 Emissions Frameworks" whitepaper; and Cefic/ECTA emissions calculation
  guidance (2025). Practical statements about shipper aggregation and consignment allocation;
  vendor and industry-body documents. **D**

> **Note on a source deliberately not used.** A search for GLEC factors returned a commercial page
> presenting a "GLEC v3.2, October 2025" factor table whose metadata does not match the official
> framework's publication history. It is not cited. Where a factor matters, read the framework
> document or the government methodology paper, not a reseller's table.

## 17. Links

**This module**

- Module map: [`_index.md`](_index.md)
- The decision ladder this topic serves: [`01_network-decision-ladder.md`](01_network-decision-ladder.md)
- The cost decomposition the emissions expression mirrors: [`02_graph-cost-to-serve.md`](02_graph-cost-to-serve.md)
- MILP anatomy, and why a cap's marginal value must be re-solved: [`03_milp-anatomy.md`](03_milp-anatomy.md)
- The model the emissions term is added to: [`04_facility-location.md`](04_facility-location.md)
- Multi-commodity and multi-period extensions: [`05_multi-commodity-period.md`](05_multi-commodity-period.md)
- Transition and brownfield analysis: [`06_greenfield-brownfield.md`](06_greenfield-brownfield.md)
- The trade-off analysis this curve is a variant of: [`07_scenario-sensitivity.md`](07_scenario-sensitivity.md)
- The formal treatment of uncertainty the emissions model inherits: [`08_robust-stochastic.md`](08_robust-stochastic.md)
- Later, solver behaviour and gap reporting: [`10_solvers-heuristics.md`](10_solvers-heuristics.md)

**Adjacent modules**

- Transport operations, where freight emissions arise: [`../12_transportation_routing_last_mile/_index.md`](../12_transportation_routing_last_mile/_index.md)
- Modal choice and emissions, at depth: [`../../06_domain_electives/36_global_freight_ports_trade_policy/04_modal-choice-emissions.md`](../../06_domain_electives/36_global_freight_ports_trade_policy/04_modal-choice-emissions.md)
- Trade policy, sanctions, and carbon: [`../../06_domain_electives/36_global_freight_ports_trade_policy/09_trade-policy-sanctions-carbon.md`](../../06_domain_electives/36_global_freight_ports_trade_policy/09_trade-policy-sanctions-carbon.md)
- Circular economy and reverse networks: [`../../06_domain_electives/37_circular_economy_reverse_networks/_index.md`](../../06_domain_electives/37_circular_economy_reverse_networks/_index.md)

**Programme navigation**

- Part II methods map: [`../_index.md`](../_index.md)
- Part I foundations: [`../../01_foundations/_index.md`](../../01_foundations/_index.md)
- Module survey file: [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)

> **Note:** all ten topic files in this module are now written. Every markdown link on this page
> resolves to a file that exists on disk.
