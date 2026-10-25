# Dimensional Analysis

> Module 01A — Foundations, Assumptions, and Units
> Topic 04 of 10
> Estimated study time: 3 hours
> Prerequisites: 01, 03
> Version 2026.3

## 1. Why this topic exists

Every equation carries a hidden contract: the units in which its variables must be expressed. When the
contract is honoured, the equation returns a number in the stated output unit. When it is violated,
the equation still returns a number. That number is wrong, plausible, and often actionable.

- A Cockcroft-Gault creatinine clearance is computed with a weight in pounds entered into a field
  labelled kilograms. The result is 161 mL/min instead of 73 mL/min, a factor of 2.2, and a
  renally-cleared drug is under-dosed.
- A vasopressor infusion is programmed in mL/h from a dose in µg/kg/min. The concentration is
  expressed in mg/mL. Three conversions are needed, one of them is skipped, and the rate is wrong by
  a factor of 1,000.
- A body surface area is computed with height in inches and weight in pounds using the Mosteller
  formula, which expects centimetres and kilograms. The result is 1.35 m² instead of 2.12 m², and a
  chemotherapy dose is reduced by more than a third.
- An oxygen content equation mixes a haemoglobin in g/L with a coefficient calibrated for g/dL. The
  dissolved oxygen term is negligible, so the error is a factor of 10 in the bound term and the
  result is physiologically impossible, yet it passes because nobody checks.
- A cost model divides cost per case by length of stay in days and multiplies by a monthly admission
  count. The output is labelled "cost per month" while carrying units of cost per bed-day-month. The
  budget is wrong by a factor of 30.

These are dimensional errors. They are distinct from unit conversion errors in one important way:
conversion errors happen when a factor is applied wrongly, while dimensional errors happen when the
structure of an equation is misunderstood. A converter cannot catch them, because there is no single
factor to apply. Only a check of the dimensions can.

The skill is cheap to learn and unusually high-yield. It takes about an hour to become fluent in
checking an equation, and the check catches a class of error that survives code review, statistical
validation, and clinical sign-off. It is also the fastest way to reconstruct a forgotten formula: if
you know the dimensions of the output and the dimensions of the inputs, the arrangement is often
forced.

Regulators care about this for a specific reason. Device labelling, drug dosing instructions, and
laboratory reporting are all written in units, and a dimensional inconsistency in a label is a
labelling defect. The FDA's infusion pump initiative documented screens that did not clearly indicate
pounds versus kilograms, leading to inappropriate dosing. That is a dimensional failure in a user
interface, not in an equation, and it caused harm.

> **Note:** This topic teaches dimensional checking as a working habit. It does not teach the
> Buckingham Pi theorem as a research method, although the theorem is stated because it explains why
> the check works.

## 2. Learning outcomes

By the end of this topic you can:

1. **Assign** a dimension vector to any clinical or engineering quantity, including composite units
   such as mg/kg/min and mL/min/1.73 m². Assessed in Exercise 1.
2. **Check** an equation for dimensional homogeneity by substituting dimension vectors and comparing
   exponents. Assessed in Exercises 2 and 3.
3. **Identify** empirical constants that carry units, such as the 72 in Cockcroft-Gault and the 3600
   in Mosteller, and state the unit contract they impose. Assessed in Exercise 4.
4. **Derive** a conversion factor from first principles using unit fractions, and chain several
   conversions without losing a factor. Assessed in Exercise 5.
5. **Reconstruct** a forgotten formula from the dimensions of its inputs and output. Assessed in
   Exercise 6.
6. **Detect** a dimensional error in an infusion or dosing calculation and compute the correct rate.
   Assessed in Exercise 7.
7. **Audit** a model or spreadsheet for dimensionally inconsistent columns and labelled-but-wrong
   output units. Assessed in Exercise 8.
8. **Write** a unit contract for a formula: input units, output unit, constant conventions, and the
   conditions under which the contract fails. Assessed in all exercises through the reporting rubric.

## 3. The core mental model

A dimension is a vector of exponents over the base quantities. A unit is a scale attached to a
dimension. A quantity is a number times a unit. Dimensional analysis is the arithmetic of those
vectors.

```text
   PHYSICAL QUANTITY
   Q = {Q} [Q]
   number x unit
        |
        |  decompose
        v
   DIMENSION VECTOR
   dim Q = L^a M^b T^c I^d Theta^e N^f J^g
        |
        |  compare across an equation
        v
   HOMOGENEITY CHECK
   dim(lhs) == dim(rhs) for every term
        |
        |  if unequal
        v
   ERROR LOCATED
   a term, a constant, or a unit contract is wrong
```

### 3.1 The one rule

**Every term in a sum must have the same dimension, and both sides of an equality must have the same
dimension.** Nothing else is required for the check. Multiplication and division combine dimensions by
adding and subtracting exponents; powers multiply them. Addition and equality compare them.

The rule has three corollaries that catch most errors:

1. You cannot add a concentration to a mass. If an equation appears to, one term has a hidden factor.
2. The argument of a logarithm, exponential, or trigonometric function must be dimensionless. A term
   like exp(−k·t) is valid only if k·t is dimensionless, which fixes the unit of k once t is chosen.
3. The exponent in a power law must be dimensionless, and the base must be dimensionless if the result
   is to be compared across unit systems.

### 3.2 The unit contract

Every empirical formula has a contract:

```text
Inputs:   weight in kg, height in cm, age in years, creatinine in mg/dL
Output:   creatinine clearance in mL/min
Constant: 72, which carries the unit conversion and is NOT dimensionless
Valid:    adults, stable creatinine, actual body weight
```

The contract is usually implicit. Making it explicit is the single most useful documentation act in a
quantitative pipeline, because it converts an unstated assumption into a checkable statement.

### 3.3 Why the check works

The Buckingham Pi theorem states that any physically meaningful equation relating n quantities with k
independent dimensions can be rewritten as a relationship among n − k dimensionless groups. The
practical consequence is that dimensions constrain the form of an equation. If the dimensions of the
inputs and output do not admit a consistent arrangement, no physically meaningful equation of that
form exists, and the formula is wrong regardless of how well it fits the data.

The theorem does not say that a dimensionally consistent equation is correct. Many wrong equations
are dimensionally consistent. It says that an inconsistent one is certainly wrong, which is enough to
make the check worth doing.

### 3.4 Dimensions versus units

Two quantities can share a dimension and differ in unit: metres and centimetres both have dimension L.
Two quantities can share a unit and differ in dimension: the litre is a volume, but litres per second
is a flow. The dimension is the invariant; the unit is the scale. A dimensional check is therefore
unit-system independent, which is why it catches errors that a unit converter cannot.

```text
Same dimension, different unit:   1 m = 100 cm          (conversion factor 100)
Different dimension, same symbol:  L (volume) vs L (length in some notations)
Same unit, different quantity:     mL/min (flow) vs mL/min/1.73 m2 (indexed flow)
```

The last row matters clinically: an eGFR in mL/min and an eGFR indexed to 1.73 m² are different
quantities with the same apparent unit unless the indexation is written. Dimensional analysis forces
the index to be explicit.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Dimension | An exponent vector over the base quantities | Confusing with unit | none |
| Base quantity | One of the seven independent quantities of the ISQ | Assuming all quantities are base | none |
| Base dimension | L, M, T, I, Θ, N, J | Using unit symbols as dimensions | none |
| Derived dimension | A product of powers of base dimensions | Treating it as independent | none |
| Dimensional exponent | The exponent of a base dimension in a derived dimension | Rounding it | dimensionless |
| Dimensional homogeneity | All terms of an equation share the same dimension | Checking only the two sides, not each term | none |
| Dimensionally inconsistent | An equation whose terms do not share a dimension | Assuming a fit implies consistency | none |
| Dimensionless quantity | A quantity with all exponents zero | Calling a unitless number dimensionless without checking | 1 |
| Quantity | A property with a magnitude and a reference | Calling a number a quantity | value × unit |
| Quantity calculus | The algebra of quantities and units treated symbolically | Treating units as labels | none |
| Quantity equation | An equation among quantities, independent of units | Writing a numerical-value equation and calling it general | none |
| Numerical-value equation | An equation valid only for specified units | Using it with other units | unit-specific |
| Unit equation | An equation among units alone | Ignoring it when checking a formula | none |
| Coherent unit | A unit with no numerical factor relative to the base | Assuming all SI units are coherent | none |
| Conversion factor | A ratio of two units equal to 1 | Applying it inverted | dimensionless ratio |
| Unit fraction | A conversion factor written as a fraction | Forgetting to cancel | dimensionless ratio |
| Chain conversion | Several unit fractions multiplied in sequence | Losing a factor in the chain | none |
| Empirical constant | A constant fitted to data, often carrying units | Treating it as dimensionless | constant's unit |
| Universal constant | A constant of nature with a fixed value | Confusing with an empirical constant | constant's unit |
| Proportionality constant | The constant in y = k·x, carrying the units of y/x | Dropping its units | y per x |
| Similarity | Equality of dimensionless groups between two systems | Assuming geometric similarity alone | none |
| Buckingham Pi theorem | n quantities with k dimensions reduce to n − k dimensionless groups | Treating it as a proof of correctness | none |
| Dimensionless group | A product of powers of quantities with zero net dimension | Confusing with a pure number | 1 |
| Reynolds number | Inertial to viscous force ratio, dimensionless | Applying it outside its regime | 1 |
| Scale invariance | Independence of a result from the unit of measurement | Assuming it for empirical formulas | none |
| Indexed quantity | A quantity normalised to a standard reference | Dropping the index | unit per reference |
| Body surface area (BSA) | Estimated surface area, used for dosing | Treating it as measured | m² |
| Body mass index (BMI) | Weight divided by height squared | Treating it as body fat | kg/m² |
| Creatinine clearance | Volume of plasma cleared of creatinine per time | Confusing with eGFR | mL/min |
| eGFR | Estimated glomerular filtration rate indexed to 1.73 m² | Reporting without the index | mL/min/1.73 m² |
| Clearance | Volume cleared per unit time | Confusing with elimination rate | volume/time |
| Elimination rate constant | Fractional removal per unit time | Reporting without the time unit | 1/time |
| Half-life | Time for concentration to halve | Applying to zero-order processes | time |
| AUC | Area under the concentration-time curve | Reporting without concentration × time units | mass·time/volume |
| Dose rate | Mass per mass per time | Dropping the mass basis | 1/time after mass cancels |
| Infusion rate | Volume per time | Confusing with dose rate | volume/time |
| Concentration | Mass or amount per volume | Mixing mass and molar concentration | mass/volume or amount/volume |
| Molar concentration | Amount of substance per volume | Writing it as mg/dL | mol/volume |
| Equivalent | One mole of charge | Using it for uncharged solutes | mol of charge |
| Osmolality | Osmoles per kilogram of solvent | Using osmolality and osmolarity interchangeably | osmol/kg |
| Anion gap | Difference of measured cations and anions | Using it without the lab's range | mmol/L |
| A-a gradient | Alveolar minus arterial oxygen partial pressure | Using it without the age correction | mmHg |
| Oxygen content | Bound plus dissolved oxygen per volume of blood | Mixing mL/dL and mL/L | volume of O2 per volume of blood |
| Unit contract | The stated input units, output unit, and validity conditions of a formula | Leaving it implicit | none |

Two definitional points require emphasis.

**Dimension is not unit.** The dimension of a clearance is volume per time, whether expressed in
mL/min or L/h. A formula that is dimensionally correct remains correct if every unit is changed
consistently; a numerical-value equation does not, because its constant encodes a specific unit
choice.

**An empirical constant has units.** The 72 in Cockcroft-Gault and the 3600 in Mosteller are not pure
numbers. They absorb the conversion between the units the formula expects and the coherent units of
the output. Treating them as dimensionless is the mechanism behind the pound-versus-kilogram failure
in section 1.

## 5. The mechanics, step by step

The procedure checks or derives a quantitative relationship. It applies to a formula in a paper, a
spreadsheet cell, a device algorithm, or a model feature.

### Step 1 — Write the equation in symbols, one term per line

**Input.** A formula, possibly written in prose or code.

**Output.** A symbolic equation with each term on its own line.

**Reasoning.** Dimensional checking is term-by-term. Prose hides terms inside parentheses and
qualifiers; code hides them inside function calls. Rewriting forces every additive term into view,
which is where the errors live.

**Branch point.** If a term cannot be written symbolically because its definition is unclear, stop.
The ambiguity is the finding.

### Step 2 — Assign a dimension vector to each variable

**Input.** The variable list.

**Output.** A vector over the base dimensions for each variable.

**Reasoning.** Use the seven ISQ dimensions: length L, mass M, time T, electric current I,
thermodynamic temperature Θ, amount of substance N, luminous intensity J. A concentration in mg/dL is
M L⁻³. A pressure in mmHg is M L⁻¹ T⁻². A rate per minute is T⁻¹. Most clinical quantities use only M,
L, T, and N.

**Branch point.** If a variable is a count or a ratio, its dimension is 1. Do not assign it a
dimension merely because it has a unit-like name.

### Step 3 — Treat each additive term separately

**Input.** The symbolic equation.

**Output.** A list of additive terms and their dimensions.

**Reasoning.** Homogeneity is required for each sum, not only for the two sides. An equation of the
form A = B + C requires dim(A) = dim(B) = dim(C). A common error is a formula whose left side matches
one term on the right but not the other, which happens when a coefficient was calibrated in different
units from the rest of the equation.

**Branch point.** If one term differs, go to Step 6 to decide whether a constant is carrying the
difference.

### Step 4 — Simplify each term by adding and subtracting exponents

**Input.** The dimension vectors.

**Output.** A simplified exponent for each base dimension.

**Reasoning.** Multiplication adds exponents, division subtracts them, and a power multiplies them.
For example, mg/kg/min is M·M⁻¹·T⁻¹ = T⁻¹. The simplification often reveals that a term thought to be
dimensional is actually dimensionless, or the reverse.

**Branch point.** If the simplification yields a fractional exponent, check whether a square root is
intended and whether the quantity under it has an even-powered dimension. A square root of a dimension
with odd exponents is a sign of a dimensional constant.

### Step 5 — Compare the left and right sides

**Input.** The simplified dimensions.

**Output.** Equality or a mismatch with the offending dimension.

**Reasoning.** Equality confirms that the equation could be physically meaningful. It does not confirm
that the coefficients are right, the model is right, or the data support it. It eliminates one class
of error, which is the purpose.

**Branch point.** If the sides match but the units do not (for example, output in mL/min but the
contract says L/h), the error is a unit contract issue, not a dimensional one. Go to Step 10.

### Step 6 — Inspect every constant for hidden units

**Input.** The constants in the equation.

**Output.** A unit for each constant, or a justification that it is dimensionless.

**Reasoning.** A constant is dimensionless only if it is a pure ratio, a mathematical constant, or a
count. A constant that makes a dimensionally inconsistent equation consistent is carrying units, and
those units define the contract. The 72 in Cockcroft-Gault is the canonical example: it exists because
the formula converts between kilograms, milligrams per decilitre, minutes, and millilitres.

**Branch point.** If a constant's units cannot be derived, the formula is empirical and its contract
must be taken from the source. Do not generalise it.

### Step 7 — Check the arguments of transcendental functions

**Input.** Any exp, ln, log, sin, cos, or tan term.

**Output.** A dimensionless argument, or an error.

**Reasoning.** The exponential e^x is defined by a power series in x, and adding x² to x requires x to
be dimensionless. The same applies to logarithms and trigonometric functions. In pharmacokinetics,
e^(−k·t) requires k·t to be dimensionless, which means k has dimension T⁻¹ when t has dimension T.
This is how the unit of a rate constant is fixed by the model rather than chosen.

**Branch point.** If the argument carries a dimension, insert the reference value that makes it
dimensionless and state it. A log of a concentration is really a log of a ratio to a reference
concentration.

### Step 8 — Check the base and exponent of power laws

**Input.** A term of the form a·x^b.

**Output.** A dimensionless exponent, and a base that is either dimensionless or has a dimension
consistent with the output.

**Reasoning.** In y = a·x^b, dim(y) = dim(a)·dim(x)^b. If b is not dimensionless, the equation cannot
be unit-independent. In allometric scaling, y = a·M^b with b a pure exponent and a carrying the
dimension of y/M^b. Reporting b as "0.75 per kilogram" is a category error.

**Branch point.** If b was estimated from data, its uncertainty propagates into the dimension of a,
which is why the constant is rarely reported with units in practice.

### Step 9 — Check derived clinical indices

**Input.** An index such as BMI, BSA, or eGFR.

**Output.** A dimension and an indexation statement.

**Reasoning.** BMI has dimension M L⁻², usually reported as kg/m². BSA has dimension L². eGFR has
dimension L³ T⁻¹ but is indexed to 1.73 m², so the reported quantity is a flow per standard surface
area and is not a plain flow. Dropping the index changes the meaning, not just the notation.

**Branch point.** If an index is reported without its reference, treat it as a different quantity and
do not compare it with indexed values.

### Step 10 — Convert units with unit fractions

**Input.** A value in a source unit and a target unit.

**Output.** A value in the target unit.

**Reasoning.** Write the conversion as a fraction equal to 1, with the source unit in the denominator
so it cancels. For example, 1 mmHg = 0.133322 kPa gives the fraction 0.133322 kPa / 1 mmHg. Multiplying
by the fraction cancels mmHg and leaves kPa. Chaining fractions is safe because each is equal to 1.

**Branch point.** If a conversion cannot be written as a fraction equal to 1, it is not a unit
conversion. An offset conversion such as °F to °C requires an affine transformation and cannot be
done by multiplication alone.

### Step 11 — Cancel units algebraically and check what remains

**Input.** The chained expression.

**Output.** The residual unit.

**Reasoning.** Cancellation is the audit trail. If the residual unit is not the desired output unit,
either a factor is inverted or a conversion is missing. Writing the units next to every number makes
the cancellation visible; doing the arithmetic first and attaching units later hides it.

**Branch point.** If the residual unit is correct but the magnitude is implausible, the error is in a
factor, not in the structure. Go to Step 22.

### Step 12 — Handle composite units component by component

**Input.** A composite unit such as mg/kg/min or µg/kg/h.

**Output.** A conversion path with one factor per component.

**Reasoning.** Convert mass, body-size basis, and time independently. Converting mg/kg/min to µg/kg/h
requires ×1,000 for mass and ×60 for time, a combined factor of 60,000. Errors occur when one
component is converted and the other is not, which produces a factor of 60 or 1,000 rather than 60,000.

**Branch point.** If the body-size basis changes (per kg to per m²), the conversion requires the
patient's height and weight. It is not a unit conversion at all; it is a re-expression using patient
data.

### Step 13 — Handle time-basis changes explicitly

**Input.** A rate per minute and a rate per hour.

**Output.** A converted rate.

**Reasoning.** The time basis is the most frequently dropped component in infusion calculations.
1 mg/min = 60 mg/h; 1 mL/h = 1/60 mL/min. Writing the basis as a unit fraction prevents the error:
(60 min / 1 h) is a valid fraction equal to 1.

**Branch point.** If the order is written "per 24 hours" and the pump is programmed "per hour", the
conversion is 24, not 60. Read the order, not the pump default.

### Step 14 — Derive an unknown unit from the equation

**Input.** An equation with one unknown unit.

**Output.** The unit of the unknown.

**Reasoning.** Rearranging the dimension equation gives the unknown dimension. If clearance = dose/AUC
and dose is mass and AUC is mass·time/volume, then clearance has dimension volume/time. This is how
the units of a fitted parameter are established, and how a mislabelled parameter is detected.

**Branch point.** If the derived unit is not a recognised quantity, the model may be misspecified. A
regression coefficient with units of "cost per mmHg per year" is interpretable; one with units of
"cost per mmHg per year per kilogram" suggests an interaction that was not intended.

### Step 15 — Test the formula's unit-specific constant

**Input.** An empirical formula and a constant.

**Output.** A demonstration of what happens when the input units change.

**Reasoning.** Substitute a value in a different unit and show that the output changes by the
conversion factor. For Cockcroft-Gault, entering 198 lb as 198 kg gives 161 mL/min instead of
73 mL/min. The demonstration is the evidence that the constant is unit-specific and that the formula
cannot be used with other units without conversion.

**Branch point.** If the formula is invariant to a unit change, either it is written in quantity
calculus with explicit units or the constant is genuinely dimensionless. Verify which before relaxing
the contract.

### Step 16 — Reconstruct a formula from dimensions

**Input.** The output dimension and the available input dimensions.

**Output.** A candidate arrangement.

**Reasoning.** If the output is a flow (L³ T⁻¹) and the inputs are volume (L³) and time (T), the only
arrangement is volume/time. If the output is a velocity (L T⁻¹) and the inputs are length and time,
the arrangement is forced. When several arrangements are possible, dimensions cannot choose between
them, and the choice requires theory or data.

**Branch point.** If no arrangement produces the output dimension, a relevant variable is missing or
a constant carries units. Add the missing quantity before forcing a fit.

### Step 17 — Check dose calculations against the prescribed basis

**Input.** A prescription in mg/kg/day or mg/m²/day and a patient's size.

**Output.** A dose per administration and a rate.

**Reasoning.** The basis matters. A dose per kg uses actual, ideal, or adjusted body weight, and the
choice changes the dose in obesity. A dose per m² uses BSA, which requires a formula with its own
unit contract. The check confirms that the basis in the prescription matches the basis used in the
calculation.

**Branch point.** If the prescription does not state the weight basis, the dose is ambiguous. Resolve
before calculating.

### Step 18 — Check infusion rates against concentration and dose rate

**Input.** A concentration, a dose rate, and a pump rate.

**Output.** A verified rate, or the correction.

**Reasoning.** The chain is dose rate (mass/mass/time) → dose per minute (mass/time) → volume per
minute (volume/time) → volume per hour (volume/time). Each arrow is a multiplication by a factor
equal to 1. A missing arrow produces a factor of 60, 1,000, or the patient's weight.

**Branch point.** If the concentration is expressed as a ratio (1:1,000), convert to mass per volume
before proceeding. Ratio expressions are a common source of a factor of 1,000.

### Step 19 — Check laboratory derived quantities

**Input.** A formula such as the anion gap or osmolality.

**Output.** A dimensionally consistent result in the laboratory's units.

**Reasoning.** The anion gap is Na − (Cl + HCO₃), all in mmol/L, so the output is mmol/L. The
calculated osmolality is 2·Na + glucose/18 + BUN/2.8, where the divisors carry units of mg/dL per
mmol/L. The divisors are empirical constants with units, not pure numbers, and they assume the
analytes are reported in mg/dL.

**Branch point.** If the laboratory reports glucose in mmol/L, the divisor must be removed, not
applied. Applying 18 to a value already in mmol/L produces an osmolality that is wrong by a factor of
18 in that term.

### Step 20 — Check statistical and economic quantities

**Input.** A regression coefficient, a rate, an odds ratio, an incremental cost-effectiveness ratio.

**Output.** A dimension for each.

**Reasoning.** A regression coefficient carries the unit of the outcome divided by the unit of the
predictor. A hazard ratio is dimensionless because it is a ratio of hazards. A rate has dimension
T⁻¹. An incremental cost-effectiveness ratio has dimension cost per QALY, which is currency per
quality-adjusted life year. Checking these prevents the common error of comparing coefficients from
models with different outcome units.

**Branch point.** If a coefficient is reported without units, its magnitude cannot be compared across
studies. Request the units or the standard deviation of the predictor.

### Step 21 — Write the unit contract into the code

**Input.** A verified formula.

**Output.** Names, docstrings, assertions, and tests that encode the contract.

**Reasoning.** A check performed once is forgotten. A check encoded in a function name
(`cockcroft_gault_ml_min(age_years, weight_kg, scr_mg_dl)`) and an assertion (`assert 0 < scr < 20`)
survives refactoring. This is the difference between a habit and a control.

**Branch point.** If the code cannot express the contract because the interface accepts untyped
numbers, add a validation layer at the boundary rather than relying on documentation.

### Step 22 — Bound the result plausibility

**Input.** The computed value.

**Output.** A pass or a flag.

**Reasoning.** A creatinine clearance of 161 mL/min in an adult with a creatinine of 1.4 mg/dL is
physiologically implausible; a BSA of 1.35 m² in an adult of 180 cm and 90 kg is implausible; an
infusion of 67 mL/h for a standard concentration is plausible while 4,000 mL/h is not. Bounds catch
the errors that dimensional checks miss because the constant was applied in the wrong units but the
structure was right.

**Branch point.** If the value is outside the bound, locate the unit error before adjusting the model.

### Step 23 — Record the check in the assumption ledger

**Input.** The verified formula and its contract.

**Output.** A ledger row: assumption, evidence, test, consequence if false.

**Reasoning.** The ledger makes the contract visible to reviewers and to the next analyst. A row such
as "Cockcroft-Gault is computed with actual body weight in kg and creatinine in mg/dL; the constant 72
is unit-specific" is testable and decision-relevant.

**Branch point.** If the contract cannot be verified from the source, mark the assumption open rather
than closed.

### Step 24 — Re-check after any change of units or population

**Input.** A modified pipeline or a new population.

**Output.** A re-run of the dimensional and plausibility checks.

**Reasoning.** Unit contracts break when a new data source arrives with different units, when a
paediatric population is added, or when a device is replaced. The check is cheap; the failure it
prevents is not.

**Branch point.** If the new source cannot supply the units the contract requires, the formula cannot
be applied. Do not substitute a default.

### 5.1 The branch map

```text
Equation or calculation
  |
  +-- term not separable? -----------> ambiguity is the finding
  |
  +-- addition of different dims? ---> hidden unit constant or wrong term
  |
  +-- transcendental argument dim? --> insert reference value
  |
  +-- exponent not dimensionless? ---> category error in the power law
  |
  +-- sides equal but units differ? -> unit contract error
  |
  +-- constant unexplained? --------> derive its units or cite the source
  |
  +-- composite unit? --------------> convert each component
  |
  +-- time basis changed? ----------> 60 or 24, stated explicitly
  |
  +-- result implausible? ----------> factor error, not structure error
```

### 5.2 The four checks that catch most dimensional errors

1. **Write the unit next to every number in the calculation.** If a line has a number without a unit,
   the check has stopped.
2. **Cancel before multiplying.** The residual unit tells you whether the arrangement is right.
3. **Substitute a value in the wrong unit deliberately.** If the output changes by a plausible factor,
   the formula has a unit-specific constant and a contract.
4. **Bound the answer.** A physiologically impossible value is a dimensional error until proven
   otherwise.

## 6. Worked example

A 58-year-old man, 180.34 cm and 89.8113 kg, has a serum creatinine of 1.4 mg/dL, haemoglobin
13.4 g/dL, arterial oxygen saturation 0.97, arterial oxygen partial pressure 92 mmHg, arterial carbon
dioxide partial pressure 40 mmHg, sodium 140 mmol/L, chloride 104 mmol/L, bicarbonate 24 mmol/L, and
is breathing room air at sea level. He is prescribed an infusion of 1 g of drug in 250 mL at
0.05 mg/kg/min.

The case checks eight formulas dimensionally, computes each, and then introduces three deliberate unit
errors to show what the checks catch.

### 6.1 Write the unit contracts

| Formula | Inputs | Output | Constant and its status |
|---|---|---|---|
| BMI = weight/height² | kg, m | kg/m² | none |
| BSA = √(height × weight / 3600) | cm, kg | m² | 3600 is dimensional |
| CaO₂ = 1.34·Hb·SaO₂ + 0.003·PaO₂ | g/dL, fraction, mmHg | mL O₂/dL | 1.34 mL/g, 0.003 mL/(dL·mmHg) |
| PAO₂ = FiO₂·(P_atm − 47) − PaCO₂/0.8 | fraction, mmHg, mmHg | mmHg | 47 mmHg, 0.8 dimensionless |
| Anion gap = Na − (Cl + HCO₃) | mmol/L | mmol/L | none |
| Osmolality = 2·Na + glucose/18 + BUN/2.8 | mmol/L, mg/dL, mg/dL | mOsm/kg | 18 and 2.8 carry units |
| Cockcroft-Gault | years, kg, mg/dL | mL/min | 72 is unit-specific |
| CrCl = ((140 − age)·weight)/(72·SCr) | | | |

The contract table is the artefact the rest of the case depends on. Each row states what the formula
expects and what it returns. Any use outside the stated units is a new formula, not the same one.

### 6.2 Body size

```text
BMI = 89.8113 kg / (1.8034 m)^2
    = 89.8113 / 3.25225
    = 27.6151 kg/m2
Reported: 27.6 kg/m2

Dimension check: M L^-2. Valid: mass divided by length squared.
```

```text
BSA = sqrt(height cm x weight kg / 3600)
    = sqrt(180.34 x 89.8113 / 3600)
    = sqrt(16196.60 / 3600)
    = sqrt(4.49906)
    = 2.1211 m2
Reported: 2.12 m2
```

The dimensional check of the BSA formula exposes the 3600. Under the square root, height × weight has
dimension L·M, so the square root has dimension L^0.5·M^0.5, not L². The constant 3600 therefore
carries dimension cm·kg/m⁴, and the formula is valid only with centimetres and kilograms. Entering
inches and pounds produces a number that is dimensionally meaningless and numerically wrong.

```text
Wrong-units BSA with 71 in and 198 lb:
  sqrt(71 x 198 / 3600) = sqrt(3.905) = 1.976 m2
Correct 2.121 m2; error = (2.121 - 1.976)/2.121 = 6.8%
```

A 6.8% BSA error changes a chemotherapy dose by 6.8%, which is inside some dosing tolerances and
outside others. It is the kind of error that survives because the result is not absurd.

### 6.3 Oxygen content and the alveolar-arterial gradient

```text
CaO2 = 1.34 mL O2/g x 13.4 g/dL x 0.97 + 0.003 mL O2/(dL mmHg) x 92 mmHg
     = 17.4173 mL O2/dL + 0.2760 mL O2/dL
     = 17.6933 mL O2/dL
     = 176.93 mL O2/L

Dimension check:
  mL/g x g/dL = mL/dL                     (bound term)
  mL/(dL mmHg) x mmHg = mL/dL             (dissolved term)
  Both terms mL/dL -> addition valid.
```

The check confirms that the two terms can be added. It also shows why the coefficient 1.34 must carry
mL/g: without it the bound term would be g/dL and could not be added to mL/dL.

```text
PAO2 = 0.21 x (760 - 47) - 40/0.8
     = 0.21 x 713 - 50
     = 149.73 - 50
     = 99.73 mmHg

A-a gradient = 99.73 - 92 = 7.73 mmHg
```

The 47 is the saturated water vapour pressure at 37 °C in mmHg, and it must be in the same unit as the
atmospheric pressure. The 0.8 is the respiratory quotient, dimensionless. At altitude, P_atm falls and
the gradient calculation must use the local value; substituting 760 at 2,000 m overestimates PAO₂ by
about 150 mmHg.

### 6.4 Anion gap and osmolality

```text
Anion gap = 140 - (104 + 24) = 12 mmol/L

Dimension: mmol/L for every term. Addition valid.
```

```text
Calculated osmolality = 2 x 140 + 126/18.0182 + 14/2.8
                      = 280 + 6.9929 + 5.0000
                      = 291.99 mOsm/kg
Reported: 292 mOsm/kg
```

The divisors 18.0182 and 2.8 are unit-carrying constants: mg/dL divided by mg/dL per mmol/L gives
mmol/L. If the laboratory reports glucose in mmol/L, the divisor must be omitted. Applying 18.0182 to
a glucose already in mmol/L gives 0.39 instead of 6.99, and the calculated osmolality falls by
6.6 mOsm/kg, which is enough to change the interpretation of an osmolal gap.

> **Note:** The measured osmolality is reported in mOsm/kg of solvent while the calculated value is
> conventionally in mOsm/kg of water. The difference is small at physiological concentrations but is a
> real dimensional distinction, not a rounding artefact.

### 6.5 Cockcroft-Gault and the pound-kilogram failure

```text
CrCl = ((140 - 58) x 89.8113) / (72 x 1.4)
     = (82 x 89.8113) / 100.8
     = 7364.53 / 100.8
     = 73.06 mL/min
Reported: 73 mL/min

Dimension check of the right side:
  years x kg / (mg/dL) = kg dL / mg  (with years treated as a pure number)
  This is not mL/min. The constant 72 supplies the missing conversion.
```

The constant 72 is not dimensionless. It absorbs the factors that convert kilograms, milligrams per
decilitre, and minutes into millilitres per minute. That is why the formula is valid only with the
stated units.

```text
If 198 lb is entered as 198 kg:
  CrCl = (82 x 198) / (72 x 1.4) = 16236 / 100.8 = 161.07 mL/min

Ratio = 161.07 / 73.06 = 2.205
```

The ratio is exactly the pound-to-kilogram factor, 2.20462, rounded. A clinician who sees 161 mL/min
may conclude that renal function is normal and prescribe a full dose of a renally cleared drug. The
correct value, 73 mL/min, may require dose reduction. The error is a factor of 2.2 and it passes every
arithmetic check because the arithmetic is right and the units are wrong.

### 6.6 Infusion rate

The order is 1 g in 250 mL at 0.05 mg/kg/min. The pump is programmed in mL/h.

```text
Concentration
  1000 mg / 250 mL = 4.0 mg/mL

Dose rate
  0.05 mg/kg/min x 89.8113 kg = 4.4906 mg/min

Volume rate
  4.4906 mg/min / 4.0 mg/mL = 1.1226 mL/min

Pump rate
  1.1226 mL/min x 60 min/h = 67.36 mL/h
Reported: 67 mL/h
```

The chain has four links and three unit fractions: kg cancels, mg cancels, min converts to h. A missing
time conversion gives 1.12 mL/h, a factor of 60 low. A missing concentration conversion, treating
1 g/250 mL as 1 mg/mL, gives 269 mL/h, a factor of 4 high. Both errors produce plausible-looking pump
settings, which is why the chain must be written out.

### 6.7 Milliequivalents

Calcium gluconate 10% contains 1 g in 10 mL. The label may express the content in mg, mmol, or mEq,
and the three differ.

```text
Moles
  1000 mg / 448.4 g/mol = 2.2302 mmol

Equivalents
  calcium is divalent, so 1 mmol = 2 mEq
  2.2302 mmol x 2 = 4.460 mEq

Concentration
  4.460 mEq / 10 mL = 0.446 mEq/mL
```

A prescription written in mEq and dispensed in mg requires both the molar mass and the valence. Using
the molar mass alone halves the dose. Using the valence alone is meaningless. The dimensional check
confirms that mEq has dimension N (amount of charge), which is why the conversion is not a simple mass
conversion.

### 6.8 What the example demonstrates

| Formula | Check performed | Error caught |
|---|---|---|
| BMI | M L⁻² | Height in cm not converted to m |
| BSA | Square root of L·M | 3600 assumed dimensionless |
| CaO₂ | Both terms mL/dL | Haemoglobin in g/L not g/dL |
| PAO₂ | mmHg throughout | Altitude ignored |
| Anion gap | All terms mmol/L | Bicarbonate in mEq/L with a different valence |
| Osmolality | Divisors carry units | Applying 18 to mmol/L glucose |
| Cockcroft-Gault | Constant 72 carries units | Weight in pounds |
| Infusion | Four-link chain | Missing time or concentration conversion |
| mEq | Valence included | Mass-to-equivalent confusion |

## 7. Code, query, or calculation

The script implements a small dimensional algebra over exponent vectors, checks six quantities,
computes the clinical values, and demonstrates three unit failures.

```python
"""Topic 04 verification: dimensional algebra and clinical formula checks."""
from math import sqrt

# Dimensions as exponent vectors over (M, L, T, N).
D = {
    "kg": {"M": 1},
    "g": {"M": 1},
    "m": {"L": 1},
    "cm": {"L": 1},
    "dL": {"L": 3},
    "L": {"L": 3},
    "mL": {"L": 3},
    "min": {"T": 1},
    "h": {"T": 1},
    "mol": {"N": 1},
    "mmHg": {"M": 1, "L": -1, "T": -2},
}

def mul(*ds):
    out = {}
    for d in ds:
        for k, v in d.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}

def div(a, b):
    return mul(a, {k: -v for k, v in b.items()})

def power(a, p):
    return {k: v * p for k, v in a.items()}

def show(d):
    if not d:
        return "1 (dimensionless)"
    order = ["M", "L", "T", "N"]
    return " ".join(f"{k}^{d[k]:g}" for k in order if k in d)

print("4.1 Dimension checks")
print(f"  BMI kg/m^2           : {show(div(D['kg'], power(D['m'], 2)))}")
print(f"  BSA sqrt(cm*kg)      : {show(power(mul(D['cm'], D['kg']), 0.5))}")
print(f"  Clearance mL/min     : {show(div(D['mL'], D['min']))}")
print(f"  AUC mg*h/L           : {show(div(mul(D['g'], D['h']), D['L']))}")
print(f"  Dose mg/kg/min       : {show(div(D['g'], mul(D['kg'], D['min'])))}")
print(f"  mmHg                 : {show(D['mmHg'])}")

print("\n4.2 Addition check: MAP = DBP + (SBP-DBP)/3")
print(f"  DBP dimension        : {show(D['mmHg'])}")
print(f"  (SBP-DBP)/3          : {show(D['mmHg'])}")
print("  same dimension -> addition valid")

print("\n4.3 Mosteller constant is dimensional")
print("  sqrt(cm * kg) has dimension", show(power(mul(D['cm'], D['kg']), 0.5)))
print("  to obtain m^2 the constant 3600 must carry cm*kg/m^4")

print("\n4.4 Numeric clinical calculations")
kg, cm, age, scr = 89.8113, 180.34, 58, 1.4
h_m = cm / 100
bmi = kg / h_m ** 2
bsa = sqrt(cm * kg / 3600)
print(f"  BMI  = {bmi:.4f} kg/m2")
print(f"  BSA  = {bsa:.4f} m2")

hb, sao2, pao2 = 13.4, 0.97, 92
cao2 = 1.34 * hb * sao2 + 0.003 * pao2
print(f"  CaO2 = {1.34*hb*sao2:.4f} + {0.003*pao2:.4f} = {cao2:.4f} mL/dL")
print(f"       = {cao2*10:.2f} mL/L")

fio2, patm, paco2 = 0.21, 760, 40
pao2_calc = fio2 * (patm - 47) - paco2 / 0.8
print(f"  PAO2 = {fio2*(patm-47):.2f} - {paco2/0.8:.2f} = {pao2_calc:.2f} mmHg")
print(f"  A-a  = {pao2_calc - pao2:.2f} mmHg")

na, cl, hco3 = 140, 104, 24
print(f"  anion gap = {na} - ({cl} + {hco3}) = {na-(cl+hco3)} mmol/L")

crcl = ((140 - age) * kg) / (72 * scr)
print(f"  Cockcroft-Gault (kg) = {crcl:.2f} mL/min")
crcl_lb = ((140 - age) * 198) / (72 * scr)
print(f"  Cockcroft-Gault (lb entered as kg) = {crcl_lb:.2f} mL/min")
print(f"  ratio = {crcl_lb/crcl:.3f}")

print("\n4.5 Osmolality")
osm = 2 * na + 126 / 18.0182 + 14 / 2.8
print(f"  2*Na = {2*na}, glucose/18.0182 = {126/18.0182:.2f}, BUN/2.8 = {14/2.8:.2f}")
print(f"  calculated osmolality = {osm:.1f} mOsm/kg")

print("\n4.6 Infusion rate")
conc_mg_per_ml = 1000 / 250
dose_mg_per_min = 0.05 * kg
rate_ml_per_min = dose_mg_per_min / conc_mg_per_ml
print(f"  concentration = {conc_mg_per_ml:.1f} mg/mL")
print(f"  dose = 0.05 mg/kg/min x {kg:.4f} kg = {dose_mg_per_min:.4f} mg/min")
print(f"  rate = {rate_ml_per_min:.4f} mL/min = {rate_ml_per_min*60:.2f} mL/h")

print("\n4.7 Unit-fraction conversion")
val_mmhg = 110.67
print(f"  {val_mmhg} mmHg x (0.133322 kPa / 1 mmHg) = {val_mmhg*0.133322:.3f} kPa")

print("\n4.8 mEq conversion for calcium gluconate")
mg = 1000.0
mw = 448.4
valence = 2
mmol = mg / mw
meq = mmol * valence
print(f"  1 g / 448.4 g/mol = {mmol:.4f} mmol")
print(f"  x valence {valence} = {meq:.3f} mEq")
```

**Expected output.**

```text
4.1 Dimension checks
  BMI kg/m^2           : M^1 L^-2
  BSA sqrt(cm*kg)      : M^0.5 L^0.5
  Clearance mL/min     : L^3 T^-1
  AUC mg*h/L           : M^1 L^-3 T^1
  Dose mg/kg/min       : T^-1
  mmHg                 : M^1 L^-1 T^-2

4.2 Addition check: MAP = DBP + (SBP-DBP)/3
  DBP dimension        : M^1 L^-1 T^-2
  (SBP-DBP)/3          : M^1 L^-1 T^-2
  same dimension -> addition valid

4.3 Mosteller constant is dimensional
  sqrt(cm * kg) has dimension M^0.5 L^0.5
  to obtain m^2 the constant 3600 must carry cm*kg/m^4

4.4 Numeric clinical calculations
  BMI  = 27.6151 kg/m2
  BSA  = 2.1211 m2
  CaO2 = 17.4173 + 0.2760 = 17.6933 mL/dL
       = 176.93 mL/L
  PAO2 = 149.73 - 50.00 = 99.73 mmHg
  A-a  = 7.73 mmHg
  anion gap = 140 - (104 + 24) = 12 mmol/L
  Cockcroft-Gault (kg) = 73.06 mL/min
  Cockcroft-Gault (lb entered as kg) = 161.07 mL/min
  ratio = 2.205

4.5 Osmolality
  2*Na = 280, glucose/18.0182 = 6.99, BUN/2.8 = 5.00
  calculated osmolality = 292.0 mOsm/kg

4.6 Infusion rate
  concentration = 4.0 mg/mL
  dose = 0.05 mg/kg/min x 89.8113 kg = 4.4906 mg/min
  rate = 1.1226 mL/min = 67.36 mL/h

4.7 Unit-fraction conversion
  110.67 mmHg x (0.133322 kPa / 1 mmHg) = 14.755 kPa

4.8 mEq conversion for calcium gluconate
  1 g / 448.4 g/mol = 2.2302 mmol
  x valence 2 = 4.460 mEq
```

**Assumptions stated by the script.**

1. The dimension vectors cover M, L, T, and N only. Luminous intensity and electric current are
   omitted because no formula in the example uses them.
2. The Mosteller constant is treated as dimensional; the script shows the dimension of the
   unconstrained square root rather than silently inserting a factor.
3. Cockcroft-Gault is evaluated with actual body weight. Ideal or adjusted body weight would change
   the result and is a separate clinical assumption.
4. The osmolality formula assumes glucose and BUN in mg/dL, which is what makes the divisors
   meaningful.
5. The infusion assumes a linear concentration and no dead-space or line-flush correction.

## 8. Data, standards, and regulatory context

### 8.1 The standards that define the algebra

**ISO 80000-1:2022**, *Quantities and units — Part 1: General*, second edition, is the normative
reference for quantity calculus. Clause 4 defines quantities, kinds, and systems of quantities;
Clause 5 defines the dimension of a quantity as a product of powers of the base dimensions,
dim Q = L^α M^β T^γ I^δ Θ^ε N^ζ J^η, and states that quantities of the same kind have the same
dimension while quantities of different dimension cannot be compared; Clause 6 defines the relation
between a quantity, its numerical value, and its unit, and distinguishes quantity equations from
numerical-value equations.

The distinction in Clause 6 is the one this topic depends on. A quantity equation is independent of
the units chosen; a numerical-value equation is valid only for a specified set of units. Cockcroft-
Gault is a numerical-value equation. Writing it as a quantity equation would require the constant 72
to carry its units explicitly, which is exactly the point of section 6.5.

**ISO 80000-2:2019** covers mathematical notation, including the convention that a logarithm without
a stated base is ambiguous.

**The SI Brochure (BIPM, 9th edition, 2019, revised 2026)** defines the units in which the base
dimensions are expressed and lists the coherent derived units. **NIST SP 811** is the US guide to
their use, including the rules for prefixes, spacing, and unit symbols.

**JCGM 200:2008**, the International Vocabulary of Metrology (VIM), defines quantity calculus
(entry 1.21), quantity equation (1.22), unit equation (1.23), and numerical-value equation (1.25).
The vocabulary is the reference for the terms used in this topic.

**UCUM** supplies the machine-readable syntax in which the units are transmitted. A dimensional check
in code is only as good as the unit codes it parses, which is why the check belongs after unit
normalisation, not before.

### 8.2 Where regulation touches dimensional consistency

**Device labelling.** FDA labelling requirements for prescription drugs and devices require the
strength, concentration, and dosage to be stated with units. The FDA's 2010 infusion pump white paper
identified user interfaces that did not clearly indicate pounds versus kilograms as a cause of
inappropriate dosing. That is a dimensional requirement expressed as a human-factors requirement.

**Drug dosing.** Paediatric and oncology dosing instructions are written per kg or per m², and the
label must state the basis. A dose calculated on the wrong basis is a dosing error even if every
arithmetic operation is correct.

**Laboratory reporting.** Results must be reported with units, and derived quantities such as the
anion gap or eGFR must state the formula and the indexation. CLIA requires the laboratory to define
its reportable range and reference intervals, both of which are unit-specific.

**Medical device software.** IEC 62304 requires software requirements to be verifiable. A requirement
that says "compute the dose" without a unit contract is not verifiable, which is a documentation
defect under the standard even when the code is correct.

### 8.3 What the standards do not constrain

- They do not choose the units an application should use internally. That is a design decision.
- They do not require a dimensional check in code. They define the algebra; the check is a practice.
- They do not make an empirical formula generalisable. A published formula remains tied to its
  population and its unit contract regardless of how it is documented.
- They do not resolve conflicts between conventions, such as glucose 18 versus 18.0182. They require
  that the convention be stated.
- They do not cover every clinical formula. Many, including the anion gap and the A-a gradient, are
  convention rather than standard.

> **Note:** The regulatory content here is educational. Labelling and software compliance require
> review by a qualified regulatory professional.

## 9. Industry practice and case evidence

### 9.1 What organisations actually do

**Hospitals and infusion pump vendors** use drug libraries and smart pumps that constrain the
programmable dose and rate to a configured range. The library is a dimensional control: it encodes the
concentration, the dose basis, and the allowed rate for each drug. Its weakness is that it depends on
correct configuration and on the clinician selecting the right entry, which is why independent double
checks are recommended for high-alert medications.

**Pharmacies and oncology services** compute doses per m² or per kg with a verified BSA or weight, and
many require a second-person check for chemotherapy. The check is a dimensional check performed by a
human: does the prescribed dose, the patient's size, and the protocol agree on the basis.

**Laboratory information systems** store results with units and compute derived quantities with
formula-specific unit contracts. A well-configured system refuses to compute an anion gap if the
electrolytes arrive in different units, rather than silently mixing them.

**Clinical data teams** encode unit contracts in function names and validation rules. The practice is
not universal. A team that names a function `crcl(age, weight, scr)` without units has left the
contract implicit, and the first caller with pounds will produce the error in section 6.5.

### 9.2 Case evidence

**The FDA infusion pump initiative.** The FDA's 2010 white paper reported approximately 56,000 adverse
event reports associated with infusion pumps between 2005 and 2009 and 87 recalls in the same period,
including 14 Class I recalls. Among the documented problems were software defects and user-interface
issues, including screens that did not clearly indicate units of measurement such as pounds versus
kilograms, leading to inappropriate dosing. The report is a documented case of dimensional ambiguity
causing harm at scale, and it led to the FDA's infusion pump improvement initiative, including
requirements for manufacturers and a programme of static code analysis offered to vendors.

**Chemotherapy dosing by body surface area.** BSA-based dosing is standard for many cytotoxic agents
because clearance correlates with surface area better than with weight alone. The practice makes the
BSA formula a safety-critical calculation: an error in height or weight units propagates directly into
the dose. The Mosteller formula's dimensional constant is therefore not an academic curiosity; it is
the mechanism by which an inches-and-pounds entry becomes a dosing error.

**The Mars Climate Orbiter.** In 1999 the orbiter was lost after navigation software produced
impulse data in pound-force seconds while the receiving system expected newton seconds. The mishap
investigation board documented the interface as lacking a unit specification. The case is not
biomedical and is included only as the clearest public example of the general failure: two systems,
each internally consistent, exchanging a number without its unit contract.

**The anion gap and the osmolality divisor.** Clinical chemistry laboratories routinely report a
calculated osmolality using divisors of 18 and 2.8 when glucose and BUN are in mg/dL. When a laboratory
changes its reporting units to SI, the divisors must change or be removed. This is a well-known
configuration hazard in laboratory information systems and a routine item in method-validation
checklists.

> **Note:** The infusion pump figures and the Mars orbiter failure are documented in the cited public
> reports. The interpretation of the BSA and osmolality cases as dimensional-contract failures is
> inference from the formulas and the standards, not a claim about a specific institution's incident.

### 9.3 The practical pattern

Teams that avoid dimensional errors do four things:

1. They put units in function names and column names, not only in documentation.
2. They write the unit chain for every dose or rate calculation, and a second person reads it.
3. They encode plausibility bounds next to the formula, in the output unit.
4. They re-run the check when a data source, population, or device changes.

The cost is minutes per formula. The failure it prevents is a wrong dose, a wrong budget, or a wrong
label, each of which is expensive to discover after the fact.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Empirical constant treated as dimensionless | A formula works only with certain units but no contract is stated | The constant absorbs unit conversions | Factor-of-2.2 or 39 error when units change | Derive the constant's units; encode the contract in the function name |
| 2 | Mixing mass and molar concentration | A term in mg/dL added to a term in mmol/L | Both look like "concentration" | Result wrong by the molar mass factor | Convert to one basis before adding |
| 3 | Missing time-basis conversion | Rate per minute entered as rate per hour | 60 is dropped | Factor of 60 under- or over-infusion | Write the unit chain with min/h fractions |
| 4 | Missing mass-basis conversion | mg and µg mixed | 1,000 is dropped | Factor of 1,000 dose error | Convert to a single mass unit before dividing |
| 5 | Body-size basis mismatch | Dose per kg computed with lb, or per m² with kg | The prescription basis differs from the calculation basis | Dose error proportional to the unit error | State the basis; convert weight and height first |
| 6 | Index dropped from an indexed quantity | eGFR reported as mL/min | The 1.73 m² reference is omitted | Values compared across body sizes incorrectly | Store the index with the unit |
| 7 | Transcendental argument with units | exp(−k·t) with k in mg/h | The exponent is not dimensionless | Nonsensical decay or growth | Check k·t is dimensionless; fix the unit of k |
| 8 | Offset scale converted multiplicatively | °F multiplied by 5/9 | The 32 offset is omitted | Impossible temperature | Affine converter with a unit test |
| 9 | Square root of an odd-powered dimension | √(mg/dL) in a formula | A dimensional constant was omitted | Output unit is meaningless | Inspect the constant under the root |
| 10 | Adding a rate to a quantity | Cumulative dose plus a rate in one sum | The terms have different dimensions | Nonsense total | Separate stock and flow; integrate the rate |
| 11 | Ratio unit expressed as a single number | 1:1,000 interpreted as 1,000 mg/mL | Ratio notation is ambiguous | Factor of 1,000 | Convert ratios to mass per volume explicitly |
| 12 | Unit contract changed by a new data source | A new lab reports in SI | The old divisors remain in code | Derived quantities wrong by 18 or 2.8 | Validate units at ingestion; version the formula |
| 13 | Coefficient compared across models with different outcome units | Two studies report "0.42" | The coefficient carries outcome units | False comparison of effect sizes | Convert to a standardised effect or request units |
| 14 | Pump programmed in the wrong mode | mL/h versus mL/min selected | The device accepts both | Factor of 60 | Verify the mode on the pump display |
| 15 | Altitude or pressure correction omitted | PAO₂ computed with 760 mmHg at altitude | P_atm is local | A-a gradient wrong by more than 100 mmHg | Use local barometric pressure |
| 16 | Valence omitted in equivalent conversion | mg to mEq using molar mass only | Equivalents depend on charge | Factor of 2 for divalent ions | Include the valence explicitly |

Two rows deserve emphasis.

**Row 1, the empirical constant.** The failure is invisible in code review because the constant looks
like a number. The tell is that the formula is described as "valid for weight in kg" in a paper but
the code accepts any weight. The mitigation is to put the unit in the parameter name and to assert a
plausible range, so a pound value fails the assertion rather than silently producing 161 mL/min.

**Row 10, stock versus flow.** Adding a cumulative quantity to a rate is dimensionally invalid, but
spreadsheets do it routinely because both columns are numeric. The result is not merely wrong; it has
no interpretation. The check is to ask what the unit of the sum would be, and to notice that the
question has no answer.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The equation is a quantity equation or a numerical-value equation, and the analyst knows which | The check differs for the two | Treating an empirical formula as general | Inspect the constants for units |
| 2 | The variables are measured in the units the formula expects | The formula's coefficients are calibrated to those units | A new source reports in SI | Assert units at the boundary |
| 3 | The base dimensions used are sufficient | A check over M, L, T, N omits I, Θ, J | An electrical or thermal term is present | Add the missing dimensions |
| 4 | The constants are correctly transcribed | A digit error in a constant is not dimensional | Manual transcription | Unit tests against a published worked example |
| 5 | The formula's population matches the patient | The coefficients were fitted in a specific population | Paediatrics, pregnancy, extremes of body size | Check the derivation population; do not extrapolate |
| 6 | The physiological model is correct | Dimensional consistency does not imply validity | A wrong model with consistent units | Compare with an independent measurement |
| 7 | The units of the output are the units the caller expects | A correct formula can return the wrong unit | Interface mismatch | Typed return values; contract tests |
| 8 | The body-size basis is the one the prescription intends | Dose per kg depends on which weight | Actual, ideal, adjusted body weight | Read the protocol; document the choice |
| 9 | The conversion factors are exact or their approximation is stated | Rounded factors introduce bias | Using 18 for 18.0182 | Versioned factor table |
| 10 | The measurement uncertainty is small relative to the decision | A dimensionally correct result can still be too imprecise | Assay imprecision near a threshold | Propagate uncertainty (Topic 03) |
| 11 | The time basis of every rate is explicit | Rates per minute, hour, and day are all common | A default assumed silently | Name the parameter with its time unit |
| 12 | The indexation of every indexed quantity is explicit | eGFR, BSA, and cardiac index all carry references | Dropping the reference | Store the index with the value |
| 13 | The calculation is re-checked when the device changes | Device defaults differ (lb versus kg, mL/h versus mL/min) | A new pump or scale | Commissioning checklist |

The limits of this topic:

- It does not prove an equation is correct. It proves only that the equation is not dimensionally
  impossible.
- It does not cover the Buckingham Pi theorem as a derivation method beyond the statement in section
  3.3.
- It does not cover uncertainty propagation. That is Topic 03.
- It does not cover the clinical validity of the formulas used as examples. Cockcroft-Gault, Mosteller,
  and the A-a gradient are illustrative and have their own literatures.
- It does not replace a pharmacist's or a regulatory reviewer's check of a dose calculation.

## 12. Exercises

### Exercise 1 — Dimension vectors (application)

Assign a dimension vector over M, L, T, N to each of the following: heart rate; cardiac output; stroke
volume; systemic vascular resistance; creatinine; creatinine clearance; dose in mg/kg; dose rate in
µg/kg/min; AUC; volume of distribution; hazard ratio; odds ratio; BMI; BSA; eGFR.

**Deliverable.** A table with the quantity, the vector, and the usual unit.

**Acceptance criteria.** Heart rate T⁻¹; cardiac output L³T⁻¹; stroke volume L³; resistance
M L⁻⁴ T⁻¹ (mmHg·min/L); hazard and odds ratios dimensionless; eGFR L³T⁻¹ with the indexation noted;
all fifteen rows present.

**Time.** 30 minutes.

### Exercise 2 — Check three equations (application)

Check each for dimensional homogeneity and state the unit contract:

1. Clearance = dose/AUC, with dose in mg and AUC in mg·h/L.
2. Velocity = distance/time, with distance in m and time in s.
3. Dose rate = concentration × flow, with concentration in mg/mL and flow in mL/h.

**Deliverable.** Three checks with the dimension of each side and the resulting unit.

**Acceptance criteria.** Clearance L³T⁻¹ in L/h; velocity L T⁻¹ in m/s; dose rate M T⁻¹ in mg/h; each
check shows the cancellation explicitly.

**Time.** 25 minutes.

### Exercise 3 — Locate the inconsistent term (critique)

An oxygen content calculation uses haemoglobin 134 g/L with the coefficient 1.34 mL/g and PaO₂
92 mmHg with the coefficient 0.003 mL/(dL·mmHg). Identify the inconsistent term, compute the wrong
result, and compute the correct result.

**Deliverable.** A two-line calculation for each version and a one-sentence diagnosis.

**Acceptance criteria.** Wrong bound term about 174.2 mL/dL (179.56 before the saturation factor) versus correct
17.4173 mL/dL; the diagnosis names the g/L versus g/dL mismatch; the correct total is 17.69 mL/dL.

**Time.** 20 minutes.

### Exercise 4 — Derive the contract of a constant (calculation)

Derive the units of the constant 72 in Cockcroft-Gault, given that the output is mL/min, age is in
years, weight in kg, and creatinine in mg/dL. Then show what happens if weight is entered in pounds.

**Deliverable.** A dimension equation for the constant and a numerical demonstration.

**Acceptance criteria.** The constant is shown to be unit-carrying, not dimensionless; the pound
demonstration gives 161.07 mL/min versus 73.06 mL/min; the ratio 2.205 is identified as the lb/kg
factor.

**Time.** 30 minutes.

### Exercise 5 — Chain conversion (calculation)

Convert a dose rate of 5 µg/kg/min to mL/h for a 70 kg patient receiving a solution of 4 mg/mL.

**Deliverable.** A four-link unit chain with every factor written as a fraction.

**Acceptance criteria.** 5 µg/kg/min × 70 kg = 350 µg/min; = 0.35 mg/min; ÷ 4 mg/mL = 0.0875 mL/min;
× 60 = 5.25 mL/h; each link shows the cancelled unit.

**Time.** 25 minutes.

### Exercise 6 — Reconstruct a formula (design)

A pharmacokineticist tells you that exposure (AUC) depends on dose and clearance. Using only
dimensions, derive the relationship among dose (mg), clearance (L/h), and AUC (mg·h/L), and state
which variable must be solved for in a dose-adjustment calculation.

**Deliverable.** A dimensional derivation and a one-paragraph explanation.

**Acceptance criteria.** AUC = dose/clearance; the dimension check shows mg/(L/h) = mg·h/L; the
explanation states that dose is solved as clearance × target AUC.

**Time.** 25 minutes.

### Exercise 7 — Detect an infusion programming error (application)

An order reads 0.1 µg/kg/min for an 80 kg patient. The bag contains 1 mg in 50 mL. The pump is
programmed at 4 mL/h. Compute the correct rate, the delivered dose at the programmed rate, and the
factor by which the patient is under- or over-dosed.

**Deliverable.** Three values and a recommendation.

**Acceptance criteria.** Concentration 20 µg/mL; correct rate 0.4 mL/min = 24 mL/h; delivered dose at
4 mL/h is 0.0167 µg/kg/min; the patient receives one sixth of the ordered dose; the recommendation is
to reprogram and verify against the bag label.

**Time.** 30 minutes.

### Exercise 8 — Audit a spreadsheet (design)

You are given a budget model with columns: admissions per month, average length of stay in days, cost
per bed-day, and total cost. The model computes total cost as admissions × length of stay × cost per
bed-day, then divides by 12 and labels the result "annual cost per bed."

Identify the dimensional error, state what each column's unit is, write the correct formula, and
propose two checks that would have caught the error.

**Deliverable.** A one-page audit with the unit table, the corrected formula, and two checks.

**Acceptance criteria.** The product has dimension cost per month, not cost per bed; dividing by 12
converts months to years only if the admission count is monthly; the corrected formula is stated
explicitly; the checks include a unit-label test and a plausibility bound.

**Time.** 40 minutes.

## 13. Self-check answers

### Answer to Exercise 1

| Quantity | Dimension | Usual unit |
|---|---|---|
| Heart rate | T⁻¹ | beats/min |
| Cardiac output | L³T⁻¹ | L/min |
| Stroke volume | L³ | mL |
| Systemic vascular resistance | M L⁻⁴ T⁻¹ | mmHg·min/L |
| Creatinine | M L⁻³ | mg/dL |
| Creatinine clearance | L³T⁻¹ | mL/min |
| Dose | M | mg |
| Dose per kg | dimensionless (M/M) | mg/kg |
| Dose rate per kg | T⁻¹ | µg/kg/min |
| AUC | M L⁻³ T | mg·h/L |
| Volume of distribution | L³ | L |
| Hazard ratio | 1 | dimensionless |
| Odds ratio | 1 | dimensionless |
| BMI | M L⁻² | kg/m² |
| BSA | L² | m² |
| eGFR | L³T⁻¹ indexed | mL/min/1.73 m² |

The dose per kg row is the one that surprises people: mass divided by mass is dimensionless. The
quantity is still useful because it is a normalised dose, but its dimension is 1, which is why a dose
rate per kg has dimension T⁻¹ rather than M T⁻¹.

### Answer to Exercise 2

```text
1. Clearance = dose/AUC
   dim(lhs) = L3 T-1
   dim(rhs) = M / (M L-3 T) = L3 T-1
   Equal. Output in L/h when dose is mg and AUC is mg*h/L.

2. Velocity = distance/time
   dim(lhs) = L T-1
   dim(rhs) = L / T = L T-1
   Equal. Output in m/s.

3. Dose rate = concentration x flow
   dim(lhs) = M T-1
   dim(rhs) = (M L-3) x (L3 T-1) = M T-1
   Equal. Output in mg/h.
```

All three are dimensionally consistent, and in each case the output unit follows from the cancellation
rather than from a convention.

### Answer to Exercise 3

```text
Wrong version, haemoglobin in g/L used with a coefficient calibrated for g/dL:
  bound term = 1.34 x 134 x 0.97 = 174.18 mL/dL
  dissolved  = 0.003 x 92 = 0.276 mL/dL
  total      = 174.46 mL/dL = 1,744.6 mL/L

Correct version, haemoglobin converted to g/dL:
  134 g/L = 13.4 g/dL
  bound term = 1.34 x 13.4 x 0.97 = 17.4173 mL/dL
  total      = 17.6933 mL/dL = 176.93 mL/L
```

The wrong total is about ten times the correct one, which is physiologically impossible because
arterial blood cannot carry 1.7 L of oxygen per litre. The dimensional check exposes the mismatch
because the coefficient 1.34 carries mL/g, so the haemoglobin must be in g per the same volume unit as
the output.

### Answer to Exercise 4

```text
Right-hand side with years treated as a pure number:
  kg / (mg/dL) = kg dL / mg

Output required: mL/min.

The constant 72 must therefore carry:
  (kg dL / mg) / (mL/min) = kg dL min / (mg mL)

Numerically, 1 dL = 100 mL, so the constant includes a factor of 100 and a factor related to
the creatinine molar mass and the empirical fit. It is not a pure number, and it is not derivable
from dimensions alone; it is an empirical constant whose units are fixed by the required output.

Pound demonstration:
  (82 x 198) / (72 x 1.4) = 161.07 mL/min
  (82 x 89.8113) / (72 x 1.4) = 73.06 mL/min
  ratio = 2.205, the lb/kg conversion factor.
```

Because the constant is empirical, the formula cannot be made unit-general by adding a conversion. It
can only be used with the units for which it was fitted, or re-derived for another unit set.

### Answer to Exercise 5

```text
5 ug/kg/min x 70 kg = 350 ug/min
350 ug/min x (1 mg / 1000 ug) = 0.35 mg/min
0.35 mg/min / (4 mg/mL) = 0.0875 mL/min
0.0875 mL/min x (60 min / 1 h) = 5.25 mL/h
```

The chain has three unit fractions and one cancellation per link. Writing it as a single expression,

```text
5 ug/kg/min x 70 kg x (1 mg/1000 ug) x (1 mL/4 mg) x (60 min/1 h) = 5.25 mL/h
```

confirms that kg, µg, mg, and min all cancel and that mL/h remains.

### Answer to Exercise 6

```text
AUC = dose / clearance

dim(rhs) = M / (L3 T-1) = M L-3 T
dim(lhs) = M L-3 T
Equal.

For dose adjustment, solve for dose:
  dose = clearance x target AUC
  dim = (L3 T-1) x (M L-3 T) = M
```

The derivation is forced: given only these three quantities, there is one arrangement that produces
the dimension of AUC. This is the practical value of dimensional analysis for reconstructing a
relationship, and it is also its limit: the derivation gives the form but not the proportionality
constant, which must come from data.

### Answer to Exercise 7

```text
Concentration
  1 mg / 50 mL = 0.02 mg/mL = 20 ug/mL

Ordered dose rate
  0.1 ug/kg/min x 80 kg = 8 ug/min

Correct volume rate
  8 ug/min / 20 ug/mL = 0.4 mL/min
  0.4 mL/min x 60 = 24 mL/h

Delivered dose at 4 mL/h
  4 mL/h / 60 = 0.0667 mL/min
  0.0667 mL/min x 20 ug/mL = 1.333 ug/min
  1.333 ug/min / 80 kg = 0.0167 ug/kg/min

Factor
  24 / 4 = 6
```

The patient receives one sixth of the ordered dose. The error is a factor of 6, which is not a power of
ten and therefore does not look like an obvious decimal slip. It arises from a missing time conversion
combined with an arithmetic slip, which is why the four-link chain must be written rather than done
mentally.

### Answer to Exercise 8 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Unit table | 25% | Admissions/month, days, currency/bed-day, currency/month for the product |
| Error identification | 25% | The product is cost per month, not cost per bed; the label is wrong |
| Corrected formula | 25% | Annual cost = admissions/month × 12 × LOS days × cost/bed-day, with units shown |
| Checks | 25% | A unit-label test on every output column and a plausibility bound on cost per bed-day |

A weak audit corrects the arithmetic without identifying that the output label is dimensionally wrong,
which leaves the same error to recur in the next version of the model.

## 14. Interview questions

### Question 1 — "What is dimensional analysis for?"

**Strong answer.** "It is a consistency check. Every additive term and both sides of an equation must
carry the same dimension. If they do not, the equation is wrong regardless of the data. It also
reconstructs formulas from the dimensions of the inputs and gives the units of fitted parameters."

**Weak answer.** "It is for converting units." Conversion is a special case; the check is the general
use.

### Question 2 — "Check this: clearance equals dose divided by AUC."

**Strong answer.** "Dose is mass, AUC is mass times time over volume, so mass divided by mass times
time over volume is volume over time. That is the dimension of clearance. The equation is consistent,
and the output is in L/h if dose is in mg and AUC in mg·h/L."

**Weak answer.** "It looks right because clearance is dose over exposure." The answer should show the
exponent arithmetic, not just the intuition.

### Question 3 — "Why can you not add mg/dL to mmol/L?"

**Strong answer.** "They have the same dimension, mass or amount per volume, but different units and
different bases. Strictly, they are different kinds of quantity: mass concentration versus substance
concentration. To add them you must convert one to the other using the molar mass, and the result is in
one chosen unit."

**Weak answer.** "Because the numbers are different." The issue is the basis, not the magnitude.

### Question 4 — "What is wrong with the Mosteller formula's 3600?"

**Strong answer.** "It is not dimensionless. Under the square root, height times weight has dimension
length times mass, so the square root has dimension L^0.5 M^0.5, not L². The 3600 carries the
conversion from cm·kg to m⁴, which is why the formula requires centimetres and kilograms. Using inches
and pounds produces a number that is dimensionally meaningless."

**Weak answer.** "It is just a constant." Every constant either is dimensionless or carries units, and
this one carries units.

### Question 5 — "A colleague computes creatinine clearance from a weight in pounds. What happens?"

**Strong answer.** "The result is 2.205 times too high, because Cockcroft-Gault expects kilograms and
the constant 72 is calibrated for kilograms. A value of 73 mL/min becomes 161 mL/min, which may lead to
a full dose in a patient who needs a reduced dose. The check is a plausibility bound and a unit-carrying
parameter name."

**Weak answer.** "The result is a bit high." The error is a known factor and can be quantified exactly.

### Question 6 — "How do you convert 5 µg/kg/min to mL/h?"

**Strong answer.** "Multiply by the patient's weight in kg to get µg/min, divide by 1,000 to get
mg/min, divide by the concentration in mg/mL to get mL/min, and multiply by 60 to get mL/h. I write
each step as a fraction equal to 1 so the units cancel and the residual unit is visible."

**Weak answer.** "Multiply by 60 and divide by the concentration." That omits the weight and the
microgram-to-milligram conversion, which are factors of 70 and 1,000.

### Question 7 — "What does it mean when an equation is dimensionally consistent?"

**Strong answer.** "It means the equation could be physically meaningful. It does not mean it is
correct: the coefficients can be wrong, the model can be wrong, and the population can be wrong. It
eliminates one class of error, which is why it is a first check rather than a validation."

**Weak answer.** "It means the equation is correct." Consistency is necessary, not sufficient.

### Question 8 — "Why is a hazard ratio dimensionless but a hazard is not?"

**Strong answer.** "A hazard is an instantaneous event rate with dimension T⁻¹. A hazard ratio divides
one hazard by another, so the time units cancel and the result is a pure number. That is why hazard
ratios can be compared across studies with different follow-up units, while hazards cannot without
conversion."

**Weak answer.** "Because it is a ratio." Many ratios carry units, such as cost per QALY; the
dimensionless status follows from the cancellation, not from being a ratio.

### Question 9 — "How would you catch a unit error in a spreadsheet model?"

**Strong answer.** "Label every column with its unit, write the unit arithmetic for each derived
column, and check that the output label matches the residual unit. Then pick one row and recompute it
by hand. Finally, bound the result: a cost per bed-day outside a plausible range is a unit error until
proven otherwise."

**Weak answer.** "Check that the totals reconcile." Totals can reconcile while every ratio is wrong.

### Question 10 — "What is an indexed quantity and why does the index matter?"

**Strong answer.** "It is a quantity normalised to a standard reference, such as eGFR per 1.73 m² or
cardiac index per m². The index is part of the unit. Dropping it turns the quantity into a different
one and makes values incomparable across body sizes."

**Weak answer.** "It is just a normalisation for convenience." The index changes the dimension of the
reported quantity and therefore its interpretation.

### Question 11 — "You are given a formula with a constant you cannot explain. What do you do?"

**Strong answer.** "I try to derive the constant's units from the required output. If that fails, I
treat the formula as empirical, take the unit contract from the source, and do not generalise it. I
would also test it against a published worked example to confirm I have transcribed the constant
correctly."

**Weak answer.** "I would assume it is a scaling factor." An unexplained constant is an unexplained
unit contract, and assuming it away is how the pound-kilogram error happens.

### Question 12 — "Why did the FDA care about pounds versus kilograms on an infusion pump screen?"

**Strong answer.** "Because the screen did not make the unit unambiguous, so a clinician could enter a
weight in pounds into a field that the pump treated as kilograms. The resulting dose error is a factor
of 2.2. That is a dimensional ambiguity expressed as a human-factors defect, and the FDA's 2010 white
paper documented it among the causes of infusion pump adverse events."

**Weak answer.** "Because clinicians make mistakes." The point is that the interface allowed a
dimensional ambiguity, which is a design defect rather than an individual error.

## 15. Key terms

- **A-a gradient** — alveolar minus arterial oxygen partial pressure, in mmHg.
- **Anion gap** — sodium minus chloride plus bicarbonate, in mmol/L.
- **AUC** — area under the concentration-time curve, dimension M L⁻³ T.
- **Base dimension** — one of L, M, T, I, Θ, N, J.
- **BMI** — body mass index, M L⁻², usually kg/m².
- **BSA** — body surface area, L², usually m².
- **Buckingham Pi theorem** — n quantities with k dimensions reduce to n − k dimensionless groups.
- **Cardiac output** — L³ T⁻¹, usually L/min.
- **Chain conversion** — sequential multiplication by unit fractions.
- **Clearance** — volume cleared per time, L³ T⁻¹.
- **Coherent unit** — a unit with no numerical factor relative to the base.
- **Concentration** — mass or amount per volume.
- **Conversion factor** — a ratio of two units equal to 1.
- **Creatinine clearance** — measured or estimated volume of plasma cleared of creatinine per time.
- **Dimension** — an exponent vector over the base quantities.
- **Dimensional homogeneity** — all terms of an equation share the same dimension.
- **Dimensional exponent** — the exponent of a base dimension in a derived dimension.
- **Dimensionless group** — a product of powers with zero net dimension.
- **Dimensionless quantity** — a quantity with all exponents zero.
- **Dose rate** — mass per mass per time, dimension T⁻¹ after mass cancels.
- **eGFR** — estimated glomerular filtration rate indexed to 1.73 m².
- **Empirical constant** — a fitted constant that often carries units.
- **Equivalent (Eq)** — one mole of charge.
- **Flow** — volume per time, L³ T⁻¹.
- **Hazard** — instantaneous event rate, T⁻¹.
- **Hazard ratio** — ratio of hazards, dimensionless.
- **Indexed quantity** — a quantity normalised to a standard reference.
- **Infusion rate** — volume per time.
- **Molar concentration** — amount of substance per volume.
- **Numerical-value equation** — an equation valid only for specified units.
- **Osmolality** — osmoles per kilogram of solvent.
- **PAO₂** — alveolar oxygen partial pressure, mmHg.
- **Power law** — y = a·x^b, with b dimensionless.
- **Proportionality constant** — the constant in y = k·x, carrying the units of y/x.
- **Quantity calculus** — the algebra of quantities and units treated symbolically.
- **Quantity equation** — an equation among quantities, independent of units.
- **Scale invariance** — independence of a result from the unit of measurement.
- **Similarity** — equality of dimensionless groups between systems.
- **Stroke volume** — L³, usually mL.
- **Systemic vascular resistance** — M L⁻⁴ T⁻¹, usually mmHg·min/L.
- **Transcendental function** — exp, ln, log, or trigonometric; argument must be dimensionless.
- **Unit contract** — the stated input units, output unit, and validity of a formula.
- **Unit equation** — an equation among units alone.
- **Unit fraction** — a conversion factor written as a fraction equal to 1.
- **Universal constant** — a constant of nature with a fixed value.

## 16. Sources

All URLs were accessed on 24 October 2026.

### Quantity calculus and dimensions

- ISO 80000-1:2022, *Quantities and units — Part 1: General*, second edition.
  https://www.iso.org/standard/76921.html — Clause 5 defines the dimension product and the
  same-kind/same-dimension rule; Clause 6 defines quantity equations versus numerical-value equations.
- ISO 80000-2:2019, *Quantities and units — Part 2: Mathematics*.
  https://www.iso.org/standards.html
- JCGM 200:2008, *International Vocabulary of Metrology — Basic and General Concepts and Associated
  Terms (VIM)*. https://www.bipm.org/en/committees/jc/jcgm/publications — entries 1.21 to 1.25.
- BIPM, *The International System of Units (SI Brochure)*, 9th edition, 2019, revised 2026.
  https://www.bipm.org/en/publications/si-brochure
- NIST, *Guide for the Use of the International System of Units (SI)*, NIST SP 811.
  https://www.nist.gov/pml/special-publication-811
- Taylor, B. N., "Quantity Calculus, Fundamental Constants, and SI Units", *Journal of Research of
  the National Institute of Standards and Technology* 123, 123008, 2018.
  https://nvlpubs.nist.gov/nistpubs/jres/123/jres.123.008.pdf

### Units in clinical data

- Regenstrief Institute, *Unified Code for Units of Measure*. https://ucum.org/
- National Library of Medicine, *Table of Example UCUM Codes for Electronic Messaging*, v1.4, 2016.
  https://ucum.nlm.nih.gov/example-UCUM-Codes-v1.4.pdf
- LOINC, *Logical Observation Identifiers Names and Codes*. https://loinc.org/

### Device safety and regulation

- US Food and Drug Administration, *White Paper: Infusion Pump Improvement Initiative*, April 2010.
  https://www.fda.gov/medical-devices/infusion-pumps/white-paper-infusion-pump-improvement-initiative
- US Food and Drug Administration, *Infusion Pump Risk Reduction Strategies for Clinicians*.
  https://www.fda.gov/medical-devices/infusion-pumps/infusion-pump-risk-reduction-strategies-clinicians
- IEC 62304:2006+AMD1:2015, *Medical device software — Software life cycle processes*.
  https://www.iec.ch/
- NASA, *Mars Climate Orbiter Mishap Investigation Board Phase I Report*, 1999.
  https://solarsystem.nasa.gov/resources/776/mars-climate-orbiter-mishap-investigation-board-phase-i-report/

### Clinical formulas used as examples

- Cockcroft, D. W., Gault, M. H., "Prediction of creatinine clearance from serum creatinine",
  *Nephron* 16(1), 31–41, 1976. https://doi.org/10.1159/000180580
- Mosteller, R. D., "Simplified calculation of body-surface area", *New England Journal of Medicine*
  317(17), 1098, 1987. https://doi.org/10.1056/NEJM198710223171717
- Inker, L. A. et al., "New Creatinine- and Cystatin C–Based Equations to Estimate GFR without Race",
  *New England Journal of Medicine* 385, 1737–1749, 2021. https://doi.org/10.1056/NEJMoa2102953

> **Note:** ISO and IEC documents are sold standards; the URLs are product or catalogue pages.
> Summaries here are from publicly available front matter and should be checked against the purchased
> document before use in a regulated process.

## 17. Links

- Up to the module map: [`_index.md`](_index.md)
- Previous topic: [`03_units-and-metrology.md`](03_units-and-metrology.md)
- Next topic: [`05_biological-basics.md`](05_biological-basics.md)
- Mathematical basis of functions and ratios: [`01_mathematical-basics.md`](01-mathematical-basics.md)
- Pharmacokinetic applications of clearance and AUC: [`06_pharmacokinetics.md`](06_pharmacokinetics.md)
- Device and infusion safety: [`08_engineering-and-device-basics.md`](08_engineering-and-device-basics.md)
- Assumption recording: [`10_the-assumptions-ledger.md`](10_the-assumptions-ledger.md)
- Root survey file: [`../../01A_foundations_assumptions_and_units.md`](../../01A_foundations_assumptions_and_units.md)
- Labs: [`../../labs/`](../../labs/) — `01A_unit_conversion.ipynb` covers conversion and dimensional
  checks; `14B_quantization.ipynb` covers resolution in sampled signals.
