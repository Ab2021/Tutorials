# Computing Basics

> Module 01A — Foundations, Assumptions, and Units
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02, 03
> Version 2026.3

## 1. Why this topic exists

Most data defects in biomedical analytics are not statistical. They are representational. A value is
stored in a type that cannot hold it, a date loses its time zone, an identifier is parsed as a number,
a missing value becomes a zero, or a file is read with the wrong encoding. The analysis then proceeds
on a corrupted object, and no amount of modelling recovers the original.

- Ten doses of 0.1 mL are summed and the result is 0.9999999999999999 mL. A reconciliation report
  shows a 0.0000000000000001 mL discrepancy, an engineer spends a day on it, and the real issue is that
  binary floating point cannot represent 0.1 exactly.
- A patient identifier `0012345` is read as an integer and stored as `12345`. It no longer matches the
  source system, and the join silently drops 4% of records.
- A timestamp is recorded without a time zone. A service that runs across a daylight-saving boundary
  computes a 48-hour interval for an event that lasted 47 hours, and a 24-hour readmission window
  includes an extra hour of admissions.
- A missing HbA1c is imputed as zero. The mean HbA1c falls by 2.4 percentage points, and a quality
  dashboard reports a dramatic improvement that is entirely an artefact of the imputation.
- A CSV containing `SEPT2` is opened in a spreadsheet and becomes `2-Sep`. The gene name is destroyed,
  and the error propagates into a supplementary table.
- A timestamp in milliseconds is interpreted as seconds, and every event is placed in the year 58,000.

These are ordinary computing failures, and they are preventable with a small number of disciplines:
typed schemas, explicit parsing, explicit missingness, time-zone-aware timestamps, string identifiers,
and tests with adversarial fixtures.

The topic matters to data scientists because the tooling encourages the failures. A CSV has no types;
a JSON number has no precision guarantee; a spreadsheet guesses; a database driver coerces. The analyst
who does not specify the representation is accepting whatever the tool chose.

The commercial consequence is direct. A pipeline that cannot reconcile to source is a pipeline that
cannot be audited. A registry that loses identifiers cannot be linked. A model trained on a corrupted
date cannot be deployed across a daylight-saving boundary. Each failure is cheap to prevent and
expensive to diagnose.

> **Note:** This topic uses Python for illustration because it is the program language of the labs.
> The concepts apply to R, SQL, Java, and any other environment. The examples are runnable with the
> standard library.

## 2. Learning outcomes

By the end of this topic you can:

1. **Explain** why binary floating point cannot represent most decimal fractions, and choose an
   appropriate representation for money, doses, and measurements. Assessed in Exercises 1 and 2.
2. **Compare** floating-point values correctly using a tolerance, and explain why equality comparison
   is a defect. Assessed in Exercise 2.
3. **Distinguish** missing, unknown, not applicable, and zero, and choose an explicit representation
   for each. Assessed in Exercise 3.
4. **Represent** timestamps as an instant plus a zone, handle daylight-saving transitions and ambiguous
   local times, and compute durations correctly. Assessed in Exercise 4.
5. **Choose** identifier representations that survive joins, and explain why identifiers must be
   strings. Assessed in Exercise 5.
6. **Select** a file format for a stated use, and state its encoding, delimiter, quoting, and schema
   behaviour. Assessed in Exercise 6.
7. **Audit** a pipeline for precision, time-zone, encoding, and missing-data defects, and write
   adversarial tests that reproduce them. Assessed in Exercise 7.
8. **Design** a data contract for a registry, including types, units, nullability, and versioning.
   Assessed in Exercise 8.

## 3. The core mental model

A value passes through three representations: the machine representation, the storage representation,
and the display representation. Each can lose information, and the losses are different.

```text
   SOURCE                MACHINE                 STORAGE                DISPLAY
   "7.20"       ->   float64 7.199999...   ->  CSV "7.2"        ->   "7.2 mmol/L"
   "0012345"    ->   int 12345              ->  "12345"          ->   "12345"
   "2026-03-08   ->  naive datetime         ->  "2026-03-08       ->  "03/08/26"
    02:30"            (no zone)                 02:30:00"
```

The rule that prevents most defects: **parse at the boundary, store a representation that preserves the
information, and convert only for display.** The boundary is where the source's format is known; after
that, every conversion is a guess.

### 3.1 The four type families that cause most defects

| Family | Failure | Correct representation |
|---|---|---|
| Decimal fractions | Binary floating point cannot represent 0.1 | Decimal or integer minor units |
| Identifiers | Leading zeros, large integers, non-numeric characters | String, never numeric |
| Time | Naive timestamps, DST, partial dates | Instant plus zone, or explicit local date |
| Missing | Null coerced to zero or to a sentinel | Explicit null with a reason code |

Each family has a characteristic failure signature. Decimal errors appear as tiny discrepancies in
sums and equality tests. Identifier errors appear as failed joins and dropped rows. Time errors appear
as off-by-one-hour or off-by-one-day shifts at boundaries. Missing-data errors appear as implausible
means and as counts that do not reconcile.

### 3.2 Binary floating point in one paragraph

IEEE 754 double precision stores a sign, an 11-bit exponent, and a 52-bit fraction, giving 53 bits of
significand precision. Most decimal fractions, including 0.1, 0.2, and 2.675, have no exact binary
representation, so they are stored as the nearest representable value. Arithmetic on those
approximations produces results that differ from the decimal result by a tiny amount, and equality
comparison fails. The precision limit is about 15 to 17 significant decimal digits; integers above
2^53 = 9,007,199,254,740,992 cannot all be represented exactly.

### 3.3 Time is an interval, not a string

A timestamp is a point on the timeline (an instant) plus a rule for displaying it in a locality (a
zone). A string such as `2026-03-08 02:30:00` is neither: it has no offset, so it denotes a different
instant in every zone, and in some zones it denotes no instant at all because the local time does not
exist.

The correct model:

```text
   instant (UTC)  +  zone (IANA identifier)  ->  local time (for display)
   local time + zone -> instant (only if the local time is unambiguous)
```

A daylight-saving spring-forward creates local times that do not exist; a fall-back creates local times
that occur twice. Both are real in clinical data, and both break naive arithmetic.

### 3.4 Missingness is a type, not a value

A missing value means "we do not have this measurement." A zero means "we measured zero." A sentinel
such as −999 means "someone encoded missing as a number, and now the mean is wrong." The representation
must distinguish:

```text
   null        no value available
   unknown     a value exists but was not recorded
   not applicable  the question does not apply to this record
   zero        a measured zero
   below LOD   measured, but below the limit of detection
```

These are different states with different analytical consequences, and collapsing them into one is a
data-modelling decision that cannot be reversed later.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Binary floating point | A number represented as sign × significand × 2^exponent | Assuming decimal exactness | IEEE 754 |
| Double precision | 64-bit float with 53-bit significand | Assuming unlimited precision | IEEE 754 binary64 |
| Significand | The significant digits of a floating-point number | Confusing with significant figures | bits |
| Machine epsilon | The gap between 1 and the next representable number | Confusing with the smallest positive number | dimensionless |
| Round-off error | Error from representing a value in finite precision | Confusing with measurement error | unit |
| Catastrophic cancellation | Loss of precision when subtracting nearly equal numbers | Ignoring it in differences | unit |
| Decimal type | A base-10 representation with a fixed precision | Assuming it is exact for division | decimal |
| Integer minor units | Money stored as integer cents | Storing money as float | cents |
| Tolerance | An allowed difference for comparison | Using exact equality | unit |
| Equality comparison | Testing whether two floats are identical | Using it on computed values | boolean |
| NaN | Not a number, a floating-point sentinel | Treating it as missing in a database | IEEE 754 |
| Infinity | A floating-point value beyond the range | Treating it as a large number | IEEE 754 |
| Overflow | A value exceeding the representable range | Ignoring it in integer arithmetic | none |
| Underflow | A value below the smallest representable positive | Ignoring it in tiny measurements | none |
| Type coercion | Implicit conversion between types | Assuming it is lossless | none |
| Parsing | Converting a string to a typed value | Using a locale-dependent parser | none |
| Locale | Regional formatting rules for numbers and dates | Assuming a decimal point is "." | language tag |
| Encoding | The mapping from characters to bytes | Assuming UTF-8 | UTF-8, UTF-16 |
| BOM | Byte order mark at the start of a file | Ignoring it in the first column name | bytes |
| Delimiter | The character separating fields | Assuming a comma | character |
| Quoting | Wrapping a field to protect delimiters | Ignoring embedded newlines | character |
| Escape | A sequence representing a special character | Double-escaping | character |
| CSV injection | A formula in a field executed by a spreadsheet | Treating CSV as inert | none |
| Schema | The declared types and constraints of a dataset | Inferring types from the first rows | none |
| Data contract | An agreed schema plus semantics and quality rules | Treating it as documentation only | none |
| Null | The absence of a value | Treating it as zero or empty string | none |
| Sentinel value | A number used to encode missing | Using −999 or 0 | unit |
| Not applicable | The question does not apply | Encoding it as null | none |
| Imputation | Filling a missing value with an estimate | Treating the estimate as observed | unit |
| Complete-case analysis | Analysing only records with no missing values | Assuming it is unbiased | none |
| Timestamp | A point in time with an offset or zone | Storing a local time without a zone | ISO 8601 |
| Instant | A point on the UTC timeline | Confusing with a local time | ISO 8601 |
| Time zone | A rule for offset from UTC, including DST | Confusing with a fixed offset | IANA identifier |
| UTC | Coordinated Universal Time | Treating it as a local time | ISO 8601 |
| Offset | The difference between local time and UTC | Assuming it is constant | ±HH:MM |
| DST | Daylight saving time | Assuming every day has 24 hours | none |
| Ambiguous local time | A local time that occurs twice at fall-back | Parsing it without a rule | none |
| Nonexistent local time | A local time skipped at spring-forward | Parsing it without a rule | none |
| Leap second | A one-second adjustment to UTC | Ignoring it in precise timing | s |
| Epoch | A reference instant for numeric timestamps | Confusing seconds and milliseconds | s or ms |
| ISO 8601 | The international date and time format | Using a locale format | ISO 8601 |
| RFC 3339 | A profile of ISO 8601 for internet timestamps | Omitting the offset | RFC 3339 |
| ISO week | A week numbered from the first Thursday of the year | Confusing with the calendar week | ISO 8601 |
| Partial date | A date with missing components, such as a year only | Parsing it as the first of the month | none |
| Identifier | A label that identifies a record | Storing it as a number | string |
| Surrogate key | A generated identifier with no meaning | Confusing it with a natural key | integer or UUID |
| Natural key | An identifier derived from the data | Assuming it is stable | string |
| UUID | A 128-bit identifier | Assuming it is human-readable | RFC 4122 |
| Hash | A fixed-length digest of a value | Assuming it is collision-free | bytes |
| Collision | Two different inputs with the same hash | Ignoring the birthday bound | probability |
| Join key | The field used to combine records | Joining on a non-unique field | none |
| Encoding normalisation | Converting to a canonical Unicode form | Ignoring accented characters | NFC |
| PII | Personally identifiable information | Assuming de-identification is complete | none |
| De-identification | Removing or transforming identifiers | Assuming it is irreversible | none |
| Safe harbour | A US HIPAA de-identification method | Assuming it covers re-identification risk | 18 identifiers |
| Provenance | The origin and processing history of a record | Storing it outside the data | none |
| Immutability | The property that a record cannot be changed | Overwriting raw data | none |
| Reproducibility | The ability to regenerate a result | Assuming a notebook is enough | none |

Three definitions deserve emphasis.

**Null is not a value.** In SQL, `NULL` is a marker for the absence of a value, and comparisons with it
return unknown rather than true or false. In pandas, `NaN` is a float sentinel that propagates through
arithmetic. In JSON, `null` is a literal. The three behave differently, and a pipeline that moves data
between them can turn a missing value into a string "nan" or into a zero.

**ISO week is not calendar week.** ISO 8601 numbers weeks from the week containing the first Thursday
of the year, so 1 January can belong to the last week of the previous ISO year. A weekly report that
groups by calendar day-of-year will not match a report that groups by ISO week.

**De-identification is not anonymisation.** Removing the 18 HIPAA identifiers reduces re-identification
risk but does not eliminate it, especially with dates, geography, and rare diagnoses. The distinction
matters for governance and for the validity of a data-sharing agreement.

## 5. The mechanics, step by step

The procedure defines a data contract and implements it so that the representation survives the
pipeline.

### Step 1 — Write the data contract before the code

**Input.** A source system and a use case.

**Output.** A schema with names, types, units, nullability, and allowed values.

**Reasoning.** The contract is the specification the code implements. Without it, types are inferred
from the first rows, and the inference is wrong as soon as a row with a missing value or a large
integer appears. The contract is also the artefact a reviewer can check.

**Branch point.** If the source cannot supply a field the contract requires, the field is nullable with
a reason code, not silently defaulted.

### Step 2 — Classify each field by type family

**Input.** The field list.

**Output.** A type assignment: decimal, integer, identifier, date, timestamp, duration, code, text,
boolean, or missing.

**Reasoning.** The family determines the representation. Money and doses are decimal, not float.
Identifiers are strings. Timestamps are instants with zones. Codes are strings even when they look
numeric (ICD-10 code `E11.9`, LOINC code `2345-7`). Booleans are not 0/1 integers unless the contract
says so.

**Branch point.** If a field is ambiguous (a number that is sometimes an identifier), treat it as a
string. The cost of a string is storage; the cost of a lost leading zero is a broken join.

### Step 3 — Choose the storage representation

**Input.** The type family.

**Output.** A physical type and format.

**Reasoning.** Decimal values are stored as `DECIMAL`/`NUMERIC` in a database, as `Decimal` in Python,
or as integer minor units. Identifiers are stored as text with a fixed length or as UUIDs. Timestamps
are stored as UTC instants with a separate zone field for display. The storage type is a decision that
must be documented.

**Branch point.** If the storage is a CSV, the type is not stored at all; the contract must be supplied
by the reader.

### Step 4 — Parse explicitly at the boundary

**Input.** The raw source.

**Output.** Typed values or a parse error.

**Reasoning.** Explicit parsing means specifying the format, the locale, and the error behaviour. A
date parser that accepts any format will interpret `03/04/2026` as March 4 in one locale and April 3 in
another. A number parser that accepts commas will interpret `1,234` as 1234 in one locale and as 1.234
in another.

**Branch point.** If a value fails to parse, route it to a quarantine table with the raw string. Do not
drop it silently.

### Step 5 — Validate ranges and allowed values

**Input.** The parsed values.

**Output.** A validation report.

**Reasoning.** Range checks catch unit errors, parsing errors, and sentinel values. A glucose of 700 is
plausible in mg/dL and impossible in mmol/L; a systolic pressure of 12 is implausible in mmHg; a
temperature of 98.6 is plausible in °F and fatal in °C. Allowed-value checks catch coding errors such
as `M`/`F`/`U` versus `male`/`female`/`unknown`.

**Branch point.** If a value is out of range, quarantine it. Do not clip it, because clipping destroys
the evidence of the defect.

### Step 6 — Represent missingness explicitly

**Input.** The raw missing markers.

**Output.** A null with a reason code.

**Reasoning.** Sources encode missing as empty string, `NA`, `NULL`, `-999`, `.`, or a blank. Each must
be mapped to a null, and the reason should be recorded where it is known (not collected, collected but
lost, not applicable). A null that is actually a zero must not be mapped to null; a zero that is
actually missing must not be mapped to zero.

**Branch point.** If the source uses a sentinel that is also a plausible value (0 for a lab result),
the mapping requires source documentation. Do not infer.

### Step 7 — Distinguish missing, unknown, and not applicable

**Input.** The nulls.

**Output.** A reason-coded missingness field.

**Reasoning.** A missing lab because the test was not ordered is different from a missing lab because
the result was lost, and both are different from a question that does not apply. The distinction
determines whether imputation is appropriate and whether the missingness is informative.

**Branch point.** If the reason is unknown, record it as unknown. Do not assign the most convenient
reason.

### Step 8 — Choose the missing-data handling

**Input.** The missingness pattern.

**Output.** A method: complete-case, imputation, model-based, or a sensitivity analysis.

**Reasoning.** Complete-case analysis is valid only under a missingness assumption. Mean imputation
distorts variance and covariance. Multiple imputation preserves uncertainty under a stated model.
The choice must be justified by the mechanism, not by convenience.

**Branch point.** If the missingness is likely informative, report the complete-case and imputed
estimates side by side and state the likely direction of the bias.

### Step 9 — Represent an instant with a zone

**Input.** A source timestamp.

**Output.** A UTC instant plus a zone identifier.

**Reasoning.** The instant is unambiguous; the zone is needed to reconstruct the local time for
display and for clinical interpretation (a drug dose at 08:00 local is not the same as 08:00 UTC). 
Storing only a local time loses the information needed to place the event on the timeline.

**Branch point.** If the source provides a local time and no zone, the zone must be supplied from the
facility record. If it cannot be, the timestamp is ambiguous and must be flagged.

### Step 10 — Handle DST transitions

**Input.** Local times near a transition.

**Output.** A resolved instant or an ambiguity flag.

**Reasoning.** At spring-forward, local times in the skipped hour do not exist; at fall-back, local
times in the repeated hour occur twice. A parser must either resolve them with a stated rule (the
first occurrence, the second occurrence, or an error) or flag them. Silent resolution is a defect.

**Branch point.** If the data contain ambiguous times and the rule is unknown, exclude them from
duration calculations or report both possibilities.

### Step 11 — Compute durations on the instant, not the local time

**Input.** Two timestamps.

**Output.** A duration.

**Reasoning.** A duration is the difference between instants. Subtracting local times across a DST
boundary gives 24 hours for a 23-hour day, or 48 hours for a 47-hour interval. The error is one hour
per transition, which is small for a monthly report and decisive for a 24-hour readmission window.

**Branch point.** If the duration is defined in local calendar days (for example, "three days of
therapy"), the calculation is a calendar arithmetic problem, not an instant difference. State which.

### Step 12 — Handle partial and imprecise dates

**Input.** A date with missing components.

**Output.** A partial date representation.

**Reasoning.** "2026-02" is a month, not 1 February. Parsing it as the first of the month introduces a
systematic bias toward early dates and makes durations wrong by up to 30 days. A partial date should
be stored as a range or with an explicit precision field.

**Branch point.** If the analysis requires a full date, use the midpoint or a stated assumption, and
report the sensitivity of the result.

### Step 13 — Fix the timestamp unit and epoch

**Input.** Numeric timestamps.

**Output.** A unit and an epoch, stated.

**Reasoning.** Seconds and milliseconds differ by a factor of 1,000; a millisecond value interpreted as
seconds places the event tens of thousands of years in the future. Some systems use a different epoch
(Windows file time, Excel serial dates). The unit and epoch must be part of the contract.

**Branch point.** If the unit is ambiguous, check the magnitude against a known event date. A value of
about 1.7 × 10⁹ is seconds; about 1.7 × 10¹² is milliseconds.

### Step 14 — Generate stable identifiers

**Input.** The record.

**Output.** A key that is unique and stable.

**Reasoning.** A surrogate key (a UUID or a sequence) is stable if it is never reused. A natural key
(medical record number, NHS number) can be reused, reassigned, or merged. A hash of identifying fields
is stable only if the fields never change and the hash is salted consistently.

**Branch point.** If the source reuses identifiers, the key must include the source system and the
assignment period. A bare identifier is not unique across systems or time.

### Step 15 — Store identifiers as strings

**Input.** The identifier.

**Output.** A string, with leading zeros preserved.

**Reasoning.** Numeric identifiers lose leading zeros, lose precision above 2^53, and change sort
order. A `BIGINT` column cannot store `0012345` and `12345` as distinct values, and a float cannot store
a 19-digit identifier exactly. Strings preserve the representation.

**Branch point.** If an identifier is genuinely numeric and used for arithmetic, it is a quantity, not
an identifier. Store both if needed.

### Step 16 — Normalise text

**Input.** Free text and coded values.

**Output.** A canonical form.

**Reasoning.** Unicode has multiple representations of the same character (composed and decomposed
accents), case differences, and whitespace variants. Normalisation to NFC, case folding where
appropriate, and whitespace trimming prevent joins from failing on visually identical strings.

**Branch point.** If the text is a code, do not case-fold blindly; some coding systems are
case-sensitive.

### Step 17 — Choose the file format

**Input.** The use case.

**Output.** A format with a stated schema behaviour.

**Reasoning.** CSV is universal and untyped; Parquet is typed, compressed, and columnar; JSON is
flexible and imprecise for large numbers; HL7 v2 and FHIR are the clinical interchange formats. The
format determines what can be preserved: a CSV cannot preserve types, and a JSON number cannot preserve
an integer above 2^53 in every parser.

**Branch point.** If the data will be read by a spreadsheet, the format must protect against formula
injection and date conversion. CSV alone does not.

### Step 18 — Specify the encoding

**Input.** The file.

**Output.** An encoding declaration and a decoding rule.

**Reasoning.** UTF-8 is the default for new systems, but legacy files may be Latin-1 or Windows-1252,
and a file may begin with a byte order mark that corrupts the first column name. The encoding must be
declared and the reader must be told to use it.

**Branch point.** If the encoding is unknown, detect it on a sample, validate the decoded output, and
record the detection as an assumption.

### Step 19 — Handle delimiters, quoting, and embedded newlines

**Input.** A delimited file.

**Output.** Correctly parsed fields.

**Reasoning.** A comma inside a quoted field, a newline inside a quoted field, and an escaped quote are
all valid CSV. A parser that splits on commas corrupts the record. The quoting rules must be applied by
a real parser, not by string splitting.

**Branch point.** If the file has inconsistent quoting, the parse will fail or silently misalign
columns. Validate the field count per row.

### Step 20 — Guard against formula injection

**Input.** Text fields destined for a spreadsheet.

**Output.** Escaped or rejected values.

**Reasoning.** A cell beginning with `=`, `+`, `-`, or `@` can be interpreted as a formula by a
spreadsheet. This is a security issue when the content is untrusted, and a data-integrity issue when
the content is a legitimate string such as `-1` or `=1+1`. The export must escape or quote the value,
and the import must not execute it.

**Branch point.** If the file will be opened by a spreadsheet, prefer a format that carries types, or
pre-escape the vulnerable fields.

### Step 21 — Prevent silent date conversion in spreadsheets

**Input.** Text that looks like a date.

**Output.** A preserved string.

**Reasoning.** Spreadsheets convert `SEPT2` to `2-Sep`, `MARCH1` to `1-Mar`, and `2310009E13` to
`2.31E+13`. The conversion is a locale-dependent date or number parse applied to a text field, and it
is destructive. Gene names, accession numbers, and identifiers are the usual victims.

**Branch point.** If a spreadsheet must be used, pre-escape the fields or use a tool designed for the
purpose. The safest rule is not to open raw data in a spreadsheet.

### Step 22 — Validate joins

**Input.** Two tables and a join key.

**Output.** A join with a reported match rate.

**Reasoning.** A join that matches 96% of rows may be correct or may indicate a type mismatch. The
match rate must be reported and compared with an expectation. A silent inner join that drops 4% of
records biases every downstream analysis.

**Branch point.** If the match rate is below expectation, investigate the key type, the normalisation,
and the identifier reuse before accepting the result.

### Step 23 — Log provenance

**Input.** Every transformation.

**Output.** A record of source, time, code version, and parameters.

**Reasoning.** Provenance is what allows a result to be reproduced and a defect to be traced. A
pipeline without provenance cannot answer "which version produced this number", and a regulator will
ask.

**Branch point.** If provenance cannot be stored with the data, store it in a manifest and link the
manifest to the dataset version.

### Step 24 — Test with adversarial fixtures

**Input.** The contract.

**Output.** A test suite with edge cases.

**Reasoning.** The tests must include: the smallest and largest values, a leading-zero identifier, a
timestamp at a DST transition, a missing value, a sentinel, a Unicode string, a quoted comma, a
formula string, and a value above 2^53. These are the cases that break naive code.

**Branch point.** If a test fails, fix the code, not the test. The fixture represents a real source.

### Step 25 — Make the environment reproducible

**Input.** The code and dependencies.

**Output.** A pinned environment and a reproducible run.

**Reasoning.** A result that depends on an unpinned library version is not reproducible. The
environment, the data version, and the code version must be recorded together.

**Branch point.** If the environment cannot be pinned, record the versions actually used and state the
limitation.

### Step 26 — Document the data dictionary

**Input.** The contract and the implementation.

**Output.** A dictionary entry per field.

**Reasoning.** The dictionary records the name, type, unit, allowed values, missingness semantics,
source, and transformation. It is the artefact that lets a new analyst use the data without repeating
the discovery process.

**Branch point.** If a field's semantics are unknown, mark them unknown. A dictionary that guesses is
worse than one that admits a gap.

### 5.1 The branch map

```text
Raw value
  |
  +-- decimal fraction? -----------> Decimal or integer minor units
  |
  +-- identifier? -----------------> string, fixed length, never numeric
  |
  +-- timestamp without zone? -----> supply zone or flag ambiguous
  |
  +-- missing sentinel? ------------> null with a reason code
  |
  +-- value above 2^53? ------------> string or 128-bit integer
  |
  +-- text in a spreadsheet? -------> escape formulas and dates
  |
  +-- join match rate low? ---------> check type, normalisation, reuse
```

### 5.2 The four checks that catch most computing defects

1. **Print the type and the raw bytes of a suspicious value.** The display hides the representation.
2. **Compare floats with a tolerance, never with equality.**
3. **Check every timestamp for an offset.** A timestamp without one is a local time, not an instant.
4. **Reconcile the row count at every join and filter.** A silent drop is the most common defect.

## 6. Worked example

A diabetes registry ingests a daily extract from a hospital prescribing system: basal insulin doses in
millilitres, a per-patient identifier, administration timestamps in local time, HbA1c results, and a
free-text note. The registry reconciles the extract to the source, links to a primary care dataset, and
computes a quarterly summary.

The case demonstrates the seven defects that the pipeline must prevent: binary floating point, rounding,
large integers, time zones, ISO weeks, missing versus zero, and CSV interpretation.

### 6.1 Binary floating point in a dose reconciliation

```text
Ten basal insulin doses of 0.1 mL each.

  naive sum: 0.1 + 0.1 + ... (ten times)
  result:    0.9999999999999999 mL

  expected:  1.0 mL
  difference: 1.1e-16 mL
```

The difference is a quadrillionth of a millilitre and is clinically irrelevant. It is not irrelevant
to the reconciliation: a test that asserts `sum == 1.0` fails, and a downstream rule that checks
"total equals prescribed" reports a discrepancy that no clinician can act on.

```text
Representation check
  0.1 in binary64 = 0.1000000000000000055511151231257827...
  0.2 in binary64 = 0.2000000000000000111022302462515654...
  sum             = 0.3000000000000000444089209850062616...
  displayed       = 0.30000000000000004

  Decimal("0.1") + Decimal("0.2") = 0.3 exactly
```

The fix is to store doses as integer microlitres or as `Decimal`, and to compare with a tolerance if
floats are unavoidable. The tolerance should be smaller than the smallest clinically meaningful
difference and larger than the accumulated representation error.

### 6.2 Money and rounding

```text
A unit cost of $0.10 for three items:
  float:   0.10 x 3 = 0.30000000000000004
  decimal: 0.10 x 3 = 0.30

Rounding 2.675 to two decimal places:
  Python round(2.675, 2) = 2.67    (binary representation is slightly below 2.675)
  Decimal half-up          = 2.68

Banker's rounding, as used by Python's round:
  round(0.5) = 0
  round(1.5) = 2
  round(2.5) = 2
```

Three different rounding results for the same decimal input are a governance problem. A financial
report must state the rounding mode: half-up, half-even, or truncation. A clinical report must state
whether a value is rounded for display only or rounded before a threshold comparison, because the two
can disagree at the boundary.

### 6.3 Identifiers above 2^53

```text
An identifier of 9,007,199,254,740,993 (2^53 + 1):
  stored as int:   9007199254740993
  stored as float: 9007199254740992.0  (the last bit is lost)

  float(2^53 + 1) = 9007199254740992.0
  1e16 + 1        = 1e+16
```

The identifier is now shared with its neighbour. If two patients have identifiers 2^53 and 2^53 + 1,
the float representation cannot distinguish them, and a join will merge the records. The fix is to store
identifiers as strings or 128-bit integers, never as floats.

A leading-zero identifier has a simpler version of the same failure:

```text
  "0012345" -> int 12345 -> str "12345"
  The string no longer matches the source.
```

### 6.4 Durations across a daylight-saving transition

```text
Start: 2026-03-07 12:00 at UTC-05:00 (EST)
End:   2026-03-09 12:00 at UTC-04:00 (EDT)

Aware difference:
  start instant = 2026-03-07 17:00 UTC
  end instant   = 2026-03-09 16:00 UTC
  difference    = 47.0 h

Naive difference (local times, no zone):
  2026-03-09 12:00 - 2026-03-07 12:00 = 48.0 h
```

The event lasted 47 hours, not 48. A 48-hour rule applied to the naive difference would exclude the
last hour of events. The error appears once a year in each direction and is easy to miss because most
days have 24 hours.

The spring-forward also creates local times that do not exist: on 2026-03-08, the clock jumps from
02:00 to 03:00, so `02:30` is not a valid local time. The fall-back creates local times that occur
twice. A parser must state how it resolves both cases.

### 6.5 ISO week versus calendar week

```text
2027-01-01 is a Friday.
  ISO year 2026, week 53, weekday 5.

A quarterly report that groups by ISO week places this date in 2026-W53,
not in 2027-W01.
```

A report that groups by calendar year and a report that groups by ISO week will disagree on the first
and last days of the year. The disagreement is small in volume and large in a reconciliation, because
the two reports will not sum to the same annual total.

### 6.6 Missing versus zero

```text
HbA1c values: 7.2, missing, 6.8, missing, 7.5, 6.9

Mean of present values = (7.2 + 6.8 + 7.5 + 6.9)/4 = 28.4/4 = 7.10%
Mean treating missing as zero = (7.2 + 0 + 6.8 + 0 + 7.5 + 6.9)/6 = 28.4/6 = 4.73%
Missing fraction = 2/6 = 33.33%
```

The zero-imputed mean is 4.73%, which would be reported as excellent glycaemic control. The true mean
of the observed values is 7.10%. The defect is not the missingness; it is the representation of
missingness as a number that participates in arithmetic.

If the missing values are informative (sicker patients are less likely to have a recent HbA1c), the
complete-case mean of 7.10% is also biased, and the direction depends on why the test was not ordered.
The pipeline must record the reason, not only the null.

### 6.7 CSV quoting and formula injection

The export writes the following rows.

```text
id,name,note
1,"Smith, John",follow-up
2,Okafor,"line1
line2"
3,=1+1,formula
```

Three properties are visible. The comma inside `Smith, John` forces quoting. The newline inside the
note forces quoting and spans two physical lines. The field `=1+1` is written unquoted, and a
spreadsheet opening the file will evaluate it as a formula.

The first two are handled by a real CSV parser and writer. The third is a security and integrity issue:
the export must escape fields beginning with `=`, `+`, `-`, or `@`, and the import must not execute
them. The same mechanism is why a gene name such as `SEPT2` can become a date when a file is opened in
a spreadsheet.

### 6.8 Timestamp units

```text
2026-10-24T00:00:00+00:00
  epoch seconds      = 1,792,800,000
  epoch milliseconds = 1,792,800,000,000

If the millisecond value is interpreted as seconds:
  the date would be in the year ~58,000, which is out of range on many platforms.
```

A value near 1.7 × 10⁹ is seconds; a value near 1.7 × 10¹² is milliseconds. The magnitude is the
quickest check, and the contract should state the unit rather than relying on the check.

### 6.9 What the example demonstrates

| Defect | Example | Representation |
|---|---|---|
| Binary float | 0.1 × 10 = 0.9999999999999999 | Decimal or integer minor units |
| Rounding mode | 2.675 → 2.67 or 2.68 | State the mode |
| Large integer | 2^53 + 1 → 2^53 | String or 128-bit |
| Leading zero | "0012345" → 12345 | String |
| DST duration | 47 h versus 48 h | Instant plus zone |
| ISO week | 2027-01-01 in 2026-W53 | ISO week function |
| Missing as zero | 7.10% versus 4.73% | Null with reason |
| CSV quoting | Comma and newline in a field | Real parser |
| Formula injection | `=1+1` in a cell | Escape on export |
| Epoch units | seconds versus milliseconds | State the unit |

## 7. Code, query, or calculation

The script reproduces every number in section 6 with the standard library.

```python
"""Topic 09 verification: floating point, rounding, dates, identifiers, missing data."""
from decimal import Decimal, ROUND_HALF_UP
from datetime import datetime, timezone, timedelta
import csv, io

print("9.1 Binary floating point")
print(f"  0.1 + 0.2 = {0.1 + 0.2!r}")
print(f"  equal to 0.3? {0.1 + 0.2 == 0.3}")
acc = 0.0
for _ in range(10):
    acc += 0.1
print(f"  ten additions of 0.1 = {acc!r}")
print(f"  Decimal 0.1 + 0.2 = {Decimal('0.1') + Decimal('0.2')}")

print("\n9.2 Rounding")
print(f"  round(2.675, 2) = {round(2.675, 2)}")
print(f"  Decimal half-up  = {Decimal('2.675').quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)}")
print(f"  round(0.5) = {round(0.5)}, round(1.5) = {round(1.5)}, round(2.5) = {round(2.5)}")

print("\n9.3 Integer precision")
big = 9007199254740993
print(f"  int  = {big}")
print(f"  float= {float(big)!r}")
print(f"  1e16 + 1 = {1e16 + 1!r}")
print(f"  2**53 = {2**53}, 2**53 + 1 as float = {float(2**53 + 1)!r}")

print("\n9.4 Time zones and DST")
est = timezone(timedelta(hours=-5))
edt = timezone(timedelta(hours=-4))
start = datetime(2026, 3, 7, 12, 0, tzinfo=est)
end = datetime(2026, 3, 9, 12, 0, tzinfo=edt)
print(f"  aware difference   = {(end - start).total_seconds() / 3600:.1f} h")
naive_start = datetime(2026, 3, 7, 12, 0)
naive_end = datetime(2026, 3, 9, 12, 0)
print(f"  naive difference   = {(naive_end - naive_start).total_seconds() / 3600:.1f} h")

print("\n9.5 ISO week")
d = datetime(2027, 1, 1)
iso = d.isocalendar()
print(f"  {d.date()} -> ISO year {iso.year}, week {iso.week}, weekday {iso.weekday}")

print("\n9.6 Identifiers")
pid = "0012345"
print(f"  string {pid!r} -> int {int(pid)} -> string {str(int(pid))!r}")
ids = ["10", "9", "100", "2"]
print(f"  lexical sort: {sorted(ids)}")
print(f"  numeric sort: {sorted(ids, key=int)}")

print("\n9.7 Missing versus zero")
values = [7.2, None, 6.8, None, 7.5, 6.9]
present = [v for v in values if v is not None]
mean_present = sum(present) / len(present)
mean_zero = sum(v if v is not None else 0 for v in values) / len(values)
print(f"  values = {values}")
print(f"  mean of present = {mean_present:.4f}")
print(f"  mean treating None as 0 = {mean_zero:.4f}")
print(f"  missing fraction = {(len(values)-len(present))/len(values):.2%}")

print("\n9.8 CSV quoting")
rows = [["id", "name", "note"],
        ["1", "Smith, John", "follow-up"],
        ["2", "Okafor", "line1\nline2"],
        ["3", "=1+1", "formula"]]
buf = io.StringIO()
w = csv.writer(buf, lineterminator="\n")
w.writerows(rows)
print(buf.getvalue(), end="")

print("\n9.9 Epoch units")
dt = datetime(2026, 10, 24, 0, 0, tzinfo=timezone.utc)
print(f"  {dt.isoformat()} -> {int(dt.timestamp())} s")
ms = int(dt.timestamp()*1000)
print(f"  milliseconds -> {ms}")
print(f"  correct conversion ms/1000 -> {datetime.fromtimestamp(ms/1000, timezone.utc).isoformat()}")
print(f"  ms treated as s -> out of range on this platform (year would be ~58,000)")
```

**Expected output.**

```text
9.1 Binary floating point
  0.1 + 0.2 = 0.30000000000000004
  equal to 0.3? False
  ten additions of 0.1 = 0.9999999999999999
  Decimal 0.1 + 0.2 = 0.3

9.2 Rounding
  round(2.675, 2) = 2.67
  Decimal half-up  = 2.68
  round(0.5) = 0, round(1.5) = 2, round(2.5) = 2

9.3 Integer precision
  int  = 9007199254740993
  float= 9007199254740992.0
  1e16 + 1 = 1e+16
  2**53 = 9007199254740992, 2**53 + 1 as float = 9007199254740992.0

9.4 Time zones and DST
  aware difference   = 47.0 h
  naive difference   = 48.0 h

9.5 ISO week
  2027-01-01 -> ISO year 2026, week 53, weekday 5

9.6 Identifiers
  string '0012345' -> int 12345 -> string '12345'
  lexical sort: ['10', '100', '2', '9']
  numeric sort: ['2', '9', '10', '100']

9.7 Missing versus zero
  values = [7.2, None, 6.8, None, 7.5, 6.9]
  mean of present = 7.1000
  mean treating None as 0 = 4.7333
  missing fraction = 33.33%

9.8 CSV quoting
id,name,note
1,"Smith, John",follow-up
2,Okafor,"line1
line2"
3,=1+1,formula

9.9 Epoch units
  2026-10-24T00:00:00+00:00 -> 1792800000 s
  milliseconds -> 1792800000000
  correct conversion ms/1000 -> 2026-10-24T00:00:00+00:00
  ms treated as s -> out of range on this platform (year would be ~58,000)
```

**Assumptions stated by the script.**

1. The DST offsets are supplied explicitly rather than derived from a tz database. The script
   demonstrates the arithmetic; a production system should use a tz database and an IANA zone
   identifier.
2. The mean calculations assume the missing values are missing at random. If they are not, both means
   are biased and the direction depends on the mechanism.
3. The CSV writer uses `\n` line endings for portability. A Windows spreadsheet may expect `\r\n`.
4. The float representation is IEEE 754 binary64, which is the default in Python, NumPy, and most
   databases.
5. The 2^53 boundary applies to exact integer representation; larger integers can be represented
   approximately, which is worse because the error is silent.

## 8. Data, standards, and regulatory context

### 8.1 The standards that define the representations

| Standard | What it defines | Binding status |
|---|---|---|
| IEEE 754-2019 | Binary floating-point arithmetic | Universal in hardware and languages |
| ISO 8601-1:2019 | Date and time representations | International standard |
| RFC 3339 | A profile of ISO 8601 for internet timestamps | IETF standard; widely required |
| RFC 9557 | Timestamps with a time-zone annotation | Updates RFC 3339 |
| RFC 4180 | CSV format | Informational; widely followed |
| RFC 8259 | JSON data interchange | IETF standard |
| RFC 3629 | UTF-8 encoding | IETF standard |
| RFC 4122 | UUIDs | IETF standard |
| HL7 FHIR R5 | Clinical data exchange, including identifiers and timestamps | Required where adopted |
| 21 CFR Part 11 | Electronic records and signatures | Binding in the US |
| HIPAA Privacy Rule | De-identification safe harbour | Binding in the US |
| GDPR | Personal data protection | Binding in the EU |

**IEEE 754** defines binary64 as a 64-bit format with a 53-bit significand. The standard also defines
rounding modes, NaN, and infinity. The practical consequence is that decimal fractions are
approximations and equality comparison is exact only for values that happen to be representable.

**ISO 8601** defines the date and time formats. **RFC 3339** restricts them to a profile suitable for
internet protocols, requiring the offset and permitting the `Z` suffix for UTC. **RFC 9557** adds an
optional bracket annotation for the time-zone identifier, for example
`2026-03-08T02:30:00-05:00[America/New_York]`. The annotation is what distinguishes an instant from a
local time with an offset.

**RFC 4180** describes the CSV conventions: comma delimiter, double-quote quoting, doubled quotes for
an embedded quote, and CRLF line endings. It is informational, not a formal standard, which is why CSV
files in the wild vary. A production parser must be configurable.

**21 CFR Part 11** requires electronic records to be trustworthy, reliable, and equivalent to paper,
with audit trails, access controls, and electronic signatures. For a data pipeline, the practical
requirements are provenance, immutability of raw data, and a record of who changed what and when.

**HIPAA** defines a safe-harbour de-identification method that removes 18 categories of identifiers,
including names, dates more specific than a year (with exceptions), geographic subdivisions smaller
than a state, and device identifiers. Removing them reduces risk; it does not eliminate it, especially
for rare conditions.

### 8.2 What the standards do not constrain

- They do not choose the storage type for a field. That is a schema decision.
- They do not prevent a tool from reinterpreting a string. A spreadsheet can convert a gene name
  regardless of the file's standard.
- They do not make a de-identified dataset anonymous. Re-identification risk is a separate analysis.
- They do not cover every clinical data format. HL7 v2, FHIR, DICOM, and proprietary formats each have
  their own rules.
- They do not replace testing. A conforming format can still contain a semantically wrong value.

> **Note:** The regulatory content is educational. Electronic-record compliance and de-identification
> require review by a qualified professional.

## 9. Industry practice and case evidence

### 9.1 What organisations actually do

**Health systems** store timestamps as UTC with a separate facility time zone, and they display local
time at the point of care. The separation is what makes a multi-site report comparable and a
single-site display correct.

**Registries** use string identifiers with a source-system prefix, and they keep a crosswalk between
identifiers when patients move between systems. The crosswalk is a governed artefact, not a lookup
table maintained by hand.

**Data platforms** use typed columnar formats (Parquet, ORC) for analytics and reserve CSV for
interchange. The typed format preserves the schema and avoids the inference errors that CSV invites.

**Regulated environments** maintain audit trails and immutable raw data. The transformation layer is
versioned, and every derived table can be traced to a raw table and a code version.

### 9.2 Case evidence

**Gene name conversion in spreadsheets.** Zeeberg et al. (2004, *BMC Bioinformatics* 5:80) documented
that Excel converts gene names such as `SEPT2` to `2-Sep` and accession numbers such as `2310009E13` to
`2.31E+13`. Ziemann, Eren, and El-Osta (2016, *Genome Biology* 17:177) screened 35,175 supplementary
Excel files from 18 genomics journals and found errors in about 19.6% of papers with gene lists.
Abeysooriya et al. (2021, *PLOS Computational Biology*) found the problem persisted, affecting about
30.9% of articles with Excel gene lists in the later sample. The documented lesson is that a
destructive conversion can persist for two decades despite being well known, because the tool's default
is convenient.

**CSV injection.** Security researchers have documented that spreadsheet applications execute formulas
in CSV fields beginning with `=`, `+`, `-`, or `@`. The vulnerability is listed in the OWASP guidance on
CSV injection. The data-integrity version is the same mechanism: a legitimate string is interpreted as
a formula or a date.

**Daylight-saving errors in clinical systems.** Reports of missed or duplicated appointments, incorrect
medication administration times, and off-by-one-hour analytics appear whenever a system stores local
time without a zone. The mechanism is the one in section 6.4; the specific incidents are usually not
public, so the general mechanism is the documented part.

**Identifier precision in JSON.** JSON numbers are specified without a precision guarantee, and many
parsers use IEEE 754 doubles. A 19-digit identifier or a large integer sent as a JSON number can lose
its last digits in transit. The FHIR specification recommends that identifiers be exchanged as strings,
which is the general fix.

> **Note:** The gene-name studies and the OWASP guidance are cited to published sources. The
> daylight-saving and JSON cases are described as mechanisms because specific incident reports are not
> public.

### 9.3 The practical pattern

Teams that avoid computing defects do four things:

1. They declare a schema and parse explicitly at the boundary.
2. They store identifiers as strings and timestamps as instants plus zones.
3. They represent missingness with a reason code, never with a sentinel.
4. They test with adversarial fixtures and reconcile row counts at every step.

Each practice is cheap, and each prevents a class of defect that is expensive to diagnose after the
analysis has been reported.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Float equality test | `if total == expected` fails intermittently | Binary representation of decimals | False alarms; flaky tests | Compare with a tolerance |
| 2 | Money stored as float | Balances off by fractions of a cent | Binary approximation | Reconciliation failures | Decimal or integer minor units |
| 3 | Rounding mode unstated | Two reports differ in the last digit | Half-up versus half-even | Inconsistent published numbers | State the mode; round once |
| 4 | Identifier stored as integer | Leading zeros disappear; joins drop rows | Numeric type discards the representation | Record loss and linkage failure | String identifiers |
| 5 | Identifier above 2^53 stored as float | Two identifiers collide | 53-bit significand | Merged patient records | String or 128-bit integer |
| 6 | Naive timestamp | No offset in the stored value | Local time treated as an instant | One-hour errors at DST | Instant plus zone |
| 7 | Duration computed on local times | 48 h for a 47 h interval | Calendar subtraction | Window boundary errors | Subtract instants |
| 8 | Ambiguous local time silently resolved | Two events at 01:30 on fall-back | DST repeated hour | Events ordered incorrectly | Flag ambiguous times |
| 9 | Partial date parsed as first of month | All events cluster on the 1st | Missing components defaulted | Bias in timing analyses | Partial date type |
| 10 | Epoch unit mismatch | Dates in the year 58,000 | Seconds versus milliseconds | All timestamps invalid | State the unit; magnitude check |
| 11 | Missing encoded as zero | Means implausibly low | Sentinel participates in arithmetic | Biased estimates | Null with reason code |
| 12 | Missing encoded as −999 | Mean below any plausible value | Sentinel not mapped | Gross bias | Explicit null mapping |
| 13 | Encoding guessed | First column name contains `ï»¿` | BOM or Latin-1 read as UTF-8 | Failed column match | Declare the encoding |
| 14 | CSV split on commas | Column count varies by row | Quoted commas and newlines | Misaligned records | Use a real parser |
| 15 | Formula injection | A cell begins with `=` | Spreadsheet evaluates the string | Security and integrity risk | Escape on export |
| 16 | Spreadsheet date conversion | `SEPT2` becomes `2-Sep` | Locale date parse on text | Gene names destroyed | Do not open raw data in a spreadsheet |
| 17 | Silent join drop | Row count falls after a join | Inner join with unmatched keys | Selection bias | Report match rates |
| 18 | Locale-dependent parsing | `03/04/2026` parsed differently | Day-first versus month-first | Dates shifted by months | Explicit format strings |

Three rows deserve emphasis.

**Row 5, identifier precision.** The failure is invisible because the value still looks like a number.
Two identifiers that differ in the last digit become the same float, and the join merges two patients.
The defect surfaces as an implausible record with two birth dates or two sexes, long after the merge.

**Row 12, sentinel values.** A sentinel is a value chosen to mean missing, and it is always a value
that could occur. The mean of a column with −999 sentinels is not merely wrong; it is wrong by an amount
that depends on the missing fraction, which makes the error look like a real effect.

**Row 17, silent join drop.** An inner join is the default in SQL and in most dataframe libraries. It
drops unmatched rows without a warning. The analyst sees a smaller table and assumes the data were
incomplete, when the cause was a type mismatch in the key.

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test |
|---:|---|---|---|---|
| 1 | The source format is known and stable | Explicit parsing | A vendor changes the export | Schema validation on every load |
| 2 | The encoding is declared | Correct decoding | Legacy files, BOM | Decode a sample; check the first field |
| 3 | The identifier is unique and stable | Joins and linkage | Reuse, reassignment, merges | Duplicate check by key and period |
| 4 | The timestamp has a zone | Instant arithmetic | Local-time-only sources | Audit the offset field |
| 5 | The epoch and unit are stated | Numeric timestamp conversion | Mixed seconds and milliseconds | Magnitude check against a known date |
| 6 | Missing markers are enumerated | Correct null mapping | An undocumented sentinel | Value-frequency check for repeated unusual values |
| 7 | The missingness mechanism is understood | Choice of imputation | Informative missingness | Compare characteristics by missingness |
| 8 | The rounding mode is stated | Reproducible reporting | Different tools, different defaults | Unit tests on boundary values |
| 9 | The numeric range fits the type | No overflow or precision loss | Large identifiers, high-precision labs | Range tests at the boundaries |
| 10 | The file will not be opened in a spreadsheet | Preservation of text | A user opens the CSV | Pre-escape or use a typed format |
| 11 | The join key is normalised | Match rate | Case, whitespace, Unicode form | Compare match rates before and after normalisation |
| 12 | The environment is pinned | Reproducibility | Library upgrades | Record versions; rebuild |
| 13 | The raw data are immutable | Audit and reprocessing | Overwriting in place | Hash the raw files |
| 14 | The de-identification is adequate | Privacy | Rare combinations, dates | Re-identification risk assessment |

The limits of this topic:

- It does not cover database design, indexing, or query optimisation.
- It does not cover distributed computing or streaming architectures.
- It does not cover security engineering beyond the specific defects named.
- It does not cover every file format. It covers the formats that recur in clinical data work.
- It does not replace a data engineer's review of a production pipeline.

## 12. Exercises

### Exercise 1 — Floating point (recall and calculation)

Explain why 0.1 + 0.2 is not exactly 0.3 in binary64. Compute the result of adding 0.1 ten times and
state the smallest tolerance you would use for a comparison of insulin doses measured to 0.01 mL.

**Deliverable.** A short explanation, the computed sum, and a tolerance with a justification.

**Acceptance criteria.** Explanation references the binary representation of 0.1; the sum is
0.9999999999999999; the tolerance is smaller than 0.005 mL (half the measurement resolution) and larger
than the accumulated error (about 1e-16 mL), for example 1e-9 mL.

**Time.** 25 minutes.

### Exercise 2 — Money and rounding (calculation)

Compute 3 × $0.10 in float and in decimal. Compute 2.675 rounded to two decimals with half-up and with
Python's default. State which mode you would specify for a financial report and why.

**Deliverable.** Four values and a recommendation.

**Acceptance criteria.** Float 0.30000000000000004; decimal 0.30; half-up 2.68; Python default 2.67;
the recommendation states the mode explicitly and notes that half-even reduces cumulative bias in
large sums.

**Time.** 25 minutes.

### Exercise 3 — Missing versus zero (calculation)

HbA1c values are 8.1, missing, 7.4, missing, 6.9, 7.7. Compute the mean of the present values, the
mean treating missing as zero, and the missing fraction. State which mean you would report and what
additional information you need.

**Deliverable.** Three values and a recommendation.

**Acceptance criteria.** Present mean 7.525%; zero-imputed mean 5.017%; missing fraction 33.33%; the
recommendation reports the present mean with the missing fraction and requests the reason for
missingness before imputing.

**Time.** 20 minutes.

### Exercise 4 — Duration across a DST transition (calculation)

An event starts at 2026-11-01 12:00 EDT (UTC−04:00) and ends at 2026-11-02 12:00 EST (UTC−05:00).
Compute the duration using the instants and using the naive local times. State which is correct and
why the difference has the opposite sign to the spring transition.

**Deliverable.** Two durations and an explanation.

**Acceptance criteria.** Instant duration 25 h; naive duration 24 h; the fall-back day is 25 hours long,
so the instant duration exceeds the naive one, which is the opposite of the spring case.

**Time.** 25 minutes.

### Exercise 5 — Identifier representation (application)

For each identifier, state the correct type and one failure that occurs if the wrong type is used:
`"00042"`; `9007199254740993`; `"E11.9"`; a UUID; a 19-digit NHS number.

**Deliverable.** A five-row table.

**Acceptance criteria.** All strings; leading zero lost for `"00042"`; precision lost above 2^53;
`"E11.9"` cannot be numeric at all; UUID is a string by definition; NHS number loses digits as a float.
The answer states that identifiers are never used for arithmetic.

**Time.** 20 minutes.

### Exercise 6 — CSV design (design)

Design the export of a table with patient names containing commas, notes containing newlines, a field
beginning with `=`, and a gene symbol `SEPT2`. Specify the delimiter, quoting, escaping, encoding, and
a protection against spreadsheet conversion.

**Deliverable.** A format specification and a sample of three rows.

**Acceptance criteria.** UTF-8 without BOM; comma delimiter with double-quote quoting; doubled quotes
for embedded quotes; fields beginning with `=`, `+`, `-`, `@` escaped; `SEPT2` protected by a
pre-escape or by delivering a typed format; a statement that the file should not be opened in a
spreadsheet without protection.

**Time.** 35 minutes.

### Exercise 7 — Audit a pipeline (critique)

You inherit a pipeline that reads a CSV with `pandas.read_csv`, converts `patient_id` to int, parses
`admit_time` with `pd.to_datetime` without a time zone, and fills missing lab values with 0. List the
defects, the data signatures that would reveal each, and the fix.

**Deliverable.** A defect table with signature and fix.

**Acceptance criteria.** Identifier: leading zeros lost, signature is a join match rate below 100%;
timestamp: naive, signature is a one-hour shift at DST; missing: zero-filled, signature is an
implausible mean; CSV: default encoding and type inference, signature is a BOM in the first column or a
misparsed column. Each fix is specific.

**Time.** 40 minutes.

### Exercise 8 — Data contract (design)

Write a data contract for a laboratory observations table with fields: patient identifier, LOINC code,
value, unit, collection time, result status, and a comment. Specify type, nullability, allowed values,
and the missingness semantics for each field.

**Deliverable.** A contract table of seven rows plus a versioning statement.

**Acceptance criteria.** Identifier string; LOINC string code; value decimal or string depending on the
analyte with a stated rule; unit UCUM string; collection time instant plus zone; status enumerated;
comment free text with a length bound; every field has a nullability and a missingness reason;
versioning includes a schema version and a change procedure.

**Time.** 50 minutes.

## 13. Self-check answers

### Answer to Exercise 1

```text
0.1 in binary64 is 0.1000000000000000055511151231257827...
0.2 in binary64 is 0.2000000000000000111022302462515654...
Their sum is 0.3000000000000000444089209850062616..., displayed as 0.30000000000000004.
0.3 in binary64 is 0.2999999999999999888977697537484346...
The two are different, so equality fails.

Ten additions of 0.1 give 0.9999999999999999.
```

The tolerance should be smaller than the measurement resolution (0.01 mL, so half is 0.005 mL) and
larger than the accumulated floating-point error (about 1e-16 mL). A tolerance of 1e-9 mL satisfies
both and is far below any clinically meaningful difference.

### Answer to Exercise 2

```text
3 x 0.10 in float  = 0.30000000000000004
3 x 0.10 in decimal = 0.30

2.675 half-up to 2 dp  = 2.68
2.675 Python default   = 2.67
```

For a financial report, specify half-up if the convention is commercial rounding, or half-even if the
goal is to avoid cumulative bias in large sums. The important point is that the mode is stated and
applied consistently, because two systems with different defaults will disagree on the last digit.

### Answer to Exercise 3

```text
Present values: 8.1, 7.4, 6.9, 7.7
Mean = (8.1 + 7.4 + 6.9 + 7.7)/4 = 30.1/4 = 7.525%

Zero-imputed: (8.1 + 0 + 7.4 + 0 + 6.9 + 7.7)/6 = 30.1/6 = 5.0167%

Missing fraction = 2/6 = 33.33%
```

Report the present mean of 7.525% with the missing fraction, and state that a third of the values are
missing. Before imputing, establish why they are missing: if the test was not ordered because the
patient was stable, the missingness may be informative in a favourable direction; if it was not ordered
because the patient was too sick, the direction reverses.

### Answer to Exercise 4

```text
Start: 2026-11-01 12:00 EDT = 2026-11-01 16:00 UTC
End:   2026-11-02 12:00 EST = 2026-11-02 17:00 UTC
Duration = 25.0 h

Naive: 2026-11-02 12:00 - 2026-11-01 12:00 = 24.0 h
```

The instant duration is correct. The fall-back day is 25 hours long because the clock repeats an hour,
so the interval spanning it is one hour longer than the naive calendar difference. The spring
transition produces the opposite sign: the day is 23 hours long and the instant duration is one hour
shorter than the naive difference.

### Answer to Exercise 5

| Identifier | Type | Failure with the wrong type |
|---|---|---|
| `"00042"` | String | Leading zeros lost, value becomes 42 |
| `9007199254740993` | String or 128-bit integer | Float rounds to 9007199254740992 |
| `"E11.9"` | String | Cannot be parsed as a number at all |
| UUID | String | Not a number; parsing fails |
| 19-digit NHS number | String | Float loses the last digits |

Identifiers are labels, not quantities. The only correct arithmetic on an identifier is equality and
concatenation, and even equality requires the representation to be preserved.

### Answer to Exercise 6

```text
Encoding:   UTF-8, no BOM
Delimiter:  comma
Quoting:    double quote, minimal (only when needed)
Escaping:   embedded double quote doubled
Protection: fields beginning with =, +, -, @ are prefixed with a single quote
            or exported in a typed format
SEPT2:      exported as text, not opened in a spreadsheet; if a spreadsheet
            is unavoidable, pre-escape as ="SEPT2"
Line ends:  \n for portability; \r\n if the consumer requires it
```

A sample row:

```text
id,name,note
1,"Smith, John","follow-up"
2,Okafor,"line1
line2"
3,'=1+1,formula
```

The single quote before `=1+1` prevents formula evaluation. The quoted fields preserve the comma and
the newline. The gene symbol must be protected by the export process, because the file itself cannot
prevent a spreadsheet from reinterpreting it.

### Answer to Exercise 7 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Identifier | 25% | Int conversion loses leading zeros; signature is a match rate below 100% or duplicate keys |
| Timestamp | 25% | Naive parsing; signature is a one-hour shift at DST or an ambiguous local time |
| Missing | 25% | Zero fill; signature is an implausible mean and a changed distribution |
| CSV | 15% | Default encoding and inference; signature is a BOM in the column name or a column parsed as object |
| Fixes | 10% | Specific: dtype str, tz-aware parse, explicit null, encoding and dtype declaration |

A weak audit lists the defects without the signatures, which makes them impossible to detect in
production.

### Answer to Exercise 8 (rubric)

| Criterion | Weight | What earns credit |
|---|---:|---|
| Types | 25% | Identifier and LOINC as strings; value as decimal or a stated rule; unit as UCUM |
| Time | 20% | Instant plus zone, with a rule for ambiguous and nonexistent local times |
| Nullability | 20% | Every field has a nullability and a missingness reason code |
| Allowed values | 15% | Status enumerated; unit from UCUM; value range rules |
| Versioning | 10% | Schema version and a change procedure |
| Provenance | 10% | Source system, extraction time, and transformation record |

A weak contract lists the fields and types but omits nullability and missingness semantics, which are
the parts that determine whether the data can be analysed at all.

## 14. Interview questions

### Question 1 — "Why is 0.1 + 0.2 not 0.3?"

**Strong answer.** "Because 0.1 and 0.2 have no exact binary representation in IEEE 754 double
precision, so they are stored as the nearest representable values. Their sum inherits the rounding
error and is not the same bit pattern as the representation of 0.3. The fix is a decimal type, integer
minor units, or a tolerance for comparison."

**Weak answer.** "Floating point is imprecise." The question asks why, and the answer is the binary
representation of decimal fractions, not a general imprecision.

### Question 2 — "How do you store money in a database?"

**Strong answer.** "As a decimal type with a fixed scale, or as integer minor units such as cents.
Never as a float, because the representation error accumulates in sums and breaks reconciliation. I
would also state the rounding mode for any division or percentage calculation."

**Weak answer.** "As a float with two decimals." There is no such thing as a float with two decimals;
the display can be rounded, the storage cannot be.

### Question 3 — "Why must identifiers be strings?"

**Strong answer.** "Because numeric types lose leading zeros, lose precision above 2^53, and change the
sort order. An identifier is a label, not a quantity, so the only valid operations are equality and
concatenation. Storing it as a number creates silent join failures and record merges."

**Weak answer.** "Because some identifiers have letters." That is true for some, but numeric-looking
identifiers fail too, which is the more common and more dangerous case.

### Question 4 — "What is the difference between a timestamp and an instant?"

**Strong answer.** "An instant is a point on the UTC timeline. A timestamp is a representation that may
include a local time and an offset or zone. A local time without a zone is not an instant, because it
maps to different instants in different zones and can be ambiguous or nonexistent at a DST
transition."

**Weak answer.** "They are the same thing." The distinction is the source of every DST and multi-site
error.

### Question 5 — "How do you compute a duration across a daylight-saving change?"

**Strong answer.** "Subtract the instants, not the local times. A spring-forward day is 23 hours and a
fall-back day is 25 hours, so local subtraction gives 24 hours for both. I would store UTC and the
zone, and compute the difference in UTC."

**Weak answer.** "Subtract the timestamps." If the timestamps are local and naive, the answer is wrong
by an hour twice a year.

### Question 6 — "What is wrong with filling missing values with zero?"

**Strong answer.** "It changes the meaning of the value from 'not measured' to 'measured as zero'. The
mean, variance, and correlations are all biased, and the direction depends on the missing fraction. A
null with a reason code, and an explicit imputation method, preserve the distinction."

**Weak answer.** "It is a simple imputation." Simplicity is not the issue; the issue is that zero is a
plausible value and the imputation is therefore undetectable.

### Question 7 — "How would you detect an encoding problem?"

**Strong answer.** "Look at the first column name for a byte order mark, check for mojibake in accented
characters, and compare the decoded byte count with the file size. I would declare the encoding in the
reader and fail loudly on a decode error rather than replacing characters silently."

**Weak answer.** "Use UTF-8." That is the right default, and the detection is still required for
legacy files.

### Question 8 — "What is CSV injection?"

**Strong answer.** "A spreadsheet interprets a cell beginning with `=`, `+`, `-`, or `@` as a formula.
If the content is untrusted, it can execute a command or exfiltrate data; if it is legitimate text, it
is corrupted. The export must escape those fields, and the import must not execute them."

**Weak answer.** "It is a security issue in CSV files." It is also a data-integrity issue, which is the
part that affects analytics.

### Question 9 — "Why did gene names get corrupted in supplementary tables?"

**Strong answer.** "Spreadsheets apply a locale-dependent date or number parse to text fields. `SEPT2`
becomes `2-Sep`, `MARCH1` becomes `1-Mar`, and accession numbers become scientific notation. The
conversion is destructive and was documented in 2004, and studies in 2016 and 2021 found it persisted
in 20% to 30% of papers with Excel gene lists."

**Weak answer.** "Users made a mistake." The default behaviour of the tool caused it, and the fix is a
process change, not user care.

### Question 10 — "How do you make a join safe?"

**Strong answer.** "Normalise the key, use the same type on both sides, and report the match rate
before and after. An inner join that drops rows silently is a selection bias. If the match rate is
below expectation, investigate the key type, the leading zeros, the whitespace, and the identifier
reuse before accepting the result."

**Weak answer.** "Use a left join so nothing is lost." A left join preserves rows but produces nulls
that then need handling; the match rate still needs to be reported.

### Question 11 — "What does reproducibility require?"

**Strong answer.** "The code version, the data version, the environment, and the parameters, all
recorded together. A notebook alone is not enough, because it does not pin the environment and it may
not record the data version. The raw data should be immutable so the pipeline can be re-run."

**Weak answer.** "Put the code in Git." Version control is necessary and not sufficient.

### Question 12 — "How would you test a pipeline for these defects?"

**Strong answer.** "With adversarial fixtures: a leading-zero identifier, a value above 2^53, a
timestamp at each DST transition, a missing value, a sentinel, a quoted comma, an embedded newline, a
formula string, a BOM, and a partial date. Each fixture asserts the expected typed output. I would also
reconcile row counts at every join and filter."

**Weak answer.** "Test on a sample of the real data." Real data rarely contains the boundary cases,
which is why they reach production.

## 15. Key terms

- **21 CFR Part 11** — US regulation for electronic records and signatures.
- **Ambiguous local time** — a local time that occurs twice at fall-back.
- **Binary floating point** — a number represented as sign × significand × 2^exponent.
- **BOM** — byte order mark at the start of a file.
- **Catastrophic cancellation** — loss of precision when subtracting nearly equal numbers.
- **Collision** — two different inputs with the same hash.
- **Complete-case analysis** — analysing only records with no missing values.
- **CSV injection** — a formula in a field executed by a spreadsheet.
- **Data contract** — an agreed schema plus semantics and quality rules.
- **De-identification** — removing or transforming identifiers.
- **Decimal type** — a base-10 representation with fixed precision.
- **Delimiter** — the character separating fields.
- **Double precision** — 64-bit float with a 53-bit significand.
- **DST** — daylight saving time.
- **Encoding** — the mapping from characters to bytes.
- **Epoch** — a reference instant for numeric timestamps.
- **Equality comparison** — testing whether two floats are identical.
- **Escape** — a sequence representing a special character.
- **Hash** — a fixed-length digest of a value.
- **HL7 FHIR** — a clinical data exchange standard.
- **Identifier** — a label that identifies a record.
- **IEEE 754** — the floating-point arithmetic standard.
- **Immutability** — the property that a record cannot be changed.
- **Imputation** — filling a missing value with an estimate.
- **Instant** — a point on the UTC timeline.
- **Integer minor units** — money stored as integer cents.
- **ISO 8601** — the international date and time format.
- **ISO week** — a week numbered from the first Thursday of the year.
- **Join key** — the field used to combine records.
- **Leap second** — a one-second adjustment to UTC.
- **Locale** — regional formatting rules for numbers and dates.
- **Machine epsilon** — the gap between 1 and the next representable number.
- **NaN** — not a number, a floating-point sentinel.
- **Natural key** — an identifier derived from the data.
- **Nonexistent local time** — a local time skipped at spring-forward.
- **Not applicable** — the question does not apply to the record.
- **Null** — the absence of a value.
- **Offset** — the difference between local time and UTC.
- **Parsing** — converting a string to a typed value.
- **Partial date** — a date with missing components.
- **PII** — personally identifiable information.
- **Provenance** — the origin and processing history of a record.
- **Quoting** — wrapping a field to protect delimiters.
- **Reproducibility** — the ability to regenerate a result.
- **RFC 3339** — a profile of ISO 8601 for internet timestamps.
- **RFC 4180** — the CSV format conventions.
- **Round-off error** — error from finite precision.
- **Safe harbour** — a US HIPAA de-identification method.
- **Schema** — the declared types and constraints of a dataset.
- **Sentinel value** — a number used to encode missing.
- **Significand** — the significant digits of a floating-point number.
- **Surrogate key** — a generated identifier with no meaning.
- **Timestamp** — a point in time with an offset or zone.
- **Time zone** — a rule for offset from UTC, including DST.
- **Tolerance** — an allowed difference for comparison.
- **Type coercion** — implicit conversion between types.
- **UTC** — Coordinated Universal Time.
- **UUID** — a 128-bit identifier.

## 16. Sources

All URLs were accessed on 24 October 2026.

### Representation standards

- IEEE 754-2019, *IEEE Standard for Floating-Point Arithmetic*. https://standards.ieee.org/ieee/754/6210/
- ISO 8601-1:2019, *Date and time — Representations for information interchange — Part 1: Basic rules*.
  https://www.iso.org/standards.html
- RFC 3339, *Date and Time on the Internet: Timestamps*. https://www.rfc-editor.org/rfc/rfc3339.html
- RFC 9557, *Date and Time on the Internet: Timestamps with Additional Information*.
  https://www.rfc-editor.org/rfc/rfc9557.html
- RFC 4180, *Common Format and MIME Type for Comma-Separated Values (CSV) Files*.
  https://www.rfc-editor.org/rfc/rfc4180.html
- RFC 8259, *The JavaScript Object Notation (JSON) Data Interchange Format*.
  https://www.rfc-editor.org/rfc/rfc8259.html
- RFC 3629, *UTF-8, a transformation format of ISO 10646*. https://www.rfc-editor.org/rfc/rfc3629.html
- RFC 4122, *A Universally Unique IDentifier (UUID) URN Namespace*.
  https://www.rfc-editor.org/rfc/rfc4122.html

### Clinical data standards and regulation

- HL7, *FHIR R5 specification*, including the Observation and Identifier resources.
  https://hl7.org/fhir/R5/
- 21 CFR Part 11, *Electronic Records; Electronic Signatures*.
  https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
- US Department of Health and Human Services, *HIPAA Privacy Rule: De-identification of Protected
  Health Information*. https://www.hhs.gov/hipaa/for-professionals/privacy/special-topics/de-identification/index.html
- Regulation (EU) 2016/679, *General Data Protection Regulation*.
  https://eur-lex.europa.eu/eli/reg/2016/679/oj

### Case evidence

- Zeeberg, B. R. et al., "Mistaken Identifiers: Gene name errors can be introduced inadvertently when
  using Excel in bioinformatics", *BMC Bioinformatics* 5, 80, 2004.
  https://doi.org/10.1186/1471-2105-5-80
- Ziemann, M., Eren, Y., El-Osta, A., "Gene name errors are widespread in the scientific literature",
  *Genome Biology* 17, 177, 2016. https://doi.org/10.1186/s13059-016-1044-7
- Abeysooriya, M., Soria, M., Kasu, M. S., Ziemann, M., "Gene name errors: Lessons not learned",
  *PLOS Computational Biology* 17(7), e1008984, 2021. https://doi.org/10.1371/journal.pcbi.1008984
- OWASP, *CSV Injection*. https://owasp.org/www-community/attacks/CSV_Injection

> **Note:** The RFCs are freely available. ISO and IEEE standards are sold documents; the URLs are
> catalogue pages. The gene-name studies are open access.

## 17. Links

- Up to the module map: [`_index.md`](_index.md)
- Previous topic: [`08_engineering-and-device-basics.md`](08_engineering-and-device-basics.md)
- Next topic: [`10_the-assumptions-ledger.md`](10_the-assumptions-ledger.md)
- Units and numerical precision: [`03_units-and-metrology.md`](03_units-and-metrology.md)
- Statistical handling of missing data: [`02_probability-and-statistics-basics.md`](02_probability-and-statistics-basics.md)
- Time-varying biological processes: [`05_biological-basics.md`](05_biological-basics.md)
- Assumption recording: [`10_the-assumptions-ledger.md`](10_the-assumptions-ledger.md)
- Root survey file: [`../../01A_foundations_assumptions_and_units.md`](../../course/01A_foundations_assumptions_and_units.md)
- Labs: [`../../labs/`](../../labs/) — `03_cohort_pipeline.ipynb` exercises joins and cohort
  construction; `14B_quantization.ipynb` covers numeric representation in signals.
