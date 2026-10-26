# Lab Conventions

> Module 24 — Hardware and Robotics Labs
> Topic 3 of 9
> Estimated study time: 4 hours
> Prerequisites: 01, 02
> Version 2026.3

## 1. Why this topic exists

A result that cannot be reproduced is an anecdote. A result that can be reproduced but cannot be traced to its raw data is a claim. A result that can be traced but whose analysis choices were made after seeing the data is a hypothesis.

The industry problem is that lab work is usually recorded for the person who did it, not for the person who will review it. The notebook says "filtered the signal and ran the detector." It does not say which filter, which order, which cutoff, which detector version, which records, which annotations, which split, or which threshold. Six months later, the team cannot tell whether a changed result is a bug fix or a regression, and a reviewer cannot tell whether the evidence supports the claim.

Three decisions depend on this topic:

1. **What to record so the result can be rebuilt.** The record must cover the physical configuration, the software configuration, and the analysis configuration. A missing vertex makes the result an anecdote.
2. **How to keep raw data separate from derived data.** Raw data is immutable. Derived data is reproducible from raw data plus a script. A pipeline that overwrites raw data destroys the evidence.
3. **How to report a result so a reviewer can weigh it.** The report must state the question, the method, the uncertainty, the acceptance criterion, the verdict, and what the result does not prove.

The commercial stakes are concrete. A design history file that cannot be reconstructed fails an audit. A verification result that cannot be traced to a requirement is not verification. A published model that cannot be reproduced costs credibility and, in a regulated product, a submission cycle.

The conventions are also a defence against self-deception. A team that records the filter order, the seed, and the exclusion rule before the run cannot quietly change them afterwards. The record does not make the team honest; it makes dishonesty visible, which is the most a process can do.

This topic defines the lab conventions for the module: the record structure, the metadata, the hashing, the calibration record, the deviation log, and the report template. It assumes the stack from Topic 2 and the safety boundary from Topic 1.

> **Safety statement for this module.** Hazards in these labs include electric shock, stored energy in capacitors and batteries, burns from soldering irons and hot water, cuts from strip tools and glass, eye injury from projectiles and ultraviolet light, pinch and crush injury from servo arms, and lithium battery fire. Required personal protective equipment (PPE) is safety glasses in every lab that solders, cuts, heats, or pressurises; a lab coat or equivalent coverage; closed shoes; and an electrostatic discharge (ESD) wrist strap whenever a bare board is handled. The electrical-safety boundary for this course is extra-low voltage only: a current-limited supply or a fused USB supply, with no mains conductor exposed on the bench. Any experiment that is mains-powered, that connects to a patient, or that uses an applied part requires qualified supervision and the institution's electrical-safety process. No lab in this module authorises such work.

## 2. Learning outcomes

By the end of this topic you can:

- Write a lab protocol with a question, a claim, an acceptance criterion, and a pre-specified analysis.
- Separate raw data, derived data, and reported results, and state which one is immutable.
- Build a manifest that records the physical, software, and analysis configuration, with hashes.
- Verify a manifest against the files it describes and report a mismatch.
- Compute a confidence interval for a proportion and a paired difference, and use it to decide whether two results differ.
- Write a deviation log that records what changed, why, and what it affected.
- Apply the ALCOA+ data-integrity attributes to a lab record and identify which attribute a given defect violates.
- Produce a lab report that a reviewer who was not in the room can follow.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A lab result is the end of a chain. Each link has an identity, a version, and a hash. The chain is only as strong as its weakest record.

### 3.1 The provenance chain

```text
  QUESTION
     |
  PROTOCOL ----------------> acceptance criterion
     |
  CONFIGURATION -----------> BOM, firmware commit, model hash
     |
  RAW DATA ----------------> hash, timestamp, instrument ID
     |
  PROCESSING SCRIPT -------> commit, parameters, seed
     |
  DERIVED DATA ------------> hash, script version
     |
  FIGURE / TABLE ----------> script, data hash
     |
  CLAIM -------------------> uncertainty, limitations
```

Every arrow is a transformation. A transformation that is not recorded breaks the chain. A transformation that is recorded but not reproducible is a weaker link than one that is.

The chain has a direction: a claim can be traced back to a question, and a question can be traced forward to a claim. A reviewer follows it in both directions. Forward, to check that the claim is supported. Backward, to check that the data is what it says it is.

### 3.2 The three configurations

Reproducibility depends on three configurations, and each has a different failure mode.

| Configuration | What it covers | Failure mode | Record |
|---|---|---|---|
| Physical | Board revision, components, wiring, instrument | A replaced part changes the measurement | BOM, photograph, instrument ID |
| Software | Firmware, libraries, OS, simulator | An update changes a default | Commit, lockfile, container digest, SBOM |
| Analysis | Script, parameters, filter, split, seed | A choice made after seeing the data | Script commit, config file, seed, pre-registration |

The physical configuration is the one most often omitted in a software-heavy lab. A result that depends on a particular board, a particular cable, or a particular probe is not reproducible from the code alone.

### 3.3 Raw, derived, and reported

The three data layers must be physically separate.

```text
  raw/        immutable, append-only, hashed
  derived/    reproducible from raw/ plus code
  reports/    figures and tables generated from derived/
```

Raw data is what the instrument produced: the ADC codes, the scope capture, the CSV export. It is never edited. If a channel needs to be inverted, the inversion happens in the processing script, not in the raw file.

Derived data is what the analysis produced: the filtered signal, the detected beats, the confusion matrix. It is reproducible from the raw data and the script. If the script changes, the derived data changes, and the report must be regenerated.

Reported results are the figures and tables in the report. They are generated from the derived data, not typed by hand. A number that is typed into a report cannot be traced to a file.

### 3.4 The reproducibility envelope

A result is reproducible within an envelope: a set of conditions under which it should hold.

| Condition | Inside the envelope | Outside the envelope |
|---|---|---|
| Board revision | Same revision, same BOM | A different revision |
| Temperature | 18–26 °C | Below 10 °C or above 35 °C |
| Supply | Same voltage and ripple | A different supply |
| Sample rate | Same rate | A different rate |
| Software | Same commit and lockfile | A different commit |
| Data | Same records and annotations | A different annotation version |

The envelope is part of the claim. A result that is reported without its envelope will be applied outside it.

### 3.5 Data integrity as a set of attributes

The ALCOA+ attributes are a checklist for a record. Each attribute catches a different defect.

| Attribute | Question | Typical defect |
|---|---|---|
| Attributable | Who created or changed this record? | An unnamed analyst, a shared login |
| Legible | Can a reviewer read it? | Unreadable scan, truncated log |
| Contemporaneous | Was it recorded at the time? | A notebook written the next day |
| Original | Is this the first record? | A transcription with no source |
| Accurate | Is it correct and free of errors? | A typo, a wrong unit, a copied value |
| Complete | Are all the data and metadata present? | Missing calibration record, missing run |
| Consistent | Is it consistent with other records? | Two files with different timestamps |
| Enduring | Will it survive the retention period? | A disk that is not backed up |
| Available | Can it be retrieved when needed? | A file on a departed student's laptop |

The attributes are not independent. A record that is not attributable is also not trustworthy as an original. A record that is not enduring is not available. The checklist is a way to find the weakest attribute before a reviewer does.

### 3.6 Verification and validation in the record

Verification and validation are different questions, and the record must answer both.

| Question | Evidence | Record |
|---|---|---|
| Did we build the thing right? | Output meets input requirement | Verification report with acceptance criterion |
| Did we build the right thing? | Device meets user needs and intended use | Validation report with simulated or actual use |
| Can someone else rebuild it? | Reproducibility envelope and manifest | Lab record with hashes and versions |

A lab report usually answers the first question. It rarely answers the second, and it often fails the third. The conventions in this topic are designed to make the third question answerable.

### 3.7 The cost of a broken link

A broken link does not always invalidate the result. It changes what the result can support.

| Broken link | What is lost | What can still be claimed |
|---|---|---|
| No question | The reason for the experiment | A measurement exists, but its relevance is unknown |
| No protocol | The pre-specification | A result, but not a confirmatory one |
| No configuration | The identity of the thing measured | A result about an unknown configuration |
| No raw data | The original evidence | A derived result, if the derivation is trusted |
| No script | The analysis method | A result, but not a reproducible one |
| No uncertainty | The weight of the evidence | A number, but not a decision |
| No limitations | The boundary of the claim | A claim that will be over-extended |

The table is the practical reason to build the chain from the start. A missing link can sometimes be reconstructed, but the reconstruction is a new record, not the original, and it must be labelled as such.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Protocol | Pre-specified plan for an experiment | Written after the data | Document |
| SOP | Standard operating procedure | Treated as a suggestion | Document |
| Raw data | Data as produced by the instrument | Edited to remove artefacts | File |
| Derived data | Data produced by processing raw data | Treated as raw | File |
| Metadata | Data about the data | Omitted | Fields |
| Provenance | Record of the origin and history of data | Confused with a file path | Record |
| Chain of custody | Record of who held and changed a sample or file | Assumed | Log |
| Data integrity | Completeness, consistency, and accuracy of data over its lifecycle | Confused with security | Property |
| ALCOA+ | Attributable, legible, contemporaneous, original, accurate, complete, consistent, enduring, available | Recited without application | Checklist |
| Audit trail | Secure, time-stamped record of create, modify, and delete actions | A text log that can be edited | Record |
| Electronic signature | Computerised signature legally equivalent to a handwritten one | An image of a signature | Record |
| 21 CFR Part 11 | US regulation for electronic records and signatures | Applied to every spreadsheet | Regulation |
| Hash | Fixed-size digest of data | Confused with encryption | Hex string |
| SHA-256 | 256-bit secure hash algorithm | Truncated in a report | Hex string |
| Commit | Immutable snapshot in version control | A saved file | Hash |
| Tag | Named reference to a commit | A branch | Name |
| Lockfile | Exact dependency versions | "Latest" | File |
| Container digest | Content hash of an image | Image tag | SHA-256 |
| SBOM | Software bill of materials | A list of licences only | Document |
| Calibration record | Evidence of comparison with a reference | A sticker | Certificate |
| Traceability | Unbroken chain of calibrations | A brand claim | Record |
| Uncertainty budget | List of uncertainty sources and their combination | A single percentage | Table |
| Acceptance criterion | Pre-defined condition for pass or fail | A criterion chosen after the result | Statement |
| Verification | Confirmation that output meets input | Confused with validation | Activity |
| Validation | Confirmation that the device meets user needs | Confused with verification | Activity |
| Deviation | Documented departure from the protocol | An undocumented change | Record |
| CAPA | Corrective and preventive action | A blame exercise | Process |
| Change control | Managed process for changing a controlled item | An edit | Process |
| DHF | Design history file | A folder of documents | Record set |
| ELN | Electronic lab notebook | A shared drive | System |
| LIMS | Laboratory information management system | A spreadsheet | System |
| Timestamp | Recorded time of an event | Time zone omitted | ISO 8601 |
| Monotonic clock | Clock that never goes backwards | Confused with wall-clock time | s |
| UTC | Coordinated Universal Time | Local time without a zone | ISO 8601 |
| NTP | Network Time Protocol | Assumed accurate to the millisecond | Protocol |
| Seed | Initial value for a random process | Omitted from the record | Integer |
| Outlier | Observation far from the others | Removed without a rule | Value |
| Exclusion criterion | Pre-specified rule for excluding data | Applied after seeing the result | Statement |
| Pre-registration | Public record of the protocol before data collection | Confused with a protocol | Record |
| SAP | Statistical analysis plan | Written after unblinding | Document |
| Reproducible | Another person can obtain the same result | The code runs on your machine | Property |
| Repeatable | The same person obtains the same result | Confused with reproducible | Property |
| Robust | The result is insensitive to small changes | Confused with reproducible | Property |

## 5. The mechanics, step by step

The procedure below runs a lab from protocol to archived report. Each step states its input, output, and rationale. The procedure is written for a two-person team: one operator and one reviewer. The reviewer can be the partner for a small lab and must be independent for a verification lab.

### Step 1. Write the question and the claim

**Input:** the engineering question and the decision it informs.

**Output:** a one-paragraph charter with the question, the claim, the decision rule, and the claim boundary.

**Rationale:** the charter determines the protocol. A claim that is not written cannot be tested, and a decision rule that is not written cannot be applied consistently.

The charter contains:

- The question, in one sentence.
- The claim that the experiment will support or refute.
- The population or unit: which board, which records, which subjects, which conditions.
- The decision rule: what result leads to which action.
- The claim boundary: what the result will not establish.

The charter is dated and versioned. A change to the charter after data collection is a deviation, not an edit.

### Step 2. Write the protocol before collecting data

**Input:** the charter.

**Output:** a protocol with the setup, the procedure, the measurements, the analysis, the acceptance criterion, and the exclusion rules.

**Rationale:** the protocol is the defence against choosing the analysis after seeing the data. A pre-specified analysis can still be wrong, but it cannot be unconsciously fitted to the result.

The protocol contains:

| Section | Content | Why it matters |
|---|---|---|
| Objective | The question and the claim | Aligns the team |
| Design | Units, conditions, repeats, randomisation | Defines the evidence |
| Setup | Instruments, wiring, fixtures, environment | Makes the bench reproducible |
| Procedure | Numbered steps with acceptance checks | Makes the run repeatable |
| Measurements | Quantity, instrument, range, resolution | Makes the data interpretable |
| Analysis | Script, filter, model, test, seed | Prevents post-hoc choices |
| Acceptance | Criterion, uncertainty, decision rule | Defines pass and fail |
| Exclusions | Rules for dropping data | Prevents selective reporting |
| Deviations | How changes will be recorded | Preserves the trail |
| Records | What will be stored, where, and for how long | Makes the result auditable |

The protocol is reviewed by a second person before the first measurement. The review checks that the acceptance criterion is achievable, that the instruments can meet the uncertainty target, and that the analysis is specified in enough detail to be reproduced.

### Step 3. Control the configuration

**Input:** the protocol and the stack record from Topic 2.

**Output:** a configuration identifier that names every physical and software element.

**Rationale:** a result is about a configuration, not about a device in general. The configuration identifier is how the report says which one.

The configuration identifier includes:

- The board revision and the BOM hash.
- The firmware commit and the build flags.
- The model hash and the conversion flags.
- The analysis commit and the lockfile hash.
- The container digest, if used.
- The instrument IDs and the calibration dates.
- The fixture ID and the wiring revision.

A configuration identifier that is a date is not an identifier. "2026-10-26" does not say which firmware was running.

### Step 4. Record the calibration before the measurement

**Input:** the instrument list and the calibration certificates.

**Output:** a calibration record with the instrument, the range, the uncertainty, the date, and the traceability statement.

**Rationale:** a calibration that is recorded after the measurement is a calibration of unknown status. The record must show that the instrument was in calibration on the day of the measurement.

The record contains:

| Field | Example | Why it matters |
|---|---|---|
| Instrument | Bench DMM, serial 12345 | Identifies the unit |
| Parameter | DC voltage | Confirms the function |
| Range | 10 V | Confirms the range used |
| Resolution | 1 µV | Sets the quantization |
| Accuracy | ±(0.0035% + 0.5 µV) | Sets the Type B term |
| Calibration date | 2026-03-14 | Confirms validity |
| Calibration due | 2027-03-14 | Confirms the interval |
| Certificate | Lab certificate 2026-1234 | Links to the record |
| Traceability | NIST via accredited lab | Links to the SI |

A calibration certificate that covers a 100 V range does not cover a 10 mV measurement. The range is part of the record.

### Step 5. Capture raw data without transformation

**Input:** the instrument and the acquisition settings.

**Output:** a raw file with the samples, the settings, and a timestamp.

**Rationale:** any transformation applied before the raw file is a choice that cannot be undone. The raw file must contain what the instrument produced.

Rules:

- Save the native format if possible, and a plain-text export if not.
- Do not apply a filter, an offset, or a scale factor in the acquisition software unless the setting is recorded.
- Record the sample rate, the range, the gain, the trigger, and the channel map.
- Record the start time in UTC, with the time zone.
- Record the instrument ID and the firmware version.
- Close the file after the capture and make it read-only.
- Compute the hash immediately.

A capture that is saved as a screenshot is not raw data. A screenshot loses the sample values, the time base, and the channel mapping.

### Step 6. Record the metadata in a manifest

**Input:** the raw file, the configuration, and the calibration record.

**Output:** a manifest that describes the run.

**Rationale:** the manifest is the index of the run. It is what a reviewer reads first, and it is what a script checks before processing.

A manifest in JSON:

| Field | Content | Example |
|---|---|---|
| run_id | Unique identifier | 2026-10-26-ecg-001 |
| question | The charter's question | Input-referred noise of AFE |
| configuration | Configuration identifier | board-r3-fw-1a2b3c |
| operator | Name or initials | A. Patel |
| reviewer | Independent reviewer | J. Chen |
| start_utc | ISO 8601 timestamp | 2026-10-26T09:14:22Z |
| instrument | ID and calibration date | DMM-12345, 2026-03-14 |
| environment | Temperature and humidity | 23.1 °C, 45% RH |
| files | Paths and SHA-256 hashes | raw/adc.csv |
| analysis | Commit and parameters | analysis@9f3c2a1 |
| result | Path and hash | derived/noise.json |
| deviations | List of deviations | None |
| limitations | What the run does not prove | One board, one temperature |

The manifest is itself hashed and committed. A manifest that can be edited without a record is not a manifest.

### Step 7. Hash and back up the raw data

**Input:** the raw file.

**Output:** a SHA-256 hash in the manifest and a backup copy in a second location.

**Rationale:** the hash proves that the analysed file is the recorded file. The backup proves that the file survives a disk failure.

Procedure:

1. Compute the SHA-256 hash of the raw file.
2. Record the hash in the manifest.
3. Copy the file to a second physical location, such as a lab server or a managed cloud store.
4. Verify the hash on the copy.
5. Record the backup location in the manifest.
6. Do not edit the file. If a correction is needed, create a new file and record the correction.

A hash is not encryption. It detects accidental or deliberate modification; it does not hide the data. For sensitive data, the hash is combined with access control and encryption.

### Step 8. Write the processing script before running it

**Input:** the raw data and the analysis plan.

**Output:** a script that reads the raw data, applies the specified processing, and writes the derived data.

**Rationale:** a script is a record of the analysis. A manual step in a spreadsheet is not reproducible and cannot be reviewed. The script must be written before the result is known, or the analysis is exploratory and must be labelled as such.

Rules:

- Read the raw file, never modify it.
- Use a configuration file for parameters, not hard-coded values.
- Set and record the random seed.
- Write the derived data with a hash.
- Print or log the input hash, the parameters, and the output hash.
- Fail loudly if the input hash does not match the manifest.
- Keep the script in version control.

A notebook that is run out of order is not a script. The final analysis must run from top to bottom in a fresh environment.

### Step 9. Run the analysis and record the log

**Input:** the script and the raw data.

**Output:** the derived data, the log, and the result summary.

**Rationale:** the log is the evidence that the script ran on the recorded data with the recorded parameters. Without it, the result is a number with no history.

The log contains:

- The start and end time.
- The script commit.
- The input hashes.
- The parameters.
- The warnings.
- The output hashes.
- The environment: interpreter, packages, platform.

A run that produced warnings and ignored them is a run with unrecorded deviations. The warnings must be reviewed and either accepted with a reason or fixed.

### Step 10. Compute the uncertainty and the interval

**Input:** the derived data and the uncertainty budget.

**Output:** the estimate, the standard uncertainty, the expanded uncertainty, and the confidence interval where applicable.

**Rationale:** a number without an uncertainty cannot be compared with a criterion. A difference without an interval cannot be distinguished from noise.

For a proportion, the Wilson interval is preferable to the normal approximation when the proportion is near 0 or 1 or the sample is small. For a paired difference, the McNemar test or a paired bootstrap is appropriate. For a continuous quantity, the GUM method applies.

The report states the method, the coverage factor, and the sample size. A confidence interval without a sample size is not interpretable.

### Step 11. Generate the figures from the data

**Input:** the derived data and a plotting script.

**Output:** figures with the data hash in the caption or the metadata.

**Rationale:** a figure that is edited by hand cannot be traced. A figure that is generated from the data can be regenerated when the data changes.

Rules:

- The plotting script reads the derived data and writes the figure.
- The figure caption states the data file and the script commit.
- The axes have units.
- The uncertainty is shown, not hidden.
- The figure is exported at a resolution that preserves the data.
- No manual annotation is added without a record.

A figure that has been "cleaned up" in an image editor is no longer evidence. If a label must be added, it is added in the script.

### Step 12. Record deviations as they happen

**Input:** any departure from the protocol.

**Output:** a deviation record with the time, the change, the reason, and the affected data.

**Rationale:** a deviation that is recorded is a documented decision. A deviation that is not recorded is a data-integrity defect.

The deviation record contains:

| Field | Example | Why it matters |
|---|---|---|
| Time | 2026-10-26T10:02Z | Orders the events |
| Protocol step | Step 6, gain setting | Locates the change |
| Original | Gain 100 | Shows what was planned |
| Actual | Gain 50 | Shows what happened |
| Reason | Clipping at gain 100 | Explains the change |
| Affected data | Runs 3–5 | Bounds the impact |
| Action | Re-ran runs 3–5 at gain 100 | Restores the plan |
| Approval | Reviewer initials | Confirms the decision |

A deviation that changes the acceptance criterion is not a deviation; it is a new protocol, and the original result must still be reported.

### Step 13. Review before reporting

**Input:** the manifest, the protocol, the derived data, the figures, and the draft report.

**Output:** a review record with findings and dispositions.

**Rationale:** the reviewer checks the chain, not only the number. A review that checks only the arithmetic misses the missing calibration, the changed filter, and the excluded record.

The review checks:

1. The question and the claim match the protocol.
2. The configuration is recorded and matches the data.
3. The raw data hash matches the manifest.
4. The analysis script reproduces the derived data.
5. The uncertainty method is stated and applied.
6. The acceptance criterion was pre-specified.
7. The exclusions follow the pre-specified rules.
8. The deviations are recorded and bounded.
9. The figures are generated from the data.
10. The limitations are stated.

A finding is recorded with a severity and a disposition: fixed, accepted with a rationale, or deferred with an owner. A deferred finding is a residual risk.

### Step 14. Archive and hand over

**Input:** the complete record.

**Output:** an archive that a reviewer can open without the original team.

**Rationale:** the record outlives the project. An archive that depends on a personal account, a local path, or a discontinued tool is not an archive.

The archive contains:

- The charter, the protocol, and the amendments.
- The manifest and the hashes.
- The raw data or a controlled-access pointer with the hash.
- The analysis scripts and the environment files.
- The derived data and the figures.
- The review record and the deviations.
- The report and the limitations.
- A README that says how to open and rerun the analysis.

The archive is tested by a person who was not on the team. If that person cannot open it, the archive is not complete.

### 5.15 Acceptance criteria for a lab record

| # | Criterion | Evidence | Fail action |
|---:|---|---|---|
| 1 | Charter is written and dated | Charter file | Write it |
| 2 | Protocol is pre-specified and reviewed | Signed protocol | Review |
| 3 | Configuration identifier is recorded | Manifest | Complete |
| 4 | Calibration is valid on the measurement date | Certificate | Recalibrate |
| 5 | Raw data is unmodified and hashed | Hash in manifest | Restore |
| 6 | Metadata covers rate, range, gain, channel map | Manifest | Complete |
| 7 | Analysis is scripted and seeded | Script and config | Rewrite |
| 8 | Derived data is reproducible | Rerun matches | Investigate |
| 9 | Uncertainty is computed and stated | Budget | Compute |
| 10 | Acceptance criterion is pre-specified | Protocol | Flag |
| 11 | Deviations are recorded | Deviation log | Record |
| 12 | Figures are generated from data | Plot script | Regenerate |
| 13 | Review is recorded | Review record | Review |
| 14 | Archive opens without the team | Test by a third party | Fix |

### 5.16 The record as a legal and regulatory object

In a regulated environment, the lab record may become a regulatory record. The conventions in this topic are a teaching version of the practices that make a record defensible.

| Context | What the record must support | Consequence of a gap |
|---|---|---|
| Design verification | A requirement-to-evidence link | The verification is incomplete |
| Design validation | Evidence of use by representative users | The validation is not established |
| Risk management | Evidence that a control was implemented and verified | The residual risk is unsupported |
| Software lifecycle | Evidence of the process and the tests | The software class claim is unsupported |
| Production | Evidence that each unit met the specification | The lot is not released |
| Post-market | Evidence for an investigation | The investigation is inconclusive |

The record does not have to be perfect. It has to be honest, traceable, and sufficient for the decision it supports.

### 5.17 What the conventions do not fix

The conventions make a result traceable. They do not make it correct.

- A well-recorded measurement of the wrong quantity is still wrong.
- A reproducible analysis of biased data is still biased.
- A complete record of an inadequate experiment is still inadequate.

The conventions are necessary and not sufficient. The scientific question, the design, and the interpretation remain the team's responsibility.

## 6. Worked example

### 6.1 The question

A team is rerunning a QRS-detection experiment from the previous semester. The previous report claimed a sensitivity of 0.94 and a positive predictive value (PPV) of 0.91 on ten records from a public arrhythmia database. The current run gives a sensitivity of 0.87 and a PPV of 0.89. The team wants to know whether the difference is a real regression or an artefact of the rerun.

### 6.2 The data

The experiment uses 1,000 annotated beats from ten records. The annotations are the reference. A beat is a true positive (TP) if the detector marks it within 150 ms of the reference annotation, a false negative (FN) if it misses it, and a false positive (FP) if it marks a beat that is not in the reference.

| Quantity | Previous run | Current run |
|---|---:|---:|
| Reference beats | 1,000 | 1,000 |
| True positives | 940 | 870 |
| False negatives | 60 | 130 |
| False positives | 93 | 108 |
| Sensitivity | 0.940 | 0.870 |
| PPV | 0.910 | 0.890 |

Sensitivity is \(TP/(TP+FN)\). PPV is \(TP/(TP+FP)\). The previous sensitivity is 940/1000 = 0.940. The current sensitivity is 870/1000 = 0.870. The difference is 0.070, or 7.0 percentage points.

### 6.2a The instrument and the annotation record

The detector is software, but the experiment still has an instrument record.

| Item | Value | Why it is recorded |
|---|---|---|
| Signal source | Public arrhythmia database, records listed | The data are licensed and versioned |
| Annotation file | Version 1.0.0, hashed | The reference beats must be identical |
| Sampling rate | 360 Hz native, resampled to 250 Hz | The detector's passband depends on it |
| Detector commit | a1b2c3d for the previous run, 9f3c2a1 for the current | The code must be identified |
| Preprocessing | Band-pass, order recorded | The root cause in this example |
| Tolerance window | 150 ms | The definition of a true positive |
| Split | By record | Prevents leakage between tuning and evaluation |

The table is short because the experiment is computational. The same fields would be longer for a bench experiment, with instrument IDs, ranges, and calibration dates. The principle is the same: the result is about a configuration, and the configuration is recorded.

### 6.3 Is the difference explained by sampling?

The same 1,000 beats are used in both runs, so the comparison is paired. The discordant pairs are the beats that one run detects and the other misses.

| Pair type | Count | Meaning |
|---|---:|---|
| Both detect | 855 | Concordant |
| Previous only | 85 | Detected before, missed now |
| Current only | 15 | Missed before, detected now |
| Neither detects | 45 | Concordant |

The McNemar statistic, with the continuity correction, is:

\[
\chi^2 = \frac{(|b - c| - 1)^2}{b + c} = \frac{(|85 - 15| - 1)^2}{85 + 15} = \frac{69^2}{100} = 47.6
\]

With one degree of freedom, 47.6 is far above the 3.84 critical value at the 0.05 level. The difference is not explained by sampling.

The standard error of the paired difference is:

\[
SE = \frac{\sqrt{b + c}}{n} = \frac{\sqrt{100}}{1000} = 0.010
\]

The 95% confidence interval for the difference is \(0.070 \pm 1.96 \times 0.010\), or 5.0 to 9.0 percentage points. The interval excludes zero.

### 6.3a What the McNemar test assumes

The McNemar test assumes that the pairs are independent of each other and that the discordant pairs are a random sample of the underlying discordance. In this experiment, the pairs are beats within records. Beats from the same record are not independent: a detector that fails on one beat in a record often fails on neighbouring beats. The test therefore understates the uncertainty.

The correct analysis would use a cluster-robust method with the record as the cluster, or a bootstrap that resamples records rather than beats. The team has ten records, which is a small number of clusters, so the interval would be wide. The reported McNemar interval of 5.0 to 9.0 points should be read as a lower bound on the uncertainty, not as the full uncertainty.

This is a limit of the example, not a flaw in the arithmetic. The arithmetic is correct for independent pairs. The claim that the difference is real is supported by the reproduction check, which is stronger evidence than the test.

### 6.4 The uncertainty of each sensitivity

Each sensitivity is a proportion. The Wilson 95% interval for the previous run is:

\[
\frac{p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z\sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}
\]

with \(p = 0.94\), \(n = 1000\), \(z = 1.96\). The centre is 0.9383 and the half-width is 0.0148, so the interval is 0.9235 to 0.9531.

For the current run, \(p = 0.87\). The centre is 0.8686 and the half-width is 0.0209, so the interval is 0.8477 to 0.8894.

The intervals do not overlap. This is consistent with the paired test, although the two intervals are not the correct comparison for paired data; the paired test is.

### 6.5 The root cause

The team checks the manifest. The raw data hashes match. The record set matches. The annotation version matches. The analysis commit differs:

| Item | Previous run | Current run | Match |
|---|---|---|---|
| Raw data hash | 3f9a...c21 | 3f9a...c21 | Yes |
| Records | 100, 101, 102, 103, 105, 106, 108, 109, 111, 112 | Same | Yes |
| Annotation version | 1.0.0 | 1.0.0 | Yes |
| Preprocessing filter order | 2 | 4 | No |
| Filter cutoff | 15 Hz | 15 Hz | Yes |
| Detection threshold | 0.35 | 0.35 | Yes |
| Analysis commit | a1b2c3d | 9f3c2a1 | No |

The current run uses a fourth-order Butterworth filter where the previous run used a second-order filter. The higher order removes more high-frequency content, which sharpens some QRS complexes and attenuates others. The change was made to reduce noise in a different experiment and was not recorded as a deviation.

This is the answer to the question: the difference is a configuration change, not a regression in the detector.

### 6.6 The reproduction check

The team reruns the analysis at the previous commit with the previous filter order and the same raw data. The result is:

| Run | Filter order | Sensitivity | PPV |
|---|---:|---:|---:|
| Previous | 2 | 0.940 | 0.910 |
| Current | 4 | 0.870 | 0.890 |
| Reproduction at previous commit | 2 | 0.940 | 0.910 |

The reproduction matches the previous result exactly. This is the evidence that the difference is caused by the filter order, not by a change in the data or the environment.

### 6.7 Two checks that would invalidate the result

**Check 1: the annotations were not versioned.** If the previous run used a different annotation version, the reference beats would not be the same, and the paired comparison would be invalid. The manifest records the annotation version, but the version string is not a content hash. The check is to hash the annotation file and compare it with the previous run's hash. A version label can be reused; a hash cannot.

**Check 2: the records were not split at the record level.** If the detector's threshold was tuned on windows from the same records that were used for evaluation, the sensitivity would be optimistically biased. The protocol says the split is by record, but the current run's log does not show the split. The check is to inspect the analysis script and the record list. A window-level split with the same records in training and test would inflate both runs, and the difference between them might still be real, but the absolute values would be meaningless.

### 6.7a What the review record should say

A review record for this example would contain findings, not a verdict.

| Finding | Severity | Evidence | Disposition |
|---|---|---|---|
| The filter order changed between runs | Major | Manifest diff | Documented as a deviation; report corrected |
| The change was not recorded at the time | Major | No deviation entry | Change-control finding; process action |
| The McNemar interval assumes independent beats | Minor | Ten records, clustered beats | Limitation added; cluster method recommended |
| The record split is not shown in the log | Major | Analysis log | Script instrumented; split list added |
| The raw data hash matches | None | Manifest | Accepted |

The record separates a technical finding from a process finding. The filter-order change is a technical finding; the missing deviation is a process finding. A review that records only the first misses the reason the change was possible.

### 6.8 A varied condition: the filter order is restored

The team reruns the current pipeline with the filter order set to 2, keeping everything else the same.

| Condition | Filter order | Sensitivity | PPV | Difference from previous |
|---|---:|---:|---:|---:|
| Previous | 2 | 0.940 | 0.910 | 0.000 |
| Current | 4 | 0.870 | 0.890 | −0.070 |
| Current pipeline, filter order 2 | 2 | 0.938 | 0.908 | −0.002 |

The difference falls from 7.0 to 0.2 percentage points. The remaining 0.2 points are within the paired confidence interval and are consistent with a small implementation difference, such as a changed threshold comparison or a floating-point difference. The result moves in the direction the root-cause analysis predicts.

### 6.9 What the example demonstrates

The arithmetic closes: 940/1000 = 0.940, 870/1000 = 0.870, a 7.0-point difference, a McNemar statistic of 47.6, and a 95% interval of 5.0 to 9.0 points. The Wilson intervals are 0.925–0.954 and 0.847–0.889.

The example also demonstrates the value of the manifest. Without it, the team would have seen a 7-point drop and assumed a regression in the detector. With it, the team found a filter-order change in a preprocessing step and reproduced the original result at the previous commit.

The example demonstrates the limit of the conventions as well. The manifest recorded the filter order, but the change was not recorded as a deviation. The conventions made the cause findable; they did not prevent the change. Prevention requires change control, which is a process, not a file format.

### 6.10 The residual risk after the correction

The team corrects the report, records the deviation, and adds the split list to the analysis log. The residual risks are:

- The detector was tuned on a subset that may overlap with the evaluation records. The split check is now documented, but the tuning history is not.
- The McNemar interval is too narrow because the beats are clustered. The report states this as a limitation.
- The ten records are a convenience sample, not a representative sample of the population. The sensitivity estimate applies to these records only.

Each residual risk is stated in the report. A residual risk that is stated is a limit of the claim; a residual risk that is hidden is an overclaim.

## 7. Code, query, or calculation

The program below verifies a lab manifest, computes the statistics from the worked example, and compares two result files. It uses only the Python standard library.

```python
import hashlib
import json
import math

# --- 1. Build a small manifest and verify hashes -------------------------
RAW = b"sample_index,adc_code\n0,2048\n1,2049\n2,2047\n"
DERIVED = b'{"sensitivity": 0.87, "ppv": 0.89}\n'

def sha256(data):
    return hashlib.sha256(data).hexdigest()

manifest = {
    "run_id": "2026-10-26-qrs-001",
    "configuration": "board-r3-fw-1a2b3c",
    "files": {
        "raw/adc.csv": sha256(RAW),
        "derived/result.json": sha256(DERIVED),
    },
    "analysis": {"commit": "9f3c2a1", "filter_order": 4, "cutoff_hz": 15.0},
}

def verify(manifest, contents):
    ok = True
    for path, expected in manifest["files"].items():
        actual = sha256(contents[path])
        status = "OK" if actual == expected else "MISMATCH"
        if actual != expected:
            ok = False
        print(f"{path:24s} {status}")
    return ok

print("manifest verification")
print("--------------------")
verified = verify(manifest, {
    "raw/adc.csv": RAW,
    "derived/result.json": DERIVED,
})
print(f"all files verified: {verified}")

# --- 2. Wilson interval for a proportion --------------------------------
def wilson(p, n, z=1.96):
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half

print("\nWilson 95% intervals")
print("--------------------")
for label, tp, n in (("previous", 940, 1000), ("current", 870, 1000)):
    p = tp / n
    lo, hi = wilson(p, n)
    print(f"{label:9s} p={p:.3f}  CI=({lo:.4f}, {hi:.4f})  width={hi-lo:.4f}")

# --- 3. McNemar paired comparison ---------------------------------------
b, c, n = 85, 15, 1000
chi2 = (abs(b - c) - 1) ** 2 / (b + c)
se = math.sqrt(b + c) / n
diff = (b - c) / n
lo = diff - 1.96 * se
hi = diff + 1.96 * se
print("\nMcNemar paired comparison")
print("-------------------------")
print(f"discordant b={b}, c={c}")
print(f"difference       = {diff:.3f}")
print(f"chi2 (1 df)      = {chi2:.1f}")
print(f"SE of difference = {se:.4f}")
print(f"95% CI           = ({lo:.4f}, {hi:.4f})")
print(f"excludes zero    = {lo > 0 or hi < 0}")

# --- 4. Compare two result files ----------------------------------------
def load_result(text):
    return json.loads(text)

prev = load_result('{"sensitivity": 0.940, "ppv": 0.910}')
curr = load_result(DERIVED.decode())
print("\nresult comparison")
print("-----------------")
for key in prev:
    d = curr[key] - prev[key]
    print(f"{key:12s} previous={prev[key]:.3f} current={curr[key]:.3f} delta={d:+.3f}")

# --- 5. Hash a configuration dictionary canonically ---------------------
def config_hash(cfg):
    canon = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    return sha256(canon)

cfg_a = {"filter_order": 2, "cutoff_hz": 15.0, "threshold": 0.35}
cfg_b = {"threshold": 0.35, "cutoff_hz": 15.0, "filter_order": 4}
print("\nconfiguration hashes")
print("---------------------")
print(f"order 2: {config_hash(cfg_a)[:16]}")
print(f"order 4: {config_hash(cfg_b)[:16]}")
print(f"identical: {config_hash(cfg_a) == config_hash(cfg_b)}")
```

Expected output:

```text
manifest verification
--------------------
raw/adc.csv              OK
derived/result.json      OK
all files verified: True

Wilson 95% intervals
--------------------
previous  p=0.940  CI=(0.9235, 0.9531)  width=0.0296
current   p=0.870  CI=(0.8477, 0.8894)  width=0.0417

McNemar paired comparison
-------------------------
discordant b=85, c=15
difference       = 0.070
chi2 (1 df)      = 47.6
SE of difference = 0.0100
95% CI           = (0.0504, 0.0896)
excludes zero    = True

result comparison
-----------------
sensitivity  previous=0.940 current=0.870 delta=-0.070
ppv          previous=0.910 current=0.890 delta=-0.020

configuration hashes
---------------------
order 2: fc79fd49308a89e6
order 4: cc3c67c2d98966ec
identical: False
```

The program demonstrates four conventions. The manifest verification catches a modified file before the analysis runs. The Wilson interval is wider for the lower proportion, which is a property of the binomial variance and not an artefact. The McNemar test uses the discordant pairs, not the total, which is why it is more sensitive than comparing two independent proportions. The canonical JSON hash changes when a parameter changes, even though the key order is different, which is why the configuration hash is a useful configuration identifier.

The hash prefixes in the output depend on the exact bytes of the configuration and the JSON canonicalisation. They will be the same on any Python 3.11 installation because the canonical form is deterministic. The manifest hashes depend on the embedded byte strings and are likewise deterministic.

## 8. Data, standards, and regulatory context

### 8.1 Design controls and the design history file

In the United States, 21 CFR 820.30 defines design controls for Class II and Class III devices and for certain Class I devices. The regulation requires design and development planning, design input, design output, design review, design verification, design validation, design transfer, design changes, and a design history file (DHF).

The DHF is the record set that demonstrates the design was developed according to the approved plan and the regulation. A lab record that is intended to support a verification activity must be linkable to a requirement, a risk control, and a test. The conventions in this topic are the minimum structure for that link.

The FDA's Quality Management System Regulation (QMSR), effective 2 February 2026, incorporates ISO 13485:2016 by reference. Clause 7.3 of ISO 13485 covers design and development, and Clause 7.3.10 requires a design and development file for each device type or family. The file is the ISO counterpart of the DHF.

### 8.2 Electronic records and signatures

21 CFR Part 11 applies to electronic records created or maintained under agency record requirements and to records submitted to the FDA. Section 11.10 lists the controls for closed systems, including validation, accurate and complete copies, record protection, access limitation, and secure, computer-generated, time-stamped audit trails. Section 11.50 specifies the signature manifestation, and Section 11.70 requires the signature to be linked to the record.

The FDA's 2003 guidance on the scope and application of Part 11 states that the agency interprets the scope narrowly and exercises enforcement discretion in several areas, while still expecting the predicate-rule requirements to be met. The practical lesson for a lab is that a record which may become a regulatory record should be created with an audit trail from the start; retrofitting an audit trail is not possible.

A teaching lab is not required to comply with Part 11. It can adopt the parts that make a record defensible: unique user identity, time-stamped entries, no destructive edits, and a retained original.

### 8.3 Data integrity attributes

The ALCOA+ attributes are defined in WHO guidance and used by the Pharmaceutical Inspection Co-operation Scheme (PIC/S), the Therapeutic Goods Administration (TGA), the Organisation for Economic Co-operation and Development (OECD), and the European Medicines Agency (EMA). The attributes are attributable, legible, contemporaneous, original, accurate, complete, consistent, enduring, and available.

The EMA's 2023 guideline on computerised systems and electronic data in clinical trials extends the set to ALCOA++ by adding traceable. The extension is useful for a lab: a record that cannot be traced to its source is not complete in the sense that matters.

### 8.4 Laboratory competence

ISO/IEC 17025:2017 Clause 7.5 requires technical records that contain the results, the conditions, and the information needed to reproduce the activity. Clause 7.8 specifies the report content, including the uncertainty, the method, and the compliance statement. Clause 8.4 requires the control of records.

An accredited laboratory's records are a useful model even for a teaching lab. The record is designed for a reviewer who was not present, and it is retained for a defined period.

### 8.5 Uncertainty and statistics

The GUM (JCGM 100:2008) defines the uncertainty evaluation. The Wilson interval is a standard method for a binomial proportion; it is preferred to the normal approximation when the proportion is near the boundaries or the sample is small. The McNemar test is a standard method for paired binary data.

A report must state the method. "95% confidence interval" without a method is ambiguous, because the normal, Wilson, and exact intervals differ.

### 8.6 Software lifecycle records

IEC 62304:2006 + Amendment 1:2015 requires a software development plan, requirements, architecture, detailed design, unit implementation, integration, system testing, release, and maintenance. The standard's Class A, B, and C classification determines the rigour. The records must show the process, not only the result.

A lab that writes analysis code is not developing medical device software unless the code is part of the device. But the habits transfer: version control, requirements, tests, and a release record.

### 8.7 Retention and availability

A record that is not retained is not a record. The retention period depends on the context.

| Context | Typical retention | Source of the requirement |
|---|---|---|
| Design history file | Life of the device plus a defined period | 21 CFR 820.180, ISO 13485 |
| Technical documentation | 10 years after the last device is placed on the market; 15 years for implantable devices | EU MDR Article 10(8) |
| Calibration record | At least the interval plus one cycle | Laboratory policy, ISO/IEC 17025 |
| Clinical trial record | Defined by the sponsor and the regulation | ICH E6, 21 CFR Part 312 |
| Teaching lab | One academic year, or as the institution requires | Course policy |

The retention period is a minimum, not a target. A record that is retained but unreadable is not retained in the sense that matters. The archive must be tested: a person who was not on the team must be able to open it and find the result.

The EU Medical Device Regulation (MDR) Article 10(8) requires the manufacturer to keep the technical documentation available for at least 10 years after the last device covered by the declaration of conformity has been placed on the market, and for implantable devices for at least 15 years. The exact period and the starting event should be checked against the current consolidated text.

### 8.8 What is binding and what is convention

| Document | Status | Binds | Does not bind |
|---|---|---|---|
| 21 CFR 820.30 | Regulation | US device manufacturers | A teaching lab |
| FDA QMSR | Regulation, effective 2026-02-02 | US device manufacturers | A teaching lab |
| 21 CFR Part 11 | Regulation, narrow scope | Electronic records under predicate rules | A lab notebook |
| ISO 13485:2016 | International standard | Quality management systems | A course project |
| ISO/IEC 17025:2017 | International standard | Accredited laboratories | A teaching lab |
| IEC 62304 | International standard | Medical device software | Analysis scripts |
| JCGM 100:2008 | International guidance | Uncertainty statements | A rough estimate |
| ALCOA+ | Guidance, adopted in inspections | Regulated records where adopted | A lab notebook |

## 9. Industry practice and case evidence

### 9.1 How device companies manage the DHF

A device company typically maintains the DHF as a controlled record set in a quality management system. Each document has an identifier, a version, an owner, and an approval. The DHF is indexed so that a reviewer can follow a requirement to a risk control, a verification test, and a result.

The practical difference from a student lab is not the tool. It is the discipline of the link. A requirement that is not linked to a test is a requirement that was not verified. A test that is not linked to a requirement is a test that does not support the design.

### 9.2 Electronic lab notebooks

An electronic lab notebook (ELN) provides time-stamped entries, attachments, and a searchable record. A laboratory information management system (LIMS) provides sample tracking, instrument integration, and result reporting. Both are common in pharmaceutical and clinical laboratories.

A small team does not need an ELN. It needs a repository with a manifest, a hash, and a README. The tool is less important than the structure.

### 9.3 Open-science reproducibility

The open-science movement has produced practices that transfer directly to a lab: a repository with code and data, a lockfile, a container, a README, and a licence. A paper that is accompanied by a runnable analysis is more useful than a paper that is accompanied by a description.

The limit is that a repository can be complete and still not reproduce the result, because the physical configuration is not in the repository. The manifest closes that gap.

### 9.4 Data-integrity findings in inspections

Regulatory inspections of pharmaceutical and device manufacturers have repeatedly found data-integrity defects: shared logins, uncontrolled spreadsheets, missing audit trails, backdated records, and deleted raw data. The specific findings vary by inspection; the pattern is consistent.

The lab response is not to buy a system. It is to make the record attributable, contemporaneous, original, and complete by design. A named operator, a timestamp at the time of the action, and an immutable raw file prevent most of the common findings.

### 9.5 Reproducibility in machine learning for health

Machine-learning papers in health often report a result that cannot be reproduced because the data split, the preprocessing, or the model version is not specified. The field has produced checklists and reporting standards to address this. The lab conventions in this topic are a small-scale version: specify the split, the preprocessing, the seed, and the version before the result is known.

### 9.6 What a reviewer does when the record is incomplete

A reviewer who finds an incomplete record has three options: request the missing evidence, accept the result with a documented limitation, or reject the result. The choice depends on the severity of the gap and the consequence of the decision.

| Gap | Consequence if accepted | Typical disposition |
|---|---|---|
| Missing raw data | The result cannot be verified | Reject, unless the result is not relied upon |
| Missing calibration | The uncertainty is unknown | Accept with a widened uncertainty, or recalibrate and repeat |
| Missing script | The analysis cannot be reproduced | Reject for a verification claim; accept for an exploratory note |
| Missing deviation | The cause of a difference is unknown | Investigate; do not accept the result as a reproduction |
| Missing exclusion rule | The sample may be biased | Re-analyse with a pre-specified rule, or report as exploratory |
| Missing configuration | The result is about an unknown configuration | Reject for a design claim |

The reviewer's disposition is part of the record. A rejection without a reason is not reviewable, and an acceptance without a limitation is an overclaim.

### 9.7 Change control in practice

A change-control procedure answers four questions: what changed, why, what does it affect, and who approved it. In a device company, the change is recorded in a change request, the impact is assessed against the requirements and the risk file, and the verification is repeated where the impact reaches it.

The filter-order change in the worked example would have triggered a change request. The impact assessment would have asked whether the change affects the detector's sensitivity, whether the acceptance criterion still holds, and whether the previous verification remains valid. The answer would have been yes to the first, no to the second for the reported result, and no to the third.

A teaching lab does not need a change-request system. It needs a deviation log and the habit of asking the four questions. The habit is what transfers.

### 9.8 Inference

It is an inference, not a documented finding, that teams which maintain a manifest from the first run spend less time reconstructing results at the end of a project. The reasoning is that the manifest is written when the information is available, and reconstruction is expensive precisely because the information is gone. The inference is consistent with the practice in regulated industries, but it has not been measured in this course.

## 10. Failure modes and red flags

| # | Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---:|---|---|---|---|---|
| 1 | Protocol written after the data | Analysis section is vague | Post-hoc choice | Biased result | Pre-register the analysis |
| 2 | Raw data edited in place | File timestamp changes after capture | Loss of original | Unverifiable result | Immutable raw, hash |
| 3 | No manifest | "It is in the shared folder" | No index | Unfindable evidence | Manifest per run |
| 4 | Shared login | Two people, one account | No attribution | Unattributable record | Unique identities |
| 5 | Audit trail editable | A text file that can be changed | No integrity | Untrustworthy record | Append-only log |
| 6 | Version drift | "It worked last week" | Dependency change | Irreproducible result | Lockfile, digest |
| 7 | Undocumented deviation | A setting differs from the protocol | No record | Unexplained difference | Deviation log |
| 8 | Exclusion after seeing data | A record dropped without a rule | Selective reporting | Inflated result | Pre-specified exclusions |
| 9 | Figure edited by hand | Caption does not match the data | Untraceable image | Misleading evidence | Generate from script |
| 10 | Uncertainty omitted | A single number with no interval | False precision | Wrong decision | Budget and interval |
| 11 | Calibration expired | Certificate date before the run | Unknown error | Invalid measurement | Check before the run |
| 12 | Time zone omitted | "10:14" with no zone | Ambiguous ordering | Wrong sequence | ISO 8601 UTC |
| 13 | Seed not recorded | "The result varies a little" | Irreproducible randomness | Unrepeatable run | Record the seed |
| 14 | Archive on a personal drive | Departed student's laptop | Loss | Evidence gone | Managed archive |
| 15 | Report number typed by hand | Number not in any file | Transcription error | Wrong claim | Generate from data |
| 16 | Reviewer checks arithmetic only | Review record lists no chain items | Missed configuration change | False assurance | Chain checklist |
| 17 | Analysis run out of order | Results depend on earlier cells | Hidden state | Irreproducible analysis | Top-to-bottom run in a fresh environment |
| 18 | Timestamp from a local clock | Two files disagree on order | Time zone or clock drift | Wrong sequence | UTC and NTP, record the source |

## 11. Assumptions and limits

| # | Assumption | Why it is needed | What breaks it | How to test it |
|---:|---|---|---|---|
| 1 | The raw file is the instrument's output | The provenance chain starts here | Acquisition software applied a filter | Compare with the instrument's native format |
| 2 | The hash was computed before any edit | The hash proves integrity | The file was edited before hashing | Hash at capture time |
| 3 | The manifest describes the run that produced the data | The index is accurate | A later run overwrote the manifest | Version the manifest; hash it |
| 4 | The analysis script is deterministic | The rerun matches | Unseeded randomness, thread order | Rerun twice; compare |
| 5 | The environment is pinned | The rerun uses the same libraries | Floating dependencies | Rebuild from the lockfile |
| 6 | The reviewer is independent | The review is a check | The reviewer is the operator | Assign a different person |
| 7 | The acceptance criterion was pre-specified | The test is valid | The criterion was changed after the result | Date the protocol; keep versions |
| 8 | The exclusions follow the protocol | The sample is unbiased | A record was dropped for a convenient reason | Log every exclusion with a reason |
| 9 | The uncertainty method is appropriate | The interval is meaningful | Normal approximation near 0 or 1 | Use Wilson or exact methods |
| 10 | The archive is readable without the team | The record outlives the project | Proprietary format, personal account | Test with a third party |
| 11 | The configuration identifier is unique | The result is about a configuration | A date used as an identifier | Hash the configuration |
| 12 | The lab record is not a regulatory record | The conventions are proportionate | The record is submitted to a regulator | Apply Part 11 and QMS controls |
| 13 | The clock is accurate | The timestamps order the events | NTP failure, time zone error | Wrong sequence | Compare with a reference clock |
| 14 | The archive medium is readable | The record survives | Obsolete format or drive | Evidence lost | Migrate on a schedule; test restore |

## 12. Exercises

### Exercise 1. Write a protocol (application)

**Deliverable:** a protocol for a 30-minute bench experiment, with the objective, design, setup, procedure, measurements, analysis, acceptance criterion, exclusions, and records.

**Acceptance criteria:** the analysis is specified in enough detail that a partner could run it without asking a question; the acceptance criterion is stated before the data; the exclusion rules are stated; the record plan names the files and the hashes.

**Time:** 60 minutes.

### Exercise 2. Build a manifest (application)

**Deliverable:** a JSON manifest for a lab run, with the fields in Section 5, Step 6, and a SHA-256 hash for every file.

**Acceptance criteria:** the manifest validates as JSON; every file hash matches the file; the configuration identifier is unique; the limitations field names at least two things the run does not prove.

**Time:** 45 minutes.

### Exercise 3. Verify a manifest (calculation)

**Deliverable:** a script that reads a manifest, recomputes the hashes, and reports a mismatch with the file name and the expected and actual values.

**Acceptance criteria:** the script exits non-zero on a mismatch; it reports every file, not only the first failure; it handles a missing file without crashing; the output is reproducible.

**Time:** 45 minutes.

### Exercise 4. Compute an interval (calculation)

**Deliverable:** a Wilson 95% interval for a sensitivity of 0.87 with 1,000 beats, and a McNemar comparison for b = 85, c = 15.

**Acceptance criteria:** the interval is computed by the formula, not by a library; the McNemar statistic includes the continuity correction; the conclusion states whether the difference is explained by sampling; the sample size is stated.

**Time:** 45 minutes.

### Exercise 5. Write a deviation log (application)

**Deliverable:** a deviation log for three changes that occurred during a lab: a gain change, a dropped record, and a filter-order change.

**Acceptance criteria:** each entry states the time, the protocol step, the original, the actual, the reason, the affected data, the action, and the approval; the log distinguishes a deviation from a new protocol; the impact on the result is stated.

**Time:** 45 minutes.

### Exercise 6. Audit a record (critique)

**Deliverable:** a review of a provided lab record against the 14 acceptance criteria in Section 5.15, with a finding and a severity for each failure.

**Acceptance criteria:** every criterion is addressed; at least three findings are substantive; each finding names the evidence and the consequence; the review separates a documentation gap from a technical error.

**Time:** 60 minutes.

### Exercise 7. Reproduce a result (design)

**Deliverable:** a plan to reproduce a result from a previous semester, including the configuration diff, the data hash check, the environment rebuild, and the acceptance criterion for "reproduced."

**Acceptance criteria:** the plan names the three configurations; it defines what counts as a reproduction (exact, within uncertainty, or not reproduced); it identifies the most likely source of a mismatch; it states what to do if the raw data is missing.

**Time:** 60 minutes.

### Exercise 8. Design a record for a regulated context (design)

**Deliverable:** a one-page record design for a verification test that may be submitted to a regulator, mapping each record element to 21 CFR 820.30, ISO 13485 Clause 7.3, and the relevant Part 11 controls.

**Acceptance criteria:** the design covers attribution, audit trail, retention, and signature; it distinguishes the elements required by regulation from those adopted by convention; it states which elements the teaching lab cannot satisfy.

**Time:** 60 minutes.

## 13. Self-check answers

### 13.1 What is the difference between raw data and derived data?

Raw data is what the instrument produced. It is immutable and hashed. Derived data is produced by processing raw data with a script. It is reproducible from the raw data and the script. A pipeline that edits raw data destroys the evidence; a pipeline that cannot regenerate derived data has no evidence.

### 13.2 Why is a hash not encryption?

A hash is a one-way digest. It detects a change but does not hide the data. Encryption transforms the data so that it cannot be read without a key. A hash is used for integrity; encryption is used for confidentiality. A lab usually needs both, for different reasons.

### 13.3 Compute the sensitivity for 870 true positives and 130 false negatives.

\[
\text{Sensitivity} = \frac{870}{870 + 130} = 0.870
\]

The result is 87.0%.

### 13.4 Compute the PPV for 870 true positives and 108 false positives.

\[
\text{PPV} = \frac{870}{870 + 108} = \frac{870}{978} = 0.8896
\]

The result is 89.0%.

### 13.5 What does the McNemar test compare?

It compares the discordant pairs in a paired binary comparison. It tests whether the number of pairs where the first condition is positive and the second is negative differs from the reverse. It does not compare the total proportions as if they were independent.

### 13.6 Why is the Wilson interval preferred to the normal approximation?

The normal approximation can produce an interval that extends below 0 or above 1, and it is inaccurate when the proportion is near a boundary or the sample is small. The Wilson interval is derived from the score test and stays within the feasible range. For a proportion of 0.87 with 1,000 samples, the difference is small but the Wilson interval is still the better default.

### 13.7 What are the ALCOA+ attributes?

Attributable, legible, contemporaneous, original, accurate, complete, consistent, enduring, and available. Some guidance adds traceable, making ALCOA++. The attributes are a checklist for a record: who created it, can it be read, was it recorded at the time, is it the original, is it correct, is it complete, is it consistent, will it last, and can it be retrieved.

### 13.8 What is a deviation, and when does it become a new protocol?

A deviation is a documented departure from the protocol that does not change the question or the acceptance criterion. If the change alters the question, the population, or the acceptance criterion, it is a new protocol, and the original result must still be reported. A deviation that is not documented is a data-integrity defect.

### 13.9 Why does a manifest need a hash of itself?

Because a manifest that can be edited without a record can be changed to match any data. Hashing the manifest and recording the hash in a separate, controlled location makes the manifest tamper-evident. The hash does not prevent editing; it makes the edit detectable.

### 13.10 What would you do if the raw data were missing?

Report the result as unsupported by raw data, reconstruct what can be reconstructed from derived data and logs, and state the limitation. Do not recreate the raw data from a figure or a summary. A reconstructed file is a new file, not the original, and must be labelled as such.

### 13.11 Rubric for Exercise 6

A strong audit addresses every criterion, distinguishes a documentation gap from a technical error, and states the consequence of each finding. It does not accept "the result looks right" as evidence. A weak audit checks only the arithmetic and the formatting.

### 13.12 What is the difference between a manifest and a protocol?

A protocol says what will be done and why. A manifest says what was done, with which files, hashes, and configuration. The protocol is written before the run; the manifest is written during or immediately after it. A run can follow the protocol and still have an incomplete manifest, and a manifest can describe a run that deviated from the protocol. Both are needed.

### 13.13 Why is a confidence interval not a probability that the result is correct?

A confidence interval describes the sampling variability of an estimate under a model. It does not include the probability that the model is wrong, that the sample is biased, or that the measurement is systematically off. A 95% interval means that, under repeated sampling and the model, about 95% of such intervals would contain the true value. It is not a 95% probability that this interval contains the truth.

### 13.14 Why should the raw data be read-only?

A read-only file cannot be edited by accident or by a later analysis step. The processing script reads it and writes a new file, so the original remains available for comparison. If a correction is needed, a new file is created with a record of the correction. The read-only attribute is a technical control that supports the procedural rule; it is not a substitute for the rule, because an administrator can change it.

### 13.15 Rubric for Exercise 7

A strong reproduction plan names the three configurations, the hash checks, the environment rebuild, and the criterion for success. It anticipates the most likely mismatch and says how to isolate it. A weak plan says "rerun the code" without saying what would count as a reproduction.

## 14. Interview questions

### 14.1 What makes a lab result reproducible?

A strong answer names the physical, software, and analysis configurations, the raw data hash, and the environment pin. It says reproducibility is a property of the whole record, not of the code alone. A weak answer says "the code is in git."

### 14.2 What is the difference between a DHF and a lab notebook?

A DHF is a controlled record set that demonstrates the design was developed according to the plan and the design-control regulation. A lab notebook is a contemporaneous record of work. A notebook can contribute to a DHF, but it is not a DHF. A strong answer notes the traceability requirement.

### 14.3 Why is raw data immutable?

Because any edit destroys the original and makes the result unverifiable. A strong answer says the processing happens on a copy and the derived data is reproducible from the raw data and the script. A weak answer says "we keep a backup."

### 14.4 What does 21 CFR Part 11 require for an audit trail?

A secure, computer-generated, time-stamped record of create, modify, and delete actions, retained as long as the record, available for review, and not obscuring prior information. A strong answer notes the narrow scope of Part 11 and the FDA's enforcement discretion. A weak answer says "we log to a text file."

### 14.5 How do you decide whether two results differ?

A strong answer uses the paired structure if the data are paired, computes the difference and its interval, and checks whether the interval excludes zero. It notes that overlapping marginal intervals are not the correct test for paired data. A weak answer compares the two point estimates.

### 14.6 What is a deviation, and why record it?

A deviation is a documented departure from the protocol. It is recorded to preserve the trail, bound the impact, and show that the change was a decision rather than an accident. A strong answer distinguishes a deviation from a new protocol. A weak answer says deviations are failures.

### 14.7 What is the difference between verification and validation?

Verification confirms that the output meets the input requirement. Validation confirms that the device meets user needs and intended use. A strong answer gives an example: a filter that meets its cutoff specification is verified; a device that a clinician can use safely is validated. A weak answer uses the terms interchangeably.

### 14.8 How would you audit a result you did not produce?

A strong answer follows the provenance chain: question, protocol, configuration, raw data hash, script, derived data, figure, claim. It checks the acceptance criterion, the uncertainty, the deviations, and the limitations. A weak answer checks the arithmetic and the formatting.

### 14.9 What is the reproducibility envelope?

The set of conditions under which the result should hold: board revision, temperature, supply, sample rate, software version, data version. A strong answer says the envelope is part of the claim and that a result applied outside it is an extrapolation. A weak answer says the result is reproducible if the code runs.

### 14.10 What would you do if a reviewer found a discrepancy between the report and the data?

A strong answer thanks the reviewer, reproduces the discrepancy, determines whether it is a transcription error, a data change, or an analysis change, corrects the record with a documented change, and reports the correction. It does not quietly edit the report. A weak answer says the reviewer misread it.

### 14.11 Why is a typed number in a report a defect?

Because it cannot be traced to a file and can be changed without a record. A strong answer says the number should be generated from the derived data and the figure or table should carry the data hash. A weak answer says typing is faster.

### 14.12 What is the minimum record for a 30-minute lab?

A strong answer names the question, the configuration, the raw data and hash, the script and parameters, the uncertainty, the acceptance criterion, the deviations, and the limitations. It notes that the minimum is the set that lets another person reproduce the result, not the set that fits on one page.

## 15. Key terms

- **ALCOA+** — nine data-integrity attributes from attributable to available.
- **Audit trail** — secure, time-stamped record of create, modify, and delete actions.
- **CAPA** — corrective and preventive action.
- **Chain of custody** — record of who held and changed a sample or file.
- **Change control** — managed process for changing a controlled item.
- **Commit** — immutable snapshot in version control.
- **Configuration identifier** — unique name for a physical and software configuration.
- **Container digest** — content hash that pins an image.
- **Contemporaneous** — recorded at the time of the action.
- **Derived data** — data produced by processing raw data.
- **Deviation** — documented departure from the protocol.
- **DHF** — design history file.
- **ELN** — electronic lab notebook.
- **Exclusion criterion** — pre-specified rule for excluding data.
- **Hash** — fixed-size digest of data.
- **ISO 8601** — international standard for date and time representation.
- **LIMS** — laboratory information management system.
- **Lockfile** — exact dependency versions.
- **Manifest** — index of a run with files, hashes, and metadata.
- **McNemar test** — paired test for binary outcomes.
- **Metadata** — data about the data.
- **Monotonic clock** — clock that never goes backwards.
- **NTP** — Network Time Protocol.
- **Pre-registration** — public record of a protocol before data collection.
- **Provenance** — origin and history of data.
- **Raw data** — data as produced by the instrument.
- **Reproducible** — another person can obtain the same result.
- **Repeatable** — the same person obtains the same result.
- **SAP** — statistical analysis plan.
- **SBOM** — software bill of materials.
- **Seed** — initial value for a random process.
- **SHA-256** — 256-bit secure hash algorithm.
- **SOP** — standard operating procedure.
- **Tag** — named reference to a commit.
- **Traceability** — unbroken chain of calibrations or records.
- **UTC** — Coordinated Universal Time.
- **Verification** — confirmation that output meets input.
- **Validation** — confirmation that the device meets user needs.
- **Wilson interval** — score-based confidence interval for a proportion.
- **Change request** — controlled record of a proposed change and its impact.
- **Exploratory analysis** — analysis not pre-specified, and labelled as such.
- **Immutable** — unable to be modified after creation.
- **Provenance chain** — the linked record from question to claim.
- **Record set** — the collection of records that together support a claim.
- **Retention period** — the minimum time a record must be kept.
- **Tamper-evident** — a record whose modification can be detected.
- **Test record** — the record of a single execution of a test.
- **Traceable** — able to be followed back to a source.

## 16. Sources

All URLs accessed 26 October 2026 unless otherwise stated.

### Regulation and standards

- 21 CFR 820.30, Design Controls. https://www.govinfo.gov/content/pkg/CFR-2025-title21-vol8/pdf/CFR-2025-title21-vol8-sec820-30.pdf
- 21 CFR Part 11, Electronic Records; Electronic Signatures. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-11
- FDA, Part 11, Electronic Records; Electronic Signatures — Scope and Application (2003). https://www.fda.gov/regulatory-information/search-fda-guidance-documents/part-11-electronic-records-electronic-signatures-scope-and-application
- FDA, Quality Management System Regulation, Design and Development. https://www.fda.gov/media/189041/download
- ISO 13485:2016, Medical devices — Quality management systems. https://www.iso.org/standard/59752.html
- ISO/IEC 17025:2017, General requirements for the competence of testing and calibration laboratories. https://www.iso.org/standard/66912.html
- IEC 62304:2006+AMD1:2015, Medical device software — Software life cycle processes. https://www.iso.org/standard/64686.html
- JCGM 100:2008, Guide to the expression of uncertainty in measurement. https://www.bipm.org/documents/20126/2071204/JCGM_100_2008_E.pdf

### Data integrity guidance

- WHO TRS 1033, Annex 4, Guideline on Data Integrity. https://www.who.int/docs/default-source/medicines/norms-and-standards/guidelines/inspections/trs1033-annex4-guideline-on-data-integrity.pdf
- OECD, GLP Data Integrity, Advisory Document No. 22 (2021). https://www.oecd.org/content/dam/oecd/en/publications/reports/2021/09/glp-data-integrity_c2f067ec/45779212-en.pdf
- EMA, Guideline on computerised systems and electronic data in clinical trials (2023). https://www.ema.europa.eu/en/documents/regulatory-procedural-guideline/guideline-computerised-systems-and-electronic-data-clinical-trials_en.pdf
- TGA, Data Management and Data Integrity. https://www.tga.gov.au/products/regulations-all-products/manufacturing/data-management-and-data-integrity-dmdi
- FDA, Electronic Systems, Electronic Records, and Electronic Signatures in Clinical Investigations: Questions and Answers (2024). https://www.fda.gov/media/166215/download

### Methods

- National Institute of Standards and Technology, Metrological Traceability. https://www.nist.gov/metrology/metrological-traceability
- Wilson, E. B. (1927). Probable inference, the law of succession, and statistical inference. Journal of the American Statistical Association.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika.

> **Note:** The standard texts are paywalled. The guidance and regulation texts above were used to check the conventions. Confirm every clause and requirement against the current edition before relying on it in a regulated record.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_bench-and-simulation-stack.md`](02_bench-and-simulation-stack.md)
- Next topic: [`04_sensor-and-signal-labs.md`](04_sensor-and-signal-labs.md)
- Sibling: [`01_safety-and-standards.md`](01_safety-and-standards.md)
- Sibling: [`09_evidence-dossier.md`](09_evidence-dossier.md)
- Survey file: [`../../course/24_hardware_robotics_labs.md`](../../course/24_hardware_robotics_labs.md)
- Lab notebooks and fixtures: [`../../labs/`](../../labs/)
