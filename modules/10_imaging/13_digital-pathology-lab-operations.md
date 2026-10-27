# Digital Pathology Lab Operations

> **Module 10 — Medical Imaging and Pathology AI**
> Topic 13 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, 11, and 12
> Version 2026.3

## 1. Why this topic exists

A pathology laboratory is a production system with a regulatory obligation. Tissue arrives,
is fixed, processed, embedded, cut, stained, scanned, read, and reported, and every step has a
clock, a custodian, and a failure mode. An artificial intelligence (AI) model is one station in
that system. It can be excellent and still fail, because the slide it needs arrived at 16:40, the
scanner was down for calibration, the barcode would not read, or the laboratory information system
(LIS) sent the result to a worklist nobody was watching.

The industry problem is that data scientists are usually taught the model and not the line. They
optimise area under the receiver operating characteristic curve (AUROC) while the laboratory
optimises turnaround time (TAT), scanner utilisation, and report accuracy. The two sets of numbers
are related, but not in the way either group assumes: a model that adds two minutes to a
pathologist's review of 600 slides adds twenty hours of pathologist time per day, which is more
than the model saves if it prevents nothing.

Four decisions are blocked until someone can read the operation:

| Decision | What operational knowledge provides |
|---|---|
| Scanner count and shift pattern | Whether capacity meets peak arrival, and with what queue |
| AI deployment model | Whether inference runs before review, during review, or after sign-out |
| Integration design | Which identifiers join the slide, the order, and the report |
| Quality control | Which failures are caught before a diagnosis, and which are caught after |

### 1.1 Why the laboratory, not the model, sets the ceiling

Three properties of the laboratory dominate model performance in practice.

**Arrival is bursty.** Slides do not arrive uniformly. A morning batch from overnight processing
and a mid-day batch from same-day biopsies create peaks that a daily average hides. Capacity
planned on the average fails at the peak.

**Identity is physical.** A slide is a piece of glass with a label. The label can be misread, the
barcode can fail, and the cassette-to-slide mapping can be wrong. Every digital system inherits
these physical errors and adds new ones of its own.

**Quality is not binary.** A scan can be adequate for one task and inadequate for another. A slide
that is slightly out of focus in one corner may be fine for a diagnosis at low power and useless
for mitotic counting. Quality control must be defined against the task, not against a generic
sharpness threshold.

### 1.2 The regulatory environment in 2026

Two regulatory facts shape deployment, and both are frequently misreported.

First, whole-slide imaging for primary diagnosis is validated under accreditation rules. In the
United States, the College of American Pathologists (CAP) publishes a validation guideline whose
three strong recommendations are at least 60 cases for one application, intraobserver concordance
of at least 95 percent between digital and glass, and a washout of at least two weeks between
readings. These are binding for accredited diagnostic use, not suggestions.

Second, the status of laboratory developed tests (LDTs) changed twice. The Food and Drug
Administration (FDA) issued a final rule in May 2024 (89 FR 37286) that would have phased out
general enforcement discretion for LDTs. A federal court vacated the rule in March 2025, and the
FDA published a conforming rule in September 2025 (90 FR 45134) restoring the earlier position. As
of October 2026 the pre-2024 position applies. A hospital strategy document that still plans for
the 2024 phase-out schedule is planning for a rule that no longer exists.

### 1.3 The operational failure that motivates the topic

A laboratory installs a breast biomarker model. The model is validated at 0.92 AUROC and
integrated into the LIS. Six weeks after go-live, the operations team notices that cases from one
satellite clinic are reported later than before. The model is not slow. The satellite's slides are
scanned on a scanner whose colour calibration has drifted, so the model's quality gate flags
18 percent of them for manual review, which routes them to a single pathologist who works three
days a week.

Nothing in the model's validation predicted this. The validation used slides from the main
campus, scanned on the calibrated scanner, during a period when the satellite scanner was new. The
failure is operational: a quality threshold interacting with a staffing pattern and a maintenance
schedule. This topic teaches how to see it before go-live.

## 2. Learning outcomes

By the end of this topic you can:

1. **Map** the specimen journey from accession to report, naming the custodian and the clock at
   each handoff.
2. **Compute** scanner capacity, utilisation, queue length, and TAT percentiles from arrival and
   service rates.
3. **Design** a slide-tracking scheme that survives barcode failure and cassette-to-slide
   mislabelling.
4. **Specify** colour calibration and drift monitoring with an action limit, not only a target.
5. **Choose** the points in the workflow where AI inference runs, and justify the choice against
   TAT and reviewer workload.
6. **Distinguish** CLIA accreditation, CAP validation, FDA device clearance, and LDT enforcement
   as four different obligations.
7. **Diagnose** an operational regression, such as a rising rescan rate or a lengthening queue,
   from routine metrics.
8. **Write** a downtime plan that keeps reports flowing when the scanner, the network, or the model
   is unavailable.

## 3. The core mental model

The laboratory is a queueing system wrapped in a chain of custody. The queue explains time; the
chain explains identity. A digital pathology deployment has to satisfy both.

```text
  accession        gross           processing        section         stain          scan
  (LIS record) --> (cassettes) --> (12-16 h) -----> (slides) -----> (H&E/IHC) ---> (WSI)
      |               |               |                |               |              |
    clock 1         clock 2         clock 3          clock 4         clock 5        clock 6
      |               |               |                |               |              |
      +---------------+-------+-------+----------------+---------------+--------------+
                              |
                              v
                    custody: who holds the block, the slide, the image
                              |
                              v
     AI inference --> worklist --> pathologist review --> report --> archive
                          |                                  |
                     clock 7                            clock 8
```

### 3.1 Two clocks per case

Every case has a **clinical clock** (how long until the result matters) and a **laboratory clock**
(how long the work takes). A frozen section has a clinical clock of 15 to 20 minutes and a
laboratory clock that must fit inside it. A routine resection has a clinical clock of days and a
laboratory clock of about a day. The same scanner serves both, and the priority rule that
resolves the conflict is an operational decision with clinical consequences.

### 3.2 The queue as the explanation of TAT

Little's law relates the three quantities of any stable queue:

```text
L = lambda x W
```

where `L` is the average number of items in the system, `lambda` is the arrival rate in items per
hour, and `W` is the average time in the system in hours. For a scanner, the service rate `mu` sets
the capacity, and utilisation is `rho = lambda / mu`. As `rho` approaches 1, waiting time grows
faster than the utilisation number suggests, and small arrival fluctuations produce long queues.

```text
rho = 0.70  ->  moderate queue, recovers from bursts
rho = 0.85  ->  visible queue at peak, sensitive to a scanner going down
rho = 0.95  ->  queue grows on any bad day, overtime becomes routine
```

A laboratory running at 0.95 utilisation is not efficient; it is fragile. The first question in
any AI deployment should be the laboratory's utilisation, because an AI station that adds
variance to a 0.95 system creates a backlog, while the same station in a 0.75 system is absorbed.

### 3.3 Identity is a chain, not a field

The accession number identifies the case. The cassette identifies the block. The slide label
identifies the section. The image identifier identifies the scan. A report is correct only if all
four links hold. Digital systems tend to assume the chain and validate only the last link, which
is why a barcode failure at scanning can produce an image attached to the wrong case without any
error being raised.

```text
accession S26-00123
   |  cassette A3
   |     |  slide A3-2 (H&E, level 2)
   |     |     |  image hash 9f2c... (scanner 2, 10:42)
   |     |     |     |  AI result r-8821 (model v3.1)
   v     v     v     v
  [joined only if every link is recorded and checked]
```

### 3.4 Quality control is a control chart, not a threshold

A scanner's colour behaviour drifts. The drift is slow, and a single slide that looks slightly
different proves nothing. The operational tool is the control chart: a daily measurement of a
reference target, plotted against a mean and action limits, with a rule for when to stop and
recalibrate. The same logic applies to the model: a monthly measurement of performance on a fixed
audit set, plotted against its validated value.

### 3.5 Where AI sits

There are three placements, and they are not interchangeable:

- **Pre-review triage.** The model runs before the pathologist opens the case and orders the
  worklist. It changes what is read first. It needs ranking quality and low latency, and its
  errors are mostly delays.
- **Concurrent assist.** The model runs while the pathologist reads and displays an overlay. It
  changes what is seen. It needs calibrated localisation and a viewer that does not slow the
  read.
- **Post-sign-out audit.** The model runs after the report and flags disagreements for review. It
  changes what is caught. It needs high sensitivity and a process for amendments.

A model validated for one placement is not validated for another, even if the AUROC is the same.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Accession | The laboratory's unique identifier for a submitted case | Confused with the patient identifier | Identifier |
| Grossing | The macroscopic examination and dissection of a specimen into cassettes | Called "cutting" | Procedure |
| Cassette | The plastic container holding a tissue block during processing | Confused with the block | Physical item |
| Block | The paraffin-embedded tissue from one cassette | Called a cassette | Physical item |
| Section | A thin slice cut from a block and mounted on a slide | Called a slide | 3 to 5 um |
| H&E | Haematoxylin and eosin stain, the routine morphological stain | Treated as a single standard colour | Stain |
| IHC | Immunohistochemistry; antibody-based staining of a specific antigen | Treated as one stain | Stain |
| Special stain | A non-H&E chemical stain, such as PAS or trichrome | Grouped with IHC | Stain |
| Frozen section | Intraoperative sectioning of fresh tissue for a rapid diagnosis | Called a rapid biopsy | Procedure |
| TAT | Turnaround time, from a defined start event to a defined end event | Quoted without its endpoints | Hours |
| STAT | Priority designation for a case that must bypass the routine queue | Used loosely for "urgent" | Priority |
| Batch | A group of slides processed or scanned together | Confused with a rack | Group |
| Rack | A physical holder for slides, typically 20 or 30 | Confused with a batch | Physical item |
| Worklist | The ordered list of cases a pathologist is expected to read | Confused with the LIS queue | Software |
| LIS | Laboratory information system; the system of record for orders and reports | Called "the EMR" | System |
| HL7 v2 | A pipe-delimited messaging standard for orders and results | Assumed to be self-describing | Standard |
| FHIR | Fast Healthcare Interoperability Resources; a modern resource-based standard | Treated as a drop-in replacement for HL7 v2 | Standard |
| Barcode | A machine-readable label on the slide | Assumed to be always readable | Identifier |
| Chain of custody | The documented sequence of who held a specimen and when | Reduced to a signature | Record |
| CLIA | Clinical Laboratory Improvement Amendments; the US laboratory accreditation statute | Confused with CAP | Regulation |
| CAP | College of American Pathologists; an accrediting body with its own checklist | Confused with CLIA | Accreditation |
| LDT | Laboratory developed test; a test designed and used within one laboratory | Assumed to be exempt from all oversight | Regulatory category |
| IVD | In vitro diagnostic; a test regulated as a medical device | Confused with LDT | Regulatory category |
| SaMD | Software as a medical device | Used for any clinical software | Regulatory category |
| IQ / OQ / PQ | Installation, operational, and performance qualification | Reduced to a single "validation" | Validation stage |
| Calibration | Adjusting a device to a reference | Confused with validation | Procedure |
| Drift | A slow change in device behaviour away from calibration | Confused with a single outlier | Phenomenon |
| Control chart | A plot of a measured quantity against limits, with rules for action | Replaced by a pass/fail threshold | Method |
| Levey-Jennings chart | A control chart with mean and standard-deviation limits | Used without a rule for runs | Method |
| Rescan rate | The share of scans repeated for quality reasons | Not measured at all in many laboratories | Proportion |
| WSI validation | The evidence that a digital system is adequate for diagnosis | Confused with scanner purchase | Process |
| Washout | The interval between digital and glass reads in a validation study | Omitted under scheduling pressure | Weeks |
| Concordance | Agreement between two readings or modalities | Reported without its denominator | Proportion |
| Telepathology | Remote interpretation of digital slides | Assumed to require live video | Practice |
| VNA | Vendor neutral archive; a multi-specialty image archive | Confused with the LIS | System |
| IMS | Image management system for pathology | Confused with the viewer | System |

## 5. The mechanics, step by step

This procedure runs a digital pathology operation from accession to post-deployment monitoring.
It is written for the person who owns the workflow, not only the model: each step names the
custodian, the record created, and the failure that must be caught before the next step.

### Step 1. Accession and identity capture

**Input:** The requisition, the specimen container, and the order in the LIS.

**Output:** An accession number, a specimen record, and the ordered test list.

**Rationale:** The accession number is the anchor for every later join. If it is wrong, every
downstream record is wrong in the same direction and no consistency check will catch it. The
patient identifier and the accession must be captured separately, because patients have multiple
accessions and accessions are sometimes reused across years.

**Branch points:**
- Single specimen: one accession, one container.
- Multiple containers: separate accession suffixes per container, never one accession with
  free-text location.
- Add-on tests: recorded as amendments to the order, with a timestamp.

**Edge case:** A specimen received without a requisition is accessioned under a temporary
identifier and must be reconciled before scanning, not after.

### Step 2. Grossing and cassette mapping

**Input:** The specimen and the accession.

**Output:** A cassette map: which part of the specimen is in which cassette, recorded.

**Rationale:** The cassette map is what makes a diagnosis localisable. "Margins positive" is only
meaningful if the margin cassette is known. For digital work it is also what lets a model's output
be tied back to an anatomical site, which is often more useful than a slide-level score.

**Branch points:**
- Standard dissection: numbered cassettes with a written key.
- Complex resection: photographic mapping plus a diagram.
- Multiple blocks per cassette: record the block count, because sectioning depth changes what the
  slide represents.

**Edge case:** A cassette map stored only in the pathologist's notes cannot be queried by a
pipeline. Structure it at grossing or accept that the linkage is lost.

### Step 3. Fixation and processing

**Input:** Cassettes of tissue.

**Output:** Processed, paraffin-infiltrated tissue.

**Rationale:** Fixation time affects morphology and antigenicity, and therefore both the
pathologist's read and the model's input distribution. Under-fixation produces pale, distorted
nuclei; over-fixation can mask IHC epitopes. The processing schedule is a batch process with a
clock, and a delay in one step shifts the whole day's scanning.

**Branch points:**
- Overnight processing: routine, with a 12- to 16-hour cycle.
- Rapid processing: same-day biopsies, with a shorter cycle and different morphology.
- Frozen section: no processing at all, a different image distribution.

**Edge case:** A processor failure overnight is discovered in the morning, when the day's scanning
is already scheduled. The recovery plan belongs in the downtime procedure, not in improvisation.

### Step 4. Embedding and sectioning

**Input:** Processed tissue.

**Output:** Blocks and glass slides with a section of 3 to 5 um.

**Rationale:** Section thickness affects staining intensity and nuclear overlap. A 3 um section
and a 5 um section of the same block produce different images, and a model trained on one may
misread the other. The microtome operator is also the first person who can catch a block that will
not cut cleanly.

**Branch points:**
- Routine sections: 4 um typical.
- IHC sections: often 3 to 4 um, with a charged slide.
- Serial sections for molecular work: order matters, and the mapping must be recorded.

**Edge case:** A block that crumbles produces a slide with holes. That is a quality event, not a
scanning event, and rescanning will not fix it.

### Step 5. Staining and coverslipping

**Input:** Mounted sections.

**Output:** Stained, coverslipped slides ready to scan.

**Rationale:** Stain concentration, time, temperature, and coverslip medium all change the colour
distribution. This is the largest single source of site-to-site variation in the images, and it is
a laboratory process variable, not a scanner variable.

**Branch points:**
- Automated stainer with a fixed protocol: stable, with drift.
- Manual staining: higher variance, and a protocol that is hard to audit.
- IHC with antigen retrieval: a separate process with its own controls.

**Edge case:** A coverslip bubble over the diagnostic area is invisible on the glass at low power
and obvious on the scan. It is a rescan or a recoverslip decision, not a model decision.

### Step 6. Verify slide identity at the scanner

**Input:** The stained slide and its barcode.

**Output:** A verified link between the physical slide and the accession.

**Rationale:** The scanner is the last point where the physical slide and the digital record can
be compared. If the barcode is unreadable and the operator types an identifier, a transposition
becomes a wrong-patient image. The verification must be a read-back against the LIS order, not
only a successful decode.

**Branch points:**
- Barcode reads and matches: proceed.
- Barcode reads but does not match the expected order: hold, do not scan.
- Barcode unreadable: manual entry with a second-person check and a recorded reason.
- No order exists for the slide: quarantine.

**Edge case:** A slide from a previous case, re-scanned for comparison, will not match today's
order. The system needs an explicit "historical rescan" path or it will either block valid work or
accept invalid work.

### Step 7. Schedule the scan

**Input:** The verified slide, the clinical priority, and the scanner queue.

**Output:** A scan job with a priority and an estimated start time.

**Rationale:** The scanner is a shared resource between frozen sections, same-day biopsies,
routine resections, and research. The priority rule determines whether a STAT case waits behind a
batch of 200. Without an explicit rule, the queue is decided by whoever walks to the scanner.

| Priority | Typical case | Target start | Pre-emption rule |
|---|---|---|---|
| STAT | Frozen section, intraoperative | Immediate | Pre-empts all batch work at the next tile boundary |
| Urgent | Same-day biopsy, suspected malignancy | Within 30 minutes | Ahead of routine, behind STAT |
| Routine | Resection, next-day sign-out | Within 4 hours | FIFO within priority |
| Batch | Research, teaching, archive | Overnight window | Never runs during clinical hours |
| Recall | Rescan after QC failure | Within 1 hour | Treated as urgent, and counted in the rescan rate |

**Edge case:** A STAT case that arrives while the scanner is calibrating waits for calibration.
The calibration schedule must therefore avoid clinical peaks, or the STAT target is unachievable
by design.

### Step 8. Acquire the scan

**Input:** The slide and the scan parameters.

**Output:** A whole-slide image with focus points, scan time, and scanner metadata.

**Rationale:** Focus is the parameter that most affects diagnostic usability. Most scanners sample
focus at a grid of points and interpolate. A tissue fold or a bubble can pull focus away from the
tissue, producing a region that is blurred in the scan but sharp on the glass. The number and
placement of focus points is a scanner setting with a direct effect on rescan rate.

**Branch points:**
- Standard focus grid: fast, adequate for most sections.
- Dense focus grid: slower, better for large or folded sections.
- Z-stack: multiple focal planes stored, larger files, used when the section is thick.

**Edge case:** A z-stack flattened to a single plane at ingestion discards the information that
made it worth scanning. Record which plane was chosen.

### Step 9. Pass image quality control

**Input:** The scanned image and its metadata.

**Output:** A pass, rescan, reprocess, or reject decision with a reason code.

**Rationale:** Quality control protects both the diagnosis and the model. A blurred or incomplete
scan is a clinical risk and a distribution shift at the same time. The decision must be recorded,
because the rescan rate is one of the few operational metrics that predicts both cost and
downstream model failure.

| Finding | Decision | Reason code | Owner |
|---|---|---|---|
| Sharp, complete, colour within limits | Pass | OK | Scanner operator |
| Focal blur in a non-diagnostic region | Pass with flag | BLUR-MINOR | Pathologist may review |
| Focal blur over diagnostic tissue | Rescan | BLUR-MAJOR | Scanner operator |
| Tissue missing or folded | Reprocess from the block | TISSUE | Histology |
| Barcode or label unreadable | Reject and re-label | ID | Accessioning |
| Colour outside the action limit | Hold for calibration | COLOUR | Laboratory supervisor |

**Edge case:** A slide that fails QC twice should not be rescanned a third time without a
histology review. The block, not the scanner, may be the problem.

### Step 10. Maintain colour calibration and monitor drift

**Input:** A reference target scanned on a schedule.

**Output:** A measured colour difference, a control chart point, and a calibration decision.

**Rationale:** Colour drifts with lamp age, sensor response, and staining. A daily or per-shift
measurement of a stable reference target turns an invisible drift into a plotted line. The action
limit should be set from the validated state of the scanner, not from an industry default, because
the acceptable difference depends on the stain and the task.

**Branch points:**
- Within limits: continue.
- One point beyond a limit: recheck, then recalibrate if it repeats.
- A run of points on one side of the mean: investigate even though no single point failed.
- Calibration fails: remove the scanner from clinical service and rescan affected slides.

**Edge case:** The slides scanned between the drift starting and the calibration being performed
are still in the archive. The laboratory needs a policy for reviewing them, and the policy should
be written before it is needed.

### Step 11. Link the image to the order

**Input:** The verified scan and the LIS order.

**Output:** An image record joined to the accession, the block, and the stain.

**Rationale:** An image that is not joined to its order is unusable, and an image joined to the
wrong order is dangerous. The join should be event-driven: the scanner emits a completion event,
the middleware resolves the accession, and the image management system stores the image with the
resolved identifier. Polling the LIS for matching filenames is a common shortcut that fails on
renames and duplicates.

**Branch points:**
- Event received and matched: publish an image-available event.
- Event received, no match: quarantine with the raw identifier.
- No event received: the scanner or the interface is down; escalate.

**Edge case:** Two orders for the same block, such as an H&E and an IHC recut, produce two images
that must be distinguished by stain, not only by accession.

### Step 12. Route the AI inference

**Input:** The image-available event, the model catalogue, and the order's test list.

**Output:** An inference job, and a rule for what happens to its result.

**Rationale:** The routing rule is where clinical risk is created or contained. A model that runs
on every slide and writes into the report is a different product from one that runs on a subset
and displays an overlay. The rule must be explicit about which stains, which specimen types, and
which clinical contexts the model is validated for.

| Situation | Route | Release rule |
|---|---|---|
| Validated stain and specimen, model available | Run | Result to worklist, not to report |
| Validated stain, model unavailable | Hold | Queue for retry; report proceeds without AI |
| Out-of-scope specimen | Do not run | No result, no message |
| Model confidence below the validated range | Run, flag low confidence | Pathologist review required before any action |
| Emergency or downtime mode | Do not run | Manual workflow, recorded |

**Edge case:** A model validated on resections is not validated on frozen sections. The routing
rule must key on the specimen type, which means the LIS order must carry it in a structured field.

### Step 13. Present the result

**Input:** The inference output and the viewer context.

**Output:** A worklist order, an overlay, or a report addendum, depending on the placement chosen
in Section 3.5.

**Rationale:** Presentation determines whether the model changes a decision or only adds a
number. A score buried in a tab is ignored; a flag that reorders the worklist changes behaviour. A
flag without a reason is dismissed as noise. The presentation is part of the product and needs its
own usability testing.

**Branch points:**
- Triage: the worklist order changes; the case content does not.
- Concurrent assist: the overlay appears in the viewer, with a confidence and a region.
- Post-sign-out audit: a disagreement list is reviewed on a schedule.

**Edge case:** A model that flags 40 percent of cases as abnormal destroys the value of the flag.
The flag rate at the chosen threshold is an operational parameter, not a technical detail.

### Step 14. Review and sign out

**Input:** The case, the images, and any AI result.

**Output:** A signed report.

**Rationale:** The pathologist remains accountable. The operational question is how long the
review takes with and without the model, and whether the model's presence changes the reading
behaviour in a way that helps or harms. A model that increases review time by 20 seconds per case
adds more than three hours per day across 600 cases.

**Branch points:**
- Model agrees and the pathologist agrees: routine sign-out.
- Model disagrees and the pathologist overrides: record the override, because it is the raw
  material for monitoring.
- Model flags something the pathologist did not see: the flag must be actionable, or it trains the
  reader to ignore flags.

**Edge case:** Override rates measured without context are misleading. A high override rate can
mean the model is wrong, or that the model is catching things the pathologist considers
out of scope. Read a sample of overrides before drawing a conclusion.

### Step 15. Handle amendments and addenda

**Input:** A revised interpretation, a corrected result, or a new stain.

**Output:** An amended report with an audit trail.

**Rationale:** Amendments are normal in pathology and are a useful signal. A rise in amendments
after an AI deployment may indicate that the model's flags are prompting late discoveries, which
is good, or that reports are being signed before review is complete, which is not. The audit trail
must link the amendment to the original image and model version.

**Branch points:**
- Addendum without change to the diagnosis: record and notify per policy.
- Change to the diagnosis: root-cause review, including whether AI contributed.
- Change caused by a data error: correct the record and check for other cases with the same error.

**Edge case:** If the model version is not recorded with the report, an amendment review cannot
distinguish a model failure from a human one.

### Step 16. Preserve custody and retention

**Input:** Blocks, slides, images, and reports.

**Output:** An archive with retention that meets accreditation and legal requirements.

**Rationale:** The digital slide is part of the diagnostic record. Its retention period is set by
accreditation and local law, and it may exceed the period for which the vendor supports the
format. A format migration plan is therefore part of retention, not an IT nicety.

**Branch points:**
- Glass retained: the image is a copy, and the glass is the primary record.
- Glass destroyed after digitisation: the image becomes the primary record, with stricter
  requirements on integrity and backup.
- Research use of clinical images: separate governance and retention.

**Edge case:** Deleting the source image but keeping derived embeddings does not preserve the
record and does not satisfy a request to re-review the slide.

### Step 17. Plan for downtime

**Input:** The dependency map: scanner, network, LIS, image archive, model service.

**Output:** A downtime procedure with a trigger, an owner, and a communication path.

**Rationale:** Every digital step removes a manual fallback that existed before it. A laboratory
that has not scanned glass slides for two years has lost the muscle memory to do so under
pressure. The downtime plan must be rehearsed, not only written.

**Branch points:**
- Scanner down: use the backup scanner, or revert to glass for urgent cases.
- Network or archive down: read from glass, queue scans locally.
- LIS down: paper requisitions and a reconciliation procedure.
- Model down: the clinical workflow continues; the AI result is marked unavailable.

**Edge case:** A model outage during a period when pathologists have come to rely on triage
ordering is a workflow outage even though no diagnosis is blocked. The plan must say how the
worklist is reordered.

### Step 18. Monitor and control change

**Input:** Operational metrics, model metrics, and change requests.

**Output:** A monthly review, a change decision, and an updated validation record.

**Rationale:** A deployed model is a moving system. Scanners age, stains change, case mix shifts,
and the model's performance decays. Monitoring must cover both the laboratory (rescan rate, TAT,
downtime, calibration) and the model (flag rate, override rate, audit-set performance). Change
control must treat a model update, a scanner replacement, and a staining protocol change as events
that can invalidate the validation.

**Branch points:**
- Metrics stable: continue, with the next scheduled review.
- Model metric degraded: investigate the pipeline before retraining.
- Scanner replaced: re-run the validation requirements for a different make or model.
- Model updated: re-validate and record the version in every report.

**Edge case:** A slow drift in the flag rate is easy to accept as the new normal. The control chart
exists to make the drift visible before it becomes the baseline.

## 6. Worked example

A hospital laboratory scans 600 slides on a working day with two whole-slide scanners. Average
tissue area is 15 mm by 15 mm at 40x. The laboratory is considering a triage model that runs after
scanning and reorders the pathologist's worklist. Before discussing the model, the operations team
has to answer a simpler question: does the scanning capacity support the day it already has?

### 6.1 Demand and capacity

The manufacturer's specification for this scanner class is 81 slides per hour at 40x for a
15 mm by 15 mm area. A vendor-published workflow study in a routine laboratory reported about 41
slides per hour for the same instrument class. The difference is handling, rack changes, label
checks, and the pauses that a specification excludes. Plan with 41.

```text
capacity per scanner = 41 slides/h
capacity, two scanners = 82 slides/h
capacity, 8-hour day = 82 x 8 = 656 slides
utilisation = 600 / 656 = 0.915
```

At 91.5 percent utilisation the laboratory has almost no slack. That is the single most important
number in this case, and it is not a model number.

### 6.2 The peak

Slides do not arrive evenly. A plausible day:

| Window | Arrivals | Rate | Capacity in window | Backlog at end |
|---|---:|---:|---:|---:|
| 08:00 to 10:00 | 120 | 60 per hour | 164 | 0 |
| 10:00 to 14:00 | 360 | 90 per hour | 328 | 32 |
| 14:00 to 18:00 | 120 | 30 per hour | 328 | 0 |

During the four-hour peak, demand exceeds capacity by 8 slides per hour:

```text
(90 - 82) x 4 h = 32 slides of backlog
```

The backlog clears at 82 slides per hour once the peak passes, which takes about 23 minutes after
14:00 before the queue is empty again. A 32-slide backlog sounds small. It means the last slide
entered the scanner about 23 minutes later than it would have in a quiet period, and if one
scanner goes down during the peak, the backlog grows at 90 minus 41, or 49 slides per hour, and
does not clear that day.

### 6.3 Turnaround time for one routine case

Follow a resection accessioned on Monday morning:

| Time | Event | Elapsed |
|---|---|---:|
| Mon 08:00 | Accessioned | 0:00 |
| Mon 09:30 | Grossed, cassettes created | 1:30 |
| Mon 10:00 | Processor starts | 2:00 |
| Mon 22:00 | Processor ends, 12-hour cycle | 14:00 |
| Tue 07:00 | Embedded and sectioned | 23:00 |
| Tue 08:00 | Stained and coverslipped | 24:00 |
| Tue 08:40 | Entered scan queue | 24:40 |
| Tue 08:43 | Scan complete, 88 seconds of scanning | 24:43 |
| Tue 08:43 | AI inference complete | 24:43 |
| Tue 09:40 | Pathologist signs out | 25:40 |

TAT is 25 hours 40 minutes, and scanning and inference together contributed about three minutes
of it. The processing cycle contributed twelve hours, the overnight wait before embedding
contributed nine, and the pathologist's review contributed 57 minutes. A model that is one minute
faster changes nothing; a model that reorders the worklist so the clinically urgent cases are read
first can change the time to the *decision* without changing the TAT at all.

### 6.4 The frozen section clock

A frozen section runs on a different clock. The surgeon is waiting in the operating room.

```text
transport to laboratory        4 min
gross and freeze              5 min
section and stain             4 min
scan at 40x, small area       2 min
pathologist review            3 min
call to operating room        1 min
total                        19 min
```

The clinical tolerance is often quoted as 15 to 20 minutes, so this workflow has no slack. A
scanner that is busy with a batch, or a calibration that starts at the wrong moment, consumes the
entire margin. This is why the priority table in Step 7 exists, and why calibration schedules
belong to the laboratory supervisor rather than to the scanner vendor's convenience.

### 6.5 Diagnostic 1: waiting time versus service time

The scanner logs two durations per job: time from queue entry to scan start, and scan duration.
Suppose the monthly medians are 18 minutes of waiting and 14 minutes of scanning at the daily
level, but at the 10:00 to 14:00 peak the waiting time rises to 65 minutes while the scan duration
stays at 14 minutes.

The diagnosis is immediate: service time is stable and waiting time is not, so the constraint is
capacity or arrival, not scanner speed. Three explanations are consistent with the data and are
distinguished by a second cut:

- If waiting rises on both scanners, demand exceeded capacity. Fix by shifting work or adding a
  scanner.
- If waiting rises on one scanner only, its effective rate is lower. Check focus failures, rack
  handling, and calibration interruptions.
- If waiting rises while scan volume is flat, jobs are arriving in larger batches. Check the
  staining and rack-release schedule.

This diagnostic is valuable because it is cheap and because it points at operations rather than at
the model. Teams that skip it tend to buy a faster scanner when the problem is a 10:00 batch
release.

### 6.6 Diagnostic 2: rescan rate by rack and scanner

The laboratory's overall rescan rate is 7 percent. Broken down by rack position, one rack on one
scanner shows 19 percent, against 5 to 7 percent elsewhere.

Two mechanisms produce this pattern. If the affected rack shares a scanner, suspect focus or
calibration on that instrument. If the affected rack spans scanners, suspect the rack itself:
slides seated at an angle, a damaged spring, or a barcode label placed outside the reader's field.
The distinction matters because one fix is maintenance and the other is consumables.

The rescan rate is also a leading indicator for the model. Every rescan is a second image of the
same slide, and a pipeline that treats images as independent will embed the same tissue twice. A
rising rescan rate therefore raises both cost and duplicate risk, and it does so before any model
metric moves.

### 6.7 A varied assumption: larger sections

Suppose the average tissue area rises from 15 mm by 15 mm to 20 mm by 20 mm, because the
laboratory takes on a new surgical service with larger resections. Scan time scales with area:

```text
area ratio = (20 x 20) / (15 x 15) = 400 / 225 = 1.78
effective rate per scanner = 41 / 1.78 = 23.0 slides/h
capacity, two scanners = 46.0 slides/h
capacity, 8-hour day = 368 slides
daily demand = 600 slides
deficit = 232 slides
```

Two scanners no longer meet the day. At 46 slides per hour, clearing 600 slides takes:

```text
600 / 46 = 13.0 hours
```

The conclusion changes from "two scanners with 8.5 percent slack" to "two scanners and five hours
of overtime, or a third scanner". A third scanner gives 69 slides per hour and clears the day in
8.7 hours, which restores a utilisation of about 0.87.

This is the calculation that a model deployment plan must include, because the model's benefit is
usually expressed in pathologist time while the constraint is often scanner time. A triage model
that saves 30 minutes of pathologist time per day is irrelevant if the slides reach the worklist
five hours late.

### 6.8 What the case establishes

The case establishes that the laboratory is capacity-constrained at the peak, that TAT is
dominated by processing and waiting rather than by scanning or inference, that waiting time and
service time separate capacity problems from equipment problems, and that a change in specimen
mix can invalidate a capacity plan without changing any model. It does not establish the value of
the triage model. That requires a workflow study measuring time to decision and reader behaviour,
which is the next piece of evidence and is outside the scope of a throughput calculation.

## 7. Code, query, or calculation

The script does two things an operations team needs every month. It simulates a day of scanner
arrivals with two machines, a priority queue for STAT cases, and a service-time distribution, and
reports utilisation, waiting-time percentiles, and the finish time. It then builds a control chart
for a daily colour measurement and detects a slow drift.

The simulation is a model. Service times are drawn from a normal distribution around the measured
rate, arrivals follow the three-window profile from Section 6, and travel time between stations is
ignored. What it produces is a queue that behaves like the real one: sensitive to the peak, and
sensitive to a scanner going down.

```python
"""lab_ops.py -- scanner queue and colour-drift arithmetic for a pathology laboratory.

Standard library only. Deterministic.
Run: python lab_ops.py
"""

import heapq
import random
import statistics

DAY_S = 8 * 3600
ARRIVALS = [(0, 2 * 3600, 120), (2 * 3600, 6 * 3600, 360), (6 * 3600, 10 * 3600, 120)]
SCANNERS = 2
MEAN_SERVICE_S = 3600 / 41          # 41 slides per hour, routine conditions
SERVICE_SD_S = 20.0
STAT_CASES = 12
SEED = 20261027


def arrivals(rng):
    """Return (time_s, priority) for each slide; priority 0 is STAT, 1 routine."""
    out = []
    for start, end, count in ARRIVALS:
        for i in range(count):
            t = start + (i + rng.random()) * (end - start) / count
            out.append((t, 1))
    for i in range(STAT_CASES):
        out.append((rng.uniform(0, 7 * 3600), 0))
    out.sort()
    return out


def service_time(rng):
    return max(30.0, rng.gauss(MEAN_SERVICE_S, SERVICE_SD_S))


def simulate(rng):
    jobs = arrivals(rng)
    free_at = [0.0] * SCANNERS
    # priority queue: (priority, arrival, index)
    queue = []
    for idx, (t, prio) in enumerate(jobs):
        queue.append((t, prio, idx))
    queue.sort()
    pending = []
    waits, done = [], []
    i = 0
    while i < len(queue) or pending:
        if not pending:
            now = queue[i][0]
        else:
            now = min(free_at)
        while i < len(queue) and queue[i][0] <= now:
            t, prio, idx = queue[i]
            heapq.heappush(pending, (prio, t, idx))
            i += 1
        if not pending:
            continue
        prio, t, idx = heapq.heappop(pending)
        s = min(range(SCANNERS), key=lambda k: free_at[k])
        start = max(free_at[s], now, t)
        dur = service_time(rng)
        free_at[s] = start + dur
        waits.append((start - t) / 60.0)
        done.append(start + dur)
    return waits, done, free_at


def control_chart(rng, days=90, drift_start=60, drift_per_day=0.02):
    base = [rng.gauss(1.0, 0.25) for _ in range(days)]
    series = [b + (drift_per_day * max(0, d - drift_start)) for d, b in enumerate(base)]
    mu = statistics.mean(series[:30])
    sd = statistics.pstdev(series[:30])
    alarms = []
    for d in range(3, days):
        window = series[d - 2:d + 1]
        if sum(1 for v in window if v > mu + 2 * sd) >= 2 or series[d] > mu + 3 * sd:
            alarms.append(d)
    return series, mu, sd, alarms


def main():
    rng = random.Random(SEED)
    waits, done, free_at = simulate(rng)
    waits.sort()
    print("== Scanner queue, 600 slides, 2 scanners ==")
    print(f"capacity          : {SCANNERS * 41} slides/h, {SCANNERS * 41 * 8} slides/day")
    print(f"utilisation       : {600 / (SCANNERS * 41 * 8):.3f}")
    print(f"mean service      : {MEAN_SERVICE_S:.0f} s per slide")
    print(f"wait p50 / p95    : {waits[len(waits)//2]:.1f} / {waits[int(0.95*len(waits))]:.1f} min")
    print(f"last scan ends    : {max(done)/3600:.2f} h after the first arrival")
    print(f"mean wait         : {statistics.mean(waits):.1f} min")

    print("\n== Utilisation sensitivity ==")
    for n, rate in ((2, 41), (2, 23), (3, 23), (1, 41)):
        cap = n * rate
        print(f"{n} scanner(s) at {rate}/h: {cap}/h, 600 slides need {600/cap:.1f} h, "
              f"8 h capacity {cap*8} slides")

    series, mu, sd, alarms = control_chart(rng)
    print("\n== Colour drift control chart ==")
    print(f"baseline mean {mu:.3f}, sd {sd:.3f}, limits {mu+2*sd:.3f} / {mu+3*sd:.3f}")
    print(f"alarms before day 60: {sum(1 for a in alarms if a < 60)}")
    print(f"first alarm after day 60: {next((a for a in alarms if a >= 60), None)}")
    print(f"value at day 60 {series[60]:.3f}, day 89 {series[89]:.3f}")


if __name__ == "__main__":
    main()
```

Output on the author's machine, Python 3.11:

```text
== Scanner queue, 600 slides, 2 scanners ==
capacity          : 82 slides/h, 656 slides/day
utilisation       : 0.915
mean service      : 88 s per slide
wait p50 / p95    : 4.3 / 22.6 min
last scan ends    : 10.00 h after the first arrival
mean wait         : 7.8 min

== Utilisation sensitivity ==
2 scanner(s) at 41/h: 82/h, 600 slides need 7.3 h, 8 h capacity 656 slides
2 scanner(s) at 23/h: 46/h, 600 slides need 13.0 h, 8 h capacity 368 slides
3 scanner(s) at 23/h: 69/h, 600 slides need 8.7 h, 8 h capacity 552 slides
1 scanner(s) at 41/h: 41/h, 600 slides need 14.6 h, 8 h capacity 328 slides

== Colour drift control chart ==
baseline mean 0.942, sd 0.190, limits 1.322 / 1.512
alarms before day 60: 0
first alarm after day 60: 64
value at day 60 1.107, day 89 1.414
```

### 7.1 Reading the output

The utilisation of 0.915 is computed against an eight-hour day, and the simulated waiting time
tells the more useful story: a median of 4.3 minutes but a 95th percentile of 22.6 minutes. The
mean, at 7.8 minutes, sits between them and understates the experience of cases that arrive in the
peak. Operations should plan against the 95th percentile, because that is the case a clinician
notices.

The finish time of 10.00 hours after the first arrival is not a capacity result. It is the last
arrival plus its service time: the scanners catch up between peaks, so the day ends when the last
rack is scanned, not when the queue drains. This is why average utilisation can look comfortable
while individual cases wait. The peak, not the day, is the constraint.

The sensitivity table makes the Section 6.7 change concrete. At 23 slides per hour, two scanners
need 13.0 hours for 600 slides; a third scanner brings that to 8.7 hours, still above an eight-hour
day but close enough to plan with overtime. This is the kind of artefact a capital request should
contain, and it takes seconds to produce.

### 7.2 Reading the control chart

The chart uses the first 30 days to set a mean and a standard deviation, then applies a simple
rule: two of the last three points above two standard deviations, or one point above three. The
simulated drift begins on day 60 and the first alarm arrives on day 64, which is the point of the
exercise: a drift of 0.02 per day is invisible day to day and unambiguous within a week when
plotted.

In a real laboratory the measurement is a colour difference between a reference target and its
baseline, in a defined colour space, under a defined illuminant. The statistical rule is the same.
What changes is the action limit, which should come from the validated state of the scanner and
from the tolerance of the task, not from a generic threshold.

### 7.3 What the script does not model

It does not model scanner failures, rack changes, or operator breaks, all of which reduce effective
capacity below the service-time distribution. It does not model the interaction between STAT
pre-emption and routine waiting beyond the priority order. It does not model the pathologist's
queue, which is the next constraint after scanning and behaves differently because a pathologist
can batch similar cases. And it does not model the model: inference time is seconds and is not the
bottleneck at this scale, which is itself the finding.

## 8. Data, standards, and regulatory context

Four obligations apply to a digital pathology operation, and they are enforced by different bodies
for different reasons. Treating them as one "compliance" task is the most common governance error.

| Obligation | Enforced by | What it requires | Binding? |
|---|---|---|---|
| Laboratory accreditation | Centers for Medicare & Medicaid Services (CMS) under CLIA, and CAP as a deemed authority | Documented procedures, competency, proficiency testing, quality control | Yes, to operate |
| Whole-slide imaging (WSI) diagnostic validation | CAP checklist and the 2021 guideline | 60 cases, 95 percent concordance, 2-week washout, 20 cases per added application | Yes, for primary diagnosis |
| Device regulation | FDA, or a notified body in the EU | Clearance or approval for the intended use, quality system, change control | Yes, to market a device |
| Privacy | Department of Health and Human Services (HHS) under the Health Insurance Portability and Accountability Act (HIPAA), supervisory authorities under the General Data Protection Regulation (GDPR) | Identifier removal, access control, retention limits | Yes, by jurisdiction |

### 8.1 CLIA and CAP

CLIA is a statute, implemented at 42 CFR Part 493, that sets minimum standards for laboratories
testing human specimens. The College of American Pathologists runs an accreditation programme that
CMS recognises as meeting or exceeding CLIA, which is why a US laboratory usually says it is "CAP
accredited" while operating under CLIA authority.

The CAP guideline for validating whole slide imaging for diagnostic purposes, updated in 2021,
makes three strong recommendations: at least 60 cases for one application, at least 95 percent
intraobserver concordance between digital and glass, and a washout of at least two weeks. It adds
20 cases for each additional application, such as immunohistochemistry or frozen sections, and nine
good-practice statements covering documentation and change management. The guideline also defines
what to do when equipment changes: a same-model replacement scanner needs an image calibration
check, while a different make or model requires full revalidation.

### 8.2 FDA device regulation and the LDT question

Software that analyses pathology images for a clinical decision may be a medical device. The
pathway depends on risk and on whether a predicate exists: 510(k) for substantial equivalence, De
Novo for a novel low-to-moderate risk device, and premarket approval for high risk. The quality
system requirement is 21 CFR Part 820, harmonised with ISO 13485.

Laboratory developed tests are the exception that has moved. The 2024 final rule (89 FR 37286)
would have ended general enforcement discretion in stages through 2028. In March 2025 the Eastern
District of Texas vacated the rule, and in September 2025 the FDA published a conforming rule
(90 FR 45134) removing the added text from 21 CFR 809.3(a). As of October 2026 the earlier position
is restored: an LDT designed, manufactured, and used in a single CLIA-certified laboratory is
generally not required to go through premarket review, though it remains subject to CLIA and to
FDA enforcement against unsafe tests. Legislation could change this again, which is why the date
on any such statement matters.

### 8.3 Interoperability standards

The order and the report travel on Health Level Seven (HL7) version 2 messages in most laboratories, typically
`ORM^O01` for an order and `ORU^R01` for a result, over a message broker or a virtual private
network connection. The message identifies the patient, the accession, the specimen, and the
requested test, but it does not carry pixels.

The image travels on DICOM, with the whole-slide object defined in Supplement 145, and the result
may be attached as a DICOM structured report or referenced from a report. FHIR R4 resources,
principally `ImagingStudy`, `Specimen`, `DiagnosticReport`, and `ServiceRequest`, are increasingly
used for newer interfaces but rarely replace the HL7 v2 order flow in an established laboratory.

The fields a digital pathology interface actually uses, in HL7 v2 notation:

| Segment | Field | Meaning | Use in the pipeline |
|---|---|---|---|
| MSH | MSH-10 | Message control identifier | Deduplication and audit |
| PID | PID-3 | Patient identifier | Patient-level splitting, never the join key alone |
| OBR | OBR-2 | Placer order number | Usually carries the accession |
| OBR | OBR-4 | Universal service identifier | Test code, such as a surgical pathology code |
| OBX | OBX-5 | Observation value | Specimen site or result text, depending on the message |
| SPM | SPM-4 | Specimen type | Routing by specimen type, such as biopsy or resection |

The integration rule that matters operationally: the LIS is the system of record for the order and
the report, the image archive is the system of record for pixels, and the join must be by
accession plus block plus stain. Any integration that joins by filename will eventually attach an
image to the wrong case.

### 8.4 Records and audit

Diagnostic records need an audit trail. In the United States, 21 CFR Part 11 defines what an
electronic record needs to be trustworthy: secure access, audit trails, and record retention.
Accreditation adds a retention period for slides and reports, set by the laboratory's own policy
and by state law. If glass is destroyed after digitisation, the image becomes the primary record
and inherits stricter integrity requirements, including verified backup and format migration.

### 8.5 What the standards do not cover

None of these instruments specifies scanner throughput, queue discipline, rescan thresholds,
colour action limits, AI threshold selection, or model monitoring frequency. Those are operational
decisions, and they are where Section 10 lives. A laboratory can be fully accredited, fully
compliant, and still run a fragile operation.

## 9. Industry practice and case evidence

### 9.1 What scanners actually deliver

Manufacturer figures describe an instrument under ideal handling. Routine laboratories report
lower numbers, and the gap is the operational reality.

| Instrument | Published figure | Conditions | Routine observation |
|---|---:|---|---|
| Leica Aperio GT 450 | 81 slides per hour | 15 mm by 15 mm at 40x | About 41 slides per hour in a published workflow study |
| Hamamatsu NanoZoomer S360 | 82 slides per hour | 15 mm by 15 mm, 20x or 40x mode | Batch capacity of 360 slides |
| Philips IntelliSite Ultra Fast | 60 seconds per slide | 15 mm by 15 mm at 40x equivalent | 300-slide capacity |

The routine figure is roughly half the specification. Capacity planning that uses the
specification will under-provision by a factor of two, which is exactly the error that turns a
comfortable 0.45 utilisation into a fragile 0.91.

### 9.2 How laboratories integrate

Most laboratories run a three-part stack: the LIS for orders and reports, an image management
system for storage and viewing, and a scanner fleet with vendor middleware. Integration is
usually HL7 v2 for orders and results and DICOM for images. The middleware is where identifiers are
resolved, and it is the component most likely to be undocumented and least likely to be tested
during a scanner replacement.

Two patterns are common and both have costs. A **single-vendor stack**, where scanner, image
management, and viewer come from one supplier, reduces integration risk and increases switching
cost. A **best-of-breed stack** with a vendor neutral archive (VNA) and an independent viewer gives flexibility and makes
the integration the laboratory's problem. Neither is wrong; the choice should be made with the
downtime plan in hand.

### 9.3 Telepathology in practice

Asynchronous store-and-forward telepathology is routine for second opinions and subspecialty
sign-out. Synchronous telepathology is used for frozen sections and for intraoperative
consultation, where the clinical clock is minutes. The operational difference is not the
technology but the staffing model: a synchronous service needs a pathologist available at the
remote site during the operating window, which is a scheduling commitment, not a software feature.

### 9.4 The AI deployment pattern that works

Published deployments and cleared products share a pattern. The model addresses a narrow,
well-defined task; it is validated on a reader study or a held-out site; it is deployed as an
adjunct with the pathologist accountable; and it is monitored with a pre-specified metric. Paige
Prostate, authorised by the FDA through the De Novo pathway in 2021, is the canonical early
example: a narrow intended use, a clinical evidence package, and a workflow in which the
pathologist reviews the flagged regions.

The pattern that fails is broader: a general model, validated on a public benchmark, deployed
across specimens and stains without a local reader study, with no monitoring beyond an occasional
complaint. It fails not because the model is bad but because the operation around it was never
specified.

### 9.5 Documented fact versus inference

| Statement | Status |
|---|---|
| A GT 450 is specified at 81 slides per hour at 40x for a 15 mm square | Documented by the manufacturer |
| A routine laboratory achieved about 41 slides per hour on the same class | Documented in a vendor workflow study |
| Two scanners at 41 slides per hour cannot absorb a 600-slide day with a 90-per-hour peak without a queue | Arithmetic, given the assumptions |
| Most laboratories run below 0.8 scanner utilisation | Inference; measure your own |
| AI triage reduces pathologist time | Inference; depends on the workflow and the flag rate |
| LDT oversight is settled for 2026 | Documented as of October 2026, and reversible by legislation |

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Capacity shortfall at peak | Waiting p95 rises while service time is flat | Arrivals exceed service rate in a window | Late reports; overtime; STAT delays | Queue model; shift or add a scanner |
| Scanner down with no backup | Single point of failure in the route | One instrument serves all STAT work | Frozen sections delayed | Backup scanner or glass fallback |
| Barcode misread | Rescan rate rises on one rack | Label damaged or outside the reader | Wrong-case image risk | Two-person check; label standard |
| Colour drift | Control chart run above the mean | Lamp or sensor ageing; stain change | Model flags rise; diagnoses questioned | Calibration schedule; review policy |
| Rescan without versioning | Duplicate images per slide | Same slide scanned twice | Duplicate embeddings; double counting | Image version key; deduplicate by hash |
| Flag rate drift | Share of flagged cases rises monthly | Case mix or model drift | Alert fatigue; ignored flags | Monitor flag rate against a baseline |
| Override rate ignored | High override share with no review | Model out of scope or wrong | Model loses credibility | Sample and categorise overrides |
| Interface mismatch | Images without orders; orders without images | Middleware join by filename | Orphaned or misattached cases | Join on accession, block, stain |
| Downtime untested | Staff improvise during an outage | Manual workflow not rehearsed | Delays and errors under pressure | Rehearse twice a year |
| Validation not re-run after change | Scanner replaced quietly | Change treated as maintenance | Diagnostic system out of validation | Change control tied to CAP rules |
| AI result in the report without review | Report contains unverified model text | Routing rule bypassed | Accountability unclear | Results go to the worklist, not the report |
| Retention shorter than the record | Archive tier expires before policy | Cost pressure on storage | Record unavailable for review | Retention policy owned by the laboratory |
| Time zone or clock skew | Events out of order in the log | Different system clocks | Wrong TAT; audit gaps | Synchronise clocks; record offsets |

Two of these deserve emphasis. Colour drift is insidious because it degrades both the human read
and the model at once, and the first symptom is often a complaint rather than a metric. Flag-rate
drift is insidious because it is gradual: a model that flags 8 percent of cases in January and
14 percent in June has changed the workload, and nobody may notice until the worklist is late.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| 600 slides per day | Sizes the capacity | Specialty mix, seasonal variation | Count arrivals for 20 working days |
| 41 slides per hour effective | Converts to capacity | Tissue area, focus settings, operator | Measure scan duration from logs |
| Two scanners available | Defines the queue | Maintenance, breakdown, calibration | Track uptime and mean time to repair (MTTR) |
| Arrivals follow a three-window profile | Models the peak | Batch release, courier schedules | Plot arrivals in 15-minute bins |
| Service times are independent | Simplifies the queue | Large sections, repeated rescans | Compare predicted and observed p95 |
| STAT cases pre-empt routine | Sets priority | Informal queue jumping | Audit scan order against priority |
| Colour measurement is daily | Detects drift | Protocol or lamp changes | Compare detection delay to change point |
| The model is not the bottleneck | Justifies the workflow design | Very large models, limited GPU | Measure inference latency per slide |
| Accreditation rules are stable | Allows a fixed validation plan | Guideline updates | Review annually |

The most consequential limit is that this topic treats the laboratory as a single queue. Real
laboratories have several interacting queues: histology, staining, scanning, and reading. A
constraint that moves from scanning to reading changes the answer, and the only way to see it is
to measure each stage rather than the end-to-end average.

## 12. Exercises

### Exercise 1. Draw the chain of custody

Take one hypothetical case and draw every handoff from accession to report, naming the custodian,
the record created, and the identifier used at each step.

**Deliverable:** A one-page diagram with a table of identifiers.

**Acceptance criteria:** At least eight handoffs; every identifier is defined; the diagram shows
where a barcode failure would break the chain and who would notice.

**Time:** 45 minutes.

### Exercise 2. Compute capacity and queue

A laboratory has 450 slides per day, two scanners at 40 slides per hour, and a peak in which
55 percent of slides arrive in four hours.

**Deliverable:** Capacity, utilisation, peak backlog, and the time to clear it.

**Acceptance criteria:** Capacity 640 slides per day; utilisation 0.70; peak arrival 61.9 slides
per hour against capacity 80, so no growing backlog, and the answer says so rather than inventing
one.

**Time:** 30 minutes.

### Exercise 3. Design the rescan policy

Write the decision rule for a slide that fails image quality control, covering minor blur, major
blur, tissue loss, barcode failure, and colour outside limits.

**Deliverable:** A decision table with owner and reason code.

**Acceptance criteria:** Every outcome names who decides; a second failure of the same type
escalates to histology; the policy states how rescans are counted and reported.

**Time:** 45 minutes.

### Exercise 4. Set a colour action limit

A scanner's daily colour measurement has a mean of 1.0 and a standard deviation of 0.2 in the
validated state. Propose an action limit and a rule for a run of points.

**Deliverable:** A control chart specification.

**Acceptance criteria:** Limits justified against the validated state, not invented; a run rule
and a single-point rule both present; the specification says what happens to slides scanned
between the drift and the detection.

**Time:** 30 minutes.

### Exercise 5. Map the AI routing rule

Design the routing rule for a model validated on H&E resections of one organ, deployed in a
laboratory that also handles frozen sections, biopsies, and IHC.

**Deliverable:** A routing table with the four possible dispositions and the fields the LIS must
supply.

**Acceptance criteria:** Out-of-scope specimens never run; the LIS fields are named and structured;
the rule states what happens when the model is unavailable.

**Time:** 45 minutes.

### Exercise 6. Write the downtime plan

Write the procedure for a four-hour outage of the image archive during clinical hours.

**Deliverable:** A one-page procedure with trigger, roles, and recovery.

**Acceptance criteria:** Patient care continues; scans are queued locally and reconciled; the
plan says how AI results are marked unavailable; a rehearsal date is named.

**Time:** 45 minutes.

### Exercise 7. Critique an AI business case

A vendor proposes a triage model that saves 30 minutes of pathologist time per day, at a cost of
200,000 USD per year, for a laboratory whose peak queue adds 23 minutes of waiting per case.

**Deliverable:** A critique identifying what the business case omits.

**Acceptance criteria:** Notes that the constraint may be scanning, not reading; asks for the
flag rate and the override rate; asks whether the saved time is redeployable; requires a
measurement plan rather than a projection.

**Time:** 45 minutes.

### Exercise 8. Build the monthly operations review

Design the one-page monthly review that the laboratory director and the model owner read
together.

**Deliverable:** A metric list with definitions, owners, and thresholds.

**Acceptance criteria:** Covers TAT, waiting time, rescan rate, calibration, downtime, flag rate,
override rate, and audit-set performance; each metric has a definition and a threshold; the page
fits on one page.

**Time:** 60 minutes.

## 13. Self-check answers

**Exercise 2.** Capacity is 2 x 40 = 80 slides per hour, or 640 in an eight-hour day. Utilisation
is 450 / 640 = 0.70. The peak carries 0.55 x 450 = 247.5 slides in four hours, or 61.9 per hour,
which is below the 80 per hour capacity. No backlog grows during the peak; the queue that exists is
short and clears. The correct answer is that the system has slack, and the exercise is designed to
catch the reflex of declaring a crisis.

**Exercise 4.** With a mean of 1.0 and a standard deviation of 0.2, a two-sigma limit is 1.4 and a
three-sigma limit is 1.6. A reasonable specification uses 1.4 as the warning and 1.6 as the action
limit, with a rule that two of three consecutive points above 1.4 triggers recalibration. The
policy must also state that slides scanned between the drift onset and detection are reviewed
against the clinical risk, because the control chart detects late by construction.

**Exercise 6.** A strong answer triggers the plan when the archive has been unavailable for 30
minutes, names a single incident owner, and separates clinical continuity (read from glass, or
from the local cache) from reconciliation (scan to local storage, import when the archive returns).
It marks AI results as unavailable rather than silently omitting them, and it names a rehearsal
date within the quarter.

**Exercises 1, 3, 5, 7, and 8 are assessed against their acceptance criteria.** A strong Exercise 1
diagram shows the barcode failure point at the scanner, because that is where the physical and
digital identities are last compared. A strong Exercise 3 policy escalates a second failure to
histology rather than rescanning again, because the block is often the cause. A strong Exercise 7
critique separates the model's benefit from the laboratory's constraint and refuses a projection
without a measurement plan.

## 14. Interview questions

**1. Walk me through a slide from accession to report.**
Strong: names the handoffs, the identifiers, and the clocks, and identifies where identity can
break. Weak: lists the steps without the records.

**2. How do you decide how many scanners a laboratory needs?**
Strong: capacity from measured throughput, utilisation, peak arrival, and a queue model; plans
against p95. Weak: divides daily volume by the specification.

**3. A scanner's rescan rate doubles. What do you check?**
Strong: splits by scanner, rack, and operator, then checks focus, calibration, and labels. Weak:
blames the model.

**4. What does CLIA require that CAP also covers?**
Strong: distinguishes the statute from the accreditation programme and notes CAP as a deemed
authority. Weak: treats them as synonyms.

**5. What changed about LDT regulation in 2025?**
Strong: the 2024 rule was vacated and the FDA removed the language in 2025, restoring the earlier
position. Weak: recites the 2024 phase-out schedule.

**6. Where should an AI result appear?**
Strong: depends on placement, triage versus assist versus audit, each with its own validation.
Weak: "in the report".

**7. How do you monitor a deployed model in a laboratory?**
Strong: flag rate, override rate, audit-set performance, and pipeline metrics, reviewed monthly.
Weak: "we retrain when it gets bad".

**8. What happens when the archive goes down for four hours?**
Strong: a rehearsed plan with clinical continuity, local queuing, reconciliation, and AI marked
unavailable. Weak: "IT handles it".

**9. How do you set a colour action limit?**
Strong: from the validated state and the task tolerance, with a control-chart rule. Weak: "use the
manufacturer's default".

**10. Why is a 0.95 utilisation a problem?**
Strong: queueing theory, no slack for a breakdown, p95 waits grow non-linearly. Weak: "it is
efficient".

## 15. Key terms

- **Accession** — the laboratory's identifier for a case.
- **Grossing** — macroscopic dissection into cassettes.
- **Cassette** — the container used during processing.
- **Block** — paraffin-embedded tissue from one cassette.
- **Section** — a thin slice mounted on a slide.
- **H&E** — haematoxylin and eosin stain.
- **IHC** — immunohistochemistry.
- **Frozen section** — intraoperative rapid sectioning.
- **TAT** — turnaround time between defined events.
- **STAT** — priority designation for urgent work.
- **LIS** — laboratory information system.
- **HL7 v2** — pipe-delimited messaging for orders and results.
- **FHIR** — resource-based interoperability standard.
- **Chain of custody** — documented sequence of specimen possession.
- **CLIA** — US laboratory accreditation statute.
- **CAP** — College of American Pathologists, an accrediting body.
- **LDT** — laboratory developed test.
- **IVD** — in vitro diagnostic device.
- **SaMD** — software as a medical device.
- **Calibration** — adjustment to a reference.
- **Drift** — slow change away from calibration.
- **Control chart** — a plot with limits and action rules.
- **Rescan rate** — share of scans repeated for quality.
- **Telepathology** — remote interpretation of slides.
- **VNA** — vendor neutral archive.

## 16. Sources

- Centers for Medicare and Medicaid Services. *Clinical Laboratory Improvement Amendments*,
  42 CFR Part 493. https://www.ecfr.gov/current/title-42/chapter-IV/subchapter-G/part-493
- Evans, A. J., Brown, R. W., Bui, M. M., et al. (2022). Validating Whole Slide Imaging Systems
  for Diagnostic Purposes in Pathology: Guideline Update from the CAP in Collaboration with ASCP
  and API. *Archives of Pathology and Laboratory Medicine*, 146(4), 440-450.
  doi:10.5858/arpa.2020-0723-CP
- College of American Pathologists. *Validating Whole Slide Imaging for Diagnostic Purposes in
  Pathology: If/Then Statements*, 2021. https://documents.cap.org/documents/wsi-if-then-statements.pdf
- US Food and Drug Administration. (2016). *Technical Performance Assessment of Digital Pathology
  Whole Slide Imaging Devices*. Guidance for Industry and FDA Staff.
- US Food and Drug Administration. (2024). *Medical Devices; Laboratory Developed Tests*. Final
  rule, 89 FR 37286.
- US Food and Drug Administration. (2025). *Medical Devices; Laboratory Developed Tests;
  Implementation of Vacatur*. Final rule, 90 FR 45134.
- *American Clinical Laboratory Association v. US Food and Drug Administration*, No. 4:24-cv-479
  (E.D. Tex., 31 March 2025).
- Health Level Seven International. *HL7 Version 2.5.1 Standard*.
  https://www.hl7.org/implement/standards/product_brief.cfm?product_id=184
- HL7 International. *FHIR R4 Specification*. https://hl7.org/fhir/R4/
- DICOM Standards Committee. *Supplement 145: Whole Slide Microscopic Image IOD and SOP Classes*,
  2010. https://dicom.nema.org/MEDICAL/Dicom/Final/sup145_ft.pdf
- International Organization for Standardization. *ISO 15189:2022 Medical Laboratories:
  Requirements for Quality and Competence*.
- Leica Biosystems. *Aperio GT 450 Specifications*, MAN-0393 Rev L.
  https://www.leicabiosystems.com/
- Hamamatsu Photonics. *NanoZoomer S360 Brochure*, SBIS0144E. https://nanozoomer.hamamatsu.com/
- Philips. *IntelliSite Ultra Fast Scanner Brochure*. https://www.philips.com/healthcare/
- US Food and Drug Administration. (2021). *Good Machine Learning Practice for Medical Device
  Development*. Guiding principles.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous: [`12_pathology-foundation-models.md`](12_pathology-foundation-models.md)
- Pipeline infrastructure: [`11_wsi-pipeline-infrastructure.md`](11_wsi-pipeline-infrastructure.md)
- Workflow integration: [`08_workflow-integration.md`](08_workflow-integration.md)
- Regulatory path: [`09_regulatory-path.md`](09_regulatory-path.md)
- Survey file: [`../../course/10_medical_imaging.md`](../../course/10_medical_imaging.md)
- Lab catalogue: [`../../labs/`](../../labs/)
