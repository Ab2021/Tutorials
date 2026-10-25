# Is It a Device?

> Module 14 — Medical Devices, SaMD, and Regulation
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01
> Version 2026.3

## 1. Why this topic exists

A team can have a validated model, a paying pilot, and a launch date, and still not know whether the thing it is selling
is a regulated product. "Is it a device?" is the first gate in this module, and it is the only gate a team can pass by
accident. Classification, pathway, evidence, and post-market obligations all sit downstream of it. If the gate answer is
wrong, every later document is written for the wrong authority.

The two errors are not symmetric. Treating a non-device as a device costs money: a quality management system (QMS) built
early, a clinical study that no reviewer required, a 90-day review that consumes a market window, a regulatory hire before
revenue. Treating a device as a non-device costs the market: a warning letter, a recall, an injunction, a qui tam action,
or a product liability claim in which the plaintiff's expert reads the sales deck aloud. The second error is also the one
regulators look for, because the statutory exclusions are self-executing. No agency approves a company's decision to call
its product non-device clinical decision support (CDS). The company simply does it, and the record of whether it was right
is created later, by someone else.

The folk rule that governed this question for years was: "a clinician makes the final decision, so it is CDS and not a
device." The 21st Century Cures Act of 2016 created the statutory hook, and the industry compressed it into that one
sentence. The US Food and Drug Administration's (FDA) final guidance of 28 September 2022 is more demanding. It reads the
statute as four conditions rather than one, and two of those conditions disqualify whole product categories that the folk
rule admitted: any function that acquires, processes, or analyzes a medical image, a signal from an in vitro diagnostic
device, or a pattern from a signal acquisition system, and any function whose basis a clinician cannot independently
review. Teams that built to the folk rule discover the gap during a submission, a diligence review, or a hospital
procurement assessment, which is the most expensive possible moment.

The question is also not about a product. Section 520(o)(2) of the Federal Food, Drug, and Cosmetic Act (FD&C Act) tells
FDA that it may not regulate an excluded software function as a device even when that function ships inside a product that
contains device functions. The statute therefore forces a function-level analysis: a platform is a portfolio of statuses,
not one status. A documentation summarizer, a risk score, a code suggester, and a scheduling assistant can ship in one
installable product and land in four different regulatory positions.

Two further properties make the question harder than it looks. First, device status decays. The FDA guidance page for CDS
software displayed an issue date of January 2026 and a content-current-as-of date of 29 January 2026 when this file was
prepared; the criteria taught here are those of the 28 September 2022 final guidance, and a reader who needs a current
determination must confirm which version is in force. A determination memo without a date and a source version is an
opinion about the past.

Second, the answer is jurisdictional. The United States has a statutory list of excluded software functions. The European
Union has no equivalent list. The same function, described in the same words, can be non-device CDS in the United States
and medical device software in the European Union. A team that writes one global claim and one global evidence plan will
discover this at the point where the European technical documentation is due.

The cost of getting the gate right early is small: a function inventory, a claim-surface audit, a two-page memo, and one
conversation with someone who does this for a living. The cost of getting it wrong late is a rebuild of the evidence
package around a different legal theory. This topic gives you the procedure for the two-page memo and the vocabulary to
make the conversation useful.

## 2. Learning outcomes

By the end of this topic, you can:

1. Apply the United States device gate in the correct order: the medical purpose test in section 201(h)
of the FD&C Act, then the five software exclusions in section 520(o)(1), then the multi-function rule of
section 520(o)(2).
2. Evaluate a CDS function against the four statutory conditions and against FDA's 2022 reading
of the word "support," including the specific-directive, time-critical, and independent-review tests.
3. Audit a claim surface across label, user interface, sales material, contracts, and pricing,
and state which register supplies the objective intent under 21 CFR 801.4.
4. Qualify software under the European Union gate using the Figure 1 decision steps in the Medical Device
Coordination Group (MDCG) guidance 2019-11 Rev.1 (June 2025), and split medical device software from in vitro
diagnostic software using the Figure 2 steps.
5. Distinguish standalone software, software in a hardware device, an accessory, and a component,
and state the submission consequence of each distinction.
6. Write a device-status memorandum with one disposition per function, a confidence statement,
and explicit re-review triggers.
7. Defend a determination against its two strongest counterarguments and name the evidence that
would settle it.

The outcomes map to the exercises and self-check items as follows.

| Outcome | Exercise | Self-check |
|---|---|---|
| 1 | Exercise 1, Exercise 3 | Q1, Q2 |
| 2 | Exercise 2, Exercise 3 | Q3, Q4 |
| 3 | Exercise 4 | Q5 |
| 4 | Exercise 5, Exercise 6 | Q6, Q7 |
| 5 | Exercise 6 | Q8 |
| 6 | Exercise 7 | Q9 |
| 7 | Exercise 8 | Q10 |

## 3. The core mental model

Device status is produced by two independent gates, each with its own questions, and each applied to a function rather
than to a product. The gates are not a hierarchy and they are not interchangeable. Passing one says nothing about the
other.

```text
                        product concept
                              |
                    [0] function inventory
                              |
          +-------------------+-------------------+
          |                                       |
   US gate                                  EU gate
   21 U.S.C. 321(h)                          MDR Art 2(1) / IVDR Art 2(2)
          |                                       |
   medical purpose?                        software acting on data
          | yes                                   | beyond storage?
   520(o)(1) screen                              | yes
   A   B   C   D   E                       benefit to individual patients?
          |                                       | yes
   exclusion applies?                       medical purpose in Art 2(1)?
     |            |                                | yes
    yes           no                        MDSW -> device or IVD
     |            |                          (MDCG 2019-11 Fig 2)
  non-device    device                            |
     |            |                                |
     +------------+---------------+----------------+
                                  |
                      class, pathway, controls
```

The left gate is a statutory exclusion machine. The right gate is a purpose test under the Medical Device Regulation
(MDR) and the In Vitro Diagnostic Medical Devices Regulation (IVDR), with a carve-out for software that only stores,
archives, communicates, searches, or compresses without loss. They produce different vocabularies and different evidence
obligations, and a determination memo needs one row per jurisdiction even when the underlying facts are identical.

The left gate has an order that matters. Section 201(h) asks whether there is a medical purpose at all. Section 520(o)(1)
then asks whether this particular function is excluded. Only if the function survives both does device status attach, and
only then does classification begin. Teams that start at the exclusion list tend to reverse-engineer a purpose to fit a
category, which is the single most common origin of a memo that does not survive review.

```text
function with a medical purpose under 201(h)
        |
        +-- (A) administrative support of a health care facility? ---> excluded
        |
        +-- (B) healthy lifestyle, unrelated to disease? ------------> excluded
        |
        +-- (C) certified electronic record, no interpretation? -----> excluded
        |
        +-- (D) display/storage of lab or device data,
        |       with no interpretation? -----------------------------> excluded
        |
        +-- (E) CDS, four conditions all met? ----------------------> excluded
        |
        +-- none of the above --------------------------------------> device
```

Read the ladder as a series of narrow doors, not as rooms. Each door has a threshold condition, and the condition is a
fact about intended use, not about capability. A product that *could* interpret a laboratory result but is not *intended*
to do so has a different status from one that is intended to. That is why the label, the interface text, and the sales
narrative are regulatory instruments and not marketing decoration.

The concept that ties all of this together is the claim surface. Objective intent under 21 CFR 801.4 is assembled from
expressions, from the design or composition of the article, and from the circumstances surrounding its distribution. In
software, that means at least five registers:

| Register | What it contains | Why it counts |
|---|---|---|
| Label and instructions for use | Indications, contraindications, warnings | The formal statement of intent |
| User interface | Screen text, default settings, button labels, alert thresholds | Design as intent; a default is a statement |
| Sales and marketing | Website, deck, demo script, conference talk | Expressions of the persons responsible for labeling |
| Contracts and pricing | Order forms, statements of work, billing codes | Circumstances of distribution |
| Integrations | Which system calls the function, in which workflow, at what moment | Setting and timing, which change the clinical action |

A function can pass the statutory test and still be a device because one register contradicts the others. A label that
says "adjunctive" alongside a sales deck that says "replaces the triage nurse" produces the broader claim, because the
broader claim is also an expression of intent.

The fourth idea is the default. There is no presumption of non-device status. If a function has a medical purpose and no
exclusion applies, it is a device. The exclusion must be established, per function, with facts. "We are probably fine" is
not a disposition.

The fifth idea is the four-cell outcome. Because the gates are independent, a product occupies one of four positions, and
the work plan differs in each.

| | EU in scope | EU out of scope |
|---|---|---|
| US device | Dual-track: FDA submission and EU conformity assessment | US-only plan; check whether EU claim can be narrowed |
| US non-device | EU-only plan; the US exclusion does not travel | Neither regime; document why and monitor |

The most expensive cell is the top-right one, because teams discover it after building a US-first evidence package. The
most misleading cell is the bottom-left one, because a US exclusion feels like a general permission and is not.

Two cautions complete the model. First, an exclusion is not the same as enforcement discretion. An excluded function is
outside the device definition; a function subject to enforcement discretion is a device that FDA has said it will not
prioritize. The two have different legal consequences, different labeling obligations, and different answers when a
customer's quality auditor asks a question. Second, failing an exclusion does not automatically make a function a device.
FDA's 2022 guidance says as much: other policies or enforcement-discretion positions may still apply. The determination is
therefore not binary: a function can be excluded, can be a device, or can remain unresolved with a named reason. A memo
that forces a binary answer is hiding its weakest step.

## 4. Definitions and vocabulary

The table below defines the terms this topic uses. The "common misuse" column records errors that have appeared in real
submissions and diligence reviews, not hypothetical ones.

| Term | Definition | Common misuse | Source and status |
|---|---|---|---|
| Device | An instrument, apparatus, implement, machine, contrivance, implant, in vitro reagent, or similar article intended for diagnosis, cure, mitigation, treatment, or prevention of disease, or to affect the structure or function of the body | Treating any health-adjacent software as a device, or any device as requiring premarket review | 21 U.S.C. 321(h); binding statute |
| Software function | A discrete capability with its own intended use and its own output; the unit at which exclusions and classifications are assessed | Assessing the product as one object and letting the most regulated feature set the status | FD&C Act 520(o); binding statute |
| Objective intent | The intent of the persons legally responsible for labeling, shown by expressions, design or composition, and circumstances of distribution | Quoting only the label and ignoring the interface, sales deck, and contract | 21 CFR 801.4; binding regulation |
| Intended use (US) | The general purpose of the device and the conditions of its use, as evidenced by objective intent | Confusing it with a feature list or a product name | 21 CFR 801.4 |
| Intended purpose (EU) | The use for which a device is intended according to the manufacturer's label, instructions, promotional material, and clinical evaluation | Copying the US intended-use sentence and missing the clinical benefit and target population | MDR Article 2(12); binding regulation |
| Non-device CDS | A software function that meets all conditions in 520(o)(1)(E) and is therefore outside the device definition | Calling any alert or score "CDS" without testing the conditions | FD&C Act 520(o)(1)(E); binding statute |
| Medical information | Information about a patient's health status, care, or condition, such as a laboratory result, a diagnosis, a medication list, or a guideline | Treating a continuous waveform as "just information" | FDA CDS guidance, 28 September 2022; guidance |
| Signal | A time-varying measurement from an acquisition system, such as an electrocardiogram lead or a photoplethysmographic trace | Assuming a single result and a continuous signal are legally the same input | FDA CDS guidance, 28 September 2022 |
| Pattern | Multiple, sequential, or repeated measurements considered together, such as a sequence of glucose readings | Assuming "we only use the numbers, not the waveform" removes the exception | FDA CDS guidance, 28 September 2022 |
| Independent review of the basis | The clinician's ability to see the inputs, logic, and limitations well enough not to rely primarily on the output | Claiming reviewability because a "Why?" link exists but opens a generic disclaimer | FD&C Act 520(o)(1)(E)(iii); binding statute |
| Specific directive | An output that names the action to take, rather than supporting the clinician's own choice | Calling an order "a recommendation" because the button says "Suggest" | FDA CDS guidance, 28 September 2022; guidance |
| Time-critical | An output whose value depends on acting within a window too short for independent review | Ignoring timing because the software is not labelled "emergency" | FDA CDS guidance, 28 September 2022; guidance |
| Enforcement discretion | A device that FDA has said it does not intend to enforce certain requirements against | Treating discretion as exclusion from the definition | FDA device software policy, 28 September 2022; guidance |
| Multi-function product | A product containing at least one excluded function and at least one device function | Regulating the whole product, or none of it | FD&C Act 520(o)(2); binding statute |
| SaMD | Software as a Medical Device: software intended for a medical purpose that performs that purpose without being part of a hardware medical device | Calling embedded firmware SaMD | IMDRF N10:2013; international guidance |
| SiMD | Software in a Medical Device: software that is part of, drives, or controls a hardware medical device | Analyzing SiMD with standalone-software assumptions | IMDRF N10:2013; convention |
| MDSW | Medical device software under the EU regulations: software intended for a purpose in the MDR or IVDR definition, whether independent or driving or influencing a device | Assuming MDSW requires standalone deployment | MDCG 2019-11 Rev.1, June 2025; EU guidance |
| IVDMDSW | Medical device software that provides information within the IVD definition, based on data from specimens | Treating any software used in a laboratory as IVD software | MDCG 2019-11 Rev.1; EU guidance |
| Accessory | An article that is not itself a device but is intended to be used with a specific device to enable its intended purpose or directly assist its medical function | Treating any add-on as an accessory, or assuming an accessory inherits the host's class | MDR Article 2(2); binding regulation |
| Parent device | A finished device whose performance is supported, supplemented, or augmented by an accessory | Confusing the parent with the platform it runs on | FDA accessories guidance, 20 December 2017; guidance |
| Component | A part or software module intended for incorporation into a finished device and supplied to a manufacturer | Selling the same module directly to clinicians and still calling it a component | 21 CFR 820.3(c); binding regulation |
| Finished device | A device packaged and labeled for distribution, ready for use | Treating a component as finished because it is downloadable | 21 CFR 820.3(l); binding regulation |
| Combination product | A product combining a drug, device, or biological product, physically or functionally | Assuming software plus a drug is always a device | 21 CFR Part 3; binding regulation |
| Primary mode of action | The single mode of action expected to provide the most important therapeutic effect, used to assign a combination product to a lead center | Choosing the lead center by which team is larger | 21 CFR 3.2(m); binding regulation |
| Qualification (EU) | The determination of whether software falls within the scope of the MDR or IVDR at all | Confusing qualification with classification | MDCG 2019-11 Rev.1; EU guidance |
| Classification | The assignment of a risk class within a regime that already applies | Performing classification before qualification | MDR Annex VIII; binding regulation |
| Substantial equivalence | The 510(k) standard: same intended use and similar technological characteristics, or different characteristics with comparable safety and effectiveness | Treating a similar name or a similar algorithm as equivalence | FD&C Act 513(i); binding statute |
| Borderline product | A product whose status is genuinely uncertain on the facts | Declaring every borderline case a device to be safe, without recording why | Institutional practice; convention |

Three terms deserve more than a row.

**Objective intent.** Under 21 CFR 801.4, intended use is the objective intent of the persons legally responsible for
labeling, and intent may be shown by expressions, by the design or composition of the article, or by the circumstances
surrounding distribution. The word "objective" is doing the work: a company's internal belief that its product is a
wellness tool does not control if every outward signal says otherwise. In software, the design register is unusually rich.
A default-on order button, a red badge, a hard stop in a workflow, and a "recommended" label are all design statements,
and a reviewer will read them as evidence.

**Medical information versus signal or pattern.** The 520(o)(1)(E) exception turns on the input, not the output. Software
that analyzes a laboratory result value is analyzing medical information; software that analyzes the continuous output of
an assay instrument is analyzing a signal from an in vitro diagnostic device. The 2022 guidance draws the line using
sampling: a single clinically meaningful result is medical information, while repeated or continuous sampling that must be
interpreted as a sequence is a signal or pattern. This distinction decides a large share of real-world determinations, and
it is the reason a model trained on waveforms cannot reach the CDS exclusion however explainable it is.

**Support versus directive.** The statute says the function must "support or provide recommendations" to a health care
professional, and must enable that professional to review the basis so as not to rely primarily on the output. The 2022
guidance reads "support" as excluding a specific directive, a time-critical output, and an output intended to replace
clinician judgment. A function that says "start 1 mL/kg/hr sodium chloride now" is not supporting a decision; it is making
one. The vocabulary matters because teams describe the same feature as "a nudge" in the deck and "an order" in the
interface, and the interface is the register a reviewer will believe.

## 5. The mechanics, step by step

The procedure below produces two artifacts: a disposition for every function in every target jurisdiction, and a list of
facts that would change a disposition. Work in order. Steps 1 and 2 are preparation, steps 3 to 9 are the United States
gate, steps 10 and 11 are the European Union gate, and step 12 is the record. Do not skip to step 5 because the product
"is obviously CDS." Step 5 is the third filter, not the first.

A note on how to record results. Use four disposition codes rather than two: `excluded`, `device`, `out of scope`, and
`unresolved`. The fourth code is the one that protects you. It forces the memo to name the fact that would resolve the
question instead of converting an assumption into a conclusion.

### Step 1 — Inventory the functions and fix the unit of analysis

**Input.** Product description, screen flows, model inventory, data-flow diagram, roadmap, and any existing claims.

**Output.** A numbered function list in which each entry has one primary output, one primary user, and one clinical or
operational action.

**Rationale.** Section 520(o) excludes *software functions*, and section 520(o)(2) requires FDA to treat an excluded
function differently from a device function inside the same product. The statute therefore makes the function the unit of
analysis. A product-level determination is not merely imprecise; it is the wrong object.

**Procedure.** (1) List every output that a user could act on, including outputs that are displayed but not stored. (2)
For each, name the user by role and the action the user would take. (3) Group outputs only when user, action, and
consequence of error all match. (4) Label functions by what they do, not by product or feature name. (5) Mark each
function as clinical, administrative, operational, or educational. (6) Record shared inputs, because a shared input can
propagate a disqualifying data type across functions.

**Branches and edge cases.** If two outputs share a user but lead to different actions, keep them separate. If one
function feeds another, decide whether the upstream function is independently marketed; an image-quality check that only
serves the detector is part of the detector's claim, while an image-quality check sold on its own has its own claim. If a
product contains a patient-facing and a clinician-facing view of the same score, treat them as two functions, because the
520(o)(1)(E) analysis differs for each.

### Step 2 — Audit the claim surface

**Input.** The function list plus every artifact that describes the product.

**Output.** A claim register with one row per function per register, and a list of contradictions.

**Rationale.** Objective intent under 21 CFR 801.4 is assembled from expressions, design, and circumstances of
distribution. A determination that reads only the label is incomplete, and a determination that ignores the interface is
wrong, because the interface is where design becomes intent.

**Procedure.** (1) Collect the label, instructions for use, and any patient-facing material. (2) Capture the interface
text for every screen that shows an output, including button labels, defaults, and empty states. (3) Capture the website,
deck, demo script, and conference abstracts. (4) Capture order forms, statements of work, and pricing pages. (5) Record
which system calls the function and at what workflow moment. (6) Mark every sentence that names a disease, a treatment, a
diagnosis, a population, or a time window. (7) Flag contradictions between registers.

**Branches and edge cases.** A negative claim ("does not diagnose") does not neutralize a positive one elsewhere. A
disclaimer in the footer does not override a headline. A sales email is an expression of intent even when it is not part
of the label. If the interface displays a severity colour that the label does not describe, the colour is an unlabelled
claim.

### Step 3 — Test for a medical purpose under the US statute

**Input.** The claim register and the function list.

**Output.** A yes, no, or unresolved answer per function, with the statutory element identified.

**Rationale.** Section 201(h) of the FD&C Act defines a device by intended purpose: diagnosis, cure, mitigation,
treatment, or prevention of disease, or affecting the structure or function of the body. If no element is present, the
analysis ends at this step and the function is not a device regardless of how sophisticated it is. Note that the United
States definition does not name prediction or prognosis explicitly, while the European definition does; a pure
risk-forecasting function therefore needs a careful mapping to "diagnosis" or "prevention" in the United States.

**Procedure.** (1) Quote the statutory element that the claim implicates. (2) Map each intended-use phrase to an element.
(3) Identify the phrase that carries the most weight. (4) Check whether the function informs a clinical decision or only
an operational one. (5) Record unresolved mappings rather than resolving them by assertion.

**Branches and edge cases.** Administrative and operational functions usually fail this step: bed management, claims
analytics, staffing, and revenue-cycle prediction are not medical purposes. A function that predicts no-shows fails; a
function that predicts deterioration passes. A function that ranks patients for outreach may pass if the ranking is based
on a medical prediction, because the medical purpose sits in the prediction, not the list. General wellness claims sit
close to the line and depend on the specific wording and population.

### Step 4 — Screen the section 520(o)(1) categories A to D

**Input.** The function list and the claim register.

**Output.** A category match or no match, with the condition that decides it.

**Rationale.** The 21st Century Cures Act, enacted in December 2016, added section 520(o) to the FD&C Act. Paragraph (1)
lists five categories of software function that are not devices. Four of them are narrow and mostly administrative; the
fifth is the CDS category handled in step 5. Screening A to D first is efficient because most non-clinical functions exit
here.

**Procedure.** (1) Ask whether the function supports facility administration, including billing, claims, scheduling,
population health, or laboratory workflow. (2) Ask whether it is a healthy lifestyle function unrelated to disease. (3)
Ask whether it is an electronic patient record that meets all three statutory conditions. (4) Ask whether it only
transfers, stores, converts, or displays laboratory or device data without interpreting it. (5) Record the match and the
condition that could break it.

**Branches and edge cases.** The record exclusion in (C) requires that records be created, stored, transferred, or
reviewed by health care professionals, that the health information technology be certified under 42 U.S.C. 300jj-11(c)(5),
and that the function not interpret or analyze records, including medical image data. FDA has said it does not intend to
enforce the certification condition alone, which is an enforcement position and not a change to the statute. The display
exclusion in (D) does not survive interpretation: FDA's 2019 guidance on section 3060 states that alarms, alerts, flags,
and multi-patient prioritization displays involve analysis and are not excluded under (D). A dashboard that sorts a
worklist by a patient-specific risk is therefore outside (D) even though it looks like a display.

| Category | Question | Excludes when | Does not exclude when |
|---|---|---|---|
| (A) Administrative | Does it support facility administration? | Billing, claims, scheduling, population health, lab workflow | The output informs a patient's diagnosis or treatment |
| (B) Healthy lifestyle | Does it encourage a healthy lifestyle? | Unrelated to diagnosis, cure, mitigation, prevention, or treatment | It claims a disease benefit or a diagnostic output |
| (C) Electronic records | Is it an electronic patient record? | Three conditions met, including no interpretation | It interprets records, including image data |
| (D) Data display | Does it store, convert, or display lab or device data? | No interpretation or analysis | It interprets, prioritizes, or flags |
| (E) CDS | Does it support a health care professional's decision? | Four conditions in step 5 all met | Any condition fails |
| None | Does any category fit? | Not applicable | A medical purpose remains, so it is a device |

### Step 5 — Apply the section 520(o)(1)(E) CDS conditions

**Input.** The function description, the input data types, the user, the output text, and the review affordances.

**Output.** `non-device CDS`, `device`, or `unresolved`, with the failing condition identified.

**Rationale.** This is where most determinations are won and lost. The statute states four conditions, all of which must
hold. FDA's final guidance of 28 September 2022 explains how the agency reads them, and the guidance is where the
practical tests live.

**Procedure.** (1) Identify every input and classify it as medical information, image, signal from an in vitro diagnostic
device, or pattern from a signal acquisition system. (2) Confirm the function displays, analyzes, or prints medical
information rather than acquiring or processing a disqualifying input. (3) Confirm the user is a health care professional
and not a patient or caregiver. (4) Test whether the output is a specific directive, is time-critical, or is intended to
replace the professional's judgment. (5) Test whether the basis can be independently reviewed: are the inputs, the logic,
the validation, and the patient-specific facts all available to the user at the point of use? (6) Record the result per
condition.

**Branches and edge cases.** Condition 1 is a hard stop: no amount of explainability cures an image, signal, or pattern
input. Condition 3 is where patient-facing products fail, because a recommendation to a patient is outside the exclusion
even when a clinician could review it later. Condition 4 is the most contested in practice. A "Why?" link that opens a
generic disclaimer does not enable independent review; a panel that shows the contributing factors, their values, and the
population the model was validated in does. A function that is time-critical fails condition 3 under FDA's reading even if
a human is nominally in the loop, because review that cannot happen in the available time is not review.

| Condition | Statutory hook | Test question | Fails when |
|---|---|---|---|
| 1. Input restriction | 520(o)(1)(E) closing text | Does it acquire, process, or analyze an image, an IVD signal, or an acquisition pattern? | Any image, waveform, or repeated-sampling input |
| 2. Medical information | 520(o)(1)(E)(i) | Does it display, analyze, or print medical information? | It processes raw acquisition data rather than information |
| 3. Professional user and support | 520(o)(1)(E)(ii) | Does it support a professional's prevention, diagnosis, or treatment decision? | Patient user, specific directive, time-critical, or replacement of judgment |
| 4. Independent review | 520(o)(1)(E)(iii) | Can the professional review the basis and not rely primarily on the output? | No inputs, logic, validation, or patient facts are exposed |
| All four | 520(o)(1)(E) | Do all conditions hold together? | Any one fails |

### Step 6 — Apply the multi-function rule in section 520(o)(2)

**Input.** Dispositions from steps 3 to 5 for every function in the product.

**Output.** A product-level plan that regulates the device functions and leaves the excluded functions alone, plus a list
of interactions between them.

**Rationale.** Section 520(o)(2) provides that when a product contains at least one excluded function and at least one
device function, the Secretary may not regulate the excluded function as a device, but may assess the excluded function's
impact when evaluating the safety and effectiveness of the device functions. The practical consequence is that a single
product can carry two regulatory regimes at once, and the boundary between them must be documented.

**Procedure.** (1) Tabulate dispositions per function. (2) Identify shared data, shared screens, and shared model
artifacts. (3) Ask whether an excluded function's output is an input to a device function; if so, the device function's
risk analysis must cover it. (4) Write the boundary into the labeling and the architecture. (5) Decide whether to ship as
one product or split it.

**Branches and edge cases.** A shared model that serves both an excluded summarizer and a device risk score is a single
technical artifact with two regulatory roles; changing it changes the device. A common authentication and audit layer is
infrastructure, not a device function. If the excluded function can alter the device function's output, the exclusion does
not protect the combined behaviour, and the safer reading is that the combined behaviour carries the device claim.

| Product composition | Disposition | Action |
|---|---|---|
| All functions excluded | Not a device | Document exclusions; retain the analysis |
| All functions are device functions | Device | Classify and select a pathway |
| Mixed, with no data dependency | Split regulation | Separate labels, release notes, and change control per function |
| Mixed, with a data dependency | Device claim covers the interaction | Risk analysis must include the excluded function's outputs |
| Mixed and inseparable in use | Unresolved | Treat the combined behaviour as a device until resolved |

### Step 7 — Locate the hardware boundary

**Input.** Architecture diagram, deployment model, commercial relationship to any hardware.

**Output.** SaMD, SiMD, embedded software, accessory, component, or non-device infrastructure.

**Rationale.** The category determines the lifecycle standard, the submission route, and who owns the evidence. Software
that drives a pump and software that reads images from a scanner can share a codebase and have nothing else in common
regulatorily.

**Procedure.** (1) Ask whether the software performs its medical purpose without being part of a hardware device. (2) Ask
whether it controls or influences a hardware medical function. (3) Ask whether it is embedded, distributed separately, or
both. (4) Ask whether it is supplied to a device manufacturer for incorporation. (5) Ask whether it is supplied to enable
a specific host device's intended use. (6) Record the category and the boundary fact.

**Branches and edge cases.** The same algorithm can be SaMD in a cloud deployment and SiMD when embedded in the scanner,
and the risk file must change with the deployment. A firmware update that changes a dose calculation is part of the device
even if it is delivered over the network. An app that displays a sensor's output may be an accessory, an independent
device, or neither, depending on whether it adds interpretation. A module sold only to manufacturers is a component, but
the identical module sold to clinicians is a finished device.

| Deployment | Category | Why | Consequence |
|---|---|---|---|
| Cloud service reading images from a separate scanner | SaMD | Performs a medical purpose without being part of hardware | Standalone submission and software lifecycle evidence |
| App that programs an infusion pump | SiMD or part of the device | Drives a hardware medical function | Device lifecycle, dose-error risk analysis, hardware interface testing |
| Firmware inside an imaging system | Part of the device | The medical function is the system's | System-level verification and validation |
| Module licensed to a device manufacturer | Component | Supplied for incorporation, not for end use | Contract controls; manufacturer owns the submission |
| App sold with a sensor to enable its use | Accessory candidate | Enables or assists a specific host's intended purpose | Own classification analysis; do not assume inheritance |
| Authentication and logging layer | Infrastructure | No medical purpose | Not a device function, but in scope for cybersecurity |

### Step 8 — Screen for in vitro diagnostic status in the United States

**Input.** The input data, the specimen pathway, and the claimed output.

**Output.** Device, in vitro diagnostic device, or unresolved.

**Rationale.** In the United States, in vitro diagnostic products are devices under section 201(h), which names "in vitro
reagent" in the definition. They are classified through the in vitro diagnostic regulations, principally 21 CFR Part 862
for clinical chemistry and clinical toxicology, and they carry different classification regulations, product codes, and
evidence expectations from imaging or monitoring software. Getting this wrong sends the predicate search into the wrong
database neighbourhood.

**Procedure.** (1) Ask whether the product examines a specimen derived from the human body. (2) Ask whether the output is
a measurement, a result, or an interpretation used for diagnosis or treatment. (3) Ask whether the software's contribution
is measurement, interpretation, or reporting. (4) Identify the analyte and the clinical use. (5) Record the category.

**Branches and edge cases.** Software that only displays a result is a display function, and the step 4 analysis applies.
Software that interprets a result for a clinical decision is doing more than displaying, and FDA's 2019 section 3060
guidance treats interpretation as outside the display exclusion. Software that controls an analyzer is part of the
analyzer. A digital pathology tool is a device but usually not an in vitro diagnostic, because the input is an image
rather than a specimen measurement, which is exactly the distinction that changes between jurisdictions.

### Step 9 — Screen for combination product status

**Input.** All product constituents, their modes of action, and the claimed therapeutic effect.

**Output.** Device, combination product, or unresolved, with the primary mode of action.

**Rationale.** A product that combines a drug, device, or biological product is assigned to a lead center under 21 CFR
Part 3 according to its primary mode of action, defined in 21 CFR 3.2(m) as the single mode of action expected to provide
the most important therapeutic effect. Most digital products are not combination products, but the ones that are have very
different submission logistics.

**Procedure.** (1) List every constituent and its mode of action. (2) Ask whether any constituent is integral to the
claimed effect rather than supportive of it. (3) Identify the primary mode of action. (4) Identify the lead center and the
applicable premarket route. (5) Record the conclusion.

**Branches and edge cases.** A drug with a companion app that only displays adherence data is not a combination product. A
drug whose safe use depends on a proprietary dosing algorithm that is part of its labeling may be. A device with an
antimicrobial coating is a combination product even though the device dominates the use case. The analysis is about the
claimed therapeutic effect, not about which component is more expensive.

| Composition | Primary mode of action | Likely lead | Note |
|---|---|---|---|
| Software only | Software | Device center | The ordinary case |
| Software plus hardware sensor | Device function | Device center | Accessory analysis may still apply |
| Drug plus adherence app | Drug | Drug center | App is labeling support, not a constituent |
| Drug plus dose-determining algorithm | Depends on the claim | Drug or device center | Analyse the claimed effect, not the code volume |

### Step 10 — Qualify under the European Union gate

**Input.** The EU intended purpose, the data the software acts on, and the population it serves.

**Output.** Out of scope, MDR software, or IVDR software, with the decision step recorded.

**Rationale.** The European Union has no CDS exclusion. It has a purpose-based qualification with a narrow carve-out for
software that performs only storage, archival, communication, simple search, or lossless compression. The Medical Device
Coordination Group's guidance MDCG 2019-11 Rev.1, published on 17 June 2025, sets out five decision steps in its Figure 1
and states that qualification depends on intended purpose alone, regardless of whether the software is embedded, on a
platform, or in the cloud.

**Procedure.** (1) Confirm the artifact is software within the guidance's definition. (2) Check whether it is an Annex XVI
product, an accessory, or software driving or influencing a device; if so, it is treated as part of that device or as an
accessory in its own right. (3) Check whether it does anything beyond storage, archival, communication, simple search, and
lossless compression. (4) Check whether the action benefits individual patients rather than only aggregating population
data or providing generic pathways. (5) Check the medical purpose against MDR Article 2(1) or IVDR Article 2(2). (6)
Record the step at which the function exits or enters.

**Branches and edge cases.** A picture archiving viewer that only stores and displays is out of scope; the same viewer
with a measurement tool that informs a diagnosis is in scope. Software that aggregates population data for a registry is
out of scope even when clinicians use it, unless the output is directed at individual patients. The absence of a
risk-of-harm criterion is deliberate: MDCG 2019-11 states that the risk of harm is not a criterion for qualification,
which is the opposite of how many teams reason. Harm enters at classification, not at qualification.

| Step | Question | If yes | If no |
|---|---|---|---|
| 1 | Is it software within the guidance's definition? | Continue | Out of this guidance; may still be in the regulations |
| 2 | Annex XVI, accessory, or driving a device? | Treated as part of the device or as an accessory | Continue |
| 3 | Does it act on data beyond storage, archival, communication, simple search, lossless compression? | Continue | Out of scope |
| 4 | Is the action for the benefit of individual patients? | Continue | Out of scope |
| 5 | Does it have a medical purpose under Article 2(1) or 2(2)? | MDSW | Out of scope |

### Step 11 — Split medical device software from in vitro diagnostic software

**Input.** The data sources and the intended purpose of the output.

**Output.** MDR software, IVDR software, or a weighted determination with the reasoning recorded.

**Rationale.** The MDR and the IVDR are separate regulations with separate classification rules and separate conformity
assessment. MDCG 2019-11 Rev.1 Figure 2 sets out three decision steps for the split, and it anticipates the mixed case
rather than forbidding it.

**Procedure.** (1) Ask whether the software provides information within the IVD definition in IVDR Article 2(2), which
covers physiological or pathological state, congenital impairments, predisposition, safety and compatibility of
recipients, prediction of treatment response, and definition or monitoring of therapeutic measures. (2) If the information
is based solely on data from in vitro diagnostic devices, treat it as IVD software. (3) If the data are mixed, ask which
source substantially drives the intended purpose; if the answer is genuinely mixed, weight the sources by their
significance to the output and record the weighting.

**Branches and edge cases.** A model that combines a laboratory biomarker with an imaging feature is the hard case, and
the guidance's answer is a documented qualitative weighting rather than a rule. The weighting is a judgment, so it needs a
written rationale and a second reviewer. A software module that interprets an assay result is IVD software even when it is
sold by an imaging vendor.

| Step | Question | Outcome |
|---|---|---|
| 1 | Does the output provide information within IVDR Article 2(2)? | If no, MDR software |
| 2 | Is the data solely from in vitro diagnostic devices? | If yes, IVDR software |
| 3 | Which source substantially drives the intended purpose? | Apply that regulation; if genuinely mixed, weight and record |

### Step 12 — Record the disposition, the confidence, and the re-review triggers

**Input.** All prior steps.

**Output.** A device-status memorandum with a disposition per function per jurisdiction, a confidence statement, and a
dated source list.

**Rationale.** A determination is a legal conclusion drawn from facts, and both change. The memo's value is that it lets a
reviewer reproduce the reasoning and see which fact would overturn it. That is also what makes it useful in a
pre-submission, a 513(g) request, or a diligence review.

**Procedure.** (1) State the function, its user, its output, and its action. (2) State the disposition and the governing
provision. (3) State the confidence: high, medium, or low, with the reason. (4) List the facts that would change the
disposition. (5) List the open questions and who will answer them. (6) Date the memo and version every source. (7) Set
triggers for re-review: a label change, a new user, a new input data type, a new jurisdiction, a guidance revision, or a
model change that alters the output's actionability.

**Branches and edge cases.** A disposition that rests on a single phrase should trigger a claim revision rather than a
longer memo. A disposition that differs between jurisdictions is normal and should be explained, not harmonized by force.
A borderline product needs external input before evidence investment, and the memo should say so explicitly.

| Disposition | Meaning | Next action |
|---|---|---|
| Excluded | A statutory exclusion applies to this function | Document the conditions; keep the analysis under change control |
| Non-device CDS | All four 520(o)(1)(E) conditions hold | Document; monitor the claim surface for drift |
| Device | Medical purpose and no exclusion | Classify and select a pathway |
| Out of scope (EU) | Fails a Figure 1 step | Document; check other EU legislation such as the AI Act |
| Unresolved | A material fact is missing or contested | Name the fact, the owner, and the deadline; do not ship a claim that depends on it |

The steps produce a matrix, not a verdict. A product with six functions can produce six rows in each jurisdiction, and the
rows do not need to agree. What they must do is name their basis, so that the next reviewer can attack the weakest row
instead of the whole document.

## 6. Worked example

The case is hypothetical. The product, sponsor, site counts, and performance figures are
constructed to show the reasoning; they do not describe a real clearance, a real predicate, or a
real study. No clearance number, product code, or regulation number is asserted, because those
must be looked up in the current databases and a fabricated identifier would be worse than no
identifier.

### 6.1 Product and claim

RenalSafe is a software module that runs inside a hospital's radiology order-entry system. When a
clinician orders a contrast-enhanced computed tomography (CT) examination, the module reads the
patient's age, most recent estimated glomerular filtration rate (eGFR), diabetes status, active
loop-diuretic orders, the planned contrast volume, and the order urgency. It computes a 48-hour
risk of contrast-associated acute kidney injury (CA-AKI), displays one of three risk bands, and
offers a pre-hydration order for clinician confirmation.

The draft intended-use statement reads:

"RenalSafe is intended for use by prescribing clinicians in hospital and outpatient imaging
settings to identify adult patients aged 18 and older with an eGFR of at least 30 mL/min/1.73 m2
who are at elevated risk of contrast-associated acute kidney injury before a contrast-enhanced
computed tomography examination, and to offer a protocol-concordant intravenous hydration order
for clinician confirmation. RenalSafe does not diagnose acute kidney injury, does not select the
contrast agent, and is not intended for patients receiving emergent imaging where the examination
cannot be delayed for risk assessment."

Four regulatory choices sit inside that statement. The user is a prescribing clinician, not a
patient. The population excludes emergent imaging, which removes the time-critical branch. The
action is risk identification plus an offered order, not an executed order. The negative clauses
exclude diagnosis and contrast-agent selection. Each choice narrows the evidence burden, and each
also narrows the market.

### 6.2 Function inventory and the United States gate

The product has four user-facing functions, and they do not share a status.

| Function | Output | User | Action | US disposition |
|---|---|---|---|---|
| F1 Risk computation | 48-hour CA-AKI risk and band | Ordering clinician | Decide whether to modify the order | Device |
| F2 Hydration offer | Specific order: 1 mL/kg/hr for 6 hours | Ordering clinician | Accept, modify, or decline an order | Device |
| F3 Ordering audit | Weekly ordering-pattern report | Imaging department manager | Operational review | Excluded under 520(o)(1)(A) |
| F4 Discharge text | Plain-language instruction to the patient | Patient | Follow advice at home | Device |

F1 and F2 are device functions. The inputs are medical information rather than a signal: a
reported eGFR is a single clinically meaningful laboratory result, not a waveform from an assay
instrument. If the module ingested the raw assay trace and derived the eGFR itself, condition 1 of
section 520(o)(1)(E) would fail and the CDS route would close immediately.

F2 fails condition 3 independently. "Start 1 mL/kg/hr for 6 hours" is a specific directive, and
FDA's 2022 guidance reads "support or provide recommendations" as excluding one. A click does not
convert the directive into support; the question is what the output tells the user to do, not who
presses the key.

F3 is excluded: it aggregates ordering patterns for departmental management and has no
patient-specific clinical output, which places it in section 520(o)(1)(A). F4 is a device, because
a recommendation to a patient falls outside section 520(o)(1)(E), which is limited to
recommendations to a health care professional. F4 is small, cheap to build, and the function most
likely to be overlooked in a product review, which is exactly why it belongs in the inventory.

Under section 520(o)(2) the product is a mixed product. FDA may not regulate F3 as a device, but it
may assess F3's impact on the device functions, and the audit report reads order data that F2
writes. The interaction belongs in the risk file even though F3 itself is excluded.

### 6.3 United States classification and pathway

F1 and F2 fall within Class II. The clinical risk is moderate: a missed high-risk patient may
develop AKI, but the clinician retains the decision, the injury is usually reversible, and the
intervention is a well-understood hydration protocol. The product-code search returns candidates in
the clinical decision support and contrast-safety neighbourhood, and the sponsor records them
without selecting one until the regulation text and cleared indications have been read.

The pathway is a 510(k) under section 513(i) of the FD&C Act. The sponsor identifies three legally
marketed predicates whose intended uses include flagging patients at risk of contrast-associated
kidney injury and offering a protocol order. None has an identical population, and one is
explicitly adjunctive with clinician confirmation required. The predicate matrix records each
device's intended use, input data, output form, and stated limitations.

The comparison is not "does the model work." It is "does the difference between RenalSafe and the
predicate raise a different question of safety and effectiveness?" Three differences could: the
three-band output rather than a binary flag; the explicit hydration order rather than a generic
recommendation; and the exclusion of emergent imaging. Each must be addressed with performance
data, or with an argument supported by evidence that the difference does not change the risk
question.

### 6.4 European Union qualification and classification

The EU analysis starts again from the intended purpose. Under MDCG 2019-11 Rev.1 Figure 1, the
module is software, it is not an Annex XVI product, it is not an accessory to a specific device,
and it does not drive or influence a hardware device. That last point deserves attention: the
module writes an order into the electronic health record, and a pharmacist compounds the bag. It
does not program an infusion pump. If a later version transmitted a rate directly to a pump,
Figure 1 step 2 would apply and the software would fall within the class of the driven device under
Annex VIII implementing rule 3.3.

The module acts on data beyond storage and display and benefits individual patients, so it passes
steps 3 and 4, and it has a medical purpose under MDR Article 2(1). Figure 2 then asks whether the
output provides information within IVDR Article 2(2). It does not: the module predicts a patient's
risk from laboratory results and clinical variables, but it does not examine a specimen or report
on an analyte. The clinical variables and the planned contrast volume are essential inputs, so the
information is not based solely on data from in vitro diagnostic devices. It is MDR software, not
IVDR software.

Under Annex VIII rule 11, the module provides information used to take a preventive decision. The
possible harm is serious deterioration requiring intervention, since a missed high-risk patient may
develop AKI requiring dialysis. The sponsor classifies it as Class IIb and records the alternative
view that the harm is irreversible in the subset progressing to end-stage renal disease, which
would support Class III. The memo states why IIb was chosen (the population excludes emergent
imaging, the clinician confirms the order, the protocol is standard of care) and records that a
notified body may disagree.

### 6.5 The evidence gap

The predicate's public summary reports an area under the receiver operating characteristic curve
(AUROC) of 0.78 on a held-out set and does not report calibration, net benefit, or performance by
eGFR stratum. RenalSafe's development set contains 48,200 orders with 3,140 events, the temporal
validation set contains 12,400 orders with 806 events, and the external site contributes 6,900
orders with 372 events. Pooled prevalence is 3,512 events in 55,100 orders, or 6.4 percent.

Discrimination is comparable, not superior: AUROC is 0.81 (95 percent confidence interval 0.78 to
0.84) at the development site and 0.77 (0.72 to 0.82) at the external site. The submission cannot
claim superiority, and it does not need to. It needs to show that performance is adequate for the
claimed action and that the output is calibrated enough for the displayed band to mean what it
says.

At a 5 percent threshold, sensitivity is 0.82 and specificity is 0.79. At the intended-use
prevalence of 6.4 percent:

- Positive predictive value is 0.82 x 0.064 divided by (0.82 x 0.064 + 0.21 x 0.936), which is
  0.0525 divided by 0.2490, or 0.211.
- Negative predictive value is 0.79 x 0.936 divided by (0.79 x 0.936 + 0.18 x 0.064), which is
  0.7394 divided by 0.7510, or 0.985.
- Per 1,000 orders the module flags 249 patients, of whom 52.5 have the outcome and 196.6 do not.
  It misses 11.5 events.

Four of every five flagged patients will not develop CA-AKI. That is not a defect in the model; it
is what a 6.4 percent prevalence does to a screening test. It is a labelling obligation: the output
must be described as a risk marker that triggers review, not as a prediction that the patient will
be harmed.

The net benefit calculation makes the same point in decision terms. At a threshold probability of
5 percent, the model's net benefit is 0.0525 minus 0.1966 x (0.05 / 0.95), or 0.0421 per patient.
Treating every patient has a net benefit of 0.064 minus 0.936 x (0.05 / 0.95), or 0.0147. The model
adds 0.0274 per patient, or 27.4 per 1,000 orders, over universal pre-hydration. That margin is the
clinical case for the product, and it is thin enough that a calibration shift can erase it.

### 6.6 The submission decision

The sponsor files a 510(k) for F1 and F2 and a Class IIb technical documentation package for the
same functions in the EU. F3 is documented as an excluded administrative function. F4 is removed
from the release, because a patient-facing recommendation would require its own analysis and would
broaden the claim without adding revenue.

The decision comes with three claim changes. The sentence "prevents contrast-associated acute
kidney injury" is removed from the website and the deck, because the product offers a
protocol-concordant order and the sponsor has no trial showing that the order prevents injury. The
default state of the order button changes from pre-checked to unchecked, so that acceptance
requires a deliberate action. The label adds the sentence "RenalSafe does not rule out acute kidney
injury, and a low-risk band does not remove the need for clinical assessment." The first change
addresses the claim surface, the second the design register, and the third the negative predictive
claim that the arithmetic makes unavoidable.

### 6.7 Two diagnostics that would sink the submission

**Diagnostic 1: calibration and informative missingness at the external site.** The calibration
slope at the development site is 0.91 with an intercept of -0.05, which is acceptable. At the
external site the slope is 0.68 with an intercept of -0.42, so the model systematically
overpredicts. The flag rate rises from 24.9 percent of orders to 41.2 percent. The cause is not
random: the planned contrast volume is missing in 38 percent of external orders, concentrated in
urgent and unscheduled examinations, and the model imputes the development-set median. In the
missing-volume stratum the observed event rate is 9.8 percent while the model predicts 5.6 percent.
The intended use excludes emergent imaging, but the external cohort contains those orders and the
site cannot reliably distinguish them at the point of order entry. A reviewer can reasonably
conclude that the evidence does not support the claimed performance in a setting the sponsor
intends to sell into, and the submission's central performance table is the document that proves
it.

**Diagnostic 2: the workflow contradicts the claim.** The intended use says the clinician confirms
the order. The formative usability study shows that 62 percent of accepted orders were accepted
within three seconds of the prompt appearing, and that 71 percent of participants could not name
the three inputs that drove the band, because the interface displayed the risk percentage and the
order button but not the contributing factors. This is a problem for the device claim, whose safety
argument depends on clinician confirmation. It is also a problem for the alternative strategy: if
the sponsor had removed the specific directive to pursue the non-device CDS route, the same facts
would defeat condition 4, because the basis was not reviewable and users relied primarily on the
output. The finding closes both doors, which is why it belongs in the first review rather than the
last.

### 6.8 Vary one assumption

Change one thing: the intended-use sentence "offers a protocol-concordant intravenous hydration
order for clinician confirmation" becomes "displays a risk band and the contributing factors, and
links to the institutional hydration protocol." The model, the data, the threshold, and the
population are unchanged.

In the United States the specific directive disappears, so condition 3 no longer fails on that
ground. If the interface lists the inputs and their values, condition 4 is supportable. If the
function is not time-critical and the user remains a clinician, all four conditions can hold, and
the function becomes non-device CDS under section 520(o)(1)(E). No 510(k) is required, and no
device QMS obligation attaches to that function. The audit function remains excluded, and the
patient-facing function would still be a device if it shipped.

In the European Union nothing changes. There is no CDS exclusion, the function still has a medical
purpose under MDR Article 2(1), and it is still Class IIb under rule 11. The product becomes
unregulated in one market and regulated in the other, on identical code and identical evidence.

| Variation | US pathway | EU status | Evidence consequence |
|---|---|---|---|
| Base: risk band plus a specific order | 510(k), Class II | MDSW, Class IIb | Analytical, calibration, external validation, reader study, human factors |
| Wording: risk band, factors, protocol link | Non-device CDS, no submission | MDSW, Class IIb | US evidence optional but EU clinical evaluation still required |
| Wording: autonomous order executed without confirmation | Device, likely De Novo if no predicate | MDSW, Class III if harm is irreversible | Prospective outcome study, stricter post-market plan |
| Predicate unavailable or withdrawn | De Novo request | Unchanged | Stand-alone safety and effectiveness evidence |
| Risk class raised to III in the EU | Unchanged | Class III, stricter conformity assessment | Clinical investigation unless equivalence is supportable |

The lesson is not that one version is better. It is that a clause in a sentence moves the product
between regimes, and the clause is cheaper to change before the evidence is collected. A withdrawn
predicate changes the US pathway without touching the EU file, and a different harm
characterization changes the EU class without touching the US file. Each axis is tested separately,
because the two gates do not move together.

## 7. Code, query, or calculation

The script below turns the decision ladder into executable tests. It is deliberately small enough to audit in one sitting.
It implements a simplified United States gate, a simplified European Union qualification gate, and the predictive-value
and net-benefit arithmetic used in section 6. It is a teaching aid and a template for a memo appendix, not a legal
determination engine.

Three deliberate simplifications are visible in the code. It collapses the section 520(o)(1)(C) record conditions into a
single flag, because the certification condition is an enforcement position rather than a fact a product team can test. It
does not implement Annex VIII rule 11 classification, which is topic 01's subject, so a European result means "in scope,"
not "Class IIb." And it treats the four CDS conditions as independent booleans, when in practice condition 3 is a judgment
about wording that needs a human reader.

```python
"""Device-status decision engine for Module 14, Topic 02.

Teaching aid, not legal advice. Simplified versions of FD&C Act 201(h) and
520(o), and of MDCG 2019-11 Rev.1 Figures 1 and 2.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Function:
    """One software function, described by the facts a reviewer needs."""

    name: str
    medical_purpose: bool = True
    category_520o: str | None = None       # "A" to "E", or None
    interprets_clinical_data: bool = False
    analyzes_image_or_signal: bool = False
    displays_medical_info: bool = True
    hcp_user: bool = True
    specific_directive: bool = False
    time_critical: bool = False
    basis_reviewable: bool = True
    eu_acts_on_data: bool = True
    eu_benefit_individual: bool = True
    eu_ivd_information: bool = False
    eu_drives_hardware: bool = False


def us_status(f: Function) -> tuple[str, str]:
    """Return (status, reason) for the US device gate."""
    if not f.medical_purpose:
        return ("not a device", "no medical purpose under 201(h)")
    cat = f.category_520o
    if cat == "A":
        return ("not a device", "520(o)(1)(A) administrative support")
    if cat == "B":
        return ("not a device", "520(o)(1)(B) healthy lifestyle, unrelated to disease")
    if cat == "C":
        return ("not a device", "520(o)(1)(C) electronic patient record")
    if cat == "D":
        if f.interprets_clinical_data:
            return ("device", "520(o)(1)(D) does not exclude interpretation")
        return ("not a device", "520(o)(1)(D) display without interpretation")
    if cat == "E":
        if f.analyzes_image_or_signal:
            return ("device", "520(o)(1)(E) exception: image, IVD signal, or pattern")
        if not f.displays_medical_info:
            return ("device", "520(o)(1)(E) no medical information shown or analyzed")
        if not f.hcp_user:
            return ("device", "520(o)(1)(E) recommendation is not to a health professional")
        if f.specific_directive:
            return ("device", "520(o)(1)(E) specific directive is not 'support'")
        if f.time_critical:
            return ("device", "520(o)(1)(E) time-critical output is not 'support'")
        if not f.basis_reviewable:
            return ("device", "520(o)(1)(E) basis cannot be independently reviewed")
        return ("non-device CDS", "520(o)(1)(E) criteria met")
    return ("device", "medical purpose; no 520(o) exclusion applies")


def eu_qualify(f: Function) -> tuple[str, str]:
    """Return (status, reason) for the EU qualification gate, MDCG 2019-11."""
    if f.eu_drives_hardware:
        return ("MDR, class of driven device", "Figure 1 step 2 and Annex VIII rule 3.3")
    if not f.eu_acts_on_data:
        return ("out of scope", "Figure 1 step 3: storage, archival, communication, "
                                "simple search, lossless compression")
    if not f.eu_benefit_individual:
        return ("out of scope", "Figure 1 step 4: not for individual patients")
    if not f.medical_purpose:
        return ("out of scope", "Figure 1 step 5: no medical purpose in Article 2(1) or 2(2)")
    if f.eu_ivd_information:
        return ("IVDR (IVD MDSW)", "Figure 2 step 1: information within the IVD definition")
    return ("MDR (MD MDSW)", "MDR Article 2(1) medical purpose")


def predictive_values(sensitivity: float, specificity: float, prevalence: float) -> dict[str, float]:
    """PPV, NPV, and the four cells per 1,000 patients."""
    for name, value in {"sensitivity": sensitivity, "specificity": specificity,
                        "prevalence": prevalence}.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    tp = sensitivity * prevalence
    fn = (1.0 - sensitivity) * prevalence
    fp = (1.0 - specificity) * (1.0 - prevalence)
    tn = specificity * (1.0 - prevalence)
    return {"ppv": tp / (tp + fp), "npv": tn / (tn + fn),
            "tp": tp * 1000, "fp": fp * 1000, "fn": fn * 1000, "tn": tn * 1000,
            "flagged": (tp + fp) * 1000}


def net_benefit(sensitivity: float, specificity: float, prevalence: float,
                threshold: float) -> dict[str, float]:
    """Decision-curve net benefit per patient, plus the treat-all comparator."""
    cells = predictive_values(sensitivity, specificity, prevalence)
    tp, fp = cells["tp"] / 1000, cells["fp"] / 1000
    odds = threshold / (1.0 - threshold)
    return {"model": tp - fp * odds,
            "treat_all": prevalence - (1.0 - prevalence) * odds}


if __name__ == "__main__":
    us_cases = [
        Function("Administrative scheduling", category_520o="A"),
        Function("Portal lab display, no interpretation", category_520o="D"),
        Function("Risk band, clinician, reviewable basis", category_520o="E"),
        Function("Risk band plus a specific hydration order",
                 category_520o="E", specific_directive=True),
        Function("ECG rhythm detection", category_520o="E", analyzes_image_or_signal=True),
        Function("Patient-facing triage advice", category_520o="E", hcp_user=False),
        Function("Radiology report summary for a clinician",
                 category_520o="E", basis_reviewable=False),
        Function("Plain medical literature search", medical_purpose=False),
    ]
    eu_cases = [
        Function("PACS viewer, storage and display only", eu_acts_on_data=False),
        Function("Population registry analytics", eu_benefit_individual=False),
        Function("Dose calculator driving an infusion pump", eu_drives_hardware=True),
        Function("CA-AKI risk score"),
        Function("Analyte interpretation for a lab assay", eu_ivd_information=True),
    ]
    for cases, gate in ((us_cases, us_status), (eu_cases, eu_qualify)):
        width = max(len(c.name) for c in cases)
        for c in cases:
            status, reason = gate(c)
            print(f"{c.name:{width}s}  ->  {status:32s}  [{reason}]")

    cells = predictive_values(0.82, 0.79, 0.064)
    nb = net_benefit(0.82, 0.79, 0.064, threshold=0.05)
    print("\nCA-AKI module at sens=0.82, spec=0.79, prevalence=0.064, threshold=0.05")
    print(f"PPV = {cells['ppv']:.3f}")
    print(f"NPV = {cells['npv']:.3f}")
    print(f"Per 1,000 orders: TP={cells['tp']:.1f} FP={cells['fp']:.1f} "
          f"FN={cells['fn']:.1f} TN={cells['tn']:.1f} flagged={cells['flagged']:.1f}")
    print(f"Net benefit, model     = {nb['model']:.4f} per patient")
    print(f"Net benefit, treat all = {nb['treat_all']:.4f} per patient")
    print(f"Delta versus treat all = {nb['model'] - nb['treat_all']:.4f} per patient "
          f"({(nb['model'] - nb['treat_all']) * 1000:.1f} per 1,000)")
```

Expected output from a Python 3.12 run:

```text
Administrative scheduling                  ->  not a device                      [520(o)(1)(A) administrative support]
Portal lab display, no interpretation      ->  not a device                      [520(o)(1)(D) display without interpretation]
Risk band, clinician, reviewable basis     ->  non-device CDS                    [520(o)(1)(E) criteria met]
Risk band plus a specific hydration order  ->  device                            [520(o)(1)(E) specific directive is not 'support']
ECG rhythm detection                       ->  device                            [520(o)(1)(E) exception: image, IVD signal, or pattern]
Patient-facing triage advice               ->  device                            [520(o)(1)(E) recommendation is not to a health professional]
Radiology report summary for a clinician   ->  device                            [520(o)(1)(E) basis cannot be independently reviewed]
Plain medical literature search            ->  not a device                      [no medical purpose under 201(h)]
PACS viewer, storage and display only     ->  out of scope                      [Figure 1 step 3: storage, archival, communication, simple search, lossless compression]
Population registry analytics             ->  out of scope                      [Figure 1 step 4: not for individual patients]
Dose calculator driving an infusion pump  ->  MDR, class of driven device       [Figure 1 step 2 and Annex VIII rule 3.3]
CA-AKI risk score                         ->  MDR (MD MDSW)                     [MDR Article 2(1) medical purpose]
Analyte interpretation for a lab assay    ->  IVDR (IVD MDSW)                   [Figure 2 step 1: information within the IVD definition]

CA-AKI module at sens=0.82, spec=0.79, prevalence=0.064, threshold=0.05
PPV = 0.211
NPV = 0.985
Per 1,000 orders: TP=52.5 FP=196.6 FN=11.5 TN=739.4 flagged=249.0
Net benefit, model     = 0.0421 per patient
Net benefit, treat all = 0.0147 per patient
Delta versus treat all = 0.0274 per patient (27.4 per 1,000)
```

Four cautions apply to the use of this output. First, the function does not know whether a claim is true; it only applies
the facts you give it. Passing a `specific_directive=False` flag to a product whose interface says "start now" produces a
wrong answer with a confident string. Second, the real procedure has four disposition codes, one of which is `unresolved`,
and the code has no equivalent. A production memo needs that case, which is why section 5 keeps a separate row for facts
that are missing rather than forcing them into a status. Third, the European gate is qualification only. It does not tell
you the class, the conformity route, or
whether the AI Act applies, and a European answer of "MDR (MD MDSW)" is the start of the analysis rather than the end.
Fourth, `net_benefit` assumes the sensitivity and specificity estimated at one threshold transfer to another population at
that same threshold. Section 6.7 shows exactly how that assumption fails when prevalence and calibration shift, and the
function's tidy 27.4 per 1,000 is the number that a calibration slope of 0.68 puts at risk.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and is not legal advice. Device status is a fact-specific legal conclusion. Confirm the current statutes, regulations, guidance, and database records with qualified regulatory counsel before relying on a determination.

The sources below are binding unless marked otherwise. A statute or regulation creates an obligation. Guidance describes
an agency's current thinking and is not legally binding, though a submission that departs from it without an alternative
rationale invites questions. A standard becomes binding when a regulation incorporates it, when a contract requires it, or
when a manufacturer adopts it into its own quality system. Convention is local practice: a template, a review habit, a
house style. A memo that mixes the four makes a recommendation look more authoritative than it is, and the mixing is easy
to spot because the citations are missing.

| Source | Status | What it constrains | What it does not do |
|---|---|---|---|
| FD&C Act 201(h), 21 U.S.C. 321(h) | Binding US statute | Defines a device by intended purpose and lists software exclusions by reference | Does not classify the device or select a pathway |
| FD&C Act 520(o), 21 U.S.C. 360j(o) | Binding US statute | Excludes five categories of software function and sets the multi-function rule | Does not create a general exemption for clinical software |
| FD&C Act 513(i) | Binding US statute | Sets the substantial equivalence standard for 510(k) | Does not determine whether a predicate exists |
| FD&C Act 513(f)(2) | Binding US statute | Creates the De Novo route for novel low- and moderate-risk devices | Does not make De Novo available for high-risk devices |
| 21 CFR 801.4 | Binding US regulation | Defines intended use and the evidence of objective intent | Does not decide whether a specific claim is a device claim |
| 21 CFR 820.3 | Binding US regulation | Defines component and finished device | Does not settle accessory status |
| 21 CFR Part 3 | Binding US regulation | Assigns combination products by primary mode of action | Does not cover software-plus-hardware systems that are not combination products |
| FDA CDS guidance, final 28 September 2022 | Guidance | Explains FDA's reading of 520(o)(1)(E), including specific directives, time-critical outputs, and independent review | Is not a safe harbour, and failing a condition does not by itself make a function a device |
| FDA device software policy, 28 September 2022 | Guidance | Describes oversight of device software functions and enforcement discretion | Does not change the statutory definition |
| FDA accessories guidance, 20 December 2017 | Guidance | Describes accessory and parent device, and accessory classification pathways | Does not classify a specific accessory |
| FDA guidance on section 3060 changes, 27 September 2019 | Guidance | Explains the exclusions in 520(o)(1)(A) to (D) and states that alerts and prioritization involve analysis | Does not address the 2022 CDS reading of 520(o)(1)(E) |
| MDR 2017/745, Articles 2 and 52 and Annex VIII | Binding EU regulation | Defines medical devices, accessories, intended purpose, and classification | Does not contain a clinical decision support exclusion |
| IVDR 2017/746, Articles 2 and 47 and Annex VIII | Binding EU regulation | Defines in vitro diagnostic medical devices and their classes | Does not apply to software that does not provide IVD information |
| MDCG 2019-11 Rev.1, 17 June 2025 | EU guidance | Sets the Figure 1 and Figure 2 qualification steps and the module boundary guidance | Is not binding on manufacturers, notified bodies, or courts |
| IMDRF N10:2013 | International guidance | Defines SaMD and SiMD vocabulary | Is not law in any jurisdiction |
| ISO 13485:2016 | Voluntary standard | Sets quality management system requirements once a product is a device | Does not establish device status |
| IEC 62304:2006+A1:2015 | Voluntary standard | Sets software lifecycle processes by software safety class | Does not set the regulatory class |

The United States analysis begins with section 201(h). The definition covers an instrument, apparatus, implement, machine,
contrivance, implant, in vitro reagent, or other similar or related article, including any component, part, or accessory,
intended for use in the diagnosis of disease or other conditions, or in the cure, mitigation, treatment, or prevention of
disease, or intended to affect the structure or any function of the body, and which does not achieve its primary intended
purposes through chemical action and is not dependent on being metabolized. The closing sentence excludes software
functions excluded under section 520(o). The definition is functional: the same code can be inside or outside it depending
on the claim.

Section 520(o) was added by section 3060 of the 21st Century Cures Act, enacted in December 2016. Paragraph (1) excludes
five categories of software function, and paragraph (2) addresses multi-function products. The categories are:
administrative support of a health care facility; healthy lifestyle functions unrelated to disease; electronic patient
records meeting three conditions; transfer, storage, format conversion, or display of clinical laboratory or device data
without interpretation; and clinical decision support meeting four conditions. Paragraph (3) preserves FDA's authority to
act by Federal Register order when a non-device function would pose a serious risk of adverse health consequences, and
paragraph (4) preserves other authorities. The exclusion is narrow and function-specific, which is why the analysis in
section 5 is function by function.

FDA's CDS guidance, finalized on 28 September 2022, is the operative interpretation. It explains that the input
restriction is absolute for image, IVD signal, and acquisition pattern data; that "medical information" includes
laboratory results, guidelines, and peer-reviewed literature; that recommendations to patients or caregivers fall outside
the exclusion; and that a specific directive, a time-critical output, or an output that replaces clinician judgment is not
"support." It also lists the information a developer should provide so that a clinician can independently review the
basis: the intended use and user, the required inputs and their quality requirements, a plain-language description of the
algorithm and its validation, and the patient-specific facts behind the output. Note on currency: at access on 25 October
2026, FDA's guidance page displayed an issue date of January 2026 and a content-current-as-of date of 29 January 2026.
This file teaches the 2022 criteria because they are the ones the course materials analyse; confirm on the page which
version is current before relying on a determination.

The device software policy of 28 September 2022 is a different document with a different job. It describes which device
software functions FDA intends to oversee and which are subject to enforcement discretion, including certain mobile
applications and software that helps patients self-manage a condition without providing a specific treatment
recommendation. Enforcement discretion is not exclusion. A product subject to discretion is still a device; the agency has
simply said it will not enforce certain requirements against it. The distinction matters at the moment a hospital's
quality department asks for a device listing, a recall notice, or a cybersecurity bill of materials.

The European analysis begins with MDR Article 2(1), which defines a medical device by intended purpose and expressly names
software as an article that can be a device. The listed purposes are diagnosis, prevention, monitoring, prediction,
prognosis, treatment, or alleviation of disease; diagnosis, monitoring, treatment, alleviation of, or compensation for
injury or disability; investigation, replacement, or modification of anatomy or of a physiological or pathological process
or state; and providing information by means of in vitro examination of specimens. The words "prediction" and "prognosis"
appear in the European definition and not in the United States definition, which is one reason a pure forecasting product
can be an easier fit in the EU than in the US. MDR Article 2(2) defines an accessory as an article that is not itself a
medical device but is intended to be used with one or several particular devices to enable them to be used in accordance
with their intended purpose, or to specifically and directly assist their medical functionality. Article 2(12) defines
intended purpose by reference to the label, instructions, promotional or sales material, and the clinical evaluation.

MDCG 2019-11 Rev.1, published on 17 June 2025, is the most detailed European guidance on software qualification. Its
Figure 1 runs five decision steps: whether the artifact is software; whether it is an Annex XVI product, an accessory, or
software driving or influencing a device; whether it acts on data beyond storage, archival, communication, simple search,
and lossless compression; whether the action benefits individual patients; and whether it has a medical purpose under
Article 2(1) or 2(2). Its Figure 2 runs three steps to separate MDR software from IVDR software, including a qualitative
weighting when the data sources are genuinely mixed. The guidance also states that the risk of harm is not a criterion for
qualification, and that the term "standalone software" is no longer used because qualification depends on intended purpose
regardless of where the code runs. The same document explains the module concept: where a product contains modules subject
to different legislation, the manufacturer must delineate the boundaries and identify which modules fall under the MDR or
IVDR.

Classification is a separate step and lives in Annex VIII. Rule 11 covers software used for decision support or
physiological monitoring. Implementing rule 3.3 provides that software which drives or influences the use of a device
falls within the same class as that device, and that software independent of any other device is classified in its own
right. Implementing rule 3.5 requires the strictest class when several rules apply. None of this is a qualification test,
and applying rule 11 to software that is out of scope produces a class for a product that is not regulated.

Data sources matter as much as legal sources. The FDA product classification database supplies product codes, regulations,
and classes. The 510(k) and De Novo databases supply cleared and authorized devices with their indications for use. The
FDA artificial intelligence-enabled device list supplies a view of authorized AI-enabled devices and is not exhaustive.
The Global Unique Device Identification Database and, in the EU, EUDAMED supply registration and traceability data. A memo
that cites a conference slide instead of a database record is not reproducible, and a memo that cites a database record
without an access date is not verifiable.

## 9. Industry practice and case evidence

Public evidence about device-status determinations is thin by construction. A company that decides its product is not a
device does not publish the memo. A company that decides it is a device publishes a submission summary, which describes
the claim it settled on rather than the claims it abandoned. The cases below separate what is documented from what is
inference.

**IDx-DR (DEN180001).** FDA granted De Novo authorization for an autonomous system that analyzes retinal images to detect
diabetic retinopathy, with a labeling that specified the user, the population, and the follow-up pathway. The documented
fact is that an autonomous detection claim was authorized at Class II with a narrow intended use. The inference, not
stated by FDA, is that the narrowness of the claim did as much work as the algorithm: the product was authorized for a
specific image type, a specific population, and a specific action. For this topic, the relevant lesson is that the claim
was narrow enough to be testable, which is the same discipline the 520(o)(1)(E) analysis demands.

**Apple ECG (DEN180044).** FDA granted De Novo authorization for an electrocardiogram app and an irregular-rhythm
notification feature, with labeling describing a screening use rather than a diagnostic replacement. The documented fact
is that a consumer product with a medical claim was regulated as a device. The inference is that consumer distribution did
not lower the evidentiary bar; the labeling did the work of limiting the claim. The product also illustrates the input
restriction in section 520(o)(1)(E): an electrocardiogram trace is a signal from a signal acquisition system, so the CDS
exclusion was never available regardless of how the output was described.

**Viz.ai large-vessel occlusion triage.** Public materials describe software that analyzes computed tomography angiography
images and alerts care teams to a suspected large-vessel occlusion. The documented fact is that the product is marketed as
a workflow intervention with a time-to-treatment value proposition. The inference is that a triage product's regulatory
case can rest on the action it triggers rather than on stand-alone accuracy. That is also why the time-critical branch of
the 2022 CDS guidance matters: a product whose value depends on acting within minutes is making a claim that the
exclusion's "support" reading does not fit. Current clearance numbers and indications should be read from the FDA database
rather than from a secondary summary.

**The Epic sepsis model.** Wong and colleagues published an external validation of a widely deployed sepsis prediction
model in JAMA Internal Medicine in 2021, reporting lower discrimination in the external health system than the vendor had
reported and a high alert volume at the deployed threshold. The documented finding is about performance, not about
regulatory status. The inference relevant here is that a product can be commercially available, integrated into a major
electronic health record, and still fail to perform in a new setting, which is the same failure that sinks a submission
when the external site is part of the claimed population.

**The 2022 CDS guidance as a market event.** The guidance's publication changed the status of product categories that had
been marketed as CDS under the looser reading. The documented fact is the guidance's content and date. The inference,
widely reported in industry commentary but not quantified in any public dataset, is that a number of products required
claim changes, additional evidence, or removal of features that supplied image or signal inputs. The practical lesson for
a team is that a determination memo written before September 2022 should be re-run, not amended.

**European practice: the module boundary.** MDCG 2019-11 Rev.1 devotes expanded guidance to modules, requiring
manufacturers to identify which modules fall under the MDR or IVDR and which fall under other legislation such as the
European Health Data Space Regulation. The documented fact is the guidance text. The inference from notified body practice
is that module documentation is now a standard technical-documentation question, and that a product which claims a single
regulatory status across an architecture with distinct functions will be asked to justify the boundary.

**Industry practice: the status workshop.** Mature organizations run a structured session before design freeze with
product, engineering, clinical, quality, and regulatory in the room, and they produce a per-function disposition table
with a confidence column. Startups more often run the analysis backwards: they build the product, then ask regulatory to
find an exclusion. The public record does not quantify the failure rate of either approach. The inference from warning
letters, refusal-to-accept decisions, and diligence findings is that claim-evidence misalignment at the gate is common and
expensive.

**Industry practice: the two-memo habit.** Companies that manage this well keep two documents: an internal determination
memo that records the reasoning, including the unresolved rows, and an external statement of intended use that records
only what the company will defend. The internal memo is privileged where possible and is not shared with customers. The
external statement is short, versioned, and reviewed whenever a sales deck changes. The separation prevents the most
common failure in this topic, which is a marketing sentence becoming the operative intended use because nobody compared
the two documents.

**What the cases do not prove.** None of these cases establishes that a particular product is or is not a device. IDx-DR
does not prove that autonomous screening is always Class II. Apple's authorization does not prove that a wearable with a
health feature is always a device. Viz.ai does not prove that a triage claim is always available. The transferable lesson
is the method: write the function, audit the claim surface, apply the gate in order, record the disposition, and test the
weakest row.

## 10. Failure modes and red flags

Gate errors rarely look like errors at the time. They look like a confident slide, a product name that implies a benefit,
or a decision made in a design review by someone with authority and no source. The table lists the failures that most
often survive until a reviewer, a customer's quality auditor, or a regulator finds them.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Exclusion by assertion | A slide says "CDS, therefore not a device" with no condition-by-condition analysis | The exclusion is treated as a category label rather than a four-part test | Unregulated device on the market | Require a written condition table with a source for each row |
| Exclusion by product name | The product is called a "wellness" or "insight" tool | Name is evidence of intent but not controlling; the claim surface decides | Status contradicted by the interface and deck | Audit the claim surface, not the name |
| Interface ignored | Default-on order button, red severity badge, hard stop in the workflow | Design is objective intent under 21 CFR 801.4 | Claim broader than the label and the evidence | Review screens as labeling artifacts |
| Signal input overlooked | Model ingests a waveform, a continuous glucose trace, or raw assay data | Condition 1 is a hard stop regardless of explainability | CDS route closed; device status attaches | Classify every input by data type before drafting the claim |
| Patient-facing recommendation | The same score is shown to patients "for transparency" | 520(o)(1)(E) applies only to recommendations to professionals | Second device function inside a non-device product | Split the function or remove the patient view |
| Single-product analysis | One disposition covers a product with six outputs | 520(o)(2) requires function-level treatment | Either over- or under-regulation of parts of the product | Build the function inventory first |
| Accessory inherits host class | App assumed covered because the sensor is cleared | Accessory status and class are determined in their own right | Missing submission or wrong class | Analyse the accessory against its own intended use |
| SiMD treated as SaMD | Embedded control software analysed with standalone assumptions | Hardware and software failures interact | Incomplete hazard analysis and lifecycle evidence | Define the system boundary and interfaces explicitly |
| US exclusion assumed global | EU launch plan written from the US memo | The EU has no CDS exclusion | EU technical documentation missing | One row per jurisdiction in every memo |
| Discretion treated as exclusion | "FDA does not enforce this" becomes "this is not a device" | Enforcement discretion leaves device status intact | Missing registration, listing, and complaint handling | Record the legal basis, not the enforcement posture |
| Stale determination | Memo cites a guidance without a version date | Guidance is revised and criteria change | Determination rests on superseded text | Version every source and set a review date |
| Undocumented multi-function interaction | An excluded summarizer feeds a device risk score | The device claim covers the interaction | Risk file incomplete; exclusion does not protect combined behaviour | Map data dependencies between functions |
| IVD boundary misread | Imaging software classified through laboratory logic, or the reverse | Device and IVD regimes have different regulations | Wrong predicate search and wrong evidence standard | Apply the specimen test and the Figure 2 weighting |
| Time-critical output missed | The output's value depends on acting within minutes | FDA's reading of "support" excludes time-critical outputs | CDS exclusion unavailable for a triage product | Test the workflow timing, not the label wording |
| Post-launch claim drift | New website copy or a new feature appears without a memo update | The claim surface changes the intended use | Status changes silently | Route marketing and release notes through the memo owner |

A red flag is not proof of a violation. It is a signal that the memo is missing a fact. The right response is to add the
fact and re-run the step, not to argue that the red flag does not apply. A team that can explain why a specific red flag
is not relevant is in a stronger position than a team that never looked.

## 11. Assumptions and limits

Every determination rests on assumptions. A memo should state them, explain why each is needed, describe what would break
it, and name the test that would detect the break. The table is a template for that ledger.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The claim surface is fully captured | Objective intent is assembled from every register | An uncollected sales deck, demo, or contract | Ask sales and customer success for everything they show a prospect |
| Intended use is separable from capability | The gate tests intent, not what the code could do | A feature that is unused but marketed as available | Compare the roadmap and the enablement flags with the label |
| The function inventory is complete | Each function carries its own status | A hidden output, an export, or an API response | Trace every screen and every data export to a function |
| Reported laboratory values are medical information | Condition 1 turns on the input type | The product ingests raw instrument output or waveforms | Inspect the data contract, not the user interface |
| The user role is stable | Conditions 3 and 4 depend on who the user is | A clinician view extended to patients or caregivers | Check role-based access and the release plan |
| Workflow timing matches the design | Time-critical outputs fail condition 3 | Real-world use is faster or slower than the prototype | Time the task in the intended setting with real users |
| US and EU claims can be separated | The regimes have different exclusions | One global label and one global deck | Write two intended-purpose statements and diff them |
| The cited guidance is current | Criteria change between versions | A revision between drafting and filing | Record the access date and recheck before submission |
| The model version is the one described | Evidence attaches to a version | A retrain or threshold change after validation | Configuration control and release manifest |
| The customer's use matches the label | Distribution circumstances are evidence of intent | A customer configures the product for another purpose | Contract terms, implementation review, and post-market surveillance |

The limits of this topic follow from its scope. It does not produce a legal opinion, and it does not cover every
product-specific classification regulation. It does not resolve borderline cases, and it does not tell you whether a
regulator will agree. It does not cover the United Kingdom, Canada, Australia, or Japan, each of which has its own
treatment of software and its own view of decision support. It does not address the European Union Artificial Intelligence
Act, which applies alongside the MDR and IVDR and has its own risk classification; a product can be a Class IIb device and
a high-risk AI system, and the two analyses are separate. It does not cover laboratory-developed tests, which in the
United States have had a volatile regulatory history and are addressed elsewhere in the module.

A determination memo should end with a confidence statement rather than a verdict. "High confidence that the risk function
is a device in both jurisdictions; medium confidence that the audit function is excluded, because the manager-facing
report includes a patient-level ranking that may be analysis; low confidence on the EU classification because the harm
characterization depends on whether the site treats AKI requiring dialysis as serious or irreversible; we will seek a
notified body opinion before locking the clinical evaluation plan." That statement tells the team where to spend the next
week, which a binary conclusion does not.

## 12. Exercises

The exercises move from recall to application to critique to design. Each states a deliverable, an acceptance criterion,
and a time estimate. Use the current legal texts and the current databases, not this file alone.

### Exercise 1 — Function inventory and disposition table (application, 35 minutes)

Take a product description with at least four user-facing outputs, either your own or a public one. Produce a table with
one row per output and columns for output, user, action, input data types, proposed disposition, and confidence.
Deliverable: the table plus a two-sentence note on the row you are least sure about. Acceptance criteria: every row names
a user and an action; input data types are classified as information, image, signal, or pattern; at least one row is
marked unresolved rather than forced.

### Exercise 2 — The four-condition test (application, 40 minutes)

Choose three functions: a risk score shown to nurses, an image-based triage alert, and a patient-facing symptom checker.
For each, write the four conditions of section 520(o)(1)(E) as rows and record pass, fail, or unresolved with the fact
that decides it. Deliverable: a 3 by 4 matrix with citations. Acceptance criteria: the image-based function fails
condition 1 and the reason is stated as an input fact, not an accuracy concern; the patient-facing function fails
condition 3 on user role; no row cites a vendor claim as evidence.

### Exercise 3 — Claim-surface audit and rewrite (critique, 45 minutes)

Take a public product page for a clinical software product. Extract every sentence that names a disease, a population, an
action, or a time window. Group the sentences by register (label, interface, sales, contract, integration). Then rewrite
the product description as one intended-use statement that a reviewer could defend. Deliverable: the sentence table plus a
six-element intended-use statement. Acceptance criteria: the statement names purpose, user, population, setting, output,
and at least one limitation; any sentence you removed is listed with the reason.

### Exercise 4 — Interface as intent (design, 40 minutes)

Sketch a screen that displays a risk band to a clinician. Produce two versions: one in which the output is supportable as
non-device CDS, and one in which it is a specific directive. Annotate the design differences. Deliverable: two wireframes
with annotations. Acceptance criteria: the differences are described in regulatory terms (directive, time-critical, basis
reviewable), not only in design terms; the supportable version exposes the inputs and the validation population.

### Exercise 5 — EU qualification under MDCG 2019-11 (application, 40 minutes)

Apply Figure 1 and Figure 2 of MDCG 2019-11 Rev.1 to the three functions from Exercise 2. Record the step at which each
enters or exits, and for any function that enters, state whether it is MDR or IVDR software and why. Deliverable: a
step-by-step table with the guidance citation for each step. Acceptance criteria: the risk of harm is not used as a
qualification criterion; the mixed-data case is resolved with a written weighting or explicitly marked unresolved; the
difference from the US result is stated.

### Exercise 6 — Hardware boundary and accessory analysis (critique, 35 minutes)

Take a product sold with a sensor, such as an app that displays and interprets a wearable's signal. Determine whether it
is standalone software, software in a device, an accessory, or a component, in both jurisdictions. Identify the two facts
that would change the answer. Deliverable: a comparison table with a conclusion per jurisdiction and the changing facts.
Acceptance criteria: the analysis does not assume the host's clearance covers the add-on; the EU accessory definition in
Article 2(2) is quoted; the US analysis distinguishes an accessory from a component using 21 CFR 820.3.

### Exercise 7 — Determination memo (design, 60 minutes)

Write a two-page determination memo for a fictional product of your choice. Include the function inventory, the
disposition per function per jurisdiction, the governing provision for each, the confidence statement, the open questions,
and the re-review triggers. Deliverable: the memo with a source table that labels each source as statute, regulation,
guidance, standard, or convention. Acceptance criteria: every disposition has a citation and a date; at least two rows are
marked unresolved with a named owner; the memo states what external advice is needed and when.

### Exercise 8 — Adversarial review (critique, 40 minutes)

Exchange a memo with a peer. Attack the weakest row: the input classification, the user role, the directive test, the
reviewability claim, or the EU qualification. Write five questions a reviewer would ask, and identify the one the memo
cannot answer. Deliverable: five questions and a one-paragraph assessment. Acceptance criteria: questions are specific to
the memo; at least one question identifies a missing fact rather than a missing document; the assessment separates a
documentation gap from a substantive gap.

## 13. Self-check answers

**Q1. Which provision excludes software functions from the United States device definition, and what does it not do?**
Section 520(o) of the FD&C Act, added by section 3060 of the 21st Century Cures Act in December 2016, excludes five
categories of software function. It does not create a general exemption for clinical software, it does not exempt a
product as a whole when only some functions qualify, and it does not prevent FDA from assessing an excluded function's
impact on a device function in the same product under section 520(o)(2).

**Q2. What is the correct order of the United States gate, and why does the order matter?** First the medical purpose
under section 201(h), then the five exclusions in section 520(o)(1), then the multi-function rule in section 520(o)(2),
then classification. The order matters because starting from the exclusion list encourages reverse-engineering a purpose
to fit a category. It also matters because a function with no medical purpose exits at step one, and no amount of
exclusion analysis improves a determination that began in the wrong place.

**Q3. State the four conditions in section 520(o)(1)(E) and give the arithmetic test that accompanies a risk output.** The
conditions are: the function does not acquire, process, or analyze a medical image, a signal from an in vitro diagnostic
device, or a pattern or signal from a signal acquisition system; it displays, analyzes, or prints medical information; it
supports or provides recommendations to a health care professional about prevention, diagnosis, or treatment; and it
enables the professional to independently review the basis so as not to rely primarily on the output. The arithmetic test
is the positive predictive value at the intended prevalence. At sensitivity 0.82, specificity 0.79, and prevalence 6.4
percent, PPV is 0.0525 divided by 0.2490, or 0.211, so roughly four of five flagged patients will not have the outcome and
the label must say so.

**Q4. Why is a specific directive not "support"?** The statute requires that the function support or provide
recommendations to a professional and enable independent review. FDA's 2022 guidance reads a specific directive, a
time-critical output, and an output intended to replace clinician judgment as outside "support," because in each case the
professional is not exercising the judgment the exclusion assumes. The test is what the output tells the user to do, not
which button the user presses.

**Q5. What are the five registers of the claim surface, and which is most often missed?** Label and instructions for use;
user interface; sales and marketing; contracts and pricing; and integrations, meaning which system calls the function and
when. The interface is most often missed, because teams treat it as a design artifact rather than as evidence of intent,
and because defaults and severity colours are claims that nobody wrote in a document.

**Q6. Walk through the Figure 1 decision steps in MDCG 2019-11 Rev.1.** Step 1 asks whether the artifact is software
within the guidance's definition. Step 2 asks whether it is an Annex XVI product, an accessory to a medical device, or
software driving or influencing a device; if so, it is treated as part of that device or as an accessory in its own right.
Step 3 asks whether it acts on data beyond storage, archival, communication, simple search, and lossless compression. Step
4 asks whether the action benefits individual patients. Step 5 asks whether it has a medical purpose under MDR Article
2(1) or IVDR Article 2(2). A function that passes all five is medical device software.

**Q7. What does "the risk of harm is not a criterion for qualification" mean in practice?** It means that a low-risk
function does not escape the MDR by being low risk, and a high-risk function does not enter it by being high risk.
Qualification depends on intended purpose. Harm enters at classification, where rule 11 uses the seriousness of possible
harm to choose between classes. Teams that reason from harm at the qualification stage usually produce the right answer
for the wrong reason, and then misapply it in a jurisdiction where the logic differs.

**Q8. Distinguish standalone software, software in a device, an accessory, and a component.** Standalone software performs
a medical purpose without being part of a hardware device. Software in a device is part of, drives, or controls a hardware
medical device. An accessory is an article that is not itself a device but is intended to be used with a specific device
to enable its intended purpose or directly assist its medical function, and it is assessed in its own right. A component
is a part or module supplied for incorporation into a finished device. The distinctions change the submission route, the
lifecycle evidence, and who owns the file.

**Q9. What belongs in a determination memo's confidence statement?** The disposition, the confidence level, the specific
step that carries the uncertainty, the fact that would resolve it, and the date by which it must be resolved. A useful
statement names the weakest row rather than summarizing the whole memo, because the weakest row is what a reviewer will
attack.

**Q10. What is the strongest counterargument to a non-device CDS determination, and how do you answer it?** The strongest
counterargument is that the function's design, not its label, makes it a directive: a default-on action, a hard stop, or
an alert that fires inside a time window. The answer is evidence, not assertion: show the interface, the workflow timing,
the user's ability to decline without friction, and the usability study that demonstrates independent review. If the
evidence does not exist, the determination is unresolved, and the honest memo says so.

## 14. Interview questions

**1. When is clinical software not a medical device in the United States?** A strong answer names section 520(o), lists
the five categories, and explains that the CDS category has four conditions that must all hold, including the input
restriction and the independent-review requirement. A weak answer says "when a clinician makes the final decision," which
is the folk rule the 2022 guidance replaced.

**2. Why can an image-based algorithm never qualify as non-device CDS?** A strong answer cites the closing text of section
520(o)(1)(E), which excludes functions intended to acquire, process, or analyze a medical image, an IVD signal, or an
acquisition pattern, and notes that this is an input test independent of accuracy or explainability. A weak answer talks
about risk or about how good the model is.

**3. What is the difference between exclusion and enforcement discretion?** A strong answer says an excluded function is
outside the device definition, while a function subject to enforcement discretion remains a device that FDA has chosen not
to prioritize for certain requirements; it also notes the practical consequences for registration, listing, and complaint
handling. A weak answer treats the two as synonyms.

**4. How would you determine the status of a documentation tool that also suggests diagnoses?** A strong answer splits the
functions, applies the gate to each, notes that the summarizer may fall under the record or display categories only if it
does not interpret, and treats the diagnostic suggestion as a device function under the multi-function rule. A weak answer
gives one status for the product.

**5. What does "independently review the basis" require in an interface?** A strong answer names the inputs and their
values, a plain-language description of the logic, the validation population and performance, and the patient-specific
facts, and it connects that to a usability test showing users can actually do it. A weak answer points to a "Why?" link.

**6. How does the European gate differ from the United States gate?** A strong answer says the EU has no CDS exclusion,
that qualification depends on intended purpose under MDR Article 2(1) or IVDR Article 2(2), and that MDCG 2019-11 Rev.1
sets out the decision steps including the storage and compression carve-out. A weak answer assumes the US exclusion
travels.

**7. What is the difference between qualification and classification in the EU?** A strong answer says qualification
decides whether the product is in scope at all, and classification assigns a class within a regime that already applies. A
weak answer uses the words interchangeably.

**8. When is software an accessory rather than an independent device?** A strong answer quotes MDR Article 2(2), explains
enabling or directly assisting a specific device's intended purpose, notes that accessories are classified in their own
right, and contrasts the US accessory framework. A weak answer says any app that works with a device is an accessory.

**9. What is the most common error you have seen in device-status memos?** A strong answer names a specific mechanism,
such as analysing the product instead of the function, or relying on the label while ignoring the interface, and describes
how the error would be detected. A weak answer gives a generic warning about regulation being complex.

**10. A product team says the model is explainable, so the CDS exclusion applies. What do you say?** A strong answer
separates explainability from the four conditions: explainability supports condition 4 but does nothing for condition 1,
and a specific directive fails condition 3 however explainable it is. A weak answer accepts the premise that
explainability is the test.

## 15. Key terms

| Term | One-line meaning |
|---|---|
| Accessory | Article enabling or directly assisting a specific device's intended purpose, classified in its own right |
| Claim surface | The set of registers (label, interface, sales, contract, integration) that supply objective intent |
| Classification | Assignment of a risk class within a regime that already applies |
| Clinical decision support (CDS) | Software that supports a professional's clinical decision; excluded in the US only if four conditions hold |
| Component | Part or module supplied for incorporation into a finished device |
| Combination product | Product combining drug, device, or biological constituents, assigned by primary mode of action |
| Enforcement discretion | A device FDA has said it will not enforce certain requirements against |
| Finished device | Device packaged and labeled for distribution |
| Independent review of the basis | The professional's ability to see inputs, logic, validation, and patient facts behind an output |
| Intended purpose (EU) | The use stated in the label, instructions, promotional material, and clinical evaluation |
| Intended use (US) | The objective intent of the persons responsible for labeling |
| IVD | In vitro diagnostic medical device |
| IVDMDSW | Medical device software providing information within the IVD definition |
| MDR | EU Medical Device Regulation (EU) 2017/745 |
| MDCG | Medical Device Coordination Group, the EU guidance body |
| MDSW | Medical device software under the MDR or IVDR |
| Medical information | Patient health information such as results, diagnoses, medications, or guidelines |
| Multi-function product | Product containing at least one excluded function and at least one device function |
| Non-device CDS | A function meeting all four conditions of 520(o)(1)(E) |
| Objective intent | Intent shown by expressions, design, and circumstances of distribution |
| Parent device | A finished device whose performance an accessory supports or augments |
| Pattern | Multiple, sequential, or repeated measurements considered together |
| Primary mode of action | The mode of action expected to provide the most important therapeutic effect |
| Qualification (EU) | Determination of whether software falls within the MDR or IVDR at all |
| SaMD | Software as a Medical Device |
| SiMD | Software in a Medical Device |
| Signal | A time-varying measurement from an acquisition system |
| Specific directive | An output naming the action to take rather than supporting the user's choice |
| Substantial equivalence | The 510(k) standard comparing intended use and technological characteristics |
| Time-critical | An output whose value depends on acting faster than independent review allows |

## 16. Sources

All URLs were accessed on 25 October 2026. Dates are the publication or revision dates shown by the source at access. Guidance is revised; verify the current version before relying on a determination.

1. United States Code, 21 U.S.C. 321(h), definition of device, including the exclusion of software functions under 21 U.S.C. 360j(o). https://www.law.cornell.edu/uscode/text/21/321
2. United States Code, 21 U.S.C. 360j(o), software functions excluded from the device definition, including the multi-function rule. https://www.law.cornell.edu/uscode/text/21/360j
3. Code of Federal Regulations, 21 CFR 801.4, meaning of intended uses. https://www.law.cornell.edu/cfr/text/21/801.4
4. Code of Federal Regulations, 21 CFR Part 3, product jurisdiction and combination products. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-3
5. Code of Federal Regulations, 21 CFR 820.3, definitions including component and finished device. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820/subpart-A/section-820.3
6. US Food and Drug Administration. "Clinical Decision Support Software." Final guidance, 28 September 2022; the guidance page displayed an issue date of January 2026 and content current as of 29 January 2026 at access. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
7. US Food and Drug Administration. "Policy for Device Software Functions and Mobile Medical Applications." Guidance, 28 September 2022. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/policy-device-software-functions-and-mobile-medical-applications
8. US Food and Drug Administration. "Medical Device Accessories: Describing Accessories and Classification Pathways." Guidance, 20 December 2017. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/medical-device-accessories-describing-accessories-and-classification-pathways
9. US Food and Drug Administration. "Changes to Existing Medical Software Policies Resulting from Section 3060 of the 21st Century Cures Act." Guidance, 27 September 2019. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/changes-existing-medical-software-policies-resulting-section-3060-21st-century-cures-act
10. US Food and Drug Administration. "Digital Health Policy Navigator." https://www.fda.gov/medical-devices/digital-health-center-excellence/digital-health-policy-navigator
11. US Food and Drug Administration. "Product Classification" database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPCD/classification.cfm
12. US Food and Drug Administration. "510(k) Premarket Notification" database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm
13. US Food and Drug Administration. "De Novo Classification Request: DEN180001." https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
14. US Food and Drug Administration. "De Novo Classification Request: DEN180044." https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf
15. European Parliament and Council. Regulation (EU) 2017/745 on medical devices (MDR). https://eur-lex.europa.eu/eli/reg/2017/745/oj
16. European Parliament and Council. Regulation (EU) 2017/746 on in vitro diagnostic medical devices (IVDR). https://eur-lex.europa.eu/eli/reg/2017/746/oj
17. Medical Device Coordination Group. MDCG 2019-11 Rev.1, "Guidance on Qualification and Classification of Software in Regulation (EU) 2017/745 and Regulation (EU) 2017/746," 17 June 2025.
https://health.ec.europa.eu/system/files/2025-06/mdcg_2019_11_en.pdf
18. European Commission, Directorate-General for Health and Food Safety. "Update MDCG 2019-11 rev.1: Qualification and classification of software," 17 June 2025. https://health.ec.europa.eu/latest-updates/update-mdcg-2019-11-rev1-qualification-and-classification-software-regulation-eu-2017745-and-2025-06-17_en
19. International Medical Device Regulators Forum. "Software as a Medical Device: Key Definitions," IMDRF/SaMD WG/N10FINAL:2013. https://www.imdrf.org/documents/software-medical-device-key-definitions
20. Wong, A., et al. "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients." JAMA Internal Medicine, 2021. https://doi.org/10.1001/jamainternmed.2021.2626
21. International Organization for Standardization. ISO 13485:2016, Medical devices: Quality management systems. https://www.iso.org/standard/59752.html
22. International Electrotechnical Commission. IEC 62304:2006+A1:2015, Medical device software: Software life cycle processes. https://webstore.iec.ch/en/publication/22794

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_definitions-and-classification.md`](01_definitions-and-classification.md)
- Next topic: [`03_fda-pathways.md`](03_fda-pathways.md)
- Related topic: [`08_eu-mdr-and-ivdr.md`](08_eu-mdr-and-ivdr.md)
- Related topic: [`09_the-evidence-package.md`](09_the-evidence-package.md)
- Survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 14.1 regulatory triage: [`../../labs/`](../../labs/)
- Lab 14.2 intended-use statements: [`../../labs/`](../../labs/)

The module index is the authoritative map. If a sibling file is not yet present, the index marks it as planned; do not
infer content from the link text.
