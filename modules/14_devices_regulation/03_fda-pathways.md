# FDA Pathways

> Module 14 — Medical Devices, SaMD, and Regulation
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: 01, 02
> Version 2026.3

## 1. Why this topic exists

A device program makes four commitments it cannot easily reverse: the intended use, the
risk class, the pathway, and the evidence standard. Topics 01 and 02 cover the first two.
This topic covers the third, which converts the first two into a submission type, a review
clock, a public record, and a set of rules for changing the product after launch. The
pathway decision is where regulatory strategy stops being a document and starts costing
money.

The decision is easy to get wrong because it looks like a preference. Teams talk about
"going 510(k)" or "doing a De Novo" as if the choice were a commercial tactic. It is not.
The pathway is determined by two facts about the product and the market: whether a legally
marketed device with the same intended use exists, and whether general and special controls
can provide reasonable assurance of safety and effectiveness. A team that picks a pathway
first and writes a claim to fit it has reversed the causation, and the reversal usually
surfaces as a refusal to accept, a not-substantially-equivalent decision, or a warning
letter two years later.

The cost asymmetry is structural rather than a matter of fees. A 510(k) is a comparison: it
asks whether the new device is as safe and effective as a device that is already legally
marketed. A De Novo request is a classification: it asks the Food and Drug Administration
(FDA) to create a new device type and to write the special controls for it. A premarket
approval (PMA) is an independent demonstration: it requires valid scientific evidence of
safety and effectiveness, a manufacturing review, and usually a prospective clinical study.
Each step up the ladder adds evidence, time, and a different kind of reviewer. The difference
between a 510(k) and a PMA is not a larger version of the same document; it is a different
burden of proof.

The timing language around pathways is also widely misread. FDA publishes review goals, not
deadlines. The 510(k) goal is 90 FDA days for 95 percent of submissions under the Medical
Device User Fee Amendments (MDUFA) V commitment for fiscal years 2023 through 2027. The
PMA statute sets 180 days from the date of filing. The De Novo goal is 150 FDA days for 70
percent of requests, not 90 days, and the module survey's table should be corrected on that
point. None of these numbers is a promise about calendar time. FDA days exclude periods
when the submission is on hold waiting for the sponsor, and a single additional-information
request can add months.

For a data scientist, the pathway decides what evidence the model must carry. A 510(k) may
be supported by bench testing, a retrospective reader study, and a comparison to a
predicate's published performance. A De Novo requires a standalone benefit-risk argument and
usually prospective data. A PMA requires the kind of clinical evidence that changes practice.
The algorithm can be identical in all three cases. What changes is the question the evidence
must answer, and the point at which the algorithm must be frozen.

The pathway also governs what happens after launch. A cleared 510(k) constrains changes: a
modification that could significantly affect safety or effectiveness, or a major change in
intended use, requires a new 510(k) under 21 CFR 807.81(a)(3). A PMA constrains changes more
tightly, through supplements. Since December 2022, the Federal Food, Drug, and Cosmetic Act
(FD&C Act) section 515C has allowed a predetermined change control plan (PCCP) to authorize
specified future changes in advance. A team that plans to retrain a model every quarter
without deciding which pathway governs those retraining events has not finished its
regulatory design.

Finally, the pathway is a communication object. Investors, partners, hospital procurement,
and auditors all ask a version of the same question: what is your regulatory route and what
is the evidence behind it? A team that can name the classification regulation, the product
code, the predicate, the differences that raise questions, and the plan to answer them is
ready to have that conversation. A team that says "we think it is probably a 510(k)" is not.

This topic gives you a procedure for making the decision, a set of decision tables for the
branches, a worked case with the arithmetic, and executable code that scores predicate fit.
It does not replace regulatory counsel, and it does not predict what FDA will decide. It
gives you the structure to ask for the right meeting, prepare the right evidence, and avoid
the errors that are expensive to fix after a filing.

## 2. Learning outcomes

By the end of this topic, you can:

1. Explain the legal difference between a 510(k) clearance, a De Novo grant, and a PMA
     approval, and name the statutory section and regulation behind each.
2. Given a product description, determine whether the device needs a 510(k), a De Novo, a
     PMA, or no premarket submission because its classification regulation exempts it.
3. Build a predicate matrix from the FDA 510(k) and De Novo databases, including the
     intended-use comparison, the technological comparison, and the "different questions of
     safety and effectiveness" analysis.
4. Decide when a pre-submission (Q-Sub), a 513(g) request, or a Request for Designation is
     the right way to resolve an open question, and when none of them is.
5. Compute the evidence consequence of a pathway choice, including prevalence-dependent
     predictive values and the event count needed to support a sensitivity claim.
6. Apply the post-clearance change rules in 21 CFR 807.81(a)(3) and FD&C Act section 515C
     to a planned model update.
7. Distinguish a binding requirement, an FDA guidance recommendation, and an industry
     convention in a pathway memo, with dates.
8. Identify the two or three findings in a validation package that would sink a submission
     and state the design change that would prevent them.

The outcomes map to the exercises and self-check items as follows.

| Outcome | Exercise | Self-check |
|---|---|---|
| 1 | Exercise 1 | Questions 1, 2 |
| 2 | Exercise 1, Exercise 4 | Questions 3, 4 |
| 3 | Exercise 2, Exercise 3 | Questions 5, 6 |
| 4 | Exercise 6 | Question 7 |
| 5 | Exercise 3 | Question 8 |
| 6 | Exercise 7 | Question 9 |
| 7 | Exercise 8 | Question 10 |
| 8 | Exercise 5, Exercise 8 | Question 11 |

## 3. The core mental model

A pathway is chosen by two questions asked in a fixed order. First: does a legally marketed
device with the same intended use exist? Second: can general controls, or general and
special controls, provide reasonable assurance of safety and effectiveness? The answers
place the product in one of four doors. The rest of the submission is detail.

```text
intended use + indications for use
                |
                v
   [1] classification regulation + product code
                |
        +-------+--------+
        |                |
     exempt          not exempt
        |                |
        v                v
 general controls   [2] class I / II / III
 registration,           |
 listing, labeling,      +--> class III --------> [4] PMA
 device reporting        |                          (valid scientific evidence,
                         |                           manufacturing, clinical)
                         +--> class I or II
                                  |
                                  v
                         [3] is there a legally marketed predicate
                             with the same intended use and no
                             different questions of safety and effectiveness?
                                  |
                          yes ----+---- no
                           |            |
                           v            v
                        510(k)       De Novo
                        (SE)         (new class, new controls)
```

Read the diagram from the top. The classification regulation, not the algorithm, assigns the
class and the default submission type. A regulation that says "exempt" removes the premarket
submission but not the general controls. A regulation that says "510(k)" establishes that a
predicate-based comparison is the expected route. A product that fits no regulation is
unclassified, and unclassified products go to De Novo or PMA depending on risk.

The second thing to notice is that the three doors answer different questions. A 510(k)
answers the substantial equivalence (SE) question: is it as safe and effective as something
already sold? A De Novo answers a different question: should this new kind of device be
allowed, and under what controls? A PMA answers a third: is it safe and effective on its own
evidence? The evidence package follows the question. A team that writes a PMA-style clinical
program for a 510(k) wastes money; a team that writes a 510(k)-style bench comparison for a
PMA fails.

The third thing is that the diagram is recursive. A granted De Novo creates a new
classification regulation and a new product code. The next product with the same intended
use can then use the newly created device as a predicate and file a 510(k). This is why the
De Novo route is strategically valuable beyond the first product, and why the 510(k)
database changes over time. Viz.ai's large-vessel occlusion (LVO) triage software is the
clean example: the original authorization was a De Novo in 2018, and later versions and
competitors moved through 510(k) clearances under the product code that the De Novo created.

The fourth thing is that "predicate" is a legal status, not a similarity judgment. A device
is a predicate if it is legally marketed: cleared through 510(k), granted through De Novo,
approved through PMA, or in commercial distribution before the 1976 amendments. A device
that FDA has removed from the market by order, or that a court has declared misbranded or
adulterated, is not a predicate under 21 CFR 807.100(b). Two products can look identical and
differ in predicate status. A product can look different and still be substantially
equivalent. The comparison is about intended use and technological characteristics, not
about appearance or brand.

The fifth thing is the difference between the three clocks. The statutory clock is set by
the FD&C Act and the regulations: 180 days from filing for a PMA, 120 days for a De Novo
under 21 CFR 860.240, and no fixed statutory deadline for a 510(k). The performance-goal
clock is set by MDUFA: 90 FDA days for 95 percent of 510(k)s, 150 FDA days for 70 percent of
De Novo requests. The calendar clock is what the program actually experiences, and it
includes acceptance review, holds, sponsor response time, and FDA workload. The three clocks
diverge, sometimes by a year. Plan against the calendar clock and report against the goal.

The sixth thing is that the pathway constrains the product after launch. A clearance is tied
to an intended use, a technology, and a label. Change any of them materially and the
clearance may no longer cover the product. The statute does not say "do not improve the
product"; it says that a change which could significantly affect safety or effectiveness, or
a major change in intended use, requires a new submission. A PCCP can pre-authorize a
defined set of changes, but only within the boundaries the plan describes. The pathway
decision therefore sets the cost of every future release.

A compact way to remember the model: claim fixes purpose, purpose fixes class, class and
predicate availability fix the door, the door fixes the evidence question, and the evidence
question fixes the change rules. If you can defend each arrow with a citation, you can run a
pathway discussion.

## 4. Definitions and vocabulary

The terms below recur in every pathway memo. The "common misuse" column records errors that
have produced real refusals, holds, and mis-filings.

| Term | Definition | Common misuse | Unit or source |
|---|---|---|---|
| Premarket notification, 510(k) | A submission demonstrating that a device is substantially equivalent to a legally marketed predicate, permitting commercial distribution | Calling it an approval, or assuming it certifies safety and effectiveness independent of the predicate | FD&C Act §510(k), 21 U.S.C. 360(k); 21 CFR 807 Subpart E |
| Substantial equivalence (SE) | Same intended use as the predicate, and either the same technological characteristics or different ones that do not raise different questions of safety and effectiveness and are supported by data | Treating SE as "similar enough" or as a similarity score | 21 CFR 807.100(b) |
| Not substantially equivalent (NSE) | FDA's decision that the device is not SE; it may not be marketed until cleared, granted De Novo, or approved | Reading NSE as a permanent rejection rather than a routing decision | 21 CFR 807.100(b)(2) |
| Predicate device | A legally marketed device to which equivalence is claimed | Choosing a predicate by product name, input modality, or vendor reputation | 21 CFR 807.92(a)(3); FDA 510(k) Program guidance, July 2014 |
| Primary predicate | The predicate whose intended use and technology drive the comparison when several are cited | Listing several predicates without saying which one controls each difference | FDA 510(k) Program guidance, July 2014 |
| Reference device | A device used to support performance testing or a scientific rationale but not relied on for SE | Calling a reference device a predicate and assuming its clearance covers the claim | FDA 510(k) Program guidance, July 2014 |
| Legally marketed | In commercial distribution under a cleared 510(k), granted De Novo, approved PMA, or pre-1976 status, and not removed by FDA order or judicial finding | Assuming a listed product is currently marketed, or that a discontinued product is unavailable as a predicate | 21 CFR 807.100(b) |
| Classification regulation | The 21 CFR section that describes a device type, assigns its class, and states whether a 510(k) is required or the type is exempt | Reading the regulation title instead of its full description | 21 CFR Parts 862–892 |
| Product code | FDA's three-letter identifier for a device type, linked to a regulation, class, and panel | Treating the code as a substitute for the regulation text | FDA Product Classification database |
| Class I / II / III | US risk classes tied to general controls, general and special controls, and premarket approval | Assigning class from an internal risk matrix rather than the regulation | FD&C Act §513(a)(1) |
| Exempt | A device type for which no premarket notification is required, while general controls still apply | Treating exempt as unregulated | 21 CFR 862–892, exemption columns |
| De Novo request | A request under §513(f)(2) to classify a novel device of low or moderate risk for which no legally marketed predicate exists | Using De Novo to avoid PMA for a high-risk device | FD&C Act §513(f)(2), 21 U.S.C. 360c(f)(2); 21 CFR Part 860 Subpart D |
| PMA | Premarket approval for a Class III device, based on valid scientific evidence and a manufacturing review | Treating a 510(k) clearance as permission for any claim | FD&C Act §515, 21 U.S.C. 360e; 21 CFR Part 814 |
| PMA supplement | A submission required to change an approved PMA, with types such as panel-track, 180-day, real-time, and 30-day notices | Assuming a manufacturing change never needs a supplement | 21 CFR 814.39 |
| HDE | Humanitarian device exemption for devices treating conditions affecting fewer than 8,000 individuals in the US per year, with limits on profit | Using HDE for a common condition | FD&C Act §520(m); 21 CFR 814 Subpart H |
| IDE | Investigational device exemption allowing an unapproved device to be used in a clinical study | Running a pivotal study without an approved IDE | 21 CFR Part 812 |
| Q-Sub / Pre-Sub | A voluntary pre-submission interaction with FDA for feedback on a planned submission | Expecting binding approval of a study design | FDA Q-Submission guidance, May 2025 |
| 513(g) request | A request for FDA's written view on the classification of a device and the requirements that apply | Using it to obtain a substantial-equivalence review | FD&C Act §513(g), 21 U.S.C. 360c(g) |
| Request for Designation (RFD) | A request assigning a combination product to a lead agency component | Filing a combination product without deciding the lead center | FD&C Act §563; 21 CFR Part 3 |
| eSTAR | FDA's interactive PDF template for electronic premarket submissions | Submitting a PDF assembly that fails technical screening | FDA eSTAR program page |
| Acceptance review | FDA's check that a submission contains the required elements before substantive review begins | Confusing acceptance with a substantive decision | 21 CFR 807.100(a)(1); FDA 510(k) process page |
| Refuse to accept (RTA) | A decision that a submission is incomplete and will not be reviewed until cured | Treating an RTA as a scientific rejection | FDA 510(k) process page |
| Additional information request | A request that places the submission on hold until the sponsor responds | Missing the response window and losing the filing slot | FDA 510(k) Submission Process page; 21 CFR 807.87(m) |
| FDA day | A day counted on the review clock, excluding days the submission is on hold for the sponsor | Reporting FDA days as elapsed calendar time | MDUFA V commitment letter, 2022 |
| MDUFA | The device user fee agreement that sets review performance goals | Treating a goal as a statutory deadline | MDUFA V, FY2023–2027 |
| PCCP | A predetermined change control plan describing modifications a device may undergo without a new submission | Using a PCCP to authorize open-ended learning | FD&C Act §515C (added by FDORA, 29 December 2022) |
| FDORA | The Food and Drug Omnibus Reform Act of 2022, which added §515C and other device provisions | Citing it for requirements it does not contain | Consolidated Appropriations Act, 2023, Division FF |
| Combination product | A product combining a drug, device, or biological product | Assuming a software-plus-drug product is always device-led | 21 CFR Part 3 |
| SaMD | Software as a Medical Device, performing a medical purpose without being part of a hardware device | Assuming SaMD always has a 510(k) route | IMDRF N10:2013; FDA SaMD page |

Three distinctions in the table deserve more than a row. The first is clearance versus
approval versus authorization. A 510(k) produces a clearance and an SE order. A PMA produces
an approval order. A De Novo produces a grant of classification and a new regulation. The
words are not interchangeable in a submission, a press release, or an investor deck, and
using them loosely signals that the team does not understand its own regulatory position.

The second is the difference between a predicate and a product that resembles one. A
predicate must be legally marketed and must share the intended use. A device with the same
sensor but a different claim is not a predicate for the new claim; it may be a reference
device that supports part of the testing. A device that was cleared for a narrower
population cannot silently support a broader one. The database record shows the device name
and the decision, but the intended use comes from the cleared labeling and the summary, and
those must be read.

The third is the meaning of an FDA day. When FDA reports that it met the 90-day goal, it is
reporting 90 days on a clock that stopped every time the sponsor had a question to answer.
The calendar elapsed time can be twice that. A program plan that assumes 90 calendar days
from submission to clearance is planning for a performance that FDA has not committed to and
that the sponsor's own response time makes impossible.

## 5. The mechanics, step by step

The procedure produces two artifacts: a pathway memo that states the route with its legal
basis, and a predicate matrix that records the comparison. Work in order. When a step
produces more than one plausible answer, record both and continue; later steps usually
resolve the ambiguity, and the record is itself useful in a pre-submission.

### Step 1 — Freeze the intended use and the indications for use

**Input.** Product description, user-interface flows, model card, draft labeling, sales
material, and the clinical problem statement.

**Output.** A one-paragraph intended use and a separate indications-for-use statement that
name the medical purpose, the user, the patient population, the setting, the output, and the
exclusions.

**Rationale.** Substantial equivalence begins with the intended-use comparison, and the
intended use is the objective intent of the person responsible for labeling under 21 CFR
801.4, not the sentence a copywriter prefers. A claim that moves from "supports triage" to
"detects" changes the predicate set, the evidence, and sometimes the class. Freezing the
wording before the predicate search prevents the team from choosing a predicate first and
then bending the claim to fit it.

**Procedure.** (1) Write the purpose as a verb phrase. (2) Name the user by role and
training. (3) Name the population with age, condition, and exclusions. (4) Name the setting
and the workflow point. (5) Name the output and its basis. (6) Add at least one explicit
negative limitation. (7) Read it against every customer-facing sentence in the deck and the
website.

**Branches and edge cases.** If the product has several outputs, split the claim by function
before continuing. If the claim uses a word such as "monitor," "predict," or "screen," check
whether the predicate uses the same word with the same meaning; the same verb can carry
different evidentiary weight across device types. If the claim includes a workflow benefit
such as time to treatment, decide whether that needs outcome evidence or is a labeling
statement about the same function.

### Step 2 — Confirm device status and identify the lead center

**Input.** The frozen intended use and the product's components.

**Output.** A device status determination for each component and, for combination products,
a lead center and the legal basis for the assignment.

**Rationale.** A product that combines a device with a drug or biologic is a combination
product under 21 CFR Part 3, and the lead center is assigned by the primary mode of action.
Software that delivers a behavioral therapy and software that controls an infusion pump can
both be "digital," but they are regulated through different centers and different submission
types. A wrong lead center wastes a filing cycle.

**Procedure.** (1) Confirm the product is a device under FD&C Act §201(h). (2) List drug,
biologic, and device constituents. (3) Determine the primary mode of action. (4) If the
assignment is unclear, prepare a Request for Designation under §563. (5) Record the lead
center and any consulting center.

**Branches and edge cases.** A device-led combination product may still need a drug
component's chemistry, manufacturing, and controls information. Software that merely displays
a drug's dosing information is usually not a combination product. A drug-led product with a
connected app may have the app reviewed in the drug application instead of as a device.

### Step 3 — Locate the classification regulation and product code

**Input.** Intended use, technology, clinical domain, and anatomical site.

**Output.** Candidate 21 CFR regulations, product codes, classes, and default submission
types, each with a citation.

**Rationale.** The classification regulation is the hinge of the whole decision. It states
the device type, the class, and whether a 510(k) is required or the type is exempt. The
product code links the regulation to the databases where predicates live. Searching by
keyword without reading the regulation produces codes that look right and carry the wrong
controls.

**Procedure.** (1) Search the FDA Product Classification database by clinical domain and
output. (2) Read the regulation description in full. (3) Record the class and submission
type. (4) Note the panel and any recognized consensus standards. (5) Search the 510(k) and
De Novo databases for devices under each candidate code. (6) Record why each candidate was
kept or rejected.

**Branches and edge cases.** A product may fit several codes; the intended use decides which
one governs. A code with no cleared devices may still be the right code. A product with no
plausible code is unclassified, which points to De Novo or PMA rather than to a 510(k). A
code's class does not guarantee that a new device under that code is Class II; the
regulation does.

### Step 4 — Determine the class and the control set

**Input.** The candidate regulations and product codes.

**Output.** A provisional class (I, II, or III), the general and special controls that
apply, and the exemption status.

**Rationale.** Class is a legal category tied to controls, not a risk score. Class I devices
are subject to general controls; Class II devices to general and special controls; Class III
devices to premarket approval. The class determines which door is available, and the special
controls determine what evidence the submission must contain even when the door is a 510(k).

**Procedure.** (1) Read the classification regulation. (2) Identify general controls:
registration, listing, labeling, device reporting, and quality system requirements. (3)
Identify special controls: performance standards, labeling, postmarket surveillance, and
registries. (4) Check whether the type is exempt. (5) Record the control set and evidence.

**Branches and edge cases.** An exempt Class II type still requires registration, listing,
correct labeling, and complaint handling. A Class I type can require a 510(k) if the
regulation says so. A Class III type normally requires a PMA, but a humanitarian device
exemption is available only for the narrow statutory population. A novel type may be
unclassified even when a similar product is Class II.

### Step 5 — Build the predicate candidate set

**Input.** The product code, the intended use, and the technological description.

**Output.** A predicate matrix with one row per candidate and the evidence for each row.

**Rationale.** The predicate search is an evidence exercise, not a search-engine exercise.
Each candidate must be verified in the current database, with its decision date, its product
code, its intended use as cleared, and its technological description. A candidate that was
cleared ten years ago under a narrower claim may still be a predicate, but the difference
must be addressed. A candidate that has been removed from the market is not a predicate at
all.

**Procedure.** (1) Search the 510(k) and De Novo databases by product code and keyword. (2)
For each candidate, record the submission number, decision date, device name, and product
code. (3) Retrieve the cleared summary or decision summary and quote the intended use. (4)
Record the technology: inputs, algorithm class, output, user, and deployment. (5) Verify
current marketing status. (6) Rank by intended-use proximity, not recency.

**Branches and edge cases.** Multiple predicates are common and legitimate, but the
submission must identify a primary predicate and explain which differences each additional
predicate addresses. A predicate from a different product code is permitted when the
intended use and technology support it. A reference device is not a predicate. A device
whose clearance was rescinded, or whose labeling changed materially after clearance, needs
re-verification before it is cited.

### Step 6 — Run the substantial equivalence comparison

**Input.** The predicate matrix and the device's technical file.

**Output.** A written SE argument with a difference-by-difference analysis and a list of the
evidence needed to close each difference.

**Rationale.** 21 CFR 807.100(b) sets the test. The device must have the same intended use
as the predicate. It must have the same technological characteristics, or different ones
that do not raise different questions of safety and effectiveness and are supported by
submitted data showing it is as safe and effective. The critical phrase is "different
questions of safety and effectiveness," not "different technology." New technology is
acceptable when the questions it raises are the ones the submission answers.

**Procedure.** (1) Compare intended use word by word and record every difference. (2) Compare
technological characteristics: materials, design, energy source, inputs, algorithm, output,
and user interface. (3) For each difference, ask what safety or effectiveness question it
raises. (4) Decide whether existing data, bench testing, or clinical data answer that
question. (5) Write the SE argument so a reviewer can follow the logic without the predicate
summary open beside it. (6) Identify the differences that cannot be closed and reconsider
the pathway.

**Branches and edge cases.** A difference in intended use is usually fatal to a 510(k);
performance data cannot answer it. A difference in technology is usually survivable with
data. For software as a device, the technology comparison covers inputs, model class, output,
and user interface, and a change in any of them can raise a new question. A difference in
user, setting, or population may be an intended-use difference when those elements are named
in the indications.

### Step 7 — Select the pathway

**Input.** Class, exemption status, predicate status, and the SE analysis.

**Output.** A pathway decision with the legal basis and the evidence consequences.

**Rationale.** The four doors are not interchangeable and the choice is not a preference. A
510(k) requires a predicate and the same intended use. A De Novo requires no legally
marketed predicate and controls sufficient for reasonable assurance. A PMA is required for
Class III. An exempt type requires no premarket submission but still requires general
controls.

**Procedure.** (1) Check exemption first. (2) If not exempt, check class. (3) For Class I and
II, check whether a predicate passes the intended-use gate and whether the differences can
be closed. (4) For Class II with no suitable predicate, check De Novo eligibility. (5) For
Class III, plan a PMA or analyze HDE eligibility. (6) Record the alternative route and why
it was rejected.

| Situation | Class | Predicate | Pathway | Evidence emphasis |
|---|---|---|---|---|
| Exempt type, general controls only | I or II | Not required | No premarket submission | Registration, listing, labeling, complaint handling |
| Same intended use, same technology | II | Strong | 510(k), traditional | Bench, software verification, labeling |
| Same intended use, different technology | II | Usable | 510(k) with performance data | Comparative performance, possibly clinical |
| No predicate with the same intended use | II | Fails the intended-use gate | De Novo | Standalone benefit-risk, prospective data |
| No predicate, controls insufficient | III | Not applicable | PMA | Valid scientific evidence, manufacturing |
| Rare condition, statutory population | III | Not applicable | HDE or PMA | Safety and probable benefit for HDE |

### Step 8 — Choose the 510(k) type

**Input.** The pathway decision and the relationship to the predicate.

**Output.** A submission type and the corresponding content plan.

**Rationale.** The 510(k) program has several variants with different evidence expectations
and review times. A Special 510(k) is for a change to the sponsor's own legally marketed
device. An Abbreviated 510(k) relies on a guidance document or a recognized consensus
standard. The Safety and Performance Based Pathway relies on performance testing against
established criteria rather than a direct predicate comparison. Choosing the wrong variant
produces a submission that does not answer the question the reviewer is asking.

**Procedure.** (1) Confirm the submission is for a new device or for a change to the
sponsor's own device. (2) Check whether an FDA guidance or recognized standard covers the
device type. (3) Check whether the Safety and Performance Based Pathway applies. (4) Select
the variant and record the basis. (5) Build the content plan against 21 CFR 807.87.

**Branches and edge cases.** A Special 510(k) cannot be used to change the intended use. An
Abbreviated 510(k) still needs a predicate and an SE discussion. The Safety and Performance
Based Pathway is optional and does not remove the need to identify a predicate for context.
The 2014 program guidance removed its discussion of Special and Abbreviated 510(k)s; the
1998 New 510(k) Paradigm and the relevant guidance still govern those routes. FDA's 510(k)
Submission Programs page states a review period of about 30 days for a Special 510(k) and
about 90 days for the other variants.

### Step 9 — Assemble and file the 510(k)

**Input.** The SE analysis, the test data, the labeling, and the quality-system records.

**Output.** An eSTAR submission with a 510(k) summary or statement, and a filing strategy.

**Rationale.** Content requirements are set by 21 CFR 807.87 and the summary requirements by
21 CFR 807.92 or 807.93. Since 1 October 2023, most 510(k)s must be submitted using eSTAR.
The format is not administrative decoration: an incomplete eSTAR is held, and a held
submission does not start the review clock.

**Procedure.** (1) Build the eSTAR from the current template. (2) Complete the device
description, predicate comparison, SE discussion, and labeling. (3) Include the 510(k)
summary with the required elements, including the predicate identification. (4) Check the
user fee and registration. (5) File, monitor technical screening, and respond to any
additional-information request within the window.

**Branches and edge cases.** A 510(k) summary becomes a public document; write it for a
competitor to read. A 510(k) statement avoids the summary but requires the data to be
available on request. Clinical data from outside the United States needs to meet the
applicable acceptance standards. A submission that relies on a predicate's unpublished data
cannot assume access to that data.

### Step 10 — Prepare a De Novo request

**Input.** The classification analysis, the benefit-risk argument, and the proposed special
controls.

**Output.** A De Novo request with the device description, the classification proposal, the
controls proposal, and the evidence.

**Rationale.** Section 513(f)(2) allows a request when no legally marketed predicate exists
and general controls, or general and special controls, can provide reasonable assurance of
safety and effectiveness. The request is a classification proposal, not only an evidence
package. A granted request creates a new classification regulation, a product code, and, for
Class II, special controls. That is why the De Novo route changes the market for everyone who
follows.

**Procedure.** (1) Document the predicate search that establishes the absence of a legally
marketed predicate. (2) Write the proposed classification and special controls. (3) Build
the benefit-risk analysis for the intended population. (4) Assemble the non-clinical and
clinical evidence. (5) Describe the labeling and the postmarket plan. (6) File using eSTAR,
mandatory for non-exempt De Novo requests since 1 October 2025. (7) Prepare for a
classification discussion rather than a comparison discussion.

**Branches and edge cases.** A De Novo can follow an NSE decision on a 510(k) or be filed
directly. A direct filing must still show that no predicate exists; a weak search invites a
question about why a 510(k) was not attempted. A De Novo is not available for a device that
needs Class III controls. The 150-day MDUFA goal applies to 70 percent of requests, and the
regulation's 120-day period runs from receipt, so the two clocks are not the same clock.

### Step 11 — Prepare a PMA

**Input.** The Class III determination, the clinical plan, and the manufacturing system.

**Output.** A PMA application with the required modules and a pre-approval inspection plan.

**Rationale.** Section 515 requires valid scientific evidence that the device is safe and
effective for its intended use, and the application includes manufacturing information that
FDA reviews. The PMA is the only route that examines the manufacturing process as part of
the marketing decision. For a software device, that means the software lifecycle records,
the configuration management, and the quality system are part of the reviewed object.

**Procedure.** (1) Confirm the Class III determination and check HDE eligibility. (2)
Assemble the PMA modules: administrative, summary, non-clinical, clinical, and manufacturing.
(3) Prepare for the filing decision within 45 days. (4) Prepare for in-depth review,
deficiency letters, and a possible advisory panel. (5) Plan the pre-approval inspection,
post-approval studies, and supplements.

**Branches and edge cases.** A first-of-a-kind device is likely to go to an advisory panel;
plan the panel package and the public meeting. A PMA for a software-only device still needs
a manufacturing section, interpreted as the software development and release process.
Changes after approval require supplements unless they fall within an authorized PCCP.
Failure to meet a post-approval requirement can lead to withdrawal.

### Step 12 — Decide whether to use a Q-Sub or a 513(g)

**Input.** The open questions from steps 1 through 11.

**Output.** A feedback strategy with a named mechanism for each question.

**Rationale.** A Q-Sub is a voluntary pre-submission interaction that gives FDA's feedback on
a planned submission; a 513(g) request asks for FDA's view on classification and applicable
requirements. They answer different questions. A Pre-Sub is not a binding decision, and a
513(g) response is not a substantial-equivalence review or a clearance. Using the wrong
mechanism produces a useful-looking letter that does not resolve the question.

**Procedure.** (1) List the open questions and classify each as classification, predicate,
study design, change control, or jurisdictional. (2) Route classification questions to
513(g), study and predicate questions to a Pre-Sub, and combination products to an RFD. (3)
Limit a Pre-Sub to four topics. (4) Record the feedback and the residual uncertainty.

**Branches and edge cases.** A Pre-Sub that asks FDA to approve a protocol invites a
non-committal answer; ask specific questions with proposed answers. A 513(g) filed for an
obvious device wastes the fee. A Q-Sub does not stop the clock and does not bind a later
reviewer, although reviewers expect an explanation for a departure from prior feedback.

| Question | Mechanism | Statutory or policy basis | Timing |
|---|---|---|---|
| Is this product a device, and what class? | 513(g) request | FD&C Act §513(g) | 60 days, statutory |
| Is this predicate acceptable for this claim? | Pre-Sub | Q-Submission guidance, May 2025 | 70 calendar days, target |
| Is this endpoint acceptable for a pivotal study? | Pre-Sub | Q-Submission guidance, May 2025 | 70 calendar days, target |
| Will a planned change need a new submission? | Pre-Sub with a draft PCCP | Q-Submission guidance, May 2025; §515C | 70 calendar days, target |
| Which center leads a combination product? | Request for Designation | FD&C Act §563 | 60 days, statutory |
| None of the above; the database answers it | No Q-Sub | Documented search | Immediate |

### Step 13 — Close the evidence gap

**Input.** The differences identified in step 6 or the benefit-risk questions in steps 10 and
11.

**Output.** An evidence matrix that maps each question to a test, a dataset, an endpoint,
and an acceptance criterion.

**Rationale.** A pathway memo that lists evidence without linking it to a reviewer question
produces a submission with redundant studies and unaddressed differences. The matrix makes
the logic auditable: this difference raises this question, this test answers it, this
criterion defines success. It also exposes evidence that cannot be generated before filing,
which is the signal to change the claim or the route.

**Procedure.** (1) List each difference or uncertainty. (2) Write the safety or effectiveness
question it raises. (3) Choose the test, study, or analysis that answers it. (4) Pre-specify
the endpoint and the analysis population. (5) State the acceptance criterion and the
decision rule. (6) Identify the evidence that depends on the predicate's data. (7) Mark any
question that cannot be answered before filing.

**Branches and edge cases.** Some differences are answered by labeling, for example a
limitation that narrows the user. Some are answered by the predicate's public summary, but
only when the comparison is valid. Some cannot be answered at all, which means the claim must
change. A difference that expands the population is rarely answerable with a subgroup
analysis powered for the narrower population.

### Step 14 — Manage the product after the decision

**Input.** The cleared or approved device, the labeling, and the change backlog.

**Output.** A change-classification procedure and, where appropriate, a PCCP.

**Rationale.** A clearance covers a specific device, intended use, and technology. Under 21
CFR 807.81(a)(3), a new 510(k) is required when a change could significantly affect safety or
effectiveness or when there is a major change in intended use. Section 515C, added by FDORA,
allows a PCCP to authorize specified changes in advance for devices including artificial intelligence (AI)-enabled
software functions. Everything else can usually be handled through the quality system with a
documented rationale.

**Procedure.** (1) Describe the change. (2) Ask whether it changes the intended use. (3) Ask
whether it could significantly affect safety or effectiveness. (4) Check whether a PCCP
covers it. (5) For a PMA, check the supplement type. (6) Record the decision and the
rationale in the design history file. (7) Re-verify the predicate and the labeling after
the change.

**Branches and edge cases.** A retrained model is a change to the device even when the code
is unchanged. A threshold change can alter clinical performance and the label. A new input
device or image protocol can invalidate the analytical validation. A population expansion is
a major change in intended use; a performance improvement within the cleared indications may
still need a new 510(k) if it shifts the risk-benefit balance.

| Change | Likely route | Why | Documentation |
|---|---|---|---|
| Cosmetic UI change, same function and output | No submission | No effect on safety or effectiveness | Change record and rationale |
| Retrained model within a PCCP protocol | No new submission, if within the authorized plan | §515C authorizes the planned change | PCCP update and verification records |
| New output category or threshold with clinical effect | New 510(k) | Could significantly affect safety or effectiveness | SE comparison and performance data |
| Expanded population or new clinical claim | New 510(k), De Novo, or PMA supplement | Major change in intended use | New clinical evidence and labeling |
| New hardware platform for the same software | New 510(k) or PMA supplement | New risks and verification burden | Platform verification and cybersecurity review |

## 6. Worked example

The case is hypothetical. The product, sponsor, predicate labels, and study numbers are
constructed to show the reasoning. Identifiers for the predicate are deliberately omitted
because the case is not real; a real analysis would record the submission number and the
decision date for every candidate, verified in the FDA databases on the day of the search.

### 6.1 Product and claim

SpiroRisk is a cloud software product that analyzes home spirometry waveforms and a daily
symptom diary from adults with chronic obstructive pulmonary disease (COPD). It returns a
30-day exacerbation risk tier, low, medium, or high, to a respiratory nurse in a
community-care practice. The nurse uses the tier to decide whom to call, whom to see, and
whom to leave on the routine schedule. The product does not diagnose an exacerbation, does
not recommend a drug or a dose, and does not replace clinical assessment.

The draft intended-use statement is:

"SpiroRisk is intended for use by respiratory nurses and primary-care clinicians to
prioritize adults aged 40 years and older with a diagnosis of COPD and no current
exacerbation for proactive review. The software analyzes home spirometry and symptom data
and assigns a 30-day risk tier. The output is an aid to outreach prioritization and is not a
diagnosis. SpiroRisk is not intended for use in patients with home mechanical ventilation,
patients with suspected pneumonia, or patients requiring urgent assessment."

Four choices narrow the evidence burden. The user is a nurse or clinician, not the patient.
The population excludes the highest-risk groups. The action is prioritization, not diagnosis
or treatment. The output is a tier, not a probability, which reduces the calibration
claim but does not remove the need for calibration evidence.

### 6.2 Classification and product code

The sponsor searches the Product Classification database for software that analyzes
spirometry or predicts exacerbation. It finds codes for spirometers, spirometry software,
and respiratory monitoring, most of which are Class II and require a 510(k). None of the
regulations describes a software function that predicts a future exacerbation from home
data. The closest code covers a device that measures and displays spirometry values; its
regulation describes measurement, not prediction.

This is the first substantive finding. A code that matches the input is not a code that
matches the function. The sponsor records two candidate codes, notes that neither regulation
describes prediction, and treats the product as potentially unclassified. That does not
settle the class, but it changes the likely route from 510(k) to De Novo and makes the
predicate search decisive.

The sponsor also checks whether the product could be an in vitro diagnostic. It is not: the
input is a physiological signal and a symptom record, not a specimen derived from the human
body. That keeps the analysis in the device framework and out of the IVD framework.

### 6.3 Predicate search

The sponsor searches the 510(k) database by product code and by keyword, then the De Novo
database. It records five candidates and evaluates each against the intended-use gate and
the technology comparison.

| Candidate | Submission type | Intended use compared with SpiroRisk | Technology | Gate | Fit |
|---|---|---|---|---|---|
| A. Spirometry display and trending | 510(k) | Measures and displays lung function; no prediction | Same input, no predictive model | Fails same use | Not a predicate |
| B. Respiratory monitoring with alerts | 510(k) | Monitors parameters and alerts on thresholds | Threshold rules, no 30-day risk model | Fails same use | Not a predicate |
| C. COPD care-management platform | 510(k) | Schedules outreach and tracks adherence | Workflow software, no patient-specific prediction | Fails same use | Not a predicate |
| D. Readmission-risk software for a different condition | 510(k) | Predicts 30-day readmission in heart failure | Similar model class, different population and outcome | Fails same use | Not a predicate |
| E. Exacerbation prediction from home data | None found | No legally marketed device with this claim | Not applicable | No predicate | Route to De Novo |

Candidate D is the interesting one. It has the same model class, a similar output format, and
a similar user. It fails the intended-use gate because the population and the predicted
event differ. A team that scored candidates on technical similarity alone would have picked
D and written a 510(k). The gate exists to prevent exactly that.

The search is documented with the date, the query, the codes searched, and the reason each
candidate was rejected. That document is the evidence that the De Novo request is not a
convenience.

### 6.4 The evidence gap

The sponsor's clinical team ran a retrospective development study in a secondary-care COPD
cohort of 4,180 patients with 1,421 exacerbation events, a prevalence of 34 percent. The
locked model achieved an area under the receiver operating characteristic curve (AUROC) of
0.81, sensitivity of 0.82 at the chosen threshold, and specificity of 0.79. The draft
labeling says that "two-thirds of high-risk alerts are true exacerbations."

That sentence is arithmetically correct in the development cohort and wrong in the intended
population. At the intended-use prevalence of 8 percent, the same sensitivity and specificity
give a positive predictive value of 0.253, not 0.668. The arithmetic is in section 7 and is
reproduced here in words: 1,000 patients at 8 percent prevalence contain 80 events; the model
finds 65.6 of them and flags 193.2 patients without an event; 65.6 divided by 258.8 is 0.253.
At 34 percent prevalence, 1,000 patients contain 340 events, the model finds 278.8, and flags
138.6 without an event; 278.8 divided by 417.4 is 0.668.

The evidence gap is therefore not "does the model discriminate." The gap is "does the model
calibrate and remain useful at the prevalence where it will be used, and does the labeling
describe the performance the user will see?" The sponsor needs a calibration analysis in a
primary-care-like cohort, a re-estimated threshold, and a labeling change that replaces the
"two-thirds" sentence with a statement of the expected alert burden and the positive
predictive value range.

There is a second arithmetic problem, in the sample size. To estimate a sensitivity of 0.82
with a 95 percent confidence interval half-width of 0.05, the required number of events is
approximately 1.96 squared times 0.82 times 0.18 divided by 0.05 squared, which is 227
events. At 34 percent prevalence, that requires about 668 patients. At 8 percent prevalence,
it requires about 2,835 patients. The development cohort had enough events for the enriched
setting and nowhere near enough for the intended one. A prospective study powered on the
enriched prevalence would produce a confidence interval that looks precise and answers the
wrong question.

### 6.5 The pathway decision

The sponsor decides on a De Novo request. The reasoning is recorded in four sentences. There
is no legally marketed device with the same intended use, so a 510(k) is not available. The
device is Class II in risk terms because a missed high-risk tier leads to delayed outreach
rather than immediate harm, and the nurse retains the decision. General and special controls
can provide reasonable assurance, so Class III controls are not required. The request will
propose special controls covering analytical validation, calibration across prevalence
ranges, subgroup performance, human-factors testing, labeling of the alert burden, and a
postmarket performance monitoring plan.

The sponsor files a Pre-Sub before locking the prospective protocol. The questions are
specific: whether the primary endpoint should be sensitivity at a fixed alert rate or
calibration across strata; whether a decentralized recruitment design is acceptable; and
whether the proposed special controls are sufficient. The Pre-Sub is not a guarantee, but it
reduces the chance that the pivotal study answers a question FDA did not ask.

### 6.6 Two findings that would sink the submission

**Finding 1: calibration fails in the intended population.** In a primary-care validation
set of 612 patients with 49 events, the calibration slope is 0.62 and the intercept is
negative. The model over-predicts risk in the low range and under-predicts at the top. The
observed event rate in the high tier is 21 percent, while the labeling implies roughly 67
percent. A reviewer comparing the labeling claim to the validation table sees a device whose
public claim is not supported in the population it is cleared for. The fix is not a
statistical adjustment after the fact; it is re-estimation and recalibration in a
representative cohort, with the labeling rewritten to the observed performance.

**Finding 2: temporal leakage in the feature pipeline.** Eleven of the 42 features are
computed from the 30-day window after the index date. Two examples are the count of rescue
corticosteroid courses and the number of unscheduled respiratory visits, both of which are
consequences of the exacerbation the model is predicting. Re-extracting the features with
strict index-date alignment drops the AUROC from 0.81 to 0.66 and the sensitivity from 0.82
to 0.64. A reviewer who asks for the feature extraction specification will find the leak,
and the primary endpoint in the submission will not be reproducible. The fix is a data
lineage review before the analysis is locked, not after.

Either finding is sufficient to stop the submission. The first is a population-validity
failure; the second is a measurement-validity failure. Both are detectable before filing,
which is the point of building the evidence matrix in step 13.

### 6.7 Vary one assumption: a predicate appears

Fourteen months before the planned filing, a competitor receives a De Novo grant for a
product with the same intended use in a different care setting. The grant creates a
classification regulation and a product code. SpiroRisk now has a legally marketed predicate.

The pathway changes from De Novo to 510(k). The evidence burden changes with it. Instead of a
standalone benefit-risk argument and a proposed set of special controls, the sponsor must
show substantial equivalence: the same intended use, and technological differences that do
not raise different questions of safety and effectiveness. The competitor's predicate uses a
different feature set and a different model class, so the technological characteristics
differ. The sponsor must argue that the differences raise questions the submission answers,
and must supply comparative performance data against the predicate's published results.

The change is not automatically cheaper. A 510(k) is faster when the predicate is close. When
the technology differs materially, the sponsor still needs clinical data, and the predicate's
special controls now define the evidence expectations. The calibration failure in finding 1
remains fatal under either route. The leakage in finding 2 remains fatal under either route.

| Assumption changed | Pathway | Evidence consequence | What stays the same |
|---|---|---|---|
| Base: no predicate, Class II | De Novo | Benefit-risk, proposed special controls, prospective study | Calibration and leakage findings |
| Predicate granted for the same use | 510(k) | SE comparison plus performance data against predicate | Calibration and leakage findings |
| Claim adds a treatment recommendation | New De Novo or PMA; class analysis required | Clinical outcome evidence, stricter controls | Nothing about the model changes |
| Population expanded to ventilated patients | De Novo with Class III analysis | Higher risk, stricter controls, mortality endpoint | Calibration becomes harder |
| Claim narrowed to spirometry quality only | 510(k) or exempt, depending on code | Analytical and usability evidence | The prediction claim disappears |

The lesson is that the pathway is a function of the claim and the market, not of the
algorithm. The same model, the same training data, and the same code can require a De Novo
in one year and a 510(k) in the next. The team that freezes the algorithm before it freezes
the claim will file the wrong submission.

## 7. Code, query, or calculation

The script below turns the decision tables into functions. It scores a candidate predicate
against the facts that drive substantial equivalence, returns a provisional pathway from the
class and the best available predicate, and computes the prevalence-dependent predictive
values used in section 6. It is small enough to audit and has no dependencies outside the
standard library.

```python
"""Predicate fit and US pathway selection (Module 14, Topic 03).

The score is a heuristic; the legal gates are `legally_marketed` and
`same_intended_use`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Predicate:
    label: str
    legally_marketed: bool = True
    same_intended_use: bool = True
    same_tech: bool = True
    different_questions: bool = False
    same_product_code: bool = True
    clinical_data_public: bool = False

    def gates(self) -> bool:
        return self.legally_marketed and self.same_intended_use

    def score(self) -> int:
        if not self.legally_marketed:
            return 0
        s = 100
        s -= 0 if self.same_intended_use else 45
        s -= 0 if self.same_tech else 15
        s -= 25 if self.different_questions else 0
        s -= 0 if self.same_product_code else 5
        s -= 0 if self.clinical_data_public else 10
        return max(s, 0)

    def verdict(self) -> str:
        if not self.gates():
            return "not a predicate for this claim"
        if self.score() >= 80:
            return "strong"
        if self.score() >= 55:
            return "usable with data"
        return "weak; expect NSE"


def pathway(cls: int, exempt: bool, best: Predicate | None) -> str:
    if cls == 3:
        return "PMA"
    if exempt:
        return "exempt; general controls only"
    usable = best is not None and best.gates() and best.score() >= 55
    if cls == 2:
        return f"510(k) via {best.label}" if usable else "De Novo"
    return "510(k) or De Novo" if usable else "De Novo"


def predictive_values(sens: float, spec: float, prevalence: float) -> dict[str, float]:
    tp, fn = sens * prevalence, (1 - sens) * prevalence
    fp, tn = (1 - spec) * (1 - prevalence), spec * (1 - prevalence)
    return {
        "ppv": tp / (tp + fp),
        "npv": tn / (tn + fn),
        "tp": tp * 1000,
        "fp": fp * 1000,
        "fn": fn * 1000,
        "tn": tn * 1000,
    }


if __name__ == "__main__":
    candidates = [
        Predicate("CADt triage, same code", clinical_data_public=True),
        Predicate(
            "adjacent triage, different model",
            same_tech=False,
            different_questions=True,
        ),
        Predicate(
            "spirometer, measurement claim",
            same_intended_use=False,
            same_product_code=False,
            clinical_data_public=True,
        ),
        Predicate("cleared then withdrawn", legally_marketed=False),
    ]

    print("Predicate fit")
    print(f"{'label':34s} {'gate':>4s} {'score':>5s}  verdict")
    for p in candidates:
        gate = "pass" if p.gates() else "fail"
        print(f"{p.label:34s} {gate:>4s} {p.score():5d}  {p.verdict()}")

    print("\nPathway")
    for label, cls, exempt, best in [
        ("class II, strong predicate", 2, False, candidates[0]),
        ("class II, weak predicate", 2, False, candidates[1]),
        ("class II, exempt code", 2, True, candidates[0]),
        ("class III", 3, False, candidates[0]),
        ("class I, no predicate", 1, False, None),
    ]:
        print(f"{label:28s} -> {pathway(cls, exempt, best)}")

    print("\nPredictive values at sens=0.82, spec=0.79")
    for prevalence in (0.08, 0.34):
        v = predictive_values(0.82, 0.79, prevalence)
        print(
            f"prevalence={prevalence:.2f}: PPV={v['ppv']:.3f} NPV={v['npv']:.3f} "
            f"TP={v['tp']:.1f} FP={v['fp']:.1f} FN={v['fn']:.1f} TN={v['tn']:.1f}"
        )
```

Exact output from a Python 3.11 run:

```text
Predicate fit
label                              gate score  verdict
CADt triage, same code             pass   100  strong
adjacent triage, different model   pass    50  weak; expect NSE
spirometer, measurement claim      fail    50  not a predicate for this claim
cleared then withdrawn             fail     0  not a predicate for this claim

Pathway
class II, strong predicate   -> 510(k) via CADt triage, same code
class II, weak predicate     -> De Novo
class II, exempt code        -> exempt; general controls only
class III                    -> PMA
class I, no predicate        -> De Novo

Predictive values at sens=0.82, spec=0.79
prevalence=0.08: PPV=0.253 NPV=0.981 TP=65.6 FP=193.2 FN=14.4 TN=726.8
prevalence=0.34: PPV=0.668 NPV=0.895 TP=278.8 FP=138.6 FN=61.2 TN=521.4
```

Three cautions apply. First, the score is not the legal test. The score can rank candidates
for a meeting, but 21 CFR 807.100(b) decides eligibility, and the two hard gates are legally
marketed status and the same intended use. The "spirometer, measurement claim" candidate
scores 50 and fails the gate; a team that read only the score might think it was close.

Second, the pathway function treats class as an input. In a real analysis the class comes
from the classification regulation, and the regulation may not exist for the product. When it
does not, the correct call is "unclassified" rather than "class II", and the route is De Novo
or PMA after a risk analysis.

Third, the predictive values assume that sensitivity and specificity transport from the
development cohort to the intended-use population. Section 6 shows why that assumption fails
when prevalence differs. The function is a communication tool for the alert-burden
conversation, not a substitute for calibration in the intended population.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and is not legal advice. Pathway eligibility is a fact-specific legal conclusion. Confirm the current statutes, regulations, guidance, and database records with qualified regulatory counsel before filing or relying on them.

The legal sources form a hierarchy. Statutes are binding. Regulations are binding and
implement the statutes. Guidance documents describe FDA's current thinking; they are not
binding, but a submission that departs from them without an explained alternative invites
questions. Standards are voluntary unless a regulation incorporates them or a manufacturer
adopts them. Local practice, such as a preferred template or a review habit, binds nobody.
A pathway memo should label every claim with its level and its date.

| Source | Status | What it constrains | What it does not do |
|---|---|---|---|
| FD&C Act §510(k), 21 U.S.C. 360(k) | Binding statute | Requires premarket notification for devices not exempt | Does not define the evidence standard |
| FD&C Act §513(a)(1) | Binding statute | Defines Classes I, II, and III and the controls for each | Does not assign a class to a product |
| FD&C Act §513(f)(2), 21 U.S.C. 360c(f)(2) | Binding statute | Creates the De Novo route for devices without a predicate | Does not guarantee that controls are sufficient |
| FD&C Act §513(g), 21 U.S.C. 360c(g) | Binding statute | Requires FDA to answer a written classification request within 60 days | Does not provide a clearance or an SE review |
| FD&C Act §515, 21 U.S.C. 360e | Binding statute | Requires premarket approval for Class III devices | Does not specify the clinical endpoint |
| FD&C Act §515C | Binding statute, added by FDORA on 29 December 2022 | Authorizes predetermined change control plans | Does not authorize changes outside the plan |
| 21 CFR 807 Subpart E | Binding regulation | Sets 510(k) procedures, content, and the SE criteria at 807.100(b) | Does not set review deadlines |
| 21 CFR 807.81(a)(3) | Binding regulation | Defines when a change requires a new 510(k) | Does not cover every software change |
| 21 CFR Part 814 | Binding regulation | Sets PMA content, filing, review, and supplement rules | Does not replace the statute's evidence standard |
| 21 CFR Part 860 Subpart D | Binding regulation | Sets De Novo procedures, including the 120-day period at 860.240 | Does not set the MDUFA performance goal |
| 21 CFR Part 3 | Binding regulation | Assigns combination products to a lead center | Does not resolve every jurisdictional question |
| MDUFA V commitment, FY2023–2027 | Binding agreement with performance goals | Sets 90 FDA days for 95 percent of 510(k)s and 150 FDA days for 70 percent of De Novo requests | Does not create a right to a decision by a date |
| 510(k) Program guidance, July 2014 | Guidance | Explains predicate selection and the SE decision flow | Is not a safe harbor |
| Q-Submission guidance, May 2025 | Guidance | Describes Pre-Sub content and the 70-calendar-day target | Does not bind a later reviewer |
| PCCP guidance, December 2024, reissued August 2025 | Guidance | Describes PCCP content for AI-enabled device software functions | Does not expand the statutory authority |
| eSTAR program page, accessed 25 October 2026 | Program requirement | Makes eSTAR mandatory for most 510(k)s and De Novo requests | Does not cover every submission type |

The statutory core is short. Section 510(k) requires a premarket notification at least 90
days before marketing a device that is not exempt, and section 513(i) defines substantial
equivalence. The operative test is in 21 CFR 807.100(b): same intended use, and either the
same technological characteristics or different ones that do not raise different questions of
safety and effectiveness and are supported by data showing the device is as safe and as
effective as the predicate. The predicate must be legally marketed and must not have been
removed from the market by FDA order or declared misbranded or adulterated by a judicial
order.

Section 513(f)(2) creates the De Novo route. A request may be submitted when no legally
marketed predicate exists and general controls, or general and special controls, can provide
reasonable assurance of safety and effectiveness. The implementing regulation at 21 CFR
860.240 sets a 120-day period after receipt for a grant or decline order. The MDUFA V goal
is 150 FDA days for 70 percent of requests. Those two numbers measure different things, and
neither is calendar time.

Section 515 requires premarket approval for Class III devices on the basis of valid
scientific evidence. The regulation at 21 CFR Part 814 provides for a filing decision within
45 days and an order within 180 days of filing. A first-of-a-kind device is likely to go to
an advisory panel. Manufacturing is part of the review, which is why a PMA for software still
requires a software development and release process that FDA can inspect.

The timing correction matters for planning. A commonly repeated claim is that the De Novo
goal is 90 days. The primary sources do not support it: the 510(k) goal is 90 FDA days for 95
percent under MDUFA V, and the De Novo goal is 150 FDA days for 70 percent. The 180-day PMA
figure is a period from filing, not from receipt, and it excludes the 45-day filing decision
and any amendment time. Report goals as goals, and plan in calendar weeks with a hold budget.

The data sources are part of the regulatory context. The 510(k) database provides cleared
devices, decision dates, product codes, and summaries. The De Novo database provides granted
requests and the classifications they created. The PMA database provides approvals and
supplements. The Product Classification database maps codes to regulations and classes. The
AI-enabled device list provides a view of authorized artificial intelligence devices but is
not exhaustive. Each source has an access date and each query should be recorded, because a
predicate search that cannot be reproduced is not evidence.

The sources do not tell you what a reviewer will decide. They tell you what has been decided
and what the agency has said it expects. The gap between those two things is where
regulatory judgment lives, and it is why a Pre-Sub exists.

## 9. Industry practice and case evidence

Public case evidence is uneven. FDA publishes De Novo decision summaries and PMA summaries,
and 510(k) summaries when the sponsor files one, but the level of performance detail varies.
Companies publish the cases that flatter them. The cases below separate the documented
record from the inference.

**IDx-DR.** FDA granted De Novo request DEN180001 for IDx-DR, an autonomous retinal image
analysis device, with a decision in April 2018. The device was classified in 21 CFR
886.1100 under product code PIB at Class II. The decision summary describes a pivotal study
of 900 patients across 10 primary-care sites, with sensitivity of 87.4 percent and
specificity of 89.5 percent for detecting more than mild diabetic retinopathy, and an
imageability rate of about 96 percent. The documented lesson is that an autonomous diagnostic
claim can be authorized at Class II when the intended use, the population, the image-acquisition
requirements, and the follow-up pathway are tightly specified. The inference, not stated by
FDA, is that the narrow claim did as much work as the algorithm.

**Apple ECG.** FDA granted DEN180044 for the Apple Watch ECG app with a decision in September
2018. The device was classified in 21 CFR 870.2345 under product code QDA at Class II, for
over-the-counter recording of a single-lead ECG and classification of atrial fibrillation
versus sinus rhythm, with an age restriction and an informational-use framing. A companion
irregular-rhythm notification feature was granted under DEN180042. The documented lesson is
that a consumer product with a medical claim is a regulated device, and that the channel does
not lower the evidence standard. The inference is that over-the-counter labeling and a
narrow classification claim were the mechanisms that made the authorization workable.

**Viz.ai large-vessel occlusion triage.** The original authorization was De Novo request
DEN170073, granted on 13 February 2018, which created product code QAS and 21 CFR 892.2080
for radiological computer-assisted triage and notification software. The decision summary
reports a 300-case computed tomography angiography study with sensitivity of 87.8 percent,
specificity of 89.6 percent, an area under the curve of 0.91, and a mean time from imaging
to notification of 7.32 minutes against 58.72 minutes in the comparator workflow. The later
510(k) K223042, cleared on 21 October 2022, is a later 510(k) for the same product line
under the product code the De Novo created. The documented lesson is
that a De Novo creates the predicate that makes the next product a 510(k), and that a
workflow claim can carry clinical value even when stand-alone accuracy is moderate. The
inference is that the time-to-notification endpoint was central to the original benefit
argument.

**Practice: the predicate search is a portfolio decision.** Mature companies maintain a
predicate watch. They track which codes are accumulating clearances, which De Novo grants
have created new codes, and which predicates are ageing or being withdrawn. A predicate that
is about to be discontinued is a strategic risk, because equivalence depends on a legally
marketed comparator. The public record does not quantify how often this happens; the
inference from recall and withdrawal patterns is that predicate continuity deserves an
explicit owner.

**Practice: the Q-Sub is a discipline, not a courtesy.** Teams that use Pre-Subs well bring
three or four specific questions with proposed answers and the data that support them. Teams
that use them badly ask FDA to endorse a program. The guidance's 70-calendar-day target is
a planning input, not a guarantee, and the feedback is not binding. The inference from
publicly discussed deficiency patterns is that specific questions get more useful answers.

**Practice: the 510(k) summary is a public document.** A summary filed under 21 CFR 807.92
is released and read by competitors, purchasers, and plaintiff experts. It should describe
the device, the predicate, the differences, and the performance data accurately enough to
survive being read alone. A summary that overstates the comparison creates a documentary
record that outlives the launch.

**Practice: clearance is not adoption.** A clearance establishes that the device may be
marketed for the cleared use. It does not establish reimbursement, clinical adoption, or
local performance. Health systems increasingly ask for local validation, and the Epic sepsis
model case in topic 01 shows that external performance can diverge from vendor claims. The
inference is that a regulatory strategy without a local-evidence strategy is incomplete.

**What the cases do not prove.** IDx-DR does not prove that every autonomous screening
product can use De Novo. Apple ECG does not prove that every wearable with a sensor is a
device. Viz.ai does not prove that every triage product can claim a workflow benefit. Each
case is a data point about a specific claim, population, and evidence package. The
transferable lesson is the method: establish the claim, find the regulation, test the
predicate, close the differences with evidence, and plan the change route.

## 10. Failure modes and red flags

Pathway errors rarely announce themselves. They look like a confident slide, a predicate
chosen because it appeared first in a search, or a timeline that assumes the review goal is a
deadline. The table records the failures that survive until a reviewer finds them.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Predicate chosen for input similarity | "It uses the same camera, so it is a predicate" | Intended use differs even when the sensor matches | NSE decision or a withdrawn submission | Apply the intended-use gate before the technology comparison |
| No documented predicate search | The De Novo draft has no search record | Absence of a predicate is an assertion that must be proved | Reviewer asks why a 510(k) was not attempted | Record queries, dates, codes, and rejection reasons |
| Stale predicate | Predicate was cleared a decade ago and is no longer sold | Legally marketed status is a current fact | Equivalence argument collapses | Re-verify status before filing and before launch |
| Class III product planned as a 510(k) | The claim implies a mortality or life-support benefit | Class comes from the regulation and the risk, not from the plan | Filing rejected or product cannot be marketed | Run the class analysis before the evidence plan |
| Review goal treated as a deadline | Launch date equals submission date plus 90 days | Goals exclude holds and calendar time | Launch slips, revenue plan fails | Plan with a hold budget and a calendar buffer |
| De Novo used to avoid PMA | The device is high risk but "novel" | De Novo requires controls sufficient for reasonable assurance | Request declined or classification disputed | Test whether general and special controls can assure safety |
| Predicate mismatch hidden in a table | Comparison table lists differences without questions | The SE argument is asserted, not written | Deficiency letter on the core question | Write the question for every difference, then the test |
| Multiple predicates without a primary | Three predicates cited, none controlling | Reviewers cannot tell which comparison governs | Confusion and delay | Name a primary predicate and assign differences to it |
| Format failure | Submission assembled as a flat PDF | eSTAR is mandatory for most 510(k)s and De Novo requests | Technical hold, clock does not start | Build in eSTAR from the current template |
| Change made without a new submission | Model retrained and deployed quietly | A significant change requires a new 510(k) under 807.81(a)(3) | Unapproved modification of a cleared device | Change procedure with a documented decision |
| PCCP treated as open-ended | Plan says "the model will improve over time" | Section 515C authorizes specified changes, not a direction | Plan rejected or change outside its scope | Bound the modifications, protocol, and acceptance criteria |
| Combination product filed with the wrong center | Software-plus-drug filed as a device without analysis | Lead center follows the primary mode of action | Refusal or transfer and lost time | Use a Request for Designation when the assignment is unclear |
| Predicate's clinical data assumed available | Team plans to rely on the predicate's unpublished study | Only public summaries are available to a competitor | Evidence gap discovered late | Plan independent data for differences that matter |
| Label broader than the indications | Marketing says "rules out" while the label says "supports" | Labeling is evidence of intended use under 21 CFR 801.4 | Claim exceeds the clearance | Review customer-facing text against the cleared label |

A red flag is not a violation. It is a missing fact. The correct response is to add the fact
to the memo, not to argue that the flag does not apply. A team that can explain why a
specific red flag is irrelevant, with a citation, is in a stronger position than a team that
never saw it.

The compound failure to watch for is the "predicate-first" program: the team finds a
cleared product that looks similar, writes a claim to match it, builds evidence to match the
claim, and discovers late that the cleared indications exclude the population the business
needs. The fix is procedural, not analytical: freeze the claim, then find the predicate.

## 11. Assumptions and limits

Every pathway memo rests on assumptions that are not in the submission. The memo should state
them, explain why they are needed, say what would break them, and name the test that would
detect the break.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The intended use is stable through the review | SE is judged against a fixed claim | A marketing rewrite or a new feature changes the claim | Version-control the intended use and review every release |
| The predicate remains legally marketed | Equivalence needs a current comparator | Withdrawal, rescission, or a judicial finding | Recheck the database before filing and before launch |
| The product code is representative | The code supplies class and controls | The code covers a different device type | Compare the regulation text and the cleared labels |
| The class comes from the regulation | Class sets the available door | An internal risk matrix is substituted for the regulation | Cite the regulation section in the memo |
| Sensitivity and specificity transport to the intended prevalence | Predictive values depend on prevalence | Case mix, referral pathway, or data quality differs | Recalibrate and report performance by stratum |
| FDA days approximate calendar planning | Teams budget by review goal | Holds, acceptance issues, and workload | Track actual elapsed time and keep a hold budget |
| Pre-Sub feedback predicts the decision | Teams rely on it to lock protocols | A different reviewer or new information | Treat feedback as advice and record residual risk |
| Guidance current at filing stays current | The evidence plan follows guidance | A revision or a new final guidance | Check the guidance page at submission and at decision |
| Clearance implies clinical utility | Commercial plans assume adoption | Local workflows and populations differ | Run local validation and monitor performance |
| One jurisdiction's route informs another | Teams reuse the US package | The EU and other regimes have separate rules | Build a jurisdiction matrix; see topic 08 |

Two limits deserve emphasis. The first is that this topic describes the US premarket
system and does not cover the European Union, the United Kingdom, Canada, or other markets
except by reference. A US clearance is not an EU conformity assessment, and the evidence
packages differ. The second is that the pathway does not determine the clinical value of the
product. A 510(k) clearance can be obtained for a device with modest benefit, and a PMA
approval can be followed by disappointing adoption. The regulatory decision and the clinical
decision are separate judgments.

A pathway memo should end with a confidence statement and a list of open facts. "We are
confident that the device is Class II and that a 510(k) is available if the predicate remains
marketed; we are less confident that the differences in the user population can be closed
with bench data; we will file a Pre-Sub on the reader-study endpoint before locking the
protocol." That statement tells the team where to spend the next month.

## 12. Exercises

The exercises move from recall to application to critique to design. Each states a
deliverable, acceptance criteria, and a time estimate. Use the current FDA databases and
guidance, not this file alone.

### Exercise 1 — Pathway triage (recall and application, 35 minutes)

Take four product concepts: a spirometry display app, a sepsis risk score for clinicians, a
closed-loop insulin dosing controller, and a claims-denial dashboard. For each, write a
one-sentence intended use, identify the likely product code family and class, and name the
likely pathway. Deliverable: a four-row table with a citation for each pathway. Acceptance
criteria: each row cites a regulation or statute; no row uses "probably exempt" without the
regulation's exemption text; the non-device row is justified by the statutory definition,
not by the absence of a code.

### Exercise 2 — Predicate matrix (application, 50 minutes)

Choose one of the Class II concepts from Exercise 1 and search the 510(k) and De Novo
databases. Record five candidate predicates with submission number, decision date, device
name, product code, and a quotation of the cleared intended use. Score each candidate against
the intended-use gate and the technology comparison. Deliverable: a predicate matrix plus a
one-paragraph recommendation of the primary predicate. Acceptance criteria: every candidate
is verified in the current database; the intended-use quotation comes from the summary or
decision document; rejected candidates state the specific failing element.

### Exercise 3 — Substantial equivalence argument (critique and application, 45 minutes)

Using the primary predicate from Exercise 2, write the SE argument in the form the reviewer
will read: same intended use, technological differences, the question each difference raises,
the data that answers it, and the conclusion. Deliverable: a 600-word argument with a difference table and the positive predictive value at
the intended prevalence. Acceptance criteria: every difference has a question and a test; the
argument does not rely on the predicate's unpublished data; the conclusion is stated in the
language of 21 CFR 807.100(b).

### Exercise 4 — De Novo eligibility memo (application, 45 minutes)

Assume the predicate in Exercise 2 is withdrawn and no equivalent device exists. Write a
one-page memo on De Novo eligibility. Include the documented search, the risk analysis, the
proposed special controls, and the evidence plan. Deliverable: the memo with a search log
appendix. Acceptance criteria: the search log lists queries, dates, and results; the proposed
controls are specific enough to be testable; the memo states why Class III controls are not
required.

### Exercise 5 — PMA transition analysis (design, 40 minutes)

Take the same product and add a claim that the output may be used to initiate treatment
without clinician review in a population at risk of a life-threatening event. Rewrite the
intended use and analyze how the pathway changes. Deliverable: a before-and-after table
covering class, pathway, evidence, manufacturing review, and change control. Acceptance
criteria: the analysis names the statutory basis for the new class; the evidence plan includes
a clinical endpoint with a decision rule; the change-control section addresses supplements.

### Exercise 6 — Feedback strategy (application, 30 minutes)

List five open questions for a planned submission and route each to a Pre-Sub, a 513(g), a
Request for Designation, or no interaction. For each Pre-Sub question, write the question and
your proposed answer in two sentences. Deliverable: a routing table with the drafted
questions. Acceptance criteria: no question is routed to a mechanism that cannot answer it;
each Pre-Sub question is specific and answerable; the table states the expected timing.

### Exercise 7 — Change-control decision set (application, 40 minutes)

Given a cleared device, classify six changes: a color change in the user interface, a
retrained model inside an authorized PCCP, a new threshold that alters the alert rate, a new
output category, an expanded age range, and a new cloud hosting region. For each, state the
route, the rationale, and the documentation. Deliverable: a six-row decision table.
Acceptance criteria: each row cites 21 CFR 807.81(a)(3), section 515C, or the PMA supplement
rules; the PCCP row states what would take the change outside the plan.

### Exercise 8 — Adversarial review (critique, 40 minutes)

Exchange a pathway memo with a peer. Attack the weakest element: the intended-use wording,
the predicate choice, the class determination, the difference analysis, or the change plan.
Write six questions a reviewer would ask and identify the two the memo cannot answer.
Deliverable: six questions plus a one-paragraph assessment. Acceptance criteria: questions
are specific to the memo; at least two identify missing facts rather than missing prose; the
assessment separates a documentation gap from a substantive gap.

## 13. Self-check answers

**Question 1. What is the difference between a 510(k) clearance, a De Novo grant, and a PMA
approval?** A 510(k) clearance is an order that the device is substantially equivalent to a
legally marketed predicate. A De Novo grant is a classification decision that creates a new
device type and its controls because no predicate exists. A PMA approval is an independent
finding that the device is safe and effective on valid scientific evidence, including a
manufacturing review. A 510(k) clearance and a De Novo grant permit marketing without a PMA;
only the 510(k) requires a predicate.

**Question 2. Which statutory section and regulation govern each route?** The 510(k) is
section 510(k) of the FD&C Act, 21 U.S.C. 360(k), implemented at 21 CFR 807 Subpart E. De Novo
is section 513(f)(2), 21 U.S.C. 360c(f)(2), implemented at 21 CFR Part 860 Subpart D. PMA is
section 515, 21 U.S.C. 360e, implemented at 21 CFR Part 814. Exemption is a feature of the
individual classification regulation in 21 CFR Parts 862 through 892.

**Question 3. Given a Class II device with a predicate that shares the intended use but uses a
different algorithm, what pathway applies?** A 510(k), provided the difference does not raise
different questions of safety and effectiveness that the data cannot answer. The intended use
gate is satisfied. The technological difference is addressed with performance data. If the
difference raises a question the data cannot answer, the route is not automatically De Novo;
the first response is to change the claim or generate the data, because a predicate exists.

**Question 4. What does exempt mean, and what still applies?** Exempt means the device type
does not require a premarket notification. General controls still apply: establishment
registration and device listing, labeling requirements, medical device reporting, and the
quality system requirements applicable to the device. Exempt is not unregulated.

**Question 5. What must a predicate matrix contain?** One row per candidate, with the
submission number, decision date, device name, applicant, product code, a quotation of the
cleared intended use, the technological characteristics, the current marketing status, the
gate result, and the reason for rejection or selection. It should also name the primary
predicate and identify any reference devices separately.

**Question 6. What does "different questions of safety and effectiveness" mean?** It means
that the technological difference creates a safety or effectiveness uncertainty that did not
exist for the predicate and that the submission must resolve. A new sensor may raise a
question about signal quality in a subgroup. A new algorithm may raise a question about
failure modes under distribution shift. The phrase is about the questions, not the novelty.

**Question 7. When do you use a Pre-Sub, a 513(g), or a Request for Designation?** A Pre-Sub
for feedback on a planned submission, including predicate strategy and study design. A 513(g)
for FDA's view on classification and applicable requirements. A Request for Designation when a
combination product's lead center is unclear. None of them is a substitute for a marketing
submission, and none produces a binding decision.

**Question 8. Compute the positive predictive value at 8 percent and 34 percent prevalence
for sensitivity 0.82 and specificity 0.79.** At 8 percent: true positives 0.0656, false
positives 0.1932, PPV 0.0656 divided by 0.2588 equals 0.253. At 34 percent: true positives
0.2788, false positives 0.1386, PPV 0.2788 divided by 0.4174 equals 0.668. The negative
predictive value moves from 0.981 to 0.895. At 8 percent prevalence the model generates 3.9
alerts for every true case; at 34 percent it generates 1.5, which changes the labeling and
the workflow.

**Question 9. When does a change require a new 510(k)?** Under 21 CFR 807.81(a)(3), when the
change could significantly affect safety or effectiveness, or when there is a major change in
intended use. A change within an authorized PCCP under section 515C does not require a new
submission. A change to a PMA device requires a supplement unless it falls within an
authorized PCCP or is eligible for a 30-day notice.

**Question 10. Distinguish a binding requirement, guidance, and convention with an example.**
21 CFR 807.100(b) is binding: it defines substantial equivalence. The 510(k) Program guidance
of July 2014 is guidance: it explains how FDA evaluates the factors but does not create a
legal obligation. A house style for predicate tables is convention: it affects readability,
not legality. A memo that mixes the three makes a recommendation look more authoritative than
it is.

**Question 11. Name two findings that would sink a submission and explain the difference.**
A population-validity failure, such as calibration that does not hold when the intended
population has 8 percent prevalence instead of 34 percent, and a measurement-validity
failure, such as features computed from the outcome window, both stop a submission. The
first says the device does not work as claimed in the intended population; the second says
the reported performance is not reproducible from the device as specified. The first is
fixed by recalibration and a new study; the second by data lineage review and re-analysis.

## 14. Interview questions

**1. Walk me through choosing a FDA pathway for a new software device.** A strong answer
starts with the intended use, finds the classification regulation and product code, states
the class, searches for a predicate, applies the intended-use gate, analyzes differences, and
then chooses among exempt, 510(k), De Novo, and PMA. A weak answer starts with the technology
or says "it depends" without naming what it depends on.

**2. Why is a 510(k) not an approval?** A strong answer explains that a 510(k) is a finding
of substantial equivalence to a predicate, not an independent finding of safety and
effectiveness, and that the clearance is bounded by the intended use and the technology
described. A weak answer treats clearance and approval as synonyms.

**3. When is De Novo the right route?** A strong answer says it is for a device with no
legally marketed predicate where general and special controls can provide reasonable
assurance, and notes that a grant creates a new regulation, a product code, and special
controls that later products can use as a predicate. A weak answer calls it a way to avoid
PMA.

**4. How do you evaluate a predicate?** A strong answer covers legally marketed status, the
intended-use comparison, the technological comparison, the questions raised by differences,
the availability of public performance data, and the current marketing status. A weak answer
compares product names or sensors.

**5. What is the review timeline, and how do you plan against it?** A strong answer separates
the statutory period, the MDUFA goal, and calendar time, and gives the figures: 90 FDA days
for 95 percent of 510(k)s, 150 FDA days for 70 percent of De Novo requests, 180 days from
filing for a PMA. A weak answer quotes 90 days as a deadline for everything.

**6. A model is retrained every quarter. What is the regulatory plan?** A strong answer
describes a PCCP under section 515C with a description of modifications, a modification
protocol, and an impact assessment, plus change-control procedures for anything outside the
plan. A weak answer says the model is "continuously learning" and implies no submission is
needed.

**7. When would you file a Pre-Sub?** A strong answer names a specific unresolved question,
the proposed answer, the data supporting it, and the timing, and notes that the feedback is
not binding. A weak answer says Pre-Subs are always a good idea, or that they guarantee
clearance.

**8. What would make you recommend PMA over 510(k)?** A strong answer cites a Class III
classification, the insufficiency of general and special controls, or a claim that requires
valid scientific evidence of a mortality or life-support benefit, and describes the
manufacturing and clinical consequences. A weak answer treats PMA as a bigger 510(k).

## 15. Key terms

| Term | One-line meaning |
|---|---|
| 510(k) | Premarket notification demonstrating substantial equivalence to a legally marketed predicate |
| 513(g) | Request for FDA's written view on classification and applicable requirements |
| Acceptance review | FDA's completeness check before substantive review |
| Additional information request | A hold that stops the review clock until the sponsor responds |
| Classification regulation | The 21 CFR section that names a device type, class, and submission requirement |
| Clearance | The marketing permission produced by a 510(k) SE order |
| Combination product | A product combining drug, device, or biologic components under 21 CFR Part 3 |
| De Novo | Classification request for a novel device with no legally marketed predicate |
| eSTAR | FDA's interactive PDF template for electronic premarket submissions |
| Exempt | A device type that needs no premarket notification but remains subject to general controls |
| FDA day | A review-clock day that excludes sponsor hold periods |
| FDORA | The 2022 law that added section 515C and other device provisions |
| HDE | Humanitarian device exemption for devices meeting the statutory rare-condition criteria |
| IDE | Investigational device exemption for clinical studies of unapproved devices |
| MDUFA | The user fee agreement that sets FDA review performance goals |
| NSE | Not substantially equivalent; the device may not be marketed on that basis |
| PCCP | Predetermined change control plan authorizing specified future modifications |
| PMA | Premarket approval for a Class III device |
| PMA supplement | A submission required to change an approved PMA |
| Predicate | A legally marketed device used as the comparator for substantial equivalence |
| Pre-Sub | A Q-Submission requesting feedback on a planned submission |
| Primary predicate | The predicate that governs the comparison when several are cited |
| Product code | FDA's three-letter identifier linking a device type to a regulation and class |
| Q-Sub | A voluntary pre-submission interaction with FDA |
| Reference device | A device used for testing or rationale but not relied on for equivalence |
| RFD | Request for Designation assigning a combination product to a lead center |
| Substantial equivalence | Same intended use plus technological characteristics that raise no unresolved questions |
| Unclassified | A device type with no classification regulation, routed to De Novo or PMA |

## 16. Sources

All URLs were accessed on 25 October 2026. Dates are the publication, revision, or content
dates shown by the source at access. Verify current versions before relying on them.

1. US Food and Drug Administration. "510(k) Submission Process." Content current as of 3 October 2022, with the eSTAR requirement effective 1 October 2023. https://www.fda.gov/medical-devices/premarket-notification-510k/510k-submission-process
2. US Food and Drug Administration. "510(k) Submission Programs." https://www.fda.gov/medical-devices/premarket-notification-510k/510k-submission-programs
3. US Food and Drug Administration. "The 510(k) Program: Evaluating Substantial Equivalence in Premarket Notifications [510(k)]." Guidance, July 2014. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/510k-program-evaluating-substantial-equivalence-premarket-notifications-510k
4. US Food and Drug Administration. "De Novo Classification Request." Content current as of 30 September 2025. https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/de-novo-classification-request
5. US Food and Drug Administration. "Premarket Approval (PMA)." https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/premarket-approval-pma
6. US Food and Drug Administration. "PMA Review Process." Content current as of 27 May 2025. https://www.fda.gov/medical-devices/premarket-approval-pma/pma-review-process
7. US Food and Drug Administration. "Requests for Feedback and Meetings for Medical Device Submissions: The Q-Submission Program." Final guidance, May 2025. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/requests-feedback-and-meetings-medical-device-submissions-q-submission-program
8. US Food and Drug Administration. "FDA and Industry Procedures for Section 513(g) Requests for Information." Guidance, 23 August 2024. https://www.fda.gov/media/78456/download
9. US Food and Drug Administration. "eSTAR Program." Content current as of 21 September 2026. https://www.fda.gov/medical-devices/how-study-and-market-your-device/estar-program
10. US Food and Drug Administration. "MDUFA Performance Goals and Procedures, Fiscal Years 2023 Through 2027." https://www.fda.gov/media/158308/download
11. US Food and Drug Administration. "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions." Final guidance, 4 December 2024, reissued 18 August 2025. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence
12. US Food and Drug Administration. "510(k) Premarket Notification Database." https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm
13. US Food and Drug Administration. "Device Classification Under Section 513(f)(2) (De Novo) Database." https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/denovo.cfm
14. US Food and Drug Administration. "Product Classification Database." https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPCD/classification.cfm
15. US Food and Drug Administration. "De Novo Classification Request DEN180001, IDx-DR." https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
16. US Food and Drug Administration. "De Novo Classification Request DEN180044, ECG App." https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf
17. US Food and Drug Administration. "De Novo Classification Request DEN170073, Viz.AI Contact." https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN170073.pdf
18. US Food and Drug Administration. "510(k) K223042, Viz LVO ContaCT, decision 21 October 2022." https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm?ID=K223042
19. US Food and Drug Administration. "FDA permits marketing of clinical decision support software for alerting providers of a potential stroke in patients." Press announcement, 13 February 2018. https://www.fda.gov/news-events/press-announcements/fda-permits-marketing-clinical-decision-support-software-alerting-providers-potential-stroke
20. Code of Federal Regulations. 21 CFR Part 807 Subpart E, Premarket Notification Procedures. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-807/subpart-E
21. Code of Federal Regulations. 21 CFR 807.100, FDA action on a premarket notification. https://www.law.cornell.edu/cfr/text/21/807.100
22. Code of Federal Regulations. 21 CFR 807.81, When a premarket notification submission is required. https://www.law.cornell.edu/cfr/text/21/807.81
23. Code of Federal Regulations. 21 CFR Part 814, Premarket Approval of Medical Devices. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-814
24. Code of Federal Regulations. 21 CFR Part 860 Subpart D, De Novo Classification. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-860/subpart-D
25. Code of Federal Regulations. 21 CFR 860.240, Review of a De Novo request. https://www.law.cornell.edu/cfr/text/21/860.240
26. Code of Federal Regulations. 21 CFR Part 3, Product Jurisdiction. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-A/part-3
27. Federal Food, Drug, and Cosmetic Act, sections 510(k), 513(a), 513(f)(2), 513(g), and 515; 21 U.S.C. 360, 360c, and 360e. Section 513 text: https://www.law.cornell.edu/uscode/text/21/360c
28. Food and Drug Omnibus Reform Act of 2022, Division FF of the Consolidated Appropriations Act, 2023 (Public Law 117-328), enacted 29 December 2022. Section 515C is the predetermined change control plan provision.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`02_is-it-a-device.md`](02_is-it-a-device.md)
- Next topic: [`04_gmlp-and-pccp.md`](04_gmlp-and-pccp.md)
- Related topic: [`07_iec-62304-software-lifecycle.md`](07_iec-62304-software-lifecycle.md)
- Related topic: [`10_post-market-surveillance.md`](10_post-market-surveillance.md)
- Survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 14.1 regulatory triage: [`../../labs/`](../../labs/)
- Lab 14.5 PCCP draft: [`../../labs/`](../../labs/)

The module index is the authoritative map. If a sibling file is not yet present, the index
marks it as planned; do not infer content from the link text.
