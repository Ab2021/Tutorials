# Definitions and Classification

> Module 14 — Medical Devices, SaMD, and Regulation
> Topic 01 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 00, 01
> Version 2026.3

## 1. Why this topic exists

A team can build an excellent model and still fail at the first regulatory question: what is
this thing legally? That question is not philosophical. It decides whether a quality
management system applies, whether a premarket submission is required, which evidence standard
governs, how changes may be made after launch, and who may sell the product in which market.

Most engineering teams arrive at the question late. They have a working prototype, a product
name, a slide deck, and a revenue plan. The regulatory lead then discovers that the product
description mixes diagnosis, triage, monitoring, and workflow messaging. One sentence implies
a screening claim; another implies a treatment recommendation. The intended-use statement
cannot be written without removing features or splitting the product. Every week spent after
that discovery is more expensive than a week spent before it.

The cost is not only submission cost. Classification changes the entire evidence pyramid. A US
Class I product may be exempt from premarket notification and need only general controls. A US
Class III product needs premarket approval (PMA) with valid scientific evidence, manufacturing
information, and likely a prospective clinical study. In the European Union, a Class IIa
software product may follow a conformity assessment route with a notified body, while a Class
III product faces stricter scrutiny and periodic audits. A borderline product that is wrongly
treated as non-device may be sold without any of those controls until a regulator, a customer,
or a plaintiff notices.

Classification is also unstable in a useful way. It follows the claim, not the code. A model
that estimates sepsis risk can be a device when it is marketed to clinicians for
patient-specific decision support and not a device when it is used only for retrospective
quality improvement inside a health system's own practice. The algorithm can be identical. The
intended use, users, setting, and consequences of output differ. This is why a data scientist
cannot hand the problem to regulatory affairs after the model is frozen. Design choices encode
claims.

This topic gives you the vocabulary and the procedure to make a first-pass classification, to
document the reasoning, and to know when the answer is too close to call. It does not replace
regulatory counsel or a formal 513(g) request in the United States or a notified body opinion
in the European Union. It gives you enough structure to ask better questions and to avoid the
common failure of treating a regulatory category as a marketing label.

The topic is deliberately jurisdictional. The United States and the European Union use
different definitions, different class names, and different logic. A product that is Class II
in the US is not automatically Class IIa in the EU. A software function that is a device in
one jurisdiction may be excluded in another. You will learn two systems side by side and a
third, the International Medical Device Regulators Forum (IMDRF) categorization, that is
useful for risk thinking but is not law anywhere.

The final reason this topic exists is communication. Regulatory documents, quality procedures,
and submission summaries use the same words with narrow meanings. "Intended use" is not a
slogan. "Indication" is not a feature list. "Accessory" is not any add-on. "SaMD" is not
synonymous with "AI." When a team uses these words loosely, reviewers infer claims the team
did not intend to support. Precise definitions are a form of risk control.

## 2. Learning outcomes

By the end of this topic, you can:

1. Distinguish a medical device, software as a medical device (SaMD),
     software in a medical device (SiMD), in vitro diagnostic (IVD)
     software, and a non-device software function using the applicable
     statutory or regulatory definition.
2. Write a function-level intended-use statement that names the medical
     purpose, user, patient population, setting, output, and
     limitations.
3. Assign a preliminary US device class and identify the regulation and
     product code family that would govern it.
4. Apply European Union Medical Device Regulation (MDR) Rule 11 and the
     implementing rules in Annex VIII to a software product, including
     the accessory and driving-software branches.
5. Explain why IMDRF risk categories are not regulatory classes and use
     them only as a risk-communication aid.
6. Produce a classification memo that separates binding requirements,
     guidance, and local convention.
7. Identify at least five situations in which classification is
     genuinely ambiguous and specify the evidence or external opinion
     needed to resolve them.
8. Critique a product description that mixes multiple intended uses and
     rewrite it as separable functions.

The outcomes map to the exercises and self-check items as follows.

| Outcome | Exercise | Self-check |
|---|---|---|
| 1 | Exercise 1, Exercise 2 | Questions 1, 2 |
| 2 | Exercise 3 | Question 3 |
| 3 | Exercise 4 | Question 4 |
| 4 | Exercise 5, Exercise 6 | Questions 5, 6 |
| 5 | Exercise 5 | Question 7 |
| 6 | Exercise 7 | Question 8 |
| 7 | Exercise 8 | Question 9 |
| 8 | Exercise 3, Exercise 8 | Question 10 |

## 3. The core mental model

Classification is a chain of narrowing questions. Each question has a legal source, and each
answer constrains the next. The chain is not a checklist that can be completed in any order.
If you start with the algorithm, you will reverse-engineer a claim to fit it. Start with the
function and the claim.

```text
Product concept
    |
    v
[1] What is the function, stated as one output for one user?
    |
    v
[2] Is there a medical purpose in the applicable statute?
    | yes                         | no
    v                             v
[3] Is it an article, accessory,  Non-device function
    or software function?          (document exclusion)
    |
    v
[4] Device vs IVD vs combination vs non-device
    |
    v
[5] Which jurisdiction and which market claim?
    |
    +-------------------+-------------------+
    | US                | EU                | Other
    v                   v                   v
[6] 21 CFR regulation  [7] MDR/IVDR         Local law
    product code           Annex VIII
    class I/II/III         class I/IIa/IIb/III
    |                      or A/B/C/D
    v                   v
[8] Pathway and controls
    |
    v
[9] Evidence obligations
    |
    v
[10] Change-control obligations
```

The first move is to define the unit of analysis. A product can contain several functions, and
each function can have a different status. A patient portal that displays a laboratory result,
explains it in plain language, and recommends a drug dose contains at least three functions.
The display may be a device or not depending on the claim; the explanation may be patient
education; the dose recommendation may be a device with a treatment claim. Regulators assess
software function by function, not by product name. The US Food and Drug Administration (FDA)
says so explicitly in its digital health policy and repeats it in the Clinical Decision
Support (CDS) guidance.

The second move is to write the intended use before choosing a class. Intended use is the
objective intent of the person legally responsible for labeling. It can be shown by words, by
design, by the circumstances of distribution, or by how sales staff describe the product. A
product called "insight" that is sold to hospitals with a claim that it "identifies patients
who need immediate review" has a clinical claim even if the word "diagnose" never appears. The
test is what the product is intended to do, not what the marketing team prefers to say.

The third move is to identify the medical purpose in the jurisdiction's own language. In the
US, section 201(h) of the Federal Food, Drug, and Cosmetic Act (FD&C Act) covers articles
intended for diagnosis, cure, mitigation, treatment, or prevention of disease, or intended to
affect the structure or any function of the body. The 21st Century Cures Act added exclusions
for certain software functions in section 520(o). In the EU, Article 2(1) of Regulation (EU)
2017/745 (MDR) defines a medical device by intended medical purpose, and Regulation (EU)
2017/746 (IVDR) defines an IVD by its purpose of examining specimens derived from the human
body. The words are similar but the exclusions and the classification logic differ.

The fourth move is to separate device from IVD. An IVD is not a device that happens to use a
sample. It is an article intended to examine specimens derived from the human body for a
medical purpose, and it is regulated under a separate European regulation and a separate US
framework. Software that analyzes a digital pathology image can be a device; software that
analyzes a blood sample's measured analyte and reports a result for diagnosis is likely an
IVD. The distinction changes the classification rules, the evidence standards, and the
submission type. When a product does both, it may need separate analysis for each function.

The fifth move is to apply the classification rules, not a risk score. US classes are tied to
the level of control needed to assure safety and effectiveness and to the availability of a
predicate. EU classes are defined by Annex VIII of the MDR or IVDR. A product does not become
Class III because a team's internal risk matrix says "high." It becomes Class III because a
rule says so, or because the US statute and a classification regulation put it there.

The sixth move is to record the answer as provisional. Classification is a legal conclusion. A
first-pass memo is a hypothesis with reasons. It should state which facts would change the
answer and what evidence would resolve them. That record is useful even when it is wrong,
because it shows the reviewer that the team reasoned from the claim rather than from the
desired timeline.

A useful shorthand for the whole chain is: claim determines purpose, purpose determines device
status, status determines rule set, rule set determines class, class determines controls, and
controls determine evidence. If you can say that sentence and defend each arrow with a source,
you understand the topic.

## 4. Definitions and vocabulary

The table below defines the terms used in this topic and later in the module. The "common
misuse" column is not pedantry. Each misuse has caused a real classification or submission
error.

| Term | Definition | Common misuse | Unit or source |
|---|---|---|---|
| Medical device | An article intended for a medical purpose, including diagnosis, prevention, monitoring, treatment, or alleviation of disease, as defined by the applicable jurisdiction | Treating any health-related product as a device, or any device as requiring premarket review | FD&C Act §201(h); MDR Article 2(1) |
| Software as a Medical Device (SaMD) | Software intended for one or more medical purposes that performs those purposes without being part of a hardware medical device | Calling all healthcare software SaMD, including software embedded in a pump or imaging system | IMDRF N10:2013; FDA SaMD page |
| Software in a Medical Device (SiMD) | Software that is part of, drives, or controls a hardware medical device, including firmware and embedded control software | Treating SiMD as SaMD and applying the wrong lifecycle or submission logic | IMDRF N10:2013; FDA digital health policy |
| Software as a Medical Device function | A discrete software capability with its own intended use and output; a product may contain several | Classifying a whole platform under the claim of its most regulated feature | FDA digital health policy; MDCG 2019-11 |
| Intended use | The objective intent of the persons legally responsible for labeling, including the general purpose and the conditions of use | Confusing it with a feature description or a marketing tagline | 21 CFR 801.4; MDR Article 2(12) |
| Indications for use | The disease or condition, patient population, and circumstances in which the device is intended to be used | Omitting the population or setting, which makes the claim broader than the evidence | FDA guidance; MDR Annex XIV |
| In vitro diagnostic (IVD) | A device intended to examine specimens derived from the human body to provide information for diagnosis, monitoring, or treatment | Calling any test-related software an IVD when it only displays results | IVDR Article 2(2); FD&C Act §201(h) |
| Accessory | An article intended to be used with a specific device to enable or assist its intended purpose, and which is itself a device under the MDR | Treating a cable, app, or algorithm as an accessory without checking the definition | MDR Article 2(2); MDR Annex VIII Rule 3.2 |
| Component | A part or software module that is not itself a finished device and is supplied to a device manufacturer | Selling a component directly to clinicians and still calling it a component | MDR Article 2(5); FDA component guidance |
| Combination product | A product composed of two or more regulated components, such as a drug and a device, physically or functionally combined | Assuming a software-plus-drug product is always a device | 21 CFR Part 3; FDA combination product policy |
| Class I (US) | Lowest-risk device class, subject to general controls; many are exempt from premarket notification | Assuming Class I means no controls or no registration | 21 CFR 860.3(c)(1) |
| Class II (US) | Moderate-risk class, usually subject to special controls and premarket notification under section 510(k) | Assuming every Class II device needs clinical trials | 21 CFR 860.3(c)(2) |
| Class III (US) | High-risk class, usually requiring premarket approval (PMA) | Assuming a novel low-risk device is Class III because no predicate exists | 21 CFR 860.3(c)(3) |
| 510(k) | Premarket notification demonstrating substantial equivalence to a legally marketed predicate | Calling it an "approval" | FD&C Act §510(k); 21 CFR 807 Subpart E |
| De Novo | Classification request for a novel device of low or moderate risk with no legally marketed predicate | Using De Novo for a high-risk device to avoid PMA | FD&C Act §513(f)(2) |
| PMA | Premarket approval for a high-risk device, requiring valid scientific evidence and manufacturing review | Treating a 510(k) clearance as permission to make any claim | FD&C Act §515 |
| Predicate | A legally marketed device used to support substantial equivalence | Choosing a predicate on similarity of name rather than intended use and technology | FDA 510(k) program |
| Product code | An FDA identifier for a device type, linked to a regulation and typically a class | Assuming a product code guarantees the same class in another jurisdiction | FDA product classification database |
| Class I (EU) | Lowest-risk MDR class, with some sterile or measuring subclasses requiring notified body involvement | Assuming Class I always allows self-certification | MDR Annex VIII; Article 52 |
| Class IIa (EU) | Medium-risk class, generally requiring notified body conformity assessment | Assuming IIa is the same as US Class II | MDR Annex VIII Rule 11 |
| Class IIb (EU) | Higher medium-risk class | Assuming a monitoring claim is always IIa | MDR Annex VIII Rule 11 |
| Class III (EU) | Highest-risk class, requiring strict conformity assessment and clinical evidence | Applying III only to implantables | MDR Annex VIII; Article 52 |
| Class A/B/C/D (IVDR) | IVD risk classes from lowest to highest under the IVDR | Using MDR classes for an IVD | IVDR Annex VIII |
| Rule 11 | The MDR classification rule specific to software | Applying it without the Chapter II implementing rules | MDR Annex VIII Rule 11 |
| Implementing rule 3.3 | Software that drives or influences a device takes the same class as that device; independent software is classified in its own right | Forgetting the hardware class when software controls a device | MDR Annex VIII Chapter II |
| Implementing rule 3.5 | If several rules apply, the strictest class applies | Choosing the most convenient rule | MDR Annex VIII Chapter II |
| IMDRF category | A four-level impact category (I–IV) based on significance of information and seriousness of the condition | Treating it as a legal classification | IMDRF N12:2014 |
| Clinical decision support (CDS) | Software that supports clinical decisions, some of which may be excluded from the US device definition | Calling any alert or score "CDS" without applying section 520(o) | FD&C Act §520(o); FDA CDS guidance |
| General controls | Basic US requirements such as registration, listing, labeling, and quality system requirements | Assuming general controls are optional for exempt devices | FD&C Act §501, §510, §513 |
| Special controls | Device-specific requirements, such as performance standards, postmarket surveillance, and labeling, used for Class II | Writing special controls as if they were guidance | FD&C Act §513(a)(1)(B) |
| Notified body | An independent EU organization that assesses conformity for higher-risk devices | Treating a notified body as a government agency | MDR Article 43 |
| Conformity assessment | The EU process of demonstrating compliance with the MDR or IVDR, including technical documentation and quality system review | Assuming a CE mark is a one-time event | MDR Articles 52–60 |
| CE marking | The mark indicating conformity with applicable EU harmonization legislation | Applying it to a product that is not yet compliant | MDR Article 20 |
| Unique Device Identification (UDI) | A globally unique identifier for a device model and production unit, used for traceability | Treating UDI as a marketing SKU | MDR Article 27; 21 CFR Part 830 |
| Intended purpose (EU) | The use for which a device is intended according to the data supplied by the manufacturer, including the clinical benefit and target population | Using the US "intended use" wording verbatim and missing EU clinical benefit language | MDR Article 2(12) |
| Clinical benefit | The positive impact on health, or a meaningful impact on diagnosis or treatment, claimed for a device | Claiming a workflow benefit as a clinical benefit without evidence | MDR Article 2(53) |
| Borderline product | A product whose status under the device definition is genuinely uncertain | Treating every borderline case as a device to be safe, without documenting why | Institutional practice; seek regulator advice |
| 513(g) request | A US request for FDA's view on the classification or regulatory status of a device | Using it as a substitute for a pre-submission meeting | FD&C Act §513(g) |
| Pre-submission (Q-Sub) | A voluntary meeting or written feedback process before a US marketing submission | Expecting a binding classification decision from a Q-Sub | FDA Q-Submission program |

Two definitions deserve more than a table row. The first is intended use. In the US, 21 CFR
801.4 says the intended use of a device is the objective intent of the persons legally
responsible for its labeling. Intent may be shown by expressions, by the design or composition
of the article, or by the circumstances surrounding its distribution. That means a product's
name, its user interface, its training materials, its sales claims, and even the customers to
whom it is sold can all be evidence of intended use. A regulatory memo that quotes only the
label is incomplete.

The second is SaMD. The IMDRF definition has three parts: software, one or more medical
purposes, and performance of those purposes without being part of a hardware medical device.
The third part is easy to misread. Software that runs on a general-purpose phone is not
thereby SaMD if it controls a connected pump; it may be SiMD or part of a device system.
Software that is embedded in a hardware device but performs an independent medical function
can still require separate analysis. The question is not where the code runs. It is what the
code is intended to do and whether it is part of a hardware device's medical function.

A third term, "accessory," is a frequent source of error. Under the MDR, an accessory is an
article that is not itself a medical device but is intended to be used together with a
specific device to enable that device to achieve its intended purpose or to specifically and
directly assist the device's medical function. The MDR treats accessories as devices and
classifies them in their own right under Annex VIII Rule 3.2, with the important exception
that accessories to active implantable devices fall under Rule 8. In the US, the concept is
similar but the statutory language and the submission logic differ. Do not assume that an app
sold alongside a cleared device inherits the device's class.

Finally, distinction between a binding requirement and guidance matters from the first page of
any memo. In the US, the FD&C Act and the Code of Federal Regulations are binding. FDA
guidance documents describe the agency's current thinking and are not legally binding,
although deviating from them without an alternative approach invites questions. In the EU,
regulations are binding and directly applicable, while MDCG guidance is not binding on
manufacturers or notified bodies. A classification memo should label each statement as
statute, regulation, guidance, standard, or local practice. That labeling is the difference
between a legal analysis and a summary of blog posts.

## 5. The mechanics, step by step

Classification is a procedure with a paper trail, not a single judgment. The procedure
produces two outputs: a provisional classification for each jurisdiction and a list of facts
that could change it. Work in order. If a step yields more than one plausible answer, record
both and continue rather than forcing an early decision.

### Step 1 — Define the unit of analysis

**Input.** Product description, user-interface flows, marketing draft, model card, data-flow
diagram, and existing claims.

**Output.** Discrete software functions, each with one primary output, one primary user, and
one clinical action.

**Rationale.** Regulators assess intended use at the level of the software function. A
platform with a sepsis alert, a documentation summarizer, and a billing dashboard is not one
regulatory object. Splitting the product prevents the most regulated feature from setting the
claim for everything else and prevents a low-risk feature from diluting a high-risk claim.

**Procedure.** (1) List every output a user could act on. (2) Name the user and the expected
clinical action for each. (3) Group outputs only when user, action, and failure consequence
match. (4) Use neutral functional labels, not marketing names. (5) Mark operational,
financial, educational, and administrative outputs.

**Branches and edge cases.** If two outputs share a user but lead to different clinical
actions, keep them separate. If one output is a precursor to another, decide whether it is
independently marketed. A summary that restates documented facts differs from a summary that
recommends a diagnosis. A ranking intended to guide outreach may be a device even when the
dashboard itself does not diagnose.

### Step 2 — Draft the intended-use statement

**Input.** The function list and the clinical problem the product addresses.

**Output.** A statement covering medical purpose, user, population, setting, output, and
limitations.

**Rationale.** Intended use anchors every later step. A useful draft has the form: "The
product is intended for use by [user] in [setting] to [clinical action] for [population] using
[input], and is not intended to [excluded use]." The negative clause prevents the reviewer
from inferring a broader claim.

**Procedure.** (1) State the purpose as a verb phrase: detect, diagnose, triage, monitor,
predict, recommend, or inform. (2) Name the user by role and training. (3) Name the
population, including age and exclusions. (4) Name the setting and workflow point. (5) Name
the output and its basis. (6) Add at least one explicit limitation. (7) Read it aloud and test
whether a clinician would infer a stronger claim.

**Branches and edge cases.** If the purpose is support rather than replace, say what the user
must independently review. If the output is a probability, say whether it is calibrated and
whether a threshold is part of the claim. "For informational purposes" does not neutralize a
patient-specific clinical claim. "Clinical decision support" describes a function, not an
exemption.

### Step 3 — Test the medical purpose against the statutory definition

**Input.** The draft intended-use statement.

**Output.** A yes, no, or unresolved answer for each jurisdiction, with the statutory
citation.

**Rationale.** Device status is legal, not technical. In the US, section 201(h) of the Federal
Food, Drug, and Cosmetic Act (FD&C Act) covers articles intended for diagnosis, cure,
mitigation, treatment, or prevention of disease, or intended to affect the structure or
function of the body. In the EU, Article 2(1) of Regulation (EU) 2017/745 (MDR) defines a
medical device by intended medical purpose.

**Procedure.** (1) Quote the current statutory text. (2) Map each intended-use element to an
element of the definition. (3) Check applicable exclusions, including section 520(o) in the
US. (4) Distinguish a complete exclusion from enforcement discretion. (5) Record unresolved
elements instead of guessing.

**Branches and edge cases.** Diagnosis, treatment, prevention, or monitoring of disease
usually means yes. Administrative, operational, or purely educational purposes may mean no.
Wellness claims depend on the specific claim and population. A no-show predictor is generally
not a device; a deterioration predictor generally is. A product used by a payer to deny care
may escape device law but attract other laws and clinical-risk review.

### Step 4 — Separate device, IVD, combination product, and non-device

**Input.** The intended-use statement and the product's inputs and outputs.

**Output.** A product category and the governing legal framework.

**Rationale.** The device/in vitro diagnostic (IVD) distinction changes the classification
rules and evidence standard. In the EU, the MDR and Regulation (EU) 2017/746 (IVDR) are
separate regulations with separate Annex VIII rules. In the US, IVDs are devices but often
have different classification regulations and submission expectations. A combination product
may be assigned to a lead center.

**Procedure.** (1) Ask whether the product examines a specimen derived from the human body.
(2) Ask whether the output is a measurement, result, or interpretation used for diagnosis or
treatment. (3) Ask whether a drug, biologic, or hardware component is integral to the primary
mode of action. (4) Determine the primary mode of action and lead center. (5) Record the
category and legal basis.

**Branches and edge cases.** A specimen-based diagnostic result points to IVD. An image or
signal from a patient that yields a clinical finding points to device. A drug-device
combination requires combination-product analysis. Digital pathology can be a device even
though the input is digitized. A laboratory system that only stores results may be non-device;
a module that interprets them may be a device or IVD.

### Step 5 — Determine whether the product is an article, accessory, or component

**Input.** The product's relationship to another device and the commercial arrangement.

**Output.** Independent device, accessory, component, or non-device, with the legal basis.

**Rationale.** Accessories and components are not free-floating labels. The MDR treats an
accessory as a device in its own right and classifies it separately under Annex VIII Rule 3.2,
with exceptions such as accessories to active implantable devices under Rule 8. In the US, an
accessory may need its own clearance even when sold with a cleared host.

**Procedure.** (1) Identify the host and the function the product enables or assists. (2) Ask
whether it enables the host's purpose or directly assists its medical function. (3) Ask
whether it is a finished device or a part for incorporation. (4) Check the jurisdiction's
accessory and component definitions. (5) Determine whether it has an independently marketed
intended use.

**Branches and edge cases.** An accessory is classified in its own right and does not
automatically inherit the host's class. A component supplied only to manufacturers is
generally not a finished device, but the same article sold to clinicians may be. A
glucose-sensor display app may be an accessory or an independent device depending on the
claim. A cable with signal conditioning may be an accessory; a passive cable usually is not.

### Step 6 — Determine the software category

**Input.** Function list, hardware architecture, and intended-use statement.

**Output.** Software as a Medical Device (SaMD), Software in a Medical Device (SiMD), embedded
software, production software, or non-device software.

**Rationale.** The category determines the lifecycle standard and submission logic. SaMD is
software intended for a medical purpose that is not part of a hardware medical device. SiMD is
software that is part of, drives, or controls a hardware device. Manufacturing software is not
a medical device function, although it may be subject to computer software assurance.

**Procedure.** (1) Ask whether the software performs its medical purpose without being part of
a hardware device. (2) Ask whether it controls or drives a hardware medical function. (3) Ask
whether it is embedded or distributed separately. (4) Ask whether it is used only to design,
manufacture, or test a device. (5) Record the category and boundary case.

**Branches and edge cases.** A cloud algorithm analyzing images from a separate scanner is
SaMD. The same algorithm embedded in the scanner may be SiMD. A firmware update that changes a
pump's dose calculation is part of the device. An app that only displays a device's output may
be an accessory or device, not automatically SaMD.

### Step 7 — Identify jurisdictions and claims

**Input.** Commercial plan, target markets, and intended-use statement.

**Output.** A jurisdiction-by-jurisdiction claim matrix.

**Rationale.** A single global claim is usually the broadest claim any regulator will accept,
which makes every submission harder. A claim matrix lets the team narrow a US claim, split an
EU claim, or defer a market. It also prevents the error of treating US clearance as evidence
of EU conformity.

**Procedure.** (1) List target markets and current status. (2) Write the claim in each
market's regulatory language. (3) Identify differences in user, population, setting, and
output. (4) Identify differences in evidence standard. (5) Decide whether to harmonize or
separate claims.

**Branches and edge cases.** Materially different claims need separate labeling and evidence
plans. Identical claims can still produce different classes. A US cleared claim does not
authorize an EU diagnostic claim. A CE-marked product may still need a US submission. A UK
product needs the applicable UK route, not just CE marking.

### Step 8 — Find the US classification regulation and product code

**Input.** Intended use, technology, and clinical domain.

**Output.** Candidate 21 CFR regulation, product code, class, and submission type.

**Rationale.** The US system is organized by regulation and product code, not abstract risk. A
product code links to a regulation, a class, and often guidance. It is the entry point to the
510(k) database and the FDA artificial intelligence (AI)-enabled device list. Finding the
right code early prevents a predicate search in the wrong neighborhood.

**Procedure.** (1) Search the FDA product classification database by domain and output. (2)
Read the regulation description, not only the code name. (3) Check class and submission type.
(4) Search the 510(k) database for cleared devices with that code. (5) Check the AI device
list and device-specific guidance. (6) Record candidates and reasons.

**Branches and edge cases.** A code match suggests Class II and a 510(k), but does not
guarantee equivalence. No matching code with low or moderate risk suggests De Novo. A Class
III code suggests premarket approval (PMA). An exempt code still carries general controls. One
product can fit several codes; intended use decides.

### Step 9 — Assign the US class and controls

**Input.** Candidate regulation and product code.

**Output.** Provisional class (I, II, or III), control set, and submission route.

**Rationale.** US class is a legal category tied to controls. Class I devices are subject to
general controls, Class II to general and special controls, and Class III to premarket
approval. The class reflects the controls needed for reasonable assurance of safety and
effectiveness, not the elegance of the algorithm.

**Procedure.** (1) Read the classification regulation. (2) Identify general controls. (3)
Identify special controls, including performance standards and labeling. (4) Identify
submission type and exempt status. (5) Check 510(k), De Novo, or PMA eligibility. (6) Record
the control set and evidence implications.

**Branches and edge cases.** Class I exempt devices still require registration, listing, and
accurate labeling. Class II usually means a 510(k) unless exempt; no predicate may mean De
Novo. Class III usually means PMA or a humanitarian device exemption. A Class II device with
new technology may still need clinical data.

### Step 10 — Qualify the product under the EU MDR or IVDR

**Input.** EU intended purpose, mode of action, and specimen or patient input.

**Output.** MDR device, IVDR device, accessory, or out-of-scope product.

**Rationale.** EU qualification is separate from US classification. The MDR and IVDR have
their own definitions, exclusions, and classification rules. A US device may be out of scope
in the EU, and a US non-device may be an EU device.

**Procedure.** (1) Read MDR Article 2(1) or IVDR Article 2(2). (2) Map intended purpose to the
definition. (3) Check exclusions and the product's relationship to a device. (4) Determine
accessory or component status. (5) Record qualification and legal basis.

**Branches and edge cases.** A medical purpose that is not an IVD points to the MDR; a
specimen-examination purpose points to the IVDR. An accessory is classified in its own right.
General wellness, cosmetic, and lifestyle products are not automatically out of scope. A
consumer wearable that predicts a condition may still yield a device.

### Step 11 — Apply MDR Rule 11 and the implementing rules

**Input.** EU intended purpose and the clinical consequence of the output.

**Output.** Provisional MDR class (I, IIa, IIb, or III) with the rule path.

**Rationale.** Rule 11 is the software-specific rule in Annex VIII. It distinguishes software
providing information for diagnostic or therapeutic decisions from software monitoring
physiological processes, and sets class by the seriousness of possible harm. Chapter II
implementing rules modify the result, especially Rule 3.3 for software that drives or
influences a device and Rule 3.5 for conflicting rules.

**Procedure.** (1) Decide whether the software provides decision information, monitors
physiological processes, or neither. (2) For decision information, assess whether possible
harm is death or irreversible deterioration, serious deterioration or surgical intervention,
or other harm. (3) For monitoring, assess whether parameters are vital and whether variation
could cause immediate danger. (4) Apply Rule 3.3. (5) Apply Rule 3.5 and take the highest
class. (6) Check MDCG 2019-11 examples and MDCG 2021-24.

**Branches and edge cases.** Decision support with possible death or irreversible
deterioration is Class III; serious deterioration or surgery is IIb; other harm is IIa.
Vital-parameter monitoring with immediate danger is IIb; non-vital monitoring is IIa; neither
is Class I. The same output can be IIa or IIb depending on population and action. Software
driving a Class III device takes Class III under Rule 3.3.

### Step 12 — Apply IVDR classification if the product is an IVD

**Input.** IVD intended purpose, analyte, specimen, and clinical use.

**Output.** Provisional IVDR class (A, B, C, or D) with the rule path.

**Rationale.** The IVDR uses a different system from the MDR. Class A is lowest and Class D
highest, with rules based on individual and public-health risk of an incorrect result.
Applying MDR classes to an IVD is a category error that leads to the wrong conformity
assessment route.

**Procedure.** (1) Confirm IVD status. (2) Identify analyte, specimen, and clinical purpose.
(3) Apply Annex VIII rules in order. (4) Determine self-testing, near-patient, or professional
use. (5) Record class and rule path. (6) Check companion-diagnostic and high-risk-analyte
rules.

**Branches and edge cases.** Class D covers the highest individual and public-health risk; C
high individual or moderate public-health risk; B moderate individual or low public-health
risk; A low risk. A software module interpreting an IVD result may itself be an IVD. A
rare-disease test can still be Class D if the result drives a high-risk treatment decision.

### Step 13 — Assign an IMDRF category as a risk communication aid

**Input.** Intended use and seriousness of the condition.

**Output.** An International Medical Device Regulators Forum (IMDRF) category (I–IV) and
rationale.

**Rationale.** The IMDRF framework communicates the significance of the information and the
state of the healthcare situation. It is not a regulatory classification and does not
determine US or EU class. Treating it as a classification is a common slide-deck error.

**Procedure.** (1) Classify significance as treat or diagnose, drive clinical management, or
inform clinical management. (2) Classify the situation as critical, serious, or non-serious.
(3) Read the category from the IMDRF N12 matrix. (4) Record the non-regulatory caveat. (5)
Explain divergences from US and EU results.

**Branches and edge cases.** Critical plus treat/diagnose is IV; critical plus drive
management is III; serious plus treat/diagnose is III; serious plus drive management is II;
non-serious plus treat/diagnose is II; other combinations are I or II. A Category IV product
can be US Class II if a predicate exists. Divergence reflects different legal systems, not an
error.

### Step 14 — Reconcile, document, and challenge

**Input.** Provisional results from Steps 1–13.

**Output.** A classification memo with conclusion, alternatives, open questions, and action
plan.

**Rationale.** A classification is only as good as its documented reasoning. The memo should
let a reviewer reproduce the conclusion and see what would change it. That makes it useful for
a pre-submission, a 513(g) request, a notified body opinion, or an internal go/no-go decision.

**Procedure.** (1) Write function-level intended use. (2) State device status and legal basis
per jurisdiction. (3) State class, rule path, and controls. (4) List evidence implications.
(5) List open questions and resolving facts. (6) Identify needed external advice. (7) Record
dates and source versions. (8) Have a second reviewer attack the weakest step.

**Branches and edge cases.** A stable conclusion proceeds to evidence planning. A conclusion
resting on one ambiguous phrase should trigger a claim revision. Divergent jurisdictions
require a decision to harmonize or separate. A borderline product needs external advice before
evidence investment. A product can be a device in one function and not another, Class II in
the US and Class III in the EU, or cleared in the US and still need an EU notified body.

### Decision table: device status by function

| Function | US status | EU status | Reason |
|---|---|---|---|
| Claims denial dashboard | Not a device | Out of scope | Administrative |
| Bed capacity forecast | Not a device | Out of scope | Operational |
| Sepsis risk score for clinicians | Device | MDR device | Patient-specific decision support |
| Wearable arrhythmia detection | Device | MDR device | Monitoring and diagnosis |
| Autonomous retinopathy screening | Device | MDR device | Diagnosis without review |
| Documentation summarizer | Depends | Depends | Summary versus diagnosis |
| Patient triage chatbot | Likely device | Likely MDR device | Directs care |
| Patient-specific dose calculator | Device | MDR device | Treatment safety |

### Decision table: software category

| Architecture | Medical purpose | Category | Lifecycle |
|---|---|---|---|
| Standalone cloud service | Yes | SaMD | IEC 62304, ISO 14971, IEC 62366-1 |
| App controlling a pump | Yes | SiMD | Device lifecycle and IEC 62304 |
| Imaging firmware | Yes | Device software | Device lifecycle and IEC 62304 |
| Manufacturing analytics | No | Production software | Computer software assurance |
| Result-display portal | Depends | Non-device or device | Claim-specific |
| Algorithm sold as a kit | Yes | Component or SaMD | Contract and intended use |
| Trial randomization | No | Research software | Trial regulations |
| Endpoint adjudication | Depends | Device or research software | Claim and use |

### Decision table: US class and route

| Class | Controls | Common route | Evidence emphasis |
|---|---|---|---|
| I | General | Exempt or 510(k) | Labeling, registration, listing |
| II | General and special | 510(k), De Novo if no predicate | Performance, often clinical |
| III | Premarket approval | PMA | Valid scientific evidence, manufacturing |
| Unclassified | Case-by-case | De Novo or PMA | Benefit-risk and novelty |

### Decision table: EU MDR Rule 11

| Software role | Possible harm | Class |
|---|---|---|
| Decision information | Death or irreversible deterioration | III |
| Decision information | Serious deterioration or surgery | IIb |
| Decision information | Other harm | IIa |
| Physiological monitoring | Vital parameter, immediate danger | IIb |
| Physiological monitoring | Non-vital parameter | IIa |
| Neither | Any | I |

### Decision table: IVDR class

| Risk profile | Class | Conformity assessment |
|---|---|---|
| Low individual and public-health risk | A | Self-declaration for most |
| Moderate individual or low public-health risk | B | Notified body |
| High individual or moderate public-health risk | C | Notified body |
| Highest individual and public-health risk | D | Notified body, stricter scrutiny |

The tables summarize; the steps are the audit trail. A reviewer who sees only the table cannot
tell whether the team considered accessory status, the IVD distinction, or the
hardware-software boundary. A reviewer who sees the steps can challenge a specific inference
rather than the conclusion as a whole.

## 6. Worked example

The case is hypothetical and is used to show the reasoning, not to describe a real clearance.
The product, sponsor, predicate label, and study numbers are constructed. Real determinations
require the current FDA database, the current EU guidance, and professional regulatory review.

### 6.1 Product and claim

RetinaTrace is a cloud-based software product that analyzes color fundus photographs. A
trained ophthalmic photographer captures one macula-centered and one disc-centered image per
eye in a primary-care or community screening setting. The software returns one of three
outputs per eye: "no referable diabetic retinopathy detected," "referable diabetic retinopathy
suspected," or "image ungradable." The output is displayed to a primary-care clinician or a
diabetic eye-screening grader. The product does not diagnose diabetic retinopathy, does not
recommend a treatment, and does not replace a dilated eye examination.

The draft intended-use statement is:

"RetinaTrace is intended for use by trained primary-care clinicians and diabetic eye-screening
graders to prioritize adults aged 22 years and older with diabetes and no known diabetic
retinopathy for specialist ophthalmic evaluation. The software analyzes color fundus
photographs and identifies images that are likely to show referable diabetic retinopathy or
that are ungradable. The output is an aid to referral prioritization and is not a diagnosis.
RetinaTrace is not intended for use in patients with known diabetic macular edema, prior
retinal surgery, or symptoms requiring urgent ophthalmic assessment."

The claim contains four regulatory choices. First, the user is a clinician or trained grader,
not the patient. Second, the population is adults with diabetes and no known retinopathy.
Third, the action is referral prioritization, not diagnosis or treatment. Fourth, the
limitations exclude known disease and urgent symptoms. Each choice narrows the evidence
burden. Each also reduces the market.

### 6.2 Function-level analysis

The product has three functions that must be analyzed separately. The image-quality function
decides whether an image is gradable. The retinopathy-detection function assigns a likelihood
of referable disease. The workflow function displays a priority queue and a referral
recommendation. The first two are medical device functions because they interpret
patient-specific images for a clinical decision. The workflow display is part of the same
intended use and does not escape regulation merely because it is a user-interface feature.

The image-quality function is not a separate product in this example, but it is a separate
failure mode. If it labels a poor-quality image as gradable, the detection function receives
an input outside its validated domain. If it labels a good image as ungradable too often, the
workflow becomes unusable and clinicians may ignore it. Both outcomes belong in the risk file
even though the product has one intended use.

### 6.3 US classification and pathway

The sponsor searches the FDA product classification database for retinal-image analysis and
diabetic retinopathy. It finds a product code for retinal image analysis software used to
detect or triage diabetic retinopathy, with Class II and a 510(k) submission requirement. The
sponsor identifies three legally marketed predicates with closely related intended uses. None
has the exact same user, population, and referral claim, but one has a clinician-facing
retinal image triage claim and a similar imaging modality. The sponsor records the predicate
identifiers and the comparison in a predicate matrix. It does not rely on a clearance number
remembered from a conference slide; it verifies each predicate in the current database.

The US pathway is a 510(k) with substantial equivalence to the closest predicate. The intended
use is similar: clinician-facing analysis of fundus images for diabetic retinopathy. The
technological characteristics are similar: a machine-learning classifier applied to color
fundus photographs. The differences are the three-output design, the image-quality function,
and the specific referral-prioritization wording. The sponsor must show that these differences
do not raise different questions of safety and effectiveness, or address them with performance
data.

The US class is Class II because the product code is Class II and the risk is moderate: a
missed referral can lead to delayed treatment, but the clinician retains responsibility and
the disease is typically slow to progress. The special controls likely include performance
testing, labeling, and postmarket surveillance. The sponsor plans a 510(k) with analytical
validation, a multi-reader multi-case study, subgroup performance, and human-factors evidence.
The sponsor does not assume that a Class II code means no clinical evidence. The code supplies
the class; the claim supplies the evidence question.

### 6.4 EU qualification and classification

The sponsor writes the EU intended purpose separately. The product is a medical device under
MDR Article 2(1) because it analyzes patient images to inform a clinical decision about
referral. It is not an IVD because it does not examine a specimen derived from the human body.
It is not an accessory because it has an independent medical purpose and is not sold to enable
a specific host device. It is standalone software, so it is SaMD.

Under MDR Annex VIII Rule 11, the software provides information used to take a diagnostic or
therapeutic decision. The possible harm from a false-negative result is delayed diagnosis of
sight-threatening retinopathy, which can lead to serious deterioration and may require
surgical intervention. On the facts of this claim, the sponsor classifies the product as Class
IIb. The classification memo records the alternative view that a referral-prioritization tool
with a clinician in the loop might be Class IIa, and explains why the sponsor chose IIb: the
claim includes patients with no known retinopathy, the output can suppress referral, and the
consequence of a missed case is serious. The memo also records that a notified body may take a
different view.

The EU conformity assessment route for Class IIb software is generally Annex IX (quality
system plus technical documentation assessment) or Annex XI (production quality assurance),
with a notified body. The sponsor needs a quality management system, technical documentation,
a clinical evaluation, a post-market surveillance plan, and a post-market clinical follow-up
plan. The sponsor also needs to check the EU artificial intelligence (AI) Act because the
product is an AI system. The current AI Act timeline must be verified; the medical-device
classification and the AI Act risk classification are separate analyses.

### 6.5 Predicate and evidence gap

The closest US predicate supports a two-output "disease suspected / not suspected" claim for a
similar population. RetinaTrace adds an "ungradable" output and a referral-priority workflow.
The predicate's public summary reports sensitivity of 0.87 and specificity of 0.85 on a
curated set, but it does not report performance in patients with darkly pigmented skin,
patients over 75, or patients with concurrent cataract. RetinaTrace's training set has 18,400
images from three sites. Only 6.2 percent of images come from patients with darkly pigmented
skin, and only 4.8 percent from patients over 75. The sponsor's pre-specified subgroup
analysis requires at least 200 images per subgroup.

The evidence gap is not "does the model work." The gap is "does the model work well enough, in
the intended population, to support a referral-prioritization claim, and does the labeling
make the limits clear?" The sponsor needs four pieces of evidence: analytical validation on a
locked algorithm, a reader study that measures the effect of the output on referral decisions,
a subgroup analysis that meets the pre-specified precision, and a human-factors study that
shows users do not misinterpret "no referable disease" as "no disease."

The arithmetic exposes a communication risk. Suppose the intended-use prevalence of referable
retinopathy is 12 percent, sensitivity is 0.90, and specificity is 0.88. The positive
predictive value is 0.108 divided by 0.108 plus 0.1056, or 0.506. The negative predictive
value is 0.7744 divided by 0.7744 plus 0.012, or 0.985. Half of the "suspected" outputs are
false positives, while only 1.5 percent of "not suspected" outputs are false negatives. A
clinician who sees a "suspected" result may over-refer; a clinician who sees "not suspected"
may under-refer if the label implies absence of disease. The labeling must state that a
negative output does not rule out retinopathy and that ungradable images require a different
pathway.

### 6.6 The submission decision

The sponsor decides to proceed with a US 510(k) and an EU Class IIb conformity assessment, but
not with the original claim. The original claim says the product "identifies patients who do
not need an eye exam." That wording implies a rule-out claim the evidence cannot support. The
revised claim says the product "prioritizes patients for specialist evaluation and does not
rule out retinopathy." The revised claim aligns with the intended use, the reader study, and
the label. The sponsor also removes a marketing sentence that says the product "replaces
manual grading," because the validation compares the product plus a clinician with a clinician
alone, not with a grading service.

The US submission includes a 510(k) summary, a predicate comparison, software documentation,
cybersecurity documentation, a human-factors report, analytical validation, a reader study,
and labeling. The EU technical documentation includes the same core evidence plus a clinical
evaluation report, a post-market surveillance plan, and a PMCF plan. The sponsor files a
pre-submission to discuss the reader-study endpoint and the ungradable-image handling before
locking the protocol. The pre-submission is not a guarantee of clearance, but it reduces the
risk of a preventable deficiency.

### 6.7 Two diagnostics that would sink the submission

**Diagnostic 1: subgroup sensitivity falls below the pre-specified floor.** The reader study
enrolls 240 cases, but only 38 are from patients with darkly pigmented skin. In that subgroup,
sensitivity is 0.71 with a 95 percent confidence interval from 0.54 to 0.85. The overall
sensitivity is 0.90. The sponsor's pre-specified subgroup floor is 0.80. The overall result
looks acceptable, but the subgroup result shows that the product may miss disease in a
population that is explicitly in the intended use. A reviewer can reasonably conclude that the
evidence does not support the claim across the intended population. The submission is not
saved by a post-hoc claim that the subgroup was "exploratory" if the protocol pre-specified it
and the labeling does not exclude the subgroup.

**Diagnostic 2: the reader study does not isolate the effect of the device.** The study
compares readers who use RetinaTrace with readers who use a sham interface, but the
RetinaTrace arm also receives a one-hour training session and a referral checklist. The
control arm receives neither. The improvement in referral accuracy cannot be attributed to the
device. A reviewer will ask whether the training or the checklist produced the effect. The
protocol must either equalize the co-interventions or use a design that isolates the device. A
second problem is that the reference standard is a single grader's opinion rather than a
clinical consensus or a masked adjudication. Either defect can sink the claim because the
study does not measure what the submission says it measures.

### 6.8 Vary one assumption

The base case is a clinician-facing triage claim, US Class II 510(k), EU Class IIb. Change the
intended-use wording from "prioritize for specialist evaluation" to "autonomously determine
whether the patient has referable diabetic retinopathy and should be treated." The user
becomes the patient or a screening technician, the action becomes diagnosis and treatment
referral without clinician review, and the population remains adults with diabetes.

The US pathway changes from a 510(k) to a De Novo request or a PMA, depending on the closest
predicate and the risk. A 510(k) requires a legally marketed predicate with the same intended
use; an autonomous diagnostic claim is unlikely to be substantially equivalent to a
clinician-facing triage predicate. The De Novo route is plausible for a novel Class II device
with special controls, as the IDx-DR precedent shows for autonomous diabetic retinopathy
screening. A PMA would be required if the device is Class III. The sponsor must not assume
that a De Novo is available simply because the technology is novel; the risk classification
still governs.

The EU class changes from IIb to III if the missed diagnosis can cause death or irreversible
deterioration. Autonomous diagnosis of sight-threatening retinopathy can lead to irreversible
vision loss, so the sponsor should analyze Class III. The conformity assessment becomes
stricter, and a clinical investigation is generally required unless the sponsor can rely on
equivalence with sufficient access to the comparator's technical documentation. The evidence
burden grows from a reader study to a prospective clinical performance study with a clinically
meaningful endpoint, plus a PMCF plan.

The variation table makes the dependency explicit.

| Assumption changed | US pathway | EU class | Evidence consequence |
|---|---|---|---|
| Base: clinician triage, predicate available | 510(k), Class II | IIb | Analytical, reader study, human factors |
| Autonomous diagnosis, predicate unavailable | De Novo or PMA | III if irreversible harm possible | Prospective clinical study, stricter PMCF |
| Predicate withdrawn or not equivalent | De Novo | IIb or III | Standalone safety and effectiveness evidence |
| Claim narrowed to image quality only | Likely 510(k) or exempt | Likely IIa | Analytical and usability evidence |
| Claim broadened to treatment recommendation | PMA likely | III likely | Clinical outcome evidence and strict controls |
| Population narrowed to specialist setting | 510(k) | IIa or IIb | Evidence in the narrowed population |

The lesson is not that one pathway is better. The lesson is that a wording change is a
regulatory change. The same model, the same data, and the same code can require a different
predicate, a different class, a different study, and a different post-market plan. The team
should decide the claim before it decides the architecture, the training data, or the launch
plan.

## 7. Code, query, or calculation

The script below turns the decision tables into executable functions. It is deliberately small
enough to audit. It covers the US route, a simplified MDR Rule 11 tree, a simplified IVDR
table, the IMDRF matrix, and the predictive-value arithmetic used in Section 6. The IVDR table
is a teaching subset, not the full Annex VIII. The MDR function does not implement Rule 3.3;
callers must supply the class of the device that the software drives, if any.

```python
"""First-pass classification helpers for Module 14, Topic 01.

These functions encode the decision tables in the topic file. They are
teaching aids, not legal advice.
"""

from __future__ import annotations


def classify_us(device_class: int, predicate_available: bool) -> str:
    """Return a provisional US route from class and predicate availability."""
    if device_class == 1:
        return "Class I: general controls; exempt or 510(k) depending on code"
    if device_class == 2:
        if predicate_available:
            return "Class II: 510(k) with a suitable predicate"
        return "Class II: De Novo (no legally marketed predicate)"
    if device_class == 3:
        return "Class III: PMA unless an eligible alternative applies"
    return "Unclassified: De Novo or PMA after risk analysis"


def classify_eu_mdr(role: str, harm: str) -> str:
    """Apply a simplified MDR Rule 11 tree.

    role: 'decision', 'monitor_vital', 'monitor_nonvital', or 'other'
    harm: 'death_or_irreversible', 'serious_or_surgery', 'other', or 'none'
    """
    role = role.strip().lower()
    harm = harm.strip().lower()
    if role == "decision":
        if harm == "death_or_irreversible":
            return "Class III"
        if harm == "serious_or_surgery":
            return "Class IIb"
        return "Class IIa"
    if role == "monitor_vital":
        return "Class IIb"
    if role == "monitor_nonvital":
        return "Class IIa"
    if role == "other":
        return "Class I"
    raise ValueError(f"unknown role: {role}")


def classify_ivdr(individual_risk: str, public_health_risk: str) -> str:
    """Return a simplified IVDR class from individual and public-health risk."""
    pair = (individual_risk.strip().lower(), public_health_risk.strip().lower())
    table = {
        ("low", "low"): "Class A",
        ("moderate", "low"): "Class B",
        ("high", "low"): "Class C",
        ("high", "moderate"): "Class C",
        ("high", "high"): "Class D",
    }
    if pair not in table:
        raise ValueError(f"risk pair not in teaching table: {pair}")
    return table[pair]


def imdrf_category(significance: str, situation: str) -> str:
    """Return the IMDRF N12 category. This is not a legal classification."""
    significance = significance.strip().lower()
    situation = situation.strip().lower()
    matrix = {
        ("treat_or_diagnose", "critical"): "IV",
        ("treat_or_diagnose", "serious"): "III",
        ("treat_or_diagnose", "non_serious"): "II",
        ("drive_management", "critical"): "III",
        ("drive_management", "serious"): "II",
        ("drive_management", "non_serious"): "I",
        ("inform_management", "critical"): "II",
        ("inform_management", "serious"): "I",
        ("inform_management", "non_serious"): "I",
    }
    if (significance, situation) not in matrix:
        raise ValueError(f"unknown IMDRF pair: {(significance, situation)}")
    return matrix[(significance, situation)]


def predictive_values(sensitivity: float, specificity: float, prevalence: float) -> dict[str, float]:
    """Compute PPV and NPV from sensitivity, specificity, and prevalence."""
    for name, value in {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "prevalence": prevalence,
    }.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    tp = sensitivity * prevalence
    fn = (1.0 - sensitivity) * prevalence
    fp = (1.0 - specificity) * (1.0 - prevalence)
    tn = specificity * (1.0 - prevalence)
    ppv = tp / (tp + fp)
    npv = tn / (tn + fn)
    return {"ppv": ppv, "npv": npv, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


if __name__ == "__main__":
    cases = [
        ("US Class II with predicate", classify_us(2, True)),
        ("US Class II without predicate", classify_us(2, False)),
        ("EU decision, serious harm", classify_eu_mdr("decision", "serious_or_surgery")),
        ("EU decision, death or irreversible", classify_eu_mdr("decision", "death_or_irreversible")),
        ("EU monitor, vital", classify_eu_mdr("monitor_vital", "none")),
        ("IVDR high/high", classify_ivdr("high", "high")),
        ("IMDRF critical treat", imdrf_category("treat_or_diagnose", "critical")),
    ]
    for label, result in cases:
        print(f"{label:36s} -> {result}")

    values = predictive_values(sensitivity=0.90, specificity=0.88, prevalence=0.12)
    print("\nPredictive values at sens=0.90, spec=0.88, prevalence=0.12")
    print(f"PPV = {values['ppv']:.3f}")
    print(f"NPV = {values['npv']:.3f}")
    print(
        "Cells per 1,000 patients: "
        f"TP={values['tp'] * 1000:.1f}, FP={values['fp'] * 1000:.1f}, "
        f"FN={values['fn'] * 1000:.1f}, TN={values['tn'] * 1000:.1f}"
    )
```

Expected output from a Python 3.12 run:

```text
US Class II with predicate           -> Class II: 510(k) with a suitable predicate
US Class II without predicate        -> Class II: De Novo (no legally marketed predicate)
EU decision, serious harm            -> Class IIb
EU decision, death or irreversible   -> Class III
EU monitor, vital                    -> Class IIb
IVDR high/high                       -> Class D
IMDRF critical treat                 -> IV

Predictive values at sens=0.90, spec=0.88, prevalence=0.12
PPV = 0.506
NPV = 0.985
Cells per 1,000 patients: TP=108.0, FP=105.6, FN=12.0, TN=774.4
```

Three cautions apply. First, the `classify_us` function treats class and predicate
availability as independent inputs. In a real analysis they are not independent: the product
code supplies the class, and the predicate search tests whether a device with the same
intended use is legally marketed. Second, `classify_eu_mdr` does not model Rule 3.3. If the
software drives a Class III device, the software is Class III under Rule 3.3 even if the Rule
11 tree returns IIa. Third, `predictive_values` assumes the sensitivity and specificity are
transportable to the intended-use prevalence. That assumption is often false. The function is
a communication tool, not a substitute for external validation.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and is not legal advice. Device status and classification are fact-specific legal conclusions. Confirm the current statutes, regulations, guidance, and database records with qualified regulatory counsel before relying on them.

The legal sources below are binding unless marked as guidance or standard. A standard becomes
binding when a regulation incorporates it, when a contract requires it, or when a manufacturer
chooses it and makes it part of its quality system. Guidance describes an agency's current
thinking; it is not a statute, but a submission that ignores it without an alternative
rationale invites questions.

| Source | Status | What it constrains | What it does not do |
|---|---|---|---|
| FD&C Act §201(h) | Binding US statute | Defines a device by intended use | Does not classify the device or choose a pathway |
| FD&C Act §520(o) | Binding US statute | Excludes certain software functions from the device definition | Does not create a general exemption for clinical software |
| 21 CFR 801.4 | Binding US regulation | Defines intended use and the evidence of objective intent | Does not decide whether a specific claim is a device claim |
| 21 CFR Parts 862–892 | Binding US regulations | Classify device types and set controls | Do not cover every new technology |
| FDA CDS guidance, final September 28, 2022 | Guidance | Describes FDA's approach to CDS software and section 520(o) | Is not a safe harbor and does not bind a court |
| FDA device software policy, September 28, 2022 | Guidance | Describes oversight of device software functions and enforcement discretion | Does not change the statutory definition |
| MDR 2017/745, Article 2 and Annex VIII | Binding EU regulation | Defines medical devices and sets classification rules | Does not classify a product without a intended purpose |
| MDCG 2019-11 Rev.1, June 2025 | EU guidance | Explains qualification and classification of medical device software | Is not binding on manufacturers, notified bodies, or courts |
| MDCG 2021-24 Rev.1 | EU guidance | Explains the Annex VIII classification rules | Is not a substitute for the legal text |
| IVDR 2017/746, Article 2 and Annex VIII | Binding EU regulation | Defines IVDs and sets IVD classes | Does not apply to non-IVD software |
| IMDRF N12:2014 | International guidance | Provides a risk-categorization framework | Is not a legal classification in any jurisdiction |
| ISO 13485:2016 | Voluntary standard | Sets quality management system requirements | Does not itself establish device status or class |
| ISO 14971:2019 | Voluntary standard | Sets risk management requirements | Does not replace the legal classification |
| IEC 62304:2006+A1:2015 | Voluntary standard | Sets software lifecycle processes by safety class | Does not set the regulatory device class |
| IEC 62366-1:2015+A1:2020 | Voluntary standard | Sets usability engineering requirements | Does not establish clinical effectiveness |

The US analysis starts with the statute. Section 201(h) defines a device as an instrument,
apparatus, implement, machine, contrivance, implant, in vitro reagent, or other similar or
related article, including a component part or accessory, intended for use in the diagnosis of
disease or other conditions, or in the cure, mitigation, treatment, or prevention of disease,
or intended to affect the structure or any function of the body. The definition is functional.
The same physical object can be a device or not depending on the claim.

The 21st Century Cures Act, enacted in 2016, amended section 520 of the FD&C Act to exclude
certain software functions from the device definition. The exclusions are narrow and
fact-specific. They cover some administrative support functions, some lifestyle and wellness
functions, and certain clinical decision support functions that meet statutory criteria. The
FDA's CDS guidance, finalized on September 28, 2022, explains how the agency evaluates those
criteria. The guidance is not a checklist that a product can self-certify against. A vendor
that writes "CDS" on a slide has not established an exclusion.

The EU analysis starts with the MDR. Article 2(1) defines a medical device by intended medical
purpose. The MDR applies to devices and accessories placed on the EU market. The IVDR applies
to in vitro diagnostic medical devices. The two regulations have separate classification
systems. The MDR's Annex VIII Rule 11 is the software-specific rule, but it is not the only
rule that can apply. Rule 3.2 requires accessories to be classified in their own right. Rule
3.3 requires software that drives or influences a device to take the same class as that
device. Rule 3.5 requires the strictest class when several rules apply.

The MDCG 2019-11 Rev.1 guidance, dated June 2025, is the most detailed EU software
qualification and classification guidance. It defines medical device software, provides
decision trees, splits Rule 11 into decision-support, monitoring, and other software, and maps
IMDRF categories to MDR classes. The mapping is a communication aid; the legal class still
comes from the rule. The MDCG 2021-24 Rev.1 guidance explains the general classification rules
and is useful for checking whether a rule other than Rule 11 applies. Neither document is
binding, but a notified body will expect the manufacturer to explain any departure from them.

The IMDRF N12:2014 framework is often misused. It places software in one of four categories
based on the significance of the information and the seriousness of the healthcare situation.
The category is an impact category, not a regulatory class. It does not tell you whether a
product is a device, which US pathway applies, or which EU class applies. It is useful for
aligning risk language across a team and for explaining why two products with similar
algorithms can have different clinical consequences.

Two standards are relevant even at the classification stage. ISO 13485:2016 defines quality
management system requirements, and ISO 14971:2019 defines risk management requirements.
Neither standard determines device status. They become relevant after the product is a device,
and they shape the evidence that a submission will expect. IEC 62304:2006+A1:2015 defines
software safety classes A, B, and C. Those classes are not the same as US or EU regulatory
classes. A software safety class describes the possible consequence of a software failure; a
regulatory class describes the controls required for the device.

The data sources for a classification memo are as important as the legal sources. The FDA
product classification database provides product codes, regulations, and classes. The 510(k)
database provides cleared devices and their intended uses. The FDA AI-enabled device list
provides a view of authorized AI-enabled devices, but it is not exhaustive and it changes. The
EUDAMED database, once fully populated, provides device registrations and certificates. The
notified body's scope and the manufacturer's technical documentation provide the EU evidence.
A memo that cites a blog post instead of the database is not reproducible.

The most common data error is treating a product code as a classification decision. A product
code is a bucket. The regulation description and the intended use decide whether the product
belongs in the bucket. A second error is treating a 510(k) clearance as evidence that the
product is safe and effective for every use in the code. A clearance is tied to a specific
intended use, a specific technology, and a specific label. A third error is treating a CE mark
as a global passport. It is not.

## 9. Industry practice and case evidence

Public case evidence is uneven. FDA publishes De Novo and PMA summaries and some 510(k)
summaries, but many 510(k) records contain limited performance detail. Companies publish case
studies that emphasize success. The result is a literature in which the failures are easier to
find in post-market studies than in premarket documents. The cases below separate what is
documented from what is inference.

**IDx-DR.** The FDA granted De Novo authorization DEN180001 for IDx-DR, an autonomous AI
system that analyzes retinal images to detect diabetic retinopathy. The authorization was
reported in 2018. The device was intended for use by health care providers in primary-care
settings, and the labeling specified the population and the follow-up pathway. The public
review document describes a prospective clinical study and a narrow intended use. The
documented lesson is that an autonomous claim can be authorized at Class II when the intended
use, population, and workflow are tightly defined and the evidence supports them. The
inference, not stated by FDA, is that the narrowness of the claim was as important as the
algorithm.

**Apple ECG.** The FDA granted De Novo authorization DEN180044 for the Apple Watch ECG app in
2018. The authorization covered irregular rhythm notification and ECG recording, with labeling
that described the device as a screening tool, not a diagnostic replacement. The documented
lesson is that a consumer-facing product can be a regulated device when it makes a medical
claim. The inference is that the consumer brand did not lower the evidence standard; the
narrow claim and the labeling did the work.

**Viz.ai large-vessel occlusion triage.** Viz.ai received FDA clearance for software that
analyzes computed tomography angiography images and alerts care teams to a suspected
large-vessel occlusion. The public record and company materials describe a workflow product,
not only a detector. The documented lesson is that the clinical benefit may come from
time-to-treatment and care coordination, not from stand-alone accuracy. The inference is that
a classifier with modest stand-alone performance can still support a claim if the workflow
evidence shows a meaningful change in action. The current 510(k) number and indications should
be checked in the FDA database rather than taken from a secondary summary.

**Epic Sepsis Model.** Wong et al. published an external validation of a widely deployed
sepsis prediction model in JAMA Internal Medicine in 2021. The study reported that the model's
discrimination in the external health system was lower than the values reported by the vendor,
and that many alerts would be generated at the deployed threshold. The documented lesson is
that regulatory status, if any, does not establish local performance. The inference is that
health systems need local validation and monitoring even when a product is commercially
available and cleared for a related use. The paper is not a regulatory finding; it is a
performance finding.

**EU borderline practice.** The MDCG 2019-11 guidance includes examples of software that is
and is not medical device software. The examples are useful because they show how the same
technology can fall on different sides of the line depending on the claim. A software tool
that stores and displays images is generally not a device; a tool that analyzes images to
support a diagnosis is. A tool that calculates a score from a validated questionnaire may be a
device if the score is intended to inform a diagnosis or treatment. The examples are guidance,
not law, and a notified body may reach a different conclusion on different facts.

**Industry practice: classify before you build.** Mature device companies write a regulatory
strategy before the design freeze. They choose the intended use, the jurisdiction, the class,
the predicate strategy, and the evidence plan as a package. Startups often do the reverse:
build a general model, look for a predicate, then discover that the claim they want is broader
than the evidence they can afford. The public record does not quantify this failure rate. The
inference from recalls, warning letters, and submission deficiencies is that claim-evidence
misalignment is common.

**Industry practice: separate the clinical claim from the workflow claim.** Companies
increasingly sell a workflow benefit alongside an accuracy claim. The workflow benefit may be
easier to measure and more valuable to the customer, but it does not erase the device claim. A
product that says "reduces time to treatment" is making a clinical outcome claim that needs
evidence. A product that says "sorts worklists" may be making a workflow claim, but if the
sorting is based on a patient-specific medical prediction, the underlying prediction may still
be a device function.

**Industry practice: use a classification memo as a living document.** Companies that manage
regulatory change well keep a classification memo with version control. When the intended use
changes, the memo changes. When a new jurisdiction is added, the memo adds a row. When a
predicate is withdrawn, the memo records the fallback. The memo is not a one-time deliverable
for a submission; it is part of the design history file and the regulatory strategy.

**What the cases do not prove.** None of these cases proves that a particular pathway is
available for a new product. IDx-DR does not prove that every autonomous screening product can
use De Novo. Apple ECG does not prove that every consumer wearable is a device. Viz.ai does
not prove that every triage product can claim a time-to-treatment benefit. Epic does not prove
that every sepsis model fails externally. Each case is a data point about a specific claim,
population, and evidence package. The transferable lesson is the method: define the claim,
classify it under the applicable law, identify the evidence gap, and test the claim in the
intended population.

## 10. Failure modes and red flags

Classification errors rarely announce themselves. They appear as a confident slide, a product
name that implies a diagnosis, or a predicate chosen because it was the first search result.
The table lists the failures that most often survive until a reviewer or a regulator finds
them.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Claim inflation | Marketing says "detects" while the label says "supports" | Sales language becomes evidence of intended use | Device status or class broader than planned | Review all customer-facing text against the label |
| Self-declared CDS exclusion | Team says "we are CDS, so we are exempt" without a memo | Section 520(o) criteria are fact-specific | Unregulated device claim | Write a function-level exclusion analysis with counsel |
| No predicate search | Predicate chosen from a conference slide | Predicate may be cleared for a different use | 510(k) rejected as not substantially equivalent | Verify each predicate in the current database |
| Wrong product code | Code chosen by keyword, not regulation text | Code carries the wrong class and controls | Wrong pathway and evidence plan | Read the regulation and compare intended uses |
| Accessory inherits host class | App assumed to be part of a cleared device | MDR Rule 3.2 and US accessory policy differ | Missing submission or wrong class | Classify the accessory in its own right |
| SiMD treated as SaMD | Embedded software analyzed with standalone assumptions | Hardware and software failures interact | Incomplete risk analysis and lifecycle evidence | Analyze the system boundary and interfaces |
| IVD/device confusion | Blood-test interpretation software called a device app | MDR and IVDR have separate rules | Wrong regulation and conformity route | Determine whether a specimen is examined |
| Multiple uses in one claim | Product description lists diagnosis, triage, and treatment | One claim becomes broader than the evidence | Submission deficiency or post-market action | Split functions and classify separately |
| US clearance treated as EU approval | Launch plan assumes global readiness after 510(k) | Jurisdictions have independent regimes | Product cannot be sold in the EU | Build a jurisdiction-by-jurisdiction matrix |
| IMDRF category used as class | Slide shows Category IV and concludes Class III | IMDRF is a communication framework | Wrong pathway and evidence | Label it non-regulatory and apply local law |
| Stale predicate | Predicate was removed or changed after search | Substantial equivalence depends on current status | Submission rejected or delayed | Recheck the database before filing |
| Label overclaim | Label says "rules out" disease without evidence | Negative predictive claim exceeds study | Recall, warning letter, or liability | Match label to endpoint and evidence |

A red flag is not proof of a violation. It is a signal that the classification memo is missing
a fact. The right response is to add the fact, not to argue that the red flag is irrelevant. A
team that can explain why a red flag does not apply is in a stronger position than a team that
never saw it.

## 11. Assumptions and limits

Every classification rests on assumptions. The memo should state them, explain why they are
needed, describe what would break them, and name the test that would detect the break. The
table is a template for that ledger.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The intended use is stable | Classification follows the claim | A new market, user, or feature changes the claim | Version-control the intended-use statement and review every release |
| The legal text is current | Statutes and regulations change | A new rule, guidance, or court decision | Check the official source and record the access date |
| The predicate remains legally marketed | 510(k) equivalence depends on a predicate | Predicate withdrawn, modified, or reclassified | Recheck the FDA database before filing and before launch |
| The product code is representative | Code supplies class and controls | Code covers a different intended use | Compare the regulation text and cleared labels |
| The population matches the evidence | Subgroup performance can differ | Training data underrepresents the intended population | Pre-specify subgroup analysis and power it |
| The algorithm is frozen | Evidence is tied to a version | Model or threshold changes after validation | Configuration control and change impact assessment |
| Input quality is controlled | The model was validated on a certain image or signal quality | New cameras, protocols, or patient preparation | Analytical validation across sites and devices |
| Clinicians use the output as intended | The claim depends on human behavior | Users ignore, over-trust, or misinterpret the output | Formative and summative usability studies |
| The jurisdiction is accessible | A class only matters if the product can be sold | Notified body capacity, local registration, or trade barriers | Regulatory intelligence and early notified body engagement |

The limits of this topic are equally important. It does not provide a legal opinion. It does
not cover every product-specific classification regulation. It does not cover the UK, Canada,
Australia, Japan, or other markets in detail. It does not resolve borderline cases. It does
not tell you whether a product will be cleared or certified. It gives you a structured first
pass and a vocabulary for the next conversation.

A classification memo should end with a confidence statement. "We are confident that this is a
device in the US and the EU; we are less confident about the EU class because the harm
characterization depends on the referral pathway; we will seek a notified body opinion before
locking the clinical evaluation plan." That statement is more useful than a binary conclusion
because it tells the team where to spend the next week.

## 12. Exercises

The exercises move from recall to application to critique to design. Each has a deliverable,
acceptance criteria, and a time estimate. Use the current legal sources, not this file alone.

### Exercise 1 — Function inventory (recall and application, 25 minutes)

Take a product description of your choice that contains at least three user-facing outputs.
Produce a table with one row per output: output, user, clinical action, failure consequence,
and preliminary device status. Deliverable: a one-page table with a short note on any output
you could not classify. Acceptance criteria: every output has a named user and action; no
output is classified solely by product name; unresolved rows are marked unresolved rather than
guessed.

### Exercise 2 — Statutory mapping (application, 30 minutes)

Choose one function from Exercise 1 and map it to the elements of FD&C Act §201(h) and MDR
Article 2(1). Quote the relevant language, identify the element that is weakest, and state
what fact would strengthen or break the mapping. Deliverable: a two-column mapping with
citations. Acceptance criteria: citations are current and specific; the weakest element is
named; the answer distinguishes statute from guidance.

### Exercise 3 — Intended-use rewrite (critique, 40 minutes)

Find a public product description that mixes several claims. Rewrite it as three separate
intended-use statements, each with purpose, user, population, setting, output, and at least
one limitation. Then write a paragraph explaining which statement is most likely to be a
device and why. Deliverable: three statements plus a rationale. Acceptance criteria: each
statement is one sentence or a short paragraph; the negative limitation is explicit; the
rationale cites a legal source rather than a risk score.

### Exercise 4 — US pathway memo (application, 45 minutes)

For one product, identify two candidate product codes, compare their regulations and classes,
and select one with a written rationale. Then state whether a predicate search supports a
510(k), whether De Novo is plausible, or whether PMA is likely. Deliverable: a one-page memo
with a product-code table and a pathway recommendation. Acceptance criteria: both codes are
checked in the current database; the predicate discussion names the intended use and
technology comparison; the recommendation states its confidence.

### Exercise 5 — EU Rule 11 decision (application, 45 minutes)

Apply MDR Annex VIII Rule 11 to the same product. Write the decision tree you used, including
the role of the software, the possible harm, the class, and the effect of Rules 3.2, 3.3, and
3.5. Then assign an IMDRF category and explain why it is not the EU class. Deliverable: a
rule-path diagram and a short memo. Acceptance criteria: the rule text is quoted; the harm
characterization is explicit; the IMDRF category is labeled non-regulatory.

### Exercise 6 — Accessory and component analysis (critique, 35 minutes)

Take a product that is sold with a host device, such as an app that displays a sensor's output
or a module that adds a feature to an imaging system. Determine whether it is an accessory, a
component, or an independent device under the MDR and the US framework. Identify the facts
that would change the answer. Deliverable: a comparison table with a conclusion for each
jurisdiction. Acceptance criteria: the MDR Rule 3.2 and Rule 8 exception are addressed; the US
analysis does not assume that the host's clearance covers the add-on; the changing facts are
specific.

### Exercise 7 — Classification memo (design, 60 minutes)

Write a two-page classification memo for a fictional product. Include the function-level
intended use, US status and class, EU status and class, IVD screen, IMDRF category, evidence
implications, open questions, and a confidence statement. Deliverable: the memo with a source
table that labels each source as statute, regulation, guidance, standard, or convention.
Acceptance criteria: every conclusion has a citation; at least three open questions are
listed; the memo states what external advice is needed.

### Exercise 8 — Adversarial review (critique, 40 minutes)

Exchange a classification memo with a peer. Attack the weakest step: the claim, the predicate,
the harm characterization, the accessory analysis, or the population. Write five questions
that a regulator or notified body would ask, and identify which one the memo cannot answer.
Deliverable: five questions and a one-paragraph assessment. Acceptance criteria: questions are
specific to the memo, not generic; at least one question identifies a missing fact; the
assessment distinguishes a documentation gap from a substantive gap.

## 13. Self-check answers

**Question 1. What is the difference between a medical device and SaMD?** A medical device is
the legal category defined by the applicable statute or regulation. SaMD is a subset of
software that performs a medical purpose without being part of a hardware medical device. A
device can be hardware, software, or a combination. SaMD is always software, but not all
device software is SaMD; embedded software that drives a hardware device is SiMD or part of
the device.

**Question 2. Why is a function-level analysis necessary?** Because intended use is assessed
per function. A single platform can contain a non-device administrative dashboard and a device
clinical alert. If the platform is analyzed as one object, either the low-risk function is
over-regulated or the high-risk function is under-regulated. Function-level analysis also
makes the evidence plan specific.

**Question 3. What belongs in an intended-use statement?** The medical purpose, the user, the
patient population, the setting, the output, and the limitations or excluded uses. A strong
statement also says whether the output supports or replaces a clinical judgment and what the
user must independently review.

**Question 4. How do you choose a US pathway?** Start with the product code and regulation,
then check the class and submission type. If a legally marketed predicate supports the same
intended use and similar technology, a 510(k) may be available. If no predicate exists and the
risk is low or moderate, consider De Novo. If the device is Class III, PMA is likely. The
pathway is a legal conclusion, not a preference.

**Question 5. How does MDR Rule 11 work?** It asks whether the software provides information
for diagnostic or therapeutic decisions, monitors physiological processes, or does neither.
Decision support is IIa, IIb, or III depending on the seriousness of possible harm. Monitoring
is IIa or IIb depending on whether the parameter is vital and whether variation could cause
immediate danger. Other software is Class I. The Chapter II implementing rules can raise the
class.

**Question 6. Why can the same product be Class II in the US and Class IIb in the EU?** The
systems use different legal criteria. The US class comes from the product code and the
controls needed for reasonable assurance. The EU class comes from Annex VIII rules and the
harm characterization. A Class II US device can be IIa, IIb, or even III in the EU. The
difference is expected, not an inconsistency.

**Question 7. What is the IMDRF category used for?** It communicates the significance of the
information and the seriousness of the healthcare situation. It is useful for risk
discussions, training, and cross-functional alignment. It is not a legal classification and
does not determine a US or EU pathway.

**Question 8. What does it mean to label a source as binding, guidance, or convention?**
Binding sources create legal obligations: statutes, regulations, and incorporated standards.
Guidance describes an agency's current thinking and is not legally binding, though it shapes
expectations. Convention is local practice, such as a preferred template or a common review
habit. Mixing them in a memo makes a recommendation look more authoritative than it is.

**Question 9. Name three genuinely ambiguous classification situations.** A clinical
documentation tool that summarizes a note and also suggests a diagnosis; a patient-facing
chatbot that triages symptoms and directs care; and a software module embedded in a cleared
device that adds a new clinical function. Other examples include apps that display implantable
sensor data, laboratory software that interprets a result, and products sold to employers or
payers rather than clinicians.

**Question 10. How do you critique a product description that mixes multiple intended uses?**
Separate the outputs, users, and clinical actions. Rewrite each as a standalone intended-use
statement. Identify which statements are medical claims and which are administrative or
educational. Then decide whether the product should be split, the claim narrowed, or the
evidence expanded. The answer should cite the applicable definition and avoid the assumption
that one label can cover all functions.

## 14. Interview questions

**1. What makes software a medical device?** A strong answer starts with intended use and the
statutory definition, distinguishes US and EU law, and notes that function-level analysis is
required. A weak answer says "if it touches patient data" or "if it uses AI," which confuses
data type and technology with legal purpose.

**2. What is the difference between SaMD and SiMD?** A strong answer explains that SaMD
performs a medical purpose without being part of a hardware device, while SiMD is part of or
controls a hardware device. It gives an example of the same algorithm deployed both ways. A
weak answer treats the terms as synonyms.

**3. How would you classify a sepsis alert?** A strong answer writes the intended use,
identifies the user and clinical action, analyzes US device status and EU Rule 11, and states
the evidence implications. A weak answer says "it is CDS, so it is exempt" without applying
the statutory criteria.

**4. What is a predicate and why does it matter?** A strong answer defines a legally marketed
device used for substantial equivalence, explains the intended-use and technology comparison,
and notes that the predicate must still be legally marketed. A weak answer treats any similar
product as a predicate.

**5. When is De Novo the right pathway?** A strong answer says it is for a novel device of low
or moderate risk with no legally marketed predicate, and notes that the classification request
creates a new classification regulation if granted. A weak answer uses De Novo as a way to
avoid PMA.

**6. What does MDR Rule 11 require?** A strong answer describes the decision-support,
monitoring, and other branches, the harm-based class, and the implementing rules. A weak
answer recites "software is IIa" without the exceptions.

**7. How do you handle a product with multiple intended uses?** A strong answer splits the
functions, classifies each, and decides whether to separate the products or narrow the claims.
A weak answer picks the highest-risk use and applies it to everything, or picks the
lowest-risk use and ignores the rest.

**8. What is an accessory under the MDR?** A strong answer defines an article that enables or
assists a specific device's intended purpose, notes that accessories are devices in their own
right, and cites Rule 3.2 and the Rule 8 exception. A weak answer says any add-on is an
accessory or that an accessory inherits the host's class.

**9. How do you distinguish a device from an IVD?** A strong answer focuses on whether the
product examines a specimen derived from the human body and reports a result for diagnosis or
treatment. A weak answer relies on the department that buys the product.

**10. What would you do if the classification is borderline?** A strong answer documents the
competing interpretations, identifies the facts that would resolve them, and seeks external
advice such as a 513(g), a pre-submission, or a notified body opinion. A weak answer guesses
and moves on, or over-classifies without documenting why.

## 15. Key terms

| Term | One-line meaning |
|---|---|
| Accessory | Article that enables or assists a specific device's medical purpose and is itself a device under the MDR |
| AI-enabled device | Device that uses artificial intelligence, including machine learning, as part of its function |
| CE marking | Mark indicating conformity with applicable EU harmonization legislation |
| Class I / II / III (US) | US device classes tied to general controls, special controls, and premarket approval |
| Class I / IIa / IIb / III (EU) | MDR risk classes defined by Annex VIII |
| Class A / B / C / D (IVDR) | IVD risk classes defined by the IVDR |
| Clinical decision support (CDS) | Software that supports clinical decisions; some functions may be excluded from the US device definition |
| De Novo | US classification request for a novel low- or moderate-risk device without a predicate |
| Device software function | Discrete software capability with its own intended use and output |
| FD&C Act | US Federal Food, Drug, and Cosmetic Act |
| Intended use | Objective intent of the persons responsible for labeling, including the general purpose and conditions of use |
| IVD | In vitro diagnostic medical device |
| IVDR | EU In Vitro Diagnostic Medical Devices Regulation (EU) 2017/746 |
| MDR | EU Medical Device Regulation (EU) 2017/745 |
| PMA | US premarket approval for a high-risk device |
| Predicate | Legally marketed device used to support substantial equivalence in a 510(k) |
| SaMD | Software as a Medical Device |
| SiMD | Software in a Medical Device |
| 510(k) | US premarket notification based on substantial equivalence |
| 513(g) | US request for FDA's view on device classification or regulatory status |

## 16. Sources


All URLs were accessed on 25 October 2026. Dates are the publication or revision dates shown
by the source at access. Verify current versions before relying on them.

1. US Food and Drug Administration. "Step 1: Is the Software Function Intended For a Medical Purpose?" Digital Health Policy Navigator. Content current as of 27 September 2022. https://www.fda.gov/medical-devices/digital-health-center-excellence/step-1-software-function-intended-medical-purpose
2. US Food and Drug Administration. "Software as a Medical Device (SaMD)." https://www.fda.gov/medical-devices/digital-health-center-excellence/software-medical-device-samd
3. US Food and Drug Administration. "Device Software Functions Including Mobile Medical Applications." https://www.fda.gov/medical-devices/digital-health-center-excellence/device-software-functions-including-mobile-medical-applications
4. US Food and Drug Administration. "Clinical Decision Support Software." Final guidance, 28 September 2022. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
5. US Food and Drug Administration. "Policy for Device Software Functions and Mobile Medical Applications." Guidance, 28 September 2022. https://www.fda.gov/media/80958/download
6. US Food and Drug Administration. "Product Classification." https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPCD/classification.cfm
7. US Food and Drug Administration. "Artificial Intelligence-Enabled Medical Devices." https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
8. US Food and Drug Administration. "De Novo Classification Request: DEN180001." https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
9. US Food and Drug Administration. "De Novo Classification Request: DEN180044." https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf
10. European Parliament and Council. Regulation (EU) 2017/745 on medical devices (MDR). https://eur-lex.europa.eu/eli/reg/2017/745/oj
11. European Parliament and Council. Regulation (EU) 2017/746 on in vitro diagnostic medical devices (IVDR). https://eur-lex.europa.eu/eli/reg/2017/746/oj
12. Medical Device Coordination Group. MDCG 2019-11 Rev.1, "Guidance on Qualification and Classification of Software in Regulation (EU) 2017/745 – MDR and Regulation (EU) 2017/746 – IVDR," June 2025. https://health.ec.europa.eu/document/download/b45335c5-1679-4c71-a91c-fc7a4d37f12b_en
13. Medical Device Coordination Group. MDCG 2021-24 Rev.1, "Guidance on classification of medical devices." https://health.ec.europa.eu/document/download/cbb19821-a517-4e13-bf87-fdc6ddd1782e_en
14. International Medical Device Regulators Forum. "Software as a Medical Device: Possible Framework for Risk Categorization and Corresponding Considerations." IMDRF/SaMD WG/N12FINAL:2014, 18 September 2014. https://www.imdrf.org/documents/software-medical-device-possible-framework-risk-categorization-and-corresponding-considerations
15. International Organization for Standardization. ISO 13485:2016, Medical devices — Quality management systems — Requirements for regulatory purposes. https://www.iso.org/standard/59752.html
16. International Organization for Standardization. ISO 14971:2019, Medical devices — Application of risk management to medical devices. https://www.iso.org/standard/72704.html
17. International Electrotechnical Commission. IEC 62304:2006+A1:2015, Medical device software — Software life cycle processes. https://webstore.iec.ch/en/publication/22794

## 17. Links

- Module index: [`_index.md`](_index.md)
- Next topic: [`02_is-it-a-device.md`](02_is-it-a-device.md)
- Related topic: [`03_fda-pathways.md`](03_fda-pathways.md)
- Related topic: [`08_eu-mdr-and-ivdr.md`](08_eu-mdr-and-ivdr.md)
- Survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 14.1 regulatory triage: [`../../labs/`](../../labs/)
- Lab 14.2 intended-use statements: [`../../labs/`](../../labs/)

The module index is the authoritative map. If a sibling file is not yet present, the index
marks it as planned; do not infer content from the link text.
