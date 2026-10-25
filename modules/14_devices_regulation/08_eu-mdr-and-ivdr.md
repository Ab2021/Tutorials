# EU MDR and IVDR

> Module 14 — Medical Devices, SaMD, and Regulation
> Topic 08 of 10
> Estimated study time: 3.5 hours
> Prerequisites: 01, 05, 06, 07
> Version 2026.3

## 1. Why this topic exists

The European Union is not a second checkbox on a United States launch plan. It is a separate
legal system with a separate definition of a device, a separate classification logic, a separate
conformity assessment procedure, and a separate post-market regime. A product can be cleared by
the US Food and Drug Administration (FDA) and still have no lawful route to the EU market. A
product can carry a CE mark and still need a US submission. The two systems reward different
evidence and punish different omissions.

Since 26 May 2021, the Medical Device Regulation (MDR), Regulation (EU) 2017/745, has applied
across the EU. Since 26 May 2022, the In Vitro Diagnostic Medical Devices Regulation (IVDR),
Regulation (EU) 2017/746, has applied to in vitro diagnostics. Both replaced directives: the
Medical Devices Directive (MDD) 93/42/EEC and the Active Implantable Medical Devices Directive
(AIMDD) 90/385/EEC for the MDR, and the In Vitro Diagnostics Directive (IVDD) 98/79/EC for the
IVDR. A regulation is directly applicable in every member state, so the old pattern of national
transposition differences is smaller, though national competent authorities still enforce.

The practical effect is a heavier evidentiary and quality-system burden than the directives
required. Clinical evaluation is not a one-time literature review. Post-market clinical follow-up
(PMCF) is planned before launch. A Periodic Safety Update Report (PSUR) is produced on a schedule
that depends on class. A Unique Device Identification (UDI) is assigned and registered. Economic
operators have named legal duties. A person responsible for regulatory compliance must be
appointed. For software, Annex VIII Rule 11 supplies a software-specific classification rule, and
the implementing rules decide whether the software inherits the class of a device it drives.

The IVDR changed the in vitro diagnostic (IVD) world more abruptly than the MDR changed the device
world. Under the IVDD, many in vitro diagnostics were self-certified. Under the IVDR, the majority of
IVDs require a notified body, and the highest class also involves an EU reference laboratory where
one is designated. Laboratories, reagent manufacturers, and software vendors that assumed a
self-declaration route discovered that their product was Class B, C, or D. The transition deadlines
were extended more than once, which is itself a lesson: the legal date and the practical capacity
date are different dates.

The costs of getting the EU question wrong are concrete. A company may spend a year building a
clinical evaluation around a claim that its chosen notified body will not accept. It may design a
Class IIa dossier and discover that the software drives a Class III device, which makes the
software Class III under implementing rule 3.3. It may treat an accessory as part of a host
device and omit the accessory's own conformity assessment. It may launch with a CE mark on the
box but no EU declaration of conformity, no UDI registration, and no post-market surveillance (PMS)
plan. Each of these is a finding waiting for an audit, a competent authority, or a customer's
supplier qualification.

The Artificial Intelligence Act (AI Act), Regulation (EU) 2024/1689, adds a second legal layer
for AI-enabled products. Medical devices that require third-party conformity assessment are
high-risk AI systems under Annex I of the AI Act. The AI Act's high-risk obligations for those
products were deferred by Regulation (EU) 2026/1744 to 2 August 2028, while Annex III systems
were deferred to 2 December 2027. The deferral does not suspend the MDR or IVDR. It changes the
date on which a second set of obligations becomes enforceable, and it gives teams time to align
data governance, logging, human oversight, and technical documentation with both regimes.

This topic exists because the EU procedure is learnable but not guessable. The procedure has
gates, each gate has an input and an output, and each output becomes evidence for the next gate.
A team that can draw the gates and name the legal source for each one can plan a CE-marking
program. A team that cannot will treat "CE" as a sticker to be purchased at the end.

The topic is also a counterweight to US-centric regulatory thinking. A US-only strategy is a
market-access decision, not a neutral default. The FDA pathway vocabulary of 510(k), De Novo, and
premarket approval (PMA) has no EU equivalent. The EU has no premarket
approval in the US sense. A notified body issues certificates, and the manufacturer issues the EU
declaration of conformity. That division of responsibility is easy to miss and it changes how a
team prepares for review: the notified body assesses, but the manufacturer asserts.

## 2. Learning outcomes

By the end of this topic, you can:

1. Qualify a product under the MDR or the IVDR and separate a device, an IVD, an accessory, a
   component, and a combination product.
2. Apply MDR Annex VIII Rule 11 and implementing rules 3.2, 3.3, and 3.5 to a software product,
   including the branch in which software drives a hardware device.
3. Assign an IVDR class A, B, C, or D from the analyte, specimen, clinical use, and self-testing
   status, and explain why companion diagnostics are treated separately.
4. Select the Article 52 or Article 48 conformity route for a given class, including the Class I
   self-certification exceptions for sterile, measuring, and reusable surgical devices.
5. Explain what a notified body is designated to assess, how its scope constrains a submission,
   and what the manufacturer still owns after certification.
6. Plan the clinical evaluation or performance evaluation, PMCF or post-market performance
   follow-up, PSUR, vigilance, UDI, and European database on medical devices (EUDAMED) obligations
   attached to a class.
7. Explain how the AI Act high-risk regime overlays the MDR and IVDR, including the deferred
   application dates in Regulation (EU) 2026/1744.
8. Produce a conformity memo that states the intended purpose, class, rule path, route, evidence
   gaps, open questions, and source dates.

The outcomes map to the exercises and self-check items as follows.

| Outcome | Exercise | Self-check |
|---|---|---|
| 1 | Exercise 1 | Question 1 |
| 2 | Exercise 2 | Question 2 |
| 3 | Exercise 3 | Question 7 |
| 4 | Exercise 4 | Questions 3, 4 |
| 5 | Exercise 6 | Question 4 |
| 6 | Exercise 7 | Question 5 |
| 7 | Exercise 8 | Question 6 |
| 8 | Exercise 2, Exercise 8 | Question 4 |

## 3. The core mental model

EU market access is a sequence of five gates followed by a loop. Each gate narrows the legal
category, and each output is the input to the next. The gates are not interchangeable. Starting
with the algorithm, or with a competitor's CE certificate, produces a conclusion that cannot be
defended when a reviewer asks for the rule path.

```text
EU intended purpose and target market
                |
                v
[Gate 1] QUALIFICATION
    MDR device | IVDR device | both (separate functions) | out of scope
                |
                v
[Gate 2] CATEGORY
    device | IVD | accessory | component | combination | system or procedure pack
                |
                v
[Gate 3] CLASSIFICATION (Annex VIII)
    MDR: I | IIa | IIb | III          IVDR: A | B | C | D
                |
                v
[Gate 4] CONFORMITY ROUTE (Article 52 MDR / Article 48 IVDR)
    self-declaration | Annex IX | Annex X + XI | Annex XI | NB limited to aspects
                |
                v
[Gate 5] EVIDENCE AND QMS
    GSPR | risk | clinical or performance evaluation | software | usability | security
                |
                v
NB certificate + EU declaration of conformity + CE marking (+ NB number)
                |
                v
PMS | PMCF or PMPF | PSUR | vigilance | UDI | EUDAMED
                |
                +---- material change ----> back to Gate 1
```

The first insight is that the intended purpose is the input to the whole chain. The MDR defines
intended purpose as the use for which a device is intended according to the data supplied by the
manufacturer. Clinical benefit is a related but separate defined term, and the clinical evaluation
must show that the claim delivers it. The IVDR uses the same idea for an IVD's intended purpose. A
claim that names a diagnosis, a treatment decision, or a monitoring action sets the classification
analysis in motion. A claim that names only workflow sorting may still be a device claim if the
sorting depends on a patient-specific medical prediction.

The second insight is that qualification precedes classification. Before asking whether software
is Class IIa or IIb, ask whether it is a device at all, whether it is an IVD, and whether it is an
accessory to something else. The MDR and IVDR are separate regulations with separate Annex VIII
rules. A product that examines a specimen is not classified under the MDR simply because a
clinician uses the result. A product that analyzes an image is not an IVD simply because a
laboratory buys it.

The third insight is that classification is rule-based, not score-based. A team's internal risk
matrix can say "high" and the product can still be Class IIa if the Annex VIII rule says so. The
inverse is also true. Rule 3.5 requires the strictest class when several rules apply, so a product
that looks like a modest decision-support tool can inherit Class III through the device it drives
or the accessory relationship it has.

The fourth insight is that the conformity route is a function of class plus a small number of
product attributes. Class I is normally self-certified, but a sterile Class I device, a Class I
device with a measuring function, and a reusable surgical instrument need a notified body for
those aspects. Class IIa, IIb, and III always need a notified body. On the IVDR side, Class A is
normally self-certified unless sterile, and Classes B, C, and D need a notified body. Class D also
involves an EU reference laboratory where one is designated.

The fifth insight is that the notified body is an assessor, not an approver. It audits the quality
management system (QMS), assesses technical documentation, and issues certificates. The
manufacturer then signs the EU declaration of conformity and applies the CE mark. The certificate
is time-limited and conditional. The declaration is the manufacturer's legal assertion that the
device meets the General Safety and Performance Requirements (GSPR) in Annex I. If the assertion
is false, the certificate does not cure it.

The sixth insight is that CE marking is not the end. The MDR and IVDR attach a continuous
obligation: PMS, PMCF, PSUR, vigilance, trend reporting, UDI maintenance,
and change control. A material change to the intended purpose, design, or software can require a new
classification analysis and notified body involvement before the changed device is placed on the
market. The loop in the diagram is the part that surprises teams trained on a one-time submission
model.

A compact way to remember the chain: purpose selects regulation, regulation selects category,
category and purpose select class, class selects route, route selects evidence and surveillance.
If you can defend each arrow with an article or annex, you have a regulatory strategy. If you can
defend only the conclusion, you have a slide.

## 4. Definitions and vocabulary

The table defines the terms used later. The "common misuse" column records errors that appear in
real dossiers and review meetings. The source column names the legal instrument or guidance, not
a secondary summary.

| Term | Definition | Common misuse | Source |
|---|---|---|---|
| MDR | Regulation (EU) 2017/745 on medical devices, applicable from 26 May 2021 | Calling it a directive or assuming national transposition | OJ L 117, 5.5.2017; Article 123 |
| IVDR | Regulation (EU) 2017/746 on in vitro diagnostic medical devices, applicable from 26 May 2022 | Applying MDR classes to an IVD | OJ L 117, 5.5.2017; Article 113 |
| MDD | Directive 93/42/EEC, replaced by the MDR | Assuming an MDD certificate remains valid without transitional conditions | Repealed by MDR Article 122 |
| AIMDD | Directive 90/385/EEC on active implantable devices, replaced by the MDR | Treating legacy implant certificates as permanent | Repealed by MDR Article 122 |
| IVDD | Directive 98/79/EC, replaced by the IVDR | Assuming most IVDs remain self-certified | Repealed by IVDR Article 112 |
| CE marking | Mark indicating conformity with the applicable EU harmonisation legislation | Treating it as a quality badge, a global passport, or a regulator's approval | MDR Article 20; Annex V |
| Conformity assessment | The process of demonstrating that a device meets the MDR or IVDR | Assuming it ends when the certificate arrives | MDR Article 52; IVDR Article 48 |
| Notified body | An independent body designated by a member state to assess conformity | Treating it as a government agency or as the decision-maker on market access | MDR Article 43; Annex VII |
| Competent authority | A national authority responsible for market surveillance, designations, and enforcement | Confusing it with the European Commission or the notified body | MDR Article 2; national implementing law |
| EU declaration of conformity | The manufacturer's signed statement that the device meets the applicable requirements | Treating it as a certificate issued by the notified body | MDR Article 19; Annex IV |
| Technical documentation | The dossier demonstrating conformity, including design, risk, verification, and clinical evidence | Reusing a US 510(k) summary as the EU dossier | MDR Annex II; IVDR Annex II |
| GSPR | General Safety and Performance Requirements in Annex I | Treating Annex I as a documentation checklist rather than a design obligation | MDR Annex I; IVDR Annex I |
| Classification | Assignment to a risk class under Annex VIII | Choosing a class from a competitor's label or a conference slide | MDR Article 51; Annex VIII |
| Generic device group | A set of devices with the same intended purpose and similar technology | Confusing it with a commercial product family | MDR Article 2; Annex IX |
| Intended purpose | The use for which a device is intended according to the data supplied by the manufacturer | Copying the US intended-use sentence without stating the clinical benefit or target population | MDR Article 2(12) |
| Clinical benefit | A positive impact on health, or a meaningful impact on diagnosis or treatment | Claiming workflow savings or cost savings as a clinical benefit | MDR Article 2(53) |
| Clinical evaluation | A systematic assessment of clinical data to verify safety and performance | Treating it as a one-time literature search | MDR Article 61; Annex XIV |
| Clinical investigation | A systematic study in human subjects to assess safety or performance | Calling a retrospective chart review a clinical investigation | MDR Article 2; Annex XV |
| PMCF | Post-market clinical follow-up; the planned collection of clinical data after launch | Confusing it with complaint handling or general PMS | MDR Annex XIV Part B |
| Performance evaluation | The IVDR assessment of scientific validity, analytical performance, and clinical performance | Calling an IVDR performance evaluation a clinical evaluation | IVDR Article 56; Annex XIII |
| Performance study | An IVDR study conducted to establish or confirm analytical or clinical performance | Assuming a method comparison is always sufficient | IVDR Article 2; Annex XIII |
| PMS | Post-market surveillance; the system for collecting and analyzing post-market data | Treating PMS as a complaints inbox | MDR Article 83 |
| PMS plan | The documented plan that describes data sources, methods, and indicators | Writing a plan that no one uses after certification | MDR Article 84 |
| PMS report | The post-market summary required for Class I devices | Producing a PSUR for a Class I device | MDR Article 85 |
| PSUR | Periodic Safety Update Report, required at class-dependent intervals | Assuming one annual report satisfies every class | MDR Article 86; MDCG 2022-21 |
| Vigilance | The system for reporting and acting on serious incidents and field safety corrective actions | Reporting every complaint as a serious incident, or reporting none | MDR Article 87 |
| Serious incident | An incident involving death, serious deterioration, or a serious public health threat | Using the term for a minor malfunction | MDR Article 2; Article 87 |
| FSCA | Field safety corrective action, taken to remove a device or reduce a risk in the field | Treating a software patch as a routine update without assessment | MDR Article 2; Article 87 |
| UDI | Unique Device Identification assigned to a device model and production unit | Using a catalog number or a national reimbursement code | MDR Article 27 |
| Basic UDI-DI | The identifier for a device model or family, used in EUDAMED and certificates | Confusing it with the packaging-level UDI-DI | MDR Article 27; Annex VI |
| EUDAMED | The European database on medical devices | Assuming all modules are fully operational before relying on them | MDR Article 33 |
| Economic operator | Manufacturer, authorised representative, importer, or distributor | Assuming only the manufacturer has legal duties | MDR Article 2; Articles 10–14 |
| PRRC | Person responsible for regulatory compliance | Treating it as a nominal signatory with no authority | MDR Article 15 |
| Legacy device | A device placed on the market under a valid MDD or AIMDD certificate during the transitional period | Assuming legacy status removes the need for an MDR plan | MDR Article 120; Regulation (EU) 2023/607 |
| Custom-made device | A device made for a named patient under a written prescription | Calling a configurable product custom-made | MDR Article 2; Annex XIII |
| Companion diagnostic | An IVD essential for the safe and effective use of a medicinal product | Treating it as an ordinary Class C test | IVDR Article 2(7); Annex VIII |
| Self-testing | An IVD intended for use by a lay person | Assuming self-test status lowers the class | IVDR Article 2; Annex VIII Rule 4 |
| Harmonised standard | A standard cited in the Official Journal that gives presumption of conformity | Treating every ISO or IEC standard as harmonised | MDR Article 8 |
| MDCG | Medical Device Coordination Group, which issues non-binding guidance | Treating MDCG guidance as regulation | MDR Article 103 |
| Equivalence | Reliance on another manufacturer's clinical data after demonstrating similarity and access | Claiming equivalence without contractual access to the technical documentation | MDR Article 61(4)–(5) |
| High-risk AI system | An AI system listed in AI Act Annex I or Annex III, including MDR devices requiring a notified body | Assuming the AI Act replaces the MDR or IVDR | AI Act Article 6; Annex I |

Two definitional points change how the rest of the topic reads. First, a regulation is not a
directive. The MDR and IVDR apply directly, and national law mainly supplies designations,
penalties, and administrative arrangements. A team that keeps looking for a national
"transposition" of the MDR will not find one. Second, the EU has no premarket approval. The
notified body issues a certificate after assessing the QMS and technical documentation; the
manufacturer issues the declaration of conformity. The CE mark is applied on the basis of that
declaration. A certificate is evidence that an assessor was satisfied; the declaration is the
legal act that places responsibility on the manufacturer.

The word "approval" is therefore a warning sign in EU discussions. A vendor that says its product
was "approved by the EU" is usually compressing a certificate, a declaration, and a registration
into one misleading verb. The precise question is: which certificate, issued by which notified
body, under which annex, covering which device and which intended purpose, valid until when, and
subject to which conditions? If the answer is vague, the claim is not verifiable.

"Legacy" is the second warning sign. A legacy device is not a device that has escaped the MDR. It
is a device that may remain on the market during a transitional period because it holds a valid
MDD or AIMDD certificate or declaration and meets the conditions in Article 120 as amended by
Regulation (EU) 2023/607. The conditions include lodging an application with a notified body and
signing a written agreement, within dates set by that regulation. Legacy status is a bridge with
conditions, not a permanent exemption.

Finally, guidance is not law. The Medical Device Coordination Group (MDCG) documents explain how
the MDR and IVDR are read, and a notified body will expect a manufacturer to address them, but
they do not bind manufacturers, notified bodies, or courts. A memo that cites only MDCG guidance
and not the article or annex is an interpretation, not a legal basis. Label each source in the
memo as regulation, guidance, harmonised standard, or local practice; the label tells the reader
how much weight the statement carries.

## 5. The mechanics, step by step

The EU procedure answers three questions in order: what is the product legally, how risky is the
claim, and which assessment route does that risk require. Each question has a legal source, and
each answer narrows the next. The procedure below produces a conformity memo, not a certificate.
The memo is the audit trail that a notified body, a competent authority, or an internal investment
committee can challenge line by line.

Work in order. If a step produces two plausible answers, record both and continue. Forcing an
early class decision hides the fact that makes the case borderline.

### Step 1 — Fix the unit of analysis and qualify under the MDR or the IVDR

**Input.** Product description, feature list, variants, labeling, distribution model, input material,
and the clinical use of the output.

**Output.** A function-level list, the EU market version, the legal manufacturer, and a qualification
(MDR device, IVDR device, both, or out of scope) with the qualifying article.

**Rationale.** Obligations attach to the device placed on the EU market and to the manufacturer under
whose name it is placed. A platform can contain a non-device administrative function and a device
clinical function, and the EU claim may differ from the US claim. Qualification then asks whether
each function examines a specimen (IVDR) or serves a medical purpose through another means (MDR).

**Procedure.** (1) List every output a user can act on. (2) Name the user and clinical action.
(3) Identify the EU labeling, instructions for use (IFU), and languages. (4) Identify the legal
manufacturer, including private-label arrangements. (5) Ask whether the product examines a specimen
derived from the human body. (6) Ask whether the output informs diagnosis, treatment, monitoring, or
prevention. (7) Check the exclusions in Article 1. (8) Record the regulation and article per function.

**Branches and edge cases.** A private-label distributor that places the device under its own name
becomes the manufacturer. An original equipment manufacturer (OEM) module may be a component. A
digital pathology image product is generally an MDR device, an analyte-measurement product is
generally an IVD, and a mixed product needs function-level separation. Laboratory software that only
stores results may be out of scope, while an in-house IVD used within a health institution may fall
under Article 5(5) conditions.

### Step 2 — Resolve the category: device, IVD, accessory, component, or combination

**Input.** The architecture, the host device relationship, the specimen flow, and any medicinal
product component.

**Output.** A category for each function, with the legal consequence for the route.

**Rationale.** An accessory is a device in its own right and is classified separately under Annex
VIII Rule 3.2. A component supplied for incorporation is not a finished device. A device that
incorporates a medicinal substance is assessed under the MDR with consultation on the medicinal
component. Misclassifying the category changes who needs a certificate and which annex applies.

**Procedure.** (1) Identify any host device and the function the product enables or assists.
(2) Ask whether the product has an independent medical purpose. (3) Ask whether it is supplied to a
manufacturer for incorporation. (4) Identify any integral medicinal, biological, or human-derived
component. (5) Record the category and the rule that follows from it.

**Branches and edge cases.** Software that controls or drives a hardware device is part of that
device, not an accessory, and Rule 3.3 applies. An add-on sold to clinicians that enables a
specific host may be an accessory with its own class. A companion diagnostic is an IVD even when
it is developed alongside a drug. A configuration option is not a custom-made device.

### Step 3 — Write the EU intended purpose and clinical benefit

**Input.** Draft claims, indications, user descriptions, and the clinical problem.

**Output.** An EU intended purpose statement with user, population, setting, input, output,
clinical benefit, and explicit limitations.

**Rationale.** Intended purpose drives qualification, classification, clinical evaluation, and
labeling. Clinical benefit is a defined term, and the clinical evaluation must demonstrate it for
the intended population. A statement that copies US intended-use language often omits the clinical
benefit and the target population.

**Procedure.** (1) State the clinical action as a verb: detect, diagnose, triage, monitor, predict,
or inform. (2) Name the user and training. (3) Name the population, including age and exclusions.
(4) Name the setting and workflow point. (5) State the output and its basis. (6) State the clinical
benefit and how it will be measured. (7) Add at least one negative limitation.

**Branches and edge cases.** "Supports" and "replaces" are different claims with different evidence.
Screening, diagnosis, and treatment selection are different clinical actions even when the same
model output supports all three. A quantitative output needs a stated threshold if the threshold
changes the action. A pediatric claim changes the population and may change the class.

### Step 4 — Classify under MDR Annex VIII

**Input.** The EU intended purpose, the harm analysis, and the product's relationship to any host
device.

**Output.** A provisional MDR class (I, IIa, IIb, or III) and the rule path that produces it.

**Rationale.** Article 51 requires classification under Annex VIII. Rule 11 is the software-specific
rule, but it is not the only rule that can apply. Rule 3.3 makes software that drives or influences
a device take the device's class. Rule 3.2 classifies accessories in their own right. Rule 3.5
requires the strictest class when several rules apply.

**Procedure.** (1) Decide whether the software provides decision information, monitors physiological
processes, or neither. (2) For decision information, characterize the possible harm. (3) For
monitoring, decide whether the parameter is vital and whether variation could cause immediate
danger. (4) Apply Rule 3.3 for driving software. (5) Apply Rule 3.2 for accessories. (6) Apply any
other relevant Annex VIII rule. (7) Apply Rule 3.5 and take the highest class. (8) Record the
alternative reading and why it was rejected.

**Branches and edge cases.** The harm question is about the consequence the claim permits, not the
consequence the team hopes for. A tool that only prioritizes a worklist may still be IIb if a missed
priority delays surgery. The same model sold to a specialist centre and to an urgent-care clinic can
carry different harm characterizations because the downstream pathway differs.

| Software role | Possible harm | Class |
|---|---|---|
| Decision information | Death or irreversible deterioration | III |
| Decision information | Serious deterioration or surgical intervention | IIb |
| Decision information | Other harm | IIa |
| Physiological monitoring | Vital parameter, variation could cause immediate danger | IIb |
| Physiological monitoring | Non-vital parameter | IIa |
| Neither decision nor monitoring | Any | I |
| Software driving a device | Any | Same class as the device (Rule 3.3) |

### Step 5 — Classify under IVDR Annex VIII

**Input.** The analyte, specimen type, clinical purpose, testing setting, and any companion
diagnostic relationship.

**Output.** A provisional IVDR class (A, B, C, or D) and the rule path.

**Rationale.** IVDR classification reflects the individual and public-health risk of an incorrect
result. Class D covers the highest individual and public-health risk. The rules are not a copy of
the MDR rules, and applying MDR classes to an IVD is a category error that leads to the wrong route.

**Procedure.** (1) Confirm IVD status. (2) Apply Rule 1 for transmissible agents and Rule 2 for
blood grouping or tissue compatibility, each subject to its listed exceptions. (3) Apply Rule 3 for
cancer, genetic testing, congenital disorders, staging or severity, infectious disease, companion
diagnostics, and drug monitoring. (4) Apply Rule 4 for self-testing and near-patient testing.
(5) Apply Rule 5 for reagents, calibrators, control materials, and specimen receptacles. (6) Apply
Rule 6 as the residual Class B rule. (7) Take the strictest class where several rules apply.

**Branches and edge cases.** A companion diagnostic is Class C under Rule 3 and triggers consultation
with the competent authority or the European Medicines Agency. A self-test is generally Class C, but
the rule lists exceptions such as pregnancy and fertility tests that remain Class B. A control
material without assigned values follows the device it is intended for.

| Risk profile | Class | Notified body involvement |
|---|---|---|
| Low individual and public-health risk | A | None unless sterile |
| Moderate individual or low public-health risk | B | Notified body required |
| High individual or moderate public-health risk | C | Notified body required |
| Highest individual and public-health risk | D | Notified body required, reference laboratory where designated |

### Step 6 — Select the conformity assessment route

**Input.** The class, the sterile or measuring status, the reusable surgical status, and the IVD
testing setting.

**Output.** The annex path, whether a notified body is required, and the depth of technical
documentation assessment.

**Rationale.** Article 52 MDR and Article 48 IVDR map class to route. The route determines the
depth of notified body involvement and whether the manufacturer can self-declare. Class I is not
synonymous with self-certification, and Class A IVD is not synonymous with no notified body.

**Procedure.** (1) Map the class to the default route. (2) Check the Class I exceptions for sterile,
measuring, and reusable surgical devices. (3) Check the IVD Class A sterile exception. (4) Check
whether Class D requires an EU reference laboratory. (5) Check custom-made and investigational
routes. (6) Record the chosen annex and the reason.

**Branches and edge cases.** A Class I device with a measuring function needs notified body
involvement for the measuring aspect even though the rest is self-declared. A Class IIa device
normally has its technical documentation assessed on a representative basis, while a Class IIb
device is assessed per generic device group. A Class III device is assessed per device and needs a
Summary of Safety and Clinical Performance (SSCP) unless it is custom-made or investigational.

| Class | Default route | Notified body | Technical documentation depth |
|---|---|---|---|
| MDR I (ordinary) | Annex II and III; EU declaration of conformity | No | Manufacturer holds the file |
| MDR I sterile, measuring, or reusable surgical | Annex IX Chapters I and III, or Annex XI Part A | Yes, limited to the special aspect | Aspect-specific |
| MDR IIa | Annex IX Chapters I and III, or Annex XI Part A | Yes | Representative device per category |
| MDR IIb | Annex IX Chapters I, II, and III, or Annex X plus XI | Yes | Per generic device group |
| MDR III | Annex IX Chapters I, II, and III, or Annex X plus XI | Yes | Per device; SSCP required |
| IVDR A (non-sterile) | Annex II and III; EU declaration of conformity | No | Manufacturer holds the file |
| IVDR A (sterile) | Annex IX Chapters I and III, or Annex XI Part A | Yes, limited to sterility | Aspect-specific |
| IVDR B | Annex IX Chapters I and III, or Annex XI | Yes | Representative device |
| IVDR C | Annex IX Chapters I, II, and III, or Annex XI | Yes | Per device |
| IVDR D | Annex IX Chapters I, II, and III, or Annex X plus XI | Yes | Per device; reference laboratory where designated |

### Step 7 — Select and contract a notified body

**Input.** The route, the technology, the device type, the target timeline, and the manufacturer's
quality-system maturity.

**Output.** A designated notified body with the right scope, a signed contract, a fee schedule, and
an audit plan.

**Rationale.** A notified body is designated for specific device types and technologies, and its
scope must cover the product. A certificate from a body whose scope does not cover the technology
is not a valid basis for CE marking. Capacity and queue position are commercial facts that affect
the launch plan as much as the science does.

**Procedure.** (1) Search the European Commission's New Approach Notified and Designated
Organisations (NANDO) database for MDR or IVDR designation. (2) Verify that the scope covers the
device type and technology. (3) Request a quote and a projected assessment schedule. (4) Use
structured dialogue or a pre-submission meeting for classification and clinical-evidence questions.
(5) Sign the contract and agree the audit plan. (6) Record the certificate scope and expiry.

**Branches and edge cases.** No designated body with the right scope may mean a delay, a change of
technology, or a different route. A small manufacturer may need extra lead time for a first
certification audit. A Class D IVD may depend on reference-laboratory availability. A notified body
can suspend, withdraw, or reduce its scope, which turns into a manufacturer problem even when the
device has not changed.

### Step 8 — Build the QMS, technical documentation, and GSPR evidence

**Input.** Existing quality processes, design outputs, risk file, and verification evidence.

**Output.** A QMS that satisfies Annex IX, technical documentation under Annex II, post-market
technical documentation under Annex III, and a GSPR checklist under Annex I.

**Rationale.** The notified body assesses the QMS and the technical documentation, but the
manufacturer remains responsible for conformity. ISO 13485:2016, harmonised through
EN ISO 13485:2016+A11:2021, supports the QMS requirement, but a certificate to ISO 13485 is not
by itself an MDR certificate.
The GSPR checklist is where design, risk, usability, software, and labeling claims meet the legal
requirements.

**Procedure.** (1) Gap-assess the QMS against Annex IX and ISO 13485. (2) Confirm design controls,
document control, supplier control, and corrective and preventive action (CAPA). (3) Structure the
technical documentation by Annex II. (4) Build a GSPR matrix with evidence references. (5) Prepare
the SSCP for Class III and implantable devices where required. (6) Confirm IFU and labeling in the
required languages.

**Branches and edge cases.** A Class I device still needs technical documentation, a GSPR checklist,
and a declaration of conformity even though no notified body reviews them before launch. Software
needs a lifecycle file under IEC 62304, a risk file under ISO 14971, and usability evidence under
IEC 62366-1. A cloud service needs security, availability, and change-management evidence that a
hardware-only dossier does not contain.

### Step 9 — Perform clinical evaluation or performance evaluation

**Input.** Literature, internal data, external data, and any clinical or performance studies.

**Output.** A clinical evaluation report (CER) or performance evaluation report (PER), a PMCF or
post-market performance follow-up (PMPF) plan, and a plan for updating the evaluation.

**Rationale.** Article 61 and Annex XIV require clinical evaluation for MDR devices. Article 56 and
Annex XIII require performance evaluation for IVDs, covering scientific validity, analytical
performance, and clinical performance. Equivalence under Article 61(4)–(5) requires similar clinical
condition, similar technology, and sufficient access to the comparator's technical documentation,
usually through a contract.

**Procedure.** (1) Define the clinical or performance question and the claim it supports. (2) Run a
systematic literature search with documented inclusion criteria. (3) Appraise the evidence and
identify gaps against the claim. (4) Decide whether equivalence, a clinical investigation, or a
performance study is needed. (5) Write the CER or PER. (6) Write the PMCF or PMPF plan. (7) Set the
review interval and the triggers for an update.

**Branches and edge cases.** A novel claim usually cannot be supported by literature alone. A Class
III implantable device generally needs a clinical investigation unless an exemption applies. A
well-established technology may rely on literature and equivalence, but the manufacturer still needs
access to the comparator's documentation. For software, the evidence must address the model,
the data, the user interface, and the workflow, not only stand-alone accuracy.

### Step 10 — Register, certify, and apply CE marking

**Input.** Device identity, packaging hierarchy, economic-operator roles, and the notified body
assessment report.

**Output.** A Basic UDI-DI and UDI-DIs, EUDAMED registrations, a signed EU declaration of conformity,
CE marking with the notified body's number where applicable, and a change-control procedure.

**Rationale.** Article 27 requires UDI assignment, Article 33 establishes EUDAMED, and Articles 10 to
15 allocate duties across the manufacturer, authorised representative, importer, distributor, and
person responsible for regulatory compliance. Article 19 requires the declaration of conformity, and
Article 20 with Annex V governs CE marking. The certificate is time-limited and conditional.

**Procedure.** (1) Assign the Basic UDI-DI and packaging-level UDI-DIs and apply the carrier.
(2) Register the economic operator, the device, and the certificate in EUDAMED. (3) Appoint the
person responsible for regulatory compliance and verify importer and distributor contracts.
(4) Close assessment findings and sign the declaration of conformity. (5) Apply the CE mark and the
notified body number where applicable. (6) Schedule surveillance audits and assess every design,
software, or labeling change before release.

**Branches and edge cases.** A non-EU manufacturer needs an authorised representative in the EU. An
implantable device needs an implant card under Article 18. A certificate suspension stops the legal
basis for the CE mark even if stock remains. A change to the intended purpose, technology, or a
safety-critical component may require assessment before the change reaches the market. A legacy
device under Article 120 has separate registration and transition conditions.

### Step 11 — Operate PMS, PMCF, PSUR, and vigilance

**Input.** Complaints, service data, literature, registries, clinical data, and security
intelligence.

**Output.** A PMS plan, a PMS report for Class I devices or a PSUR for higher classes, a PMCF or PMPF
evaluation report, and vigilance reports within the legal timelines.

**Rationale.** Articles 83 to 87 require a proactive PMS system, a plan, class-dependent reporting,
and vigilance. The PSUR interval depends on class: every two years for Class IIa and annually for
Class IIb and Class III. Under the IVDR, Class A and B devices produce a PMS report, while Class C
and D devices produce a PSUR. Serious incidents are reported within 15 days, death or unanticipated
serious deterioration within 10 days, and a serious public health threat within 2 days.

**Procedure.** (1) Write the PMS plan with data sources and indicators. (2) Collect and analyze the
data against the plan. (3) Produce the PMS report or PSUR on schedule. (4) Evaluate PMCF or PMPF
results. (5) Report serious incidents and field safety corrective actions within the applicable
timeline. (6) Feed findings into risk management, labeling, and design changes.

**Branches and edge cases.** A software update can be a routine change, a field safety corrective
action, or a reportable incident depending on its effect. A cybersecurity vulnerability that could
cause harm is a safety issue, not only an information-technology issue. A cluster of minor
malfunctions can trigger trend reporting even when no single event is a serious incident.

### Step 12 — Overlay the AI Act

**Input.** The AI system description, the MDR or IVDR class, and the conformity route.

**Output.** The AI Act role (provider or deployer), the high-risk determination, the applicable
obligations, and the application date.

**Rationale.** AI Act Annex I includes medical devices that require third-party conformity
assessment, which makes them high-risk under Article 6(1). Regulation (EU) 2026/1744 deferred the
Annex I high-risk obligations to 2 August 2028 and Annex III obligations to 2 December 2027. The
MDR and IVDR continue to apply, so the AI Act is an additional layer rather than a replacement.

**Procedure.** (1) Decide whether the product is an AI system under Article 3(1). (2) Decide whether
it is an Annex I product requiring notified body assessment. (3) Identify the provider and any
deployers. (4) Map the high-risk obligations: risk management, data governance, technical
documentation, logging, transparency, human oversight, accuracy, robustness, and cybersecurity.
(5) Confirm how the MDR or IVDR conformity assessment interacts with AI Act Article 43. (6) Plan
registration and post-market monitoring.

**Branches and edge cases.** A Class I device that does not require a notified body is not high-risk
through Annex I, but it can still be caught by Annex III if it falls into a listed use. Prohibited
practices under Article 5 apply on an earlier timeline. A general-purpose AI component supplied by
a third party creates a provider-versus-deployer allocation question. MDCG 2025-6 explains the
interplay but is guidance, not law.

### Step 13 — Reconcile, document, and maintain

**Input.** The outputs of Steps 1 to 12.

**Output.** A conformity memo with the conclusion, the rule path, the route, the evidence gaps, the
open questions, the source dates, and the review triggers.

**Rationale.** A conclusion without a rule path cannot be reviewed. A memo that records alternatives
and open questions lets a notified body or an investor see where the risk sits. The memo is also the
change-control baseline: when the intended purpose changes, the analysis restarts at Step 1.

**Procedure.** (1) Write the function-level intended purpose. (2) State qualification and category
with citations. (3) State class and rule path, including rejected alternatives. (4) State the route
and the notified body scope. (5) List evidence gaps and owners. (6) Record source versions and
access dates. (7) Set review triggers: new guidance, new class, new market, or material change.

**Branches and edge cases.** A memo that depends on one ambiguous phrase in the claim should trigger
a claim rewrite before an evidence investment. A memo that assumes a notified body will accept an
equivalence argument should record that assumption as unverified. A memo prepared for a board
decision should separate regulatory risk from commercial risk rather than blending them. A genuinely
borderline product should trigger a notified body or competent authority opinion before investment.

### Cross-cutting edge cases

| Edge case | What changes | Where to record it |
|---|---|---|
| Standalone software | Annex VIII Rule 11 applies; Rule 3.3 only if it drives a device | Steps 2 and 4 |
| Software embedded in a device | Part of the device, not a separate CE-marked product unless separately placed on the market | Step 2 |
| IVD versus device | Specimen examination points to the IVDR; image or signal analysis generally points to the MDR | Step 1 |
| Accessory | Classified in its own right under Rule 3.2, not automatically the host's class | Step 2 |
| Borderline product | Record both readings and seek a notified body or competent authority opinion | Step 13 |
| Combination product | MDR applies with consultation on the medicinal component | Step 2 |
| Class I self-certification | Exceptions for sterile, measuring, and reusable surgical devices | Step 6 |
| Legacy device | Article 120 conditions and the Regulation (EU) 2023/607 dates | Steps 7 and 10 |
| AI-enabled device | AI Act high-risk layer with deferred dates; MDR or IVDR still applies | Step 12 |

The procedure is deliberately longer than a classification table. A table can tell you that a
product is Class IIb. The procedure tells a reviewer why, which evidence the class implies, and
which fact would change the answer. That is the difference between a regulatory strategy and a
guess with a citation.

## 6. Worked example

The case is constructed to show the reasoning. OsteoTrace, the sponsor, the sites, the performance
figures, and the study results are hypothetical. No certificate, notified body number, or approval
date is implied. A real assessment requires the current legal text, current guidance, and a
notified body's written view.

### 6.1 Product and claim

OsteoTrace is a cloud-based software as a medical device (SaMD) product that analyzes adult
radiographs of the distal appendicular skeleton: ankle, wrist, elbow, hand, and foot. A radiographer
or emergency clinician uploads the images. The software returns one of three outputs per study:
"fracture suspected," "no fracture suspected," or "image inadequate." The output is displayed in the
urgent-care workflow before the radiologist's report is available. A radiologist still reads every
study, and the software does not remove a study from that queue.

The draft EU intended purpose is:

"OsteoTrace is intended for use by emergency clinicians and radiographers in urgent-care and
primary-care settings to identify adult patients aged 18 years and older with a suspected acute
fracture of the distal appendicular skeleton who may benefit from earlier immobilization or earlier
orthopedic review. The software analyzes radiographs and flags studies as fracture suspected, no
fracture suspected, or image inadequate. The output is an adjunct to, and not a replacement for,
radiologist interpretation. OsteoTrace is not intended for definitive diagnosis, for children or
adolescents, for the spine, skull, pelvis, or hip, or for patients with a known healing fracture."

The clinical benefit claim is "earlier immobilization and earlier orthopedic review for patients
with a suspected fracture." That is a health-benefit claim, not a workflow claim, and it must be
measured. The negative limitations do real work: they exclude the populations in which a missed
fracture most plausibly causes death or irreversible deterioration, and they keep the product out
of the most sensitive anatomical regions.

### 6.2 Qualification and classification

The product is a medical device under MDR Article 2(1). It analyzes patient-specific images to
inform a clinical decision about immobilization and orthopedic review. It is not an IVD because it
does not examine a specimen derived from the human body. It is not an accessory because it has an
independent medical purpose and is not sold to enable a specific radiography system. It does not
drive or control the acquisition device, so implementing rule 3.3 does not apply. It is standalone
software, so it is SaMD.

Under Annex VIII Rule 11, the software provides information used to take a diagnostic or
therapeutic decision. The possible harm from a false-negative output is a missed fracture that
leads to delayed immobilization, delayed orthopedic review, or surgery. On the stated claim, that
is a serious deterioration of health or a surgical intervention, which places the product in Class
IIb. The sponsor records two alternative readings. A notified body could argue Class IIa if the
radiologist reads every study before disposition and the output only changes queue order. It could
argue Class III if the claim included hip fractures in frail elderly patients, where a missed
fracture can lead to death or irreversible deterioration.

The sponsor chooses IIb for three reasons. The output is displayed before the radiologist report,
so a clinician can act on it. The claim includes earlier immobilization, which is a treatment
action. The population includes adults over 75, who are at higher risk of serious consequence. The
classification memo records that a notified body may take a different view and that the class would
change if the anatomical scope or the user workflow changed.

### 6.3 Conformity route and notified body

Class IIb has two route families under Article 52: Annex IX (quality management system and technical
documentation assessment) or Annex X (type examination) plus Annex XI (production quality assurance).
The sponsor chooses Annex IX because it already operates an ISO 13485:2016 quality system and wants
one assessment covering design, production, and post-market obligations. The technical documentation
is assessed per generic device group, and the notified body audits the quality system.

The notified body must be designated for the MDR and its scope must cover medical device software
and radiology. The sponsor verifies the designation in the NANDO database rather than relying on a
vendor's brochure. The contract covers the initial certification audit, the technical documentation
assessment, and surveillance audits. The sponsor requests a structured dialogue on the Rule 11
classification and the clinical evaluation design before locking the protocol.

The evidence package includes the technical documentation under Annex II, the post-market technical
documentation under Annex III, a GSPR checklist, a risk management file under ISO 14971, a software
lifecycle file under IEC 62304, a usability file under IEC 62366-1, a cybersecurity file, a clinical
evaluation report under Article 61 and Annex XIV, a PMCF plan, and a PMS plan. Because the device is
Class IIb and not Class III or implantable, an SSCP is not required. A PSUR is required annually.

### 6.4 The clinical evaluation gap

The sponsor has a retrospective dataset of 24,800 radiographs from three sites collected between
2017 and 2021. The labels come from the original radiology reports. The model's area under the
receiver operating characteristic curve (AUC) is 0.94 on an internal hold-out set and 0.88 at the
largest external site. At the chosen threshold, sensitivity is 0.88 and specificity is 0.83.

At the intended-use prevalence of 12 percent, the positive predictive value is 0.414 and the
negative predictive value is 0.981. The arithmetic is:

```text
PPV = (0.88 x 0.12) / ((0.88 x 0.12) + (0.17 x 0.88))
    = 0.1056 / (0.1056 + 0.1496)
    = 0.414

NPV = (0.83 x 0.88) / ((0.83 x 0.88) + (0.12 x 0.12))
    = 0.7304 / (0.7304 + 0.0144)
    = 0.981
```

The numbers describe a triage tool, not a rule-out test. Four of every ten "fracture suspected"
outputs are false positives at this prevalence, and about two of every hundred "no fracture
suspected" outputs are false negatives. The label must say that a negative output does not exclude
fracture and that radiologist review remains required.

The clinical evaluation gap is not the AUC. The gap is the absence of evidence for the clinical
benefit claim and the weakness of the reference standard. There is no prospective study, no reader
study measuring the effect of the output on immobilization or orthopedic review, and no usability
study for the "image inadequate" path. The labels are radiology reports, not a masked consensus
with follow-up imaging. The training data include only 2.1 percent of patients over 80 and 8.4
percent with prior orthopedic hardware, both of which are common in the intended population.

The sponsor considered equivalence to two CE-marked fracture-detection products. Equivalence under
Article 61(4)–(5) requires similar clinical condition, similar technology, and sufficient access to
the comparator's technical documentation. The sponsor cannot obtain that access, so equivalence is
not available. Published literature on other algorithms does not close the gap because the claims,
populations, and reference standards differ.

### 6.5 Two diagnostics that would sink CE marking

**Diagnostic 1: subgroup sensitivity falls below the pre-specified floor.** The sponsor pre-specified
a sensitivity floor of 0.80 in each of three subgroups: patients aged 75 and older, patients with
prior orthopedic hardware, and patients imaged on the oldest detector at each site. In the
validation set, 184 patients are aged 75 and older, and 41 have a fracture. Sensitivity in that
subgroup is 0.68 with a 95 percent confidence interval from 0.54 to 0.82. The overall sensitivity is
0.88. The overall result passes; the subgroup result does not.

A notified body will ask why a population that is explicitly inside the intended use has a
sensitivity 20 points below the overall figure. The sponsor cannot answer that the subgroup was
exploratory, because the protocol pre-specified it. It cannot exclude the subgroup without narrowing
the intended use and the clinical benefit claim, and that exclusion would remove a large share of
the clinical need. The diagnostic is fatal to the current CER because the evidence does not support
the claim across the intended population.

**Diagnostic 2: the reference standard is contaminated by the index test.** The labels come from
radiology reports produced in routine practice. In 38 percent of the validation studies, the
radiologist had access to a prior OsteoTrace output or a similar triage flag in the worklist. The
sponsor re-adjudicates 120 studies originally labeled "no fracture" using computed tomography (CT),
magnetic resonance imaging (MRI), or six-week follow-up. Two blinded musculoskeletal radiologists
find 14 fractures that the original reports missed. The model had flagged 3 of those 14.

Recomputing on the re-adjudicated set changes the picture. The original 264 fractures with
sensitivity 0.88 yield 232 true positives and 32 false negatives. The 14 new fractures add 3 true
positives and 11 false negatives. The new sensitivity is 235 divided by 278, or 0.845, with a 95
percent confidence interval from 0.80 to 0.89. The acceptance criterion required a point estimate
of at least 0.85, so the result fails. Specificity changes only slightly, from 0.83 to 0.831.

The re-adjudication sampled only 120 of the 936 originally negative studies. If the 11.7 percent
miss rate is representative, the full negative set contains roughly 109 additional fractures, and
the true sensitivity would be far lower than 0.845. That extrapolation assumes random sampling and a
constant miss rate; it is an inference, not a measurement, but it is the inference a reviewer will
make. A reference standard that was produced with the index test visible is not independent, and an
independent reference standard is the minimum condition for a credible sensitivity estimate.

### 6.6 The certification decision

The sponsor decides not to file the Annex IX technical documentation for certification. The CER
cannot demonstrate the clinical benefit claim, and the sensitivity estimate fails both the subgroup
floor and the overall acceptance criterion once the reference standard is corrected. Filing would
consume a notified body assessment slot and produce a deficiency that is predictable from the
sponsor's own data.

The remediation plan has four parts. First, re-adjudicate the full negative cohort, or a
statistically powered random sample, with CT, MRI, or follow-up and blinded readers. Second, run a
prospective reader study with a pre-specified endpoint of time to immobilization or time to
orthopedic review, with the control arm receiving the current workflow. Third, enroll enough
patients aged 75 and older and enough patients with hardware to meet the subgroup precision target.
Fourth, revise the label so that "no fracture suspected" explicitly does not rule out fracture, and
validate the "image inadequate" path in a usability study.

The projected delay is 9 to 12 months. The quality system, the risk file, and the software
lifecycle evidence remain useful; the clinical evidence is rebuilt. The sponsor asks the notified
body for a structured dialogue on the reader-study endpoint before enrolling. The certification
decision is therefore "not certifiable now, credible path later," not "rejected."

### 6.7 Vary one assumption

The base case is an MDR Class IIb SaMD with a clinician-facing triage claim and an Annex IX route.
Change one assumption at a time and the route and evidence change with it.

| Assumption changed | Class and framework | Route | Evidence consequence |
|---|---|---|---|
| Base: triage before radiologist read, adults 18+ | MDR IIb | Annex IX, notified body | CER, PMCF, annual PSUR, reader study |
| Claim narrowed to worklist ordering; radiologist reads before disposition | MDR IIa | Annex IX, notified body | Representative-device assessment, PSUR every two years, smaller clinical study |
| Scope extended to hip fractures in frail elderly, or autonomous rule-out | MDR III | Annex IX or Annex X plus XI, notified body | Clinical investigation generally required, SSCP, annual PSUR to the notified body |
| Input changed to a serum biomarker instead of an image | IVDR Class B under Rule 6 | Annex IX or Annex XI, notified body | Performance evaluation under Annex XIII, no CER, PMCF becomes PMPF |
| Same biomarker sold as a self-test | IVDR Class C under Rule 4 | Annex IX or Annex XI, notified body | Analytical and clinical performance study in lay users, stricter labeling |
| Sold as an add-on that enables a specific radiography system | MDR accessory under Rule 3.2 | Class in its own right; notified body if IIa or higher | Accessory technical documentation and host-interface evidence |
| Software drives the acquisition device | MDR class of the host under Rule 3.3 | Host class route | Software inherits the host's class and assessment depth |

The table is the point of the exercise. A change in wording, population, input, or commercial
relationship is a change in legal category, route, and evidence. The model, the code, and the
training data can stay the same. The regulatory program cannot.

## 7. Code, query, or calculation

The script below maps an MDR or IVDR class to a route family and checks a documentation set for
missing items. It encodes the decision tables in this topic and is deliberately small enough to
audit. It is a teaching aid, not a conformity assessment tool. It does not model every Article 52
nuance, and it does not replace the legal text or a notified body's view.

```python
"""EU MDR/IVDR route and documentation completeness helper.

Teaching aid for Module 14, Topic 08. It encodes route families from
Article 52 MDR and Article 48 IVDR, not every annex nuance.
"""

CORE = {
    "MDR": {"technical_documentation", "risk_management_file", "gspr_checklist",
            "clinical_evaluation_report", "pms_plan", "labeling", "udi_assignment",
            "declaration_of_conformity", "vigilance_sop"},
    "IVDR": {"technical_documentation", "risk_management_file", "gspr_checklist",
             "performance_evaluation_report", "pms_plan", "labeling", "udi_assignment",
             "declaration_of_conformity", "vigilance_sop"},
}
EXTRAS = {
    ("MDR", "I"): {"pms_report"},
    ("MDR", "IIA"): {"psur", "pmcf_plan", "pmcf_evaluation_report"},
    ("MDR", "IIB"): {"psur", "pmcf_plan", "pmcf_evaluation_report"},
    ("MDR", "III"): {"psur", "pmcf_plan", "pmcf_evaluation_report", "sscp",
                     "clinical_investigation_report"},
    ("IVDR", "A"): {"pms_report"},
    ("IVDR", "B"): {"pms_report", "pmpf_plan"},
    ("IVDR", "C"): {"psur", "pmpf_plan"},
    ("IVDR", "D"): {"psur", "pmpf_plan"},
}


def mdr_route(cls, sterile=False, measuring=False, reusable=False):
    """Return the Article 52 route family for an MDR class."""
    c = cls.upper()
    if c == "I":
        special = [n for n, f in (("sterile", sterile), ("measuring", measuring),
                                  ("reusable surgical", reusable)) if f]
        if special:
            return {"annex": "Annex IX (I and III) or Annex XI Part A", "nb": True,
                    "td": "limited to the special aspect",
                    "note": "Notified body limited to: " + ", ".join(special)}
        return {"annex": "Annex II and III; EU declaration of conformity", "nb": False,
                "td": "none", "note": "Self-certification"}
    if c == "IIA":
        return {"annex": "Annex IX (I and III) or Annex XI Part A", "nb": True,
                "td": "representative device per category", "note": ""}
    if c == "IIB":
        return {"annex": "Annex IX (I, II and III) or Annex X plus XI", "nb": True,
                "td": "per generic device group", "note": ""}
    if c == "III":
        return {"annex": "Annex IX (I, II and III) or Annex X plus XI", "nb": True,
                "td": "per device", "note": "SSCP unless custom-made or investigational"}
    raise ValueError(f"unknown MDR class: {cls}")


def ivdr_route(cls, sterile=False, companion=False):
    """Return the Article 48 route family for an IVDR class."""
    c = cls.upper()
    if c == "A" and not sterile:
        return {"annex": "Annex II and III; EU declaration of conformity", "nb": False,
                "td": "none", "note": "Self-declaration"}
    if c == "A":
        return {"annex": "Annex IX (I and III) or Annex XI Part A", "nb": True,
                "td": "limited to sterility", "note": "Notified body limited to sterility"}
    return {"annex": "Annex IX or Annex XI", "nb": True,
            "td": "representative device" if c == "B" else "per device",
            "note": "Competent authority or EMA consulted" if companion else ""}


def docs(framework, cls, route, sterile=False, companion=False):
    """Return the expected documentation set for a route."""
    required = set(CORE[framework]) | EXTRAS[(framework, cls.upper())]
    if route["nb"]:
        required |= {"qms_certificate", "nb_contract"}
    if sterile:
        required.add("sterility_validation")
    if companion:
        required.add("companion_dx_consultation_record")
    return required


def check(framework, cls, present, **kwargs):
    """Return the route and the sorted list of missing documents."""
    route = (mdr_route if framework == "MDR" else ivdr_route)(cls, **kwargs)
    missing = sorted(docs(framework, cls, route, **kwargs) - set(present))
    return route, missing


if __name__ == "__main__":
    cases = [
        ("MDR", "IIb", {}),
        ("MDR", "I", {"sterile": True}),
        ("IVDR", "C", {"companion": True}),
    ]
    present = {
        ("MDR", "IIb"): CORE["MDR"] | {"psur", "pmcf_plan"},
        ("MDR", "I"): CORE["MDR"] | {"pms_report", "sterility_validation"},
        ("IVDR", "C"): CORE["IVDR"] | {"psur", "pmpf_plan"},
    }
    for framework, cls, kw in cases:
        route, missing = check(framework, cls, present[(framework, cls)], **kw)
        print(f"{framework} {cls}: nb={route['nb']} td={route['td']}")
        print(f"  {route['annex']}")
        if route["note"]:
            print(f"  note: {route['note']}")
        print(f"  missing {len(missing)}: {', '.join(missing)}")
```

Expected output from a CPython 3.11 run:

```text
MDR IIb: nb=True td=per generic device group
  Annex IX (I, II and III) or Annex X plus XI
  missing 3: nb_contract, pmcf_evaluation_report, qms_certificate
MDR I: nb=True td=limited to the special aspect
  Annex IX (I and III) or Annex XI Part A
  note: Notified body limited to: sterile
  missing 2: nb_contract, qms_certificate
IVDR C: nb=True td=per device
  Annex IX or Annex XI
  note: Competent authority or EMA consulted
  missing 3: companion_dx_consultation_record, nb_contract, qms_certificate
```

The output shows three behaviours worth checking in a real review. First, the Class I sterile
device has a notified body even though Class I is normally self-certified, because the sterility
aspect is outside the manufacturer's self-declaration. Second, the Class IIb and IVDR Class C cases
are missing the notified body contract and the quality-system certificate, which are route
prerequisites rather than clinical evidence. Third, the companion diagnostic is missing the
consultation record, which is a procedural obligation specific to that IVD category.

Three cautions apply. The `present` sets are illustrative and omit documents that a real file would
contain, so the missing lists are not audit findings. The route strings compress annex chapters;
read the annex before relying on a chapter number. The completeness check treats documents as
present or absent, not as adequate, current, or linked to the claim. A complete folder of weak
documents passes this script and fails a notified body.

## 8. Data, standards, and regulatory context

> **Note:** This section is educational and is not legal advice. Qualification, classification, and
> conformity are fact-specific legal conclusions. Confirm the current regulations, guidance,
> harmonised standards, and database records with qualified regulatory counsel before relying on
> them.

The MDR and IVDR are binding regulations. They apply directly in every member state, and the
articles and annexes are the primary source. Guidance explains how the European Commission and the
Medical Device Coordination Group read the text, but guidance is not binding on manufacturers,
notified bodies, or courts. A standard becomes relevant when it is harmonised and cited in the
Official Journal, because then it gives a presumption of conformity with the requirements it
covers. A standard that is not harmonised can still be evidence of the state of the art, but the
presumption does not attach automatically.

| Source | Status | What it constrains | What it does not do |
|---|---|---|---|
| MDR, Regulation (EU) 2017/745 | Binding EU regulation, applicable 26 May 2021 | Device definition, obligations, classification, routes, PMS | Does not apply to IVDs |
| IVDR, Regulation (EU) 2017/746 | Binding EU regulation, applicable 26 May 2022 | IVD definition, classification, performance evaluation, PMS | Does not apply to non-IVD devices |
| MDR Annex VIII | Binding classification rules | Rules 1–22 and implementing rules 3.1–3.5 | Does not classify without an intended purpose |
| MDR Article 52; IVDR Article 48 | Binding route rules | Self-declaration, Annex IX, Annex X plus XI, Annex XI | Do not remove the need for evidence |
| MDR Article 61; Annex XIV | Binding clinical evaluation and PMCF rules | Clinical evaluation, PMCF plan, PMCF evaluation report | Does not prescribe a study design |
| IVDR Article 56; Annex XIII | Binding performance evaluation rules | Scientific validity, analytical performance, clinical performance | Does not make a literature review sufficient for a novel claim |
| MDR Articles 83–87 | Binding PMS and vigilance rules | PMS plan, PMS report, PSUR, serious incident reporting | Do not specify the analytics tooling |
| MDR Article 27; Annex VI | Binding UDI rules | Basic UDI-DI, UDI-DI, carrier, database attributes | Does not replace national reimbursement codes |
| MDR Article 33 | Binding EUDAMED rule | Actor, device, certificate, vigilance registration | Does not guarantee that every module is operational |
| MDR Article 120; Regulation (EU) 2023/607 | Binding transitional rules | Legacy device conditions and staged deadlines | Do not make a legacy certificate permanent |
| AI Act, Regulation (EU) 2024/1689; Regulation (EU) 2026/1744 | Binding EU regulation with deferred high-risk dates | High-risk AI obligations, Annex I and Annex III timelines | Does not replace the MDR or IVDR |
| MDCG 2019-11 Rev.1, June 2025 | EU guidance | Software qualification and classification under MDR and IVDR | Is not binding and does not decide a borderline case |
| MDCG 2021-24 Rev.1 | EU guidance | Interpretation of the Annex VIII classification rules | Is not a substitute for the legal text |
| MDCG 2022-21 | EU guidance | Structure and content of the PSUR | Does not change the Article 86 reporting interval |
| MDCG 2025-6 | EU guidance | Interplay between the AI Act and the medical device regulations | Does not amend either regulation |
| ISO 13485:2016; EN ISO 13485:2016+A11:2021 | Harmonised standard | Quality management system requirements | Does not by itself certify a device or a class |
| ISO 14971:2019; EN ISO 14971:2019+A11:2021 | Harmonised standard | Risk management process and file | Does not set the regulatory class |
| IEC 62304:2006+A1:2015 | Standard, harmonised in some contexts | Software lifecycle processes by safety class | Does not determine the MDR or IVDR class |
| IEC 62366-1:2015+A1:2020 | Harmonised standard | Usability engineering process | Does not establish clinical benefit |
| IMDRF N12:2014 | International guidance | Software risk categorization for communication | Is not law in any jurisdiction |

The classification rules are where most teams need the legal text open beside the guidance. Rule 11
is written as a set of conditions, not a score. It asks whether the software provides information
used to take a diagnostic or therapeutic decision, whether it monitors physiological processes, or
whether it does neither. The class then depends on the seriousness of the possible harm or on
whether the monitored parameter is vital and could vary dangerously. The implementing rules modify
the result: accessories are classified in their own right, driving software inherits the device
class, and the strictest applicable rule wins.

The IVDR rules are organized around the consequence of an incorrect result. Rule 1 covers
transmissible agents at Class D, subject to listed exceptions, and Rule 2 covers blood grouping and
tissue compatibility at Class D, subject to exceptions for control materials.
Rule 3 covers cancer markers, genetic testing, congenital disorders, staging or severity,
infectious disease, companion diagnostics, and drug monitoring at Class C, with listed exceptions.
Rule 4 covers self-testing and near-patient testing, Rule 5 covers reagents and specimen
receptacles, Rule 6 is the residual Class B rule, and Rule 7 covers control materials. The rules
interact, and the strictest result governs.

Two standards deserve a caution. ISO 13485 certification is not an MDR certificate. It supports the
quality-system requirement, and a notified body will use it as input, but the MDR assessment also
covers technical documentation, clinical evaluation, and post-market obligations. ISO 14971 is a
process standard: it tells you how to manage risk, not what risk is acceptable. The acceptability
decision belongs to the manufacturer's risk policy and, ultimately, to the benefit-risk judgment
that the regulation requires.

The law also has limits that matter to a data scientist. It does not tell you which model
architecture to use, how to split data, or which metric to report. It tells you that the
performance must be valid for the intended population, that the claim must match the evidence, that
changes must be controlled, and that post-market performance must be monitored. The AI Act adds
data-governance, logging, transparency, and human-oversight obligations for high-risk systems, but
it does not supply a validation protocol. MDCG 2025-6 is the current guidance on the interplay, and
it is guidance rather than an amendment.

Source hygiene is part of the analysis. Record the version and access date of every guidance
document, because MDCG documents are revised. Check the NANDO database for a notified body's current
designation and scope, because a scope can be reduced. Check the EUDAMED status before promising a
registration date, because module availability has changed over time. Check the AI Act timeline in
the amending regulation, not in a conference summary, because the dates were deferred once and may
change again.

## 9. Industry practice and case evidence

Public EU device cases separate into two kinds: failures that shaped the law, and operational
evidence about how the system behaves under load. The first kind is well documented. The second is
mostly reported by industry associations, notified bodies, and the European Commission, and it is
easier to overstate. The notes below label documented fact and inference separately.

**Poly Implant Prothèse (PIP) breast implants.** French authorities ordered a recall of PIP breast
implants in 2010 after regulators found that the manufacturer had used industrial-grade silicone
instead of medical-grade silicone. The notified body that had certified the device had its
certificate withdrawn, and the case became a central example in the argument for stronger EU
oversight. The documented facts are the recall, the fraud, and the certificate withdrawal. The
inference, widely drawn but not a formal causal finding, is that periodic audits and document review
did not detect a deliberate material substitution, which strengthened the case for unannounced
audits and tighter notified body designation under the MDR.

**DePuy ASR metal-on-metal hip implants.** DePuy Orthopaedics recalled the ASR hip resurfacing
system in 2010 after registry data showed high revision rates. The case exposed the difficulty of
detecting long-term implant failure through premarket evidence alone and reinforced the role of
registries and PMCF. The documented facts are the recall and the revision-rate signal. The inference
is that the MDR's explicit PMCF and PSUR obligations reflect a policy choice to make post-market
performance a continuous legal duty rather than a voluntary activity.

**The MDR transition and notified body capacity.** Regulation (EU) 2023/607 amended the MDR
transitional provisions and extended the period during which certain legacy devices could remain on
the market, subject to conditions such as lodging an application with a notified body and signing a
written agreement by set dates. The Commission's stated rationale included notified body capacity
and the risk of device shortages. The documented fact is the extension and its conditions. The
operational inference is that notified body queue position is a real constraint on launch timing,
and that a certificate strategy which assumes unlimited assessment capacity is fragile.

**The IVDR transition.** The IVDR moved a large share of IVDs from self-certification to notified
body assessment, and the transition period was extended by Regulation (EU) 2022/112 and subsequent
measures. Laboratories and reagent manufacturers reported that some tests, particularly low-volume
and rare-disease tests, faced uncertain routes to certification. The documented fact is the
extension and the reclassification. The inference is that the economic viability of a low-volume
test can depend on the certification route, which is a product-portfolio question, not only a
regulatory one.

**Industry practice: engage the notified body before the evidence is locked.** Companies that
certify predictably tend to request a structured dialogue on classification and clinical evidence
before the protocol is final. They bring a draft intended purpose, a rule-path memo, a clinical
evaluation plan, and a gap list. The practice is visible in notified body communications and
industry guidance. The inference is that early dialogue reduces the chance of a preventable
deficiency, not that it guarantees certification.

**Industry practice: treat the classification memo as a controlled document.** Mature manufacturers
version the memo with the technical documentation. When the intended purpose changes, the class and
route are reassessed. When a new guidance document appears, the memo records the impact. When a
notified body questions a rule path, the answer is already written down. The practice is inferred
from audit findings and remediation plans, not from a published survey.

**What the cases do not prove.** PIP does not prove that every notified body is ineffective. ASR
does not prove that every implant needs a registry. The MDR extension does not prove that the MDR
was wrong. The IVDR transition does not prove that every small IVD will disappear. Each case is a
data point about a specific failure, population, and institutional response. The transferable
lesson is procedural: define the claim, identify the rule path, test the evidence against the
claim, and plan for the assessment queue.

## 10. Failure modes and red flags

EU failures usually appear first as a confident sentence in a planning deck: "we are Class I, so no
notified body," "the app is part of the cleared device," or "the FDA clearance covers Europe." The
table lists the failures that survive until a reviewer, an auditor, or a competent authority finds
them, with the signal that should trigger a check.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Class I assumed self-certified | "Class I, so no notified body" | Sterile, measuring, or reusable surgical status triggers notified body involvement | Device placed on the market without the required certificate | Check the Article 52 exceptions at the route step |
| Accessory inherits host class | Add-on sold with a CE-marked host and no own file | Rule 3.2 classifies accessories in their own right | Missing conformity assessment for the add-on | Classify the accessory separately and document the host interface |
| MDR Rule 11 applied to an IVD | Someone says "Class IIa IVD" | The IVDR uses A/B/C/D, not MDR classes | Wrong regulation, route, and performance evidence | Qualify device versus IVD before classifying |
| Rule 3.3 ignored | Software controls a pump but is analyzed as standalone | Driving software takes the host's class | Under-classification and an inadequate assessment | Map the architecture and the control path |
| US clearance treated as CE | "FDA cleared, so Europe is a formality" | The systems have different definitions and routes | No lawful EU market access | Maintain a jurisdiction-by-jurisdiction matrix |
| Novel claim supported by literature only | CER cites similar products without matching claims | Article 61 requires evidence for the actual claim | Notified body deficiency and delay | Generate prospective or primary data |
| Equivalence claimed without access | "We are equivalent to product X" and no contract | Article 61(5) requires sufficient access to technical documentation | Equivalence rejected | Obtain a contract or generate own data |
| Contaminated reference standard | Labels come from reports produced with the AI output visible | The index test influenced the ground truth | Inflated sensitivity and an invalid CER | Blind adjudication with independent follow-up |
| Subgroup analysis added after results | "We will look at age once we see the data" | Post-hoc analysis is not confirmation | Claim not supported in a named population | Pre-specify subgroups and power them |
| Wrong PSUR cadence | Annual report for Class IIa or no report for Class IIb | Article 86 sets class-dependent intervals | Non-compliance finding | Build a class-based compliance calendar |
| Late vigilance reporting | Incident logged in the complaint system but not assessed | The 2, 10, and 15-day clocks start on awareness | Enforcement and liability | Write an SOP with a clock and an owner |
| UDI left until launch | Label artwork has no UDI carrier | Article 27 and Annex VI require assignment and registration | Cannot register or ship compliantly | Assign UDI during design transfer |
| No PRRC or authorised representative | Non-EU manufacturer has no EU contact | Articles 11 and 15 require named roles | Placement on the market is unlawful | Appoint before the first placement |
| AI Act ignored because of the deferral | "2028 is far away" | The AI Act is an additional regime, not a replacement | Costly retrofit of data governance and logging | Map the obligations and evidence now |
| Notified body scope mismatch | Quote accepted without checking designation | Article 43 and Annex VII limit what the body may assess | Certificate does not cover the technology | Verify scope in NANDO before contracting |
| Legacy certificate treated as permanent | "We have an MDD certificate" and no MDR plan | Article 120 conditions and staged deadlines | Abrupt loss of market access | Build a transition plan with dated milestones |

Two red flags deserve separate treatment because they are easy to rationalize. The first is a claim
that the product is "only a workflow tool." A workflow claim can still be a medical device claim if
the workflow depends on a patient-specific prediction, and the classification follows the clinical
action, not the user-interface label. The second is a claim that the product is "not AI" because the
model is a logistic regression or a rules engine. The AI Act definition is broader than deep
learning, and the MDR applies regardless of the algorithm. The relevant question is what the product
does, not which technique it uses.

The failures compound. A classification error changes the route; the route changes the depth of
technical documentation; the documentation depth changes the clinical evidence plan; and the
evidence plan changes the launch date. A team that discovers the class is wrong after a prospective
study has enrolled has not made a paperwork error. It has spent the study budget on the wrong
question. That is why the classification memo belongs at the start of the program, not in the
submission binder.

A red flag is not proof of a violation. It is a signal that the memo is missing a fact. The correct
response is to add the fact, not to explain why the red flag does not matter. A team that can show
why an exception does not apply is in a stronger position than a team that never checked.

## 11. Assumptions and limits

Every EU conformity plan rests on assumptions that are not in the regulation. The memo should state
them, explain why each is needed, name what would break it, and describe the test that would detect
the break. The ledger below is a template.

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The intended purpose is stable | Class, route, and evidence all follow the claim | A new user, population, or output changes the claim | Version-control the intended purpose and review every release |
| The legal text and guidance are current | The MDR and IVDR are amended and guidance is revised | A new amending regulation or MDCG revision | Record versions and access dates; recheck before filing |
| The notified body scope covers the technology | A certificate is only valid within the designation | Scope reduction, withdrawal, or a technology the body does not assess | Verify the designation in NANDO and in the contract |
| EU clinical evidence applies to the EU population | A US-trained model may not transport | Different prevalence, care pathways, or imaging equipment | Pre-specify subgroup and site analyses in the EU population |
| Equivalence access is available | Article 61 reliance requires the comparator's file | The comparator refuses a contract or the file is incomplete | Obtain written access before relying on equivalence |
| The reference standard is independent | Sensitivity estimates require ground truth not produced by the index test | Reports generated with the AI output visible | Blind adjudication and follow-up confirmation |
| Subgroup prevalence matches the intended use | Overall performance can hide subgroup failure | Underrepresentation of elderly, pigmented skin, or comorbidity | Pre-specify and power subgroup analyses |
| PMS data sources are complete | PSUR conclusions depend on the data | Complaints, service records, or registries are excluded | Audit the data-source coverage annually |
| EUDAMED modules are available | Registration deadlines depend on the database | Module delays or changed submission rules | Check the Commission's EUDAMED status page |
| AI Act dates and scope are stable | The compliance roadmap depends on them | A further amending regulation or new guidance | Track EUR-Lex and MDCG publications |
| The certificate remains valid | CE marking depends on a current certificate | Suspension, withdrawal, or a missed surveillance audit | Track expiry and audit findings in the QMS |
| The manufacturer has EU representation | Non-EU manufacturers need an authorised representative | Contract lapse or a representative without capacity | Verify the mandate and the PRRC appointment |

The most expensive assumptions are the two that concern evidence access: equivalence and the
reference standard. Equivalence determines whether the manufacturer can rely on someone else's
clinical data, and it fails silently when the contract is absent. The reference standard determines
whether the manufacturer's own sensitivity estimate is credible, and it fails silently when the
labels were produced with the product visible. Both are cheap to test at the start and expensive to
discover at the end.

The limits of this topic follow from its scope. It does not provide a legal opinion, does not cover
every member state's language and administrative requirements, does not reproduce the full text of
Annex VIII or Annex IX, and does not cover the UK, Switzerland, Canada, Australia, or Japan. It does
not tell you whether a specific notified body will accept a specific equivalence argument. It gives
you the structure to ask the right question and the vocabulary to understand the answer.

A conformity memo should end with a confidence statement, not a binary conclusion. "We are confident
that this is an MDR Class IIb SaMD; we are less confident that the notified body will accept the
clinical benefit endpoint without a prospective study; we will request a structured dialogue before
locking the protocol." That statement tells the team where to spend the next month. A memo that says
only "Class IIb, Annex IX" hides the risk that matters.

## 12. Exercises

The exercises move from recall to application to critique to design. Each states a deliverable,
acceptance criteria, and a time estimate. Use the current legal text and guidance, not this file
alone.

### Exercise 1 — Qualification screen (application, 30 minutes)

Take three products: a sepsis risk score sold to hospitals, a software module that interprets a
next-generation sequencing (NGS) panel from a tumor specimen, and a radiograph fracture-triage
tool. Decide for each whether it is an MDR device, an IVDR device, both, or out of scope, and name
the article that supports the answer. Deliverable: a three-row table with a one-paragraph rationale
per row. Acceptance criteria: each row cites Article 2 of the applicable regulation; the NGS module
is analyzed as a possible IVD rather than assumed to be a device; unresolved rows are marked.

### Exercise 2 — Rule 11 classification memo (application, 45 minutes)

Choose one software product and write a one-page classification memo. Include the intended purpose,
the Rule 11 branch, the harm characterization, the effect of Rules 3.2, 3.3, and 3.5, and one
alternative class reading. Deliverable: the memo with a rule-path diagram. Acceptance criteria: the
rule text is quoted; the harm argument names the clinical consequence; the alternative reading is
rejected with reasons, not ignored.

### Exercise 3 — IVDR classification (application, 40 minutes)

Classify three IVDs: a companion diagnostic for a targeted oncology therapy, a pregnancy self-test
sold directly to consumers, and a blood-grouping reagent used in a hospital laboratory. For each,
state the Annex VIII rule, the class, and whether a notified body is required. Deliverable: a table
with rule, class, route, and one sentence of reasoning. Acceptance criteria: the companion
diagnostic is recognized as Class C with consultation; the self-test is analyzed under Rule 4 and
its exception; blood grouping is analyzed under Rule 2.

### Exercise 4 — Route selection (application, 30 minutes)

For each case, select the Article 52 or Article 48 route and state whether a notified body is
required: a Class I reusable surgical instrument, a Class I sterile dressing, a Class IIb imaging
software, a Class III implantable device, and a non-sterile IVDR Class A reagent. Deliverable: a
five-row table with route, notified body status, and technical documentation depth. Acceptance
criteria: the Class I exceptions are applied correctly; the IVDR Class A case is not assumed to
need a notified body; the Class III case mentions the SSCP.

### Exercise 5 — Clinical evaluation gap critique (critique, 50 minutes)

You are given a retrospective dataset of 30,000 images from two sites, labels from radiology
reports, an external AUC of 0.86, and a claim of earlier treatment. Write a two-page critique that
identifies the evidence gaps, states which gaps are fatal and which are fixable, and proposes one
prospective study with an endpoint, comparator, and sample-size rationale. Deliverable: the
critique and a one-page study outline. Acceptance criteria: the critique separates analytical
performance from clinical benefit; the reference standard is assessed for independence; the study
endpoint matches the claim.

### Exercise 6 — Notified body engagement plan (design, 40 minutes)

For a Class IIb SaMD, write a plan for selecting and engaging a notified body. Include the scope
search, the questions for a structured dialogue, the documents to bring, and a timeline from
contract to certificate. Deliverable: a two-page plan with a milestone table. Acceptance criteria:
the plan verifies designation and scope; the questions target classification and clinical evidence;
the timeline includes surveillance and recertification, not only initial certification.

### Exercise 7 — PMS, PSUR, and vigilance calendar (design, 45 minutes)

Build a 24-month compliance calendar for a Class IIb MDR device and an IVDR Class C companion
diagnostic. Include PMS data reviews, PSUR submissions, PMCF or PMPF updates, and the reporting
clock for a serious incident. Deliverable: a month-by-month calendar with owners and triggers.
Acceptance criteria: the PSUR intervals match Article 86 and the IVDR; the vigilance clocks are
stated as 2, 10, and 15 days; the calendar includes a trigger for a software change.

### Exercise 8 — AI Act overlay memo (design, 45 minutes)

Take the Class IIb SaMD from Exercise 2 and write a one-page AI Act overlay memo. State whether it
is a high-risk AI system, which annex applies, the obligations that are not already covered by the
MDR, and the applicable date under Regulation (EU) 2026/1744. Deliverable: the memo with an
obligation table. Acceptance criteria: the memo does not treat the AI Act as a replacement for the
MDR; the date is the deferred date, not the original date; at least three obligations are mapped to
existing MDR evidence.

## 13. Self-check answers

**Question 1. What is the difference between the MDR and IVDR scope?** The MDR applies to medical
devices, including software that serves a medical purpose without examining a specimen. The IVDR
applies to in vitro diagnostic medical devices, which examine specimens derived from the human body
to provide information for diagnosis, monitoring, or treatment. A product can contain functions under
both regulations, in which case each function is analyzed separately. Applying MDR classes to an IVD
is a category error.

**Question 2. When does software take the class of a device?** Under MDR Annex VIII implementing
rule 3.3, software intended to control, monitor, or directly influence the performance of a device
is classified in the same class as that device. If the software is independent of the device, it is
classified in its own right, usually under Rule 11. A cloud algorithm that analyzes images from a
separate scanner is independent; firmware that controls a pump is part of the pump.

**Question 3. What are the Class I self-certification exceptions?** A Class I device is normally
self-certified under Article 52, with technical documentation under Annex II and III and an EU
declaration of conformity. The exceptions are devices that are sterile, have a measuring function,
or are reusable surgical instruments. For those aspects, a notified body must assess the relevant
requirements under Annex IX or Annex XI Part A. The rest of the conformity assessment can remain
self-declared.

**Question 4. How do you choose a conformity route for Class IIb?** Article 52 offers Annex IX
(quality system and technical documentation assessment) or Annex X (type examination) plus Annex XI
(production quality assurance). Both require a notified body. Annex IX is the common choice for a
software manufacturer that already runs an ISO 13485 quality system, because one assessment covers
the QMS and the technical documentation. The technical documentation is assessed per generic device
group.

**Question 5. What are the PSUR intervals and vigilance timelines?** Under MDR Article 86, the PSUR
is produced every two years for Class IIa and annually for Class IIb and Class III. Under Article
87, a serious public health threat is reported within 2 days, death or unanticipated serious
deterioration within 10 days, and other serious incidents within 15 days of awareness. The clock
starts when the manufacturer becomes aware of the incident, not when the investigation concludes.

**Question 6. Why does the AI Act not replace the MDR?** The AI Act and the MDR are separate legal
instruments with different scopes. The MDR governs the device's safety, performance, and clinical
evidence. The AI Act adds obligations for AI systems, including data governance, logging,
transparency, human oversight, and post-market monitoring. For medical devices that require
third-party conformity assessment, the AI Act treats them as high-risk under Annex I, but the MDR
conformity assessment still applies. Regulation (EU) 2026/1744 deferred the Annex I high-risk date
to 2 August 2028; it did not repeal the MDR.

**Question 7. How does IVDR classification differ from MDR classification?** The IVDR uses
Classes A, B, C, and D, based on the individual and public-health risk of an incorrect result.
Rules 1 and 2 place the highest-risk tests in Class D, subject to their exceptions, and Rule 3 places
cancer, genetic, congenital, infectious, companion diagnostic, and drug-monitoring tests in Class C.
Rule 4 governs self-testing, Rule 5 governs reagents and receptacles, and Rule 6 is the residual
Class B rule.
The MDR classes I, IIa, IIb, and III do not apply to an IVD, and using them is a category error.

**Rubric for design exercises (Exercises 6, 7, and 8).** A strong submission states the legal basis
for every step, names the owner and the trigger for every calendar item, and separates what is known
from what is assumed. A competent submission covers the main obligations but leaves one dependency
unstated, such as notified body scope or EUDAMED availability. A weak submission lists documents
without linking them to a class, a route, or a decision, and treats a guidance document as if it
were binding.

## 14. Interview questions

**1. Walk me through how you would take a software product to the EU market.** A strong answer
starts with the intended purpose, qualifies under the MDR or IVDR, classifies under Annex VIII,
selects the Article 52 or Article 48 route, and plans clinical and post-market evidence. A weak
answer starts with "find a notified body" and never states the class or the rule path.

**2. Why is Rule 11 not enough on its own?** A strong answer explains that Rule 11 is the
software-specific rule but that the implementing rules can change the result: Rule 3.2 for
accessories, Rule 3.3 for driving software, and Rule 3.5 for the strictest applicable rule. A weak
answer recites "software is IIa" and ignores the device relationship.

**3. What does a notified body actually assess for a Class IIb device?** A strong answer describes
the quality-system audit and the technical documentation assessment, the certificate's scope and
validity, and the manufacturer's continuing obligations. A weak answer treats the notified body as
an approver that makes the product legal.

**4. How would you find the clinical evaluation gap for an AI device?** A strong answer compares the
claim to the evidence, checks the reference standard, the population, the subgroups, the workflow,
and the clinical benefit endpoint, and proposes the study that closes the largest gap. A weak answer
reports only the AUC and concludes that the model performs well.

**5. What changes when the product is an IVD rather than a device?** A strong answer names the IVDR,
the A/B/C/D classes, performance evaluation under Annex XIII, companion diagnostic consultation, and
the notified body requirement for Class B and above. A weak answer says the process is the same but
with different paperwork.

**6. How do you keep post-market obligations under control?** A strong answer describes the PMS
plan, the class-based PSUR calendar, PMCF or PMPF, the vigilance clocks, and the link back to risk
management. A weak answer says "we monitor complaints" and stops.

**7. How does the AI Act change the plan?** A strong answer distinguishes the AI Act's high-risk
obligations from the MDR, names Annex I and the deferred date of 2 August 2028, and maps the
additional evidence. A weak answer says the AI Act replaces the MDR or that the deferral means
nothing needs to be done.

**8. When would you delay a CE submission?** A strong answer names a specific gap that invalidates
the claim, such as a contaminated reference standard, a failed subgroup, or a missing clinical
benefit endpoint, and explains why filing would waste an assessment slot. A weak answer says "when
the documentation is not ready" without naming the decision that the missing evidence supports.

## 15. Key terms

| Term | One-line meaning |
|---|---|
| MDR | Regulation (EU) 2017/745 on medical devices, applicable from 26 May 2021 |
| IVDR | Regulation (EU) 2017/746 on in vitro diagnostic medical devices, applicable from 26 May 2022 |
| MDD | Directive 93/42/EEC, replaced by the MDR |
| IVDD | Directive 98/79/EC on in vitro diagnostics, replaced by the IVDR |
| CE marking | Mark indicating conformity with the applicable EU harmonisation legislation |
| Conformity assessment | The process of demonstrating compliance with the MDR or IVDR |
| Notified body | Independent body designated by a member state to assess conformity |
| Competent authority | National authority responsible for market surveillance and enforcement |
| EU declaration of conformity | The manufacturer's signed statement that the device meets the requirements |
| Technical documentation | The dossier demonstrating conformity under Annex II or the IVDR equivalent |
| GSPR | General Safety and Performance Requirements in Annex I |
| Annex VIII | The classification rules for MDR devices and IVDR IVDs |
| Rule 11 | The MDR software classification rule |
| Implementing rule 3.2 | Accessories are classified in their own right |
| Implementing rule 3.3 | Software driving or influencing a device takes the device's class |
| Implementing rule 3.5 | The strictest applicable rule determines the class |
| Class IIa / IIb / III | MDR risk classes requiring notified body assessment |
| Class A / B / C / D | IVDR risk classes, from lowest to highest |
| Clinical evaluation | Systematic assessment of clinical data for safety and performance |
| PMCF | Post-market clinical follow-up |
| Performance evaluation | IVDR assessment of scientific validity, analytical performance, and clinical performance |
| PMPF | Post-market performance follow-up under the IVDR |
| PMS | Post-market surveillance |
| PSUR | Periodic Safety Update Report required at class-dependent intervals |
| Vigilance | Reporting and action on serious incidents and field safety corrective actions |
| Serious incident | Incident involving death, serious deterioration, or a serious public health threat |
| FSCA | Field safety corrective action |
| UDI | Unique Device Identification |
| Basic UDI-DI | The device-model identifier used in EUDAMED and certificates |
| EUDAMED | The European database on medical devices |
| PRRC | Person responsible for regulatory compliance |
| Authorised representative | The EU-established representative of a non-EU manufacturer |
| Legacy device | Device relying on transitional provisions under Article 120 |
| Companion diagnostic | IVD essential for the safe and effective use of a medicinal product |
| Self-testing | IVD intended for use by a lay person |
| SSCP | Summary of Safety and Clinical Performance for Class III and implantable devices |
| Harmonised standard | Standard cited in the Official Journal that gives presumption of conformity |
| MDCG | Medical Device Coordination Group, which issues non-binding guidance |
| Equivalence | Reliance on another manufacturer's clinical data with demonstrated similarity and access |
| High-risk AI system | AI Act category that includes MDR devices requiring notified body assessment |

## 16. Sources

All URLs were accessed on 25 October 2026. Dates are the publication, revision, or application dates
shown by the source at access. Verify current versions before relying on them.

1. European Parliament and Council. Regulation (EU) 2017/745 on medical devices (MDR). https://eur-lex.europa.eu/eli/reg/2017/745/oj
2. European Parliament and Council. Regulation (EU) 2017/746 on in vitro diagnostic medical devices (IVDR). https://eur-lex.europa.eu/eli/reg/2017/746/oj
3. European Parliament and Council. Regulation (EU) 2024/1689 laying down harmonised rules on artificial intelligence (AI Act). https://eur-lex.europa.eu/eli/reg/2024/1689/oj
4. European Parliament and Council. Regulation (EU) 2026/1744 amending the AI Act timelines (Digital Omnibus). https://eur-lex.europa.eu/eli/reg/2026/1744/oj/eng
5. European Parliament and Council. Regulation (EU) 2023/607 amending the MDR and IVDR transitional provisions. https://eur-lex.europa.eu/eli/reg/2023/607/oj
6. Medical Device Coordination Group. MDCG 2019-11 Rev.1, "Guidance on Qualification and Classification of Software in Regulation (EU) 2017/745 – MDR and Regulation (EU) 2017/746 – IVDR," June 2025. https://health.ec.europa.eu/document/download/b45335c5-1679-4c71-a91c-fc7a4d37f12b_en
7. Medical Device Coordination Group. MDCG 2021-24 Rev.1, "Guidance on classification of medical devices." https://health.ec.europa.eu/document/download/cbb19821-a517-4e13-bf87-fdc6ddd1782e_en
8. Medical Device Coordination Group. MDCG 2022-21, "Guidance on Periodic Safety Update Report (PSUR) according to Regulation (EU) 2017/745." https://health.ec.europa.eu/system/files/2023-01/mdcg_2022-21_en.pdf
9. Medical Device Coordination Group. MDCG 2025-6, "Guidance on the interplay between Regulation (EU) 2024/1689 (AI Act) and the EU medical devices regulations," June 2025. https://health.ec.europa.eu/system/files/2025-06/mdcg_2025-6_en.pdf
10. International Organization for Standardization. ISO 13485:2016, Medical devices — Quality management systems — Requirements for regulatory purposes. https://www.iso.org/standard/59752.html
11. International Organization for Standardization. ISO 14971:2019, Medical devices — Application of risk management to medical devices. https://www.iso.org/standard/72704.html
12. International Electrotechnical Commission. IEC 62304:2006+A1:2015, Medical device software — Software life cycle processes. https://webstore.iec.ch/en/publication/22794
13. International Electrotechnical Commission. IEC 62366-1:2015+A1:2020, Medical devices — Application of usability engineering to medical devices. https://webstore.iec.ch/
14. European Commission. New Approach Notified and Designated Organisations (NANDO) database. https://ec.europa.eu/growth/tools-databases/nando/
15. European Commission. EUDAMED, the European database on medical devices. https://ec.europa.eu/tools/eudamed/
16. International Medical Device Regulators Forum. "Software as a Medical Device: Possible Framework for Risk Categorization and Corresponding Considerations." IMDRF/SaMD WG/N12FINAL:2014, 18 September 2014. https://www.imdrf.org/documents/software-medical-device-possible-framework-risk-categorization-and-corresponding-considerations

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`07_iec-62304-software-lifecycle.md`](07_iec-62304-software-lifecycle.md)
- Related topic: [`01_definitions-and-classification.md`](01_definitions-and-classification.md)
- Related topic: [`05_qms-and-design-controls.md`](05_qms-and-design-controls.md)
- Related topic: [`09_the-evidence-package.md`](09_the-evidence-package.md)
- Related topic: [`10_post-market-surveillance.md`](10_post-market-surveillance.md)
- Survey file: [`../../course/14_medical_devices_samd_regulation.md`](../../course/14_medical_devices_samd_regulation.md)
- Labs: [`../../labs/`](../../labs/)
- Lab 14.1 regulatory triage: [`../../labs/`](../../labs/)
- Lab 14.2 intended-use statements: [`../../labs/`](../../labs/)

The module index is the authoritative map. If a sibling file is not yet present, the index marks it
as planned; do not infer content from the link text.
