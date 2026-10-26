# Medical Equipment Classes

> Module 14A — Medical Hardware, Sensors, Chips, and Robotics
> Topic 09 of 14
> Estimated study time: 4 hours
> Prerequisites: 01, 08
> Version 2026.3

## 1. Why this topic exists

"Class" is one of the most overloaded words in medical device engineering. A device has an FDA class, an EU MDR class, an applied-part class, a software safety class, an ingress-protection class, and an electromagnetic-emissions class. The classes are independent, and confusing them changes the evidence, the timeline, and the cost.

The industry problem is that a team treats one class as if it implied another. A class II device is assumed to need a 510(k) but not a clinical trial; a class IIa device is assumed to need a notified body but not a PMA; a type BF applied part is assumed to be safer than type B; a software class B is assumed to need less testing than class C. Each assumption can be wrong in a way that delays the launch or weakens the safety case.

Three decisions depend on classification literacy:

1. **Which regulatory pathway applies.** The FDA class and the EU MDR class determine the submission, the clinical evidence, and the post-market obligations.
2. **Which safety requirements apply.** The applied-part class determines the leakage limits and the isolation; the ingress class determines the cleaning and sterilization; the emissions class determines the EMC limits.
3. **How much evidence is enough.** The software class and the risk class determine the verification, the clinical evaluation, and the documentation.

This topic covers the equipment categories (monitoring, imaging, therapeutic, laboratory, implantable), the classification systems, and the mapping from class to evidence. It assumes the hardware, firmware, and power material from earlier topics. It does not give legal advice; the classification of a specific device requires a regulatory professional and the current text of the applicable regulation.

A useful framing: a class is a claim about risk, and each claim has an evidence burden. The device's function, its contact with the patient, its energy, and its failure modes determine the claim. The class is not a label on the box; it is a summary of the evidence the regulator expects.

## 2. Learning outcomes

By the end of this topic you can:

- Distinguish the FDA device class, the EU MDR class, the IEC 60601 applied-part type, the IEC 62304 software class, the IEC 60529 ingress class, and the CISPR emissions class.
- Map a device's intended use and technology to its regulatory class and pathway.
- Explain how the applied-part class changes the leakage limit and the isolation design.
- Explain how the software class changes the verification evidence.
- Identify the equipment category (monitoring, imaging, therapeutic, laboratory, implantable) and its dominant failure modes.
- Estimate the verification and clinical evidence burden for a device class.
- Critique a claim that a device is "class II" without naming the classification system.
- Build a classification record that states each class, its basis, and its evidence.

Each outcome maps to an exercise in Section 12 or a self-check item in Section 13.

## 3. The core mental model

A device has several independent classifications. Each answers a different question and has a different evidence burden.

```text
                         DEVICE
                            |
        +-------------------+-------------------+
        |                   |                   |
   REGULATORY          SAFETY                PERFORMANCE
   FDA class           applied part          software class
   EU MDR class        ingress class         emissions class
        |                   |                   |
   510(k)/De Novo/PMA   B/BF/CF, IP          A/B/C, CISPR A/B
        |                   |                   |
        +-------------------+-------------------+
                            |
                    EVIDENCE BURDEN
```

The model has five classification axes:

| Axis | Values | Question | Evidence |
|---|---|---|---|
| FDA risk class | I, II, III | How much premarket review? | 510(k), De Novo, PMA |
| EU MDR class | I, IIa, IIb, III | How much conformity assessment? | Self-declaration or notified body |
| Applied part | B, BF, CF | How much patient protection? | Leakage, isolation, defibrillation |
| Software safety | A, B, C | How much software rigour? | Plan, architecture, tests |
| Ingress / emissions | IP, CISPR A/B | How much environmental protection? | IP tests, EMC tests |

A device can be FDA class II, EU class IIa, type BF, software class B, IP22, and CISPR B. Each value is chosen for a different reason, and each has its own evidence.

### 3.6 The six axes are independent

The independence is the key idea. A change on one axis does not automatically change another.

| Change | Axis affected | Other axes |
|---|---|---|
| Add a cardiac catheter | Applied part (BF to CF) | FDA/EU may also change |
| Add a therapy function | FDA/EU class, software class | Applied part may not change |
| Move from hospital to home | Emissions, ingress, usability | FDA/EU class may not change |
| Add a diagnostic claim | Clinical evidence, EU class | Applied part may not change |
| Add a wireless radio | Emissions/immunity | Software class may change |
| Change the enclosure | Ingress | Applied part may not change |

A team that assumes "class II means no clinical evidence" or "type BF means safe" has collapsed the axes. The collapse is the source of most classification errors.

### 3.1 The FDA classes

| Class | Risk | Pathway | Examples |
|---|---|---|---|
| I | Low | Exempt or 510(k) | Bandages, tongue depressors, manual stethoscopes |
| II | Moderate | 510(k) or De Novo | Most monitors, infusion pumps, ultrasound |
| III | High | PMA | Implantable pacemakers, heart valves, some implants |

The FDA class is assigned by regulation (21 CFR Parts 862–892) and by product code. A device that is not substantially equivalent to a predicate may need De Novo or PMA. The class is not the same as the pathway: a class II device can be exempt from 510(k) if the regulation says so.

### 3.2 The EU MDR classes

| Class | Risk | Conformity assessment | Examples |
|---|---|---|---|
| I | Low | Self-declaration (with exceptions) | Non-invasive devices, some software |
| IIa | Moderate | Notified body | Many monitors, ultrasound |
| IIb | Moderate-high | Notified body | Ventilators, infusion pumps, some implants |
| III | High | Notified body, clinical evaluation | Implants, high-risk devices |

The EU class is determined by the classification rules in Annex VIII of the MDR. The rules consider the duration of contact, the invasiveness, the active or non-active nature, the energy, and the clinical consequence. The class determines the conformity assessment route and the clinical evaluation.

### 3.3 The applied-part classes

| Type | Definition | Leakage limit (AC, normal) | Typical use |
|---|---|---:|---|
| B | Not suitable for direct cardiac contact | 100 µA | General monitoring |
| BF | Floating applied part, not cardiac | 100 µA | ECG, EMG, external sensors |
| CF | Floating, cardiac contact | 10 µA | Intracardiac leads, dialysis |

The applied-part class is defined by IEC 60601-1. The limits in clause 8.7.3 are 100 µA normal and 500 µA single-fault for B and BF, and 10 µA normal and 50 µA single-fault for CF, at 50/60 Hz. The class determines the isolation, the protection, and the defibrillation withstand.

A type BF device is not "safer" than type B in every sense; it is safer for the patient because the applied part is floating, but it may have a higher leakage through the enclosure. The class is a specific protection scheme, not a general quality grade.

### 3.4 The software classes

| Class | Harm if software fails | Evidence |
|---|---|---|
| A | No injury | Basic lifecycle |
| B | Non-serious injury | Architecture, unit verification |
| C | Death or serious injury | Detailed design, unit and integration verification |

The software class is assigned per software item by the risk analysis, following IEC 62304. A device can have a class C control loop and a class A logging function. The class determines the rigour, not the processor.

### 3.5 The environmental classes

| Class | Meaning | Test |
|---|---|---|
| IP20 | No water protection; fingers protected | IEC 60529 |
| IP22 | Dripping water, 15° tilt | IEC 60529 |
| IPX7 | Temporary immersion | IEC 60529 |
| CISPR A | Industrial emissions | CISPR 11 |
| CISPR B | Residential emissions | CISPR 11 |

The ingress class determines the cleaning and sterilization, and the emissions class determines the EMC limits. A device used in the home must meet CISPR B, which is stricter than CISPR A. A device that is cleaned with a spray must meet at least IPX4.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| FDA class | Risk class I, II, or III | Confused with the EU class | I, II, III |
| EU MDR class | Risk class I, IIa, IIb, III | Confused with the FDA class | I, IIa, IIb, III |
| Applied part | Part in contact with the patient | Confused with the enclosure | B, BF, CF |
| Software class | IEC 62304 safety class | Confused with the device class | A, B, C |
| Ingress protection | IP rating | Confused with sterilization | IP code |
| Emissions class | CISPR A or B | Confused with immunity | A, B |
| 510(k) | Premarket notification | Confused with PMA | Submission |
| De Novo | New device classification | Confused with 510(k) | Submission |
| PMA | Premarket approval | Confused with 510(k) | Submission |
| Predicate | Legally marketed device for equivalence | Confused with a competitor | Device |
| Substantial equivalence | Similar safety and effectiveness | Confused with identical | Determination |
| Product code | FDA code for a device type | Confused with the regulation | Code |
| Notified body | EU conformity assessment body | Confused with the FDA | Organization |
| Clinical evaluation | EU evidence of clinical safety and performance | Confused with a clinical trial | Report |
| Post-market surveillance | EU monitoring after launch | Confused with a recall | System |
| Intended use | What the device is for | Confused with the label | Statement |
| Indication | Clinical condition the device addresses | Confused with the intended use | Statement |
| Contraindication | Condition in which the device should not be used | Confused with a warning | Statement |
| Type B | Applied part not floating, not cardiac | Confused with BF | Classification |
| Type BF | Floating applied part, not cardiac | Confused with CF | Classification |
| Type CF | Floating applied part, cardiac | Confused with BF | Classification |
| MOPP | Means of patient protection | Confused with MOOP | Insulation |
| MOOP | Means of operator protection | Confused with MOPP | Insulation |
| CISPR 11 | Emissions standard | Confused with IEC 60601-1-2 | Standard |
| IEC 60529 | Ingress protection standard | Confused with sterilization | Standard |
| Sterilization | Killing microorganisms | Confused with cleaning | Method |
| Disinfection | Reducing microorganisms | Confused with sterilization | Method |
| Single-use | Device intended for one use | Confused with disposable | Label |
| Reusable | Device intended for reprocessing | Confused with single-use | Label |
| Implantable | Device placed in the body | Confused with invasive | Classification |
| Invasive | Device that enters the body | Confused with implantable | Classification |
| Active device | Device that depends on an energy source | Confused with a powered device | Classification |
| Non-active device | Device that does not depend on energy | Confused with a passive device | Classification |

## 5. The mechanics, step by step

The procedure below classifies a device and maps each class to its evidence. Each step states its input, output, and rationale. The required edge cases are mapped to the steps and summarised in Section 5.13.

### Step 1. State the intended use

**Input:** the clinical problem, the user, the patient, and the environment.

**Output:** an intended-use statement and an indications-for-use statement.

**Rationale:** the intended use determines the class, the pathway, and the evidence. A device that "monitors vital signs" is different from one that "detects atrial fibrillation and alerts the clinician". The first may be a general monitor; the second makes a diagnostic claim that requires specific evidence.

The intended use must name the patient population, the setting, the user, and the clinical decision. A statement that is too broad increases the evidence burden; one that is too narrow may exclude the intended patients.

### Step 2. Identify the equipment category

**Input:** the intended use and the technology.

**Output:** a category: monitoring, imaging, therapeutic, laboratory, or implantable.

**Rationale:** the category determines the dominant failure modes, the standards, and the clinical evidence. A monitor's failure is a missed alarm or a wrong value; a ventilator's failure is a wrong volume or pressure; an implant's failure is a lead fracture or a battery depletion; a laboratory analyzer's failure is a wrong result.

The category is not the same as the class. A laboratory analyzer can be class II or class III depending on the assay; an implant is usually class III. The category informs the risk analysis, and the class informs the pathway.

### Step 3. Determine the FDA class and product code

**Input:** the intended use, the technology, and the FDA classification database.

**Output:** an FDA class, a product code, a regulation, and a pathway.

**Rationale:** the FDA class is assigned by the regulation for the device type. The product code identifies the specific type and the predicate. A device that fits an existing code may use the 510(k) pathway; one that does not may need De Novo.

The classification database is the source of truth, not a competitor's label. A device that is marketed without a 510(k) may be exempt, or may be misclassified; the manufacturer's determination must be documented.

### Step 4. Determine the EU MDR class

**Input:** the intended use, the duration, the invasiveness, the energy, and the Annex VIII rules.

**Output:** an EU class and the applicable rules.

**Rationale:** the EU class is determined by the classification rules, which consider the device's characteristics and the clinical risk. The rules are applied in order, and the most stringent applicable rule wins. A device that is class IIa under one rule may be class IIb under another.

The classification must be documented rule by rule, with the rationale. A classification that cites only the final class is not auditable.

### Step 5. Determine the applied-part type

**Input:** the patient contact, the electrical connection, and the cardiac contact.

**Output:** a type (B, BF, or CF) and the leakage budget.

**Rationale:** the applied-part type determines the leakage limit and the isolation. A device that contacts the heart, directly or through a conductive path, needs type CF. A device that contacts the skin and is not connected to the heart can be type BF. A device with no patient contact has no applied part.

The type must be justified by the contact, not by a preference. A type CF design is more expensive and more complex; a type BF design may be adequate and simpler.

### Step 6. Determine the software class

**Input:** the software functions and the risk analysis.

**Output:** a class per software item (A, B, or C).

**Rationale:** the software class is determined by the harm that a software failure could cause, assuming the hardware and the user are functioning. A control loop that can overdose a patient is class C; a logging function is class A.

The class is per item, not per device. A device with a class C control loop and a class A logger has two classes and two evidence burdens.

### Step 7. Determine the ingress and emissions classes

**Input:** the cleaning method, the environment, and the EMC requirement.

**Output:** an IP rating and a CISPR class.

**Rationale:** the ingress class determines the cleaning and sterilization, and the emissions class determines the EMC limits. A home device must meet CISPR B; a professional device may meet CISPR A. A device that is sprayed with cleaner must meet IPX4 or better.

The IP rating is not a sterilization claim. A device can be IPX7 and not sterile; a device can be sterile and not waterproof. The two are independent.

### Step 8. Map the classes to the evidence

**Input:** the classes and the applicable standards.

**Output:** an evidence matrix: tests, documents, and clinical data.

**Rationale:** each class has an evidence burden. The FDA class determines the submission; the EU class determines the conformity assessment; the applied part determines the leakage and isolation tests; the software class determines the lifecycle evidence; the ingress and emissions classes determine the environmental tests.

The evidence matrix is the bridge between the classification and the project plan. A class without an evidence line is a class that has not been understood.

### Step 9. Estimate the verification effort

**Input:** the evidence matrix and the test methods.

**Output:** an estimate in test hours, cost, and calendar time.

**Rationale:** the effort depends on the number of tests, the number of units, the test duration, and the laboratory availability. A full IEC 60601-1 test can take weeks; an EMC test can take days; a clinical study can take months.

The estimate must include the re-test time after a failure. A design that fails EMC once may need a board revision and a full re-test, which doubles the effort.

### Step 10. Identify the clinical evidence

**Input:** the intended use, the class, and the pathway.

**Output:** a clinical evaluation plan or a clinical trial plan.

**Rationale:** the FDA pathway determines whether a clinical study is needed. A 510(k) may rely on bench and animal data plus a literature review; a PMA usually requires a pivotal trial; an EU class IIa/IIb requires a clinical evaluation, which may be based on literature and post-market data.

The clinical evidence must match the claim. A device that claims to "improve outcomes" needs an outcome study; a device that claims to "measure heart rate" needs an accuracy study.

### Step 11. Plan the post-market obligations

**Input:** the class and the pathway.

**Output:** a post-market surveillance plan, a complaint system, and a recall procedure.

**Rationale:** a higher class brings more post-market obligations: periodic reporting, post-market clinical follow-up, and a vigilance system. The obligations are part of the lifecycle cost, not an afterthought.

The post-market plan must include the data needed to detect a drift in performance, and the trigger for a corrective action.

### Step 12. Document and review

**Input:** the classification decisions and the evidence.

**Output:** a classification record with the basis for each class.

**Rationale:** the classification record is the audit trail. It must state each class, the rule or regulation, the rationale, the evidence, and the reviewer. A record that states only the final class is not sufficient.

The record must be reviewed by regulatory, quality, and engineering. A classification that is chosen by one function without review is a classification that may not survive an audit.

### 5.13 Edge cases and where they enter

| Edge case | Step | Mechanism | Detection | Consequence |
|---|---|---|---|---|
| Saturation | 3, 9 | Wrong class increases evidence | Audit | Delay; cost |
| Common-mode rejection failure | 5 | Wrong applied-part type | Leakage test | Safety failure |
| Electrode offset | 5 | Wrong input range | Bench test | Saturation |
| Clock jitter | 6 | Wrong software class | Risk review | Insufficient evidence |
| Brown-out | 9 | Missing power-fail test | Test plan review | Field failure |
| Watchdog reset loop | 6, 9 | Missing fault test | Test plan review | Field failure |
| Thermal throttling | 9 | Missing thermal test | Test plan review | Performance failure |
| Firmware update failure | 6, 11 | Missing update evidence | Audit | Recall |
| Calibration drift | 9, 11 | Missing post-market plan | Audit | Silent bias |

The classification edge cases are mostly process failures: a class that is wrong, an evidence line that is missing, or a post-market obligation that is ignored. The technical edge cases enter through the tests that the class requires.

### 5.14 Evidence matrix template

| Class axis | Value | Evidence | Standard | Owner |
|---|---|---|---|---|
| FDA | II | 510(k) | 21 CFR 807 | Regulatory |
| EU | IIa | Notified body | MDR Annex IX | Regulatory |
| Applied part | BF | Leakage, isolation | IEC 60601-1 | Electrical |
| Software | B | Architecture, unit tests | IEC 62304 | Software |
| Ingress | IP22 | IP test | IEC 60529 | Mechanical |
| Emissions | CISPR B | EMC test | CISPR 11 | EMC |

The matrix is the project plan in one page. Each row has an owner, a standard, and an evidence item. A row without an owner is a row that will be late.

### 5.15 Effort estimate template

| Activity | Hours | Cost at $150/h | Calendar |
|---|---:|---:|---|
| Electrical safety | 120 | $18,000 | 3 weeks |
| EMC | 60 | $9,000 | 2 weeks |
| Software verification | 200 | $30,000 | 6 weeks |
| Biocompatibility | 80 | $12,000 | 8 weeks |
| Usability | 60 | $9,000 | 4 weeks |
| Clinical evaluation | 300 | $45,000 | 12 weeks |
| Regulatory submission | 100 | $15,000 | 6 weeks |
| **Total** | **920** | **$138,000** | **12 weeks critical path** |

The estimate is illustrative. The point is that the class determines the rows, and the rows determine the cost and the calendar. A class I device may need a fraction of this; a class III implant may need multiples.

### 5.16 Equipment categories and dominant failure modes

| Category | Function | Dominant failure | Class tendency |
|---|---|---|---|
| Monitoring | Measure and display physiology | Missed alarm; wrong value | II / IIa–IIb |
| Imaging | Form an image of anatomy or function | Artifact; wrong measurement | II / IIa–IIb |
| Therapeutic | Deliver energy, drug, or support | Over/under-delivery | II–III / IIb–III |
| Laboratory | Analyze a sample | Wrong result; contamination | II–III / IIa–III |
| Implantable | Replace or support a function | Lead failure; battery; infection | III / III |

The category is not the class, but it predicts the failure mode and the evidence. A monitor's clinical evidence is an accuracy study; a ventilator's is a performance and safety study; an implant's is a long-term outcome study.

### 5.17 Classification decision table

| Question | If yes | If no |
|---|---|---|
| Does the device contact the heart or a cardiac path? | Type CF; likely class III | Continue |
| Does the device deliver a therapy that can injure? | Class IIb/III; software C | Continue |
| Is the device implantable? | Class III; PMA | Continue |
| Does the device make a diagnostic claim? | Clinical evidence; class II/IIa+ | Continue |
| Is the device used at home? | CISPR B; IP rating; usability | Continue |
| Is the device active? | MDR active-device rules | Non-active rules |

The table is a first pass, not a classification. Each answer must be documented with the regulation or rule that supports it.

### 5.18 Pathway comparison

| Pathway | When | Evidence | Time |
|---|---|---|---|
| 510(k) | Substantial equivalence to a predicate | Bench, software, biocompatibility, usability | 3–12 months |
| De Novo | No predicate; low-moderate risk | Similar to 510(k) plus classification request | 6–18 months |
| PMA | High risk | Clinical trial, manufacturing, inspection | 1–3 years |
| EU self-declaration | Class I | Technical file, clinical evaluation | 3–6 months |
| EU notified body | Class IIa+ | Technical file, clinical evaluation, audit | 6–18 months |

The pathway determines the clinical evidence and the calendar. A 510(k) with a good predicate can be faster than a De Novo; a PMA is a multi-year program.

### 5.19 Post-market obligations

| Class | Obligations |
|---|---|
| FDA I | Establish complaint files; MDR if applicable |
| FDA II | MDR, recalls, annual reporting for some |
| FDA III | MDR, PMA supplements, annual reports |
| EU I | PMS, vigilance |
| EU IIa | PMS, PSUR, vigilance |
| EU IIb/III | PMS, PSUR, PMCF, vigilance |

The post-market obligations are part of the lifecycle cost. A device that is launched without a surveillance plan is a device that cannot detect a drift.

### 5.20 Ten classification review questions

1. What is the intended use, and what decision does the device support?
2. What is the FDA class, product code, and regulation?
3. What is the EU class, and which Annex VIII rules apply?
4. What is the applied-part type, and what is the leakage budget?
5. What is the software class for each item?
6. What is the ingress class, and what cleaning does it support?
7. What is the emissions class, and what environment does it support?
8. What clinical evidence does the claim require?
9. What post-market obligations apply?
10. What is the evidence matrix, and who owns each row?

A review that answers all ten has covered the classification. A review that answers only the second and third has covered the submissions, not the device.

### 5.21 Common classification mistakes

- Treating the FDA class and the EU class as equivalent.
- Assuming class II means no clinical evidence.
- Assuming type BF and type B have the same single-fault limit.
- Assigning one software class to the whole device.
- Ignoring the home-use emissions class.
- Confusing IP rating with sterilization.
- Classifying after the design freeze.
- Using a competitor's label as the classification basis.
- Ignoring the post-market obligations in the budget.

Each mistake is a schedule or safety risk. The fix is a classification record with a named reviewer.

### 5.27 Combination products and multiple codes

A device that combines a drug and a device, or two device types, may be a combination product. In the United States, the lead center (CDRH, CDER, or CBER) is assigned by the product's primary mode of action. In the EU, the combination may be regulated as a device with a medicinal substance, or as a medicinal product with a device.

A device with multiple functions may have multiple product codes. The most stringent code governs the submission, and the evidence must cover all the functions. A device that adds a class III function to a class II base is a class III device for that function.

### 5.28 Classification worksheet

| Item | Entry | Basis |
|---|---|---|
| Intended use | Monitor sleep apnea at home | Clinical claim |
| User | Patient, with clinician review | Home use |
| Patient contact | Skin, nasal, finger | Applied part |
| Duration | One night, repeated | Contact duration |
| Energy | Battery, low voltage | Active device |
| FDA class | II | Product code |
| EU class | IIa | Annex VIII rules |
| Applied part | BF | No cardiac path |
| Software class | A/B | Harm analysis |
| Ingress | IP22 | Cleaning method |
| Emissions | CISPR B | Home environment |
| Clinical | Literature + usability | Claim |
| Post-market | PMS | Class |

The worksheet is the record. A worksheet with an empty basis column is a classification without a rationale.

### 5.22 Applied-part leakage arithmetic

The patient leakage limit is a current, not a voltage. For a type CF applied part, the limit is 10 µA at 50/60 Hz in normal condition. The current through the patient is the source voltage divided by the impedance of the path. A 250 V mains source with a 25 MΩ isolation impedance produces 10 µA, which is exactly the limit. A 10 MΩ isolation produces 25 µA, which fails.

The isolation impedance is not a single component. It includes the transformer, the optocouplers, the PCB spacing, and the enclosure. The measurement is made with the patient lead connected to a measuring device that represents the patient. The test is a system test, not a component test.

### 5.23 Software class examples

| Function | Failure consequence | Class |
|---|---|---|
| Log sensor data | Data gap; clinician can see | A |
| Display a trend | Misleading trend; clinician may act | B |
| Detect an alarm condition | Missed alarm | B or C |
| Control a drug dose | Overdose | C |
| Control a ventilator | Over/under-ventilation | C |
| Manage a firmware update | Bricked device | B or C |

The class depends on the harm, not on the complexity. A simple dose calculator can be class C; a complex display can be class B.

### 5.24 Ingress test table

| IP rating | First digit (solids) | Second digit (water) | Typical use |
|---|---|---|---|
| IP20 | Fingers | None | Office, clean room |
| IP22 | Fingers | Dripping, 15° | Home, clinic |
| IP44 | 1 mm | Splashing | Cleaning |
| IP54 | Dust-protected | Splashing | Harsh environment |
| IP67 | Dust-tight | Immersion 1 m | Portable, outdoor |
| IPX7 | Not specified | Immersion 1 m | Wearable |

The IP rating is a pair of digits; a missing digit means "not specified". IPX7 does not imply dust protection. A device that is cleaned with a spray needs at least IPX4; one that is immersed needs IPX7.

### 5.25 Pathway decision tree

| Condition | FDA pathway | EU pathway |
|---|---|---|
| Predicate exists; moderate risk | 510(k) | IIa/IIb notified body |
| No predicate; low-moderate risk | De Novo | IIa/IIb notified body |
| High risk; no predicate | PMA | III notified body |
| Low risk; exempt | Exempt | I self-declaration |
| Novel technology; high risk | PMA plus advisory panel | III notified body |

The tree is a first pass. The actual pathway depends on the product code, the predicate, and the clinical claim. A device that is "novel" but low-risk may use De Novo; a device that is "similar" but high-risk may need PMA.

### 5.26 The classification record

- Intended use and indications.
- FDA class, product code, regulation, and predicate.
- EU class and the Annex VIII rules applied.
- Applied-part type and leakage budget.
- Software class per item.
- Ingress and emissions classes.
- Clinical evidence plan.
- Post-market plan.
- Reviewer and date.

## 6. Worked example

### 6.1 The device

A home sleep apnea monitor records nasal airflow, respiratory effort, oxygen saturation, and body position. It is used by a patient at home for one night, and the data are reviewed by a clinician. The device is not intended to diagnose sleep apnea on its own; it provides data for a clinician's interpretation.

### 6.2 The FDA classification

The device is a class II device under the sleep apnea monitoring product code. The pathway is 510(k), with a predicate device that has the same sensors and the same intended use. The submission includes bench performance, software documentation, biocompatibility for the patient-contact materials, and a usability study.

The classification is not class III because the device does not make a therapeutic decision and does not contact the heart. It is not class I because the data influence a clinical diagnosis and the device has patient contact with sensors.

### 6.3 The EU classification

Under the MDR, the device is class IIa. It is an active device used for monitoring a physiological process, and it is not intended for a critical clinical decision without clinician review. The conformity assessment requires a notified body, a clinical evaluation based on literature and a small usability study, and post-market surveillance.

The classification is not class I because the device is active and provides information used for diagnosis. It is not class IIb because it does not administer or control a therapy and its failure is not immediately life-threatening.

### 6.4 The applied-part type

The device has three patient contacts: a nasal cannula (airflow), a chest effort belt (piezoelectric), and a finger pulse oximeter. The cannula and the belt are type BF; the pulse oximeter is type BF. The device is battery-powered and isolated, so the leakage path is small.

The applied-part type is not CF because there is no cardiac contact and no conductive path to the heart. A type CF design would add isolation and cost without a clinical benefit.

### 6.5 The software class

The software has three items:

- The acquisition and storage of the signals: class A, because a failure produces a data gap that the clinician can see.
- The automatic event detection (apnea/hypopnea index): class B, because a failure could mislead the clinician but is reviewed.
- The alarm for low oxygen saturation: class B, because a missed alarm could delay care but the device is used at home with a clinician review.

None of the items is class C, because the device does not deliver a therapy and does not make an autonomous diagnosis.

### 6.6 The ingress and emissions classes

The device is used at home, so it must meet CISPR B emissions. It is cleaned with a damp cloth, not immersed, so IP22 is adequate. The pulse oximeter finger probe is IPX1 at most; the main unit is IP22.

### 6.7 The evidence burden

The evidence matrix is:

| Class axis | Value | Evidence | Test hours |
|---|---|---|---|
| FDA | II | 510(k) | 80 |
| EU | IIa | Notified body | 40 |
| Applied part | BF | Leakage, isolation | 40 |
| Software | B | Architecture, unit tests | 120 |
| Ingress | IP22 | IP test | 8 |
| Emissions | CISPR B | EMC test | 40 |
| Biocompatibility | Surface, prolonged | ISO 10993 | 60 |
| Usability | Home use | Formative and summative | 50 |
| Clinical | Literature plus usability | Clinical evaluation | 120 |
| **Total** | | | **558** |

At $150 per hour, the direct test cost is $83,700. The calendar critical path is the clinical evaluation and the notified-body review, at about 16 weeks.

The arithmetic closes: 558 hours, $83,700, 16 weeks. The estimate is not a quote; it is a planning number that shows the order of magnitude. A class III implant with a pivotal trial would be an order of magnitude larger.

### 6.8 Diagnostic 1: the device is misclassified as class I

An advisor suggests that the device is class I because it is "just a monitor" and does not treat the patient. The team accepts the suggestion and plans a self-declaration, with no 510(k) and no notified body.

The diagnostic is to check the FDA product code and the MDR classification rules. The product code for sleep apnea monitoring is class II; the MDR rule for active monitoring devices used for diagnosis is IIa. The device does not fit class I.

The design flaw is the classification, not the device. The consequence is a launch delay of 6–12 months, a retroactive submission, and a possible recall. The fix is to classify before the design freeze, not after.

### 6.9 Diagnostic 2: the applied part is misclassified as B

A supplier provides a pulse oximeter probe that is marked type B, not BF. The team assumes that B and BF are equivalent because both have a 100 µA normal-condition limit.

The diagnostic is to check the isolation: a type B applied part is not floating, so the patient current can flow through the device to earth. A type BF applied part is floating, so the current is limited by the isolation. The leakage test with the probe connected to a patient simulator shows a current above the BF limit in a single-fault condition.

The design flaw is the assumption that the normal-condition limit is the only requirement. The single-fault limit and the isolation are different. The fix is to use a BF probe or to add isolation, which changes the cost and the schedule.

### 6.10 Vary one assumption: the device adds a therapy function

Change the intended use to include an automatic positive airway pressure (APAP) therapy. The device now delivers a therapy, not only monitors. The FDA class becomes II (or III, depending on the claim), the EU class becomes IIb, the software class for the therapy control becomes C, and the clinical evidence requires a trial.

The evidence burden rises from 558 hours to about 1,400 hours, and the calendar from 16 weeks to 40 weeks. The arithmetic:

\[
\frac{1{,}400}{558} = 2.5
\]

The effort more than doubles. The change is not a feature; it is a new device with a new class and a new evidence burden.

### 6.11 What the example demonstrates

The arithmetic closes: 558 test hours, $83,700, 16 weeks, and a 2.5× increase when the therapy function is added. The two diagnostics reveal different flaws: one is a classification error, the other is an applied-part error. The sensitivity analysis shows that a change in intended use changes every class and every evidence line.

The general method is to state the intended use, identify each class on its own axis, map each class to evidence, and estimate the effort. A classification that is done after the design is a classification that will be redone.

### 6.12 A second device: the infusion pump

An infusion pump delivers a drug at a programmed rate. It is FDA class II (510(k)) and EU class IIb (notified body). The applied part is type CF if it connects to a central line; otherwise type BF. The software class is C for the dose control and B for the user interface. The pump is used in a professional facility, so CISPR A is acceptable, but many hospitals require CISPR B.

| Axis | Sleep monitor | Infusion pump |
|---|---|---|
| FDA | II, 510(k) | II, 510(k) |
| EU | IIa | IIb |
| Applied part | BF | BF or CF |
| Software | A/B | B/C |
| Emissions | CISPR B | CISPR A/B |
| Clinical | Literature + usability | Performance + usability |
| Post-market | PMS | PMS + PSUR + PMCF |

The pump's evidence burden is higher because the therapy can injure. The class difference (IIa vs IIb) reflects the consequence, not the technology.

### 6.13 Verification matrix

| Test | Sleep monitor | Infusion pump |
|---|---|---|
| Leakage | BF limits | BF/CF limits |
| EMC | CISPR B | CISPR A/B |
| Software | Class B | Class C |
| Biocompatibility | Surface, prolonged | Surface, prolonged; fluid path |
| Usability | Home use | Clinical use; dose errors |
| Clinical | Accuracy | Dose accuracy; occlusion |
| Post-market | PMS | PMS + PMCF |

The matrix shows that the pump needs more evidence in every row. The difference is the harm, not the complexity.

### 6.14 Cost sensitivity

If the sleep monitor's software is reclassified from B to C, the software verification hours rise from 120 to 220, and the total from 558 to 658, a 18% increase. If the pump adds a CF applied part, the leakage and isolation tests add 40 hours, a 7% increase. If the pump adds a home-use claim, the EMC test changes from CISPR A to B and the usability study expands, a 15% increase.

The arithmetic shows that the class changes are not linear. A software class change is larger than an applied-part change for this device; a home-use claim is larger than a CF change. The project plan must be updated when a class changes.

### 6.15 Regulatory timeline

| Phase | Sleep monitor | Infusion pump |
|---|---:|---:|
| Design and verification | 12 weeks | 20 weeks |
| Clinical evaluation | 12 weeks | 24 weeks |
| Submission preparation | 6 weeks | 10 weeks |
| Agency review | 12 weeks | 24 weeks |
| **Total** | **42 weeks** | **78 weeks** |

The pump's timeline is 1.9× the monitor's. The difference is the clinical evidence and the review, not the hardware. A team that plans the hardware and forgets the regulatory timeline will miss the launch.

### 6.18 The cost of a class change

| Change | Added hours | Added cost | Added calendar |
|---|---:|---:|---:|
| Software B to C | 100 | $15,000 | 4 weeks |
| BF to CF | 40 | $6,000 | 3 weeks |
| CISPR A to B | 20 | $3,000 | 2 weeks |
| Add clinical trial | 400 | $60,000 | 24 weeks |
| Class IIa to IIb | 120 | $18,000 | 8 weeks |

The table shows that the clinical trial is the largest single cost, followed by the software class change. A team that plans a class change without these numbers will underestimate the launch.

The numbers are illustrative; the ratios are the point. A software class change is cheaper than a clinical trial but more expensive than an EMC re-test. The class change should be decided before the design freeze, when the cost is still a design choice.

### 6.16 A third device: the implantable pacemaker

An implantable pacemaker is FDA class III (PMA) and EU class III. The applied part is type CF, because it contacts the heart. The software class is C for the therapy control. The device is not user-serviceable; the battery must last years, and the firmware update, if any, is through a proprietary programmer.

| Axis | Pacemaker |
|---|---|
| FDA | III, PMA |
| EU | III, notified body |
| Applied part | CF |
| Software | C |
| Ingress | Hermetic, not IP |
| Clinical | Pivotal trial, long-term follow-up |
| Post-market | MDR, PMA supplements, registry |

The pacemaker's evidence burden is the largest because the failure can be immediate and the device cannot be easily replaced. The class reflects the consequence, not the number of functions.

The residual risks are the battery depletion, the lead fracture, the infection, and the firmware update. Each has a mitigation and a post-market surveillance plan. The class does not remove the risks; it determines the evidence that the risks are controlled.

The classification record for the pacemaker includes the PMA, the CF leakage evidence, the class C software, the hermetic packaging, and the long-term clinical follow-up.
## 7. Code, query, or calculation

The script reproduces the evidence-hour total, the direct cost, and the therapy-function ratio in Section 6. The evidence items are the rows of the matrix; the hours are estimates for planning, not quotes.

**Assumptions.** The nine evidence items from Section 6.7, a blended rate of $150 per hour, and a 1,400-hour estimate for the therapy version of the device.

```python
evidence = [
    ("FDA 510(k)", 80),
    ("EU notified body", 40),
    ("Leakage and isolation", 40),
    ("Software class B", 120),
    ("Ingress IP22", 8),
    ("EMC CISPR B", 40),
    ("Biocompatibility", 60),
    ("Usability", 50),
    ("Clinical evaluation", 120),
]
total = sum(h for _, h in evidence)
for name, hours in evidence:
    print(f"{name:28s} {hours:4d} h")
print(f"{'TOTAL':28s} {total:4d} h")
print(f"cost at $150/h: ${total*150:,}")
print(f"ratio with therapy function: {1400/total:.2f}")
```

**Expected output.**

```text
FDA 510(k)                     80 h
EU notified body               40 h
Leakage and isolation          40 h
Software class B              120 h
Ingress IP22                    8 h
EMC CISPR B                    40 h
Biocompatibility               60 h
Usability                      50 h
Clinical evaluation           120 h
TOTAL                         558 h
cost at $150/h: $83,700
ratio with therapy function: 2.51
```

The output shows that the software and the clinical evaluation are the two largest items, at 120 hours each, or 43% of the total. The hardware safety tests (leakage, ingress, EMC) are 88 hours, or 16%. The classification effort is dominated by software and clinical evidence, not by the electrical tests.

The therapy-function ratio of 2.51 is the sensitivity result: adding a therapy function more than doubles the evidence burden. The ratio is not a law; it depends on the device and the claim. It is a planning number that shows the order of magnitude.

**A note on the estimate.** The hours are illustrative. A real estimate must use the laboratory's quotes, the number of units, the test duration, and the re-test allowance. A design that fails EMC once may need a board revision and a full re-test, which can add 50% to the EMC line.

### 7.1 Class lookup

The script prints the six classification axes for the sleep monitor. The dictionary is a record, not a classifier; each value has a basis in the classification worksheet.

```python
classes = {
    "FDA": "II",
    "EU": "IIa",
    "applied part": "BF",
    "software": "B",
    "ingress": "IP22",
    "emissions": "CISPR B",
}
for axis, value in classes.items():
    print(f"{axis:14s} {value}")
print(f"axes {len(classes)}")
```

**Expected output.**

```text
FDA            II
EU             IIa
applied part   BF
software       B
ingress        IP22
emissions      CISPR B
axes 6
```

The output is deliberately simple. The value of the record is not the print; it is the discipline of naming each axis and its basis. A device with six axes and six bases is a device that can survive an audit.

## 8. Data, standards, and regulatory context

The classification systems are defined by regulations and standards. The table separates the binding regulations from the consensus standards.

| Instrument | Scope | Status | What it classifies |
|---|---|---|---|
| FDA 21 CFR 860 | Classification procedures | Binding regulation (U.S.) | FDA class and product code |
| FDA 21 CFR 862–892 | Device regulations by medical specialty | Binding regulation (U.S.) | Device type and class |
| FDA 21 CFR 807 | Premarket notification | Binding regulation (U.S.) | 510(k) pathway |
| EU MDR 2017/745, Annex VIII | Classification rules | Binding regulation (EU) | EU class |
| IEC 60601-1:2005+AMD1:2012+AMD2:2020 | General safety | Consensus standard | Applied-part type, leakage |
| IEC 62304:2006+AMD1:2015 | Software lifecycle | Consensus standard | Software class A/B/C |
| ISO 14971:2019 | Risk management | Consensus standard | Risk file |
| IEC 60529:1989+AMD1:1999+AMD2:2013 | Ingress protection | Consensus standard | IP code |
| CISPR 11 / IEC 60601-1-2 | Emissions and immunity | Consensus standard | Emissions class |
| ISO 10993-1:2025 | Biocompatibility | Consensus standard | Contact category and duration |

### 8.1 The FDA classification

The FDA assigns a device to class I, II, or III by regulation. Class I devices are low risk and often exempt from premarket notification; class II devices are moderate risk and usually require a 510(k); class III devices are high risk and require a PMA. The product code identifies the specific device type and the regulation.

The classification is not a single decision. A device can have multiple product codes if it combines functions, and the most stringent code governs the submission. A device that adds a function with a class III code may need a PMA even if the base device is class II.

### 8.2 The EU classification

The MDR's Annex VIII contains the classification rules. The rules consider the duration of contact, the invasiveness, whether the device is active, whether it delivers energy or a drug, and the clinical consequence. The rules are applied in order; the most stringent applicable rule determines the class.

A class I device can be self-declared, with exceptions for sterile devices, measuring functions, and reusable surgical instruments. Class IIa, IIb, and III require a notified body. The clinical evaluation is required for all classes, but its depth depends on the class and the claim.

### 8.3 The applied-part classification

IEC 60601-1 clause 8.7.3 sets the patient leakage limits by applied-part type. For alternating current at 50/60 Hz, type B and BF have a 100 µA normal-condition and 500 µA single-fault limit; type CF has 10 µA and 50 µA. The type also determines the isolation and the defibrillation withstand.

The type is a property of the applied part and its connection, not of the whole device. A device can have a type BF sensor and a type CF catheter connected at different times; each configuration must be evaluated.

### 8.4 The software classification

IEC 62304 assigns class A, B, or C by the harm that a software failure could cause. The class determines the lifecycle evidence: class A needs basic planning; class B needs architecture and unit verification; class C needs detailed design, unit and integration verification, and regression testing.

The class is per software item. A device with a class C therapy control and a class A logger has two classes. The classification must be documented in the software development plan.

### 8.5 The environmental classifications

IEC 60529 defines the IP code. The first digit is solids, the second is water; a missing digit means "not specified". CISPR 11 defines the emissions classes: class A for industrial, class B for residential. IEC 60601-1-2 sets the immunity test levels by environment.

A home-use device must meet CISPR B and the home healthcare immunity levels. A professional device may meet CISPR A and the professional levels. The environment is part of the intended use, not a marketing choice.

### 8.6 What the classifications do not cover

The classifications do not specify:

- The clinical evidence's size or design.
- The software's architecture or language.
- The applied part's material.
- The post-market surveillance's frequency.

Those are determined by the risk analysis, the clinical evaluation, and the quality system. A class is a summary of the evidence, not a substitute for it.

### 8.7 Global markets

A device launched in multiple markets needs a classification for each. The FDA class, the EU class, the Health Canada class, the TGA class, and the PMDA class are different systems with different evidence. A device that is class II in the United States may be class IIb in the EU and class II in Canada.

The differences are not only paperwork. A market may require a local clinical study, a local representative, or a different labelling. The global plan must include the classification and the evidence for each market, not only the home market.

### 8.8 A note on the current regulation

The MDR and the FDA regulations change. The MDR has been amended since 2017, and the FDA issues guidance and proposed rules. A classification that was correct at the design freeze may be wrong at launch.

The regulatory team must monitor the changes and re-review the classification before the submission. The monitoring is a process, not a one-time check.

## 9. Industry practice and case evidence

### 9.1 Most monitoring devices are class II

A multi-parameter patient monitor, a pulse oximeter, and an ultrasound system are typically FDA class II and EU class IIa. The class reflects the moderate risk: a wrong value or a missed alarm can harm, but the device does not deliver a therapy and the clinician is in the loop.

The class II pathway is a 510(k) with a predicate. The evidence includes bench accuracy, software documentation, biocompatibility, and usability. A clinical trial is usually not required, but a literature review and a usability study are.

**Documented fact:** the device types and their classes are listed in the FDA classification database. **Inference:** the specific class for a new device depends on its intended use and product code.

### 9.2 Therapeutic devices are class IIb or III

An infusion pump, a ventilator, and a defibrillator are typically FDA class II or III and EU class IIb or III. The class reflects the higher risk: the device delivers energy or a drug, and a failure can injure or kill.

The evidence includes performance testing, software class C, usability with use-error analysis, and often a clinical study. The post-market obligations include periodic reporting and field safety notices.

**Documented fact:** the device types and their classes are listed in the FDA classification database and the MDR. **Inference:** the specific class depends on the therapy and the claim.

### 9.3 Implants are class III

A pacemaker, a heart valve, and a cochlear implant are FDA class III and EU class III. The class reflects the highest risk: the device is implanted, cannot be easily removed, and a failure can be immediate.

The evidence includes a pivotal trial, long-term follow-up, manufacturing inspection, and a PMA. The post-market obligations include supplements for changes, annual reports, and sometimes a registry.

**Documented fact:** the device types and their classes are listed in the FDA classification database. **Inference:** the specific evidence depends on the implant and the claim.

### 9.4 Classification errors are expensive

A device that is misclassified as class I can be launched without a 510(k), then be found non-compliant and recalled or removed from the market. A device that is misclassified as class II when it needs a PMA can be launched and then stopped. The cost is not only the submission; it is the delay, the remediation, and the reputational damage.

The defence is a classification record with a named reviewer and a documented basis. The record should be reviewed at the design freeze and updated when the intended use changes.

### 9.5 What good classification practice looks like

Teams that avoid classification surprises tend to:

- Write the intended use before the design.
- Check the FDA product code and the MDR rules.
- Classify each axis separately.
- Map each class to an evidence row.
- Estimate the effort and the calendar.
- Review the classification with regulatory and quality.
- Re-review when the intended use or the technology changes.

The practices are ordinary. The failures they prevent are not.

### 9.6 A note on product codes

The FDA product code is a three-letter code that identifies a device type. A device may have multiple codes if it combines functions. The code determines the regulation, the class, and the predicate.

A team that chooses a code by searching for a similar device may choose the wrong code. The code must be confirmed against the regulation and the device's intended use. A wrong code can lead to a wrong predicate and a wrong pathway.

### 9.7 Examples of class assignments

| Device | FDA | EU | Applied part | Software |
|---|---|---|---|---|
| Wearable ECG patch | II | IIa | BF | B |
| Home sleep monitor | II | IIa | BF | B |
| Infusion pump | II | IIb | BF/CF | C |
| Ventilator | II | IIb | BF | C |
| Pacemaker | III | III | CF | C |
| Ultrasound | II | IIa | BF | B |

The table is illustrative, not a classification. The actual class depends on the intended use and the current regulation.

### 9.8 A note on the notified body

A notified body is an independent organisation that assesses conformity. The notified body's scope must cover the device's technology and the classification. A notified body that is not designated for the device's code cannot issue the certificate.

The choice of notified body is a project decision. The queue, the fee, and the scope vary. A device that is ready for submission but has no notified body with the right scope is not ready.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong FDA class | No predicate; unusual product code | Misread regulation | Launch delay; recall | Classification review |
| Wrong EU class | Rule not documented | Misapplied Annex VIII | Notified body mismatch | Rule-by-rule record |
| Wrong applied part | Type B used for cardiac path | Misunderstood contact | Leakage failure | Contact analysis; CF design |
| Wrong software class | One class for the whole device | Misread IEC 62304 | Insufficient evidence | Per-item classification |
| Wrong ingress class | IP rating assumed | Cleaning method mismatch | Water damage | IP test; cleaning validation |
| Wrong emissions class | CISPR A for home use | Environment mismatch | Interference; non-compliance | CISPR B design |
| Missing clinical evidence | Claim broader than data | Claim/evidence mismatch | Submission rejection | Clinical evaluation plan |
| Missing post-market plan | No PMS budget | Obligation overlooked | Audit finding | PMS plan |
| Classification after design | Late change | Design not aligned | Redesign | Classify at concept |
| Competitor label used | Assumed class | No basis | Wrong pathway | Database check |
| Multiple product codes ignored | One code considered | Combined functions | Wrong pathway | Code review |
| Exemption assumed | No 510(k) filed | Exemption misread | Illegal marketing | Regulation check |

The most dangerous failure is the wrong applied-part class, because it is a safety failure, not only a regulatory one. A type B device connected to a cardiac path can exceed the CF leakage limit and injure the patient.

### 10.1 Ten red flags in classification records

1. One class for the whole device.
2. No stated intended use.
3. No product code or regulation.
4. No Annex VIII rule.
5. Applied part chosen by preference, not contact.
6. Software class equal to the device class.
7. Home use without CISPR B.
8. Clinical claim without clinical evidence.
9. Post-market plan missing from the budget.
10. Classification dated after the design freeze.

Each is a prompt to review, not a conclusion.

### 10.2 What to do when the class is disputed

1. Re-read the intended use and the claim.
2. Check the product code and the regulation.
3. Check the Annex VIII rules.
4. Document the basis for each axis.
5. Ask regulatory and quality to review.
6. If the dispute remains, request a pre-submission meeting with the agency.

A dispute is cheaper before the submission than after. The pre-submission meeting is the formal way to resolve it.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The intended use is stable | Classification | Scope change | Design review |
| The product code is correct | FDA pathway | New function | Database check |
| The MDR rules are correctly applied | EU class | Rule misread | Regulatory review |
| The applied part is correctly identified | Leakage design | Contact change | Contact analysis |
| The software class is per item | Evidence | One class for all | Software plan |
| The environment is correctly stated | IP and EMC | Home versus professional | Intended-use review |
| The clinical claim matches the evidence | Submission | Overclaim | Clinical evaluation |
| The post-market obligations are funded | Lifecycle | Budget cut | PMS plan |
| The classification is reviewed | Audit | No reviewer | Record |
| The regulation is current | Compliance | Amendment | Regulatory monitoring |
| The device's functions are fully listed | Product codes | Hidden function | Requirements review |
| The exemption is applicable | No 510(k) | Misread | Regulation check |

The dominant assumption is that the intended use is stable. A change in the intended use, even a small one, can change the class and the evidence. The classification must be re-reviewed whenever the intended use changes.

A second limit is that the classification is jurisdiction-specific. A device that is class II in the United States may be class IIb in the EU, and the evidence may differ. A global launch needs a classification for each market.

A third limit is that the classifications are simplified in this topic. The actual rules are detailed and change over time. The topic teaches the structure; the specific classification requires the current regulation and a regulatory professional.

### 11.1 Assumption audit questions

- Which assumption would change the clinical decision if it were false?
- Which assumption has no regulatory basis?
- Which assumption was true for a predicate but false for this device?
- Which assumption depends on the intended use or the claim?
- Which assumption is encoded in a product code or a rule?
- Which assumption is specific to one market?
- Which assumption is monitored after launch, and who acts when it fails?
- Which assumption has a safe fallback?

A review that cannot answer the last two has risks without controls.

## 12. Exercises

### Exercise 1. Distinguish the classes (recall, 15 minutes)

For a wearable ECG patch, name the likely FDA class, EU class, applied-part type, software class, and ingress class. State one reason for each.

**Deliverable:** a five-row table.

**Acceptance criteria:** each class named; reason; no class inferred from another.

**Time:** 15 minutes.

### Exercise 2. Map a device to a pathway (application, 25 minutes)

A new home sleep monitor has no predicate. What FDA pathway applies, and what evidence is needed? What EU class applies?

**Deliverable:** a pathway and an evidence list.

**Acceptance criteria:** De Novo or 510(k) justified; EU class with rules; evidence list.

**Time:** 25 minutes.

### Exercise 3. Compute a leakage limit (application, 20 minutes)

A type CF applied part is connected to a 250 V source through a 30 MΩ isolation impedance. Compute the leakage current and compare it with the 10 µA limit.

**Deliverable:** a calculation and a verdict.

**Acceptance criteria:** Ohm's law; units; comparison.

**Time:** 20 minutes.

### Exercise 4. Classify a software item (application, 20 minutes)

A software function computes an insulin dose from a glucose reading and a patient parameter. What class is it, and why? What evidence does the class require?

**Deliverable:** a class and an evidence list.

**Acceptance criteria:** class C justified by harm; evidence matches IEC 62304.

**Time:** 20 minutes.

### Exercise 5. Diagnose a misclassification (critique, 30 minutes)

A device is launched as class I with a self-declaration. A clinician uses it to diagnose a condition. What is the classification error, and what is the consequence?

**Deliverable:** a diagnosis and a consequence.

**Acceptance criteria:** diagnostic claim identified; class and pathway; consequence.

**Time:** 30 minutes.

### Exercise 6. Build an evidence matrix (design, 35 minutes)

Build an evidence matrix for a home infusion pump. Include FDA, EU, applied part, software, ingress, emissions, biocompatibility, usability, clinical, and post-market.

**Deliverable:** a matrix with at least 10 rows.

**Acceptance criteria:** each class mapped to evidence; owner; standard.

**Time:** 35 minutes.

### Exercise 7. Estimate the effort (application, 30 minutes)

Using the template in Section 5.15, estimate the effort for a class IIb pump with a class C control loop and a clinical study. State the assumptions.

**Deliverable:** an estimate with hours and cost.

**Acceptance criteria:** rows justified; total; assumptions.

**Time:** 30 minutes.

### Exercise 8. Evaluate a class claim (critique, 25 minutes)

A vendor says its device is "class II, so no clinical evidence is needed." Write five questions that would test the claim.

**Deliverable:** five questions with a rationale each.

**Acceptance criteria:** questions cover intended use, claim, predicate, EU class, and post-market; no question is answerable by the class alone.

**Time:** 25 minutes.

| Exercise | Outcome assessed | Evidence produced |
|---|---|---|
| 1 | Class distinctions | Five-row table |
| 2 | Pathway mapping | Pathway and evidence |
| 3 | Leakage | Calculation |
| 4 | Software class | Class and evidence |
| 5 | Misclassification | Diagnosis |
| 6 | Evidence matrix | Matrix |
| 7 | Effort estimate | Hours and cost |
| 8 | Claim evaluation | Five questions |

### 12.1 How the exercises map to a submission

| Exercise | Submission section | Evidence |
|---|---|---|
| 1 | Device description | Class table |
| 2 | Regulatory strategy | Pathway memo |
| 3 | Electrical safety | Leakage calculation |
| 4 | Software documentation | Class and lifecycle |
| 5 | Risk analysis | Misclassification hazard |
| 6 | Evidence matrix | Matrix |
| 7 | Project plan | Hours and cost |
| 8 | Claim review | Questions |

The exercises are not academic. Each produces an artifact that a submission or a technical file would contain, in a simplified form.

## 13. Self-check answers

**Answer 1 (Exercise 1).** FDA class II (510(k)), EU class IIa, type BF, software class B, IP22. Reasons: the device monitors a physiological signal, does not deliver therapy, contacts the skin but not the heart, and is used at home with a clinician review.

**Answer 2 (Exercise 2).** With no predicate, the FDA pathway is De Novo, unless a predicate exists under a different code. The evidence includes bench accuracy, software documentation, biocompatibility, usability, and a clinical evaluation. The EU class is IIa, with a notified body and a clinical evaluation.

**Answer 3 (Exercise 3).** I = 250 V / 30 MΩ = 8.33 µA, below the 10 µA CF limit. A 20 MΩ isolation would give 12.5 µA, above the limit. The margin is 1.67 µA, or 17%.

**Answer 4 (Exercise 4).** The function is class C because a wrong dose can cause severe hypoglycemia or death. The evidence includes detailed design, unit verification, integration testing, regression testing, and a hazard analysis.

**Answer 5 (Exercise 5).** The device makes a diagnostic claim without the class II/IIa evidence. The classification error is the omission of the diagnostic claim and the clinical evidence. The consequence is a non-compliant device, a possible recall, and a safety risk if the diagnosis is wrong.

**Answer 6 (Exercise 6).** The matrix includes FDA II (510(k)), EU IIb, type CF or BF, software class C, IP22, CISPR B, ISO 10993, usability, clinical performance, and PMS/PMCF. Each row has an owner and a standard.

**Answer 7 (Exercise 7).** The estimate: electrical safety 120 h, EMC 60 h, software class C 220 h, biocompatibility 80 h, usability 80 h, clinical 400 h, regulatory 120 h. Total 1,080 h; at $150/h, $162,000. The assumptions are the device's complexity, the clinical study size, and the laboratory quotes.

**Answer 8 (Exercise 8).** Questions: (1) What is the intended use and the clinical claim? (2) What is the predicate, and is the equivalence supported? (3) What clinical evidence supports the claim? (4) What is the EU class, and does it require a notified body? (5) What post-market obligations apply? The class alone does not answer any of them.

**Rubric for the design exercises.** A strong design states each class, its basis, its evidence, and its cost. A weak design names one class and assumes the others.

**Numeric answer key.**

| Item | Expected value | Tolerance |
|---|---:|---|
| Exercise 3: leakage | 8.33 µA | ±0.05 µA |
| Exercise 3: margin | 1.67 µA | ±0.05 µA |
| Exercise 7: total hours | 1,080 h | ±50 h |
| Exercise 7: cost | $162,000 | ±$5,000 |
| Section 6: evidence total | 558 h | ±1 h |
| Section 6: cost | $83,700 | ±$100 |
| Section 6: therapy ratio | 2.51 | ±0.02 |

A correct method with a rounding difference receives full credit. A wrong unit receives partial credit only.

**Time budget for the exercises.**

| Exercise | Reading | Calculation | Writing | Total |
|---|---:|---:|---:|---:|
| 1 | 5 | 5 | 5 | 15 |
| 2 | 5 | 10 | 10 | 25 |
| 3 | 5 | 10 | 5 | 20 |
| 4 | 5 | 10 | 5 | 20 |
| 5 | 5 | 15 | 10 | 30 |
| 6 | 5 | 20 | 10 | 35 |
| 7 | 5 | 15 | 10 | 30 |
| 8 | 5 | 10 | 10 | 25 |

### 13.1 Rubric dimensions

| Dimension | Novice | Competent | Expert |
|---|---|---|---|
| Axes | Names one class | Names three | Names all six with bases |
| Evidence | Lists tests | Maps class to evidence | Quantifies hours and cost |
| Applied part | Names BF | Explains leakage | Designs isolation |
| Software | Names class | Maps lifecycle | Quantifies verification |
| Regulatory | Names pathway | Explains predicate | Plans pre-submission |

The rubric rewards the separation of axes and the quantification of evidence, not the memorisation of class numbers.

## 14. Interview questions

**1. What is the difference between FDA class and EU class?**
A strong answer explains the different systems, the different pathways, and the fact that they are not equivalent. A weak answer treats them as the same.

**2. What determines the applied-part type?**
A strong answer names the patient contact, the cardiac path, and the leakage limit. A weak answer says "the amount of contact".

**3. What does type CF mean?**
A strong answer explains the floating applied part and the 10 µA normal-condition limit. A weak answer says "cardiac".

**4. How is the software class determined?**
A strong answer ties it to the harm from a software failure and the IEC 62304 evidence. A weak answer says it is the device class.

**5. What is a predicate?**
A strong answer defines it as a legally marketed device for substantial equivalence. A weak answer says it is a competitor.

**6. When is a PMA required?**
A strong answer names class III and the high-risk, no-predicate case. A weak answer says "for implants".

**7. What is the difference between IP rating and sterilization?**
A strong answer explains that IP is ingress protection and sterilization is microbial kill; they are independent. A weak answer confuses them.

**8. Why does the home environment change the EMC class?**
A strong answer explains CISPR B versus A and the home healthcare immunity levels. A weak answer says "home devices are quieter".

**9. What evidence does a class IIb device need?**
A strong answer names the technical file, clinical evaluation, notified body, and PMS. A weak answer says "a 510(k)".

**10. What would make you reject a classification?**
A strong answer names an unsupported intended use, a missing rule, a wrong applied part, or an overclaim. A weak answer rejects on the class number.

**11. How do you classify a device with two functions?**
A strong answer classifies each function and applies the most stringent class; it also considers combination-product rules. A weak answer picks the higher number without a basis.

**12. Why is the intended use the first step?**
A strong answer explains that the intended use determines the class, the claim, and the evidence. A weak answer starts with the technology.

## 15. Key terms

- **510(k)** — premarket notification.
- **Applied part** — part of the device in contact with the patient.
- **CISPR 11** — emissions standard.
- **Class I/II/III** — FDA risk classes.
- **Class I/IIa/IIb/III** — EU risk classes.
- **Clinical evaluation** — EU evidence of safety and performance.
- **De Novo** — new device classification pathway.
- **Emissions class** — CISPR A or B.
- **FDA class** — U.S. risk class.
- **IFU** — instructions for use.
- **Indication** — clinical condition addressed.
- **Ingress protection** — IP rating.
- **Intended use** — what the device is for.
- **MDR** — Medical Device Regulation.
- **MOPP** — means of patient protection.
- **MOOP** — means of operator protection.
- **Notified body** — EU conformity assessment body.
- **PMA** — premarket approval.
- **PMS** — post-market surveillance.
- **PMCF** — post-market clinical follow-up.
- **Predicate** — legally marketed device for equivalence.
- **Product code** — FDA device type code.
- **PSUR** — periodic safety update report.
- **Software class** — IEC 62304 A/B/C.
- **Sterilization** — microbial kill.
- **Substantial equivalence** — similarity to a predicate.
- **Type B/BF/CF** — applied-part classifications.
- **UDI** — unique device identification.
- **Vigilance** — EU adverse-event reporting.
- **Combination product** — device plus drug or biologic.
- **Contact duration** — limited, prolonged, or long-term.
- **Exemption** — class I devices not requiring a 510(k).
- **Invasive device** — device that enters the body.
- **Primary mode of action** — the main effect of a combination product.
- **Product code** — FDA three-letter code.
- **Reusable device** — device intended for reprocessing.
- **Single-use device** — device intended for one use.
- **TGA** — Australian regulator.
- **Health Canada** — Canadian regulator.
- **CDRH** — FDA Center for Devices and Radiological Health.
- **CER** — clinical evaluation report.
- **GSPR** — general safety and performance requirements.
- **IFU** — instructions for use.
- **MDR Annex VIII** — classification rules.
- **PMDA** — Japan regulator.
- **Pre-submission** — FDA meeting to discuss a planned submission.
- **Technical file** — EU documentation of conformity.
- **TGA** — Australian regulator.
- **UDI** — unique device identification.

## 16. Sources

All URLs were accessed on 25 October 2026.

**Regulation**

1. U.S. Food and Drug Administration, 21 CFR Part 860, *Medical Device Classification Procedures*. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-860
2. U.S. Food and Drug Administration, 21 CFR Part 807, *Establishment Registration and Device Listing; Premarket Notification*. https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-807
3. U.S. Food and Drug Administration, device classification database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfPCD/classification.cfm
4. Regulation (EU) 2017/745, *Medical Device Regulation*, Annex VIII. https://eur-lex.europa.eu/eli/reg/2017/745/oj
5. Regulation (EU) 2017/746, *In Vitro Diagnostic Regulation*. https://eur-lex.europa.eu/eli/reg/2017/746/oj

**Standards**

6. IEC 60601-1:2005+AMD1:2012+AMD2:2020, clause 8.7.3. https://webstore.iec.ch/en/publication/26169
7. IEC 62304:2006+AMD1:2015. https://webstore.iec.ch/en/publication/22794
8. ISO 14971:2019. https://www.iso.org/standard/72704.html
9. IEC 60529:1989+AMD1:1999+AMD2:2013, *Degrees of protection provided by enclosures*. https://webstore.iec.ch/en/publication/2458
10. CISPR 11:2015+AMD1:2016+AMD2:2019, *Industrial, scientific and medical equipment — Radio-frequency disturbance characteristics*. https://webstore.iec.ch/en/publication/28565
11. IEC 60601-1-2:2014+AMD1:2020. https://webstore.iec.ch/en/publication/67554
12. ISO 10993-1:2025. https://www.iso.org/standard/87505.html

**Background**

13. J. G. Webster (ed.), *Medical Instrumentation: Application and Design*, 4th ed., Wiley, 2009.
14. U.S. Food and Drug Administration, *Overview of Device Regulation*. https://www.fda.gov/medical-devices/device-advice-comprehensive-regulatory-assistance/overview-device-regulation
15. European Commission, *Medical Devices — Regulatory Framework*. https://health.ec.europa.eu/medical-devices-sector_en
16. U.S. Food and Drug Administration, recall database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfres/res.cfm
17. Health Canada, *Medical Devices Regulations*. https://www.canada.ca/en/health-canada/services/drugs-health-products/medical-devices.html
18. Therapeutic Goods Administration, *Medical devices*. https://www.tga.gov.au/
19. Pharmaceuticals and Medical Devices Agency (Japan). https://www.pmda.go.jp/english/
20. U.S. Food and Drug Administration, *Combination Products*. https://www.fda.gov/combination-products
21. ISO 13485:2016, quality management. https://www.iso.org/standard/59752.html
22. IEC 62366-1:2015+AMD1:2020, usability. https://webstore.iec.ch/en/publication/26570
23. U.S. Food and Drug Administration, *Pre-Submission Program*. https://www.fda.gov/medical-devices/premarket-submissions-selecting-and-preparing-correct-submission/pre-submission-program
24. European Commission, *Notified Bodies*. https://health.ec.europa.eu/medical-devices-topics-interest/notified-bodies_en
25. IMDRF, *Medical Device Cybersecurity* guidance. https://www.imdrf.org/
26. ISO 13485:2016, quality management systems.
27. U.S. Food and Drug Administration, *Device Advice*. https://www.fda.gov/medical-devices/device-advice-comprehensive-regulatory-assistance
28. European Commission, *MDR Guidance*. https://health.ec.europa.eu/medical-devices-sector/new-regulations/guidance-mdcg-endorsed-documents-and-other-guidance_en
29. ISO 14971:2019, risk management.
30. IEC 60601-1:2005, clause 8.7.3.
31. IEC 62304:2006+AMD1:2015, software lifecycle.
32. CISPR 11:2015, emissions.
33. IEC 60529:1989, ingress protection.
34. ISO 10993-1:2025, biocompatibility.
35. FDA, 21 CFR Part 860.

## 17. Links

- Module map: [`_index.md`](_index.md)
- Previous topic: [`08_firmware-and-rtos.md`](08_firmware-and-rtos.md)
- Next topic: [`10_robotics-kinematics.md`](10_robotics-kinematics.md)
- Related: [`13_standards-and-emc.md`](13_standards-and-emc.md)
- Module survey: [`../../course/14A_medical_hardware_sensors_chips_robotics.md`](../../course/14A_medical_hardware_sensors_chips_robotics.md)
- Lab on device teardown: see Lab 14A.6 in the survey file.
- Lab on usability: see Lab 14A.11 in the survey file.
- Lab index: [`../../labs/`](../../labs/)
