# Supplied Clinical Library

> Module 22 — Sources and Further Reading
> Topic 02 of 09
> Estimated study time: 2.5 hours
> Prerequisites: 01_how-to-read-this-list
> Version 2026.3

## 1. Why this topic exists

The course supplies a small clinical library: physical examination, electrocardiography, radiology, cardiovascular and respiratory physiology, neuroanatomy, neurology, and a book on neurodiversity. The books exist for a specific reason. A data scientist working in healthcare has to be able to read a clinical note, understand what a measurement means, and judge whether a model's inputs and outputs are clinically plausible. No amount of statistics substitutes for knowing what a troponin is, why an ECG lead is where it is, or what a radiologist means by an opacity.

The library is also a source category with its own failure modes, and they are different from the ones in journal articles:

- A team cites a textbook statement on heart failure treatment that predates the trials that changed practice, because the book on the shelf is the seventh edition and the eighth has been available for a year.
- A reader treats a textbook as a guideline. The book explains the disease; the guideline states what to do this year, and the two can disagree.
- A citation gives the title and author but not the edition, so nobody can tell which text was read.
- A chapter is assumed to be written by the named author, when it is written by a guest contributor with a different view.
- An online version is updated between printings, so two readers with the same ISBN see different sentences.
- An erratum changes a dose, a threshold, or a classification, and the erratum is published on a companion website that nobody visits.

The people affected are specific. The analyst cannot tell whether a feature is clinically meaningful. The clinical lead cannot reconcile a model with current practice. The regulatory writer cannot cite a textbook as evidence and does not know why. The student cannot tell which of two conflicting statements is current.

This topic teaches three things: how to identify a book precisely, how to judge whether its content is current for a particular claim, and how to use a book for what it is good for without asking it to do something it cannot. The library itself is listed in the survey file; this topic is the method for using it.

The library divides by the kind of question each book can answer, which is the first thing to know before opening one:

| Book | Primary use | Secondary use | Do not use for |
|---|---|---|---|
| Bates' Guide | Examination and history vocabulary | Documentation conventions | Treatment recommendations |
| Goldberger's Electrocardiography | ECG interpretation | Clinical correlation of waveforms | Signal-processing methods |
| The Only EKG Book | ECG fundamentals | Rapid pattern recall | Quantitative ECG analysis |
| Learning Radiology | Modality overview | Common imaging findings | Reporting standards |
| Pathophysiology of Heart Disease | Cardiovascular mechanism | Disease vocabulary | Current therapy |
| West's Respiratory Physiology | Gas exchange and acid-base | Ventilator physiology | Clinical management |
| Neuroanatomy through Clinical Cases | Localization | Neurological examination | Current neurological therapy |
| Bradley and Daroff's Neurology | Disease reference | Diagnostic vocabulary | Quick lookup under time pressure |
| The Power of Neurodiversity | Language and design | Ethical framing | Clinical guidance |
| Schwinger review | Heart failure mechanism | A citable peer-reviewed review | Treatment recommendations |

The distinction between a primary use and a secondary use is not about quality. It is about the kind of claim each use supports, and the next section makes that distinction operational.

A second distinction is between a book that teaches and a book that reminds. Bates and West teach; Thaler and the review books remind. A reminding book is useful when the vocabulary is already understood and dangerous when it is not, because compression hides the qualifications that make a clinical statement true.

> **Note:** A textbook is tertiary material. It summarises and interprets primary evidence, usually several years before the edition's publication date. It is excellent for vocabulary, mechanisms, and pattern recognition. It is weak as evidence for a therapeutic or regulatory claim, and it is never a substitute for the current guideline on a question of treatment.

## 2. Learning outcomes

By the end of this topic you can:

1. **Identify** a book unambiguously by title, edition, year, and ISBN, and distinguish an edition from a printing or a revised reprint. Assessed by Exercise 1 and self-check item 1.
2. **Determine** whether a cited edition is current, and whether any gap matters for the claim class being supported. Assessed by Exercise 2 and the worked example.
3. **Classify** the role a book can play for a claim: definition, mechanism, pattern recognition, procedure, or evidence. Assessed by Exercise 3.
4. **Detect** the specific failure modes of textbook citation, including edition drift, chapter authorship, errata, and online-versus-print divergence. Assessed by Exercise 4.
5. **Compare** a textbook statement with the current guideline and explain any divergence in terms of publication lag rather than error. Assessed by Exercise 5 and the worked example.
6. **Apply** the edition-checking script to a library and interpret its verdicts, including the case where an edition is current but the field has moved. Assessed by Exercise 6 and section 7.
7. **Write** a citation that a reader can resolve to the exact text used, including the chapter and the access route for an electronic copy. Assessed by Exercise 7.
8. **Decide** when a clinical question requires a guideline, a trial, or a specialist rather than a book, and say so in writing. Assessed by Exercise 8.

The outcomes are ordered so that each one depends on the previous. A learner who cannot identify the edition cannot decide currency, and a learner who cannot classify the claim cannot decide the role.

Each outcome is observable in a document a reviewer can mark, which is the standard used throughout this module.

## 3. The core mental model

A clinical book is an object with a publication history and a set of claims with different shelf lives. The object is identified by edition; the claims are identified by class. The two are separate questions, and conflating them produces both false confidence and unnecessary scepticism.

```text
                    CLINICAL QUESTION
        vocabulary | mechanism | pattern | therapy | dose
                              |
                              v
                    IDENTIFY THE OBJECT
        title | edition | year | ISBN | format (print/online)
                              |
                              v
                    ESTABLISH CURRENCY
        latest edition | printing | errata | online updates
                              |
                              v
                    CLASSIFY THE CLAIM
        definition | mechanism | pattern | therapy | dose
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
   STABLE SUBJECT        MOVING SUBJECT        THERAPY OR DOSE
   anatomy, physical     pathophysiology,      always check the
   exam technique        imaging patterns      current guideline
        |                     |                     |
        +---------------------+---------------------+
                              |
                              v
                    ROLE DECISION
        definition source | explanation | pattern reference | not evidence
                              |
                              v
                    CITATION WITH CHAPTER AND EDITION
```

Read the diagram downwards. The question type determines how much currency matters. A question about the origin of the second heart sound is stable; a question about which diuretic to use in acute decompensation is not. The same book can be authoritative for the first and outdated for the second, and the edition number does not distinguish them.

**Books are tertiary by construction.** A textbook chapter is a synthesis written months or years before publication, drawing on primary literature that is itself years old. The publication year on the copyright page is the end of a process, not the date of the evidence. A 2024 edition can contain a sentence that was true in 2019 and has since been overtaken.

**Edition, printing, and reprint are different objects.** An edition is a substantive revision with a new ISBN. A printing is a re-run of the same edition, sometimes with minor corrections. A revised reprint can carry corrections without a new edition number, and an international edition can differ in pagination, figures, and occasionally content. A citation that names only the author and title cannot distinguish these.

**Chapter authorship matters.** Many textbooks are edited volumes. The named editors wrote the preface, not every chapter. A chapter's author is the person accountable for that chapter's claims, and a chapter can reflect an individual view that differs from the book's overall stance. When precision matters, cite the chapter author.

**The guideline is a different genre.** A guideline is a recommendation system with a stated evidence review and a date. A textbook is an explanation. When they disagree, the guideline is usually more current for treatment and the textbook is usually better for mechanism. Neither is a primary study.

**Online access changes the object.** A book accessed through a publisher platform can be updated between printings, and the platform can show a different version to two readers. The citation then needs the platform, the access date, and the section, because the ISBN no longer identifies the text.

| Claim class | Currency sensitivity | Best book role | What to check instead |
|---|---|---|---|
| Vocabulary and definitions | Low to moderate | Definition source | Current consensus statement if the term is contested |
| Mechanism and physiology | Low | Explanation | Recent review if the mechanism is an active research area |
| Pattern recognition | Moderate | Atlas or teaching text | Local practice and latest imaging criteria |
| Examination technique | Low | Procedural reference | Local protocol for technique and documentation |
| Therapy | High | Background only | Current guideline and pivotal trials |
| Dose | Very high | Never | Formulary, label, and pharmacist |

The model also implies a division of labour. The clinician on the team judges whether a mechanism sentence is plausible and whether a therapy sentence is current. The data scientist judges whether the claim is stated precisely enough to be checked and whether the citation resolves. Neither can do the other's part, and most bad citations come from one role answering the other's question.

A book can be current and still wrong for the model. If the chapter describes a definition that the local laboratory no longer uses, the feature derived from it will not match the data. That is an applicability failure, not a currency failure, and it is caught by step 20 rather than step 6.

## 4. Definitions and vocabulary

### 4.1 The book as a physical and bibliographic object

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Edition | A substantive revision of a book, usually with a new ISBN and new pagination. | Confused with a printing. | Ordinal number |
| Printing | A production run of the same edition, possibly with minor corrections. | Treated as a new edition. | Printing number |
| Revised reprint | A reprint with corrections but the same edition number. | Assumed identical to the first printing. | Reprint statement |
| Impression | A synonym for printing in some publishers' usage. | Used interchangeably with edition. | Publisher statement |
| Copyright year | The year the copyright was registered, which may precede release. | Treated as the date of the evidence. | Year |
| Publication date | The date the book was released, which may differ by country and format. | Confused with the copyright year. | Date |
| ISBN-13 | The 13-digit identifier for a specific edition and format. | Assumed unique to the title rather than the format. | 13-digit code |
| eISBN | The identifier for an electronic edition, which can differ from print. | Used for a print citation. | 13-digit code |
| DOI | A persistent identifier, increasingly assigned to book chapters. | Assumed present for every chapter. | Prefix and suffix |
| Chapter author | The person accountable for a specific chapter in an edited volume. | Assumed to be the book's editor. | Person and affiliation |
| Editor | The person who commissions and assembles chapters. | Treated as author of every chapter. | Person and affiliation |
| International edition | A separate printing for a territory, sometimes differing in content. | Assumed page-identical to the domestic edition. | ISBN and territory |
| Companion website | A publisher site with figures, questions, and errata. | Ignored after purchase. | URL and access date |
| Errata | Published corrections to a specific printing. | Assumed absent because the book is printed. | Erratum list |
| Out of print | No longer produced by the publisher. | Assumed invalid; it may still be the best text. | Publisher status |
| Monograph | A single-author work on one subject. | Confused with an edited textbook. | Publication type |
| Successor edition | An announced or published replacement edition. | Assumed to be available immediately. | Edition number and date |
| Cutoff statement | The preface's statement of the literature reviewed. | Ignored in favour of the copyright year. | Date |
| Chapter byline | The named author of a chapter. | Replaced by the editors' names. | Person and affiliation |

### 4.2 Roles a clinical book can play

| Role | What the book supplies | What it cannot supply | Example use in this course |
|---|---|---|---|
| Definition source | A working definition and its vocabulary | The current consensus definition if contested | Terms in a data dictionary |
| Mechanism explanation | A causal account of physiology or disease | Evidence that an intervention changes outcomes | Feature plausibility |
| Pattern reference | Typical appearances and their differential | Prevalence in a specific population | Imaging and ECG features |
| Procedural reference | Technique and documentation conventions | Local policy | Examination variables |
| Examination preparation | Structured recall for board examinations | Research-grade evidence | Vocabulary building |
| Historical record | What was believed at a date | Current practice | Understanding legacy labels |
| Evidence | Nothing beyond a pointer to primary studies | Comparative effectiveness | Never cite as primary evidence |

### 4.3 Currency and version terms

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Edition cycle | The interval between editions, typically three to six years. | Assumed uniform across subjects. | Years |
| Publication lag | The gap between the evidence and the printed statement. | Ignored when reading a recent copyright year. | Years |
| Content volatility | How quickly the subject's claims change. | Assumed to be a property of the book. | Slow, moderate, fast |
| Superseded edition | An edition replaced by a newer one. | Cited because it is the copy on the shelf. | Status |
| Within tolerance | An edition gap small enough for the claim class. | Applied to therapy claims. | Policy threshold |
| Erratum | A published correction to a specific statement. | Assumed to be incorporated in every copy. | Notice |
| Platform version | The version served by an online book platform. | Assumed to match the print ISBN. | Platform and date |
| Access route | How the text was obtained: print, institutional platform, personal purchase. | Omitted from the citation. | Route |
| Shelf life | The period for which a book's statement remains adequate. | Assumed to equal the edition cycle. | Years |

### 4.4 Terms that describe a claim's dependence on the book

| Term | Definition | Common misuse | Unit or coding |
|---|---|---|---|
| Textbook claim | A statement whose authority rests on the book's synthesis. | Treated as a primary finding. | Claim type |
| Guideline divergence | A difference between the book and the current recommendation. | Read as an error in one of them. | Comparison |
| Definition drift | A change in a term's meaning across editions or guidelines. | Ignored because the term is unchanged. | Version comparison |
| Chapter-level view | A position that reflects the chapter author rather than the editors. | Attributed to the book as a whole. | Attribution |
| Local practice | What a specific institution does, which may differ from both book and guideline. | Assumed to follow the guideline. | Site-specific |
| Examination fact | A stable fact about technique or anatomy. | Confused with a management recommendation. | Claim class |

| Question type | Where the answer lives | What to check before quoting |
|---|---|---|
| What is this term? | Glossary or introductory chapter | Whether the definition is contested |
| Why does this happen? | Mechanism chapter | Whether the chapter cites primary literature |
| What does it look like? | Atlas or figure-heavy chapter | Whether the figure matches your modality |
| What should we do? | Guideline, not the book | The guideline date and recommendation strength |
| How much should we give? | Formulary and label, not the book | Local protocol and renal dosing |
| How is it measured? | Methods chapter or standard | Units, assay, and reference range |
| How common is it? | Epidemiology source, not the book | Denominator and population |
| Is this still true? | Guideline and recent trials | The edition's literature cutoff |

## 5. The mechanics, step by step

The procedure has 18 steps in five phases. It is longer than the procedure for journal articles because a book has more identifying detail and a slower, less visible revision process. The first four steps identify the object; steps 5 to 8 establish currency; steps 9 to 11 establish applicability; steps 12 to 15 establish role and provenance; steps 16 to 18 produce the verdict, the citation, and the review trigger.

The cost of the whole procedure for one chapter is about thirty minutes. The cost of skipping it is a citation that cannot be resolved, or a clinical statement that a reviewer can date to a superseded edition in one search.

| Book type | Identifying feature | Extra check required |
|---|---|---|
| Authored textbook | One or two named authors for the whole book | Author's own research programme |
| Edited textbook | Editors plus a contributor list | Chapter author and chapter references |
| Handbook | Short, protocol-like chapters | Local protocol divergence |
| Atlas | Figure-led with brief text | Figure edition and modality |
| Review or recall book | Questions, summaries, mnemonics | Not a source for mechanism |
| Monograph | Single author, one subject, research-led | Author's programme and funding |
| Translated text | Translator named, original edition year | Translation lag and omissions |
| Local adaptation | Territory-specific chapters | Replaced therapy and pathway content |

### Phase A — Identify the object

**Step 1. Record the title, edition, year, and ISBN.**
Input: the physical book, the PDF, or the platform page.
Output: a bibliographic line with all four elements, plus the format.
Rationale: the edition is the object. Without it, two readers can hold different texts with the same title and argue about sentences that only one of them can see.
Branch: if the ISBN is unavailable, use the publisher's product page and record the product identifier.
Edge case: the same title can exist in a professional and a student edition, with different content and pagination.

**Step 2. Distinguish edition, printing, and revised reprint.**
Input: the copyright page and the publisher's product page.
Output: one of four object states: first printing of an edition, later printing with corrections, revised reprint, or international edition.
Rationale: corrections can be incorporated without a new edition number, so two copies of "the same edition" can differ.
Branch: if the copy is a revised reprint, record the reprint date as well as the edition year.
Edge case: an international edition may have different pagination and, occasionally, different content. Page-based citations do not transfer.

| Object state | How to recognise it | Citation consequence |
|---|---|---|
| First printing of an edition | Edition number, single year, no reprint line | Cite edition and year |
| Later printing with corrections | Reprint line with a later year | Cite edition and reprint year |
| Revised reprint | "Revised reprint" on the copyright page | Cite edition and reprint date |
| International edition | Territory statement and a different ISBN | Cite the edition and note the territory |
| Online platform version | Platform name and access date | Cite platform, section, and access date |

**Step 3. Identify the chapter author.**
Input: the chapter opening page and the contributor list.
Output: the person accountable for the chapter, with affiliation.
Rationale: edited books are collections. Attributing a chapter's view to the editors is a common and sometimes material error.
Branch: if the chapter is unsigned, attribute it to the editors and say so.
Edge case: a chapter can be updated by a different author in a later edition, so the same chapter number can carry a different view.

**Step 4. Classify the claim the book will support.**
Input: the sentence being written.
Output: one of five classes: definition, mechanism, pattern, therapy, or dose.
Rationale: the class determines how much currency matters and whether a book can carry the claim at all.
Branch: if the claim is therapy or dose, the book is background only, and the procedure continues only to find a pointer to the guideline or label.
Edge case: a sentence can contain a mechanism claim and a therapy claim; split it before proceeding.

### Phase B — Establish currency

**Step 5. Find the latest edition.**
Input: the publisher's product page, a library catalogue, or a bookseller listing.
Output: the latest edition number and year, with the URL and access date.
Rationale: currency is measured against the latest edition, not against the copy in hand.
Branch: if a new edition is announced but not published, record the announcement date and treat the current edition as current.
Edge case: a book can be out of print with no successor. Out of print is not the same as superseded.

**Step 6. Compare the cited edition with the latest.**
Input: the two edition numbers and years.
Output: an edition gap and an age, plus a status from the table below.
Rationale: the gap is the primary currency signal, and the age catches the case where the edition is current but the field has moved since publication.
Branch: a gap of one edition is material for therapy, often immaterial for anatomy.
Edge case: a book with a very long edition cycle can be current by edition and stale by age.

When a new edition exists, four responses are defensible, and one is not:

- Replace the citation if the claim is therapy, dose, or anything the new edition is likely to have revised.
- Keep the citation and add the edition if the claim is mechanism or examination technique and the edition gap is within tolerance.
- Cite both editions if the point is that the field changed between them.
- Cite the new edition from a library copy after checking the passage, even if you do not own it.
- The indefensible response is to keep citing the old edition without saying which edition it is, which is what most documents do.

The lag between evidence and citation is longer than most readers assume, and it accumulates at every step:

```text
primary evidence published        2016 ------ 2019
guideline or consensus            2019 ------ 2020
chapter drafted                   2020 ------ 2021
book printed and released         2022
reader cites the sentence         2026
```

The reader in 2026 is citing a synthesis that closed around 2021, which is why the copyright year is a poor proxy for currency. The lag is not a defect of the book; it is the cost of synthesis. The defect is treating the synthesis as if it were the primary evidence.

| Signal that a chapter may be outdated | What it looks like | What to check |
|---|---|---|
| No references after a certain year | Reference list ends several years before publication | The edition's stated cutoff |
| Guideline not mentioned | A major trial or guideline absent from the text | Publication date versus trial date |
| Definitive language about therapy | "The treatment of choice is" without a date | Current guideline |
| Old classification | A staging or grading system that has been revised | Current staging manual |
| Units or ranges that look unfamiliar | Conventional units, old assay names | Local laboratory and current reference |
| A drug named by brand only | Brand without generic, or a withdrawn brand | Formulary and label |

| Status | Definition | Typical action |
|---|---|---|
| Current | Cited edition is the latest | Use within its role |
| Superseded, within tolerance | One edition behind, stable subject | Use, note the edition |
| Superseded, beyond tolerance | One edition behind, moving subject | Check the change or replace |
| Materially superseded | Two or more editions behind | Replace for any claim that matters |
| Out of print, no successor | No current edition exists | Use with a date caveat; look for a guideline |
| Withdrawn | Publisher has withdrawn the title | Do not cite |

**Step 7. Check the errata and companion site.**
Input: the publisher's companion page and any errata list.
Output: a statement of whether corrections affect the passage used.
Rationale: errata are published outside the book and are not incorporated into copies already sold.
Branch: if an erratum changes a number, dose, or classification, cite the corrected value and note the erratum.
Edge case: companion sites disappear when editions retire, so archive the erratum page at the time of use.

| Erratum type | Effect on the citation | Action |
|---|---|---|
| Typographical | No change in meaning | Cite the corrected text if available |
| Numerical | Changes a value, dose, or threshold | Cite the correction explicitly |
| Figure replacement | Changes a visual reference | Cite the edition containing the corrected figure |
| Reference correction | Changes the evidence pointer | Follow the corrected reference |
| Withdrawal of a chapter | Removes the claim's support | Do not cite the chapter |
| No errata published | Status unknown, not status clean | Check the publisher page and record the access date |

**Step 8. Check the online platform version.**
Input: the platform page and the access date.
Output: whether the online text differs from the print edition, and the section identifier.
Rationale: platform content can be updated between printings, so the ISBN no longer identifies the text.
Branch: if the platform shows a different sentence, cite the platform version and date as the object.
Edge case: institutional access can lapse, leaving a citation that nobody in the organisation can check.

### Phase C — Establish applicability

**Step 9. Check scope: does the chapter actually cover the claim?**
Input: the chapter heading, the index, and the relevant pages.
Output: a pass, partial, or fail on scope.
Rationale: a book can be authoritative in its subject and silent on the specific question. Silence is not support.
Branch: if the chapter mentions the topic in one sentence, the book is a pointer, not a source.
Edge case: a sentence in an introductory chapter can be a simplification that the specialist chapter contradicts.

| Scope outcome | What the chapter does | Verdict for this claim |
|---|---|---|
| Direct coverage | A section with the same population and question | Usable within its role |
| Partial coverage | A paragraph or a related chapter | Pointer only |
| Background mention | One sentence in an introduction | Not support |
| Contradiction | The chapter states the opposite | Investigate before use |
| Silence | No mention | Not support; find another source |
| Out of scope | The book excludes the topic by design | Not applicable |

**Step 10. Check population and setting.**
Input: the chapter's stated scope and examples.
Output: a statement of the population and setting the chapter addresses, and any mismatch.
Rationale: clinical textbooks often describe tertiary-care practice in high-income settings, which may not match the deployment context.
Branch: if the chapter addresses adults and the claim concerns children, the book is not applicable.
Edge case: a book can describe a condition in a way that assumes resources, such as imaging availability, that the deployment setting lacks.

**Step 11. Compare with the current guideline.**
Input: the relevant society guideline and its date.
Output: agreement, divergence, or no corresponding recommendation.
Rationale: the guideline is the current statement of what to do; the book explains why. Divergence is information, not necessarily error.
Branch: if the book and guideline disagree on treatment, the guideline governs and the book supplies the mechanism.
Edge case: a guideline can be silent on a question the book discusses, which is common for mechanism and examination detail.

### Phase D — Role and provenance

**Step 12. Decide the role the book can play.**
Input: the claim class and the results of steps 9 to 11.
Output: one of four roles: definition source, explanation, pattern reference, or not evidence.
Rationale: the role decision prevents a book from being cited for something it cannot support.
Branch: for therapy and dose, the role is never "evidence"; the book can only point to the guideline or label.
Edge case: a book can be the best available definition source when no consensus statement exists, provided the definition is labelled as the book's.

| Claim class | Permitted book role | Citation form | Escalation if the claim matters |
|---|---|---|---|
| Definition | Definition source | Chapter, edition, page | Consensus statement or standard |
| Mechanism | Explanation | Chapter, edition, page | Recent review or primary study |
| Pattern | Pattern reference | Chapter, figure, edition | Local imaging or ECG criteria |
| Therapy | Background only | Chapter, edition, page, plus guideline | Current guideline and trials |
| Dose | None | Not cited as a source | Formulary and product label |

**Step 13. Assess the chapter's provenance.**
Input: the chapter author, affiliations, and any disclosed relationships.
Output: a statement of the author's standing for this topic and any conflict.
Rationale: a chapter reflects its author's synthesis, and the book's editorial review is not the same as peer review.
Branch: if the chapter author is also an author of a guideline, the two sources are not independent.
Edge case: a chapter that summarises a single research programme can present that programme's view as the field's.

**Step 14. Check for a primary or guideline source that would be better.**
Input: the claim and a search of the guideline and trial literature.
Output: a better source, or a documented finding that the book is the best available.
Rationale: the book is often a convenient source, not the best one. The check makes the choice deliberate.
Branch: if a current guideline covers the claim, cite the guideline and use the book only for explanation.
Edge case: for historical statements, the book may be the best source precisely because it records what was believed.

**Step 15. Record the access route.**
Input: how the text was obtained.
Output: print, institutional platform, personal e-book, or interlibrary copy, with the date.
Rationale: a reader must be able to reach the same text, and the route determines whether they can.
Branch: if the route is a personal licence that others cannot use, say so and provide the page reference.
Edge case: interlibrary copies can differ in printing, which matters for corrected passages.

### Phase E — Verdict, citation, and review

**Step 16. Assign a role-specific verdict.**
Input: the outputs of steps 5 to 15.
Output: use, use with caveat, background only, or replace.
Rationale: the verdict is about the claim, not the book. The same book can receive different verdicts for different sentences.
Branch: if the verdict is "replace", name the replacement source rather than leaving a gap.
Edge case: a superseded edition can remain the best source for a historical claim, with the date stated.

**Step 17. Write the citation with chapter and edition.**
Input: the bibliographic record.
Output: author of the chapter, chapter title, book title, editors, edition, publisher, year, pages, ISBN, and access route.
Rationale: the citation has to resolve to the exact text, and the edition is the part most often omitted.
Branch: if the source is an online platform, cite the platform and access date instead of pages.
Edge case: a translated edition needs the translator and the original edition year.

**Step 18. Set the review trigger.**
Input: the edition cycle and the claim class.
Output: a review date or an event trigger, such as a new edition or a guideline update.
Rationale: books go stale silently, and the next edition is the natural trigger.
Branch: if the claim is therapy, the trigger is the guideline update cycle, not the book.
Edge case: a new edition can appear without announcement on a bookseller page before the publisher's own site updates.

The procedure produces a citation that a reader can resolve and a verdict that a reviewer can audit. It does not produce a judgement about whether the book is good; that judgement belongs to the field, and the procedure assumes the supplied library has already passed it.

### The same book, three claims

The role decision is easier to see with one book and three sentences. Take Lilly, *Pathophysiology of Heart Disease*, seventh edition, 2021, with the eighth edition published in 2025.

**Claim A: "Heart failure with reduced ejection fraction involves activation of the sympathetic nervous system and the renin-angiotensin-aldosterone system."**
Class: mechanism. The seventh edition is one edition behind, but the mechanism is stable and the chapter is a synthesis of decades of work. Verdict: use, with the edition cited. The eighth edition would be preferable, and the difference is unlikely to change the sentence.

**Claim B: "The first-line pharmacological treatment for heart failure with reduced ejection fraction is an angiotensin-converting enzyme inhibitor or an angiotensin receptor-neprilysin inhibitor, with a beta-blocker and a mineralocorticoid receptor antagonist."**
Class: therapy. The seventh edition is one edition behind in a fast-moving subject, and the guideline has changed since 2021. Verdict: background only. Cite the current guideline for the recommendation and use the book for the mechanism.

**Claim C: "Heart failure with preserved ejection fraction has no treatment that improves outcomes."**
Class: therapy, and the statement was defensible when the seventh edition was written. Two large trials published after it, one in 2021 and one in 2022, changed the answer for a class of drugs. Verdict: replace, not merely caveat. This is the worked example in section 6.

The three sentences come from the same chapter of the same book. A rule that treated the book as current or outdated would be wrong for at least two of them.

The mechanism sentence is the easiest case and the one most often over-corrected. Teams that have just learned about edition drift sometimes replace every textbook citation with a guideline, and then lose the explanation that made the feature meaningful in the first place.

The therapy sentence is the case most often under-corrected, because the textbook is authoritative on the disease and the reader extends that authority to the treatment.

- Mechanism claims age slowly and are usually safe within one edition.
- Therapy claims age quickly and are never safe without a guideline check.
- Definition claims depend on whether the term is contested; if it is, the book is a historical source.

### What goes in the record

- The exact object: edition, year, ISBN, and format.
- The chapter author and page range.
- The claim class and the permitted role.
- The comparison edition and any errata consulted.
- The access route and date, and the review trigger.

| Phase | Steps | Typical effort | Artefact |
|---|---|---|---|
| A Identify | 1 to 4 | 5 minutes | Bibliographic record |
| B Currency | 5 to 8 | 10 minutes | Edition status and errata check |
| C Applicability | 9 to 11 | 8 minutes | Scope and guideline comparison |
| D Role and provenance | 12 to 15 | 5 minutes | Role decision and access route |
| E Verdict and citation | 16 to 18 | 4 minutes | Verdict and citation |
| F Cross-checks | 19 to 24 | 8 minutes | Final record |

The total is about forty minutes for a chapter that matters, and under ten for a chapter used only for vocabulary. The phase that is most often skipped is C, and it is the one that catches the mismatch between what the book covers and what the sentence claims.

### Phase F — Cross-checks before the citation leaves the desk

**Step 19. Check the index and cross-references for a qualifying statement.**
Input: the index entry and any "see also" references.
Output: a statement of whether the passage is qualified elsewhere in the book.
Rationale: textbooks often state a rule in one chapter and its exceptions in another, and the index is the map between them.
Branch: if a qualification exists, cite both passages or summarise the pair.
Edge case: a qualification can appear only in a figure legend, which the index does not capture.

**Step 20. Check units and reference ranges against the local laboratory.**
Input: the book's stated units and ranges, and the local laboratory's report.
Output: a conversion or a note that the local range differs.
Rationale: reference ranges vary by assay and population, and a book's range is illustrative rather than universal.
Branch: if the model uses a threshold from the book, replace it with the local range or state the source of the threshold.
Edge case: the book may use conventional units while the local system reports SI units.

**Step 21. Check the preface for the literature cutoff.**
Input: the preface or "how to use this book" section.
Output: the date to which the literature was reviewed, if stated.
Rationale: the cutoff is a better currency signal than the copyright year, and it is often two years earlier.
Branch: if no cutoff is stated, assume at least eighteen months before publication.
Edge case: a new edition can reuse chapters from the previous edition without updating their references.

**Step 22. Check whether the title is a derivative or review book.**
Input: the title, subtitle, and publisher description.
Output: whether the book is an original text, a board-review derivative, or a compilation.
Rationale: review books compress and sometimes simplify to the point of error, and they are designed for examination recall rather than practice.
Branch: if the book is a review derivative, use it for vocabulary only and prefer the parent text for mechanism.
Edge case: a derivative can be the only accessible version of a text in some settings.

**Step 23. Check the figure or table version.**
Input: the figure number and the edition.
Output: whether the figure has changed between editions.
Rationale: figures are often revised without a change to the surrounding text, and a model trained on an old figure inherits its errors.
Branch: if the figure changed, cite the edition that matches the figure used.
Edge case: platform versions can update figures while the text remains the same.

**Step 24. Check the local protocol and formulary.**
Input: the institutional protocol, formulary, and order set.
Output: agreement or divergence with the book.
Rationale: local practice is what the model will encounter, and it can differ from both book and guideline.
Branch: if local practice diverges, the model's labels and features reflect the local practice, not the book.
Edge case: a protocol can be outdated locally, which is itself a finding worth recording.

### Common errors at each phase

- **Identify.** Citing the editor for a chapter; omitting the edition; treating a revised reprint as a new edition.
- **Currency.** Using the copyright year as the evidence date; missing a new edition because the shelf copy looks recent; ignoring errata.
- **Applicability.** Assuming a chapter covers a question it mentions once; applying tertiary-centre examples to a community setting.
- **Role.** Citing a textbook as evidence for a therapy; using a review book for mechanism; treating a definition as consensus.
- **Cross-check.** Skipping the index; ignoring unit differences; not checking the local protocol.

Each error is cheap to detect and expensive to defend. The cross-check phase exists because the first four phases produce a defensible citation, and the fifth produces a citation that survives contact with the clinical team.
## 6. Worked example

A clinical lead is writing a slide for a heart failure model. The sentence reads: "Heart failure with preserved ejection fraction has no treatment that has been shown to improve outcomes." The citation is to the seventh edition of Lilly, *Pathophysiology of Heart Disease*, published in 2021. The slide will be shown to a cardiology group, and the lead wants to know whether the citation is adequate.

### Framing the claim

The sentence has three parts:

1. A population: heart failure with preserved ejection fraction (HFpEF), usually defined as an ejection fraction of 40 percent or more in the trials discussed below.
2. An outcome: a clinical outcome such as cardiovascular death or heart failure hospitalisation, not a surrogate such as exercise capacity.
3. A universal negative: no treatment has been shown to improve outcomes.

The universal negative is the difficult part. It requires either a systematic review showing consistent null results or a statement from a guideline that reflects one. A single textbook sentence cannot establish it, because the absence of a treatment in one synthesis is not evidence that none exists.

| Sub-claim | Type | Source that could support it |
|---|---|---|
| HFpEF is defined by ejection fraction at or above 40 percent | Definition | Current guideline or consensus statement |
| No treatment improves outcomes | Universal negative | Systematic review or guideline with an evidence review |
| This statement is current in 2026 | Currency | Guideline date and trial dates |

### Candidate sources

| # | Candidate | Type | Role |
|---|---|---|---|
| C1 | Lilly, 7th edition, 2021, heart failure chapter | Textbook | The current citation |
| C2 | Lilly, 8th edition, 2025, heart failure chapter | Textbook | The successor edition |
| C3 | 2022 AHA/ACC/HFSA heart failure guideline | Guideline | The recommendation source |
| C4 | EMPEROR-Preserved trial, 2021 | Randomised trial | Primary evidence for empagliflozin |
| C5 | DELIVER trial, 2022 | Randomised trial | Primary evidence for dapagliflozin |
| C6 | A 2019 narrative review of HFpEF treatment | Review | Context and prior state |

The citation chain is short: the slide cites C1, which cites a guideline from before 2021 and trials that predate the two large trials. C3, C4, and C5 are absent from the slide's reference list.

### Scoring the candidates

**C1, Lilly seventh edition.**
- Authority: high for mechanism, moderate for therapy. A textbook is a synthesis, not an evidence review.
- Currency: superseded by the eighth edition in 2025, four years after the cited edition.
- Applicability: the chapter covers HFpEF, so scope passes; the therapy claim is exactly the class where a textbook is weakest.
- Accessibility: passes, print copy with page numbers.
- Provenance: chapter authorship and references are visible, but the synthesis cannot establish a universal negative.
- Verdict: replace for this claim. It is still usable for the mechanism sentence on the same slide.

**C2, Lilly eighth edition.**
- Authority: same genre limitation.
- Currency: current as an edition.
- Applicability: same scope.
- Accessibility: available in the library.
- Provenance: same limitation, though the references are newer.
- Verdict: better, but still background only for a universal negative. The edition being current does not change the genre problem.

**C3, 2022 guideline.**
- Authority: high. A society guideline with a stated evidence review and graded recommendations.
- Currency: current at the review date, though a later update should be checked.
- Applicability: the guideline addresses the population and the outcome.
- Accessibility: open access.
- Provenance: methods, evidence review, and conflicts are disclosed.
- Verdict: primary support for the recommendation, with the date stated.

**C4 and C5, the two trials.**
- Authority: high, as randomised evidence.
- Currency: current; both are within five years of the review date.
- Applicability: adults with HFpEF, mostly outpatient, with the trial's inclusion criteria.
- Accessibility: abstracts and, for one, open full text.
- Provenance: methods, sponsors, and analyses are reported.
- Verdict: primary support for the specific drug and population, not for the class as a whole.

**C6, the 2019 review.**
- Authority: moderate; a narrative review without a systematic search.
- Currency: predates both trials, so it cannot speak to the current question.
- Applicability: broad.
- Accessibility: paywalled.
- Provenance: methods are not systematic.
- Verdict: background only, and only for the state of the field in 2019.

### The arithmetic

Both trials reported a statistically significant reduction in a composite outcome. The absolute differences are modest, and they are what a clinical audience will ask about.

| Trial | Patients | Event rate, treatment | Event rate, control | Absolute difference | Number needed to treat |
|---|---:|---:|---:|---:|---:|
| EMPEROR-Preserved | 5,988 | 13.8% | 17.1% | 3.3 percentage points | 31 |
| DELIVER | 6,263 | 16.4% | 19.5% | 3.1 percentage points | 33 |

The arithmetic is 17.1 minus 13.8 equals 3.3 percentage points, and 100 divided by 3.3 equals 30.3, which rounds up to 31 patients treated for one additional patient to avoid the composite event. For DELIVER, 19.5 minus 16.4 equals 3.1, and 100 divided by 3.1 equals 32.3, which rounds up to 33.

Two cautions belong with the table. The number needed to treat is over the trial's follow-up period, not per year, and the composite includes heart failure hospitalisation, which the audience may weight differently from death. A relative reduction of about 20 percent and an absolute reduction of about 3 percentage points are the same result described two ways, and the slide should say which one it is using.

### Reading the effect size

- The relative reduction is about 20 percent in both trials, which sounds large.
- The absolute reduction is about 3 percentage points, which is what a clinic sees.
- The number needed to treat is about 31 to 33 over the trial period, which is a reasonable basis for a treatment decision and a poor basis for a cost-saving claim.
- The composite is dominated by heart failure hospitalisation, so the effect on mortality alone is smaller and less certain.
- The confidence intervals exclude 1, so the direction is not in doubt; the size is.

A slide that reports only the relative reduction is not lying, but it is choosing the presentation that makes the effect look largest. A clinical audience will usually ask for the absolute difference within one question, so it is better to show both.

### Textbook statement versus guideline statement

| Element | Textbook (7th edition, 2021) | Guideline (2022) |
|---|---|---|
| Genre | Explanatory synthesis | Recommendation with graded evidence |
| Currency | Predates two pivotal trials | Includes both trials |
| Claim form | "There is no treatment that..." | "We recommend SGLT2 inhibitors..." |
| Action implied | None | Initiate or continue therapy |
| Appropriate use | Explain the prior state of the field | State current practice |

The two statements are not contradictory in the way they first appear. The textbook statement was accurate for the evidence available when it was written, and the guideline statement reflects evidence published afterwards. The error is temporal, not factual.

### How the guideline was found

The guideline was located by searching the society's own site rather than a general search engine, which matters because the top results for a heart failure query are often summaries of the guideline rather than the guideline itself.

- The society's guideline page gave the current version and its date.
- The recommendation table gave the class of recommendation and the level of evidence.
- The reference list pointed to the two trials.
- The guideline's methods section described the evidence review and the conflicts of interest process.

The class and level matter for the slide. A class 1 recommendation with level A evidence supports a strong sentence; a class 2b recommendation with level C evidence supports a much weaker one. Quoting the recommendation without its class is a common way to overstate a guideline.

| Guideline element | What it tells the reader | Common omission |
|---|---|---|
| Class of recommendation | How strongly the society endorses the action | Quoted without the class |
| Level of evidence | The quality and consistency of the evidence | Confused with the class |
| Date of the guideline | The evidence cutoff | Omitted from the slide |
| Scope and population | Whether the recommendation applies | Applied outside the population |
| Conflicts of interest | Whether the panel was independent | Not checked |

### What the original claim got right

The seventh edition was correct that, before 2021, no therapy had shown a clear outcome benefit in HFpEF. It was also correct that the condition is heterogeneous and that earlier trials had failed for reasons that were not fully understood. Those parts of the chapter remain useful.

The claim went wrong in the tense. "There is no treatment" is a present-tense universal, and a textbook written in 2021 can only support a past-tense statement about the evidence up to its cutoff. Changing "is" to "was" would have made the sentence defensible and would have signalled that it needed rechecking.

The episode is a useful example of a broader pattern: the facts in an outdated source are often right, and the grammar of the sentence is what makes them wrong.

### Review questions for the clinical lead

- What is the date of the evidence in the sentence, as distinct from the date of the citation?
- Does the claim have a universal quantifier ("no", "all", "never") that one new trial could falsify?
- Is the population in the model the same as the population in the trials?
- Does the audience need the mechanism, the recommendation, or both?
- Which sentence would a cardiologist challenge first, and what is the answer?
- What is the review trigger for this slide if it is reused next year?

### What the slide should say

| Claim | Source | Sentence |
|---|---|---|
| Historical state | Textbook, 7th edition | "Before 2021, no therapy had shown a clear outcome benefit in HFpEF." |
| Current recommendation | 2022 guideline | "Current guidelines recommend SGLT2 inhibitors for HFpEF." |
| Effect size | EMPEROR-Preserved and DELIVER | "Two trials showed about a 3 percentage point absolute reduction in a composite endpoint." |
| Mechanism | Textbook, 7th edition | "The pathophysiology involves..." |

A slide with those four rows is more work than one sentence and is far more likely to survive a cardiology review.

### Verdict

The sentence "no treatment has been shown to improve outcomes" is not supportable at the review date. Two randomised trials published after the cited edition showed benefit on a composite outcome, and the guideline recommends the drug class. The textbook was not wrong when it was written; it is being asked to carry a claim that later evidence has overtaken.

| Sub-claim | Verdict | Supportable sentence |
|---|---|---|
| No treatment improves outcomes | Reject | "Until 2021, no therapy had shown a clear outcome benefit in HFpEF." |
| A specific drug class reduces composite events | Primary support | "SGLT2 inhibitors reduced the composite of cardiovascular death or heart failure hospitalisation by about 3 percentage points in two large trials." |
| The textbook statement is current | Reject | "The seventh edition, published in 2021, predates both trials." |

The replacement sentence is narrower and more useful. It tells the audience what changed, when, and by how much, which is the information a clinical group actually needs.

### Two diagnostics that would change the verdict

**Diagnostic 1: a later trial with a neutral primary endpoint in the same population.** If a well-powered trial of the same class in HFpEF had shown no benefit on a comparable composite, the claim would become contested rather than false, and the verdict would shift to "cite with limitation" with both results presented. A single neutral trial does not erase two positive ones, but it changes the strength of the recommendation.

**Diagnostic 2: a population definition that excludes the trial populations.** HFpEF is heterogeneous, and the trials enrolled particular phenotypes. If the model's population is, for example, patients with transthyretin cardiac amyloidosis or a specific comorbidity profile excluded from both trials, the original sentence could remain true for that subgroup. The diagnostic is a comparison of the trial's inclusion and exclusion criteria with the model's cohort definition, not a general appeal to heterogeneity.

A third diagnostic, weaker, is a guideline update that withdraws the recommendation. It would change the clinical action but not the historical fact that the trials reported benefit.

### What the team should record

The ledger row records the claim, the four sources, the edition comparison, the trial dates, and the verdict. The review trigger is the next guideline update and any new trial in HFpEF. The original sentence is retained in the slide's history with a note, because a later author will otherwise restore it from an older deck.

The case also produces a reusable rule for the library: a textbook may explain why a treatment works, and a guideline must state whether it is recommended. The two roles are different, and a slide that needs both should cite both.

## 7. Code, query, or calculation

The script encodes the edition decision as a rule rather than a judgement. It compares the cited edition with the latest known edition, then applies a tolerance that depends on the subject's volatility and the claim class. Therapy claims are always replaced when the edition is behind, because the therapy tolerance is zero by construction.

```python
from dataclasses import dataclass

REVIEW_YEAR = 2026

# Latest edition known at the review date. Update this table, not the prose.
LATEST = {
    "Bates": (13, 2023, "moderate"), "Goldberger": (10, 2023, "moderate"),
    "Thaler": (10, 2022, "moderate"), "Herring": (5, 2024, "fast"),
    "Lilly": (8, 2025, "fast"), "West": (11, 2021, "slow"),
    "Blumenfeld": (3, 2021, "slow"), "Bradley": (8, 2022, "moderate"),
    "Armstrong": (1, 2010, "slow"),
}

# Maximum acceptable edition gap by volatility and claim class.
TOLERANCE = {
    ("slow", "definition"): 2, ("slow", "mechanism"): 1,
    ("moderate", "definition"): 1, ("moderate", "mechanism"): 1,
    ("fast", "definition"): 0, ("fast", "mechanism"): 1,
}


@dataclass
class Citation:
    book: str
    edition: int
    year: int
    claim_class: str      # definition | mechanism | therapy


def assess(c):
    latest_ed, latest_year, volatility = LATEST[c.book]
    gap = latest_ed - c.edition
    age = REVIEW_YEAR - c.year
    if gap == 0:
        status, verdict = "current", "use"
    elif c.claim_class == "therapy":
        status, verdict = "superseded", "replace"
    elif gap <= TOLERANCE[(volatility, c.claim_class)]:
        status, verdict = "superseded, within tolerance", "use"
    elif gap == TOLERANCE[(volatility, c.claim_class)] + 1:
        status, verdict = "superseded, beyond tolerance", "caveat"
    else:
        status, verdict = "materially superseded", "replace"
    note = "field may have moved" if age > 8 else ""
    return dict(book=c.book, cited=f"{c.edition}e/{c.year}",
                latest=f"{latest_ed}e/{latest_year}", gap=gap, age=age,
                status=status, verdict=verdict, note=note)


CITATIONS = [
    Citation("Lilly", 7, 2021, "therapy"),
    Citation("Lilly", 8, 2025, "therapy"),
    Citation("Thaler", 9, 2018, "definition"),
    Citation("West", 11, 2021, "mechanism"),
    Citation("Herring", 5, 2024, "therapy"),
    Citation("Blumenfeld", 3, 2021, "mechanism"),
    Citation("Armstrong", 1, 2010, "definition"),
]

rows = [assess(c) for c in CITATIONS]
print(f"review year {REVIEW_YEAR}   citations {len(rows)}")
print("-" * 104)
print(f"{'book':12s} {'cited':11s} {'latest':11s} {'gap':>3s} {'age':>3s} "
      f"{'status':29s} {'verdict':8s} note")
print("-" * 104)
for r in rows:
    print(f"{r['book']:12s} {r['cited']:11s} {r['latest']:11s} {r['gap']:3d} {r['age']:3d} "
          f"{r['status']:29s} {r['verdict']:8s} {r['note']}")
print("-" * 104)
print("replace:", sum(r["verdict"] == "replace" for r in rows),
      " caveat:", sum(r["verdict"] == "caveat" for r in rows),
      " use:", sum(r["verdict"] == "use" for r in rows))
```

Expected output:

```text
review year 2026   citations 7
--------------------------------------------------------------------------------------------------------
book         cited       latest      gap age status                        verdict  note
--------------------------------------------------------------------------------------------------------
Lilly        7e/2021     8e/2025       1   5 superseded                    replace  
Lilly        8e/2025     8e/2025       0   1 current                       use      
Thaler       9e/2018     10e/2022      1   8 superseded, within tolerance  use      
West         11e/2021    11e/2021      0   5 current                       use      
Herring      5e/2024     5e/2024       0   2 current                       use      
Blumenfeld   3e/2021     3e/2021       0   5 current                       use      
Armstrong    1e/2010     1e/2010       0  16 current                       use      field may have moved
--------------------------------------------------------------------------------------------------------
replace: 1  caveat: 0  use: 6
```

Four features of the output deserve comment. Thaler's ninth edition is two editions behind and eight years old, but the claim class is definition and the subject is stable, so it passes with the edition cited. Lilly's seventh edition is only one edition behind and fails, because the claim class is therapy. Armstrong is current by edition and sixteen years old, which is why the note column exists: an edition can be current and the field can still have moved. And the script cannot see the trials that changed the HFpEF answer, which is why the tolerance table is a starting point for judgement rather than a substitute for it.

To adapt the script, replace `LATEST` with the books you actually hold, and adjust `TOLERANCE` to the organisation's risk appetite. Keep the therapy rule at zero. A therapy recommendation from a superseded edition is the one case where the cost of being wrong is borne by a patient rather than by a reviewer.

The script deliberately does not score authority or provenance. Those are not reducible to a table, and a numeric score would create false confidence. What it does is make the edition decision consistent across reviewers, which is the part that is mechanical.

A useful extension is to add a `last_checked` field per book and flag any entry not verified in the last twelve months. The registry is only as good as its review date, and a stale registry produces confident verdicts about editions that have since changed.

## 8. Data, standards, and regulatory context

Books sit inside a bibliographic standards system and outside the evidence system that regulates medical products. Both facts matter.

### 8.1 Bibliographic standards

| Standard or scheme | Issuer | What it constrains | Binding status |
|---|---|---|---|
| ISO 2108 | International Organization for Standardization | ISBN structure, assignment, and check digit | Technical standard |
| ISO 3297 | ISO | ISSN for serials, not books | Technical standard |
| DOI Handbook | International DOI Foundation | DOI syntax and resolution | Technical standard |
| ONIX | Book industry standards body | Metadata exchange between publishers and retailers | Industry convention |
| MARC 21 | Library of Congress | Library catalogue records | Library convention |
| ICMJE recommendations | ICMJE | Authorship and disclosure in biomedical publications | Voluntary, journal-adopted |
| Accessibility standards for e-books | W3C and publishers | Structure and navigation of electronic texts | Varies by contract and jurisdiction |

The practical consequence is that an ISBN identifies a format-specific edition, and a DOI identifies a chapter or a book at the object level. A citation that gives a DOI can be resolved without the ISBN; a citation that gives only a title cannot be resolved at all.

### 8.2 Copyright, licensing, and access

A textbook is a copyrighted work. Purchasing a copy does not transfer the right to reproduce figures, tables, or long passages in a slide deck or a publication. Institutional licences vary: some permit classroom use and limited distribution, others permit nothing beyond reading. Four rules cover most situations:

- Quote short passages with attribution; do not reproduce whole figures without permission or a licence that allows it.
- Check whether the publisher's platform licence covers internal training materials, which is often a separate permission.
- Treat a personal e-book licence as non-transferable; colleagues may not be able to open the citation.
- Record the access route so that a reader knows which licence governs.

### 8.3 What a textbook is not

A textbook is not a regulatory document, not a validated training curriculum, and not evidence for a device submission. In a regulatory context, the applicable sources are the regulation, the recognised standards, the guidance, and the clinical evidence in the submission. A textbook can inform the author's understanding, and it cannot be cited as the basis for a claim about safety or performance.

The distinction is sometimes blurred by the phrase "clinical evidence," which in a regulatory setting means evidence about the device, not general medical knowledge. A textbook chapter about the disease does not become device evidence because it is authoritative.

### 8.4 The library as a curriculum artefact

The supplied library is a reading list, not a validated curriculum. Its selection reflects the course's view of what a data scientist needs, and it is not a systematic review of medical education. The books are widely used standard texts, which is a reason to trust their vocabulary and a reason not to treat any single sentence as settled.

### 8.5 How a book is produced, and why provenance is visible

A textbook chapter passes through more hands than a journal article, and each hand leaves a trace that a reader can check.

| Role | What they control | What a reader can check |
|---|---|---|
| Commissioning editor | Whether the book exists and its scope | The preface and table of contents |
| Volume editor | Chapter selection and overall stance | The contributor list |
| Chapter author | The content and references of the chapter | The chapter byline |
| Development editor | Structure, consistency, and length | The preface acknowledgements |
| External reviewers | Critique before revision | Sometimes named in the preface |
| Production editor | Typesetting, figures, and errata | The copyright page and errata list |
| Sales and marketing | Title, subtitle, and cover claims | The publisher's product page |

The provenance test for a book is therefore different from the test for a paper. There is usually no data availability statement and no code, and there is often no transparent peer-review record. What is available is the chapter byline, the reference list, the edition history, and the errata. Those are enough to judge whether the chapter is a synthesis of primary literature or a summary of the author's own programme.

### 8.6 Translations, local editions, and adaptations

A translated textbook is a new object. The translation can lag the original by years, can omit chapters, and can adapt examples to a local health system. A citation to the translation should name the translator and the year, and the original edition year should be recorded separately, because the underlying evidence is as old as the original.

Local adaptations are common in some markets: a publisher licenses a text and replaces chapters on health systems, guidelines, or formulary. The result can be excellent and can differ materially from the original on exactly the questions a data team cares about, such as coding practice and care pathways.

Three rules follow:

- Cite the edition you read, not the edition you assume it derives from.
- Record the original edition year alongside the translation year.
- Check whether a local adaptation has replaced the therapy chapters, which is where divergence is most likely.

## 9. Industry practice and case evidence

The documented facts below were checked on 26 October 2026. They illustrate how quickly a clinical library drifts, even when every individual book is reputable.

| Book | Edition verified | Year | Note |
|---|---|---|---|
| Bickley et al., *Bates' Guide to Physical Examination and History Taking* | 13th | 2023 | Revised reprint; four-author credit |
| Goldberger et al., *Goldberger's Clinical Electrocardiography* | 10th | 2023 | Earlier editions remain in circulation |
| Thaler, *The Only EKG Book You'll Ever Need* | 10th | 2022 | Ninth edition still widely held |
| Herring, *Learning Radiology* | 5th | 2024 copyright | Sixth edition scheduled for 2027 |
| Lilly, *Pathophysiology of Heart Disease* | 8th | 2025 | Seventh edition is the one many teams own |
| West and Luks, *West's Respiratory Physiology* | 11th | 2021 | Released late 2020, 2021 copyright |
| Blumenfeld, *Neuroanatomy through Clinical Cases* | 3rd | 2021 | Some catalogues list 2022 |
| Jankovic et al., *Bradley and Daroff's Neurology in Clinical Practice* | 8th | 2022 | Two-volume set |
| Armstrong, *The Power of Neurodiversity* | 1st | 2010 | Current by edition, old by field |

Three observations follow from the table, and they are inferences rather than documented facts.

First, the survey file for this module lists these books by author and title without editions. That is normal for a reading list and is a defect for a citation. A reader who wants to check a sentence cannot tell which text was intended, and a team that standardises on the list can end up with three editions in use.

Second, the edition cycle is not uniform. Physiology texts run on cycles of five years or more; a cardiology text in a fast-moving therapeutic area can be revised while the previous edition is still being sold as current. The volatility column in section 7 is a crude encoding of this, and it is the part most worth adjusting locally.

Third, the presence of a newer edition does not automatically invalidate the older one. The seventh edition of Lilly remains an excellent explanation of heart failure pathophysiology. It is the therapy sentence that fails, and it fails because the field moved, not because the book was careless.

**What practitioners do (inference).** Clinical teams usually keep a small number of current reference texts and rely on guidelines for anything therapeutic. Data teams often inherit whichever PDF was available when the project started, which is why edition checks matter more in data work than in clinical work. The asymmetry is not deliberate; it is a consequence of how PDFs circulate.

**What goes wrong in diligence (inference).** A reviewer asks for the source of a clinical statement, receives a textbook page, and cannot tell whether the page is current. The team then spends a meeting defending the book rather than the sentence. The edition check in section 7 prevents the meeting.

### Edition drift by subject

| Subject | Typical cycle | What changes between editions | Risk of using the older edition |
|---|---|---|---|
| Physical examination | 5 to 8 years | Technique, documentation, some evidence links | Low for technique, moderate for evidence |
| Electrocardiography | 4 to 6 years | Criteria, device formats, some guidelines | Low for pattern recognition |
| Radiology | 4 to 6 years | Modalities, AI chapters, reporting standards | Moderate; imaging criteria change |
| Cardiovascular physiology | 4 to 6 years | Mechanisms, trials, therapy | High for therapy, low for mechanism |
| Respiratory physiology | 6 to 10 years | Physiology stable, clinical correlations move | Low |
| Neuroanatomy | 6 to 10 years | Anatomy stable, imaging correlations move | Low |
| Neurology | 4 to 6 years | Classification, trials, therapy | High for therapy |
| Social and ethical topics | No cycle | Language and framing evolve | Moderate to high for framing |

The table is a heuristic, not a measurement. Its purpose is to make the tolerance decision explicit rather than habitual.

### The case of a delayed edition

*Learning Radiology* illustrates a subtlety. The fifth edition carries a 2024 copyright and was published in 2023. A sixth edition has been listed with a 2027 publication date. A team that checks only the copyright year sees a current book; a team that checks the publisher's catalogue sees an announced successor and can decide whether to wait. The correct action in most cases is to use the fifth edition and record that a successor is expected, because a book that is current today does not become wrong because a new edition is announced.

### What the library does not contain

The supplied library is clinical, not methodological. Knowing what is missing prevents a book from being stretched to cover a question it was never designed for.

| Missing category | Why it matters for data work | Where to look instead |
|---|---|---|
| Epidemiology and study design | Prevalence, bias, and causal reasoning | Module 05 and the methods sources |
| Biostatistics | Estimation, uncertainty, and testing | Module 04 and a methods text |
| Health economics | Cost, budget impact, and reimbursement | Module 17 and payer sources |
| Pharmacology and dosing | Drug features and interactions | Formulary and product label |
| Coding and terminologies | How clinical concepts become data | LOINC, SNOMED CT, ICD, and Module 02 |
| Regulatory science | What evidence a device needs | Module 14 and the regulator's guidance |
| Implementation science | Whether a workflow change sticks | Module 18 and implementation literature |
| Data ethics and law | Consent, privacy, and secondary use | Module 16 and the relevant regulation |

A team that uses a pathophysiology text to answer an epidemiology question will produce a sentence that sounds clinical and is methodologically unsupported. The gap is not a defect of the book; it is a mismatch of genre.

### Documented facts and inferences in this topic

| Statement | Status |
|---|---|
| Editions and years in the table above | Documented at publisher or catalogue pages on 26 October 2026 |
| The survey file lists titles without editions | Documented by inspection of the survey file |
| Edition cycles differ by subject | Documented by the edition histories; the specific intervals are approximate |
| Data teams inherit PDFs rather than current editions | Inference from practice, not measured |
| Diligence meetings defend the book rather than the sentence | Inference from observed review behaviour |

The inferences are labelled because they are the kind of claim that a reader should be able to challenge. The documented rows can be checked in a minute; the inferred rows are the ones this topic is trying to make visible.

One further inference is worth stating because it drives the module's advice. The cost of an edition check is minutes, and the cost of an outdated therapy statement in a clinical deck is a credibility loss that is hard to reverse. The asymmetry is large enough that the check should be routine rather than discretionary.

In practice, the check is easiest when the library is small. A team with three current texts and a guideline source per therapeutic area will outperform a team with thirty PDFs of unknown provenance, because the first team knows what it has.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Edition omitted | Citation has author, title, and year only | The object is under-specified | Reader cannot resolve the text | Add edition and ISBN |
| Old edition cited | Shelf copy is several years old | Convenience beats checking | Therapy statement is outdated | Check the publisher's current edition |
| Copyright year as evidence date | Citation uses the copyright year | Publication lag is invisible | Evidence is years older than it appears | Check the preface cutoff |
| Chapter attributed to editors | Citation names the editors only | Edited volume treated as authored | Wrong person held accountable | Cite the chapter author |
| Textbook as evidence | A therapy claim cites a chapter | Genre confusion | Claim has no comparative support | Cite the guideline and trials |
| Guideline divergence ignored | Book and guideline disagree | Different publication dates | Practice statement contradicts guidance | Compare and state the difference |
| Errata missed | A number differs from other sources | Corrections published off-book | Dose or threshold is wrong | Check the companion site |
| Platform version drift | Two readers see different text | Online content updated between printings | Citation does not resolve to the text | Cite platform and access date |
| International edition assumed identical | Different ISBN or territory statement | Pagination and content can differ | Page citation is wrong | Cite the edition actually used |
| Review book used for mechanism | Title includes "review" or "recall" | Compression for exams | Simplification treated as explanation | Prefer the parent text |
| Units taken from the book | Values look unfamiliar | Conventional versus SI units | Order-of-magnitude error | Convert and record the unit |
| Local protocol assumed | Model built on book thresholds | Local practice differs | Model does not match workflow | Check the local protocol |
| Definition drift | Term unchanged, meaning changed | Guideline revised the definition | Cohort mixes definitions | Check definition dates |
| Out-of-print treated as invalid | No current edition exists | Availability confused with validity | A usable source is discarded | Use with a date caveat |
| Translated edition cited as original | Translator not named | Translation lags and can omit content | Evidence date is earlier than it appears | Cite translator and original year |
| Figure reused without permission | Slide uses a scanned figure | Copyright not checked | Takedown or legal exposure | Check licence and seek permission |
| Chapter updated by a new author | Same chapter number, different view | Edition revision changed authorship | Attribution is wrong | Check the byline in the edition used |
| Reference list stale | Newest reference predates the edition | Chapter not updated in revision | Statement is older than it appears | Check the newest reference date |
| Online access lapsed | Colleagues cannot open the citation | Subscription ended | Citation cannot be verified | Provide a print locator or archive |

Three red flags are specific to a clinical library and worth naming separately.

- A PDF with no title page. The edition is often on the copyright page, which is the first thing a scanned PDF loses.
- A citation to a page number that does not match any edition the team holds. This usually means the page came from a different edition or a different format.
- A slide that cites the same book for a mechanism claim and a treatment claim. The first is appropriate and the second is not.

### A library audit in ten checks

- Pick the ten most-used books and record edition, year, and ISBN.
- Find the latest edition for each at the publisher or a library catalogue.
- Mark the therapy chapters in any superseded edition.
- Check the errata page for every book published in the last three years.
- Identify any title that is a review or recall book and restrict its use.
- Identify any translated or locally adapted title and record the original edition.
- Check whether the online platform version differs from the print copy.
- Confirm that every citation in active documents names an edition.
- Flag any book whose field has moved despite a current edition.
- Set a review date tied to the edition cycle, not to the calendar year.

The audit takes an afternoon for a small library and pays for itself the first time a reviewer asks which edition was used.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test it |
|---|---|---|---|
| The book is a reputable standard text | The library is preselected | A title is a derivative or a predatory publication | Check the publisher and reviews |
| The edition can be identified | Currency depends on it | PDF lacks a title page | Obtain the copyright page or a catalogue record |
| The chapter reflects the book's stance | Attribution assumes coherence | Chapter author dissents | Read the chapter, not the blurb |
| The field's volatility is known | Tolerance depends on it | A fast-moving area is misclassified | Ask a specialist; check guideline cadence |
| The guideline is more current than the book | Role hierarchy assumes it | The guideline is older than the edition | Compare dates |
| Local practice resembles the book | Applicability assumes it | Site-specific protocol differs | Read the local protocol |
| Units are stated | Safety depends on it | Units omitted or conventional | Check against the local laboratory |
| The reader can obtain the same text | Verification depends on it | Personal licence or platform-only | Record the access route |
| The erratum page is current | Corrections depend on it | Page retired with the edition | Archive the page at use |
| A newer edition exists for every book | Currency checking assumes a successor | Out of print with no successor | Treat as a dated source, not a false one |
| The citation's page numbers match the edition | Page-based citation assumes pagination | Different printing or format | Verify against the edition cited |
| The book's figures are reproducible | Figures inform features and slides | Copyright restricts reuse | Check the licence before reproducing |
| The chapter's references are current | The reference list signals the evidence base | References stop at the cutoff | Check the newest reference date |
| The book's units match the local laboratory | Numeric claims depend on units | Conventional versus SI | Convert and record |
| The figure is the one in the cited edition | Figures change between editions | Revised figure, same caption | Compare the figure number and edition |
| The publisher's platform shows the same text | Online access is assumed stable | Platform updates | Check the passage in the platform version |
| The book's population matches the model's | Applicability assumes a comparison | Different age range or setting | Read the chapter's stated scope |

The largest limit is genre. A textbook cannot supply comparative evidence, and no amount of edition checking changes that. The second largest is the lag: even a current edition is a synthesis of evidence that closed years earlier. The procedure manages both limits; it does not remove them.

A practical consequence is that the library should be paired with a guideline source for every therapeutic area the project touches. The library explains; the guideline decides. Teams that maintain both spend less time arguing about which book is current and more time on the actual question.

## 12. Exercises

**Exercise 1 — Identify the object (recall, 20 minutes).**
Take three books from the supplied library and record title, edition, year, ISBN, format, and chapter author for a chapter you have used.
Deliverable: a three-row bibliographic table.
Acceptance criteria: every row has an edition and an ISBN; at least one row distinguishes a printing or reprint from an edition; the chapter author is named, not the editor alone.
Time: 20 minutes.

**Exercise 2 — Edition audit (application, 35 minutes).**
For each book, find the latest edition from the publisher or a library catalogue and compare it with the copy you hold.
Deliverable: an edition-gap table with cited edition, latest edition, gap, and status.
Acceptance criteria: each latest edition is confirmed at a publisher or catalogue page with an access date; at least one book has a gap; the status uses the vocabulary in section 5.
Time: 35 minutes.

**Exercise 3 — Classify the role (application, 30 minutes).**
Take five sentences from your own work that cite a textbook. Classify each as definition, mechanism, pattern, therapy, or dose, and state whether the book can carry it.
Deliverable: a five-row table with claim, class, permitted role, and verdict.
Acceptance criteria: at least one sentence is reclassified as background only; no therapy claim is left citing a textbook as evidence; a replacement source is named where the role fails.
Time: 30 minutes.

**Exercise 4 — Failure-mode hunt (critique, 35 minutes).**
Find a document, slide deck, or report that cites a clinical textbook and identify every failure from section 10 that it contains.
Deliverable: a marked-up excerpt with at least four findings, each naming the failure and the fix.
Acceptance criteria: findings are sentence-specific; at least one concerns the edition and one concerns the genre; the fixes are concrete.
Time: 35 minutes.

**Exercise 5 — Textbook versus guideline (critique, 40 minutes).**
Choose a clinical statement in your work and compare the textbook passage with the current guideline recommendation.
Deliverable: a two-column comparison with dates, a statement of divergence or agreement, and a revised sentence.
Acceptance criteria: both sources are dated; the divergence is explained by publication timing or by a genuine disagreement; the revised sentence is supportable by the guideline.
Time: 40 minutes.

**Exercise 6 — Run the edition checker (calculation, 45 minutes).**
Adapt the script in section 7 to the books you actually use, add your own volatility judgements, and run it on ten citations.
Deliverable: the modified script, its output, and a note on every verdict you disagree with.
Acceptance criteria: the script runs with the standard library only; the output is reproducible; at least one verdict is overridden with a written reason.
Time: 45 minutes.

**Exercise 7 — Write a resolvable citation (application, 20 minutes).**
Rewrite three existing textbook citations so that a reader can resolve each to the exact text, including chapter, edition, and access route.
Deliverable: three citations in a consistent style, each with a resolvability check.
Acceptance criteria: each citation includes the edition; each includes a chapter or section; an electronic source includes the platform and access date.
Time: 20 minutes.

**Exercise 8 — Escalation decision (design, 30 minutes).**
For a clinical question in your current project, decide whether the answer requires a book, a guideline, a trial, or a specialist, and write the decision with its rationale.
Deliverable: a one-page decision memo with the question, the source class chosen, the reason, and what would change the choice.
Acceptance criteria: the memo distinguishes vocabulary from evidence; it names the specific guideline or specialist; it states the cost of getting the answer wrong.
Time: 30 minutes.

### Marking guide

| Exercise | Evidence of a pass | Common failure |
|---|---|---|
| 1 Identify | Edition and ISBN present | Omitting the edition |
| 2 Edition audit | Latest edition confirmed at source | Using a bookseller's "newest" sort |
| 3 Role | Therapy claims escalated | Defending the textbook |
| 4 Failure hunt | Four specific findings | General comments about quality |
| 5 Book versus guideline | Both dates shown | Quoting the guideline without its class |
| 6 Script | Runs and is overridden with reasons | Hard-coded verdicts |
| 7 Citation | Resolvable to the text | Missing chapter or access route |
| 8 Escalation | Names the source class and the cost | "Ask a clinician" |

Two notes on marking. A submission that finds fewer than four failure modes in Exercise 4 has usually read the document rather than the sources. A submission that cannot name a replacement source in Exercise 3 has not finished the exercise, because the point is to move the claim to a source that can carry it.

The exercises are designed to be done on real documents. A learner who uses a synthetic example will produce a correct answer and will not experience the moment when a familiar sentence turns out to be unsupported, which is the moment the topic is trying to teach.

## 13. Self-check answers

**1. What is the difference between an edition and a printing?**
An edition is a substantive revision with a new ISBN and usually new pagination. A printing is a production run of the same edition and may incorporate minor corrections without changing the edition number. A citation that names only the title cannot distinguish them.

**2. Why is the copyright year a poor proxy for the date of the evidence?**
The copyright year marks the end of a production process that began years earlier. Chapters are drafted, reviewed, and typeset before release, so the evidence cutoff is usually one to three years before the copyright year. The preface sometimes states the cutoff and is a better guide.

**3. A book is one edition behind. When is that acceptable?**
It is acceptable when the claim class is stable (definition, mechanism, or examination technique) and the subject's volatility is low or moderate. It is not acceptable for therapy or dose, where a single edition can span a practice-changing trial.

**4. Why can a textbook not support a universal negative?**
A universal negative ("no treatment works") requires a systematic search and appraisal. A textbook synthesis can report that no treatment was known at its cutoff, which is a historical statement, not a claim about all possible treatments at the review date.

**5. What does a revised reprint change?**
It incorporates corrections without a new edition number, so it can change a sentence, a figure, or a number while the edition and ISBN remain the same. Two readers with the same edition can therefore hold different text.

**6. Why does the procedure require the chapter author?**
Because edited textbooks are collections, and a chapter reflects its author's synthesis. Attributing a chapter's view to the editors can misidentify who is accountable and can hide a dissent from the book's overall stance.

**7. When is a textbook the best available source?**
When the claim is about established mechanism, vocabulary, or examination technique and no current consensus statement covers it. It is also the best source for a historical claim about what was believed at a date.

**8. What should a citation to an online book include that a print citation does not?**
The platform name, the section identifier, and the access date, because the platform can serve a version that differs from the print edition and can change over time.

**Rubric for Exercises 6 and 8.** A strong Exercise 6 submission has a working script, a documented volatility judgement, and at least one reasoned override of the output. A weak submission hard-codes verdicts or removes the therapy rule. A strong Exercise 8 memo names the source class, the specific guideline or specialist, and the cost of error. A weak memo says "consult a clinician" without naming what kind of clinician or what question.

**9. Why is a review book unsuitable for mechanism, even when it is current?**
A review book compresses content for examination recall, and compression removes the qualifications that make a mechanism statement accurate. The parent text has the space to explain exceptions and uncertainty. The review book is useful for checking that you remember the vocabulary, not for learning why the mechanism works.

**10. What does an erratum change about a citation?**
It changes the text without changing the edition, so the citation should point to the corrected value and note the erratum. If the erratum changes a dose or threshold, the citation should also state the date of the correction, because a reader with an uncorrected copy will see a different number.

**11. Why does the procedure ask for the access route?**
Because a reader has to be able to reach the same text. A citation to a personal e-book licence may be unreadable by colleagues, and an institutional platform may lose access when a subscription ends. The access route tells the reader what they need.

**12. When is an out-of-print book still the right source?**
When the claim is historical, when no successor edition exists, or when the book is the definitive treatment of a narrow topic. The date should be stated, and any therapy claim should be checked against a current guideline.

**13. What does a page-stable citation require?**
An edition, a printing or reprint where relevant, and a format whose pagination does not change. An online platform may not have stable pagination, so a section identifier is safer than a page number.

**14. Why is a locally adapted edition risky for a data team?**
Because it can replace the chapters on care pathways, coding, and formulary, which are exactly the parts a data team relies on, and the divergence from the original is rarely marked. The adaptation can be better for local practice and worse for comparability.

**15. What should a library owner do when a book is out of print with no successor?**
Keep it, label it with its date, restrict its use to stable content, and identify the guideline that supersedes it for therapy. Removing it loses a useful explanation; keeping it unlabelled invites misuse.

## 14. Interview questions

**1. How do you know a textbook statement is still current?**
Strong answer: checks the latest edition, the preface cutoff, and the current guideline for therapy claims, and distinguishes stable from moving content. Weak answer: checks the copyright year and assumes recency.

**2. Why would you cite a chapter author rather than the editors?**
Strong answer: edited books are collections, the chapter author is accountable for that chapter's claims, and the author can differ from the editors' view. Weak answer: says it is a formatting preference.

**3. When is a textbook not acceptable evidence?**
Strong answer: for therapy, dose, comparative effectiveness, and any regulatory claim about a device. Weak answer: says textbooks are never acceptable.

**4. A book and a guideline disagree. What do you do?**
Strong answer: compares dates, populations, and the evidence each relies on; for treatment, the current guideline governs; documents the divergence. Weak answer: picks the newer source without reading either.

**5. What is publication lag and why does it matter?**
Strong answer: the gap between evidence and printed synthesis, which accumulates through drafting, review, and production, and which makes a recent copyright year misleading. Weak answer: confuses it with the time since the edition was published.

**6. How would you audit a clinical library?**
Strong answer: builds a table of editions, finds the latest for each, classifies volatility, and flags therapy chapters in superseded editions. Weak answer: checks that the books exist.

**7. What is the risk of using a review book for mechanism?**
Strong answer: review books compress for examination recall and can simplify to the point of error; the parent text is preferable for explanation. Weak answer: assumes all books are equivalent.

**8. How do you cite an online textbook?**
Strong answer: platform, section, access date, and edition, with the ISBN where available. Weak answer: gives a URL with no section or date.

**9. What do you do when the only available copy is an old edition?**
Strong answer: uses it for stable content, states the edition and date, and finds a guideline for anything therapeutic. Weak answer: cites it as if it were current.

**10. How does this connect to model features?**
Strong answer: features derived from textbook definitions inherit their version, so a definition change can create a label or feature mismatch. Weak answer: treats clinical definitions as timeless.

**11. What is the difference between a handbook and a textbook?**
Strong answer: a handbook is organised for lookup and often protocol-like, while a textbook is organised for explanation; the handbook is closer to local policy and dates faster. Weak answer: treats them as interchangeable.

**12. Why might a locally adapted edition be better and riskier at the same time?**
Strong answer: better because it reflects local pathways and formulary; riskier because it diverges from the original on exactly the questions a data team asks, and the divergence is rarely flagged. Weak answer: assumes the adaptation is a translation.

**13. How would you decide whether to wait for a new edition?**
Strong answer: compares the publication date, the claim class, and the cost of waiting; for a therapy claim the guideline is the better source anyway, so waiting is rarely necessary. Weak answer: waits because newer sounds better.

**14. What makes a clinical citation resolvable?**
Strong answer: edition, chapter or section, page or platform location, and access route. Weak answer: author, title, and year.

**15. How do you handle a book that is current but whose field has moved?**
Strong answer: treats the edition as current and the content as dated, checks the specific claim against a guideline or recent review, and records the reason. Weak answer: assumes the edition number settles the question.

**16. What is the first thing you check when a colleague sends you a PDF of a chapter?**
Strong answer: the title page and copyright page, because a scanned PDF usually loses them first, and without them the edition is unknown. Weak answer: starts reading the chapter.

## 15. Key terms

- **Edition** — a substantive revision with a new ISBN.
- **Printing** — a production run of the same edition.
- **Revised reprint** — a reprint with corrections but the same edition number.
- **Impression** — a synonym for printing.
- **Copyright year** — the year of copyright registration, often before release.
- **Publication lag** — the gap between evidence and printed synthesis.
- **ISBN** — identifier for a specific edition and format.
- **eISBN** — identifier for an electronic edition.
- **Chapter author** — the person accountable for a chapter in an edited book.
- **Editor** — the person who assembles and commissions chapters.
- **Companion website** — publisher site with figures, questions, and errata.
- **Errata** — published corrections to a specific printing.
- **International edition** — a territory-specific printing that can differ in content.
- **Tertiary source** — a synthesis that points to primary evidence.
- **Definition source** — a book used for vocabulary and definitions.
- **Mechanism source** — a book used for causal explanation.
- **Pattern reference** — a book used for typical appearances.
- **Content volatility** — how quickly a subject's claims change.
- **Edition gap** — the difference between the cited and latest editions.
- **Within tolerance** — a gap small enough for the claim class.
- **Materially superseded** — two or more editions behind.
- **Out of print** — no longer produced, not necessarily invalid.
- **Platform version** — the version served by an online book platform.
- **Access route** — how the text was obtained.
- **Shelf life** — the period for which a statement remains adequate.
- **Universal negative** — a claim that no treatment or case exists.
- **Guideline divergence** — a difference between book and recommendation.
- **Definition drift** — a change in a term's meaning over time.
- **Board-review book** — a derivative text for examination preparation.
- **Local protocol** — the institution's own procedure, which may differ from both book and guideline.
- **Translation lag** — the interval between an original edition and its translation.
- **Local adaptation** — a licensed edition with territory-specific chapters.
- **Figure edition** — the edition in which a specific figure appears.
- **Reference cutoff** — the newest reference date in a chapter.
- **Page-stable citation** — a citation that resolves to the same page across copies.
- **Volatility class** — the slow, moderate, or fast label used in the tolerance table.
- **Edition registry** — the maintained list of latest known editions.
- **Successor edition** — an announced or published replacement for the edition held.
- **Cutoff statement** — the preface's statement of the literature reviewed.
- **Chapter byline** — the named author of a chapter.
- **Format** — print, e-book, or platform, each with its own identifier.
- **Resolvability** — whether a reader can reach the exact text cited.
- **Shelf copy** — the physical copy available locally, often an older edition.
- **Library licence** — the terms under which an institution may use an electronic text.
- **Errata page** — the publisher page listing corrections for a title.

## 16. Sources

All URLs were checked on 26 October 2026. Edition details were verified against publisher or library catalogue pages.

**Bickley, L. S., Szilagyi, P. G., Hoffman, R. M., and Soriano, R. P.** *Bates' Guide to Physical Examination and History Taking*, 13th edition. Wolters Kluwer, 2023. ISBN 9781975210533.
Good for: examination technique, history taking, documentation, and clinical reasoning vocabulary. Not for: data-science methods or current treatment recommendations.

**Goldberger, A. L., Goldberger, Z. D., and Shvilkin, A.** *Goldberger's Clinical Electrocardiography: A Simplified Approach*, 10th edition. Elsevier, 2023. ISBN 9780323824750.
Good for: ECG interpretation and the clinical meaning of waveform features. Not for: signal-processing methods or device-grade algorithm validation.

**Thaler, M. S.** *The Only EKG Book You'll Ever Need*, 10th edition. Wolters Kluwer, 2022. ISBN 9781975185831.
Good for: introductory ECG pattern recognition. Not for: edge cases or quantitative ECG analysis.

**Herring, W.** *Learning Radiology: Recognizing the Basics*, 5th edition. Elsevier, 2024 copyright. ISBN 9780323878173.
Good for: modality overview and common findings. Not for: radiology-level interpretation or reporting standards.

**Lilly, L. S. (ed.)** *Pathophysiology of Heart Disease*, 8th edition. Wolters Kluwer, 2025. ISBN 9781975216627.
Good for: cardiovascular physiology and disease mechanisms. Not for: current therapy, which changes faster than the edition cycle.

**West, J. B., and Luks, A. M.** *West's Respiratory Physiology: The Essentials*, 11th edition. Wolters Kluwer, 2021. ISBN 9781975139186.
Good for: ventilation, gas exchange, and acid-base physiology. Not for: clinical management of respiratory failure.

**Blumenfeld, H.** *Neuroanatomy through Clinical Cases*, 3rd edition. Oxford University Press, 2021. ISBN 9781605359625.
Good for: localization and neuroanatomy through cases. Not for: current neurological therapy.

**Jankovic, J., Mazziotta, J. C., Pomeroy, S. L., and Newman, N. J. (eds.)** *Bradley and Daroff's Neurology in Clinical Practice*, 8th edition. Elsevier, 2022. ISBN 9780323642613.
Good for: neurological disease reference and diagnostic vocabulary. Not for: a quick read; it is a two-volume reference.

**Armstrong, T.** *The Power of Neurodiversity*. Da Capo Lifelong Books, 2010.
Good for: neurodiversity-affirming language and design considerations. Not for: current clinical guidance; the framing has evolved since publication.

**Schwinger, R. H. G.** "Pathophysiology of heart failure." *Cardiovascular Diagnosis and Therapy* 11(1):263-276, 2021. https://doi.org/10.21037/cdt-20-302
Good for: a peer-reviewed review of heart failure mechanisms, citable as a review rather than a textbook. Not for: treatment recommendations, which require a guideline.

**U.S. National Library of Medicine.** NLM Catalog. https://www.ncbi.nlm.nih.gov/nlmcatalog/
Good for: verifying editions, ISBNs, and publisher details. Not for: content review.

**International Organization for Standardization.** ISO 2108 (ISBN). https://www.iso.org/standard/65483.html
Good for: the identifier structure and check-digit rules. Not for: free access; the standard is paywalled.

**International Committee of Medical Journal Editors.** Recommendations. https://www.icmje.org/recommendations/
Good for: disclosure and authorship conventions that also apply to book chapters. Not for: book-specific guidance.

**World Wide Web Consortium (W3C).** Web Content Accessibility Guidelines. https://www.w3.org/WAI/standards-guidelines/wcag/
Good for: accessibility expectations for electronic texts and course materials. Not for: publisher licence terms, which are separate.

**U.S. Copyright Office.** Fair use index. https://www.copyright.gov/fair-use/
Good for: understanding the factors that govern reuse of figures and passages. Not for: legal advice on a specific use.

## 17. Links

- Module index: [`_index.md`](_index.md)
- Previous topic: [`01_how-to-read-this-list.md`](01_how-to-read-this-list.md)
- Next topic: [`03_regulation-and-policy.md`](03_regulation-and-policy.md)
- Clinical background: [`05_clinical-background.md`](05_clinical-background.md)
- Survey file: [`../../course/22_sources_and_further_reading.md`](../../course/22_sources_and_further_reading.md)
- Labs: [`../../labs/`](../../labs/)
- Module 01, clinical foundations: [`../01_clinical_foundations/`](../01_clinical_foundations/)
- Module 11, ECG signal work: [`../11_signals/`](../11_signals/)

The next topic moves from books to the regulatory instruments that govern devices and AI, where version and status are legal questions rather than editorial ones.

