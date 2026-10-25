# Regulatory and High-Risk Claim Verification Log

> **Purpose:** Track every regulatory, legal, clinical-safety, and market claim that could mislead a learner or product team.
>
> **Status:** Desk-checked against public sources as of 24 October 2026. **No claim has been reviewed by legal counsel or a regulatory professional.** High-risk claims remain pending until signed off.
>
> **Rule:** a claim is not “verified” because a URL resolves. It is verified when a qualified reviewer confirms the interpretation, jurisdiction, date, and applicability.

---

## 1. Verification workflow

1. **Identify the claim.** One row per claim, not one per paragraph.
2. **Assign a risk tier.**
   - **Tier 1:** law, regulation, device classification, reimbursement, privacy.
   - **Tier 2:** clinical safety, validation, performance, recall.
   - **Tier 3:** market size, salary, dataset size, competition status.
3. **Find the primary source.** Guidance, statute, standard, FDA decision, peer-reviewed paper, official dataset page.
4. **Record the exact date and status** (final, draft, proposed, withdrawn, vacated, reissued).
5. **Check jurisdiction and applicability.**
6. **Have a qualified reviewer sign off.**
7. **Set a recheck date.**
8. **Log corrections and propagate them to every module.**

## 2. Status definitions

| Status | Meaning |
|---|---|
| Desk-checked | Author checked a public primary source; not professionally reviewed |
| Pending counsel | Legal/regulatory interpretation needed |
| Pending reviewer | Clinical/technical reviewer needed |
| Verified | Qualified reviewer signed off |
| Superseded | A newer source or rule replaces it |
| Unverified | Could not confirm; do not present as fact |

## 3. Claim register

### Tier 1 — Regulatory, legal, reimbursement, privacy

| ID | Claim | Module | Primary source | Date/status | Verification | Recheck |
|---|---|---|---|---|---|---|
| R-001 | FDA has authorized more than 1,600 AI-enabled devices as of September 2026; roughly three-quarters are radiology | 00, 14, 18 | FDA AI-enabled device list | Sep 2026; ongoing | Desk-checked; Pending counsel | Apr 2027 |
| R-002 | FDA PCCP final guidance issued 4 Dec 2024 and reissued 18 Aug 2025 | 14, 22 | FDA PCCP guidance | Final; reissued 2025 | Desk-checked; Pending counsel | Apr 2027 |
| R-003 | FDA QMSR effective 2 Feb 2026 and incorporates ISO 13485:2016 | 14, 14A | FDA QMSR page | Final; effective 2026 | Desk-checked; Pending counsel | Apr 2027 |
| R-004 | FDA LDT rule vacated 31 Mar 2025 and rescinded 19 Sep 2025 | 14 | Federal Register / FDA LDT page | Final; rescinded 2025 | Desk-checked; Pending counsel | Apr 2027 |
| R-005 | FDA GMLP has ten guiding principles, published 2021 | 14, 14A, 14B | FDA GMLP | Final; 2021 | Desk-checked; Pending counsel | Apr 2027 |
| R-006 | FDA CDS final guidance dated January 2026 | 14 | FDA CDS guidance page | Final; Jan 2026 | Desk-checked; Pending counsel | Apr 2027 |
| R-007 | FDA generative-AI device discussion paper issued Aug 2026; comments closed 19 Oct 2026 | 14 | FDA discussion paper, docket FDA-2026-N-7874 | Discussion paper; not guidance | Desk-checked; Pending counsel | Apr 2027 |
| R-008 | FAERS was renamed/merged into AEMS in March 2026 | 13, 21, 22 | FDA AEMS page | Current; 2026 | Desk-checked; Pending counsel | Apr 2027 |
| R-009 | EU AI Act 2024/1689 in force Aug 2024; Digital Omnibus 2026/1744 delays Annex I high-risk medical devices to 2 Aug 2028 | 14, 14B | EUR-Lex; Commission AI page | Amended 2026 | Desk-checked; Pending counsel | Apr 2027 |
| R-010 | EU MDR 2017/745 and IVDR 2017/746 apply in parallel with the AI Act | 14, 14A | EUR-Lex | In force | Desk-checked; Pending counsel | Apr 2027 |
| R-011 | EHDS Regulation 2025/327 phases in governance 2027, secondary use 2029, omics/trials 2031 | 14 | EUR-Lex / EC EHDS page | In force 2025 | Desk-checked; Pending counsel | Apr 2027 |
| R-012 | HTI-1 uses USCDI v3 baseline from 1 Jan 2026 and includes DSI transparency criteria | 14, 22 | ONC HTI-1 | Final; 2026 | Desk-checked; Pending counsel | Apr 2027 |
| R-013 | CMS-0057-F phases prior-authorization APIs from Jan 2026 to Jan 2027 | 14, 22 | CMS interoperability page | Final rule | Desk-checked; Pending counsel | Apr 2027 |
| R-014 | HIPAA Security Rule NPRM was published Jan 2025; final rule not published at cutoff | 14, 16 | HHS NPRM fact sheet | Proposed; pending | Desk-checked; Pending counsel | Apr 2027 |
| R-015 | FDA premarket cybersecurity expectations include SBOM, vulnerability monitoring, and a response plan | 14, 14A, 14B | FDA cybersecurity guidance | Final; 2025 | Desk-checked; Pending counsel | Apr 2027 |
| R-016 | IEC 60601-1 Edition 3.2 is the current general medical electrical safety standard | 14A, 14B | IEC webstore | Current edition | Desk-checked; Pending reviewer | Apr 2027 |
| R-017 | IEC 62304 is the medical device software lifecycle standard; Edition 2 is not final | 14, 14A, 14B | IEC | Current edition 2006+A1:2015 | Desk-checked; Pending reviewer | Apr 2027 |
| R-018 | ISO 14971:2019 is the risk-management standard for medical devices | 14, 14A, 14B | ISO | Current | Desk-checked; Pending reviewer | Apr 2027 |
| R-019 | ISO 13485:2016 is the medical device QMS standard | 14, 14A, 14B | ISO | Current | Desk-checked; Pending reviewer | Apr 2027 |
| R-020 | ISO 10993-1:2025 is the current biological evaluation standard | 14A | ISO | Published 2025 | Desk-checked; Pending reviewer | Apr 2027 |
| R-021 | IEC 80601-2-77 covers robotically assisted surgical equipment | 14A, 25 | IEC | 2019 + AMD1:2023 | Desk-checked; Pending reviewer | Apr 2027 |
| R-022 | IEC 80601-2-78 covers rehabilitation/assistive medical robots | 14A, 25 | IEC | 2019 + AMD1:2024 | Desk-checked; Pending reviewer | Apr 2027 |
| R-023 | ISO/TS 15066 addresses collaborative industrial robots and is informative, not a substitute for medical standards | 14A, 25 | ISO | 2016; under revision | Desk-checked; Pending reviewer | Apr 2027 |
| R-024 | IEC 81001-5-1:2021 is the health software security lifecycle standard | 14A, 14B | IEC | 2021 | Desk-checked; Pending reviewer | Apr 2027 |
| R-025 | CLIA/CAP govern US clinical laboratory testing; LDTs have a separate regulatory history | 12, 21 | CMS/CLIA; CAP | Current | Desk-checked; Pending counsel | Apr 2027 |
| R-026 | Medicare coverage is separate from FDA authorization | 14, 17 | CMS | Current | Desk-checked; Pending counsel | Apr 2027 |

### Tier 2 — Clinical, device, and safety claims

| ID | Claim | Module | Primary source | Date/status | Verification | Recheck |
|---|---|---|---|---|---|---|
| R-027 | IDx-DR received FDA De Novo authorization in 2018 for autonomous diabetic retinopathy screening | 02, 10, 14 | FDA DEN180001 | 2018 | Desk-checked; Pending reviewer | Apr 2027 |
| R-028 | Apple ECG app received FDA De Novo authorization (DEN180044) | 14 | FDA DEN180044 | 2018 | Desk-checked; Pending reviewer | Apr 2027 |
| R-029 | Apple hypertension notification cleared 11 Sep 2025 (K250507); sensitivity 41.2%, specificity 92.3%, n=2,229 | 11, 21 | FDA K250507; Apple validation paper | 2025 | Desk-checked; Pending reviewer | Apr 2027 |
| R-030 | da Vinci 5 received FDA 510(k) clearance in March 2024 | 14A | Intuitive press release; FDA | 2024 | Desk-checked; Pending reviewer | Apr 2027 |
| R-031 | Hugo received FDA 510(k) K250725 in December 2025 for urologic procedures | 14A | Medtronic; FDA K250725 | 2025 | Desk-checked; Pending reviewer | Apr 2027 |
| R-032 | Ottava received FDA market authorization DEN250068 in July 2026 | 14A | J&J; FDA DEN250068 | 2026 | Desk-checked; Pending reviewer | Apr 2027 |
| R-033 | Mendaera Focalist received FDA 510(k) K250524 in July 2025 | 14A | Mendaera; FDA | 2025 | Desk-checked; Pending reviewer | Apr 2027 |
| R-034 | ReWalk 7 received FDA 510(k) K241822 in March 2025 | 14A | Lifeward; FDA K241822 | 2025 | Desk-checked; Pending reviewer | Apr 2027 |
| R-035 | CyberKnife recall Z-1023-2025 concerned snap-ring detachment and uncontrolled roll-axis rotation | 14A | FDA recall database | 2025 | Desk-checked; Pending reviewer | Apr 2027 |
| R-036 | Among 950 cleared AI devices, 6.3% had recalls; 43.4% within 12 months; diagnostic/measurement errors were the most common cause (109 of 182) | 18 | JAMA Health Forum study | 2025 | Desk-checked; Pending reviewer | Apr 2027 |
| R-037 | Among 903 FDA-authorized AI devices (76.6% radiology), 55.9% reported clinical performance studies; 38.2% retrospective, 8.1% prospective, 2.4% randomized, 16.2% AUC | 18 | JAMA Network Open study | 2025 | Desk-checked; Pending reviewer | Apr 2027 |
| R-038 | Epic Sepsis Model external validation found lower discrimination than vendor claims and high alert burden | 01, 05, 06, 08, 15 | JAMA Internal Medicine 2021 | 2021 | Desk-checked; Pending reviewer | Apr 2027 |
| R-039 | Pulse oximetry bias by skin pigmentation is documented | 01, 11, 16 | NEJM 2020; FDA communications | Current | Desk-checked; Pending reviewer | Apr 2027 |
| R-040 | GRAIL NHS-Galleri primary endpoint IRR 1.03 (95% CI 0.92–1.14); stage IV −14%, stage I/II +16%; reported 2026 | 21 | ASCO/peer-reviewed publication | 2026 | Desk-checked; Pending reviewer | Apr 2027 |
| R-041 | Insilico rentosertib Phase IIa: 71 patients; FVC +98.4 mL vs −20.3 mL | 13 | Nature Medicine / company | 2025 | Desk-checked; Pending reviewer | Apr 2027 |
| R-042 | No AI-discovered drug had been approved as of the source cutoff | 13 | Regulatory landscape | 2026 | Desk-checked; Pending reviewer | Apr 2027 |

### Tier 3 — Datasets, market, and competition

| ID | Claim | Module | Primary source | Date/status | Verification | Recheck |
|---|---|---|---|---|---|---|
| R-043 | MIMIC-IV v3.1: 364,627 patients; 546,028 hospitalizations; 94,458 ICU stays; 2008–2022 | 02, 21 | PhysioNet | v3.1 | Desk-checked | Oct 2027 |
| R-044 | All of Us CDR v9: 747,000+ participants; 535,000+ short-read WGS; 553,000 arrays; ~96,000 SV; ~14,000 long-read | 21 | All of Us | v9, June 2026 | Desk-checked; verify release | Oct 2027 |
| R-045 | HealthBench: 5,000 conversations; 48,562 rubric criteria; 262 physicians; best models ~32% on Hard | 07, 18, 21 | OpenAI HealthBench | 2025 | Desk-checked | Oct 2027 |
| R-046 | MedAgentBench: 100 patients, 300 physician-written FHIR tasks | 07, 21 | Stanford ML Group | 2025 | Desk-checked | Oct 2027 |
| R-047 | EHRSHOT: 6,739 patients, 15 tasks | 21 | EHRSHOT | Current | Desk-checked | Oct 2027 |
| R-048 | PhysioNet 2025 Chagas task uses precision in top 5% as the metric | 19, 21 | PhysioNet | 2025 | Desk-checked | Oct 2027 |
| R-049 | PhysioNet 2026 PSG task uses age-conditioned AUROC | 19, 21 | PhysioNet | 2026 | Desk-checked | Oct 2027 |
| R-050 | RSNA 2026 knee challenge closed 22 Oct 2026 with 12 findings and macro ROC AUC | 19 | RSNA/Kaggle | 2026 | Desk-checked; team count unverified | Oct 2027 |
| R-051 | BLS median data scientist salary $120,230 (May 2025) | 20 | BLS | 2025 | Desk-checked | Oct 2027 |
| R-052 | Change Healthcare breach affected ~192.7M individuals | 16 | HHS OCR; GAO | 2024–25 | Desk-checked | Oct 2027 |
| R-053 | Epic Cosmos covers ~2,358 hospitals and ~310M de-identified patients | 00, 02 | Epic Cosmos (vendor) | 2026 | Desk-checked; vendor claim | Oct 2027 |
| R-054 | BLS/industry salary ranges are dated snapshots, not guarantees | 20 | BLS; postings | 2025–26 | Desk-checked | Oct 2027 |

## 4. High-risk items requiring professional review first

These must be reviewed before any public claim:

- FDA device classification and CDS exclusion analysis (R-001, R-006, R-026).
- PCCP scope and whether a retrain/quantization change is covered (R-002, R-015, R-035).
- EU AI Act/MDR dual compliance and 2028 timing (R-009, R-010).
- HIPAA Security Rule status and de-identification (R-014).
- CLIA/LDT and lab-developed AI (R-025).
- Robot safety standards and claims (R-021 to R-024).
- Any statement that a device is “cleared,” “approved,” or “validated” (R-027 to R-035).

## 5. Claim-change propagation

If a source changes:

1. Mark the row **Superseded**.
2. Add the new source and date.
3. Find every module that cites the claim.
4. Update the module and the `CHANGELOG.md`.
5. Re-run the link checker and DOI checker.
6. Notify reviewers and learners if the change affects a graded artifact.

## 6. Sign-off block

```markdown
Claim ID:
Reviewer:
Role/qualification:
Date:
Jurisdiction:
Source reviewed:
Interpretation:
Status: [ ] Verified [ ] Corrected [ ] Rejected [ ] Needs more evidence
Notes:
```

## 7. Limitations

- Desk checks are not legal advice.
- Public FDA pages can move; a resolved URL is not a verified interpretation.
- Guidance can be draft, final, withdrawn, or rescinded.
- EU implementation dates can change.
- Device clearances are indication-specific.
- Market and salary figures are snapshots.
- No reviewer has signed this log yet.
