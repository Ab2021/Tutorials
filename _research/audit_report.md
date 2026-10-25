# Audit Report — Biomedical Data Science for Industry (2026)

**Audit date:** 24 October 2026, 18:20–18:35 IST
**Package snapshot:** 25 module files, 13,794 module lines; README 393 lines; COURSE_OUTLINE 637 lines.
**Important:** The package changed while this audit was running. Modules `01A_foundations_assumptions_and_units.md` and `14A_medical_hardware_sensors_chips_robotics.md` appeared at 18:22 and 18:27; README, COURSE_OUTLINE, CHANGELOG, Module 00, Module 19 and the research notes were edited repeatedly between 18:21 and 18:31. Findings below refer to the state hashed at 18:33 (14A `9efc0c2a`, README `42b182a7`, OUTLINE `45b51089`, CHANGELOG `57ad0086`). Several structural defects may be transient artifacts of that edit burst; the substantive findings (sources, arithmetic, assessment design) are not.

**Method:** full reads of README, COURSE_OUTLINE, CHANGELOG, `_course_spec.md`, 01A and 14A; headings plus targeted sections of all other modules; programmatic checks for internal links, module references, duplicate paragraphs, rubric sums and line targets; Crossref REST metadata for every DOI; direct fetch/search verification of 35+ regulatory, dataset, paper, device and benchmark sources; independent re-derivation of the schedule arithmetic.

---

## Summary (10 lines)

1. The package is unusually strong on industry framing, regulation, dataset caveats and capstone design; the two newly added modules close the largest content gap (units, measurement, sensors, chips, firmware, electrical safety, robotics).
2. It is not internally consistent: 690–730 guided hours cannot fit 36 weeks at the stated 12–15 h/week (that requires 19–20 h/week), and Term 3 gives 10 weeks to 208–248 hours of modules plus a 10–12-week capstone.
3. 01A and 14A are absent from the program map, term tables and every track's required list, despite being labelled "common core" and "MedTech core"; 14A is filed under the Term 3 heading in COURSE_OUTLINE.
4. 35% of the grade — the midterm evidence dossier (20%) and track playbook (15%) — has no brief, template, rubric or owning module anywhere in the package.
5. Source hygiene is good but the verification pass overstates itself: its "no genuine 404" conclusion misclassifies at least four genuinely moved FDA pages as bot-blocking, and its DOI sweep silently skipped the one DOI that is wrong.
6. Confirmed citation error: MERIT-HF is cited twice with DOI `10.1016/S0140-6736(99)01350-3`; the correct DOI is `10.1016/S0140-6736(99)04440-2` (Crossref).
7. Confirmed staleness: GRAIL NHS-Galleri results are dated 2025 but were reported at ASCO on 30 May 2026 and published 22–23 September 2026; All of Us is described at CDR v8 although v9 shipped 26 June 2026.
8. Several hard numbers are sourced only to organisation homepages (denial rates, revenue leakage) or are slightly mislabelled ("903 imaging devices", "software issues dominated").
9. Module 15 (Deployment/MLOps, 368 lines) is the thinnest technical module and below its own spec target, in the area employers weight most.
10. Fix the schedule, integrate the two new modules, repair the FDA/AEMS links and the MERIT-HF DOI, define the two missing assessments, then deepen lifecycle/reliability/HRI and Module 15.

---

## 1. Critical fixes (must do)

### C1. The time architecture is arithmetically impossible

| Source | Claim | Arithmetic |
|---|---|---|
| README §3 | 36 weeks; 12–15 hours/week | 432–540 h |
| README §3 | Total guided time 690–730 h | 19.2–20.3 h/week over 36 weeks |
| Module 00 §4.3 | "A 12–20 hour week for 36 weeks is roughly 430–720 hours"; module estimates "690–730" | 12–20 h/wk × 36 = 432–720 h; 730 exceeds even the 20 h/wk bound |
| Sum of module headers | 204 (Term 1) + 276 (Term 2) + 208–248 (Term 3) = **688–728 h** | matches 690–730 only to ±2 h, not to 12–15 h/wk |

Term-level load is worse than the average:

- Term 1: 204 h over 12 weeks = **17.0 h/week**.
- Term 2: 276 h over 14 weeks = **19.7 h/week**.
- Term 3: 208–248 h over 10 weeks = **20.8–24.8 h/week**, before the capstone.

Term 3 also contains the capstone, which Module 19 schedules as a **twelve-week** plan and README describes as 10–12 weeks. Ten weeks cannot hold a 10–12-week capstone plus Modules 15–22.

**Fix (choose one and make every document agree):**
- (a) Extend to 44–48 weeks and keep 690–730 h; or
- (b) Keep 36 weeks and cut to ~500–540 h (reduce labs, merge 17/18/20/22, shorten capstone to 8 weeks); or
- (c) Keep both and state explicitly that the programme is 19–20 h/week and that Term 3 is 14 weeks (weeks 27–40), with the capstone running in parallel from week 29.
Whichever is chosen, recompute `README §3`, `README §5`, `COURSE_OUTLINE §1–4`, `00 §4.1/4.3`, `19 §4`, and the assessment summary.

### C2. 01A and 14A are not integrated into the programme structure

The modules exist and are good, but the architecture was not updated around them:

- `COURSE_OUTLINE §1` program map still says Term 1 = `00–06`, Term 2 = `07–14`. Neither range includes 01A or 14A.
- `COURSE_OUTLINE` files Module 14A under the **"## 4. Term 3"** heading, although README places it in Term 2 and its prerequisites (01, 01A, 11, 14) are all Term 1–2 modules.
- `00 §4.1` says Term 2 = `7–14`, Term 3 = `15–21` (omits both 14A and 22).
- Track tables (`COURSE_OUTLINE §1.2`, `00 §4.4`) list MedTech required as `00–07, 10, 11, 14–19` — a numeric range that excludes 14A — despite 14A declaring "MedTech core". No track lists 01A as required despite "Common core".
- README §2 entry points still send ML engineers entering medtech to "00, 01, 14–16", and the self-study path still says "Modules 01–06 in order". Neither mentions 01A/14A.
- The dependency map (`00 §4.2`) gives 14A parents 01A and 14 but omits 11, which 14A lists as a prerequisite.

**Fix:** use explicit lists rather than ranges; add 01A after 01 in Term 1 and 14A after 14 in Term 2; add 01A to all three tracks and 14A to MedTech required; move the 14A section of COURSE_OUTLINE into Term 2; update README §2 and §12.

### C3. Two assessments worth 35% of the grade are undefined

| Assessment | Weight | Where defined |
|---|---:|---|
| Midterm evidence dossier | 20% | Only two table rows (`00 §4.5`, `COURSE_OUTLINE §5`). No brief, template, milestone, rubric, or module owner. |
| Track playbook | 15% | Only two table rows. No definition of "playbook", no template, no rubric, no module owner. |

Every other assessment (labs, case critiques, capstone) has a rubric or a described artifact somewhere. These two do not. A student cannot know what is being graded, and an instructor cannot grade it consistently.

**Fix:** add a one-page brief plus rubric for each in Module 19 (or Module 00 for the midterm), specify page limits, required sections, evidence standard and submission week, and name the assessor.

### C4. The source-verification pass draws a false conclusion

`_research/link_verification_summary.md` states: "after replacement of genuinely dead URLs, no cited source remains a confirmed 404" and attributes all 22 remaining FDA 404s to "abuse detection". This is wrong for at least four URLs, which are genuinely moved or superseded and which I confirmed independently:

| URL in package | Reality | Evidence |
|---|---|---|
| `/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device` | Page no longer exists at this path; the current AI-device page is under `digital-health-center-excellence` | FDA page title "Artificial Intelligence-Enabled Medical Devices", "over 1,600 … as of September 2026" |
| `/medical-devices/artificial-intelligence-enabled-medical-devices/artificial-intelligence-enabled-medical-devices` | Returns 404; the index lives under `digital-health-center-excellence` | as above |
| `/medical-devices/quality-and-compliance-medical-devices/quality-management-system-regulation-qmsr` | Moved to `/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr` | FDA QMSR page, updated 2 Feb 2026 |
| `/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers` | 301 to the **AEMS** page; FAERS was renamed/merged in March 2026 | FDA AEMS page, "formerly FDA Adverse Event Reporting System (FAERS)"; Reuters 11 Mar 2026 |

The blanket rule "FDA 404 = bot detection" caused the package to miss real breaks. A bot-block test needs a second signal: fetch the page title in a browser, query the site's search endpoint, or search `site:fda.gov "<exact title>"`. Only then classify as bot-blocked.

### C5. One DOI is wrong and the checker structurally could not catch it

Module 04 cites MERIT-HF twice (lines 365 and 613) with `https://doi.org/10.1016/S0140-6736(99)01350-3`. Crossref returns 404. The correct DOI, confirmed by bibliographic search, is `10.1016/S0140-6736(99)04440-2` ("Effect of metoprolol CR/XL in chronic heart failure: MERIT-HF", Lancet, June 1999).

The reason it was missed: the DOI extractor splits on `)`, so every DOI containing parentheses was dropped from `doi_check.tsv`. The "42/42 DOIs resolved" claim therefore excludes exactly the class of DOI that contains the error. The Ghassemi DOI `10.1016/S2589-7500(21)00208-9` is correct, but it was also excluded from the check.

**Fix:** correct both MERIT-HF occurrences; extend the extractor to handle balanced parentheses and re-run; report the denominator as "39/41 extractable + 2 manually checked".

### C6. Stale facts in a package dated October 2026

| Claim | Location | Problem | Correct state |
|---|---|---|---|
| "NHS-Galleri (primary endpoint reported 2025)" | `21` line 349 | Results were reported at ASCO on **30 May 2026** and published in NEJM / Nature Medicine on **22–23 September 2026** | Update year and add the publication citation |
| "All of Us CDR v8 … 447,278 genotyped; 414,830 WGS; 97,061 SV; 2,800 long-read" | `21` line 110 | These v8 figures are numerically correct but **v9 shipped 26 June 2026**, before the stated cutoff | v9: 747,000+ participants, 535,000+ short-read WGS, 553,000 arrays, ~96,000 SV, ~14,000 long-read, first clinical notes |
| "FAERS" used as the current system name | `13` line 248, `21` dataset table, `_research/industry_brief.md` | FAERS became **AEMS** on 11 March 2026; MAUDE is being folded in | Rename to AEMS (formerly FAERS); keep "FAERS" only as a historical alias |

### C7. README quality-check claims are not true

`README §17.1` asserts:

- "Regulatory dates flagged for verification" — there are **zero** `[VERIFY]` markers in the entire package (`grep -r '\[VERIFY\]'` = 0), despite `_course_spec.md` principle 8 and the changelog claiming the guidance was added.
- "Internal Markdown links resolve" — technically true but vacuous: the package contains exactly **one** internal Markdown link (README's "Start here"). All ~60 cross-references are plain text ("Module 07"), and every module footer "Next:" is plain text. The spec's quality gate asks for cross-links.
- "Track and capstone mappings cross-checked" — contradicted by C2 (01A/14A missing from all track lists; recommended electives are subsets of required lists in all three tracks).

**Fix:** either add the markers/links or remove the claims. A false quality checklist is worse than no checklist, because it stops the next reviewer from looking.

### C8. Changelog contradicts the package

`CHANGELOG.md` line 31 says "Corrected the course module count from 22 to 23 (Modules 00–22)" while line 9 of the same file says "Complete 25-module curriculum" and README/outline say 25. Remove or rewrite the line as "22 → 23 core modules; 01A and 14A added later, total 25".

### C9. Malformed or unverified links in the new hardware module

- `https://www.iso.org/standard/10993-1` — not a valid ISO catalogue identifier (ISO uses numeric document IDs, e.g. `/standard/59752.html`). This looks constructed. Replace with the ISO catalogue search result for ISO 10993-1:2025, or cite the standard without a URL.
- `https://www.mdpi.com/2072-666X/17/1/51` is cited as 2025, but *Micromachines* volume 17 corresponds to 2026 (volume 16 = 2025). Verify year/volume or drop.
- `https://www.fda.gov/medical-devices/medical-device-recalls-and-early-alerts/medtronic-recalls-stealthstation-s8-application-versions-120-110-103-102-and-101-software-error-may` returns 404 and could not be confirmed by search. Verify against the FDA recall database or remove.
- `Z-1927-2025` (EPIONE) and `K251314` (Rosa) could not be confirmed in the time available. Mark `[VERIFY]` rather than presenting as fact.

### C10. Module 21 does not contain the datasets that 14A and the new briefs require

14A teaches and cites Cholec80, CholecT50, JIGSAWS, MISAW, HeiChole, dVRK, SurRoL, Surgical Gym, and CGM. Capstone briefs 38–41 use Cholec80, dVRK, OhioT1DM and gait data. **None of these appear in Module 21's catalogue**, which is the package's designated dataset reference and is a stated prerequisite for the capstone. A student following the intended path cannot licence or find the data.

**Fix:** add a "Surgical, robotic, and implantable-device data" section to Module 21 with Cholec80/CholecT50 (registration, non-commercial), JIGSAWS (Johns Hopkins, DUA), dVRK (custom), OhioT1DM (request), and a public IMU/gait set; add device post-market data (AEMS/MAUDE).

---

## 2. High-value additions (should do)

### H1. Expand Module 15 (Deployment and MLOps)

At 368 lines it is 18% below its own spec target (450–550) and is the thinnest technical module in the package, despite deployment being the skill employers screen for. It has only seven `###` subsections and five labs. Missing:

- Edge/embedded operations: OTA update, signed firmware, rollback, device fleet versioning, battery/thermal budget.
- Device and EHR integration: IEEE 11073, IHE PCD, FHIR `Device`/`DeviceMetric`, time synchronization (NTP/PTP) — 14A introduces these but 15 never operationalises them.
- Drift taxonomy with worked examples and detection thresholds; delayed-label monitoring; proxy metrics.
- Incident response with a real post-mortem (e.g., a sepsis-alert false-negative incident) and a rollback drill.
- Cost model: inference cost, storage, on-call, validation cost per model change.
- Change control under PCCP: how a retrain maps to a pre-authorised modification versus a new submission.

### H2. Device lifecycle depth (the one item on the user's checklist still only partially covered)

14A covers firmware lifecycle and supply chain, and Module 14 covers QMS/design controls conceptually, but the package never teaches the **artifacts**: Design History File, Device Master Record, design transfer, process validation (IQ/OQ/PQ), supplier qualification, CAPA workflow, UDI, complaint handling, recall execution, or end-of-life/service planning. This is the difference between "knows the standards" and "can work in a device company". See the proposed Module 14B outline in §8.

### H3. Reliability and measurement-systems engineering

No treatment of FMEA/FMECA, fault-tree analysis, derating, MTBF, HALT/HAST, accelerated life testing, shelf-life/packaging validation, or Gauge R&R. These are standard vocabulary in device and semiconductor roles and are cheap to add as one section plus one lab.

### H4. Human–robot interaction depth

14A §7.11 is a ten-line bullet list. For a 2026 course that claims robotics, add: trust calibration and over-trust, situation awareness, workload measurement (NASA-TLX), handover and shared-control modes, team communication in the OR, training/credentialing, and a usability-study method mapped to IEC 62366-1. Pair with a summative-usability lab.

### H5. EMC depth

14A §5.6 names EMC in one paragraph. Add IEC 60601-1-2 test categories (emissions, immunity, ESD, radiated RF, conducted disturbances), pre-compliance testing, and the interaction with electrosurgery and MRI. This matters because EMC findings routinely force filter/sampling changes that alter the data a model sees.

### H6. Clinical basics still missing

01A §6–7 substantially improves biology, physiology, pharmacology and epidemiology. Still absent: pathology/histology principles (staining, WSI, grading), pediatrics and obstetrics, geriatrics and palliative care, psychiatry beyond neurodiversity, and laboratory-medicine operations (CLIA, QC, proficiency testing). Add a 2-hour primer on pathology and one on lab operations; both are needed for Modules 10, 12 and 21.

### H7. Datasets

Beyond C10: the catalogue has no CGM dataset, no IMU/gait dataset, no semiconductor or manufacturing dataset, and no device-log dataset. Add OhioT1DM, UCI HAR/Opportunity, and a public fab/process dataset if one can be licensed; if not, state explicitly that this modality is not represented.

### H8. Assessment design

- Add an individual technical assessment (oral or written) in Term 2. At present the only individual gate is the capstone defence; a team can carry a weak member through 70% of the grade.
- Add a rubric for case critiques (15%) — currently "one-page critique" with no criteria.
- Define the track playbook and midterm dossier (C3).
- State how AI assistance is handled in assessed work (Module 00 discusses disclosure but no assessment rule exists).

### H9. Module 22 source hygiene

Three of the most important 2025–2026 guidance items link to the generic FDA guidance **search page** rather than the document: PCCP, Computer Software Assurance, and AI in drug development. A student cannot verify a claim from a search index. Replace with direct document pages:
- PCCP: `https://www.fda.gov/media/166704/download` (issued 18 Aug 2025; originally 4 Dec 2024).
- CSA: `https://www.fda.gov/regulatory-information/search-fda-guidance-documents/computer-software-assurance-production-and-quality-management-system-software` (Feb 2026 revision).
- AI in drug development: cite by title and date and verify the current URL; do not leave the generic search page.

### H10. Regulatory currency

Missing from the modules though present in the research brief: HIPAA Security Rule NPRM (Jan 2025), HTI-2/TEFCA/QHIN governance, information blocking, UDI, NIST AI RMF operationalisation, ISO/IEC 42001, and the FDA AI device list as a data source. Add a short "regulatory watchlist" update discipline with an owner and a date, rather than a static table.

### H11. Make cross-references clickable and bidirectional

Convert "Module NN" to `[Module NN](NN_file.md)` at least in the footer "Next:" lines and in the prerequisite headers. Add reverse links in the dataset catalogue to the modules that use each dataset. This is mechanical and would materially improve usability.

### H12. Book chapter mapping

`_course_spec.md` principle 10 requires chapter-level mapping of the ten supplied clinical books. There are **zero** chapter references in the package; books appear only as a list in Module 01/22. Add a table in Module 01 mapping chapter → concept → module section, or drop the claim from the spec.

---

## 3. Source corrections table

Legend: **DEAD** = genuine 404; **MOVED** = 301/URL change; **SUPERSEDED** = content replaced; **BOT** = valid but bot-blocked; **PAYWALL** = 403 subscription; **WRONG** = citation incorrect; **STALE** = valid but outdated; **UNVERIFIED** = could not confirm.

| # | Location | Old URL / value | Status | Correct value / action | Confidence |
|---:|---|---|---|---|---|
| 1 | `14` L544; `22` L39 | `fda.gov/medical-devices/software-medical-device-samd/artificial-intelligence-and-machine-learning-software-medical-device` | DEAD/MOVED | `https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices` | High (fetched) |
| 2 | `14A` L900? / link check | `fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/artificial-intelligence-enabled-medical-devices` | DEAD | Same as #1 | High |
| 3 | `14` L552; `22` L47 | `.../quality-and-compliance-medical-devices/quality-management-system-regulation-qmsr` | MOVED | `https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr` | High (search) |
| 4 | `14` L546, L573; `22` L40 | `.../software-medical-device-samd/good-machine-learning-practice-medical-device-development-guiding-principles` | MOVED (301) | `https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles` | High (server redirect) |
| 5 | `13` L248 | `.../drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers` | SUPERSEDED | `https://www.fda.gov/drugs/surveillance-post-drug-approval-activities/fda-adverse-event-monitoring-system-aems` | High (fetched) |
| 6 | `_research/industry_brief` §14; `21` | "OpenFDA/FAERS" | STALE | AEMS (formerly FAERS); note MAUDE migration | High |
| 7 | `14` L557; `22` L57 | `https://www.cms.gov/priorities/electronic-prior-authorization/overview` | WEAK (valid but generic) | `https://www.cms.gov/priorities/electronic-prior-authorization/overview` | High (fetched) |
| 8 | `04` L365, L613 | `10.1016/S0140-6736(99)01350-3` | WRONG | `10.1016/S0140-6736(99)04440-2` | High (Crossref) |
| 9 | `21` L561 | `https://arxiv.org/abs/2505.16941` | NONCANONICAL | `https://arxiv.org/abs/2505.16941` | High (resolves, wrong domain) |
| 10 | `10` L473; `21` L548 | `https://nihcc.app.box.com/v/ChestXray-NIHCC` | DEAD/BLOCKED | `https://www.kaggle.com/datasets/nih-chest-xrays/data` (mirror) or the NIH announcement page; label provenance | Medium |
| 11 | `21` L570 | `https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection` | DEAD (redirects to 404) | `https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection` | High (fetched) |
| 12 | `21` L565 | `https://www.kaggle.com/competitions/prostate-cancer-grade-assessment` | DEAD | Kaggle competition page retired; cite PANDA via the original challenge paper and state access is unavailable, or find the current host | Medium |
| 13 | `19` L615 | `https://www.decide-ai.org/` | ERR | Cite the DECIDE-AI statement (BMJ 2022;379:e070904) and EQUATOR; verify a live URL before use | Medium |
| 14 | `09` L550; `22` L118 | `https://www.i2b2.org/` | ERR (781) | Server returns nonstandard 781; test in browser; prefer `https://www.i2b2.org/` or the tranSMART Foundation page | Low |
| 15 | `21` L110 | All of Us CDR v8 figures | STALE | CDR v9 (26 Jun 2026): 747k+ participants, 535k+ WGS, 553k arrays, ~96k SV, ~14k long-read | High (NIH/All of Us) |
| 16 | `21` L349 | "GRAIL … reported 2025" | WRONG DATE | ASCO 30 May 2026; NEJM/Nature Medicine 22–23 Sep 2026 | High |
| 17 | `14A` L951 | `https://www.iso.org/standard/10993-1` | MALFORMED | ISO catalogue entry for ISO 10993-1:2025, or cite without URL | Medium |
| 18 | `14A` L922 | MDPI *Micromachines* 2025;17(1):51 | UNVERIFIED | Volume 17 is 2026; verify year/volume or replace | Medium |
| 19 | `14A` L916 | Medtronic StealthStation recall URL | DEAD/UNVERIFIED | Verify in FDA recall database or remove | Low |
| 20 | `14A` L914–915 | MIRA `Z-1905-2025`; EPIONE `Z-1927-2025` | UNVERIFIED | Verify each recall number in the FDA database | Low |
| 21 | `14A` L383 | Rosa Knee `K251314` | UNVERIFIED | FDA database returned 503 during audit; recheck | Low |
| 22 | `19` L258 area | "~6,425 teams" (RSNA 2026) | UNVERIFIED | Competition and Oct 2026 close confirmed; team count not confirmed | Low |

**Do not change:** the following were checked and are correct — PCCP dates (4 Dec 2024 / reissued 18 Aug 2025), CSA (Sep 2025 / revised Feb 2026), LDT vacatur (31 Mar 2025) and rescission (19 Sep 2025), QMSR effective 2 Feb 2026, CDS final guidance Jan 2026, EU AI Act Digital Omnibus 2026/1744 and Aug 2028 Annex I date, EHDS 2025/327, HTI-1/USCDI v3 Jan 2026, CMS-0057-F 2026/2027 phases, MDCG 2025-6, Apple K250507 metrics, Insilico rentosertib figures, Change Healthcare 192.7M, Epic Cosmos 2,358/310M, Aidoc CARE1 vendor claim, BLS $120,230, MIMIC-IV v3.1 counts, EHRSHOT 6,739/15, MedAgentBench 300 tasks, HealthBench 262/5,000/48,562, PhysioNet 2025 and 2026 task/metric, and the 903-device and 950-device recall studies.

---

## 4. Source verification log (35+ critical sources)

| Source | Claim checked | Result | Confidence |
|---|---|---|---|
| FDA AI-enabled devices page | >1,600 authorised, ~75–76% radiology, as of Sep 2026 | Confirmed (1,614 per FDA dataset, 4 Sep 2026) | High |
| FDA CDS guidance | Final, January 2026 | Confirmed (page says Final, Jan 2026) | High |
| FDA QMSR | Effective 2 Feb 2026, incorporates ISO 13485:2016 | Confirmed | High |
| FDA PCCP guidance | Final 4 Dec 2024, reissued 18 Aug 2025 | Confirmed ("originally issued Dec 4 2024") | High |
| FDA CSA guidance | Final Sep 2025, revised Feb 2026 | Confirmed (3 Feb 2026, supersedes Sep 2025) | High |
| FDA LDT page | Rule vacated; rescinded 19 Sep 2025 | Confirmed | High |
| FDA generative-AI discussion paper | Aug 2026, docket FDA-2026-N-7874, comments closed 19 Oct 2026 | Confirmed | High |
| FDA GMLP | 10 principles, Oct 2021 | Confirmed; URL now 301 | High |
| FDA cybersecurity page | Live under DHCoE | Confirmed | High |
| FDA De Novo DEN180001 / DEN180044 | IDx-DR / Apple ECG | PDFs exist (content not text-extractable) | Medium |
| CMS-0057-F | Jan 2026 process; Jan 2027 APIs | Confirmed | High |
| CMS e-PA page | Five APIs from 1 Jan 2027 | Confirmed | High |
| ASTP HTI-1 | USCDI v3 baseline 1 Jan 2026; DSI criterion | Confirmed | High |
| EHDS 2025/327 | In force 26 Mar 2025; 2029/2031 phases | Confirmed | High |
| MDCG 2025-6 | June 2025, AI Act/MDR interplay | PDF exists (463 KB); content not extracted | Medium |
| EU AI Act 2024/1689 | In force Aug 2024 | Confirmed (EUR-Lex 202) | High |
| Digital Omnibus 2026/1744 | Adopted 8 Jul 2026; Annex I → 2 Aug 2028 | Confirmed via secondary legal sources; OJ text not fetched | Medium |
| Apple K250507 | 11 Sep 2025; 2,229 enrolled; sens 41.2%; spec 92.3% | Confirmed exactly | High |
| Insilico rentosertib | 71 patients; +98.4 mL vs −20.3 mL | Confirmed (Nat Med 2025) | High |
| GRAIL NHS-Galleri | IRR 1.03; stage IV −14%; stage I/II +16% | Confirmed; **date in course wrong** | High |
| Change Healthcare | 192.7M individuals | Confirmed (HHS OCR) | High |
| Epic Cosmos | 2,358 hospitals; 310M patients | Confirmed (Epic site) | High |
| Aidoc CARE1 | Feb 2025; "first foundation-model device" | Confirmed as vendor claim, not independent | High |
| BLS data scientists | $120,230 median; 35% growth | Confirmed | High |
| MIMIC-IV v3.1 | 364,627 / 546,028 / 94,458 | Confirmed | High |
| EHRSHOT | 6,739 patients; 15 tasks | Confirmed (site also says 6,729 once) | High |
| MedAgentBench | 100 patients; 300 tasks; success rate | Confirmed; metric is SR, not "pass@1" | High |
| HealthBench | 262 physicians; 5,000 conversations; 48,562 rubrics; 32% Hard | Confirmed | High |
| PhysioNet 2025 | Chagas; top-5% metric; CODE-15% ~300k | Confirmed | High |
| PhysioNet 2026 | PSG → cognitive impairment; age-conditioned AUROC | Confirmed (5 US institutions, EDF, 214 GiB/1.2 TiB) | High |
| RSNA 2026 | Knee MRI, 12 findings, closed Oct 2026 | Competition confirmed; team count not | Medium |
| 903-device study | 55.9% clinical performance; 38.2/8.1/2.4/16.2% | Confirmed (JAMA Netw Open 2025) | High |
| 950-device recall study | 6.3% recalled; 43.4% within 12 months | Confirmed; top cause is diagnostic/measurement error, not "software" | High |
| Ottava DEN250068 | Jul 2026, upper-abdominal general surgery | Confirmed (21 Jul 2026) | High |
| Hugo K250725 | Dec 2025, urologic | Confirmed (3 Dec 2025) | High |
| Mendaera K250524 | Jul 2025, handheld robotic ultrasound | Confirmed (2 Jul 2025) | High |
| ReWalk 7 K241822 | Mar 2025 | Confirmed (12 Mar 2025) | High |
| CyberKnife Z-1023-2025 | Jan 2025 snap-ring recall | Confirmed | High |
| Naeotom Alpha class | Feb 2025 | Confirmed (K243523, 12 Feb 2025) | High |
| Ekso Indego K1007 | ~$91,032, Apr 2024 | Confirmed ($91,031.93) | High |
| Crossref DOI sweep | 39/41 extractable DOIs | 38 consistent; 1 wrong (MERIT-HF); 2 parenthetical DOIs skipped | High |

---

## 5. Structural inconsistencies

| # | Issue | Where | Severity |
|---:|---|---|---|
| S1 | 14A filed under "Term 3" heading though it is a Term 2 module | `COURSE_OUTLINE` §4 | High |
| S2 | Program maps omit 01A/14A (`00–06`, `07–14`) | `COURSE_OUTLINE` §1; `00` §4.1 | High |
| S3 | 01A/14A absent from every track's required list despite "Common core"/"MedTech core" | `COURSE_OUTLINE` §1.2; `00` §4.4 | High |
| S4 | "Recommended electives" are subsets of "Required" in all three tracks (17,18 / 05,18 / 15,16) | `COURSE_OUTLINE` §1.2; `00` §4.4 | Medium |
| S5 | 690–730 h vs 12–15 h/wk × 36 weeks (432–540 h); 00 says 12–20 h/wk | README §3; `00` §4.3 | Critical |
| S6 | Term 3 = 10 weeks but contains 208–248 h plus a 10–12-week capstone | README, outline, `19` §4 | Critical |
| S7 | Module 21 "required before the capstone" yet sequenced after Module 19; footer chain 20→21→19 is circular | `21` header/footer; README §12 | High |
| S8 | Changelog says module count corrected to 23; package has 25 | `CHANGELOG` line 31 | Medium |
| S9 | Module 21 titled "Datasets, Benchmarks, and Tool Stacks" in-file vs "Datasets, Tools, and Benchmarks" everywhere else | `21` L1 |Low |
| S10 | Heading level error: `## 9.3` inside section 9 | `14` L270 | Low |
| S11 | Template subheadings rendered as H2 (`## Decision`, `## Data`, `## Model details` …), breaking the outline | `00`, `18`, `19`, `20` | Medium |
| S12 | Exactly one internal Markdown link in 13,000+ lines; cross-refs and "Next" are plain text | package-wide | Medium |
| S13 | Duplicate lab: 11.5 "Edge deployment" and 14A.3 "TinyML quantization" are the same exercise; 11 is a prerequisite of 14A | `11` §12, `14A` §11 | Medium |
| S14 | Overlapping content: sampling/aliasing/filter tables repeated in 01A §8 and 11 §4 | `01A`, `11` | Low |
| S15 | Module 21 catalogue omits every dataset introduced by 14A and briefs 38–41 | `21` | High |
| S16 | Spec line targets missed: 00 536 (target 420–520), 15 368 (450–550), 20 292 (400–500), 22 270 (400–600) | `_course_spec.md` table | Medium |
| S17 | README entry points and self-study path omit 01A/14A | README §2, §12 | High |
| S18 | Dependency map omits 11 as a prerequisite of 14A | `00` §4.2 | Low |
| S19 | `_research/link_verification_summary.md` is listed in README package contents but was created mid-audit; other research files are not listed | README §11 | Low |

---

## 6. Factual red flags

| # | Claim | Location | Problem | Suggested action |
|---:|---|---|---|---|
| F1 | "median final denial rates 2.5% → 2.7%; net revenue leakage $38.6B → $48.4B; AHA ~$18B" | `08` L274 | Specific, load-bearing numbers sourced only to `hfma.org` and `aha.org` homepages, with no report title, year or URL | Cite the actual Kaufman Hall / HFMA / AHA report with title and date, or label as industry estimate |
| F2 | "software issues dominated" AI device recalls | `18` L236; `_research/industry_brief` | The cited JAMA Health Forum study's top category is diagnostic/measurement error (109 recalls), not "software" | Rewrite as "diagnostic/measurement errors were the most common cause (109 of 182 recalls)" |
| F3 | "A review of 903 AI-enabled **imaging** devices" | `18` L224 | The study covers all AI-enabled devices (76.6% radiology), not imaging only | Change to "903 FDA-authorised AI-enabled devices (76.6% radiology)" |
| F4 | "NHS England did not proceed to a pilot" placed after 2025/2026 results | `21` L349 | The decision was made in 2024 on first-year data, not as a consequence of the 2026 primary results | Separate the 2024 pilot decision from the 2026 trial result |
| F5 | "By mid-2025 >75 AI-designed molecules … up from none in 2020" | `13` L252 | Supported by the cited *Pharmacological Reviews* perspective, but the module cites only the journal and a company release; the "75" is a review estimate, not a registry count | Attribute explicitly to the review and note it is an estimate |
| F6 | Apple hypertension case presents sensitivity 41.2% without interpretation | `11` L339 | Accurate but incomplete: 41.2% sensitivity means ~59% of hypertensive users are not notified; the 2026 JAMA/AHA critiques make this the central point | Add the false-negative implication and the PPV/NPV context |
| F7 | "FDA has authorized 1,600+ … ~75% radiology" | README §3 | Correct; "authorized" is used properly (not "approved") | None |
| F8 | "1,356 through September 2025" | `_research/industry_brief` §4 | Consistent with FDA's growth curve, but no URL | Add the FDA dataset URL and date |
| F9 | GRAIL "publication venue verify" in the brief but asserted in `21` | `_research` vs `21` | The uncertainty marker was dropped when the fact moved into the module | Carry uncertainty markers into modules or resolve them |
| F10 | MERIT-HF DOI | `04` (twice) | Wrong DOI resolves to nothing | Fix (C5) |
| F11 | ISO 10993-1 URL | `14A` L951 | Malformed catalogue ID | Fix (C9) |
| F12 | "~6,425 teams" RSNA 2026 | `19` | Unverified; presented as a fact in the module table | Mark unverified or remove |
| F13 | "Naeotom Alpha class received FDA clearance in February 2025" | `14A` L188 | Correct (K243523, 12 Feb 2025) but easily confused with the 2021 original clearance; the sentence reads as if photon-counting CT was first cleared in 2025 | Clarify "the Alpha.Peak/Pro/Prime class" |
| F14 | Denial/leakage figures also appear in README industry signals | README §3.1 | Same unsourced statistic propagated to the front page | Fix once at source (F1) |

---

## 7. Coverage matrix for the requested topics

Status: **Covered** = dedicated section + lab or worked example; **Partial** = named but shallow; **Missing** = absent.

| Topic | Status | Where | Depth | Residual gap |
|---|---|---|---|---|
| Units and metrology | Covered | `01A` §5; Lab 01A.1 | Strong: SI, conversions, significant figures, error gallery | No GUM uncertainty-budget exercise |
| Measurement error | Covered | `01A` §5.4; Lab 01A.2 (Bland-Altman) | Strong conceptual | No misclassification/regression-dilution lab |
| Sensor physics | Covered | `14A` §4 (electrodes, PPG/SpO2, pressure, thermal, IMU, chemical, ultrasound) | Strong survey | Depth per modality is 2–5 lines; no derivation |
| Embedded systems | Covered | `14A` §5.4–5.5; Lab 14A.3 | Medium | No RTOS/firmware lab; no OTA exercise |
| Chips / semiconductors | Covered | `14A` §5.1–5.3, §5.8 | Medium | No process/qualification detail; supply-chain is a list |
| Firmware | Covered | `14A` §5.5 | Medium | Secure boot/SBOM discussed, not practised |
| Electrical safety | Covered | `14A` §5.6; `01A` §8.4 | Medium | One page; no 60601-1 test-category walkthrough |
| EMC | Partial | `14A` §5.6 (one paragraph) | Shallow | No IEC 60601-1-2 categories, no pre-compliance lab |
| Robotics kinematics/control/safety | Covered | `14A` §7.5–7.8; Labs 14A.4–14A.5 | Strong intro (FK/IK, Jacobian, singularity, PID/impedance, registration, standards) | No ROS 2/Gazebo lab; no control-loop stability exercise |
| Human–robot interaction | Partial | `14A` §7.11 (10 bullets) | Shallow | No trust/SART/workload method, no usability lab |
| Device lifecycle | Partial | `14A` §5.5, §5.8; `14` §8; `18` §8 | Medium | No DHF/DMR, design transfer, process validation, CAPA, UDI, recall execution |
| Clinical basics | Covered | `01` + `01A` §6–7 | Strong for a data-science audience | Pathology/histology, pediatrics/OB, geriatrics/palliative, lab operations |
| Engineering basics | Partial | `01A` §8; `03`; `14A` | Good software/signal basics | Reliability engineering, FMEA, Gauge R&R, systems engineering |
| Regulatory topics | Covered | `14`, `14A`, `16`, `18` | Very strong for software/SaMD/AI | UDI, recall/MAUDE-AEMS workflow, design-control artifact mapping, NIST AI RMF, ISO 42001 |
| Datasets | Covered | `21` (25 sections, 7 labs) | Very strong for EHR/imaging/signals/omics | No surgical/robotics, CGM, IMU/gait, device-log or semiconductor data |
| Labs | Covered | 4–7 per module, 7 in 14A | Strong; acceptance criteria present | Duplicate 11.5/14A.3; no ROS, firmware, EMC or reliability lab |
| Assessment components | Partial | Rubrics in every module; capstone rubric | Strong per-module | Midterm dossier (20%) and track playbook (15%) undefined; no individual technical exam; no case-critique rubric |

**Direct answers to the explicit checklist:** units/metrology — yes; measurement error — yes; sensor physics — yes; embedded systems — yes; chips — yes; firmware — yes; electrical safety — yes; EMC — only nominally; robotics kinematics/control/safety — yes; human–robot interaction — only nominally; device lifecycle — partially.

---

## 8. Suggested new module outlines

The originally missing hardware and foundations modules now exist as 14A and 01A. The outlines below are **gap-filling addenda**, not replacements: they cover the residual lifecycle/reliability/HRI and measurement-science depth identified above. Each is sized to 20–24 hours so it can be added without worsening the schedule problem — ideally by trimming Module 15 and Module 17 rather than extending the programme.

### Module 14B — Device Lifecycle, Reliability, and Human Factors (24 hours)

*Prerequisites: 14, 14A. Core artifact: a design-history and post-market evidence file for a fictional device.*

1. Design controls end to end: user needs → design inputs → outputs → review → verification → validation → transfer.
2. The Design History File and Device Master Record: what goes in, who owns it, how it is audited.
3. Design transfer to manufacturing: process validation (IQ/OQ/PQ), first-article inspection, scale-up.
4. Supplier qualification, incoming inspection, and supplier change control.
5. Reliability engineering: FMEA/FMECA, fault-tree analysis, derating, MTBF, reliability block diagrams.
6. Accelerated and environmental testing: HALT/HAST, drop, vibration, thermal cycling, shelf life.
7. Sterilization validation (ISO 11135, 11137, 17665) and biocompatibility (ISO 10993-1) as evidence, not checkboxes.
8. Human factors engineering: use specification, task analysis, use-related risk, formative and summative evaluation (IEC 62366-1).
9. Human–robot interaction: trust calibration, situation awareness, workload (NASA-TLX/SART), shared control, handover, team communication.
10. Complaint handling, MDR/vigilance, recalls, CAPA, and UDI: the post-market data pipeline.
11. Post-market surveillance and PMCF: designing registries and real-world device performance studies.
12. Obsolescence, end-of-life, service models, and the total cost of ownership.
13. Metrics that matter: complaint rate, recall rate, time-to-correction, service uptime, complaint-to-CAPA cycle time.
14. Lab: DHF traceability audit against a 510(k) summary.
15. Lab: FMEA plus a summative usability test on a device interface, with a use-error taxonomy.

### Module 01B — Measurement Science and Assumption Testing (20 hours)

*Prerequisites: 01A. Core artifact: a measurement-uncertainty and bias-analysis report for one variable in a real study.*

1. Metrology foundations: traceability chain, calibration hierarchy, reference materials, SI redefinition.
2. The GUM uncertainty framework: Type A and Type B evaluation, combined and expanded uncertainty, coverage factors.
3. Accuracy, precision, trueness, repeatability, reproducibility; ICC and Bland-Altman with limits of agreement.
4. Gauge R&R and variance-component analysis for measurement systems.
5. Classical vs Berkson measurement-error models; regression dilution and correction.
6. Misclassification of labels and exposures: differential vs non-differential, sensitivity/specificity of a label, PPV of a phenotype.
7. Limits of detection, censoring, and rounding as measurement processes.
8. Missingness as a measurement process: why "not tested" is informative.
9. Assumption taxonomy: statistical, causal, ML, clinical, device, economic, ethical (extends the 01A ledger).
10. Testability and falsification: which assumptions can be checked, which must be bounded, which must be argued.
11. Sensitivity analysis toolkit: probabilistic bias analysis, E-values, tipping-point analysis, negative controls.
12. External validity: transportability, covariate shift, site effects, and why random splits mislead.
13. Benchmark validity: construct validity, contamination, leaderboard overfitting, and "what is this metric a proxy for?"
14. Lab: full uncertainty budget for a blood-pressure or SpO2 measurement.
15. Lab: bias analysis for a misclassified sepsis label, with a tipping-point statement.

---

## 9. What is already strong (brief)

- **Decision-first pedagogy.** Every module opens with an industry question, a decision owner, and the cost of error. This is rare and is the package's main differentiator.
- **Regulatory depth.** Modules 14 and 14A are better than most professional short courses: definitions, pathways, GMLP/PCCP, QMS, EU MDR/IVDR/AI Act, EHDS, HTI-1, CMS-0057-F, and now IEC 60601/80601/10993 and ISO 14971. Claims are mostly dated and correctly characterised as final/draft/proposed.
- **Dataset realism.** Module 21 is the strongest section of the package: access mechanics, licence vs ethics, per-dataset pitfalls, and explicit "do not use this for that claim" guidance. The field notes on MIMIC/eICU/UK Biobank/All of Us are accurate and unusually candid.
- **Capstone design.** 41 briefs with decision, data, metric, regulatory angle and difficulty; honest labelling of concluded/closed/announced/unverified competitions; an explicit refusal to invent winners; a leakage and validation playbook.
- **Measurement bias and equity.** Pulse-oximetry and PPG skin-tone bias run through Modules 01, 11, 14A and 16 with primary sources, not as a diversity appendix.
- **Assessment rubrics.** Every taught module has a weighted rubric with novice/competent/expert descriptors, and most sum to 100.
- **The new 01A.** A genuinely useful foundations module: SI units, clinical conversions, PK/PD, epidemiology, an assumptions ledger, and labs that catch real errors (unit mix-ups, aliasing, NNT). It is the kind of module most programmes assume students already had.
- **The new 14A.** Covers the full hardware stack at the right altitude for data scientists, with correct kinematics, credible standards mapping, and device claims that — in the 12 cases I checked — were accurate to the K/DEN number and date.
- **Verification infrastructure.** Keeping `link_check.tsv`, a post-fix pass, a DOI TSV and a written method note is good practice; the flaws are in the interpretation, not the existence, of the evidence.

---

## 10. Method, limitations, and uncertainty

**What was checked directly:** file structure and headings; line counts against spec; internal link resolution; duplicate paragraphs; rubric sums; module cross-references; 41 DOIs via Crossref; 35+ sources via direct fetch or search; schedule arithmetic; 12 device claims in Module 14A.

**What was not checked:** the full text of every module (headings plus targeted sections were read); the content of paywalled papers beyond metadata; the exact text of the EU OJ for Regulation 2026/1744 (secondary legal sources only); every one of the 375 URLs in the post-fix pass; and the ~100 device/recall identifiers in 14A beyond the 12 sampled.

**Known uncertainties:**
- The package was being edited during the audit; line counts and some structural findings may already be stale. Re-hash before acting.
- RSNA 2026 team count, Rosa K251314, EPIONE Z-1927-2025, MIRA Z-1905-2025 and the StealthStation recall URL remain unverified.
- The MDCG 2025-6 PDF exists but its text could not be extracted; the characterisation of its content is taken from the course and the Commission page, not from the document itself.
- "NHS England did not proceed to a pilot" is factually true but refers to a 2024 decision; the phrasing in Module 21 implies a later causal link.

**Recommended next step:** fix C1–C5 and C7–C10, re-run the link checker with a browser-based secondary check for every FDA/FTC 404, and re-freeze the package under a version tag before any further content is added. The package is close to excellent; the remaining risks are architectural and editorial, not substantive.
