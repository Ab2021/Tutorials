# Changelog

All substantive changes to the course package are recorded here.

## 2026.2.1 — Reference production stacks

### Added
- `labs/cloud/`: Terraform modules for VPC, S3/KMS, ECR, and EKS.
- `labs/cloud/k8s/`: namespace, model config, deployment, service, ingress, HPA, PDB, network policy, service account.
- `labs/cloud/kafka/`: Redpanda compose, Pydantic schema, idempotent producer, deduplicating consumer, DLQ, sample events.
- `labs/mlops/`: FastAPI service, model artifact, tests, Dockerfile, CI, K8s manifests, Prometheus rules, drift script, runbook, model card.
- Module 15 §9A and Module 26 §9A now walk through the reference stacks.

### Verified
- MLOps tests: 5 passed.
- Python modules compile; YAML manifests parse.
- Terraform and Docker/Kubernetes were not executed in the authoring environment; verification limits are documented.

## 2026.2 — P0 validation infrastructure and P1 depth

### Added
- `COMPETENCY_MAP.md`: outcome-to-module/lab/assessment crosswalk.
- `GOVERNANCE_AND_EXTERNAL_REVIEW.md`: reviewer terms, COI, rubric, sign-off.
- `REGULATORY_VERIFICATION_LOG.md`: 54 tiered claims with desk-check status.
- `INSTRUCTOR_KIT.md`, `GRADING_CALIBRATION.md`, `PILOT_PROTOCOL.md`.
- `labs/`: five starter notebooks, requirements, and two sample instructor solutions.
- `VALIDATION_STATUS.md`: claims boundary and exit criteria.
- Module 23: specialty electives (oncology, cardiology, pathology, critical care, pediatrics/OB, public health).
- Module 24: hardware, firmware, EMC, HIL, and robotics labs.
- Module 25: trial operations, HTA/market access, global regulation.
- Module 26: cloud, Kubernetes, and streaming production labs.

### Changed
- Package now 30 modules and >20,000 lines.
- Advanced electives are optional; the 40-week core remains intact.
- Claims language now distinguishes drafted, desk-checked, reviewed, and validated.

### Still pending
- External reviewers have not been convened.
- No regulatory claim has been professionally signed off.
- No learner pilot has been run.
- The course must not be described as validated yet.

## 2026.1.1 — Audit and hardware expansion

### Added
- Module 01A: foundations, units, measurement, clinical/biology/engineering basics, and the assumptions ledger.
- Module 14A: medical hardware, sensors, analog front ends, semiconductors, firmware, electrical safety, EMC, device lifecycle/reliability, and robotics.
- Module 14B: distillation, quantization, pruning, inference runtimes, target-hardware validation, and PCCP-aligned change control.
- 44 capstone briefs, including hardware/sensor/robotics and efficient-AI/on-device briefs.
- Defined the midterm evidence dossier, track playbook, and case-critique assessments with rubrics.
- Surgical/robotic, CGM, IMU, and device post-market datasets in Module 21.
- Digital twins, population health, pathology/lab medicine, special populations, and device lifecycle material.
- Regulatory watchlist additions: HIPAA Security NPRM, HTI-2/TEFCA, information blocking, UDI/EUDAMED, NIST AI RMF, ISO/IEC 42001, AEMS.

### Changed
- Schedule corrected to 40 weeks at 17–19 hours/week; Term 3 is weeks 27–40 and can hold the 12-week capstone.
- 01A and 14A integrated into program maps, term tables, tracks, prerequisites, and capstone briefs.
- Module 15 expanded with device-fleet operations, drift taxonomy, delayed labels, post-mortem, cost model, and PCCP change control.
- Module 21 updated to All of Us CDR v9 and GRAIL 2026 results; AEMS naming applied.
- MERIT-HF and operating-room DOIs corrected; moved FDA/CMS/dataset links replaced.
- Module 01 book-to-module map added.

### Verified
- 46/46 DOI citations resolve in Crossref.
- 385 current user-facing URLs checked; 267 reachable, 83 bot/paywall, 16 FDA/FTC anti-bot, 8 network/TLS errors.
- No known non-FDA/FTC dead URL in the current set.
- Internal links and Markdown tables checked.

### Known limitations
- FDA/FTC automated checks remain bot-blocked; browser verification is still required before formal publication.
- Some device K/recall identifiers remain marked `[VERIFY]`.
- Regulatory and market facts are current to 24 October 2026.

## 2026.1 — 24 October 2026

### Added

- Complete 26 core/advanced modules covering clinical foundations, units/measurement/assumptions, data engineering, statistics, causal inference, machine learning, deep learning, LLMs, EHR/claims, clinical NLP, imaging, signals, omics, biopharma, regulation, medical hardware/sensors/chips/robotics, efficient AI/distillation/quantization/on-device inference, MLOps, privacy, product, quality, capstone, careers, datasets, and sources.
- Three tracks: Health Systems Analytics; Biopharma & Translational; MedTech, Sensors & Imaging.
- 40-week program map and 12-week capstone.
- 44 capstone/hackathon briefs, including seven hardware/sensor/robotics briefs and three efficient-AI/on-device briefs.
- Module 01A: mathematical, biological, clinical, engineering, and data foundations; unit/metrology reference; assumptions ledger.
- Module 14A: device signal chains, sensors, analog front ends, semiconductors, firmware, electrical safety, and medical robotics.
- Industry research source pack (2025–2026).
- Hackathon and competition research source pack (2023–2026).
- Module-level labs, assignments, rubrics, interview questions, and source lists.
- 2026 regulatory watchlist covering FDA PCCP, QMSR, CSA, generative AI devices, LDT vacatur, EU AI Act timing, EHDS, HTI-1, and CMS prior authorization.
- Current industry anchors for providers, pharma, devices, wearables, diagnostics, payers, and public health.

### Changed

- Replaced uncertain DOI citations for two 2025 JAMA articles with stable journal URLs.
- Added source verification rules and `[VERIFY]` guidance.
- Corrected the FDA PCCP source date to note the 2024 issuance and 2025 reissue.
- Added the 2025 single-cell foundation-model benchmark caveat.
- Added the 2025 Apple hypertension notification case.
- Added the 2025 AI-designed molecule clinical-trial case.
- Added a 2026 revenue-cycle leakage and denial-analytics case.
- Added primary-source links for QMSR, computer software assurance, LDTs, generative AI devices, HTI-1, CMS prior authorization, MDCG 2025-6, and EHDS.
- Core module range is 00–22 (23 modules); Modules 01A, 14A, and 14B bring the package total to 26.
- Corrected dataset and benchmark source URLs for FoMoH, MedAgentBench, and MedHELM.
- Rechecked 394 external URLs and 46 DOIs; corrected moved FDA pages, AEMS/FAERS naming, MERIT-HF and operating-room DOIs, CMS links, WESAD, PANDA, DECIDE-AI, and All of Us/GRAIL dates.
- Added 40-week schedule consistency, 26-module maps at that time, and defined the midterm dossier, track playbook, and case-critique assessments.

### Known limitations

- Regulatory facts change frequently. Re-verify before use.
- Some datasets require credentialing and cannot be redistributed.
- Several 2026 competitions are announced, closed, or pending; status is labeled in Module 19.
- Market and salary figures are dated snapshots, not forecasts.
- The course is educational and not legal, medical, or regulatory advice.

### Next review

- Regulatory review: April 2027.
- Full curriculum review: October 2027.
- Dataset access review: every 6 months.
- Competition status review: quarterly.
