# Claims, Fraud, Customer and Operations — Cheat Sheets

## 1. Overview
This collection will teach reusable, auditable methods for claims intake, extraction, triage, fraud investigation, customer contact, capacity, monitoring, and human review. It assumes insurance decision vocabulary and never treats a score, extraction, or graph connection as authority to deny a claim. No listed topic is authored in this collection; linked case material is separate evidence, not a substitute for a 12-section sheet.

## 2. Topic ladder
| Level | File / planned topic | Testable outcome | Prerequisites | Status |
|---|---|---|---|---|
| Foundation | Claims journey and automation controls | Map decision-time evidence, action, owner, queue, and recovery state. | Claim lifecycle basics. | Planned. |
| Core | Document extraction and evidence validation | Compute field and claim completeness; route unsupported fields. | Claims journey; schema concepts. | Planned; covers PLAN-2026-IR011/014. |
| Core | Triage, severity, and human review | Select a threshold subject to review capacity and error cost. | Conditional probability and queue basics. | Planned. |
| Core | Fraud investigation and graph enrichment | Evaluate incremental value at fixed investigator capacity. | Temporal splits and delayed labels. | Planned; covers PLAN-2026-IR036. |
| Core | Customer communication and call analytics | Preserve evidence spans and distinguish ASR, role, and extraction error. | Consent/authorization boundary; evaluation basics. | Planned; covers PLAN-2026-IR015/016. |
| Advanced | Operational capacity, monitoring, and release recovery | Trace idempotent processing, dead-letter replay, and compatible release recovery. | Above controls; state/version concepts. | Planned; covers PLAN-2026-IR005/031. |
| Final | Common Issues | Diagnose at least 25 complete distinct failures. | All completed sheets. | Planned. |

## 3. Quick topic lookup
| Question | Planned teaching home | Use condition | Failure warning |
|---|---|---|---|
| Can this field bypass review? | Document extraction and evidence validation §7–8 | Schema and source evidence are available. | Valid syntax is not business correctness. |
| Does graph enrichment help investigators? | Fraud investigation §5, §9 | Investigation capacity is fixed and labels are mature enough. | Connectivity is not proof of fraud. |
| Is a retry safe? | Operational capacity §4, §8 | Side effects have idempotency keys and durable state. | Replaying an unkeyed action can duplicate it. |
| Are coaching themes trustworthy? | Customer communication §9 | Speaker/evidence quality is measured. | Transcript match does not prove audio or role fidelity. |

## 4. Reading paths
- **Beginner:** Claims journey → document extraction → triage/human review.
- **Practitioner:** Beginner path → fraud investigation → capacity/monitoring → customer communication.
- **Advanced:** Practitioner path → compatible release/recovery, authorization-aware hierarchy retrieval, and final Common Issues.

## 5. Prerequisite skills
Define a claim and field unit, preserve event versus processing time, calculate precision/recall and rates, interpret queues, and distinguish a source document from a derived extraction. The existing [document extraction case](../Case_Studies/Medical_Claims_Document_Extraction_MediConCen.md) is a published-source critique and illustrative design exercise; it supplies no general benchmark.

## 6. Exclusions and handoffs
This collection does not make coverage, medical, legal, fraud, or employment determinations. Subrogation and recovery policy require relevant owners. Pricing and reserves hand off to `../../01_pricing_underwriting_and_reserving/`; model governance and climate hand off to `../../03_insurance_mlops_model_risk_and_climate/`.

## 7. Master symbol glossary
| Symbol / term | Meaning | Units/type | Domain / warning | Planned first use |
|---|---|---|---|---|
| Claim completeness | Required fields correct for one claim | proportion | Requires an explicit eligible field set | Document extraction |
| Review capacity | Cases an authorized team can complete per time window | claims/time | Not model throughput | Triage |
| Mature label | Outcome fixed enough for stated metric | status/event | Selective investigation labels are biased | Fraud investigation |
| Idempotency key | Stable identity for one intended side effect | key | Must survive retry and replay | Operational capacity |
| Evidence span | Source page/region supporting a claim | structured reference | Similar text alone is not evidence | Customer communication |

## 8. Quick Reference Card directory
No completed cards exist. Planned cards will cover extraction acceptance, capacity-constrained triage, graph-alert evaluation, evidence-grounded communication, and idempotent recovery. Cards become links only after their authored sheets exist.

## 9. Common Issues locator
No Common Issues file exists. The final artifact must retain 12 headings and contain at least 25 distinct, complete issues with category, provenance, competing causes, diagnostic signal, mitigation risk, correction, prevention, and verification.

## 10. Sources, versions and maintenance
| Commitment | Basis and limitation | Status | Recheck trigger |
|---|---|---|---|
| Original 12-topic module map | Local INDEX; original topic names only. | Planned. | Collection scope changes. |
| PLAN-2026-IR005/011/014/015/016/031/036 | Reference-informed planning briefs; referenced bodies are examples only. | Planned; no source-body reconstruction. | Reference correction or authored teaching scope. |
| MediConCen case | Public customer-story analysis with explicit evidence gaps. | Draft case; not a completed sheet. | Primary evaluation or source correction. |

## 11. Collection acceptance checklist
- [x] Original and reference-informed commitments are visible in the topic ladder.
- [x] Prerequisite order, handoffs, source limits, and planned issue target are stated.
- [ ] Every topic has an actual 12-section, 1,000-line sheet.
- [ ] Final Common Issues artifact contains at least 25 complete issues.
- [ ] Navigation, numerical examples, sources, and editorial status receive independent review.
