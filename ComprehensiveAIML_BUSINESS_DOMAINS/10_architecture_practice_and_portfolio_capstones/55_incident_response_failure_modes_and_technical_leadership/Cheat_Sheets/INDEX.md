# Module 55 Cheat Sheets — Incident Response, Failure Modes and Technical Leadership

## 1. Overview
This collection will teach incident command, data/model and LLM/agent failures, resilience drills, and prevention. Existing streaming cases are illustrative drafts, not empirical incident reports. No individual cheat sheet or Common Issues artifact exists.

## 2. Topic ladder
| Level | Topic | Outcome | Prerequisite | Status |
|---|---|---|---|---|
| Foundation | incident-response playbook | establish roles, evidence, containment, and decision clock | systems basics | planned |
| Core | failure-mode catalog | distinguish signals, causes, mitigations, and verification | playbook | planned |
| Core | postmortem template | derive prevention from an invariant failure | playbook | planned |
| Advanced | IR005 release bundle | promote/recover compatible model/prompt/tool/index/policy bundle | versioning | planned |
| Advanced | IR028 bounded parsing | contain collision/regex CPU amplification | DSA/security | planned |
| Advanced | IR029 bounded graph traversal | return explicit partial results | graphs | planned |
| Advanced | IR030 invariant failures | replay diverse algorithmic failure mechanisms | DSA | planned |
| Final | Common Issues | >=25 complete issues | all sheets | planned; no file |

## 3. Quick topic lookup
| Question | File | Use condition | Warning |
|---|---|---|---|
| Who decides containment? | planned playbook | active incident | restart is not diagnosis |
| Why did CPU spike? | planned IR028 | adversarial parsing | size limit alone may fail |
| Can traversal return partial output? | planned IR029 | bounded graph expansion | partial must be labeled |

## 4. Reading paths — three options
**Beginner:** playbook → failure catalog → postmortem. **Practitioner:** beginner path plus one existing streaming case and planned resilience drills. **Advanced:** all planned reference additions, recovery rehearsals, and the final issue artifact.

## 5. Prerequisite skills
Timestamped logs/metrics, queues and backpressure, data/model versioning, idempotency, complexity analysis, and operational communication. Source incident narratives referenced in the module are unavailable or reference-only.

## 6. Exclusions and handoffs
This is not an on-call authorization, security/legal policy, vendor runbook, or evidence of a real organization’s incident. Product-specific SLOs, safety decisions, and privacy notifications need local owners and primary sources.

## 7. Master symbol glossary
| Term | Meaning | Units/type | First planned use |
|---|---|---|---|
| `lambda`, `mu` | ingress/service rate | events/s | capacity diagnosis |
| `B` | backlog | events | recovery calculation |
| RTO | proposed recovery-time objective | time | playbook |
| invariant | condition preserved across allowed transitions | predicate | postmortem |

## 8. Quick Reference Card directory
No cards exist; every future sheet must finish with a card. The two case cards do not fulfill this directory.

## 9. Common Issues locator
No final issue artifact exists. It must retain all 12 headings and contain at least 25 distinct complete entries with stable ID, category, provenance, causes, diagnostic signal, mitigation risk, correction, prevention, and verification.

## 10. Sources, versions and maintenance
| Source | Scope | Limitation | Trigger |
|---|---|---|---|
| [module index](../INDEX.md) | original topics and plans | planned curriculum | index change |
| [stream incidents brief](../../../00_SOURCE_COVERAGE_EDGE_INDUSTRY.md#stream-incidents) | source scope | titles are not source bodies | body/version change |
| [Lang issues brief](../../../00_SOURCE_COVERAGE_AGENTS.md#lang-issues) | issue syllabus | not completed issues | source review |

## 11. Collection acceptance checklist
- [ ] Each topic is authored as a 12-section substantive sheet.
- [ ] All calculations/traces are checked and evidence boundaries honest.
- [ ] Final >=25-issue artifact exists.
- [ ] Primary sources, links, and changing dependency semantics are reviewed.
