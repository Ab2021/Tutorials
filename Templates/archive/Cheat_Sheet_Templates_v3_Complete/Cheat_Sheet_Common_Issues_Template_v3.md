# [Collection] — Common Issues (`NN_<N>_Common_Issues.md`)

> **Purpose:** Final file in a cheat-sheet collection. Capture at least **25 distinct, diagnosable issues**; use 50 for a comprehensive collection.
> **Status:** Blank template. The rows below are **placeholders, not completed issues**.

## Issue provenance standard

Each issue belongs to one original Tip 8 category: `VIOLATED_ASSUMPTION`, `NUMERICAL_EDGE`, `SCALE_FAILURE`, or `CONCEPT_CONFUSION`. Record provenance as `OBSERVED` (real documented failure), `REPRODUCED` (test confirmed by the author) or `HYPOTHETICAL` (explicit teaching example). Don't invent real-world incident claims. Use a reproduction if reproducible; otherwise give a mathematical diagnostic check and a bounded fix.

## Issue index

| ID | Category | Symptom | Root cause | Diagnostic / expected signal | Fix | Provenance | Related file |
|---|---|---|---|---|---|---|---|
| ISSUE-001 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-002 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-003 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-004 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-005 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-006 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-007 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-008 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-009 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-010 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-011 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-012 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-013 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-014 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-015 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-016 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-017 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-018 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-019 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-020 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-021 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-022 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-023 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-024 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |
| ISSUE-025 | [one of 4] | [observable] | [underlying mechanism] | [check and expected signal] | [corrective action] | [OBSERVED / REPRODUCED / HYPOTHETICAL] | [NN_Topic.md] |


## Detailed issue format (repeat for each filled row)

### ISSUE-001 — [Concise symptom]

- **Trigger / environment:** [inputs, versions and scale]
- **Symptoms:** [what a practitioner sees]
- **Root cause:** [what actually fails in the model, code or assumptions]
- **Diagnostic:** [step-by-step check, expected output and what a failure proves]
- **Correct action:** [remediation]
- **Prevention:** [test, invariant, alert or validation gate]
- **Provenance:** [OBSERVED / REPRODUCED / HYPOTHETICAL with pointer to evidence]
- **Related topics:** [relative link to a collection sheet]
- **Open limitation:** [when the fix does not apply]

## Publication checks

- [ ] At least 25 rows have genuine, distinct, non-placeholder content.
- [ ] Every issue has a diagnostic that distinguishes its stated cause from competing causes.
- [ ] Every fix addresses the cause, not only the symptom.
- [ ] All empirical incident claims carry evidence or are clearly marked hypothetical.
- [ ] The file is the final numbered file and INDEX links point to it.
