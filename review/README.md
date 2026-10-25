# External Review Workspace

This folder is a **template** for the private review of the course. It is not a record of completed review. Do not put reviewer personal data or confidential findings in a public repository.

## Contents to create

| File | Purpose |
|---|---|
| `membership.md` | Reviewer names, roles, qualifications, conflicts |
| `coi/` | Signed conflict-of-interest disclosures |
| `package/` | Frozen review package with version tag |
| `findings/` | Module and claim findings |
| `signoff.md` | Final reviewed/approved/validated statement |
| `pilot/` | Pilot protocol, consent, data, report |

## Minimum reviewer set

- Chair
- Clinician
- Regulatory/quality lead
- Hardware/robotics engineer
- Privacy/ethics reviewer
- Patient/community representative (recommended)
- Learner representative (recommended)
- Legal counsel (advisory)

## Freeze the package

Before review:

1. Tag the version (e.g., `2026.2-review`).
2. Record file hashes.
3. Freeze the change log.
4. Send the package and rubric.
5. Set a review window and a response deadline.

## Finding severity

- **Critical:** clinical harm, illegal data use, false regulatory claim, unsafe lab.
- **Major:** material inaccuracy, missing evidence, unteachable section.
- **Minor:** clarity, citation, consistency.

## Decision language

Do not use “validated” until the pilot is complete and critical/major findings are closed. Use:

- “Reviewed” for a read.
- “Approved for pilot” after critical findings close.
- “Approved for delivery” after pilot corrections.
- “Validated” only after external sign-off and pilot evidence.

See `../GOVERNANCE_AND_EXTERNAL_REVIEW.md` for the full process.
