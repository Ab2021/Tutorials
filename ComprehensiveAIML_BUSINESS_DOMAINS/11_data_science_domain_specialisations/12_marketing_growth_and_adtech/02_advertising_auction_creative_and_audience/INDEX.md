# Advertising, Auction, Creative and Audience

> **Family:** [Marketing, Growth and AdTech](../INDEX.md)  
> **Purpose:** Ad delivery systems for targeting, retrieval, ranking, auctions, creative, pacing, safety, and platform operations.  
> **Company / context examples:** Retail media, publishers, ad platforms

## Detailed topic map

The source crosswalk has no reconstructed source body assigned to this module as
of 2026-11-09. These are original local-catalog commitments. They must be
taught as methods and illustrative exercises unless a primary source supports a
specific platform claim; a topic title is not an authored artifact.

| # | Topic | Testable outcome | Required predecessor | Teaching evidence state |
|---:|---|---|---|---|
| 1 | ad stack | Trace a request from eligible campaign to logged impression and identify the decision owner at each boundary. | Event and API basics. | planned |
| 2 | audience | Select a segment definition with eligibility, consent, reach, and exclusion semantics. | 1; privacy basics. | planned |
| 3 | retrieval | Compare candidate-generation recall, latency, and eligibility constraints. | 1–2; ranking basics. | planned |
| 4 | ranking | Specify a ranking objective, calibration check, and business guardrails. | 3; supervised learning basics. | planned |
| 5 | auctions | Compute a stated auction payment and distinguish mechanism from platform policy. | 4; probability and utility basics. | planned |
| 6 | bidding | Set a bid from value, probability, constraint, and uncertainty; reject invalid inputs. | 4–5. | planned |
| 7 | pacing | Derive spend error and choose a stable control response under a budget horizon. | 5–6; feedback basics. | planned |
| 8 | creative | Design a creative experiment that separates asset effect, audience mix, and delivery effects. | 2, 4; experiments. | planned |
| 9 | frequency | Apply a frequency cap with deduplication, horizon, and incremental-reach trade-offs. | 2, 7. | planned |
| 10 | brand safety | Define policy classifications, reviewer escalation, and false-positive cost. | 1; content-risk basics. | planned |
| 11 | fraud | Diagnose invalid-traffic signals without treating a proxy as proof of fraud. | 1; adversarial measurement basics. | planned |
| 12 | privacy | Choose a measurement/targeting design under stated consent, retention, and access constraints. | 2, 9–11. | planned |

**Completion boundary:** files in the applied collections, not this map, carry
the required derivations, checks, interview answers, and final cards.

## Applied collections

- [Case Studies](./Case_Studies/INDEX.md): decision-centred cases using the local v2 catalog and integrated outline standard.
- [Cheat Sheets](./Cheat_Sheets/INDEX.md): concise references for recurring design, analysis, and operating decisions.

## Scope boundary

Use public company material only where cited. Treat company-specific systems as unknown unless a primary or clearly labelled published source supports the claim. Learning cases may be simulated or illustrative, but cannot claim measured production impact.