# Module 24 — Recommendation Systems and Personalization

> **Part of:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026), expanded edition.
> **Survey file:** [`../../../course/24_recommendation_systems_personalization.md`](../../../course/24_recommendation_systems_personalization.md)
> **Folder:** `modules/03_commerce/24_recommendation_systems_personalization/`
> **Version:** 2026.5

---

## 1. Module purpose

This module covers retrieval and ranking for supply chain-aware recommendations: baselines, collaborative filtering, two-tower models, evaluation, feedback loops, and inventory constraints.

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 06, 07, 17 |
| Unlocks | 25 |
| Assessment | Ranking evaluation and re-ranking policy (Week 30) |
| Survey file | `24_recommendation_systems_personalization.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_recommendation-tasks.md`](01_recommendation-tasks.md) | Recommendation Tasks | Related items, substitutions, complements, and next-best-offer. | 1,200–1,800 | planned |
| 2 | [`02_popularity-content-baselines.md`](02_popularity-content-baselines.md) | Popularity and Content-Based Baselines | The baselines every recommender must beat. | 1,200–1,800 | planned |
| 3 | [`03_collaborative-filtering.md`](03_collaborative-filtering.md) | Collaborative Filtering and ALS | User-item structure, implicit feedback, and matrix factorization. | 1,200–1,800 | planned |
| 4 | [`04_two-tower-retrieval.md`](04_two-tower-retrieval.md) | Two-Tower Retrieval | Candidate generation at catalogue scale and the embedding contract. | 1,200–1,800 | planned |
| 5 | [`05_sequential-session.md`](05_sequential-session.md) | Sequential and Session Models | Intent within a session and the features that capture it. | 1,200–1,800 | planned |
| 6 | [`06_cold-start.md`](06_cold-start.md) | Cold Start | New users, new items, and the fallback ladder. | 1,200–1,800 | planned |
| 7 | [`07_offline-online-evaluation.md`](07_offline-online-evaluation.md) | Offline and Online Evaluation | Recall@k, NDCG, MAP, and the gap to business outcomes. | 1,200–1,800 | planned |
| 8 | [`08_feedback-loops.md`](08_feedback-loops.md) | Feedback Loops and Bias | Exposure bias, position bias, and the model that trains on its own choices. | 1,200–1,800 | planned |
| 9 | [`09_constraints-re-ranking.md`](09_constraints-re-ranking.md) | Business Constraints and Re-Ranking | Margin, diversity, availability, and policy constraints at serving time. | 1,200–1,800 | planned |
| 10 | [`10_inventory-aware-substitution.md`](10_inventory-aware-substitution.md) | Inventory-Aware Recommendation and Substitution | What to show when the first choice is unavailable. | 1,200–1,800 | planned |
| 11 | [`11_llm-privacy-fairness.md`](11_llm-privacy-fairness.md) | LLM Recommenders, Privacy, and Fairness | Generative ranking, consent, and distributional harm. | 1,200–1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Recommendation Tasks ->
02 Popularity and Content-Based Baselines ->
03 Collaborative Filtering and ALS ->
04 Two-Tower Retrieval ->
05 Sequential and Session Models ->
06 Cold Start ->
07 Offline and Online Evaluation ->
08 Feedback Loops and Bias ->
09 Business Constraints and Re-Ranking ->
10 Inventory-Aware Recommendation and Substitution ->
11 LLM Recommenders, Privacy, and Fairness
```

## 5. What this module does not cover

- It does not cover advertising auctions or large-scale feature-store engineering.
- Hands-on lab execution: see the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.

## 6. Status

| Status | Count |
|---|---:|
| planned | 11 |
| draft | 0 |
| complete | 0 |

Topic files are indexed but not yet written. Status moves from `planned` to `draft` to `complete`
as each file is produced at its target length. The survey file linked above is complete at survey
depth and can be used on its own.

## 7. Sources for this module

See section 14 of the survey file (`24_recommendation_systems_personalization.md`) and the per-topic source sections once written.
Evidence grading follows Module 28: A peer-reviewed or audited, B regulatory or official statistic,
C company case study or investor claim, D vendor marketing, E press summary or anecdote.
