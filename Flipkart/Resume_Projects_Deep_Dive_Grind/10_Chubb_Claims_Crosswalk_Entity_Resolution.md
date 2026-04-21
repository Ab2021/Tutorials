# 🔥 PROJECT GRIND: Claims Crosswalk (NLP Entity Resolution)
### Company: Chubb | Role: Senior Data Scientist II

> **Resume Bullet:** Built a Claims Crosswalk tool using advanced NLP and entity resolution techniques to intelligently link and reconcile data from multiple external sources, accelerating data engineering workflows for internal analytics projects.

---

## 🏗️ 1. PRODUCTION ARCHITECTURE

```
[Source 1: Internal Claims DB]
[Source 2: External Vendor Feed A]
[Source 3: External Vendor Feed B]
              │
              ▼
┌──────────────────────────────────┐
│  Schema Alignment & Mapping     │
│  (Standardize column names,     │
│   date formats, ID formats)     │
└──────────────┬───────────────────┘
               │
┌──────────────▼───────────────────┐
│  Entity Feature Engineering     │
│  - Name normalization           │
│  - Address geocoding/hash       │
│  - Phone/Email standardization  │
│  - Fuzzy key: Soundex + Metaphone│
└──────────────┬───────────────────┘
               │
┌──────────────▼───────────────────┐
│  Blocking (Reduce Comparisons)  │
│  - Block on ZIP + First 3 chars │
│  - Reduces O(n²) to manageable │
└──────────────┬───────────────────┘
               │
┌──────────────▼───────────────────┐
│  Pairwise Similarity Features   │
│  - Jaro-Winkler (names)         │
│  - Levenshtein (addresses)      │
│  - Exact match (SSN/PolicyNum)  │
│  - Date proximity (DOB/DOL)     │
└──────────────┬───────────────────┘
               │
┌──────────────▼───────────────────┐
│  Match Classifier (XGBoost)     │
│  - Match / Possible / No-Match  │
│  - Calibrated probabilities     │
└──────────────┬───────────────────┘
               │
      ┌────────┴────────┐
      │                 │
  Auto-Merge         Human Review
  (P > 0.95)        (0.6 < P < 0.95)
      │                 │
      └────────┬────────┘
               │
┌──────────────▼───────────────────┐
│  Golden Record Assembly         │
│  (Conflict resolution: newest   │
│   source wins for mutable       │
│   fields, union for immutable)  │
└──────────────────────────────────┘
```

---

## 🛠️ 2. PHASE-BY-PHASE DEEP DIVE & UNUSUAL EDGE CASES

### A. Data Ingestion & Schema Alignment
*   **What you did:** Mapped heterogeneous vendor schemas to a canonical internal schema. Handled dates in 15+ formats, addresses with/without suite numbers, names with/without middle initials.
*   **The "Unusual" Issue:** **"The Encoding Nightmare."** Vendor B sends data in Windows-1252 encoding; Vendor A uses UTF-8. Special characters in names (é, ñ, ü) were silently mangled during ingestion, causing downstream matching failures on perfectly identical records.
*   **The Fix:** Forced all ingest pipelines to decode to UTF-8 with error handling: `text.encode('utf-8', errors='replace').decode('utf-8')`. Added a data quality test that counts non-ASCII characters per vendor batch — if the count drops suddenly, it's an encoding regression.

### B. Matching & Classification
*   **What you did:** Built feature vectors for candidate pairs (Jaro-Winkler on name, date proximity on DOB, exact on policy number) and trained XGBoost to classify Match/No-Match.
*   **The "Unusual" Issue:** **"The Married Name Problem."** Claimant "Jane Smith" in Source 1 was the same person as "Jane Doe-Smith" or "Jane Doe" in Source 2 (name change after marriage/divorce). All string similarity metrics scored low.
*   **The Fix:** Added a secondary matching pass on invariant features: SSN hash (if available), DOB + Gender + ZIP combination. If these hard identifiers match, override the name-based similarity threshold entirely. This increased recall by 12%.

### C. Production & Quality Monitoring
*   **The "Unusual" Issue:** **"Match Drift."** New vendor feeds introduced new formatting conventions (e.g., switching from "Last, First" to "First Last"). The classifier's precision degraded silently because features shifted.
*   **The Fix:** Weekly precision monitoring by human-sampling 100 random auto-merged pairs and verifying. Alert on precision drop below 95%. Retrain the classifier quarterly with freshly labeled data.

---

## ⚔️ 3. FLIPKART ROUND-SPECIFIC GRINDING QUESTIONS

### 🔴 DDS (System Design)
1.  **"Flipkart acquires a company. You need to merge their 5M customer records with Flipkart's 200M user database. Design the entity resolution pipeline."**
    *   *Ans:* (1) Blocking on phone_hash + email_hash (exact). (2) For non-exact, TF-IDF on names + address embedding similarity. (3) Classifier with confidence tiers: auto-merge / human review / no-match. (4) Dedup within the acquired DB first (they may have internal dupes). (5) Monitor downstream impact on recommendation engines (sudden identity merges can confuse collaborative filtering).
2.  **"What data quality checks do you run before AND after entity resolution?"**
    *   *Ans:* Before: completeness (null rates per field), format validation (phone regex), encoding checks. After: merge rate sanity (if >50% merge, something is wrong), cluster size distribution (if one cluster has 10K records, it's a false transitive closure chain), downstream metric impact (did orders-per-user suddenly double?).

### 🔵 DMM (Mathematical Modeling)
1.  **"Prove mathematically that Jaro-Winkler similarity is bounded [0,1] and explain when it outperforms Levenshtein for name matching."**
2.  **"In your blocking step, you accept some false negatives (missed true pairs filtered out). How do you quantify the recall loss from blocking and decide the blocking key?"**
    *   *Ans:* Pairs Completeness (PC) = $\frac{|True\_Pairs \cap Candidate\_Pairs|}{|True\_Pairs|}$. Measure PC on a labeled sample. If PC < 0.95, the blocking key is too aggressive. Trade-off: looser blocking → higher PC but more computation. Use multiple blocking passes with different keys (union of candidate sets) to boost PC.

### 🟢 HO (Hands-On)
1.  **"Write a Python function that computes Jaro-Winkler similarity between two strings from scratch (no libraries)."**
2.  **"Write a PySpark pipeline that performs blocking on `first_3_chars_of_lastname + zip_code`, then computes pairwise string similarity features within each block."**

### 🟡 HM (Hiring Manager)
1.  **"Entity resolution errors can create terrible downstream consequences (merging two different people's insurance claims). How do you communicate the risk tolerance to non-technical stakeholders?"**
    *   *STAR:* Framed it as "We can auto-merge X% with 99% precision, and route Y% for human review. The business question is: what's the cost of a false merge vs. the cost of a missed merge?" Let the business define the threshold, not engineering.
