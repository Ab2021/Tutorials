# 🔥 PROJECT GRIND: NLP-based Plan Sponsor Entity Matching (TF-IDF + Spark MLlib)
### Company: EXL (CVS Health/Aetna) | Role: Assistant Manager → Manager

> **Resume Bullet:** Designed scalable string matching algorithm using TF-IDF and cosine similarity on Spark MLlib, achieving 75% accuracy matching 20K plan sponsors across 40K company name records — saving hundreds of hours of manual work.

---

## 🏗️ 1. PRODUCTION ARCHITECTURE

```
[Source A: 20K Plan Sponsor Names]     [Source B: 40K Company Records]
              │                                     │
              ▼                                     ▼
    ┌─────────────────────────────────────────────────────┐
    │         Text Normalization & Preprocessing          │
    │  (lowercase, remove Inc/LLC/Corp, standardize "&")  │
    └─────────────────────┬───────────────────────────────┘
                          │
              ┌───────────▼───────────────┐
              │  Spark MLlib TF-IDF       │
              │  Vectorization            │
              │  (Character n-grams 2-4)  │
              └───────────┬───────────────┘
                          │
              ┌───────────▼───────────────┐
              │  Blocking / LSH           │
              │  (Min-Hash + LSH to       │
              │   reduce O(n²) to O(n))   │
              └───────────┬───────────────┘
                          │
              ┌───────────▼───────────────┐
              │  Pairwise Cosine Sim      │
              │  (only within blocks)     │
              └───────────┬───────────────┘
                          │
              ┌───────────▼───────────────┐
              │  Match Classification     │
              │  (Random Forest on        │
              │   sim features + rules)   │
              └───────────┬───────────────┘
                          │
              ┌───────────▼───────────────┐
              │  Transitive Closure       │
              │  (A=B, B=C → A=C)         │
              │  Union-Find               │
              └───────────┬───────────────┘
                          │
              ▼ Final Matched Entity Table → Downstream Analytics
```

---

## 🛠️ 2. PHASE-BY-PHASE DEEP DIVE & UNUSUAL EDGE CASES

### A. Feature Engineering & Text Preprocessing
*   **What you did:** Normalized company names aggressively. Removed legal suffixes ("Inc.", "LLC", "Corp.", "Ltd."). Standardized "&" vs "and". Generated character-level n-gram (2-4) TF-IDF vectors in Spark MLlib.
*   **The "Unusual" Issue:** **"The Acronym Ambiguity Problem."** "IBM" and "International Business Machines" have zero character overlap in TF-IDF space. Also "CVS" could be "CVS Health", "CVS Pharmacy", or "CVS Caremark" — all different legal entities, but sometimes meant the same.
*   **The Fix:** Dual vectorization: (1) Character n-gram TF-IDF for typo tolerance (e.g., "Jhonson" ≈ "Johnson"). (2) Word-level TF-IDF for semantic matching. Both similarity scores fed as features to the Random Forest classifier. Also maintained a manual "acronym expansion dictionary" (IBM → International Business Machines) as a lookup pre-processing step.

### B. Scalability Phase
*   **What you did:** Needed to compare 20K × 40K = 800 million pairs. Brute force was computationally impossible.
*   **The "Unusual" Issue:** **"The Quadratic Explosion."** Even on a Spark cluster, computing all 800M cosine similarities was intractable (days of compute, memory explosions).
*   **The Fix:** **Locality-Sensitive Hashing (LSH) blocking.** Spark MLlib's `MinHashLSH` reduces candidate pairs to only those that share hash buckets. This brought the comparison set from 800M down to ~2M candidate pairs (99.75% reduction). Then pairwise cosine similarity was computed only within these candidate blocks.

### C. Evaluation & Edge Cases
*   **What you did:** Evaluated using Precision, Recall, F1 on a manually labeled holdout of 1,000 pairs.
*   **The "Unusual" Issue:** **"The Parent-Subsidiary Trap."** "Johnson & Johnson" and "Janssen Pharmaceuticals" are parent-subsidiary with completely different names. String matching can NEVER catch this. These were flagged as false negatives.
*   **The Fix:** Accepted that string matching has an inherent ceiling (~75% accuracy). For the remaining 25%, the system generated a "low confidence" queue for human review. In a v2 proposal, I recommended augmenting with external data sources (D&B or SEC EDGAR filings) that contain parent-subsidiary hierarchies as a supplementary lookup.

---

## ⚔️ 3. FLIPKART ROUND-SPECIFIC GRINDING QUESTIONS

### 🔴 DDS (System Design)
1.  **"Flipkart receives product listings from 1 million sellers. Many list the same product with different names ('iPhone 15 Pro Max 256GB' vs 'Apple iPhone 15Pro Max 256 GB Black'). Design a product deduplication system at scale."**
    *   *Ans:* Direct parallel to this project. (1) Text normalization (strip seller-specific prefixes). (2) TF-IDF + LSH blocking on product titles. (3) Add structured features (brand, category, price similarity). (4) Train a gradient boosted classifier on labeled pairs. (5) Transitive closure to merge canonical products. (6) For images: CLIP image embeddings for visual dedup as a secondary signal.
2.  **"What metric do you optimize for in entity matching: Precision or Recall? Why?"**
    *   *Ans:* Depends on business cost. If false merges (merging two DIFFERENT products into one listing) cause bad customer experience → optimize Precision. If missed merges (having 50 duplicate listings) dilute search results → optimize Recall. In practice, use an asymmetric threshold: high-confidence auto-merge (Precision@0.99), low-confidence → human review queue.

### 🔵 DMM (Mathematical Modeling)
1.  **"Derive mathematically why character n-grams are more robust to typos than word-level tokens for string matching."**
    *   *Ans:* A single character typo ("Jhonson" → "Johnson") changes 1 word token completely (Jaccard = 0 for that token). But in character 3-grams: "Jho", "hon", "ons", "nso", "son" vs "Joh", "ohn", "hns", "nso", "son" — they share "nso", "son" → partial match preserved. The overlap ratio degrades gracefully with edit distance rather than dropping to zero.
2.  **"Explain how MinHash approximates the Jaccard Similarity and what the error bound is."**
    *   *Ans:* MinHash uses random permutations. $P(\text{MinHash}(A) = \text{MinHash}(B)) = J(A,B)$. The Jaccard estimate using $k$ hash functions has variance $\frac{J(1-J)}{k}$. With 128 hash functions, the standard error of the Jaccard estimate is ~0.04.

### 🟢 HO (Hands-On)
1.  **"Write a PySpark pipeline that takes two DataFrames of company names, computes TF-IDF character n-grams, and uses MinHashLSH to find candidate pairs with Jaccard > 0.5."**
2.  **"Implement a Union-Find class in Python that performs transitive closure on match pairs."** (Already in `01_Complete_Implementation_Playbook.md` Challenge 5)

### 🟡 HM (Hiring Manager)
1.  **"This project saved hundreds of hours of manual matching. How did you quantify the ROI to justify the engineering investment?"**
    *   *STAR:* **S:** Manual matching took 2 analysts 3 weeks per quarter. **T:** Quantify time savings. **A:** Measured: Model matched 75% automatically (15K/20K) in 2 hours. Remaining 5K went to human queue but with suggested matches → reduced analyst effort by 80%. **R:** Annualized savings: ~$200K in analyst FTE time. Project cost: ~$30K in engineering. ROI > 6x.
