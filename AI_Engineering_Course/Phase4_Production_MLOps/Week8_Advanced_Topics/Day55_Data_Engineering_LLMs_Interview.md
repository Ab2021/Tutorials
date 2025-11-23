# Day 55: Data Engineering for LLMs
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What are the main steps in an LLM data pipeline?

**Answer:**
1. **Collection:** Web crawl, books, code, academic papers
2. **Filtering:** Language detection, quality heuristics, toxicity
3. **Deduplication:** Exact (MD5 hash) + near (MinHash LSH)
4. **Cleaning:** Remove HTML, normalize whitespace, remove PII
5. **Mixing:** Combine sources with weights (e.g., 60% web, 20% books)
6. **Tokenization:** BPE or SentencePiece
7. **Formatting:** Instruction or conversation format
8. **Versioning:** DVC for reproducibility

#### Q2: How does MinHash LSH work for near-deduplication?

**Answer:**
1. **Shingles:** Create n-grams (e.g., 3-grams) from document
2. **MinHash:** Hash shingles, keep minimum hashes (signature)
3. **LSH (Locality-Sensitive Hashing):** Group similar signatures
4. **Query:** Check if new document matches existing signatures
5. **Threshold:** Jaccard similarity >0.8 = duplicate

**Benefits:** O(1) query time, finds near-duplicates efficiently

#### Q3: What quality heuristics are used for web data filtering?

**Answer:**
- **Word count:** 50-10,000 words (remove too short/long)
- **Mean word length:** 3-10 characters (detect gibberish)
- **Symbol-to-word ratio:** <0.1 (remove code/noise)
- **Uppercase ratio:** <0.3 (remove SHOUTING)
- **Perplexity:** <1000 (remove low-quality text)
- **Language detection:** Keep only target language

#### Q4: How do you mix multiple data sources?

**Answer:**
**Weighted Sampling:**
- Define weights: `{'web': 0.6, 'books': 0.2, 'code': 0.2}`
- Sample proportionally from each source
- Shuffle combined dataset

**Upsampling/Downsampling:**
- Upsample high-quality sources (repeat examples)
- Downsample low-quality sources (skip examples)

**Example (LLaMA):** 67% CommonCrawl, 15% C4, 4.5% GitHub, etc.

#### Q5: What is perplexity-based filtering?

**Answer:**
- **Perplexity:** Measure of how "surprised" a language model is by text
- **Lower perplexity** = higher quality, more natural text
- **Process:**
  1. Train small LM on high-quality data
  2. Compute perplexity for each document
  3. Keep documents with perplexity <threshold (e.g., 1000)
- **Benefit:** Removes low-quality, unnatural text

---

### Production Challenges

#### Challenge 1: Deduplication Too Slow

**Scenario:** MinHash deduplication takes 1 week for 1B documents.
**Root Cause:** Sequential processing, inefficient implementation.
**Solution:**
- **Parallel Processing:** Use multiprocessing (8-16 workers).
- **Batch Processing:** Process 10K documents at a time.
- **Optimized Library:** Use `datasketch` library (C++ backend).
- **Distributed:** Use Spark for >1B documents.

#### Challenge 2: Quality Filtering Too Aggressive

**Scenario:** Filtered 90% of data, left with too little training data.
**Root Cause:** Thresholds too strict.
**Solution:**
- **Relax Thresholds:** Word count 20-20,000 (instead of 50-10,000).
- **Remove Filters:** Drop less important filters (e.g., uppercase ratio).
- **Manual Review:** Sample filtered data, check if good data was removed.
- **Iterative:** Start lenient, gradually tighten based on model performance.

#### Challenge 3: Data Mixing Imbalance

**Scenario:** Model performs poorly on code despite 20% code data.
**Root Cause:** Code data is lower quality or insufficient.
**Solution:**
- **Increase Weight:** 20% → 30% code data.
- **Upsample:** Repeat code examples 2x.
- **Better Filtering:** Improve code quality filtering.
- **More Data:** Collect more high-quality code.

#### Challenge 4: PII Leakage

**Scenario:** Model generates real email addresses from training data.
**Root Cause:** PII not fully removed from training data.
**Solution:**
- **Better Regex:** Improve PII detection patterns.
- **NER Model:** Use Named Entity Recognition for names, addresses.
- **Manual Review:** Sample data, check for PII.
- **Scrubbing:** Use tools like `scrubadub` for comprehensive PII removal.

#### Challenge 5: Tokenizer Vocabulary Too Large

**Scenario:** 100K vocabulary uses too much memory.
**Root Cause:** Vocabulary size set too high.
**Solution:**
- **Reduce Size:** 100K → 50K (GPT-3 uses 50K).
- **Subword Regularization:** Use SentencePiece with regularization.
- **Prune Rare Tokens:** Remove tokens appearing <10 times.
- **Trade-off:** Smaller vocab = longer sequences, but less memory.

### Summary Checklist for Production
- [ ] **Deduplication:** Use **MinHash LSH** for near-duplicates.
- [ ] **Quality Filtering:** Apply **word count**, **symbol ratio**, **perplexity**.
- [ ] **PII Removal:** Use **regex + NER** for comprehensive removal.
- [ ] **Data Mixing:** Use **weighted sampling** (e.g., 60% web, 20% books).
- [ ] **Parallel Processing:** Use **8-16 workers** for speed.
- [ ] **Versioning:** Use **DVC** for reproducibility.
- [ ] **Monitoring:** Track **filter rates**, **quality metrics**.
