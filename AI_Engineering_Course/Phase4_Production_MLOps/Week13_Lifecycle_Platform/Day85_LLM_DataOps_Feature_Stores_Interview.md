# Day 64: LLM DataOps & Feature Stores
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between a Feature Store and a Vector Database?

**Answer:**
- **Feature Store (Feast, Tecton):** Manages **structured** data (user ID, subscription plan, last login, aggregate stats). Used to retrieve context *about* entities. Served via key-value lookups.
- **Vector Database (Pinecone, Milvus):** Manages **unstructured** data embeddings (documents, chat history). Used to perform semantic search (Nearest Neighbor).
- **In LLMs:** You need both. Feature store provides user context ("Who is asking?"), Vector DB provides knowledge context ("What do we know about this topic?").

#### Q2: Why is Data Versioning critical for LLMs?

**Answer:**
- **Reproducibility:** If a fine-tuned model behaves badly, you need to know exactly which dataset version it was trained on.
- **Lineage:** Did the bad behavior come from the "Reddit" dump or the "Wikipedia" dump?
- **Rollback:** Ability to revert to a previous "clean" dataset snapshot.
- **Tools:** DVC, Git LFS, LakeFS.

#### Q3: Explain the "Self-Instruct" method for synthetic data generation.

**Answer:**
- **Concept:** Bootstrapping a large dataset from a small set of human-written seed tasks.
- **Process:**
  1.  Take 100 seed (Instruction, Output) pairs.
  2.  Ask an LLM (GPT-4) to generate *new* instructions similar to the seeds.
  3.  Ask the LLM to generate outputs for the new instructions.
  4.  Filter low-quality generations.
  5.  Add to pool and repeat.
- **Benefit:** Create 50k training examples from 100 manual ones.

#### Q4: How do you handle PII in LLM training data?

**Answer:**
- **Detection:** Use NER (Named Entity Recognition) tools like Presidio or regex to find Names, Emails, SSNs.
- **Redaction:** Replace with `<PERSON>`, `<EMAIL>`.
- **Synthetic Replacement:** Replace "John Smith" with "Alice Bob" to maintain sentence structure but remove real info.
- **Vault:** Store mapping `USER_123 -> John Smith` securely, only detokenize at the very last step of UI rendering (if needed).

#### Q5: What is a "Golden Set" in evaluation?

**Answer:**
- A curated, high-quality dataset of inputs and *ground truth* answers used for evaluation.
- It should cover edge cases, adversarial prompts, and typical usage.
- It is *never* used for training (to prevent data leakage).
- Performance on the Golden Set determines if a model is ready for production.

---

### Production Challenges

#### Challenge 1: Data Drift causing Hallucinations

**Scenario:** RAG system starts answering questions about "iPhone 15" with "iPhone 14" specs.
**Root Cause:** The Vector DB has old documents. New documents haven't been ingested.
**Solution:**
- **Real-time Ingestion:** Move from batch indexing (nightly) to streaming indexing (Kafka -> Vector DB).
- **Freshness Score:** Penalize old documents during retrieval.
- **Monitoring:** Track "Document Age" of retrieved chunks.

#### Challenge 2: Training Data Contamination

**Scenario:** Model scores 100% on benchmarks but fails in real world.
**Root Cause:** The benchmark questions were accidentally included in the training set (Data Leakage).
**Solution:**
- **Decontamination:** N-gram overlap check between Training Set and Test Set. Remove overlaps.
- **Canary String:** Insert a unique string (e.g., "2025-AI-COURSE-CANARY") into the test set. If model generates it, it memorized the test set.

#### Challenge 3: Feedback Loop Poisoning

**Scenario:** Users start trolling the bot, giving "Thumbs Up" to toxic answers.
**Root Cause:** Blindly trusting user feedback for fine-tuning.
**Solution:**
- **Moderation:** Run feedback through a moderation model before adding to training set.
- **Trusted Users:** Only use feedback from verified/internal users for training.
- **Outlier Detection:** Discard feedback patterns that deviate significantly.

#### Challenge 4: Feature Skew (Training-Serving Skew)

**Scenario:** Training used clean, normalized text. Production receives noisy, raw text. Model fails.
**Root Cause:** Preprocessing pipeline differs between Training (offline) and Inference (online).
**Solution:**
- **Unified Pipeline:** Use the exact same code (e.g., Python package) for cleaning in both DVC pipeline and API server.
- **Feature Store:** Compute features once, store, and serve.

#### Challenge 5: Vector DB Cost Explosion

**Scenario:** Storing 1 Billion vectors in Pinecone is costing $5k/month.
**Root Cause:** Storing everything in RAM/High-performance index.
**Solution:**
- **DiskANN:** Use disk-based vector indexes (LanceDB, Weaviate) for colder data.
- **Quantization:** Compress vectors (Binary Quantization, Scalar Quantization) to reduce RAM usage by 10-30x.
- **Tiering:** Move old data to S3, keep hot data in Vector DB.

### System Design Scenario: Data Pipeline for Financial News Analyst

**Requirement:** Ingest news, analyze sentiment, serve to LLM.
**Design:**
1.  **Ingest:** Kafka stream of news articles.
2.  **Process:** Spark job cleans HTML, chunks text.
3.  **Embed:** Embedding model converts chunks to vectors.
4.  **Store:**
    - Vectors -> Milvus (for search).
    - Metadata (Ticker, Date, Sentiment) -> Postgres/Feature Store.
5.  **Serve:** LLM queries Milvus for "AAPL news", filters by "Date > yesterday", retrieves Sentiment from Feature Store.
6.  **Lineage:** DVC tracks which embedding model version was used.

### Summary Checklist for Production
- [ ] **Versioning:** Use **DVC** or **LakeFS**.
- [ ] **Quality:** Implement **PII Scrubbing** and **Decontamination**.
- [ ] **Store:** Use **Vector DB** + **Redis**.
- [ ] **Feedback:** Log **User Feedback** for RLHF.
- [ ] **Synthetic:** Use **Self-Instruct** to augment small datasets.
