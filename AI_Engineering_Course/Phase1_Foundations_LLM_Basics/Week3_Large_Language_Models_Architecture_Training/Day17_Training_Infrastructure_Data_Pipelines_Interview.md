# Day 17: Training Infrastructure & Data Pipelines
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why do we need "Sharding" for large datasets? Why not just read files randomly?

**Answer:**
- **S3 Limitations:** Cloud object stores (S3) are optimized for sequential throughput, not random access (high latency). Reading small files randomly kills performance.
- **OS Limitations:** Opening millions of small files exhausts file descriptors and inode caches.
- **Sharding:** Grouping 10,000 examples into one large (1GB) file allows for efficient sequential reads, better compression, and easier distribution across worker nodes.

#### Q2: Explain the concept of "Sequence Packing" (or Concatenation) in LLM training.

**Answer:**
- **Problem:** Documents vary in length. If we batch them and pad to the max length (e.g., 4096), we waste compute on padding tokens (often >30% waste).
- **Solution:** Concatenate multiple documents into a single sequence of length 4096, separated by `<EOS>` tokens.
- **Attention:** We must ensure that tokens from Document B do not attend to tokens from Document A. This requires a "Block Diagonal" attention mask (or simply ignoring it and relying on the model to learn the separation).

#### Q3: How does MinHash deduplication work?

**Answer:**
- **Goal:** Find near-duplicates (e.g., 95% similar text).
- **Method:**
    1.  Convert text to a set of n-grams (shingles).
    2.  Hash each n-gram using $K$ hash functions.
    3.  Keep the minimum hash value for each function.
    4.  This creates a "Signature" of length $K$.
    5.  The probability that the min-hash of two sets is equal is exactly their Jaccard Similarity.
    6.  Use LSH (Locality Sensitive Hashing) to quickly find pairs with matching signatures.

#### Q4: What is the difference between "Pre-tokenization" and "On-the-fly Tokenization"?

**Answer:**
- **Pre-tokenization:** Tokenize the entire dataset once and save it to disk as integers (`uint16`).
    - *Pros:* Faster training (CPU doesn't work during training loop), exact data size known.
    - *Cons:* Static (cannot change tokenizer or augmentation easily).
- **On-the-fly:** Tokenize raw text in the DataLoader.
    - *Pros:* Flexible (can apply random augmentations, switch tokenizers).
    - *Cons:* CPU bottleneck if tokenization is slow (Python).

#### Q5: How do you handle "Bad Data" that causes training to crash (loss spike)?

**Answer:**
- **Prevention:** Strict filtering in the pipeline (length checks, UTF-8 validation).
- **Detection:** Monitor loss. If it spikes, save the batch indices.
- **Mitigation:**
    - **Skip Batch:** If a batch causes NaN, skip the update.
    - **Gradient Clipping:** Prevents massive updates.
    - **Resume:** Load previous checkpoint and skip the specific data shard.

---

### Production Challenges

#### Challenge 1: The "CPU Bottleneck"

**Scenario:** You have 8x H100 GPUs (very fast). Training is slow. `nvidia-smi` shows GPU utilization fluctuating between 0% and 100%.
**Root Cause:** The DataLoader cannot feed data fast enough. The GPUs are starving.
**Diagnosis:** Check CPU usage. If 100%, you are CPU bound.
**Solution:**
- Increase `num_workers` in DataLoader.
- Use a faster tokenizer (Rust-based).
- Move to Pre-tokenized data (remove CPU work entirely).
- Use `pin_memory=True` to speed up transfer to GPU.

#### Challenge 2: S3 Throttling

**Scenario:** Training on 1000 GPUs. All GPUs try to download shards from S3 at the start of an epoch. S3 returns `503 Slow Down`.
**Root Cause:** Too many requests per second to a single S3 prefix.
**Solution:**
- **Sharding:** Ensure dataset is sharded into enough files.
- **Randomization:** Have different GPUs start reading from different shards/prefixes.
- **Local Caching:** Download shards to local NVMe once and reuse.
- **CDN:** Use a high-performance storage layer (like FSx for Lustre) if budget permits.

#### Challenge 3: Training on "Infinite" Streams

**Scenario:** You are training on a continuous stream of Twitter data. There is no "Epoch".
**Issue:** How do you ensure you don't overfit to recent data or forget old data?
**Solution:**
- **Replay Buffer:** Mix the incoming stream with a buffer of "Gold" data (Wikipedia, Books) at a fixed ratio (e.g., 80% new, 20% old).
- **Checkpointing:** Save checkpoints by step count (e.g., every 1000 steps), not by epoch.

#### Challenge 4: Deduplication at Scale (Petabytes)

**Scenario:** You have 1PB of CommonCrawl. You need to dedup it. You can't fit it in RAM.
**Solution:**
- **MapReduce / Spark:**
    - **Map:** Calculate MinHash signature for each document. Emit `(band_hash, doc_id)`.
    - **Shuffle:** Group by `band_hash`.
    - **Reduce:** In each group, compare full signatures. Emit duplicate pairs.
    - **Filter:** Remove duplicates from the dataset.

### Summary Checklist for Production
- [ ] **Format:** Use Parquet or Arrow with Snappy compression.
- [ ] **Dataloader:** Use `streaming=True` with local NVMe caching.
- [ ] **Dedup:** Run MinHash LSH on the entire corpus.
- [ ] **Monitoring:** Track "Data Loading Time" vs "Compute Time".
