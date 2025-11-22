# Day 17: Training Infrastructure & Data Pipelines
## Core Concepts & Theory

### The Unsung Hero: Data Engineering

Training a 70B model requires feeding 1.4 Trillion tokens to thousands of GPUs without stalling.
If your data pipeline is slow, your expensive GPUs sit idle.
**Goal:** GPU Utilization > 50% (MFU).

### 1. The Data Pipeline Stages

1.  **Acquisition:** Crawling (CommonCrawl), APIs (Reddit, Twitter), Dumps (Wikipedia, StackOverflow).
2.  **Cleaning:** Removing HTML tags, boilerplate, PII (Personally Identifiable Information), and toxic content.
3.  **Deduplication:** Exact (hashing) and Fuzzy (MinHash) dedup.
4.  **Tokenization:** Converting text to integers.
5.  **Sharding:** Splitting data into manageable chunks (e.g., 1GB files).
6.  **Loading:** Streaming data to GPUs during training.

### 2. Storage Formats

**JSONL (JSON Lines):**
- Human readable.
- Slow to parse.
- Bad for random access.
- **Verdict:** Good for raw data, bad for training.

**TFRecord / WebDataset:**
- Binary formats.
- Optimized for sequential streaming.
- **Verdict:** Good for image/video, okay for text.

**Arrow / Parquet (HuggingFace Datasets):**
- Columnar, memory-mapped.
- Zero-copy reads (fast).
- Compression support (Snappy/Zstd).
- **Verdict:** The industry standard for LLM training.

### 3. Streaming vs. Pre-loading

**Pre-loading:** Load entire dataset into RAM.
- **Impossible** for LLMs (TB of data).

**Streaming (Lazy Loading):**
- Keep data on disk (or S3).
- Stream chunks into RAM on demand.
- Shuffle using a "Shuffle Buffer" (load 10k examples, shuffle, yield 1).
- **Tools:** `datasets.load_dataset(..., streaming=True)`, `WebDataset`.

### 4. Distributed Data Loading

When training on 1000 GPUs:
- **Naive:** 1000 GPUs read from S3 simultaneously. -> **S3 Throttling / Network Congestion.**
- **Better:**
    - **Sharding:** Divide dataset into 1000 shards. Each GPU reads 1 shard.
    - **Caching:** Cache shards on local NVMe SSD of the node.
    - **Prefetching:** CPU loads batch $N+1$ while GPU processes batch $N$.

### 5. Quality Filtering

**Heuristics:**
- **Perplexity:** Train a small model (KenLM) on Wikipedia. If a web page has high perplexity (looks nothing like Wikipedia), discard it.
- **Length:** Discard very short documents.
- **Language ID:** Use `fastText` to keep only target languages.
- **Gopher Rules:** Remove lines with no stop words, lines with mostly symbols, etc.

### Summary of Infrastructure

| Component | Tool | Purpose |
| :--- | :--- | :--- |
| **Storage** | Parquet / Arrow | Fast, compressed storage |
| **Loading** | HF Datasets / MosaicML Streaming | Stream from S3/Disk |
| **Tokenization** | HuggingFace Tokenizers (Rust) | Fast subword encoding |
| **Deduplication** | datasketch (MinHash) | Remove duplicates |
| **Orchestration** | Slurm / Kubernetes | Manage GPU jobs |

### Next Steps
In the Deep Dive, we will build a high-performance streaming dataloader and analyze the MinHash deduplication algorithm.
