# Day 55: Data Engineering for LLMs
## Core Concepts & Theory

### Data Pipeline Fundamentals

**Challenges:**
- **Scale:** Billions of tokens
- **Quality:** Noisy web data
- **Diversity:** Multiple sources and formats
- **Privacy:** PII and sensitive data

### 1. Data Sources

**Web Crawl:**
- **Common Crawl:** 250+ billion web pages
- **C4 (Colossal Clean Crawled Corpus):** Filtered Common Crawl
- **RefinedWeb:** High-quality web data

**Books:**
- **Books3:** ~200K books
- **Gutenberg:** Public domain books

**Code:**
- **The Stack:** 6TB of code from GitHub
- **CodeParrot:** Filtered code dataset

**Academic:**
- **ArXiv:** Research papers
- **PubMed:** Medical literature

**Conversational:**
- **Reddit:** Discussions
- **StackOverflow:** Q&A

### 2. Data Filtering

**Quality Filters:**
- **Language detection:** Keep only target language
- **Deduplication:** Remove exact/near duplicates
- **Length filtering:** Remove too short/long documents
- **Perplexity filtering:** Remove low-quality text

**Content Filters:**
- **Toxicity:** Remove harmful content
- **PII:** Remove personal information
- **Adult content:** Filter NSFW

### 3. Deduplication

**Exact Deduplication:**
```python
seen = set()
for doc in documents:
    hash_val = hashlib.md5(doc.encode()).hexdigest()
    if hash_val not in seen:
        seen.add(hash_val)
        yield doc
```

**Near Deduplication (MinHash):**
- Create shingles (n-grams)
- Hash shingles to signatures
- Find similar documents via LSH
- **Benefit:** Remove similar documents

### 4. Data Cleaning

**Text Normalization:**
- Remove HTML tags
- Fix encoding issues
- Normalize whitespace
- Remove special characters

**Quality Heuristics:**
- **Word count:** 50-10,000 words
- **Mean word length:** 3-10 characters
- **Symbol-to-word ratio:** <0.1
- **Uppercase ratio:** <0.3

### 5. Data Mixing

**Concept:** Combine multiple datasets with weights

**Example:**
```
Web: 60%
Books: 20%
Code: 10%
Academic: 10%
```

**Upsampling/Downsampling:**
- Upsample high-quality sources
- Downsample low-quality sources

### 6. Tokenization

**Byte-Pair Encoding (BPE):**
```
1. Start with character vocabulary
2. Merge most frequent pairs
3. Repeat until vocab size reached
```

**SentencePiece:**
- Language-agnostic
- Handles rare words via subwords

**Vocabulary Size:**
- **Small:** 32K (GPT-2)
- **Medium:** 50K (GPT-3)
- **Large:** 100K (LLaMA)

### 7. Data Formatting

**Instruction Format:**
```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{response}
```

**Conversation Format:**
```
User: {user_message}
Assistant: {assistant_message}
```

### 8. Data Versioning

**DVC (Data Version Control):**
```bash
dvc add data/train.jsonl
git add data/train.jsonl.dvc
git commit -m "Add training data"
```

**Benefits:**
- Track data changes
- Reproduce experiments
- Share large datasets

### 9. Data Quality Metrics

**Perplexity:**
- Measure with language model
- Lower = higher quality

**Diversity:**
- Unique n-grams
- Vocabulary size

**Toxicity:**
- Perspective API score
- Target: <0.1

### 10. Real-World Examples

**LLaMA Training Data:**
- **CommonCrawl:** 67%
- **C4:** 15%
- **GitHub:** 4.5%
- **Wikipedia:** 4.5%
- **Books:** 4.5%
- **ArXiv:** 2.5%
- **StackExchange:** 2%

**GPT-3 Training Data:**
- **CommonCrawl (filtered):** 60%
- **WebText2:** 22%
- **Books1:** 8%
- **Books2:** 8%
- **Wikipedia:** 3%

### Summary

**Data Pipeline:**
1. **Collection:** Web crawl, books, code
2. **Filtering:** Quality, toxicity, PII
3. **Deduplication:** Exact + near duplicates
4. **Cleaning:** Normalize, remove noise
5. **Mixing:** Combine sources with weights
6. **Tokenization:** BPE, SentencePiece
7. **Formatting:** Instruction, conversation
8. **Versioning:** DVC, Git

### Next Steps
In the Deep Dive, we will implement complete data pipeline with filtering, deduplication, and quality checks.
