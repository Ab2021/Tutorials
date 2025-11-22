# Day 17: Training Infrastructure & Data Pipelines
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Fuzzy Deduplication: MinHash & LSH

**Problem:** Exact deduplication (MD5 hash of file) misses near-duplicates (e.g., same article with one word changed, or different ads).
**Solution:** MinHash + Locality Sensitive Hashing (LSH).

**MinHash Algorithm:**
1.  **Shingling:** Convert document to set of n-grams (e.g., 5-grams).
    - "The cat sat" -> {"The cat", "cat sat"}
2.  **Hashing:** Apply $K$ different hash functions to every n-gram.
3.  **Min-Value:** For each hash function, keep the *minimum* hash value seen in the document.
4.  **Signature:** The document is represented by a vector of $K$ integers (the min values).
5.  **Similarity:** The Jaccard similarity between two documents is approximately equal to the fraction of matching values in their signatures.

**LSH (Locality Sensitive Hashing):**
- Comparing every pair of signatures is $O(N^2)$. Too slow for billions of docs.
- **Banding:** Divide signature into $b$ bands of $r$ rows.
- Hash each band to a bucket.
- Candidate pairs are documents that hash to the same bucket in *at least one* band.
- Only check candidates. Complexity $\approx O(N)$.

### 2. High-Performance Streaming Dataloader

**Requirements:**
- **Infinite Dataset:** Cannot fit in RAM.
- **Random Access:** Impossible on S3.
- **Shuffling:** Essential for convergence.
- **Throughput:** Must exceed GPU consumption rate.

**Architecture (MosaicML Streaming / WebDataset):**
1.  **Shard Creation:**
    - Data is written as thousands of `.mds` or `.tar` files (shards).
    - Each shard contains ~100MB - 1GB of data.
    - Shards are uploaded to cloud storage (S3/GCS).
2.  **Deterministic Shuffling:**
    - Master node generates a random permutation of *shard IDs*.
    - Assigns shards to worker nodes.
3.  **Local Caching:**
    - Worker downloads assigned shard to local NVMe.
    - Future epochs read from NVMe (no network).
4.  **Intra-Shard Shuffling:**
    - Load full shard into memory.
    - Shuffle samples within the shard.
    - Yield samples.

**Why this works:**
- Global shuffling is approximated by shuffling shards + shuffling within shards.
- Sequential I/O from S3 is fast.
- NVMe cache prevents network bottlenecks.

### 3. Tokenization Pipeline Optimization

**Bottleneck:** Python tokenizers are slow.
**Solution:** Rust-based tokenizers (HuggingFace).

**Parallelism:**
- **Pre-tokenization:** Tokenize the entire dataset *offline* and save as `uint16` (2 bytes per token) or `int32`.
- **On-the-fly:** Tokenize in the DataLoader workers (`num_workers > 0`).

**Packing (Concatenation):**
- LLMs train on fixed context length (e.g., 4096).
- Documents are rarely exactly 4096 tokens.
- **Naive:** Pad every document. -> Wasted compute on padding.
- **Packing:** Concatenate documents with an `<EOS>` token until 4096 is reached.
    - `[Doc A] <EOS> [Doc B] <EOS> [Doc C (partial)]`
    - **Attention Masking:** Need "Block Diagonal" masking so Doc A doesn't attend to Doc B. (Or just ignore it and let the model learn `<EOS>` resets context).

### Code: MinHash Implementation

```python
import hashlib
import struct

def get_minhash_signature(text, num_perm=128):
    # 1. Shingling (3-grams)
    words = text.split()
    shingles = set()
    for i in range(len(words) - 2):
        shingles.add(" ".join(words[i:i+3]))
    
    # 2. Hashing
    signature = [float('inf')] * num_perm
    
    for shingle in shingles:
        # Create deterministic hashes
        h = int(hashlib.sha1(shingle.encode('utf-8')).hexdigest(), 16)
        
        for i in range(num_perm):
            # Simulate k permutations using linear transformation
            # h_i(x) = (a_i * x + b_i) % p
            # Here we just use a simple XOR for demo
            permuted_hash = h ^ i 
            if permuted_hash < signature[i]:
                signature[i] = permuted_hash
                
    return signature

# Jaccard Estimation
def estimate_similarity(sig1, sig2):
    matches = sum(1 for i in range(len(sig1)) if sig1[i] == sig2[i])
    return matches / len(sig1)
```

### Code: Streaming Dataset with Packing

```python
class PackedIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, file_paths, block_size=4096):
        self.file_paths = file_paths
        self.block_size = block_size
        
    def __iter__(self):
        buffer = []
        for path in self.file_paths:
            with open(path, 'r') as f:
                tokens = json.load(f)['tokens'] # Assume pre-tokenized
                buffer.extend(tokens)
                
                while len(buffer) >= self.block_size:
                    yield torch.tensor(buffer[:self.block_size])
                    buffer = buffer[self.block_size:]
```
