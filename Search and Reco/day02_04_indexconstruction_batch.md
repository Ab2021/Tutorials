# Day 2: Batch Index Construction Algorithms

## Table of Contents
1. [Introduction to Batch Index Construction](#introduction)
2. [Memory-Based Construction](#memory-based-construction)
3. [External Sorting Approaches](#external-sorting-approaches)
4. [MapReduce Index Construction](#mapreduce-construction)
5. [Distributed Construction](#distributed-construction)
6. [Study Questions](#study-questions)
7. [Code Examples](#code-examples)

---

## Introduction to Batch Index Construction

Batch index construction builds complete inverted indexes from scratch, typically used for **initial index creation** or **periodic rebuilds** when documents don't change frequently.

### Construction Challenges

#### **Scale Requirements**
- **Web Collections**: Billions of documents, petabytes of text
- **Memory Limitations**: Cannot fit entire collection in RAM
- **Time Constraints**: Index must be built within acceptable timeframe
- **Quality Requirements**: Must produce optimal index structure

#### **Resource Trade-offs**
- **CPU**: Tokenization, sorting, compression
- **Memory**: Temporary data structures, buffering
- **Disk I/O**: Reading documents, writing index
- **Network**: Distributed construction coordination

### Construction Strategies

#### **Single-Machine Approaches**
1. **In-Memory Construction**: Entire index fits in RAM
2. **External Sort-Based**: Use disk for temporary storage
3. **Streaming Construction**: Process documents sequentially

#### **Distributed Approaches**  
1. **MapReduce/Hadoop**: Commodity cluster construction
2. **Spark-Based**: In-memory distributed processing
3. **Custom Distributed**: Specialized index construction systems

---

## Memory-Based Construction

Simple approach when the entire inverted index fits in available memory.

### Algorithm Overview

```python
def build_inverted_index_memory(documents):
    """Build inverted index assuming it fits in memory"""
    inverted_index = defaultdict(list)
    
    for doc_id, content in documents:
        terms = tokenize(content)
        term_positions = defaultdict(list)
        
        # Track term positions
        for pos, term in enumerate(terms):
            term_positions[term].append(pos)
        
        # Add to inverted index
        for term, positions in term_positions.items():
            posting = Posting(doc_id, len(positions), positions)
            inverted_index[term].append(posting)
    
    # Sort posting lists by document ID
    for term in inverted_index:
        inverted_index[term].sort(key=lambda p: p.doc_id)
    
    return inverted_index
```

### Memory Requirements

#### **Size Estimation**
```python
def estimate_index_size(collection_stats):
    """Estimate memory requirements for in-memory construction"""
    
    # Basic posting: doc_id (4 bytes) + tf (4 bytes) = 8 bytes
    basic_posting_size = 8
    
    # Positional posting: + positions (4 bytes per position)
    avg_positions_per_posting = collection_stats.avg_term_frequency
    positional_posting_size = basic_posting_size + (4 * avg_positions_per_posting)
    
    # Total postings = total term occurrences
    total_postings = collection_stats.total_term_occurrences
    
    # Index size
    index_size = total_postings * positional_posting_size
    
    # Add term dictionary overhead (assume 50 bytes per unique term)
    dict_size = collection_stats.vocabulary_size * 50
    
    # Total memory requirement (with 2x safety factor)
    total_memory = (index_size + dict_size) * 2
    
    return {
        'index_size_gb': index_size / (1024**3),
        'dict_size_mb': dict_size / (1024**2),
        'total_memory_gb': total_memory / (1024**3)
    }
```

### Optimization Techniques

#### **Memory Pool Management**
```python
class MemoryPool:
    def __init__(self, block_size=1024*1024):  # 1MB blocks
        self.block_size = block_size
        self.blocks = []
        self.current_block = None
        self.current_offset = 0
    
    def allocate(self, size):
        """Allocate memory from pool"""
        if self.current_block is None or self.current_offset + size > self.block_size:
            # Need new block
            self.current_block = bytearray(self.block_size)
            self.blocks.append(self.current_block)
            self.current_offset = 0
        
        # Allocate from current block
        start_offset = self.current_offset
        self.current_offset += size
        return memoryview(self.current_block[start_offset:start_offset + size])
```

#### **Streaming Document Processing**
```python
def build_index_streaming(document_stream):
    """Build index without loading all documents into memory"""
    inverted_index = defaultdict(list)
    
    for doc_id, content in document_stream:
        # Process one document at a time
        terms = tokenize(content)
        term_freq = Counter(terms)
        
        for term, tf in term_freq.items():
            posting = Posting(doc_id, tf)
            inverted_index[term].append(posting)
        
        # Optional: garbage collect document content
        del content
    
    return inverted_index
```

---

## External Sorting Approaches

When the index is too large for memory, use external sorting techniques with disk-based temporary storage.

### SPIMI Algorithm (Single-Pass In-Memory Indexing)

#### **Core Algorithm**
```python
def spimi_invert(document_stream, memory_limit):
    """Single-Pass In-Memory Indexing with external merge"""
    block_number = 0
    intermediate_files = []
    
    while True:
        # Build in-memory index for one block
        block_index = {}
        memory_used = 0
        
        for doc_id, content in document_stream:
            terms = tokenize(content)
            
            for term in set(terms):  # Unique terms only
                if term not in block_index:
                    block_index[term] = []
                    memory_used += len(term) * 2  # Approximate string overhead
                
                posting = Posting(doc_id, terms.count(term))
                block_index[term].append(posting)
                memory_used += 12  # Posting size estimate
                
                if memory_used > memory_limit:
                    break
            
            if memory_used > memory_limit:
                break
        
        if not block_index:
            break  # No more documents
        
        # Sort terms and write block to disk
        filename = f"block_{block_number}.idx"
        write_block_to_disk(block_index, filename)
        intermediate_files.append(filename)
        block_number += 1
    
    # Merge all intermediate files
    final_index = merge_blocks(intermediate_files)
    return final_index
```

#### **Block Writing**
```python
def write_block_to_disk(block_index, filename):
    """Write sorted block to disk"""
    with open(filename, 'wb') as f:
        # Sort terms lexicographically
        sorted_terms = sorted(block_index.keys())
        
        for term in sorted_terms:
            # Sort postings by document ID
            postings = sorted(block_index[term], key=lambda p: p.doc_id)
            
            # Write term and postings
            write_string(f, term)
            write_int(f, len(postings))
            
            for posting in postings:
                write_int(f, posting.doc_id)
                write_int(f, posting.tf)
```

#### **Multi-Way Merge**
```python
def merge_blocks(block_files):
    """Merge multiple sorted block files"""
    import heapq
    
    # Open all block files
    file_iterators = []
    heap = []
    
    for i, filename in enumerate(block_files):
        iterator = BlockFileIterator(filename)
        try:
            term, postings = next(iterator)
            heapq.heappush(heap, (term, postings, i, iterator))
        except StopIteration:
            pass
    
    merged_index = {}
    
    while heap:
        term, postings, file_id, iterator = heapq.heappop(heap)
        
        # Merge postings for same term
        if term in merged_index:
            merged_index[term].extend(postings)
        else:
            merged_index[term] = postings
        
        # Get next term from same file
        try:
            next_term, next_postings = next(iterator)
            heapq.heappush(heap, (next_term, next_postings, file_id, iterator))
        except StopIteration:
            pass
    
    # Sort all posting lists
    for term in merged_index:
        merged_index[term].sort(key=lambda p: p.doc_id)
    
    return merged_index
```

### Two-Pass Algorithm

#### **Pass 1: Term Collection**
```python
def collect_vocabulary(document_stream):
    """First pass: collect all unique terms"""
    vocabulary = set()
    term_frequencies = defaultdict(int)
    
    for doc_id, content in document_stream:
        terms = tokenize(content)
        for term in terms:
            vocabulary.add(term)
            term_frequencies[term] += 1
    
    return sorted(vocabulary), term_frequencies
```

#### **Pass 2: Index Construction**
```python
def build_index_second_pass(document_stream, vocabulary):
    """Second pass: build inverted index with known vocabulary"""
    # Create posting lists for all terms
    inverted_index = {term: [] for term in vocabulary}
    
    for doc_id, content in document_stream:
        terms = tokenize(content)
        term_counts = Counter(terms)
        
        for term, tf in term_counts.items():
            if term in inverted_index:  # Should always be true
                posting = Posting(doc_id, tf)
                inverted_index[term].append(posting)
    
    return inverted_index
```

---

## MapReduce Index Construction

Distributed index construction using the MapReduce paradigm for web-scale collections.

### MapReduce Overview

#### **Map Phase**: Document Processing
```python
def map_function(doc_id, content):
    """Map function: emit (term, doc_id, tf) tuples"""
    terms = tokenize(content)
    term_counts = Counter(terms)
    
    results = []
    for term, tf in term_counts.items():
        results.append((term, (doc_id, tf)))
    
    return results
```

#### **Reduce Phase**: Index Building
```python
def reduce_function(term, posting_list):
    """Reduce function: build posting list for term"""
    # posting_list contains all (doc_id, tf) pairs for this term
    postings = []
    
    for doc_id, tf in posting_list:
        postings.append(Posting(doc_id, tf))
    
    # Sort by document ID
    postings.sort(key=lambda p: p.doc_id)
    
    return (term, postings)
```

### Hadoop Implementation

#### **Mapper Class**
```python
class IndexMapper:
    def map(self, key, value):
        """Process one document"""
        doc_id = key
        content = value
        
        terms = self.tokenize(content)
        term_positions = defaultdict(list)
        
        for pos, term in enumerate(terms):
            term_positions[term].append(pos)
        
        # Emit postings
        for term, positions in term_positions.items():
            posting_data = {
                'doc_id': doc_id,
                'tf': len(positions),
                'positions': positions
            }
            self.emit(term, posting_data)
```

#### **Reducer Class**
```python
class IndexReducer:
    def reduce(self, term, posting_data_list):
        """Build posting list for one term"""
        postings = []
        
        for posting_data in posting_data_list:
            posting = PositionalPosting(
                posting_data['doc_id'],
                posting_data['positions']
            )
            postings.append(posting)
        
        # Sort postings by document ID
        postings.sort(key=lambda p: p.doc_id)
        
        # Write to output
        self.write_posting_list(term, postings)
```

### Optimization Strategies

#### **Combiner Functions**
```python
def combiner_function(term, local_postings):
    """Combine postings locally before sending to reducer"""
    # Group by document ID (in case of duplicate emissions)
    doc_postings = defaultdict(list)
    
    for posting_data in local_postings:
        doc_id = posting_data['doc_id']
        doc_postings[doc_id].append(posting_data)
    
    # Merge postings for same document
    combined_postings = []
    for doc_id, postings in doc_postings.items():
        if len(postings) == 1:
            combined_postings.append(postings[0])
        else:
            # Merge multiple postings for same document
            merged_positions = []
            for p in postings:
                merged_positions.extend(p['positions'])
            
            combined_posting = {
                'doc_id': doc_id,
                'tf': len(merged_positions),
                'positions': sorted(merged_positions)
            }
            combined_postings.append(combined_posting)
    
    return combined_postings
```

---

## Distributed Construction

Modern approaches for building indexes across multiple machines.

### Spark-Based Construction

#### **RDD Processing**
```python
def build_index_spark(spark_context, documents_rdd):
    """Build inverted index using Apache Spark"""
    
    # Map: tokenize documents and emit (term, posting) pairs
    term_postings = documents_rdd.flatMap(
        lambda doc: extract_term_postings(doc[0], doc[1])
    )
    
    # Group by term
    grouped_postings = term_postings.groupByKey()
    
    # Reduce: build posting lists
    inverted_index = grouped_postings.mapValues(
        lambda postings: sorted(list(postings), key=lambda p: p.doc_id)
    )
    
    return inverted_index

def extract_term_postings(doc_id, content):
    """Extract (term, posting) pairs from document"""
    terms = tokenize(content)
    term_positions = defaultdict(list)
    
    for pos, term in enumerate(terms):
        term_positions[term].append(pos)
    
    results = []
    for term, positions in term_positions.items():
        posting = PositionalPosting(doc_id, positions)
        results.append((term, posting))
    
    return results
```

### Sharding Strategies

#### **Term-Based Sharding**
```python
def shard_by_term(term, num_shards):
    """Assign terms to shards based on hash"""
    return hash(term) % num_shards

def distribute_by_term(inverted_index, num_shards):
    """Distribute index by term across shards"""
    shards = [defaultdict(list) for _ in range(num_shards)]
    
    for term, postings in inverted_index.items():
        shard_id = shard_by_term(term, num_shards)
        shards[shard_id][term] = postings
    
    return shards
```

#### **Document-Based Sharding**
```python
def shard_by_document(doc_id, num_shards):
    """Assign documents to shards"""
    return doc_id % num_shards

def build_distributed_index(documents, num_shards):
    """Build index with document-based sharding"""
    # Distribute documents to shards
    document_shards = [[] for _ in range(num_shards)]
    
    for doc_id, content in documents:
        shard_id = shard_by_document(doc_id, num_shards)
        document_shards[shard_id].append((doc_id, content))
    
    # Build local indexes on each shard
    local_indexes = []
    for shard_docs in document_shards:
        local_index = build_inverted_index_memory(shard_docs)
        local_indexes.append(local_index)
    
    return local_indexes
```

---

## Study Questions

### Beginner Level
1. What are the main differences between memory-based and external sorting approaches?
2. Why is SPIMI considered a "single-pass" algorithm?
3. What role does the combiner play in MapReduce index construction?
4. How do you estimate memory requirements for in-memory index construction?

### Intermediate Level
1. Compare the time and space complexity of different batch construction algorithms.
2. How does term-based vs document-based sharding affect query processing?
3. What are the advantages of using Spark over traditional MapReduce for index construction?
4. How would you handle duplicate documents during batch construction?

### Advanced Level
1. Design a fault-tolerant distributed index construction system.
2. Analyze the I/O complexity of external sorting approaches for index construction.
3. How would you optimize the multi-way merge phase for very large numbers of intermediate files?
4. Design a hybrid approach combining streaming and batch construction.

### Tricky Questions
1. **Memory Paradox**: When might using less memory actually slow down index construction?
2. **Distribution Challenge**: How do you ensure load balancing when terms have very different frequencies?
3. **Failure Recovery**: How would you resume construction after a machine failure in a distributed setting?
4. **Quality vs Speed**: How do you balance index quality with construction time in production systems?

---

## Code Examples

### Complete SPIMI Implementation
```python
import heapq
import pickle
from collections import defaultdict, Counter

class SPIMIIndexBuilder:
    def __init__(self, memory_limit_mb=100):
        self.memory_limit = memory_limit_mb * 1024 * 1024  # Convert to bytes
        self.block_counter = 0
        self.intermediate_files = []
    
    def build_index(self, document_stream):
        """Build complete inverted index using SPIMI"""
        # Phase 1: Create intermediate blocks
        self.create_intermediate_blocks(document_stream)
        
        # Phase 2: Merge all blocks
        final_index = self.merge_all_blocks()
        
        # Cleanup intermediate files
        self.cleanup_intermediate_files()
        
        return final_index
    
    def create_intermediate_blocks(self, document_stream):
        """Create memory-sized intermediate blocks"""
        current_block = defaultdict(list)
        current_memory = 0
        
        for doc_id, content in document_stream:
            terms = self.tokenize(content)
            term_freq = Counter(terms)
            
            # Check if adding this document would exceed memory limit
            doc_memory = self.estimate_document_memory(term_freq)
            
            if current_memory + doc_memory > self.memory_limit and current_block:
                # Write current block and start new one
                self.write_block(current_block)
                current_block = defaultdict(list)
                current_memory = 0
            
            # Add document to current block
            for term, tf in term_freq.items():
                posting = SimplePosting(doc_id, tf)
                current_block[term].append(posting)
            
            current_memory += doc_memory
        
        # Write final block
        if current_block:
            self.write_block(current_block)
    
    def estimate_document_memory(self, term_freq):
        """Estimate memory usage for adding a document"""
        memory = 0
        for term, tf in term_freq.items():
            memory += len(term.encode('utf-8'))  # Term string
            memory += 16  # Posting object overhead
        return memory
    
    def write_block(self, block):
        """Write block to disk"""
        filename = f"spimi_block_{self.block_counter}.pkl"
        
        # Sort terms and postings
        sorted_block = {}
        for term in sorted(block.keys()):
            postings = sorted(block[term], key=lambda p: p.doc_id)
            sorted_block[term] = postings
        
        # Write to disk
        with open(filename, 'wb') as f:
            pickle.dump(sorted_block, f)
        
        self.intermediate_files.append(filename)
        self.block_counter += 1
        print(f"Written block {self.block_counter} with {len(sorted_block)} terms")
    
    def merge_all_blocks(self):
        """Merge all intermediate blocks"""
        if not self.intermediate_files:
            return {}
        
        if len(self.intermediate_files) == 1:
            # Only one block, just load it
            with open(self.intermediate_files[0], 'rb') as f:
                return pickle.load(f)
        
        # Multi-way merge
        return self.multi_way_merge()
    
    def multi_way_merge(self):
        """Perform multi-way merge of intermediate files"""
        # Open all files and create iterators
        file_handles = []
        heap = []
        
        for i, filename in enumerate(self.intermediate_files):
            f = open(filename, 'rb')
            block = pickle.load(f)
            f.close()
            
            # Create iterator over terms in this block
            block_iter = iter(sorted(block.items()))
            
            try:
                term, postings = next(block_iter)
                heapq.heappush(heap, (term, postings, i, block_iter, block))
            except StopIteration:
                pass
        
        merged_index = defaultdict(list)
        
        while heap:
            term, postings, block_id, block_iter, block_dict = heapq.heappop(heap)
            
            # Add postings to merged index
            merged_index[term].extend(postings)
            
            # Get next term from same block
            try:
                next_term, next_postings = next(block_iter)
                heapq.heappush(heap, (next_term, next_postings, block_id, block_iter, block_dict))
            except StopIteration:
                pass
        
        # Sort all posting lists by document ID
        final_index = {}
        for term, postings in merged_index.items():
            final_index[term] = sorted(postings, key=lambda p: p.doc_id)
        
        return final_index
    
    def tokenize(self, text):
        """Simple tokenization"""
        import re
        text = text.lower()
        terms = re.findall(r'\b\w+\b', text)
        return terms
    
    def cleanup_intermediate_files(self):
        """Remove intermediate files"""
        import os
        for filename in self.intermediate_files:
            if os.path.exists(filename):
                os.remove(filename)

class SimplePosting:
    def __init__(self, doc_id, tf):
        self.doc_id = doc_id
        self.tf = tf
    
    def __repr__(self):
        return f"Posting({self.doc_id}, {self.tf})"

# Example usage
def demo_spimi():
    # Create sample documents
    documents = [
        (1, "machine learning algorithms for data analysis"),
        (2, "deep learning neural networks and artificial intelligence"),
        (3, "machine learning is a subset of artificial intelligence"),
        (4, "data science uses machine learning and statistical methods"),
        (5, "neural network architectures in deep learning"),
        (6, "statistical analysis of machine learning performance"),
        (7, "artificial intelligence applications in data science"),
        (8, "deep neural networks for pattern recognition"),
        (9, "machine learning optimization techniques"),
        (10, "data analysis using statistical learning methods")
    ]
    
    # Build index with small memory limit to force multiple blocks
    builder = SPIMIIndexBuilder(memory_limit_mb=1)  # Very small for demo
    index = builder.build_index(documents)
    
    # Display results
    print(f"\nBuilt index with {len(index)} terms")
    for term in sorted(list(index.keys())[:10]):  # Show first 10 terms
        postings = index[term]
        print(f"{term}: {postings}")

if __name__ == "__main__":
    demo_spimi()
```

---

## Key Takeaways
1. **Scalability**: External sorting enables index construction beyond memory limits
2. **Efficiency**: SPIMI algorithm provides optimal single-pass construction
3. **Distribution**: MapReduce/Spark enable web-scale index construction
4. **Trade-offs**: Balance memory usage, I/O operations, and construction time
5. **Flexibility**: Different algorithms suit different collection sizes and requirements

---

**Next**: In day2_indexconstruction_incremental.md, we'll explore incremental index updates and real-time index maintenance strategies.