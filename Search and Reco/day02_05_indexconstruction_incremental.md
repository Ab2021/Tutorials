# Day 2: Incremental Index Updates

## Table of Contents
1. [Introduction to Incremental Updates](#introduction)
2. [Dynamic Index Structures](#dynamic-index-structures)
3. [Update Operations](#update-operations)
4. [Delta Index Approach](#delta-index-approach)
5. [Real-time Indexing](#real-time-indexing)
6. [Study Questions](#study-questions)
7. [Code Examples](#code-examples)

---

## Introduction to Incremental Updates

Incremental updates enable **real-time index maintenance** without rebuilding the entire index, essential for dynamic document collections like web pages, news feeds, and social media.

### Update Requirements

#### **Performance Constraints**
- **Low Latency**: Updates should not block query processing
- **High Throughput**: Handle thousands of updates per second
- **Consistency**: Maintain index correctness during updates
- **Availability**: System remains searchable during updates

#### **Update Types**
1. **Document Addition**: New documents enter collection
2. **Document Deletion**: Remove documents from collection  
3. **Document Modification**: Update existing document content
4. **Bulk Updates**: Process large batches of changes

### Challenges in Incremental Updates

#### **Index Structure Modifications**
```python
# Adding new document may require:
# 1. New terms → Expand vocabulary
# 2. Existing terms → Insert into sorted posting lists
# 3. Statistics updates → Recalculate IDF values
# 4. Storage management → Handle index growth
```

#### **Concurrency Issues**
- **Reader-Writer Problem**: Queries during updates
- **Consistency**: Atomic updates across multiple structures
- **Lock Contention**: Minimize blocking of concurrent operations

---

## Dynamic Index Structures

Data structures optimized for frequent updates while maintaining query efficiency.

### B+ Tree Based Indexes

#### **Structure Benefits**
```python
class BTreeInvertedIndex:
    def __init__(self, degree=100):
        self.term_btree = BPlusTree(degree)  # term -> posting_list_id
        self.posting_storage = PostingListStorage()
        self.doc_metadata = {}
    
    def add_document(self, doc_id, content):
        """Add document with B+ tree updates"""
        terms = self.tokenize(content)
        term_freq = Counter(terms)
        
        # Store document metadata
        self.doc_metadata[doc_id] = {
            'length': len(terms),
            'terms': set(terms)
        }
        
        # Update each term's posting list
        for term, tf in term_freq.items():
            self.update_term_posting_list(term, doc_id, tf)
    
    def update_term_posting_list(self, term, doc_id, tf):
        """Update posting list for a term"""
        # Find existing posting list
        posting_list_id = self.term_btree.search(term)
        
        if posting_list_id is None:
            # New term - create posting list
            posting_list = [Posting(doc_id, tf)]
            posting_list_id = self.posting_storage.create_list(posting_list)
            self.term_btree.insert(term, posting_list_id)
        else:
            # Existing term - update posting list
            posting_list = self.posting_storage.get_list(posting_list_id)
            self.insert_posting_sorted(posting_list, Posting(doc_id, tf))
            self.posting_storage.update_list(posting_list_id, posting_list)
```

#### **Insertion Complexity**
- **Term Lookup**: O(log V) where V = vocabulary size
- **Posting Insertion**: O(log P) where P = posting list length
- **Total**: O(U × log V × log P) for U unique terms

### Skip List Implementation

#### **Probabilistic Balanced Structure**
```python
import random

class SkipListPosting:
    def __init__(self, doc_id, tf, max_level=16):
        self.doc_id = doc_id
        self.tf = tf
        self.forward = [None] * (max_level + 1)
        self.level = self.random_level(max_level)
    
    def random_level(self, max_level):
        level = 0
        while random.random() < 0.5 and level < max_level:
            level += 1
        return level

class SkipListPostingList:
    def __init__(self, max_level=16):
        self.max_level = max_level
        self.header = SkipListPosting(-1, 0, max_level)  # Sentinel
        self.level = 0
    
    def insert(self, doc_id, tf):
        """Insert posting maintaining sorted order"""
        update = [None] * (self.max_level + 1)
        current = self.header
        
        # Find insertion point
        for i in range(self.level, -1, -1):
            while (current.forward[i] and 
                   current.forward[i].doc_id < doc_id):
                current = current.forward[i]
            update[i] = current
        
        # Create new posting
        new_posting = SkipListPosting(doc_id, tf, self.max_level)
        
        # Update pointers
        for i in range(new_posting.level + 1):
            new_posting.forward[i] = update[i].forward[i]
            update[i].forward[i] = new_posting
        
        # Update list level
        if new_posting.level > self.level:
            self.level = new_posting.level
```

### Hash-Based Dynamic Structures

#### **Extendible Hashing for Term Dictionary**
```python
class ExtendibleHashIndex:
    def __init__(self, initial_depth=2):
        self.global_depth = initial_depth
        self.buckets = {}
        self.directory = [None] * (2 ** initial_depth)
        
        # Initialize buckets
        for i in range(2 ** initial_depth):
            bucket = HashBucket(initial_depth)
            self.buckets[i] = bucket
            self.directory[i] = bucket
    
    def insert(self, term, posting_list_ptr):
        """Insert term with dynamic hash table growth"""
        hash_val = hash(term)
        bucket_index = hash_val & ((1 << self.global_depth) - 1)
        bucket = self.directory[bucket_index]
        
        if bucket.insert(term, posting_list_ptr):
            return  # Successful insertion
        
        # Bucket overflow - split required
        if bucket.local_depth == self.global_depth:
            self.expand_directory()
        
        self.split_bucket(bucket_index)
        self.insert(term, posting_list_ptr)  # Retry after split
```

---

## Update Operations

Detailed implementation of fundamental update operations.

### Document Addition

#### **Multi-Step Process**
```python
def add_document_incremental(self, doc_id, content):
    """Add document with full incremental update"""
    
    # Step 1: Tokenize and analyze document
    terms = self.tokenize(content)
    term_freq = Counter(terms)
    doc_length = len(terms)
    
    # Step 2: Update collection statistics
    self.update_collection_stats(doc_length, len(term_freq))
    
    # Step 3: Process each unique term
    for term, tf in term_freq.items():
        self.add_term_occurrence(term, doc_id, tf)
    
    # Step 4: Store document metadata
    self.doc_store[doc_id] = {
        'content': content,
        'length': doc_length,
        'terms': set(term_freq.keys()),
        'timestamp': time.time()
    }
    
    # Step 5: Update derived statistics (IDF, etc.)
    self.update_derived_statistics(term_freq.keys())

def add_term_occurrence(self, term, doc_id, tf):
    """Add single term occurrence to index"""
    if term not in self.inverted_index:
        # New term
        self.inverted_index[term] = SortedList()
        self.term_stats[term] = {'df': 0, 'cf': 0}
    
    # Insert posting in sorted order
    posting = Posting(doc_id, tf)
    self.inverted_index[term].add(posting)
    
    # Update statistics
    self.term_stats[term]['df'] += 1
    self.term_stats[term]['cf'] += tf
```

### Document Deletion

#### **Tombstone Approach**
```python
def delete_document_lazy(self, doc_id):
    """Lazy deletion using tombstones"""
    if doc_id not in self.doc_store:
        return False
    
    # Mark document as deleted
    self.deleted_docs.add(doc_id)
    
    # Update collection statistics
    doc_info = self.doc_store[doc_id]
    self.total_docs -= 1
    self.total_terms -= doc_info['length']
    
    # Note: Posting lists not immediately updated
    # Cleanup happens during query processing or periodic maintenance
    
    return True

def is_document_deleted(self, doc_id):
    """Check if document is deleted"""
    return doc_id in self.deleted_docs
```

#### **Immediate Deletion**
```python
def delete_document_immediate(self, doc_id):
    """Immediate deletion with posting list updates"""
    if doc_id not in self.doc_store:
        return False
    
    doc_info = self.doc_store[doc_id]
    
    # Remove from each term's posting list
    for term in doc_info['terms']:
        posting_list = self.inverted_index[term]
        
        # Find and remove posting
        for i, posting in enumerate(posting_list):
            if posting.doc_id == doc_id:
                del posting_list[i]
                break
        
        # Update term statistics
        self.term_stats[term]['df'] -= 1
        self.term_stats[term]['cf'] -= posting.tf
        
        # Remove term if no documents contain it
        if self.term_stats[term]['df'] == 0:
            del self.inverted_index[term]
            del self.term_stats[term]
    
    # Remove document
    del self.doc_store[doc_id]
    
    # Update collection statistics
    self.total_docs -= 1
    self.total_terms -= doc_info['length']
    
    return True
```

### Document Modification

#### **Differential Update**
```python
def update_document(self, doc_id, new_content):
    """Update document using differential approach"""
    
    if doc_id not in self.doc_store:
        # New document
        return self.add_document_incremental(doc_id, new_content)
    
    # Get old document info
    old_doc = self.doc_store[doc_id]
    old_terms = set(old_doc['terms'])
    
    # Analyze new content
    new_terms = self.tokenize(new_content)
    new_term_freq = Counter(new_terms)
    new_term_set = set(new_term_freq.keys())
    
    # Find differences
    removed_terms = old_terms - new_term_set
    added_terms = new_term_set - old_terms
    common_terms = old_terms & new_term_set
    
    # Remove deleted terms
    for term in removed_terms:
        self.remove_term_occurrence(term, doc_id)
    
    # Add new terms
    for term in added_terms:
        self.add_term_occurrence(term, doc_id, new_term_freq[term])
    
    # Update common terms (frequency may have changed)
    for term in common_terms:
        self.update_term_frequency(term, doc_id, new_term_freq[term])
    
    # Update document storage
    self.doc_store[doc_id] = {
        'content': new_content,
        'length': len(new_terms),
        'terms': new_term_set,
        'timestamp': time.time()
    }
```

---

## Delta Index Approach

Maintain separate indexes for new/updated documents, periodically merging with main index.

### Two-Level Architecture

#### **Main Index + Delta Index**
```python
class DeltaIndexSystem:
    def __init__(self):
        self.main_index = StaticInvertedIndex()      # Large, optimized
        self.delta_index = DynamicInvertedIndex()    # Small, updatable
        self.deleted_docs = set()
        self.last_merge_time = time.time()
        self.merge_threshold = 10000  # Documents
    
    def add_document(self, doc_id, content):
        """Add document to delta index"""
        self.delta_index.add_document(doc_id, content)
        
        # Check if merge is needed
        if self.delta_index.size() > self.merge_threshold:
            self.schedule_merge()
    
    def search(self, query):
        """Search across both indexes"""
        # Search main index
        main_results = self.main_index.search(query)
        
        # Search delta index  
        delta_results = self.delta_index.search(query)
        
        # Merge results, handling deletions
        merged_results = self.merge_search_results(
            main_results, delta_results
        )
        
        return merged_results
    
    def merge_search_results(self, main_results, delta_results):
        """Merge results from both indexes"""
        # Convert to dictionaries for easier merging
        main_dict = {doc_id: score for doc_id, score in main_results}
        delta_dict = {doc_id: score for doc_id, score in delta_results}
        
        # Start with main results
        merged = main_dict.copy()
        
        # Add/update with delta results
        for doc_id, score in delta_dict.items():
            if doc_id in merged:
                # Document exists in both - delta takes precedence
                merged[doc_id] = score
            else:
                # New document in delta
                merged[doc_id] = score
        
        # Remove deleted documents
        for doc_id in self.deleted_docs:
            merged.pop(doc_id, None)
        
        # Convert back to sorted list
        result_list = [(doc_id, score) for doc_id, score in merged.items()]
        result_list.sort(key=lambda x: x[1], reverse=True)
        
        return result_list
```

#### **Periodic Merge Process**
```python
def merge_indexes(self):
    """Merge delta index into main index"""
    print("Starting index merge...")
    
    # Create new main index
    new_main_index = StaticInvertedIndex()
    
    # Get all terms from both indexes
    all_terms = set(self.main_index.get_terms()) | set(self.delta_index.get_terms())
    
    for term in all_terms:
        # Get posting lists from both indexes
        main_postings = self.main_index.get_postings(term) or []
        delta_postings = self.delta_index.get_postings(term) or []
        
        # Merge posting lists
        merged_postings = self.merge_posting_lists(main_postings, delta_postings)
        
        # Remove deleted documents
        filtered_postings = [
            p for p in merged_postings 
            if p.doc_id not in self.deleted_docs
        ]
        
        # Add to new main index
        if filtered_postings:
            new_main_index.add_term_postings(term, filtered_postings)
    
    # Replace main index
    self.main_index = new_main_index
    
    # Clear delta index and deleted documents
    self.delta_index = DynamicInvertedIndex()
    self.deleted_docs.clear()
    
    self.last_merge_time = time.time()
    print("Index merge completed")

def merge_posting_lists(self, main_postings, delta_postings):
    """Merge two sorted posting lists"""
    merged = []
    i, j = 0, 0
    
    while i < len(main_postings) and j < len(delta_postings):
        main_posting = main_postings[i]
        delta_posting = delta_postings[j]
        
        if main_posting.doc_id < delta_posting.doc_id:
            merged.append(main_posting)
            i += 1
        elif main_posting.doc_id > delta_posting.doc_id:
            merged.append(delta_posting)
            j += 1
        else:
            # Same document - delta takes precedence
            merged.append(delta_posting)
            i += 1
            j += 1
    
    # Add remaining postings
    merged.extend(main_postings[i:])
    merged.extend(delta_postings[j:])
    
    return merged
```

---

## Real-time Indexing

Systems that provide immediate searchability of new documents.

### Streaming Updates

#### **Event-Driven Architecture**
```python
import asyncio
from queue import Queue
import threading

class RealTimeIndexer:
    def __init__(self):
        self.index = DynamicInvertedIndex()
        self.update_queue = Queue()
        self.is_running = True
        
        # Start background update processor
        self.update_thread = threading.Thread(target=self.process_updates)
        self.update_thread.start()
    
    def add_document_async(self, doc_id, content):
        """Add document asynchronously"""
        update_event = {
            'type': 'add',
            'doc_id': doc_id,
            'content': content,
            'timestamp': time.time()
        }
        self.update_queue.put(update_event)
    
    def delete_document_async(self, doc_id):
        """Delete document asynchronously"""
        update_event = {
            'type': 'delete',
            'doc_id': doc_id,
            'timestamp': time.time()
        }
        self.update_queue.put(update_event)
    
    def process_updates(self):
        """Background thread to process updates"""
        while self.is_running:
            try:
                # Get update with timeout
                update = self.update_queue.get(timeout=1.0)
                
                if update['type'] == 'add':
                    self.index.add_document(update['doc_id'], update['content'])
                elif update['type'] == 'delete':
                    self.index.delete_document(update['doc_id'])
                elif update['type'] == 'update':
                    self.index.update_document(update['doc_id'], update['content'])
                
                # Mark task as done
                self.update_queue.task_done()
                
            except:
                # Timeout or other error - continue processing
                continue
```

### Lock-Free Updates

#### **Compare-and-Swap Operations**
```python
import threading
from concurrent.futures import ThreadPoolExecutor

class LockFreePostingList:
    def __init__(self):
        self.postings = []
        self.version = 0
        self.lock = threading.RLock()  # Minimal locking for version
    
    def insert_posting(self, new_posting):
        """Insert posting using optimistic concurrency"""
        max_attempts = 10
        
        for attempt in range(max_attempts):
            # Read current state
            current_postings = self.postings.copy()
            current_version = self.version
            
            # Find insertion point
            insert_index = self.find_insertion_point(current_postings, new_posting.doc_id)
            
            # Create new posting list
            new_postings = current_postings.copy()
            new_postings.insert(insert_index, new_posting)
            
            # Attempt atomic update
            if self.compare_and_swap(current_version, new_postings):
                return True
        
        return False  # Failed after max attempts
    
    def compare_and_swap(self, expected_version, new_postings):
        """Atomic compare-and-swap operation"""
        with self.lock:
            if self.version == expected_version:
                self.postings = new_postings
                self.version += 1
                return True
            return False
```

### Memory-Mapped Files

#### **Persistent Real-time Updates**
```python
import mmap
import struct

class MemoryMappedIndex:
    def __init__(self, filename, initial_size=1024*1024*100):  # 100MB
        self.filename = filename
        self.file = open(filename, 'r+b')
        self.mmap = mmap.mmap(self.file.fileno(), 0)
        
        # Index header: [magic_number, version, term_count, free_space_ptr]
        self.header_size = 16
        
        if self.mmap.size() == 0:
            # Initialize new file
            self.mmap.resize(initial_size)
            self.write_header(0x4D4D4944, 1, 0, self.header_size)  # "MMID" magic
    
    def write_header(self, magic, version, term_count, free_ptr):
        """Write index header"""
        header = struct.pack('<IIII', magic, version, term_count, free_ptr)
        self.mmap[0:16] = header
    
    def read_header(self):
        """Read index header"""
        header = struct.unpack('<IIII', self.mmap[0:16])
        return {
            'magic': header[0],
            'version': header[1], 
            'term_count': header[2],
            'free_ptr': header[3]
        }
    
    def add_term_posting(self, term, posting):
        """Add posting to memory-mapped index"""
        header = self.read_header()
        
        # Find term location or allocate new space
        term_offset = self.find_or_create_term(term, header)
        
        # Append posting to term's posting list
        posting_data = struct.pack('<II', posting.doc_id, posting.tf)
        posting_offset = header['free_ptr']
        
        # Write posting data
        self.mmap[posting_offset:posting_offset+8] = posting_data
        
        # Update free space pointer
        header['free_ptr'] += 8
        self.write_header(header['magic'], header['version'], 
                         header['term_count'], header['free_ptr'])
        
        # Ensure data is written to disk
        self.mmap.flush()
```

---

## Study Questions

### Beginner Level
1. What are the main differences between immediate and lazy deletion strategies?
2. How does the delta index approach reduce update costs?
3. Why are B+ trees suitable for dynamic inverted indexes?
4. What challenges arise when updating posting lists concurrently?

### Intermediate Level
1. Compare the trade-offs between tombstone deletion and immediate deletion.
2. How do you handle document updates that change the vocabulary significantly?
3. What are the benefits and drawbacks of lock-free update algorithms?
4. How do you determine when to merge delta indexes with the main index?

### Advanced Level
1. Design a distributed incremental indexing system with consistency guarantees.
2. Analyze the amortized complexity of different dynamic index structures.
3. How would you implement transactional updates across multiple index components?
4. Design a real-time indexing system that guarantees bounded query latency.

### Tricky Questions
1. **Consistency Problem**: How do you ensure searchers see a consistent view during updates?
2. **Memory Pressure**: What happens when incremental updates cause memory fragmentation?
3. **Failure Recovery**: How do you recover from crashes during incremental updates?
4. **Performance Paradox**: When might batch updates be faster than incremental updates?

---

## Code Examples

### Complete Incremental Index System
```python
import threading
import time
from collections import defaultdict, Counter
from queue import Queue, Empty

class IncrementalInvertedIndex:
    def __init__(self):
        self.main_index = defaultdict(list)
        self.delta_index = defaultdict(list)
        self.deleted_docs = set()
        self.doc_store = {}
        self.term_stats = defaultdict(lambda: {'df': 0, 'cf': 0})
        
        # Statistics
        self.total_docs = 0
        self.total_terms = 0
        
        # Concurrency control
        self.read_write_lock = threading.RWLock()
        self.update_queue = Queue()
        
        # Background processing
        self.is_running = True
        self.update_processor = threading.Thread(target=self._process_updates)
        self.update_processor.daemon = True
        self.update_processor.start()
    
    def add_document(self, doc_id, content):
        """Add document (queued for background processing)"""
        update = {
            'type': 'add',
            'doc_id': doc_id,
            'content': content,
            'timestamp': time.time()
        }
        self.update_queue.put(update)
    
    def delete_document(self, doc_id):
        """Delete document (queued for background processing)"""
        update = {
            'type': 'delete',
            'doc_id': doc_id,
            'timestamp': time.time()
        }
        self.update_queue.put(update)
    
    def _process_updates(self):
        """Background thread to process queued updates"""
        while self.is_running:
            try:
                update = self.update_queue.get(timeout=1.0)
                
                if update['type'] == 'add':
                    self._add_document_immediate(update['doc_id'], update['content'])
                elif update['type'] == 'delete':
                    self._delete_document_immediate(update['doc_id'])
                
                self.update_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"Error processing update: {e}")
    
    def _add_document_immediate(self, doc_id, content):
        """Immediately add document to delta index"""
        with self.read_write_lock.writer():
            # Tokenize content
            terms = self._tokenize(content)
            term_freq = Counter(terms)
            
            # Store document
            self.doc_store[doc_id] = {
                'content': content,
                'length': len(terms),
                'terms': set(term_freq.keys()),
                'timestamp': time.time()
            }
            
            # Update delta index
            for term, tf in term_freq.items():
                posting = SimplePosting(doc_id, tf)
                self._insert_posting_sorted(self.delta_index[term], posting)
                
                # Update statistics
                self.term_stats[term]['df'] += 1
                self.term_stats[term]['cf'] += tf
            
            # Update collection statistics
            self.total_docs += 1
            self.total_terms += len(terms)
    
    def _delete_document_immediate(self, doc_id):
        """Immediately delete document (tombstone approach)"""
        with self.read_write_lock.writer():
            if doc_id in self.doc_store:
                # Mark as deleted
                self.deleted_docs.add(doc_id)
                
                # Update statistics
                doc_info = self.doc_store[doc_id]
                self.total_docs -= 1
                self.total_terms -= doc_info['length']
                
                # Update term statistics
                for term in doc_info['terms']:
                    self.term_stats[term]['df'] -= 1
                    # Note: cf update would require scanning posting lists
    
    def search(self, query):
        """Search with read lock"""
        with self.read_write_lock.reader():
            query_terms = self._tokenize(query)
            
            if not query_terms:
                return []
            
            # Get posting lists from both indexes
            result_postings = None
            
            for term in query_terms:
                # Merge postings from main and delta indexes
                main_postings = self.main_index.get(term, [])
                delta_postings = self.delta_index.get(term, [])
                
                merged_postings = self._merge_posting_lists(main_postings, delta_postings)
                
                # Filter deleted documents
                filtered_postings = [
                    p for p in merged_postings 
                    if p.doc_id not in self.deleted_docs
                ]
                
                if result_postings is None:
                    result_postings = filtered_postings
                else:
                    # Intersect with previous results
                    result_postings = self._intersect_postings(result_postings, filtered_postings)
            
            # Convert to document IDs and scores
            results = []
            for posting in result_postings or []:
                if posting.doc_id in self.doc_store:
                    score = self._calculate_score(posting, query_terms)
                    results.append((posting.doc_id, score))
            
            # Sort by score
            results.sort(key=lambda x: x[1], reverse=True)
            return results
    
    def _insert_posting_sorted(self, posting_list, new_posting):
        """Insert posting maintaining sorted order"""
        # Binary search for insertion point
        left, right = 0, len(posting_list)
        
        while left < right:
            mid = (left + right) // 2
            if posting_list[mid].doc_id < new_posting.doc_id:
                left = mid + 1
            else:
                right = mid
        
        posting_list.insert(left, new_posting)
    
    def _merge_posting_lists(self, list1, list2):
        """Merge two sorted posting lists"""
        merged = []
        i, j = 0, 0
        
        while i < len(list1) and j < len(list2):
            if list1[i].doc_id < list2[j].doc_id:
                merged.append(list1[i])
                i += 1
            elif list1[i].doc_id > list2[j].doc_id:
                merged.append(list2[j])
                j += 1
            else:
                # Same document - delta takes precedence
                merged.append(list2[j])
                i += 1
                j += 1
        
        # Add remaining postings
        merged.extend(list1[i:])
        merged.extend(list2[j:])
        
        return merged
    
    def _intersect_postings(self, list1, list2):
        """Intersect two sorted posting lists"""
        result = []
        i, j = 0, 0
        
        while i < len(list1) and j < len(list2):
            if list1[i].doc_id == list2[j].doc_id:
                result.append(list1[i])
                i += 1
                j += 1
            elif list1[i].doc_id < list2[j].doc_id:
                i += 1
            else:
                j += 1
        
        return result
    
    def _calculate_score(self, posting, query_terms):
        """Simple scoring function"""
        # Basic TF score (could be enhanced with IDF)
        return posting.tf
    
    def _tokenize(self, text):
        """Simple tokenization"""
        import re
        text = text.lower()
        return re.findall(r'\b\w+\b', text)
    
    def get_statistics(self):
        """Get index statistics"""
        return {
            'total_docs': self.total_docs,
            'total_terms': self.total_terms,
            'vocabulary_size': len(self.term_stats),
            'main_index_terms': len(self.main_index),
            'delta_index_terms': len(self.delta_index),
            'deleted_docs': len(self.deleted_docs),
            'pending_updates': self.update_queue.qsize()
        }

# Reader-Writer Lock Implementation
import threading

class RWLock:
    def __init__(self):
        self._readers = 0
        self._writers = 0
        self._read_condition = threading.Condition(threading.RLock())
        self._write_condition = threading.Condition(threading.RLock())
    
    def reader(self):
        return self._ReaderLock(self)
    
    def writer(self):
        return self._WriterLock(self)
    
    class _ReaderLock:
        def __init__(self, rw_lock):
            self.rw_lock = rw_lock
        
        def __enter__(self):
            with self.rw_lock._read_condition:
                while self.rw_lock._writers > 0:
                    self.rw_lock._read_condition.wait()
                self.rw_lock._readers += 1
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            with self.rw_lock._read_condition:
                self.rw_lock._readers -= 1
                if self.rw_lock._readers == 0:
                    self.rw_lock._read_condition.notify_all()
    
    class _WriterLock:
        def __init__(self, rw_lock):
            self.rw_lock = rw_lock
        
        def __enter__(self):
            with self.rw_lock._write_condition:
                while self.rw_lock._writers > 0 or self.rw_lock._readers > 0:
                    self.rw_lock._write_condition.wait()
                self.rw_lock._writers += 1
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            with self.rw_lock._write_condition:
                self.rw_lock._writers -= 1
                self.rw_lock._write_condition.notify_all()

# Add to threading module for convenience
threading.RWLock = RWLock

# Example usage
def demo_incremental_index():
    index = IncrementalInvertedIndex()
    
    # Add documents
    docs = [
        (1, "machine learning algorithms"),
        (2, "deep learning neural networks"),
        (3, "artificial intelligence research"),
        (4, "machine learning applications"),
        (5, "neural network architectures")
    ]
    
    for doc_id, content in docs:
        index.add_document(doc_id, content)
    
    # Wait for processing
    time.sleep(1)
    
    # Search
    results = index.search("machine learning")
    print("Search results for 'machine learning':")
    for doc_id, score in results:
        content = index.doc_store[doc_id]['content']
        print(f"  Doc {doc_id} (score: {score}): {content}")
    
    # Delete a document
    index.delete_document(1)
    time.sleep(0.5)
    
    # Search again
    results = index.search("machine learning")
    print("\nAfter deleting doc 1:")
    for doc_id, score in results:
        content = index.doc_store[doc_id]['content']
        print(f"  Doc {doc_id} (score: {score}): {content}")
    
    # Show statistics
    stats = index.get_statistics()
    print(f"\nIndex Statistics: {stats}")

if __name__ == "__main__":
    demo_incremental_index()
```

---

## Key Takeaways
1. **Real-time Capability**: Incremental updates enable immediate document searchability
2. **Concurrency Control**: Reader-writer locks essential for maintaining consistency
3. **Performance Trade-offs**: Balance update speed with query performance
4. **Delta Architecture**: Separate updatable index reduces update complexity
5. **Asynchronous Processing**: Background update processing prevents blocking

---

**Next**: In day2_compression_techniques.md, we'll explore index compression methods to reduce storage requirements and improve cache efficiency.