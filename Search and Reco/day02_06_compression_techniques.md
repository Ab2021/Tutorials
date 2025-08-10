# Day 2: Index Compression Techniques

## Table of Contents
1. [Introduction to Index Compression](#introduction)
2. [Gap Encoding and Delta Compression](#gap-encoding)
3. [Variable-Byte Encoding](#variable-byte-encoding)
4. [PForDelta Compression](#pfordelta-compression)
5. [Dictionary Compression](#dictionary-compression)
6. [Block-Based Compression](#block-based-compression)
7. [Study Questions](#study-questions)
8. [Code Examples](#code-examples)

---

## Introduction to Index Compression

Index compression reduces storage requirements and improves cache efficiency, crucial for large-scale search systems where indexes can consume terabytes of storage.

### Why Compress Indexes?

#### **Storage Economics**
- **Web-scale indexes**: Google's index estimated at 100+ petabytes
- **Cost reduction**: 50-90% compression saves millions in storage costs
- **Memory efficiency**: More index data fits in RAM
- **Network bandwidth**: Faster index distribution and replication

#### **Performance Benefits**
- **Cache efficiency**: Compressed data has better cache locality
- **I/O reduction**: Less disk reads for same amount of information
- **Decompression speed**: Modern algorithms decompress faster than disk reads
- **Memory pressure**: Reduced memory footprint enables larger working sets

### Compression Challenges

#### **Access Pattern Requirements**
```python
# Index access patterns require different compression strategies:
# 1. Sequential access: Simple compression works well
# 2. Random access: Need block-based or indexed compression
# 3. Range queries: Require efficient decompression of ranges
# 4. Updates: Compression must support incremental changes
```

#### **CPU vs Storage Trade-off**
- **High compression**: Better storage efficiency, higher CPU cost
- **Fast decompression**: Lower compression ratio, faster query processing
- **Adaptive compression**: Different strategies for hot vs cold data

---

## Gap Encoding and Delta Compression

Most fundamental compression technique for sorted sequences like posting lists.

### Delta Encoding Principle

#### **Basic Concept**
Instead of storing absolute document IDs, store differences between consecutive IDs:

```python
# Original posting list (document IDs)
original = [15, 127, 892, 1205, 2367, 3456, 4789]

# Delta encoded (gaps between consecutive IDs)
deltas = [15, 112, 765, 313, 1162, 1089, 1333]

# Observation: Deltas are typically much smaller than original values
# Smaller values require fewer bits to represent
```

#### **Mathematical Foundation**
For a sorted sequence S = [s₁, s₂, ..., sₙ]:
```
delta_encoding(S) = [s₁, s₂ - s₁, s₃ - s₂, ..., sₙ - sₙ₋₁]
```

**Space savings depend on**:
- Document ID distribution
- Collection size vs vocabulary size
- Term frequency patterns

### Delta Encoding Implementation

#### **Encoding Process**
```python
def delta_encode(posting_list):
    """Convert document IDs to delta encoding"""
    if not posting_list:
        return []
    
    deltas = [posting_list[0]]  # First ID stored as-is
    
    for i in range(1, len(posting_list)):
        delta = posting_list[i] - posting_list[i-1]
        deltas.append(delta)
    
    return deltas

def delta_decode(delta_list):
    """Reconstruct original document IDs from deltas"""
    if not delta_list:
        return []
    
    reconstructed = [delta_list[0]]  # First value is absolute
    
    for i in range(1, len(delta_list)):
        next_id = reconstructed[i-1] + delta_list[i]
        reconstructed.append(next_id)
    
    return reconstructed
```

#### **Advanced Delta Strategies**
```python
class AdaptiveDeltaEncoder:
    def __init__(self):
        self.reference_points = []  # Store periodic absolute values
        self.reference_interval = 128  # Store absolute value every 128 positions
    
    def encode_with_references(self, posting_list):
        """Delta encoding with periodic reference points"""
        encoded = []
        references = []
        
        for i, doc_id in enumerate(posting_list):
            if i % self.reference_interval == 0:
                # Store absolute value as reference point
                encoded.append(doc_id)
                references.append(i)
                last_reference = doc_id
            else:
                # Store delta from last reference or previous value
                if i == 1 or (i - 1) % self.reference_interval == 0:
                    delta = doc_id - last_reference
                else:
                    delta = doc_id - posting_list[i-1]
                encoded.append(delta)
        
        return encoded, references
```

### Compression Ratio Analysis

#### **Theoretical Analysis**
For uniformly distributed document IDs in collection of size N:
```python
def estimate_compression_ratio(collection_size, posting_list_length):
    """Estimate compression ratio for delta encoding"""
    
    # Average document ID
    avg_doc_id = collection_size // 2
    
    # Average gap size
    avg_gap = collection_size // posting_list_length
    
    # Bits needed for original IDs
    bits_original = math.ceil(math.log2(collection_size))
    
    # Bits needed for gaps (approximately)
    bits_gap = math.ceil(math.log2(avg_gap))
    
    # Compression ratio
    compression_ratio = bits_gap / bits_original
    
    return compression_ratio, avg_gap, bits_original, bits_gap
```

---

## Variable-Byte Encoding

Efficient encoding scheme for small integers using variable-length byte sequences.

### VByte Encoding Principle

#### **Encoding Scheme**
- Use 7 bits per byte for data
- Use 1 bit per byte as continuation flag
- **Continuation bit = 0**: Last byte of number
- **Continuation bit = 1**: More bytes follow

#### **Example Encoding**
```python
# Number 127 (fits in 7 bits):
# Binary: 1111111
# VByte: 01111111 (single byte, continuation bit = 0)

# Number 128:
# Binary: 10000000  
# VByte: 10000001 00000000 (two bytes, first has continuation bit = 1)

# Number 16384:
# Binary: 100000000000000
# VByte: 10000010 00000000 00000000
```

### VByte Implementation

#### **Encoding Algorithm**
```python
def vbyte_encode(number):
    """Encode integer using variable-byte encoding"""
    if number == 0:
        return bytes([0])
    
    result = []
    
    while number >= 128:
        # Extract 7 bits, set continuation bit
        byte_val = (number % 128) | 0x80  # Set MSB to 1
        result.append(byte_val)
        number //= 128
    
    # Last byte - no continuation bit
    result.append(number)
    
    return bytes(result)

def vbyte_decode(byte_sequence):
    """Decode variable-byte encoded integers"""
    numbers = []
    current_number = 0
    shift = 0
    
    for byte_val in byte_sequence:
        # Extract 7 data bits
        data_bits = byte_val & 0x7F
        current_number |= (data_bits << shift)
        
        # Check continuation bit
        if (byte_val & 0x80) == 0:
            # End of current number
            numbers.append(current_number)
            current_number = 0
            shift = 0
        else:
            # More bytes for this number
            shift += 7
    
    return numbers
```

#### **VByte for Posting Lists**
```python
def compress_posting_list_vbyte(doc_ids):
    """Compress posting list using delta + VByte encoding"""
    if not doc_ids:
        return b''
    
    # Step 1: Delta encode
    deltas = delta_encode(doc_ids)
    
    # Step 2: VByte encode each delta
    compressed = b''
    for delta in deltas:
        compressed += vbyte_encode(delta)
    
    return compressed

def decompress_posting_list_vbyte(compressed_data):
    """Decompress VByte + delta encoded posting list"""
    if not compressed_data:
        return []
    
    # Step 1: VByte decode to get deltas
    deltas = vbyte_decode(compressed_data)
    
    # Step 2: Delta decode to get original document IDs
    doc_ids = delta_decode(deltas)
    
    return doc_ids
```

### Performance Characteristics

#### **Space Efficiency**
```python
def vbyte_space_analysis(max_value):
    """Analyze VByte space requirements"""
    
    ranges = [
        (1, 127, 1),          # 1 byte
        (128, 16383, 2),      # 2 bytes  
        (16384, 2097151, 3),  # 3 bytes
        (2097152, 268435455, 4), # 4 bytes
    ]
    
    for min_val, max_val, bytes_needed in ranges:
        if max_value <= max_val:
            bits_fixed = math.ceil(math.log2(max_value + 1))
            bits_vbyte = bytes_needed * 8
            
            return {
                'fixed_bits': bits_fixed,
                'vbyte_bytes': bytes_needed,
                'vbyte_bits': bits_vbyte,
                'efficiency': bits_fixed / bits_vbyte
            }
    
    return None
```

---

## PForDelta Compression

Patched Frame of Reference Delta compression - optimized for modern CPUs with SIMD instructions.

### PForDelta Principle

#### **Core Concept**
1. **Frame processing**: Process integers in fixed-size blocks (e.g., 128 integers)
2. **Bit packing**: Pack most values using minimum bits required
3. **Exception handling**: Store outliers (patches) separately
4. **SIMD optimization**: Vectorized operations for fast decompression

#### **Algorithm Steps**
```python
def pfordelta_analyze_frame(delta_values, frame_size=128):
    """Analyze frame to determine optimal bit width"""
    
    frame_stats = []
    
    for i in range(0, len(delta_values), frame_size):
        frame = delta_values[i:i+frame_size]
        
        # Find bit requirements for different percentiles
        sorted_frame = sorted(frame)
        
        # Try different bit widths (90% of values should fit)
        percentile_90 = sorted_frame[int(len(sorted_frame) * 0.9)]
        
        bit_width = math.ceil(math.log2(percentile_90 + 1)) if percentile_90 > 0 else 1
        
        # Count exceptions (values that don't fit in bit_width)
        exceptions = [val for val in frame if val >= (1 << bit_width)]
        
        frame_stats.append({
            'start_index': i,
            'bit_width': bit_width,
            'exceptions': exceptions,
            'exception_positions': [j for j, val in enumerate(frame) if val >= (1 << bit_width)]
        })
    
    return frame_stats
```

### PForDelta Implementation

#### **Compression Algorithm**
```python
import struct

class PForDeltaCompressor:
    def __init__(self, frame_size=128):
        self.frame_size = frame_size
    
    def compress(self, integers):
        """Compress integer sequence using PForDelta"""
        # Step 1: Delta encode
        deltas = delta_encode(integers)
        
        # Step 2: Process in frames
        compressed_frames = []
        
        for i in range(0, len(deltas), self.frame_size):
            frame = deltas[i:i+self.frame_size]
            compressed_frame = self.compress_frame(frame)
            compressed_frames.append(compressed_frame)
        
        return self.serialize_frames(compressed_frames)
    
    def compress_frame(self, frame):
        """Compress a single frame"""
        if not frame:
            return {'bit_width': 0, 'packed_data': b'', 'exceptions': []}
        
        # Determine optimal bit width (90th percentile)
        sorted_frame = sorted(frame)
        percentile_idx = min(int(len(sorted_frame) * 0.9), len(sorted_frame) - 1)
        max_normal = sorted_frame[percentile_idx]
        
        bit_width = math.ceil(math.log2(max_normal + 1)) if max_normal > 0 else 1
        bit_width = max(1, bit_width)  # Minimum 1 bit
        
        # Separate normal values and exceptions
        normal_values = []
        exceptions = []
        
        for pos, value in enumerate(frame):
            if value < (1 << bit_width):
                normal_values.append(value)
            else:
                exceptions.append({'position': pos, 'value': value})
                normal_values.append(0)  # Placeholder
        
        # Pack normal values
        packed_data = self.bit_pack(normal_values, bit_width)
        
        return {
            'bit_width': bit_width,
            'packed_data': packed_data,
            'exceptions': exceptions,
            'frame_size': len(frame)
        }
    
    def bit_pack(self, values, bit_width):
        """Pack values using specified bit width"""
        if bit_width == 0:
            return b''
        
        packed_bytes = []
        current_byte = 0
        bits_in_byte = 0
        
        for value in values:
            # Add value to current accumulation
            current_byte |= (value << bits_in_byte)
            bits_in_byte += bit_width
            
            # Flush complete bytes
            while bits_in_byte >= 8:
                packed_bytes.append(current_byte & 0xFF)
                current_byte >>= 8
                bits_in_byte -= 8
        
        # Flush remaining bits
        if bits_in_byte > 0:
            packed_bytes.append(current_byte & 0xFF)
        
        return bytes(packed_bytes)
    
    def decompress(self, compressed_data):
        """Decompress PForDelta encoded data"""
        frames = self.deserialize_frames(compressed_data)
        
        all_deltas = []
        for frame_data in frames:
            frame_deltas = self.decompress_frame(frame_data)
            all_deltas.extend(frame_deltas)
        
        # Delta decode to get original integers
        return delta_decode(all_deltas)
```

### SIMD Optimization

#### **Vectorized Bit Unpacking**
```python
import numpy as np

def simd_bit_unpack(packed_bytes, bit_width, count):
    """SIMD-optimized bit unpacking using NumPy"""
    
    # Convert bytes to bit array
    bit_array = np.unpackbits(np.frombuffer(packed_bytes, dtype=np.uint8))
    
    # Reshape and extract values
    values = []
    bit_pos = 0
    
    for i in range(count):
        if bit_pos + bit_width <= len(bit_array):
            # Extract bit_width bits
            value_bits = bit_array[bit_pos:bit_pos + bit_width]
            
            # Convert to integer
            value = 0
            for j, bit in enumerate(value_bits):
                value |= (int(bit) << j)
            
            values.append(value)
            bit_pos += bit_width
        else:
            break
    
    return values
```

---

## Dictionary Compression

Compress frequently occurring terms and patterns using dictionary-based methods.

### Term Dictionary Compression

#### **Front Coding**
Exploit common prefixes in lexicographically sorted terms:

```python
def front_code_compress(sorted_terms):
    """Compress term dictionary using front coding"""
    if not sorted_terms:
        return []
    
    compressed = []
    
    # First term stored completely
    compressed.append({
        'prefix_length': 0,
        'suffix': sorted_terms[0]
    })
    
    for i in range(1, len(sorted_terms)):
        prev_term = sorted_terms[i-1]
        curr_term = sorted_terms[i]
        
        # Find common prefix length
        prefix_len = 0
        min_len = min(len(prev_term), len(curr_term))
        
        while prefix_len < min_len and prev_term[prefix_len] == curr_term[prefix_len]:
            prefix_len += 1
        
        # Store prefix length and suffix
        suffix = curr_term[prefix_len:]
        compressed.append({
            'prefix_length': prefix_len,
            'suffix': suffix
        })
    
    return compressed

def front_code_decompress(compressed_terms):
    """Decompress front-coded terms"""
    if not compressed_terms:
        return []
    
    terms = []
    
    # First term
    current_term = compressed_terms[0]['suffix']
    terms.append(current_term)
    
    for entry in compressed_terms[1:]:
        prefix_len = entry['prefix_length']
        suffix = entry['suffix']
        
        # Reconstruct term
        prefix = current_term[:prefix_len]
        current_term = prefix + suffix
        terms.append(current_term)
    
    return terms
```

### String Compression for Terms

#### **Huffman Coding for Characters**
```python
import heapq
from collections import defaultdict, Counter

class HuffmanNode:
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right
    
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_codes(text):
    """Build Huffman codes for character compression"""
    
    # Count character frequencies
    char_freq = Counter(text)
    
    # Build priority queue
    heap = [HuffmanNode(char, freq) for char, freq in char_freq.items()]
    heapq.heapify(heap)
    
    # Build Huffman tree
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)
    
    # Generate codes
    root = heap[0] if heap else None
    codes = {}
    
    def generate_codes(node, code=''):
        if node:
            if node.char is not None:
                codes[node.char] = code or '0'  # Handle single character case
            else:
                generate_codes(node.left, code + '0')
                generate_codes(node.right, code + '1')
    
    generate_codes(root)
    return codes, root

def huffman_compress_terms(terms):
    """Compress terms using Huffman coding"""
    
    # Combine all terms for frequency analysis
    all_text = ''.join(terms)
    
    # Build Huffman codes
    codes, tree = build_huffman_codes(all_text)
    
    # Compress each term
    compressed_terms = []
    for term in terms:
        compressed_bits = ''.join(codes[char] for char in term)
        
        # Convert bit string to bytes
        compressed_bytes = []
        for i in range(0, len(compressed_bits), 8):
            byte_bits = compressed_bits[i:i+8].ljust(8, '0')  # Pad if needed
            byte_val = int(byte_bits, 2)
            compressed_bytes.append(byte_val)
        
        compressed_terms.append({
            'compressed': bytes(compressed_bytes),
            'bit_length': len(compressed_bits)
        })
    
    return compressed_terms, tree
```

---

## Block-Based Compression

Compress posting lists in blocks for better compression ratios and efficient random access.

### Block Organization

#### **Fixed-Size Blocks**
```python
class BlockCompressedIndex:
    def __init__(self, block_size=1024):
        self.block_size = block_size  # Documents per block
        self.blocks = {}  # term -> list of compressed blocks
        self.block_metadata = {}  # term -> block metadata
    
    def compress_posting_list(self, term, doc_ids):
        """Compress posting list into blocks"""
        blocks = []
        metadata = []
        
        for i in range(0, len(doc_ids), self.block_size):
            block_docs = doc_ids[i:i + self.block_size]
            
            # Compress block
            compressed_block = self.compress_block(block_docs)
            blocks.append(compressed_block)
            
            # Store metadata for random access
            block_meta = {
                'start_doc_id': block_docs[0],
                'end_doc_id': block_docs[-1],
                'doc_count': len(block_docs),
                'compressed_size': len(compressed_block)
            }
            metadata.append(block_meta)
        
        self.blocks[term] = blocks
        self.block_metadata[term] = metadata
    
    def compress_block(self, doc_ids):
        """Compress a single block of document IDs"""
        # Use delta + VByte compression
        deltas = delta_encode(doc_ids)
        compressed = b''
        
        for delta in deltas:
            compressed += vbyte_encode(delta)
        
        return compressed
    
    def search_blocks(self, term, min_doc_id, max_doc_id):
        """Find relevant blocks for range query"""
        if term not in self.block_metadata:
            return []
        
        relevant_blocks = []
        metadata_list = self.block_metadata[term]
        
        for i, meta in enumerate(metadata_list):
            # Check if block overlaps with query range
            if (meta['start_doc_id'] <= max_doc_id and 
                meta['end_doc_id'] >= min_doc_id):
                
                relevant_blocks.append({
                    'block_index': i,
                    'metadata': meta,
                    'compressed_data': self.blocks[term][i]
                })
        
        return relevant_blocks
```

### Adaptive Block Compression

#### **Variable Block Sizes**
```python
def adaptive_block_compression(posting_list, target_compression_ratio=0.3):
    """Use different block sizes based on compression effectiveness"""
    
    best_blocks = []
    current_pos = 0
    
    while current_pos < len(posting_list):
        # Try different block sizes
        best_block = None
        best_ratio = 0
        
        for block_size in [64, 128, 256, 512, 1024]:
            if current_pos + block_size > len(posting_list):
                block_size = len(posting_list) - current_pos
            
            if block_size == 0:
                break
            
            # Test compression ratio for this block size
            test_block = posting_list[current_pos:current_pos + block_size]
            compressed = compress_posting_list_vbyte(test_block)
            
            original_size = len(test_block) * 4  # 4 bytes per int
            compressed_size = len(compressed)
            ratio = compressed_size / original_size
            
            if ratio <= target_compression_ratio or best_block is None:
                best_block = {
                    'size': block_size,
                    'compressed': compressed,
                    'ratio': ratio
                }
                best_ratio = ratio
        
        best_blocks.append(best_block)
        current_pos += best_block['size']
    
    return best_blocks
```

---

## Study Questions

### Beginner Level
1. Why is delta encoding effective for document ID compression?
2. How does variable-byte encoding handle integers of different sizes?
3. What are the trade-offs between compression ratio and decompression speed?
4. When would you use block-based compression over simple delta encoding?

### Intermediate Level
1. Compare the compression effectiveness of VByte vs PForDelta for different data distributions.
2. How does front coding exploit redundancy in term dictionaries?
3. What factors determine optimal block size in block-based compression?
4. How do you handle updates in compressed indexes?

### Advanced Level
1. Design a hybrid compression scheme that adapts to different posting list characteristics.
2. Analyze the impact of compression on query processing algorithms (intersection, union).
3. How would you implement compressed indexes that support efficient range queries?
4. Design a compression scheme optimized for SSDs vs traditional hard drives.

### Tricky Questions
1. **Compression Paradox**: When might a less compressed index actually perform better?
2. **Update Challenge**: How do you efficiently update a heavily compressed index?
3. **Access Pattern**: How does random vs sequential access affect compression strategy choice?
4. **Memory Hierarchy**: How should compression strategies differ for different levels of the memory hierarchy?

---

## Code Examples

### Complete Compression Framework
```python
import math
import struct
from abc import ABC, abstractmethod

class CompressionStrategy(ABC):
    @abstractmethod
    def compress(self, integers):
        pass
    
    @abstractmethod
    def decompress(self, compressed_data):
        pass

class DeltaVByteCompression(CompressionStrategy):
    def compress(self, integers):
        if not integers:
            return b''
        
        # Delta encode
        deltas = [integers[0]]
        for i in range(1, len(integers)):
            deltas.append(integers[i] - integers[i-1])
        
        # VByte encode
        result = b''
        for delta in deltas:
            result += self.vbyte_encode(delta)
        
        return result
    
    def decompress(self, compressed_data):
        if not compressed_data:
            return []
        
        # VByte decode
        deltas = self.vbyte_decode(compressed_data)
        
        # Delta decode
        result = [deltas[0]]
        for i in range(1, len(deltas)):
            result.append(result[i-1] + deltas[i])
        
        return result
    
    def vbyte_encode(self, number):
        result = []
        while number >= 128:
            result.append((number % 128) | 0x80)
            number //= 128
        result.append(number)
        return bytes(result)
    
    def vbyte_decode(self, data):
        numbers = []
        current = 0
        shift = 0
        
        for byte in data:
            current |= ((byte & 0x7F) << shift)
            if (byte & 0x80) == 0:
                numbers.append(current)
                current = 0
                shift = 0
            else:
                shift += 7
        
        return numbers

class CompressedPostingList:
    def __init__(self, compression_strategy=None):
        self.compression_strategy = compression_strategy or DeltaVByteCompression()
        self.compressed_data = b''
        self.doc_count = 0
        self.min_doc_id = None
        self.max_doc_id = None
    
    def compress_and_store(self, doc_ids):
        """Compress and store document IDs"""
        if not doc_ids:
            return
        
        # Store metadata
        self.doc_count = len(doc_ids)
        self.min_doc_id = min(doc_ids)
        self.max_doc_id = max(doc_ids)
        
        # Compress
        sorted_ids = sorted(doc_ids)
        self.compressed_data = self.compression_strategy.compress(sorted_ids)
    
    def decompress(self):
        """Decompress and return document IDs"""
        return self.compression_strategy.decompress(self.compressed_data)
    
    def intersect_with(self, other_posting_list):
        """Intersect with another compressed posting list"""
        # Decompress both lists
        docs1 = self.decompress()
        docs2 = other_posting_list.decompress()
        
        # Perform intersection
        result = []
        i, j = 0, 0
        
        while i < len(docs1) and j < len(docs2):
            if docs1[i] == docs2[j]:
                result.append(docs1[i])
                i += 1
                j += 1
            elif docs1[i] < docs2[j]:
                i += 1
            else:
                j += 1
        
        # Return compressed result
        compressed_result = CompressedPostingList(self.compression_strategy)
        compressed_result.compress_and_store(result)
        return compressed_result
    
    def get_compression_stats(self):
        """Get compression statistics"""
        if self.doc_count == 0:
            return {}
        
        uncompressed_size = self.doc_count * 4  # 4 bytes per doc ID
        compressed_size = len(self.compressed_data)
        
        return {
            'doc_count': self.doc_count,
            'uncompressed_size': uncompressed_size,
            'compressed_size': compressed_size,
            'compression_ratio': compressed_size / uncompressed_size if uncompressed_size > 0 else 0,
            'space_savings': 1 - (compressed_size / uncompressed_size) if uncompressed_size > 0 else 0
        }

# Example usage and benchmarking
def benchmark_compression():
    import random
    import time
    
    # Generate test data - realistic posting list
    doc_ids = sorted(random.sample(range(1, 1000000), 10000))
    
    print(f"Original data: {len(doc_ids)} document IDs")
    print(f"Range: {min(doc_ids)} to {max(doc_ids)}")
    
    # Test compression
    compressor = CompressedPostingList()
    
    start_time = time.time()
    compressor.compress_and_store(doc_ids)
    compression_time = time.time() - start_time
    
    # Test decompression
    start_time = time.time()
    decompressed = compressor.decompress()
    decompression_time = time.time() - start_time
    
    # Verify correctness
    assert decompressed == doc_ids, "Decompression failed!"
    
    # Show statistics
    stats = compressor.get_compression_stats()
    print(f"\nCompression Statistics:")
    print(f"  Uncompressed size: {stats['uncompressed_size']:,} bytes")
    print(f"  Compressed size: {stats['compressed_size']:,} bytes")
    print(f"  Compression ratio: {stats['compression_ratio']:.3f}")
    print(f"  Space savings: {stats['space_savings']:.1%}")
    print(f"  Compression time: {compression_time:.4f} seconds")
    print(f"  Decompression time: {decompression_time:.4f} seconds")

if __name__ == "__main__":
    benchmark_compression()
```

---

## Key Takeaways
1. **Delta Encoding Foundation**: Most effective compression starts with delta encoding for sorted sequences
2. **Variable-Length Codes**: VByte and similar schemes efficiently handle small integers
3. **Block-Based Benefits**: Block compression enables better ratios and random access
4. **Trade-off Management**: Balance compression ratio, speed, and update complexity
5. **Adaptive Strategies**: Different data characteristics require different compression approaches

---

**Next**: In day2_compression_storage.md, we'll explore storage optimization strategies including caching, tiered storage, and distributed index management.