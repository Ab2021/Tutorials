# Day 55: Data Engineering for LLMs
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Complete Data Pipeline

```python
import hashlib
import re
from collections import Counter
from typing import Iterator, Dict
import langdetect

class LLMDataPipeline:
    def __init__(self, config: Dict):
        self.config = config
        self.seen_hashes = set()
        self.stats = Counter()
    
    def process(self, documents: Iterator[str]) -> Iterator[str]:
        """Process documents through complete pipeline."""
        for doc in documents:
            # Language detection
            if not self._is_target_language(doc):
                self.stats['filtered_language'] += 1
                continue
            
            # Quality filtering
            if not self._passes_quality_checks(doc):
                self.stats['filtered_quality'] += 1
                continue
            
            # Deduplication
            if self._is_duplicate(doc):
                self.stats['filtered_duplicate'] += 1
                continue
            
            # Cleaning
            doc = self._clean_text(doc)
            
            # Toxicity filtering
            if self._is_toxic(doc):
                self.stats['filtered_toxic'] += 1
                continue
            
            # PII filtering
            doc = self._remove_pii(doc)
            
            self.stats['kept'] += 1
            yield doc
    
    def _is_target_language(self, text: str) -> bool:
        """Check if text is in target language."""
        try:
            lang = langdetect.detect(text)
            return lang == self.config.get('target_language', 'en')
        except:
            return False
    
    def _passes_quality_checks(self, text: str) -> bool:
        """Apply quality heuristics."""
        words = text.split()
        
        # Word count
        if not (50 <= len(words) <= 10000):
            return False
        
        # Mean word length
        mean_word_len = sum(len(w) for w in words) / len(words)
        if not (3 <= mean_word_len <= 10):
            return False
        
        # Symbol-to-word ratio
        symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if symbols / len(words) > 0.1:
            return False
        
        # Uppercase ratio
        uppercase = sum(1 for c in text if c.isupper())
        if uppercase / len(text) > 0.3:
            return False
        
        return True
    
    def _is_duplicate(self, text: str) -> bool:
        """Check for exact duplicates."""
        hash_val = hashlib.md5(text.encode()).hexdigest()
        if hash_val in self.seen_hashes:
            return True
        self.seen_hashes.add(hash_val)
        return False
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+', '', text)
        
        # Remove extra punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)
        
        return text.strip()
    
    def _is_toxic(self, text: str) -> bool:
        """Check for toxic content."""
        # Simplified toxicity check (use Perspective API in production)
        toxic_words = ['hate', 'kill', 'violence']  # Placeholder
        text_lower = text.lower()
        return any(word in text_lower for word in toxic_words)
    
    def _remove_pii(self, text: str) -> str:
        """Remove personal identifiable information."""
        # Email
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)
        
        # Phone
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '[PHONE]', text)
        
        # SSN
        text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', text)
        
        return text
```

### 2. MinHash Near-Deduplication

```python
import numpy as np
from datasketch import MinHash, MinHashLSH

class NearDuplicateDetector:
    def __init__(self, threshold=0.8, num_perm=128):
        self.threshold = threshold
        self.num_perm = num_perm
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        self.seen_docs = {}
    
    def add_document(self, doc_id: str, text: str):
        """Add document to LSH index."""
        # Create MinHash
        minhash = MinHash(num_perm=self.num_perm)
        
        # Add shingles (3-grams)
        for shingle in self._get_shingles(text, n=3):
            minhash.update(shingle.encode('utf8'))
        
        # Add to LSH
        self.lsh.insert(doc_id, minhash)
        self.seen_docs[doc_id] = minhash
    
    def is_duplicate(self, text: str) -> bool:
        """Check if document is near-duplicate."""
        # Create MinHash for query
        minhash = MinHash(num_perm=self.num_perm)
        for shingle in self._get_shingles(text, n=3):
            minhash.update(shingle.encode('utf8'))
        
        # Query LSH
        result = self.lsh.query(minhash)
        
        return len(result) > 0
    
    def _get_shingles(self, text: str, n: int = 3):
        """Generate n-gram shingles."""
        words = text.split()
        for i in range(len(words) - n + 1):
            yield ' '.join(words[i:i+n])
```

### 3. Data Mixing

```python
class DataMixer:
    def __init__(self, sources: Dict[str, float]):
        """
        sources: {source_name: weight}
        Example: {'web': 0.6, 'books': 0.2, 'code': 0.2}
        """
        self.sources = sources
        # Normalize weights
        total = sum(sources.values())
        self.sources = {k: v/total for k, v in sources.items()}
    
    def mix(self, datasets: Dict[str, Iterator], total_samples: int):
        """Mix datasets according to weights."""
        # Calculate samples per source
        samples_per_source = {
            source: int(total_samples * weight)
            for source, weight in self.sources.items()
        }
        
        # Sample from each source
        mixed_data = []
        for source, num_samples in samples_per_source.items():
            dataset = datasets[source]
            samples = list(itertools.islice(dataset, num_samples))
            mixed_data.extend(samples)
        
        # Shuffle
        random.shuffle(mixed_data)
        
        return mixed_data
```

### 4. Perplexity-Based Filtering

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class PerplexityFilter:
    def __init__(self, model_name='gpt2', threshold=1000):
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.threshold = threshold
        self.model.eval()
    
    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text."""
        encodings = self.tokenizer(text, return_tensors='pt')
        
        with torch.no_grad():
            outputs = self.model(**encodings, labels=encodings['input_ids'])
            loss = outputs.loss
        
        perplexity = torch.exp(loss).item()
        return perplexity
    
    def should_keep(self, text: str) -> bool:
        """Keep if perplexity below threshold."""
        perplexity = self.compute_perplexity(text)
        return perplexity < self.threshold
```

### 5. BPE Tokenizer Training

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

def train_bpe_tokenizer(texts: Iterator[str], vocab_size: int = 50000):
    """Train BPE tokenizer."""
    # Initialize tokenizer
    tokenizer = Tokenizer(models.BPE())
    
    # Pre-tokenizer (split on whitespace)
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Trainer
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<s>", "</s>", "<unk>", "<mask>"]
    )
    
    # Train
    tokenizer.train_from_iterator(texts, trainer=trainer)
    
    return tokenizer
```

### 6. Data Versioning with DVC

```python
import dvc.api

class DataVersioning:
    def __init__(self, repo_path='.'):
        self.repo_path = repo_path
    
    def add_dataset(self, data_path: str):
        """Add dataset to DVC."""
        import subprocess
        subprocess.run(['dvc', 'add', data_path])
        subprocess.run(['git', 'add', f'{data_path}.dvc'])
    
    def get_dataset(self, data_path: str, version: str = None):
        """Get dataset at specific version."""
        with dvc.api.open(
            data_path,
            repo=self.repo_path,
            rev=version
        ) as f:
            return f.read()
```

### 7. Parallel Processing

```python
from multiprocessing import Pool
from functools import partial

def process_chunk(chunk, pipeline):
    """Process chunk of documents."""
    return list(pipeline.process(chunk))

def parallel_process(documents, pipeline, num_workers=8, chunk_size=1000):
    """Process documents in parallel."""
    # Split into chunks
    chunks = []
    chunk = []
    for doc in documents:
        chunk.append(doc)
        if len(chunk) >= chunk_size:
            chunks.append(chunk)
            chunk = []
    if chunk:
        chunks.append(chunk)
    
    # Process in parallel
    with Pool(num_workers) as pool:
        process_fn = partial(process_chunk, pipeline=pipeline)
        results = pool.map(process_fn, chunks)
    
    # Flatten results
    return [doc for chunk_result in results for doc in chunk_result]
```

### 8. Quality Metrics Dashboard

```python
class DataQualityMetrics:
    def __init__(self):
        self.metrics = {}
    
    def compute_metrics(self, documents):
        """Compute quality metrics for dataset."""
        total_docs = len(documents)
        total_tokens = 0
        unique_tokens = set()
        perplexities = []
        
        for doc in documents:
            tokens = doc.split()
            total_tokens += len(tokens)
            unique_tokens.update(tokens)
        
        self.metrics = {
            'total_documents': total_docs,
            'total_tokens': total_tokens,
            'unique_tokens': len(unique_tokens),
            'avg_doc_length': total_tokens / total_docs,
            'vocabulary_size': len(unique_tokens)
        }
        
        return self.metrics
    
    def print_report(self):
        """Print quality report."""
        print("=== Data Quality Report ===")
        for key, value in self.metrics.items():
            print(f"{key}: {value:,.0f}")
```
