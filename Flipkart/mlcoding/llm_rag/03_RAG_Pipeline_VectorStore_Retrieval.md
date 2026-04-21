# 🤖 LLM & RAG Coding — Part 3: Full RAG Pipeline
> **Difficulty:** Medium → Hard | **Focus:** Vector Store, Retrieval, Reranking, RAG E2E
> **Flipkart Relevance:** 🔥🔥🔥 — Catalog Q&A, Search, Support ChatBot

---

## THE RAG PIPELINE — FULL ARCHITECTURE

```
           INDEXING PHASE (offline)
           ┌─────────────────────────────────────┐
Documents ─→ [Chunker] → [Embedding Model] → [Vector Store]
           └─────────────────────────────────────┘

           RETRIEVAL PHASE (online)
           ┌─────────────────────────────────────┐
Query ─→ [Embedding] → [ANN Search] → [Reranker] → Top-K Chunks
           └─────────────────────────────────────┘

           GENERATION PHASE (online)
           ┌─────────────────────────────────────┐
[Prompt Builder: Query + Chunks] → [LLM] → Answer
           └─────────────────────────────────────┘
```

---

## 🟡 PROBLEM 1: Vector Store from Scratch (Flat/Exact Search)

### Theory
A vector store stores embedding vectors and enables fast nearest-neighbor search.

**Flat (exact) search:** Compute cosine/dot similarity against ALL stored vectors.
- Complexity: O(n × d) per query
- Perfect recall (no approximation)
- Feasible up to ~millions of vectors for d=768

**For production scale:** Use FAISS/Chroma/Pinecone (ANN algorithms).

### Things to Focus On
- ✅ Cosine similarity = dot product of L2-normalized vectors = most common for text
- ✅ Always L2-normalize before storing (enables dot product == cosine similarity)
- ✅ Metadata filter: filter before or after retrieval? → Pre-filtering more scalable
- ✅ Hybrid search: sparse (BM25) + dense (embedding) scores combined

### Implementation
```python
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import json

@dataclass
class Document:
    """A stored document chunk in the vector store."""
    doc_id: str
    text: str
    embedding: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)

class VectorStore:
    """
    Simple in-memory flat vector store (exact cosine similarity).
    
    For production: replace with FAISS/Pinecone/Chroma/Weaviate.
    Interface remains identical — dependency injection pattern.
    """
    
    def __init__(self, d: int):
        """d: embedding dimension."""
        self.d = d
        self.documents: List[Document] = []
        self.embedding_matrix: Optional[np.ndarray] = None  # (n, d) cache
        self._dirty = True  # Rebuild cache when docs added
    
    def add_document(self, doc_id: str, text: str, 
                      embedding: np.ndarray, metadata: Dict = None) -> None:
        """Add a document to the store."""
        # L2 normalize for cosine similarity via dot product
        norm = np.linalg.norm(embedding)
        normalized_emb = embedding / (norm + 1e-8)
        
        doc = Document(doc_id=doc_id, text=text, 
                        embedding=normalized_emb, 
                        metadata=metadata or {})
        self.documents.append(doc)
        self._dirty = True
    
    def add_documents_batch(self, docs: List[Dict]) -> None:
        """Add many documents efficiently."""
        for doc in docs:
            self.add_document(
                doc['doc_id'], doc['text'], doc['embedding'], 
                doc.get('metadata', {})
            )
    
    def _build_matrix(self) -> None:
        """Build embedding matrix cache for fast batch search."""
        if not self.documents:
            self.embedding_matrix = np.zeros((0, self.d))
        else:
            self.embedding_matrix = np.stack([d.embedding for d in self.documents])
        self._dirty = False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5,
                filter_fn=None) -> List[Dict]:
        """
        Retrieve top-k most similar documents.
        
        query_embedding: (d,) query vector
        filter_fn: optional function(doc) → bool for metadata filtering
        Returns: list of {doc_id, text, score, metadata}
        """
        if self._dirty:
            self._build_matrix()
        
        if len(self.documents) == 0:
            return []
        
        # Apply metadata filter before search
        if filter_fn is not None:
            valid_indices = [i for i, doc in enumerate(self.documents) if filter_fn(doc)]
            if not valid_indices:
                return []
            search_matrix = self.embedding_matrix[valid_indices]
            index_map = valid_indices
        else:
            search_matrix = self.embedding_matrix
            index_map = list(range(len(self.documents)))
        
        # L2 normalize query
        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        
        # Cosine similarity = dot product (since all embeddings are normalized)
        scores = search_matrix @ q_norm  # (n,)
        
        # Top-K via argpartition (O(n) vs O(n log n) for argsort)
        k = min(top_k, len(scores))
        top_indices = np.argpartition(-scores, k-1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]  # Sort top-K
        
        results = []
        for idx in top_indices:
            original_idx = index_map[idx]
            doc = self.documents[original_idx]
            results.append({
                'doc_id':   doc.doc_id,
                'text':     doc.text,
                'score':    float(scores[idx]),
                'metadata': doc.metadata,
            })
        
        return results
    
    def delete_document(self, doc_id: str) -> bool:
        """Remove document from store."""
        original_len = len(self.documents)
        self.documents = [d for d in self.documents if d.doc_id != doc_id]
        self._dirty = True
        return len(self.documents) < original_len
    
    def __len__(self) -> int:
        return len(self.documents)

# Test
d = 64  # Embedding dimension
store = VectorStore(d=d)

# Index some documents
docs_data = [
    {'doc_id': 'doc_1', 'text': 'Flipkart offers free delivery on orders above ₹499', 
     'embedding': np.random.randn(d), 'metadata': {'category': 'shipping', 'source': 'faq'}},
    {'doc_id': 'doc_2', 'text': 'Return policy: 7-day easy returns for electronics',
     'embedding': np.random.randn(d), 'metadata': {'category': 'returns', 'source': 'faq'}},
    {'doc_id': 'doc_3', 'text': 'Track your order using the Flipkart app',
     'embedding': np.random.randn(d), 'metadata': {'category': 'tracking', 'source': 'help'}},
]
store.add_documents_batch(docs_data)

# Query
query_emb = np.random.randn(d)
results = store.search(query_emb, top_k=2)
print(f"Found {len(results)} results:")
for r in results:
    print(f"  [{r['score']:.4f}] {r['doc_id']}: {r['text'][:60]}...")

# Filtered search: only FAQ documents
results_filtered = store.search(
    query_emb, top_k=2,
    filter_fn=lambda doc: doc.metadata.get('source') == 'faq'
)
print(f"\nFiltered to FAQ docs: {len(results_filtered)} results")
```

---

## 🟡 PROBLEM 2: BM25 (Sparse Retrieval)

### Theory
BM25 is a probabilistic sparse retrieval function. Better than TF-IDF for most retrieval tasks.

$$\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t,d)(k_1+1)}{f(t,d) + k_1(1 - b + b \cdot \frac{|d|}{avg\_dl})}$$

- f(t,d) = term frequency in document
- k1 = 1.5 (controls TF saturation)
- b = 0.75 (document length normalization)
- |d| = document length, avg_dl = corpus average length

**BM25 vs TF-IDF:**
- BM25: TF saturates (diminishing returns for repeated terms)
- BM25: normalizes for document length
- Both: sparse (only non-zero for matching terms)

### Implementation
```python
from collections import Counter
import math
from typing import List, Dict

class BM25Retriever:
    """BM25 sparse retrieval from scratch."""
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus = []        # List of tokenized docs
        self.doc_freqs = []     # Term frequency per doc
        self.idf = {}           # IDF scores
        self.doc_lengths = []
        self.avg_dl = 0
        self.N = 0              # Number of documents
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenizer: lowercase, split on non-alphanumeric."""
        return text.lower().split()
    
    def fit(self, documents: List[str]) -> 'BM25Retriever':
        """Build BM25 index from corpus."""
        self.corpus = [self._tokenize(doc) for doc in documents]
        self.N = len(self.corpus)
        self.doc_freqs = [Counter(doc) for doc in self.corpus]
        self.doc_lengths = [len(doc) for doc in self.corpus]
        self.avg_dl = np.mean(self.doc_lengths) if self.doc_lengths else 0
        
        # Compute IDF for each unique term
        all_terms = set(term for doc in self.corpus for term in doc)
        for term in all_terms:
            df = sum(1 for df_dict in self.doc_freqs if term in df_dict)
            # BM25 IDF: log((N - df + 0.5) / (df + 0.5) + 1)
            self.idf[term] = math.log((self.N - df + 0.5) / (df + 0.5) + 1)
        
        return self
    
    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for a query against a specific document."""
        query_terms = self._tokenize(query)
        doc_tf = self.doc_freqs[doc_idx]
        dl = self.doc_lengths[doc_idx]
        
        score = 0.0
        for term in query_terms:
            if term not in self.idf:
                continue
            
            tf = doc_tf.get(term, 0)
            idf = self.idf[term]
            
            # BM25 TF component (saturating)
            tf_component = (tf * (self.k1 + 1)) / \
                          (tf + self.k1 * (1 - self.b + self.b * dl / (self.avg_dl + 1e-8)))
            
            score += idf * tf_component
        
        return score
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve top-k documents by BM25 score."""
        scores = [self.score(query, i) for i in range(self.N)]
        
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [
            {'index': int(i), 'score': scores[i], 
             'text': ' '.join(self.corpus[i])}
            for i in top_indices if scores[i] > 0
        ]

# Test
corpus = [
    "Flipkart shipping policy free delivery 499 rupees",
    "Electronics mobile phones laptops deals offers Flipkart",  
    "Return refund policy easy returns 7 days",
    "Track order status Flipkart app delivery date",
    "Flipkart plus membership free delivery fast shipping",
]

bm25 = BM25Retriever().fit(corpus)
results = bm25.search("free delivery shipping", top_k=3)
print("BM25 results for 'free delivery shipping':")
for r in results:
    print(f"  [{r['score']:.3f}] {r['text']}")
```

---

## 🔴 PROBLEM 3: Hybrid Search — Sparse + Dense Fusion

### Theory
Combine BM25 (sparse, keyword matching) with dense embeddings (semantic similarity):

**Reciprocal Rank Fusion (RRF):**
$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + r(d)}$$

Where r(d) = rank of document d in retrieval system r, k = 60 (default).

**Why RRF over weighted average?**
- Robust to different score scales across retrievers
- No tuning of weights required
- Works well in practice across many tasks

### Implementation
```python
def reciprocal_rank_fusion(ranked_lists: List[List[str]], k: int = 60) -> List[str]:
    """
    Combine multiple ranked lists using RRF.
    
    ranked_lists: list of ranked document ID lists (each sorted by relevance)
    k: RRF smoothing constant (default 60)
    
    Returns: combined ranking of document IDs
    """
    rrf_scores = defaultdict(float)
    
    for ranked_list in ranked_lists:
        for rank, doc_id in enumerate(ranked_list, start=1):
            rrf_scores[doc_id] += 1 / (k + rank)
    
    # Sort by RRF score (descending)
    return sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)

class HybridRetriever:
    """
    Hybrid retriever combining BM25 (sparse) and vector search (dense).
    """
    
    def __init__(self, bm25_retriever: BM25Retriever, 
                 vector_store: VectorStore,
                 k_rrf: int = 60):
        self.bm25 = bm25_retriever
        self.vector_store = vector_store
        self.k_rrf = k_rrf
    
    def retrieve(self, query: str, query_embedding: np.ndarray,
                  top_k: int = 5, fetch_k: int = 20) -> List[Dict]:
        """
        Hybrid retrieval: BM25 + dense, fused via RRF.
        
        fetch_k: retrieve more candidates, then RRF to get top_k
        """
        # Sparse retrieval (BM25)
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        bm25_ranked = [str(r['index']) for r in bm25_results]
        
        # Dense retrieval
        dense_results = self.vector_store.search(query_embedding, top_k=fetch_k)
        dense_ranked = [r['doc_id'] for r in dense_results]
        
        # RRF fusion
        fused_ranking = reciprocal_rank_fusion([bm25_ranked, dense_ranked], k=self.k_rrf)
        
        # Return top-k with scores
        # Build lookup
        dense_lookup = {r['doc_id']: r for r in dense_results}
        final_results = []
        for doc_id in fused_ranking[:top_k]:
            if doc_id in dense_lookup:
                final_results.append(dense_lookup[doc_id])
        
        return final_results

print("Hybrid search with RRF — combining BM25 + semantic search")
```

---

## 🔴 PROBLEM 4: Complete RAG Pipeline

### Implementation
```python
class RAGPipeline:
    """
    End-to-end RAG pipeline.
    Production-grade structure with all key components.
    """
    
    def __init__(self, vector_store: VectorStore,
                 embedding_model,      # Any model with .encode(text) → ndarray
                 llm,                  # Any LLM with .generate(prompt) → str
                 chunker: TextChunker = None,
                 top_k: int = 5,
                 rerank: bool = False):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm = llm
        self.chunker = chunker or TextChunker(chunk_size=512, overlap=50)
        self.top_k = top_k
        self.rerank = rerank
    
    def index_documents(self, documents: List[Dict]) -> int:
        """
        Index a batch of documents.
        
        documents: list of {'id': str, 'text': str, 'metadata': dict}
        Returns: number of chunks indexed
        """
        total_chunks = 0
        
        for doc in documents:
            # Step 1: Chunk the document
            chunks = self.chunker.chunk_by_sentences(doc['text'])
            
            for chunk in chunks:
                # Step 2: Embed each chunk
                chunk_text = chunk['text']
                embedding = self.embedding_model.encode(chunk_text)
                
                # Step 3: Store in vector store
                chunk_id = f"{doc['id']}_chunk_{chunk['chunk_id']}"
                self.vector_store.add_document(
                    doc_id=chunk_id,
                    text=chunk_text,
                    embedding=embedding,
                    metadata={**doc.get('metadata', {}), 
                               'source_doc': doc['id'],
                               'chunk_id': chunk['chunk_id']}
                )
                total_chunks += 1
        
        return total_chunks
    
    def retrieve(self, query: str, filter_fn=None) -> List[Dict]:
        """
        Retrieve relevant chunks for a query.
        """
        # Step 1: Embed query
        query_embedding = self.embedding_model.encode(query)
        
        # Step 2: Vector search
        results = self.vector_store.search(
            query_embedding, top_k=self.top_k, filter_fn=filter_fn
        )
        
        return results
    
    def build_prompt(self, query: str, retrieved_chunks: List[Dict],
                     system_prompt: str = None) -> str:
        """
        Build the prompt for the LLM with retrieved context.
        """
        default_system = """You are a helpful assistant. Answer the user's question using ONLY the provided context.
If the context doesn't contain enough information, say "I don't have enough information to answer this."
Do not make up information."""
        
        system = system_prompt or default_system
        
        # Format retrieved chunks
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk['metadata'].get('source_doc', 'Unknown')
            context_parts.append(f"[Source {i}: {source}]\n{chunk['text']}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""{system}

CONTEXT:
{context}

QUESTION: {query}

ANSWER:"""
        
        return prompt
    
    def answer(self, query: str, filter_fn=None, 
                system_prompt: str = None) -> Dict:
        """
        Full RAG pipeline: retrieve + generate.
        Returns: {answer, retrieved_chunks, prompt}
        """
        # Step 1: Retrieve
        chunks = self.retrieve(query, filter_fn=filter_fn)
        
        if not chunks:
            return {
                'answer': "No relevant information found.",
                'retrieved_chunks': [],
                'prompt': None,
                'has_context': False,
            }
        
        # Step 2: Build prompt
        prompt = self.build_prompt(query, chunks, system_prompt)
        
        # Step 3: Generate
        answer = self.llm.generate(prompt)
        
        return {
            'answer': answer,
            'retrieved_chunks': chunks,
            'prompt': prompt,
            'has_context': True,
        }

# --- Mock implementations for testing ---
class MockEmbeddingModel:
    """Mock embedding model (replace with SentenceTransformers/OpenAI)."""
    def __init__(self, d: int = 64):
        self.d = d
    
    def encode(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % 10000)
        return np.random.randn(self.d)

class MockLLM:
    """Mock LLM (replace with OpenAI/Anthropic/local LLaMA)."""
    def generate(self, prompt: str) -> str:
        # Extract context and return first relevant chunk
        lines = prompt.split('\n')
        for line in lines:
            if len(line) > 50 and 'CONTEXT' not in line and 'QUESTION' not in line:
                return f"Based on the context: {line[:100]}..."
        return "Based on the provided context, I can help answer your question."

# Test the full pipeline
d = 64
embedding_model = MockEmbeddingModel(d=d)
llm = MockLLM()
vector_store = VectorStore(d=d)

rag = RAGPipeline(vector_store, embedding_model, llm, top_k=3)

# Index documents
documents = [
    {'id': 'faq_shipping', 'text': 'Flipkart offers free delivery on all orders above ₹499. Orders below ₹499 have a delivery charge of ₹40. Express delivery is available in select cities for an additional charge.', 'metadata': {'category': 'shipping'}},
    {'id': 'faq_returns', 'text': 'Flipkart has an easy 7-day return policy for electronics. For fashion items, the return window is 30 days. Items must be unused and in original packaging.', 'metadata': {'category': 'returns'}},
    {'id': 'faq_payment', 'text': 'Flipkart accepts UPI, credit cards, debit cards, net banking, and EMI options. Flipkart Pay Later is available for eligible customers.', 'metadata': {'category': 'payment'}},
]

n_chunks = rag.index_documents(documents)
print(f"Indexed {n_chunks} chunks")

# Query
result = rag.answer("What is the minimum order for free delivery?")
print(f"\nAnswer: {result['answer']}")
print(f"Retrieved {len(result['retrieved_chunks'])} chunks")
print(f"Top chunk score: {result['retrieved_chunks'][0]['score']:.4f}")
```

---

## 🔴 PROBLEM 5: Reranking — Cross-Encoder

### Theory
**Bi-encoder (what we've done):** Encode query and doc SEPARATELY → fast but less accurate.

**Cross-encoder:** Feed [QUERY; DOC] together → full attention across both → much more accurate but slow.

**Re-ranking pipeline:**
1. Bi-encoder: retrieve top-K (fast, high recall)
2. Cross-encoder: re-score top-K pairs (slower, high precision)

### Implementation
```python
class CrossEncoderReranker:
    """
    Cross-encoder reranker (mock implementation).
    In production: use sentence-transformers CrossEncoder or Cohere Rerank API.
    """
    
    def __init__(self, model=None):
        self.model = model  # Cross-encoder model (query, doc) → score
    
    def rerank(self, query: str, documents: List[Dict], 
                top_k: int = 3) -> List[Dict]:
        """
        Rerank retrieved documents using cross-encoder scores.
        
        documents: list from bi-encoder retrieval (with 'text' and 'doc_id')
        """
        if not documents:
            return documents
        
        # Score each (query, doc) pair
        scored = []
        for doc in documents:
            if self.model:
                score = self.model.predict([(query, doc['text'])])[0]
            else:
                # Mock: use simple keyword overlap as proxy
                q_words = set(query.lower().split())
                d_words = set(doc['text'].lower().split())
                score = len(q_words & d_words) / (len(q_words | d_words) + 1e-8)
            
            scored.append({**doc, 'rerank_score': score})
        
        # Re-sort by cross-encoder score
        scored.sort(key=lambda x: x['rerank_score'], reverse=True)
        return scored[:top_k]

# Test
reranker = CrossEncoderReranker()  # Using mock scoring
results = rag.retrieve("free delivery minimum order")
print(f"Before reranking: {[r['doc_id'] for r in results]}")

reranked = reranker.rerank("free delivery minimum order", results, top_k=2)
print(f"After reranking:  {[r['doc_id'] for r in reranked]}")
```

---

## 🎯 INTERVIEW FOLLOW-UP QUESTIONS

1. **"When would you use BM25 over dense retrieval?"** → BM25 for: exact keyword matching (product SKUs, model numbers, version codes), low latency (no GPU needed), small corpus (&lt;100k docs). Dense: semantic similarity, paraphrased queries, multilingual. Hybrid always beats either alone.
2. **"How do you handle hallucination in RAG?"** → (a) Confidence threshold: if all retrieved chunks have score &lt;0.7 → hedge answer. (b) Attribution: force LLM to cite which chunk source. (c) NLI-based verification: use a separate NLI model to check if answer is "entailed" by retrieved context. (d) Reduce temperature.
3. **"RAG vs fine-tuning — when to use RAG?"** → RAG: dynamic knowledge (new products, news), domain knowledge that changes frequently, need source citation. Fine-tuning: static behavioral changes (tone, format), capabilities (math, coding), when you have large labeled task-specific dataset.
4. **"How do you evaluate a RAG system?"** → (a) Retrieval: Recall@K (are relevant docs in top-K?), MRR, NDCG. (b) Generation: faithfulness (answer grounded in context), relevance (answer matches query), RAGAS score. (c) End-to-end: user satisfaction, hallucination rate.
