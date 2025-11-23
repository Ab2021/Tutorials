# Day 38: RAG (Retrieval-Augmented Generation) Fundamentals
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Embedding Generation Pipeline

**Text Preprocessing:**
```python
def preprocess_text(text: str) -> str:
    # Remove extra whitespace
    text = " ".join(text.split())
    # Remove special characters (optional)
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    return text.strip()
```

**Chunking Implementation:**
```python
from typing import List

def chunk_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
    
    return chunks
```

**Embedding Generation:**
```python
import openai

def generate_embeddings(texts: List[str], model="text-embedding-3-small"):
    """Generate embeddings for a list of texts."""
    response = openai.Embedding.create(
        input=texts,
        model=model
    )
    
    embeddings = [item['embedding'] for item in response['data']]
    return embeddings
```

### 2. Vector Search Implementation

**Naive Search (Brute Force):**
```python
import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return np.dot(a, b) / (np.linalm.norm(a) * np.linalg.norm(b))

def search_naive(
    query_embedding: np.ndarray,
    doc_embeddings: List[np.ndarray],
    top_k: int = 5
) -> List[int]:
    """Brute force search. O(n) complexity."""
    similarities = [
        cosine_similarity(query_embedding, doc_emb)
        for doc_emb in doc_embeddings
    ]
    
    # Get top K indices
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return top_indices.tolist()
```

**Approximate Nearest Neighbor (FAISS):**
```python
import faiss

def build_faiss_index(embeddings: np.ndarray):
    """Build FAISS index for fast search."""
    dimension = embeddings.shape[1]
    
    # Create index
    index = faiss.IndexFlatIP(dimension)  # Inner Product (for normalized vectors)
    
    # Normalize embeddings
    faiss.normalize_L2(embeddings)
    
    # Add to index
    index.add(embeddings)
    
    return index

def search_faiss(
    query_embedding: np.ndarray,
    index: faiss.Index,
    top_k: int = 5
):
    """Fast approximate search. O(log n) complexity."""
    # Normalize query
    query_embedding = query_embedding.reshape(1, -1)
    faiss.normalize_L2(query_embedding)
    
    # Search
    distances, indices = index.search(query_embedding, top_k)
    
    return indices[0].tolist(), distances[0].tolist()
```

### 3. Complete RAG System

```python
import openai
import numpy as np
from typing import List, Dict

class RAGSystem:
    def __init__(
        self,
        documents: List[str],
        chunk_size: int = 512,
        overlap: int = 50,
        embedding_model: str = "text-embedding-3-small",
        llm_model: str = "gpt-4"
    ):
        self.documents = documents
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        
        # Build index
        self.chunks, self.embeddings, self.index = self._build_index()
    
    def _build_index(self):
        """Chunk documents, generate embeddings, build FAISS index."""
        all_chunks = []
        
        # Chunk all documents
        for doc in self.documents:
            chunks = chunk_text(doc, self.chunk_size, self.overlap)
            all_chunks.extend(chunks)
        
        # Generate embeddings
        embeddings = generate_embeddings(all_chunks, self.embedding_model)
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Build FAISS index
        index = build_faiss_index(embeddings_array)
        
        return all_chunks, embeddings_array, index
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve top K relevant chunks."""
        # Generate query embedding
        query_embedding = generate_embeddings([query], self.embedding_model)[0]
        query_embedding = np.array(query_embedding).astype('float32')
        
        # Search
        indices, scores = search_faiss(query_embedding, self.index, top_k)
        
        # Return chunks
        retrieved_chunks = [self.chunks[i] for i in indices]
        return retrieved_chunks
    
    def generate(self, query: str, context: List[str]) -> str:
        """Generate answer using LLM with retrieved context."""
        # Build prompt
        context_str = "\n\n".join([f"[{i+1}] {chunk}" for i, chunk in enumerate(context)])
        
        prompt = f"""Answer the question based on the provided context. Cite sources using [1], [2], etc.

Context:
{context_str}

Question: {query}

Answer:"""
        
        # Generate
        response = openai.ChatCompletion.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        return response.choices[0].message.content
    
    def query(self, question: str, top_k: int = 3) -> Dict:
        """End-to-end RAG query."""
        # Retrieve
        retrieved_chunks = self.retrieve(question, top_k)
        
        # Generate
        answer = self.generate(question, retrieved_chunks)
        
        return {
            "question": question,
            "answer": answer,
            "sources": retrieved_chunks
        }

# Usage
documents = [
    "The company offers 15 days of paid time off per year...",
    "Employees can work remotely up to 3 days per week...",
    "Health insurance is provided to all full-time employees..."
]

rag = RAGSystem(documents)
result = rag.query("What is the vacation policy?")
print(result["answer"])
print("\nSources:")
for i, source in enumerate(result["sources"]):
    print(f"[{i+1}] {source[:100]}...")
```

### 4. Hybrid Search (BM25 + Dense)

```python
from rank_bm25 import BM25Okapi

class HybridRAG(RAGSystem):
    def __init__(self, *args, alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha  # Weight for dense vs sparse
        
        # Build BM25 index
        tokenized_chunks = [chunk.split() for chunk in self.chunks]
        self.bm25 = BM25Okapi(tokenized_chunks)
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Hybrid retrieval: BM25 + Dense."""
        # Dense retrieval
        query_embedding = generate_embeddings([query], self.embedding_model)[0]
        query_embedding = np.array(query_embedding).astype('float32')
        dense_indices, dense_scores = search_faiss(query_embedding, self.index, len(self.chunks))
        
        # Sparse retrieval (BM25)
        tokenized_query = query.split()
        sparse_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores
        dense_scores_norm = (dense_scores - dense_scores.min()) / (dense_scores.max() - dense_scores.min() + 1e-10)
        sparse_scores_norm = (sparse_scores - sparse_scores.min()) / (sparse_scores.max() - sparse_scores.min() + 1e-10)
        
        # Combine scores
        combined_scores = self.alpha * dense_scores_norm + (1 - self.alpha) * sparse_scores_norm
        
        # Get top K
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        return [self.chunks[i] for i in top_indices]
```

### 5. Reranking

**Cross-Encoder Reranking:**
```python
from sentence_transformers import CrossEncoder

class RerankedRAG(RAGSystem):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def retrieve(self, query: str, top_k: int = 3, rerank_top_n: int = 10) -> List[str]:
        """Retrieve with reranking."""
        # Initial retrieval (get more candidates)
        query_embedding = generate_embeddings([query], self.embedding_model)[0]
        query_embedding = np.array(query_embedding).astype('float32')
        indices, _ = search_faiss(query_embedding, self.index, rerank_top_n)
        
        candidates = [self.chunks[i] for i in indices]
        
        # Rerank
        pairs = [[query, chunk] for chunk in candidates]
        scores = self.reranker.predict(pairs)
        
        # Get top K after reranking
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [candidates[i] for i in top_indices]
```

### 6. Evaluation Framework

```python
def evaluate_rag(
    rag_system: RAGSystem,
    test_questions: List[str],
    ground_truth_answers: List[str]
) -> Dict:
    """Evaluate RAG system."""
    correct = 0
    total = len(test_questions)
    
    for question, gt_answer in zip(test_questions, ground_truth_answers):
        result = rag_system.query(question)
        predicted_answer = result["answer"]
        
        # Simple exact match (in practice, use more sophisticated metrics)
        if gt_answer.lower() in predicted_answer.lower():
            correct += 1
    
    accuracy = correct / total
    
    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total
    }
```
