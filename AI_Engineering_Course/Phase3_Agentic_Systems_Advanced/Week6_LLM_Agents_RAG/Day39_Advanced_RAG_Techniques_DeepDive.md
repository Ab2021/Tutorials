# Day 39: Advanced RAG Techniques
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. HyDE (Hypothetical Document Embeddings) Implementation

**Theory:**
- Question embeddings are far from answer embeddings in vector space.
- Generate a hypothetical answer, embed it, retrieve similar documents.

**Complete Implementation:**
```python
import openai
import numpy as np

class HyDERAG:
    def __init__(self, documents, embedding_model="text-embedding-3-small"):
        self.documents = documents
        self.embedding_model = embedding_model
        self.doc_embeddings = self._embed_documents()
    
    def _embed_documents(self):
        """Embed all documents."""
        response = openai.Embedding.create(
            input=self.documents,
            model=self.embedding_model
        )
        return np.array([item['embedding'] for item in response['data']])
    
    def retrieve_with_hyde(self, query: str, top_k: int = 3):
        """Retrieve using HyDE."""
        # Generate hypothetical answer
        hypothetical_answer = self._generate_hypothetical_answer(query)
        
        # Embed hypothetical answer
        response = openai.Embedding.create(
            input=[hypothetical_answer],
            model=self.embedding_model
        )
        query_embedding = np.array(response['data'][0]['embedding'])
        
        # Retrieve
        similarities = np.dot(self.doc_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [self.documents[i] for i in top_indices]
    
    def _generate_hypothetical_answer(self, query: str) -> str:
        """Generate hypothetical answer to the query."""
        prompt = f"""Write a detailed answer to this question. Be specific and factual.

Question: {query}

Answer:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        
        return response.choices[0].message.content
```

### 2. Self-RAG: Iterative Retrieval with Reflection

**Algorithm:**
```python
class SelfRAG:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
        self.max_iterations = 3
    
    def query(self, question: str):
        """Self-RAG with reflection."""
        context = []
        
        for iteration in range(self.max_iterations):
            # Retrieve
            new_docs = self.retriever.retrieve(question, context)
            context.extend(new_docs)
            
            # Generate
            answer, needs_more = self._generate_with_reflection(question, context)
            
            if not needs_more:
                return answer
        
        return answer
    
    def _generate_with_reflection(self, question: str, context: List[str]):
        """Generate answer and check if more retrieval is needed."""
        context_str = "\n\n".join(context)
        
        prompt = f"""Context:
{context_str}

Question: {question}

Answer the question based on the context. If you need more information, say "NEED_MORE_INFO: <what you need>". Otherwise, provide the answer.

Response:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        answer = response.choices[0].message.content
        
        if "NEED_MORE_INFO" in answer:
            # Extract what's needed
            needed_info = answer.split("NEED_MORE_INFO:")[1].strip()
            return answer, True
        else:
            return answer, False
```

### 3. Reciprocal Rank Fusion (RRF)

**Mathematical Foundation:**
$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

**Implementation:**
```python
def reciprocal_rank_fusion(
    rankings: List[List[int]],
    k: int = 60
) -> List[int]:
    """
    Combine multiple rankings using RRF.
    
    rankings: List of rankings, where each ranking is a list of document indices
    k: Constant (typically 60)
    """
    scores = {}
    
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1 / (k + rank + 1)
    
    # Sort by score
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return [doc_id for doc_id, score in sorted_docs]

# Example usage
bm25_ranking = [0, 2, 1, 3]  # doc IDs in order of relevance
dense_ranking = [1, 0, 3, 2]

fused_ranking = reciprocal_rank_fusion([bm25_ranking, dense_ranking])
print(fused_ranking)  # Combined ranking
```

### 4. Parent-Child Chunking System

**Implementation:**
```python
from typing import List, Dict, Tuple

class ParentChildRAG:
    def __init__(self, documents: List[str]):
        self.parent_chunks, self.child_chunks, self.child_to_parent = \
            self._create_parent_child_chunks(documents)
        
        # Embed only child chunks
        self.child_embeddings = self._embed(self.child_chunks)
    
    def _create_parent_child_chunks(
        self,
        documents: List[str],
        parent_size: int = 512,
        child_size: int = 128
    ) -> Tuple[List[str], List[str], Dict[int, int]]:
        """Create parent and child chunks."""
        parent_chunks = []
        child_chunks = []
        child_to_parent = {}
        
        for doc in documents:
            words = doc.split()
            
            # Create parent chunks
            for i in range(0, len(words), parent_size):
                parent = " ".join(words[i:i + parent_size])
                parent_id = len(parent_chunks)
                parent_chunks.append(parent)
                
                # Create child chunks within this parent
                parent_words = parent.split()
                for j in range(0, len(parent_words), child_size):
                    child = " ".join(parent_words[j:j + child_size])
                    child_id = len(child_chunks)
                    child_chunks.append(child)
                    child_to_parent[child_id] = parent_id
        
        return parent_chunks, child_chunks, child_to_parent
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieve using child chunks, return parent chunks."""
        # Embed query
        query_embedding = self._embed([query])[0]
        
        # Search child chunks
        similarities = np.dot(self.child_embeddings, query_embedding)
        top_child_indices = np.argsort(similarities)[-top_k:][::-1]
        
        # Get corresponding parent chunks
        parent_indices = [self.child_to_parent[i] for i in top_child_indices]
        
        # Deduplicate parents
        unique_parents = list(dict.fromkeys(parent_indices))
        
        return [self.parent_chunks[i] for i in unique_parents]
```

### 5. Contextual Compression with LLM

**Implementation:**
```python
class ContextualCompressor:
    def __init__(self, model="gpt-3.5-turbo"):
        self.model = model
    
    def compress(self, query: str, documents: List[str]) -> List[str]:
        """Extract only relevant information from documents."""
        compressed = []
        
        for doc in documents:
            prompt = f"""Extract ONLY the sentences that are relevant to answering this question. Return them verbatim.

Question: {query}

Document:
{doc}

Relevant sentences (return empty if none):"""
            
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=500
            )
            
            relevant = response.choices[0].message.content.strip()
            if relevant and relevant != "empty":
                compressed.append(relevant)
        
        return compressed
```

### 6. Multi-Query RAG with Deduplication

**Implementation:**
```python
class MultiQueryRAG:
    def __init__(self, retriever):
        self.retriever = retriever
    
    def generate_queries(self, original_query: str, num_queries: int = 3) -> List[str]:
        """Generate multiple variations of the query."""
        prompt = f"""Generate {num_queries} different ways to ask this question. Each should capture a different aspect or use different wording.

Original question: {original_query}

Variations (one per line):"""
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        
        variations = response.choices[0].message.content.strip().split("\n")
        return [original_query] + variations[:num_queries]
    
    def retrieve(self, query: str, top_k: int = 3) -> List[str]:
        """Multi-query retrieval with deduplication."""
        # Generate query variations
        queries = self.generate_queries(query)
        
        # Retrieve for each query
        all_results = []
        for q in queries:
            results = self.retriever.retrieve(q, top_k)
            all_results.extend(results)
        
        # Deduplicate (simple exact match)
        unique_results = list(dict.fromkeys(all_results))
        
        return unique_results[:top_k * 2]  # Return more results
```

### 7. RAG with Citation Tracking

**Implementation:**
```python
class CitedRAG:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator
    
    def query_with_citations(self, question: str):
        """Generate answer with inline citations."""
        # Retrieve
        documents = self.retriever.retrieve(question, top_k=5)
        
        # Build prompt with numbered sources
        context = "\n\n".join([f"[{i+1}] {doc}" for i, doc in enumerate(documents)])
        
        prompt = f"""Answer the question using the provided sources. Add citations using [1], [2], etc. after each claim.

Sources:
{context}

Question: {question}

Answer with citations:"""
        
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        answer = response.choices[0].message.content
        
        return {
            "answer": answer,
            "sources": documents
        }
```
