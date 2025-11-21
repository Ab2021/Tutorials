# Day 30: RAG & LangChain - Theory & Implementation

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Retrieval Augmented Generation, Vector DBs, and Agents

## 1. Theoretical Foundation: Hallucination & Knowledge Cutoff

LLMs have two problems:
1.  **Hallucination**: Making things up.
2.  **Knowledge Cutoff**: Training stopped in 2023.

**RAG (Retrieval Augmented Generation)**:
Instead of relying on internal parameters (Parametric Memory), fetch relevant data from an external source (Non-Parametric Memory) and feed it to the LLM.

## 2. The RAG Pipeline

1.  **Ingestion**: Chunk documents.
2.  **Embedding**: Convert chunks to vectors (OpenAI Ada, SBERT).
3.  **Indexing**: Store in Vector DB (FAISS, Chroma, Pinecone).
4.  **Retrieval**: Query $\to$ Vector. Find Top-K similar chunks.
5.  **Generation**: Prompt = `Context: {Chunks} Question: {Query}`.

## 3. LangChain

A framework to glue LLMs with tools.
*   **Chains**: Sequence of calls.
*   **Agents**: LLM decides which tool to call.
*   **Memory**: Chat history management.

## 4. Implementation: Simple RAG with LangChain

```python
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Load & Split
loader = TextLoader("my_knowledge_base.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 2. Embed & Store
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

# 3. Retrieve & Generate
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=retriever)

query = "What does the document say about X?"
print(qa.run(query))
```

## 5. Advanced RAG

*   **Hybrid Search**: Keyword (BM25) + Semantic (Vector).
*   **Re-ranking**: Retrieve 50 docs, use a Cross-Encoder to rank top 5 accurately.
*   **Query Transformations**: Rewrite user query to be more search-friendly.
