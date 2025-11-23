# Day 53: Agentic RAG (Self-Querying & Filter Extraction)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing Self-Querying Retrieval

We will build a Self-Querying Retriever using LangChain that translates natural language into a structured Qdrant/Chroma filter.

### 1. Defining the Metadata Schema

The LLM needs to know what it can filter on.

```python
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 1. Define Metadata Fields
metadata_field_info = [
    AttributeInfo(
        name="genre",
        description="The genre of the movie",
        type="string or list[string]",
    ),
    AttributeInfo(
        name="year",
        description="The year the movie was released",
        type="integer",
    ),
    AttributeInfo(
        name="rating",
        description="A 1-10 rating for the movie",
        type="float",
    ),
]

document_content_description = "Brief summary of a movie"

# 2. Setup Vector Store (Mock)
llm = ChatOpenAI(temperature=0)
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())

# 3. Create Self-Query Retriever
retriever = SelfQueryRetriever.from_llm(
    llm,
    vectorstore,
    document_content_description,
    metadata_field_info,
    verbose=True
)

# 4. Run
# retriever.invoke("I want to watch sad movies from the 90s rated highly")
```

### 2. Under the Hood: The Query Constructor

What is the LLM actually doing? It's outputting a structured object.
LangChain uses a prompt that looks like this:

```text
Your goal is to structure the user's query to match the request schema.

Data Schema:
genre: string
year: integer
rating: float

User Query: "sad movies from the 90s rated highly"

Structured Request:
```

The LLM outputs:
```json
{
    "query": "sad",
    "filter": {
        "operator": "and",
        "arguments": [
            {"comparator": "eq", "attribute": "genre", "value": "drama"},
            {"comparator": "gte", "attribute": "year", "value": 1990},
            {"comparator": "lt", "attribute": "year", "value": 2000},
            {"comparator": "gt", "attribute": "rating", "value": 8.5}
        ]
    }
}
```
The `SelfQueryRetriever` then translates this JSON into the specific syntax of your Vector DB (e.g., Qdrant Filter, Pinecone Filter).

### 3. Parent Document Retriever (Recursive)

Indexing small chunks, retrieving large chunks.

```python
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Splitters
# Child: Small chunks for vector search (high precision)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
# Parent: Large chunks for context (high coherence)
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)

# 2. Storage
# Vectorstore holds Child chunks
vectorstore = Chroma(embedding_function=OpenAIEmbeddings())
# Docstore holds Parent chunks (Key-Value store)
store = InMemoryStore()

# 3. Retriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# 4. Ingest
# retriever.add_documents(docs)
# When you search, it finds the child, looks up the parent_id, and returns the parent.
```

### 4. Corrective RAG (CRAG) Logic

A simplified implementation of the CRAG flow.

```python
def corrective_rag_node(state):
    query = state['query']
    docs = retrieve(query)
    
    # Evaluate Relevance
    eval_prompt = f"Does this document answer '{query}'? Yes/No.\nDoc: {docs[0].content}"
    score = llm.invoke(eval_prompt)
    
    if "No" in score.content:
        print("Retrieval failed. Falling back to Web Search.")
        web_results = search_web(query)
        return {"context": web_results, "source": "web"}
    else:
        print("Retrieval good.")
        return {"context": docs, "source": "vector_db"}
```

### 5. Handling Date Filters (The Hard Part)

LLMs are bad at "relative dates".
*   *User:* "Last month's reports."
*   *LLM:* Needs to know "Today's Date".
*   **Solution:** Inject `current_date` into the system prompt.
*   **Prompt:** "Today is 2023-10-25. If user says 'last month', filter for `date >= 2023-09-01 AND date <= 2023-09-30`."

### Summary

Agentic RAG is about **control**. Instead of blindly trusting the embedding model, we use the LLM's reasoning capabilities to construct precise queries, filter noise, and verify results. This is essential for enterprise applications where "close enough" isn't good enough.
