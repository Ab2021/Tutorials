# Day 52: RAG with Tools (Query Decomposition & Routing)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing Advanced RAG Patterns

We will implement a **Query Router** and a **Sub-Question Engine** using LlamaIndex and LangChain concepts.

### 1. The LLM Router (Selector)

We want to route queries between a `VectorStore` (Docs) and a `SQLDatabase` (Stats).

```python
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

class Route(BaseModel):
    destination: str = Field(..., description="The tool to use: 'vector_db' or 'sql_db'")
    reasoning: str = Field(..., description="Why you chose this tool")

system_prompt = """
You are a router. 
Use 'vector_db' for questions about policies, history, or text.
Use 'sql_db' for questions about numbers, counts, or tables.
"""

llm = ChatOpenAI(model="gpt-3.5-turbo")
router_chain = llm.with_structured_output(Route)

def route_query(query: str):
    route = router_chain.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ])
    return route

# Usage
# print(route_query("How do I apply for leave?")) -> destination='vector_db'
# print(route_query("How many employees are on leave?")) -> destination='sql_db'
```

### 2. Query Decomposition (Manual Implementation)

Breaking a complex question into sub-questions.

```python
class SubQuestions(BaseModel):
    questions: List[str] = Field(..., description="List of simple sub-questions")

decomposer = llm.with_structured_output(SubQuestions)

def answer_complex_query(complex_query, retrieval_tool):
    # 1. Decompose
    plan = decomposer.invoke(f"Break down this question: {complex_query}")
    print(f"Plan: {plan.questions}")
    
    answers = []
    # 2. Execute Sub-Questions
    for q in plan.questions:
        # Assume retrieval_tool returns a string answer
        ans = retrieval_tool(q)
        answers.append(f"Q: {q}\nA: {ans}")
        
    # 3. Synthesize
    context = "\n\n".join(answers)
    final_answer = llm.invoke(f"Answer the original question based on these findings:\n{context}\nOriginal Question: {complex_query}")
    return final_answer.content

# Usage
# answer_complex_query("Compare the battery life of iPhone 15 and Pixel 8", search_tool)
# Plan: ["What is the battery life of iPhone 15?", "What is the battery life of Pixel 8?"]
```

### 3. HyDE (Hypothetical Document Embeddings)

Implementing HyDE from scratch.

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()

def hyde_search(query, vector_store):
    # 1. Generate Hypothetical Answer
    hypothesis = llm.invoke(f"Write a hypothetical passage that answers this question: {query}").content
    print(f"Hypothesis: {hypothesis}")
    
    # 2. Embed Hypothesis
    hypo_vector = embeddings.embed_query(hypothesis)
    
    # 3. Search using Hypothesis Vector
    docs = vector_store.similarity_search_by_vector(hypo_vector, k=5)
    return docs

# Why it works: The hypothesis might be factually wrong, but it will use the 
# right vocabulary ("vectors", "embeddings", "cosine") that matches the documents.
```

### 4. LlamaIndex SubQuestionQueryEngine

LlamaIndex has this built-in.

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# Assume we have two engines: engine_2022 and engine_2023
tool_2022 = QueryEngineTool(
    query_engine=engine_2022,
    metadata=ToolMetadata(name="docs_2022", description="Financials for 2022")
)
tool_2023 = QueryEngineTool(
    query_engine=engine_2023,
    metadata=ToolMetadata(name="docs_2023", description="Financials for 2023")
)

query_engine = SubQuestionQueryEngine.from_defaults(
    query_engine_tools=[tool_2022, tool_2023],
    use_async=True,
)

# response = query_engine.query("Compare revenue in 2022 vs 2023")
# It will auto-generate sub-queries to the respective tools.
```

### 5. Self-RAG (Retrieval Augmented Generation with Self-Reflection)

A simplified loop.

```python
def self_rag(query):
    docs = retrieve(query)
    
    # Grade Relevance
    relevant_docs = []
    for doc in docs:
        grade = llm.invoke(f"Is this doc relevant to '{query}'? Yes/No.\nDoc: {doc.page_content}")
        if "Yes" in grade.content:
            relevant_docs.append(doc)
            
    if not relevant_docs:
        # Rewrite query
        new_query = llm.invoke(f"The search for '{query}' failed. Suggest a better keyword search.").content
        return self_rag(new_query) # Recursive
        
    # Generate
    return generate(query, relevant_docs)
```

### Summary

By treating retrieval as a programmable step, we can overcome the limitations of semantic search.
*   **Routing** optimizes cost/accuracy.
*   **Decomposition** handles complexity.
*   **HyDE** handles ambiguity.
*   **Self-Correction** handles noise.
