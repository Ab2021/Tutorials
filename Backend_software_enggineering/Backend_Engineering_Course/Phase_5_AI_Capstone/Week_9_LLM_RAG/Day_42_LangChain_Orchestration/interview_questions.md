# Day 42: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the main benefit of using LangChain over raw API calls?
**Answer:**
*   **Abstraction**: Switch from OpenAI to Anthropic by changing one line of code.
*   **Tooling**: Built-in support for Document Loaders (PDF, CSV), Vector Stores, and Memory.
*   **Chaining**: Easy to compose complex workflows (A -> B -> C).

### Q2: Explain "ConversationSummaryMemory".
**Answer:**
*   **Problem**: `ConversationBufferMemory` keeps *all* messages. Eventually, you hit the Token Limit (and pay a lot).
*   **Solution**: `ConversationSummaryMemory` uses an LLM to periodically summarize the conversation so far.
    *   *Old*: "Hi", "Hello", "My name is Bob", "Nice to meet you Bob".
    *   *Summary*: "The user's name is Bob."
*   **Trade-off**: Saves tokens, but loses specific details.

### Q3: What is LCEL (LangChain Expression Language)?
**Answer:**
*   **Syntax**: A declarative way to compose chains using the pipe `|` operator.
*   **Example**: `chain = prompt | model | output_parser`.
*   **Benefit**: Standard interface (`invoke`, `stream`, `batch`) for all chains.

---

## Scenario-Based Questions

### Q4: You are building a chatbot that needs to answer questions about a 500-page PDF. How do you structure the chain?
**Answer:**
1.  **Loader**: `PyPDFLoader` to read the file.
2.  **Splitter**: `RecursiveCharacterTextSplitter` to chunk it into 1000-token pieces.
3.  **Vector Store**: Embed and store chunks in Pinecone/Chroma.
4.  **Retrieval Chain**:
    *   User Query -> Embed -> Search Vector Store.
    *   Retrieve Top 3 Chunks.
    *   Prompt: "Answer the query using these chunks: ..."

### Q5: Your chain fails because the OpenAI API is rate-limited. How do you handle it?
**Answer:**
*   **Retry**: Use LangChain's built-in `max_retries` configuration.
*   **Backoff**: Implement exponential backoff.
*   **Fallback**: Configure a `FallbackLLM` (e.g., if GPT-4 fails, try GPT-3.5 or Anthropic).

---

## Behavioral / Role-Specific Questions

### Q6: A developer says "LangChain is too heavy/complex, let's just write Python". Do you agree?
**Answer:**
*   **It depends**.
*   **Agree**: For a simple "Chat with GPT" app, raw API is fine. LangChain adds overhead.
*   **Disagree**: For a RAG app with PDF parsing, vector search, and memory, writing it from scratch is reinventing the wheel. LangChain provides standard interfaces for all these hard parts.
