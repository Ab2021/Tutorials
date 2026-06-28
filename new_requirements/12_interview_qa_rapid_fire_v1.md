# AI ENGINEER INTERVIEW Q&A: RAPID FIRE BANK (v1 - CONCEPTUAL)
## The ultimate prep sheet for Italian SME Consulting Roles (No Code)

---

## 1. RAG & CORE ARCHITECTURE

**Q1: A RAG system is retrieving the right documents, but the LLM keeps saying "I don't know." What is the architectural root cause?**
**Answer:** The root cause is likely a mismatch in Chunk Size vs. Context Window, or a strictness configuration in the prompt. If the retrieved chunk is too small, it might contain the keyword but lack the surrounding sentences that give the LLM confidence to answer. Alternatively, the prompt might have a "temperature" set to 0.0 with a hyper-strict instruction to "only answer if 100% certain." The fix is to increase chunk size (or implement Parent-Child chunking) and slightly loosen the confidence boundary in the prompt.

**Q2: Explain Cross-Encoder Reranking and why it is necessary.**
**Answer:** Standard vector search (Bi-Encoders) is fast because it pre-computes vectors and compares them using simple math (Cosine Similarity). However, it misses nuanced context. A Cross-Encoder takes the user's query AND the retrieved document, feeds them together into a deep neural network, and outputs a highly accurate relevance score. Because it is computationally expensive, it cannot be run on 1 million documents. The architecture uses Bi-Encoders to quickly fetch the top 30 documents, and the Cross-Encoder to rerank those 30 into the top 5 for the LLM. It acts as a massive quality multiplier.

**Q3: How do you handle indexing a massive 500-page PDF so a user can ask "Summarize the entire document"?**
**Answer:** Standard RAG fails here because 500 pages exceed the context window, and standard vector search will only retrieve 5 random chunks. To solve this, I implement a Map-Reduce architecture during indexing. I create a hierarchical index. I chunk the document, summarize each chunk, combine those summaries into section summaries, and combine those into a master summary. When the user asks for a full summary, the system retrieves the pre-computed master summary node, rather than attempting to retrieve and read the entire raw text at query time.

---

## 2. VECTOR DATABASES

**Q4: We want to use Pinecone because it's managed, but we have strict GDPR data residency rules. What is your advice?**
**Answer:** Pinecone is a U.S.-based SaaS. While they offer EU regions, an SME with strict data residency rules (like an Italian legal firm) often requires total infrastructure control. I would pivot the architecture to Qdrant. Qdrant is open-source and can be hosted locally via Docker on the client's own hardware, or deployed on a private EU-based cloud server. This provides the exact same vector search capability with absolute legal and physical data sovereignty.

**Q5: What is HNSW and why does it matter?**
**Answer:** HNSW (Hierarchical Navigable Small World) is the underlying algorithm that makes vector databases fast. Instead of comparing a query to every single document (which is impossible at scale), it builds a multi-layered graph. The top layers have few nodes and long connections for fast traversal, while bottom layers contain the dense data. It allows the database to find the closest match in milliseconds using an Approximate Nearest Neighbor search, rather than a brute-force exact search.

**Q6: What happens if your vector database runs out of RAM?**
**Answer:** Because HNSW keeps the graph in memory for speed, running out of RAM causes catastrophic crashes or heavy swapping to disk, killing latency. To fix this architecturally without buying more RAM, I would implement Scalar Quantization (converting 32-bit float vectors to 8-bit integers). This shrinks the RAM footprint by 4x. I would also move the payload (the actual text metadata) to disk-based storage, keeping only the vectors in RAM.

---

## 3. AGENTIC SYSTEMS (LANGGRAPH)

**Q7: Your agent is supposed to check inventory and then send an email. Sometimes it skips checking inventory and just sends a hallucinated email. How do you stop this?**
**Answer:** The agent is operating in an unconstrained ReAct loop. I would refactor it using a State Machine framework like LangGraph. I would physically draw a graph where the `Send_Email` node cannot be reached unless the state object contains a variable `inventory_checked == True`, which is only populated by the `Check_Inventory` node. By enforcing strict topological routing, the LLM is physically barred from skipping mandatory steps.

**Q8: What is the difference between a Tool and a RAG pipeline?**
**Answer:** A RAG pipeline is a fixed process: Retrieve then Generate. A Tool is an arbitrary function exposed to an LLM. Interestingly, a RAG pipeline can *be* a tool. In an Agentic system, I might give the LLM three tools: `Calculator`, `Web_Search`, and `Search_Internal_KnowledgeBase` (which triggers the RAG pipeline). The LLM dynamically decides if the question requires math, recent news, or internal documents.

**Q9: How do you prevent an Agent from getting stuck in an infinite error loop?**
**Answer:** Implement a hard limit on the orchestration layer. In LangGraph, you track a `recursion_depth` variable in the state. If the agent calls a tool, fails, and retries more than 3 times, the orchestrator overrides the LLM, terminates the execution graph, and returns a graceful error to the user. Never rely on the LLM to realize it is stuck.

---

## 4. DEPLOYMENT & MLOPS

**Q10: Why do we use Docker for AI applications?**
**Answer:** AI applications rely on complex, OS-level dependencies (like C++ compilers for vector libraries, or specific versions of PyTorch). If deployed directly onto a client's Windows server, it will almost certainly fail due to environment mismatches. Docker encapsulates the OS, the Python environment, and the code into a single immutable artifact. This guarantees that the exact same environment tested on my laptop is what runs on the client's production server.

**Q11: How do you handle semantic caching in an AI API?**
**Answer:** I deploy a Redis instance alongside the FastAPI application. When a query hits the API, I embed the query. I run a blazing-fast vector search against the Redis cache. If I find a historical query with a cosine similarity > 0.95, I return the cached historical answer instantly. This drops latency to 10ms and API cost to zero. If there is no match, I route to the LLM, generate the answer, and write the new Query-Answer pair to the Redis cache.

**Q12: A client's AI system goes down. The API logs show HTTP 429 Too Many Requests from OpenAI. What is the architectural failure?**
**Answer:** The system lacks an asynchronous queue and rate-limiting middleware. If 50 users query the system simultaneously, the API forwards all 50 to OpenAI, hitting the account's token-per-minute limit. I would architect a queuing system (like Celery). Requests are placed in a queue and processed at a controlled rate. If OpenAI returns a 429, the system must implement Exponential Backoff, pausing for a few seconds before retrying the request, rather than crashing.

---

## 5. DOCUMENT PROCESSING & INGESTION

**Q13: How do you handle extracting data from an Excel file with thousands of rows?**
**Answer:** Do not use RAG. Converting a 10,000-row grid into text destroys its utility and blows out the context window. I would architect a "Data Agent." The Excel file is loaded into a secure, sandboxed Pandas DataFrame. When the user asks "What was Q3 revenue?", the LLM writes a Python script to sum the relevant columns, executes it in the sandbox, and returns the mathematically perfect answer.

**Q14: You have a mix of digital PDFs and scanned PDFs. How do you process them efficiently?**
**Answer:** I build a routing pipeline. The script checks every PDF for a digital text layer. If it exists, it extracts the text directly using PyMuPDF (fast and free). If it detects an image, it routes the PDF to a local Tesseract OCR instance (for basic scans) or Azure Document Intelligence (for complex, dirty scans). This tiered routing saves massive amounts of time and API costs compared to running everything through OCR.

**Q15: How do you handle tables in PDFs during chunking?**
**Answer:** If you use a standard token splitter, it will cut the table in half, ruining the data. I use a layout-aware parser to identify tables. I extract the table and convert it into a Markdown format. I then ensure my chunking algorithm treats the entire Markdown table as a single, atomic chunk. If the table is too large for one chunk, I inject the column headers into every subsequent chunk so the LLM always knows what the numbers mean.
