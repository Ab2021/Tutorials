# 🔗 LangChain — Technical Deep Dive & Interview Q&A
### Optum Sr. AI/ML Engineer — Domain Round Preparation

---

## PART 1: LANGCHAIN ARCHITECTURE INTERNALS

### 1.1 Core Abstraction Layers

```
LANGCHAIN STACK (Bottom → Top)
─────────────────────────────────────────────────────
Layer 5: CHAINS / AGENTS        ← Compose everything together
Layer 4: MEMORY                 ← State persistence across turns
Layer 3: TOOLS / RETRIEVERS     ← External capabilities
Layer 2: PROMPT TEMPLATES       ← Structured LLM inputs
Layer 1: LLMs / CHAT MODELS     ← Model wrappers (OpenAI, Bedrock, Vertex)
Layer 0: SCHEMA / RUNNABLES     ← Base interfaces (LCEL primitives)
─────────────────────────────────────────────────────
```

### 1.2 LCEL — LangChain Expression Language (Critical to Know)

LCEL uses the `|` pipe operator to compose Runnables:

```python
# Every component in LCEL is a Runnable with:
# .invoke(input)  → synchronous single call
# .batch(inputs)  → parallel batch calls
# .stream(input)  → streaming output tokens
# .astream(input) → async streaming

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

# --- Basic LCEL chain ---
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a clinical fraud analyst. Be concise and factual."),
    ("human", "Analyze this claim for fraud indicators:\n{claim_text}")
])

llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
parser = StrOutputParser()

# Compose chain with pipe operator
chain = prompt | llm | parser

# Invoke
result = chain.invoke({"claim_text": "Patient claims surgery on 2024-01-01 but hospital records show..."})

# Stream (critical for UX — shows tokens as they generate)
for chunk in chain.stream({"claim_text": "..."}):
    print(chunk, end="", flush=True)

# --- RunnablePassthrough: pass input unchanged to next step ---
rag_chain = (
    {
        "context": retriever | format_docs,   # retriever runs on input, formats docs
        "question": RunnablePassthrough()      # passes original question unchanged
    }
    | prompt
    | llm
    | parser
)

# --- RunnableLambda: wrap any Python function as a Runnable ---
def validate_claim_length(claim: str) -> str:
    if len(claim) < 50:
        raise ValueError("Claim text too short for analysis")
    return claim

chain_with_validation = (
    RunnableLambda(validate_claim_length)
    | prompt
    | llm
    | parser
)
```

### 1.3 Prompt Templates — All Types

```python
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder
)

# --- 1. Simple string template ---
template = PromptTemplate.from_template(
    "Summarize the following clinical note in 3 bullet points:\n{note}"
)

# --- 2. Chat prompt with system + human ---
chat_template = ChatPromptTemplate.from_messages([
    ("system", "You are a {specialty} specialist. Follow HIPAA guidelines."),
    ("human", "{query}")
])

# --- 3. Few-shot chat template (teach format via examples) ---
examples = [
    {
        "claim": "Patient submitted 3 claims for same procedure same day.",
        "analysis": '{"risk": "HIGH", "indicator": "duplicate_billing", "confidence": 0.95}'
    },
    {
        "claim": "Lab test ordered but no diagnosis code matches test type.",
        "analysis": '{"risk": "MEDIUM", "indicator": "procedure_diagnosis_mismatch", "confidence": 0.72}'
    }
]

example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{claim}"),
    ("ai", "{analysis}")
])

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze insurance claims for fraud. Output JSON only."),
    few_shot_prompt,   # inject examples
    ("human", "{claim}")
])

# --- 4. MessagesPlaceholder for conversation history ---
memory_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a clinical assistant."),
    MessagesPlaceholder(variable_name="chat_history"),  # injects conversation history
    ("human", "{input}")
])
```

### 1.4 Output Parsers

```python
from langchain_core.output_parsers import (
    StrOutputParser,
    JsonOutputParser,
    PydanticOutputParser
)
from pydantic import BaseModel, Field
from typing import Literal

# --- Pydantic parser: enforce typed structured output ---
class FraudAssessment(BaseModel):
    risk_level: Literal["LOW", "MEDIUM", "HIGH", "CRITICAL"] = Field(
        description="Overall fraud risk level"
    )
    indicators: list[str] = Field(
        description="List of specific fraud indicators identified"
    )
    confidence: float = Field(
        description="Confidence score between 0.0 and 1.0",
        ge=0.0, le=1.0
    )
    recommended_action: Literal["AUTO_APPROVE", "FLAG_REVIEW", "ESCALATE_SIU"] = Field(
        description="Recommended disposition"
    )
    supporting_evidence: list[str] = Field(
        description="Specific quotes from claim supporting the assessment"
    )

parser = PydanticOutputParser(pydantic_object=FraudAssessment)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Analyze claims for fraud. {format_instructions}"),
    ("human", "{claim_text}")
]).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser
result: FraudAssessment = chain.invoke({"claim_text": "..."})
print(result.risk_level)          # "HIGH"
print(result.confidence)          # 0.87
print(result.recommended_action)  # "ESCALATE_SIU"
```

---

## PART 2: RETRIEVAL & RAG IN LANGCHAIN

### 2.1 Complete RAG Pipeline

```python
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.runnables import RunnablePassthrough

# ── STEP 1: Load documents ──────────────────────────────────────────
loader = DirectoryLoader("./claims_kb/", glob="**/*.pdf", loader_cls=PyPDFLoader)
docs = loader.load()

# ── STEP 2: Chunk documents ─────────────────────────────────────────
# RecursiveCharacterTextSplitter respects sentence/paragraph boundaries
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # characters per chunk
    chunk_overlap=200,     # overlap prevents cutting context at boundaries
    separators=["\n\n", "\n", ". ", " ", ""]  # tries these in order
)
chunks = splitter.split_documents(docs)

# ── STEP 3: Embed and store ─────────────────────────────────────────
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = FAISS.from_documents(chunks, embeddings)

# Save/load index
vectorstore.save_local("./fraud_kb_index")
vectorstore = FAISS.load_local("./fraud_kb_index", embeddings)

# ── STEP 4: Hybrid Retriever (BM25 + Dense) ────────────────────────
# Critical for healthcare: ICD-10 codes need exact match (BM25)
# while clinical context needs semantic match (dense)
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 5

dense_retriever = vectorstore.as_retriever(
    search_type="mmr",        # Maximal Marginal Relevance: diverse + relevant
    search_kwargs={"k": 5, "fetch_k": 20}
)

hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.3, 0.7]        # 30% BM25, 70% semantic
)

# ── STEP 5: RAG Chain ───────────────────────────────────────────────
def format_docs(docs):
    return "\n\n---\n\n".join([
        f"[Source: {d.metadata.get('source', 'Unknown')}, "
        f"Page: {d.metadata.get('page', 'N/A')}]\n{d.page_content}"
        for d in docs
    ])

rag_prompt = ChatPromptTemplate.from_template("""
You are a clinical fraud investigator at Optum.
Using ONLY the retrieved context below, assess the fraud risk of the input claim.
If the context is insufficient, state: INSUFFICIENT_EVIDENCE.

Retrieved Context:
{context}

Claim to Analyze:
{question}

Respond in JSON: {{"risk_level": "...", "indicators": [...], "evidence": [...], "confidence": 0.0}}
""")

rag_chain = (
    {
        "context": hybrid_retriever | format_docs,
        "question": RunnablePassthrough()
    }
    | rag_prompt
    | ChatOpenAI(model="gpt-4o", temperature=0.0)
    | JsonOutputParser()
)

result = rag_chain.invoke("Patient submitted claims for both inpatient and outpatient on same date...")
```

### 2.2 Advanced Retrieval Strategies

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.multi_query import MultiQueryRetriever

# ── Contextual Compression: extract relevant passages only ──────────
compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=dense_retriever
)
# Retrieves docs then extracts only the relevant sentences → cleaner context

# ── Multi-Query Retrieval: generate multiple query variations ───────
multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=dense_retriever,
    llm=llm
)
# Generates 3-5 variants of the query, retrieves for each, deduplicates
# Good for: queries that may be phrased in different clinical terminologies
```

---

## PART 3: MEMORY TYPES

```python
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory
)

# ── Buffer Memory: stores full conversation history ─────────────────
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True          # returns Message objects, not strings
)

# ── Window Memory: stores last k turns only (controls context length) ─
window_memory = ConversationBufferWindowMemory(
    k=5,                          # keep last 5 exchanges
    memory_key="chat_history",
    return_messages=True
)

# ── Summary Memory: LLM summarizes old turns to save tokens ─────────
summary_memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o-mini"),  # cheap model for summarization
    memory_key="chat_history",
    return_messages=True
)
# Best for: long clinical consultations where full history exceeds context limit
```

---

## PART 4: TOOLS & AGENTS

### 4.1 Defining Custom Tools

```python
from langchain.tools import tool, StructuredTool
from langchain_core.tools import BaseTool
from pydantic import BaseModel

# ── Decorator-based tool ────────────────────────────────────────────
@tool
def lookup_claim_history(member_id: str, months: int = 12) -> str:
    """
    Retrieves the claim history for a member from the claims database.
    Use this when you need to check a member's historical claim patterns.
    
    Args:
        member_id: The unique member identifier (format: MBR-XXXXXXXX)
        months: Number of months of history to retrieve (default: 12)
    """
    # In production: query BigQuery / claims DB
    return f"[Claims DB] Member {member_id}: 3 claims in last {months} months. " \
           f"Total amount: $12,450. No prior fraud flags."

@tool
def check_provider_sanctions(npi: str) -> str:
    """
    Checks if a healthcare provider (identified by NPI) has any OIG sanctions,
    exclusions, or prior fraud investigations.
    Use this when a provider NPI appears in a suspicious claim.
    """
    # In production: query OIG LEIE database
    return f"[OIG Check] NPI {npi}: No active exclusions. 1 advisory action in 2021."

@tool
def calculate_icd_procedure_validity(icd_code: str, procedure_code: str) -> str:
    """
    Validates whether a procedure (CPT code) is clinically appropriate
    for a given diagnosis (ICD-10 code). Returns validity and explanation.
    """
    # In production: query clinical coding rules database
    return f"[Coding Validation] ICD {icd_code} + CPT {procedure_code}: VALID combination."

# ── Structured tool for complex inputs ─────────────────────────────
class ClaimLookupInput(BaseModel):
    claim_id: str = Field(description="The claim identifier")
    include_attachments: bool = Field(default=False, description="Include claim attachments")

def lookup_claim_details(claim_id: str, include_attachments: bool = False) -> dict:
    """Fetches full claim details from the claims management system."""
    return {"claim_id": claim_id, "status": "PENDING_REVIEW", "amount": 5430.00}

claim_tool = StructuredTool.from_function(
    func=lookup_claim_details,
    name="lookup_claim_details",
    description="Fetches complete claim details. Use when you need full claim information.",
    args_schema=ClaimLookupInput
)
```

### 4.2 Agent Types & When to Use Each

```python
from langchain.agents import create_react_agent, create_tool_calling_agent, AgentExecutor
from langchain import hub

# ── ReAct Agent (Reason + Act) ──────────────────────────────────────
# Pattern: Thought → Action → Observation → Thought → ...
# Best for: Complex reasoning tasks needing intermediate steps visible
# Weakness: No structured output, relies on LLM to parse tool calls correctly

react_prompt = hub.pull("hwchase17/react")  # Standard ReAct prompt
tools = [lookup_claim_history, check_provider_sanctions, calculate_icd_procedure_validity]

react_agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
executor = AgentExecutor(
    agent=react_agent,
    tools=tools,
    verbose=True,          # Shows thought/action/observation loop
    max_iterations=10,     # Prevents infinite loops
    handle_parsing_errors=True,   # Graceful handling of malformed LLM output
    early_stopping_method="generate"
)

result = executor.invoke({
    "input": "Investigate claim CLM-12345 for member MBR-78901. "
             "Check their history and validate the provider NPI 1234567890."
})

# ── Tool Calling Agent (preferred for modern LLMs) ─────────────────
# Uses native OpenAI function calling / Anthropic tool use
# More reliable than ReAct — structured JSON tool calls, not text parsing
# Best for: Production systems with modern LLMs

tool_calling_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a fraud investigator. Use available tools to investigate claims thoroughly."),
    MessagesPlaceholder("chat_history", optional=True),
    ("human", "{input}"),
    MessagesPlaceholder("agent_scratchpad")   # where tool results accumulate
])

tool_agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=tool_calling_prompt)
tool_executor = AgentExecutor(
    agent=tool_agent,
    tools=tools,
    verbose=True,
    max_iterations=8,
    return_intermediate_steps=True   # captures full tool call trace for evaluation
)
```

---

## PART 5: INTERVIEW Q&A — LANGCHAIN TECHNICAL

---

**Q1: What is LCEL and why was it introduced?**

> "LCEL (LangChain Expression Language) is a declarative pipeline composition system using the `|` pipe operator. It was introduced to address problems with the original Chain classes:
>
> 1. **Streaming support:** LCEL natively supports `.stream()` — output tokens flow through the entire pipeline as generated, giving users real-time feedback. Old chains buffered the full response first.
> 2. **Async support:** Every LCEL component has `.ainvoke()`, `.astream()`, `.abatch()` — critical for production APIs serving many users concurrently.
> 3. **Parallelism:** LCEL automatically runs dict branches in parallel: `{'context': retriever, 'question': RunnablePassthrough()}` — retriever runs while question passes through simultaneously.
> 4. **Observability:** Every step is traceable via LangSmith without code changes.
> 5. **Composability:** Any Runnable can compose with any other — chains, agents, tools are all interchangeable.
>
> In production at Optum, LCEL streaming is critical for clinical note summarization — physicians see summary building in real-time rather than waiting 10-15 seconds for a large response."

---

**Q2: How does LangChain's retriever interface work? What are the key retriever types?**

> "All retrievers implement `BaseRetriever` with a single method: `get_relevant_documents(query: str) -> List[Document]`. This abstraction means you can swap retrieval strategies without changing the chain.
>
> Key types:
> - **VectorStoreRetriever:** Cosine/MMR similarity search against a vector store. Most common.
> - **BM25Retriever:** Sparse keyword retrieval — fast, exact-match for codes and IDs. No embeddings needed.
> - **EnsembleRetriever:** Combines multiple retrievers with Reciprocal Rank Fusion (RRF) — takes rank from each retriever and combines. Better than picking one.
> - **MultiQueryRetriever:** Generates N rephrasings of the query, retrieves for each, deduplicates — increases recall for ambiguous queries.
> - **ContextualCompressionRetriever:** Wraps another retriever, then uses an LLM to extract only relevant passages from retrieved docs — reduces noise in context.
> - **SelfQueryRetriever:** LLM parses natural language into a structured metadata filter + semantic query — 'Claims from provider NPI 1234 in 2024' becomes a metadata filter + embedding query.
>
> For healthcare claims: I'd use EnsembleRetriever(BM25 + dense) because ICD-10 codes need exact match and clinical context needs semantic understanding."

---

**Q3: What are the failure modes of LangChain agents in production?**

> "Five major failure modes:
>
> 1. **Tool call parsing failures (ReAct):** The LLM produces malformed tool call syntax. ReAct agents parse text output — any deviation breaks the loop. Solution: Use Tool Calling agents with function-calling LLMs instead; set `handle_parsing_errors=True` as a safety net.
>
> 2. **Infinite loops:** Agent keeps calling the same tool without progress. Solution: `max_iterations` hard cap; implement loop detection by tracking tool call history.
>
> 3. **Context window overflow:** Long tool outputs (e.g., large DB query results) fill the context. Solution: Truncate tool outputs to a max length; summarize long results before adding to agent scratchpad.
>
> 4. **Hallucinated tool calls:** Agent calls a tool that doesn't exist or passes wrong parameter types. Solution: Use Pydantic-validated tool schemas; the function-calling API enforces schema compliance.
>
> 5. **Non-determinism:** Same query produces different tool call sequences → different results. Solution: `temperature=0` for agent LLM; add intermediate checkpointing; log full execution trace for debugging."

---

**Q4: How do you evaluate a LangChain RAG pipeline?**

> "I use a 4-layer evaluation:
>
> **Layer 1 — Retrieval quality** (independent of generation):
> - Precision@k: Of top-k retrieved docs, how many are relevant? (need labeled evaluation set)
> - Recall@k: Of all relevant docs, how many appear in top-k?
> - MRR: Is the most relevant doc ranked first?
>
> **Layer 2 — Generation quality** (full pipeline via RAGAS):
> - Faithfulness: Does generated output stay grounded in retrieved context?
> - Answer Relevance: Does output actually answer the question?
> - Context Recall: Did retrieval surface the right evidence?
> - Context Precision: Are retrieved docs actually relevant?
>
> **Layer 3 — Business validation** (shadow mode):
> - Compare pipeline outputs to expert-labeled gold standard
> - At Chubb: Compared RAG fraud flags to investigator verdicts (60-day lag)
>
> **Layer 4 — Production monitoring**:
> - Weekly faithfulness score sampling
> - Input distribution drift (PSI on query embeddings)
> - Retrieval score trends (cosine similarity dropping = KB staleness)
>
> The tricky part: RAGAS requires a test dataset with ground truth. I build this by having domain experts (fraud investigators, clinicians) label 100-200 query-relevant_document pairs manually. That investment pays off in every subsequent evaluation cycle."

---

**Q5: What is the difference between `ConversationBufferMemory`, `ConversationWindowMemory`, and `ConversationSummaryMemory`? When do you use each?**

> "All three manage conversation history but with different token cost vs. fidelity trade-offs:
>
> - **BufferMemory:** Stores the complete conversation verbatim. Highest fidelity, but token count grows unbounded. Fine for short conversations (<20 turns). Risk: hits context limit mid-session.
>
> - **WindowMemory(k):** Keeps only the last k turns, drops older ones. Constant token cost. Risk: loses information from early conversation. Good for: sessions where recent context dominates and historical context is less important.
>
> - **SummaryMemory:** Uses an LLM to progressively summarize older turns into a compressed summary. Balances fidelity and token cost. Best for: long clinical consultations where both early context (chief complaint) and recent context (latest lab results) matter.
>
> For Optum's clinical assistant: I'd use SummaryMemory. A patient conversation might start with 'I have chest pain' (critical to retain) and evolve over 30 turns. SummaryMemory would preserve the chief complaint in summary form while retaining the last 5 turns verbatim."

---

**Q6: How do you implement streaming in LangChain for a production API?**

```python
# Production streaming with FastAPI
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from langchain_core.output_parsers import StrOutputParser

app = FastAPI()

@app.post("/summarize-note")
async def summarize_note(note_text: str):
    """Stream clinical note summary token by token"""
    
    chain = (
        ChatPromptTemplate.from_template(
            "Summarize this clinical note in structured SOAP format:\n{note}"
        )
        | ChatOpenAI(model="gpt-4o", temperature=0.1, streaming=True)
        | StrOutputParser()
    )
    
    async def token_generator():
        async for chunk in chain.astream({"note": note_text}):
            # SSE format: "data: <chunk>\n\n"
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        token_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )
```

> "Streaming is critical for healthcare UX — physicians shouldn't wait 15 seconds staring at a blank screen while the full SOAP note generates. The key points: use `streaming=True` on the LLM, use `.astream()` on the chain (async), and return a `StreamingResponse` from FastAPI. In production, also add error handling — if the stream breaks mid-response, the client needs to know so it can retry or show a partial result warning."

---

**Q7: How would you use LangChain Callbacks for observability?**

```python
from langchain_core.callbacks import BaseCallbackHandler
from datetime import datetime
import json

class HealthcareAuditCallback(BaseCallbackHandler):
    """
    HIPAA-compliant audit logger for all LLM interactions.
    Required: every prompt + response must be logged with user ID and timestamp.
    """
    
    def __init__(self, user_id: str, session_id: str):
        self.user_id = user_id
        self.session_id = session_id
        self.start_time = None
    
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = datetime.utcnow()
        # Log prompt (after PHI redaction in production)
        audit_log = {
            "event": "llm_start",
            "user_id": self.user_id,
            "session_id": self.session_id,
            "timestamp": self.start_time.isoformat(),
            "prompt_length": sum(len(p) for p in prompts)
            # DO NOT log raw prompts if they may contain PHI
            # Log de-identified version or prompt hash only
        }
        self._write_audit_log(audit_log)
    
    def on_llm_end(self, response, **kwargs):
        latency_ms = (datetime.utcnow() - self.start_time).total_seconds() * 1000
        audit_log = {
            "event": "llm_end",
            "user_id": self.user_id,
            "session_id": self.session_id,
            "latency_ms": latency_ms,
            "token_usage": response.llm_output.get("token_usage", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        self._write_audit_log(audit_log)
    
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"[AUDIT] Tool called: {serialized['name']} | Input: {input_str[:100]}...")
    
    def on_tool_error(self, error, **kwargs):
        print(f"[AUDIT] Tool error: {str(error)}")
        # Page on-call engineer for repeated tool failures
    
    def _write_audit_log(self, log: dict):
        # In production: write to CloudWatch / BigQuery / S3 audit bucket
        print(json.dumps(log))

# Usage
callback = HealthcareAuditCallback(user_id="dr.smith@optum.com", session_id="sess-abc123")
result = chain.invoke({"query": "..."}, config={"callbacks": [callback]})
```

---

## PART 6: LANGCHAIN ANTI-PATTERNS (What NOT to Do)

| Anti-Pattern | Problem | Fix |
|---|---|---|
| **Using old `LLMChain` class** | Deprecated, verbose, no streaming | Migrate to LCEL: `prompt \| llm \| parser` |
| **Random split for RAG docs** | Cuts sentences mid-thought | Use `RecursiveCharacterTextSplitter` with overlap |
| **Not caching embeddings** | Re-embeds same docs on every run | Use `CacheBackedEmbeddings` with Redis/local cache |
| **Logging raw prompts with PHI** | HIPAA violation | Log prompt hash or de-identified version only |
| **`max_iterations` not set** | Agent loops forever on ambiguous queries | Always set `max_iterations=8-15` |
| **Using ReAct with small models** | Frequent parsing failures | Use Tool Calling agents; ReAct needs GPT-4+ |
| **Single retriever for medical codes** | Semantic search misses exact ICD code matches | Use EnsembleRetriever(BM25 + dense) |
| **Temperature > 0 for clinical tasks** | Non-deterministic, may fabricate | Set `temperature=0.0` or `0.1` max |

---

*End of LangChain Deep Dive — See 07_LangGraph for state machine patterns*
