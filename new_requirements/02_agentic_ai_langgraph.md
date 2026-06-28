# AGENTIC AI — LangChain, LangGraph, LlamaIndex, Agentic Patterns
## Complete production-grade guide for AI Engineer interviews

---

## SECTION 1: WHAT IS AN AGENTIC SYSTEM?

### Definition
An agentic AI system is one where the LLM **decides what actions to take** based on observations, rather than following a fixed script.

**Key difference from standard RAG:**
- **RAG:** Fixed pipeline — User asks → Retrieve → Generate → Return
- **Agentic:** Variable pipeline — User asks → LLM decides (search? compute? ask for clarification? return?) → Act → Observe → Decide again

**When to use agents vs simple RAG:**
| Use RAG | Use Agents |
|---------|-----------|
| Fixed Q&A on documents | Multi-step task execution |
| Single knowledge base | Multiple tools/APIs needed |
| Answer a question | Complete a workflow |
| Document search | Research + synthesis + action |

**Practical SME examples of agentic workflows:**
- "Extract all invoices from this month, check against ERP, flag discrepancies, and email the report"
- "Research competitor pricing from websites, compare to our catalog, and generate a pricing recommendation"
- "Process incoming order email, check inventory in ERP API, confirm if deliverable, send response"

---

## SECTION 2: LANGCHAIN — THE BUILDING BLOCKS

### Core Abstractions

**1. LLMs / Chat Models**
```python
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

# Standard instantiation
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,          # 0 for factual tasks, 0.7 for creative
    max_tokens=2000,
    timeout=30,             # Critical in production — set timeouts!
    max_retries=3,          # Auto-retry on rate limits
    api_key=settings.OPENAI_API_KEY
)

# Use Claude for longer documents (200K context window)
claude = ChatAnthropic(
    model="claude-3-5-sonnet-20241022",
    max_tokens=4000
)
```

**2. Prompts and Templates**
```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Template with system prompt + history + user input
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an AI assistant for {company_name}. 
    You help employees find information in company documents.
    Always cite your sources. If you don't know, say so.
    Today's date is {date}."""),
    MessagesPlaceholder("history"),  # For conversation memory
    ("human", "{question}")
])

# Fill template
formatted = prompt.invoke({
    "company_name": "Acme SRL",
    "date": "2025-01-15",
    "history": [],
    "question": "What is our return policy?"
})
```

**3. Output Parsers**
```python
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

class InvoiceExtraction(BaseModel):
    invoice_number: str = Field(description="Invoice number (numero fattura)")
    date: str = Field(description="Invoice date in YYYY-MM-DD format")
    total_amount: float = Field(description="Total amount including VAT")
    supplier_name: str = Field(description="Supplier company name")
    vat_number: str = Field(description="Supplier VAT number (P.IVA)")
    line_items: list[dict] = Field(description="List of items with quantity and price")

parser = JsonOutputParser(pydantic_object=InvoiceExtraction)

# The parser adds format instructions to the prompt automatically
prompt_with_format = prompt | parser  # LCEL (LangChain Expression Language)
```

**4. LCEL — LangChain Expression Language**
```python
# Chain composition using | operator (pipe)
from langchain_core.runnables import RunnablePassthrough

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# Equivalent long form:
class RAGChain:
    def invoke(self, question: str) -> str:
        context = retriever.invoke(question)
        prompt_value = prompt.invoke({"context": context, "question": question})
        response = llm.invoke(prompt_value)
        return StrOutputParser().invoke(response)
```

**5. Tools**
```python
from langchain_core.tools import tool

@tool
def search_invoices(query: str, date_from: str = None, date_to: str = None) -> str:
    """
    Search company invoices by keyword and optional date range.
    Returns matching invoice summaries.
    
    Args:
        query: Search term (e.g., supplier name, invoice number)
        date_from: Start date in YYYY-MM-DD format
        date_to: End date in YYYY-MM-DD format
    """
    results = invoice_db.search(query=query, date_from=date_from, date_to=date_to)
    return format_invoice_results(results)

@tool  
def get_erp_stock(product_code: str) -> str:
    """Check current stock level for a product code in the ERP system."""
    stock = erp_api.get_stock(product_code)
    return f"Product {product_code}: {stock.quantity} units available, reorder point: {stock.reorder_point}"

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email. Use ONLY when explicitly asked to communicate externally."""
    # Note: this is a HIGH-RISK tool — always confirm before executing
    email_client.send(to=to, subject=subject, body=body)
    return f"Email sent to {to} with subject: {subject}"
```

---

## SECTION 3: LANGGRAPH — STATEFUL AGENTIC WORKFLOWS

### Why LangGraph Over Simple LangChain Agents?

**LangChain agents (ReAct):** LLM repeatedly chooses tools until it's done. Works for simple tasks but:
- No control over execution flow
- Hard to add human approval steps
- Difficult to handle complex multi-step workflows
- No visibility into what's happening mid-execution

**LangGraph:** Define explicit states and transitions:
- Full control over execution graph
- Easy to add conditional logic, loops, human-in-the-loop
- Persistent state across steps (survives failures)
- Production-grade observability

### LangGraph Core Concepts

```python
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, Annotated
import operator

# 1. Define the State (what persists across graph nodes)
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]  # Accumulate messages
    documents: list[str]                      # Retrieved documents
    current_task: str                         # What the agent is doing
    human_approval_needed: bool               # Flag for high-risk actions
    iteration_count: int                      # Prevent infinite loops
    final_answer: str | None

# 2. Define nodes (functions that transform state)
def retrieve_node(state: AgentState) -> AgentState:
    """Retrieve documents based on the question in messages"""
    question = state["messages"][-1].content
    docs = retriever.invoke(question)
    return {
        "documents": [d.page_content for d in docs],
        "current_task": "retrieved_documents"
    }

def generate_node(state: AgentState) -> AgentState:
    """Generate answer from retrieved documents"""
    question = state["messages"][-1].content
    context = "\n\n".join(state["documents"])
    
    response = llm.invoke(RAG_PROMPT.format(context=context, question=question))
    
    return {
        "messages": [response],
        "final_answer": response.content,
        "current_task": "generated_answer"
    }

def grade_documents_node(state: AgentState) -> str:
    """Check if retrieved documents are relevant → routing function"""
    question = state["messages"][-1].content
    documents = state["documents"]
    
    # LLM judges relevance
    grade = grader_llm.invoke(f"Are these documents relevant to '{question}'? Relevant/Irrelevant\n{documents}")
    
    if "Relevant" in grade.content:
        return "generate"  # → go to generate_node
    else:
        return "web_search"  # → go to web search node

def human_approval_node(state: AgentState) -> AgentState:
    """Pause for human approval on high-risk actions"""
    # In production: send notification, wait for webhook response
    # LangGraph supports interrupts here
    return {"human_approval_needed": True, "current_task": "awaiting_approval"}

# 3. Build the graph
def build_rag_agent_graph() -> StateGraph:
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("human_approval", human_approval_node)
    
    # Set entry point
    workflow.set_entry_point("retrieve")
    
    # Add edges (transitions)
    workflow.add_edge("retrieve", "grade_documents")
    
    # Conditional edge (routing based on function return value)
    workflow.add_conditional_edges(
        "grade_documents",
        grade_documents_node,  # Returns "generate" or "web_search"
        {
            "generate": "generate",
            "web_search": "web_search"
        }
    )
    
    workflow.add_edge("web_search", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

# 4. Add persistence (memory across conversations)
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")  # In-memory for dev
# In production: PostgreSQL checkpointer
graph = workflow.compile(checkpointer=memory)

# 5. Invoke with thread_id for persistence
result = graph.invoke(
    {"messages": [HumanMessage(content="What invoices are overdue?")]},
    config={"configurable": {"thread_id": "client_session_123"}}
)
```

### Human-in-the-Loop Pattern (CRITICAL for SME deployments)

```python
from langgraph.checkpoint.memory import MemorySaver

# Define interrupt point
workflow.add_node("send_email", send_email_node)
workflow.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["send_email"]  # ALWAYS pause before sending emails
)

# When agent reaches "send_email" node, it pauses and saves state
# Human reviews and approves via API:

# 1. Agent runs until interrupt:
initial_result = graph.invoke(user_message, config={"thread_id": "thread_1"})
# → Returns with status "interrupted" at "send_email" node

# 2. Human reviews pending action
pending = graph.get_state(config={"thread_id": "thread_1"})
email_to_send = pending.values["pending_email"]  # Show to human

# 3. Human approves → resume execution
if human_approved:
    graph.invoke(None, config={"thread_id": "thread_1"})  # Resume from checkpoint
else:
    # Human rejected → update state and re-plan
    graph.update_state(config, {"email_cancelled": True, "messages": [rejection_reason]})
    graph.invoke(None, config={"thread_id": "thread_1"})
```

**Interview answer on Human-in-the-loop:**
> "For SME clients, I always add human approval gates before irreversible actions — sending emails, updating ERP records, creating invoices. LangGraph's interrupt_before feature lets the graph pause at those nodes, serialize state to a database, and wait for a human webhook response. The client can approve or reject via a simple web interface. If approved, the graph resumes exactly where it stopped. This is critical for building client trust in early deployments."

---

## SECTION 4: REACT AGENT PATTERN

### How ReAct Works
```
Thought: [LLM reasons about what to do]
Action: [LLM chooses a tool]
Observation: [Tool returns result]
Thought: [LLM reasons based on observation]
Action: [Another tool or Final Answer]
...
Final Answer: [LLM synthesizes]
```

```python
from langgraph.prebuilt import create_react_agent

tools = [search_invoices, get_erp_stock, search_web, send_email]

# Simple ReAct agent
agent = create_react_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are an AI assistant for an Italian SME.
    Help users with business tasks involving their documents and ERP system.
    Always ask for clarification before sending emails or modifying records.
    Be concise and factual."""
)

# But in production, DON'T use prebuilt — build with LangGraph for control
```

### Preventing Agent Failures in Production

**Problem 1: Infinite loops**
```python
# In state: track iteration count
class AgentState(TypedDict):
    iteration_count: int

def should_continue(state: AgentState) -> str:
    if state["iteration_count"] > 10:  # Hard limit
        return "force_end"
    if state["messages"][-1].type == "ai" and not state["messages"][-1].tool_calls:
        return "end"  # LLM chose not to use tools → done
    return "continue"
```

**Problem 2: Tool call hallucination (calling tools with wrong args)**
```python
# Validate tool inputs before execution
def safe_tool_execution(tool_call: ToolCall) -> str:
    tool = tools_map[tool_call["name"]]
    
    try:
        # Pydantic validation of arguments
        validated_args = tool.args_schema(**tool_call["args"])
        return tool.invoke(validated_args.dict())
    except ValidationError as e:
        return f"Tool call failed due to invalid arguments: {e}. Please check the required format."
    except Exception as e:
        logger.error(f"Tool execution error: {tool_call['name']} - {e}")
        return f"Tool '{tool_call['name']}' failed: {str(e)}. Try a different approach."
```

**Problem 3: Context window overflow in long agent runs**
```python
def trim_messages_for_context(messages: list, max_tokens: int = 50000) -> list:
    """Keep recent messages, summarize older ones"""
    total_tokens = sum(count_tokens(m.content) for m in messages)
    
    if total_tokens <= max_tokens:
        return messages
    
    # Summarize older messages
    old_messages = messages[:-10]  # Keep last 10 messages
    summary = llm.invoke(f"Summarize this conversation: {old_messages}")
    
    return [SystemMessage(content=f"[Conversation summary]: {summary.content}")] + messages[-10:]
```

---

## SECTION 5: LLAMAINDEX — ALTERNATIVE FRAMEWORK

### When to Choose LlamaIndex vs LangChain

| Feature | LangChain | LlamaIndex |
|---------|-----------|------------|
| Primary strength | Agentic workflows, tool use | Document indexing + Q&A |
| Document connectors | Good | Excellent (150+ connectors) |
| Multi-document RAG | Manual setup | Built-in |
| Agent framework | LangGraph (strong) | Workflows + SubQuestionQueryEngine |
| Learning curve | Medium | Medium |
| Community | Larger | Focused on RAG |

**Use LlamaIndex when:** Primary use case is document indexing, complex multi-document synthesis
**Use LangChain/LangGraph when:** Agentic workflows, multi-tool orchestration, complex state machines

### LlamaIndex Key Components

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore

# Global settings (set once)
Settings.llm = OpenAI(model="gpt-4o", temperature=0)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
Settings.node_parser = SentenceSplitter(chunk_size=500, chunk_overlap=100)
Settings.context_window = 4096

# Load and index documents
documents = SimpleDirectoryReader("/path/to/sme_docs").load_data()

# With Qdrant as vector store
client = QdrantClient(url="http://localhost:6333")
vector_store = QdrantVectorStore(client=client, collection_name="sme_docs")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    show_progress=True
)

# Query
query_engine = index.as_query_engine(
    similarity_top_k=5,
    response_mode="tree_summarize"  # Synthesizes across multiple nodes
)
response = query_engine.query("What are our payment terms?")
print(response.response)
print(response.source_nodes)  # See which chunks were used
```

### SubQuestion Query Engine (Multi-Document Complex Queries)

```python
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool

# Create separate indices for different document types
contracts_engine = contract_index.as_query_engine()
invoices_engine = invoice_index.as_query_engine()
emails_engine = email_index.as_query_engine()

# Combine with SubQuestion engine
query_tools = [
    QueryEngineTool.from_defaults(contracts_engine, description="Legal contracts and agreements"),
    QueryEngineTool.from_defaults(invoices_engine, description="Invoices and billing documents"),
    QueryEngineTool.from_defaults(emails_engine, description="Email correspondence"),
]

# This engine breaks complex questions into sub-questions per source
query_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools=query_tools)

# "What are the payment terms in our contract with Fornitore X, 
#  and how much have we actually paid them this year?"
# → Auto-breaks into: contract question + invoice question → synthesizes
response = query_engine.query(complex_question)
```

---

## SECTION 6: TOOL DESIGN PATTERNS FOR PRODUCTION

### The CRUD Pattern for SME Workflows

```python
# Pattern: Read operations are always safe, Write operations need confirmation

# SAFE tools (no confirmation needed)
@tool(return_direct=False)
def search_documents(query: str) -> str:
    """Search company documents. READ-ONLY."""
    ...

@tool
def get_customer_info(customer_id: str) -> str:
    """Get customer information from CRM. READ-ONLY."""
    ...

@tool
def calculate_invoice_total(items: list[dict]) -> float:
    """Calculate invoice total. COMPUTATION ONLY."""
    ...

# RISKY tools (ALWAYS require human confirmation via interrupt)
@tool
def create_invoice(customer_id: str, items: list[dict]) -> str:
    """
    Create a new invoice in the ERP system. 
    WARNING: This creates an official fiscal document. Requires human approval.
    """
    ...

@tool  
def send_customer_email(customer_id: str, subject: str, body: str) -> str:
    """
    Send email to customer.
    WARNING: External communication. Requires human approval.
    """
    ...
```

### Tool Result Formatting (Critical for Agent Reliability)

```python
@tool
def search_invoices(query: str) -> str:
    """Search invoices by keyword"""
    results = db.search_invoices(query)
    
    if not results:
        return "No invoices found matching the query."
    
    # Format for LLM consumption — structured but readable
    formatted = f"Found {len(results)} invoices:\n"
    for inv in results[:5]:  # Limit output size
        formatted += f"""
- Invoice {inv.number} | Date: {inv.date} | Amount: €{inv.total:.2f}
  Supplier: {inv.supplier_name} | Status: {inv.payment_status}
  Due: {inv.due_date}
"""
    if len(results) > 5:
        formatted += f"\n[{len(results)-5} more results not shown. Refine your search to see more.]"
    
    return formatted
```

---

## SECTION 7: N8N — WORKFLOW AUTOMATION (Nice to Have)

### What n8n Is

n8n is a visual workflow automation tool (like Zapier, but self-hosted). It connects apps and services without coding:
- Trigger: "When new email arrives" → Action: "Extract attachment → Run AI analysis → Post to Slack"
- Used by SMEs without technical teams to automate repetitive tasks

### How n8n Integrates with AI Systems

```
n8n workflow example for Italian accounting firm:
1. Trigger: New PDF file uploaded to Google Drive/OneDrive
2. n8n: Download file, extract text (using built-in PDF node)
3. n8n: HTTP Request to YOUR FastAPI endpoint (/extract-invoice-data)
4. FastAPI + AI: Extract fields using LLM, validate, return JSON
5. n8n: Map extracted fields to ERP API calls (create vendor invoice in Zucchetti)
6. n8n: Send Slack/email notification with summary
7. n8n: Log to Google Sheets for audit trail
```

**Interview answer on n8n:**
> "n8n is excellent for connecting AI capabilities to existing SME workflows without requiring the client to write code. I use n8n as the orchestration layer that handles triggers (file uploads, emails, schedules) and integrates with existing SME tools (email, ERP, CRM). The AI-specific logic — extraction, classification, generation — lives in a FastAPI microservice that n8n calls via HTTP. This separation means the client's operations team can modify the workflow in n8n's visual interface without touching the AI code."

---

## SECTION 8: PRODUCTION AGENT FAILURES AND DEBUGGING

### Failure 1: Agent Gets Stuck in Tool Call Loop
**Symptom:** Agent keeps calling the same tool with slightly different args
**Detection:** Log every tool call with timestamps. Alert if same tool called >3 times in a row.
**Fix:** 
```python
# Track tool call history in state
if state["tool_call_counts"].get(tool_name, 0) >= 3:
    return "I've tried this approach multiple times without success. Let me try a different strategy or ask for clarification."
```

### Failure 2: Hallucinated Tool Arguments
**Symptom:** Agent calls tool with wrong argument types or fabricated values
**Detection:** Pydantic validation on all tool inputs catches this at runtime
**Fix:** Clear tool descriptions with explicit parameter types and examples

### Failure 3: Context Accumulation → Token Limit
**Symptom:** Long agent runs fail with context window errors
**Detection:** Count tokens before each LLM call, log when > 80% of limit
**Fix:** Summarize old messages, use streaming for long outputs

### Failure 4: LLM Not Using Tools When It Should
**Symptom:** Agent answers from training knowledge instead of calling tools
**Root cause:** System prompt not clear enough, or LLM "thinks" it knows the answer
**Fix:**
```
System prompt: "ALWAYS use the search_documents tool to find information. 
Never answer from memory alone. If no relevant documents are found, 
say so explicitly — do not guess."
```

### Failure 5: Rate Limiting Mid-Agent Run
**Symptom:** Agent fails partway through due to API rate limits
**Fix:** Exponential backoff with jitter + queue management

```python
import time
import random

def call_llm_with_retry(prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            return llm.invoke(prompt)
        except RateLimitError:
            if attempt == max_retries - 1:
                raise
            # Exponential backoff: 2^attempt + random jitter
            sleep_time = (2 ** attempt) + random.uniform(0, 1)
            logger.warning(f"Rate limited. Retry {attempt+1} after {sleep_time:.1f}s")
            time.sleep(sleep_time)
```
