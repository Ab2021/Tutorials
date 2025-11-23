# Day 51: Building Custom Tools (LangChain & LlamaIndex)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing Custom Tools in Frameworks

We will implement a "Database Research Agent" using LangChain and a "Document Expert" using LlamaIndex.

### 1. LangChain: The SQL Toolkit Approach

We'll build a custom tool that doesn't just run SQL, but explains the schema first.

```python
from langchain.tools import tool
from langchain.pydantic_v1 import BaseModel, Field
from typing import List
import sqlite3

# 1. Define Input Schema
class SQLQueryInput(BaseModel):
    query: str = Field(description="The SQL query to execute")

# 2. Define the Tool
class SQLiteTool:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    @tool("execute_sql", args_schema=SQLQueryInput)
    def execute_sql(self, query: str) -> str:
        """Execute a SELECT statement against the database. 
        Returns the rows or an error message."""
        try:
            # Security: Read-only check (Basic)
            if "drop" in query.lower() or "delete" in query.lower():
                return "Error: Read-only access."
                
            cursor = self.conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            
            # Truncation logic
            if len(rows) > 10:
                return f"Result: {str(rows[:10])}... (and {len(rows)-10} more rows)"
            return str(rows)
        except Exception as e:
            return f"SQL Error: {str(e)}"

    @tool
    def get_schema(self) -> str:
        """Get the list of tables and their columns."""
        # Logic to query sqlite_master
        return "Table: Users (id, name, email)..."

# 3. Bind to Agent
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain import hub

llm = ChatOpenAI(model="gpt-4-turbo")
db_tool = SQLiteTool("my_db.sqlite")
tools = [db_tool.execute_sql, db_tool.get_schema]

# Get standard prompt
prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run
# agent_executor.invoke({"input": "How many users are there?"})
```

### 2. LlamaIndex: Wrapping RAG as a Tool

This is where LlamaIndex shines. We turn a vector store index into a tool that an agent can call.

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent

# 1. Load Data & Build Index
documents = SimpleDirectoryReader("./data/financial_reports").load_data()
index = VectorStoreIndex.from_documents(documents)

# 2. Create Query Engine
query_engine = index.as_query_engine(similarity_top_k=3)

# 3. Wrap as Tool
finance_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="financial_report_search",
        description=(
            "Useful for looking up financial data, revenue, and quarterly results. "
            "Input should be a specific question about the finance reports."
        ),
    ),
)

# 4. Create Agent
agent = OpenAIAgent.from_tools(
    [finance_tool],
    verbose=True,
    system_prompt="You are a financial analyst. Use the search tool to find data."
)

# 5. Run
# response = agent.chat("What was the Q3 revenue growth?")
```

### 3. Advanced: Tool Retrieval (LangChain)

When you have too many tools, use a Retriever to fetch the right ones.

```python
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

# 1. Index Tool Descriptions
tool_docs = [
    Document(page_content=t.description, metadata={"index": i})
    for i, t in enumerate(all_tools)
]
vector_store = FAISS.from_documents(tool_docs, OpenAIEmbeddings())
retriever = vector_store.as_retriever()

# 2. Custom Step to Fetch Tools
def get_tools(query):
    docs = retriever.get_relevant_documents(query)
    return [all_tools[d.metadata["index"]] for d in docs]

# This logic would sit inside a custom Agent loop
```

### 4. Handling "Human in the Loop" Tool

A tool that pauses execution to ask the user for confirmation.

```python
from langchain.tools import tool

@tool
def ask_human(question: str) -> str:
    """Ask the human user for input or confirmation."""
    # In a real app, this would send a WebSocket message to the UI
    # and block until a response is received.
    print(f"🤖 Agent asks: {question}")
    return input("👤 You answer: ")

# Usage in Agent
# If the agent needs a password or clarification, it calls `ask_human`.
```

### Summary

*   **LangChain** is great for general-purpose tools and chaining logic.
*   **LlamaIndex** is superior for data-centric tools (RAG engines).
*   **Best Practice:** Use Pydantic schemas everywhere. It's the contract between your messy Python code and the structured world of the LLM.
