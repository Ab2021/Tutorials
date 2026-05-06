# Bedrock Agents, Multi-Agent & Tool Integration
## AIP-C01 – Domain 2 (26%) – Implementation and Integration

---

## 🤖 Amazon Bedrock Agents Overview

A **Bedrock Agent** is a managed AI agent that can:
- Reason about complex multi-step tasks (ReAct framework)
- Execute actions by calling tools/APIs
- Access knowledge bases for contextual retrieval
- Maintain conversation memory across turns

```
User Request
     ↓
[Agent Orchestrator]
     ↓ Plans action
[Tool Selection] → Which action group to call?
     ↓
[Action Group] → Lambda / API / Knowledge Base
     ↓
[Tool Response] → Back to orchestrator
     ↓
[Reasoning Step] → Is more action needed?
     ↓ (loop until goal achieved)
[Final Response] → Back to user
```

---

## 🧩 Agent Components

### 1. Instructions (System Prompt)

Defines the agent's persona, scope, and behavior:

```
You are a financial advisor assistant. 
You have access to:
- Portfolio lookup tools to retrieve account balances
- Market data tools to get real-time prices
- Policy database for investment guidelines

Always cite policy numbers when referencing investment rules.
Never provide specific investment advice; direct users to a human advisor.
```

### 2. Foundation Model

The reasoning engine. Common choices:
- **Claude 3.5 Sonnet** – Best reasoning for complex agentic tasks
- **Amazon Nova Pro** – Cost-effective, large context
- **Claude 3 Haiku** – Low-latency, simple tool use

### 3. Action Groups

Define the tools available to the agent:

| Action Group Type | Description |
|------------------|-------------|
| **Lambda function** | Custom Python/Node.js code |
| **OpenAPI schema** | Describes REST API endpoints |
| **Return Control** | Agent returns decision to app (app executes) |
| **User Input** | Agent asks user for clarification |

```json
// OpenAPI action group schema example
{
  "openapi": "3.0.0",
  "info": {"title": "Portfolio API", "version": "1.0"},
  "paths": {
    "/portfolio/{account_id}": {
      "get": {
        "summary": "Get portfolio by account ID",
        "parameters": [{"name": "account_id", "in": "path", "required": true}],
        "responses": {"200": {"description": "Portfolio data"}}
      }
    }
  }
}
```

### 4. Knowledge Bases

Optional; agents can query knowledge bases as part of their reasoning chain.

### 5. Memory

Types of memory available to agents:

| Memory Type | Scope | Purpose |
|-------------|-------|---------|
| **Session Memory** | Within a single conversation | Maintains context for current turn |
| **Agent Core Memory** | Cross-session | Shared state for multi-agent systems |
| **External Memory** | DynamoDB / ElasticCache | Custom long-term storage |

---

## 🔍 Agent Traces – Debugging Tool ⭐

**Agent traces** provide visibility into the agent's reasoning chain:

```
Step 1: [rationale] User wants to check portfolio balance
Step 2: [invocationInput] Calling get_portfolio(account_id="ACC123")
Step 3: [observation] Portfolio value: $45,234. Holdings: AAPL(30%), MSFT(20%)...
Step 4: [rationale] I have the portfolio data, now format the response
Step 5: [finalResponse] Your portfolio is valued at $45,234...
```

### Why Traces Matter for Debugging

| Problem | What Traces Reveal | What Metrics Miss |
|---------|-------------------|-------------------|
| Infinite loops | Exact step where agent keeps re-calling same tool | Only shows high call count |
| Wrong tool selection | Which tool was chosen and why | Cannot identify reasoning error |
| Ambiguous responses | What response caused re-processing | No visibility into tool output |
| Slow performance | Which step is the bottleneck | Shows only total latency |

> **Exam Pattern:** "Agent is stuck in an infinite loop" → **Enable Agent Traces** to see reasoning chain  
> ❌ Wrong answers: Increase Lambda timeout, reduce context window, add CloudWatch alarms

---

## 🔄 ReAct (Reason + Act) Framework

Bedrock Agents implement the **ReAct** pattern:

```
Thought: I need to find the user's account balance
Action: call_tool("get_balance", {"account_id": "ACC123"})
Observation: {"balance": 45234, "currency": "USD"}

Thought: I have the balance. Should I also show recent transactions?
Action: call_tool("get_transactions", {"account_id": "ACC123", "limit": 5})
Observation: [{"date": "2025-01-15", "amount": -250, "description": "..."}]

Thought: I have enough context to answer the user
Final Answer: Your current balance is $45,234...
```

---

## 🌐 Amazon Bedrock AgentCore

**AgentCore** is the managed runtime platform for production Bedrock Agents:

### AgentCore Components

| Component | Description |
|-----------|-------------|
| **AgentCore Runtime** | Managed execution environment for agents |
| **AgentCore Memory** | Shared state across multiple specialized agents |
| **AgentCore Gateway** | Exposes REST APIs as tools via OpenAPI specs |
| **AgentCore Observability** | Traces, logs, performance metrics |
| **AgentCore Policy** | Deterministic governance at the gateway layer |

### AgentCore Gateway ⭐

**Purpose:** Automatically expose existing REST APIs as agent tools using OpenAPI specifications.

```
Existing REST API (e.g., CRM, ERP, Database)
         ↓
[AgentCore Gateway]
         ↓ Auto-generates tool definitions from OpenAPI spec
[Bedrock Agent] → Can now call the API as a tool
```

> **Exam Pattern:** "Expose REST APIs as agent tools without manual coding" → **AgentCore Gateway with OpenAPI spec**

### AgentCore Memory ⭐

**Purpose:** Shared conversation state across multiple specialized agents.

```
Orchestrator Agent
    ├── Specialist Agent A (Customer Service)  ─┐
    ├── Specialist Agent B (Finance)            ├── All access AgentCore Memory
    └── Specialist Agent C (Technical Support) ─┘
```

> **Exam Pattern:** "Multiple agents needing access to shared conversation context" → **Amazon Bedrock AgentCore Memory**

---

## 🔌 Model Context Protocol (MCP)

**MCP** is an open standard for connecting AI agents to external tools and data sources.

### Why MCP?

```
Without MCP:
Agent Framework A → Custom wrapper for Tool 1
Agent Framework A → Custom wrapper for Tool 2
Agent Framework B → Different custom wrapper for Tool 1
Agent Framework B → Different custom wrapper for Tool 2

With MCP:
Agent Framework A → MCP Client → MCP Server → Tool 1
Agent Framework B → MCP Client → MCP Server → Tool 1
(Same tool works with any framework)
```

### MCP Architecture

```
MCP Client (Agent) ←→ MCP Server (Tool Provider)

MCP Server exposes:
- Resources: Data sources (files, databases, APIs)
- Tools: Executable functions
- Prompts: Reusable prompt templates
```

### MCP in AWS Ecosystem

| Component | Role |
|-----------|------|
| **AgentCore Gateway** | Hosts MCP servers for REST APIs |
| **Lambda** | Can act as MCP server for custom tools |
| **ECS** | For CPU-intensive MCP server workloads |

> **Exam Pattern:** "Ensure tool consistency across different agent frameworks" → **Implement MCP (Model Context Protocol)**  
> **Exam Pattern:** "Lightweight tools vs CPU-intensive tools in same agent" → **Hybrid: Lambda (lightweight) + ECS (heavy)**

---

## 🏗️ Multi-Agent Architectures

### Supervisor Pattern

```
User → [Supervisor Agent]
            ├── Delegates to [Specialist Agent: Research]
            ├── Delegates to [Specialist Agent: Analysis]  
            └── Delegates to [Specialist Agent: Report Writing]
            ↓
       [Aggregates results]
            ↓
       Final Response → User
```

### Pipeline Pattern

```
User → [Agent 1: Data Extraction] → [Agent 2: Analysis] → [Agent 3: Formatting] → Response
```

### Fan-Out Pattern

```
User → [Orchestrator]
            ├── [Agent A] ─┐
            ├── [Agent B] ─┤ All run in parallel
            └── [Agent C] ─┘
            ↓
       [Aggregates results]
```

---

## 🔧 Human-in-the-Loop with Step Functions

When agents need **human approval for high-risk decisions:**

```python
# Step Functions state machine with task token
{
  "states": {
    "GenerateRecommendation": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "Next": "WaitForHumanApproval"
    },
    "WaitForHumanApproval": {
      "Type": "Task",
      "Resource": "arn:aws:states:::sqs:sendMessage.waitForTaskToken",
      "Parameters": {
        "QueueUrl": "...",
        "MessageBody": {
          "taskToken.$": "$$.Task.Token",
          "recommendation.$": "$.recommendation"
        }
      },
      "Next": "ExecuteApprovedAction"
    }
  }
}
```

**Key Concept: Task Tokens**
- Workflow PAUSES when sending to SQS/SNS
- Stays paused (no resources consumed) until human approves/rejects
- On approval, human sends task token back → workflow resumes
- Provides full state management and audit trail

> **Exam Pattern:** "High-risk AI decisions requiring human approval before execution" → **Step Functions with Task Tokens**

---

## 🌊 Bedrock Flows (Low-Code Orchestration)

### When Flows vs Step Functions vs Agents

| Tool | Best For | Key Differentiator |
|------|---------|-------------------|
| **Bedrock Agents** | Autonomous multi-step reasoning | Self-directed tool use |
| **Bedrock Flows** | Deterministic LLM pipelines | Visual, LLM-native workflow builder |
| **Step Functions** | Complex workflow with conditionals, retries, human approval | Enterprise orchestration, non-LLM steps |

### Circuit Breaker Pattern with Step Functions

```
Normal Request → Try Bedrock Call
                      ↓
               Success? → Return response
                      ↓ Failure
               Failure count < threshold? → Retry
                      ↓ Threshold exceeded
               CIRCUIT OPEN → Return fallback response immediately
               (Don't hammer failing service)
```

> **Exam Pattern:** "Prevent cascading failures during Bedrock service instability" → **Circuit Breaker pattern with Step Functions**  
> "Fail fast instead of waiting for timeouts" → Circuit Breaker returns cached/fallback response

---

## 📊 Agent Observability

### Distributed Tracing with X-Ray

```python
# Add X-Ray tracing to Lambda functions handling agent callbacks
import aws_xray_sdk.core as xray_core
xray_core.patch_all()

@xray_core.capture('process_agent_request')
def lambda_handler(event, context):
    with xray_core.in_subsegment('bedrock_invoke') as subsegment:
        response = bedrock.invoke_agent(...)
        subsegment.put_annotation('model_id', model_id)
    return response
```

**What X-Ray Reveals for Agents:**
- End-to-end latency breakdown
- Which step (model, tool, Lambda cold start) is the bottleneck
- Error propagation across services

> **Exam Pattern:** "Debugging slow agent responses" → **AWS X-Ray with subsegments** to isolate model vs tool latency

---

## 🔐 Agent Security

### IAM Execution Roles

Each agent requires an IAM role with least-privilege permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel"],
      "Resource": "arn:aws:bedrock:*::foundation-model/anthropic.claude-*"
    },
    {
      "Effect": "Allow",
      "Action": ["bedrock:Retrieve"],
      "Resource": "arn:aws:bedrock:*:*:knowledge-base/KB123"
    },
    {
      "Effect": "Allow",
      "Action": ["lambda:InvokeFunction"],
      "Resource": "arn:aws:lambda:*:*:function:portfolio-lookup"
    }
  ]
}
```

> **Exam Key:** Access to Bedrock models is managed via **IAM policies with model ARNs** — NOT bucket policies, NOT network isolation alone

---

## 📝 Practice Questions (Agents)

**Q1:** A Bedrock Agent designed to answer customer queries has started producing infinite loops where it repeatedly calls the same tool without resolving the query. What is the BEST first debugging step?

- A. Increase the Lambda function timeout to allow more processing time  
- B. Reduce the agent's context window to limit processing overhead  
- C. Enable agent traces to see the model's reasoning chain and identify the stuck step  
- D. Add CloudWatch alarms to detect high tool invocation counts  

**Answer: C** – Traces reveal WHY the loop happens (ambiguous tool response, unclear stopping condition). Timeout and context changes mask the symptom. CloudWatch only detects THAT looping occurs.

---

**Q2:** A company has three specialized agents (research, analysis, formatting) that all need access to the same conversation history. What service provides shared state for these agents?

- A. Amazon DynamoDB with a shared session table  
- B. Amazon Bedrock AgentCore Memory  
- C. Amazon ElastiCache for Redis  
- D. AWS Step Functions with shared context variable  

**Answer: B** – AgentCore Memory is specifically designed for shared state across specialized agents in multi-agent architectures.

---

**Q3:** A team wants to expose 20 existing REST APIs as tools for a Bedrock Agent without writing custom Lambda wrappers for each. What is the most efficient approach?

- A. Create 20 separate Lambda functions, one per API  
- B. Use Amazon Bedrock AgentCore Gateway with OpenAPI specifications  
- C. Write a single Lambda function that routes to all 20 APIs  
- D. Use API Gateway as a proxy and reference endpoint URLs in agent instructions  

**Answer: B** – AgentCore Gateway automatically exposes REST APIs as tools using OpenAPI specs, eliminating manual coding.

---

*Next: [05_guardrails_safety.md](./05_guardrails_safety.md)*
