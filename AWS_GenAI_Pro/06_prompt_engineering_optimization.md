# Prompt Engineering, Optimization & Management
## AIP-C01 – Domain 1 & 4 – Prompt Patterns

---

## 🎨 Prompt Engineering Fundamentals

Prompt engineering is the practice of designing inputs to guide FM behavior without changing model weights.

### Anatomy of an Effective Prompt

```
┌────────────────────────────────────────────────────────────┐
│ SYSTEM PROMPT                                              │
│ - Persona definition                                       │
│ - Behavioral constraints                                   │
│ - Output format specification                              │
│ - Context/domain information                               │
├────────────────────────────────────────────────────────────┤
│ FEW-SHOT EXAMPLES (optional)                               │
│ - 2-5 examples of desired input/output pairs               │
├────────────────────────────────────────────────────────────┤
│ RETRIEVED CONTEXT (RAG)                                    │
│ - Document chunks from knowledge base                      │
├────────────────────────────────────────────────────────────┤
│ USER QUERY                                                 │
│ - Current user input                                       │
└────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Core Prompting Techniques

### 1. Zero-Shot Prompting

```
No examples provided; relies on model's pre-trained knowledge

Prompt: "Classify the sentiment of this review as POSITIVE, NEGATIVE, or NEUTRAL:
'The product arrived quickly but the packaging was damaged.'"

Response: "NEGATIVE"
```

**Best for:** Tasks the model understands well from training.

---

### 2. Few-Shot Prompting ⭐

```
Provide 2-5 examples before the actual query

System: "Classify customer feedback sentiment."

Example 1:
Input: "Great product, fast shipping!"
Output: POSITIVE

Example 2:
Input: "Broke after one week, terrible quality."
Output: NEGATIVE

Example 3:
Input: "Works as described, nothing special."
Output: NEUTRAL

Now classify:
Input: "The interface is confusing but the results are impressive."
Output: ???
```

**Best for:** Specialized formats, domain-specific tasks, consistent output structure.  
**Exam Note:** Few-shot examples are great candidates for **Prompt Caching** (static prefix).

---

### 3. Chain-of-Thought (CoT) Prompting

```
Ask the model to show reasoning steps before answering

Prompt: "Analyze whether this loan application should be approved. 
Think step by step:
1. Credit score assessment
2. Debt-to-income ratio
3. Employment stability
4. Final recommendation"
```

**Best for:** Math, logic, multi-step reasoning, reducing hallucinations.

---

### 4. System Prompt Design

```python
system_prompt = """You are a specialized financial assistant for AcmeCorp.

ROLE: Help employees understand company financial policies.
SCOPE: Only answer questions about AcmeCorp financial procedures.
TONE: Professional, concise, cite policy numbers when applicable.
RESTRICTIONS: 
- Never provide specific investment advice
- Never discuss competitor products
- If asked about regulated topics, redirect to the compliance team

FORMAT:
- Lead with a direct answer
- Follow with policy reference (e.g., "Per Policy FIN-2025-03...")
- End with next steps if action is required
"""
```

---

### 5. Structured Output Prompting

```python
prompt = """Extract the following information from this invoice and return as JSON:
{
  "vendor_name": "string",
  "invoice_date": "YYYY-MM-DD",
  "total_amount": "number",
  "line_items": [{"description": "string", "quantity": "number", "unit_price": "number"}],
  "payment_terms": "string"
}

Invoice text: {invoice_content}"""
```

**Best for:** Downstream processing, API integrations, consistent parsing.

---

## 💰 Prompt Caching

### How Prompt Caching Works

Bedrock caches the **static prefix** of prompts across requests. Subsequent requests that share the same prefix pay only ~10% of the cost for the cached portion.

```
Request 1 (full cost):
┌─────────────────────────────────┐
│ System Prompt (2,000 tokens)    │ ← Computed + CACHED
│ Few-Shot Examples (1,000 tokens)│ ← Computed + CACHED
├─────────────────────────────────┤ ← CACHE CHECKPOINT
│ User Query (50 tokens)          │ ← Computed (small)
└─────────────────────────────────┘

Request 2 (up to 90% cost reduction):
┌─────────────────────────────────┐
│ System Prompt (2,000 tokens)    │ ← SERVED FROM CACHE (~10% cost)
│ Few-Shot Examples (1,000 tokens)│ ← SERVED FROM CACHE (~10% cost)
├─────────────────────────────────┤
│ User Query (different, 75 tokens)│ ← Computed
└─────────────────────────────────┘
```

### Caching Rules

```
1. Static content must come BEFORE the cache checkpoint
2. Content AFTER the checkpoint is always computed
3. Cache is per-model and per-region
4. Cache TTL: ~5 minutes (refreshed on each hit)
```

```python
# Enable prompt caching with cachePoint marker
messages = [
    {
        "role": "user",
        "content": [
            {"text": "Large static context or system instructions..."},
            {"cachePoint": {"type": "default"}}  # Cache everything above this
        ]
    },
    {"role": "user", "content": [{"text": "Actual user question"}]}
]
```

### When Caching Helps vs. Doesn't Help

| Scenario | Cache Effective? | Why |
|----------|-----------------|-----|
| FAQ bot with fixed system prompt | ✅ Yes | System prompt is static prefix |
| Long few-shot examples reused | ✅ Yes | Examples are static prefix |
| Power users with unique long prompts | ❌ No | Each prompt is unique; no repeated prefix |
| Real-time personalization per request | ❌ No | Personalized content changes prefix |

> **Exam Trap:** "Power users sending long unique prompts driving costs" → Caching doesn't help  
> **Correct solution:** Per-user token budgets via API Gateway usage plans + CountTokens API

---

## 📊 Per-User Token Budget Management

### Problem: High-Cost Power Users

```
Top 5% of users generate 60% of token costs
→ Unfair to penalize all users with global limits
→ Model downgrade hurts quality for everyone
```

### Solution: API Gateway Usage Plans + CountTokens API

```python
# CountTokens API to measure prompt size before inference
response = bedrock.count_tokens(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    systemPrompt="You are a helpful assistant.",
    messages=[{"role": "user", "content": [{"text": user_message}]}]
)
token_count = response["inputTokenCount"]

# Check against user's remaining budget
if token_count > user_budget.remaining_tokens:
    raise Exception("Token budget exceeded for this period")

# Proceed with inference if within budget
```

```
API Gateway Usage Plan:
- Request rate limit: 100 req/min per user
- Burst limit: 200 req
- Per-user API keys track consumption
- Usage Plans enforce limits at the API level
```

> **Exam Key:** Combine `CountTokens API` (measure) + `API Gateway Usage Plans` (enforce)

---

## 🏗️ Bedrock Prompt Management

### Features Overview

| Feature | Description | Exam Relevance |
|---------|-------------|---------------|
| **Templates** | Reusable parameterized prompts with `{{variable}}` syntax | Multi-team consistency |
| **Variants** | Multiple versions for A/B testing | Comparing prompt approaches |
| **Versioning** | Immutable version snapshots | Approval workflows |
| **IAM Control** | Who can create/read/publish prompts | Access governance |
| **CloudTrail Integration** | All prompt operations logged | Audit trail |
| **Cross-Region** | Templates shared across regions | Enterprise governance |

### Prompt Template with Variables

```
Template Name: "SupportTicketSummarizer-v3"
Variables: {{department}}, {{ticket_priority}}, {{language}}

System: You are a {{department}} support analyst. 
Analyze tickets with {{ticket_priority}} priority.
Respond in {{language}}.

Task: Summarize the following support ticket and classify it by:
1. Category (Technical/Billing/Account)
2. Urgency (Critical/High/Medium/Low)
3. Suggested resolution path

Ticket: {{ticket_content}}
```

### Approval Workflow

```
Developer creates Draft version
    ↓
[Bedrock Prompt Management] → Draft state
    ↓
Reviewer approves via IAM-controlled action
    ↓
Published version (immutable)
    ↓
CloudTrail logs the approval event
```

> **Exam Pattern:** "Manage hundreds of prompt templates across multiple teams with version control and audit trails" → **Bedrock Prompt Management + CloudTrail**

---

## 🔁 Conversation Context Management

### Problem: Long Conversations Hit Context Limits

```
Turn 1 (100 tokens)
Turn 2 (100 tokens)
...
Turn 50 (100 tokens per turn = 5,000 tokens in context)
Turn 51 → Context window exceeded or degraded performance
```

### Solution: Conversation Summarization

```python
def manage_conversation_context(history: list, max_tokens: int = 4000):
    """Summarize old turns when approaching context limit."""
    total_tokens = count_tokens(history)
    
    if total_tokens > max_tokens * 0.8:  # 80% threshold
        # Summarize all but last N turns
        old_turns = history[:-5]
        recent_turns = history[-5:]
        
        summary = invoke_bedrock(
            f"Summarize this conversation preserving key facts and preferences:\n{old_turns}"
        )
        
        return [{"role": "system", "content": f"Conversation so far: {summary}"}] + recent_turns
    
    return history
```

**What to Preserve in Summary:**
- User preferences and explicit requirements
- Key decisions made
- Important facts stated by user
- Current task context

> **Exam Pattern:** "Managing long conversations without hitting context limits or causing performance degradation" → **Conversation Summarization** (NOT truncation, NOT larger context window)

---

## 🎯 Converse API Message Structure

### Correct Format (Critical!)

```python
# ✅ CORRECT - Structured messages array
response = bedrock.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    system=[{"text": "You are a helpful assistant."}],  # Separate system
    messages=[
        {
            "role": "user",  # Always 'user' or 'assistant'
            "content": [    # Always an ARRAY
                {"text": "What is quantum computing?"}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"text": "Quantum computing uses quantum mechanical phenomena..."}
            ]
        },
        {
            "role": "user",
            "content": [
                {"text": "How is it different from classical computing?"}
            ]
        }
    ]
)

# ❌ WRONG - Plain text string (not supported)
response = bedrock.converse(
    modelId="...",
    messages="What is quantum computing?"  # This will fail!
)
```

---

## 🔄 Prompt Routing for Cost Optimization

### Intelligent Prompt Routing

```
Incoming Request
    ↓
[Complexity Classifier] ← Internal ML model evaluates request complexity
    ↓
Simple query? → Claude Haiku (cheap, fast)
Complex query? → Claude Sonnet (powerful, expensive)
```

```python
# Use inference profile with intelligent routing
response = bedrock.converse(
    modelId="us.amazon.nova-intelligent-routing",  # Routing profile
    messages=[{"role": "user", "content": [{"text": user_query}]}]
)
```

**Cost Impact Example:**
```
Traffic mix: 70% simple FAQs + 30% complex analysis
Without routing: All traffic → Sonnet = $1.00 per 1K requests
With routing:   70% → Haiku + 30% → Sonnet = ~$0.40 per 1K requests
Savings: ~60%
```

### Manual Routing (AppConfig)

```python
# Fetch routing config from AppConfig without code deployment
import json
import boto3

appconfig = boto3.client('appconfigdata')

# Get current configuration
response = appconfig.get_latest_configuration(
    ConfigurationToken=token
)
config = json.loads(response['Configuration'].read())

# Route based on config
model_id = config['models']['complex'] if is_complex(query) else config['models']['simple']
```

> **Exam Pattern:** "Switch between foundation models without code deployment" → **AWS AppConfig**  
> More powerful than Lambda environment variables (which cause cold starts and require manual management)

---

## 📐 AWS AppConfig for Configuration Management

### Problem with Environment Variables

```
Traditional approach: Lambda environment variable → Model ID
Issue: 
- Changing env var requires Lambda config update
- Config update causes cold starts
- Propagation to 1000s of concurrent instances is slow
- No built-in rollback
- No gradual deployment
```

### AppConfig Solution

```
AppConfig profile: {"model_id": "anthropic.claude-3-5-sonnet-20241022-v2:0"}
    ↓
Deploy new config: {"model_id": "amazon.nova-pro-v1:0"}
    ↓
Canary deployment: 5% of Lambda instances get new config
    ↓ Monitor for errors
Rollout: Gradual increase to 100%
    ↓ If errors spike
Auto-rollback: Revert to previous config
```

```python
# Lambda with AppConfig Agent
import urllib.request

def get_config():
    url = "http://localhost:2772/applications/GenAI/environments/prod/configurations/ModelConfig"
    response = urllib.request.urlopen(url)
    return json.loads(response.read())

def lambda_handler(event, context):
    config = get_config()
    model_id = config["model_id"]
    # Use model_id for Bedrock calls
```

> **Exam Pattern:** "Switch models without redeploying code, with gradual rollouts and automatic rollback" → **AWS AppConfig**

---

## 📝 Practice Questions (Prompting & Optimization)

**Q1:** A company's RAG chatbot has a 2,000-token system prompt and 1,000-token few-shot examples that never change. User queries average 50 tokens. What will MOST reduce token costs?

- A. Switch to a smaller foundation model  
- B. Reduce the number of few-shot examples to 2  
- C. Enable prompt caching with cache checkpoint after the few-shot examples  
- D. Implement conversation summarization for long sessions  

**Answer: C** – 3,000 static tokens cached; subsequent requests pay only ~10% for that portion. This is the highest-impact optimization for this pattern.

---

**Q2:** An enterprise needs to centrally manage prompt templates for 5 business units with different tones, ensure all changes go through approval before deployment, and maintain an audit trail. Which solution has LEAST maintenance overhead?

- A. Store prompts in DynamoDB with Lambda for retrieval and custom approval via SNS notifications  
- B. Use Bedrock Prompt Management with versioning, IAM-controlled approval, and CloudTrail logging  
- C. Store prompt files in S3 with S3 versioning and CodePipeline for approval workflow  
- D. Use SageMaker notebooks to manage prompts with Git for version control  

**Answer: B** – Prompt Management is purpose-built: versioning, approvals via IAM, automatic CloudTrail logging. Others require significant custom development.

---

**Q3:** The Converse API request is returning an error. The developer is sending messages as a plain text string. What is the correct fix?

- A. Use InvokeModel API instead, which supports plain text  
- B. Wrap the text in a messages array with role and content structure  
- C. Add a Content-Type header of text/plain  
- D. URL-encode the message text  

**Answer: B** – Converse API requires structured messages array: `[{"role": "user", "content": [{"text": "..."}]}]`

---

*Next: [07_model_evaluation.md](./07_model_evaluation.md)*
