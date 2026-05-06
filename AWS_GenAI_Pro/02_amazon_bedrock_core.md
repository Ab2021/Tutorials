# Amazon Bedrock Core – APIs, Services & Architecture
## AIP-C01 – Domain 1 & 2 Deep Dive

---

## 🏗️ Amazon Bedrock Architecture Overview

```
                         ┌─────────────────────────────────────┐
                         │         Amazon Bedrock              │
                         │                                     │
  Your App ──────────────►  ┌─────────────────────────────┐   │
                         │  │     Bedrock Runtime APIs     │   │
  SDK / CLI ─────────────►  │  InvokeModel                │   │
                         │  │  InvokeModelWithStream       │   │
                         │  │  Converse / ConverseStream   │   │
                         │  └──────────────┬──────────────┘   │
                         │                 │                   │
                         │  ┌──────────────▼──────────────┐   │
                         │  │       Foundation Models      │   │
                         │  │  Claude | Nova | Titan |     │   │
                         │  │  Llama | Mistral | Cohere   │   │
                         │  └─────────────────────────────┘   │
                         │                                     │
                         │  ┌─────────────────────────────┐   │
                         │  │    Bedrock Features          │   │
                         │  │  Knowledge Bases | Agents   │   │
                         │  │  Guardrails | Evaluations   │   │
                         │  │  Flows | Prompt Mgmt        │   │
                         │  └─────────────────────────────┘   │
                         └─────────────────────────────────────┘
```

---

## 📡 Core Bedrock APIs

### 1. InvokeModel API

**Use:** Synchronous, single-turn inference. Waits for the complete response.

```python
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

response = bedrock.invoke_model(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "messages": [{"role": "user", "content": "Explain RAG in 3 sentences."}]
    }),
    contentType="application/json"
)

result = json.loads(response["body"].read())
print(result["content"][0]["text"])
```

**Exam Context:** Use when full response is acceptable (not interactive). NOT for streaming.

---

### 2. InvokeModelWithResponseStream API

**Use:** Streaming – delivers tokens as they are generated.

```python
response = bedrock.invoke_model_with_response_stream(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    body=json.dumps({...}),
    contentType="application/json"
)

stream = response.get("body")
for event in stream:
    chunk = event.get("chunk")
    if chunk:
        delta = json.loads(chunk.get("bytes").decode())
        print(delta.get("completion", ""), end="")
```

**Exam Context:** 
- Required for "character-by-character" / "real-time streaming" UI requirements
- Reduces **Time-To-First-Token (TTFT)** perceived by users
- Pair with **WebSocket API Gateway** for browser-facing apps

---

### 3. Converse API ⭐ (Most Important for Exam)

**Use:** Unified, model-agnostic conversational API. Handles multi-turn dialogue.

```python
response = bedrock.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    system=[{"text": "You are a helpful assistant."}],
    messages=[
        {"role": "user", "content": [{"text": "What is RAG?"}]},
        {"role": "assistant", "content": [{"text": "RAG stands for..."}]},
        {"role": "user", "content": [{"text": "How does chunking affect it?"}]}
    ],
    inferenceConfig={"maxTokens": 512, "temperature": 0.1}
)
```

**Critical Exam Facts:**
- Uses **structured `messages` array** — NOT plain text strings
- Each message object has `role` (`user` or `assistant`) and `content` array
- `content` is always an **array of content blocks** (text, image, document)
- Does NOT support legacy plain-text strings

---

### 4. ConverseStream API

**Use:** Streaming version of Converse API.

```python
response = bedrock.converse_stream(
    modelId="...",
    messages=[...],
    system=[...]
)
stream = response.get("stream")
for event in stream:
    if "contentBlockDelta" in event:
        print(event["contentBlockDelta"]["delta"]["text"], end="")
```

**Exam Context:** When requirements say:
- "Display responses character by character" → Use ConverseStream or InvokeModelWithResponseStream
- For interactive chat interfaces → Pair with API Gateway WebSocket + Lambda

---

### 5. RetrieveAndGenerate API

**Use:** Full RAG in a single API call using Bedrock Knowledge Bases.

```python
response = bedrock_agent_runtime.retrieve_and_generate(
    input={"text": "What are the refund policies?"},
    retrieveAndGenerateConfiguration={
        "type": "KNOWLEDGE_BASE",
        "knowledgeBaseConfiguration": {
            "knowledgeBaseId": "KB123",
            "modelArn": "arn:aws:bedrock:...",
            "retrievalConfiguration": {
                "vectorSearchConfiguration": {
                    "numberOfResults": 5,
                    "filter": {"equals": {"key": "department", "value": "finance"}}
                }
            }
        }
    }
)
```

**Exam Context:** 
- Timeout issues with complex queries → Use **AWS Amplify AI Kit** for streaming GraphQL
- Simpler than building custom RAG; use when managed RAG is acceptable

---

### 6. Retrieve API

**Use:** Retrieve-only without generation. Returns chunks for custom processing.

```python
response = bedrock_agent_runtime.retrieve(
    knowledgeBaseId="KB123",
    retrievalQuery={"text": "What are the treatment protocols?"},
    retrievalConfiguration={
        "vectorSearchConfiguration": {
            "numberOfResults": 10,
            "filter": {"greaterThan": {"key": "version", "value": "2.0"}}
        }
    }
)
```

**Exam Context:** Used for **retrieve-only evaluation** — isolating retrieval quality from generation quality.

---

## 🤖 Amazon Bedrock Knowledge Bases

### Architecture

```
S3 / SharePoint / Confluence / Salesforce / Web Crawl
          ↓
    [Data Source Connector]
          ↓
    [Data Automation / Parsing]
          ↓
    [Chunking Strategy]
          ↓
    [Embedding Model] → Vector Store (OpenSearch / Aurora / Pinecone / Redis / Mongo)
```

### Supported Data Sources

| Source | Method |
|--------|--------|
| Amazon S3 | Native; supports PDF, TXT, DOCX, HTML, MD |
| Web Crawler | URL-based crawling |
| Confluence | Native connector |
| SharePoint | Native connector |
| Salesforce | Native connector |
| Custom (Lambda) | Custom processing logic |

### Ingestion Pipeline

```python
# Trigger ingestion programmatically
response = bedrock_agent.start_ingestion_job(
    knowledgeBaseId="KB123",
    dataSourceId="DS456"
)
```

**Exam Pattern (Event-Driven Ingestion):**
```
S3 Put Event → S3 Event Notification → Lambda → StartIngestionJob
```
This is preferred over:
- EventBridge polling (too slow)  
- Glue periodic scanning (not real-time)

---

## 💬 Amazon Bedrock Prompt Management

**Purpose:** Centrally manage, version, and deploy prompt templates across teams and regions.

### Key Features

| Feature | Description |
|---------|-------------|
| **Versioning** | Immutable versions; draft stage for editing |
| **Variables** | `{{variable_name}}` placeholder syntax |
| **Variants** | Multiple prompt templates for A/B testing |
| **Audit Logging** | CloudTrail captures all prompt activities |
| **IAM Control** | Control who can create/approve prompts |

```python
# Use a prompt from Prompt Management
response = bedrock.converse(
    modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
    promptId="PMPT123",
    promptVersion="1",
    variables={"business_unit": "legal", "query": "Summarize this contract"}
)
```

**Exam Context:**
- Version control + approval workflow + audit trail → **Bedrock Prompt Management + CloudTrail**
- Team-specific tone/format → **Prompt Variants** per business unit
- NOT SageMaker Canvas, NOT S3 with tags for version control

---

## 🔄 Amazon Bedrock Flows

**Purpose:** Visual, low-code workflow builder for chaining LLM calls, conditions, and tools.

### Flow Node Types

| Node | Description |
|------|-------------|
| **Input** | Entry point for user data |
| **Output** | Returns final result |
| **LLM** | Calls a foundation model |
| **Prompt** | Uses a Prompt Management template |
| **Condition** | Branches flow based on logic |
| **Iterator** | Loops over list items |
| **Collector** | Aggregates loop results |
| **Lambda** | Custom code execution |
| **Knowledge Base** | RAG retrieval |
| **Agent** | Invokes a Bedrock Agent |

**Exam Context:** Use Flows for multi-step LLM orchestration with minimal code. Distinguished from Step Functions by being LLM-native.

---

## 🧮 Bedrock Pricing Models

### 1. On-Demand Inference (Default)

- Pay per 1,000 input tokens + per 1,000 output tokens
- No minimum commitment
- Best for: Variable, unpredictable workloads, development/testing

### 2. Provisioned Throughput

- Reserve model capacity in **Model Units (MUs)** for 1 or 6 months
- Guarantees consistent throughput
- Best for: **Sustained, high-volume, predictable traffic** (e.g., product launches, peak periods)

```
Provisioned Throughput = Guaranteed Capacity (Use for predictable, sustained high-volume)
On-Demand = Pay-as-you-go (Use for variable, spiky traffic)
```

> **Exam Trap:** For "high-demand events like product launches where traffic is predictable" → **Provisioned Throughput**  
> For "spiky, unpredictable workloads" → **On-Demand**

### 3. Batch Inference

- Submit large batches of requests via S3 JSON lines file
- 24-48 hour SLA; up to 50% cost savings vs on-demand
- Best for: **High-volume, time-tolerant offline processing** (50M items, nightly analytics)

```python
response = bedrock.create_model_invocation_job(
    roleArn="arn:aws:iam::...",
    modelId="amazon.titan-text-express-v1",
    clientRequestToken="unique-token",
    jobName="batch-embedding-job",
    inputDataConfig={"s3InputDataConfig": {"s3Uri": "s3://bucket/input/"}},
    outputDataConfig={"s3OutputDataConfig": {"s3Uri": "s3://bucket/output/"}}
)
```

### 4. Prompt Caching

- Cache the "prefix" (static system prompt + examples) across requests
- Reduces cost by up to **90%** for the cached portion
- Best for: Prompts with large, **static system instructions or few-shot examples**

```
Without caching: Pay for full prompt every request
With caching:    Pay for full prompt once; subsequent requests pay ~10%

Place static content BEFORE the cache checkpoint
```

> **Exam Trap:** Prompt caching does NOT help for unique, novel prompts (power user long queries) → For those, use **per-user token budgets via API Gateway usage plans + CountTokens API**

---

## 🌍 Cross-Region Inference

**Purpose:** Automatically route inference to other AWS regions when the primary region hits capacity limits.

```python
# Use inference profile instead of model ID directly
response = bedrock.invoke_model(
    modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",  # 'us.' prefix = cross-region
    body=json.dumps({...})
)
```

**Exam Context:**
- "Automatic recovery during regional disruptions" → **Cross-Region Inference Profiles**
- Handles failover automatically within chosen geographic scope
- Does NOT require complex multi-region infrastructure stacks

---

## 📊 Model Invocation Logging

**Purpose:** Log all Bedrock API calls for monitoring, compliance, and debugging.

```python
# Enable via console or API
bedrock.put_model_invocation_logging_configuration(
    loggingConfig={
        "cloudWatchConfig": {
            "logGroupName": "/aws/bedrock/model-invocations",
            "roleArn": "arn:aws:iam::...",
            "largeDataDeliveryS3Config": {"bucketName": "bedrock-logs"}
        },
        "s3Config": {"bucketName": "bedrock-logs"},
        "textDataDeliveryEnabled": True,
        "imageDataDeliveryEnabled": False,
        "embeddingDataDeliveryEnabled": False
    }
)
```

**What's Logged:** Model ID, request/response payload, token counts, latency, timestamps

**Exam Context:**
- Centralize visibility into token usage, latency, costs → **Model Invocation Logging to CloudWatch**
- For cost attribution by team/app → **Combine with cost allocation tags**

---

## 🚦 Intelligent Prompt Routing

**Purpose:** Automatically route requests to the most cost-effective model based on complexity.

```
Complex query → Large, expensive model (e.g., Claude Sonnet)
Simple FAQ    → Small, cheap model (e.g., Claude Haiku)
```

**Exam Context:**
- "70% simple FAQs, 30% complex tasks" → **Bedrock Intelligent Prompt Routing**
- Routes without client-side code changes
- More sophisticated than deterministic rule-based routing

---

## 📝 Practice Questions

**Q1:** A developer needs to display AI responses character-by-character in a chat interface. The application uses a React frontend with API Gateway. Which combination satisfies the requirements?

- A. API Gateway REST API + Lambda + InvokeModel  
- B. API Gateway WebSocket API + Lambda + InvokeModelWithResponseStream  
- C. API Gateway HTTP API + Lambda + RetrieveAndGenerate  
- D. Direct Bedrock API calls from the React app using IAM user credentials  

**Answer: B** – WebSocket enables bidirectional streaming; InvokeModelWithResponseStream delivers tokens as generated. Answer C doesn't stream. Answer D exposes credentials in the browser.

---

**Q2:** A company needs to process 100 million product descriptions to generate embeddings overnight. Which approach minimizes cost?

- A. On-demand inference via Lambda functions in parallel  
- B. Bedrock batch inference with an S3 input file  
- C. Provisioned Throughput with maximum model units  
- D. Real-time API calls distributed across multiple regions  

**Answer: B** – Batch inference is designed for high-volume, time-tolerant workloads at significantly lower cost.

---

**Q3:** After a code deployment, a RAG application begins generating responses from outdated company policies. Without changing the Foundation Model, what is the fastest fix?

- A. Re-train the foundation model  
- B. Apply metadata filters at query time to exclude old documents  
- C. Increase the chunk size to include more context  
- D. Switch to a higher-capacity embedding model  

**Answer: B** – Metadata filters (e.g., `effective_date > cutoff`) exclude outdated documents without model changes.

---

*Next: [03_rag_architecture.md](./03_rag_architecture.md)*
