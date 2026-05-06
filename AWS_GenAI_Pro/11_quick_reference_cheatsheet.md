# Quick Reference Cheat Sheet — AIP-C01
## Last-Minute Exam Revision Guide

---

## 🚀 BEDROCK APIS

| API | Use When |
|-----|---------|
| `InvokeModel` | Synchronous, full response, non-streaming |
| `InvokeModelWithResponseStream` | Streaming tokens to client |
| `Converse` | Multi-turn chat, model-agnostic |
| `ConverseStream` | Streaming multi-turn chat |
| `RetrieveAndGenerate` | Full RAG in one API call |
| `Retrieve` | Retrieval only (no generation) |
| `CountTokens` | Measure prompt size before inference |
| `StartIngestionJob` | Trigger KB sync from data source |

**Converse API Message Format (Must Know):**
```json
{
  "messages": [{"role": "user", "content": [{"text": "query"}]}],
  "system": [{"text": "You are..."}]
}
```

---

## 📦 CHUNKING STRATEGIES

| Strategy | Best For | Exam Trigger |
|----------|---------|-------------|
| Fixed-size | General purpose | Default starting point |
| Fixed-size + overlap | Boundary continuity | "Prevent info loss at boundaries" |
| Semantic | Narrative docs | Complex, topic-coherent docs |
| **Hierarchical** ⭐ | Long structured docs | "Precise retrieval + full context" |
| No chunking | Short atomic docs | Self-contained small documents |
| Custom Lambda | Domain-specific | Proprietary structure |

---

## 💰 INFERENCE MODES

| Mode | Cost | Latency | When |
|------|------|---------|------|
| On-Demand | Per token | Real-time | Variable/spiky traffic |
| **Provisioned Throughput** | Committed | Guaranteed | Sustained high-volume, predictable |
| **Batch Inference** | 50% cheaper | 24-48h | High-volume, time-tolerant |
| Prompt Caching | -90% on cached | Same | Static system prompt + examples |

---

## 🛡️ GUARDRAIL TYPES

| Type | Blocks | Example |
|------|--------|---------|
| Content Filters | Harmful content (hate/violence/sexual) | Set HATE=HIGH |
| **Denied Topics** ⭐ | Business topic areas | "No competitor discussion" |
| **Word Filters** ⭐ | Specific words/brands | Block "CompetitorX" |
| PII Filtering | Personal data | ANONYMIZE SSN |
| Grounding Check | Ungrounded responses | HIGH threshold = strictest |

**High grounding threshold** = STRICT (blocks ungrounded responses)  
**Low grounding threshold** = PERMISSIVE (allows creative latitude)

---

## 🔑 SECURITY PATTERNS

| Requirement | Solution |
|-------------|---------|
| Private Bedrock network | Interface VPC Endpoint (PrivateLink) |
| Department model access | IAM Identity Center + Permission Sets |
| Column-level cross-account | Lake Formation LF-tag grants |
| PII in logs blocked | Guardrails + Model Invocation Logging |
| Physical data residency | AWS Outposts + de-identify first |
| Training data immutability | S3 Object Lock (COMPLIANCE) + CloudTrail |
| Model access control | IAM policies on model ARNs |
| Data lineage | Glue Data Catalog + SageMaker Model Registry |

---

## 📊 EVALUATION METRICS

| Symptom | Metric to Check |
|---------|----------------|
| Wrong answers | Correctness, Faithfulness |
| Incomplete answers | Completeness |
| Cites irrelevant docs | **Citation Precision, Context Relevance** |
| Retrieves wrong docs | Context Recall, Context Precision |
| Inappropriate tone | Coherence, Helpfulness |
| Bias in outputs | SageMaker Clarify fairness metrics |

**Why embedding similarity fails:** Measures semantic closeness NOT factual accuracy. 10mg vs 100mg = ~0.97 cosine similarity but factually wrong.

**Why response hashing fails:** LLMs are non-deterministic → same correct answer = different hash each run.

---

## 🏗️ SERVICE SELECTION SHORTCUTS

| "Need to..." | Service |
|-------------|---------|
| Stream tokens to browser | API GW WebSocket + InvokeModelWithResponseStream |
| Switch models without code deploy | **AWS AppConfig** |
| Route simple vs complex queries | Bedrock Intelligent Prompt Routing |
| Debug agent infinite loops | **Agent Traces** |
| Process 50M items overnight | **Bedrock Batch Inference** |
| Parse complex PDFs for KB | **Bedrock Data Automation (BDA)** |
| Extract invoice forms/tables | Amazon Textract |
| Audit who called which model | **AWS CloudTrail** |
| Debug latency across services | **AWS X-Ray** |
| Monitor Bedrock costs by team | Cost Allocation Tags + Model Invocation Logs |
| Human approval in workflow | **Step Functions + Task Tokens** |
| Continuous bias monitoring | SageMaker Model Monitor + Clarify |
| Human review of bad responses | **Amazon A2I** |
| Tool consistency across frameworks | **MCP (Model Context Protocol)** |
| Auto-ingest S3 files to KB | S3 Event → Lambda → StartIngestionJob |
| Manage prompt templates centrally | Bedrock Prompt Management + CloudTrail |
| Enterprise assistant + corp data | **Amazon Q Business** |
| Rollback bad Lambda deployment | **CodeDeploy Canary + CloudWatch alarm** |
| Validate data before training | **AWS Glue Data Quality** |
| Eliminate Lambda cold starts | **Provisioned Concurrency** |
| Multi-agent shared memory | **AgentCore Memory** |
| Expose REST APIs as agent tools | **AgentCore Gateway + OpenAPI** |
| LLM pipeline with visual editor | **Bedrock Flows** |
| Complex orchestration + retries | **Step Functions** |
| Cross-region failover | **Cross-Region Inference Profiles** |
| Mobile 5G low latency | **AWS Wavelength** |

---

## ⚙️ VECTOR SEARCH OPTIMIZATION

```
Problem: Many small shards → high latency
Fix: Consolidate to 30-50 GB per shard

Problem: High storage at scale
Fix: Reduce embedding dimensions (1024→512 or 256)

Problem: Misses exact terms (acronyms)
Fix: Hybrid search (vector + BM25 keyword)

Problem: Retrieves outdated documents
Fix: Metadata filters at query time

Problem: Correct docs but insufficient context
Fix: Hierarchical chunking
```

---

## 🧠 ARCHITECTURAL PRINCIPLES (Apply to Every Question)

1. **Managed > Custom** — AWS managed services over custom code
2. **Root cause > Symptom** — Fix data quality at source, not model behavior
3. **IAM layer > App layer** — Security at AWS resource layer
4. **Defense-in-depth** — WAF → Comprehend → Guardrails → Lambda
5. **Event-driven > Polling** — S3 events, not EventBridge schedules
6. **Batch for bulk** — >1M items, time-tolerant → Batch Inference
7. **AppConfig for config** — Config changes without code deployment
8. **Traces for WHY, Metrics for WHAT**
9. **Streaming for interactive** — WebSocket + streaming API
10. **Summarize, don't truncate** — Long conversation management

---

## 📌 DOMAIN WEIGHTS REMINDER

```
Domain 1 (FM Integration/Data)  ████████████ 31% ← Heaviest!
Domain 2 (Implementation)       ██████████  26%
Domain 3 (Safety/Security)      ████████    20%
Domain 4 (Optimization)         █████       12%
Domain 5 (Testing)              ████        11%
```

**65 scored + 10 unscored = 75 questions | 130 minutes | Pass = 750/1000**

---

*Good luck! Focus on the WHY behind each answer, not just memorization.*
