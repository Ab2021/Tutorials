# Performance, Cost Optimization & Operational Excellence
## AIP-C01 – Domain 4 (12%) – Complete Guide

---

## 💰 Cost Optimization Strategy

### Cost Driver Hierarchy

```
Biggest Cost Drivers:
1. Output tokens (2-5x more expensive than input tokens)
2. Model size (large models cost significantly more)
3. Real-time inference for batch-able tasks
4. Redundant context (static content re-processed every request)
5. Inefficient vector search (too many dimensions at scale)
```

### Cost Optimization Levers

| Lever | Savings Potential | Implementation |
|-------|------------------|----------------|
| **Prompt Caching** | Up to 90% on cached tokens | Cache static system prompt + examples |
| **Batch Inference** | Up to 50% vs on-demand | S3 JSONL input file + batch job |
| **Intelligent Prompt Routing** | 40-60% on mixed traffic | Route simple queries to smaller models |
| **Lower Embedding Dimensions** | 50-75% storage/compute | Use 256-512 instead of 1024 dimensions |
| **Optimized Chunking** | Reduce index size | Semantic or hierarchical chunking |
| **Per-User Budgets** | Control outlier costs | API Gateway + CountTokens API |

---

## 📊 Inference Modes Decision Tree

```
New Request Arrives
     ↓
Is response needed in < 5 seconds?
     ├─ YES → Is it a one-off complex task?
     │          ├─ YES → On-Demand Inference
     │          └─ NO → Is traffic predictable and sustained high-volume?
     │                    ├─ YES → Provisioned Throughput
     │                    └─ NO → On-Demand Inference
     └─ NO → Can it wait 24-48 hours?
               └─ YES → Batch Inference (50% cheaper)
```

### On-Demand Inference

**Characteristics:**
- Pay per input + output tokens
- No minimum commitment
- Automatic scaling
- Best for variable/unpredictable workloads

**Pricing example (Claude 3.5 Sonnet):**
```
Input: $0.003 per 1,000 tokens
Output: $0.015 per 1,000 tokens
On-demand request of 1,000 in + 500 out = ~$0.0105
```

---

### Provisioned Throughput

**Characteristics:**
- Reserve capacity in Model Units (MUs)
- 1-month or 6-month commitment
- Guaranteed response capacity
- Best for predictable, sustained high-volume

**When to choose provisioned:**
```
✅ Product launches with known traffic patterns
✅ Daily batch workloads with consistent peaks
✅ SLA-bound applications hitting quota limits
✅ 24/7 high-volume production systems

❌ Development/testing
❌ Unpredictable spiky traffic
❌ Low-volume applications
```

> **Exam Trap:** "Variable workloads with spiky traffic" → **On-Demand** (NOT Provisioned Throughput)  
> **Exam Trap:** "Guaranteed capacity for product launch" → **Provisioned Throughput** (NOT caching)

---

### Batch Inference ⭐

**Characteristics:**
- Submit large batch via S3 JSONL file
- 24-48 hour completion SLA
- Up to 50% cost reduction vs on-demand
- Fully managed parallelism

**Input File Format (JSONL):**
```jsonl
{"recordId": "1", "modelInput": {"inputText": "Summarize: ...first product..."}}
{"recordId": "2", "modelInput": {"inputText": "Summarize: ...second product..."}}
{"recordId": "50000000", "modelInput": {"inputText": "Summarize: ...last product..."}}
```

**Job Configuration:**
```python
response = bedrock.create_model_invocation_job(
    jobName="product-embedding-batch-20250115",
    modelId="amazon.titan-embed-text-v2:0",
    roleArn="arn:aws:iam::ACCOUNT:role/BedrockBatchRole",
    inputDataConfig={
        "s3InputDataConfig": {
            "s3Uri": "s3://data-bucket/batch-input/",
            "s3InputFormat": "JSONL"
        }
    },
    outputDataConfig={
        "s3OutputDataConfig": {
            "s3Uri": "s3://data-bucket/batch-output/"
        }
    }
)
```

> **Exam Pattern:** "Process 50 million items where 24-hour lead time is acceptable" → **Batch Inference** (NOT Lambda parallel, NOT ECS)

---

## ⚡ Performance Optimization

### Latency Components

```
User → API Gateway (5-15ms)
         ↓
       Lambda (0ms warm / 300-800ms cold start)
         ↓
       Bedrock Knowledge Base Retrieve (50-200ms)
         ↓
       Bedrock Model Inference (200ms-30s depending on output length)
         ↓
       Response to User
```

### Optimization by Component

| Component | Optimization | Impact |
|-----------|-------------|--------|
| **Lambda cold start** | Provisioned Concurrency | Eliminates 300-800ms cold start |
| **Model TTFT** | Streaming + latency-optimized config | Improves perceived responsiveness |
| **KB retrieval latency** | Optimize shard size in OpenSearch | Reduces coordination overhead |
| **Network latency** | Edge deployment (Wavelength, CloudFront) | Lower physical distance |
| **Token processing** | Reduce output token count | Direct correlation to latency |

---

### Cross-Region Inference for Resiliency

```python
# Cross-region inference profile (us.* prefix)
# Automatically routes to available region within US scope
response = bedrock.invoke_model(
    modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    # 'us.' prefix = will route across US regions (us-east-1, us-west-2) as needed
    body=json.dumps({...})
)
```

**Available Scopes:**
- `us.*` – Routes across US regions
- `eu.*` – Routes across EU regions (for data residency requirements)

**Benefits:**
- Automatic failover during regional outages
- Better capacity utilization across regions
- No complex multi-region infrastructure needed

> **Exam Pattern:** "Automatic recovery during regional service disruptions" → **Cross-Region Inference Profiles** (NOT custom Route 53 health checks + manual failover)

---

## 🔢 Vector Search Optimization

### OpenSearch Index Optimization

#### The Shard Size Problem

```
Problematic configuration: 10 million documents × 1,000 documents/shard = 10,000 shards
Result: Every query must coordinate across 10,000 shards → Massive overhead

Optimal configuration: 10 million documents → ~20 shards @ 30-50 GB each
Result: Minimal coordination overhead → Fast queries
```

#### Optimal Shard Sizing

```
Target: 30-50 GB per shard
Formula: Total index size / Target shard size = Number of shards

Example:
- 10M documents × 4 KB per vector (1024 dims) = ~40 GB
- 40 GB / 35 GB per shard ≈ 2 primary shards (with 1-2 replicas)
```

#### HNSW Parameter Tuning

```json
{
  "settings": {
    "index": {
      "knn": true,
      "knn.algo_param.ef_search": 512  // Higher = more accurate, slower
    }
  },
  "mappings": {
    "properties": {
      "embedding": {
        "type": "knn_vector",
        "dimension": 1024,
        "method": {
          "name": "hnsw",
          "space_type": "cosine",
          "parameters": {
            "ef_construction": 512,  // Higher = better index quality, slower build
            "m": 16                  // Connections per node; 16-48 typical
          }
        }
      }
    }
  }
}
```

#### Embedding Dimensionality Trade-off

```
                  Accuracy
                    ↑
High (1024) ────────────── High precision, high cost
                 ↑ Sweet spot for most use cases
Medium (512) ──────────── Good precision, moderate cost
                 ↑
Low (256) ──────────────── Lower precision, lowest cost
                    
                    Cost/Latency →
```

> **Exam Pattern:** "10M item catalog with high query latency and storage costs" → **Reduce embedding dimensionality** (NOT remove data, NOT switch models)

---

## 📡 Edge Deployment Patterns

### AWS Wavelength for Mobile Users

```
Mobile User (5G Network)
    ↓ (Sub-millisecond)
[AWS Wavelength Zone] ← Compute at 5G network edge
    ↓ (Private, low-latency)
[Amazon API Gateway + Lambda] ← Application logic
    ↓ (AWS backbone)
[Amazon Bedrock] ← Regional inference
```

**Use Case:** Mobile apps needing ultra-low latency (real-time gaming, AR, interactive AI assistants on 5G)

> **Exam Pattern:** "Low-latency access for mobile 5G users" → **AWS Wavelength**

### CloudFront for Global Distribution

```python
# CloudFront in front of API Gateway for:
# 1. Global edge caching of static AI responses (e.g., FAQ answers)
# 2. Lower latency for geographically distributed users
# 3. DDoS protection via AWS Shield

cloudfront_distribution = {
    "DefaultCacheBehavior": {
        "TargetOriginId": "api-gateway-origin",
        "ViewerProtocolPolicy": "https-only",
        "CachePolicyId": "deterministic-caching-policy"
    }
}
```

---

## 📊 Cost Attribution & Monitoring

### Cost Allocation Tags

```python
# Tag Bedrock resources for cost allocation
bedrock_agent.create_knowledge_base(
    name="CustomerSupportKB",
    tags={
        "Department": "CustomerService",
        "Application": "SupportChatbot",
        "Environment": "Production",
        "CostCenter": "CC-1234"
    }
)
```

### Model Invocation Logging for Cost Attribution

```python
# Log configuration with S3 delivery
logging_config = {
    "cloudWatchConfig": {
        "logGroupName": "/aws/bedrock/invocations",
        "roleArn": "...",
    },
    "s3Config": {
        "bucketName": "bedrock-invocation-logs",
        "keyPrefix": "logs/"
    },
    "textDataDeliveryEnabled": True,
    "embeddingDataDeliveryEnabled": True
}

# Analyze with Athena
query = """
SELECT 
    json_extract(request, '$.modelId') as model,
    json_extract(response, '$.usage.inputTokens') as input_tokens,
    json_extract(response, '$.usage.outputTokens') as output_tokens,
    tags['Application'] as application,
    DATE(eventTime) as date
FROM bedrock_invocation_logs
GROUP BY model, application, date
"""
```

> **Exam Pattern:** "Granular insight into Bedrock usage by team or application" → **Cost Allocation Tags + Model Invocation Logs**

### CloudWatch Metrics for Bedrock

```
Key Bedrock CloudWatch Metrics:
- InvocationsThrottled: Count of throttled requests
- InvocationsClientErrors: 4xx errors
- InvocationsServerErrors: 5xx errors
- InvocationsFirstByteLatency: Time to first byte (TTFT)
- InvocationsTotalDuration: End-to-end latency
- InvocationsInputTokenCount: Input tokens per request
- InvocationsOutputTokenCount: Output tokens per request
- GuardrailsInvocationsIntervened: Guardrail blocks
```

---

## 🔄 Operational Patterns

### Serverless GenAI API Architecture

```
Client
  ↓ HTTPS
[API Gateway] ← WAF protection, rate limiting, auth
  ↓
[Lambda with Provisioned Concurrency] ← No cold starts
  ├── Pre-processing (Comprehend)
  ├── Bedrock Retrieval
  ├── Bedrock Generation
  └── Post-processing
  ↓
[Response]

Optional: [SQS Queue] for async high-volume processing
```

### SQS for Asynchronous GenAI Processing

```python
# For long-running generation tasks (>30 seconds):
# 1. API Gateway → SQS (immediate acknowledgment to client)
# 2. Lambda consumes from SQS → Bedrock → Store result
# 3. Client polls for result (or receives SNS notification)

sqs.send_message(
    QueueUrl=SQS_URL,
    MessageBody=json.dumps({
        "request_id": "req-123",
        "prompt": user_prompt,
        "callback_url": "https://app.example.com/results/req-123"
    })
)
```

### Step Functions Circuit Breaker

```python
# Step Functions state machine - Circuit Breaker pattern
{
  "states": {
    "TryBedrockCall": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "Retry": [{"ErrorEquals": ["States.TaskFailed"], "MaxAttempts": 2}],
      "Catch": [{"ErrorEquals": ["States.ALL"], "Next": "CircuitOpen"}]
    },
    "CircuitOpen": {
      "Type": "Task",
      "Resource": "arn:aws:states:::lambda:invoke",
      "Parameters": {"FunctionName": "get-fallback-response"},
      "End": True
    }
  }
}
```

---

## 🏗️ Well-Architected GenAI Lens

The AWS Well-Architected Framework – Generative AI Lens defines best practices across 6 pillars:

| Pillar | Key Practices |
|--------|--------------|
| **Operational Excellence** | Model Invocation Logging, CI/CD with evaluation gates, AppConfig for config management |
| **Security** | IAM least-privilege, VPC endpoints, Guardrails, KMS encryption |
| **Reliability** | Cross-region inference, Circuit Breaker, fallback strategies, retry with backoff |
| **Performance Efficiency** | Streaming APIs, optimal chunking, shard sizing, TTFT optimization |
| **Cost Optimization** | Batch inference, prompt caching, intelligent routing, dimensionality reduction |
| **Sustainability** | Right-size models, batch workloads, avoid over-provisioning |

> **Exam Pattern:** "Authoritative source for GenAI architectural guidance" → **AWS Well-Architected Framework – Generative AI Lens** (NOT internal wikis, NOT general whitepapers)

---

## 📝 Practice Questions (Performance & Cost)

**Q1:** A company's OpenSearch vector index has millions of very small shards and experiences high query latency during peak hours. What is the recommended fix?

- A. Add more OpenSearch instances to the cluster  
- B. Increase embedding dimensions from 512 to 1024  
- C. Consolidate shards into fewer, larger shards in the 30-50 GB range  
- D. Switch to Aurora PostgreSQL with pgvector  

**Answer: C** – Shard consolidation reduces coordination overhead. More instances don't fix shard coordination issues. Higher dimensions increase cost without fixing latency caused by shard count.

---

**Q2:** A media company needs to generate summaries for 200 million articles. The summaries don't need to be ready for 48 hours. What minimizes cost?

- A. Lambda with 10 concurrent executions calling on-demand Bedrock  
- B. ECS tasks with auto-scaling processing articles in parallel  
- C. Amazon Bedrock batch inference with S3 input and output  
- D. Provisioned Throughput with maximum model units  

**Answer: C** – Batch inference is purpose-built for this pattern: high-volume, time-tolerant, 50% cheaper than on-demand.

---

**Q3:** After a new Lambda function version is deployed, the AI response quality degraded significantly. Before the next deployment, what mechanism should be implemented?

- A. Increase Lambda memory to improve processing speed  
- B. Add CodeDeploy canary deployment with CloudWatch alarms triggering auto-rollback  
- C. Deploy to a separate region and compare responses manually  
- D. Enable CloudTrail logging for the deployment  

**Answer: B** – CodeDeploy canary routing with CloudWatch error rate alarms provides automatic rollback protection during deployments.

---

*Next: [09_architecture_patterns.md](./09_architecture_patterns.md)*
