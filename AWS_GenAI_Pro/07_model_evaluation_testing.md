# Model Evaluation, Testing & Troubleshooting
## AIP-C01 – Domain 5 (11%) – Complete Guide

---

## 🎯 Bedrock Model Evaluation Framework

Amazon Bedrock Model Evaluations is a managed service for assessing FM quality:

```
Test Dataset (JSONL with prompts + ground truth)
    ↓
[Bedrock Model Evaluation Job]
    ├── Automated (LLM-as-Judge approach)
    └── Human (bring your own workforce)
    ↓
[Evaluation Report] → Metrics per prompt, averages, distributions
    ↓
[CloudWatch] → Metrics published for monitoring
    ↓
[CI/CD Quality Gate] → Pass/Fail deployment decision
```

---

## 📊 Evaluation Metrics Reference

### Automated Metrics (LLM-as-Judge)

| Metric | What It Measures | When to Use |
|--------|-----------------|-------------|
| **Correctness** | Factual accuracy vs. ground truth | Agent responses, factual Q&A |
| **Completeness** | Coverage of required key points | Summarization, comprehensive answers |
| **Faithfulness** | Grounded in retrieved context? | RAG hallucination detection |
| **Helpfulness** | Does it address the user's need? | General assistant quality |
| **Coherence** | Logical flow and readability | Tone/style complaints |
| **Context Precision** | % of retrieved chunks that are relevant | RAG retrieval quality |
| **Context Recall** | % of relevant chunks actually retrieved | RAG coverage |
| **Citation Precision** | Citations support the claims made | RAG citation accuracy |

### Metric Selection Guide

```
Problem: "Responses are factually wrong"
→ Use: Correctness, Faithfulness

Problem: "Responses miss key requirements"  
→ Use: Completeness

Problem: "Cites irrelevant documents"
→ Use: Citation Precision, Context Relevance

Problem: "Tone is inappropriate"
→ Use: Coherence, Helpfulness

Problem: "Users prefer old prompts vs new prompts"
→ Use: Bedrock Model Evaluations Compare Feature
```

---

## ❌ Why Embedding Similarity ≠ Accuracy

> **Critical Exam Concept**

```
Response A: "The treatment dosage is 10mg twice daily."
Response B: "The treatment dosage is 100mg twice daily."

Cosine similarity between A and B: ~0.97 (very high!)
But B is FACTUALLY WRONG and could harm patients.
```

**Why Embedding Similarity Fails:**
- Measures SEMANTIC PROXIMITY, not FACTUAL CORRECTNESS
- Similar wording ≠ similar meaning
- Cannot detect subtle factual errors (10mg vs 100mg)
- Handles paraphrasing but not factual deviation

**Why Correctness/Completeness Metrics Succeed:**
- Compare against known-correct ground truth
- Model-based judge evaluates factual equivalence
- Handles natural paraphrasing (unlike exact keyword match)
- Detects both omissions (completeness) and errors (correctness)

---

## 🔁 RAG Regression Testing

### The Non-Determinism Problem

```
Same input → LLM → Different output (temperature > 0)

Therefore:
Response hash at time T1: "8f3a2b..."
Response hash at time T2: "4c9d7e..." (different hash, same correct answer)
```

**Why response hashing fails for regression testing:**
- LLMs sample probabilistically → different words each time
- Even correct answers will produce different hashes
- 100% false positive rate for detecting "regressions"

### Correct Approach: Fixed Test Dataset + Model Evaluation

```python
# Regression test dataset (stored permanently)
test_dataset = [
    {
        "prompt": "What is the refund policy for digital products?",
        "referenceResponse": "Digital products are non-refundable within 30 days of purchase unless..."
    },
    {
        "prompt": "How do I reset my password?",
        "referenceResponse": "Go to the login page, click 'Forgot Password', enter your email..."
    }
]

# After each KB update, run evaluation
eval_job = bedrock.create_evaluation_job(
    evaluationConfig={
        "automated": {
            "datasetMetricConfigs": [{
                "taskType": "QuestionAndAnswer",
                "dataset": {"name": "regression-v1"},
                "metricNames": ["Correctness", "Completeness"]
            }]
        }
    }
)

# In CI/CD pipeline - fail if quality drops below baseline
if eval_results["correctness_avg"] < BASELINE_THRESHOLD:
    raise Exception("Quality regression detected! Halting deployment.")
```

### RAG Regression Testing Flow

```
Document Update (S3 event)
    ↓
[StartIngestionJob] → KB updated
    ↓
[Run Regression Test Suite]
    ├── Fixed 50 test questions with expected answers
    ├── Bedrock Model Evaluation → Correctness/Completeness
    └── Compare to baseline metrics
    ↓
Quality OK? → Proceed to production
Quality Degraded? → Alert, rollback document, investigate
```

---

## 🤖 LLM-as-Judge Evaluation

### How LLM-as-Judge Works

```
Reference Answer: "..."
Generated Answer: "..."
Judge Prompt: "On a scale of 1-5, rate the correctness of the Generated Answer 
               compared to the Reference Answer. Consider: factual accuracy, 
               completeness, no hallucinations. Explain your rating."
Judge LLM: "Rating: 4/5. The generated answer correctly identifies..."
```

**Advantages over human evaluation:**
- Scalable to millions of evaluations
- Consistent scoring criteria
- Handles paraphrasing (unlike keyword overlap)
- Fast (minutes vs. days for human review)

**Limitations:**
- Judge model can have its own biases
- May struggle with domain-specific technical accuracy
- Not suitable for subjective quality assessments

---

## 📈 Pre-Production Model Comparison

### Bedrock Model Evaluation Compare Feature

```
Test Dataset
    ├── Run with Prompt Variant A → Metric Set A
    └── Run with Prompt Variant B → Metric Set B
    ↓
Side-by-side comparison report
```

**Use Cases:**
- Compare two prompt approaches before launch
- Evaluate model upgrade (e.g., Claude 3 Haiku → Claude 3.5 Haiku)
- A/B test system prompts for tone/format
- Validate fine-tuned model vs. base model

> **Exam Tip:** "Compare foundation models on the same test data offline" → **Bedrock Model Evaluations with Compare Feature**  
> NOT A/B testing on live production traffic (risky, delayed feedback)

---

## 🩺 RAG Evaluation – Specific Scenarios

### Scenario 1: Retrieval Correct, Generation Wrong

```
Symptom: Retrieved docs contain the answer, but model response is incorrect
Metrics to check: Faithfulness, Correctness
Root cause: Model hallucinating beyond context, or context not clearly structured
Fix: Improve system prompt, increase grounding threshold in Guardrails
```

### Scenario 2: Generation Correct, Citations Wrong

```
Symptom: Answers are factually correct but cite irrelevant documents
Metrics to check: Citation Precision, Context Relevance
Root cause: Vector search returning semantically similar but contextually irrelevant chunks
Fix: Implement hybrid search, tune retrieval filter thresholds
```

### Scenario 3: Retrieval Missing Key Documents

```
Symptom: Some questions get wrong answers because relevant docs aren't retrieved
Metrics to check: Context Recall
Root cause: Chunking too large (semantic dilution), missing metadata, wrong embedding model
Fix: Adjust chunk size, improve metadata tagging, consider semantic chunking
```

### Retrieve-Only Evaluation ⭐

```python
# Evaluate ONLY retrieval quality (isolate from generation)
response = bedrock_agent_runtime.retrieve(
    knowledgeBaseId="KB123",
    retrievalQuery={"text": "What are the treatment protocols for diabetes?"}
)

# Manually assess retrieved chunks:
# - Are these chunks relevant to the query?
# - Do they contain the expected information?
# - Is context recall sufficient?
```

> **Exam Pattern:** "Validating retrieval relevance before deployment" → **Retrieve-only evaluation to isolate retrieval layer from generation layer**

---

## 🔍 Observability & Debugging

### AWS X-Ray for GenAI Pipelines

```
End-to-end trace for a single user request:

[API Gateway: 5ms] → [Lambda (cold start: 800ms)] → [Bedrock (TTFT: 350ms, total: 2.1s)]
                                                     [Knowledge Base retrieve: 120ms]
```

**X-Ray subsegments for Bedrock:**

```python
import aws_xray_sdk.core as xray_core

@xray_core.capture('rag_pipeline')
def handle_request(query):
    with xray_core.in_subsegment('retrieve') as subsegment:
        chunks = retrieve_from_kb(query)
        subsegment.put_metadata('chunks_retrieved', len(chunks))
    
    with xray_core.in_subsegment('generate') as subsegment:
        response = invoke_bedrock(query, chunks)
        subsegment.put_metadata('output_tokens', response['usage']['output_tokens'])
    
    return response
```

**What X-Ray Identifies:**
- Lambda cold starts vs warm execution time
- Bedrock model inference latency
- Knowledge Base retrieval time
- Network hops between services

> **Exam Pattern:** "Debug slow Bedrock agent where latency source is unknown" → **AWS X-Ray with subsegments**

---

## 🚀 CI/CD Quality Gates for GenAI

### Pipeline with Bedrock Evaluation

```
Developer pushes code
    ↓
[CodePipeline Stage 1: Build & Unit Test]
    ↓
[CodePipeline Stage 2: RAG Evaluation]
    ├── Run evaluation job against test dataset
    ├── Check: Correctness ≥ 0.85
    ├── Check: Faithfulness ≥ 0.90
    └── Check: No regression from baseline
    ↓ (Pass)
[CodePipeline Stage 3: Canary Deploy via CodeDeploy]
    ├── 10% traffic to new Lambda version
    ├── Monitor CloudWatch error rates
    └── Auto-rollback if error rate > 1%
    ↓ (Pass)
[Full Production Deploy]
```

### CodeDeploy for Safe Lambda Deployment

```python
# CodeDeploy deployment configuration
{
  "deploymentConfigName": "LambdaCanary10Percent5Minutes",
  "computePlatform": "Lambda",
  "trafficRoutingConfig": {
    "type": "TimeBasedCanary",
    "timeBasedCanary": {
      "canaryPercentage": 10,
      "canaryInterval": 5  # minutes
    }
  }
}
```

**CodeDeploy Auto-Rollback Triggers:**
```python
"autoRollbackConfiguration": {
    "enabled": True,
    "events": ["DEPLOYMENT_FAILURE", "DEPLOYMENT_STOP_ON_ALARM"],
    "alarms": [{"name": "BedrockErrorRateHigh"}]
}
```

> **Exam Pattern:** "Automatically rollback Lambda if error rates spike after deployment" → **AWS CodeDeploy with Canary/Linear traffic shifting**

---

## ⚡ Performance Diagnostics

### Time-To-First-Token (TTFT) Optimization

```
TTFT = Time from sending request to receiving first token

TTFT is affected by:
1. Network latency (between client and API Gateway)
2. Lambda cold start (provisioned concurrency eliminates this)
3. Model inference startup
4. Bedrock processing overhead

Solution for minimizing TTFT:
1. Use streaming APIs (tokens arrive as generated)
2. Use latency-optimized performance configuration
3. Lambda Provisioned Concurrency (no cold starts)
4. Choose region closest to users
```

### Lambda Provisioned Concurrency

```python
lambda_client.put_provisioned_concurrency_config(
    FunctionName="bedrock-proxy-lambda",
    Qualifier="prod",
    ProvisionedConcurrentExecutions=50  # Always-warm execution environments
)
```

> **Exam Pattern:** "Minimize TTFT for interactive chat" → **Streaming APIs + latency-optimized performance config**  
> "Eliminate Lambda cold starts for GenAI API" → **Lambda Provisioned Concurrency**

---

## 🧹 Data Quality & Pre-Processing

### Amazon Comprehend for Text Normalization

```python
# Clean noisy transcripts before sending to Bedrock
comprehend = boto3.client('comprehend')

def normalize_transcript(raw_text: str) -> str:
    """Remove fillers, normalize entities."""
    # Detect language
    language = comprehend.detect_dominant_language(Text=raw_text)
    
    # Detect entities (for normalization context)
    entities = comprehend.detect_entities(
        Text=raw_text,
        LanguageCode=language['Languages'][0]['LanguageCode']
    )
    
    # Apply normalization rules
    clean_text = remove_filler_words(raw_text)  # um, uh, like
    clean_text = standardize_numbers(clean_text)  # fifteen → 15
    
    return clean_text
```

> **Exam Pattern:** "Normalize noisy transcripts from Amazon Transcribe before inference" → **Lambda + Amazon Comprehend to clean at source** (NOT fine-tune model on noisy data)

### Why NOT Fine-Tune on Noisy Data

```
❌ Fine-tuning on noisy data:
1. Teaches model to expect and perpetuate noise
2. Expensive to re-train
3. Doesn't fix the root cause
4. Noise patterns change over time → must retrain again

✅ Clean at source:
1. Addresses root cause
2. One-time pre-processing logic
3. Works with any downstream model
4. Cheaper than fine-tuning
```

---

## 🔬 AWS Glue Data Quality

### Automated Data Validation

```python
# AWS Glue Data Quality rule set
rules = """
Rules = [
    IsComplete "customer_id",
    ColumnValues "age" between 18 and 120,
    ColumnLength "email" > 5,
    IsUnique "order_id",
    ColumnValues "status" in ["pending", "processing", "shipped", "delivered"]
]
"""
```

> **Exam Pattern:** "Enforce schema and data integrity before AI model training" → **AWS Glue Data Quality with declarative rule sets**  
> Preferred over Lambda-based custom validation (less code, managed service)

---

## 📝 Practice Questions (Evaluation & Testing)

**Q1:** A RAG application for a retail company returns correct answers but frequently cites documents from outdated product catalogs. Which evaluation metrics should the team examine?

- A. Faithfulness and Coherence  
- B. Citation Precision and Context Relevance  
- C. ROUGE score and BLEU score  
- D. Embedding cosine similarity  

**Answer: B** – Citation precision measures whether citations actually support the claims; context relevance measures whether retrieved passages are relevant to the query.

---

**Q2:** After updating the knowledge base content, a team notices that some previously correct answers are now wrong. What is the MOST reliable approach to detect these regressions automatically?

- A. Compare response hashes before and after update  
- B. Monitor user satisfaction scores after deployment  
- C. Maintain a regression test dataset with expected answers; run Bedrock model evaluations after each ingestion  
- D. Compare embedding similarity of new responses to old responses  

**Answer: C** – Fixed test dataset + model evaluations with correctness metrics is the only reliable approach. Hashes fail due to LLM non-determinism; user satisfaction and embedding similarity are inadequate.

---

**Q3:** A company wants to compare two prompt variants for their customer support assistant to determine which produces more accurate responses BEFORE deploying to production. What is the recommended approach?

- A. Deploy both variants to production and run A/B testing  
- B. Use Bedrock Model Evaluations with the compare feature on a test dataset  
- C. Ask customer service managers to manually review 50 sample responses per variant  
- D. Use embedding similarity to compare responses from both variants  

**Answer: B** – Pre-production, offline evaluation using the compare feature is safer than A/B testing live users. More scalable than manual review. More accurate than embedding similarity.

---

*Next: [08_performance_cost_optimization.md](./08_performance_cost_optimization.md)*
