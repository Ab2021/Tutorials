# 📊 EVALUATION DEEP DIVE — ALL ML SYSTEMS
### The Single Most Important Topic Across ALL 4 Rounds

> "How did you evaluate your model?" is asked 100% of the time at Senior DS level.
> This document covers: RAG, GenAI, Agents, Classification, Regression, Ranking, NLP, Credit Risk, CLV, Survival

---

## ╔══════════════════════════════════════════════════════╗
## ║          SECTION 1: RAG SYSTEM EVALUATION          ║
## ╚══════════════════════════════════════════════════════╝

### 1.1 THE COMPLETE RAGAS FRAMEWORK (Know Every Metric Deeply)

**Why RAGAS emerged:**
Traditional NLP metrics (ROUGE, BLEU) measure surface similarity to reference. 
For RAG: (1) No single ground truth exists. (2) Quality depends on retrieval AND generation jointly. (3) Need to detect when LLM hallucinates vs. correctly extrapolates.

**The 4 Core RAGAS Metrics:**

```
METRIC 1: FAITHFULNESS
Definition: Are all claims in the generated answer supported by the retrieved context?

Mathematical formulation:
  Faithfulness = |{statements in answer that are entailed by context}| 
                 / |{total statements in answer}|

Implementation:
  1. LLM extracts individual atomic claims from the answer
     Example answer: "The claim was filed on Jan 15 by Dr. Smith for $5,000"
     → Claims: ["filed on Jan 15", "by Dr. Smith", "for $5,000"]
  
  2. For each claim: NLI check → does context entail this claim?
     Using an NLI model (cross-encoder trained on MNLI):
     - ENTAILMENT → claim grounded (score = 1)
     - CONTRADICTION → hallucination detected
     - NEUTRAL → neither supported nor contradicted
  
  3. Faithfulness = # ENTAILED claims / # total claims

Production threshold: Faithfulness > 0.85
What 0.88 means: "12% of statements in outputs are NOT grounded in retrieved context"

Failure patterns and fixes:
  - LLM parametric memory override: LLM uses pre-training knowledge instead of context
    Fix: Add "Answer ONLY using the provided context" to system prompt
  - Missing evidence in KB: Correct claim but evidence not in retrieved docs
    Fix: Improve retrieval recall (reduce this BEFORE worrying about LLM generation)
  - Hedging misclassified: "The claim appears to be fraudulent" → atomic claim "fraudulent"
    Fix: Distinguish opinion statements (LLM inference) from factual claims

METRIC 2: ANSWER RELEVANCE
Definition: How well does the generated answer address the actual question asked?

Mathematical formulation:
  Answer Relevance = cos(embed(answer), embed(question))
  
  But RAGAS computes it more cleverly:
  1. Generate n synthetic questions from the answer using LLM
     (What question would this answer be answering?)
  2. Average cosine similarity between each synthetic question and original question
  
  Answer_Relevance = (1/n) Σ cos(q_orig, q_synthetic_i)

Why this way: Catches when answer is factually grounded BUT doesn't address the question
Example failure: Q: "When was the claim filed?" A: "The claimant has 15 years of insurance history"
→ Faithful (grounded in context) but irrelevant to question

Production threshold: > 0.80

Failure patterns:
  - Topic drift: Retrieved context about related topic → answer drifts
    Fix: Add relevance filter on retrieved docs BEFORE passing to LLM
  - Verbose but empty: Answer is long but says "I don't have enough information to..."
    Fix: Distinguish between "I don't know" and "irrelevant"

METRIC 3: CONTEXT RECALL
Definition: Does the retrieved context cover all the information needed to answer?

Mathematical formulation:
  Context Recall = |{gold reference sentences covered by context}| 
                   / |{total sentences in gold reference}|

Requires: Ground truth answer for each query (expensive to collect)

Practical implementation:
  1. Have domain experts write "ideal answers" for 100-200 test queries
  2. Break ideal answer into sentences
  3. For each sentence: does any retrieved chunk contain the same information?
     (using NLI/semantic similarity)
  4. Context Recall = % of ideal-answer sentences covered by retrieved context

What low context recall (0.71) means:
  "29% of information needed to answer correctly is NOT being retrieved"
  → Retrieval is the bottleneck, not generation

Debugging low context recall:
  Step 1: Which query types have lowest recall? (cluster failing queries)
  Step 2: For failing queries: is the relevant doc in the KB at all?
    If not in KB → cover gap (add content)
    If in KB → retrieval algorithm is failing to find it
  Step 3: If in KB but not retrieved:
    - Check: is relevant content split across chunk boundaries?
    - Check: does query vocabulary match document vocabulary?
    - Test: BM25 vs. dense retrieval — sometimes BM25 works better for exact terms

METRIC 4: CONTEXT PRECISION
Definition: Of all retrieved documents, how many are actually relevant?

Mathematical formulation:
  Context Precision = |{retrieved docs that are relevant}| / |{total retrieved docs}|

Why it matters: Irrelevant retrieved docs pollute the LLM context window
  → LLM might use irrelevant information → hallucination risk increases

Low context precision example:
  Query: "What is the return policy for electronics?"
  Retrieved: [electronics policy doc ✓, general FAQ ✗, fashion return policy ✗, ...]
  Context Precision = 1/5 = 0.20 → 80% of context is irrelevant

Impact of low context precision:
  - Wastes context window budget (fewer relevant docs fit)
  - LLM might conflate policies from different categories
  - Increases hallucination risk on details

Fixes:
  - Re-ranking: cross-encoder reranker filters irrelevant docs before LLM
  - Category filtering: metadata filter on retrieved docs (only electronics category)
  - MMR: Maximal Marginal Relevance penalizes redundant/irrelevant docs
```

### 1.2 RETRIEVAL-SPECIFIC EVALUATION METRICS

```python
def evaluate_retrieval_comprehensive(
    retriever, 
    test_queries: list[str],
    relevant_docs: list[list[str]],  # Ground truth relevant docs per query
    k_values: list[int] = [1, 3, 5, 10]
) -> dict:
    """Full retrieval evaluation suite."""
    
    metrics = {k: {"precision": [], "recall": [], "ndcg": [], "mrr": []} 
               for k in k_values}
    
    for query, gt_docs in zip(test_queries, relevant_docs):
        retrieved = retriever.retrieve(query, k=max(k_values))
        retrieved_ids = [doc.id for doc in retrieved]
        
        for k in k_values:
            retrieved_k = retrieved_ids[:k]
            
            # Precision@k
            hits = len(set(retrieved_k) & set(gt_docs))
            precision = hits / k
            
            # Recall@k
            recall = hits / len(gt_docs) if gt_docs else 0
            
            # NDCG@k (with binary relevance)
            dcg = sum(1/np.log2(i+2) for i, d in enumerate(retrieved_k) if d in gt_docs)
            idcg = sum(1/np.log2(i+2) for i in range(min(len(gt_docs), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            
            # MRR (Mean Reciprocal Rank)
            mrr = 0
            for i, doc_id in enumerate(retrieved_k):
                if doc_id in gt_docs:
                    mrr = 1 / (i + 1)
                    break
            
            metrics[k]["precision"].append(precision)
            metrics[k]["recall"].append(recall)
            metrics[k]["ndcg"].append(ndcg)
            metrics[k]["mrr"].append(mrr)
    
    # Aggregate
    return {
        f"Precision@{k}": np.mean(metrics[k]["precision"]),
        f"Recall@{k}": np.mean(metrics[k]["recall"]),
        f"NDCG@{k}": np.mean(metrics[k]["ndcg"]),
        f"MRR@{k}": np.mean(metrics[k]["mrr"])
        for k in k_values
    }
```

### 1.3 PRODUCTION RAG MONITORING DASHBOARD

```
DAILY MONITORING (automated):
┌────────────────────────────────────────────────────────────────┐
│ Metric                │ Yesterday │ 7d Trend │ Alert Threshold │
├────────────────────────────────────────────────────────────────┤
│ Faithfulness          │   0.88    │   →      │ < 0.85          │
│ Answer Relevance      │   0.84    │   ↑      │ < 0.80          │
│ Context Precision     │   0.72    │   →      │ < 0.65          │
│ Retrieval Avg Cosine  │   0.76    │   ↓⚠️    │ < 0.70          │
│ Escalation Rate       │   18%     │   ↑⚠️    │ > 25%           │
│ CSAT (sampled 100)    │   4.2/5   │   →      │ < 3.8           │
│ Latency P95           │  3.2s     │   →      │ > 5s            │
│ Token/query (avg)     │  3,800    │   →      │ > 5,000         │
│ Cache hit rate        │   61%     │   ↑      │ < 40%           │
│ Error rate            │   0.3%    │   →      │ > 1%            │
└────────────────────────────────────────────────────────────────┘

WEEKLY MONITORING (human review):
- Sample 50 queries: human evaluation of response quality
- KB staleness audit: are any documents >90 days old that need update?
- New query cluster analysis: are new question types emerging?
- LLM version performance: if LLM API updated, run full benchmark

MONTHLY:
- Full RAGAS evaluation on 500-query benchmark
- Retrieval model evaluation (has accuracy drifted?)
- Cost analysis: token consumption trends, cache ROI
- KB expansion: identify coverage gaps from low-recall queries
```

---

## ╔══════════════════════════════════════════════════════╗
## ║       SECTION 2: AGENT SYSTEM EVALUATION           ║
## ╚══════════════════════════════════════════════════════╝

### 2.1 FULL AGENT EVALUATION TAXONOMY

**Why agents are harder to evaluate than ML models:**
1. Multi-step: failure at step 3 of 7 cascades to total failure
2. Combinatorial: exponential tool call sequences possible
3. No single "ground truth" trajectory (multiple valid paths to correct answer)
4. Safety: must also evaluate what it DOESN'T do (refuse harmful requests)

```
EVALUATION DIMENSION HIERARCHY:

Level 1: COMPONENT (is each tool working correctly?)
  ├── Tool availability: does tool return valid output?
  ├── Tool schema: are inputs/outputs correctly formatted?
  └── Tool latency: is each tool within SLA?

Level 2: TRAJECTORY (is the agent taking the right path?)
  ├── Tool selection accuracy: right tool for each step?
  ├── Tool parameter accuracy: correct arguments?
  ├── Planning efficiency: minimal steps to goal?
  └── Recovery: does agent recover from tool failure?

Level 3: TASK (is the final output correct?)
  ├── Task completion rate: % of tasks fully completed
  ├── Output correctness: is final answer correct?
  ├── Output completeness: are all required elements present?
  └── Output grounding: is output based on tool results (not hallucinated)?

Level 4: SAFETY (does agent respect boundaries?)
  ├── Adversarial robustness: resists harmful instructions?
  ├── Scope adherence: stays within authorized domain?
  ├── Data protection: no PII leakage in outputs?
  └── Failure graceful: errors handled without unexpected behavior?

Level 5: SYSTEM (end-to-end business impact)
  ├── Throughput: tasks completed per hour
  ├── Cost per task: total tokens + API costs
  ├── User satisfaction: rating post-agent interaction
  └── Business KPI: did agent actions improve the business metric?
```

### 2.2 TRAJECTORY EVALUATION IN DETAIL

```python
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class ToolCall:
    tool_name: str
    parameters: dict
    output: str
    latency_ms: int
    success: bool

@dataclass
class Trajectory:
    task: str
    steps: List[ToolCall]
    final_output: str
    total_tokens: int
    total_time_s: float

def evaluate_trajectory(
    actual: Trajectory,
    expected: Trajectory,
    parameter_weight: float = 0.5
) -> dict:
    """
    Compare actual vs expected agent trajectory.
    
    Scoring:
    - Correct tool + correct params = 1.0
    - Correct tool + wrong params = 0.5
    - Wrong tool = 0.0
    """
    
    scores = []
    step_analysis = []
    
    # Align trajectories (allow for different lengths)
    for i, actual_step in enumerate(actual.steps):
        # Find best matching expected step
        best_match = None
        best_score = 0
        
        for expected_step in expected.steps:
            if actual_step.tool_name == expected_step.tool_name:
                # Tool matches — check parameters
                param_similarity = compute_param_similarity(
                    actual_step.parameters, 
                    expected_step.parameters
                )
                score = 1.0 if param_similarity > 0.9 else parameter_weight
                if score > best_score:
                    best_score = score
                    best_match = expected_step
        
        scores.append(best_score)
        step_analysis.append({
            "step": i,
            "actual_tool": actual_step.tool_name,
            "expected_tool": best_match.tool_name if best_match else "N/A",
            "score": best_score,
            "correct": best_score == 1.0
        })
    
    # Penalties for extra unnecessary steps
    extra_steps = max(0, len(actual.steps) - len(expected.steps))
    efficiency_penalty = extra_steps * 0.1
    
    trajectory_score = np.mean(scores) - efficiency_penalty
    
    return {
        "trajectory_score": max(0, trajectory_score),
        "tool_accuracy": np.mean([s == 1.0 for s in scores]),
        "efficiency": len(expected.steps) / len(actual.steps),  # 1.0 = same length
        "step_analysis": step_analysis,
        "extra_steps": extra_steps
    }


def evaluate_agent_benchmark(agent, benchmark_tasks: list) -> dict:
    """Run full agent evaluation on benchmark."""
    
    results = {
        "task_completion_rate": [],
        "trajectory_scores": [],
        "answer_correctness": [],
        "safety_pass": [],
        "avg_steps": [],
        "avg_tokens": []
    }
    
    for task in benchmark_tasks:
        # Run agent
        output = agent.run(task.query)
        trajectory = agent.get_last_trajectory()
        
        # Task completion
        completed = output is not None and len(output) > 0
        results["task_completion_rate"].append(completed)
        
        # Trajectory evaluation
        traj_score = evaluate_trajectory(trajectory, task.expected_trajectory)
        results["trajectory_scores"].append(traj_score["trajectory_score"])
        
        # Answer correctness
        if task.has_ground_truth:
            correctness = evaluate_answer(output, task.ground_truth)
        else:
            correctness = llm_judge_evaluate(task.query, output)
        results["answer_correctness"].append(correctness)
        
        # Safety
        results["safety_pass"].append(check_safety(output, task.safety_constraints))
        
        # Efficiency
        results["avg_steps"].append(len(trajectory.steps))
        results["avg_tokens"].append(trajectory.total_tokens)
    
    return {
        "task_completion_rate": np.mean(results["task_completion_rate"]),
        "avg_trajectory_score": np.mean(results["trajectory_scores"]),
        "avg_answer_correctness": np.mean(results["answer_correctness"]),
        "safety_pass_rate": np.mean(results["safety_pass"]),
        "avg_steps_per_task": np.mean(results["avg_steps"]),
        "avg_tokens_per_task": np.mean(results["avg_tokens"])
    }
```

### 2.3 LLM-AS-JUDGE IMPLEMENTATION

```python
def llm_judge_evaluate(
    query: str,
    response: str,
    context: str = "",
    judge_model: str = "gpt-4o"
) -> dict:
    """
    Use a strong LLM to evaluate another LLM's output.
    Key: Judge prompt must be explicit, with concrete scoring rubric.
    """
    
    judge_prompt = f"""
You are an expert evaluator assessing an AI assistant's response quality.

USER QUERY: {query}

RETRIEVED CONTEXT (what the AI had access to):
{context}

AI RESPONSE: {response}

Evaluate the response on FIVE dimensions. For each, provide:
- Score: 1 (very poor) to 5 (excellent)  
- One-sentence justification

DIMENSIONS:
1. ACCURACY: Are all factual claims in the response correct and supported by the context?
   (5=all correct, 1=multiple errors)

2. COMPLETENESS: Does the response fully address the user's query?
   (5=fully addresses all aspects, 1=misses main point)

3. FAITHFULNESS: Are all claims grounded in the provided context (not hallucinated)?
   (5=fully grounded, 1=significant hallucination detected)

4. CLARITY: Is the response clear, well-structured, and appropriate in length?
   (5=excellent, 1=confusing or inappropriate)

5. SAFETY: Does the response avoid harmful outputs, PII exposure, or scope violations?
   (5=completely safe, 1=safety violation detected)

Return ONLY valid JSON:
{{
  "accuracy": {{"score": X, "justification": "..."}},
  "completeness": {{"score": X, "justification": "..."}},
  "faithfulness": {{"score": X, "justification": "..."}},  
  "clarity": {{"score": X, "justification": "..."}},
  "safety": {{"score": X, "justification": "..."}},
  "overall_score": X,
  "primary_issue": "main issue if any or 'none'"
}}
"""
    
    response = call_llm(judge_prompt, model=judge_model, temperature=0)
    return parse_json_response(response)


class LLMJudgePipeline:
    """Production LLM evaluation pipeline with bias mitigation."""
    
    def __init__(self, judge_model="gpt-4o", sample_rate=0.01):
        self.judge = judge_model
        self.sample_rate = sample_rate  # Evaluate 1% of prod traffic
        
    def evaluate_with_position_bias_mitigation(self, query, response_A, response_B):
        """
        When comparing two responses, avoid position bias.
        Evaluate both orders and average.
        """
        score_AB = self.compare(query, response_A, response_B, order="AB")
        score_BA = self.compare(query, response_B, response_A, order="BA")
        
        # Average to cancel position bias
        prefer_A = (score_AB["winner"] == "A" and score_BA["winner"] == "B")
        prefer_B = (score_AB["winner"] == "B" and score_BA["winner"] == "A")
        
        return {
            "winner": "A" if prefer_A else "B" if prefer_B else "tie",
            "confidence": abs(score_AB["margin"] + score_BA["margin"]) / 2
        }
```

---

## ╔══════════════════════════════════════════════════════╗
## ║     SECTION 3: TRADITIONAL ML EVALUATION           ║
## ╚══════════════════════════════════════════════════════╝

### 3.1 FRAUD DETECTION — COMPLETE EVALUATION SUITE

```python
def evaluate_fraud_model_complete(
    model, X_test, y_test, 
    cost_fn={"FN": 8500, "FP": 200},  # Business costs in ₹
    capacity=500  # Max investigations/day
) -> dict:
    """Complete fraud model evaluation — 4 layers."""
    
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # === LAYER 1: RANK-ORDERING METRICS ===
    auc_roc = roc_auc_score(y_test, y_proba)
    auc_pr = average_precision_score(y_test, y_proba)
    
    # KS Statistic (separation between fraud/non-fraud distributions)
    fraud_scores = y_proba[y_test == 1]
    good_scores = y_proba[y_test == 0]
    all_thresholds = np.sort(y_proba)
    ks_values = []
    for thresh in all_thresholds:
        cum_fraud = (fraud_scores <= thresh).mean()
        cum_good = (good_scores <= thresh).mean()
        ks_values.append(abs(cum_fraud - cum_good))
    ks_stat = max(ks_values)
    gini = 2 * auc_roc - 1
    
    # === LAYER 2: CALIBRATION ===
    brier = brier_score_loss(y_test, y_proba)
    # Reliability diagram buckets
    bins = np.linspace(0, 1, 11)
    reliability = []
    for i in range(len(bins)-1):
        mask = (y_proba >= bins[i]) & (y_proba < bins[i+1])
        if mask.sum() > 0:
            predicted_rate = y_proba[mask].mean()
            actual_rate = y_test[mask].mean()
            reliability.append((predicted_rate, actual_rate))
    
    # === LAYER 3: BUSINESS METRICS ===
    # Threshold based on investigation capacity constraint
    def find_capacity_threshold(y_proba, capacity, n_samples):
        target_flag_rate = capacity / n_samples
        return np.percentile(y_proba, (1 - target_flag_rate) * 100)
    
    thresh = find_capacity_threshold(y_proba, capacity, len(y_test))
    y_pred = (y_proba >= thresh).astype(int)
    
    tp = ((y_pred == 1) & (y_test == 1)).sum()
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()
    
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    
    # Business value
    value_saved = tp * cost_fn["FN"]  # Fraud caught
    fp_cost = fp * cost_fn["FP"]      # Customer friction
    net_value = value_saved - fp_cost
    
    # Decile lift
    df_result = pd.DataFrame({"score": y_proba, "label": y_test})
    df_result["decile"] = pd.qcut(df_result["score"], 10, labels=False, duplicates="drop")
    decile_stats = df_result.groupby("decile")["label"].mean().sort_index(ascending=False)
    baseline_rate = y_test.mean()
    top_decile_lift = decile_stats.iloc[0] / baseline_rate
    
    # === LAYER 4: FAIRNESS ===
    # Would need demographic features — placeholder
    fairness = {"note": "Run subgroup AUC when demographic features available"}
    
    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "ks_statistic": ks_stat,
        "gini": gini,
        "brier_score": brier,
        "reliability_diagram": reliability,
        "operating_threshold": thresh,
        "precision_at_capacity": precision,
        "recall_at_capacity": recall,
        "net_business_value_rs": net_value,
        "top_decile_lift": top_decile_lift,
        "fairness": fairness
    }
```

### 3.2 CREDIT RISK — VINTAGE ANALYSIS

```python
def vintage_analysis(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gold standard for credit model evaluation over time.
    
    predictions_df columns: user_id, approval_date, score_band, default_flag
    
    Shows: for cohort approved in period X, what % defaulted by 3/6/12 months?
    """
    
    predictions_df['approval_month'] = pd.to_datetime(
        predictions_df['approval_date']
    ).dt.to_period('M')
    
    vintage_table = []
    
    for cohort, cohort_df in predictions_df.groupby('approval_month'):
        cohort_result = {"cohort": str(cohort)}
        
        for months in [1, 3, 6, 12]:
            cutoff_date = cohort_df['approval_date'] + pd.DateOffset(months=months)
            # Only users who have been observed for 'months' duration
            eligible = cohort_df[cohort_df['approval_date'] <= 
                                  pd.Timestamp.now() - pd.DateOffset(months=months)]
            
            if len(eligible) > 0:
                default_rate = (eligible['default_flag'] == 1).mean()
                cohort_result[f"dpd_{months}m"] = round(default_rate, 4)
        
        vintage_table.append(cohort_result)
    
    return pd.DataFrame(vintage_table).set_index('cohort')

# Expected output shape:
# Cohort  | DPD 1M | DPD 3M | DPD 6M | DPD 12M
# 2024-01 |  1.2%  |  2.8%  |  4.1%  |   5.6%
# 2024-02 |  1.1%  |  2.6%  |  3.9%  |   5.2%  ← stable trend = model working
# 2024-03 |  1.5%  |  3.4%  |    ?   |     ?   ← check if 3M rate rising = concern
```

### 3.3 RECOMMENDATION SYSTEM EVALUATION

```python
def evaluate_recommendation_system(
    recommender,
    test_interactions: pd.DataFrame,  # user_id, item_id, timestamp, purchased
    k_values: list = [5, 10, 20],
    use_temporal_split: bool = True
) -> dict:
    """
    Complete recommendation system evaluation.
    CRITICAL: Always use temporal split, never random split.
    """
    
    if use_temporal_split:
        # Last 20% of each user's interactions as test set
        test_interactions = test_interactions.sort_values('timestamp')
        split_point = int(len(test_interactions) * 0.8)
        test_set = test_interactions.iloc[split_point:]
        train_set = test_interactions.iloc[:split_point]
    
    results = {k: [] for k in k_values}
    diversity_scores = []
    novelty_scores = []
    
    for user_id in test_set['user_id'].unique():
        # Ground truth: items user actually engaged with in test period
        gt_items = set(test_set[test_set['user_id'] == user_id]['item_id'])
        
        for k in k_values:
            recs = recommender.recommend(user_id, k=k)
            rec_ids = [r.item_id for r in recs]
            
            # Precision@k
            hits = len(set(rec_ids) & gt_items)
            precision = hits / k
            
            # NDCG@k
            dcg = sum(1/np.log2(i+2) for i, item in enumerate(rec_ids) if item in gt_items)
            idcg = sum(1/np.log2(i+2) for i in range(min(len(gt_items), k)))
            ndcg = dcg / idcg if idcg > 0 else 0
            
            # HitRate@k
            hit_rate = 1 if len(set(rec_ids) & gt_items) > 0 else 0
            
            results[k].append({
                "precision": precision,
                "ndcg": ndcg,
                "hit_rate": hit_rate
            })
        
        # Diversity: Are recommendations diverse in category?
        categories = [get_item_category(r.item_id) for r in recommender.recommend(user_id, k=10)]
        diversity = len(set(categories)) / len(categories)  # ILD (Intra-List Diversity)
        diversity_scores.append(diversity)
        
        # Novelty: Are recommendations unpopular/novel items (not just bestsellers)?
        popularities = [get_item_popularity(r.item_id) for r in recommender.recommend(user_id, k=10)]
        novelty = 1 - np.mean(popularities)  # Higher = more novel
        novelty_scores.append(novelty)
    
    return {
        **{f"Precision@{k}": np.mean([r["precision"] for r in results[k]]) for k in k_values},
        **{f"NDCG@{k}": np.mean([r["ndcg"] for r in results[k]]) for k in k_values},
        **{f"HitRate@{k}": np.mean([r["hit_rate"] for r in results[k]]) for k in k_values},
        "Diversity": np.mean(diversity_scores),
        "Novelty": np.mean(novelty_scores),
        "Coverage": len(all_recommended_items) / len(all_items)  # Catalog coverage
    }
```

### 3.4 NLP SYSTEMS EVALUATION

```python
# ─── BERT NER Evaluation ───
from seqeval.metrics import classification_report, f1_score

def evaluate_ner_complete(predictions, ground_truth, entity_types):
    """Per-entity-type evaluation with adversarial test cases."""
    
    # Standard evaluation
    overall_report = classification_report(ground_truth, predictions)
    
    # Per-entity evaluation
    per_entity = {}
    for entity_type in entity_types:
        # Filter only this entity type
        pred_filtered = [[t if t.endswith(entity_type) or t == 'O' else 'O' 
                         for t in sent] for sent in predictions]
        gt_filtered = [[t if t.endswith(entity_type) or t == 'O' else 'O'
                       for t in sent] for sent in ground_truth]
        
        per_entity[entity_type] = {
            "f1": f1_score(gt_filtered, pred_filtered),
            "entity_count": sum(1 for sent in gt_filtered for t in sent 
                              if t.startswith("B"))
        }
    
    # Adversarial case evaluation
    adversarial_cases = [
        ("negation", "patient shows NO signs of {entity}"),
        ("abbreviation", "Dx: ICD {code}, Proc: {procedure}"),
        ("cross_sentence", "[sentence1] ... [sentence2] references entity from sentence1"),
        ("multi_occurrence", "Date of {event1}: {date1}. Date of {event2}: {date2}.")
    ]
    
    adversarial_results = {}
    for case_name, template in adversarial_cases:
        adversarial_results[case_name] = evaluate_on_adversarial_set(
            predictions, case_name
        )
    
    return {
        "overall_f1": f1_score(ground_truth, predictions),
        "per_entity": per_entity,
        "adversarial": adversarial_results,
        "inter_annotator_agreement": compute_cohens_kappa(ground_truth)
    }

# ─── Summarization Evaluation ───
from rouge_score import rouge_scorer
from bert_score import score as bert_score

def evaluate_summarization(generated_summaries, reference_summaries):
    """Multi-metric summarization evaluation."""
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_scores = {k: [] for k in ['rouge1_f', 'rouge2_f', 'rougeL_f']}
    
    for gen, ref in zip(generated_summaries, reference_summaries):
        scores = scorer.score(ref, gen)
        rouge_scores['rouge1_f'].append(scores['rouge1'].fmeasure)
        rouge_scores['rouge2_f'].append(scores['rouge2'].fmeasure)
        rouge_scores['rougeL_f'].append(scores['rougeL'].fmeasure)
    
    # BERTScore (semantic similarity, language-model based)
    P, R, F1 = bert_score(
        generated_summaries, reference_summaries, 
        lang='en', model_type='bert-base-uncased'
    )
    
    # Factual consistency (LLM-as-judge)
    factual_scores = [
        llm_judge_factual_consistency(gen, ref) 
        for gen, ref in zip(generated_summaries, reference_summaries)
    ]
    
    return {
        "ROUGE-1 F1": np.mean(rouge_scores['rouge1_f']),
        "ROUGE-2 F1": np.mean(rouge_scores['rouge2_f']),
        "ROUGE-L F1": np.mean(rouge_scores['rougeL_f']),
        "BERTScore F1": F1.mean().item(),
        "Factual Consistency": np.mean(factual_scores)
    }
```

---

## ╔══════════════════════════════════════════════════════╗
## ║    SECTION 4: EVALUATION INTERVIEW Q&A             ║
## ╚══════════════════════════════════════════════════════╝

### Q: "How did you evaluate your fraud RAG system?" (Perfect 90-second answer)

> "I evaluated at four layers. **Component** first: I independently assessed the BERT extraction layer on 200 gold-labeled claims achieving F1 >0.85 per entity type, and evaluated the embedding model separately using Precision@5 and MRR on 200 query-relevant document pairs — my domain fine-tuned model improved Precision@5 by 18% over the OpenAI ada-002 baseline.
>
> **Pipeline level** using RAGAS: Faithfulness 0.88 (12% of statements had grounding issues — fixed with stricter JSON schema output), Answer Relevance 0.84, Context Recall 0.71 (retrieval missed 29% of relevant evidence — addressed by expanding KB with synthetic fraud descriptions), Context Precision 0.72.
>
> **Business validation**: 60-day shadow mode comparing system flags to delayed investigator verdicts — 78% recall at 22% FPR, which the business found acceptable.
>
> **Production monitoring**: Evidently AI for weekly PSI on claim distributions, weekly faithfulness sampling via LLM-as-judge, and investigator precision tracking for the feedback loop."

---

### Q: "How do you evaluate an agent beyond task completion rate?"

> "Task completion rate alone is binary and coarse. I use 5 dimensions:
>
> **Trajectory accuracy** — I record the full execution trace and compare tool selection and parameters against expected sequences I manually annotated for 50 benchmark tasks. Scored with a matrix: correct tool + correct params = 1.0, correct tool + wrong params = 0.5, wrong tool = 0.
>
> **Answer correctness** — for factual outputs I compare against ground truth SQL queries (exact match). For analytical summaries I use LLM-as-judge with explicit rubric: accuracy, completeness, faithfulness, and a safety check.
>
> **Reliability** — running the same query 10 times at temperature=0, expecting < 5% output variance. And paraphrase consistency: 5 rephrased versions of same query should produce semantically equivalent answers.
>
> **Safety robustness** — explicit adversarial test suite: SQL injection in query, requests for unauthorized data access, out-of-scope questions. Agent must gracefully refuse all of these with 0 safety failures.
>
> **Efficiency** — average tokens per task and tool calls per task. An agent becoming less efficient over time (rising token count) is flagging degraded reasoning."

---

### Q: "What's the difference between ROUGE and BERTScore? When is one better?"

> "ROUGE is n-gram overlap: ROUGE-1 is unigram, ROUGE-2 bigram, ROUGE-L longest common subsequence. It's fast and interpretable but purely lexical — 'fraudulent claim' and 'suspicious case' score zero overlap, even though they're semantically equivalent.
>
> BERTScore computes token-level cosine similarity in contextual embedding space (BERT representations). Captures paraphrases, synonyms, domain-specific terminology differences.
>
> For claims summarization, I use BERTScore as primary and ROUGE as secondary: BERTScore catches that 'Dr. Smith identified potential fraud' and 'Fraudulent activity was flagged by the reviewing physician' are semantically equivalent. ROUGE would penalize this.
>
> Neither catches factual errors. That's why I add LLM-as-judge factual consistency: a summary that says 'filed February 15' when the claim was filed January 15 can score high on ROUGE and BERTScore but is factually wrong. Only LLM-as-judge catches this."

---

*See companion files: 02_Project_Deep_Dives_All_Projects.md, 10_Edge_Cases_And_Failure_Modes.md*
