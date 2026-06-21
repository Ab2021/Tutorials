# Solutions Architect ML/AI — Interview Prep
## 08: Production Deployment Deep Dive + Marketing AI Alignment
**Candidate:** Abhishek Bhardwaj | **Company:** Huge | **Role:** Solutions Architect — ML/AI

> **Reading strategy:** Part A is technical depth for engineering/architecture rounds. Part B is business-to-technical translation for leadership and account team rounds. Know both cold.

---

# PART A: PRODUCTION DEPLOYMENT DEEP DIVE

---

## A1: Multi-Region Deployment for AI Systems

### Architecture Decision: Active-Active vs Active-Passive

| Dimension | Active-Active | Active-Passive |
|---|---|---|
| Availability | Higher (both regions serve traffic) | Lower (failover latency ~30–120s) |
| Cost | 2× compute cost | ~1.1× (passive is scaled down) |
| Consistency | Harder (write conflicts, stale reads) | Easier (single write source) |
| Latency | Local-region serving | Passive region adds latency during failover |
| Best for | Latency-sensitive inference (chatbots, ad personalization) | Batch scoring, analytics workloads |

**Rule of thumb for ML serving:** Use **active-active** for real-time inference endpoints where p99 latency SLA < 200ms. Use **active-passive** for batch pipelines and model training where a 2-minute failover is acceptable.

---

### GCP Multi-Region: us-central1 + europe-west1 + asia-southeast1

**Baseline topology for a Huge marketing AI platform:**

```
Global Load Balancer (anycast IP)
├── us-central1: Vertex AI Endpoint (us-central1-a, b, c)
│   └── Cloud Spanner (us-region)
├── europe-west1: Vertex AI Endpoint (europe-west1-b, c, d)
│   └── Cloud Spanner (eur3 multi-region config)
└── asia-southeast1: Vertex AI Endpoint (asia-southeast1-a, b, c)
    └── Cloud Spanner (asia-region)
```

**Latency targets with this topology:**
- NA users → us-central1: ~15ms median
- EU users → europe-west1: ~12ms median
- APAC users → asia-southeast1: ~20ms median
- Cross-region fallback: +60–120ms (still sub-200ms for most use cases)

---

### Replicating a Vertex AI Vector Search Index Across Regions

This is a nuanced question. Vertex AI Vector Search **does not have native cross-region replication** as of 2025. The correct answer:

**Step 1: Source of truth in Cloud Storage**
```bash
# Export index from primary region
gcloud ai indexes export \
  --index=projects/my-project/locations/us-central1/indexes/INDEX_ID \
  --export-destination=gs://my-bucket/vector-indexes/us-central1/

# Trigger replication to other regions via Cloud Storage Transfer
gsutil -m cp -r gs://my-bucket/vector-indexes/us-central1/ \
               gs://my-bucket/vector-indexes/europe-west1/
```

**Step 2: Deploy indexes independently per region**
```python
from google.cloud import aiplatform

def deploy_regional_index(project_id, region, gcs_path):
    aiplatform.init(project=project_id, location=region)

    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=f"campaign-embeddings-{region}",
        contents_delta_uri=gcs_path,
        dimensions=768,
        approximate_neighbors_count=150,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
    )
    return index

# Deploy to all three regions
for region, path in [
    ("us-central1",      "gs://us-bucket/indexes/"),
    ("europe-west1",     "gs://eu-bucket/indexes/"),   # EU-resident bucket!
    ("asia-southeast1",  "gs://apac-bucket/indexes/"),
]:
    deploy_regional_index("my-project", region, path)
```

**Step 3: Sync strategy — event-driven via Pub/Sub**
```
New document ingested
    → Embedding generated in primary region
    → Embedding + ID published to Pub/Sub topic (global)
    → Regional subscribers (europe-west1, asia-southeast1) upsert into local index
    → Propagation lag: ~1–5 minutes (acceptable for content that changes hourly)
```

---

### Latency-Based Routing: Cloud Load Balancer Configuration

```yaml
# backend-service.yaml
name: marketing-ai-backend
loadBalancingScheme: EXTERNAL
protocol: HTTPS
backends:
  - group: projects/my-proj/regions/us-central1/instanceGroups/ai-us
    balancingMode: RATE
    maxRatePerInstance: 1000
    capacityScaler: 1.0
  - group: projects/my-proj/regions/europe-west1/instanceGroups/ai-eu
    balancingMode: RATE
    maxRatePerInstance: 1000
    capacityScaler: 1.0
  - group: projects/my-proj/regions/asia-southeast1/instanceGroups/ai-apac
    balancingMode: RATE
    maxRatePerInstance: 1000
    capacityScaler: 1.0
localityLbPolicy: LEAST_REQUEST  # route to nearest healthy backend
connectionDraining:
  drainingTimeoutSec: 30
```

Use **Premium Tier networking** for anycast routing — this routes EU traffic over Google's backbone, not the public internet, shaving ~40ms vs standard tier.

---

### Data Residency: EU AI Act + GDPR Compliance

**Key constraint:** EU user personal data (behavioral signals, cookie IDs, purchase history) cannot leave EU soil for processing.

**Implementation pattern:**

| Layer | US | EU | APAC |
|---|---|---|---|
| Embedding model | Vertex AI us-central1 | Vertex AI europe-west1 (separate deployment) | Vertex AI asia-southeast1 |
| Vector index | us-central1 | europe-west1 (EU-resident index with EU user data only) | asia-southeast1 |
| Feature Store | us-central1 | europe-west1 (separate entity group per region) | asia-southeast1 |
| Model weights | Shared via Cloud Storage Transfer | Weights are not PII — OK to replicate | Weights are not PII |
| PII/behavioral data | US users only | EU users only (GDPR Art.44 restricted) | APAC users only |

**Critical distinction in interviews:** Model *weights* are not PII and can cross borders. *User behavioral data* (inputs to the model) must be geo-fenced. Never conflate the two.

---

### RTO and RPO for an AI System

| Component | RPO Target | RTO Target | Recovery Mechanism |
|---|---|---|---|
| Vertex AI endpoint | 0 (stateless) | < 5 min | Redeploy from model registry; Cloud Run autoscales from 0 |
| Vector Search index | 1 hour (last hourly snapshot) | < 30 min | Redeploy from GCS snapshot |
| Feature Store | 15 min (Cloud Spanner PITR) | < 15 min | Cloud Spanner point-in-time restore |
| Model artifacts (GCS) | 0 (multi-region GCS) | < 1 min | Multi-region bucket; automatic failover |
| Training pipeline | 24 hours (last checkpoint) | < 4 hours | Restart from Vertex Pipelines checkpoint |
| Prompt templates (Config) | 0 (Git-versioned, config stored in GCS) | < 2 min | Reapply from Git tag |

**Key insight for interviews:** AI systems have *multiple* RPO/RTO dimensions. A traditional web app has one database. An AI system has a model, an index, a feature store, a prompt template, and potentially fine-tuning data — each with different recovery profiles.

---

### Disaster Recovery Runbook

**Scenario: Vertex AI endpoint failure in us-central1 during McDonald's breakfast rush (6–9 AM)**

```
T+0:00  Alert fires: Error rate > 5% on us-central1 endpoint
        Cloud Monitoring → PagerDuty → On-call SA

T+0:02  Automated health check: Cloud Scheduler pings /health every 30s
        If 3 consecutive failures → trigger failover runbook via Cloud Functions

T+0:03  Automated: Update Cloud Armor policy to route us-central1 traffic → us-east4
        (us-east4 is warm standby, min-instances=2 keeps it from cold start)

T+0:05  Manual verification: On-call SA confirms fallback endpoint serving
        Latency check: NA users on us-east4 → p50 <50ms, p99 <200ms ✓

T+0:15  Root cause investigation begins
        Check: Vertex AI quota exhaustion? Node pool failure? Model corruption?

T+1:00  If root cause is model corruption: redeploy from model registry
        gcloud ai endpoints undeploy-model --endpoint=ENDPOINT_ID --deployed-model-id=MODEL_ID
        gcloud ai endpoints deploy-model ENDPOINT_ID --model=MODEL_ID --display-name=recovery

T+2:00  Traffic gradually shifted back to us-central1 (10% → 25% → 50% → 100%)
        Use Cloud Load Balancer traffic splitting: capacityScaler gradual increase

T+4:00  Post-mortem initiated. Metrics collected. Blameless review scheduled.
```

**Vector Search index corruption:**
1. Stop all write traffic to the index immediately (Feature Flags → kill switch)
2. Identify last known-good snapshot in GCS (check hourly snapshots)
3. Create new index from snapshot: `aiplatform.MatchingEngineIndex.create_tree_ah_index(contents_delta_uri=last_good_snapshot)`
4. Deploy to endpoint: `index_endpoint.deploy_index(index=new_index)`
5. Run validation suite: top-k recall test on 1000 known query-document pairs
6. If recall > 0.90: restore write traffic via Feature Flag toggle

---

## A2: Cost Architecture for AI Systems at Scale

### TCO Breakdown: Production RAG System on GCP

**Assumptions:** 1M queries/day, 10M document corpus, 768-dim embeddings, Gemini 1.5 Flash for generation

| Component | Unit | Volume/Day | Unit Cost | Daily Cost | Monthly Cost |
|---|---|---|---|---|---|
| **Embedding (query)** | Gecko embeddings | 1M queries | $0.000025/1K chars (~150 chars avg) | $3.75 | $112 |
| **Vector Search** | Queries to index | 1M queries | $0.000004/query | $4.00 | $120 |
| **Vector Search index storage** | 10M × 768-dim | Static | $0.00029/GB-hr | $56/mo | $56 |
| **LLM Generation** | Gemini 1.5 Flash | 1M queries × 500 output tokens | $0.000075/1K tokens output | $37.50 | $1,125 |
| **LLM Input tokens** | Context (3K tokens avg) | 1M queries | $0.000019/1K tokens input | $57.00 | $1,710 |
| **Cloud Run (serving)** | vCPU-seconds | Scaled for 1M req | $0.000024/vCPU-s | ~$40 | $1,200 |
| **Cloud Armor** | Security policy | 1M requests | $0.000006/req | $6.00 | $180 |
| **Logging/Monitoring** | Log ingestion | ~50GB/day | $0.50/GB | $25.00 | $750 |
| **TOTAL** | | | | **~$229** | **~$5,253** |

**Key insight:** LLM generation tokens are 80–85% of variable cost. Retrieval and embedding are nearly negligible.

---

### Cost Per Query Decomposition

```
Total cost per query at 1M/day: $5,253/month ÷ 30M queries = $0.000175/query = 0.0175¢

Breakdown:
  Embedding:       0.00038¢  (2.2%)
  Vector Search:   0.00040¢  (2.3%)
  LLM input:       0.00570¢  (32.6%)
  LLM output:      0.00375¢  (21.4%)
  Infra (serving): 0.00400¢  (22.9%)
  Storage/logging: 0.00277¢  (15.8%)
  Other:           0.00050¢  (2.9%)

Levers to reduce cost:
  1. Reduce output token length: prompt engineering (biggest lever)
  2. Cache frequent queries: Redis on Cloud Memorystore (hit rate 20–30% = 20–30% LLM cost reduction)
  3. Model tiering: use Flash for simple queries, only route complex to Pro
  4. Batch embedding ingestion: use batch prediction API (30% cheaper than online)
```

---

### GCP Committed Use Discounts for ML Workloads

| Workload Type | Recommended Commitment | Savings |
|---|---|---|
| Persistent serving (Cloud Run / GKE) | 1-year CUD on vCPUs | 20–37% |
| Vertex AI training (GPU) | On-demand — training is bursty | 0% (use Spot instead) |
| Vertex AI endpoints | 1-year CUD on prediction nodes | ~30% |
| Cloud Spanner (Feature Store backend) | 1-year CUD | 20% |
| BigQuery (analytics / feature engineering) | Flat-rate slots (100 slots) if >$3K/mo query cost | 40–60% |

**Spot/Preemptible VMs for Training:**
- ✅ **Safe to use:** Non-resumable training < 4 hours; jobs with Vertex checkpoint callbacks every 10 min; hyperparameter tuning (individual trials are short)
- ❌ **Unsafe:** Single-shot training runs > 8 hours with no checkpointing; jobs where preemption means restarting from scratch; production inference (never use spot for serving)

---

### Autoscaling Strategy: Cold Start vs Cost Cap

```python
# Cloud Run service config for RAG serving
resource "google_cloud_run_v2_service" "rag_api" {
  name     = "rag-api"
  location = "us-central1"

  template {
    scaling {
      min_instance_count = 2    # Avoid cold starts; ~$200/mo always-on cost
      max_instance_count = 100  # Hard cap: prevent runaway cost
    }
    containers {
      image = "gcr.io/my-project/rag-api:latest"
      resources {
        limits = {
          cpu    = "2"
          memory = "4Gi"
        }
        cpu_idle = false  # Keep CPU allocated between requests (faster response)
      }
    }
  }
}
```

**Cold start math:** At 2 min-instances × $0.000024/vCPU-s × 3600s × 24h × 30d = ~$124/month to eliminate cold starts. For a client SLA with p99 < 200ms, this is non-negotiable.

---

### "The client's AI budget is $50K/month — what can you build?"

**Model answer:**

"$50K/month is a solid budget that can support an enterprise-grade AI platform. Here's how I'd allocate it:

| Category | Allocation | What It Buys |
|---|---|---|
| LLM inference (Gemini) | $15,000 (30%) | ~5M complex queries/month or 15M Flash queries |
| Vector Search + embeddings | $3,000 (6%) | 100M document corpus, 10M queries/month |
| Infrastructure (Cloud Run, GKE) | $8,000 (16%) | Active-active 2-region deployment |
| Data pipeline (Dataflow, BQ) | $7,000 (14%) | Daily feature engineering for 10M customers |
| MLOps (Vertex Pipelines, Experiments) | $5,000 (10%) | Weekly model retraining, experiment tracking |
| Security + Compliance | $4,000 (8%) | VPC SC, audit logs, CMEK, Cloud Armor |
| Monitoring + Observability | $3,000 (6%) | Cloud Monitoring, custom dashboards |
| Buffer / spike capacity | $5,000 (10%) | Handles 2–3× traffic spikes without budget breach |

What I'd *not* build at this budget: GPU-based fine-tuning in production (costs $20K+ alone), multi-model ensemble serving (too expensive). I'd use Gemini APIs and focus budget on data quality and retrieval quality instead."

---

## A3: SLA Design and Incident Response

### SLI / SLO / SLA Framework for AI Systems

**Standard web SLAs are insufficient for AI.** You need additional dimensions:

| Dimension | SLI (what you measure) | SLO (internal target) | SLA (contractual) |
|---|---|---|---|
| Availability | % of requests returning 2xx | 99.9% | 99.5% |
| Latency | p99 end-to-end response time | < 500ms | < 1,000ms |
| Accuracy | Retrieval recall@5 on golden set | > 0.85 | > 0.80 |
| Hallucination rate | % responses flagged by guardrail model | < 2% | < 5% |
| Staleness | Age of most recent model/index update | < 24 hours | < 48 hours |
| Throughput | Queries/second sustained | 500 QPS | 400 QPS |

**Error budget math:**
- 99.9% uptime SLO → 8.7 hours downtime/year → 43.8 minutes/month
- If you spend your error budget in week 1 on a botched deployment, all planned changes are frozen until month resets
- This disciplines deployment cadence for AI systems

---

### Incident Response: AI-Specific Failures

**Scenario: Fraud detection model's false positive rate tripled overnight**

```
T+0:00  Alert: false_positive_rate > 15% (threshold: 5%)
        SLI breach → error budget consumed → P1 incident

T+0:05  On-call checks: Was there a model deployment in the last 24 hours?
        YES → immediate rollback candidate
        NO  → data drift or upstream data pipeline issue

T+0:10  Pull feature distribution report (Vertex Model Monitoring):
        Compare today's input feature distribution vs training distribution
        Using KL divergence / Population Stability Index

T+0:20  Hypothesis confirmed: 'transaction_country' feature shows new value
        distribution — likely a new market launched by client

T+0:30  Immediate mitigation: raise decision threshold from 0.5 → 0.7
        This reduces false positives at the cost of more false negatives
        Acceptable short-term (missing fraud < blocking legitimate customers)

T+1:00  Longer-term fix: retrain model with new country distribution
        Data team: pull last 30 days including new market transactions
        ML team: retrain + validate + shadow deploy before production swap

T+48:00 New model deployed via blue-green; false positive rate returns to 4.8%
```

**Post-mortem template:**
```markdown
## Incident Post-Mortem: [Incident ID] [Date]

### Summary
One-paragraph description: what failed, duration, customer impact.

### Timeline
| Time | Event |
|------|-------|
| T+0  | Alert fired |
| T+5  | On-call acknowledged |
| ...  | ... |

### Root Cause
Technical description of what actually went wrong.

### Contributing Factors
- Factor 1 (e.g., no feature drift monitoring in place)
- Factor 2 (e.g., new market launched without model retraining)

### Impact
- X customers affected, Y transactions incorrectly blocked
- Revenue impact: $Z

### Detection Gap
Why did it take N minutes to detect?

### Action Items
| Action | Owner | Due Date | Priority |
|--------|-------|----------|----------|
| Add PSI monitoring for all categorical features | ML Eng | +1 week | P1 |
| Automated retraining trigger on PSI > 0.2 | MLOps | +2 weeks | P2 |
| Pre-launch checklist for new market expansions | PM | +1 week | P1 |

### What Went Well
- Alert fired within 5 minutes
- Threshold adjustment as mitigation was effective

### Lessons Learned
Short-form insights for org-wide sharing.
```

---

## A4: Security Architecture for Multi-Tenant AI

### Project Isolation: Project-per-Client Model

```
Organization: huge-marketing-ai.com
├── Folder: production/
│   ├── Project: client-nike-prod         (Nike data, models, endpoints)
│   ├── Project: client-mcdonalds-prod    (McDonald's data, models, endpoints)
│   ├── Project: client-verizon-prod      (Verizon data, models, endpoints)
│   └── Project: shared-infra-prod        (shared Cloud Armor, logging sink, monitoring)
├── Folder: staging/
│   └── ... (mirrors of prod per client)
└── Folder: dev/
    └── ... (shared sandbox)
```

**Why project-per-client?**
- IAM policies are project-scoped; no accidental cross-client data access
- Billing is isolated: each client's cloud costs are attributable
- Quota is isolated: Nike's traffic spike can't starve McDonald's endpoints
- VPC Service Controls perimeter is cleanest at project boundary

---

### VPC Service Controls: Preventing Cross-Client Data Exfiltration

```python
# VPC SC perimeter for Nike project
resource "google_access_context_manager_service_perimeter" "nike_perimeter" {
  name   = "accessPolicies/POLICY_ID/servicePerimeters/nike_prod"
  title  = "Nike Production Perimeter"

  status {
    restricted_services = [
      "aiplatform.googleapis.com",
      "bigquery.googleapis.com",
      "storage.googleapis.com",
      "secretmanager.googleapis.com",
    ]
    resources = ["projects/nike-prod-project-number"]

    access_levels = [
      "accessPolicies/POLICY_ID/accessLevels/corp_network",
      "accessPolicies/POLICY_ID/accessLevels/service_accounts_nike",
    ]
  }
}
```

This means: even if a bug in the shared inference layer tried to read Nike's BigQuery tables from McDonald's service account, VPC SC would deny it at the API level — before the data is ever returned.

---

### "How do you prove to Nike's CISO that their data is isolated from McDonald's?"

**Model answer (five layers):**

"I'd walk the CISO through five concrete isolation layers:

**1. Network isolation:** Nike's data lives in a dedicated GCP project with a VPC Service Controls perimeter. McDonald's project is in a completely separate perimeter. Cross-perimeter API calls are denied by default at the GCP control plane — not enforced by application logic, but by the infrastructure itself.

**2. Identity isolation:** Nike's Vertex AI endpoints use a dedicated service account `vertex-sa@client-nike-prod.iam.gserviceaccount.com`. This SA has no permissions on any McDonald's project resources. Workload Identity Federation means this SA never has a long-lived key — only short-lived OIDC tokens scoped to Nike's resources.

**3. Encryption isolation:** Nike's BigQuery datasets and GCS buckets use CMEK with a Cloud KMS key that only Nike's service accounts can use (`roles/cloudkms.cryptoKeyEncrypterDecrypter` scoped to Nike's KMS keyring). McDonald's cannot decrypt Nike's data even if they could read the raw bytes.

**4. Audit isolation:** Cloud Audit Logs record every data access with the principal identity, resource, and timestamp. Nike's logs flow to a dedicated Log Sink in their project. We can provide Nike with real-time audit log export to their own SIEM on demand.

**5. Verified by penetration test:** Before any client goes live, we run a cross-tenant penetration test: can a McDonald's service account read Nike data? We document the result and share the report with Nike's security team."

---

### Workload Identity Federation

```bash
# No service account keys needed — bind GKE workload to SA
gcloud iam service-accounts add-iam-policy-binding \
  vertex-sa@client-nike-prod.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="serviceAccount:nike-prod.svc.id.goog[nike-namespace/nike-sa]"
```

**Why this matters in a SA interview:** Traditional approach (download JSON key → store in K8s secret) has key rotation risk, key leakage risk, and audit gaps. WIF eliminates all three — no key material ever leaves GCP.

---

## A5: Deployment Patterns for LLM Applications

### Blue-Green Deployment for LLM Apps

**Traditional blue-green:** swap DNS, both environments are stateless. LLM apps have state: prompt templates, few-shot examples, system instructions, tool definitions. The "state" is the *prompt*, not just code.

```
Blue environment  (current production):
  - Model: gemini-1.5-flash-001
  - Prompt template version: v12
  - Retrieval config: top_k=5, score_threshold=0.7
  - Traffic: 100%

Green environment (new version under test):
  - Model: gemini-1.5-flash-002
  - Prompt template version: v13 (new persona, structured output)
  - Retrieval config: top_k=8, score_threshold=0.65
  - Traffic: 0% → (after shadow validation) → 5% → 25% → 100%
```

**State management difference:** Roll back for an LLM app means reverting prompt templates AND model version AND retrieval config atomically. Store all three in a versioned config bundle (Git tag + GCS object), not separately.

---

### Shadow Mode Testing

```python
import asyncio
from google.cloud import aiplatform

async def shadow_query(query: str, user_context: dict) -> dict:
    """Run production and shadow model in parallel; log both; return production result."""

    prod_task = asyncio.create_task(
        production_endpoint.predict(instances=[{"query": query, "context": user_context}])
    )
    shadow_task = asyncio.create_task(
        shadow_endpoint.predict(instances=[{"query": query, "context": user_context}])
    )

    prod_result, shadow_result = await asyncio.gather(prod_task, shadow_task)

    # Log shadow result for offline comparison — never return to user
    log_shadow_comparison({
        "query_id": query_id,
        "prod_response": prod_result.predictions[0],
        "shadow_response": shadow_result.predictions[0],
        "prod_latency_ms": prod_result.metadata.get("latency_ms"),
        "shadow_latency_ms": shadow_result.metadata.get("latency_ms"),
    })

    return prod_result  # Always return prod to the user
```

Run shadow mode for 48–72 hours. Evaluate: win rate (shadow better than prod on human evals), latency delta, hallucination rate delta. Only promote shadow to production if win rate > 55% and latency regression < 20%.

---

### Rollback Strategy for Prompt Degradation

```
Problem: New prompt v13 deployed. CSAT drops from 4.2 → 3.6 in first hour.
         Guardrail model detects 8% hallucination rate (SLO: 2%).

Immediate (< 5 min):
  Feature Flag: toggle PROMPT_VERSION from "v13" → "v12"
  (No redeployment needed — prompt loaded from GCS at request time)

Validation:
  Wait 10 minutes, observe guardrail hallucination rate return to 1.8%
  Confirm CSAT trend recovering

Root cause:
  v13 had "be concise" instruction that caused model to omit citations
  → hallucinations uncaught because model gave confident but unsourced answers

Fix:
  v14 prompt: restore citation requirement + brevity instruction balanced
  Shadow test v14 for 48 hours before re-promoting
```

---

### Multi-Model Routing in Production

```python
def route_to_model(query: str, context: dict) -> str:
    """Route query to the cheapest model that can handle it."""

    complexity_score = classify_query_complexity(query)
    # 0.0 = simple FAQ, 1.0 = multi-step reasoning required

    if complexity_score < 0.3:
        return "gemini-1.5-flash"        # 10× cheaper than Pro
    elif complexity_score < 0.7:
        return "gemini-1.5-pro"          # balanced
    elif context.get("requires_vision"):
        return "gemini-1.5-pro-vision"   # multimodal
    elif context.get("requires_coding"):
        return "claude-3-5-sonnet"       # superior coding
    else:
        return "gemini-1.5-pro"          # default complex path
```

**Cost impact of routing:** If 60% of queries are simple (Flash), 30% medium (Pro), 10% complex (Pro): blended cost is $0.000052/query vs $0.000075/query all-Pro. At 1M queries/day: saves $690/month, or $8,280/year. Meaningful at scale.

---

---

# PART B: MARKETING AI — DEEP GRILLING Q&A

---

## B1: Marketing Mix Modeling — Deep Grilling

### Primary Question: Mathematical Formulation of MMM

**Interviewer:** *"Walk me through the mathematical formulation of your MMM. What's the likelihood function?"*

**Model answer:**

"The core MMM formulation I used at Axtria is a Bayesian hierarchical model. The sales response function is:

**y_t = α + Σ_i β_i · adstock_i(x_{i,t}) + Σ_j γ_j · z_{j,t} + ε_t**

Where:
- **y_t**: KPI (sales, revenue) at time t
- **α**: baseline (organic demand without any marketing)
- **β_i**: coefficient for channel i (the 'ROI weight')
- **adstock_i(x_{i,t})**: transformed spend with memory decay: `adstock_t = x_t + λ · adstock_{t-1}` where λ ∈ [0,1] is the retention rate
- **γ_j · z_{j,t}**: control variables (price, seasonality, promotions, economic indicators)
- **ε_t ~ N(0, σ²)**: observation noise

For the Bayesian version, the likelihood function is:

**L(β, γ, σ | y) = Π_t N(y_t | α + Σ β_i · adstock_i(x_t) + Σ γ_j · z_t, σ²)**

With priors:
- **β_i ~ HalfNormal(σ_β)** — enforcing non-negative ROI (spending can't reduce sales)
- **λ_i ~ Beta(2, 2)** — decay rates between 0 and 1, centered at 0.5
- **σ ~ HalfNormal(1)** — weakly informative noise prior

I fit this using NUTS (No-U-Turn Sampler) via PyMC. The posterior gives us full uncertainty distributions over ROI — critical for budget recommendations where we want to say 'TV ROI is 2.1 with 95% CI [1.7, 2.5]' not just a point estimate."

---

### Cross-Question 1: Validating Adstock Decay Rate

**Q:** *"How did you validate that your adstock decay rate was correct and not just overfitting?"*

"Three validation approaches:

**1. Prior elicitation with media experts:** For TV, industry research suggests 2–8 week carryover. I set my Beta prior to concentrate mass in that range, preventing the model from learning λ=0.99 (permanent memory) from noise.

**2. Hold-out validation:** Withheld the last 3 months of data. Compared predicted vs actual sales during the hold-out period. If adstock was overfitting, predictions would degrade faster during hold-out than training.

**3. Sensitivity analysis:** Plotted sales response curves at λ = [0.3, 0.5, 0.7, 0.9]. Asked the client's marketing team: 'Does this carryover curve match your intuition about how TV works for your brand?' Subject-matter validation catches overfitting that statistical metrics miss.

The key trap is using RMSPE in-sample only — you'll find the decay that fits historical data perfectly and then give nonsense budget recommendations."

---

### Cross-Question 2: New Channel Mid-Year

**Q:** *"What happens to your MMM when a client launches a new channel mid-year?"*

"This is a real problem I faced at Axtria when a client launched Connected TV mid-Q3. Three options:

**Option A (simplest):** Include CTV from launch date, treat pre-launch as zero spend. The model will learn CTV's ROI from only 2–3 months of data — low confidence, wide credible interval. Flag this to the client: 'CTV estimate is preliminary, revisit after 6 months of data.'

**Option B (transfer learning):** Use linear TV's adstock and saturation parameters as an informative prior for CTV. CTV is similar enough to TV that this accelerates learning without requiring a full year of CTV data.

**Option C (hierarchical partial pooling):** If the client has multiple brands, pool CTV parameters across brands where CTV was already running. A brand launching CTV in Q3 borrows statistical strength from sibling brands.

I'd go with B or C depending on data availability. The honest answer to a client is always: 'New channels need 3–4 months of varied spend to estimate ROI reliably. Run a spend variation experiment (hold-out test in some markets) to accelerate this.'"

---

### Cross-Question 3: Endogeneity Problem in MMM

**Q:** *"How do you handle the endogeneity problem in MMM?"*

"This is the hardest econometric challenge in MMM. Endogeneity occurs when media spend is correlated with the error term — which happens because brands increase spend when they *expect* sales to be high (e.g., seasonal ramp-up), and those expectations are already partially captured in the baseline.

**Detection:** Hausman test — regress media spend on instruments, check if residuals predict the outcome. If yes, you have endogeneity.

**Solutions I've used:**

1. **Control for confounders explicitly:** Add seasonality indices, promotional calendars, search trend data (Google Trends) as exogenous controls. This absorbs the correlated variation that was causing endogeneity.

2. **Instrumental variables (IV):** Use media *price* (CPM/CPC rates) as an instrument for spend. CPM rates are correlated with spend decisions but uncausally related to sales outcomes. Two-stage least squares with CPM as instrument.

3. **Natural experiment approach:** Geo-holdout tests provide exogenous variation in spend (some geos get media, others don't by design), breaking the endogeneity. Results from geo tests can be used to calibrate and validate the observational MMM.

In practice at Axtria, I used approach 1 (richest confounder set) as the primary defense and approach 3 (geo experiments) as external validation."

---

### Trap Question: TV ROI vs Agency Disagreement

**Q:** *"Your MMM says TV has the highest ROI. The client's agency says paid social is best. What do you do?"*

**Trap:** Don't cave to the client/agency, but don't be arrogant either. The right answer is intellectual humility + methodological rigor.

"I'd treat this as a hypothesis worth investigating, not a reason to distrust either source. I'd start by asking: what methodology did the agency use? Typically agencies use last-click or platform-reported attribution — which famously over-credits digital channels and under-credits TV (because TV's impact is indirect via search lift and brand recall).

I'd then look at whether our MMM has sufficient TV variation to estimate ROI reliably — if the client ran TV at constant spend, there's no variation to regress on, and the TV coefficient will be noisy.

The resolution: propose a **geo holdout test** — run TV in some DMAs, hold it dark in others, measure sales difference after 4 weeks. This gives causal evidence that neither platform attribution nor MMM can dispute. I'd present this as 'here's how we settle the debate with data' — not as 'you're wrong.'"

---

### Meridian vs Robyn vs Custom Bayesian MMM

| Framework | Best For | Weaknesses | Huge Context |
|---|---|---|---|
| **Google Meridian** | Clients with heavy Search + YouTube spend; Google data integration | Google-ecosystem bias; less flexible priors | Strong choice if client is Google-heavy (most retail) |
| **Meta Robyn** | Clients heavy on Meta/Facebook/Instagram; open-source, R-based | Meta-centric design; Python support still maturing | Good for DTC brands with heavy social |
| **Custom Bayesian (PyMC/Stan)** | Complex hierarchical needs; multi-brand, multi-country; full prior control | Requires deep ML/stats expertise; longer build time | Axtria's approach — best for pharma, complex B2B |
| **Lightweight frequentist** | Small clients, fast turnaround, limited data | No uncertainty quantification; overconfident ROI | Quick-start pilot for skeptical CMOs |

---

## B2: Multi-Touch Attribution — Deep Grilling

### Primary Question: Markov Chain Defense

**Q:** *"Markov chains assume memory-less transitions. That's not realistic. Defend your choice or propose an alternative."*

"You're right — the memoryless (Markov) assumption means the next touchpoint depends only on the current state, ignoring sequence history. In reality, a user who saw TV → Search is probably more purchase-ready than a user who saw Display → Search, even though both are in the 'Search' state.

**Defense of Markov chains in practice:** Despite the theoretical limitation, Markov chains outperform last-click and linear in empirical lift tests because they correctly distribute credit based on path structure — they capture first-order sequential effects even without memory. For most marketing channels where the customer journey is 3–5 touchpoints, first-order transitions explain ~80% of variance.

**When to propose an alternative:**
- If journeys are 8+ touchpoints: use higher-order Markov chains (state = last 2 touchpoints)
- If you have enough data (1M+ conversions): use **LSTM-based sequential attribution** — this learns arbitrary-length path dependencies
- For causal attribution: **Shapley values** (cooperative game theory) — don't assume any path structure at all

**What I added at Axtria:** An attention mechanism on the path sequence before Markov transition estimation — this gave the model a way to weight earlier touchpoints differently based on their predicted purchase intent signal. Empirically improved attribution accuracy by 12% on held-out conversion paths."

---

### iOS ATT Problem

**Q:** *"How did you handle the iOS ATT problem in your Axtria attribution model?"*

"iOS 14.5+ ATT killed user-level tracking for ~40% of iOS users who declined tracking. This breaks the conversion path because you can't stitch the ad impression to the downstream conversion.

**Three-layer response:**

**1. Aggregated measurement tier:** Use Meta's Conversions API (server-side events) + Google Enhanced Conversions to recover hashed-identity signals. This doesn't require consent for aggregated reporting.

**2. Statistical imputation:** For unattributed conversions, use a Bayesian propensity model to probabilistically assign credit. Inputs: device type, geo, time of day, ad campaign targeting. Output: probability distribution over which campaigns likely drove this conversion.

**3. MMM as the ground truth layer:** Post-ATT, MMM becomes more important — not less — because it works on aggregate data and doesn't depend on user-level tracking at all. I explicitly repositioned MMM to clients as 'ATT-proof measurement.'

The honest message to clients: 'User-level attribution is permanently degraded for iOS. Build a two-layer measurement stack: MMM for strategic budget allocation, Conversions API for tactical campaign optimization.'"

---

### Shapley Values vs Markov Chains

| Dimension | Shapley Values | Markov Chain |
|---|---|---|
| Mathematical basis | Cooperative game theory; each channel = player; Shapley = average marginal contribution across all coalition permutations | Stochastic process; transition matrix P[i→j] estimated from observed paths |
| Computational complexity | O(2^n) channels — exponential; approximate with Monte Carlo sampling | O(n²) — polynomial; tractable for 20+ channels |
| Path order sensitivity | No — Shapley values are symmetric; order doesn't matter | Yes — first-order Markov captures immediate sequence |
| Data requirement | Needs conversion-level path data | Needs conversion-level path data |
| Interpretability | High — "each channel's fair contribution in isolation" | Medium — "removal effect" intuition |
| Best for | 5–10 channel portfolios; fairness emphasis | 10+ channels; path-sequence importance |

---

### Pharma Attribution Design (Connecting to Axtria)

**Q:** *"Design an attribution system for a pharma company where the 'conversion' is a doctor writing a prescription months after seeing a detail rep."*

"This is exactly the HCP (Healthcare Professional) attribution problem I worked on at Axtria. It differs from consumer attribution in three critical ways:

**1. Conversion lag:** The 'journey' from detail rep visit to Rx can be 4–6 months. You can't use session-based attribution windows. You need survival analysis — specifically a Cox proportional hazards model — to model time-to-first-prescription as a function of touchpoint history.

**2. Multi-channel HCP touchpoints:** Rep visits, speaker programs, medical conference interactions, digital banner ads (HCP-targeted programmatic), peer-to-peer recommendations. These are fundamentally different signal types. Rep visits are scheduled and logged in CRM (Veeva). Digital is tracked via cookie. Peer influence is latent (inferred via physician network analysis).

**3. Prescription as proxy for patient demand:** Rx is the outcome, but the physician is the decision-maker; the patient is the consumer. You're modeling HCP behavior, not patient behavior. This matters for feature engineering — HCP-level features (specialty, prescribing history, patient panel) dominate over demographic features.

**My solution architecture at Axtria:**
- Matched control design: for each HCP who received a detail visit, find a demographically and geographically matched HCP who didn't, compare 6-month Rx rates
- Incremental attribution: treatment effect = (Rx rate in treated group) - (Rx rate in control group)
- Multi-touch extension: weighted by touchpoint recency and type (rep visit > digital ad > conference attendance in empirically validated hierarchy)

This is causal inference, not just correlational attribution — the gold standard."

---

## B3: Customer Lifetime Value (CLV) Architecture

### Primary Question: BG/NBD vs ML-based CLV

**Q:** *"BG/NBD model vs ML-based CLV — when do you use each?"*

"The **BG/NBD model** (Beta-Geometric / Negative-Binomial Distribution) models two latent processes simultaneously: purchase frequency (while alive) and customer death (churn). It's mathematically elegant and interpretable — you get P(alive) for each customer.

**Use BG/NBD when:**
- Non-contractual, repeat purchase business (e-commerce, CPG retail)
- Limited feature data — BG/NBD only needs recency, frequency, monetary value (RFM)
- Interpretability is paramount (regulators, CFO wants P(alive) distribution)
- < 500K customers — tractable MCMC sampling

**Use ML-based CLV (gradient boosted, LSTM, or survival models) when:**
- Rich feature set available: browsing behavior, app engagement, customer service history, demographics
- Non-stationary purchase patterns (promotions cause bursty behavior that violates BG/NBD stationarity)
- > 5M customers — ML scales better
- You want CLV as an input feature (e.g., bid multiplier in Google Ads), not just a management tool

**At CVS Health context via EXL:** I'd use a **two-stage approach**: BG/NBD for the 'how many purchases' prediction, combined with an XGBoost model for monetary value per purchase, then multiply. This captures BG/NBD's theoretical strength on purchase timing while using ML's strength on basket size prediction where features matter."

---

### Real-Time CLV Scoring System

**Q:** *"Design a real-time CLV scoring system that updates every 24 hours for 10M customers."*

```
Architecture: Daily Batch + Low-Latency Serving

DAILY PIPELINE (11 PM trigger):
  BigQuery: Feature engineering query
  → Extract: 10M customers × 150 features (~15GB)
  → Vertex AI Batch Prediction: CLV model scoring
  → Output: CLV scores to BigQuery table
  → Cloud Bigtable: Write CLV score + decile + segment per customer_id
  → Total pipeline time: ~45 minutes

SERVING LAYER (real-time reads):
  Cloud Bigtable: key = customer_id, value = {clv_score, updated_at, segment}
  → p99 read latency: 5ms
  → Throughput: 100K reads/second
  → Used by: ad platforms (DV360 bid multiplier), CRM (priority routing), website (offer personalization)

COLD-START (new customers):
  - Default to cohort median CLV based on acquisition channel + first purchase category
  - After 3 purchases: switch to individual CLV model
  - Use survival analysis hazard rate for first 90 days

MONITORING:
  - Daily: PSI of CLV score distribution vs last 30-day rolling average
  - Alert if Gini coefficient of CLV distribution shifts > 0.05 (indicates model drift)
```

---

### How CLV Informs Marketing Decisions

**Q:** *"How do you use CLV in a marketing context? What decisions does it actually inform?"*

| Decision | Without CLV | With CLV |
|---|---|---|
| Google Ads bidding | Bid same for all users | Bid multiplier proportional to predicted CLV: high-CLV user = bid 3× |
| Retention campaign targeting | Target all customers with discount | Target only customers where CLV > acquisition cost threshold — don't discount high-CLV customers who wouldn't churn anyway |
| Customer service routing | FIFO queue | Route high-CLV customers to senior agents; reduce wait time proportionally |
| Loyalty tier design | Fixed spend thresholds | Set tier thresholds based on CLV quantiles — top 20% CLV customers get elite status |
| Channel mix for new customer acquisition | Optimize for CAC (cost per acquisition) | Optimize for CAC/LTV ratio — willing to pay higher CAC for channels that attract high-CLV segments |

---

## B4: Huge-Specific Marketing AI Scenarios

### Scenario 1: Verizon Churn Reduction System

**Q:** *"Verizon wants to reduce churn. Design the end-to-end AI system."*

"I'd structure this in three phases:

**Phase 1 — Churn Propensity Model:**
- Features: billing patterns (missed payments, plan changes), usage trends (data consumption decline), service calls (complaint frequency, NPS scores), competitor pricing signals (from public data), device age, contract tenure
- Target: 90-day voluntary churn (not involuntary/payment churn — separate model needed)
- Model: XGBoost with SHAP explanations — carrier executives need to understand *why* a customer is at risk, not just that they are
- Output: P(churn) per customer + top 3 risk factors per customer (e.g., 'Contract expires in 30 days + data usage declining + 2 service complaints this month')

**Phase 2 — Intervention Optimization (the part most teams miss):**
- Don't just identify at-risk customers — optimize the *intervention*
- Causal ML: use uplift modeling (Two-Model approach or Causal Forest) to identify customers where intervention *works* vs those who'd churn anyway or stay anyway
- Treatment options: proactive discount offer, free device upgrade, priority customer service routing, early contract renewal incentive
- Cost-constrained optimization: maximize churn reduction given budget B using linear programming on the intervention × uplift matrix

**Phase 3 — Closed-Loop Measurement:**
- A/B test: hold-out 20% of high-risk customers (no intervention), treat 80%, measure churn difference at 90 days
- Incremental churn reduction = churn rate in control - churn rate in treated
- Feed results back into uplift model quarterly

**At Huge:** 60-day pilot (model build + validation), 90-day A/B test, then productionize on GCP. Realistic timeline: 6 months to production. ROI frame: if Verizon has 10M subscribers, 2% monthly churn = 200K churners/month at $50 ARPU = $120M annual revenue at risk. Even a 5% churn reduction = $6M/year saved."

---

### Scenario 2: Nike Air Jordan Propensity Model

**Q:** *"Nike wants to predict which customers will buy their next Air Jordan drop. Design the propensity model."*

"**Data signals I'd prioritize:**
- Nike app engagement: wishlist behavior, 'notify me' events, SNKRS app opens in last 30 days
- Past Jordan purchase history: recency + frequency + style affinity (high-cut vs low-cut, colorway preferences)
- Social signals: Jordan-tagged post engagement, influencer content interaction
- Geolocation: proximity to Nike flagship stores (in-store pickup intent proxy)
- Demographic proxies: age segment, household income band

**Model: Gradient Boosted Propensity + Collaborative Filtering Hybrid**
- Propensity score: P(purchases Air Jordan | features) — XGBoost as base
- Complementary layer: collaborative filtering from past Jordan buyers to surface lookalikes who haven't bought yet
- Output: propensity score (0–1) + recommended outreach channel

**Actionability by score tier:**
- P > 0.7: SNKRS app push notification + priority queue access for launch day
- P 0.4–0.7: Email campaign with early access offer + retargeting
- P < 0.4: Social retargeting with aspirational content (don't burn goodwill with a hard sell)

**Drop-specific calibration:** The model recalibrates with each drop — Jordan 5 buyers ≠ Jordan 1 buyers necessarily. Key insight: run an online feature importance analysis after each drop to see which signals shifted. The feature set for a retro colorway release will differ from a new silhouette launch."

---

### Scenario 3: IKEA In-Store Attribution

**Q:** *"IKEA wants to know which marketing channels are actually driving in-store visits, not just online. Design the measurement framework."*

"This is an omnichannel attribution problem — the conversion (in-store visit) happens offline but is driven by online channels. Three measurement layers:

**Layer 1 — Mobile Location Data (probabilistic):**
- Partner with a location intelligence provider (Foursquare, Placer.ai)
- Match IKEA's ad impression IDs to device location data
- Measure: 'Did users exposed to Channel X visit an IKEA store within 14 days at a higher rate than unexposed users?'
- Limitation: ~60–70% match rate; opt-in dependent

**Layer 2 — IKEA Family Card (deterministic):**
- IKEA's loyalty program links online behavior (browsing, wishlist) to in-store purchases via card scan
- This is gold — deterministic identity resolution at purchase
- Build: digital-to-store attribution model using IKEA Family Card as ground truth identity spine

**Layer 3 — Geo-Lift Experiments (causal):**
- Run paid social and display campaigns in selected DMAs; hold dark in matched control DMAs
- Measure in-store visit rate difference using location data + IKEA Family Card
- This gives causal lift, not just correlation

**Synthesis:** Layer 3 calibrates Layers 1 and 2. Once you know the true lift from a geo experiment, you can use that as a multiplier to scale up the probabilistic Layer 1 estimates. Present this as a hybrid model: 'Causal lift from geo experiments × probabilistic reach from impression data = estimated in-store visits driven per channel.'"

---

### Scenario 4: McDonald's Promotional Calendar Optimization

**Q:** *"McDonald's wants to optimize their promotional calendar using AI. Walk me through your approach."*

"I'd structure this as a hierarchical Bayesian optimization problem with three components:

**Component 1 — Response Model:** Build an MMM-style response model that maps promotion type × timing × channel to incremental transactions. Features: promotion type (BOGO, discount, new item launch), promotional channel (TV, paid social, in-app offer, Drive-Thru signage), day-of-week, time-of-year, competitive activity, weather (rain increases McDelivery demand significantly).

**Component 2 — Saturation + Interaction Effects:** Model promotion fatigue — running BOGO every week trains customers to wait for the deal. Model channel interaction effects — TV announcement + in-app offer has multiplicative effect vs either alone.

**Component 3 — Genetic Algorithm Optimization (connecting to Abhishek's Axtria work):**
```python
# Simplified GA for promotional calendar optimization
population = generate_initial_calendars(n=500)  # 500 random 52-week promotional plans

for generation in range(100):
    # Score each calendar against the response model
    fitness = [response_model.predict(calendar) for calendar in population]

    # Selection: keep top 20%
    survivors = population[argsort(fitness)[-100:]]

    # Crossover: combine weeks from two parent calendars
    offspring = [crossover(survivors[i], survivors[j])
                 for i, j in random_pairs(survivors)]

    # Mutation: randomly swap promotion types or weeks
    population = [mutate(individual, rate=0.05)
                  for individual in survivors + offspring]

optimal_calendar = population[argmax(fitness)]
```

**Output:** A 52-week promotional calendar that maximizes incremental transaction volume subject to constraints: promotion cost budget, operational complexity (can't run >2 simultaneous promotions), brand guidelines (no discounting flagship products in certain windows)."

---

### Scenario 5: The Skeptical CMO

**Q:** *"A client's CMO says AI is just hype and wants ROI proof before investing. What do you say and what do you build first?"*

**Trap:** Don't oversell. Don't get defensive. Show you can speak business language, not AI language.

"I'd say: 'You're right to be skeptical — there's a lot of noise in this space and most AI projects fail to show business value because they're built as technology projects instead of business outcome projects. Let me propose something different: a 90-day value pilot with a specific, measurable business question.'

**What I'd build first: Marketing Mix Model on existing data**

Rationale: MMM requires only data the client already has (sales, media spend, promotional calendar), can be completed in 8–10 weeks, and produces immediately actionable budget reallocations. Typical result: 5–15% budget efficiency improvement from reallocation alone — without spending a dollar more on media.

**The pitch:** 'We're not going to ask you to invest in AI infrastructure or change your technology stack. We're going to answer one question: Are you spending your marketing budget in the optimal way across channels? We'll answer that in 8 weeks using your existing data.'

**ROI frame for the CMO:** If the client spends $50M/year on media and we identify a 10% efficiency gain via reallocation: that's $5M in recovered media value. The 8-week pilot costs $150K. That's a 33× ROI on the pilot cost alone.

**Why this works:** It builds trust with a quick win. It creates institutional familiarity with data-driven marketing. It surfaces data quality issues in a low-stakes setting. And it creates the baseline for the next, more ambitious AI project."

---

## B5: Solutions Architect Client-Facing Skills

### Scoping an AI Project: Discovery Process

**Q:** *"How do you scope an AI project? Walk me through your discovery process."*

"My discovery process has four layers across three sessions:

**Session 1 — Business Problem Definition (with business stakeholders):**
- What decision will this AI system inform? (Not 'build a recommendation engine' — but 'which products should we recommend to which customers in the email cadence?')
- What is the cost of a wrong decision? (Asymmetric error costs → different model design)
- What does success look like in 6 months? (Define KPI: conversion rate lift, cost reduction, churn reduction rate)
- What decisions are you making today without AI that this would replace?

**Session 2 — Data Audit (with data/analytics team):**
- What data exists? Where does it live? (BQ, Salesforce, S3, on-prem?) → data map
- What is the quality of each data source? (completeness, consistency, recency)
- What is the historical depth? (need 24+ months for seasonality-aware models)
- Are there regulatory constraints on data use? (HIPAA, GDPR, CCPA)

**Session 3 — Technical Fit Assessment (with IT/engineering):**
- What is the deployment target? (API, batch, embedded in existing platform?)
- What is the serving latency requirement? (real-time < 100ms vs batch overnight)
- Who maintains the system after build? (client ML team? Or Huge-managed service?)
- What is the integration surface? (CRM, CDP, ad platforms, website CMS)

**Output from discovery:**
- Problem statement (2 sentences, business language)
- Data readiness score (1–5) with gaps identified
- Recommended approach (ML model type, architecture)
- Rough effort estimate: S/M/L (4 weeks / 12 weeks / 24 weeks)
- Risk register: top 5 risks with mitigation"

---

### Writing a Statement of Work for an AI Project

**Q:** *"How do you write a Statement of Work for an AI project? What are the key clauses?"*

"Standard SOW clauses for software projects are insufficient for AI. Key AI-specific additions:

**1. Data Responsibility Clause:** Who provides data, in what format, by what date? What happens if data quality is insufficient — is Huge responsible for data cleaning, or does the timeline extend?

**2. Model Performance Definition:** Specify the success metric and minimum threshold: 'The propensity model will achieve AUC-ROC ≥ 0.75 on held-out test set.' Without this, 'the model doesn't work' is not actionable.

**3. Baseline Comparison:** Specify what the AI is being compared against. 'Model outperforms current rule-based system on precision@K' is measurable. 'Model is better' is not.

**4. Model Drift and Monitoring SLA:** Who monitors the model post-deployment? What triggers retraining? What is the SLA for a drift alert to retraining completion?

**5. IP Ownership of Model Artifacts:** Client owns the trained model weights and training data. Huge retains rights to the architecture, code framework, and methodology. This is critical — never give away the methodology.

**6. Hallucination / Output Liability (for LLM projects):** Define responsibility boundary: 'Huge is responsible for the system producing outputs within the configured guardrails. Client is responsible for human review of high-stakes outputs before acting on them.'

**7. Change Order Threshold:** AI projects always have scope changes when data reality doesn't match discovery assumptions. Define: 'Any change requiring > 40 hours of additional effort requires a signed change order.'"

---

### Estimating Effort for a RAG System

**Q:** *"How do you estimate effort for a RAG system? Give me a rough estimate for a mid-size enterprise knowledge base."*

**Assumptions:** 50,000 documents, enterprise internal use (HR policies, product manuals, SOPs), 200 concurrent users, integration with Slack and web portal.

| Phase | Activities | Weeks | Team |
|---|---|---|---|
| Data ingestion pipeline | Crawl, parse (PDF/Word/HTML), chunk, embed, index | 2 | 1 ML Eng |
| Retrieval layer | Vertex AI Vector Search deployment, hybrid search (dense + BM25) | 1 | 1 ML Eng |
| Generation layer | Prompt engineering, grounding, citation formatting, hallucination guardrails | 2 | 1 ML Eng + 1 SA |
| API + auth layer | REST API, OAuth2 integration, rate limiting | 1.5 | 1 Backend Eng |
| Frontend (web portal) | Chat UI, source citation display, feedback thumbs | 2 | 1 Frontend Eng |
| Slack integration | Slack bot, context threading | 1 | 1 Backend Eng |
| Evaluation framework | Golden Q&A set (500 pairs), RAGAS eval harness | 1.5 | 1 ML Eng |
| QA + UAT | Business user testing, edge case handling | 2 | Full team |
| Deployment + MLOps | Cloud Run, monitoring, alerting, CI/CD | 1.5 | 1 DevOps/SA |
| **Total** | | **~14 weeks** | **3–4 people** |

**Cost estimate:** At a blended rate of $250/hr × 3.5 FTE × 14 weeks × 40 hrs/week = **~$490,000**. If Huge uses offshore delivery for engineering tasks, compress to $350–400K. GCP infrastructure for this scale: ~$3,000–5,000/month ongoing.

---

### The "Client Wants GPT-4 for Everything" Scenario

**Q:** *"The client wants to use GPT-4 for everything. How do you handle vendor lock-in and cost?"*

"I validate the instinct — GPT-4 is genuinely excellent — then redirect the conversation to risk and economics.

**The conversation I'd have:**

'GPT-4 is a strong choice, and I understand the preference — it's what most people have experienced and trust. Let me share a few considerations so we can make the right architectural decision:

**Cost at scale:** GPT-4o costs $5/M input tokens and $15/M output tokens. At 1M queries/day with 3K input + 500 output tokens, that's $22,500/day or $675,000/month. Gemini 1.5 Pro at comparable quality costs ~$7/M input and $21/M output for long context — and we can serve many use cases with Flash at $0.075/M input, reducing to $2,250/day.

**Vendor risk:** OpenAI has had outages (November 2023 DDoS, multiple service degradations). If your entire AI system is GPT-4-only, any OpenAI outage = your AI system is down. I'd recommend an abstraction layer — LangChain or a custom model router — that allows model swapping without application code changes.

**What I'd recommend:** Use GPT-4o for the use cases where it's genuinely best (complex reasoning, specific domains where benchmarks favor it). Use Gemini for the majority of queries — the cost savings fund 2× the scale at the same budget. Build with a model-agnostic abstraction layer so the client can switch models as the landscape evolves.

The goal is not to fight the client's preference — it's to show we're thinking about their long-term costs and resilience, not just the cool technology.'"

---

### Presenting Technical Architecture to C-Suite

**Q:** *"How do you present technical architecture to a C-suite audience?"*

"The C-suite presentation follows a strict 'pyramid principle' — business outcome first, then implications, then evidence.

**What I never do:**
- Start with 'so we have a RAG system on GCP with Vertex AI Vector Search...'
- Show architecture diagrams in slide 1
- Use acronyms without spelling them out (LLM, RAG, CMEK)

**Structure I use:**

**Slide 1 — The Business Outcome:** 'This system will enable your service team to answer customer questions 40% faster, reducing AHT from 6 minutes to 3.6 minutes. At your current call volume of 50,000 calls/month, that's 1,250 hours saved per month = $62,500/month in labor costs.'

**Slide 2 — How it works (non-technical):** Analogy-driven. 'Imagine Google, but it only searches your internal knowledge base and then summarizes the answer in plain English for your agents — with the source document cited so they can verify it.'

**Slide 3 — Security and governance:** C-suite cares about risk. 'Your data never leaves your GCP environment. Every access event is logged. We can show Nike's CISO a complete audit trail of who accessed what data, when.'

**Slide 4 — Investment and ROI:** Total cost (build + run), timeline, expected ROI, break-even point. Simple 3-row table.

**Slide 5 (optional, for technical C-suite):** Architecture overview — one diagram, maximum 5 boxes, annotated with business labels not engineering labels.

**The rule:** If a CMO with no engineering background can repeat back what you built and why, you've communicated successfully."

---

### "You're 3 Weeks Into a Project and Data Quality Is Terrible"

**Q:** *"You're 3 weeks into a project and realize the client's data quality is terrible. What do you do?"*

**Trap:** Do not just "fix the data" quietly and delay the project silently. Do not panic and escalate without a plan.

"Three things simultaneously:

**1. Quantify before escalating.** Run a data quality scorecard: completeness (% null values per field), consistency (duplicate records, conflicting keys), accuracy (spot-check against source systems), and timeliness (data lag). Translate to business impact: 'Field X is 40% null — this field is a top-5 predictor in the model. This will reduce AUC by an estimated 0.05–0.08.'

**2. Bring solutions, not just problems.** Before the stakeholder call, have three options ready:
- Option A: Data remediation (client fixes data, timeline extends 4–6 weeks)
- Option B: Model adaptation (remove the problematic feature, substitute proxies, reduce model scope)
- Option C: Scope reduction (build the model on the subset of data that is clean; expand later)

**3. Document in writing.** Send a written summary of the data quality findings and the three options to the engagement lead and client sponsor — the same day as the verbal discussion. This creates a paper trail that shows Huge identified the issue proactively, not that it was caught by the client at delivery.

**What I'd say to the client:** 'We've completed our data assessment and we've found some quality issues that will affect the model's performance if we proceed as-is. Here's what we found, here's the business impact, and here are three options for how we proceed. We'd like your guidance on which direction to take.'

This is the professional move: transparent, quantified, options-based. Never bury data quality problems until the model performs poorly at demo time."

---

## Quick Reference: Interview Day Cheatsheet

### Numbers to Know Cold

| Metric | Value |
|---|---|
| 99.9% uptime = downtime/year | 8.76 hours |
| 99.95% uptime = downtime/year | 4.38 hours |
| Gemini 1.5 Flash input price | $0.075/M tokens |
| Gemini 1.5 Pro input price | $7.00/M tokens |
| GPT-4o input price | $5.00/M tokens |
| Cloud Run cold start (no min-instances) | 2–8 seconds |
| Vertex Vector Search p99 latency | ~10ms at 1M vectors |
| Typical MMM training data requirement | 2–3 years, weekly granularity |
| BG/NBD minimum transactions for reliability | 3+ purchases per customer |
| Adstock decay λ range for TV | 0.5–0.9 (2–10 week half-life) |
| Typical RAG chunk size (tokens) | 256–512 tokens |
| RAG top-k retrieval standard | k=5 to k=10 |
| RAGAS faithfulness benchmark target | > 0.85 |
| KL divergence threshold for feature drift alert | > 0.1 |
| PSI threshold for retraining trigger | > 0.2 |
| Geo holdout test minimum duration | 4–6 weeks |
| iOS ATT opt-out rate (approx.) | ~40% of iOS users |

---

### Phrases That Signal Top-1% Thinking

- *"I'd validate that with a geo holdout experiment, not just the model..."*
- *"The cost of being wrong asymmetrically is X, so I'd optimize for precision over recall..."*
- *"Before I answer that, I want to understand what decision this will actually inform..."*
- *"That's the right concern — let me show you the five-layer isolation argument..."*
- *"We should distinguish between correlation in the attribution data and causal lift — they're different questions that require different methods..."*
- *"The model is the easy part. The hard parts are data quality, change management, and ongoing monitoring..."*
- *"I'd frame this to the CMO as: what's the cost of the decision we're currently making without data?"*
- *"We can build this in two phases: a 6-week proof of value on existing data, then a full productionization if the POV shows lift..."*

---

### Axtria → Huge Mapping (Your Story Arc)

| At Axtria | Translates To at Huge |
|---|---|
| MMM for pharma brands | MMM for retail/CPG/QSR clients (Nike, McDonald's, Verizon) |
| HCP attribution (Markov + attention) | Digital MTA for consumer brands |
| Genetic algorithm budget optimization | Promotional calendar optimization, media allocation |
| Bayesian hierarchical models | Scalable multi-brand, multi-market measurement |
| Data quality and governance for HIPAA | Data governance for GDPR/CCPA-regulated client data |
| Client-facing deliverables (pharma executives) | C-suite presentations to CMOs and CISOs |
| GCP + Vertex AI production experience | Huge's GCP-based AI platform architecture |

---

*Document: 08_deployment_marketing_alignment.md | Prepared for: Abhishek Bhardwaj | Role: Solutions Architect ML/AI @ Huge | Date: June 2026*
