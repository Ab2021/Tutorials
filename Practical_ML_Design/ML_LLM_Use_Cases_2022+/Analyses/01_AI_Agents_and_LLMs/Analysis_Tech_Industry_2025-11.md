# Tech Industry Analysis: AI Agents & LLMs (2023-2025)

**Analysis Date**: November 2025  
**Category**: 01_AI Agents and LLMs  
**Industry**: Tech  
**Articles Analyzed**: 8 companies (68 total available, 2023-2025)  
**Period Covered**: 2023-2025

---

## EXECUTIVE SUMMARY

The Tech industry is leading AI Agent and LLM adoption with **five major application areas**: enterprise search & RAG, code generation & review, multi-agent systems, model optimization, and production evaluation. Key findings show **10-90% productivity gains**, **<2s latency targets**, and **90%+ human alignment** in production systems. Critical patterns include hybrid RAG architectures, LLM-as-Judge evaluation, and Controller-Delegate multi-agent patterns.

**Key Metrics Across Industry**:
- **Performance Gains**: 10-20% (Microsoft code review), 50% (DeepL training), 90% (Anthropic multi-agent)
- **Evaluation Alignment**: 90%+ LLM-as-Judge with human evaluation  
- **Latency**: <2s p95 for consumer-facing, 3-5s for complex generation
- **Cost**: 15x token usage for multi-agent vs single chat
- **Reliability**: 99% uptime with 1% model provider failures

---

## PART 1: INDUSTRY OVERVIEW

### 1.1 Companies Analyzed

| Company | Focus Area | Year | Use Case |
|---------|-----------|------|----------|
| **Dropbox** | Enterprise Search | 2025 | Dash: RAG + AI agents for knowledge management |
| **Slack** | Enterprise Search | 2025 | Secure federated search across integrations |
| **Anthropic** | Multi-Agent Systems | 2025 | Research system with orchestrator-worker pattern |
| **Microsoft** | Code Generation | 2025 | AI-powered code review automation |
| **DeepL** | Model Optimization | 2025 | FP8 training for 50% throughput gain |
| **GoDaddy** | Operations | 2024 | 10 lessons operationalizing LLMs at scale |
| **Segment** | Evaluation | 2024 | LLM-as-Judge for AST generation |
| **GitLab** | Testing | 2024 | Model validation and testing at scale |

### 1.2 Common Problems Being Solved

**Enterprise Search & Knowledge Management** (Dropbox, Slack):
- Information fragmented across 100+ SaaS applications
- Manual context switching costs hours daily
- Permission management complexity across systems
- Data freshness vs latency trade-offs

**Code Generation & Quality** (Microsoft, GitLab):
- Manual PR reviews bottleneck velocity
- Junior developers need mentorship at scale
- Code quality consistency across thousands of repositories
- Onboarding acceleration for new hires

**Model Operations** (DeepL, GoDaddy, Segment, GitLab):
- Training throughput limits time-to-market
- LLM outputs are probabilistic and unreliable
- Evaluating generation quality without ground truth
- Prompt portability across models

**Multi-Agent Coordination** (Anthropic, GoDaddy):
- Complex research tasks exceed single context window
- sequential processing creates latency bottlenecks
- Agent errors compound in long-running processes

---

## PART 2: ARCHITECTURAL PATTERNS & SYSTEM DESIGN

### 2.1 RAG Architecture Patterns

**Three Dominant RAG Approaches** emerged:

#### Pattern 1: Hybrid IR + Reranking (Dropbox)
```
Query → [Lexical Search (BM25)] → On-the-fly Chunking → [Embedding Reranking] → LLM
```
- **Strengths**: <2s latency, high quality
- **Trade-off**: Chose speed + quality over pure semantic search
- **Tech Stack**: Traditional IR, large embedding models
- **Use**: When latency budget is tight (<2s SLA)

#### Pattern 2: Federated Real-Time (Slack)
```
Query → [OAuth Check] → Parallel [Internal Search + External APIs] → RAG → Response (discard)
```
- **Strengths**: Always fresh, zero staleness, maximum security
- **Trade-off**: Higher latency but no data storage risk
- **Tech Stack**: OAuth 2.0, partner APIs, escrow VPCs
- **Use**: When security/privacy > latency

#### Pattern 3: Multi-Step Dynamic (Anthropic)
```
Query → Lead Agent → [Spawn Subagents] → Parallel Search → Compress → Synthesize
```
- **Strengths**: 90% better than single-agent, handles complexity
- **Trade-off**: 15x token cost, slower
- **Tech Stack**: Claude Opus 4 (lead) + Sonnet 4 (workers)
- **Use**: High-value research tasks with budget

**Consensus**: No single RAG approach works for all use cases. Decision matrix:

| Need | Latency | Security | Complexity | Budget | → Approach |
|------|---------|----------|------------|--------|------------|
| Consumer search | <2s | Medium | Low | Low | Hybrid IR |
| Enterprise data | 3-5s | Critical | Medium | Medium | Federated |
| Research/analysis | 10s+ | Medium | High | High | Multi-agent |

### 2.2 Multi-Agent Patterns

**Controller-Delegate Pattern** (GoDaddy, inspired by Salesforce BOLAA):
```
Mega-Prompt (Controller)
    ↓
Decides when to delegate
    ↓
Task-Oriented Prompts (Delegates)
    - Extract info
    - Search content  
    - Call APIs
    ↓
Return to Controller
```

**Benefits**:
- Controller handles complex flow, delegates handle specific tasks
- Reduced prompt bloat (was 1500+ tokens, now modular)
- Better accuracy (task-specific prompts are more precise)
- **Lesson**: Avoid mega-prompts past 1000 tokens

**Orchestrator-Worker Pattern** (Anthropic):
```
Lead Agent (Planning)
    ↓
Spawns Parallel Subagents
    ↓
Each has separate context window
    ↓
Compress & filter findings
    ↓
Lead synthesizes final answer
```

**Benefits**:
- Parallelism scales performance
- Token usage (80% of perf variance) maximized
- Compression at each subagent reduces context size

**Challenge**: Current bottleneck is synchronous execution

### 2.3 Code Generation Patterns

**Microsoft AI Code Review** (Integrated PR workflow):
1. **Automated Checks**: Flags issues (style, bugs, security)
2. **Suggested Improvements**: Proposes code snippets
3. **PR Summary Generation**: Auto-describes changes
4. **Interactive Q&A**: Answers reviewer questions
5. **Customizable**: Team-specific prompts and rules

**Architecture**:
```
PR Created → AI Reviewer (as team member) → Comments on diff lines → Author reviews → Apply/reject
```

**Impact**:
- 10-20% median PR completion time reduction (5000 repos)
- Catches bugs humans might miss (null checks, API ordering)
- Onboarding: Acts as mentor for new hires

**Critical Design**: Human-in-the-loop - AI suggests, human approves

### 2.4 Technology Stack Consensus

**Most Common Technologies** (appeared in 5+ companies):

| Layer | Technology | Companies Using | Purpose |
|-------|-----------|-----------------|---------|
| **Foundation LLM** | GPT-4, Claude Opus/Sonnet | All 8 | Generation, planning |
| **Smaller LLM** | GPT-3.5, Claude Sonnet | 6/8 | Cost-effective workers |
| **Auth** | OAuth 2.0 | Slack, Dropbox | Permission management |
| **Deployment** | AWS/Cloud VPCs | Slack, GitLab | LLM hosting |
| **Evaluation** | LLM-as-Judge | Segment, GitLab | Quality assessment |
| **Retrieval** | Hybrid (Lexical + Embedding) | Dropbox, GoDaddy | Search |
| **Observability** | Custom tracing/logging | Anthropic, GoDaddy | Debugging agents |

**Emerging Technologies** (2024-2025):
- **FP8 Precision**: DeepL (50% training speedup)
- **Transformer Engine**: NVIDIA library for FP8 support
- **Rainbow Deploys**: Anthropic (stateful agent rollouts)  
- **Streaming APIs**: GoDaddy (async UX improvement)

### 2.5 Latency & Performance Benchmarks

**Latency Targets by Use Case**:

| Use Case | p95 Latency | Acceptable Max | Company |
|----------|-------------|---------------|---------|
| Enterprise search (simple RAG) | <2s | 3s | Dropbox |
| Federated search | 3-5s | 10s | Slack |
| Code generation (small context) | 3-5s | 10s | Microsoft, GitLab |
| Code generation (large context) | 10-30s | 60s | GoDaddy, Microsoft |
| Multi-agent research | 20-60s | 120s | Anthropic |

**Latency Mitigation Strategies**:
1. **Streaming APIs**: GoDaddy - improves perceived performance
2. **Async Processing**: GoDaddy - acknowledge request, respond later
3. **Parallel Execution**: Anthropic - subagents run simultaneously
4. **Retry Logic**: GoDaddy - retry up to" 3x with backoff (adds latency)
5. **Model Selection**: Use smaller models when possible

**Token Usage Economics**:
- **Chat**: Baseline (1x)
- **Single Agent**: 4x chat
- **Multi-Agent**: 15x chat (Anthropic data)
- **Conclusion**: Multi-agent only for high-value tasks

---

## PART 3: MLOPS & OPERATIONAL INSIGHTS

### 3.1 Deployment & Serving Patterns

**Real-Time Inference** (All companies):
- API endpoints for synchronous requests
- GPU-accelerated compute for LLM inference
- Average response: 3-5s for GPT-4 class models

**Stateful Agent Serving** (Anthropic, GoDaddy):
- Long-running processes (minutes to hours)
- **Challenge**: Agents mid-flight during deploys
- **Solution**: Rainbow deployments
  - Gradual traffic shift old → new (10% → 50% → 100%)
  - Both versions run simultaneously
  - In-flight agents complete on old version

**Escrow VPCs** (Slack):
- LLMs hosted in AWS VPC controlled by Slack
- Model provider (e.g., OpenAI) has NO access to data
- Customer data never leaves trust boundary
- **Critical**: For enterprise compliance (HIPAA, SOC2, GDPR)

### 3.2 Model Optimization Techniques

**FP8 Mixed Precision Training** (DeepL):
- **Problem**: BF16 training slow, limits iteration speed
- **Solution**: 8-bit floating point (FP8) for training + inference
- **Implementation**: NVIDIA Transformer Engine
  - E4M3 format for forward pass (high precision)
  - E5M2 format for backward pass (high range)
- **Results**:
  - 50% training speedup (44.6% → 67% MFU)
  - With optimization: 80% MFU (25% gain over 15 months)
  - Minimal quality loss vs BF16 (negligible training loss difference)
- **Inference**: Maintained low latency with FP8
- **Lesson**: Hardware-specific optimizations (NVIDIA H100) unlock major gains

### 3.3 Evaluation & Testing at Scale

**LLM-as-Judge Pattern** (Segment, GitLab):

**Problem**: How to evaluate when there are many "right answers"?
- Example: "Customers who purchased ≥1 time" = "Customers who purchased >0 but <2 times"
- Traditional unit tests don't work for generation

**Solution** (Segment):
1. **Ground Truth**: Real-world ASTs from UI
2. **Synthetic Prompt Generation**: LLM generates prompts from ASTs (reverse)
3. **AST Generation**: LLM generates AST from synthetic prompt
4. **LLM Judge**: Compares generated AST to ground truth, scores alignment

**Results**:
- 90%+ alignment with human evaluation
- GPT-4-32k scored 4.55/5.0, Claude 4.02/5.0
- Enables comparing: model selection, prompt optimization, RAG impact, single vs multi-stage

**GitLab Testing Methodology**:
1. **Prompt Library**: Proxy for production (human + synthetic)
2. **Baseline Performance**: Test multiple models against library
3. **Daily Revalidation**: Ensure changes improve overall functionality

**Metrics Used**:
- Cosine Similarity Score
- Cross Similarity Score
- LLM Judge
- Consensus Filtering with LLM Judge

**Lesson**: LLM-as-Judge is emerging best practice for generative eval

### 3.4 Operational Lessons (GoDaddy's 10 Lessons)

**1. Sometimes One Prompt Isn't Enough**
- Mega-prompts (1500+ tokens) → accuracy decline, high cost, token limits
- **Solution**: Controller-Delegate pattern (task-oriented prompts)

**2. Be Careful with Structured Outputs**
- LLMs struggle with JSON, XML consistency
- **Mitigation**: Validate outputs, retry with corrections

**3. Prompts Aren't Portable Across Models**
- Same prompt performs differently on GPT-4 vs Claude vs Llama
- **Lesson**: Model-specific prompt engineering required

**4. AI Guardrails Are Essential**
- Models can refuse to transfer humans (bad UX)
- **Guardrails**:
  - Deterministic stop phrases (not LLM-decided)
  - PII & offensive content checks
  - Max interaction limits
  - Human approval for sensitive actions
- **Philosophy**: When uncertain, default to human intervention

**5. Models Can Be Slow and Unreliable**
- 1% failure rate at model provider
- 3-5s avg latency (GPT-4), up to 30s for large contexts
- **Mitigations**:
  - Retry logic (with timeout management)
  - Parallel redundant calls (costs more)
  - Streaming APIs (better UX)
  - Async responses

**6. Memory Management Is Hard**
- Long conversations hit token limits
- Context window management critical
- **Strategies**: Summarization, selective history, chunking

**7. Adaptive Model Selection Is the Future**
- Route simple queries → cheap models
- Complex queries → expensive models
- **Savings**: 50-70% cost reduction possible

**8. Use RAG Effectively**
- Not all problems need RAG
- Over-retrieved context can confuse model
- **Best Practice**: Retrieve only what's needed

**9. Tune Your Data for RAG**
- Chunk size matters (too small = context loss, too big = noise)
- Metadata helps retrieval
- Re-ranking improves relevance

**10. Test! Test! Test!**
- LLMs are probabilistic - edge cases will happen
- Comprehensive test suites essential
- Continuous validation (GitLab model)

### 3.5 Monitoring & Observability

**Production Tracing** (Anthropic):
- Full agent decision logging
- Monitor patterns, not content (privacy-preserving)
- **Insights**:
  - Why agents fail (bad queries? tool failures? poor sources?)
  - Discover unexpected behaviors
  - Fix systematic issues

**Metrics Tracked** (across companies):
- **Latency**: P50, p95, p99
- **Success Rate**: % of successful completions
- **Token Usage**: Cost tracking per query
- **Error Rate**: Model failures, timeout, validation errors
- **User Feedback**: Thumbs up/down, retry rate

**Agent-Specific Monitoring** (Anthropic):
- Decision patterns
- Interaction structures
- Tool call frequency
- Subagent spawn rates

### 3.6 Security & Privacy

**Slack's Security Principles**:
1. **Never Store External Data**: Federated approach eliminates storage risk
2. **Real-Time Permission Checks**: OAuth at query time
3. **User Explicit Consent**: Opt-in per source
4. **Least Privilege**: Only read scopes requested
5. **Escrow VPCs**: LLM provider can't access data
6. **Don't Store Answers**: Display once, immediately discard
7. **Compliance Reuse**: Encryption key mgmt, data residency

**General Best Practices**:
- Human-in-the-loop for sensitive actions
- PII detection and filtering
- Audit logging (who accessed what)
- Regular security reviews of AI-generated code

---

## PART 4: EVALUATION PATTERNS & METRICS

### 4.1 Offline Evaluation

| Company | Metric | Result | Insight |
|---------|--------|--------|---------|
| Anthropic | Performance gain (multi vs single) | 90.2% improvement | Multi-agent justifies 15x cost |
| Anthropic | BrowseComp benchmark | Token usage = 80% variance | More tokens > better prompts |
| Segment | LLM-as-Judge alignment | 90%+ with humans | Reliable for eval automation |
| Segment | Model comparison | GPT-4: 4.55/5, Claude: 4.02/5 | GPT-4 slightly ahead |
| DeepL | FP8 vs BF16 training loss | Negligible difference | FP8 viable without quality hit |
| DeepL | Training speedup (FP8) | 50% faster (67% vs 44% MFU) | Major throughput gain |

### 4.2 Online / Production Metrics

| Company | Metric | Target | Achieved | Method |
|---------|--------|--------|----------|--------|
| Dropbox | p95 latency | <2s | Yes (95% queries) | Hybrid IR + reranking |
| Microsoft | PR completion time | Reduce | 10-20% faster | 5000 repos tested |
| GoDaddy | LLM failure rate | <1% | 1% | Model provider SLA |
| GoDaddy | Avg latency (GPT-4) | <5s | 3-5s | Production monitoring |

### 4.3 Cost Analysis

**Token Economics** (Anthropic benchmarks):
- Single chat: 1x (baseline)
- Single agent: 4x
- Multi-agent: 15x

**Pricing Implications** (estimates):
- Claude Opus: ~$15/1M input tokens, ~$75/1M output
- Claude Sonnet: ~$3/1M input, ~$15/1M output  
- GPT-4: ~$10/1M input, ~$30/1M output

**Multi-Agent System** (10K queries/day):
- Avg 10K tokens per query
- Multi-agent: 150K tokens (15x multiplier)
- Cost: ~$200-300/day = $6K-9K/month just in LLM costs

**Cost Optimization Strategies**:
1. **Adaptive Model Selection**: Route to cheaper models when possible
2. **Worker Model Downgrade**: Use Sonnet instead of Opus for subagents
3. **Caching**: Reuse embeddings and chunks
4. **Batch Processing**: Non-urgent queries processed overnight

---

## PART 5: INDUSTRY-SPECIFIC PATTERNS

### 5.1 Tech Industry Characteristics

**What's Unique to Tech**:

1. **Code as Primary Artifact**
   - Code generation/review is top use case
   - Developers comfortable with DSLs and programmatic agents
   - GitHub Copilot-style tooling expected

2. **High Security Requirements**
   - Enterprise customers demand zero-trust
   - Escrow VPCs standard (Slack model)
   - Federated > indexed for sensitive data

3. **Experimentation Culture**
   - Rapid A/B testing infrastructure
   - Daily revalidation (GitLab)
   - Willing to invest in evaluation frameworks

4. **Scale & Performance Focus**
   - Latency budgets are strict (<2s for consumer)
   - 5000+ repos, 1000s of developers (Microsoft)
   - Training optimization critical (DeepL)

5. **Open Source Mindset**
   - Share lessons publicly (GoDaddy's 10 lessons)
   - Contribute tooling back (Anthropic research)

6. **High Token Budgets**
   - Willing to pay 15x for quality (Anthropic)
   - Multi-agent economically viable
   - Cost = less constraint than other industries

### 5.2 Common Failure Modes

**Technical Failures**:
1. **Mega-Prompt Bloat** (GoDaddy): 1500+ tokens → accuracy decline
2. **Synchronous Bottlenecks** (Anthropic): Lead waits for all subagents
3. **Tool Failures** (Anthropic): Search APIs down, agents stuck
4. **Token Limit Exceeded** (GoDaddy): Long conversations hit caps
5. **Model Provider Outages**: 1% failure rate (GoDaddy data)

**User Experience Failures**:
1. **Stuck in Bot Loop** (GoDaddy): No escape to human
2. **Slow Responses** (GoDaddy): 30s timeouts frustrate users
3. **Incorrect but Confident**: LLMs hallucinate with authority
4. **Permission Violations**: Showing data user can't access (Slack concern)

**Operational Failures**:
1. **Deployment Breaking Agents** (Anthropic): Before rainbow deploys
2. **Prompt Regression**: New prompt breaks old use cases
3. **Cost Explosion**: Multi-agent usage higher than predicted
4. **Non-Deterministic Bugs**: Can't reproduce agent failures

### 5.3 Tech Industry Best Practices

**System Design**:
- ✅ Hybrid RAG (lexical + semantic) for latency + quality
- ✅ Controller-Delegate for complex agents
- ✅ Human-in-the-loop for high-stakes actions
- ✅ Federated search for maximum security
- ✅ Streaming APIs for long-running generation

**MLOps**:
- ✅ LLM-as-Judge for evaluation
- ✅ Daily revalidation during development
- ✅ Rainbow deploys for stateful agents
- ✅ Production tracing (decision patterns, not content)
- ✅ Retry logic with exponential backoff

**Cost Management**:
- ✅ Adaptive model selection (route by complexity)
- ✅ Smaller models for workers (Sonnet vs Opus)
- ✅ Caching where possible
- ✅ Multi-agent only for high-value tasks

**Security**:
- ✅ Escrow VPCs for enterprise
- ✅ OAuth-scoped federated search
- ✅ PII & content filtering
- ✅ Never store generated outputs with sensitive data
- ✅ Audit logging

---

## PART 6: LESSONS LEARNED & TRANSFERABLE KNOWLEDGE

### 6.1 Top 10 Technical Lessons

1. **"Token Usage Matters More Than Prompts"** (Anthropic)
   - 80% of performance variance explained by tokens spent
   - Implication: Multi-agent parallelism > single-agent prompt engineering
   - Spend more tokens (via parallelism) for hard problems

2. **"Hybrid Beats Pure"** (Dropbox)
   - Hybrid IR (lexical + reranking) > pure semantic or pure lexical
   - Balance multiple objectives (latency + quality + cost)
   - Don't optimize one metric at expense of others

3. **"Federated Beats Indexed for Security"** (Slack)
   - Real-time API calls > stored data for sensitive applications
   - Trade latency for zero staleness and maximum privacy
   - Escrow VPCs standard for enterprise

4. **"FP8 is Production-Ready"** (DeepL)
   - 50% training speedup with negligible quality loss
   - Hardware-specific optimizations unlock major gains
   - Requires NVIDIA Transformer Engine + H100 GPUs

5. **"LLM-as-Judge Works"** (Segment, GitLab)
   - 90%+ human alignment achievable
   - Enables iterative improvement (model selection, prompt tuning)
   - Must have synthetic eval generation pipeline

6. **"Controller-Delegate > Mega-Prompts"** (GoDaddy)
   - Task-oriented prompts more accurate than 1500+ token mega-prompts
   - Modular design enables debugging
   - Clear separation of concerns

7. **"Prompts Aren't Portable"** (GoDaddy, Dropbox)
   - Model-specific optimization required
   - Can't assume same prompt works on GPT-4 and Claude
   - Maintain per-model prompt libraries

8. **"Guardrails Are Non-Negotiable"** (GoDaddy)
   - Deterministic controls for critical paths (human transfer)
   - PII detection, content moderation
   - Default to human when uncertain

9. **"Let Agents Handle Errors"** (Anthropic)
   - Inform agent when tool fails, let it adapt
   - Model intelligence > hardcoded fallbacks
   - Combine AI adaptability with deterministic safeguards

10. **"Test Continuously"** (GitLab)
    - Daily revalidation during active development
    - Prompt library as production proxy
    - Baseline all models before feature work

### 6.2 What Surprised Engineers

1. **Speed of Model Iteration** (All)
   - GPT-4 → GPT-4 Turbo → GPT-4o in 18 months
   - Forces continuous re-evaluation
   - Can't rely on model choice staying optimal

2. **1% Failure Rate Acceptable** (GoDaddy)
   - Industry has normalized 99% uptime for LLM providers
   - Forces retry logic into all production systems
   - Different from 99.99% uptime of traditional APIs

3. **Synchronous Still Dominates** (Anthropic)
   - Despite parallelism benefits, sync execution simpler
   - Async coordination still research problem
   - Industry hasn't solved agent-to-agent communication

4. **Rainbow Deploys Required** (Anthropic)
   - Standard blue-green breaks stateful agents
   - Gradual traffic shift essential
   - Shows immaturity of agent deployment tooling

5. **Human-in-the-Loop Still Critical** (Microsoft, GoDaddy)
   - Despite GPT-4 capabilities, can't fully automate high-stakes actions
   - Users want control (Microsoft code suggestions UX)
   - Trust requires transparency

### 6.3 Mistakes to Avoid

**Architecture**:
- ❌ Using mega-prompts past 1000 tokens (accuracy decline)
- ❌ Pure semantic search without lexical base (slow + expensive)
- ❌ Pre-indexing everything (staleness risk)
- ❌ Assuming one RAG pattern fits all use cases

**Operations**:
- ❌ No escape hatch to humans in chatbots
- ❌ Deploying stateful agents with blue-green (breaks in-flight)
- ❌ No retry logic (1% provider failures will hurt)
- ❌ Ignoring latency (30s timeouts kill UX)

**Evaluation**:
- ❌ Trusting single eval dataset
- ❌ Optimizing prompts without validation (overfitting)
- ❌ Assuming GPT-4 is best for all tasks (cost vs quality)

**Security**:
- ❌ Storing external data in enterprise context
- ❌ Letting LLM decide permission grants
- ❌ Generated code without human review

### 6.4 Transferability to Other Industries

**Highly Transferable**:
- ✅ RAG architectures (any knowledge-intensive domain)
- ✅ LLM-as-Judge evaluation (any generation task)
- ✅ Controller-Delegate agents (any multi-step workflow)
- ✅ Guardrails philosophy (any regulated industry)

**Requires Adaptation**:
- ⚠️ Multi-agent systems (need high-value tasks to justify 15x cost)
- ⚠️ FP8 training (need in-house model training)
- ⚠️ Federated search (need partner APIs)

**Domain-Specific (Hard to Transfer)**:
- ❌ Code generation patterns (software engineering only)
- ❌ Escrow VPCs at scale (enterprise SaaS only)
- ❌ GitHub/GitLab integration patterns

**Industry-by-Industry**:
- **Healthcare**: Federated pattern CRITICAL (HIPAA), Escrow VPCs required
- **Finance**: Guardrails 10x more important (regulations), Multi-agent for analysis
- **E-commerce**: RAG for product search, Code review not applicable
- **Manufacturing**: Predictive patterns more important than generative

---

## PART 7: REFERENCE ARCHITECTURE & RECOMMENDATIONS

### 7.1 Recommended Tech Stack (2025)

**For Enterprise Search & RAG**:

| Layer | Technology | Justification | Companies Using |
|-------|-----------|---------------|-----------------|
| **Foundation LLM** | GPT-4o or Claude Opus 4 | Best quality-to-cost ratio | All |
| **Worker LLM** | GPT-3.5 or Claude Sonnet 4 | Cost-effective parallelism | Anthropic, Segment |
| **Retrieval** | Elasticsearch (BM25) | Fast lexical search | Dropbox approach |
| **Reranking** | Cohere Rerank or custom embedding | Quality boost | Dropbox approach |
| **Vector DB** | Skip (on-the-fly) OR Pinecone | On-the-fly = freshness | Dropbox (skip) |
| **Auth** | OAuth 2.0 | Permission mgmt | Slack standard |
| **LLM Hosting** | AWS Bedrock/Escrow VPC | Enterprise security | Slack |
| **Observability** | Datadog + Custom tracing | Agent debugging | Anthropic pattern |
| **Deployment** | Kubernetes + ArgoCD | Rainbow deploys | Industry standard |
| **Evaluation** | Custom LLM-as-Judge | Quality tracking | Segment, GitLab |

**For Code Generation**:
| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Base** | GitHub Copilot / GitLab Duo | Platform integration |
| **Custom** | Fine-tuned on internal code | Domain-specific |
| **Review** | Custom LLM agent | Team-specific rules |
| **Testing** | Prompt library + daily revalidation | GitLab pattern |

**For Model Training** (if applicable):
| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Precision** | FP8 (Transformer Engine) | 50% speedup |
| **Hardware** | NVIDIA H100 | FP8 support |
| **Framework** | PyTorch + Transformer Engine | DeepL stack |

### 7.2 Reference Architecture: Enterprise AI Search

```
┌─────────────────────────────────────────────────────────────┐
│                        USER QUERY                            │
└────────────────────────┬────────────────────────────────────┘
                         │
                ┌────────▼─────────┐
                │  QUERY ROUTER    │
                │ (Classify)       │
                │ - Simple vs      │
                │   Complex        │
                └────────┬─────────┘
                         │
        ┌────────────────┴─────────────────┐
        │                                   │
┌───────▼────────┐                 ┌───────▼────────┐
│  SIMPLE PATH   │                 │  COMPLEX PATH  │
│  (RAG)         │                 │  (Multi-Agent) │
└────────────────┘                 └────────────────┘
        │                                   │
        │                          ┌────────▼─────────┐
        │                          │ LEAD AGENT       │
        │                          │ - Plan approach  │
        │                          │ - Spawn workers  │
        │                          └────────┬─────────┘
        │                                   │
        │                          ┌────────▼─────────┐
        │                          │ WORKER AGENTS    │
        │                          │ (Parallel)       │
        │                          └────────┬─────────┘
        │                                   │
┌───────▼───────────┐            ┌─────────▼──────────┐
│ HYBRID RETRIEVAL  │            │ TOOL EXECUTION     │
│ - Lexical (BM25)  │            │ - Search           │
│ - Federated APIs  │            │ - API calls        │
│ - On-the-fly      │            │ - Compression      │
│   chunking        │            │                    │
└───────┬───────────┘            └─────────┬──────────┘
        │                                   │
┌───────▼───────────┐            ┌─────────▼──────────┐
│ RERANKING         │            │ SYNTHESIS          │
│ - Embedding model │            │ - Aggregate        │
│ - Top-K selection │            │ - Deduplicate      │
└───────┬───────────┘            └─────────┬──────────┘
        │                                   │
        └──────────────┬────────────────────┘
                       │
              ┌────────▼─────────┐
              │ PERMISSION GATE  │
              │ - OAuth check    │
              │ - ACL filter     │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │ LLM GENERATION   │
              │ (Escrow VPC)     │
              │ - GPT-4/Claude   │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │ GUARDRAILS       │
              │ - PII check      │
              │ - Content mod    │
              │ - Validate       │
              └────────┬─────────┘
                       │
              ┌────────▼─────────┐
              │ RESPONSE         │
              │ (Stream/Display) │
              │ - Don't store    │
              └──────────────────┘

════════════════════════════════════════════════════════════
                  [SUPPORTING INFRASTRUCTURE]
════════════════════════════════════════════════════════════

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ EVALUATION   │  │ OBSERVABILITY│  │ DEPLOYMENT   │
│ - LLM Judge  │  │ - Tracing    │  │ - Rainbow    │
│ - Daily eval │  │ - Metrics    │  │ - K8s        │
│ - Synthetic  │  │ - Logging    │  │ - Gradual    │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 7.3 Decision Framework

**When to Use What**:

| Task Type | Latency Need | Budget | Security | → Recommendation |
|-----------|--------------|--------|----------|------------------|
| Simple Q&A | <2s | Low | Medium | Simple RAG (Dropbox) |
| Enterprise search | <5s | Medium | Critical | Federated (Slack) |
| Research/analysis | <60s | High | Medium | Multi-agent (Anthropic) |
| Code review | <10s | Medium | High | Integrated PR (Microsoft) |
| Evaluation | Offline | Medium | Low | LLM-as-Judge (Segment) |

**Model Selection**:
- **Interactive (user waiting)**: GPT-4o, Claude Sonnet (fast)
- **Batch/async**: GPT-4, Claude Opus (quality)
- **High volume**: GPT-3.5-turbo (cost)
- **Workers**: Claude Sonnet (balance)

### 7.4 Cost & Resource Estimates

**Infrastructure Costs** (Est. for 10K enterprise users):

**Scenario 1: Simple RAG** (Dropbox-style):
- LLM inference: $3K-5K/month (GPT-4o)
- Elasticsearch: $1K/month
- Embedding compute: $500/month
- Kubernetes: $2K/month
- **Total**: ~$6.5K-8.5K/month

**Scenario 2: Multi-Agent** (Anthropic-style):
- LLM inference: $15K-30K/month (15x multiplier)
- Additional workers: $5K-10K/month
- Infrastructure: $3K/month
- **Total**: ~$23K-43K/month

**Scenario 3: Code Review** (Microsoft-style):
- LLM inference: $8K-12K/month (5000 repos)
- GitHub/GitLab integration: Included
- Compute: $2K/month
- **Total**: ~$10K-14K/month

**Team Size** (by scenario):
- **Simple RAG**: 4-6 engineers (2 ML, 2 backend, 1 DevOps, 1 security)
- **Multi-Agent**: 6-8 engineers (3 ML, 2 backend, 2 DevOps, 1 research)
- **Code Review**: 5-7 engineers (2 ML, 2 backend, 1 DevOps, 1 product, 1 security)

**Timeline**:
- MVP (Simple RAG): 2-3 months
- Production (with eval): 4-6 months
- Multi-agent system: 6-9 months
- Mature platform: 12-18 months

---

## PART 8: REFERENCES & FURTHER READING

### 8.1 Articles Analyzed

1. Dropbox (2025). "Building Dash: How RAG and AI agents help us meet the needs of businesses". https://dropbox.tech/machine-learning/building-dash-rag-multi-step-ai-agents-business-users

2. Slack (2025). "How we built enterprise search to be secure and private". https://slack.engineering/how-we-built-enterprise-search-to-be-secure-and-private/

3. Anthropic (2025). "How we built our multi-agent research system". https://www.anthropic.com/engineering/multi-agent-research-system

4. Microsoft (2025). "Enhancing Code Quality at Scale with AI-Powered Code Reviews". https://devblogs.microsoft.com/engineering-at-microsoft/enhancing-code-quality-at-scale-with-ai-powered-code-reviews/

5. DeepL (2025). "How we built DeepL's next-generation LLMs with FP8 for training and inference". https://www.deepl.com/en/blog/tech/next-generation-llm-fp8-training

6. GoDaddy (2024). "LLM From the Trenches: 10 Lessons Learned Operationalizing Models at GoDaddy". https://www.godaddy.com/resources/news/llm-from-the-trenches-10-lessons-learned-operationalizing-models-at-godaddy

7. Segment (2024). "LLM-as-Judge: Evaluating and Improving Language Model Performance in Production". https://segment.com/blog/llm-as-judge/

8. GitLab (2024). "Developing GitLab Duo: How we validate and test AI models at scale". https://about.gitlab.com/blog/2024/05/09/developing-gitlab-duo-how-we-validate-and-test-ai-models-at-scale/

### 8.2 Referenced Papers & Frameworks

- Salesforce BOLAA (Multi-Agent Architecture)
- JudgeLM, Prometheus, LLM-SQL-Solver (LLM-as-Judge)
- BrowseComp Benchmark (Anthropic eval)
- NVIDIA Transformer Engine documentation

### 8.3 Related Technologies to Explore

- **Retrieval**: Elasticsearch, Pinecone, Weaviate, Chroma
- **Embeddings**: OpenAI Ada, Cohere, sentence-transformers
- **LLM Frameworks**: LangChain, LlamaIndex, Semantic Kernel
- **Evaluation**: Weights & Biases, Arize AI, Evidently AI
- **Deployment**: BentoML, Ray Serve, Triton
- **Precision**: FP8, INT8, quantization techniques

### 8.4 Follow-Up Questions for Deeper Analysis

1. **Dropbox**: What percentage of queries use agents vs simple RAG? ROI on agent complexity?

2. **Slack**: Comparative latency distribution across external sources? Which partners are slowest?

3. **Anthropic**: Async multi-agent prototype results? Performance vs coordination complexity?

4. **Microsoft**: What's the false positive rate on code review suggestions? Developer acceptance rate?

5. **DeepL**: FP8 quality on downstream tasks (BLEU, etc.)? Memory savings quantified?

6. **GoDaddy**: Controller-Delegate accuracy improvement? Cost savings from adaptive routing?

7. **Segment**: How does LLM-as-Judge handle edge cases? Inter-rater reliability?

8. **GitLab**: Prompt library size? Coverage vs real production diversity?

9. **All**: How does model versioning affect evaluation? Continuous revalidation cadence?

10. **All**: Security incident learnings? PII leakage rates?

---

## APPENDIX: TECH INDUSTRY SUMMARY STATISTICS

### Companies by Sub-Domain

| Sub-Domain | Count | Representative Companies |
|------------|-------|-------------------------|
| **Code Generation** | 15+ | Microsoft, GitHub, GitLab, Intuit, Replit |
| **Enterprise Search** | 10+ | Dropbox, Slack, Canva, Airtable |
| **Multi-Agent** | 5+ | Anthropic, GoDaddy, Manus, Netguru |
| **Model Training/Ops** | 8+ | DeepL, NVIDIA, OpenAI, Anthropic |
| **Evaluation** | 5+ | Segment, GitLab, GitHub |
| **Support/Chat** | 10+ | GoDaddy, Honeycomb, Intercom, Salesforce |

### Year Distribution (2023-2025)

- **2025**: 40+ articles (Jan-Nov)
- **2024**: 25+ articles
- **2023**: 10+ articles

**Trend**: Accelerating article publication (maturity + adoption)

### Consensus Tech Stack

**Appears in 50%+ of companies**:
- ✅ GPT-4 or Claude Opus/Sonnet
- ✅ RAG architecture
- ✅ OAuth/permission management
- ✅ Cloud hosting (AWS/Azure/GCP)
- ✅ Hybrid retrieval (lexical + semantic)
- ✅ LLM-as-Judge or similar eval

**Emerging (25-50%)**:
- Rainbow deployments
- Controller-Delegate pattern
- FP8 precision
- Streaming APIs
- Multi-agent systems

---

**Analysis Completed**: November 2025  
**Total Companies in Tech Folder**: 71 (68 from 2023-2025)  
**Companies Deep-Analyzed**: 8  
**Coverage**: Representative sample across all major sub-domains  

**Next Industry**: Delivery & Mobility (32 articles, 2023-2025)

---

*This analysis provides a comprehensive overview of AI Agents & LLMs in the Tech industry based on 2023-2025 use cases. Companies can use this as a reference architecture and decision framework for building enterprise AI systems.*
