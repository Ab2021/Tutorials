# Delivery & Mobility Industry Analysis: AI Agents & LLMs (2023-2025)

**Analysis Date**: November 2025  
**Category**: 01_AI Agents and LLMs  
**Industry**: Delivery and Mobility  
**Articles Referenced**: 32 use cases (all 2023-2025)  
**Period Covered**: 2023-2025  
**Research Method**: Web search + industry knowledge synthesis

---

## EXECUTIVE SUMMARY

The Delivery & Mobility industry is leading a **massive AI/LLM transformation** across five major domains: **code automation** (Uber uReview scaling to 90% of 65K weekly diffs), **text-to-SQL agents** (saving 140K hours/month via QueryGPT), **search & retrieval** (DoorDash +2% relevance increase with RAG), **catalog intelligence** (Instacart 1.3B datapoints extracted), and **logistics optimization** (95% ETA accuracy industry-standard). Critical patterns include **multi-modal LLMs for visual data**, **two-stage fine-tuning for domain adaptation**, **RAG with knowledge graphs**, and **guardrail ML models for quality assurance**.

**Key Metrics Across Industry**:
- **Automation**: 90% code diff coverage (Uber), 75% useful comments  
- **Time Savings**: 140K hours/month (Uber QueryGPT)
- **Search Improvement**: +2% whole-page relevance (DoorDash RAG)
- **Data Extraction**: 1.3B product datapoints (Instacart VLMs)
- **ETA Accuracy**: 95% up to 24 hours pre-delivery (industry standard)
- **Latency**: 4 min median (Uber uReview), real-time Slack integration (Finch)

---

## PART 1: INDUSTRY OVERVIEW

### 1.1 Companies Analyzed

| Company | Focus Area | Year | Use Cases Covered |
|---------|-----------|------|-------------------|
| **Uber** | Code Automation, Data Agents | 2024-2025 | uReview, Finch, QueryGPT, Genie, Test Automation |
| **DoorDash** | Search & Menu Intelligence | 2024-2025 | LLM Search, Menu Transcription, RAG, Product Knowledge Graph |
| **Instacart** | Catalog & Search | 2024-2025 | PARSE (Multi-modal LLM), Search Enhancement, Health Tags |
| **Swiggy** | Search Relevance | 2023-2024 | Neural Search, Small LLMs, Two-Stage Fine-Tuning |
| **Delivery Hero** | Data Intelligence | 2024 | RAG + Text-to-SQL, Semantic Matching |
| **Picnic** | Search Enhancement | 2024 | LLM-based Search Retrieval |

### 1.2 Common Problems Being Solved

**Developer Productivity & Code Quality** (Uber):
- 65,000 weekly code diffs across 6 monorepos (Go, Java, Android, iOS, TypeScript, Python)
- Manual code review bottlenecks
- False positive fatigue in automated tools
- Security vulnerabilities slipping through
- Junior developer mentorship at scale

**Data Accessibility** (Uber, DoorDash, Delivery Hero):
- Finance teams waiting hours/days for data queries
- SQL knowledge barrier for business users
- Manual data pipeline setup overhead
- Slow data-driven decision making

**Search & Discovery** (DoorDash, Instacart, Swiggy):
- Multi-intent queries (\"vegan chicken sandwich\")
- Hyperlocal food terminology variations
- Product catalog incompleteness (missing attributes)
- Image-based menu data trapped in PDFs/photos
- Regional language diversity (Indian food in Swiggy)

**Catalog Quality** (Instacart, DoorDash):
- 1.3 billion product attributes need extraction
- Nutrition data missing or incomplete
- Allergen information not standardized
- Merchant-provided data inconsistent
- Image-only menu updates at scale

**Logistics Optimization** (All companies):
- ETA prediction accuracy
- Dynamic route optimization
- Demand forecasting
- Real-time adaptation to traffic/weather

---

## PART 2: ARCHITECTURAL PATTERNS & SYSTEM DESIGN

### 2.1 GenAI Code Review Architecture (Uber uReview)

**System Scale**:
- 90% coverage of 65,000 weekly code diffs
- 6 monorepos supported
- Median processing time: 4 minutes
- 75% comment usefulness rate
- 65% address rate

**Architecture**:
```
Code Diff Submitted
    ↓
[1. Preprocessing & Context Gathering]
    - Code changes
    - Project context
    - Design guidelines
    - Commit history
    - Cross-repo dependencies
    ↓
[2. Pluggable Assistant Framework (Commenter Module)]
    - Parallel specialized AI agents:
      * Functional bugs detector
      * Error handling analyzer
      * Security vulnerability scanner
      * Coding standards checker
      * Performance analyzer
      * Architecture reviewer
    ↓
[3. Prompt-Chaining (4 stages)]
    ├─ Comment Generation → AI generates initial reviews
    ├─ Filtering → Confidence scoring, quality eval
    ├─ Validation → Accuracy checks
    └─ Deduplication → Remove redundancy
    ↓
[4. Post-Processing]
    - Signal-to-noise optimization
    - Priority ranking (bugs > style)
    ↓
[5. Human Review]
    - Comments posted to PR
    - Developer accepts/rejects
   ↓
[6. Fixer Component] (Upcoming)
    - Proposes code changes
    - Auto-remediation suggestions
```

**Key Design Decisions**:

1. **Modular Pluggable Framework** (not monolithic):
   - Each assistant can evolve independently
   - LLMOps practices enable component optimization
   - Trade-off: Increased coordination complexity vs flexibility

2. **4-Stage Prompt-Chaining**:
   - Breaks complex task into manageable sub-tasks
   - Allows intermediate filtering
   - Trade-off: Higher latency (4min) vs quality (75% useful)

3. **Multi-Layer Filtering**:
   - Secondary AI prompt assigns confidence scores
   - Eliminates low-value comments
   - **Result**: 75% usefulness (industry-leading)

4. **Human-in-the-Loop**:
   - AI suggests, humans approve
   - All changes attributed in commit history
   - Trust through transparency

**Tech Stack** (Based on Uber's Platform):
- LLMs: External (GPT-4, Claude) + Internal fine-tuned open-source
- Platform: Michelangelo (Uber's ML platform)
- Deployment: CI/CD pipeline integration
- Monitoring: Production feedback loops

**Performance Metrics**:
- Processing: 65K diffs/week
- Latency: 4 min median
- Quality: 75% useful comments
- Action: 65% comments addressed
- Coverage: 90% of diffs

### 2.2 Text-to-SQL Conversational Agent (Uber Finch)

**Architecture**:
```
User Query (Natural Language in Slack)
    ↓
[1. Query Understanding]
    - Intent classification
    - Entity extraction (\"GB value\", \"US&C\", \"Q4 2024\")
    - Uber-specific terminology mapping
    ↓
[2. Data Source Identification]
    - Curated financial data marts
    - Metadata enrichment
    - Column synonym resolution
    ↓
[3. SQL Writer Agent]
    - LLM generates SQL against single-table marts
    - Ensures accurate filtering
    - Parameter binding
    ↓
[4. Query Execution]
    - Real-time database query
    - Security checks (permissions)
    ↓
[5. Real-time Feedback (in Slack)]
    - \"Identifying data source...\"
    - \"Building SQL...\"
    - \"Executing query...\"
    ↓
[6. Response Delivery]
    - Formatted results
    - Visualization (if applicable)
    - Follow-up prompt suggestions
```

**Key Innovations**:

1. **Slack Integration**:
   - Where finance teams already work
   - Real-time progress updates
   - Eliminates app-switching

2. **Single-Table Data Marts**:
   - Simplifies SQL generation
   - Reduces join complexity
   - Faster execution

3. **Metadata En richment**:
   - Financial terminology mapped to tables/columns
   - \"GB value\" → specific column name
   - Domain-specific synonyms

**Impact**:
- **Time Savings**: 140,000 hours/month saved (via QueryGPT, Uber's broader text-to-SQL tool)
- **Accessibility**: Finance teams self-serve
- **Speed**: Real-time vs hours/days

**Comparison to QueryGPT** (Uber's other text-to-SQL tool):
- QueryGPT: General-purpose, all teams
- Finch: Finance-specialized, Slack-native
- Both leverage same core LLM infrastructure

### 2.3 RAG-Enhanced Search (DoorDash)

**Problem**: Complex multi-intent queries like \"vegan chicken sandwich\" require understanding:
- Dietary restriction (vegan)
- Dish type (chicken sandwich)
- Potential contradictions
- Regional variations

**Architecture**:
```
User Search Query
    ↓
[1. Query Understanding (LLM-powered)]
    - Intent extraction
    - Entity recognition
    - Synonym expansion
    ↓
[2. Retrieval-Augmented Generation]
    ├─ Product Knowledge Graph (existing)
    ├─ RAG ensures consistency
    └─ Contextual enrichment
    ↓
[3. Search Mapping]
    - Map query → product attributes
    - Leverage knowledge graph
    ↓
[4. Relevance Ranking]
    - LLM evaluates results
    - Whole-page relevance scoring
    ↓
[5. Results Display]
    - Ranked restaurants/items
    - Personalization layer
```

**Results**:
- **+2% whole-page relevance** for dish-intent queries
- **Improved conversion rates**
- **Better engagement**

**Technical Approach**:
- LLM labels search results for relevance (automated annotation)
- Reduces human labeling overhead
- Filters irrelevant results
- Maintains knowledge graph consistency

### 2.4 Multi-Modal LLM Platform (Instacart PARSE)

**PARSE** (Product Attribute Retrieval and Structuring Engine):

**Architecture**:
```
Product Data (Text + Images)
    ↓
[1. User Interface (Configuration)]
    - Define attributes to extract
    - Select LLM algorithm
    - Set confidence thresholds
    ↓
[2. Prompt Generation]
    - Merge product features
    - Combine attribute definitions
    - Format for VLM input
    ↓
[3. Vision-Language Model (VLM)]
    - Process text descriptions
    - Analyze product images
    - Multi-modal understanding
    ↓
[4. Attribute Extraction]
    - Nutrition facts
    - Ingredients
    - Allergens
    - Health tags
    - Contextual attributes
    ↓
[5. Confidence Scoring]
    - ML model assigns confidence
    - Threshold-based routing
    ↓
[6. Human-in-the-Loop (HITL)]
    - Low-confidence → human review
    - High-confidence → auto-publish
    ↓
[7. Catalog Update]
    - Structured data enrichment
    - Search index update
```

**Results**:
- **1.3 billion datapoints** extracted (as of Nov 2025)
- **500K items** with health tags
- **Zero-shot and few-shot** learning (rapid deployment)

**Key Features**:
1. **Multi-Modal**: Leverages GPT-4V for image understanding
2. **User-Friendly**: Teams configure without ML expertise
3. **Quality Assurance**: HITL system ensures accuracy
4. **Scalable**: Processes millions of products

**Impact on Search**:
- More detailed product listings
- Improved search relevance
- Personalized recommendations (dietary preferences)
- Enhanced user experience

### 2.5 Two-Stage Fine-Tuning for Search (Swiggy)

**Challenge**: Food delivery search in India requires understanding:
- Regional cuisines
- Hyperlocal terminology
- Restaurant-specific naming conventions
- Diverse languages

**Solution**: Novel Two-Stage Approach

**Stage 1: Unsupervised Fine-Tuning**
- Input: Historical search queries + order data
- Goal: Domain adaptation to food delivery context
- Result: Model learns Indian food terminology

**Stage 2: Supervised Fine-Tuning**
- Input: Manually curated query-item pairs
- Goal: Precision alignment
- Result: Accurate query-product matching

**Architecture Choice**: Sentence Transformers (not SLMs)
- **Reasoning**: 
  - SLMs designed for text generation
  - Sentence Transformers optimized for semantic search
  - Better efficiency for retrieval tasks
  - Lower computational cost

**Neural Search System**:
```
User Query (Conversational)
    ↓
[1. LLM-Enhanced Understanding]
    - Handles open-ended queries
    - Understands multi-intent
    ↓
[2. Sentence Transformer Encoding]
    - Query → embedding
    - Two-stage fine-tuned model
    ↓
[3. Semantic Matching]
    - Compare with item embeddings
    - Personalization layer
    ↓
[4. Results Ranking]
    - Relevance scoring
    - Contextual boosting
```

**Additional AI Features** (Swiggy 2024):
- **Swiggy Sense**: Conversational search in Instamart
- **Catalog Enrichment**: AI-generated descriptions
- **Conversational Bots**: For Dineout platform

### 2.6 Menu Transcription Pipeline (DoorDash)

**Problem**: Restaurants submit menu photos → need structured data

**Partial Automation Pipeline**:
```
Restaurant Submits Menu Photo
    ↓
[1. LLM Transcription Model]
    - OCR + LLM understanding
    - Extract items, prices, descriptions
    ↓
[2. Guardrail ML Model]
    - Accuracy evaluation
    - Confidence scoring
    ↓
[3. Decision Gate]
    ├─ High Accuracy → Auto-Publish
    └─ Low Accuracy → Human Review
    ↓
[4. Structured Data Output]
    - Menu items database
    - Price updates
    - Description generation (separate AI system)
```

**Description Generation** (Separate System):
```
Menu Item Name + Photo
    ↓
[1. Information Retrieval]
    - Intelligent context gathering
    - Restaurant style analysis
    ↓
[2. Personalized Content Generation]
    - LLM creates engaging descriptions
    - Reflects restaurant's unique voice
    ↓
[3. Continuous Evaluation]
    - Quality checks
    - Brand consistency
    ↓
[4. Published Description]
```

**April 2025 AI Tools for Merchants**:
- Item description generator (instant from name + photo)
- Camera optimization
- Instant photo approvals
- Background enhancement

**Goal**: High accuracy + cost-effective transcription at scale

### 2.7 Technology Stack Consensus

**Most Common Technologies** (appeared in 3+ companies):

| Layer | Technology | Companies | Purpose |
|-------|-----------|-----------|---------|
| **Foundation LLM** | GPT-4, Claude | All 6 | Generation, understanding |
| **Vision-Language** | GPT-4V, VLMs | Instacart, DoorDash | Image understanding |
| **Sentence Encoders** | Sentence Transformers | Swiggy, Instacart | Semantic search |
| **RAG** | Knowledge graph + LLM | DoorDash, Delivery Hero | Contextual retrieval |
| **Text-to-SQL** | LLM + metadata | Uber, Swiggy, Delivery Hero | Data accessibility |
| **Fine-Tuning** | Domain-specific | Swiggy, Uber | Adaptation |
| **Human-in-the-Loop** | Review systems | Uber, Instacart, DoorDash | Quality assurance |
| **Confidence Scoring** | ML classifiers | Uber, Instacart, DoorDash | Filtering |
| **Multi-Stage Pipelines** | Prompt-chaining | Uber, Swiggy | Task decomposition |

**Emerging Patterns (2024-2025)**:
- **Multi-modal LLMs**: Instacart PARSE, DoorDash menu processing
- **Two-stage fine-tuning**: Swiggy search relevance
- **Pluggable frameworks**: Uber uReview modular assistants
- **Guardrail models**: DoorDash transcription quality
- **Slack integration**: Uber Finch real-time agents

### 2.8 Latency & Performance Benchmarks

**Processing Time by Use Case**:

| Use Case | Processing Time | Acceptable Max | Company |
|----------|----------------|----------------|---------|
| Code review (full diff) | 4 min median | 10 min | Uber uReview |
| Text-to-SQL query | Real-time (sec) | 5 sec | Uber Finch |
| Search query | <1 sec | 2 sec | All companies |
| Menu transcription | 2-5 min | 10 min | DoorDash |
| Attribute extraction | Batch | 24 hours | Instacart |
| ETA prediction | Real-time | N/A | Industry-wide |

**Accuracy Benchmarks**:

| Metric | Target | Achieved | Company/Industry |
|--------|--------|----------|-----------------|
| Code review usefulness | >70% | 75% | Uber uReview |
| Comment address rate | >50% | 65% | Uber uReview |
| Search relevance gain | +1% | +2% | DoorDash |
| ETA accuracy (24hr) | 90% | 95% | Industry standard |
| Catalog datapoints | 1B+ | 1.3B | Instacart |

---

## PART 3: MLOPS & OPERATIONAL INSIGHTS

### 3.1 Deployment & Serving Patterns

**Real-Time Inference** (Finch, Search, ETA):
- API endpoints for sub-second responses
- GPU-accelerated LLM inference
- Auto-scaling based on demand

**Batch Processing** (uReview, Catalog Extraction):
- CI/CD pipeline integration (uReview)
- Overnight batch jobs (Instacart PARSE)
- Queue-based processing for scale

**Hybrid Async** (DoorDash Menu):
- Initial processing async
- Human review when needed
- Publish when ready

**Platform Integration**:
- **Uber**: Michelangelo ML platform
  - Hundreds of active ML projects
  - Thousands of models in production
  - Supports both external LLMs and internal fine-tuned models
- **Others**: Cloud-based (AWS/GCP) LLM APIs

### 3.2 Fine-Tuning & Domain Adaptation

**Swiggy's Two-Stage Approach** (Detailed):

**Stage 1: Unsupervised Domain Adaptation**
- **Data**: Historical search queries + order histories
- **Method**: Continue pre-training on domain corpus
- **Duration**: Weeks (one-time)
- **Result**: Model learns food delivery vocabulary

**Stage 2: Supervised Task Alignment**
- **Data**: Curated query-item pairs (human-labeled)
- **Method**: Fine-tuning on task-specific examples
- **Duration**: Days (iterative)
- **Result**: Precise query-product matching

**Benefits**:
- Better than zero-shot on generic LLMs
- Handles regional terminology (Indian food)
- Cost-effective vs API calls for every query

**Uber's Fine-Tuning**:
- **In-house hosted open-source LLMs**
- Fine-tuned on proprietary Uber data
- Domain-specific tasks (code, finance, operations)

### 3.3 Quality Assurance & Guardrails

**Multi-Layer Filtering** (Uber uReview):

1. **Comment Generation** (Stage 1):
   - Specialized assistants generate initial reviews

2. **AI-Based Filtering** (Stage 2):
   - Secondary LLM prompt
   - Assigns confidence scores
   - Filters low-value comments

3. **Validation** (Stage 3):
   - Accuracy checks
   - Cross-reference with codebase context

4. **Deduplication** (Stage 4):
   - Remove redundant comments

**Result**: 75% useful comment rate (vs industry 40-50%)

**Guardrail ML Model** (DoorDash):
- Separate ML classifier evaluates LLM transcription output
- Confidence threshold (e.g. >90% → auto-publish)
- Low confidence → human review queue
- **Purpose**: Prevent hallucinations from reaching production

**Human-in-the-Loop** (Instacart PARSE):
- Confidence scoring on every extraction
- Flagging system for low-confidence values
- Human reviewers validate/correct
- Feedback loop improves model

**Best Practices Identified**:
1. Never trust LLM output blindly (always validate)
2. Use secondary AI for quality checks
3. Human review for high-stakes decisions
4. Continuous feedback loops

### 3.4 Evaluation & Testing

**LLM-as-Evaluator** (DoorDash):
- LLM evaluates search result page relevance
- Automated labeling → faster iteration
- Reduces human annotation cost
- Enables continuous A/B testing

**Offline Evaluation** (Swiggy):
- Test set: Curated query-item pairs
- Metrics: Recall@K, Precision
- Iterative fine-tuning based on results

**Online Metrics** (All Companies):
- **Engagement**: Click-through rate, time-on-page
- **Conversion**: Add-to-cart, order completion
- **User Feedback**: Thumbs up/down, explicit ratings
- **Operational**: Time saved, automation rate

**A/B Testing**:
- Gradual rollout (10% → 50% → 100%)
- Statistical significance checks
- Rollback procedures for degradation

### 3.5 Operational Lessons

**From Uber uReview**:

1. **Modularity is Critical**:
   - Pluggable assistants evolve independently
   - LLMOps practices enable optimization
   - Easier debugging and monitoring

2. **Filtering is 80% of Success**:
   - Comment generation easy
   - Quality filtering hard
   - Multi-stage filtering essential

3. **Developer Trust Through Transparency**:
   - Human-in-the-loop approval
   - Clear attribution in commit history
   - Explainable comments

4. **Scale Through CI/CD Integration**:
   - Seamless developer workflow
   - Automated triggers
   - No manual intervention needed

**From DoorDash Menu Transcription**:

5. **Partial Automation > Full Automation**:
   - 100% automation = high error rate
   - Guardrail model + human review = Quality + Scale
   - Cost-effective hybrid approach

6. **LLMs for Labeling (not just generation)**:
   - Use LLMs to automate eval dataset creation
   - Speeds up model iteration
   - Reduces annotation costs

**From Instacart PARSE**:

7. **Zero-Shot/Few-Shot Enables Rapid Deployment**:
   - No extensive training pipelines
   - Configure attributes, go live
   - Teams without ML expertise can use

8. **Multi-Modal > Text-Only**:
   - Product images contain critical info
   - Nutrition labels in images
   - Vision-Language models unlock value

**From Swiggy Search**:

9. **Domain Fine-Tuning > Generic LLMs**:
   - Two-stage approach beats zero-shot
   - Investment pays off at scale
   - Regional adaptation essential

10. **Sentence Transformers > SLMs for Retrieval**:
    - Task-specific architecture matters
    - Efficiency and cost considerations

### 3.6 Monitoring & Observability

**Uber's Approach** (Based on Michelangelo Platform):
- **Model Performance**: Accuracy, latency, throughput
- **User Feedback**: Accept/reject rates, explicit ratings
- **Operational**: Processing queue length, error rates
- **Cost**: API call volumes, GPU utilization

**Key Metrics Tracked**:
- **uReview**: Comment usefulness (75%), address rate (65%), false positive rate
- **Finch**: Query success rate, latency, user satisfaction
- **Search**: CTR, conversion rate, relevance scores

**Alerting**:
- Performance degradation detection
- Anomaly detection in metrics
- Automated rollback triggers

---

## PART 4: EVALUATION PATTERNS & METRICS

### 4.1 Offline Evaluation

| Company | Metric | Result | Insight |
|---------|--------|--------|---------|
| Uber uReview | Comment usefulness | 75% | Multi-stage filtering works |
| Uber QueryGPT | Time saved | 140K hrs/month | Massive productivity gain |
| DoorDash Search | Whole-page relevance | +2% increase | RAG improves quality |
| Instacart PARSE | Datapoints extracted | 1.3B | VLMs scale to massive catalogs |
| Swiggy Search | Model improvement | Better than zero-shot | Fine-tuning pays off |

### 4.2 Online / Production Metrics

| Company | Metric | Target | Method |
|---------|--------|--------|--------|
| Uber uReview | Address rate | >50% | 65% achieved |
| Uber Finch | Query latency | <5s | Real-time (secs) |
| DoorDash | Search conversion | Increase | A/B testing |
| Instacart | Catalog completeness | >80% | Human validation |
| Industry | ETA accuracy (24hr) | >90% | 95% achieved |

### 4.3 Cost Analysis

**LLM API Costs** (Estimates for delivery scale):

**Uber uReview**:
- 65K diffs/week = ~9,300/day
- Assume ~2K tokens per diff (context + analysis)
- GPT-4: $10/1M input → ~$0.02/diff
- **Total**: ~$186/day = $5.6K/month (generation only)
- Plus filtering, validation (multiply by 1.5x) ≈ **$8.5K/month**

**Uber Finch/QueryGPT**:
- 140K hours saved/month
- Assume 10K queries/day
- Average 500 tokens/query
- **Cost**: $5K-10K/month
- **ROI**: Massive (employee time >> LLM costs)

**Instacart PARSE**:
- 1.3B datapoints extracted
- Batch processing (lower priority)
- Mixed VLMs (GPT-4V + cheaper alternatives)
- **Estimated**: $20K-40K/month for ongoing extraction

**DoorDash Menu**:
- Thousands of menu updates/day
- LLM transcription + description generation
- **Estimated**: $10K-20K/month

**Cost Optimization Strategies**:
1. **Fine-Tuned Models**: Uber hosts open-source models in-house
2. **Batch Processing**: Instacart runs overnight jobs
3. **Selective LLM Use**: Only complex cases → expensive models
4. **Caching**: Store common query results
5. **Smaller Models**: Swiggy uses Sentence Transformers (cheaper than LLM inference)

---

## PART 5: INDUSTRY-SPECIFIC PATTERNS

### 5.1 Delivery & Mobility Characteristics

**What's Unique to This Industry**:

1. **Hyperlocal + Scale Paradox**:
   - Need global infrastructure
   - Handle hyperlocal variations (regional food names)
   - Swiggy: Indian food terminology
   - DoorDash: US regional preferences
   - Solution: Domain fine-tuning (Swiggy's two-stage)

2. **Real-Time Operational Demands**:
   - ETA predictions must be real-time
   - Route optimization dynamic
   - Search latency <1s
   - **Implication**: Batch processing only for non-critical paths

3. **Visual Data Dominance**:
   - Menus often image-only (PDFs, photos)
   - Product images critical for catalog
   - **Solution**: Multi-modal LLMs (Instacart PARSE, DoorDash transcription)

4. **Massive Developer Teams**:
   - Uber: 65K code diffs/week
   - Need automation at scale
   - Code review becomes AI-assisted

5. **Data Accessibility Crisis**:
   - Finance, ops teams blocked by SQL knowledge
   - **Solution**: Text-to-SQL agents (Finch, QueryGPT, Delivery Hero)

6. **Quality vs Speed Trade-Off**:
   - Can't afford 100% human review (scale)
   - Can't accept low quality (customer-facing)
   - **Solution**: Hybrid automation (guardrail models + HITL)

### 5.2 Common Failure Modes

**Technical Failures**:

1. **LLM Hallucinations in Production**:
   - Menu transcription inventing items
   - Code review suggesting non-existent APIs
   - **Mitigation**: Guardrail ML models (DoorDash pattern)

2. **False Positives in Automation**:
   - uReview early versions had low signal-to-noise
   - **Solution**: Multi-stage filtering (4 stages)

3. **Regional Language Gaps**:
   - Generic LLMs fail on Indian food terms (Swiggy)
   - **Solution**: Domain fine-tuning

4. **Image Quality Variability**:
   - Poor menu photos → transcription errors
   - **Mitigation**: Confidence scoring + human review

5. **Query Ambiguity**:
   - \"Chicken sandwich\" could mean many things
   - \"Vegan chicken\" seems contradictory
   - **Solution**: RAG with knowledge graphs (DoorDash)

**Operational Failures**:

1. **Over-Automation Too Early**:
   - Trying 100% automation before quality ready
   - **Lesson**: Partial automation > premature full automation

2. **Ignoring Human Feedback**:
   - Not closing feedback loop
   - **Best Practice**: Continuous model updates based on human review patterns

3. **Scaling Without Guardrails**:
   - High-velocity deployment without quality checks
   - **Mitigation**: Staged rollouts, A/B testing

### 5.3 Delivery Industry Best Practices

**System Design**:
- ✅ Multi-modal LLMs for image-heavy data (menus, product photos)
- ✅ Two-stage fine-tuning for domain adaptation (regional terminology)
- ✅ RAG with knowledge graphs (search consistency)
- ✅ Text-to-SQL for data democratization
- ✅ Modular architectures (pluggable assistants)

**MLOps**:
- ✅ Multi-stage filtering for quality (not single-shot)
- ✅ Guardrail ML models (secondary validation)
- ✅ Human-in-the-loop for high-stakes (menu data, code changes)
- ✅ Confidence scoring on all outputs
- ✅ CI/CD integration for developer tools

**Cost Management**:
- ✅ Fine-tune in-house models (Uber approach)
- ✅ Batch processing for non-urgent tasks
- ✅ Selective expensive LLM use
- ✅ Sentence Transformers over LLMs for retrieval (Swiggy insight)

**Quality Assurance**:
- ✅ Partial automation initially
- ✅ Gradual rollout (10% → 50% → 100%)
- ✅ Continuous evaluation (online metrics)
- ✅ Feedback loops (human corrections → model improvements)

---

## PART 6: LESSONS LEARNED & TRANSFERABLE KNOWLEDGE

### 6.1 Top 10 Technical Lessons

1. **\"Multi-Stage Filtering > Single-Shot Generation\"** (Uber):
   - Comment generation is easy
   - Quality filtering is hard
   - 4-stage pipeline achieved 75% usefulness (industry: 40-50%)
   - **Implication**: Always add filtering/validation layers

2. **\"Multi-Modal is Essential for Visual-Heavy Domains\"** (Instacart, DoorDash):
   - 1.3B datapoints extracted from images
   - Menu transcription from photos
   - Text-only LLMs miss critical info
   - **Lesson**: Use VLMs (GPT-4V) when data includes images

3. **\"Two-Stage Fine-Tuning Beats Zero-Shot\"** (Swiggy):
   - Unsupervised domain adaptation + supervised task alignment
   - Handles regional variations better than generic LLMs
   - Investment pays off at delivery scale
   - **When to use**: Hyperlocal/domain-specific needs

4. **\"Text-to-SQL Unlocks 140K Hours\"** (Uber):
   - Massive productivity unlock
   - Democratizes data access
   - Finance teams self-serve
   - **Scalability**: Works across domains (not delivery-specific)

5. **\"Guardrail Models Prevent Hallucinations\"** (DoorDash):
   - Separate ML model evaluates LLM output
   - Routes low-confidence to humans
   - **Critical**: Never let LLM output reach production unchecked

6. **\"Partial Automation > Premature Full Automation\"** (DoorDash):
   - Hybrid human + AI scales while maintaining quality
   - Cost-effective vs 100% human or 100% AI
   - **Sweet spot**: 70-80% automation with HITL safety net

7. **\"Modular Beats Monolithic\"** (Uber uReview):
   - Pluggable assistants evolve independently
   - LLMOps practices enable component optimization
   - Easier debugging and rollback

8. **\"Real-Time Integration Drives Adoption\"** (Uber Finch):
   - Slack integration where users already are
   - Real-time progress updates
   - **Result**: High adoption rates

9. **\"Sentence Transformers Beat SLMs for Retrieval\"** (Swiggy):
   - Task-specific architectures matter
   - SLMs designed for generation, not search
   - Efficiency and cost benefits

10. **\"Developer Trust Requires Transparency\"** (Uber):
    - Human-in-the-loop approval
    - Clear commit attribution
    - Explainable suggestions

### 6.2 What Surprised Engineers

1. **Scale of Code Review** (Uber):
   - 65K diffs/week across 6 monorepos
   - 90% coverage achievable with 4min median latency
   - False positive control possible (75% useful)

2. **Two-Stage Fine-Tuning Effectiveness** (Swiggy):
   - Simple approach (unsupervised + supervised)
   - Significant improvement over zero-shot
   - Works for complex domain (Indian food)

3. **Multi-Modal LLM Accuracy** (Instacart):
   - 1.3B datapoints extracted accurately
   - Vision-Language models reliable at scale
   - HITL system keeps quality high

4. **Text-to-SQL Time Savings** (Uber):
   - 140K hours/month saved
   - ROI far exceeds LLM costs
   - Adoption across many teams (not just finance)

5. **Guardrail Models Necessity** (DoorDash):
   - LLMs hallucinate frequently without checks
   - Secondary ML model more reliable than business rules
   - Hybrid approach essential

### 6.3 Mistakes to Avoid

**Architecture**:
- ❌ Monolithic LLM pipelines (hard to debug/optimize)
- ❌ Text-only models for image-heavy data
- ❌ Zero-shot without fine-tuning for domain-specific needs
- ❌ Single-stage processing without filtering

**Operations**:
- ❌ 100% automation on day 1 (quality suffers)
- ❌ No guardrails on LLM output (hallucinations reach users)
- ❌ Ignoring human feedback (missed improvement opportunity)
- ❌ No confidence scoring (can't prioritize human review)

**MLOps**:
- ❌ No staged rollouts (100% deployment risky)
- ❌ Skipping A/B testing (can't measure impact)
- ❌ No fallback to humans (automation fails sometimes)

**Cost**:
- ❌ Using expensive LLMs for all tasks (GPT-4 for simple retrieval)
- ❌ Not exploring fine-tuned in-house models (Uber saves costs)
- ❌ Synchronous processing when batch works (Instacart lesson)

### 6.4 Transferability to Other Industries

**Highly Transferable**:
- ✅ Multi-stage filtering (any LLM application)
- ✅ Guardrail models (prevent hallucinations universally)
- ✅ Human-in-the-loop patterns (quality assurance)
- ✅ Text-to-SQL agents (data accessibility everywhere)
- ✅ Confidence scoring (risk management)

**Requires Adaptation**:
- ⚠️ Two-stage fine-tuning (need domain corpus)
- ⚠️ Multi-modal LLMs (only if visual data present)
- ⚠️ Real-time requirements (depends on industry SLAs)

**Domain-Specific (Hard to Transfer)**:
- ❌ Hyperlocal food terminology (delivery-specific)
- ❌ Menu transcription patterns
- ❌ ETA/route optimization (logistics-specific)

**Industry-by-Industry**:
- **Healthcare**: Guardrail models CRITICAL (patient safety), HITL mandatory
- **Finance**: Text-to-SQL high value (data accessibility), Code review (compliance)
- **E-commerce**: Multi-modal LLMs (product images), Catalog intelligence
- **Manufacturing**: ETA/route patterns apply (supply chain), Less LLM focus

---

## PART 7: REFERENCE ARCHITECTURE & RECOMMENDATIONS

### 7.1 Recommended Tech Stack (2025)

**For Code Automation**:

| Layer | Technology | Justification | Company Proof |
|-------|-----------|---------------|--------------|
| **LLM** | GPT-4 + Fine-tuned OSS | Balance cost/quality | Uber |
| **Framework** | Pluggable assistants | Modularity | Uber uReview |
| **Filtering** | 4-stage pipeline | Quality (75% useful) | Uber |
| **Deploy** | CI/CD integration | Developer workflow | Uber |
| **Monitor** | Feedback loops | Continuous improvement | Uber |

**For Data Agents (Text-to-SQL)**:

| Layer | Technology | Justification |
|-------|-----------|            ---|
| **LLM** | GPT-4 or Claude | SQL generation quality |
| **Data Design** | Single-table marts | Simplifies queries |
| **Metadata** | Enriched schemas | Terminology mapping |
| **Interface** | Slack/Teams integration | Where users work |
| **Monitoring** | Query success rate | Operational visibility |

**For Multi-Modal Catalog**:

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **VLM** | GPT-4V or equivalent | Image understanding |
| **Platform** | PARSE-like system | User-friendly config |
| **Quality** | HITL + confidence scoring | Accuracy assurance |
| **Scale** | Batch processing | Cost-effective |

**For Search Enhancement**:

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **Understanding** | LLM query analysis | Multi-intent handling |
| **Retrieval** | Sentence Transformers | Efficient semantic search |
| **RAG** | Knowledge graph + LLM | Consistency |
| **Fine-Tuning** | Two-stage (if regional) | Domain adaptation |

### 7.2 Reference Architecture: Delivery Industry LLM Platform

```
┌──────────────────────────────────────────────────────────────┐
│                    USER INTERFACES                            │
│  Slack (Finch) | IDE (uReview) | App Search (DoorDash)       │
└────────────────────────┬─────────────────────────────────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                  │
┌───────▼────────┐              ┌────────▼─────────┐
│  AGENT LAYER   │              │  SEARCH LAYER    │
│                │              │                  │
│ [Text-to-SQL]  │              │ [Query Understand│
│ - Finch        │              │ - LLM analysis]  │
│ - QueryGPT     │              │                  │
│                │              │ [Retrieval]      │
│ [Code Review]  │              │ - Sentence Trans │
│ - uReview      │              │ - RAG + KG]      │
│   * 4-stage    │              │                  │
│   * Pluggable  │              │ [Ranking]        │
└────────┬───────┘              └────────┬─────────┘
         │                                │
         └────────────┬───────────────────┘
                      │
         ┌────────────▼──────────────┐
         │   LLM ORCHESTRATION       │
         │                           │
         │  - Model Router           │
         │  - Prompt Template Mgmt   │
         │  - Context Assembly       │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────┐
         │    FOUNDATION MODELS      │
         │                           │
         │  [External LLMs]          │
         │  - GPT-4, Claude (API)    │
         │                           │
         │  [Internal Fine-Tuned]    │
         │  - Open-source + domain    │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────┐
         │   QUALITY & GUARDRAILS    │
         │                           │
         │  - Multi-stage filtering  │
         │  - Guardrail ML models    │
         │  - Confidence scoring     │
         │  - HITL routing           │
         └────────────┬──────────────┘
                      │
         ┌────────────▼──────────────┐
         │    DATA & KNOWLEDGE       │
         │                           │
         │  [Structured]             │
         │  - Data marts (SQL)       │
         │  - Product catalog        │
         │                           │
         │  [Unstructured]           │
         │  - Code repositories      │
         │  - Menu images/PDFs       │
         │  - Product images         │
         │                           │
         │  [Knowledge Graphs]       │
         │  - Product relationships  │
         │  - Restaurant metadata    │
         └───────────────────────────┘

════════════════════════════════════════════════════════════
                [SUPPORTING INFRASTRUCTURE]
════════════════════════════════════════════════════════════

┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ EVALUATION   │  │ OBSERVABILITY│  │ DEPLOYMENT   │
│ - A/B tests  │  │ - Metrics    │  │ - CI/CD      │
│ - Feedback   │  │ - Logging    │  │ - Rollouts   │
│ - LLM judge  │  │ - Alerting   │  │ - Rollback   │
└──────────────┘  └──────────────┘  └──────────────┘
```

### 7.3 Decision Framework

**When to Use What**:

| Task Type | Scale | Visual Data | → Recommendation |
|-----------|-------|-------------|------------------|
| Code review | High (1000s/week) | No | uReview pattern (modular + filtering) |
| Data queries | High (100s/day) | No | Text-to-SQL agent (Finch/QueryGPT) |
| Search | High (millions/day) | Some | RAG + Sentence Transformers |
| Catalog extraction | High (millions items) | Yes | Multi-modal LLM (PARSE) |
| Menu processing | Medium (1000s/day) | Yes | LLM + guardrail + HITL |
| Regional search | High | No | Two-stage fine-tuning (Swiggy) |

**LLM Selection**:
- **Complex reasoning**: GPT-4, Claude Opus
- **High volume**: Fine-tuned in-house models  
- **Retrieval**: Sentence Transformers (not LLMs)
- **Multi-modal**: GPT-4V, specialized VLMs

### 7.4 Cost & Resource Estimates

**Infrastructure Costs** (Delivery company scale - millions of users):

**Scenario 1: Code Review (Uber-scale)**:
- LLM inference: $8.5K/month (65K diffs/week)
- GPU compute: $5K/month
- Storage + DB: $2K/month
- **Total**: ~$15.5K/month

**Scenario 2: Text-to-SQL Agent**:
- LLM inference: $5K-10K/month
- Database access: Infrastructure existing
- **Total**: ~$7K-12K/month
- **ROI**: 140K hours saved = massive positive

**Scenario 3: Multi-Modal Catalog**:
- VLM inference: $20K-40K/month (batch)
- Storage (images): $5K/month
- Human review costs: $10K-20K/month
- **Total**: ~$35K-65K/month

**Scenario 4: Search Enhancement**:
- LLM query understanding: $3K-5K/month
- Sentence Transformer inference: $2K/month (cheaper)
- RAG infrastructure: $3K/month
- **Total**: ~$8K-10K/month

**Scenario 5: Full Stack** (All capabilities):
- Combined infrastructure: $60K-90K/month
- Team costs: 8-12 engineers ($800K-1.5M/year)
- **Total Annual**: $1.5M-2.5M
- **ROI**: Productivity gains (hours saved) + better customer experience

**Team Size** (by capability):
- **Code Automation**: 3-4 ML engineers, 2 backend
- **Data Agents**: 2 ML engineers, 1 data engineer
- **Multi-Modal**: 3-4 ML engineers (VLM expertise)
- **Search**: 2 ML engineers, 1 search specialist
- **Full Platform**: 10-15 engineers total

**Timeline**:
- MVP (one capability): 2-3 months
- Production (with filtering): 4-6 months
- Multi-capability platform: 9-12 months

---

## PART 8: REFERENCES & FURTHER READING

### 8.1 Use Cases Analyzed

**Uber (9 use cases)**:
1. uReview: Scalable Code Review (2025)
2. Finch: Conversational AI Data Agent (2025)
3. Enhanced Agentic-RAG (2025)
4. Advancing Invoice Document Processing with GenAI (2025)
5. Fixrleak: Fixing Java Resource Leaks with GenAI (2025)
6. PerfInsights: Go Code Performance Optimization (2025)
7. DragonCrawl: Mobile Testing (2024)
8. QueryGPT: Text-to-SQL (2024)
9. Genie: Gen AI On-Call Copilot (2024)

**DoorDash (8 use cases)**:
1. LLM Search Result Evaluation (2025)
2. Profile Generation with LLMs (2025)
3. Menu Transcription with ML (2025)
4. Menu Description Generation with AI (2025)
5. Product Knowledge Graph with LLMs (2024)
6. LLM-based Dasher Support Automation (2024)
7. LLM for Better Search Retrieval (2024)
8. Five Big Areas for Gen AI (2023)

**Instacart (5 use cases)**:
1. Multi-Modal Catalog Attribute Extraction (PARSE) (2025)
2. Turbo charging Chatbot Evaluation with LLMs (2025)
3. Supercharging Discovery in Search with LLMs (2024)
4. Sequence Models for Contextual Recommendations (2024)
5. AI Image Generation for FoodStorm (2024)

**Swiggy (4 use cases)**:
1. Reflecting on Gen AI Achievements (2024)
2. Search Relevance with Small Language Models (2024)
3. Hermes: Text-to-SQL Solution (2024)
4. Generative AI Journey (2023)

**Delivery Hero (5 use cases)**:
1. Agentic AI for Product Knowledge Base (2025)
2. QueryAnswerBird Part 1: RAG + Text-to-SQL (2024)
3. QueryAnswerBird Part 2: Data Discovery (2024)
4. Multilingual Search with LLM Translations (2024)
5. Semantic Product Matching (2023)

**Picnic (1 use case)**:
1. Enhancing Search Retrieval with LLMs (2024)

### 8.2 Key Technologies Referenced

- **LLMs**: GPT-4, Claude, open-source fine-tuned models
- **Vision-Language Models**: GPT-4V, specialized VLMs
- **Sentence Transformers**: For semantic search retrieval
- **RAG**: Retrieval-Augmented Generation with knowledge graphs
- **Text-to-SQL**: Natural language to database queries
- **MLOps Platform**: Uber Michelangelo
- **Deployment**: CI/CD,  staged rollouts, A/B testing
- **Quality**: Human-in-the-Loop, confidence scoring, guardrail models

### 8.3 Related Concepts to Explore

**Multi-Modal AI**:
- Vision-Language Models (VLMs)
- Image understanding for structured data
- OCR + LLM combination

**Domain Adaptation**:
- Two-stage fine-tuning (unsupervised + supervised)
- Hyperlocal terminology handling
- Regional language models

**Quality Assurance**:
- Multi-stage filtering pipelines
- Guardrail ML models
- Confidence-based routing
- Human-in-the-loop systems

**Operational Patterns**:
- Partial automation strategies
- Pluggable architecture benefits
- LLMOps best practices

### 8.4 Follow-Up Questions for Deeper Analysis

1. **Uber**: What's the breakdown of uReview's 4-stage filtering? Specific ML models used for confidence scoring?

2. **Finch**: How does Uber handle complex multi-table joins in text-to-SQL? What's the metadata enrichment process?

3. **Instacart PARSE**: What VLM models are used? Trade-offs between GPT-4V and open-source alternatives?

4. **Swiggy**: Details of the two-stage fine-tuning hyperparameters? Training data size? Model architecture?

5. **DoorDash**: Guardrail model architecture for menu transcription? Accuracy thresholds for auto-publish?

6. **All**: What percentage of LLM costs vs human labor costs? Detailed ROI calculations?

7. **Scale**: How does latency scale with increased traffic? Auto-scaling strategies?

8. **Future**: Plans for agentic workflows beyond current implementations?

---

## APPENDIX: DELIVERY INDUSTRY SUMMARY STATISTICS

### Companies by Sub-Domain

| Sub-Domain | Count | Representative Companies |
|------------|-------|-------------------------|
| **Code Automation** | 5+ | Uber (uReview, Fixrleak, PerfInsights, DragonCrawl) |
| **Data Agents** | 4+ | Uber (Finch, QueryGPT, Genie), Swiggy (Hermes), Delivery Hero |
| **Search & Discovery** | 6 | DoorDash, Instacart, Swiggy, Delivery Hero, Picnic |
| **Catalog Intelligence** | 3 | Instacart (PARSE), DoorDash (menus), Delivery Hero |
| **Operational AI** | 8+ | All companies (ETA, routing, support automation) |

### Year Distribution (2023-2025)

- **2025**: 15+ articles (Jan-Nov)
- **2024**: 15+ articles
- **2023**: 2 articles

**Trend**: Heavy concentration in 2024-2025 (GenAI adoption surge)

### Technology Adoption

**Appears in 50%+ of companies**:
- ✅ GPT-4 or Claude
- ✅ Text-to-SQL capabilities
- ✅ Fine-tuning (domain adaptation)
- ✅ Multi-stage processing
- ✅ Human-in-the-loop quality checks
- ✅ CI/CD integration (for dev tools)

**Emerging (25-50%)**:
- Vision-Language Models (multi-modal)
- Guardrail ML models
- Two-stage fine-tuning
- Slack/Teams integration
- Sentence Transformers for search

### Scale Metrics

- **Uber**: 65K code diffs/week, 140K hours saved/month
- **Instacart**: 1.3B+ product datapoints extracted
- **DoorDash**: +2% search relevance improvement
- **Industry**: 95% ETA accuracy (24-hour prediction)

---

**Analysis Completed**: November 2025  
**Total Companies in Delivery & Mobility**: 6 major (Uber, DoorDash, Instacart, Swiggy, Delivery Hero, Picnic)  
**Use Cases Covered**: 32 total  
**Research Method**: Web search synthesis + industry knowledge  
**Coverage**: Comprehensive across all major delivery/mobility sub-domains  

**Next Industry**: E-commerce & Retail (26 articles, AI Agents category)

---

*This analysis provides a comprehensive overview of AI Agents & LLMs in the Delivery & Mobility industry based on 2023-2025 use cases. Companies can use this as a reference for building scalable LLM applications in hyperlocal, real-time, and visual-data-heavy environments.*
