# AI-SPECIFIC DEEP DIVES — Master Index
> This directory covers advanced AI/ML architectures and production systems from ai_ref_rpojects.txt.
> These topics are **NOT covered** in Revision_SystemDesign_MLOPS_engineering — they are additive.
> Read these files when interviewing for AI Engineer / Applied Scientist / Vision AI / Clinical AI / GenAI roles.

---

## FILE MAP

| File | Domain | Key Topics | Must Read For |
|---|---|---|---|
| 01_computer_vision_video_analytics.md | CV / Video AI | YOLO, EfficientNet, DeepStream, TensorRT, Synthetic Data (DINOv2+ControlNet), 200-camera PPE deployment | CV interviews, edge AI, industrial AI |
| 02_clinical_ai_healthcare_nlp.md | Clinical NLP / Healthcare AI | ConcertBERT, NER/NERC/ABSA/Assertion/RE, SapBERT, LoRA/QLoRA, Flash Attention, Paged Attention, oncology AI | Healthcare AI, clinical NLP, LLM fine-tuning interviews |
| 03_llm_agentic_jio_langraph.md | LLM / Agentic AI | LangGraph, LangFuse evaluation, CLIP fine-tuning, CoT reasoning, sampling methods (Top-P, temperature), 10K+ daily query evaluation | GenAI, Agentic AI, LLM platform interviews |
| 04_document_ai_layoutlmv3.md | Document AI | LayoutLMv3, KV extraction, Indian invoice domain adaptation, Triton/TensorRT serving, Locust load testing | Document AI, OCR+ML, enterprise document processing interviews |

---

## WHAT THESE FILES COVER THAT IS NOT IN REVISION_SYSTEMDESIGN_MLOPS_ENGINEERING

The main revision folder covers: MLOps, system design, fraud analytics, algorithms, behavioral questions.

These ai_specific files fill these unique gaps:

### Gap 1: Computer Vision Architectures
- YOLO internals (anchor boxes, grid cells, loss functions, NMS variants)
- Two-stage vs one-stage detectors (RCNN family vs YOLO family)
- EfficientNet multi-head classification architecture
- Synthetic data generation using DINOv2 + ControlNet-style diffusion
- NVIDIA DeepStream pipeline (GStreamer, NVDEC, multi-camera)
- TensorRT: layer fusion, INT8/FP16 quantization, calibration datasets
- NVIDIA Triton: dynamic batching, concurrent model execution, ensemble models
- Scaling from 200 to 10,000 cameras: Kafka event streaming, edge vs cloud inference

### Gap 2: Clinical NLP & SLM Fine-Tuning
- Clinical domain pre-training (why general BERT fails for EHR notes)
- NER/NERC with BIO tagging for clinical entities
- Assertion classification (Confirmatory: Affirmation/Negation/Uncertainty; Temporal; Subject)
- Aspect-Based Sentiment Analysis (ABSA) in clinical context
- Relation Extraction (Drug-treats-Disease, Gene-associated-with-Tumor)
- SapBERT for biomedical entity linking (synonym-aware embedding)
- Two-stage retrieval: Bi-Encoder (recall) → Cross-Encoder (precision)
- LoRA/QLoRA/PEFT for parameter-efficient LLM fine-tuning
- Custom loss functions: triplet loss, InfoNCE, Circle loss, self-alignment loss
- Flash Attention: IO-aware exact attention (2-4x speedup for long sequences)
- Paged Attention (vLLM): KV cache memory management
- Chain-of-Thought for clinical tumor progression reasoning
- RECIST criteria and clinical reasoning
- Visio-linguistic models (CLIP, LLaVA) for medical report analysis

### Gap 3: LLM Agentic Platform (Jio Scale)
- LangGraph StateGraph with conditional routing for multi-service agents
- LangFuse evaluation at 10K+ daily queries: scoring dimensions, CI/CD gates
- CLIP fine-tuning with contrastive loss and hard negative mining
- Recall@1, Recall@K evaluation for visual search
- Chain-of-Thought (zero-shot, few-shot, self-consistency, Tree of Thought)
- Sampling strategy: temperature, Top-K, Top-P (nucleus sampling) and when to use each
- Idempotency keys for safe tool call retries at scale
- Production differences: temperature near zero for tool selection vs 0.4-0.7 for generation

### Gap 4: Document AI (LayoutLMv3)
- LayoutLM family evolution (v1 → v2 → v3): what each version added
- LayoutLMv3 architecture: text tokens + 2D position embeddings + image patches in one unified transformer
- Pre-training objectives: MLM + MIM + WPA (Word-Patch Alignment)
- Continued pre-training (domain adaptive) vs full pre-training
- Token classification with BIO tagging for KV extraction
- Entity-level F1 vs token-level accuracy (why token accuracy is misleading)
- Class imbalance handling in token classification (class-weighted cross-entropy)
- GSTIN validation: checksum, format rules
- Locust load testing: finding bottlenecks before production

---

## KEY NUMBERS TO MEMORIZE FROM AI_REF_RPOJECTS.TXT

### Computer Vision — Jamnagar PPE System
- 200+ cameras deployed
- 45 FPS throughput
- ₹93 Cr/year cost savings
- Scalable to 10K cameras
- Won: GULF Energy Information Excellence Awards 2024
- PPE types: helmet, gloves, PVC suit, IFR suit

### Document AI — Indian Invoice KV Extraction
- 55% → 80% extraction accuracy (via LayoutLMv3 pretraining + fine-tuning)
- 30K documents/day throughput
- Modular FastAPI CRUD services
- TensorRT + Triton serving

### Jio LLM / Agentic AI
- 10K+ daily use-case-specific queries evaluated via LangFuse
- CLIP recall@1: 25% → 56% (fine-tuned)
- Services: JioMart grocery ordering, cab booking

### Clinical AI / Healthcare
- Models: ConcertBERT (pretrained), + fine-tuned suite (NER, NERC, ABSA, Assertion, RE, Entity Linking)
- LLMs: Llama 3.1 (entity extraction), GPT-4 (protocol abstraction), SapBERT (entity linking)
- GPU optimization: Flash Attention + Paged Attention + optimal batching
- Applied over cancer patients data to assist Oncologists

---

## HOW TO USE THESE FILES IN INTERVIEWS

### If asked about Computer Vision:
1. Open with: "I deployed a PPE safety system across 200 cameras at Reliance Jamnagar at 45 FPS..."
2. Use SOAR structure, cite the ₹93 Cr impact
3. Explain architecture: YOLO (detection) → EfficientNet multi-head (attribute classification) → DeepStream pipeline → TensorRT on Triton
4. Defensive answers: justify YOLO over Faster RCNN (speed vs accuracy tradeoff), justify synthetic data (rare class problem)

### If asked about Clinical AI or Healthcare NLP:
1. Open with: "I built a clinical NLP pipeline for oncology — extracting entities, relations, and assertions from unstructured EHR notes to support oncologist decision-making..."
2. Cover the SLM suite (NER, assertion, RE) before mentioning LLMs — shows you know when SLMs are sufficient
3. Flash Attention and Paged Attention are strong signals of GPU optimization experience

### If asked about Agentic AI (beyond Axtria/Chubb):
1. Reference Jio: "At Jio, I built agentic workflows on LangGraph for JioMart ordering and cab booking, evaluated at 10K+ daily queries via LangFuse..."
2. CLIP fine-tuning shows multimodal experience
3. CoT and sampling research shows theoretical depth in LLM reasoning

### If asked about Document AI:
1. LayoutLMv3 is a strong, specific answer to "how do you handle document understanding"
2. The 55%→80% improvement from domain adaptive pretraining shows depth in transfer learning
3. GSTIN validation shows you think end-to-end: model accuracy is not the only thing that matters

---

## QUICK REFERENCE: ARCHITECTURAL DECISIONS & RATIONALE

| Decision | Alternative Considered | Why This Choice |
|---|---|---|
| YOLO over Faster RCNN | Faster RCNN (two-stage, more accurate on small objects) | 45 FPS real-time requirement; YOLO's one-stage is 5-10x faster at inference |
| EfficientNet multi-head over separate models | Train one model per attribute | Shared backbone amortizes compute; joint training improves features |
| Synthetic data (DINOv2+ControlNet) over data collection | Collect more real violation images | Safety violations are rare by design; impossible to collect sufficient real examples |
| TensorRT FP16 over INT8 | INT8 for higher speedup | INT8 accuracy degradation too risky for safety-critical detection |
| LayoutLMv3 over OCR+rules | Rule-based extraction | Indian invoice diversity makes rules brittle; 200+ vendor templates |
| Continued pre-training over fine-tuning only | Fine-tune directly on labeled data | Domain vocabulary gap (GSTIN, HSN) requires pre-training to learn representations |
| SapBERT Bi-Encoder + Cross-Encoder over just Cross-Encoder | Cross-Encoder only for entity linking | Cross-Encoder over full corpus is too slow; Bi-Encoder first for candidate retrieval |
| LangGraph StateGraph over ReAct | ReAct for Jio agents | Service workflows have known optimal paths; state machines are testable and auditable |
| Self-consistency CoT over standard CoT | Standard CoT | Eliminates random reasoning errors without requiring human review or additional training |
