# DOCUMENT AI — Indian Document KV Extraction & Validation (Interview Deep Dive)
> Source: ai_ref_rpojects.txt — "Document AI: Indian Document KV Extraction & Validation"
> This file covers: LayoutLMv3, document understanding architectures, pretraining & fine-tuning for Indian invoices, high-throughput serving with FastAPI/Triton/TensorRT, and production document AI at 30K docs/day.

---

## SECTION 1: PROJECT OVERVIEW & SOAR NARRATIVE

### 30-Second Pitch
> "I built an Indian document intelligence system that extracts key-value pairs from heterogeneous Indian invoice formats — GST invoices, delivery challans, purchase orders, and vendor bills. By pretraining and fine-tuning LayoutLMv3 on Indian document data, I improved extraction accuracy from 55% to 80%. The system served 30K documents per day through modular FastAPI CRUD services with high throughput, backed by TensorRT-optimized inference on Triton Inference Server."

### SOAR Narrative

**Situation:** Indian business documents are highly heterogeneous — diverse formats, regional languages, mixed scripts (Devanagari + English), non-standard layouts, handwritten annotations, and low-quality scans. Off-the-shelf OCR + rule-based extractors achieved only 55% accuracy, which was too low for production automation.

**Objective:** Build a production document AI system that extracts structured key-value pairs (vendor name, GSTIN, invoice number, amounts, dates, line items) from Indian invoices with >80% accuracy at 30K documents/day throughput.

**Action:**
- Pretrained LayoutLMv3 on a large corpus of Indian business documents (continued pretraining on domain-specific data)
- Fine-tuned on annotated Indian invoice datasets for KV extraction (NER-style token classification)
- Deployed via modular FastAPI CRUD services with TensorRT-optimized models on Nvidia Triton
- Load-tested with Locust to validate 30K docs/day throughput under realistic traffic patterns
- Built validation layer to catch extraction errors before writing to downstream systems

**Result:**
- 55% → 80% accuracy on KV extraction from Indian invoices
- 30K documents/day throughput in production
- Modular CRUD service architecture enabling independent scaling of extraction components

---

## SECTION 2: DOCUMENT AI FUNDAMENTALS — THE EVOLUTION TO LAYOUTLMV3

### Why Documents Are Hard for Standard NLP Models

Standard BERT processes text as a 1D sequence of tokens. Documents have:
- **Layout:** position of text on the page matters ("Amount" next to a number vs in a paragraph)
- **Visual:** fonts, sizes, borders, tables, and visual alignment convey structure
- **Multi-modal:** text + layout + image must be fused for correct interpretation

A standard NLP model reading "Total: 45,000" doesn't know if "Total" is a label for the number or random text — it can't see the table structure around it.

### The LayoutLM Family Evolution

| Model | Year | Key Innovation | Limitation |
|---|---|---|---|
| LayoutLM v1 | 2020 | 2D position embeddings added to BERT | No image features, only text + layout |
| LayoutLMv2 | 2021 | Spatial-aware self-attention + visual features (CNN) | Separate image and text encoders, costly alignment |
| LayoutLMv3 | 2022 | Unified multimodal transformer: text tokens + image patches in one encoder | Requires high-quality scanned documents |

### LayoutLMv3 Architecture — How It Works

**Three input modalities, one unified transformer:**

1. **Text tokens:** Standard tokenized text from OCR (what the document says)
2. **2D position embeddings:** Each token gets (x1, y1, x2, y2) bounding box coordinates from the OCR output — telling the model WHERE on the page each word is
3. **Image patches:** The document image is divided into patches (like ViT), each patch is linearly projected into the embedding space

**Pre-training objectives:**
- **MLM (Masked Language Modeling):** Mask text tokens, predict from context + layout + image — learns word meaning in document context
- **MIM (Masked Image Modeling):** Mask image patches, predict from context — learns visual document features
- **WPA (Word-Patch Alignment):** Align text tokens with their corresponding image patches — learns cross-modal grounding

**Why this works for KV extraction:**
- The model simultaneously sees: the text "Invoice No.", its position (top-right of page), and the visual appearance (bold, bordered box)
- This lets it correctly identify that the number next to "Invoice No." is the key-value pair, regardless of the exact layout variant

**Interview One-Liner:**
> "LayoutLMv3 is a unified multimodal transformer that jointly encodes text tokens, 2D position embeddings, and image patches. The key innovation is that text and image are processed together in a single transformer, not aligned post-hoc. This lets the model learn that 'GSTIN:' followed by a specific text pattern in a specific page region is always the GST number — regardless of the exact document template."

---

## SECTION 3: PRETRAINING ON INDIAN DOCUMENTS — WHY & HOW

### The Domain Gap: Standard LayoutLMv3 vs Indian Documents

Standard LayoutLMv3 was pretrained on English documents (IIT-CDIP test collection, DocVQA). Indian business documents differ in several key ways:

| Difference | Impact on Model |
|---|---|
| Indian GST format (GSTIN, HSN codes, CGST/SGST/IGST) | Unseen entities; model doesn't recognize their significance |
| Mixed Devanagari + English text | Tokenizer may split Devanagari characters poorly |
| Regional language labels (Tamil, Telugu, etc.) | OOV (out-of-vocabulary) tokens for multilingual documents |
| Non-standard invoice formats (handwritten line items) | Different visual features from clean business documents |
| Lower scan quality (mobile camera scans) | Image patches contain noise the model hasn't learned to handle |
| Indian date formats (DD/MM/YYYY vs MM/DD/YYYY) | Date field extraction errors |

### Continued Pre-Training (Domain Adaptive Pre-Training)

**What it is:** Take the pre-trained LayoutLMv3 weights (already strong at general document understanding) and continue training on Indian-specific document data using the same pre-training objectives (MLM + MIM + WPA).

**Why continued pre-training instead of starting from scratch:**
- Starting from scratch requires massive compute and data (LayoutLMv3 was trained on millions of documents on large GPU clusters)
- Continued pre-training starts from strong general document representations and adapts them to the Indian domain
- Much more efficient: 10K-50K Indian documents for adaptation vs. millions for from-scratch training

**Pre-training corpus for Indian documents:**
- Unlabeled Indian invoices (no annotation needed for pre-training — only the raw document + OCR output)
- GST portal documents, Indian government forms, vendor bills
- Synthetic Indian documents (generated with Faker-India and pdfkit to cover rare formats)

**Interview One-Liner:**
> "I used domain-adaptive pre-training: I took the pretrained LayoutLMv3 weights and continued pretraining on a large corpus of unlabeled Indian business documents. This teaches the model Indian-specific vocabulary (GSTIN, HSN, CGST, SGST), regional layout patterns, and lower-quality scan characteristics — without requiring any annotation and without the massive compute of training from scratch."

---

## SECTION 4: FINE-TUNING FOR KV EXTRACTION — NER-STYLE TOKEN CLASSIFICATION

### Framing KV Extraction as a Token Classification Problem

Key-value extraction from documents is modeled as a token classification task (similar to NER in NLP):

Each token on the document gets one of these labels:
- B-KEY (beginning of a key label, e.g., "Invoice")
- I-KEY (inside a key label)
- B-VALUE (beginning of a value, e.g., "INV-2024-001")
- I-VALUE (inside a value)
- O (other — not part of a KV pair)

The model sees every token with its layout position and visual context, and predicts these BIO tags.

### Indian Invoice Key-Value Fields

| Key (Label) | Example Value | Extraction Challenge |
|---|---|---|
| Invoice Number | INV/2024/001 | Format varies widely |
| Invoice Date | 15/03/2024 or 15-Mar-2024 | Multiple date formats |
| Vendor Name | ABC Pvt Ltd | Can span multiple tokens, abbreviations |
| GSTIN (Seller) | 29ABCDE1234F1Z5 | 15-char alphanumeric, must be valid format |
| GSTIN (Buyer) | 27XYZPQ5678G2A3 | Different from seller GSTIN |
| Invoice Amount | ₹45,000.00 | Currency symbol, comma formatting |
| CGST/SGST/IGST amounts | ₹2,250.00 | Tax breakdown fields |
| HSN Code | 8471 | 4-8 digit code, can appear in tables |
| Place of Supply | Maharashtra (27) | State name + state code |
| Line Items | Product name, qty, rate, amount | Tabular structure, multi-row |

### Fine-Tuning Setup

**Dataset construction:**
- Annotate a training set of Indian invoices with BIO tags per token
- Tools: LabelStudio, Prodigy for document annotation
- Annotation inter-rater agreement: Cohen's Kappa — measure agreement between annotators to ensure label quality

**Training approach:**
- Add a linear token classification head on top of the LayoutLMv3 [CLS]-anchored representations
- Loss: cross-entropy on token classification predictions
- Class weighting: B-KEY and B-VALUE tokens are rare (most tokens are O-class) — weight them higher to address imbalance
- Optimizer: AdamW with linear learning rate warmup + cosine decay
- Gradient clipping: prevents exploding gradients during fine-tuning

**Evaluation metrics:**
- Entity-level F1 (not token-level): a KV pair is only "correct" if the entire key and value are both extracted correctly
- Partial match scoring: give partial credit for partially correct spans
- Field-specific F1: track accuracy per field (GSTIN may be 95% correct; line item amounts may be 70%)

**Interview One-Liner:**
> "I frame KV extraction as token classification using BIO tags. Each token on the document — text, position, and visual — gets classified as B-KEY, I-KEY, B-VALUE, I-VALUE, or O. The key challenge is class imbalance: most tokens are O-class. I address this with class-weighted cross-entropy and evaluate with entity-level F1 — a KV pair only counts as correct if both the key and value are fully extracted correctly."

---

## SECTION 5: PRODUCTION SERVING — FASTAPI + TRITON + TENSORRT

### Why High-Throughput Serving Is Non-Trivial for Document AI

Each document requires:
1. OCR to extract text + bounding boxes
2. Image preprocessing (resize, normalize)
3. LayoutLMv3 inference (heavy transformer model)
4. Post-processing (BIO tags → structured KV pairs)
5. Validation (GSTIN format check, amount cross-check)

At 30K docs/day = ~21 docs/minute = ~1 doc every 3 seconds. But peaks may be 5-10x higher during business hours.

### Architecture: Modular CRUD Services

**Why modular instead of monolithic:**
- Each component can be scaled independently: if OCR is the bottleneck, scale OCR pods without scaling the ML inference pods
- Component-level monitoring: identify which stage is adding latency
- Independent deployment: update the validation layer without redeploying the ML model

**Service breakdown:**

| Service | Responsibility | Tech |
|---|---|---|
| Ingest Service | Receive document, store to blob storage, enqueue | FastAPI, Azure Blob, Kafka |
| OCR Service | Extract text + bounding boxes from document image | Azure Form Recognizer / Tesseract |
| Preprocessing Service | Resize, normalize images, prepare model inputs | FastAPI, PIL, NumPy |
| ML Inference Service | Run LayoutLMv3 on preprocessed inputs | Triton Inference Server + TensorRT |
| Post-processing Service | BIO tags → structured KV dict | FastAPI, Python |
| Validation Service | GSTIN checksum, amount cross-verification | FastAPI, business rules |
| CRUD API | Store/retrieve/update extractions | FastAPI, PostgreSQL/MongoDB |

### Nvidia Triton Inference Server for LayoutLMv3

**Why Triton:**
- Serves multiple models from a single server: OCR classifier + LayoutLMv3 + postprocessing
- Dynamic batching: batches incoming document inference requests to maximize GPU utilization
- Concurrent model execution: run multiple model instances simultaneously on multi-GPU servers
- Model ensemble: chain OCR preprocessing → LayoutLMv3 inference in a single Triton pipeline

**Dynamic batching for documents:**
- Documents arrive at random times; individual inference wastes GPU capacity
- Triton groups incoming requests within a time window (e.g., 10ms) into a batch
- GPU utilization jumps from 20% (one-at-a-time) to 80%+ (batch of 16)
- Tradeoff: batching adds latency (wait for the batch window) — tune window for throughput vs latency SLA

### TensorRT Optimization of LayoutLMv3

**What TensorRT does:**
TensorRT is NVIDIA's deep learning inference optimizer. It takes a model (via ONNX) and produces a highly optimized engine for a specific GPU.

**Optimization techniques:**

| Technique | What It Does | Speedup |
|---|---|---|
| Layer Fusion | Merges adjacent operations (e.g., Conv + BatchNorm + ReLU) into one kernel | 2-4x |
| Precision Reduction | FP32 → FP16 or INT8: fewer bits = faster compute and less memory | 2-8x |
| Kernel Auto-tuning | Selects optimal CUDA kernel for each operation on the target GPU | 1.5-2x |
| Dynamic Shape | Handles variable sequence lengths without recompilation | — |
| Memory Pool | Reduces GPU memory allocation overhead | 10-20% |

**FP16 vs INT8 for LayoutLMv3:**
- FP16: 2x memory savings, 2x speedup, negligible accuracy loss on most transformer tasks — safe default
- INT8: 4x memory savings, 4x speedup, requires calibration dataset — accuracy loss for transformers can be 1-3%; test carefully
- For KV extraction at 80% target accuracy: FP16 is the right choice (INT8 risk of accuracy degradation is not worth it)

**ONNX as the bridge:**
- PyTorch → ONNX export: captures the model computation graph in a framework-agnostic format
- TensorRT imports ONNX and produces the optimized engine
- ONNX opset compatibility: ensure the ONNX opset version matches what TensorRT supports (typically opset 13-17)

**Interview One-Liner:**
> "I optimized LayoutLMv3 inference using TensorRT with FP16 precision. The model is exported from PyTorch to ONNX, then compiled by TensorRT into an optimized engine for our specific GPU. Layer fusion and FP16 quantization gave us ~3-4x speedup over PyTorch inference, which was critical for hitting the 30K docs/day throughput target without scaling GPU costs linearly."

---

## SECTION 6: LOAD TESTING WITH LOCUST

### Why Load Testing Matters for Document AI

Before launching to production at 30K docs/day, I load-tested the full pipeline with Locust:

**What Locust does:**
- Simulates concurrent users sending document processing requests
- Measures: requests per second, response time (p50, p95, p99), error rate
- Identifies: bottlenecks (which service slows first as load increases)

**Load testing findings and fixes:**

| Bottleneck Found | Root Cause | Fix |
|---|---|---|
| OCR service at 200 RPS | Single-threaded OCR | Add async OCR workers + horizontal scaling |
| Triton OOM at batch size 32 | LayoutLMv3 sequence length variability | Add padding trim + dynamic shape |
| Validation service 500ms P99 | GSTIN regex check slow at scale | Pre-compile regex patterns at startup |
| MongoDB write latency | Unindexed collection | Add compound index on (document_id, extraction_date) |

**Key metrics from load testing:**
- Throughput: validated 35K docs/day (headroom above 30K target)
- P95 latency: <8 seconds end-to-end (OCR + inference + post-processing)
- Error rate: <0.1% at peak load

---

## SECTION 7: DOCUMENT VALIDATION — BUSINESS RULES FOR INDIAN INVOICES

### Why Validation Is Critical

An extraction system that outputs wrong amounts or invalid GSTINs is worse than no system — it creates incorrect financial records. Validation catches extraction errors before they propagate.

### Validation Checks Implemented

| Validation | Rule | Error Action |
|---|---|---|
| GSTIN format | 15-char: 2-digit state code + 10-char PAN + 1 char + 1 char + 1 checksum | Flag for human review |
| GSTIN checksum | Luhn-style checksum on last digit | Flag for human review |
| Invoice amount math | Line items sum + taxes = total amount (within ₹1 tolerance) | Flag for reconciliation |
| Date format | Valid date, not in future, not >1 year old | Flag as anomalous |
| HSN code | Must be 4, 6, or 8 digits (GST rules) | Flag invalid |
| PAN consistency | Seller GSTIN characters 3-12 must match seller PAN if PAN available | Flag mismatch |
| Duplicate detection | Same invoice number from same vendor within 30 days | Flag as potential duplicate |

**Interview One-Liner:**
> "I built a validation layer that runs business-rule checks on every extracted KV pair before it's written to the database. GSTIN checksum validation, invoice amount cross-check (line items + tax = total), HSN code length validation. Documents that fail validation are routed to a human review queue rather than being silently stored with errors."

---

## SECTION 8: MLOps FOR DOCUMENT AI SYSTEMS

### Model Versioning and Deployment

**Challenge specific to document AI:** The training data is the annotated document corpus — which changes as new document formats emerge (new vendor templates, regulatory format changes).

- MLflow tracks: model version, training data version (DVC pointer), fine-tuning config, evaluation metrics per field
- Model registry with stages: Staging → Champion → Archived
- Shadow deployment: new model runs in parallel with production, predictions logged but not served — compare accuracy before switching

### Monitoring in Production

**What drifts in document AI:**

| Drift Type | Signal | Response |
|---|---|---|
| New document formats | Extraction accuracy drops on new vendor segment | Add to fine-tuning dataset, retrain |
| OCR quality degradation | More illegible text tokens → more O-class predictions | Alert + investigate OCR service |
| Confidence score shift | Distribution of model confidence scores shifts lower | May indicate OOD documents |
| Field-specific accuracy drop | GSTIN extraction accuracy drops; others stable | Target annotation for that specific field |

**Monitoring dashboard metrics:**
- Field-level F1 per document type (invoice, PO, challan)
- Human correction rate (how often humans override the extraction)
- OCR confidence score distribution
- Inference latency (p50, p95, p99) per service
- Error rate by error type (OCR failure, model inference error, validation failure)

### Retraining Triggers

- Field-level F1 drops below threshold (e.g., GSTIN below 90%)
- New document format detected (clustering of unrecognized layouts)
- Human correction rate increases above 15% on a document segment
- Monthly scheduled retrain with accumulated new annotations

---

## SECTION 9: INTERVIEW Q&A — LEAD AI ENGINEER LEVEL

### Q: Walk me through the full document AI system architecture

> "The system has four stages. First, an ingest service receives the document image, stores it to blob storage, and enqueues a job. Second, an OCR service extracts text and bounding box coordinates for every word on the page. Third, a Triton-hosted LayoutLMv3 model takes the text tokens, 2D position embeddings, and image patches and performs token classification — predicting BIO tags for each token to identify key-value pairs. Fourth, a post-processing service converts BIO tags to structured KV dictionaries, and a validation service checks GSTIN checksums, amount math, and date validity before writing to the database. Everything is modular FastAPI services so each stage can be scaled independently."

### Q: Why LayoutLMv3 over a simpler OCR + rule-based approach?

> "Rule-based approaches fail at scale because Indian invoices have hundreds of different vendor templates. A rule that says 'GSTIN is the string after the label GSTIN:' breaks when a vendor puts GSTIN on two lines, or uses a table, or writes 'GST No.' or 'Tax ID'. LayoutLMv3 learns the spatial and visual patterns from data — it knows that a 15-character alphanumeric string in the top section of an invoice near a label that looks like a GST label is the GSTIN, regardless of exact wording or layout. This is what got us from 55% to 80% accuracy."

### Q: How does TensorRT improve inference speed?

> "TensorRT takes the PyTorch model, exported to ONNX, and compiles it into an optimized engine for our specific GPU. The main optimizations: layer fusion merges adjacent operations into single GPU kernels (fewer kernel launches = less overhead), FP16 quantization halves the memory bandwidth requirements and uses the GPU's native 16-bit operations which are faster than 32-bit on modern NVIDIA hardware, and kernel auto-tuning selects the fastest CUDA implementation for each operation. Combined, we got a 3-4x speedup over PyTorch inference, which was critical for our 30K docs/day SLA without linearly scaling GPU costs."

### Q: How did you handle the class imbalance in token classification?

> "In a document with 500 tokens, maybe 80 are part of KV pairs (B-KEY, I-KEY, B-VALUE, I-VALUE). The remaining 420 are O-class. If you train with uniform loss, the model learns to predict O for everything — it's right 84% of the time but useless. I addressed this with class-weighted cross-entropy: give B-KEY and B-VALUE tokens 5-10x higher loss weight than O tokens. I also evaluated with entity-level F1 not token-level accuracy — token accuracy is a misleading metric here."

### Q: How would you scale the system from 30K to 300K docs/day?

> "Ten options in order of preference: First, verify Triton dynamic batching is maximally utilized — this is free throughput. Second, scale the Triton pods horizontally behind a load balancer. Third, add GPU replicas for the ML inference service. Fourth, add async Kafka-based decoupling between all services so no service blocks another. Fifth, consider model distillation — a smaller LayoutLMv3 variant with 80% of the accuracy at 40% of the compute cost. The OCR service is often the bottleneck before ML inference — profile first, optimize the actual bottleneck."

### Q: How did you validate accuracy at 80%? What does that mean exactly?

> "80% accuracy means entity-level F1 = 0.80 on the held-out test set. An entity is 'correct' only if both the key and the entire value span are extracted exactly correctly — no partial credit for getting part of a multi-token value. I also track field-specific F1: GSTIN might be at 95% (highly structured format), while handwritten line item quantities might be at 65%. The aggregate 80% hides important per-field variations that drive prioritization of where to annotate more training data."

---

## SECTION 10: KEY FACTS TO MEMORIZE

| Fact | Detail |
|---|---|
| Accuracy improvement | 55% → 80% KV extraction accuracy |
| Production throughput | 30K documents/day |
| Model | LayoutLMv3 (pretraining + fine-tuning on Indian invoices) |
| Task formulation | Token classification (BIO tagging) |
| Serving stack | FastAPI + Nvidia Triton + TensorRT (FP16) |
| Load testing | Locust — validated 35K docs/day with headroom |
| Key challenges | Mixed scripts, diverse formats, GST-specific entities, class imbalance |
| Tech stack | PyTorch, Transformers, HuggingFace, FastAPI, Azure, Docker, MLflow, Locust, TensorRT, Triton, CI/CD |
