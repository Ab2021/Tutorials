# Day 73: Multi-Modal RAG & Agents
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How does CLIP work?

**Answer:**
- **Contrastive Language-Image Pre-training.**
- Trains two encoders (Text Encoder, Image Encoder).
- **Objective:** Maximize cosine similarity between correct (Image, Text) pairs and minimize it for incorrect pairs in a batch.
- **Result:** A shared embedding space where "A photo of a dog" and an image of a dog are close vectors.

#### Q2: What is the difference between "Early Fusion" and "Late Fusion" in Multi-modal models?

**Answer:**
- **Early Fusion:** Concatenate image features and text embeddings *before* the transformer layers. (e.g., GPT-4V). The model processes them together.
- **Late Fusion:** Process image and text separately, then combine the outputs (e.g., CLIP retrieval).
- **Trend:** Early fusion (native multi-modal) is winning for reasoning tasks.

#### Q3: Why is "Text-to-Image Retrieval" hard for charts/graphs?

**Answer:**
- CLIP is trained on natural images (dogs, cats). It struggles with abstract data like "Bar chart showing 20% growth".
- **Solution:** Use a VLM (GPT-4o) to generate a text *caption* or *table* from the chart first. Then index the text.

#### Q4: What are the privacy implications of Screen Agents?

**Answer:**
- **Risk:** The agent "sees" everything on your screen (passwords, private chats).
- **Mitigation:**
  - **PII Masking:** Blur sensitive areas before sending to cloud.
  - **Local Models:** Run the vision model locally (e.g., LLaVA) so data never leaves the device.

#### Q5: How do you handle the cost of Multi-modal RAG?

**Answer:**
- **Problem:** Sending 10 images to GPT-4o per query is expensive.
- **Solution:**
  - **Thumbnailing:** Resize images to lower resolution.
  - **Two-Stage:** Retrieve 10 images, use a cheap model (LLaVA) to filter to top 2, send top 2 to GPT-4o.

---

### Production Challenges

#### Challenge 1: The "Unreadable PDF"

**Scenario:** PDF contains scanned images of text. Standard text extractors return empty strings.
**Root Cause:** No OCR.
**Solution:**
- **OCR Pipeline:** Use Tesseract or Amazon Textract to convert image-based PDFs to text.
- **VLM:** Pass the page image to GPT-4o (expensive but accurate).

#### Challenge 2: Image-Text Misalignment

**Scenario:** User searches "Red car". System retrieves a "Blue car" because the text description was generic ("A nice car").
**Root Cause:** Relying only on text captions.
**Solution:**
- **Hybrid Search:** Search both Image Embeddings (CLIP) and Text Captions (BM25). Combine scores.

#### Challenge 3: Latency of Vision Models

**Scenario:** Screen Agent takes 5 seconds to process a screenshot. User has already moved the mouse.
**Root Cause:** Network latency + Inference time.
**Solution:**
- **Fast Models:** Use specialized small vision models (Moondream).
- **Streaming:** Stream the action tokens.
- **Optimistic Execution:** Predict the next click while processing.

#### Challenge 4: Hallucination in Visual Reasoning

**Scenario:** User asks "What is the value in row 3?" Model says "50" but it's "500".
**Root Cause:** VLMs struggle with precise OCR/spatial grounding.
**Solution:**
- **Set-of-Mark Prompting:** Overlay the image with numbered boxes/grids to help the model reference specific areas.

#### Challenge 5: Storage Bloat

**Scenario:** Storing 1M images + vectors costs a fortune.
**Root Cause:** Storing full-res images.
**Solution:**
- **Reference:** Store images in S3 (cheap), store only vectors/metadata in Vector DB (expensive).
- **Compression:** Store WebP thumbnails for preview.

### System Design Scenario: Insurance Claim Processor

**Requirement:** Process photos of car accidents and estimate damage.
**Design:**
1.  **Ingest:** User uploads photos.
2.  **Check:** Use VLM to verify it's a car and not a cat.
3.  **Retrieve:** Search Vector DB for similar past accidents (Image-to-Image search).
4.  **Estimate:** Pass Photo + Similar Past Claims (Cost) to GPT-4o. "Given this damage and these past examples, estimate repair cost."
5.  **Output:** Report with confidence interval.

### Summary Checklist for Production
- [ ] **Ingestion:** Use **OCR** for scanned docs.
- [ ] **Indexing:** Use **Hybrid (CLIP + Caption)** indexing.
- [ ] **Model:** Use **GPT-4o** for complex visual reasoning.
- [ ] **Cost:** **Resize images** before sending to API.
- [ ] **Privacy:** **Mask PII** in screenshots.
