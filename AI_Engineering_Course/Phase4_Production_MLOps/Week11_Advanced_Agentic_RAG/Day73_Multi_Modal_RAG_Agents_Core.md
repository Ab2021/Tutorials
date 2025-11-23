# Day 73: Multi-Modal RAG & Agents
## Core Concepts & Theory

### Beyond Text

**Reality:** The world is visual. Documents have charts. Users send screenshots.
**Goal:** Build systems that understand Text + Images + Audio.

### 1. Multi-Modal Embeddings (CLIP / SigLIP)

**Concept:**
- Map images and text into the *same* vector space.
- **Contrastive Learning:** Train model to maximize similarity between (Image of Dog, Text "A dog") and minimize others.
- **Retrieval:**
  - Text-to-Image: Query "Dog" -> Find image of dog.
  - Image-to-Image: Query Image -> Find similar images.

### 2. Multi-Modal RAG Pipeline

**Architecture:**
1.  **Ingestion:**
    - PDF Parser extracts Text and Images.
    - **Option A:** Caption images using VLM (GPT-4o). Index captions as text.
    - **Option B:** Embed images using CLIP. Index in Multi-modal Vector DB.
2.  **Retrieval:**
    - Retrieve relevant Text chunks.
    - Retrieve relevant Images (using text query or image query).
3.  **Generation:**
    - Pass Text + Images to a VLM (GPT-4o / Gemini 1.5).
    - "Based on the text and this chart, what is the trend?"

### 3. Vision Language Models (VLMs)

**Models:**
- **GPT-4o:** SOTA. Native multi-modal.
- **Gemini 1.5 Pro:** 1M context, native video/audio.
- **Claude 3.5 Sonnet:** Excellent vision capabilities.
- **LLaVA / PaliGemma:** Open source alternatives.

**Capabilities:**
- OCR (Optical Character Recognition).
- Object Detection.
- Visual Reasoning ("Why is this meme funny?").

### 4. Multi-Modal Vector Stores

**Tools:**
- **Weaviate / Milvus / Qdrant:** Support multi-modal schemas.
- **Store:** Vectors for text, vectors for images.
- **Metadata:** Link image to its source document page.

### 5. ColPali (Late Interaction for Vision)

**Concept:**
- Instead of one vector per image, use *patches*.
- **ColBERT for Images:** Map image patches to tokens.
- **Benefit:** extremely fine-grained retrieval. Can find a specific small object in a large image.

### 6. Screen Agents (Computer Use)

**Concept:**
- Agent that "sees" the screen and controls mouse/keyboard.
- **Input:** Screenshot of desktop.
- **Action:** `click(x, y)`, `type("hello")`.
- **Use Case:** Automating legacy software with no API.
- **Example:** Anthropic Computer Use, Adept Fuyu.

### 7. Audio RAG

**Concept:**
- **Ingestion:** Transcribe Audio (Whisper) -> Text.
- **Retrieval:** Search Transcript.
- **Advanced:** Audio-native models (Gemini) can "listen" to the raw audio to detect tone/emotion, which text misses.

### 8. Challenges

- **Cost:** Image tokens are expensive.
- **Latency:** Processing images takes longer.
- **Alignment:** Text embedding space and Image embedding space might not be perfectly aligned.

### 9. Summary

**Multi-Modal Strategy:**
1.  **Ingest:** Use **Unstructured.io** or **LlamaParse** to extract images from PDFs.
2.  **Index:** Use **CLIP** for image vectors or **GPT-4o** to generate captions.
3.  **Model:** Use **GPT-4o** or **Claude 3.5** for reasoning.
4.  **Agent:** Use **Screen Agents** for UI automation.

### Next Steps
In the Deep Dive, we will implement a Multi-Modal RAG pipeline (Text+Image retrieval) and a simple Screen Agent loop.
