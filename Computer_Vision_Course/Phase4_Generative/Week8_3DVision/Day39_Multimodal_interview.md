# Day 39 Interview Questions: Multimodal Vision

## Q1: What is "Zero-Shot Transfer" in CLIP?
**Answer:**
*   The ability to classify images into categories that were not explicitly seen during training.
*   CLIP is trained to match images to *any* text.
*   At test time, we provide the class names as text prompts. The model matches the image to the most similar prompt.
*   No fine-tuning required.

## Q2: Why does CLIP use a Contrastive Loss?
**Answer:**
*   Generative Loss (Predict next word) is hard for images (too much variability in captions).
*   Contrastive Loss (Match correct pair, push away incorrect pairs) is more efficient.
*   It learns the **joint embedding space** structure rather than trying to generate exact pixels or words.

## Q3: What is the "Modality Gap"?
**Answer:**
*   Even in aligned models like CLIP, the embedding of an image and its corresponding text often occupy different regions of the vector space (a constant offset).
*   This is due to the different nature of the encoders and the training objective (temperature scaling).

## Q4: How does LLaVA connect Vision to Language?
**Answer:**
*   It uses a simple **Linear Projection Layer** (or MLP).
*   It takes the grid of visual features from CLIP (e.g., $256 \times 1024$).
*   Projects them to the dimension of the LLM's word embeddings (e.g., $256 \times 4096$).
*   These projected features are treated as "Visual Tokens" and prepended to the text tokens.

## Q5: What is "Visual Instruction Tuning"?
**Answer:**
*   Standard fine-tuning uses (Image, Label) pairs.
*   Instruction Tuning uses (Image, Instruction, Response) triplets.
*   Example:
    *   Instruction: "Describe the unusual aspect of this image."
    *   Response: "The man is ironing clothes on top of a taxi."
*   Teaches the model to follow complex user commands.

## Q6: Explain the "Perceiver Resampler" in Flamingo.
**Answer:**
*   A mechanism to handle variable-length visual inputs (images/videos).
*   It uses a fixed number of learned latent queries (e.g., 64).
*   These queries attend to the visual features (Cross-Attention) and compress them into 64 tokens.
*   This ensures the LLM always receives a constant number of visual tokens, regardless of input size.

## Q7: What is "Grounding"?
**Answer:**
*   Linking abstract concepts (words) to concrete pixels (regions).
*   Example: The word "Dog" in the sentence corresponds to the bounding box $[x, y, w, h]$ in the image.
*   Crucial for robotics and detailed understanding.

## Q8: Why is SAM considered a "Foundation Model"?
**Answer:**
*   It was trained on a massive dataset (11M images, 1B masks).
*   It generalizes to almost any image domain (underwater, microscopy, space) without fine-tuning.
*   It supports flexible prompting (points, boxes, text).

## Q9: What is the difference between CLIP and ALIGN?
**Answer:**
*   **CLIP:** OpenAI. Curated dataset (filtered for quality). 400M pairs.
*   **ALIGN:** Google. Noisy dataset (raw Alt-text). 1.8B pairs.
*   ALIGN showed that scale can overcome noise.

## Q10: Implement CLIP Zero-Shot logic (Conceptual).
**Answer:**
```python
def zero_shot_classify(image, class_names, model):
    # 1. Encode Image
    image_features = model.encode_image(image)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    
    # 2. Encode Text
    text_prompts = [f"A photo of a {c}" for c in class_names]
    text_tokens = clip.tokenize(text_prompts)
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 3. Cosine Similarity
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    return similarity
```
