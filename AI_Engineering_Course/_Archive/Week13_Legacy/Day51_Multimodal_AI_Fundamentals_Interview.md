# Day 51: Multimodal AI Fundamentals
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How does CLIP work and what makes it powerful?

**Answer:**
- **Architecture:** Dual encoders (image encoder + text encoder).
- **Training:** Contrastive learning on 400M image-text pairs.
- **Objective:** Maximize similarity for matching pairs, minimize for non-matching.
- **Power:** Zero-shot classification without task-specific training.
- **Use Cases:** Image search, classification, retrieval.

#### Q2: What is the difference between early and late fusion in multimodal models?

**Answer:**
**Early Fusion:**
- Combine features early: `concat(image_features, text_features) → model`
- **Pros:** Rich cross-modal interaction.
- **Cons:** Computationally expensive, less modular.

**Late Fusion:**
- Process separately: `image → model_A`, `text → model_B`, then combine outputs.
- **Pros:** Modular, efficient, can train separately.
- **Cons:** Limited cross-modal interaction.

**Cross-Attention (Middle Ground):**
- Process separately but with cross-attention between modalities.
- **Best of both:** Efficient + rich interaction.

#### Q3: How does Vision Transformer (ViT) work?

**Answer:**
1. **Patch Embedding:** Split image into 16×16 patches, flatten to vectors.
2. **Positional Embedding:** Add position information.
3. **Class Token:** Prepend learnable [CLS] token.
4. **Transformer Encoder:** Standard Transformer layers.
5. **Classification:** Use [CLS] token output for classification.

**Benefits:** Scales better than CNNs, SOTA with sufficient data.

#### Q4: What are the main challenges in multimodal AI?

**Answer:**
**Modality Gap:**
- Different modalities have different distributions.
- **Solution:** Alignment via contrastive learning (CLIP).

**Data Scarcity:**
- Less paired multimodal data than text-only.
- **Solution:** Web scraping, synthetic data.

**Computational Cost:**
- Processing images/video is expensive.
- **Solution:** Efficient architectures, compression.

**Evaluation:**
- Hard to evaluate multimodal outputs.
- **Solution:** Human evaluation, multimodal benchmarks.

#### Q5: How do you implement zero-shot image classification with CLIP?

**Answer:**
```python
1. Encode image: image_embed = clip.encode_image(image)
2. Create text prompts: ["a photo of a cat", "a photo of a dog"]
3. Encode prompts: text_embeds = clip.encode_text(prompts)
4. Compute similarity: similarities = image_embed @ text_embeds.T
5. Softmax: probs = softmax(similarities)
6. Predict: class = argmax(probs)
```

**No training needed** for new classes!

---

### Production Challenges

#### Challenge 1: CLIP Modality Gap

**Scenario:** CLIP image and text embeddings are in different regions of embedding space.
**Root Cause:** Image and text encoders trained separately, not perfectly aligned.
**Solution:**
- **Temperature Scaling:** Adjust logit_scale parameter.
- **Fine-tuning:** Fine-tune on domain-specific data.
- **Normalization:** L2-normalize embeddings before similarity.
- **Projection:** Add learned projection layer to align spaces.

#### Challenge 2: ViT Data Hunger

**Scenario:** ViT performs poorly when trained on small dataset (e.g., 10K images).
**Root Cause:** ViT lacks inductive bias of CNNs (e.g., translation invariance). Needs more data.
**Solution:**
- **Pre-training:** Use pre-trained ViT (ImageNet-21K).
- **Data Augmentation:** Heavy augmentation (RandAugment, Mixup).
- **Hybrid:** Use CNN stem + ViT (better with less data).
- **Smaller ViT:** Use ViT-Small instead of ViT-Large.

#### Challenge 3: Multimodal Fusion Overfitting

**Scenario:** Multimodal model overfits to training data. Test accuracy 20% lower than train.
**Root Cause:** Too many parameters in fusion layer.
**Solution:**
- **Simpler Fusion:** Use concatenation instead of complex cross-attention.
- **Dropout:** Add dropout (0.3-0.5) in fusion layers.
- **Regularization:** L2 regularization, early stopping.
- **More Data:** Collect more multimodal training data.

#### Challenge 4: VQA Bias

**Scenario:** VQA model always predicts "yes" for yes/no questions (80% accuracy but useless).
**Root Cause:** Dataset bias (80% of yes/no questions have "yes" answer).
**Solution:**
- **Balanced Dataset:** Balance yes/no answers in training data.
- **Adversarial Training:** Train with adversarial examples.
- **Debiasing:** Use debiasing techniques (e.g., learned mixin).
- **Evaluation:** Use balanced test set, not just accuracy.

#### Challenge 5: Image Captioning Repetition

**Scenario:** Image captioning model generates repetitive captions: "a dog a dog a dog".
**Root Cause:** Beam search or sampling issues.
**Solution:**
- **N-gram Blocking:** Block repeated n-grams during generation.
- **Diverse Beam Search:** Use diverse beam search instead of standard.
- **Temperature:** Adjust sampling temperature (0.7-0.9).
- **Top-p Sampling:** Use nucleus sampling instead of greedy.

### Summary Checklist for Production
- [ ] **CLIP:** Use for **zero-shot classification**, **image search**.
- [ ] **ViT:** Pre-train on **ImageNet-21K** for best results.
- [ ] **Fusion:** Use **cross-attention** for rich interaction.
- [ ] **VQA:** Use **balanced datasets** to avoid bias.
- [ ] **Captioning:** Use **diverse beam search** to avoid repetition.
- [ ] **Evaluation:** Use **multimodal benchmarks** (VQAv2, COCO Captions).
