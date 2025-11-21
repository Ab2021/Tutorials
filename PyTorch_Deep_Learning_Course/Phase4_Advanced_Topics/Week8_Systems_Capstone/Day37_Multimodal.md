# Day 37: Multimodal Models - Theory & Implementation

> **Phase**: 4 - Advanced Topics
> **Week**: 8 - Systems & Capstone
> **Topic**: CLIP, BLIP, and LLaVA

## 1. Theoretical Foundation: Connecting Modalities

We have powerful Vision models (ResNet/ViT) and Language models (BERT/GPT).
**Multimodal**: Connecting them.
*   **Contrastive Learning**: Aligning the vector spaces of Image and Text.
*   **Generative**: Generating Text from Image (Captioning/VQA).

## 2. CLIP (Contrastive Language-Image Pre-training)

OpenAI (2021).
*   **Architecture**: Image Encoder (ViT) + Text Encoder (Transformer).
*   **Data**: 400M (Image, Text) pairs from web.
*   **Objective**: Maximize cosine similarity of correct pairs $(I_i, T_i)$ and minimize incorrect pairs $(I_i, T_j)$ in a batch.
*   **Result**: Zero-Shot Classification. "A photo of a dog".

## 3. LLaVA (Large Language-and-Vision Assistant)

Visual Instruction Tuning.
*   **Architecture**: Pre-trained ViT (CLIP) + Projection Layer + Pre-trained LLM (LLaMA).
*   **Training**:
    1.  **Feature Alignment**: Freeze ViT & LLM. Train Projection.
    2.  **Fine-Tuning**: Freeze ViT. Train Projection & LLM on visual instructions.
*   Allows chatting with images.

## 4. Implementation: Using CLIP with Hugging Face

```python
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1) 
# [0.99, 0.01]
```

## 5. Implementation: Simple VQA Model

Concatenate Image Features and Text Embeddings.

```python
class VQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = resnet50(pretrained=True)
        self.rnn = nn.LSTM(embed_dim, hidden_dim)
        self.fc = nn.Linear(2048 + hidden_dim, vocab_size)
        
    def forward(self, img, question):
        img_feat = self.cnn(img) # [Batch, 2048]
        q_feat = self.rnn(question) # [Batch, Hidden]
        combined = torch.cat((img_feat, q_feat), dim=1)
        return self.fc(combined)
```
