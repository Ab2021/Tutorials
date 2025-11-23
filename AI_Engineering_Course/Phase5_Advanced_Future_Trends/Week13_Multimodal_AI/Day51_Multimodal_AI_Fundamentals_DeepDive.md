# Day 51: Multimodal AI Fundamentals
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. CLIP Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel

class CLIPFromScratch(nn.Module):
    def __init__(self, image_encoder, text_encoder, embed_dim=512):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # Projection heads
        self.image_projection = nn.Linear(image_encoder.output_dim, embed_dim)
        self.text_projection = nn.Linear(text_encoder.output_dim, embed_dim)
        
        # Learnable temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def encode_image(self, images):
        """Encode images to embeddings."""
        image_features = self.image_encoder(images)
        image_embeds = self.image_projection(image_features)
        image_embeds = F.normalize(image_embeds, dim=-1)
        return image_embeds
    
    def encode_text(self, text):
        """Encode text to embeddings."""
        text_features = self.text_encoder(text)
        text_embeds = self.text_projection(text_features)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return text_embeds
    
    def forward(self, images, text):
        """Compute contrastive loss."""
        image_embeds = self.encode_image(images)
        text_embeds = self.encode_text(text)
        
        # Cosine similarity
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_embeds @ text_embeds.T
        logits_per_text = logits_per_image.T
        
        # Contrastive loss
        batch_size = images.shape[0]
        labels = torch.arange(batch_size, device=images.device)
        
        loss_i = F.cross_entropy(logits_per_image, labels)
        loss_t = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i + loss_t) / 2
        
        return loss

# Training loop
def train_clip(model, dataloader, optimizer, epochs=10):
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        
        for images, texts in dataloader:
            optimizer.zero_grad()
            
            loss = model(images, texts)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch}: Loss = {total_loss / len(dataloader):.4f}")

# Zero-shot classification
def zero_shot_classify(model, image, class_names):
    """Classify image using text prompts."""
    # Encode image
    image_embed = model.encode_image(image.unsqueeze(0))
    
    # Encode class names
    text_prompts = [f"a photo of a {name}" for name in class_names]
    text_embeds = model.encode_text(text_prompts)
    
    # Compute similarities
    similarities = (image_embed @ text_embeds.T).squeeze(0)
    probs = F.softmax(similarities, dim=0)
    
    return probs
```

### 2. Vision Transformer (ViT) Implementation

```python
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        
        # Convolutional patch embedding
        self.projection = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.projection(x)  # (batch, embed_dim, H/P, W/P)
        x = x.flatten(2)  # (batch, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (batch, num_patches, embed_dim)
        return x

class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12
    ):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (batch, num_patches, embed_dim)
        
        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Transformer
        x = self.transformer(x)
        
        # Classification from class token
        cls_output = x[:, 0]
        logits = self.head(cls_output)
        
        return logits
```

### 3. Multimodal Fusion with Cross-Attention

```python
class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
    
    def forward(self, query_features, key_value_features):
        """
        query_features: (batch, query_len, dim) - e.g., text features
        key_value_features: (batch, kv_len, dim) - e.g., image features
        """
        batch_size = query_features.shape[0]
        
        # Project
        Q = self.q_proj(query_features)
        K = self.k_proj(key_value_features)
        V = self.v_proj(key_value_features)
        
        # Reshape for multi-head
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous()
        output = output.view(batch_size, -1, self.num_heads * self.head_dim)
        output = self.o_proj(output)
        
        return output

class MultimodalFusion(nn.Module):
    def __init__(self, dim=768, num_layers=6):
        super().__init__()
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'text_to_image': CrossModalAttention(dim),
                'image_to_text': CrossModalAttention(dim),
                'text_self': nn.MultiheadAttention(dim, num_heads=8, batch_first=True),
                'image_self': nn.MultiheadAttention(dim, num_heads=8, batch_first=True),
                'text_ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                ),
                'image_ffn': nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.GELU(),
                    nn.Linear(dim * 4, dim)
                )
            })
            for _ in range(num_layers)
        ])
    
    def forward(self, text_features, image_features):
        for layer in self.layers:
            # Self-attention
            text_features = text_features + layer['text_self'](
                text_features, text_features, text_features
            )[0]
            image_features = image_features + layer['image_self'](
                image_features, image_features, image_features
            )[0]
            
            # Cross-attention
            text_features = text_features + layer['text_to_image'](
                text_features, image_features
            )
            image_features = image_features + layer['image_to_text'](
                image_features, text_features
            )
            
            # FFN
            text_features = text_features + layer['text_ffn'](text_features)
            image_features = image_features + layer['image_ffn'](image_features)
        
        return text_features, image_features
```

### 4. Visual Question Answering (VQA)

```python
class VQAModel(nn.Module):
    def __init__(self, image_encoder, text_encoder, num_answers=3000):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        
        # Fusion
        self.fusion = MultimodalFusion(dim=768)
        
        # Answer classifier
        self.classifier = nn.Linear(768, num_answers)
    
    def forward(self, image, question):
        # Encode
        image_features = self.image_encoder(image)
        question_features = self.text_encoder(question)
        
        # Fuse
        text_fused, image_fused = self.fusion(question_features, image_features)
        
        # Pool and classify
        pooled = torch.cat([
            text_fused.mean(dim=1),
            image_fused.mean(dim=1)
        ], dim=-1)
        
        logits = self.classifier(pooled)
        
        return logits
```

### 5. Image Captioning

```python
class ImageCaptioningModel(nn.Module):
    def __init__(self, image_encoder, text_decoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_decoder = text_decoder
    
    def forward(self, images, captions):
        # Encode image
        image_features = self.image_encoder(images)
        
        # Decode with cross-attention to image
        outputs = self.text_decoder(
            input_ids=captions,
            encoder_hidden_states=image_features
        )
        
        return outputs.logits
    
    def generate_caption(self, image, max_length=50):
        """Generate caption for image."""
        image_features = self.image_encoder(image.unsqueeze(0))
        
        # Start with [BOS] token
        generated = torch.tensor([[self.text_decoder.config.bos_token_id]])
        
        for _ in range(max_length):
            outputs = self.text_decoder(
                input_ids=generated,
                encoder_hidden_states=image_features
            )
            
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if next_token.item() == self.text_decoder.config.eos_token_id:
                break
        
        return generated
```
