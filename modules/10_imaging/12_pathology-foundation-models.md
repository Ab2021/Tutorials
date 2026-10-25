# Foundation Models for Pathology and Computational Pathomics

> **Module 10 — Medical Imaging**
> Topic 12 of 12
> Estimated study time: 4 hours
> Prerequisites: Topics 05, 07, 10, and 11
> Version 2026.3

## 1. Why this topic exists

Until recently, computational pathology relied on supervised convolutional neural networks trained on relatively small, task-specific datasets with hand-annotated polygon masks drawn by licensed pathologists. This approach was inherently unscalable: pathologist annotation time costs hundreds of dollars per hour, inter-observer diagnostic concordance for manual segmentation is notoriously low ($\kappa \approx 0.55\text{--}0.70$), and models trained on one cancer type or hospital scanner routinely collapsed when evaluated across external health systems.

Between 2024 and 2026, the field underwent a profound paradigm shift driven by **Self-Supervised Pathology Foundation Models**—such as UNI, CONCH, GigaPath, Prov-GigaPath, CHIEF, and Virchow. Pre-trained on tens of millions of whole-slide images encompassing billions of tissue patches without human labels, these models learn generalizable representations of human histology, cellular morphology, and tissue microarchitecture.

Coupled with weakly supervised **Multiple Instance Learning (MIL)** and pathomics, these foundation models achieve feats that exceed human visual capabilities:
1. Predicting patient overall survival and immunotherapy response directly from routine H&E slides.
2. Detecting molecular mutations (e.g., *EGFR*, *KRAS*, *BRAF*, *TP53*) and Microsatellite Instability (MSI-H) without requiring immediate, expensive Next-Generation Sequencing (NGS).
3. Zero-shot visual question answering and automated microscopic reporting using multimodal pathology-language models (e.g., PathChat, CONCH).

Understanding how to fine-tune, evaluate, and deploy these foundation models within regulated diagnostic frameworks (FDA 510(k), De Novo, and CLIA Laboratory Developed Tests) is now the defining skill in computational pathology.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Compare** leading pathology foundation models (UNI, CONCH, GigaPath, Virchow, CHIEF) across pre-training objectives, token architectures, and clinical benchmark performance.
2. **Implement** weakly supervised Multiple Instance Learning (Attention-MIL, CLAM, TransMIL) to aggregate thousands of patch-level foundation embeddings into slide-level diagnostic predictions.
3. **Execute** pathomic biomarker prediction pipelines inferring molecular alterations (MSI status, oncogenic mutations) directly from standard H&E stained biopsy tiles.
4. **Deploy** multimodal pathology-language models for zero-shot rare tumor identification and automated microscopic description generation.
5. **Mitigate** batch effects and site-specific shortcut learning using stain-invariant contrastive losses and adversarial domain adaptation.
6. **Evaluate** computational pathology algorithms under FDA pre-market submission standards for automated cancer detection and grading software.

---

## 3. The Modern Pathology Foundation Model Landscape

Modern pathology models combine self-supervised learning (DINOv2, masked autoencoding) with massive multi-institution slide corpora:

```
+-------------------------------------------------------------------------------+
|                      2024–2026 PATHOLOGY FOUNDATION MODELS                    |
+-------------------------------------------------------------------------------+
| Model      | Pre-training Corpus        | Architecture | Primary Strength      |
+------------+----------------------------+--------------+-----------------------+
| **UNI**    | 100M+ patches, 20 cancer   | ViT-Large    | General histology     |
| (Mahmood)  | types across 100k+ slides  | (DINOv2 SSL) | classification / MIL  |
+------------+----------------------------+--------------+-----------------------+
| **CONCH**  | 1.17M image-caption pairs  | CoCa         | Vision-Language;      |
| (Mahmood)  | from pathology reports     | Multimodal   | zero-shot retrieval   |
+------------+----------------------------+--------------+-----------------------+
| **GigaPath**| 1.3B patches, 30,000 WSIs | LongNet +    | Whole-slide context;  |
| (Providence)| from Providence Health    | ViT-Giant    | pan-cancer mutation   |
+------------+----------------------------+--------------+-----------------------+
| **Virchow**| 1.5M whole-slide images    | ViT-Huge     | Rare cancer & sub-type|
| (Memorial) | from MSKCC pathology       | (14x14 patch)| diagnostic accuracy   |
+------------+----------------------------+--------------+-----------------------+
| **CHIEF**  | 60k WSIs across 19 organs  | Hierarchical | Pan-cancer survival & |
| (Harvard)  | international multi-center | Transformer  | treatment response    |
+-------------------------------------------------------------------------------+
```

---

## 4. Weakly Supervised Multiple Instance Learning (MIL)

In clinical practice, slide-level labels (e.g., *"Gleason 4+3 Adenocarcinoma present"* or *"Patient responded to Pembrolizumab"*) are readily available from electronic pathology reports, but patch-level ground truth bounding boxes do not exist.

**Multiple Instance Learning (MIL)** treats each Whole-Slide Image as a **Bag** of $N$ instances (patches):

$$\text{Bag } X = \{x_1, x_2, \dots, x_N\}$$

```
Whole-Slide Image (WSI)
           |
           v
[ Patching & Tiling ] ---> 10,000 Patches ($x_1, \dots, x_N$)
                                    |
                                    v
[ Frozen Foundation Model ] ---> $N \times D$ Feature Embeddings ($h_1, \dots, h_N$)
  (e.g., UNI / Virchow)             |
                                    v
                       [ Attention-MIL / CLAM Layer ]
                                    |
             +----------------------+----------------------+
             |                                             |
             v                                             v
  Learned Attention Weights                         Aggregated Slide Vector:
  $a_i = \frac{\exp(w^T \tanh(V h_i))}{\sum \dots}$   $z = \sum_{i=1}^N a_i h_i$
             |                                             |
             v                                             v
  High-Attention Interpretability Map              Slide Classifier Head
  (Pathologist Review / Verification)              (Cancer Grade / Mutation Call)
```

### The Attention Mechanism
In **Clustering-constrained Attention-directed Multiple Instance Learning (CLAM)**, attention weights $a_i$ quantify how much patch $i$ contributes to the slide-level diagnostic decision:

$$a_i = \frac{\exp\left\{W_a^T \left(\tanh(V_a h_i^T) \odot \text{sigm}(U_a h_i^T)\right)\right\}}{\sum_{j=1}^N \exp\left\{W_a^T \left(\tanh(V_a h_j^T) \odot \text{sigm}(U_a h_j^T)\right)\right\}}$$

This produces a fully interpretable heatmap: patches receiving highest attention weights are rendered back onto the whole-slide viewer, allowing the surgical pathologist to visually confirm the tumor cells driving the algorithm's prediction.

---

## 5. Implementation: Attention Multiple Instance Learning (MIL) in PyTorch

The following PyTorch module implements an enterprise Attention-MIL aggregator that ingests pre-computed patch embeddings from a pathology foundation model and outputs slide-level predictions with instance attention weights:

```python
"""
Attention-MIL Aggregator for Slide-Level Pathology Classification.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMIL(nn.Module):
    def __init__(self, feature_dim: int = 1024, hidden_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # Gated Attention Network (Ilse et al., 2018)
        self.attention_v = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh()
        )
        self.attention_u = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Sigmoid()
        )
        self.attention_weights = nn.Linear(hidden_dim, 1)

        # Slide-level classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: Tensor of shape (N, feature_dim), where N is number of tissue patches in slide.
        Returns:
            logits: (1, num_classes)
            attention: (N, 1) attention scores for interpretability heatmap
        """
        # 1. Compute Gated Attention
        v = self.attention_v(h)  # (N, hidden_dim)
        u = self.attention_u(h)  # (N, hidden_dim)
        
        # Element-wise gate and linear projection
        a_unnorm = self.attention_weights(v * u)  # (N, 1)
        
        # Softmax over all N patches in the bag
        a = F.softmax(a_unnorm, dim=0)  # (N, 1)

        # 2. Compute Weighted Slide Representation (Bag Embedding)
        # z = sum(a_i * h_i)
        z = torch.sum(a * h, dim=0, keepdim=True)  # (1, feature_dim)

        # 3. Slide-level classification
        logits = self.classifier(z)  # (1, num_classes)

        return logits, a

# Demonstration with synthetic foundation model embeddings
if __name__ == "__main__":
    # Simulate a slide with 4,000 patches embedded by UNI (1024-dim vectors)
    num_patches = 4000
    embedding_dim = 1024
    
    mock_slide_features = torch.randn(num_patches, embedding_dim)
    
    # 50 patches simulate cancer cells (shift their distribution)
    mock_slide_features[1200:1250] += 2.5
    
    model = AttentionMIL(feature_dim=embedding_dim, num_classes=2)
    model.eval()
    
    with torch.no_grad():
        slide_logits, patch_attentions = model(mock_slide_features)
        
    probs = F.softmax(slide_logits, dim=-1)
    top_patch_idx = torch.argmax(patch_attentions).item()
    
    print(f"Slide Prediction Probabilities: {probs.numpy()[0]}")
    print(f"Total Patches Evaluated: {num_patches}")
    print(f"Highest Attention Patch Index: {top_patch_idx}")
    print(f"Max Attention Value: {patch_attentions[top_patch_idx].item():.6f}")
    print(f"Mean Attention Baseline: {1.0 / num_patches:.6f}")
```

---

## 6. Computational Pathomics: Predicting Genotypes from Phenotypes

A transformative breakthrough in 2026 computational pathology is **Pathomics**—the direct computational inference of genetic mutations, epigenetic silencing, and molecular subtypes from standard H&E morphology alone, without destroying tissue for NGS:

```
+-------------------------------------------------------------------------------+
|                      MOLECULAR PREDICTION FROM H&E HISTOLOGY                  |
+-------------------------------------------------------------------------------+
| Clinical Biomarker   | Primary Cancer Indication | AUROC (Foundation Model)   |
+----------------------+---------------------------+----------------------------+
| **MSI-H / dMMR**     | Colorectal, Endometrial   | $0.93\text{--}0.96$        |
| **BRAF V600E**       | Melanoma, Thyroid         | $0.88\text{--}0.92$        |
| **EGFR Mutation**    | Non-Small Cell Lung Cancer| $0.84\text{--}0.89$        |
| **HRD Status**       | Ovarian, Triple-Neg Breast| $0.81\text{--}0.86$        |
| **HER2 Status**      | Invasive Breast Carcinoma | $0.87\text{--}0.91$        |
+-------------------------------------------------------------------------------+
```

### The Clinical Utility Mechanism: Rapid Triage
Direct slide prediction does not replace definitive molecular sequencing; it functions as a **high-speed triage screener**. Standard NGS takes 14 to 28 days to return results from reference molecular labs. An AI model running on the initial biopsy slide provides an MSI-H prediction in **90 seconds**, allowing medical oncologists to immediately initiate life-saving immune checkpoint inhibitors while awaiting formal sequencing confirmation.

---

## 7. Regulatory Pathways: FDA Clearances in Computational Pathology

Commercialization of computational pathology algorithms requires navigating FDA Center for Devices and Radiological Health (CDRH) Class II De Novo or 510(k) pathways:

```
[ Pre-Market Approval Path for Computational Pathology Software ]

1. Multi-Reader Multi-Case (MRMC) Clinical Study
   - Must evaluate 15-25 licensed pathologists reviewing >= 300 enriched slides
   - Primary Endpoint: Non-inferiority or superiority in sensitivity / specificity
     WITH AI assistance versus WITHOUT AI assistance (washout period >= 4 weeks)

2. Technical Interoperability Validation
   - Testing across >= 3 commercial whole-slide scanners (e.g. Philips, Leica, Hamamatsu)
   - Proof of robustness against slide preparation variations (thickness, stain age)

3. Predetermined Change Control Plan (PCCP)
   - Pre-specifying boundaries for periodic foundation model weight updates
```

---

## 8. Exercises and Self-Check

### Exercise 1: Multiple Instance Learning Aggregation Strategies
Contrast three distinct MIL pooling functions: Mean-Pooling, Max-Pooling, and Attention-Pooling.
1. Why does Mean-Pooling fail when detecting focal micrometastases (where only 10 out of 5,000 patches are malignant)?
2. Why is Max-Pooling vulnerable to single-patch artifacts (e.g., an out-of-focus smudge or knife mark that resembles a tumor giant cell)?
3. How does Attention-Pooling resolve the failure modes of both Mean- and Max-Pooling?

### Exercise 2: Evaluating Spatial Saliency Concordance
Suppose an Attention-MIL model achieves an AUROC of 0.94 for predicting breast cancer lymph node metastasis. However, when you render the top 1% attention patches on the whole slide, 80% of the attention weights fall on non-neoplastic lymphocytes and fat necrosis rather than metastatic carcinoma cells.
1. What regulatory risk does this represent under FDA Good Machine Learning Practice (GMLP)?
2. What metric (e.g., Intersection-over-Union between attention heatmap and pathologist manual tumor annotation) should be included in the validation dossier to detect shortcut learning?

---

## 9. Key Terms

- **Pathology Foundation Model:** A large-scale vision transformer pre-trained on tens of millions of unlabeled microscopic histology images using self-supervised objectives to produce generalizable representations.
- **Multiple Instance Learning (MIL):** A form of weakly supervised machine learning where labels are associated with sets of instances (bags) rather than individual instances.
- **Pathomics:** The extraction and mathematical analysis of quantitative cellular and tissue features from microscopic histology images, used to predict molecular alterations and clinical outcomes.
- **Microsatellite Instability (MSI-H):** A hypermutable phenotype caused by defective DNA mismatch repair (dMMR), highly predictive of dramatic clinical response to PD-1 immunotherapy.
- **MRMC Study (Multi-Reader Multi-Case):** The regulatory gold standard study design for evaluating whether diagnostic software improves human clinician performance.

---

## 10. Sources and Further Reading

- Chen, R. J., et al. (2024). "Towards a Visual-Language Foundation Model for Computational Pathology (CONCH)." *Nature Medicine*, 30(3), 863–874.
- Lu, M. Y., et al. (2021). "Data-Efficient and Weakly Supervised Computational Pathology on Whole-Slide Images (CLAM)." *Nature Biomedical Engineering*, 5(6), 555–570.
- Xu, H., et al. (2024). "A Whole-Slide Foundation Model for Digital Pathology from Real-World Data (GigaPath)." *Nature*, 630(8015), 191–199.
- Vorontsov, E., et al. (2024). "A Foundation Model for Clinical-Grade Computational Pathology (Virchow)." *Nature Medicine*, 30(5), 1412–1422.
- Ilse, M., Tomczak, J., & Welling, M. (2018). "Attention-based Deep Multiple Instance Learning." *International Conference on Machine Learning (ICML)*, 2127–2136.
