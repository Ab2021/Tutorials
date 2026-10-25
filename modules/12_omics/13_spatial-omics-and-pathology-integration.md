# Spatial Omics and Pathology Integration

> **Module 12 — Omics and Precision Medicine**
> Topic 13 of 14
> Estimated study time: 4 hours
> Prerequisites: Topics 03, 04, 05, and Module 10 Topics 11–12
> Version 2026.3

## 1. Why this topic exists

For decades, biological understanding of human disease was polarized between two incompatible resolutions:
1. **Histopathology (Morphology without High-Plex Genomics):** Pathologists examine Hematoxylin and Eosin (H&E) or low-plex Immunohistochemistry (IHC) glass slides under light microscopy, observing microscopic tissue architecture, cellular infiltration, and nuclear pleomorphism at sub-micron resolution, but with virtually zero molecular readouts (measuring only 1 to 4 protein markers at a time).
2. **Bulk and Single-Cell Genomics (Omics without Spatial Architecture):** Next-Generation Sequencing (NGS) and single-cell RNA sequencing (scRNA-seq) measure 20,000+ genes across individual cells. However, preparing tissue for scRNA-seq requires mechanical dissociation and enzymatic digestion—literally putting a solid tumor into a blender—destroying the spatial architecture, cell-cell contact networks, and tumor-immune microenvironment (TME).

The convergence of high-throughput sequencing with microscopic imaging created **Spatial Omics**—technologies like 10x Genomics Visium, 10x Xenium, NanoString CosMx Spatial Molecular Imager (SMI), and Akoya PhenoCycler (CODEX). In 2026, spatial omics profiles thousands of RNA transcripts and proteins directly in intact, formalin-fixed paraffin-embedded (FFPE) clinical tissue sections.

The central engineering challenge is the **multimodal bridge between spatial omics and standard clinical pathology**. Spatial profiling is expensive (\$1,500 to \$4,000 per slide) and destructive or slow. Standard H&E is cheap (\$2 per slide) and universal. By training multimodal deep learning architectures (e.g., ST-Net, HisToGene, Bleeker) that co-register spatial transcriptomics with digitized H&E whole-slide images, data scientists can **predict local gene expression directly from routine histology images**, turning routine hospital pathology slides into virtual spatial molecular assays.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Compare** the primary commercial spatial omics technologies (sequencing-based array capture vs. in-situ imaging/hybridization), evaluating resolution, plex, and sample requirements.
2. **Implement** multi-modal affine and non-rigid image registration co-registering high-resolution H&E histological whole-slide images with spatial transcriptomic spot coordinate matrices.
3. **Execute** cell-type deconvolution algorithms (Seurat, RCTD, Cell2location) mapping single-cell RNA-seq reference profiles onto multi-cellular spatial capture spots.
4. **Architect** convolutional and graph neural network models (ST-Net, HisToGene) that predict spatial transcriptomic gene expression profiles directly from H&E morphological patches.
5. **Model** the Tumor Microenvironment (TME) using spatial point process models and Cell-Cell Interaction (CCI) graphs to predict immunotherapy resistance.
6. **Deploy** open-source spatial data frameworks (AnnData, SpatialData, Squidpy, Giotto) for high-dimensional spatial omics analysis.

---

## 3. Spatial Omics Technology Taxonomy

Spatial biology technologies split into two distinct physical methodologies:

```
+-------------------------------------------------------------------------------+
|                      SPATIAL OMICS TECHNOLOGICAL TAXONOMY                     |
+-------------------------------------------------------------------------------+
| Modality   | Primary Platforms         | Spatial Resolution  | Plex / Genes    |
+------------+---------------------------+---------------------+-----------------+
| **Array-   | 10x Genomics Visium,      | $55\,\mu\text{m}$   | Transcriptome-  |
| Based**    | BGI Stereo-seq, Curio     | (1-10 cells/spot)   | wide (20,000+)  |
|            | Seeker                    | down to sub-micron  | unbiased        |
+------------+---------------------------+---------------------+-----------------+
| **In-Situ  | 10x Xenium, NanoString    | Sub-cellular        | Targeted Panel  |
| Imaging**  | CosMx SMI, Vizgen MERSCOPE| ($0.1\,\mu\text{m}$)| (300 to 5,000   |
| (FISH)     |                           | single-molecule)    | curated genes)  |
+------------+---------------------------+---------------------+-----------------+
| **Spatial  | Akoya PhenoCycler (CODEX),| Single-cell         | 40 to 100       |
| Proteomics| Standard BioTools Hyperion| ($0.5\,\mu\text{m}$)| targeted        |
|            |                           |                     | proteins        |
+-------------------------------------------------------------------------------+
```

---

## 4. Multi-Modal Co-Registration: Mapping Spots to Pixels

In sequencing-based spatial transcriptomics (e.g., 10x Visium), tissue is mounted on a glass slide containing a grid of spatially barcoded capture spots. The slide is stained with H&E, imaged under light microscopy, and then permeabilized to allow mRNA to hybridize to underlying spot oligonucleotides.

```
Glass Slide Surface
+---------------------------------------------------------+
| [ High-Resolution H&E Whole-Slide Image ]               |
|   Coordinates: Pixel Space $(u, v) \in [0, 100000]^2$   |
|                           ^                             |
|                           | Affine Transformation       |
|                           | Matrix ($M_{3 \times 3}$)   |
|                           v                             |
| [ Spatial Transcriptomics Barcode Grid ]                |
|   Coordinates: Spot Space $(x, y) \in [0, 78]^2$        |
|   Each spot has: Gene Expression Vector $Y_i \in \mathbb{R}^{20000}$|
+---------------------------------------------------------+
```

### The Coordinate Mapping Matrix:
$$\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \begin{bmatrix} s_x \cos\theta & -s_y \sin\theta & t_x \\ s_x \sin\theta & s_y \cos\theta & t_y \\ 0 & 0 & 1 \end{bmatrix} \begin{bmatrix} x \\ y \\ 1 \end{bmatrix}$$

Accurate registration is clinically non-negotiable: a misalignment of $20\,\mu\text{m}$ (half a cell diameter) causes the model to associate an immune checkpoint transcript (*PDCD1 / CD274*) with adjacent healthy stroma rather than infiltrating cytotoxic T-lymphocytes.

---

## 5. Implementation: Spatial Spot-to-Image Extraction and Prediction in Python

The following Python script implements the core data processing pipeline: it loads a spatial transcriptomics coordinate matrix, extracts corresponding image patches from the registered whole-slide image, and trains a vision model to predict local gene expression (e.g., *FASN* or *ERBB2 / HER2*) from morphological image tiles:

```python
"""
Spatial Transcriptomics: Spot-Image Co-Registration and Patch Extraction.
"""
from typing import List, Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn

class SpatialOmicsPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        mock_wsi_image: np.ndarray, 
        spot_coordinates: np.ndarray, 
        gene_expression_matrix: np.ndarray, 
        patch_size: int = 128
    ):
        """
        mock_wsi_image: (H, W, 3) image array
        spot_coordinates: (N, 2) array of (x_center, y_center) pixel coordinates
        gene_expression_matrix: (N, G) log1p normalized expression values for G target genes
        """
        self.image = mock_wsi_image
        self.coords = spot_coordinates
        self.y = torch.tensor(gene_expression_matrix, dtype=torch.float32)
        self.patch_size = patch_size
        self.half_size = patch_size // 2

    def __len__(self) -> int:
        return len(self.coords)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        cx, cy = self.coords[idx]
        x1, x2 = cx - self.half_size, cx + self.half_size
        y1, y2 = cy - self.half_size, cy + self.half_size
        
        # Crop patch around spatial spot
        patch = self.image[y1:y2, x1:x2, :] # (H, W, 3)
        
        # Transform to PyTorch channel-first tensor: (3, H, W) normalized to [0, 1]
        patch_tensor = torch.tensor(patch, dtype=torch.float32).permute(2, 0, 1) / 255.0
        target_genes = self.y[idx]
        
        return patch_tensor, target_genes

class HistologyToGenePredictor(nn.Module):
    """Deep network predicting continuous local gene expression from histology patches."""
    def __init__(self, num_target_genes: int = 5):
        super().__init__()
        # Backbone: Lightweight CNN feature extractor
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_target_genes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.regressor(features)

# Example Demonstration
if __name__ == "__main__":
    # Simulate a small tissue image (1000 x 1000 px)
    mock_slide = np.random.randint(100, 240, size=(1000, 1000, 3), dtype=np.uint8)
    
    # 50 spatial transcriptomics spots
    np.random.seed(42)
    spot_x = np.random.randint(100, 900, size=50)
    spot_y = np.random.randint(100, 900, size=50)
    mock_spot_coords = np.column_stack([spot_x, spot_y])
    
    # 5 oncogenic target genes (e.g., MKI67, ERBB2, TP53, CD8A, PDCD1)
    mock_expression = np.random.lognormal(mean=0.5, sigma=0.8, size=(50, 5))
    
    dataset = SpatialOmicsPatchDataset(mock_slide, mock_spot_coords, mock_expression, patch_size=64)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
    
    model = HistologyToGenePredictor(num_target_genes=5)
    model.train()
    
    # Run single training pass demonstration
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    for patches, genes in dataloader:
        optimizer.zero_grad()
        predictions = model(patches)
        loss = criterion(predictions, genes)
        loss.backward()
        optimizer.step()
        break
        
    print(f"Spatial Patch Dataset Initialized: {len(dataset)} spots.")
    print(f"Model Loss on First Batch: {loss.item():.4f}")
    print(f"Predicted Expression Vector Shape: {predictions.shape}")
```

---

## 6. Cell-Type Deconvolution & Microenvironment Modeling

A critical limitation of first-generation array-based spatial transcriptomics (e.g., standard Visium) is that each $55\,\mu\text{m}$ capture spot captures mRNA from **5 to 30 individual cells**. It is not single-cell resolution.

### Deconvolution Mechanics (Cell2location / RCTD)
To infer cell-type proportions within each spot, bioinformaticians use **Probabilistic Cell-Type Deconvolution**:

$$\mathbf{Y}_{s, g} \approx \sum_{k=1}^K w_{s, k} \cdot \mathbf{\beta}_{k, g}$$

Where:
- $\mathbf{Y}_{s, g}$ is the observed expression of gene $g$ at spatial spot $s$.
- $\mathbf{\beta}_{k, g}$ is the reference expression profile of cell-type $k$ derived from single-cell RNA-seq (scRNA-seq).
- $w_{s, k}$ is the latent abundance (proportion) of cell-type $k$ at spot $s$.

```
[ Single-Cell Reference ]            [ Spatial Transcriptomics Spot ]
  - CD8+ Cytotoxic T-cells             - Observed Spot RNA count
  - Regulatory T-cells (Treg)                          |
  - Cancer Epithelial cells                            v
  - Cancer-Associated Fibroblasts     [ Robust Cell Type Deconvolution (RCTD) ]
                                                       |
                                                       v
Deconvoluted Spatial Map:
  - Spot #102: 60% Malignant Luminal B, 30% Fibroblast, 10% CD8+ T-cell
  - Proximity Analysis: Are CD8+ T-cells trapped in stroma or contacting tumor?
```

---

## 7. Clinical Translation: Spatial Biomarkers for Immunotherapy

Spatial omics changes oncology by moving beyond simple "bulk biomarker abundance" to **Spatial Point Process Metrics**:

```
+-------------------------------------------------------------------------------+
|                      SPATIAL ARCHITECTURE OF IMMUNE RESISTANCE                |
+-------------------------------------------------------------------------------+

PHENOTYPE A: "IMMUNE DESERT"
  - Definition: Complete absence of cytotoxic T-cells within tumor or stroma.
  - Molecular Profile: High beta-catenin signaling; low chemokine expression.
  - Clinical Outcome: Complete resistance to PD-1 / PD-L1 immune checkpoint blockade.

PHENOTYPE B: "IMMUNE EXCLUDED" (STROMA TRAPPED)
  - Definition: Abundant CD8+ T-cells present, but restricted to peritumoral stroma;
    zero infiltration into tumor parenchyma.
  - Molecular Profile: High TGF-beta, dense cross-linked collagen matrix (fibrosis).
  - Clinical Outcome: Requires combination therapy (Anti-TGF-beta + Anti-PD-L1).

PHENOTYPE C: "IMMUNE INFLAMMATORY" (HOT TUMOR)
  - Definition: CD8+ T-cells in direct physical contact with PD-L1+ tumor cells.
  - Molecular Profile: High interferon-gamma signature, high CXCL9/CXCL10.
  - Clinical Outcome: High probability of complete objective response to Pembrolizumab.
```

---

## 8. Exercises and Self-Check

### Exercise 1: Spatial Autocorrelation with Moran's I
In spatial transcriptomics, gene expression is not distributed randomly across a tissue.
1. Write down the mathematical formulation of **Moran's I** spatial autocorrelation statistic.
2. If an oncogene exhibits a Moran's I value of $+0.78$ ($p < 0.001$), what does this indicate about the spatial clustering of cancer cells?
3. How does spatial cross-validation (spatial block holdout) prevent spatial data leakage when training neural networks to predict gene expression?

### Exercise 2: Computational Complexity of Single-Molecule Spatial Imaging
A 10x Xenium assay runs on an area of $1.5 \times 2.0\,\text{cm}$ measuring a 500-gene panel. The instrument detects 45 million discrete fluorescent puncta (transcripts), each localized with 3D coordinates $(x, y, z)$ alongside DAPI nuclear boundaries.
1. What data structure (e.g., R-tree, KD-tree, SpatialIndex) is required to query which transcripts reside within a specific nuclear boundary in $O(\log N)$ time?
2. What are the storage requirements of raw transcript coordinate tables compared to pre-binned matrix formats?

---

## 9. Key Terms

- **Spatial Transcriptomics:** Technologies that measure gene expression across biological tissue sections while preserving spatial coordinates.
- **Cell-Type Deconvolution:** A statistical technique that decomposes bulk or multi-cellular spot expression into proportions of distinct cell types using a single-cell reference atlas.
- **Tumor Microenvironment (TME):** The cellular environment around a tumor, including blood vessels, immune cells, fibroblasts, signaling molecules, and the extracellular matrix.
- **Spatial Autocorrelation (Moran's I):** A statistical measure of the spatial clustering of values across geographic or physical space.
- **ST-Net / HisToGene:** Deep learning architectures designed to predict high-dimensional spatial gene expression profiles directly from H&E histology image patches.

---

## 10. Sources and Further Reading

- Ståhl, P. L., et al. (2016). "Visualization and Analysis of Gene Expression in Tissue Sections by Spatial Transcriptomics." *Science*, 353(6294), 78–82.
- He, B., et al. (2020). "Integrating Spatial Gene Expression and Breast Morphology via Deep Learning (ST-Net)." *Nature Biomedical Engineering*, 4(8), 827–834.
- Cable, D. M., et al. (2022). "Robust Decomposition of Cell Type Mixtures in Spatial Transcriptomics (RCTD)." *Nature Biotechnology*, 40(4), 517–526.
- Kleshchevnikov, V., et al. (2022). "Cell2location Maps Fine-Grained Cell Types in Spatial Transcriptomics." *Nature Biotechnology*, 40(5), 661–671.
- Moses, L., & Pachter, L. (2022). "Museum of Spatial Transcriptomics." *Nature Methods*, 19(5), 534–546.
