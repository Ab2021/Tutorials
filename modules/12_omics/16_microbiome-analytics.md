# Microbiome Analytics and Metagenomics

> **Module 12 — Omics and Precision Medicine**
> Topic 16 of 16
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, and 06
> Version 2026.3

## 1. Why this topic exists

The human body is home to over 38 trillion microbial cells—collectively termed the **Human Microbiome**—spanning the gastrointestinal tract, skin, oral cavity, and urogenital system. The microbial metagenome encodes over 3.3 million non-redundant bacterial genes, outnumbering the human host genome by more than 150-fold. In 2026, the microbiome has transitioned from an observational descriptive science into a major therapeutic and diagnostic frontier:
1. **Immunotherapy Modulation:** Groundbreaking clinical trials demonstrated that gut microbiome composition dictates whether melanoma and lung cancer patients respond to immune checkpoint inhibitors (anti-PD-1 therapy); fecal microbiota transplantation (FMT) can convert non-responders into responders.
2. **Live Biotherapeutic Products (LBPs):** The FDA approval of standardized live biotherapeutic microbial consortia (e.g., Rebyota, Vowst) for recurrent *Clostridioides difficile* established formal regulatory paths for microbial drugs.
3. **Cardiometabolic & Neurological Biomarkers:** Microbial metabolic byproducts—such as Trimethylamine N-oxide (TMAO), short-chain fatty acids (SCFAs: butyrate, acetate, propionate), and secondary bile acids—modulate atherosclerosis, non-alcoholic steatohepatitis (NASH/MASH), and gut-brain axis neurological signaling.

However, analyzing microbiome sequencing data introduces severe statistical and computational challenges that invalidate standard machine learning and statistical methods:
- **Compositionality:** High-throughput sequencers measure relative proportions, not absolute cell counts. The sum of all taxa in a sample is constrained to an arbitrary simplex ($1.0$ or $100\%$), inducing false negative correlations.
- **Extreme Sparsity:** Metagenomic feature matrices contain 70% to 90% zero values (taxa absent or below sequencing detection limits).
- **Taxonomic Resolution:** Choosing between targeted 16S ribosomal RNA amplicon sequencing (cheap, genus-level) versus whole-genome shotgun (WGS) metagenomics (expensive, species- and strain-level with functional metabolic gene mapping).

Bioinformaticians and data scientists must master compositional data analysis (CoDa), alpha and beta diversity metrics, taxonomic profilers (MetaPhlAn, Kraken2), and differential abundance testing (ANCOM-BC, ALDEx2).

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Compare** 16S rRNA hypervariable region amplicon sequencing with Whole-Genome Shotgun (WGS) metagenomics across cost, resolution, and functional gene profiling.
2. **Execute** microbiome bioinformatics pipelines using QIIME2, DADA2 (for Amplicon Sequence Variants / ASVs), and MetaPhlAn4 / Kraken2 (for taxonomic profiling).
3. **Compute** Alpha Diversity metrics (Shannon index, Simpson, Chao1 richness) and Beta Diversity distance matrices (Bray-Curtis dissimilarity, UniFrac).
4. **Apply** Compositional Data Analysis (CoDa) principles, transforming relative abundance matrices using Centered Log-Ratio (CLR) and Isometric Log-Ratio (ILR) transformations.
5. **Implement** robust differential abundance testing using ANCOM-BC2 (Analysis of Compositions of Microbiomes with Bias Correction) and ALDEx2, avoiding spurious false positives.
6. **Navigate** FDA Center for Biologics Evaluation and Research (CBER) regulatory guidelines for Live Biotherapeutic Products (LBPs).

---

## 3. 16S Amplicon Sequencing vs. Shotgun Metagenomics

Microbiome profiling utilizes two fundamentally distinct molecular sequencing strategies:

```
+-------------------------------------------------------------------------------+
|                      16S rRNA AMPLICON VS. SHOTGUN METAGENOMICS               |
+-------------------------------------------------------------------------------+
| Attribute           | 16S rRNA Amplicon Sequencing  | Whole-Genome Shotgun (WGS) Metagenomics|
+---------------------+-------------------------------+--------------------------------------+
| **Target**          | PCR amplicon of 16S gene      | Unbiased fragmentation of ALL DNA    |
|                     | hypervariable regions (V3-V4) | in biological sample (bacteria, virus)|
| **Resolution**      | Genus level (Rarely species)  | Species, Strain, and Sub-strain level|
| **Functional Info** | Inferred via databases (PICRUSt)| Direct sequencing of metabolic genes |
|                     | (Cannot see novel plasmids)   | (Antibiotic resistance, enzymes)     |
| **Host DNA Noise**  | Zero (primers specific to 16S)| High (in saliva/biopsy, 90%+ host)   |
| **Cost per Sample** | $30 to $60                    | $150 to $400                         |
| **Best Clinical Fit**| Broad community profiling     | Strain-level pathogen virulence & LBP|
+-------------------------------------------------------------------------------+
```

---

## 4. The Compositionality Trap & Centered Log-Ratio (CLR)

The single greatest source of published irreproducibility in microbiome literature is **ignoring compositionality**. Because sequencers have a fixed library capacity, doubling the abundance of Taxon A in a gut sample mathematically forces the measured relative proportions of all other 500 taxa to decrease, even if their absolute biological concentrations remained completely unchanged.

Standard statistical tests (Pearson correlation, standard Student's t-test) produce **up to 80% false-positive associations** on raw compositional tables.

### The Compositional Solution: Aitchison Geometry & CLR
John Aitchison proved that compositional data must be analyzed on the **Simplex ($\mathbb{S}^D$)** using log-ratio transformations. The standard transformation is the **Centered Log-Ratio (CLR)**:

$$\text{CLR}(x) = \left[ \ln\left(\frac{x_1}{g(x)}\right), \ln\left(\frac{x_2}{g(x)}\right), \dots, \ln\left(\frac{x_D}{g(x)}\right) \right]$$

Where $g(x) = \left(\prod_{j=1}^D x_j\right)^{1/D}$ is the geometric mean of all taxa in that sample. The CLR transforms closed simplex proportions into an open, unbounded Euclidean space where standard machine learning algorithms (Random Forests, Lasso, PCA) can be legitimately applied.

---

## 5. Implementation: Microbiome Diversity and CLR Transformation in Python

The following Python script illustrates how a data scientist ingests a taxonomic abundance matrix, computes within-sample Alpha Diversity (Shannon entropy), applies pseudocount handling, and executes a Centered Log-Ratio (CLR) transformation for downstream biomarker classification:

```python
"""
Microbiome Analytics: Alpha Diversity and Centered Log-Ratio (CLR) Transformation.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.stats import entropy

class MicrobiomeAnalyticsEngine:
    def __init__(self, pseudocount: float = 1.0):
        self.pseudocount = pseudocount

    def calculate_alpha_diversity(self, count_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates within-sample Alpha Diversity metrics:
        - Richness: Number of observed taxa
        - Shannon Index: Entropy of taxonomic distribution
        - Gini-Simpson Index: Probability that two randomly sampled microbes belong to different species
        """
        results = []
        for sample_id, row in count_matrix.iterrows():
            counts = row.values
            total_reads = np.sum(counts)
            if total_reads == 0:
                continue
                
            proportions = counts / total_reads
            # Non-zero proportions for entropy
            p_nonzero = proportions[proportions > 0]
            
            shannon_h = entropy(p_nonzero, base=np.e)
            simpson_d = 1.0 - np.sum(proportions ** 2)
            richness = np.sum(counts > 0)
            
            results.append({
                "sample_id": sample_id,
                "total_read_depth": int(total_reads),
                "taxonomic_richness": int(richness),
                "shannon_diversity": round(float(shannon_h), 3),
                "gini_simpson_index": round(float(simpson_d), 3)
            })
            
        return pd.DataFrame(results).set_index("sample_id")

    def centered_log_ratio_transform(self, count_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Applies Aitchison Centered Log-Ratio (CLR) transformation to handle compositionality.
        Adds pseudocount to handle zero-count sparsity.
        """
        # Add pseudocount
        imputed = count_matrix.values + self.pseudocount
        
        # Calculate geometric mean per sample (row-wise)
        # log(geometric_mean) = mean(log(values))
        log_vals = np.log(imputed)
        log_geometric_means = np.mean(log_vals, axis=1, keepdims=True)
        
        # CLR = log(x_i) - log(g(x))
        clr_values = log_vals - log_geometric_means
        
        return pd.DataFrame(clr_values, index=count_matrix.index, columns=count_matrix.columns)

# Example Demonstration
if __name__ == "__main__":
    engine = MicrobiomeAnalyticsEngine(pseudocount=1.0)
    
    # Mock Metagenomic Abundance Table (3 Patients x 5 Key Bacterial Species)
    # Healthy vs. Inflammatory Bowel Disease (IBD) Patient Profiles
    raw_counts = pd.DataFrame([
        # Patient 1: Healthy (High Faecalibacterium prausnitzii, high diversity)
        {"Bacteroides_fragilis": 450, "Faecalibacterium_prausnitzii": 1200, "Akkermansia_muciniphila": 350, "Escherichia_coli": 40, "Prevotella_copri": 800},
        # Patient 2: Active IBD (Dysbiosis, E. coli bloom, depleted Faecalibacterium)
        {"Bacteroides_fragilis": 1100, "Faecalibacterium_prausnitzii": 15, "Akkermansia_muciniphila": 20, "Escherichia_coli": 950, "Prevotella_copri": 0},
        # Patient 3: Post-Antibiotic Washout (Low total diversity)
        {"Bacteroides_fragilis": 80, "Faecalibacterium_prausnitzii": 5, "Akkermansia_muciniphila": 0, "Escherichia_coli": 12, "Prevotella_copri": 0}
    ], index=["PT_Healthy_01", "PT_IBD_Active", "PT_PostAntibiotic"])
    
    # 1. Compute Alpha Diversity
    alpha_df = engine.calculate_alpha_diversity(raw_counts)
    print("=== MICROBIOME ALPHA DIVERSITY ===")
    print(alpha_df[['taxonomic_richness', 'shannon_diversity', 'gini_simpson_index']])
    
    # 2. Compute CLR Transformation
    clr_df = engine.centered_log_ratio_transform(raw_counts)
    print("\n=== CENTERED LOG-RATIO (CLR) COMPOSITIONAL MATRIX ===")
    print(np.round(clr_df, 2))
```

---

## 6. Beta Diversity and Ordination: Bray-Curtis vs. UniFrac

While Alpha Diversity measures diversity within a single patient, **Beta Diversity** quantifies community dissimilarity *between* patients:

```
+-------------------------------------------------------------------------------+
|                       BETA DIVERSITY DISSIMILARITY METRICS                    |
+-------------------------------------------------------------------------------+

1. Bray-Curtis Dissimilarity (Non-Phylogenetic)
   - Measures differences in taxonomic abundance:
     $BC_{jk} = \frac{\sum_i |C_{ij} - C_{ik}|}{\sum_i (C_{ij} + C_{ik})}$
   - Bounded between 0.0 (identical communities) and 1.0 (share zero taxa).

2. Weighted vs. Unweighted UniFrac (Phylogenetic Tree Aware)
   - Incorporates the evolutionary phylogenetic distance between bacterial species.
   - Unweighted UniFrac: Measures fraction of phylogenetic tree branches unique to 
     either sample (sensitive to presence/absence of rare species).
   - Weighted UniFrac: Weights tree branch lengths by relative taxonomic abundance
     (sensitive to shifts in dominant bacterial phyla).

3. Dimensionality Reduction & Visualization
   - Principal Coordinate Analysis (PCoA / Classical Multidimensional Scaling)
   - Maps $N \times N$ pairwise UniFrac distance matrix into 2D/3D scatter plots.
```

---

## 7. Regulatory Standards: Live Biotherapeutic Products (LBPs)

The FDA Center for Biologics Evaluation and Research (CBER) regulates microbiome drugs under the **Live Biotherapeutic Products (LBP)** framework (21 CFR Part 312 / 600):

| Regulatory Attribute | LBP Regulatory Requirement | Bioinformatic Evidence Required |
|---|---|---|
| **Strain Identity & Purity** | Complete genomic identification down to exact sub-strain variant. | Whole-Genome Shotgun (WGS) deep sequencing proving 100% sequence identity to master cell bank. |
| **Antibiotic Resistance Transfer** | The live bacterial strain must not harbor transmissible antimicrobial resistance (AMR) genes. | CARD (Comprehensive Antibiotic Resistance Database) search proving AMR genes are absent or located strictly on non-mobile chromosomal regions, not conjugative plasmids. |
| **Virulence Factor Absence** | Strain must not encode bacterial toxins or invasive enzymes. | Virulence Factor Database (VFDB) screening. |

---

## 8. Exercises and Self-Check

### Exercise 1: Proving Compositional Spurious Correlation
Suppose a microbial community contains three bacteria: Species A ($N_A = 100$), Species B ($N_B = 100$), and Species C ($N_C = 100$). Across 10 patients, Species A and Species B remain completely independent and constant ($r = 0.0$).
1. Suppose in Patient 11, an infection causes Species C to multiply 10-fold ($N_C = 1000$). What happens to the *relative proportions* of Species A and Species B?
2. Explain mathematically why relative proportion normalization induces a false negative correlation between Species A and B. How does the Centered Log-Ratio (CLR) transformation resolve this?

### Exercise 2: Phylogenetic UniFrac Calculation
Consider two patients whose gut microbiomes are composed of completely different species, but all species in both patients belong to the family *Bacteroidaceae*. A third patient is dominated by *Enterococcus faecalis* (Phylum *Firmicutes*).
1. Why will Bray-Curtis dissimilarity treat Patient 1 and Patient 2 as maximally distant ($BC \approx 1.0$), while UniFrac recognizes their close evolutionary similarity?
2. In what clinical scenarios (e.g., FMT engraftment monitoring) is UniFrac vastly superior to non-phylogenetic distance metrics?

---

## 9. Key Terms

- **Microbiome:** The collection of microorganisms (bacteria, archaea, fungi, viruses) living in a particular environment, along with their genetic elements and environmental interactions.
- **Amplicon Sequence Variant (ASV):** An exact single-nucleotide DNA sequence resolved by denoising algorithms (e.g., DADA2), replacing arbitrary 97% similarity Operational Taxonomic Units (OTUs).
- **Compositional Data Analysis (CoDa):** A branch of statistics dealing with data constrained to a constant sum (simplex), utilizing log-ratio transformations.
- **Centered Log-Ratio (CLR):** A mathematical transformation that maps compositional proportions into unconstrained real space by dividing by the geometric mean.
- **Live Biotherapeutic Product (LBP):** A biological product that contains live organisms (such as bacteria) and is intended to prevent, treat, or cure a disease in humans, excluding vaccines.

---

## 10. Sources and Further Reading

- Aitchison, J. (1982). "The Statistical Analysis of Compositional Data." *Journal of the Royal Statistical Society: Series B*, 44(2), 139–160.
- Lin, H., & Peddada, S. D. (2020). "Analysis of Compositions of Microbiomes with Bias Correction (ANCOM-BC)." *Nature Communications*, 11(1), 4114.
- Lozupone, C., & Knight, R. (2005). "UniFrac: An Effective Distance Metric for Comparing Microbial Communities." *Applied and Environmental Microbiology*, 71(12), 8228–8235.
- Food and Drug Administration (FDA). (2016). *Early Clinical Trials with Live Biotherapeutic Products: Chemistry, Manufacturing, and Control Information*. Guidance for Industry.
- Bolyen, E., et al. (2019). "Reproducible, Interactive, Scalable and Extensible Microbiome Data Science Using QIIME 2." *Nature Biotechnology*, 37(8), 852–857.
