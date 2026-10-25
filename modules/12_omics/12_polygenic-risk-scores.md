# Polygenic Risk Scores in Clinical Practice

> **Module 12 — Omics and Precision Medicine**
> Topic 12 of 14
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, and 08
> Version 2026.3

## 1. Why this topic exists

Monogenic genetic disorders—diseases caused by a single high-penetrance mutation, such as *BRCA1* in hereditary breast cancer or *LDLR* in familial hypercholesterolemia—are devastating but rare, explaining less than 5% of chronic disease in the general population. The overwhelming majority of human morbidity and mortality—including Coronary Artery Disease (CAD), Type 2 Diabetes, Breast Cancer, and Schizophrenia—is **polygenic and multifactorial**, driven by the cumulative effect of hundreds or thousands of common genetic variants, each conferring a minuscule increment of risk.

A **Polygenic Risk Score (PRS)** aggregates the effects of millions of Single Nucleotide Polymorphisms (SNPs) across the human genome into a single quantitative metric estimating an individual's lifetime genetic predisposition to a complex disease. In 2026, PRS has crossed the threshold from academic research into clinical translation:
- Health systems use CAD polygenic scores to initiate early statin therapy in 35-year-olds with normal LDL cholesterol who carry top-decile genetic risk.
- Breast cancer screening programs use PRS to stratify mammography initiation age, recommending earlier MRI screening for high-polygenic-risk women without *BRCA* mutations.

However, clinical PRS deployment faces profound computational and ethical hurdles:
1. **The Ancestry Portability Gap:** Because 80%+ of historical Genome-Wide Association Study (GWAS) participants were of European descent, standard PRS models lose **50% to 75% of their predictive accuracy** when evaluated in individuals of African, Hispanic, or South Asian ancestry.
2. **Clinical Utility Metrics:** A high odds ratio does not prove clinical value; models must demonstrate Net Reclassification Improvement (NRI) and positive net clinical benefit via Decision Curve Analysis (DCA).

Biomedical data scientists must understand PRS mathematical formulation (C+T, LDpred2, PRS-CS), multi-ancestry calibration, clinical reporting standards, and the CLIA / LDT regulatory environment.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Formulate** the mathematical foundation of Polygenic Risk Scores using GWAS summary statistics and linkage disequilibrium (LD) matrices.
2. **Implement** modern PRS modeling algorithms comparing Clumping and Thresholding (C+T), Bayesian continuous shrinkage (PRS-CS), and penalized regression (LDpred2).
3. **Quantify** the ancestry performance gap and implement multi-ancestry fine-mapping and score harmonization pipelines.
4. **Evaluate** clinical utility of a PRS using Net Reclassification Improvement (NRI), Integrated Discrimination Improvement (IDI), and Decision Curve Analysis (DCA).
5. **Integrate** PRS metrics with established clinical risk algorithms (e.g., combining CAD PRS with the ACC/AHA 10-Year ASCVD Pooled Cohort Equation).
6. **Navigate** the regulatory status of PRS as Laboratory Developed Tests (LDTs) under the FDA final rule on LDT oversight and CLIA validation standards.

---

## 3. Mathematical Formulation of Polygenic Risk Scores

A Polygenic Risk Score for individual $i$ is calculated as a weighted linear combination of risk-allele counts across $M$ single nucleotide polymorphisms:

$$\text{PRS}_i = \sum_{j=1}^M \beta_j \cdot G_{ij}$$

Where:
- $G_{ij} \in \{0, 1, 2\}$ is the dosage (number of minor/risk alleles) for variant $j$ in individual $i$.
- $\beta_j$ is the per-allele effect size (log odds ratio for binary diseases, or regression coefficient for quantitative traits) estimated from a primary GWAS.

```
+-------------------------------------------------------------------------------+
|                            PRS COMPUTATIONAL PIPELINE                         |
+-------------------------------------------------------------------------------+
1. Primary GWAS Summary Statistics
   - SNP ID (rsID), Chromosome, Position, Effect Allele, Effect Size (beta), P-value
                          |
                          v
2. Linkage Disequilibrium (LD) Modeling & Selection
   - Resolve correlation between neighboring SNPs in the genome
   - Algorithm choices:
     * Method A: Clumping & Thresholding (C+T) -> Prunes correlated SNPs
     * Method B: LDpred2 / PRS-CS -> Shrinks beta weights using Bayesian priors
                          |
                          v
3. Target Cohort Genotype Scoring (PLINK / Hail)
   - Matrix multiplication: [ Patient Genotypes (N x M) ] x [ Adjusted Betas (M x 1) ]
                          |
                          v
4. Standardization & Population Percentile Mapping
   - Normalize against ancestry-matched reference distribution:
     Z = (Raw PRS - Mean_ref) / SD_ref
   - Assign percentile (e.g., 94th percentile -> 2.8x fold increased risk)
```

---

## 4. Modeling Paradigms: C+T vs. LDpred2 vs. PRS-CS

Choosing a PRS algorithm involves balancing statistical assumptions and computational complexity:

| Method | Statistical Principle | LD Handling | Predictive Power | Compute Demands |
|---|---|---|---|---|
| **Clumping & Thresholding (C+T)** | Heuristic pruning | Drops SNPs with $r^2 > \text{threshold}$; selects lead SNP below $p$-value cutoff | Moderate (discards correlated signal) | Very Low (minutes on CPU) |
| **LDpred2** | Bayesian regression with spike-and-slab prior | Adjusts $\beta_j$ conditioned on full LD correlation matrix; models genetic architecture | High | Moderate (requires reference LD panel) |
| **PRS-CS** | Continuous shrinkage (horseshoe) prior | Continuous Bayesian shrinkage adapted across local LD blocks | Highest (robust across diverse architectures) | High (MCMC sampling) |

---

## 5. Implementation: Polygenic Scoring and Standardization in Python

The following Python script illustrates how a genomics pipeline ingests GWAS summary statistics, performs score calculation on a patient genotype matrix, and normalizes scores against a population reference distribution:

```python
"""
Polygenic Risk Score Engine: Scoring, Standardization, and Percentile Mapping.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from scipy.stats import norm

class PolygenicRiskEngine:
    def __init__(self, gwas_weights_df: pd.DataFrame):
        """
        gwas_weights_df columns: ['rsid', 'effect_allele', 'adjusted_beta']
        """
        self.weights = gwas_weights_df.set_index('rsid')['adjusted_beta'].to_dict()

    def calculate_raw_prs(self, genotype_dosages: pd.DataFrame) -> pd.Series:
        """
        genotype_dosages: DataFrame where rows are patients, columns are rsIDs (values: 0, 1, 2)
        """
        valid_snps = [snp for snp in genotype_dosages.columns if snp in self.weights]
        sub_dosages = genotype_dosages[valid_snps]
        
        # Vectorized dot product
        beta_vector = np.array([self.weights[snp] for snp in valid_snps])
        raw_scores = np.dot(sub_dosages.values, beta_vector)
        
        return pd.Series(raw_scores, index=genotype_dosages.index, name="raw_prs")

    def standardize_and_map_percentile(
        self, 
        raw_scores: pd.Series, 
        ancestry_mean: float, 
        ancestry_std: float
    ) -> pd.DataFrame:
        """
        Transforms raw score to Z-score and percentile using ancestry reference distribution.
        """
        z_scores = (raw_scores - ancestry_mean) / ancestry_std
        percentiles = norm.cdf(z_scores) * 100.0
        
        # Approximate Relative Risk compared to median (assuming log-normal risk distribution)
        relative_risks = np.exp(z_scores * 0.40) # 0.40 is typical SD effect size in CAD
        
        results = pd.DataFrame({
            "raw_prs": raw_scores,
            "z_score": np.round(z_scores, 2),
            "percentile": np.round(percentiles, 1),
            "relative_risk": np.round(relative_risks, 2)
        })
        
        # Clinical Risk Stratification Category
        results['risk_tier'] = pd.cut(
            results['percentile'], 
            bins=[-0.1, 20.0, 80.0, 95.0, 100.1],
            labels=['Low Risk', 'Average Risk', 'High Risk', 'Very High Risk']
        )
        
        return results

# Example Demonstration
if __name__ == "__main__":
    # 1. Mock GWAS Summary Statistics (CAD Risk Alleles)
    mock_gwas = pd.DataFrame([
        {"rsid": "rs10757274", "effect_allele": "G", "adjusted_beta": 0.28}, # 9p21 locus
        {"rsid": "rs1333049",  "effect_allele": "C", "adjusted_beta": 0.22},
        {"rsid": "rs20455",    "effect_allele": "G", "adjusted_beta": 0.15},
        {"rsid": "rs9349379",  "effect_allele": "A", "adjusted_beta": 0.18}
    ])
    
    # 2. Mock Patient Genotypes (Dosages: 0, 1, or 2 risk alleles)
    mock_patients = pd.DataFrame([
        {"rs10757274": 2, "rs1333049": 2, "rs20455": 1, "rs9349379": 2}, # High risk patient
        {"rs10757274": 0, "rs1333049": 1, "rs20455": 0, "rs9349379": 0}, # Low risk patient
        {"rs10757274": 1, "rs1333049": 1, "rs20455": 1, "rs9349379": 1}  # Average patient
    ], index=["PT_Alpha", "PT_Beta", "PT_Gamma"])
    
    engine = PolygenicRiskEngine(mock_gwas)
    raw = engine.calculate_raw_prs(mock_patients)
    
    # Reference population distribution (simulated from 1000 Genomes EUR)
    report = engine.standardize_and_map_percentile(raw, ancestry_mean=0.83, ancestry_std=0.35)
    
    print("=== POLYGENIC RISK SCORE CLINICAL STRATIFICATION ===")
    print(report[['z_score', 'percentile', 'relative_risk', 'risk_tier']])
```

---

## 6. The Ancestry Portability Crisis & Multi-Ancestry PRS

The central technical and ethical vulnerability of polygenic risk scoring in 2026 is **Ancestry Transferability Decay**:

```
Prediction Accuracy (Relative R^2 or AUC) across Ancestries (Standard European GWAS PRS)
100% |======================================== [ European Ancestry (Reference) ]
 80% |
 60% |======================== [ South Asian Ancestry (~60% transferability) ]
 40% |================ [ East Asian Ancestry (~45% transferability) ]
 20% |======== [ African Ancestry (~25% transferability) ]
  0% +--------------------------------------------------------------------------
```

### Why Does PRS Fail Across Ancestries?
1. **Linkage Disequilibrium (LD) Diversity:** Tagged SNPs identified in European GWAS are not in LD with the true causal variant in African populations, which possess much older genomes with smaller, highly fragmented LD blocks.
2. **Allele Frequency Divergence:** A variant that is common in Europe ($MAF = 0.25$) may be exceedingly rare ($MAF < 0.01$) or absent in Africa.
3. **Clinical Hazard:** Deploying an uncalibrated European PRS in an urban safety-net hospital results in massive false reassurance for African American and Hispanic patients, who may harbor high genetic disease risk that the European-trained score completely fails to measure.

### Engineering Mitigations
Modern workflows implement **Multi-Ancestry Bayesian Models (PRS-CSx, Multi-PRS)** that jointly model GWAS summary statistics from diverse ancestral cohorts (e.g., Million Veteran Program, BioBank Japan, All of Us), allowing local LD matrices and causal weights to vary across ancestral genetic backgrounds.

---

## 7. Clinical Integration: Decision Curve Analysis and Integrated Risk Scores

In clinical cardiology, a PRS is never interpreted in isolation; it is integrated with traditional clinical risk calculators (e.g., the **ACC/AHA ASCVD Risk Estimator**):

$$\text{Integrated 10-Year Risk} = 1 - (1 - \text{ClinicalRisk})^{\text{RelativeRisk}_{\text{PRS}}}$$

```
+-------------------------------------------------------------------------------+
|                    CLINICAL RECLASSIFICATION VIA PRS INTEGRATION              |
+-------------------------------------------------------------------------------+
Patient Case: 42-year-old male, non-smoker, LDL 128 mg/dL
  - Baseline ACC/AHA 10-Year ASCVD Risk: 4.8% ("Borderline Risk" -> No Statin)
  - Genomic Profiling: CAD Polygenic Risk Score in 97th percentile (RR = 2.4)
  - Integrated 10-Year Risk: 11.1% ("High Risk" -> Initiates Statin Therapy)
  - Clinical Outcome: Prevents early myocardial infarction at age 48.
```

---

## 8. Exercises and Self-Check

### Exercise 1: Net Reclassification Improvement (NRI)
Suppose a clinical guidelines committee is evaluating whether to add a Breast Cancer PRS to the Gail Model.
1. Define **Net Reclassification Improvement (NRI)** for cases and controls.
2. If the integrated model reclassifies 12% of women who subsequently develop cancer into a higher-risk category warranting earlier MRI, but also falsely up-classifies 4% of women who remain cancer-free, what is the net clinical benefit?

### Exercise 2: Auditing Ancestry Reference Panels
A commercial direct-to-consumer genomics company standardizes all customers against a single reference distribution drawn from the UK Biobank (95% British European ancestry).
1. Explain mathematically why an individual of West African ancestry will receive an artificially shifted Z-score and distorted percentile rank.
2. How does the American College of Medical Genetics and Genomics (ACMG) 2024 guidance regulate ancestry-matched reference reporting?

---

## 9. Key Terms

- **Polygenic Risk Score (PRS):** A numerical score reflecting an individual's cumulative genetic predisposition to a complex trait or disease based on thousands or millions of variants.
- **Linkage Disequilibrium (LD):** The non-random association of alleles at different loci in a given population.
- **Clumping and Thresholding (C+T):** A heuristic method that selects significant independent SNPs while discarding variants in linkage disequilibrium.
- **Ancestry Portability Gap:** The substantial reduction in PRS prediction accuracy observed when a model trained in one ancestral population (typically European) is applied to individuals of different ancestral backgrounds.
- **Decision Curve Analysis (DCA):** A statistical method to evaluate whether a clinical prediction model improves decision-making across a range of clinical threshold probabilities.

---

## 10. Sources and Further Reading

- Khera, A. V., et al. (2018). "Genome-wide Polygenic Scores for Common Diseases Identify Individuals with Risk Equivalent to Monogenic Mutations." *Nature Genetics*, 50(9), 1219–1224.
- Martin, A. R., et al. (2019). "Clinical Use of Current Polygenic Risk Scores May Exacerbate Health Disparities." *Nature Genetics*, 51(4), 584–591.
- Ge, T., et al. (2019). "Polygenic Prediction via Bayesian Regression and Continuous Shrinkage Priors (PRS-CS)." *Nature Communications*, 10(1), 1776.
- Privé, F., et al. (2020). "LDpred2: Better, Faster, Stronger." *Bioinformatics*, 36(22-23), 5424–5431.
- Wand, H., et al. (2021). "Improving Reporting Standards for Polygenic Scores in Risk Prediction Studies: The PRS-RS Framework." *Nature*, 591(7849), 211–219.
