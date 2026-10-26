# Module 12 — Omics and Precision Medicine

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/12_omics_precision_medicine.md`](../../course/12_omics_precision_medicine.md)
> **Folder:** `modules/12_omics/`
> **Version:** 2026.3

---

## 1. Module purpose

The survey file gives the module's narrative arc in one pass. This folder breaks that arc into
teaching units. Each topic file is a self-contained 2-4 hour block with mechanics, a worked
example, code or calculation, failure modes, assumptions, and exercises.

Read `_index.md` first, then the topics in order. Topics assume the prerequisites listed below,
not the whole program.

## 2. Position in the program

| Item | Value |
|---|---|
| Prerequisites | 01A, 04 |
| Unlocks | 13, 23 |
| Assessment | Biomarker evidence plan (Week 20) |
| Survey file | `12_omics_precision_medicine.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | [`01_data-types.md`](01_data-types.md) | Data Types | Genomics, transcriptomics, epigenomics, proteomics, metabolomics, and microbiomics. | 1,200-1,800 | complete |
| 2 | [`02_genomics-and-variants.md`](02_genomics-and-variants.md) | Genomics and Variants | Sequencing, alignment, calling, annotation, and clinical interpretation. | 1,200-1,800 | complete |
| 3 | [`03_transcriptomics.md`](03_transcriptomics.md) | Transcriptomics | Bulk RNA-seq, normalization, differential expression, and batch effects. | 1,200-1,800 | complete |
| 4 | [`04_single-cell.md`](04_single-cell.md) | Single-Cell | Cell capture, QC, clustering, annotation, and trajectory analysis. | 1,200-1,800 | complete |
| 5 | [`05_spatial-transcriptomics.md`](05_spatial-transcriptomics.md) | Spatial Transcriptomics | Spatial assays, deconvolution, and tissue architecture. | 1,200-1,800 | complete |
| 6 | [`06_proteomics-and-metabolomics.md`](06_proteomics-and-metabolomics.md) | Proteomics and Metabolomics | Mass spectrometry, quantification, and identification confidence. | 1,200-1,800 | complete |
| 7 | [`07_multi-omics-integration.md`](07_multi-omics-integration.md) | Multi-Omics Integration | Early, intermediate, and late fusion; when integration helps. | 1,200-1,800 | complete |
| 8 | [`08_biomarker-discovery.md`](08_biomarker-discovery.md) | Biomarker Discovery | Discovery, qualification, validation, and the replication crisis. | 1,200-1,800 | complete |
| 9 | [`09_drug-discovery.md`](09_drug-discovery.md) | Drug Discovery | Target identification, screening, ADMET, and AI's actual role. | 1,200-1,800 | complete |
| 10 | [`10_clinical-genomics.md`](10_clinical-genomics.md) | Clinical Genomics | ACMG classification, reporting, and genetic counseling interfaces. | 1,200-1,800 | complete |
| 11 | [`11_pharmacogenomics-cds-and-cpic.md`](11_pharmacogenomics-cds-and-cpic.md) | Pharmacogenomics Clinical Decision Support and CPIC Integration | Over **99% of individuals** carry at least one clinically actionable genetic variant that significantly alters drug m... | 1,200-1,800 | draft (258 lines) |
| 12 | [`12_polygenic-risk-scores.md`](12_polygenic-risk-scores.md) | Polygenic Risk Scores in Clinical Practice | Monogenic genetic disorders—diseases caused by a single high-penetrance mutation, such as *BRCA1* in hereditary breas... | 1,200-1,800 | draft (253 lines) |
| 13 | [`13_spatial-omics-and-pathology-integration.md`](13_spatial-omics-and-pathology-integration.md) | Spatial Omics and Pathology Integration | For decades, biological understanding of human disease was polarized between two incompatible resolutions: | 1,200-1,800 | draft (288 lines) |
| 14 | [`14_ai-for-protein-engineering.md`](14_ai-for-protein-engineering.md) | AI for Protein Engineering and Structural Biology | For over fifty years, the "Protein Folding Problem"—predicting the unique three-dimensional tertiary structure of a p... | 1,200-1,800 | draft (248 lines) |
| 15 | [`15_crispr-and-gene-therapy-pipelines.md`](15_crispr-and-gene-therapy-pipelines.md) | CRISPR and Gene Therapy Data Pipelines | The commercial approval of the first CRISPR-Cas9 curative gene-editing therapy (Casgevy / exagamglogene autotemcel fo... | 1,200-1,800 | draft (259 lines) |
| 16 | [`16_microbiome-analytics.md`](16_microbiome-analytics.md) | Microbiome Analytics and Metagenomics | The human body is home to over 38 trillion microbial cells—collectively termed the **Human Microbiome**—spanning the... | 1,200-1,800 | draft (241 lines) |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 Data Types ->
02 Genomics and Variants ->
03 Transcriptomics ->
04 Single-Cell ->
05 Spatial Transcriptomics ->
06 Proteomics and Metabolomics ->
07 Multi-Omics Integration ->
08 Biomarker Discovery ->
09 Drug Discovery ->
10 Clinical Genomics ->
11 Pharmacogenomics Clinical Decision Support and CPIC Integration ->
12 Polygenic Risk Scores in Clinical Practice ->
13 Spatial Omics and Pathology Integration ->
14 AI for Protein Engineering and Structural Biology ->
15 CRISPR and Gene Therapy Data Pipelines ->
16 Microbiome Analytics and Metagenomics
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 0 |
| draft | 6 |
| complete | 10 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

