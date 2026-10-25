# Module 21 — Datasets, Tools, and Benchmarks

> **Part of:** Biomedical Data Science for Industry (2026), expanded edition.
> **Survey file:** [`../../course/21_datasets_tools_benchmarks.md`](../../course/21_datasets_tools_benchmarks.md)
> **Folder:** `modules/21_datasets/`
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
| Prerequisites | 02 |
| Unlocks | 19 |
| Assessment | Dataset and benchmark dossier (Week 31) |
| Survey file | `21_datasets_tools_benchmarks.md` |

## 3. Topics

| # | File | Topic | Scope | Target | Status |
|---:|---|---|---|---:|---|
| 1 | `01_five-gate-decision.md` | The Five-Gate Dataset Decision | Access, licence, fitness, bias, and reproducibility gates. | 1,200-1,800 | planned |
| 2 | `02_access-and-licensing.md` | Access and Licensing | Credentialing, DUAs, BAAs, and commercial restrictions. | 1,200-1,800 | planned |
| 3 | `03_ehr-claims-and-icu.md` | EHR, Claims, and ICU Data | MIMIC, eICU, All of Us, NHANES, Marketscan-like sources, and their limits. | 1,200-1,800 | planned |
| 4 | `04_imaging-datasets.md` | Imaging Datasets | Public and credentialed imaging collections and their label quality. | 1,200-1,800 | planned |
| 5 | `05_signal-datasets.md` | Signal Datasets | PhysioNet collections, wearable corpora, and annotation depth. | 1,200-1,800 | planned |
| 6 | `06_omics-datasets.md` | Omics Datasets | TCGA, GTEx, single-cell atlases, and access terms. | 1,200-1,800 | planned |
| 7 | `07_pathology-datasets.md` | Pathology Datasets | Whole-slide collections and the annotation bottleneck. | 1,200-1,800 | planned |
| 8 | `08_text-and-llm-benchmarks.md` | Text and LLM Benchmarks | Clinical QA sets, summarization sets, and contamination risk. | 1,200-1,800 | planned |
| 9 | `09_public-health-datasets.md` | Public Health Datasets | Surveillance, vital statistics, and geographic data. | 1,200-1,800 | planned |
| 10 | `10_pharma-and-safety-datasets.md` | Pharma and Safety Datasets | Trial data sharing, FAERS/AEMS, and pharmacovigilance corpora. | 1,200-1,800 | planned |
| 11 | `11_tool-stack.md` | Tool Stack | What to use for each function and what to avoid. | 1,200-1,800 | planned |
| 12 | `12_benchmark-pitfalls.md` | Benchmark Pitfalls | Leaderboard chasing, contamination, and non-comparable splits. | 1,200-1,800 | planned |

## 4. How the topics fit together

Each topic is written to stand alone, but they are ordered so that later topics can assume the
vocabulary of earlier ones. If you are short on time, the minimum path is the first topic plus
any topic that names your current decision problem.

```text
01 The Five-Gate Dataset Decision ->
02 Access and Licensing ->
03 EHR, Claims, and ICU Data ->
04 Imaging Datasets ->
05 Signal Datasets ->
06 Omics Datasets ->
07 Pathology Datasets ->
08 Text and LLM Benchmarks ->
09 Public Health Datasets ->
10 Pharma and Safety Datasets ->
11 Tool Stack ->
12 Benchmark Pitfalls
```

## 5. What this module does not cover

- Hands-on lab execution: see `labs/` and the lab sections of the survey file.
- Legal advice: regulatory content is educational and requires professional review.
- Clinical validation: no topic file here validates a product or a model.

## 6. Status

| Status | Count |
|---|---:|
| planned | 12 |
| draft | 0 |
| complete | 0 |

Regenerate this table with `python tools/scaffold_modules.py --status`.

## 7. Sources for this module

See section 17 of the survey file and the per-topic source sections.

