# Module 22 — Sources and Further Reading

> **Program:** Biomedical Data Science for Industry (2026)
> **Source cutoff:** 24 October 2026
> **Use:** Start here when you need a primary source, a regulatory text, a dataset, or a professional community. Verify time-sensitive items before citing.

---

## 1. How to read this list

Sources are grouped by use:

- **Primary regulation and policy:** use for compliance questions.
- **Methods:** use for study design and analysis.
- **Clinical background:** use for vocabulary and plausibility.
- **Datasets and benchmarks:** use for labs and replication.
- **Industry evidence:** use for market and product context.
- **Professional communities:** use for staying current.

Prefer primary sources. A news article about a guidance is not the guidance.

## 2. Supplied clinical library

| Book | Use in this course | Caution |
|---|---|---|
| Bickley, *Bates' Guide to Physical Examination and History Taking* | History, exam, documentation, clinical reasoning | Not a data-science methods text |
| Goldberger, *Clinical Electrocardiography* | ECG interpretation, clinical correlation | Pattern recognition; pair with signal processing |
| Thaler, *The Only EKG Book You'll Ever Need* | ECG fundamentals | Introductory |
| Herring, *Learning Radiology* | Modality overview, common findings | Not a substitute for radiology training |
| Lilly, *Pathophysiology of Heart Disease* | Cardiovascular physiology | Disease mechanisms, not ML |
| West, *Respiratory Physiology* | Ventilation, gas exchange, acid-base | Physiological principles |
| Blumenfeld, *Neuroanatomy through Clinical Cases* | Localization, neuro exam | Case-based |
| Jankovic, *Bradley and Daroff's Neurology in Clinical Practice* | Neurologic disease and diagnostics | Reference text |
| Armstrong, *The Power of Neurodiversity* | Neurodiversity-affirming language and design | Social/ethical lens |
| Schwinger, "Pathophysiology of heart failure," *Cardiovasc Diagn Ther*, 2021 | Heart failure mechanisms | Review article |

## 3. Regulation and policy

### United States

- FDA, *Artificial Intelligence-Enabled Medical Devices* (device list and program page). https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
- FDA, *Good Machine Learning Practice for Medical Device Development: Guiding Principles*. https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/good-machine-learning-practice-medical-device-development-guiding-principles
- FDA, *Marketing Submission Recommendations for a Predetermined Change Control Plan for AI-Enabled Device Software Functions*. https://www.fda.gov/media/166704/download
- FDA, *Clinical Decision Support Software*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/clinical-decision-support-software
- FDA, *Quality Management System Regulation (QMSR)*. https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr
- FDA, *Computer Software Assurance for Production and Quality System Software*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/computer-software-assurance-production-and-quality-management-system-software
- FDA, *Considerations for the Use of Artificial Intelligence to Support Regulatory Decision-Making for Drug and Biological Products*. https://www.fda.gov/regulatory-information/search-fda-guidance-documents/considerations-use-artificial-intelligence-support-regulatory-decision-making-drug-and-biological
- FDA, *Generative AI-Enabled Medical Devices: Discussion Paper and Request for Comment*, 2026. https://www.fda.gov/medical-devices/digital-health-center-excellence/considerations-regulation-generative-ai-enabled-medical-devices-discussion-paper-and-request
- FDA, *Real-World Evidence*. https://www.fda.gov/science-research/science-and-research-special-topics/real-world-evidence
- FDA, *Cybersecurity*. https://www.fda.gov/medical-devices/digital-health-center-excellence/cybersecurity
- FDA 510(k) database. https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/pmn.cfm
- FDA De Novo DEN180001 (IDx-DR). https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180001.pdf
- FDA De Novo DEN180044 (Apple ECG). https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN180044.pdf
- HHS, *HIPAA Privacy Rule*. https://www.hhs.gov/hipaa/for-professionals/privacy/index.html
- HHS, *HIPAA Security Rule*. https://www.hhs.gov/hipaa/for-professionals/security/index.html
- ONC, *HTI-1 Final Rule*. https://www.healthit.gov/topic/laws-regulation-and-policy/health-data-technology-and-interoperability-certification-program
- CMS, *Interoperability and Prior Authorization*. https://www.cms.gov/priorities/electronic-prior-authorization/overview
- CMS, *Coverage with Evidence Development*. https://www.cms.gov/medicare/coverage/coverage-with-evidence-development

### European Union and international

- EU MDR 2017/745. https://eur-lex.europa.eu/eli/reg/2017/745/oj
- EU IVDR 2017/746. https://eur-lex.europa.eu/eli/reg/2017/746/oj
- EU AI Act 2024/1689. https://eur-lex.europa.eu/eli/reg/2024/1689/oj
- European Commission, *AI Act policy page*. https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai
- MDCG 2025-6, *AI Act and medical devices*. https://health.ec.europa.eu/system/files/2025-06/mdcg_2025-6_en.pdf
- EHDS Regulation 2025/327. https://health.ec.europa.eu/ehealth-digital-health-and-care/european-health-data-space-regulation-ehds_en
- ICH, *E9(R1) Estimands*. https://www.ich.org/
- ICH, *GCP*. https://www.ich.org/page/efficacy-guidelines
- IMDRF. https://www.imdrf.org/
- ISO 13485. https://www.iso.org/standard/59752.html
- ISO 14971. https://www.iso.org/standard/72704.html
- IEC 62304. https://www.iec.ch/
- IEC 62366. https://www.iec.ch/

### Standards and interoperability

- HL7 FHIR. https://hl7.org/fhir/
- OMOP CDM. https://ohdsi.github.io/CommonDataModel/
- DICOM. https://www.dicomstandard.org/
- LOINC. https://loinc.org/
- SNOMED CT. https://www.snomed.org/
- RxNorm. https://www.nlm.nih.gov/research/umls/rxnorm/
- UCUM. https://ucum.org/
- CDISC. https://www.cdisc.org/
- USCDI. https://www.healthit.gov/isp/united-states-core-data-for-interoperability-uscdi
- HPO. https://hpo.jax.org/

## 4. Methods

### Statistics and causal inference
- Hernán, M. A., Robins, J. M., *Causal Inference: What If*. https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/
- Hernán, M. A., Robins, J. M., "Using Big Data to Emulate a Target Trial," *American Journal of Epidemiology*, 2016. https://doi.org/10.1093/aje/kwv254
- Rosenbaum, P. R., Rubin, D. B., "The central role of the propensity score," *Biometrika*, 1983. https://doi.org/10.1093/biomet/70.1.41
- Pearl, J., *Causality*.
- Imbens, G. W., Rubin, D. B., *Causal Inference for Statistics, Social, and Biomedical Sciences*.
- Harrell, F. E., *Regression Modeling Strategies*.
- Steyerberg, E. W., *Clinical Prediction Models*.
- Little, R. J. A., Rubin, D. B., *Statistical Analysis with Missing Data*.
- Efron, B., Tibshirani, R. J., *An Introduction to the Bootstrap*.
- Vickers, A. J., Elkin, E. B., "Decision curve analysis," *Medical Decision Making*, 2006. https://doi.org/10.1177/0272989X06295361

### Machine learning and evaluation
- Hastie, T., Tibshirani, R., Friedman, J., *The Elements of Statistical Learning*.
- James, G. et al., *An Introduction to Statistical Learning*.
- LeCun, Y., Bengio, Y., Hinton, G., "Deep learning," *Nature*, 2015. https://doi.org/10.1038/nature14539
- Vaswani, A. et al., "Attention Is All You Need," *NeurIPS*, 2017. https://arxiv.org/abs/1706.03762
- TRIPOD+AI. https://www.tripod-statement.org/
- CONSORT-AI / SPIRIT-AI. https://www.consort-spirit.org/
- STARD-AI. https://www.equator-network.org/
- PROBAST. https://www.probast.org/
- CLAIM. https://www.equator-network.org/
- Model Cards. https://arxiv.org/abs/1810.03993
- Datasheets for Datasets. https://arxiv.org/abs/1803.09010

### Clinical NLP and LLMs
- n2c2. https://n2c2.dbmi.hms.harvard.edu/
- i2b2. https://www.i2b2.org/
- Savova, G. K. et al., "Mayo clinical Text Analysis and Knowledge Extraction System (cTAKES)," *JAMIA*, 2010. https://doi.org/10.1136/jamia.2009.001560
- Chapman, W. W. et al., "A simple algorithm for identifying negated findings," *Journal of Biomedical Informatics*, 2001. https://doi.org/10.1006/jbin.2001.1029
- Kraljevic, Z. et al., "MedCAT," *Artificial Intelligence in Medicine*, 2021. https://doi.org/10.1016/j.artmed.2021.102083
- Singhal, K. et al., "Large language models encode clinical knowledge," *Nature*, 2023. https://doi.org/10.1038/s41586-023-06291-2
- Thirunavukarasu, A. J. et al., "Large language models in medicine," *Nature Medicine*, 2023. https://doi.org/10.1038/s41591-023-02448-8
- HealthBench. https://openai.com/index/healthbench/
- MedAgentBench. https://stanfordmlgroup.github.io/projects/medagentbench/
- MedHELM. https://medhelm.org/

### Imaging and signals
- Irvin, J. et al., "CheXpert," *AAAI*, 2019. https://doi.org/10.1609/aaai.v33i01.3301590
- Ma, J. et al., "Segment anything in medical images," *Nature Communications*, 2024. https://doi.org/10.1038/s41467-024-44824-z
- McKinney, S. M. et al., "International evaluation of an AI system for breast cancer screening," *Nature*, 2020. https://doi.org/10.1038/s41586-019-1799-6
- Perez, M. V. et al., "Large-Scale Assessment of a Smartwatch to Identify Atrial Fibrillation," *NEJM*, 2019. https://doi.org/10.1056/NEJMoa1901183
- Chinoy, E. D. et al., "Performance of seven consumer sleep-tracking devices," *Sleep*, 2021. https://doi.org/10.1093/sleep/zsaa291
- PhysioNet. https://physionet.org/
- MONAI. https://monai.io/

### Omics
- Jumper, J. et al., "Highly accurate protein structure prediction with AlphaFold," *Nature*, 2021. https://doi.org/10.1038/s41586-021-03819-2
- Theodoris, C. V. et al., "Transfer learning enables predictions in network biology," *Nature*, 2023. https://doi.org/10.1038/s41586-023-06139-9
- Cui, H. et al., "scGPT," *Nature Methods*, 2024. https://doi.org/10.1038/s41592-024-02201-0
- Richards, S. et al., "Standards and guidelines for the interpretation of sequence variants," *Genetics in Medicine*, 2015. https://doi.org/10.1038/gim.2015.30
- Torkamani, A. et al., "The personal and clinical utility of polygenic risk scores," *Nature Reviews Genetics*, 2018. https://doi.org/10.1038/s41576-018-0018-x
- Open Targets. https://www.opentargets.org/
- Open Problems. https://openproblems.bio/
- CAGI. https://genomeinterpretation.org/

## 5. Clinical background

- Bickley, L. S., *Bates' Guide to Physical Examination and History Taking*.
- Goldberger, A. L. et al., *Goldberger's Clinical Electrocardiography*.
- Thaler, M. S., *The Only EKG Book You'll Ever Need*.
- Herring, W., *Learning Radiology*.
- Lilly, L. S. et al., *Pathophysiology of Heart Disease*.
- West, J. B. et al., *West's Respiratory Physiology*.
- Blumenfeld, H., *Neuroanatomy through Clinical Cases*.
- Jankovic, J. et al., *Bradley and Daroff's Neurology in Clinical Practice*.
- Armstrong, T., *The Power of Neurodiversity*.
- Schwinger, R. H. G., "Pathophysiology of heart failure," *Cardiovasc Diagn Ther*, 2021. https://doi.org/10.21037/cdt-20-302
- Royal College of Physicians, *NEWS2*. https://www.rcp.ac.uk/improving-care/resources/national-early-warning-score-news-2/
- Surviving Sepsis Campaign. https://www.sccm.org/SurvivingSepsisCampaign/Guidelines
- GOLD reports. https://goldcopd.org/
- WHO, *Global report on health equity for persons with disabilities*, 2022. https://www.who.int/publications/i/item/9789240063600

## 6. Datasets and benchmarks

| Source | Use | Access |
|---|---|---|
| MIMIC-IV | ICU/EHR | Credentialed PhysioNet |
| eICU-CRD | Multi-site ICU | Credentialed PhysioNet |
| MIMIC-CXR | Imaging + reports | Credentialed |
| CheXpert | Chest X-ray | Registration |
| BraTS | Brain MRI | Challenge registration |
| PTB-XL | ECG | Open |
| Sleep-EDF | Sleep | Open |
| TUH EEG | EEG | Registration |
| UK Biobank | Population omics/EHR | Application/fee |
| All of Us | Diverse cohort | Controlled tier |
| TCGA | Cancer multi-omics | Open/controlled |
| CELLxGENE | Single-cell | Open |
| Open Targets | Target evidence | Open |
| ChEMBL | Bioactivity | Open |
| PubChem | Chemistry | Open |
| OpenFDA/AEMS | Safety | Open |
| CMS Open Payments | Payments | Open |
| HCUP | Discharge/ED | Purchase/training |
| NHANES | Population health | Open |
| Synthea | Synthetic EHR | Open |
| Eunomia | OMOP synthetic | Open |
| HealthBench | LLM evaluation | Open |
| EHRSHOT | Few-shot EHR | Credentialed |
| FoMoH | EHR foundation benchmark | Code/data links |

## 7. Industry and market

- Kaufman Hall, *National Hospital Flash Report*. https://www.kaufmanhall.com/
- KLAS Research, *Healthcare AI Update 2025*. https://klasresearch.com/
- KLAS, *Arch Collaborative 2026*. https://klasresearch.com/archcollaborative/
- HFMA, *Hospital Operating Margin Trends*. https://www.hfma.org/
- AHA, *Costs of Caring 2026*. https://www.aha.org/
- Epic Cosmos. https://cosmos.epic.com/
- Rock Health, *Digital Health Funding*. https://rockhealth.com/
- BLS, *Data Scientists*. https://www.bls.gov/ooh/math/data-scientists.htm
- LinkedIn, *Jobs on the Rise 2026*. https://www.linkedin.com/pulse/linkedin-jobs-rise-2026-25-fastest-growing-roles-us-linkedin-news-dlb1c
- FDA, *AI-enabled medical devices*. https://www.fda.gov/medical-devices/digital-health-center-excellence/artificial-intelligence-enabled-medical-devices
- *JAMA Network Open*, imaging AI evidence, 2025. https://jamanetwork.com/journals/jamanetworkopen/fullarticle/2833324
- *JAMA Health Forum*, AI device recalls, 2025. https://jamanetwork.com/journals/jama-health-forum/fullarticle/2837802

## 8. Professional communities and continuing education

- AMIA: https://amia.org/
- HIMSS: https://www.himss.org/
- OHDSI: https://ohdsi.org/
- ML4H: https://ml4h.cc/
- CHIL: https://www.cilhealth.org/
- MICCAI: https://www.miccai.org/
- RSNA: https://www.rsna.org/
- NeurIPS: https://neurips.cc/
- ACM Conference on Health, Inference, and Learning (CHIL): https://www.cilhealth.org/
- Digital Medicine Society: https://www.dimesociety.org/
- Coalition for Health AI: https://www.chai.org/
- Fast.ai: https://course.fast.ai/
- Coursera/edX: courses in biostatistics, epidemiology, clinical informatics.

## 9. Newsletters and ongoing sources

- FDA guidance updates.
- ONC HTI updates.
- CMS rules.
- *NEJM AI*.
- *Nature Medicine*.
- *Lancet Digital Health*.
- *JAMA* and *JAMA Network Open*.
- *npj Digital Medicine*.
- *Journal of the American Medical Informatics Association*.
- *Medical Image Analysis*.
- *Computers in Biology and Medicine*.
- *Health Affairs*.
- *Stat News*.
- *Fierce Healthcare*.
- *The Lancet*.

## 10. Evaluation checklist

Use this before citing any source:

- [ ] Is it primary?
- [ ] Is it dated within the relevant period?
- [ ] Does it state population, setting, and outcome?
- [ ] Does it report uncertainty?
- [ ] Is it peer-reviewed or regulatory?
- [ ] Is there a conflict of interest?
- [ ] Does the claim match the evidence?
- [ ] Is the access date recorded?
- [ ] Is the URL stable?
- [ ] Is the license compatible with use?

## 11. Keeping current

- Set Google Scholar alerts for key authors and topics.
- Follow FDA, EMA, ONC, CMS, and NICE update pages.
- Join OHDSI, AMIA, and ML4H communities.
- Read one primary paper per week.
- Keep a source log with dates and why each source matters.
- Re-verify regulatory facts every 6 months.

## 12. Final note

The field changes quickly. The durable skills are framing, evidence, validation, communication, and governance. Use this bibliography as a starting point, not a fixed canon.

> **End of core modules.** Return to `README.md` for the program map and capstone.
