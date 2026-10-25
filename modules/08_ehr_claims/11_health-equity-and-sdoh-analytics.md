# Health Equity Analytics and Social Determinants of Health

> **Module 08 — EHR, Claims, and Operations Analytics**
> Topic 11 of 11
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, 03, and 06
> Version 2026.3

## 1. Why this topic exists

Clinical care accounts for only **10% to 20%** of modifiable health outcomes. The remaining **80% to 90%** of health status, disease incidence, premature mortality, and readmission risk are driven by **Social Determinants of Health (SDOH)**—the conditions in the environments where people are born, live, learn, work, play, and age. Factors such as housing instability, food insecurity, lack of reliable transportation, environmental toxicant exposure, and neighborhood economic deprivation govern whether a diabetic patient can afford insulin, store it under refrigeration, or access a primary care clinic.

Historically, predictive models in healthcare (e.g., readmission risk scores, emergency department utilization forecasts) ignored SDOH data, relying exclusively on clinical lab values and billing codes. This created two disastrous consequences:
1. **Model Inaccuracy and Blind Spots:** Models routinely failed to identify vulnerable patients whose clinical biomarkers appeared stable but who lacked transportation to pick up discharge medications.
2. **Algorithmic Bias Amplification:** Prominently exposed in a landmark *Science* study (Obermeyer et al., 2019), commercial risk-scoring algorithms used healthcare cost as a proxy for healthcare need. Because less money is spent on Black patients relative to white patients with identical disease burdens due to unequal access and systemic bias, the algorithm systematically assigned Black patients lower risk scores, disqualifying them from specialized care management programs.

In 2026, healthcare payers, accountable care organizations (ACOs), and health systems operate under strict mandates from CMS (CMS Health Equity Framework) and the Joint Commission to systematically collect, link, analyze, and mitigate SDOH disparities. Biomedical data scientists must know how to ingest geospatial neighborhood indices, capture ICD-10 Z-codes, compute formal inequality metrics, and audit algorithms for equitable clinical allocation.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Ingest** and geocode public and commercial SDOH indices, including the Area Deprivation Index (ADI), CDC Social Vulnerability Index (SVI), and AHRQ SDOH Database at census-tract and 9-digit ZIP code resolution.
2. **Audit** electronic health record datasets for ICD-10-CM Z-codes (Z55–Z65: Social and economic circumstances) to quantify documentation rates and undercoding bias.
3. **Compute** formal epidemiological and economic disparity metrics, including the Disparity Rate Ratio, Slope Index of Inequality (SII), and Relative Index of Inequality (RII).
4. **Identify** and prevent proxy bias when selecting labels in healthcare operations and predictive machine learning models.
5. **Implement** algorithmic fairness audits using intersectional sub-population stratification across race, ethnicity, rurality, and socioeconomic deprivation.
6. **Comply** with CMS Health Equity Index (HEI) scoring rules in Medicare Advantage and the ONC USCDI+ Social Determinants of Health data standards.

---

## 3. The SDOH Data Landscape: Geocoding and Neighborhood Indices

Individual-level social risk factors are rarely documented comprehensively in medical charts. Consequently, data engineering pipelines link patient addresses to geographic socioeconomic indices:

```
+-------------------------------------------------------------------------------+
|                        GEOSPATIAL SDOH DATA LINKAGE PIPELINE                  |
+-------------------------------------------------------------------------------+
Patient Address in EHR: "742 Evergreen Terrace, Springfield, OR 97477"
                         |
                         v
[ HIPAA-Compliant De-identified Geocoder ] (e.g., DeGAUSS / US Census Batch)
                         |
                         v
Spatial Join Keys: Census Tract (FIPS 41039003201) & 9-Digit ZIP (97477-1234)
                         |
       +-----------------+-----------------+
       |                                   |
       v                                   v
[ Area Deprivation Index (ADI) ]    [ CDC Social Vulnerability Index (SVI) ]
  - 17 Census variables               - 16 Social variables grouped into:
  - Income, education, housing,         1. Socioeconomic status
    employment                          2. Household composition & disability
  - Output: 1 to 100 National           3. Minority status & language
    percentile (100 = most deprived)    4. Housing type & transportation
```

### Comparative Framework: ADI vs. SVI

| Index Name | Source Institution | Primary Focus | Best Use Case |
|---|---|---|---|
| **Area Deprivation Index (ADI)** | University of Wisconsin / HRSA | Pure socioeconomic deprivation (poverty, education, housing quality) | Predicting clinical chronic disease outcomes, 30-day readmissions, and medication adherence. |
| **Social Vulnerability Index (SVI)** | CDC / ATSDR | Community disaster resilience and structural vulnerability | Pandemic resource allocation, mobile clinic routing, and emergency response planning. |
| **CDC PLACES** | CDC / Robert Wood Johnson | Local chronic disease and behavioral risk estimates (Census tract level) | Community health needs assessments and preventive intervention targeting. |

---

## 4. ICD-10-CM Z-Codes: The Clinical Documentation Gap

The medical coding system includes dedicated categories for social risks: **ICD-10 Z55 to Z65**:

```
+-------------------------------------------------------------------------------+
|                         ICD-10-CM SDOH Z-CODE CATEGORIES                      |
+-------------------------------------------------------------------------------+
| Code Range | Clinical Domain                    | Specific Example            |
+------------+------------------------------------+-----------------------------+
| **Z55**    | Education and Literacy             | Z55.0 Illiteracy            |
| **Z56**    | Employment and Unemployment        | Z56.0 Unemployment          |
| **Z57**    | Occupational Exposure              | Z57.3 Exposure to dust      |
| **Z59**    | Housing and Economic Circumstances | Z59.0 Homelessness          |
|            |                                    | Z59.4 Food insecurity       |
| **Z60**    | Social Environment                 | Z60.2 Living alone          |
| **Z62**    | Upbringing and Parent Problems     | Z62.81 Personal history of  |
|            |                                    | childhood abuse             |
| **Z63**    | Primary Support Group / Family     | Z63.0 Disruption of family  |
+-------------------------------------------------------------------------------+
```

### The Undercoding Crisis
In commercial claims and EHR repositories, **fewer than 2% of patients with documented social need have a recorded Z-code**. Why?
1. Physicians are not reimbursed for coding Z-codes under fee-for-service payment models.
2. Social work and case management notes containing social risk data are stored as unstructured text rather than billing codes.
3. **Data Engineering Solution:** NLP extraction engines must scan social work assessments and triage narratives to infer social risk factors, cross-validating against geographic ADI scores.

---

## 5. Implementation: Disparity Rate Ratio and Algorithmic Fairness Audit in Python

The following Python script computes formal disparity metrics (Disparity Rate Ratio) and performs an algorithmic fairness audit across socioeconomic deprivation quartiles, evaluating whether a clinical risk model exhibits equal predictive performance across vulnerable populations:

```python
"""
Health Equity Analytics: Disparity Rate Ratio and Stratified Fairness Audit.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

class HealthEquityAuditor:
    def __init__(self):
        pass

    def compute_disparity_rate_ratio(
        self, 
        df: pd.DataFrame, 
        group_col: str, 
        outcome_col: str, 
        reference_group: str
    ) -> pd.DataFrame:
        """
        Calculates Disparity Rate Ratio (DRR) for an adverse clinical outcome
        comparing disadvantaged groups to a privileged reference group.
        DRR = Rate(Group) / Rate(Reference)
        """
        rates = df.groupby(group_col)[outcome_col].agg(['count', 'mean']).reset_index()
        rates.rename(columns={'mean': 'outcome_rate', 'count': 'sample_size'}, inplace=True)
        
        ref_rate = rates[rates[group_col] == reference_group]['outcome_rate'].values[0]
        
        rates['reference_rate'] = ref_rate
        rates['disparity_rate_ratio'] = rates['outcome_rate'] / ref_rate
        
        return rates

    def audit_model_equity(
        self, 
        df: pd.DataFrame, 
        strata_col: str, 
        y_true_col: str, 
        y_pred_col: str
    ) -> Dict[str, Any]:
        """
        Evaluates algorithmic fairness across socioeconomic or racial strata,
        calculating AUROC, false positive rate (FPR), and false negative rate (FNR).
        """
        audit_results = {}
        
        for stratum, group_df in df.groupby(strata_col):
            y_true = group_df[y_true_col].values
            y_pred = group_df[y_pred_col].values
            
            if len(np.unique(y_true)) < 2:
                continue
                
            auroc = roc_auc_score(y_true, y_pred)
            
            # Binary classification metrics at 0.50 threshold
            y_pred_binary = (y_pred >= 0.50).astype(int)
            tn = np.sum((y_true == 0) & (y_pred_binary == 0))
            fp = np.sum((y_true == 0) & (y_pred_binary == 1))
            fn = np.sum((y_true == 1) & (y_pred_binary == 0))
            tp = np.sum((y_true == 1) & (y_pred_binary == 1))
            
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
            
            audit_results[stratum] = {
                "sample_size": len(group_df),
                "auroc": round(auroc, 3),
                "false_positive_rate": round(fpr, 3),
                "false_negative_rate": round(fnr, 3),
                "base_prevalence": round(np.mean(y_true), 3)
            }
            
        return audit_results

# Example Demonstration
if __name__ == "__main__":
    auditor = HealthEquityAuditor()
    
    # Simulate a cohort of 5,000 patients stratified by ADI Quartile
    # Q1: Wealthiest; Q4: Most Deprived
    np.random.seed(42)
    n = 5000
    adi_quartiles = np.random.choice(['Q1_Affluent', 'Q2_MidAffluent', 'Q3_MidDeprived', 'Q4_SeverelyDeprived'], size=n, p=[0.25, 0.25, 0.25, 0.25])
    
    # 30-day readmission outcome (elevated in Q4 due to food/housing insecurity)
    readmit_probs = {'Q1_Affluent': 0.08, 'Q2_MidAffluent': 0.12, 'Q3_MidDeprived': 0.18, 'Q4_SeverelyDeprived': 0.26}
    readmitted = [np.random.binomial(1, readmit_probs[q]) for q in adi_quartiles]
    
    # Model predictions (simulating lower model accuracy in Q4 due to missing data)
    model_preds = []
    for q, y in zip(adi_quartiles, readmitted):
        noise = np.random.normal(0, 0.25 if 'Q4' in q else 0.10)
        pred = np.clip(y * 0.7 + noise + 0.15, 0.01, 0.99)
        model_preds.append(pred)
        
    study_df = pd.DataFrame({
        "adi_strata": adi_quartiles,
        "readmitted": readmitted,
        "predicted_risk": model_preds
    })
    
    # 1. Compute Disparity Rate Ratio
    drr = auditor.compute_disparity_rate_ratio(
        study_df, group_col="adi_strata", outcome_col="readmitted", reference_group="Q1_Affluent"
    )
    print("=== DISPARITY RATE RATIO ACROSS ADI STRATA ===")
    print(drr[['adi_strata', 'outcome_rate', 'disparity_rate_ratio']])
    
    # 2. Audit Model Fairness
    fairness = auditor.audit_model_equity(
        study_df, strata_col="adi_strata", y_true_col="readmitted", y_pred_col="predicted_risk"
    )
    print("\n=== STRATIFIED MODEL FAIRNESS AUDIT ===")
    for q, metrics in fairness.items():
        print(f"Stratum: {q:<22} | AUROC: {metrics['auroc']} | FNR (Missed High Risk): {metrics['false_negative_rate']}")
```

---

## 6. The Proxy Variable Trap in Healthcare AI

Algorithmic bias in health systems rarely arises from malicious intent; it is almost always caused by **unconscious proxy selection**:

```
What the Clinical Team Wants to Predict:
    "Which patients have the highest complex healthcare NEED?"
                              |
                              v
The Fatal Proxy Substitution (Because 'Need' has no clean SQL column):
    "Let's predict which patients will have the highest healthcare SPEND ($)!"
                              |
                              v
The Real-World Systemic Bias Mechanism:
    - Wealthy patients have insurance, transportation, and paid sick leave ->
      They generate $50,000 in spend for routine preventive and elective care.
    - Deprived / Minority patients face cost barriers and lack access ->
      They generate $10,000 in spend despite having stage IV cancer or brittle diabetes.
                              |
                              v
The Algorithmic Failure:
    The model identifies wealthy patients as having "higher need",
    allocating 80% of nurse-care-manager phone calls to individuals who do not need them.
```

### Remedying the Proxy Trap
Never use historical costs, billed charges, or count of prior specialist visits as the target label for clinical need. Instead, predict **biological and functional outcomes**: emergency hospital admissions for ambulatory care sensitive conditions (ACSC), avoidable mortality, or biometric disease progression.

---

## 7. Regulatory and Quality Frameworks (2026)

| Authority / Program | Health Equity Mandate | Technical Deliverable |
|---|---|---|
| **CMS Health Equity Index (HEI)** | Medicare Advantage Star Ratings | Rewards health plans that achieve high clinical quality outcomes in disadvantaged populations (dual-eligible / low-income subsidy). |
| **The Joint Commission (NPSG 16)** | Hospital Accreditation | Hospitals must identify healthcare disparities in their patient populations and implement written equity quality improvement plans. |
| **ONC USCDI+ SDOH Standards** | Health IT Certification | Mandates standardized capturing of SDOH data elements: food insecurity, housing instability, transportation problems, and financial strain. |

---

## 8. Exercises and Self-Check

### Exercise 1: Auditing Algorithmic Equality of Opportunity
Suppose a health system deploys an AI model to allocate a free diabetes home-delivery meal program.
1. What is the mathematical definition of **Equality of Opportunity** (True Positive Rate parity) across racial/socioeconomic groups?
2. If the model exhibits a True Positive Rate of 88% in affluent suburban patients but only 61% in rural impoverished patients, what clinical and ethical harm occurs?

### Exercise 2: Geocoding Resolution and Privacy
Under HIPAA Safe Harbor rules:
1. What geographic restrictions apply to publishing ZIP codes?
2. How does using census-tract level ADI data balance high-resolution social risk modeling with HIPAA de-identification requirements?

---

## 9. Key Terms

- **Social Determinants of Health (SDOH):** Non-medical conditions in the environments where people live and work that substantially influence health risks and outcomes.
- **Area Deprivation Index (ADI):** A composite metric reflecting 17 socioeconomic indicators from the US Census, measuring neighborhood-level socioeconomic disadvantage.
- **Disparity Rate Ratio (DRR):** The ratio of an adverse outcome rate in an underserved group compared to a privileged reference population.
- **Proxy Bias:** Systematic error occurring when an observable surrogate variable (e.g., healthcare spending) does not accurately reflect the latent construct of interest (e.g., healthcare need).
- **ICD-10 Z-Codes:** Diagnostic codes (Z55–Z65) used in electronic health records to document social, environmental, and economic circumstances influencing health.

---

## 10. Sources and Further Reading

- Obermeyer, Z., et al. (2019). "Dissecting Racial Bias in an Algorithm Used to Manage the Health of Populations." *Science*, 366(6464), 447–453.
- Kind, A. J., & Buckingham, W. R. (2018). "Making Neighborhood-Disadvantage Metrics Accessible—The Neighborhood Atlas." *New England Journal of Medicine*, 378(26), 2456–2458.
- Centers for Medicare & Medicaid Services (CMS). (2023). *The CMS Framework for Health Equity 2022–2032*.
- Office of the National Coordinator for Health IT (ONC). (2024). *United States Core Data for Interoperability Plus (USCDI+) Social Determinants of Health*.
- Braveman, P., & Gottlieb, L. (2014). "The Social Determinants of Health: It's Time to Consider the Causes of the Causes." *Public Health Reports*, 129(1_suppl2), 19–31.
