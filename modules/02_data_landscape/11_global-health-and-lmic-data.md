# Global Health and LMIC Data Systems

> **Module 02 — The Biomedical Data Landscape**
> Topic 11 of 11
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 02, 03, and 08
> Version 2026.3

## 1. Why this topic exists

Over **85% of the world's population** resides in Low- and Middle-Income Countries (LMICs) across sub-Saharan Africa, South Asia, Latin America, and Southeast Asia. Yet, over 90% of commercial biomedical data science and clinical machine learning is developed using datasets originating from high-income, tertiary academic centers in North America and Western Europe (e.g., Epic EHR data, UK Biobank, TCGA).

When algorithms trained on high-income datasets are evaluated in global health and LMIC settings, they collapse. The data infrastructure and clinical realities of LMICs are fundamentally distinct:
1. **Infrastructure & Connectivity Constraints:** Intermittent electrical grids, low-bandwidth satellite internet, and reliance on battery-operated mobile devices mean centralized, real-time cloud inference is often impossible. Systems must run on the edge or support asynchronous store-and-forward syncing.
2. **Distinct Open-Source Ecosystems:** Instead of commercial enterprise EHRs (Epic, Cerner), global health systems run on open-source public health platforms: **DHIS2** (District Health Information Software 2, operating in 80+ countries covering 3.2 billion people) for national aggregate reporting, and **OpenMRS** for patient-level clinical records.
3. **Alternative Digital Footprints:** In regions lacking formal credit histories or universal vital statistics registration, data scientists leverage novel digital proxies: mobile phone call detail records (CDRs) and mobile money transactions (e.g., M-Pesa) to track epidemic spread, population displacement, and socioeconomic vulnerability.
4. **Severe Data Quality and Coding Biases:** ICD-10/11 coding is rarely performed by certified medical coders; verbal autopsy algorithms are required to infer cause of death; and public health disease surveillance relies on early warning networks (EWARN, eIDSR) rather than automated lab telemetry.

To operate effectively in global health organizations (WHO, Gates Foundation, PATH, Africa CDC) and multinational biopharma clinical trial networks in 2026, data scientists must master LMIC health data infrastructure, DHIS2 data models, and epidemic surveillance analytics.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Deconstruct** the architecture of DHIS2, distinguishing Aggregate Data Models from Tracker Program event-level schemas.
2. **Navigate** OpenMRS data models and FHIR-based sync protocols in low-connectivity, offline-first clinical environments.
3. **Implement** automated epidemic surveillance and outbreak anomaly detection algorithms for syndromic and event-based health systems (EWARN, eIDSR).
4. **Analyze** verbal autopsy records using automated NLP and probabilistic algorithms (InSilicoVA, InterVA) to assign cause-of-death mortality codes.
5. **Mitigate** LMIC-specific data collection biases, including missing vital registrations, seasonal agricultural migration gaps, and clinic stockout confounding.
6. **Deploy** privacy-preserving mobility and mobile money data mining pipelines to forecast infectious disease vector transmission under WHO International Health Regulations (IHR).

---

## 3. The LMIC Health IT Architecture: DHIS2 and OpenMRS

The global health digital ecosystem is anchored by two foundational open-source platforms:

```
+-------------------------------------------------------------------------------+
|                       LMIC DIGITAL HEALTH ARCHITECTURE                        |
+-------------------------------------------------------------------------------+

[ Point of Care: Rural Clinics & District Hospitals ]
  OpenMRS (Open Medical Record System)
  - Patient Registration, HIV/TB Care Cards, Encounter Records
  - Offline-first SQLite/MySQL database on local clinic server
  - Asynchronous JSON/FHIR sync when internet becomes available
                                |
                                v
[ District & National Level: Public Health Surveillance ]
  DHIS2 (District Health Information Software 2)
  +-------------------------------------+-------------------------------------+
  | DHIS2 Aggregate Data Model          | DHIS2 Tracker (Event-Level Model)   |
  | - Monthly facility counts:          | - Longitudinal individual tracking: |
  |   "Clinic A: 42 Malaria Cases"      |   "Mother ID: 902 -> ANC Visits,    |
  | - Stored in DataValue table:        |    Vaccination, Birth Outcome"      |
  |   (DataElement, OrgUnit, Period, Val)| - Complex event-driven analytics    |
  +-------------------------------------+-------------------------------------+
                                |
                                v
[ Global Surveillance & Outbreak Response ]
  WHO EWARN (Early Warning, Alert and Response Network)
  Africa CDC Pathogen Genomics & Infectious Disease Dashboards
```

---

## 4. Verbal Autopsy Analytics: Measuring Mortality in the Absence of Death Certificates

In many LMIC settings, over **60% of deaths occur at home** outside of formal hospital facilities without medical certification of cause of death. To quantify disease burden, public health epidemiologists utilize **Verbal Autopsy (VA)**: trained interviewers administer a standardized WHO questionnaire to the deceased's family, capturing symptoms prior to death.

```
Unstructured / Structured Questionnaire:
  "Did the deceased have a cough lasting > 2 weeks?" [YES]
  "Did the deceased experience night sweats and weight loss?" [YES]
  "Did the deceased spit blood (hemoptysis)?" [YES]
                               |
                               v
[ Probabilistic & NLP Verbal Autopsy Algorithms ]
  - InSilicoVA / InterVA-5 / Tariff 2.0
  - Bayesian network computing posterior probability over 64 cause-of-death categories
                               |
                               v
Predicted Primary Cause of Death:
  ICD-10 A15.0: Tuberculosis of Lung (Posterior Probability: 0.94)
```

---

## 5. Implementation: DHIS2 API Extraction and Outbreak Detection in Python

The following Python script illustrates how a public health data scientist authenticates against a national DHIS2 Web API, extracts monthly malaria case counts across district clinics, and runs an automated cumulative sum (CUSUM) anomaly detection algorithm to flag emerging infectious outbreaks:

```python
"""
Global Health Data Science: DHIS2 API Extraction and Outbreak Detection.
"""
from typing import Dict, List, Any
import numpy as np
import pandas as pd

class GlobalHealthOutbreakDetector:
    def __init__(self, outbreak_z_threshold: float = 2.5):
        self.threshold = outbreak_z_threshold

    def parse_dhis2_analytics_payload(self, mock_dhis2_response: Dict[str, Any]) -> pd.DataFrame:
        """
        Parses standard DHIS2 Web API /api/analytics JSON response.
        Columns: [dx (dataElement), pe (period), ou (organisationUnit), value]
        """
        headers = [h["name"] for h in mock_dhis2_response.get("headers", [])]
        rows = mock_dhis2_response.get("rows", [])
        
        df = pd.DataFrame(rows, columns=headers)
        df["value"] = pd.to_numeric(df["value"], errors="coerce").fillna(0.0)
        return df

    def detect_syndromic_outbreaks(
        self, 
        monthly_series: pd.Series, 
        baseline_lookback_months: int = 12
    ) -> List[Dict[str, Any]]:
        """
        Runs early warning outbreak detection on monthly case counts.
        Compares current month against historical baseline mean and standard deviation.
        """
        alerts = []
        counts = monthly_series.values
        periods = monthly_series.index
        
        for t in range(baseline_lookback_months, len(counts)):
            current_period = periods[t]
            current_count = counts[t]
            
            # Baseline: Previous 12 months excluding current
            baseline = counts[t - baseline_lookback_months : t]
            b_mean = np.mean(baseline)
            b_std = np.std(baseline) + 1e-6
            
            z_score = (current_count - b_mean) / b_std
            is_outbreak = z_score >= self.threshold
            
            alerts.append({
                "period": current_period,
                "reported_cases": current_count,
                "baseline_mean": round(float(b_mean), 1),
                "z_score": round(float(z_score), 2),
                "outbreak_alert": is_outbreak
            })
            
        return alerts

# Example Demonstration
if __name__ == "__main__":
    detector = GlobalHealthOutbreakDetector(outbreak_z_threshold=2.5)
    
    # Mock DHIS2 analytics API response for District X Malaria Cases
    mock_api = {
        "headers": [{"name": "dx"}, {"name": "pe"}, {"name": "ou"}, {"name": "value"}],
        "rows": [
            ["Malaria_Cases", "202401", "Dist_01", "120"],
            ["Malaria_Cases", "202402", "Dist_01", "135"],
            ["Malaria_Cases", "202403", "Dist_01", "110"],
            ["Malaria_Cases", "202404", "Dist_01", "145"],
            ["Malaria_Cases", "202405", "Dist_01", "130"],
            ["Malaria_Cases", "202406", "Dist_01", "125"],
            ["Malaria_Cases", "202407", "Dist_01", "140"],
            ["Malaria_Cases", "202408", "Dist_01", "130"],
            ["Malaria_Cases", "202409", "Dist_01", "115"],
            ["Malaria_Cases", "202410", "Dist_01", "120"],
            ["Malaria_Cases", "202411", "Dist_01", "135"],
            ["Malaria_Cases", "202412", "Dist_01", "140"],
            # Outbreak spike in early 2025 following flood
            ["Malaria_Cases", "202501", "Dist_01", "345"]
        ]
    }
    
    df = detector.parse_dhis2_analytics_payload(mock_api)
    series = df.set_index("pe")["value"]
    
    alerts = detector.detect_syndromic_outbreaks(series, baseline_lookback_months=12)
    print("=== DHIS2 INFECTIOUS DISEASE OUTBREAK DETECTION ===")
    for a in alerts:
        flag = " [!] EPIDEMIC OUTBREAK DETECTED" if a['outbreak_alert'] else ""
        print(f"Period {a['period']}: Cases = {a['reported_cases']:<4} | Baseline = {a['baseline_mean']} | Z = {a['z_score']:<5}{flag}")
```

---

## 6. Structural Data Biases in Low-Resource Settings

When training machine learning models on LMIC health data, distinct systemic biases emerge that never appear in Western academic medical centers:

```
+-------------------------------------------------------------------------------+
|                       LMIC STRUCTURAL DATA BIAS TAXONOMY                      |
+-------------------------------------------------------------------------------+

1. Stockout Confounding (Supply-Side Bias)
   - Mechanism: A clinic records 0 diagnoses of malaria in June not because the
     disease disappeared, but because the clinic ran out of rapid diagnostic tests (RDTs).
   - Machine Learning Hazard: A model predicts malaria risk based on recorded
     diagnoses and erroneously concludes June has zero epidemic risk!
   - Mitigation: Always cross-reference clinical encounter counts with pharmacy
     and supply-chain inventory logs.

2. Seasonal Road Inaccessibility (Harvest & Monsoonal Bias)
   - Heavy monsoon rains flood dirt roads, making travel to district hospitals impossible.
   - Clinical consequence: Patient attendance plummets; mortality shifts from
     inpatient hospital records to unrecorded home deaths.

3. Differential Cell Phone Ownership in Digital Health
   - mHealth SMS reporting programs exhibit severe gender and age selection bias:
     mobile phones in rural households are overwhelmingly controlled by adult males,
     leading to severe under-representation of maternal and child health outcomes.
```

---

## 7. Global Health Regulations and Ethical Principles

| Framework / Treaty | Sponsoring Body | Mandate for Data Science |
|---|---|---|
| **International Health Regulations (IHR 2005)** | World Health Organization (WHO) | Legally binding treaty requiring 196 countries to build national capacity for real-time disease detection, assessment, and notification of public health emergencies of international concern (PHEIC). |
| **Principles for Digital Development** | International Donor Consortia (USAID, Gates, WHO) | Mandates: Design with the user, understand the existing ecosystem, design for scale, build for sustainability, be data-driven, and use open standards/open source. |
| **Nagoya Protocol on Access & Benefit-Sharing** | United Nations (CBD) | Governs the fair and equitable sharing of benefits arising from the utilization of genetic resources and genomic pathogen data sourced from LMIC nations. |

---

## 8. Exercises and Self-Check

### Exercise 1: Handling Stockout Artefacts in DHIS2
You are building an automated forecasting model to predict monthly antibiotic consumption across 40 rural health centers in Malawi using historical DHIS2 data.
1. How do you distinguish a true decrease in respiratory infections from a complete stockout of amoxicillin?
2. What feature engineering approach (e.g., incorporating supply delivery receipts and test kit inventory numbers) prevents the model from under-predicting future need?

### Exercise 2: Evaluating Verbal Autopsy Algorithms
Compare InSilicoVA (probabilistic Bayesian algorithm) with Tariff 2.0 (rank-based scoring algorithm):
1. How do these algorithms handle missing responses (e.g., family members who cannot recall if the patient had a rash)?
2. Why is calculating the Cause-Specific Mortality Fraction (CSMF) at the population level mathematically more robust than individual-level classification accuracy?

---

## 9. Key Terms

- **DHIS2 (District Health Information Software 2):** An open-source, web-based health management information system platform used by national ministries of health in over 80 LMICs.
- **OpenMRS:** An open-source electronic medical record system platform specifically designed for low-resource environments and developing countries.
- **Verbal Autopsy (VA):** A structured interview conducted with the family or caregivers of a deceased individual to determine the probable cause of death in regions lacking medical certification.
- **EWARN / eIDSR:** Early Warning, Alert and Response Network / electronic Integrated Disease Surveillance and Response; standardized frameworks for detecting acute outbreaks in humanitarian and low-resource settings.
- **Stockout Confounding:** The false appearance of disease absence in healthcare data caused by the exhaustion of diagnostic supplies or medications.

---

## 10. Sources and Further Reading

- De Savigny, D., et al. (2017). "Integrating Community-Level Health Information into National Health Information Systems: A Case Study from Malawi (DHIS2)." *BMC Health Services Research*, 17(1), 1–11.
- McCormick, T. H., et al. (2016). "Probabilistic Cause-of-Death Assignment Using Verbal Autopsies (InSilicoVA)." *Journal of the American Statistical Association*, 111(515), 1036–1049.
- World Health Organization (WHO). (2022). *Verbal Autopsy Standards: The 2022 WHO Verbal Autopsy Instrument*.
- Seebregts, C. J., et al. (2009). "The OpenMRS Implementers Network." *International Journal of Medical Informatics*, 78(11), 711–720.
- World Health Organization (WHO). (2005). *International Health Regulations (2005) - 3rd Edition*. WHO Press.
