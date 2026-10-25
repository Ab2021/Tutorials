# CDS Hooks and SMART on FHIR Architecture

> **Module 03 — Data Engineering and Interoperability**
> Topic 13 of 13
> Estimated study time: 4 hours
> Prerequisites: Topics 01, 05, 11, and 12
> Version 2026.3

## 1. Why this topic exists

The ultimate goal of biomedical data science and clinical machine learning is to change decisions at the point of care. A sepsis alert model with an area under the curve of 0.94 is clinically useless if it runs as an offline batch script or resides in an isolated web dashboard that physicians never open. To affect patient outcomes, predictive models and clinical decision algorithms must be embedded directly into the daily clinical workflow inside Electronic Health Record (EHR) systems like Epic, Oracle Health (Cerner), and MEDITECH.

Historically, integrating an external algorithm into an EHR required proprietary vendor APIs, custom HL7 v2 interfaces, or specialized clinical engineering services costing hundreds of thousands of dollars per hospital deployment.

The modern industry standard for clinical workflow integration is defined by two open specifications:
1. **SMART on FHIR:** An open standards architecture based on OAuth2, OpenID Connect, and FHIR REST APIs that allows external web applications to launch securely inside an EHR session with seamless single-sign-on (SSO) and scoped patient data access.
2. **CDS Hooks:** An event-driven architecture that triggers external Clinical Decision Support (CDS) services automatically at specific points in the clinician's EHR workflow (e.g., opening a chart, selecting a medication, signing an order), rendering real-time guidance directly within the native EHR interface via standardized "Cards".

Mastering the mechanics of CDS Hooks and SMART on FHIR enables machine learning engineers and data architects to build "write-once, deploy-anywhere" clinical AI products across the entire healthcare ecosystem.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Architect** an event-driven Clinical Decision Support service compliant with the HL7 CDS Hooks 1.0/2.0 specification.
2. **Implement** the four primary CDS Hook event triggers (`patient-view`, `order-select`, `order-sign`, `appointment-book`) with optimized prefetch queries.
3. **Construct** standard CDS Hook response Cards, selecting appropriate card types (Information, Suggestion with pre-built FHIR actions, and SMART App Link).
4. **Deploy** SMART on FHIR authorization flows, handling launch context tokens, OAuth2 scopes (`launch`, `patient/*.read`), and token exchange.
5. **Mitigate** clinician "alert fatigue" and automation bias through targeted hook filtering, confidence thresholding, and non-disruptive card designs.
6. **Test** and validate integration services using the open-source SMART Health IT Sandbox and Logica Sandbox environments.

---

## 3. The CDS Hooks Lifecycle and Event Triggers

CDS Hooks is a vendor-agnostic specification defining how an EHR invokes an external REST service during user interaction:

```
Clinician Interacts with EHR
            |
            v
[ EHR Trigger Event Occurs ] (e.g., Doctor selects an antibiotic in Epic)
            |
            v
EHR compiles HTTP POST request:
  - hook: "order-select"
  - hookInstance: "uuid-1234-abcd"
  - context: { patientId: "101", draftOrders: [...] }
  - prefetch: { patient: {...}, activeMedications: [...] }
            |
            v
[ External CDS ML Service (Your Cloud / Container) ]
  - Evaluates ML model / clinical logic (e.g., Sepsis Risk, Drug-Lab Interaction)
  - Execution deadline: < 500 milliseconds
            |
            v
External Service returns HTTP 200 with CDS Cards:
[
  {
    "summary": "High Risk of Acute Kidney Injury",
    "indicator": "critical",
    "source": { "label": "NephroAI Clinical Risk Engine" },
    "suggestions": [
      {
        "label": "Switch Vancomycin to Daptomycin",
        "actions": [ { "type": "delete", ... }, { "type": "create", ... } ]
      }
    ],
    "links": [
      { "label": "Open Renal Risk Dashboard", "type": "smart", "url": "https://..." }
    ]
  }
]
            |
            v
EHR displays native warning banner & clickable action buttons to clinician
```

### The Standard CDS Hook Trigger Taxonomy

| Hook Name | EHR Workflow Trigger Point | Clinical Use Case | Latency Tolerance |
|---|---|---|---|
| **`patient-view`** | Opening a patient chart / selecting encounter | Sepsis early warning, 30-day readmission risk score | $< 1,000\,\text{ms}$ |
| **`order-select`** | Selecting a medication, lab, or imaging order | Drug-drug interaction, alternative low-cost medication | $< 500\,\text{ms}$ (Must be fast to avoid UI freeze) |
| **`order-sign`** | Clicking "Sign / Submit Orders" | Final safety check, duplicate order prevention | $< 800\,\text{ms}$ |
| **`appointment-book`** | Scheduling a clinic visit or procedure | Pre-visit diagnostic testing requirements | $< 2,000\,\text{ms}$ |

---

## 4. Anatomy of a CDS Card: Three Response Modalities

The external service communicates decisions back to the EHR using standardized **Cards**:

```
+-------------------------------------------------------------------------------+
|                             ANATOMY OF A CDS HOOK CARD                        |
+-------------------------------------------------------------------------------+
| 1. Indicator Badge: | 'info' (Blue), 'warning' (Yellow), or 'critical' (Red)  |
| 2. Summary:         | Short headline (< 140 chars) describing recommendation  |
| 3. Detail:          | Markdown explanation, evidence citation, and confidence |
| 4. Source:          | Attributed algorithm, institutional review board name   |
+---------------------+---------------------------------------------------------+
|                                  ACTION MODALITIES                            |
|                                                                               |
| [ Modality A: Informational Card ]                                            |
|   Displays text only; no state change in EHR.                                 |
|                                                                               |
| [ Modality B: Suggestion Card with 1-Click Action ]                           |
|   Contains a pre-built FHIR transaction bundle. If clinician clicks "Accept",  |
|   the EHR executes the change directly (e.g. cancels drug order, adds lab).   |
|                                                                               |
| [ Modality C: SMART App Link Card ]                                           |
|   Clicking button launches interactive SMART on FHIR UI in an embedded iframe |
|   passing patient launch context seamlessly.                                  |
+-------------------------------------------------------------------------------+
```

---

## 5. Implementation: Building a CDS Hooks Microservice in Python (FastAPI)

The following production script implements an enterprise CDS Hooks microservice using FastAPI. It evaluates an `order-select` hook for pediatric patients prescribed adult medication dosages and emits a Suggestion Card:

```python
"""
CDS Hooks Microservice: Order-Select Pediatric Dosing Safety Service.
"""
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

app = FastAPI(title="Pediatric Dosing CDS Service", version="2.0.0")

# CDS Hooks Discovery Endpoint
@app.get("/cds-services")
def discover_services():
    """EHR calls this endpoint at boot to discover available CDS hooks and prefetch requirements."""
    return {
        "services": [
            {
                "hook": "order-select",
                "name": "pediatric-dosing-check",
                "id": "pediatric-dosing-guard",
                "description": "Evaluates patient age and weight to prevent adult dosage overdoses in pediatrics.",
                "prefetch": {
                    "patient": "Patient/{{context.patientId}}",
                    "weight": "Observation?patient={{context.patientId}}&code=29463-7&_sort=-date&_count=1"
                }
            }
        ]
    }

# CDS Hook Execution Endpoint
@app.post("/cds-services/pediatric-dosing-guard")
async def evaluate_hook(payload: Dict[str, Any]):
    context = payload.get("context", {})
    prefetch = payload.get("prefetch", {})
    
    # 1. Parse Patient Demographics from Prefetch
    patient_res = prefetch.get("patient", {})
    birth_date = patient_res.get("birthDate", "2000-01-01")
    
    # Simple age calculation (assume 4 years old for demonstration)
    patient_age_years = 4 
    
    cards = []
    
    # 2. Inspect Draft Orders in Context
    draft_orders = context.get("draftOrders", {}).get("entry", [])
    for entry in draft_orders:
        med_req = entry.get("resource", {})
        med_name = med_req.get("medicationCodeableConcept", {}).get("text", "Unknown Drug")
        
        # Check if adult dose ordered for pediatric patient
        if patient_age_years < 12 and "Acetaminophen 500mg" in med_name:
            card = {
                "uuid": "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "summary": "Adult Acetaminophen Dose Prescribed to Pediatric Patient",
                "indicator": "critical",
                "detail": f"Patient is {patient_age_years} years old. Recommended pediatric dosing is 15mg/kg suspension, not 500mg adult tablets.",
                "source": {
                    "label": "Pediatric Clinical Pharmacy Protocol",
                    "url": "https://hospital.org/guidelines/pediatric-acetaminophen"
                },
                "suggestions": [
                    {
                        "label": "Replace with Acetaminophen 160mg/5mL oral liquid",
                        "uuid": "sug-001",
                        "actions": [
                            {
                                "type": "delete",
                                "description": "Cancel adult tablet order",
                                "resource": med_req
                            },
                            {
                                "type": "create",
                                "description": "Order pediatric suspension",
                                "resource": {
                                    "resourceType": "MedicationRequest",
                                    "status": "draft",
                                    "intent": "order",
                                    "medicationCodeableConcept": {
                                        "text": "Acetaminophen 160 mg/5 mL oral liquid"
                                    }
                                }
                            }
                        ]
                    }
                ],
                "links": [
                    {
                        "label": "Open Pediatric Weight-Based Dose Calculator",
                        "url": "https://apps.hospital.org/smart/peds-dosing-calc",
                        "type": "smart"
                    }
                ]
            }
            cards.append(card)
            
    return {"cards": cards}
```

---

## 6. SMART on FHIR OAuth2 Launch Sequence

When a clinician clicks a SMART App Link on a CDS Card, the EHR launches an interactive web application inside an embedded secure frame using the **SMART App Launch Framework**:

```
EHR Session (Clinician active in Chart)          External SMART App (Browser)
                     |                                       |
                     | 1. Launch App in embedded iframe      |
                     |    URL?launch=xyz123&iss=https://ehr  |
                     |-------------------------------------->|
                     |                                       |
                     | 2. GET /oauth/authorize               |
                     |    scope=launch+patient/*.read        |
                     |<--------------------------------------|
                     |                                       |
                     | 3. Authenticates & Authorizes         |
                     |    Returns Authorization Code         |
                     |-------------------------------------->|
                     |                                       |
                     | 4. POST /oauth/token                  |
                     |    code + client_id                   |
                     |<--------------------------------------|
                     |                                       |
                     | 5. Returns Access Token + Context:    |
                     |    {                                  |
                     |      "access_token": "ey...",         |
                     |      "patient": "pt-4491",            |
                     |      "encounter": "enc-902"           |
                     |    }                                  |
                     |-------------------------------------->|
                     |                                       |
                     | 6. GET /r4/Observation?patient=pt-4491|
                     |    Header: Authorization: Bearer ey...|
                     |<--------------------------------------|
```

---

## 7. Clinical Human Factors: Combating Alert Fatigue

In real-world hospital environments, clinicians override or dismiss **90% to 96%** of electronic health record alerts. If a machine learning model fires too frequently or interrupts the physician for benign conditions, clinicians develop subconscious **alert fatigue**—blindly clicking "Dismiss" and missing the rare, life-saving warning.

### Production Rules for Alert Governance
1. **Never Interrupt Without an Actionable Alternative:** An alert that says *"High Mortality Risk"* without an associated protocol or button to order intervention will be ignored. Always offer a pre-built Suggestion Card with a one-click action.
2. **Strict Prefetch Optimization:** Prefetch all necessary FHIR resources during the initial hook call. If an external CDS service makes secondary synchronous REST queries back to the EHR, latency exceeds 1,500 ms, freezing the clinician's typing interface and causing IT to disable the hook.
3. **Suppression Windows:** If an alert is acknowledged or dismissed by a clinician, suppress identical alerts for the same patient for a minimum refractory period (e.g., 24 hours).

---

## 8. Exercises and Self-Check

### Exercise 1: Designing Prefetch Templates
You are building an acute sepsis alerting CDS Hook triggered on `patient-view`. The model requires the patient's age, last 24 hours of heart rate observations, and white blood cell (WBC) lab results.
1. Write down the CDS Hooks JSON `prefetch` dictionary using FHIR R4 search queries parameterized with `{{context.patientId}}`.
2. Why is using prefetch drastically superior to having your Python service query the EHR FHIR server during request handling?

### Exercise 2: Auditing OAuth2 Scopes
An external SMART on FHIR cardiology risk app requests the following scopes:
`launch patient/*.read user/*.write openid profile`
1. Explain what each scope authorizes.
2. As a hospital Chief Information Security Officer (CISO), why would you reject the `user/*.write` scope for a simple read-only risk calculation tool?

---

## 9. Key Terms

- **CDS Hooks:** An HL7 open specification defining an event-driven mechanism where EHR workflow events trigger external clinical decision support REST services.
- **SMART on FHIR:** A standardized protocol based on OAuth2 and OpenID Connect allowing third-party healthcare applications to run securely inside an EHR session.
- **Prefetch:** Data bundles retrieved proactively by the EHR and pushed inside the initial CDS Hook request payload to minimize round-trip network latency.
- **Alert Fatigue:** The psychological phenomenon where clinicians exposed to high volumes of electronic alerts become desensitized, leading to automatic dismissal and missed critical warnings.
- **Suggestion Card:** A CDS Card containing a pre-configured FHIR resource modification that the user can accept into their EHR workflow with a single click.

---

## 10. Sources and Further Reading

- HL7 International. (2022). *CDS Hooks Specification Version 1.0 & 2.0*. https://cds-hooks.hl7.org/
- Mandel, J. C., et al. (2016). "SMART on FHIR: A Standards-Based, Interoperable Apps Platform for Electronic Health Records." *Journal of the American Medical Informatics Association (JAMIA)*, 23(5), 899–908.
- SMART Health IT. (2024). *SMART App Launch Framework Implementation Guide Version 2.2.0*. https://hl7.org/fhir/smart-app-launch/
- Ancker, J. S., et al. (2017). "Effects of an Electronic Health Record Alert on Inappropriate Medication Prescribing in Ambulatory Care." *JAMA Internal Medicine*, 177(5), 652–659.
- Kawamoto, K., et al. (2021). "Scalable and Interoperable Clinical Decision Support Using Standards-Based Technology (CDS Hooks and SMART on FHIR)." *Yearbook of Medical Informatics*, 30(1), 163–175.
