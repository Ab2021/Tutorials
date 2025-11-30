# Open Insurance APIs (Part 1) - Standards & Portability - Theoretical Deep Dive

## Overview
"Data is the new Oil, but APIs are the Pipelines."
**Open Insurance** is the concept that the *Customer* owns their data, not the Insurer.
By exposing standardized APIs, insurers allow customers to share their data with FinTechs, Aggregators, and other Insurers to get better rates.

---

## 1. Conceptual Foundation

### 1.1 Open Banking vs. Open Insurance

*   **Open Banking (PSD2):** EU regulation forcing banks to share transaction data (with consent).
*   **Open Insurance (OPIN):** The equivalent movement in Insurance.
    *   *Goal:* Allow a 3rd party app to say: "Connect your Geico account to see if Progressive is cheaper."

### 1.2 The API Ecosystem

1.  **Product APIs:** "Get Quote", "Bind Policy". (Used by Aggregators).
2.  **Servicing APIs:** "Get Policy Details", "Download ID Card". (Used by Wallets).
3.  **Claims APIs:** "File FNOL", "Upload Photo". (Used by Repair Shops).

---

## 2. Mathematical Framework

### 2.1 Data Portability Value

$$ \text{Value} = \sum (\text{Premium}_{\text{Old}} - \text{Premium}_{\text{New}}) - \text{SwitchingCost} $$

*   **Without APIs:** Switching Cost is High (Re-typing 50 fields).
*   **With APIs:** Switching Cost is Low (1 Click).
*   **Result:** Market efficiency increases. Prices drop.

### 2.2 Standardization (The "Tower of Babel" Problem)

*   **Insurer A:** Uses field `car_year`.
*   **Insurer B:** Uses field `vehicle_model_year`.
*   **Aggregator:** Needs a translator.
*   **Solution:** **ACORD Standards**.
    *   Standard JSON Schema: `{"Vehicle": {"ModelYear": 2024}}`.

---

## 3. Theoretical Properties

### 3.1 The "Walled Garden" Defense

*   **Incumbent Strategy:** "Keep the data siloed so the customer can't leave."
*   **Regulator View:** This is anti-competitive.
*   **Outcome:** Regulation (like GDPR/PSD2) forces the walls down.

### 3.2 Security & Consent (OAuth 2.0)

*   **Mechanism:**
    1.  User clicks "Connect Geico".
    2.  Redirected to Geico.com.
    3.  User logs in and clicks "Approve".
    4.  Geico issues an **Access Token** to the App.
    5.  App uses Token to fetch data. (App never sees the Password).

---

## 4. Modeling Artifacts & Implementation

### 4.1 OpenAPI Specification (Swagger)

```yaml
openapi: 3.0.0
info:
  title: Open Insurance API
  version: 1.0.0
paths:
  /policies/{policyId}:
    get:
      summary: Get Policy Details
      parameters:
        - name: policyId
          in: path
          required: true
          schema:
            type: string
      responses:
        '200':
          description: Successful response
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Policy'
components:
  schemas:
    Policy:
      type: object
      properties:
        policyNumber:
          type: string
        effectiveDate:
          type: string
          format: date
        coverages:
          type: array
          items:
            type: object
            properties:
              code:
                type: string
              limit:
                type: integer
```

### 4.2 Python Client (Consuming the API)

```python
import requests

# 1. Authenticate (OAuth 2.0)
auth_response = requests.post('https://auth.insurer.com/token', data={
    'grant_type': 'client_credentials',
    'client_id': 'MY_APP_ID',
    'client_secret': 'MY_SECRET'
})
token = auth_response.json()['access_token']

# 2. Fetch Policy
headers = {'Authorization': f'Bearer {token}'}
response = requests.get('https://api.insurer.com/policies/POL-123', headers=headers)

# 3. Parse Data
policy = response.json()
print(f"Coverage Limit: {policy['coverages'][0]['limit']}")
```

---

## 5. Evaluation & Validation

### 5.1 API Governance

*   **Metric:** Breaking Changes.
*   **Rule:** You cannot change a field name in `v1`. You must create `v2`.
*   **Tool:** Spectral (Linter for OpenAPI).

### 5.2 Performance (Rate Limiting)

*   **Risk:** A competitor scrapes your API to reverse-engineer your pricing model.
*   **Defense:** Rate Limit (100 requests/minute). API Keys.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Screen Scraping" Legacy**
    *   *Scenario:* Before APIs, apps used bots to log in and "scrape" the HTML.
    *   *Risk:* Brittle (breaks when UI changes). Insecure (requires password sharing).
    *   *Fix:* Ban scraping. Build APIs.

2.  **Trap: Data Minimization**
    *   *Principle:* Only share what is needed.
    *   *Scenario:* App asks for "All Policy Data" just to show the Expiration Date.
    *   *Fix:* Granular Scopes (`read:policy_date` vs `read:full_policy`).

---

## 7. Advanced Topics & Extensions

### 7.1 "Insurance-as-a-Service" (IaaS)

*   **Concept:** A White-Label Insurer (e.g., Boost, Tint).
*   **Offering:** They provide the License + Capital + API. You provide the Brand + UX.
*   **Result:** Any startup can become an "Insurer" in 2 weeks.

### 7.2 The "Super App"

*   **Vision:** WeChat / Grab.
*   **Function:** One App for Ride, Food, Payment, and Insurance.
*   **Enabler:** Open APIs allowing deep integration.

---

## 8. Regulatory & Governance Considerations

### 8.1 FIDA (Financial Data Access) - EU

*   **Proposal:** Extends Open Banking to Insurance.
*   **Right:** Customers have the legal right to share their data. Insurers *must* provide the API.

---

## 9. Practical Example

### 9.1 Worked Example: The "Wallet" Integration

**Scenario:** Apple Wallet.
*   **Goal:** User wants their Auto ID Card in Apple Wallet.
*   **Old Way:** PDF via Email.
*   **API Way:**
    1.  User taps "Add to Wallet" in Insurer App.
    2.  App calls `POST /wallet/pass`.
    3.  API generates a `.pkpass` file signed with a Certificate.
    4.  Pass appears on Lock Screen when near the car (Geofence).
*   **Value:** Convenience. No fumbling for paper when pulled over.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Open Insurance** democratizes data.
2.  **Standards (ACORD)** prevent chaos.
3.  **OAuth** secures the bridge.

### 10.2 When to Use This Knowledge
*   **CTO:** "Should we build a public API developer portal?" (Yes, it's the future of distribution).
*   **Compliance Officer:** "How do we ensure we don't leak PII via API?"

### 10.3 Critical Success Factors
1.  **Developer Experience (DX):** Good documentation (Stripe standard).
2.  **Uptime:** 99.99% SLA.

### 10.4 Further Reading
*   **OPIN:** "The Open Insurance Initiative".

---

## Appendix

### A. Glossary
*   **REST:** Representational State Transfer.
*   **JSON:** JavaScript Object Notation.
*   **OAuth:** Open Authorization.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **API Value** | $N_{\text{Partners}} \times \text{Vol}_{\text{Partner}}$ | Network Effect |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
