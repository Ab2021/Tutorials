# Cyber Risk Modelling (Part 1) - Theoretical Deep Dive

## Overview
Cyber is the "New Hurricane". It's a catastrophic risk that can hit the entire world simultaneously (Accumulation Risk). This session covers the **Cyber Kill Chain**, **CVEs**, and why Actuaries lose sleep over Ransomware.

---

## 1. Conceptual Foundation

### 1.1 The Cyber Risk Landscape

1.  **Data Breach:** Theft of PII (Personally Identifiable Information). Cost: Legal fees, Notification costs, Fines (GDPR).
2.  **Ransomware:** Encryption of company data. Cost: Ransom payment, Business Interruption (BI).
3.  **DDoS (Distributed Denial of Service):** Flooding a server to crash it. Cost: BI.

### 1.2 The Cyber Kill Chain (Lockheed Martin)

*   **Reconnaissance:** Hacker scans for vulnerabilities (e.g., open ports).
*   **Weaponization:** Creating a malware payload.
*   **Delivery:** Phishing email or USB drive.
*   **Exploitation:** Malware executes on the victim's machine.
*   **Installation:** Backdoor installed for persistent access.
*   **Command & Control (C2):** Hacker takes control.
*   **Actions on Objectives:** Steal data or Encrypt files.

### 1.3 The Actuarial Nightmare

*   **Lack of History:** We have 100 years of Hurricane data. We have 10 years of Cyber data, and 90% of it is obsolete (Windows XP exploits don't matter anymore).
*   **Non-Independence:** If a cloud provider (AWS/Azure) goes down, *thousands* of policyholders claim at once. (Systemic Risk).

---

## 2. Mathematical Framework

### 2.1 Frequency-Severity Modeling

*   **Frequency:** Modeled using Poisson or Negative Binomial.
    *   *Driver:* Number of Employees, Industry (Healthcare is high target), Revenue.
*   **Severity:** Modeled using Lognormal or Pareto (Fat Tail).
    *   *Driver:* Record Count (How many credit cards do they store?).

### 2.2 CVE Scoring (CVSS)

*   **CVE (Common Vulnerabilities and Exposures):** A dictionary of known bugs.
*   **CVSS (Common Vulnerability Scoring System):** Score 0-10.
    *   **Base Score:** Intrinsic qualities (Is it easy to exploit? Does it give root access?).
    *   **Temporal Score:** Does a patch exist? Is exploit code available?
    *   *Actuarial Use:* A company with many unpatched High-CVSS vulnerabilities gets a higher premium.

---

## 3. Theoretical Properties

### 3.1 Silent Cyber

*   **Definition:** Cyber losses covered under non-cyber policies (e.g., Property policy covering a fire caused by a hacked thermostat).
*   **Trend:** Insurers are adding "Cyber Exclusions" to Property/GL policies to force clients to buy standalone Cyber.

### 3.2 Accumulation Risk

*   **Scenario:** A zero-day vulnerability in Windows 11.
*   **Impact:** Millions of computers infected instantly.
*   **Modeling:** We use "Counterfactual Scenarios" (e.g., "What if a solar storm hits?").

---

## 4. Modeling Artifacts & Implementation

### 4.1 Analyzing CVE Data (Python)

```python
import pandas as pd
import requests

# 1. Fetch CVE Data (NIST NVD Feed)
url = "https://services.nvd.nist.gov/rest/json/cves/2.0?keywordSearch=Microsoft Exchange"
response = requests.get(url)
data = response.json()

# 2. Parse Vulnerabilities
cves = []
for item in data['vulnerabilities']:
    cve_id = item['cve']['id']
    try:
        score = item['cve']['metrics']['cvssMetricV31'][0]['cvssData']['baseScore']
    except:
        score = 0
    cves.append({'ID': cve_id, 'Score': score})

df = pd.DataFrame(cves)
print(f"Average CVSS Score for Exchange: {df['Score'].mean():.2f}")
```

### 4.2 Shodan (The Hacker's Search Engine)

*   **Concept:** Shodan scans the entire internet 24/7.
*   **Risk Assessment:**
    *   Input: Company IP Range.
    *   Output: "You have 5 servers running Windows Server 2008 (End of Life)."
    *   *Action:* Deny coverage or demand patching.

---

## 5. Evaluation & Validation

### 5.1 Scenario Analysis

*   Instead of Backtesting (which fails due to lack of history), we use **Stress Testing**.
*   *Scenario:* "NotPetya 2.0". Assume a worm spreads via accounting software.
*   *Calculate:* Expected Loss = Exposure $\times$ Infection Rate $\times$ Cost per Victim.

### 5.2 Vendor Models

*   **RMS / AIR / CyberCube:** Specialized vendors who model the "Internet Topology".
*   Actuaries often aggregate these vendor outputs rather than building from scratch.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Inside-Out vs. Outside-In**
    *   **Outside-In:** Scanning IPs (Shodan). Easy but shallow.
    *   **Inside-Out:** Installing a probe on the client's network. Deep but intrusive.
    *   *Fix:* Use Outside-In for underwriting, Inside-Out for risk management services.

2.  **Trap: The "Human Factor"**
    *   You can have the best Firewall, but if an employee clicks a Phishing link, you are hacked.
    *   *Fix:* Include "Phishing Training" as a rating factor.

### 6.2 Implementation Challenges

1.  **Attribution:**
    *   Who hacked us? Russia? China? A teenager?
    *   *War Exclusion:* Policies exclude "Acts of War". Is a state-sponsored hack an Act of War? (Legally gray area).

---

## 7. Advanced Topics & Extensions

### 7.1 Cyber Catastrophe Bonds (ILS)

*   Insurers sell "Cyber Bonds" to investors to transfer the accumulation risk.
*   If losses > $1 Billion, investors lose their principal.

### 7.2 Zero-Trust Architecture

*   "Never trust, always verify."
*   Even if you are inside the network, you need MFA (Multi-Factor Authentication) to access files.
*   *Impact:* Reduces the "Lateral Movement" phase of the Kill Chain.

---

## 8. Regulatory & Governance Considerations

### 8.1 GDPR & CCPA

*   **Fines:** Up to 4% of Global Revenue for data breaches.
*   **Impact:** Cyber Insurance limits must be high enough to cover these massive fines.

---

## 9. Practical Example

### 9.1 Worked Example: The "Ransomware" Pricing

**Scenario:**
*   Manufacturing company. Revenue $100M.
*   **Risk Assessment:**
    *   Backups: Offline (Good).
    *   MFA: Enabled (Good).
    *   RDP Ports: Open to internet (BAD).
*   **Decision:**
    *   Base Premium: $50,000.
    *   Load for Open RDP: +50%.
    *   Subjectivity: "We will bind coverage ONLY if you close RDP ports."

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Kill Chain:** Stop the hack early.
2.  **Accumulation:** The biggest risk is a systemic cloud failure.
3.  **Human Factor:** Phishing is the #1 entry point.

### 10.2 When to Use This Knowledge
*   **Pricing:** Cyber Liability.
*   **Reserving:** Estimating IBNR for long-tail privacy claims.

### 10.3 Critical Success Factors
1.  **Dynamic Pricing:** Cyber risk changes monthly. Annual policies are too slow.
2.  **Partnerships:** Work with Cybersecurity firms (CrowdStrike, SentinelOne).

### 10.4 Further Reading
*   **Lockheed Martin:** "The Cyber Kill Chain".
*   **Eling & Wirfs:** "Cyber Risk: Too Big to Insure?".

---

## Appendix

### A. Glossary
*   **Zero-Day:** A vulnerability known to hackers but not the software vendor.
*   **White Hat:** Ethical hacker.
*   **Air Gap:** Keeping backups physically disconnected from the network.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **CVSS** | 0 to 10 | Vulnerability Severity |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
