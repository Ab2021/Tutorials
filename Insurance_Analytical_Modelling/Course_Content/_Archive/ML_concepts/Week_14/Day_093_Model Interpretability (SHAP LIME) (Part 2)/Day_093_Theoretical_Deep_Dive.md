# Cyber Risk Modelling (Part 2) - Theoretical Deep Dive

## Overview
Cyber risk isn't just about one computer getting hacked. It's about the *network*. Today, we treat Malware like a Virus (Epidemiology) and the Internet like a Social Network (Graph Theory) to model **Accumulation Risk**.

---

## 1. Conceptual Foundation

### 1.1 Network Theory in Cyber

*   **Nodes:** Computers, Servers, Routers.
*   **Edges:** Physical cables, Wi-Fi connections, Shared Software (e.g., everyone using SolarWinds).
*   **Hubs:** Nodes with massive connections (e.g., AWS, Google Cloud). If a Hub fails, the network collapses.

### 1.2 SIR Models (Epidemiology)

*   **Susceptible (S):** Unpatched computers.
*   **Infected (I):** Computers running the malware.
*   **Recovered (R):** Patched or disconnected computers.
*   *Analogy:* A computer virus spreads exactly like a biological virus (COVID-19), but faster.

### 1.3 Accumulation Risk (The "Cloud Down" Scenario)

*   **Scenario:** AWS US-East-1 goes down for 48 hours.
*   **Impact:** Netflix, Uber, Slack, and 10,000 small businesses stop working.
*   **Insurance Impact:** Business Interruption (BI) claims from *all* of them simultaneously.

---

## 2. Mathematical Framework

### 2.1 The SIR Differential Equations

$$ \frac{dS}{dt} = -\beta S I $$
$$ \frac{dI}{dt} = \beta S I - \gamma I $$
$$ \frac{dR}{dt} = \gamma I $$
*   $\beta$: Infection Rate (How contagious is the malware?).
*   $\gamma$: Recovery Rate (How fast do IT teams patch it?).
*   *Basic Reproduction Number ($R_0$):* If $R_0 > 1$, the malware becomes a pandemic.

### 2.2 Probable Maximum Loss (PML)

*   **Definition:** The loss level that will not be exceeded with 99.5% probability (1-in-200 year event).
*   **Calculation:** Run 10,000 simulations of the SIR model on the portfolio network. Take the 99.5th percentile loss.

---

## 3. Theoretical Properties

### 3.1 Scale-Free Networks

*   The Internet is a **Scale-Free Network** (Power Law distribution).
*   Most nodes have few connections. A few nodes (Hubs) have millions.
*   *Vulnerability:* The network is robust to random failure, but fragile to targeted attacks on Hubs.

### 3.2 Percolation Threshold

*   The point at which a virus spreads to the "Giant Component" (the whole network).
*   For Scale-Free networks, the percolation threshold is effectively zero (viruses always spread).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Simulating Malware with NetworkX

```python
import networkx as nx
import matplotlib.pyplot as plt
import random

# 1. Create a Network (Barabasi-Albert Scale-Free Graph)
G = nx.barabasi_albert_graph(n=1000, m=3) # 1000 computers

# 2. Initialize Status
status = {node: 'S' for node in G.nodes()}
initial_infected = random.sample(G.nodes(), 5)
for node in initial_infected:
    status[node] = 'I'

# 3. Simulation Step
def step(G, status, beta=0.1, gamma=0.05):
    new_status = status.copy()
    for node in G.nodes():
        if status[node] == 'I':
            # Recovery
            if random.random() < gamma:
                new_status[node] = 'R'
            # Infect Neighbors
            for neighbor in G.neighbors(node):
                if status[neighbor] == 'S' and random.random() < beta:
                    new_status[neighbor] = 'I'
    return new_status

# 4. Run Simulation
history = []
for t in range(50):
    status = step(G, status)
    counts = {'S': 0, 'I': 0, 'R': 0}
    for s in status.values():
        counts[s] += 1
    history.append(counts)

# Plot
import pandas as pd
pd.DataFrame(history).plot()
plt.title("Malware Propagation (SIR Model)")
plt.show()
```

### 4.2 Calculating Portfolio Accumulation

*   **Input:** List of Policyholders and their Cloud Provider (AWS, Azure, GCP).
*   **Matrix:** $A_{ij} = 1$ if Policyholder $i$ uses Provider $j$.
*   **Scenario:** Provider $j$ fails.
*   **Loss:** $\sum_i (\text{Limit}_i \times A_{ij})$.

---

## 5. Evaluation & Validation

### 5.1 Counterfactual Analysis

*   "What if WannaCry (2017) happened today?"
*   We replay historical events on the *current* portfolio to estimate loss.

### 5.2 Sensitivity Testing

*   What if the Infection Rate ($\beta$) doubles?
*   What if the Patching Rate ($\gamma$) is half as fast?
*   *Result:* Cyber risk is extremely sensitive to $\beta$. A slightly better virus causes 10x losses.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Assuming Independence**
    *   "Company A getting hacked doesn't affect Company B."
    *   *Wrong:* If they both use the same MSP (Managed Service Provider), they fall together.

2.  **Trap: The "Air Gap" Myth**
    *   "Our critical servers are not connected to the internet."
    *   *Reality:* Stuxnet jumped an air gap via a USB stick.

### 6.2 Implementation Challenges

1.  **Data Mapping:**
    *   We know who our policyholders are. We *don't* know who their vendors are.
    *   *Fix:* Supply Chain Risk Management (SCRM) tools.

---

## 7. Advanced Topics & Extensions

### 7.1 SIS Models (Endemic Malware)

*   **Susceptible -> Infected -> Susceptible.**
*   Computers get infected, cleaned, and then infected again (no immunity).
*   Models persistent threats like Botnets.

### 7.2 Game Theory (Attacker vs. Defender)

*   Modeling the Hacker's incentives.
*   If we increase defense cost, does the hacker give up or just try harder?

---

## 8. Regulatory & Governance Considerations

### 8.1 Solvency II & Cyber

*   Regulators require insurers to hold capital for a "1-in-200 year Cyber Event".
*   Since we can't estimate this from history, we rely heavily on these SIR/Network models.

---

## 9. Practical Example

### 9.1 Worked Example: The "Cloud Outage" PML

**Scenario:**
*   Portfolio: 500 Tech Companies.
*   **Dependency Mapping:**
    *   300 use AWS.
    *   150 use Azure.
    *   50 use On-Premise.
*   **Scenario:** AWS East Region fails for 24 hours.
*   **Loss Calculation:**
    *   300 companies claim Business Interruption.
    *   Avg Limit: $1M.
    *   Waiting Period: 12 hours.
    *   Claimable Hours: 12.
    *   Hourly Loss: $10k.
    *   Total Loss: $300 \times 12 \times 10k = $36M.
*   **Result:** This single event wipes out 5 years of premium.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **SIR Models** predict how fast malware spreads.
2.  **Hubs** (Cloud Providers) are the single points of failure.
3.  **Accumulation** is the main reason Cyber Insurance is expensive.

### 10.2 When to Use This Knowledge
*   **Capital Modeling:** Determining how much money to hold.
*   **Reinsurance:** Buying protection against the "Big One".

### 10.3 Critical Success Factors
1.  **Map the Network:** You can't model what you can't see.
2.  **Speed:** In Cyber, 24 hours makes the difference between a nuisance and a catastrophe.

### 10.4 Further Reading
*   **Newman:** "Networks: An Introduction".
*   **Anderson:** "Security Engineering".

---

## Appendix

### A. Glossary
*   **Botnet:** A network of infected computers controlled by a hacker.
*   **MSP:** Managed Service Provider (IT support for small businesses).
*   **PML:** Probable Maximum Loss.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **SIR Infection** | $dI/dt = \beta SI - \gamma I$ | Malware Spread |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
