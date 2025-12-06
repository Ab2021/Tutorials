# Day 146: Ethics in Robotics (Jobs, Safety, Bias)
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 21: Collaborative Robotics (Cobots)

---

> **📝 Content Creator Instructions:**
> With great power comes great liability.
> - **Focus:** Algorithmic Bias in Vision (Pedestrian detection failure rates), The "Trolley Problem" in AVs, Legal Liability (Manufacturer vs Operator), and Job Displacement economics.
> - **Code:** A Python script `fairness_audit.py` that analyzes a synthetic "Pedestrian Detection" dataset, identifying bias (Higher False Negative rates for certain groups) and applying a Fairness Constraint (Equalized Odds) to re-threshold the model.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Identify** sources of bias in robotic perception (Data Imbalance, Sensor Limitations).
2.  **Debate** the ethical dilemma of Autonomous Vehicles (Utilitarian vs Deontological ethics).
3.  **Explain** "Algorithmic Fairness" metrics (Demographic Parity, Equal Opportunity).
4.  **Audit** a classifier for disparate impact.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install numpy matplotlib sklearn
```

### Prior Knowledge
- Machine Learning metrics (Precision, Recall, ROC Curve).
- Part 1 of the Course (Computer Vision).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Bias in the Machine

Cameras are physics. Algorithms are math. How can they be racist?
*   **Data Bias:** If 90% of training pedestrians are adult males, the model might fail to detect children or wheelchair users.
*   **Sensor Bias:** Some Time-of-Flight sensors struggle with dark surfaces (absorbing IR) or certain hair types.
*   **Impact:** A self-driving car might brake later for some demographic groups.

### 🔹 Part 2: The Trolley Problem 2.0

An AV loses brakes.
*   Path A: Hit 1 Pedestrian.
*   Path B: Hit Concrete Wall (Kill Passenger).
*   **Ethics:** Who decides? The Engineer? The Government? The Machine?
*   **Reality:** Most AV companies program "Maintain Lane" and "Brake Hard". They avoid making "decisions" about value of life.

### 🔹 Part 3: Liability

Since a Neural Net is a "Black Box", who is sued when it crashes?
*   **The Operator:** Did they ignore warnings?
*   **The Manufacturer:** Did they ship a defect?
*   **The Regulation:** Currently evolving to "Strict Liability" for manufacturers.

---

## 💻 Implementation: The Fairness Audit

We simulate a Pedestrian Detector that is biased against "Group B" (e.g., Children/Wheelchairs) due to less training data.

### 🛠️ Project Structure
```text
day146_ethics/
├── src/
│   ├── fairness_audit.py
└── output/
    ├── fairness_report.png
```

### 👨‍💻 Audit Script (`src/fairness_audit.py`)

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

def generate_data(n=1000):
    # Group A: Majority (Adults) - 80%
    # Group B: Minority (Children) - 20%
    
    n_a = int(n * 0.8)
    n_b = int(n * 0.2)
    
    # Ground Truth (Is there a person?) 50/50 split
    y_true_a = np.random.randint(0, 2, n_a)
    y_true_b = np.random.randint(0, 2, n_b)
    
    # Model Scores (Confidence 0.0-1.0)
    # Simulator: Model is BETTER at detecting Group A
    
    # Group A (Good Distribution)
    # If True=1, Score ~ Normal(0.8, 0.1)
    # If True=0, Score ~ Normal(0.2, 0.1)
    scores_a = np.zeros(n_a)
    scores_a[y_true_a==1] = np.random.normal(0.8, 0.1, np.sum(y_true_a))
    scores_a[y_true_a==0] = np.random.normal(0.2, 0.1, np.sum(y_true_a==0))
    
    # Group B (Bad Distribution - Harder to see)
    # If True=1, Score ~ Normal(0.6, 0.2) -- Lower mean, higher variance
    # If True=0, Score ~ Normal(0.3, 0.2)
    scores_b = np.zeros(n_b)
    scores_b[y_true_b==1] = np.random.normal(0.6, 0.2, np.sum(y_true_b))
    scores_b[y_true_b==0] = np.random.normal(0.3, 0.2, np.sum(y_true_b==0))
    
    return (y_true_a, scores_a), (y_true_b, scores_b)

def calculate_metrics(y_true, scores, threshold):
    preds = scores > threshold
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    tpr = tp / (tp + fn) # Recall
    fpr = fp / (fp + tn)
    return tpr, fpr

def main():
    (ya, sa), (yb, sb) = generate_data()
    
    # 1. Blind Thresholding
    # Standard approach: Pick threshold=0.5 for everyone
    thresh = 0.5
    
    tpr_a, fpr_a = calculate_metrics(ya, sa, thresh)
    tpr_b, fpr_b = calculate_metrics(yb, sb, thresh)
    
    print("--- Default Algo (Threshold 0.5) ---")
    print(f"Group A (Majority) Limit: TPR={tpr_a:.2f}, FPR={fpr_a:.2f}")
    print(f"Group B (Minority) Limit: TPR={tpr_b:.2f}, FPR={fpr_b:.2f}")
    print(f"Bias Gap (Recall): {tpr_a - tpr_b:.2f}")
    print("Result: Group B is hit by car more often (Low Recall).")
    
    # 2. Fairness Intervention (Equalized Odds)
    # We want TPR_a ~= TPR_b
    # Lower threshold for Group B to boost recall?
    
    thresh_b_new = 0.35 # Iterate to find this
    tpr_b_new, fpr_b_new = calculate_metrics(yb, sb, thresh_b_new)
    
    print("\n--- Fairness Adjusted (Threshold B = 0.35) ---")
    print(f"Group B New Metrics: TPR={tpr_b_new:.2f}, FPR={fpr_b_new:.2f}")
    print(f"New Bias Gap: {tpr_a - tpr_b_new:.2f}")
    print("Result: Recall is equal, BUT False Positive Rate for B went up.")
    print("Tradeoff: Car stops for 'ghosts' (FP) more often for Group B to ensure safety.")

    # Plot Distributions
    plt.figure()
    plt.hist(sa[ya==1], bins=20, alpha=0.5, label='A True', color='blue')
    plt.hist(sb[yb==1], bins=20, alpha=0.5, label='B True', color='orange')
    plt.axvline(0.5, color='black', linestyle='--', label='Thresh Default')
    plt.axvline(0.35, color='red', linestyle='--', label='Thresh Adjusted')
    plt.title("Confidence Distributions (Positives Only)")
    plt.xlabel("Model Confidence")
    plt.ylabel("Count")
    plt.legend()
    plt.savefig("output/fairness_report.png")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "Pareto Frontier"

### 1. Lab Objectives
- **Run:** The script.
- **Observe:** Default threshold misses more Group B (Children).
- **Modify:** Adjust threshold to equalize Recall (Safety).
- **Result:** FPR (False Stops) increases dramatically for Group B.
- **Discussion:** Is it ethical to have more "Phantom Braking" if it saves lives? Yes. But it degrades the user experience (Comfort). This is the engineering trade-off.

---

## 🚀 Project: "Model Card"

**Goal:** Documentation.
1.  **Create:** A Markdown file `model_card.md` for a Pedestrian Detector.
2.  **Sections:**
    *   **Intended Use:** Autonomous Driving.
    *   **Limitations:** Poor performance in heavy rain. Poor performance on recumbent bicycles.
    *   **Training Data:** 100k images from California (Sunny).
    *   **Bias Warning:** May underperform in snowy regions.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Ignoring Bias"
*   **Symptom:** "My accuracy is 99%!".
*   **Reality:** 99% on majority, 60% on minority. Global accuracy hides local failure.
*   **Fix:** Report metrics *per subgroup*.

#### 2. "Simpson's Paradox"
*   **Symptom:** Trend appears in different groups of data but disappears or reverses when these groups are combined.

---

## ⚡ Optimization: Active Learning for Bias

If model is unsure about Group B:
1.  Detect Low Confidence scenarios.
2.  Request human labeling specifically for these edge cases.
3.  Retrain.
*   **Result:** Targeted data collection is cheaper than "Collect Everything".

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Automation Bias"?
    *   **A:** Humans tend to trust the machine too much. "The GPS said turn left into the lake, so I did." Operators must remain vigilant.
2.  **Q:** Impact of Cobots on Jobs?
    *   **A:** Dull/Dangerous jobs (Welding, Palletizing) go to Cobots. New jobs: Cobot Programmer, Maintenance. Net effect debated, but shift in skill requirement is real.
3.  **Q:** Deontological vs Consequentialist?
    *   **A:** Deontological: "Rules" (Never cross double line). Consequentialist: "Outcome" (Cross line to avoid hitting child). AVs struggle with strict rule following vs safety.

### Challenge Task
> **Task:** The CEO Decision.
> 1. YourAV saves 5 lives but runs over a dog.
> 2. CompetitorAV hits the wall, saving the dog but killing the passenger.
> 3. Which car do people *buy* vs which car do people *want to exist*? (Social Dilemma).

---

## 📚 Further Reading
- **MIT Moral Machine:** Study on human perspective of Trolley Problem.
- **Algorithmic Justice League:** Bias in AI.

---

**Day 146 Complete**
