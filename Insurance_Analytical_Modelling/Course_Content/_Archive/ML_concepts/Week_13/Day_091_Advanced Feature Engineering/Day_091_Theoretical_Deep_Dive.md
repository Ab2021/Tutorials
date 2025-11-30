# Telematics Data Analysis (Part 2) - Theoretical Deep Dive

## Overview
Telematics isn't just about "Hard Braking". It's about context. Is this a commute? Is the driver texting? Is it even the policyholder driving? This session covers **Trip Classification**, **Driver Fingerprinting**, and **Distracted Driving**.

---

## 1. Conceptual Foundation

### 1.1 Trip Classification

*   **Goal:** Label each trip (Commute, Shopping, School Run, Road Trip).
*   **Why?**
    *   *Commute:* High frequency, rush hour traffic. Moderate risk.
    *   *Road Trip:* Highway miles. Low risk per mile.
    *   *Late Night:* High risk.
*   **Features:** Time of Day, Trip Duration, Start/End Location, Road Type mix.

### 1.2 Driver Fingerprinting

*   **Problem:** A policy has 3 drivers (Dad, Mom, Teen). Who is driving right now?
*   **Solution:** Every driver has a unique "Signature".
    *   *Dad:* Accelerates slowly, brakes early.
    *   *Teen:* Accelerates fast, brakes late, corners hard.
*   **Algorithm:** Supervised Classification (Random Forest) or Anomaly Detection (Isolation Forest).

### 1.3 Distracted Driving

*   **Phone Handling:** Accelerometer detects "Pick up" motion.
*   **Phone Usage:** App detects "Screen On" or "Call Active".
*   **Risk:** Texting increases accident risk by 23x.

---

## 2. Mathematical Framework

### 2.1 Feature Extraction for Fingerprinting

*   We don't use raw data. We calculate **Summary Statistics** per trip:
    *   Mean Speed, Max Speed.
    *   Std Dev of Acceleration (Jerky vs. Smooth).
    *   % of time above Speed Limit.
    *   Turn radius vs. Speed (Cornering aggression).

### 2.2 Mileage Verification (PAYD)

*   **Traditional:** User self-reports "10,000 miles". (Lies).
*   **Telematics:** GPS calculates exact mileage.
*   **Equation:**
    $$ \text{Premium} = \text{Base Rate} + (\text{Rate per Mile} \times \text{Miles}) $$

---

## 3. Theoretical Properties

### 3.1 The "Commute" Pattern

*   **Regularity:** Commutes happen at the same time, same route, M-F.
*   **Clustering:** DBSCAN on Start/End points can identify "Home" and "Work" locations automatically.

### 3.2 Anomaly Detection for Theft

*   If the car is driven at 3 AM by a driver with a "Smooth" signature (unlike the owner), and the location is new...
*   *Alert:* Potential Theft.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Driver Identification (Random Forest)

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 1. Load Feature Table (One row per trip)
# Features: [Mean_Accel, Max_Speed, Cornering_Intensity, Time_of_Day]
# Label: Driver_ID (0=Mom, 1=Dad, 2=Teen)
df = pd.read_csv('driver_signatures.csv')

X = df.drop('Driver_ID', axis=1)
y = df['Driver_ID']

# 2. Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# 3. Predict New Trip
new_trip = [[0.5, 65, 0.2, 14]] # Smooth accel, Highway speed, Gentle turn, 2 PM
print(f"Predicted Driver: {clf.predict(new_trip)}")
```

### 4.2 Trip Purpose Clustering

```python
from sklearn.cluster import DBSCAN

# Cluster End Locations to find "Frequent Destinations"
coords = df[['End_Lat', 'End_Lon']]
db = DBSCAN(eps=0.001, min_samples=5).fit(coords) # eps in degrees (~100m)

df['Destination_Cluster'] = db.labels_
# Cluster -1 = Random spot (Shopping?)
# Cluster 0 = Home
# Cluster 1 = Work
```

---

## 5. Evaluation & Validation

### 5.1 Confusion Matrix (Driver ID)

*   **Accuracy:** How often do we get the driver right?
*   **Impact:** If we mistake the Teen (High Risk) for the Mom (Low Risk), we underprice the trip.

### 5.2 Privacy vs. Utility

*   **Fingerprinting** is invasive.
*   **Validation:** Ensure the model relies on *driving style*, not *destination* (to protect privacy).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Shared Styles**
    *   Husband and Wife might drive very similarly.
    *   *Fix:* Use "Time of Day" as a tie-breaker (e.g., Wife works night shift).

2.  **Trap: Short Trips**
    *   A 2-minute trip to the grocery store doesn't have enough data to identify the driver.
    *   *Fix:* Label as "Unknown" or aggregate with previous trip.

### 6.2 Implementation Challenges

1.  **Battery Optimization:**
    *   Uploading data in real-time kills battery.
    *   *Fix:* Upload via Wi-Fi at home. Process in batch.

---

## 7. Advanced Topics & Extensions

### 7.1 Crash Detection

*   **Algorithm:** High G-Force spike + Sudden Stop + Airbag Deployment signal (if OBD).
*   **Action:** Auto-dial 911.

### 7.2 Gamification

*   **App:** "You are in the top 10% of drivers! Here is a Starbucks badge."
*   **Psychology:** Positive reinforcement works better than "You are a bad driver".

---

## 8. Regulatory & Governance Considerations

### 8.1 Discrimination

*   **Risk:** "Zip Code" is a proxy for Race.
*   **Telematics:** "Hard Braking" is NOT a proxy for Race. It is behavioral.
*   **Benefit:** Telematics is arguably the fairest rating factor.

---

## 9. Practical Example

### 9.1 Worked Example: The "Teen" Tracker

**Scenario:**
*   Family Policy. Teenager added. Premium doubles.
*   **Offer:** "Install this App. If Teen drives safely, 20% cash back."
*   **Data:**
    *   Teen drives smoothly (Score 90).
    *   BUT, Teen drives at 1 AM on Fridays (High Risk).
*   **Outcome:**
    *   App alerts Parents: "Late night driving detected."
    *   Parents intervene. Risk reduced.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Fingerprinting** solves the "Who is driving?" problem.
2.  **Trip Classification** adds context.
3.  **Distraction** is the new drunk driving.

### 10.2 When to Use This Knowledge
*   **Product:** Designing a UBI App.
*   **Claims:** Verifying facts of loss.

### 10.3 Critical Success Factors
1.  **User Experience:** The App must not drain battery.
2.  **Transparency:** Tell the user *why* their score dropped.

### 10.4 Further Reading
*   **Diro et al.:** "Driver Identification via Brake Pedal Signals".

---

## Appendix

### A. Glossary
*   **OBD-II:** On-Board Diagnostics port.
*   **PAYD:** Pay-As-You-Drive (Mileage based).
*   **PHYD:** Pay-How-You-Drive (Behavior based).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Premium** | $Base + (Rate \times Miles)$ | PAYD |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
