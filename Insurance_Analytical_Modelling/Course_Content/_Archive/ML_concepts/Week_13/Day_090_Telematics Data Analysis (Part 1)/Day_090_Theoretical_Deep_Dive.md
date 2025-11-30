# Telematics Data Analysis (Part 1) - Theoretical Deep Dive

## Overview
Insurance is moving from "Who you are" (Age, Gender) to "How you drive" (Telematics). This session covers the data engineering pipeline: From raw GPS pings to a **Driver Safety Score**.

---

## 1. Conceptual Foundation

### 1.1 The Data Source

*   **OBD-II Dongle:** Plugs into the car. Reads speed, RPM, fuel.
*   **Mobile App:** Uses phone's GPS and Accelerometer.
*   **OEM (Embedded):** Built into the car (Tesla, GM OnStar).
*   **Data Frequency:** usually 1 Hz (1 reading per second).

### 1.2 Key Events (The "Big 3")

1.  **Hard Braking:** Deceleration > $x$ g-force (e.g., -0.3g). Correlates highly with accident risk.
2.  **Rapid Acceleration:** Acceleration > $y$ g-force. Indicates aggression.
3.  **Cornering:** Lateral g-force. Indicates taking turns too fast.

### 1.3 Map Matching

*   **Problem:** GPS is noisy. A point might look like it's in a river or a building.
*   **Solution:** Snap the points to the nearest logical road segment.
*   **OSRM (Open Source Routing Machine):** A tool to find the most likely path a car took given a sequence of noisy points.

---

## 2. Mathematical Framework

### 2.1 Calculating G-Force

*   If you have Speed ($v$) and Time ($t$):
    $$ a = \frac{v_t - v_{t-1}}{\Delta t} $$
*   Convert $a$ (in $m/s^2$) to g-force ($1g = 9.8 m/s^2$).
*   *Example:* Going from 60mph to 0mph in 3 seconds is a Hard Brake.

### 2.2 The PHYD Score (Pay-How-You-Drive)

$$ Score = 100 - (w_1 \times \text{Brakes} + w_2 \times \text{Speeding} + w_3 \times \text{NightDriving}) $$
*   Weights ($w$) are determined by a GLM predicting Loss Frequency.
*   *Night Driving:* Driving at 2 AM is riskier than 2 PM (Drunk drivers, fatigue).

---

## 3. Theoretical Properties

### 3.1 Sampling Rate

*   **1 Hz (1 sec):** Good for speed and location.
*   **10 Hz (0.1 sec):** Needed for crash detection (impact happens in milliseconds).
*   **50 Hz:** Needed for analyzing suspension health (potholes).

### 3.2 Context Matters

*   **Speeding:** Driving 80mph is bad?
    *   On a School Zone (Limit 20): **Yes.**
    *   On a Highway (Limit 75): **No.**
*   *Requirement:* You need an external database of Speed Limits (e.g., OpenStreetMap or Here Maps).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Feature Engineering with Pandas

```python
import pandas as pd
import numpy as np

# Load Data (Time, Lat, Lon, Speed_mps)
df = pd.read_csv('trip_data.csv')

# Calculate Acceleration
df['delta_v'] = df['Speed_mps'].diff()
df['delta_t'] = df['Time'].diff()
df['accel'] = df['delta_v'] / df['delta_t']

# Flag Events
HARD_BRAKE_THRESH = -3.0 # m/s^2
df['is_hard_brake'] = df['accel'] < HARD_BRAKE_THRESH

print(f"Total Hard Brakes: {df['is_hard_brake'].sum()}")
```

### 4.2 Visualization with Folium

```python
import folium

# Create Map centered on start
m = folium.Map(location=[df['Lat'][0], df['Lon'][0]], zoom_start=14)

# Plot Route
points = list(zip(df['Lat'], df['Lon']))
folium.PolyLine(points, color="blue", weight=2.5, opacity=1).add_to(m)

# Add Markers for Hard Brakes
brakes = df[df['is_hard_brake']]
for idx, row in brakes.iterrows():
    folium.Marker(
        [row['Lat'], row['Lon']],
        popup=f"Hard Brake: {row['accel']:.2f} m/s^2",
        icon=folium.Icon(color='red', icon='exclamation-sign')
    ).add_to(m)

m.save('trip_map.html')
```

---

## 5. Evaluation & Validation

### 5.1 Correlation with Claims

*   Does the "Score" actually predict risk?
*   **Validation:** Split drivers into Deciles based on Score.
*   *Check:* Does Decile 1 (Best Drivers) have a lower Loss Ratio than Decile 10 (Worst Drivers)?
*   *Target:* We want a "Lift" of 3x-5x between best and worst.

### 5.2 False Positives

*   **Phone Drop:** If the phone falls off the dashboard, the accelerometer registers a huge spike.
*   *Fix:* Filter out events where the GPS speed didn't change significantly.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Passenger vs. Driver**
    *   The App is running, but the user is a passenger in a Taxi.
    *   *Fix:* Machine Learning classification based on phone movement (distracted handling) or connection to car Bluetooth.

2.  **Trap: Tunnel Vision**
    *   GPS cuts out in tunnels.
    *   *Fix:* Dead Reckoning (interpolate position based on last known speed).

### 6.2 Implementation Challenges

1.  **Battery Drain:**
    *   Continuous GPS kills phone battery.
    *   *Fix:* Only wake up GPS when the accelerometer detects sustained motion.

---

## 7. Advanced Topics & Extensions

### 7.1 Map Matching (Hidden Markov Model)

*   OSRM uses HMMs.
*   It considers:
    1.  **Emission Probability:** How close is the point to the road?
    2.  **Transition Probability:** Is it physically possible to move from Road A to Road B in 1 second?

### 7.2 Crash Reconstruction

*   Using the 5 seconds of data *before* a crash to determine liability.
*   "Driver was speeding (85mph) and braking late (-0.8g) before impact."

---

## 8. Regulatory & Governance Considerations

### 8.1 Privacy

*   **GDPR:** Tracking location is highly sensitive.
*   **Consent:** User must explicitly opt-in.
*   **Right to be Forgotten:** User can delete their trip history.

---

## 9. Practical Example

### 9.1 Worked Example: The "Safe Driver" Discount

**Scenario:**
*   Insurer offers 10% discount for signing up. Up to 30% at renewal.
*   **Driver A:** 0 hard brakes, drives mostly on highways. Score: 95. Discount: 30%.
*   **Driver B:** 5 hard brakes/100 miles, drives at 2 AM. Score: 60. Discount: 0%.
*   **Result:** Driver A stays (Retention). Driver B leaves (Adverse Selection avoided).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **G-Force** measures aggression.
2.  **Map Matching** cleans the data.
3.  **Context** (Speed Limits) is crucial.

### 10.2 When to Use This Knowledge
*   **UBI:** Usage-Based Insurance.
*   **Fleet:** Monitoring delivery trucks.

### 10.3 Critical Success Factors
1.  **Filter Noise:** Don't punish drivers for dropping their phone.
2.  **Feedback:** Show the driver *where* they braked hard so they can improve.

### 10.4 Further Reading
*   **Handel et al.:** "Insurance Telematics: Opportunities and Challenges".

---

## Appendix

### A. Glossary
*   **Accelerometer:** Measures proper acceleration.
*   **Gyroscope:** Measures rotation (yaw, pitch, roll).
*   **Geofence:** A virtual boundary (e.g., "Leaving the state").

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Acceleration** | $\Delta v / \Delta t$ | Event Detection |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
