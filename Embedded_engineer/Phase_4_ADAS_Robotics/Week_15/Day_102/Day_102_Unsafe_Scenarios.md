# Day 102: Known and Unknown Unsafe Scenarios
## Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)

---

> **📝 Day 102 Focus:**
> The scariest bugs are the ones you don't know exist. In SOTIF, we classify scenarios into 4 quadrants. The goal is to move everything from **Area 3 (Unknown Unsafe)** to **Area 1 (Known Safe)**. Today, we build a database to track these monsters.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Classify** scenarios into the SOTIF Matrix (Known/Unknown vs Safe/Unsafe).
2.  **Implement** a Scenario Database (SQLite/JSON) to track edge cases.
3.  **Design** a "Trigger Monitor" to detect new scenarios in fleet data.
4.  **Explain** the concept of ODD (Operational Design Domain) refinement.
5.  **Simulate** the discovery of an "Unknown Unknown".

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 101:** SOTIF Basics.
-   **Databases:** Basic SQL.

### Hardware Requirements
-   **None:** Data engineering day.

### Software Stack
-   **Python:** `sqlite3`, `pandas`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Taxonomy of Edge Cases

1.  **Known Safe (Area 1):** Highway driving, clear weather. (99% of data).
2.  **Known Unsafe (Area 2):** Heavy snow, Sensor blockage.
    -   *Action:* Design mitigations (e.g., "If snow, hand over to driver").
3.  **Unknown Unsafe (Area 3):** "The Kangaroo Problem".
    -   Volvo tested detection in Sweden (Moose/Deer).
    -   Deployed in Australia. Car crashed into Kangaroo.
    -   Why? Kangaroos jump. The AI had never seen a jumping animal.
    -   *Action:* Discover via testing -> Move to Area 2 -> Fix -> Move to Area 1.
4.  **Unknown Safe (Area 4):** Scenarios that happen but don't cause crashes. (e.g., A pink car).

### 🔹 Part 2: Discovery Strategies

How do we find Area 3?
-   **Fuzzing:** Randomize simulation parameters (Sun angle, Friction).
-   **Fleet Monitoring:** Trigger data recording on "Near Miss" (Hard braking, Swerving).
-   **Crowdsourcing:** "Disengagement Reports" from test drivers.

### 🔹 Part 3: ODD Refinement

When we find an Area 2 scenario we can't fix, we shrink the ODD.
-   *Initial ODD:* "All Highways".
-   *Discovery:* Tunnel exits cause camera blindness.
-   *Refined ODD:* "All Highways EXCEPT Tunnels".

---

## 💻 Implementation: Scenario Database

**Scenario:**
-   We have a fleet of cars.
-   We receive "Events" (Disengagements).
-   We classify them and update our ODD.

### 🛠️ Setup
Create `week15_day102` and `scenario_db.py`.

```bash
mkdir -p ~/ros2_ws/src/week15_day102
cd ~/ros2_ws/src/week15_day102
touch scenario_db.py
```

### 👨‍💻 Code: The Edge Case Tracker

```python
import sqlite3
import pandas as pd
from datetime import datetime

class ScenarioDB:
    def __init__(self):
        self.conn = sqlite3.connect("scenarios.db")
        self.create_tables()
        
    def create_tables(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS scenarios (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                description TEXT,
                weather TEXT,
                location_type TEXT,
                sotif_area INTEGER, -- 1, 2, 3, 4
                mitigation_status TEXT -- 'OPEN', 'FIXED', 'ODD_EXCLUDED'
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scenario_id INTEGER,
                timestamp DATETIME,
                severity INTEGER
            )
        ''')
        self.conn.commit()

    def add_scenario(self, desc, weather, loc, area):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO scenarios (description, weather, location_type, sotif_area, mitigation_status) VALUES (?, ?, ?, ?, 'OPEN')",
                       (desc, weather, loc, area))
        self.conn.commit()
        return cursor.lastrowid

    def log_event(self, scenario_id, severity):
        cursor = self.conn.cursor()
        cursor.execute("INSERT INTO events (scenario_id, timestamp, severity) VALUES (?, ?, ?)",
                       (scenario_id, datetime.now(), severity))
        self.conn.commit()

    def get_area_3_report(self):
        return pd.read_sql_query("SELECT * FROM scenarios WHERE sotif_area = 3", self.conn)

    def promote_to_area_2(self, scenario_id):
        # We found it! Now it's Known Unsafe.
        cursor = self.conn.cursor()
        cursor.execute("UPDATE scenarios SET sotif_area = 2 WHERE id = ?", (scenario_id,))
        self.conn.commit()
        print(f"Scenario {scenario_id} moved to Area 2 (Known Unsafe).")

    def fix_scenario(self, scenario_id):
        # We fixed the bug. Now it's Known Safe.
        cursor = self.conn.cursor()
        cursor.execute("UPDATE scenarios SET sotif_area = 1, mitigation_status = 'FIXED' WHERE id = ?", (scenario_id,))
        self.conn.commit()
        print(f"Scenario {scenario_id} moved to Area 1 (Known Safe).")

def main():
    db = ScenarioDB()
    
    # 1. Initial Knowledge (Design Phase)
    s1 = db.add_scenario("Highway Sunny", "Clear", "Highway", 1)
    s2 = db.add_scenario("Heavy Snow", "Snow", "Highway", 2)
    
    # 2. Fleet Operation (Discovery)
    print("--- Fleet Running ---")
    # Driver reports a disengagement at a Tunnel Exit
    print("Event: Disengagement at Tunnel Exit due to Glare.")
    
    # Is this in DB? No. It's an Unknown.
    # Add it as Area 2 (Now we know it)
    s3 = db.add_scenario("Tunnel Exit Glare", "Clear", "Tunnel", 2)
    db.log_event(s3, severity=3)
    
    # 3. Another Event
    # Car swerves for a plastic bag.
    print("Event: Phantom swerve for plastic bag.")
    s4 = db.add_scenario("Plastic Bag on Road", "Windy", "City", 2)
    db.log_event(s4, severity=1)
    
    # 4. Analysis
    print("\n--- Scenario Database ---")
    df = pd.read_sql_query("SELECT * FROM scenarios", db.conn)
    print(df[['id', 'description', 'sotif_area', 'mitigation_status']])
    
    # 5. Engineering Fix
    print("\n--- Engineering Action ---")
    # We train the AI on plastic bags.
    db.fix_scenario(s4)
    
    # We decide Tunnels are too hard. Exclude from ODD.
    cursor = db.conn.cursor()
    cursor.execute("UPDATE scenarios SET mitigation_status = 'ODD_EXCLUDED' WHERE id = ?", (s3,))
    db.conn.commit()
    print(f"Scenario {s3} Excluded from ODD.")
    
    print("\n--- Final Status ---")
    df = pd.read_sql_query("SELECT * FROM scenarios", db.conn)
    print(df[['id', 'description', 'sotif_area', 'mitigation_status']])

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Unknown Unknown

### Lab Objectives
1.  Run the script.
2.  **Observation:**
    -   We start with knowns.
    -   Events happen (Tunnel Glare, Plastic Bag).
    -   We catalog them.
    -   We decide: Fix it (Plastic Bag) or Exclude it (Tunnel).
3.  **Thought Experiment:**
    -   Imagine a "Reflection of a Stop Sign in a puddle".
    -   Is it Area 3? Yes, until you see it.
    -   How to find it? **Anomaly Detection**. If the car brakes but there is no object in the LiDAR, flag the clip for review.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Data Deluge
**Symptom:** Fleet uploads TBs of data. Engineers overwhelmed.
**Cause:** Triggering on everything.
**Solution:** **Smart Triggers**. Only upload 10s before/after a disengagement or hard brake.

#### 2. Duplicate Scenarios
**Symptom:** "Plastic Bag" and "Trash on Road" listed separately.
**Cause:** Manual entry.
**Solution:** Use **Clustering**. Group events by similarity (embedding vectors of the scene) to find common patterns.

---

## ⚡ Optimization & Best Practices

### 1. Synthetic Data Generation
Once you find an Area 3 scenario (e.g., Kangaroo), don't wait for another one.
-   **Simulate it:** Create a 3D Kangaroo model.
-   **Vary it:** Change size, color, jump height.
-   **Train:** Feed this to the AI.
-   This turns one data point into 10,000 training examples.

### 2. Continuous Integration
Every night, run the "Regression Test Suite".
-   Includes all Area 2 scenarios we fixed.
-   Ensures the new model didn't forget how to handle plastic bags.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the goal of SOTIF testing?
    *   **A:** To reduce the size of Area 2 (Unsafe) and Area 3 (Unknown).
2.  **Q:** What is an ODD Exclusion?
    *   **A:** Explicitly stating "The system does not work here". (e.g., "Not for use in Tunnels").
3.  **Q:** Why is "Shadow Mode" useful?
    *   **A:** It allows testing new code on millions of miles without risking safety. The code runs silently and we check if it *would have* crashed.

### Challenge Task
**Task:** ODD Validator.
1.  Write a function `check_odd(weather, location)`.
2.  If `weather == 'Snow'` or `location == 'Tunnel'`, return `False`.
3.  Else return `True`.
4.  This function runs on the car to enable/disable the feature.

---

## 📚 Further Reading & References
-   [Waymo Safety Report](https://waymo.com/safety/)
-   [Koopman: Edge Cases](https://users.ece.cmu.edu/~koopman/pubs/koopman18_edge_cases.pdf)

---

**Day 102 Complete** | Phase 4: ADAS & Robotics Systems | Week 15: Safety Standards (ISO 26262 & SOTIF)
