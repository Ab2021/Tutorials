# Day 203: Week 29 Review & Milestone
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 29: Capstone Project Part 1

---

> **📝 Content Creator Instructions:**
> We have a robot. It moves. It picks. But does it work *well*?
> - **Focus:** Analyzing the performance of the Agri-Bot V1. Metrics, Logging, and preparing for the final Optimization phase.
> - **Code:** `analyze_bag.py`. Processing `ros2 bag` data to extract picking cycle times and failure rates.
> - **Review:** Architecture, Vision, Nav, Control.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Record** a full mission using `ros2 bag`.
2.  **Extract** KPIs (Key Performance Indicators) from the bag data.
3.  **Identify** bottlenecks (e.g., "Scanning takes 50% of the time").
4.  **Plan** the optimizations for Week 30 (Speed, Robustness, UI).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
pip install rosbags pandas matplotlib
```

### Prior Knowledge
- All of Week 29.

---

## 📖 Theoretical Analysis

### 🔹 The V1 Audit

We built a working prototype.
*   **Success Rate:** 3/5 fruits picked. (60%).
*   **Avg Time per Fruit:** 15 seconds.
    *   Nav: 5s.
    *   Scan: 2s.
    *   Pick: 8s.
*   **Failure Modes:**
    1.  Missed Grasp (Calibration error).
    2.  Fruit blocked by leaf (Perception).

### 🔹 Optimization Strategy (Amdahl's Law)
Speed up the slowest part.
*   The Pick (8s) is the bottleneck.
*   **Fix:** Use asynchronous motion? (Move Base while retracting Arm?).

---

## 💻 Implementation: Performance Analysis

We assume we recorded a bag during Day 202's mission:
`ros2 bag record -a -o picking_mission`

### 🛠️ Project Structure
```text
agribot_analysis/
├── bags/
│   └── picking_mission/
├── src/
│   └── analyze_bag.py
└── output/
    └── cycle_time_plot.png
```

### 👨‍💻 Log Analyzer (`src/analyze_bag.py`)

We look for state transitions in the FSM log (published to `/mission_state`).

```python
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from std_msgs.msg import String

def read_ros2_bag(bag_path):
    # ROS 2 recording format is SQLite3 by default
    conn = sqlite3.connect(f"{bag_path}/picking_mission_0.db3")
    cursor = conn.cursor()
    
    # Get Topic ID
    cursor.execute("SELECT id FROM topics WHERE name = '/mission_status'")
    topic_id = cursor.fetchone()[0]
    
    # Get Messages
    cursor.execute(f"SELECT timestamp, data FROM messages WHERE topic_id = {topic_id}")
    rows = cursor.fetchall()
    
    data = []
    for ts, blob in rows:
        msg = deserialize_message(blob, String)
        data.append({'timestamp': ts, 'status': msg.data})
        
    return pd.DataFrame(data)

def analyze_cycle_times(df):
    # Statuses: SEARCHING, DETECTED, PICKING, DEPOSITING
    
    # We want time spent in "PICKING"
    df['dt'] = df['timestamp'].diff() / 1e9 # Nanoseconds to Seconds
    
    picking_times = df[df['status'] == 'PICKING']['dt'].sum()
    nav_times = df[df['status'] == 'SEARCHING']['dt'].sum()
    
    print(f"Total Time Picking: {picking_times:.2f}s")
    print(f"Total Time Navigating: {nav_times:.2f}s")
    
    # Cycles (Count how many times status switched to DEPOSITING)
    counts = df[df['status'] == 'DEPOSITING'].shape[0]
    print(f"Fruits Picked: {counts}")
    if counts > 0:
        print(f"Avg Time Per Fruit: {picking_times/counts:.2f}s")

    # Plot Gantt-ish chart
    # (Simplified viz code omitted for brevity)

def main():
    try:
        df = read_ros2_bag("bags/picking_mission")
        analyze_cycle_times(df)
    except Exception as e:
        print(f"Analysis failed (Did you record the bag?): {e}")
        # Mock Output
        print("Mock Data Result:")
        print("Fruits Picked: 5")
        print("Avg Time: 12.4s")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "The Benchmark"

### 1. Lab Objectives
- **Run:** The mission 5 times.
- **Vary:** Change the fruit layout (clustered vs sparse).
- **Measure:** Does clustered fruit improve cycle time? (Yes, less Driving).
- **Hypothesis:** "If we pick 2 fruits with one arm cycle (dual gripper), we save 30% time."

---

## 🚀 Project Part 1 Milestone Checklist

*   [x] **Simulation:** Farm World + Robot spawned.
*   [x] **Drivers:** Nav2 and MoveIt configured.
*   [x] **Perception:** Fruits Detected and 3D localized.
*   [x] **Control:** FSM picks fruit/places in basket.
*   [x] **Integration:** One launch file runs it all.

If you checked all these, you have passed Week 29.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "DB3 Locked"
*   **Cause:** Trying to read the bag while recording it.
*   **Fix:** Stop the recording (Ctrl+C) before running analysis.

#### 2. "Corrupt Bag"
*   **Cause:** Hard crash during write.
*   **Fix:** `ros2 bag reindex`.

---

## ⚡ Optimization: Parallelism

Plan for Week 30:
*   Can we Detect *while* Driving? (Currently: Stop-Scan-Go).
*   Can we Plan *while* Executing? (Replanning).

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What is "Cycle Time"?
    *   **A:** The total time from "Start of Task A" to "Completion of Task A". (e.g., Pick-to-Pick time).
2.  **Q:** Why log to a Bag file?
    *   **A:** Reproducibility. If the robot fails at 12:00, you can replay the sensor data at 12:05 to see *why* the code crashed, without needing the physical robot setup.

### Challenge Task
> **Task:** "Heatmap".
> 1. Plot the robot's X,Y path from the bag.
> 2. Overlay the detected fruit locations.
> 3. Does the robot travel in a straight line? Or does it "wiggle" (poor control tuning)?

---

## 📚 Further Reading
- **Foxglove Studio:** Advanced ROS 2 Data Visualization (Web based).
- **PlotJuggler:** Time series plotting tool.

---

**Day 203 Complete. End of Week 29.**
**Next: Phase 5 Finale - Capstone Optimization & Certification.**
