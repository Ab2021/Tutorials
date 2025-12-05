# Day 104: Diagnostics & System Monitoring
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 15: Production-Grade ROS 2

---

> **📝 Content Creator Instructions:**
> Is the robot healthy? Don't guess. Check.
> - **Focus:** The `diagnostic_updater` library, standard `DiagnosticArray` messages, and the `diagnostic_aggregator`.
> - **Code:** A Battery Monitor Node that publishes diagnostics (OK/WARN/ERROR) based on voltage levels, and a launch file with an aggregator configuration.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** a `DiagnosticTask` to monitor hardware (Temperature, Voltage, Frequency).
2.  **Publish** `DiagnosticArray` messages at a throttled rate (1Hz).
3.  **Configure** a `diagnostic_aggregator` to group errors (e.g., `/hardware/sensors`, `/hardware/powers`).
4.  **Visualize** the dashboard using `rqt_runtime_monitor` or `rqt_robot_monitor`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- None.

### Software Environment
```bash
sudo apt install ros-humble-diagnostic-updater ros-humble-diagnostic-aggregator ros-humble-rqt-robot-monitor
```

### Prior Knowledge
- ROS 2 Parameters.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Standard format

Robots have thousands of checks. Custom messages (`/battery`, `/wifi`, `/temp`) are hard to visualize.
**Solution:** `diagnostic_msgs/DiagnosticArray`.
*   **Key/Value Pairs:** `{"Voltage": "12.4", "Temp": "45C"}`.
*   **Level:** `OK`, `WARN`, `ERROR`, `STALE`.

### 🔹 Part 2: The Updater

`diagnostic_updater::Updater` maps the update loop to the publishing loop.
*   You don't publish diagnostics in your `while(true)` loop.
*   You set the hardware ID and add tasks.
*   The Updater manages the 1Hz publishing rate automatically.

### 🔹 Part 3: The Aggregator

A critical node that subscribes to `/diagnostics` (raw) and republishes to `/diagnostics_agg` (organized).
*   Uses a `.yaml` analyzer file.
*   Groups `l_wheel_motor`, `r_wheel_motor` -> `Motors`.
*   If *any* child is ERROR, the Parent becomes ERROR.

---

## 💻 Implementation: Battery Monitor

We will create a node that simulates a battery.

### 🛠️ Project Structure
```text
day104_diagnostics/
├── src/
│   ├── battery_monitor.cpp
├── config/
│   └── aggregator.yaml
└── launch/
    └── monitor.launch.py
```

### 👨‍💻 Monitor Node (`src/battery_monitor.cpp`)

```cpp
#include <rclcpp/rclcpp.hpp>
#include <diagnostic_updater/diagnostic_updater.hpp>

class BatteryMonitor : public rclcpp::Node
{
public:
  BatteryMonitor() : Node("battery_monitor")
  {
    // Updater
    updater_ = std::make_shared<diagnostic_updater::Updater>(this);
    updater_->setHardwareID("Battery-Pack-001");
    
    // Add check function
    updater_->add("Power Status", this, &BatteryMonitor::check_battery);
    
    // Simulated voltage
    voltage_ = 12.0;

    // Timer to drain battery
    timer_ = this->create_wall_timer(std::chrono::seconds(1), [this](){
        voltage_ -= 0.1;
        if(voltage_ < 10.0) voltage_ = 12.6; // Reset
        
        // Force update (usually handled by internal timer of Updater, 
        // but we want to ensure data freshness check)
        // Note: Updater spins its own timer, we just update member vars.
    });
  }

  void check_battery(diagnostic_updater::DiagnosticStatusWrapper & stat)
  {
    stat.add("Voltage", voltage_);
    
    if (voltage_ > 12.0) {
        stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Fully Charged");
    } else if (voltage_ > 11.0) {
        stat.summary(diagnostic_msgs::msg::DiagnosticStatus::WARN, "Discharging");
    } else {
        stat.summary(diagnostic_msgs::msg::DiagnosticStatus::ERROR, "Critical Low!");
    }
  }

private:
  std::shared_ptr<diagnostic_updater::Updater> updater_;
  rclcpp::TimerBase::SharedPtr timer_;
  double voltage_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<BatteryMonitor>());
  rclcpp::shutdown();
  return 0;
}
```

### 👨‍💻 Aggregator Config (`config/aggregator.yaml`)

```yaml
diagnostic_aggregator:
  ros__parameters:
    analyzers:
      sensors:
        type: diagnostic_aggregator/GenericAnalyzer
        path: Sensors
        contains: ['lidar', 'camera']
      power:
        type: diagnostic_aggregator/GenericAnalyzer
        path: Power System
        contains: ['Battery', 'BMS']
```

---

## 🔬 Lab Exercise: "The Dashboard"

### 1. Lab Objectives
- **Launch:** `monitor.launch.py` (Starts node + aggregator).
- **Run:** `rqt_robot_monitor` (Not just `rqt`).
- **Observe:**
    *   Hierarchy tree on the left.
    *   "Power System" entry.
    *   Green -> Yellow -> Red as voltage drops.
- **Fail Check:** Kill the battery node.
- **Result:** Status becomes `STALE` (Purple) in `rqt`. The aggregator noticed the silence.

---

## 🚀 Project: "Watchdog"

**Goal:** Automatic Safety Stop.
1.  **Node:** `safety_watchdog`.
2.  **Sub:** `/diagnostics_agg`.
3.  **Logic:**
    *   Check if `Power System` is OK.
    *   Check if `Sensors` is OK.
    *   If NOT, publish `cancel` to Navigation Stack or `zero_velocity` to cmd_vel.
4.  **Why:** Human operators look at dashboards. Robot software needs to look at *aggregated* diagnostics to react.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Missing Hardware ID"
*   **Symptom:** Diagnostics warns about "None" ID.
*   **Fix:** `updater_->setHardwareID("...")`. Critical for logs to know *which* battery failed.

#### 2. "Aggregator not grouping"
*   **Cause:** Mismatch in `contains` string vs Task Name.
*   **Fix:** The string in YAML must match the string passed to `updater_->add("Name", ...)` or the Node name. Use `ros2 topic echo /diagnostics` to see exact names.

---

## ⚡ Optimization: Frequency Check

Is the Lidar running at 10Hz?
*   `diagnostic_updater::FrequencyStatus`.
*   Pass it every message using `.tick()`.
*   It automatically calculates Avg/Min/Max frequency and errors out if it drops below tolerance (e.g., < 9Hz).
*   No manual math needed!

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What sends the `STALE` status?
    *   **A:** The Analyzer (Aggregator) or the subscriber if it hasn't heard from the node in X seconds. The Node itself typically sends OK/WARN/ERROR. It can't send sth if it's dead.
2.  **Q:** Can I inspect diagnostics via CLI?
    *   **A:** Yes, `ros2 topic echo /diagnostics`. But it's verbose. `rqt_robot_monitor` is better.
3.  **Q:** Relationship to Lifecycle?
    *   **A:** A Lifecycle node should probably report `WARN (Inactive)` content when in Inactive state, and `OK` when Active.

### Challenge Task
> **Task:** CPU Monitor.
> 1. Use `psutil` in Python.
> 2. Create a node that reports CPU Load % and RAM Usage.
> 3. Add Frequency Check for `/scan` topic in the same node (Topic Monitor).

---

## 📚 Further Reading
- **REP 107:** Standard for Diagnostic System.
- **RQT Plugin:** Robot Monitor Docs.

---

**Day 104 Complete**
