# Day 107: Radar for Robotics
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 16: Advanced Sensors

---

> **📝 Content Creator Instructions:**
> Lidar stops at rain. Radar sees through it.
> - **Focus:** Frequency Modulated Continuous Wave (FMCW) principles, Doppler Velocity, and the Radar Data Cube. Reading data from a TI mmWave or Continental radar.
> - **Code:** A Radar-Camera Fusion node that projects Radar targets (Range/Azimuth/Velocity) onto an Image plane to identify *which* car is moving at what speed.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the FMCW chirp mechanism and how it extracts both Range and Velocity.
2.  **Contrast** Radar (Sparse, Doppler, Weather-proof) vs Lidar (Dense, Geometric, Weather-sensitive).
3.  **Parses** ROS 2 `radar_msgs` (or custom TI mmWave messages).
4.  **Filter** static clutter using the Doppler velocity channel.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- TI mmWave EVM (IWR6843) or Continental ARS408 (Industry Standard).
- Optional: Virtual Radar simulation in Gazebo.

### Software Environment
```bash
sudo apt install ros-humble-radar-msgs
```

### Prior Knowledge
- Doppler Effect.
- Fourier Transform (FFT) basics.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: FMCW Basics

Lidar fires a pulse (ToF). Radar emits a **Chirp**.
*   **Frequency increases linearly** with time.
*   **Echo:** Returns with a delay.
*   **Mixer:** The difference between TX freq and RX freq (Beat Frequency) is proportional to **Range**.
*   **Phase Shift:** Across multiple chirps determines **Velocity** (Doppler).

### 🔹 Part 2: The Data

Most robotics radars don't output raw ADC data (too fast, GB/s). They output **Point Clouds** or **Object Lists**.
*   **Point:** $(X, Y, Z, V_{doppler}, SNR)$.
*   **Object (Clustered):** $(X, Y, V_x, V_y, Width, Class)$.
*   **Key Advantage:** Direct measurement of radial velocity. Lidar has to differentiate position over time (noisy).

### 🔹 Part 3: The "Ghost" Problem

Radar reflections are weird.
*   **Multipath:** Beam bounces off a guardrail, hits car, hits guardrail, returns. Radar thinks car is behind the wall.
*   **Clutter:** Ground reflections.
*   **RCS (Radar Cross Section):** A person has low RCS. A soda can has high RCS. Size $\neq$ Return Strength.

---

## 💻 Implementation: Radar Filtering

We will simulate a radar stream and filter it.

### 🛠️ Project Structure
```text
day107_radar/
├── src/
│   ├── radar_processor.cpp
│   └── visualize_doppler.py
└── config/
    └── radar_params.yaml
```

### 👨‍💻 Radar Processor (`src/radar_processor.cpp`)

Filters out static objects (Velocity ~ 0) to find moving targets.

```cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Custom Point Type with Velocity
struct PointRadar
{
    PCL_ADD_POINT4D;
    float velocity;
    float intensity;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT(PointRadar,
    (float, x, x)(float, y, y)(float, z, z)
    (float, velocity, velocity)(float, intensity, intensity)
)

class RadarProcessor : public rclcpp::Node
{
public:
  RadarProcessor() : Node("radar_processor")
  {
    sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
      "ti_mmwave/radar_scan", 10, std::bind(&RadarProcessor::cb, this, std::placeholders::_1));
      
    pub_markers_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("radar_blobs", 10);
  }

  void cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    pcl::PointCloud<PointRadar>::Ptr cloud(new pcl::PointCloud<PointRadar>);
    pcl::fromROSMsg(*msg, *cloud);
    
    visualization_msgs::msg::MarkerArray markers;
    int id = 0;

    for(const auto& pt : cloud->points) {
        // FILTER: Doppler Threshold (0.1 m/s)
        if (std::abs(pt.velocity) > 0.5) {
            // It's moving!
            visualization_msgs::msg::Marker m;
            m.header = msg->header;
            m.ns = "moving_targets";
            m.id = id++;
            m.type = visualization_msgs::msg::Marker::SPHERE;
            m.action = visualization_msgs::msg::Marker::ADD;
            m.pose.position.x = pt.x;
            m.pose.position.y = pt.y;
            m.pose.position.z = pt.z;
            m.scale.x = 0.5; m.scale.y = 0.5; m.scale.z = 0.5;
            
            // Color by Velocity (Red = Fast Approaching)
            if (pt.velocity < 0) { // Approaching
                 m.color.r = 1.0; m.color.g = 0.0; m.color.b = 0.0;
            } else { // Departing
                 m.color.r = 0.0; m.color.g = 0.0; m.color.b = 1.0;
            }
            m.color.a = 0.8;
            markers.markers.push_back(m);
        }
    }
    pub_markers_->publish(markers);
  }

private:
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_markers_;
};
```

---

## 🔬 Lab Exercise: "Fog of War"

### 1. Lab Objectives
- **Scenario:** Gazebo world with thick fog (`<fog>` tag in SDF).
- **Comparison:**
    *   **Lidar:** Verify points disappear or get noisy in fog.
    *   **Camera:** White screen.
    *   **Radar:** Perfectly clear signal.
- **Task:** Drive the robot using *only* Radar Markers in Rviz through the fog.

---

## 🚀 Project: "Radar-Camera Fusion"

**Goal:** Automatic Emergency Braking (AEB) reliability.
1.  **Issue:** Camera sees a poster of a car. Lidar sees a flat wall. Radar sees... nothing (paper has low RCS).
2.  **Issue:** Fog exists. Camera blind. Lidar blind. Radar sees a Truck.
3.  **Sensor Fusion:**
    *   Project Radar Point to Image.
    *   Check Distance.
    *   Logic: `IF RadarRange < 5m AND RadarVelocity < -2m/s THEN BRAKE`.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Ghost Targets"
*   **Cause:** Multipath reflections.
*   **Fix:** Use Tracker Logic (Kalman Filter). Ghosts often move in unrealistic ways (appearing/disappearing instantly).

#### 2. "Low Angular Resolution"
*   **Context:** Unlike Lidar (0.1 deg), Radar is coarse (5-10 deg).
*   **Result:** Two cars side-by-side might look like one big blob.
*   **Fix:** Use "Imaging Radar" (4D Radar) or fuse with Camera for lateral separation.

---

## ⚡ Optimization: CFAR (Constant False Alarm Rate)

How does the radar chip decide "This is a point"?
*   It compares the signal peak to the noise floor.
*   **Adaptive Threshold:** CFAR adjusts the threshold dynamically based on background noise.
*   **Tuning:** If you see too much clutter, increase CFAR threshold in the radar config YAML.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What does "4D Radar" mean?
    *   **A:** It measures X, Y, Z, and Velocity. (Traditional automotive radar was 2D: X, Y, V). 4D adds elevation (Z).
2.  **Q:** Why can radar measure velocity instantly?
    *   **A:** Doppler shift. The frequency of the return wave is shifted by $\Delta f = \frac{2v}{\lambda}$.
3.  **Q:** Is Radar safe for humans?
    *   **A:** Yes, typically very low power (milliwatts) compared to communication towers.

### Challenge Task
> **Task:** Speeding Ticket Bot.
> 1. Set up Radar on side of "road".
> 2. Monitor velocity of passing Turtlebots.
> 3. If $|v| > 1.0 m/s$, take a picture with Camera.

---

## 📚 Further Reading
- **TI mmWave SDK:** Deep dive into chirp configuration.
- **Ainstein Radar:** Commercial drone radars using CAN bus.

---

**Day 107 Complete**
