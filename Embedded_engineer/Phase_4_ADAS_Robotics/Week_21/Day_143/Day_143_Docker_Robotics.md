# Day 143: Docker & Containerization for Robotics
## Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone

---

> **📝 Day 143 Focus:**
> "It works on my machine" is the enemy of robotics. **Docker** ensures it works on *every* machine. We will package our entire ROS 2 stack into a container, enabling easy deployment to the car's computer (Jetson/IPC).

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Write** a `Dockerfile` for a ROS 2 application.
2.  **Build** and **Run** a ROS 2 container.
3.  **Enable** GPU acceleration (NVIDIA Container Toolkit).
4.  **Mount** volumes for development (Hot Reload).
5.  **Use** `rocker` or `ade` for simplified container management.

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Linux:** Command line basics.
-   **ROS 2:** Workspace structure.

### Hardware Requirements
-   **NVIDIA GPU:** For GPU passthrough (Optional but recommended).

### Software Stack
-   **Docker Engine:** Installed.
-   **NVIDIA Container Toolkit:** Installed (if using GPU).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Docker?

-   **Dependency Hell:** ROS 2 Humble needs Ubuntu 22.04. What if your laptop runs 20.04? Docker fixes this.
-   **Reproducibility:** The exact versions of `numpy`, `pytorch`, and `ros-core` are locked in the image.
-   **Isolation:** Crashing the container doesn't crash the host OS.

### 🔹 Part 2: The Dockerfile

-   `FROM`: Base image (e.g., `ros:humble`).
-   `RUN`: Execute commands (apt-get, pip).
-   `COPY`: Copy files from host to container.
-   `ENTRYPOINT`: Script to run when container starts (usually `source /opt/ros/humble/setup.bash`).

### 🔹 Part 3: GPU Passthrough

Standard Docker doesn't see the GPU.
-   **NVIDIA Container Toolkit:** Exposes the GPU driver to the container.
-   Flag: `--gpus all`.
-   Essential for Deep Learning (YOLO, PointPillars) and Simulation (Gazebo).

---

## 💻 Implementation: Containerizing the Stack

**Scenario:**
-   We want to run our `week21_day142` launch file inside Docker.

### 🛠️ Setup
Create `week21_day143` folder (not a ROS package, just a folder for Docker config).

```bash
mkdir -p ~/ros2_ws/src/week21_day143
cd ~/ros2_ws/src/week21_day143
touch Dockerfile
touch entrypoint.sh
```

### 📄 Dockerfile

```dockerfile
# Base Image: ROS 2 Humble (Desktop Full includes Gazebo/Rviz)
FROM osrf/ros:humble-desktop-full

# Set Environment
ENV DEBIAN_FRONTEND=noninteractive
SHELL ["/bin/bash", "-c"]

# 1. Install Dependencies
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-nav2-lifecycle-manager \
    && rm -rf /var/lib/apt/lists/*

# Install Python libs
RUN pip3 install numpy opencv-python

# 2. Create Workspace
WORKDIR /root/ros2_ws/src

# 3. Copy Source Code
# In dev, we usually mount this volume instead of copying.
# For production, we copy.
COPY . /root/ros2_ws/src/

# 4. Build
WORKDIR /root/ros2_ws
RUN source /opt/ros/humble/setup.bash && \
    colcon build --symlink-install

# 5. Setup Entrypoint
COPY week21_day143/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
```

### 📄 Entrypoint Script (`entrypoint.sh`)

```bash
#!/bin/bash
set -e

# Source ROS 2
source /opt/ros/humble/setup.bash

# Source Local Workspace
if [ -f "/root/ros2_ws/install/setup.bash" ]; then
    source /root/ros2_ws/install/setup.bash
fi

# Execute the command passed to docker run
exec "$@"
```

---

## 🔬 Lab Exercise: Build and Run

### Lab Objectives
1.  **Build the Image:**
    Go to the root of your workspace (`~/ros2_ws`).
    ```bash
    docker build -t my_adas_bot -f src/week21_day143/Dockerfile .
    ```
    *(Note: This copies the whole workspace context. It might be slow if you have huge build artifacts. Use `.dockerignore` to exclude `build/`, `install/`, `log/`).*

2.  **Run the Container:**
    ```bash
    docker run -it --rm \
        --net=host \
        --ipc=host \
        --pid=host \
        --gpus all \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        my_adas_bot
    ```
    -   `--net=host`: Share network (for ROS 2 discovery).
    -   `--gpus all`: Enable GPU.
    -   `-e DISPLAY`: Enable GUI (Rviz).

3.  **Test inside Container:**
    ```bash
    ros2 launch week21_day142 system.launch.py
    ```

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. GUI Not Showing (Rviz)
**Symptom:** `qt.qpa.xcb: could not connect to display`.
**Cause:** X11 permissions.
**Solution:** On host, run `xhost +local:root` (Allow root to access display). Or use `rocker`.

#### 2. ROS 2 Communication Fails
**Symptom:** Nodes in container can't see nodes on host.
**Cause:** Network isolation.
**Solution:** Use `--net=host`. Also ensure `ROS_DOMAIN_ID` matches.

#### 3. Shared Memory (DDS)
**Symptom:** High latency or crashes with large data (Lidar).
**Cause:** Docker default shared memory is small (64MB).
**Solution:** Use `--ipc=host` or `--shm-size=512m`.

---

## ⚡ Optimization & Best Practices

### 1. Multi-Stage Builds
Reduce image size.
-   **Stage 1 (Builder):** Install compilers, build source code.
-   **Stage 2 (Runtime):** Copy only the `install/` folder from Stage 1.
-   Result: 500MB image instead of 5GB.

### 2. Rocker
A tool to simplify Docker arguments for ROS.
```bash
rocker --nvidia --x11 --network=host my_adas_bot
```
Automatically handles GPU, Display, and User ID mapping.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the purpose of `entrypoint.sh`?
    *   **A:** To source the ROS setup scripts (`setup.bash`) automatically every time a new shell is opened in the container.
2.  **Q:** Why do we use `--net=host`?
    *   **A:** ROS 2 uses DDS (Multicast/UDP). Docker's default bridge network blocks multicast. Host networking bypasses this.
3.  **Q:** How do I persist data (e.g., Maps) generated inside the container?
    *   **A:** Use Volumes (`-v /host/path:/container/path`).

### Challenge Task
**Task:** Development Container.
1.  Run the container with your source code mounted as a volume:
    `-v ~/ros2_ws/src:/root/ros2_ws/src`
2.  Edit code on your host (VS Code).
3.  Run `colcon build` inside the container.
4.  See changes immediately. This is the standard "Dev Container" workflow.

---

## 📚 Further Reading & References
-   [Official ROS 2 Docker Images](https://hub.docker.com/_/ros)
-   [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

---

**Day 143 Complete** | Phase 4: ADAS & Robotics Systems | Week 21: System Integration & Capstone
