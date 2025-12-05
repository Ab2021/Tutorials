# Day 101: Docker for Robotics
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 15: Production-Grade ROS 2

---

> **📝 Content Creator Instructions:**
> "It works on my machine" is not an excuse.
> - **Focus:** Containerization fundamentals, Multi-stage builds (reducing image size), GPU Passthrough (NVIDIA Container Toolkit), and Networking (Host vs Bridge).
> - **Code:** A Production `Dockerfile` that builds a ROS 2 workspace and sets up the entrypoint, plus a `docker-compose.yml` for multi-container orchestration.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Write** a Multi-Stage Dockerfile to separate Build dependencies (GCC, CMake) from Runtime artifacts.
2.  **Pass** Hardware access (GPU, USB, Network) to the container.
3.  **Orchestrate** a complex robot stack using `docker-compose`.
4.  **Debug** containerized nodes (Shell access, Visual Studio Code Remote Containers).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Linux Machine (preferred). WSL2 works with some caveats.

### Software Environment
```bash
# Install Docker Engine & NVIDIA Container Toolkit
sudo apt install docker.io
```

### Prior Knowledge
- Linux CLI.
- Dependency Management (`rosdep`).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: Why Docker in Robotics?

*   **Dependency Hell:** Robot A needs PCL 1.10. Robot B needs PCL 1.12. Same OS? Impossible without Docker.
*   **Reproducibility:** If it builds in the container, it builds everywhere.
*   **Deployment:** OTA (Over-The-Air) updates become "Pull new image and restart".

### 🔹 Part 2: The Layers

*   **Base Image:** `ros:humble-ros-base` (Official OSRF image).
*   **Overlay:** Your compiled workspace (`install/`).
*   **Entrypoint:** A script that `source /opt/ros/humble/setup.bash` automatically.

### 🔹 Part 3: Hardware Access

Containers are isolated. Robots need hardware.
*   **GPU:** `--gpus all` (Needs `nvidia-container-toolkit`).
*   **USB:** `--device /dev/ttyUSB0`.
*   **GUI:** Share X11 socket `/tmp/.X11-unix` to run Rviz from Docker.
*   **Network:** `--net host` (Simplest for ROS 2 Discovery).

---

## 💻 Implementation: Production Dockerfile

We will build a Docker image for our `day99_lifecycle` package.

### 🛠️ Project Structure
```text
day101_docker/
├── Dockerfile
├── docker-compose.yml
└── entrypoint.sh
```

### 👨‍💻 Multi-Stage Dockerfile

```dockerfile
# STAGE 1: BUILDER
FROM osrf/ros:humble-desktop AS builder

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-colcon-common-extensions \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /ros2_ws
COPY ./src ./src

# Install dependencies
RUN apt-get update && rosdep update && \
    rosdep install --from-paths src --ignore-src -y

# Build (Release mode)
RUN . /opt/ros/humble/setup.sh && \
    colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release

# STAGE 2: RUNTIME
FROM osrf/ros:humble-ros-base

# Install runtime libs (e.g. OpenCV runtime, not dev headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopencv-core-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy artifacts from builder
COPY --from=builder /ros2_ws/install /ros2_ws/install

# Setup Entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "day99_lifecycle", "managed_system.launch.py"]
```

### 👨‍💻 Entrypoint Script (`entrypoint.sh`)

Crucial for sourcing.

```bash
#!/bin/bash
set -e

# Source ROS 2 Base
source /opt/ros/humble/setup.bash

# Source Our Overlay
source /ros2_ws/install/setup.bash

# Exec the CMD passed to docker run
exec "$@"
```

### 👨‍💻 Orchestration (`docker-compose.yml`)

Start the Robot Node and a separate Monitor Node.

```yaml
version: '3.8'

services:
  robot_core:
    build: .
    image: my_robot_image:latest
    network_mode: host
    privileged: true
    devices:
      - /dev/video0:/dev/video0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  monitor:
    image: osrf/ros:humble-desktop
    network_mode: host
    command: ros2 topic echo /image_raw
    depends_on:
      - robot_core
```

---

## 🔬 Lab Exercise: "The Isolation Test"

### 1. Lab Objectives
- **Build:** `docker build -t day101 .`
- **Run:** `docker run --rm -it --net host day101`
- **Check:**
    *   Does it see `rostopic list`? (Yes, if `--net host`).
    *   Does it see `/dev/video0`? (Only if passed).
    *   Is the image size smaller than the builder? (Check `docker images`).
- **Visualize:** Run Rviz OUTSIDE docker. Can it see the topics from INSIDE docker? (Yes, with host networking).

---

## 🚀 Project: "DevContainer for VSCode"

**Goal:** Develop *inside* the container.
1.  **Create:** `.devcontainer/devcontainer.json`.
2.  **Config:**
    ```json
    {
        "name": "ROS 2 Dev",
        "dockerFile": "../Dockerfile",
        "runArgs": ["--net=host", "--gpus=all"],
        "extensions": ["ms-iot.vscode-ros"]
    }
    ```
3.  **Action:** "Reopen in Container".
4.  **Result:** VSCode IntelliSense works, compilation works, debugging works. No pollution of host OS.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. "Display not found" (GUI Apps)
*   **Cause:** Docker doesn't have access to X11.
*   **Fix:** `xhost +local:root` on host. Pass `-e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix`.

#### 2. "Shared Memory Error" (FastDDS)
*   **Cause:** Docker default shm size is 64MB. Lidar clouds need more.
*   **Fix:** `--shm-size=512m` (or 2gb).

---

## ⚡ Optimization: Rocker

Tool from OSRF to simplify Docker GPU/GUI args.
*   `rocker --nvidia --x11 osrf/ros:humble-desktop`
*   Automatically mounts rights volumes and sets env vars.

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why `--net host`?
    *   **A:** ROS 2 DDS uses random UDP ports for discovery. Bridged networking requires mapped ranges which is painful. Host networking shares the IP stack, making discovery instant.
2.  **Q:** What is a Multi-stage build?
    *   **A:** Using intermediate images to compile code, then discarding the compiler/headers in the final image to save space (e.g., 2GB $\to$ 200MB).
3.  **Q:** Can I run Docker on Jetson?
    *   **A:** Yes! But you must use `FROM nvcr.io/nvidia/l4t-ros:humble` base images (ARM64 specific).

### Challenge Task
> **Task:** Persist Data.
> 1. Run a generic mapping node.
> 2. Save map.
> 3. Kill container.
> 4. Map is gone!
> 5. **Fix:** Use Volumes. `-v $(pwd)/maps:/root/maps`.

---

## 📚 Further Reading
- **Docker Docs:** "Multi-stage builds".
- **OSRF Docker Images:** Official GitHub repo.

---

**Day 101 Complete**
