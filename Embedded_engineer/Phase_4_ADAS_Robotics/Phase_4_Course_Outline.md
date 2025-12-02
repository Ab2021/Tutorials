# Phase 4: ADAS & Robotics Systems
## Course Outline - 24 Weeks (168 Days)

---

## Course Overview

**Duration:** 25 weeks / 175 days  
**Focus:** Advanced Driver Assistance Systems (ADAS) and Autonomous Robotics  
**Prerequisites:** Phase 3 completion (Camera Systems & ISP Development)

**Learning Path:**
- Low-level sensor processing → High-level decision making
- ROS 2 ecosystem → Production autonomous systems
- Simulation → Real-world deployment
- Individual algorithms → Integrated autonomous stack

---

## Week 1: ROS 2 Fundamentals (Days 1-7)

### Day 1: ROS 2 Architecture & DDS
- ROS 2 vs ROS 1 comparison
- DDS (Data Distribution Service) middleware
- QoS (Quality of Service) policies
- Discovery mechanisms
- **Lab:** Setup ROS 2 Humble workspace

### Day 2: Nodes, Topics, and Publishers/Subscribers
- Node lifecycle management
- Topic-based communication
- Publisher/Subscriber pattern
- Message serialization
- **Lab:** Create multi-node talker/listener system

### Day 3: Services and Actions
- Request/Response pattern (Services)
- Goal-based pattern (Actions)
- Feedback and result handling
- Concurrent service calls
- **Lab:** Implement a service-based calculator

### Day 4: Custom Messages and Interfaces
- msg, srv, and action definitions
- Package dependencies
- IDL (Interface Definition Language)
- Cross-language compatibility
- **Lab:** Design sensor data messages

### Day 5: Launch Files and Parameters
- Python launch files
- XML/YAML parameters
- Dynamic reconfiguration
- Composition and lifecycle nodes
- **Lab:** Multi-robot launch system

### Day 6: TF2 (Transform Library)
- Coordinate frame trees
- Static vs dynamic transforms
- Time travel and interpolation
- Map → Odom → Base_link → Sensor frames
- **Lab:** Mobile robot TF tree

### Day 7: Week 1 Review & Project (Multi-Robot Communication)
- Project: Fleet coordination system
- Multiple robots with shared map
- Centralized vs decentralized coordination
- Performance benchmarking

---

## Week 2: Sensor Fusion Basics (Days 8-14)

### Day 8: Kalman Filter Theory
- State estimation problem
- Prediction and update steps
- Gaussian assumption
- Filter gain and covariance
- **Lab:** 1D tracking example

### Day 9: Extended Kalman Filter (EKF)
- Nonlinear system models
- Jacobian linearization
- EKF for robot localization
- Error analysis
- **Lab:** 2D robot pose estimation

### Day 10: Unscented Kalman Filter (UKF)
- Sigma point selection
- Unscented transform
- UKF vs EKF comparison
- Higher-order accuracy
- **Lab:** Vehicle state estimation

### Day 11: Particle Filters
- Monte Carlo localization
- Importance sampling
- Resampling strategies
- Particle depletion
- **Lab:** Global localization

### Day 12: Sensor Models (Camera, LiDAR, Radar, IMU)
- Measurement noise characteristics
- Sensor fusion architectures
- Complementary filtering
- Outlier rejection
- **Lab:** Multi-sensor characterization

### Day 13: Time Synchronization & Timestamping
- Hardware time sync (PTP)
- Software sync (NTP)
- Message filters (ApproximateTime)
- Buffering and latency
- **Lab:** Sensor sync validation

### Day 14: Week 2 Review & Project (IMU + GPS Fusion)
- Project: Outdoor navigation system
- EKF-based sensor fusion
- Dead reckoning
- Performance in GPS-denied areas

---

## Week 3: LiDAR Processing & PCL (Days 15-21)

### Day 15: LiDAR Fundamentals (Velodyne, OS1, Livox)
- Rotating vs solid-state LiDAR
- Point cloud data structure
- Range, intensity, and ring
- ROS2 drivers (velodyne_driver, ouster_driver)
- **Lab:** LiDAR data acquisition

### Day 16: Point Cloud Library (PCL) Basics
- PCL data types (PointXYZ, PointXYZI, PointXYZRGB)
- Coordinate transformations
- I/O operations (PCD, PLY)
- Visualization (pcl_viewer)
- **Lab:** Point cloud file processing

### Day 17: Filtering (Voxel, Statistical, Radius Outlier)
- VoxelGrid downsampling
- PassThrough filtering
- Statistical outlier removal
- Conditional removal
- **Lab:** Real-time filtering pipeline

### Day 18: Segmentation (RANSAC, Euclidean Clustering)
- Plane segmentation (RANSAC)
- Euclidean cluster extraction
- Region growing
- Min-cut segmentation
- **Lab:** Road and obstacle separation

### Day 19: Registration (ICP, NDT)
- Iterative Closest Point (ICP)
- Normal Distributions Transform (NDT)
- Point-to-plane ICP
- Global vs local registration
- **Lab:** Point cloud alignment

### Day 20: Ground Plane Removal & Obstacle Detection
- Ground plane estimation
- Height-based filtering
- Connected component analysis
- 3D bounding box fitting
- **Lab:** Pedestrian detection from LiDAR

### Day 21: Week 3 Review & Project (LiDAR Object Detection)
- Project: Parking lot object detection
- Real-time processing pipeline
- Object classification (car, pedestrian, cyclist)
- Distance and velocity estimation

---

## Week 4: SLAM Fundamentals (Days 22-28)

### Day 22: SLAM Problem Formulation
- Simultaneous Localization and Mapping
- Front-end vs back-end
- Online vs offline SLAM
- Full SLAM vs filtering approaches
- **Lab:** SLAM simulation in RViz

### Day 23: Graph-based SLAM
- Pose graph representation
- Factor graphs
- Optimization frameworks (g2o, Ceres)
- Sparse matrix solvers
- **Lab:** 2D pose graph optimization

### Day 24: Loop Closure Detection
- Place recognition problem
- Bag-of-Words (BoW)
- DBoW2 and DBoW3
- False positive handling
- **Lab:** Loop closure in indoor environment

### Day 25: Pose Graph Optimization (g2o)
- Graph vertices and edges
- Robust kernels
- Incremental optimization
- Constraint types (odometry, loop closure)
- **Lab:** g2o custom solver

### Day 26: Occupancy Grid Mapping
- Inverse sensor model
- Log-odds representation
- Bresenham ray tracing
- Map update rules
- **Lab:** 2D occupancy grid builder

### Day 27: Gmapping and Cartographer
- Gmapping (Rao-Blackwellized PF)
- Google Cartographer (submap-based)
- Parameter tuning
- Large-scale mapping
- **Lab:** Map a building

### Day 28: Week 4 Review & Project (2D SLAM)
- Project: Warehouse mapping robot
- Real-time SLAM with mobile robot
- Map saving and loading
- Navigation using generated map

---

## Week 5: Visual SLAM & ORB-SLAM (Days 29-35)

### Day 29: Monocular SLAM Theory
- Structure from Motion (SfM)
- Scale ambiguity
- Depth uncertainty
- Initialization problem
- **Lab:** Monocular camera motion estimation

### Day 30: Feature Extraction (ORB, SIFT, SURF)
- ORB (Oriented FAST and Rotated BRIEF)
- Scale invariance
- Rotation invariance
- Descriptor matching
- **Lab:** Feature detection comparison

### Day 31: Feature Matching and Tracking
- Brute-force vs FLANN matching
- Optical flow (Lucas-Kanade)
- KLT tracker
- Tracking quality metrics
- **Lab:** Real-time feature tracking

### Day 32: Bundle Adjustment
- Reprojection error minimization
- Sparse bundle adjustment
- Local vs global BA
- Marginalization
- **Lab:** Small-scale BA problem

### Day 33: ORB-SLAM3 Architecture
- Tracking, local mapping, loop closing threads
- Atlas (multiple maps)
- IMU integration
- Place recognition
- **Lab:** ORB-SLAM3 deployment

### Day 34: Stereo and RGB-D SLAM
- Disparity-based depth
- Depth camera characteristics
- Dense vs sparse SLAM
- RTAB-Map
- **Lab:** Indoor RGB-D SLAM

### Day 35: Week 5 Review & Project (Indoor Navigation)
- Project: Autonomous indoor robot
- Visual SLAM navigation
- Dynamic obstacle avoidance
- Multi-floor mapping

---

## Week 6: Path Planning Algorithms (Days 36-42)

### Day 36: Graph Search (Dijkstra, A*)
- Shortest path problem
- Heuristic design
- Admissibility and consistency
- Tie-breaking strategies
- **Lab:** Grid-based A* planner

### Day 37: Sampling-based Planning (RRT, RRT*)
- Rapidly-exploring Random Trees
- RRT* asymptotic optimality
- Informed RRT*
- Configuration space
- **Lab:** High-dimensional planning

### Day 38: Lattice Planners
- State lattice construction
- Motion primitives
- Kinodynamic constraints
- Search on lattice
- **Lab:** Car-like robot planner

### Day 39: Hybrid A* for Parking
- Continuous state space
- Kinematic constraints
- Reeds-Shepp curves
- Analytic expansion
- **Lab:** Parallel parking planner

### Day 40: Dynamic Window Approach (DWA)
- Velocity space sampling
- Collision checking
- Objective function design
- Local minima
- **Lab:** DWA for mobile robot

### Day 41: Timed Elastic Band (TEB)
- Trajectory optimization
- Online replanning
- Human-aware planning
- Multi-objective optimization
- **Lab:** TEB local planner

### Day 42: Week 6 Review & Project (Warehouse Navigation)
- Project: Automated forklift system
- Global path planning
- Local obstacle avoidance
- Multi-robot coordination

---

## Week 7: Motion Control & PID Tuning (Days 43-49)

### Day 43: Vehicle Kinematics (Bicycle Model)
- Front-wheel steering model
- Ackermann geometry
- Curvature and turning radius
- Kinematic vs dynamic models
- **Lab:** Bicycle model simulation

### Day 44: PID Control Theory
- Proportional, Integral, Derivative gains
- Tuning methods (Ziegler-Nichols)
- Windup prevention
- Discrete-time implementation
- **Lab:** PID controller for line following

### Day 45: Pure Pursuit Controller
- Look-ahead distance
- Path tracking error
- Curvature calculation
- Speed-dependent lookahead
- **Lab:** Pure pursuit implementation

### Day 46: Stanley Controller
- Cross-track error
- Heading error
- Front axle reference
- Gain scheduling
- **Lab:** Stanley vs Pure Pursuit comparison

### Day 47: Model Predictive Control (MPC)
- Receding horizon control
- Optimization problem formulation
- Linear vs nonlinear MPC
- Constraint handling
- **Lab:** MPC for lane keeping

### Day 48: Ackermann Steering
- Steering geometry
- Inner and outer wheel angles
- Wheelbase and track width
- ROS2 Ackermann messages
- **Lab:** Ackermann vehicle control

### Day 49: Week 7 Review & Project (Lane Keeping)
- Project: Highway lane keeping system
- MPC-based lateral control
- PID-based longitudinal control
- Feedforward + feedback control

---

## Week 8: Object Detection for Autonomous Driving (Days 50-56)

### Day 50: 3D Object Detection (SECOND, PointPillars)
- Voxelization strategies
- Sparse convolutions
- Pillar feature extraction
- 3D bounding box regression
- **Lab:** PointP illars training

### Day 51: Multi-Modal Detection (PointPainting)
- LiDAR + camera fusion
- Semantic point cloud painting
- Late fusion strategies
- Calibration requirements
- **Lab:** Multi-modal fusion pipeline

### Day 52: BEV (Bird's Eye View) Representation
- Spatial transformation
- BEVFusion architecture
- Temporal aggregation
- Occupancy prediction
- **Lab:** BEV generation from cameras

### Day 53: Temporal Fusion
- Multi-frame aggregation
- Motion compensation
- Recurrent networks
- Temporal consistency
- **Lab:** Video object detection

### Day 54: Anchor-Free Detectors (CenterPoint)
- Center-based detection
- Heatmap prediction
- Velocity estimation
- NMS-free approaches
- **Lab:** CenterPoint deployment

### Day 55: Deployment on NVIDIA Drive
- TensorRT optimization
- INT8 quantization
- DLA offloading
- Batching strategies
- **Lab:** Real-time inference pipeline

### Day 56: Week 8 Review & Project (Pedestrian Detection)
- Project: Crosswalk pedestrian detector
- Multi-modal detection (LiDAR + camera)
- Distance and velocity estimation
- Alert system integration

---

## Week 9: Multi-Object Tracking (Days 57-63)

### Day 57: Tracking Fundamentals (Data Association)
- Track initialization and termination
- State prediction
- Measurement association
- Track management
- **Lab:** Simple 2D tracker

### Day 58: Hungarian Algorithm
- Assignment problem
- Cost matrix construction
- Munkres algorithm
- Gating and validation
- **Lab:** Optimal assignment solver

### Day 59: Joint Probabilistic Data Association (JPDA)
- Multiple hypothesis tracking
- Probability calculation
- Track coalescence
- Computational complexity
- **Lab:** JPDA implementation

### Day 60: SORT and DeepSORT
- Simple Online Realtime Tracking
- IoU-based association
- Appearance embeddings (ReID)
- Kalman filter integration
- **Lab:** DeepSORT on KITTI dataset

### Day 61: AB3DMOT
- 3D multi-object tracking
- 3D IoU calculation
- Motion models for vehicles
- Occlusion handling
- **Lab:** 3D tracking from LiDAR

### Day 62: Track Management and Lifecycle
- Track confidence scoring
- Tentative vs confirmed tracks
- Track deletion strategies
- ID switching prevention
- **Lab:** Robust tracker implementation

### Day 63: Week 9 Review & Project (Vehicle Tracking)
- Project: Highway vehicle tracker
- Multi-sensor tracking
- Lane-based filtering
- Track prediction and smoothing

---

## Week 10: Behavior Planning & Decision Making (Days 64-70)

### Day 64: Finite State Machines
- State enumeration
- Transition logic
- Hierarchical FSMs
- Event-driven vs polling
- **Lab:** Traffic light FSM

### Day 65: Behavior Trees
- Sequence, selector, parallel nodes
- Decorators and conditions
- Reactive planning
- BehaviorTree.CPP
- **Lab:** Navigation behavior tree

### Day 66: Frenet Frame Planning
- Frenet coordinate system
- Lateral and longitudinal planning
- Polynomial trajectory generation
- Cost function design
- **Lab:** Highway trajectory planner

### Day 67: Velocity Planning
- Speed profile generation
- Acceleration constraints
- Jerk minimization
- Stop line handling
- **Lab:** Comfortable velocity planner

### Day 68: Collision Checking
- Swept volume analysis
- Time-to-collision (TTC)
- Conservative vs aggressive checking
- Uncertainty propagation
- **Lab:** Real-time collision checker

### Day 69: Scenarios (Merge, Overtake, Yield)
- Lane change decision
- Merge gap selection
- Yielding logic
- Right-of-way rules
- **Lab:** Scenario-based planner

### Day 70: Week 10 Review & Project (Highway Pilot)
- Project: Level 2 highway autopilot
- Lane keeping + adaptive cruise
- Lane change execution
- Driver monitoring integration

---

## Week 11: Localization (GPS/IMU Fusion) (Days 71-77)

### Day 71: GNSS Fundamentals (GPS, GLONASS, Galileo)
- Satellite positioning
- Trilateration
- Atmospheric errors
- Multipath effects
- **Lab:** GNSS data parsing (NMEA)

### Day 72: RTK (Real-Time Kinematic) GPS
- Carrier phase measurements
- Base station corrections
- Fix types (Float vs Fixed)
- Centimeter-level accuracy
- **Lab:** RTK GPS integration

### Day 73: IMU Error Models (Bias, Drift, Noise)
- Accelerometer and gyroscope errors
- Allan variance analysis
- Temperature effects
- Calibration procedures
- **Lab:** IMU characterization

### Day 74: Wheel Odometry
- Encoder-based odometry
- Slip and skid detection
- 2-wheel differential drive
- Uncertainty modeling
- **Lab:** Odometry calibration

### Day 75: Visual-Inertial Odometry (VIO)
- Tightly vs loosely coupled
- VINS-Mono/Fusion
- Initialization
- Marginalization strategies
- **Lab:** VIO on mobile platform

### Day 76: Robot Localization (AMCL)
- Adaptive Monte Carlo Localization
- Particle filter for localization
- Sensor models (laser, odometry)
- Global localization
- **Lab:** AMCL parameter tuning

### Day 77: Week 11 Review & Project (Urban Localization)
- Project: City driving localization
- Multi-sensor fusion (GPS/IMU/Wheel/Vision)
- Urban canyon handling
- Map-based correction

---

## Week 12: HD Maps & Map Matching (Days 78-84)

### Day 78: HD Map Format (OpenDRIVE, Lanelet2)
- OpenDRIVE specification
- Lanelet2 format
- Road network topology
- Coordinate systems
- **Lab:** HD map visualization

### Day 79: Map Layers (Lanes, Signs, Signals)
- Lane geometry and connectivity
- Traffic signs and signals
- Regulatory elements
- Semantic attributes
- **Lab:** Multi-layer map parsing

### Day 80: Map Matching Algorithms
- GPS to lane association
- Hidden Markov Model approach
- Particle-based matching
- Confidence scoring
- **Lab:** Real-time map matcher

### Day 81: Semantic Mapping
- LiDAR-based semantic segmentation
- Map annotation
- Dynamic vs static elements
- Map updating
- **Lab:** Semantic map builder

### Day 82: Online Map Updates
- Crowdsourced map changes
- Change detection algorithms
- Incremental map updates
- Version control
- **Lab:** Diff-based map updater

### Day 83: Apollo HD Map
- Apollo map format
- Base map and routing map
- Map creation tools
- Integration with planning
- **Lab:** Apollo map generation

### Day 84: Week 12 Review & Project (Map-Based Navigation)
- Project: Campus autonomous shuttle
- HD map-based path planning
- Lane-level localization
- Stop line and crosswalk handling

---

## Week 13: V2X Communication (Days 85-91)

### Day 85: V2X Standards (DSRC, C-V2X)
- DSRC (802.11p) vs C-V2X (5G)
- V2V, V2I, V2P, V2N
- Communication range and latency
- Security and privacy
- **Lab:** V2X message simulation

### Day 86: BSM (Basic Safety Messages)
- SAE J2735 message set
- Position, velocity, acceleration
- Message frequency and priority
- Broadcast vs unicast
- **Lab:** BSM generator and parser

### Day 87: CAM/DENM Messages
- Cooperative Awareness Messages
- Decentralized Environmental Notification
- European ITS-G5 standard
- Event triggered messages
- **Lab:** CAM/DENM implementation

### Day 88: Cooperative Perception
- Sensor data sharing
- Object list fusion
- Common coordinate frame
- Bandwidth optimization
- **Lab:** Multi-vehicle perception

### Day 89: Platooning
- Vehicle following algorithms
- String stability
- CACC (Cooperative ACC)
- Platoon formation and dissolution
- **Lab:** 3-vehicle platoon sim

### Day 90: Security and Privacy
- PKI (Public Key Infrastructure)
- Message authentication
- Certificate management
- Location privacy
- **Lab:** Secure V2X messaging

### Day 91: Week 13 Review & Project (Intersection Management)
- Project: Smart intersection system
- V2I communication
- Collision avoidance
- Priority-based scheduling

---

## Week 14: Simulation (CARLA & Gazebo) (Days 92-98)

### Day 92: Gazebo Worlds and Models
- SDF (Simulation Description Format)
- World files and models
- Physics engine
- Sensor plugins
- **Lab:** Custom Gazebo world

### Day 93: CARLA Simulator Setup
- CARLA installation
- Python API
- Synchronous vs asynchronous mode
- Client-server architecture
- **Lab:** Basic CARLA scenario

### Day 94: Scenario Runner
- OpenSCENARIO format
- Scenario definition
- Traffic manager
- Weather and lighting control
- **Lab:** Complex traffic scenario

### Day 95: Sensor Simulation (LiDAR, Camera, Radar)
- LiDAR ray tracing
- Camera rendering
- Radar detection simulation
- Sensor noise models
- **Lab:** Multi-sensor fusion in sim

### Day 96: Traffic Simulation
- NPC vehicle behavior
- Pedestrian AI
- Traffic light synchronization
- Realistic traffic patterns
- **Lab:** Urban traffic scenario

### Day 97: Hardware-in-the-Loop (HIL)
- Real vehicle controller + virtual env
- CAN bus simulation
- Timing synchronization
- Integration testing
- **Lab:** ECU-in-the-loop test

### Day 98: Week 14 Review & Project (Virtual Testing)
- Project: AV software validation
- 1000+ scenario testing
- Corner case generation
- Sim-to-real gap analysis

---

## Week 15: Safety Standards (ISO 26262 & SOTIF) (Days 99-105)

### Day 99: ASIL Decomposition
- ASIL levels (A, B, C, D)
- Redundancy strategies
- Decomposition rules
- Homogeneous vs heterogeneous
- **Lab:** ASIL decomposition exercise

### Day 100: Hazard Analysis and Risk Assessment (HARA)
- Hazardous events
- Severity, Exposure, Controllability
- ASIL determination
- Safety goals
- **Lab:** HARA for lane keeping

### Day 101: SOTIF (Safety of the Intended Functionality)
- ISO 21448 overview
- Performance limitations
- Triggering conditions
- Verification and validation
- **Lab:** SOTIF scenario catalog

### Day 102: Known and Unknown Unsafe Scenarios
- Scenario identification
- Edge case taxonomy
- Unknown unknowns
- Continuous monitoring
- **Lab:** Scenario database builder

### Day 103: Fault Injection Testing
- Hardware faults (stuck-at, bit-flip)
- Software faults (timing, logic)
- Byzantine faults
- Fault coverage metrics
- **Lab:** Sensor fault injection

### Day 104: Redundancy Architectures
- 1oo2, 2oo3 voting
- Dual-redundant systems
- Diverse redundancy
- Fail-operational design
- **Lab:** Redundant perception system

### Day 105: Week 15 Review & Project (Safety Case)
- Project: Safety argumentation
- Goal Structuring Notation (GSN)
- Evidence collection
- Safety case document

---

## Week 16: Radar Signal Processing (Days 106-112)

### Day 106: FMCW Radar Principles
- Frequency Modulated Continuous Wave
- Beat frequency
- Range resolution
- Maximum unambiguous range
- **Lab:** Radar waveform simulation

### Day 107: Range-Doppler FFT
- 2D FFT processing
- Range bins and Doppler bins
- Velocity resolution
- MTI (Moving Target Indication)
- **Lab:** Range-Doppler map generation

### Day 108: CFAR Detection
- Constant False Alarm Rate
- Cell-Averaging CFAR
- CFAR threshold calculation
- Edge effects
- **Lab:** CFAR detector implementation

### Day 109: Angle Estimation (MUSIC, ESPRIT)
- Array signal processing
- Direction of Arrival (DOA)
- MUSIC algorithm
- ESPRIT algorithm
- **Lab:** Angle estimation comparison

### Day 110: Micro-Doppler Signatures
- Pedestrian classification
- Gesture recognition
- Spectrogram analysis
- Feature extraction
- **Lab:** Pedestrian vs cyclist classification

### Day 111: Radar Cross Section (RCS)
- Target reflectivity
- RCS modeling
- Radar equation
- SNR calculation
- **Lab:** RCS measurement simulation

### Day 112: Week 16 Review & Project (Radar Tracking)
- Project: 77GHz radar tracker
- Multi-target tracking
- Ghost target filtering
- Radar-camera fusion

---

## Week 17: Sensor Calibration & Synchronization (Days 113-119)

### Day 113: Camera-LiDAR Calibration
- Extrinsic calibration
- Checkerboard-based method
- Targetless calibration
- Optimization objectives
- **Lab:** Manual vs automatic calibration

### Day 114: LiDAR-LiDAR Calibration
- Multi-LiDAR systems
- ICP-based alignment
- Feature-based methods
- Calibration verification
- **Lab:** Dual-LiDAR calibration

### Day 115: Camera-Radar Calibration
- Reflector-based targets
- Sparse radar data
- Calibration challenges
- Validation metrics
- **Lab:** Camera-radar extrinsic calib

### Day 116: Hand-Eye Calibration
- Eye-in-hand vs eye-to-hand
- AX=XB problem
- Rotation and translation decoupling
- Multiple pose estimation
- **Lab:** Robot arm calibration

### Day 117: Online Calibration
- Real-time parameter estimation
- Drift detection
- Self-calibration methods
- Adaptive calibration
- **Lab:** Online IMU calibration

### Day 118: Time Synchronization (PTP, NTP)
- IEEE 1588 Precision Time Protocol
- Network Time Protocol
- Hardware timestamping
- Latency compensation
- **Lab:** Multi-sensor time sync

### Day 119: Week 17 Review & Project (Multi-Sensor Rig)
- Project: Complete sensor suite calibration
- Camera-LiDAR-Radar-IMU
- Calibration verification
- Accuracy assessment

---

## Week 18: Deep Learning for Perception (Days 120-126)

### Day 120: Transformers for Detection (DETR)
- Attention mechanisms
- Object queries
- Bipartite matching
- Transformer decoder
- **Lab:** DETR training on COCO

### Day 121: Occupancy Networks
- 3D scene representation
- Voxel occupancy prediction
- MonoScene, BEVFormer
- Future prediction
- **Lab:** Occupancy grid prediction

### Day 122: Neural Radiance Fields (NeRF)
- Implicit 3D representation
- Volume rendering
- Camera pose optimization
- Novel view synthesis
- **Lab:** NeRF for scene reconstruction

### Day 123: Self-Supervised Learning
- Pretext tasks
- Contrastive learning
- MAE (Masked Autoencoders)
- Pseudo-labeling
- **Lab:** Self-supervised pre-training

### Day 124: Domain Adaptation
- Sim-to-real transfer
- Adversarial training
- Style transfer
- Test-time adaptation
- **Lab:** Weather domain adaptation

### Day 125: Continual Learning
- Catastrophic forgetting
- Rehearsal strategies
- Progressive networks
- Lifelong learning
- **Lab:** Incremental class learning

### Day 126: Week 18 Review & Project (Adaptation Pipeline)
- Project: All-weather perception
- Multi-domain training
- Online adaptation
- Performance monitoring

---

## Week 19: End-to-End Autonomous Driving (Days 127-133)

### Day 127: Imitation Learning (NVIDIA PilotNet)
- Behavioral cloning
- Dataset collection
- Steering angle prediction
- Limitations
- **Lab:** Simple imitation learner

### Day 128: Reinforcement Learning (DQN, PPO)
- Deep Q-Networks
- Policy gradient methods
- Reward shaping
- Exploration strategies
- **Lab:** RL for lane keeping

### Day 129: CARLA Challenge
- Leaderboard participation
- Route following
- Traffic rule compliance
- Metrics (success rate, infraction)
- **Lab:** Challenge submission

### Day 130: Uncertainty Estimation
- Aleatoric vs epistemic uncertainty
- Monte Carlo Dropout
- Ensemble methods
- Confidence calibration
- **Lab:** Uncertainty-aware planner

### Day 131: Explainability (Grad-CAM, Attention)
- Attention visualization
- Saliency maps
- Counterfactual analysis
- Trust and transparency
- **Lab:** Attention analysis

### Day 132: Sim-to-Real Transfer
- Domain randomization
- CycleGAN
- Transfer learning
- Reality gap
- **Lab:** Sim-trained policy on real robot

### Day 133: Week 19 Review & Project (End-to-End Parking)
- Project: Neural network parking
- Imitation + RL hybrid
- Safety constraints
- Real-world deployment

---

## Week 20: Hardware Platforms (Orin/Drive AGX) (Days 134-140)

### Day 140: NVIDIA Drive Orin Architecture
- SoC overview (CPU, GPU, DLA, PVA)
- Power modes
- Thermal management
- Linux for Tegra (L4T)
- **Lab:** Orin developer kit setup

### Day 135: DriveWorks SDK
- Sensor abstraction
- Core modules
- Calibration tools
- Visualization
- **Lab:** DriveWorks sample apps

### Day 136: Sensor Abstraction Layer (SAL)
- Camera, LiDAR, Radar, CAN interfaces
- Plugin architecture
- Custom sensor integration
- Data recording
- **Lab:** Custom camera plugin

### Day 137: DLA (Deep Learning Accelerator)
- INT8 inference
- DLA-supported layers
- Compiler optimization
- Power efficiency
- **Lab:** Model deployment on DLA

### Day 138: PVA (Programmable Vision Accelerator)
- Stereo processing
- Optical flow
- Feature detection
- Custom kernels
- **Lab:** Stereo disparity on PVA

### Day 139: Hypervisor and QNX
- Safety Island concept
- QNX RTOS
- VM isolation
- Inter-VM communication
- **Lab:** Mixed-criticality system

### Day 140: Week 20 Review & Project (Platform Bring-Up)
- Project: Full stack on Orin
- Multi-sensor pipeline
- DLA + PVA utilization
- Performance benchmarking

---

## Week 21: Fleet Management & OTA Updates (Days 141-147)

### Day 141: Fleet Management Architecture
- Cloud infrastructure
- Edge computing
- Data pipeline
- Monitoring dashboard
- **Lab:** AWS IoT fleet management

### Day 142: Data Collection and Logging
- ROS2 bag recording
- Selective logging
- Cloud upload
- Data annotation
- **Lab:** Automated data collection

### Day 143: Remote Diagnostics
- DTC (Diagnostic Trouble Codes)
- Log analysis
- Remote debugging
- Telemetry streaming
- **Lab:** Remote diagnostic tool

### Day 144: OTA Update Mechanisms
- Binary diff updates
- Incremental updates
- Download management
- Verification
- **Lab:** OTA update server

### Day 145: A/B Partitioning
- Dual-boot partitions
- Atomic updates
- Rollback capability
- Partition switching
- **Lab:** A/B update implementation

### Day 146: Rollback and Recovery
- Update verification
- Automatic rollback
- Recovery mode
- Factory reset
- **Lab:** Fault-tolerant update

### Day 147: Week 21 Review & Project (Fleet Dashboard)
- Project: Fleet monitoring system
- Real-time vehicle status
- OTA update deployment
- Analytics and reporting

---

## Week 22: Edge Cases & Corner Cases (Days 148-154)

### Day 148: Weather Conditions (Rain, Snow, Fog)
- Sensor degradation
- Detection under rain
- Snow accumulation
- Fog penetration
- **Lab:** Weather augmentation

### Day 149: Lighting Conditions (Night, Glare)
- Low-light enhancement
- HDR imaging
- Sun glare handling
- Headlight reflections
- **Lab:** Night perception

### Day 150: Occlusions and Shadows
- Partial occlusion handling
- Shadow detection
- Occluded pedestrian prediction
- Visibility estimation
- **Lab:** Occlusion-aware tracker

### Day 151: Road Conditions (Construction, Potholes)
- Temporary signage
- Lane markings absence
- Road damage detection
- Detour handling
- **Lab:** Construction zone navigation

### Day 152: Rare Events (Emergency Vehicles)
- Siren detection
- Emergency vehicle detection
- Yield behavior
- Unusual object handling
- **Lab:** Emergency response logic

### Day 153: Adversarial Scenarios
- Adversarial patches
- Spoofing attacks
- Robustness testing
- Defense mechanisms
- **Lab:** Adversarial robustness

### Day 154: Week 22 Review & Project (Stress Testing)
- Project: Comprehensive edge case testing
- 1000+ corner case scenarios
- Failure mode analysis
- Mitigation strategies

---

## Week 23: Testing & Validation (MIL/SIL/HIL) (Days 155-161)

### Day 155: Model-in-the-Loop (MIL)
- Algorithm validation
- MATLAB/Simulink
- Requirement tracing
- Code generation
- **Lab:** MIL test suite

### Day 156: Software-in-the-Loop (SIL)
- Compiled code testing
- Rapid prototyping
- Continuous integration
- Automated regression
- **Lab:** SIL test framework

### Day 157: Hardware-in-the-Loop (HIL)
- Real ECU testing
- Restbus simulation
- Fault injection
- Timing validation
- **Lab:** HIL test bench

### Day 158: Vehicle-in-the-Loop (VIL)
- Testbed vehicle
- Controlled environment
- Repeatability
- Safety driver
- **Lab:** Proving ground test

### Day 159: Scenario-Based Testing
- OpenSCENARIO
- Logical scenarios
- Concrete scenarios
- Parameterized testing
- **Lab:** Scenario library

### Day 160: Regression Testing
- Test case management
- Automated execution
- Diff analysis
- CI/CD integration
- **Lab:** Jenkins pipeline

### Day 161: Week 23 Review & Project (Test Automation)
- Project: Complete test framework
- MIL/SIL/HIL integration
- Automated reporting
- Coverage metrics

---

## Week 24: Capstone Project - Autonomous Valet Parking (Days 162-168)

### Day 162: Project Planning & Architecture
- System requirements
- Sensor configuration
- Module interfaces
- Development plan
- **Deliverable:** Architecture document

### Day 163: Environment Mapping
- Parking lot mapping
- Parking space detection
- Multi-floor handling
- Map updates
- **Deliverable:** HD map of parking lot

### Day 164: Path Planning in Parking Lots
- Narrow space navigation
- Hybrid A* implementation
- Parking spot assignment
- Collision-free paths
- **Deliverable:** Path planner module

### Day 165: Perception Integration
- Multi-sensor fusion
- 360° coverage
- Close-range detection
- Parking line detection
- **Deliverable:** Perception pipeline

### Day 166: Low-Speed Control
- Precise maneuvering
- Parking controller
- Safety monitoring
- Emergency stop
- **Deliverable:** Control module

### Day 167: Integration & Testing
- Full system integration
- Simulation testing
- Real vehicle testing (if available)
- Performance validation
- **Deliverable:** Test results

### Day 168: Final Demo & Presentation
- Live demonstration
- Project presentation
- Lessons learned
- Future improvements
- **Deliverable:** Final report & demo video

---

## Week 25: Advanced ADAS Topics & Production Readiness (Days 169-175)

### Day 169: Ultrasonic Sensors & Parking Assistance
- Ultrasonic sensor principles (ToF)
- Multi-sensor array configuration
- Parking slot detection algorithms
- Surround view synthesis
- Cross-traffic alert
- **Lab:** Ultrasonic-based parking system

### Day 170: CAN Bus Protocol Deep Dive
- CAN 2.0 vs CAN FD
- Message arbitration and priorities
- DBC (Database CAN) file format
- J1939 for heavy vehicles
- Diagnostic protocols (UDS, OBD-II)
- **Lab:** CAN bus sniffer and simulator

### Day 171: Trajectory Prediction & Intent Estimation
- Constant velocity/acceleration models
- Social pooling networks
- Interaction-aware prediction
- Multi-modal prediction
- Goal-oriented behavior models
- **Lab:** Vehicle trajectory forecasting

### Day 172: Lane Detection & Traffic Sign/Light Recognition
- Traditional lane detection (Hough, RANSAC)
- Deep learning approaches (LaneNet, SCNN)
- Traffic sign classification (GTSRB dataset)
- Traffic light detection (LISA dataset)
- Temporal smoothing and tracking
- **Lab:** Complete lane+sign recognition pipeline

### Day 173: Semantic Segmentation & Cost Map Generation
- Road/sidewalk/vehicle segmentation
- DeepLab, SegNet, U-Net architectures
- Real-time segmentation (BiSeNet, STDC)
- Cost map layers (static, dynamic, semantic)
- Traversability analysis
- **Lab:** Drivable area segmentation

### Day 174: Driver Monitoring & Fail-Safe Mechanisms
- Eye tracking and gaze estimation
- Drowsiness detection
- Distraction classification
- Hand position monitoring
- Minimal Risk Condition (MRC)
- Degraded operation modes
- **Lab:** Driver attention monitoring system

### Day 175: Week 25 Review & Cyber Security
- Intrusion Detection Systems (IDS)
- Secure boot and chain of trust
- ECU security (HSM, SHE)
- Penetration testing
- UNECE WP.29 cyber security regulation
- **Project:** Security hardening assessment

---

## Assessment & Certification

### Weekly Projects (25 projects)
- Hands-on implementation
- Code review
- Performance metrics
- Documentation

### Final Capstone Project
- Autonomous Valet Parking System
- Complete autonomous stack
- Real-world deployment considerations
- Industry-standard quality

### Certification Requirements
1. Complete all 175 days of content
2. Submit all 25 weekly projects
3. Pass final capstone project
4. Demonstrate proficiency in:
   - ROS 2 development
   - Sensor fusion
   - Path planning
   - Perception systems
   - Safety-critical design
   - Cyber security
   - Production readiness

---

## Tools & Technologies

### Software
- ROS 2 Humble
- Python 3.10+
- C++17
- PyTorch, TensorFlow
- Point Cloud Library (PCL)
- OpenCV
- CARLA Simulator
- Gazebo
- NVIDIA DriveWorks

### Hardware (Recommended)
- NVIDIA Jetson Orin / Drive AGX
- LiDAR (Velodyne VLP-16 or equivalent)
- Camera (1080p stereo or mono)
- IMU (9-DOF)
- GPS (RTK capable)
- CAN bus interface

---

## Next Steps After Completion

1. **Industry Deployment:** Work on production AV systems
2. **Research:** Publish in IROS, ICRA, CVPR
3. **Specialization:** Focus on perception, planning, or control
4. **Leadership:** Lead autonomous systems teams

---

**Phase 4: ADAS & Robotics Systems** | 168 Days to Autonomous Driving Mastery
