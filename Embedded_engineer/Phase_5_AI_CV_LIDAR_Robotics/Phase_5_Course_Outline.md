# Phase 5: AI/CV/LIDAR End-to-End Robotics
## Course Outline - 30 Weeks (210 Days)

---

## Course Overview

**Duration:** 30 weeks / 210 days  
**Focus:** Advanced AI, Computer Vision, LIDAR Processing & End-to-End Robotics Systems  
**Prerequisites:** Phase 4 completion (ADAS & Robotics Systems)

**Learning Path:**
- Foundation Models → Production Deployment
- Individual Perception → End-to-End Systems
- Simulation Training → Real-World Transfer
- Single Robot → Multi-Robot Systems
- Research Prototypes → Industry-Ready Solutions

---

## Featured Open-Source Projects & Frameworks

| Category | Key Projects |
|----------|-------------|
| **Navigation** | Nav2, AMCL, SLAM Toolbox, Cartographer |
| **Manipulation** | MoveIt 2, MoveIt Task Constructor, CRISP |
| **Perception** | PointPillars, VoxelNeXt, BEVFusion, CenterPoint |
| **Foundation Models** | RT-1, RT-2, PaLM-E, OpenVLA, LeRobot |
| **Simulation** | Isaac Sim/Lab, MuJoCo, Gazebo, CARLA |
| **Learning** | Diffusion Policy, DreamerV3, Humanoid-Gym |
| **Edge Deployment** | TensorRT, ONNX Runtime, JetPack SDK |
| **SLAM** | ORB-SLAM3, RTAB-Map, 3DGS-SLAM |
| **Humanoids** | Poppy, Reachy 2, Berkeley Humanoid Lite |
| **Multi-Robot** | Open-RMF, PlanSys2, Swarm Robotics |

---

## Week 1: Deep Learning Foundations for Robotics (Days 1-7)

### Day 1: Neural Network Architectures for Robotics
- CNN architectures (ResNet, EfficientNet, ConvNeXt)
- Vision Transformers (ViT, DeiT, Swin Transformer)
- Hybrid architectures for robotics
- Real-time inference considerations
- **Lab:** Benchmark different architectures on Jetson
- **Project:** NASA-JPL Open-Source Rover perception module

### Day 2: Attention Mechanisms & Transformers
- Self-attention and cross-attention
- Multi-head attention design
- Positional encodings for spatial data
- Flash Attention for efficiency
- **Lab:** Implement attention-based feature extraction
- **Project:** TurtleBot4 attention-based object tracking

### Day 3: Feature Pyramid Networks & Multi-Scale Processing
- FPN, PANet, BiFPN architectures
- Multi-scale feature fusion
- Object detection at different scales
- Anchor-free detection heads
- **Lab:** Build multi-scale detection pipeline
- **Project:** Mini Pupper object detection system

### Day 4: Point Cloud Neural Networks
- PointNet and PointNet++ architectures
- Point cloud convolutions (PointConv, KPConv)
- Sparse convolutions for 3D data
- Point Transformer architecture
- **Lab:** Point cloud classification and segmentation
- **Project:** ROS2 point cloud processing node

### Day 5: Graph Neural Networks for Robotics
- Scene graphs and spatial relationships
- Message passing networks
- Graph attention networks
- 3D scene graph generation
- **Lab:** Build scene graph from sensor data
- **Project:** Spatial relationship reasoning system

### Day 6: Self-Supervised Learning for Robotics
- Contrastive learning (SimCLR, MoCo, DINO)
- Masked autoencoders (MAE)
- Pretext tasks for robotics
- Multi-modal self-supervision
- **Lab:** Pre-train vision model on robot data
- **Project:** Self-supervised feature learning pipeline

### Day 7: Week 1 Review & Project (Real-Time Inference)
- Project: Real-time multi-modal perception
- Optimized inference pipeline
- Latency benchmarking
- **Popular Project:** NVIDIA Isaac ROS integration

---

## Week 2: Advanced LiDAR Perception (Days 8-14)

### Day 8: 4D LiDAR & Velocity Estimation
- Aeva 4D LiDAR principles
- Instant velocity detection
- Motion-compensated point clouds
- Dynamic object handling
- **Lab:** Process 4D LiDAR data
- **Project:** Velocity-enhanced object detection

### Day 9: VoxelNeXt Architecture
- Fully sparse 3D detection
- Voxel-based feature extraction
- Efficient backbone design
- NMS-free detection
- **Lab:** Deploy VoxelNeXt on nuScenes
- **Project:** ROS2 VoxelNeXt integration

### Day 10: PointPillars Deep Dive
- Pillar feature network
- 2D backbone efficiency
- KITTI and nuScenes training
- Real-time deployment strategies
- **Lab:** Train PointPillars from scratch
- **Project:** Jetson Orin PointPillars deployment

### Day 11: CenterPoint 3D Detection
- Center-based detection paradigm
- Velocity and attribute prediction
- Two-stage refinement
- Temporal aggregation
- **Lab:** CenterPoint multi-frame fusion
- **Project:** Highway vehicle detection system

### Day 12: LiDAR Panoptic Segmentation
- Panoptic segmentation fundamentals
- Thing vs stuff classification
- Instance segmentation on point clouds
- Panoptic-DeepLab adaptation
- **Lab:** Panoptic segmentation pipeline
- **Project:** Semantic mapping with panoptic labels

### Day 13: Range Image Representation
- Range image projection
- RangeNet++ architecture
- Efficient inference
- Multi-view fusion
- **Lab:** Range image-based segmentation
- **Project:** Real-time drivable area detection

### Day 14: Week 2 Review & Project (LiDAR Perception Stack)
- Project: Complete LiDAR perception pipeline
- Detection + Segmentation + Tracking
- Performance optimization
- **Popular Project:** Autoware.Universe LiDAR stack

---

## Week 3: BEV (Bird's Eye View) Representation (Days 15-21)

### Day 15: BEV Fundamentals & Geometry
- Camera to BEV transformation
- Homography and IPM limitations
- Depth estimation for BEV
- Multi-camera BEV generation
- **Lab:** Implement IPM and learned BEV
- **Project:** Surround-view BEV system

### Day 16: BEVFormer Architecture
- Spatiotemporal attention
- BEV queries and deformable attention
- Temporal self-attention
- Multi-scale feature sampling
- **Lab:** BEVFormer on nuScenes
- **Project:** Multi-camera perception node

### Day 17: BEVFusion Multi-Modal Fusion
- Camera-LiDAR BEV fusion
- Unified BEV representation
- Late fusion strategies
- Geometric and semantic alignment
- **Lab:** Implement BEVFusion pipeline
- **Project:** Sensor fusion for autonomous driving

### Day 18: BEV Segmentation & Map Prediction
- HD map prediction from sensors
- Lane and road topology
- Semantic BEV segmentation
- Vectorized map representation
- **Lab:** MapTR-style map prediction
- **Project:** Online HD map generation

### Day 19: Occupancy Networks
- 3D occupancy prediction
- MonoScene and TPVFormer
- Voxel occupancy grids
- Semantic occupancy
- **Lab:** Build 3D occupancy predictor
- **Project:** Collision-free space estimation

### Day 20: Temporal BEV Aggregation
- Multi-frame BEV fusion
- Motion compensation
- Recurrent BEV features
- Long-horizon prediction
- **Lab:** Temporal BEV tracking
- **Project:** Predictive BEV for planning

### Day 21: Week 3 Review & Project (Complete BEV System)
- Project: End-to-end BEV perception
- Multi-camera + LiDAR fusion
- Real-time deployment
- **Popular Project:** StreamPETR implementation

---

## Week 4: 3D Gaussian Splatting & Neural Scene Representation (Days 22-28)

### Day 22: 3D Gaussian Splatting Fundamentals
- Gaussian primitives for scene representation
- Differentiable rasterization
- Training from images
- Novel view synthesis
- **Lab:** Train 3DGS on custom scene
- **Project:** Robot environment modeling

### Day 23: 3DGS-SLAM Systems
- Real-time 3DGS reconstruction
- Camera pose optimization
- Incremental Gaussian addition
- RP-SLAM and RTG-SLAM
- **Lab:** Implement monocular 3DGS-SLAM
- **Project:** Indoor robot mapping with 3DGS

### Day 24: Dynamic Scene Handling
- Dynamic object segmentation
- 4D Gaussian Splatting
- Temporal consistency
- Object removal and inpainting
- **Lab:** Dynamic scene reconstruction
- **Project:** Remove moving objects from map

### Day 25: Semantic 3D Gaussians
- NEDS-SLAM architecture
- Semantic feature integration
- Language-embedded Gaussians
- Open-vocabulary scene understanding
- **Lab:** Semantic 3DGS scene
- **Project:** Language-guided robot navigation

### Day 26: Active SLAM with Gaussians
- AG-SLAM exploration
- Fisher Information for planning
- Next-best-view selection
- Uncertainty quantification
- **Lab:** Active mapping strategy
- **Project:** Autonomous exploration robot

### Day 27: Neural Radiance Fields (NeRF) Comparison
- NeRF fundamentals
- Instant-NGP acceleration
- 3DGS vs NeRF trade-offs
- Hybrid approaches
- **Lab:** Compare NeRF and 3DGS quality
- **Project:** Scene representation benchmark

### Day 28: Week 4 Review & Project (Neural Scene Mapping)
- Project: Complete neural mapping system
- Real-time reconstruction
- Semantic understanding
- **Popular Project:** SplaTAM implementation

---

## Week 5: Foundation Models for Robotics I (Days 29-35)

### Day 29: Vision-Language Models for Robotics
- CLIP and OpenCLIP
- Vision-language pre-training
- Zero-shot object recognition
- Open-vocabulary detection (OWL-ViT)
- **Lab:** CLIP-based object detection
- **Project:** Open-vocabulary robot perception

### Day 30: RT-1 (Robotics Transformer 1)
- Multi-task robot learning
- Tokenizing robot inputs/outputs
- 700+ task training
- Real-time action generation
- **Lab:** Study RT-1 architecture
- **Project:** Multi-task manipulation policy

### Day 31: RT-2 (Vision-Language-Action Model)
- VLM to VLA adaptation
- Chain-of-thought reasoning
- Emergent capabilities
- Web-scale knowledge transfer
- **Lab:** Implement RT-2 style inference
- **Project:** Language-conditioned manipulation

### Day 32: PaLM-E (Embodied Multimodal LLM)
- Embodied language models
- Multi-modal input fusion
- Visual chain-of-thought
- Cross-domain transfer
- **Lab:** PaLM-E style reasoning
- **Project:** Visual question answering for robots

### Day 33: OpenVLA & LeRobot
- Open-source VLA models
- Hugging Face robotics ecosystem
- Fine-tuning strategies
- Community datasets
- **Lab:** Fine-tune OpenVLA
- **Project:** Custom task VLA training, **Popular Project:** Hugging Face LeRobot

### Day 34: RT-X & Open X-Embodiment
- Cross-embodiment learning
- Large-scale robot datasets
- Transfer across robot morphologies
- Generalization capabilities
- **Lab:** Explore Open X-Embodiment data
- **Project:** Cross-robot policy transfer

### Day 35: Week 5 Review & Project (VLA Deployment)
- Project: Deploy VLA on real robot
- Language-guided manipulation
- Task generalization testing
- **Popular Project:** Google DeepMind RT-X reproduction

---

## Week 6: Foundation Models for Robotics II (Days 36-42)

### Day 36: GPT-4V & Multimodal LLMs for Robotics
- GPT-4V for scene understanding
- Structured output generation
- Task planning with LLMs
- API integration for robotics
- **Lab:** GPT-4V robotics interface
- **Project:** LLM-based task decomposition

### Day 37: Grounding DINO & SAM
- Open-set object detection
- Segment Anything Model (SAM)
- Grounded segmentation
- Instance-level understanding
- **Lab:** Grounding DINO + SAM pipeline
- **Project:** Zero-shot pick-and-place

### Day 38: DETIC & Open-Vocabulary Detection
- Large vocabulary detection
- CLIP integration
- Novel category generalization
- Real-time deployment
- **Lab:** DETIC for robot perception
- **Project:** Novel object manipulation

### Day 39: SigLIP & Efficient Vision-Language
- Improved vision-language training
- Sigmoid loss for contrastive learning
- Efficient fine-tuning
- Mobile deployment
- **Lab:** SigLIP for robotics
- **Project:** Edge-deployed VLM

### Day 40: Video Understanding Models
- Video-LLMs for robotics
- Temporal reasoning
- Action recognition
- Future prediction
- **Lab:** Video understanding pipeline
- **Project:** Activity monitoring robot

### Day 41: Speech & Language Integration
- Speech-to-text for robotics
- Natural language commands
- Multi-turn dialogue
- Voice-controlled manipulation
- **Lab:** Voice command interface
- **Project:** Voice-controlled robot arm

### Day 42: Week 6 Review & Project (Multimodal Robot Assistant)
- Project: Complete multimodal robot
- Vision + Language + Speech
- Natural interaction
- **Popular Project:** Reachy 2 + LLM integration

---

## Week 7: Diffusion Models for Robotics (Days 43-49)

### Day 43: Diffusion Model Fundamentals
- DDPM theory
- Score-based generative models
- Noise scheduling
- Conditioning mechanisms
- **Lab:** Train simple diffusion model
- **Project:** Action distribution modeling

### Day 44: Diffusion Policy
- Iterative action denoising
- Multi-modal action distributions
- Behavior cloning with diffusion
- Stable training dynamics
- **Lab:** Implement Diffusion Policy
- **Project:** Manipulation task learning

### Day 45: Policy Composition (PoCo)
- Multi-policy combination
- Dataset mixing strategies
- Generalized manipulation
- Task transfer
- **Lab:** Compose multiple policies
- **Project:** Multi-task robot, **Popular Project:** MIT CSAIL Diffusion Policy

### Day 46: Motion Planning Diffusion (MPD)
- Trajectory distribution learning
- B-spline trajectory representation
- Cost-guided sampling
- Constraint satisfaction
- **Lab:** MPD for manipulation
- **Project:** Collision-free motion planning

### Day 47: Multi-Robot Diffusion Planning (MMD)
- Multi-agent trajectory generation
- MAPF integration
- Collision avoidance
- Scalable multi-robot planning
- **Lab:** Multi-robot path planning
- **Project:** Warehouse robot coordination

### Day 48: Diffusion for Data Augmentation
- Scene reconstruction with diffusion
- Synthetic data generation
- Domain randomization
- Dataset scaling
- **Lab:** Generate training data
- **Project:** Augmented manipulation dataset

### Day 49: Week 7 Review & Project (Diffusion-Based Control)
- Project: Complete diffusion control system
- Multi-task manipulation
- Real-world deployment
- **Popular Project:** Diffusion Contact Model (Panasonic)

---

## Week 8: World Models & Video Prediction (Days 50-56)

### Day 50: World Model Fundamentals
- Internal environment simulation
- Latent dynamics models
- Action-conditioned prediction
- Planning in imagination
- **Lab:** Simple world model training
- **Project:** Predictive robot controller

### Day 51: DreamerV3 Architecture
- Recurrent state-space model
- Categorical latent representations
- Actor-critic in imagination
- Scalable training
- **Lab:** Train DreamerV3 policy
- **Project:** Vision-based robot control

### Day 52: RoboDreamer & Compositional World Models
- Language-guided video generation
- Compositional generalization
- Action primitive decomposition
- Novel task synthesis
- **Lab:** Compositional video prediction
- **Project:** Language-conditioned planning

### Day 53: WorldDreamer for General Video
- Universal world physics modeling
- Masked visual token prediction
- Video inpainting and editing
- Text-to-video generation
- **Lab:** General video prediction
- **Project:** Future state visualization

### Day 54: Language-Guided World Models
- Natural language environment control
- Text-to-simulation
- Efficient agent programming
- Interactive world building
- **Lab:** Language-world interface
- **Project:** Verbal instruction following

### Day 55: Action-Conditioned Video Prediction
- Next-frame prediction
- Long-horizon video synthesis
- Action consequences modeling
- Planning with predictions
- **Lab:** Action-video prediction model
- **Project:** Visual model predictive control

### Day 56: Week 8 Review & Project (World Model Robot)
- Project: Complete world model system
- Imagination-based planning
- Sim-to-real transfer
- **Popular Project:** DreamerV3 on Isaac Lab

---

## Week 9: Reinforcement Learning for Robotics (Days 57-63)

### Day 57: Deep RL Fundamentals for Robots
- Policy gradient methods (PPO, SAC)
- Value function estimation
- On-policy vs off-policy
- Continuous action spaces
- **Lab:** Train simple manipulation policy
- **Project:** ROS2 RL integration

### Day 58: Isaac Lab & Isaac Sim
- NVIDIA Isaac platform overview
- GPU-accelerated simulation
- Parallel environment training
- Isaac Lab workflow
- **Lab:** Setup Isaac Lab environment
- **Project:** Train locomotion policy

### Day 59: MuJoCo for Robot Learning
- MuJoCo physics simulation
- Contact dynamics modeling
- Articulated body simulation
- Python API and custom environments
- **Lab:** Custom MuJoCo environment
- **Project:** Dexterous manipulation task

### Day 60: Sim-to-Real Transfer Techniques
- Domain randomization
- System identification
- Domain adaptation
- Sim-to-sim verification
- **Lab:** Domain randomization pipeline
- **Project:** Transfer policy to real robot

### Day 61: RialTo Real-to-Sim-to-Real
- Digital twin creation
- Real environment scanning
- Hybrid training approach
- Efficient policy transfer
- **Lab:** Build digital twin
- **Project:** Real-to-sim-to-real manipulation, **Popular Project:** MIT CSAIL RialTo

### Day 62: Humanoid Locomotion with RL
- Humanoid-Gym framework
- Zero-shot sim-to-real
- XBot locomotion
- Bipedal walking control
- **Lab:** Train humanoid walking
- **Project:** Quadruped locomotion, **Popular Project:** Humanoid-Gym

### Day 63: Week 9 Review & Project (RL Robot System)
- Project: Complete RL-based robot
- Simulation training
- Real-world deployment
- Performance analysis

---

## Week 10: Robot Manipulation & Grasping (Days 64-70)

### Day 64: MoveIt 2 Deep Dive
- Motion planning architecture
- OMPL integration
- Collision checking
- Kinematics plugins
- **Lab:** MoveIt 2 setup and planning
- **Project:** Pick-and-place system

### Day 65: MoveIt Task Constructor
- Hierarchical task planning
- Motion planning pipeline
- Complex manipulation sequences
- Error recovery
- **Lab:** Multi-stage pick-and-place
- **Project:** Assembly task planning

### Day 66: Grasping Neural Process (MIT)
- Predictive physics for grasping
- Hidden property inference
- Real-time grasp adaptation
- Stable grasp generation
- **Lab:** Implement grasp predictor
- **Project:** Novel object grasping, **Popular Project:** MIT Grasping Neural Process

### Day 67: Dexterous Manipulation
- Multi-finger hand control
- BiDexHand and DG-5F
- Contact-rich manipulation
- Tactile sensing integration
- **Lab:** Dexterous hand simulation
- **Project:** In-hand manipulation

### Day 68: Manipulate-Anything (VLM Grasping)
- Vision-language manipulation
- Zero-shot task solving
- Any-object manipulation
- Autonomous skill generation
- **Lab:** VLM-guided grasping
- **Project:** Open-world manipulation

### Day 69: Force & Compliance Control
- Impedance and admittance control
- Force sensing integration
- Contact-rich tasks
- CRISP ROS2 framework
- **Lab:** Force-controlled assembly
- **Project:** Peg-in-hole insertion

### Day 70: Week 10 Review & Project (Advanced Manipulation)
- Project: Complete manipulation system
- Multi-object tasks
- Error recovery
- **Popular Project:** Franka Emika ROS2 integration

---

## Week 11: Navigation 2 (Nav2) Mastery (Days 71-77)

### Day 71: Nav2 Architecture Deep Dive
- Behavior trees for navigation
- Planner and controller servers
- Costmap layers
- Recovery behaviors
- **Lab:** Custom Nav2 configuration
- **Project:** Warehouse robot navigation

### Day 72: SLAM Toolbox & Mapping
- 2D SLAM algorithms
- Loop closure detection
- Map serialization
- Lifelong mapping
- **Lab:** Large-scale mapping
- **Project:** Office building map

### Day 73: Costmap Configuration & Layers
- Inflation layer tuning
- Obstacle layer configuration
- Voxel layer for 3D
- Custom costmap layers
- **Lab:** Multi-layer costmap
- **Project:** Complex environment navigation

### Day 74: Custom Planners & Controllers
- NavFn vs Smac planners
- DWB vs MPPI controllers
- Regulated Pure Pursuit
- Custom plugin development
- **Lab:** Implement custom controller
- **Project:** Ackermann vehicle navigation

### Day 75: Behavior Tree Navigation
- BT.CPP integration
- Custom navigation behaviors
- Complex mission planning
- Conditional execution
- **Lab:** Multi-goal navigation
- **Project:** Patrol robot behavior

### Day 76: Outdoor Navigation & GPS Integration
- GPS waypoint navigation
- RTK-GPS integration
- Costmap for unstructured terrain
- Agricultural robots
- **Lab:** GPS-guided navigation
- **Project:** Field robot navigation

### Day 77: Week 11 Review & Project (Complete Nav2 System)
- Project: Autonomous mobile robot
- Multi-floor navigation
- Elevator integration
- **Popular Project:** TurtleBot4 Nav2 deployment

---

## Week 12: Multi-Robot Systems (Days 78-84)

### Day 78: Open-RMF (Robot Management Framework)
- Fleet management architecture
- Traffic management
- Task allocation
- Inter-robot communication
- **Lab:** Open-RMF setup
- **Project:** Multi-robot fleet, **Popular Project:** Open-RMF

### Day 79: Multi-Robot Path Finding (MAPF)
- CBS (Conflict-Based Search)
- Priority-based planning
- Deadlock prevention
- Scalable algorithms
- **Lab:** MAPF implementation
- **Project:** Warehouse coordination

### Day 80: Swarm Robotics Fundamentals
- Decentralized control
- Self-organization
- Emergent behaviors
- Robustness and scalability
- **Lab:** Swarm formation control
- **Project:** Robot swarm simulation

### Day 81: Multi-Robot SLAM
- Distributed mapping
- Map merging algorithms
- Multi-robot localization
- Collaborative exploration
- **Lab:** Multi-robot mapping
- **Project:** Collaborative warehouse mapping

### Day 82: Task Allocation & Scheduling
- Market-based allocation
- Auction algorithms
- Optimization approaches
- Dynamic task reallocation
- **Lab:** Task allocation system
- **Project:** Delivery robot fleet

### Day 83: Multi-Robot Manipulation
- Cooperative manipulation
- Dual-arm coordination
- Load sharing
- Synchronized actions
- **Lab:** Dual-arm manipulation
- **Project:** Heavy object transport

### Day 84: Week 12 Review & Project (Multi-Robot Warehouse)
- Project: Complete warehouse automation
- Multiple AMRs
- Central coordination
- **Popular Project:** Amazon-style warehouse robots

---

## Week 13: Humanoid Robotics (Days 85-91)

### Day 85: Humanoid Robot Platforms
- Poppy Project open-source
- Reachy 2 (Hugging Face)
- Berkeley Humanoid Lite
- Commercial platforms overview
- **Lab:** Poppy robot simulation
- **Project:** Humanoid setup

### Day 86: Bipedal Locomotion Control
- ZMP (Zero Moment Point)
- Capture point dynamics
- Walking pattern generation
- Dynamic balance
- **Lab:** Bipedal walking simulation
- **Project:** Walking controller design

### Day 87: Whole-Body Control
- Inverse kinematics
- Task prioritization
- Constraint handling
- Real-time optimization
- **Lab:** Whole-body motion
- **Project:** Multi-task execution

### Day 88: Human-Robot Interaction
- Safe collaboration
- Intent recognition
- Force limiting
- Natural interaction
- **Lab:** Human-aware navigation
- **Project:** Collaborative assembly

### Day 89: Teleoperation & Imitation
- VR teleoperation
- Motion capture integration
- Imitation learning from demos
- Autonomous handover
- **Lab:** Teleoperation interface
- **Project:** Demo-to-policy learning

### Day 90: Foundation Models for Humanoids
- NVIDIA Isaac GR00T
- Humanoid foundation models
- Language-guided actions
- Multi-modal interaction
- **Lab:** Foundation model integration
- **Project:** Language-controlled humanoid

### Day 91: Week 13 Review & Project (Humanoid Robot System)
- Project: Complete humanoid application
- Manipulation + Locomotion
- Human interaction
- **Popular Project:** Figure AI style robot (simulation)

---

## Week 14: Edge AI Deployment (Days 92-98)

### Day 92: NVIDIA Jetson Platform Deep Dive
- Jetson Orin family (Nano, NX, AGX)
- JetPack SDK
- Power modes and thermal management
- Linux for Tegra (L4T)
- **Lab:** Jetson development setup
- **Project:** Robot compute platform

### Day 93: TensorRT Optimization
- Model import and parsing
- Layer fusion and optimization
- INT8/FP16 quantization
- Engine serialization
- **Lab:** TensorRT model optimization
- **Project:** Real-time detection deployment

### Day 94: ONNX Runtime for Robotics
- ONNX model export
- Cross-framework compatibility
- Execution providers
- Dynamic shapes
- **Lab:** ONNX pipeline
- **Project:** Multi-model inference

### Day 95: DLA (Deep Learning Accelerator)
- DLA architecture
- Supported operations
- DLA vs GPU trade-offs
- Power efficiency
- **Lab:** DLA deployment
- **Project:** Low-power perception

### Day 96: PVA & Vision Pipeline
- Programmable Vision Accelerator
- Stereo and optical flow
- Custom vision kernels
- Hardware acceleration
- **Lab:** PVA stereo processing
- **Project:** Depth estimation accelerator

### Day 97: Jetson Thor for Humanoids
- Thor architecture overview
- 2,070 TFLOPS AI performance
- Multi-sensor processing
- Real-time control
- **Lab:** Thor capabilities
- **Project:** Advanced humanoid compute

### Day 98: Week 14 Review & Project (Edge AI Robot)
- Project: Complete edge AI system
- Multi-model deployment
- Real-time performance
- **Popular Project:** NVIDIA Isaac ROS deployment

---

## Week 15: Imitation Learning & Behavior Cloning (Days 99-105)

### Day 99: Behavior Cloning Fundamentals
- Expert demonstration collection
- State-action mapping
- Distribution shift problem
- Data augmentation
- **Lab:** Simple BC pipeline
- **Project:** Manipulation from demos

### Day 100: DAgger & Interactive Learning
- Dataset aggregation
- Expert intervention
- Distribution correction
- Safe exploration
- **Lab:** DAgger implementation
- **Project:** Improved policy learning

### Day 101: Inverse Reinforcement Learning (IRL)
- Reward learning from demonstrations
- Maximum entropy IRL
- Adversarial IRL (GAIL)
- Preference learning
- **Lab:** IRL reward recovery
- **Project:** Complex task learning

### Day 102: Action Chunking Transformer (ACT)
- Temporal action prediction
- Transformer for imitation
- Sequence modeling
- Multi-step action output
- **Lab:** ACT implementation
- **Project:** Fine manipulation learning

### Day 103: ALOHA & Mobile ALOHA
- Low-cost teleoperation
- Bimanual manipulation
- Mobile manipulation
- Data collection efficiency
- **Lab:** ALOHA-style demo collection
- **Project:** Bimanual task learning, **Popular Project:** ALOHA 2

### Day 104: Learning from Human Video
- Video demonstration learning
- Cross-embodiment transfer
- Motion retargeting
- Action abstraction
- **Lab:** Video-to-robot transfer
- **Project:** Human demo utilization

### Day 105: Week 15 Review & Project (Imitation Learning System)
- Project: Complete IL pipeline
- Data collection to deployment
- Multi-task generalization

---

## Week 16: Tactile Sensing & Contact Perception (Days 106-112)

### Day 106: Tactile Sensor Technologies
- Capacitive and resistive sensors
- GelSight vision-based tactile
- BioTac sensors
- DIGIT sensor
- **Lab:** Tactile sensor integration
- **Project:** Touch-enabled gripper

### Day 107: Tactile Image Processing
- Contact geometry estimation
- Force and slip detection
- Texture classification
- Object recognition from touch
- **Lab:** GelSight processing
- **Project:** Tactile-based grasping

### Day 108: Visuotactile Learning
- Combining vision and touch
- Multi-modal fusion
- Cross-modal prediction
- Tactile imagination
- **Lab:** Visuotactile dataset
- **Project:** Multi-modal manipulation

### Day 109: Contact-Rich Manipulation
- Assembly tasks
- Deformable object handling
- Insertion and fitting
- Force adaptation
- **Lab:** Contact-rich control
- **Project:** Precision assembly

### Day 110: In-Hand Manipulation with Tactile
- Object rotation and translation
- Continuous touch feedback
- Dexterous skills
- Tactile-guided planning
- **Lab:** In-hand rotation
- **Project:** Tool manipulation

### Day 111: Sim-to-Real for Tactile
- Tactile simulation
- Domain gap challenges
- Sensor noise modeling
- Transfer learning
- **Lab:** Tactile sim-to-real
- **Project:** Simulated tactile training

### Day 112: Week 16 Review & Project (Tactile Robot)
- Project: Complete tactile system
- Manipulation with sensing
- Adaptive grasping
- **Popular Project:** DIGIT sensor integration

---

## Week 17: Semantic Understanding & Scene Reasoning (Days 113-119)

### Day 113: 3D Semantic Segmentation
- Point cloud segmentation networks
- Cylinder3D, RandLA-Net
- Real-time 3D segmentation
- SemanticKITTI dataset
- **Lab:** 3D semantic segmentation
- **Project:** Scene understanding for navigation

### Day 114: Open-Vocabulary 3D Understanding
- CLIP for 3D scenes
- OpenScene and ConceptFusion
- Language-queryable 3D
- Zero-shot identification
- **Lab:** Open-vocab 3D segmentation
- **Project:** Natural language scene queries

### Day 115: Scene Graphs for Robotics
- 3D scene graph construction
- Object relationships
- Hierarchical representations
- Dynamic scene graphs
- **Lab:** Scene graph building
- **Project:** Semantic navigation planning

### Day 116: Affordance Detection
- Object affordances
- Interaction prediction
- Contact point detection
- Task-relevant features
- **Lab:** Affordance network
- **Project:** Affordance-guided grasping

### Day 117: Spatial Reasoning
- Object localization
- Spatial relationship understanding
- Reference frame handling
- Language-guided localization
- **Lab:** Spatial reasoning module
- **Project:** "Pick the object on the left"

### Day 118: Semantic Mapping & Memory
- Long-term semantic maps
- Object persistence
- Map updates
- Memory-augmented navigation
- **Lab:** Semantic memory system
- **Project:** Room-scale semantic map

### Day 119: Week 17 Review & Project (Semantic Robot)
- Project: Complete semantic system
- Understanding + Reasoning + Action
- Natural interaction
- **Popular Project:** ConceptFusion integration

---

## Week 18: Safety & Robustness (Days 120-126)

### Day 120: Safety-Critical Robot Systems
- Safety constraints
- Control barrier functions
- Safe reinforcement learning
- Backup controllers
- **Lab:** CBF implementation
- **Project:** Safe navigation

### Day 121: Adversarial Robustness
- Adversarial attacks on perception
- Patch attacks on detectors
- Defense mechanisms
- Certified robustness
- **Lab:** Attack and defense
- **Project:** Robust perception system

### Day 122: Out-of-Distribution Detection
- OOD detection methods
- Uncertainty estimation
- Anomaly detection
- Fallback behaviors
- **Lab:** OOD detector
- **Project:** Novel situation handling

### Day 123: Uncertainty Quantification
- Epistemic vs aleatoric
- Bayesian neural networks
- Ensemble methods
- Monte Carlo Dropout
- **Lab:** Uncertainty pipeline
- **Project:** Uncertainty-aware planning

### Day 124: Fail-Safe Design
- Degraded operation modes
- Minimal Risk Condition (MRC)
- Recovery strategies
- System monitoring
- **Lab:** Fail-safe controller
- **Project:** Graceful degradation

### Day 125: Safety Standards for Robots
- ISO 10218 industrial robots
- ISO/TS 15066 collaborative
- Functional safety (ISO 26262)
- Risk assessment
- **Lab:** Safety documentation
- **Project:** Compliant robot system

### Day 126: Week 18 Review & Project (Safe Robot)
- Project: Complete safety system
- Multi-layer protection
- Certified operation
- **Popular Project:** SafeDiffuser deployment

---

## Week 19: Simulation Environments (Days 127-133)

### Day 127: Isaac Sim Advanced Features
- PhysX 5 physics
- Replicator for synthetic data
- Domain randomization tools
- Sensor simulation
- **Lab:** Isaac Sim custom scene
- **Project:** Training data generation

### Day 128: MuJoCo Advanced Usage
- Custom model creation
- Contact parameter tuning
- Parallel simulation
- MuJoCo XLA integration
- **Lab:** Complex MuJoCo environment
- **Project:** Manipulation benchmark

### Day 129: Gazebo & Ignition
- Gazebo Harmonic features
- SDF model creation
- Plugin development
- ROS2 integration
- **Lab:** Gazebo custom world
- **Project:** Mobile robot simulation

### Day 130: Synthetic Data Generation
- Procedural scene generation
- Automatic annotation
- Domain gap mitigation
- Dataset scaling
- **Lab:** Synthetic dataset pipeline
- **Project:** Large-scale training data

### Day 131: Photo-Realistic Simulation
- Ray tracing for robotics
- Material and lighting
- Weather and time-of-day
- Sensor realism
- **Lab:** Photo-realistic rendering
- **Project:** Realistic test scenarios

### Day 132: Hardware-in-the-Loop (HIL)
- Real ECU with simulation
- Timing synchronization
- Interface bridging
- Validation testing
- **Lab:** HIL setup
- **Project:** ECU validation

### Day 133: Week 19 Review & Project (Complete Simulation)
- Project: Full simulation pipeline
- Data generation to training
- Sim-to-real validation
- **Popular Project:** Isaac Sim ROS2 pipeline

---

## Week 20: Mobile Manipulation (Days 134-140)

### Day 134: Mobile Manipulator Platforms
- MoMa design patterns
- Base + arm coordination
- Kinematic coupling
- Workspace analysis
- **Lab:** Mobile manipulator setup
- **Project:** Fetch-style robot

### Day 135: Whole-Body Motion Planning
- Combined base-arm planning
- Unified configuration space
- Prioritized control
- Singularity handling
- **Lab:** Whole-body planner
- **Project:** Picking from shelves

### Day 136: Navigation for Manipulation
- Pre-grasp positioning
- Manipulation-aware navigation
- Seamless transitions
- Dynamic replanning
- **Lab:** Integrated nav-manipulation
- **Project:** Mobile pick-and-place

### Day 137: Loco-Manipulation
- Legged robot manipulation
- Dynamic balance during manipulation
- Force-aware locomotion
- ANYmal manipulation
- **Lab:** Quadruped manipulation
- **Project:** Spot-style manipulation

### Day 138: Long-Horizon Mobile Manipulation
- Task and motion planning
- Hierarchical planning
- Semantic task understanding
- Multi-room operations
- **Lab:** Long-horizon planner
- **Project:** Multi-step task execution

### Day 139: Human-Following & Assistance
- Person detection and tracking
- Following behavior
- Collaborative carrying
- Assistance tasks
- **Lab:** Human-following robot
- **Project:** Shopping assistant

### Day 140: Week 20 Review & Project (Mobile Manipulation System)
- Project: Complete mobile manipulator
- Autonomous operation
- Multi-task capability
- **Popular Project:** Hello Robot Stretch integration

---

## Week 21: Data Collection & Dataset Engineering (Days 141-147)

### Day 141: Robot Data Collection Systems
- Teleoperation interfaces
- Autonomous data collection
- Multi-camera synchronization
- Annotation pipelines
- **Lab:** Data collection setup
- **Project:** Custom dataset creation

### Day 142: Open X-Embodiment Dataset
- RT-X style datasets
- Multi-robot data format
- Cross-embodiment learning
- Data standardization
- **Lab:** X-Embodiment contribution
- **Project:** Dataset formatting

### Day 143: Simulation Data Generation
- Automatic scene generation
- Task randomization
- Failure case injection
- Balanced dataset creation
- **Lab:** Automated data collection
- **Project:** Large-scale sim dataset

### Day 144: Data Augmentation for Robotics
- Geometric transformations
- Lighting and texture variation
- Background replacement
- Physics-based augmentation
- **Lab:** Augmentation pipeline
- **Project:** Data multiplication

### Day 145: Active Learning for DData
- Uncertainty-based sampling
- Diversity sampling
- Query-by-committee
- Efficient annotation
- **Lab:** Active learning system
- **Project:** Efficient dataset growth

### Day 146: Dataset Analysis & Quality
- Dataset statistics
- Bias detection
- Coverage analysis
- Quality metrics
- **Lab:** Dataset analysis tools
- **Project:** Dataset health report

### Day 147: Week 21 Review & Project (Dataset Engineering)
- Project: Production dataset pipeline
- Collection to training
- Version control
- **Popular Project:** LeRobot dataset contribution

---

## Week 22: Production Deployment (Days 148-154)

### Day 148: Robot Software Architecture
- Modular design patterns
- Message-driven architecture
- State machines
- Error handling
- **Lab:** Production architecture
- **Project:** Scalable robot software

### Day 149: CI/CD for Robotics
- Automated testing
- Simulation-based validation
- Deployment pipelines
- Rollback strategies
- **Lab:** CI/CD pipeline
- **Project:** Automated deployment

### Day 150: Fleet Management Systems
- Cloud infrastructure
- Edge computing
- Data pipelines
- Monitoring dashboards
- **Lab:** Fleet management setup
- **Project:** Multi-robot monitoring

### Day 151: OTA Updates & Remote Management
- Binary diff updates
- A/B partitioning
- Remote diagnostics
- Log aggregation
- **Lab:** OTA update system
- **Project:** Remote robot updates

### Day 152: Performance Monitoring
- Real-time telemetry
- Resource monitoring
- Anomaly detection
- SLA tracking
- **Lab:** Monitoring system
- **Project:** Production metrics

### Day 153: Debugging Production Robots
- Remote debugging tools
- Replay systems
- Root cause analysis
- Postmortem processes
- **Lab:** Debug workflow
- **Project:** Issue resolution system

### Day 154: Week 22 Review & Project (Production Robot)
- Project: Production-ready deployment
- Complete CI/CD
- Fleet management
- **Popular Project:** AWS IoT RoboMaker

---

## Week 23: Aerial Robotics & Drones (Days 155-161)

### Day 155: Drone Fundamentals & PX4
- Multicopter dynamics
- PX4 autopilot
- MAVLink protocol
- ROS2 integration
- **Lab:** PX4 SITL simulation
- **Project:** Basic drone control

### Day 156: Visual SLAM for Drones
- Lightweight SLAM
- Altitude estimation
- VIO for drones
- Loop closure in 3D
- **Lab:** Drone SLAM implementation
- **Project:** Indoor drone mapping

### Day 157: 3D Path Planning for UAVs
- 3D configuration space
- RRT for drones
- Trajectory optimization
- Dynamic constraints
- **Lab:** 3D planner
- **Project:** Cluttered environment navigation

### Day 158: Perception from Aerial Platforms
- Downward-looking cameras
- LiDAR for drones
- Inspection applications
- Mapping and survey
- **Lab:** Aerial perception
- **Project:** Infrastructure inspection

### Day 159: Multi-Drone Coordination
- Swarm formation
- Collision avoidance
- Distributed planning
- Communication protocols
- **Lab:** Multi-drone simulation
- **Project:** Coordinated survey

### Day 160: Drone Delivery & Manipulation
- Package delivery
- Aerial manipulation
- Precision landing
- Dynamic grasping
- **Lab:** Delivery simulation
- **Project:** Package handling drone

### Day 161: Week 23 Review & Project (Complete Drone System)
- Project: Autonomous inspection drone
- Navigation + Perception
- Mission planning
- **Popular Project:** PX4 + ROS2 delivery drone

---

## Week 24: Underwater & Field Robotics (Days 162-168)

### Day 162: Underwater Robot Platforms
- ROV and AUV systems
- BlueROV2 platform
- Underwater navigation
- Pressure and depth sensors
- **Lab:** Underwater simulation
- **Project:** BlueROV2 control

### Day 163: Underwater Perception
- Sonar imaging
- Acoustic positioning
- Low-visibility vision
- 3D underwater mapping
- **Lab:** Sonar processing
- **Project:** Underwater object detection

### Day 164: Agricultural Robots
- Row crop navigation
- Yield estimation
- Weed detection
- Precision spraying
- **Lab:** Agricultural perception
- **Project:** Crop monitoring robot

### Day 165: Construction & Mining Robots
- Rough terrain navigation
- Material handling
- Site mapping
- Autonomous excavation
- **Lab:** Construction simulation
- **Project:** Site survey robot

### Day 166: Search & Rescue Robots
- Disaster environments
- Victim detection
- Communication relay
- Multi-modal sensing
- **Lab:** SAR robot simulation
- **Project:** Search and rescue system

### Day 167: Space Robotics
- Planetary rovers
- Low-gravity manipulation
- Long-range navigation
- Autonomous operation
- **Lab:** Rover simulation
- **Project:** Mars rover navigation

### Day 168: Week 24 Review & Project (Field Robot)
- Project: Complete field robot
- Extreme environment operation
- Robust perception
- **Popular Project:** NASA-JPL rover prototype

---

## Week 25: Robot Learning at Scale (Days 169-175)

### Day 169: Large-Scale Robot Training
- Distributed training infrastructure
- Multi-GPU pipelines
- Data parallelism
- Model parallelism
- **Lab:** Distributed training setup
- **Project:** Scalable training system

### Day 170: Foundation Model Fine-Tuning
- LoRA and adapter methods
- Efficient fine-tuning
- Task-specific adaptation
- Resource optimization
- **Lab:** LoRA fine-tuning
- **Project:** Custom VLA adapter

### Day 171: Continuous Learning
- Lifelong robot learning
- Catastrophic forgetting mitigation
- Elastic weight consolidation
- Memory replay
- **Lab:** Continual learning pipeline
- **Project:** Incrementally learning robot

### Day 172: Multi-Task Learning
- Task interference handling
- Gradient manipulation
- Task scheduling
- Architecture search
- **Lab:** Multi-task optimization
- **Project:** General-purpose robot

### Day 173: Curriculum Learning
- Task difficulty ordering
- Automatic curriculum
- Progressive complexity
- Success-based scaling
- **Lab:** Curriculum design
- **Project:** Curriculum-trained robot

### Day 174: Federated Robot Learning
- Privacy-preserving learning
- Distributed data
- Model aggregation
- Heterogeneous robots
- **Lab:** Federated setup
- **Project:** Multi-site robot learning

### Day 175: Week 25 Review & Project (Scalable Learning)
- Project: Production learning system
- Large-scale training
- Continuous improvement
- **Popular Project:** Open X-Embodiment training

---

## Week 26: Human-Robot Collaboration (Days 176-182)

### Day 176: Collaborative Robot Safety
- ISO/TS 15066 compliance
- Speed and separation monitoring
- Power and force limiting
- Safety-rated monitored stop
- **Lab:** Cobot safety configuration
- **Project:** Safe collaborative cell

### Day 177: Intent Recognition
- Human motion prediction
- Gaze tracking
- Gesture recognition
- Activity recognition
- **Lab:** Intent prediction model
- **Project:** Anticipatory robot

### Day 178: Shared Autonomy
- Variable autonomy
- Human-robot teaming
- Blended control
- Trust calibration
- **Lab:** Shared control interface
- **Project:** Assisted manipulation

### Day 179: Natural Language Instruction
- Task grounding
- Ambiguity resolution
- Clarification dialogue
- Multi-turn interaction
- **Lab:** Language interface
- **Project:** Verbal task instruction

### Day 180: Ergonomic Assistance
- Heavy lifting assistance
- Exoskeleton integration
- Fatigue monitoring
- Adaptive force support
- **Lab:** Assistance system
- **Project:** Industrial assistance robot

### Day 181: Social Robots
- Emotion recognition
- Expressive behaviors
- Social navigation
- Human-like interaction
- **Lab:** Social behavior design
- **Project:** Reception robot

### Day 182: Week 26 Review & Project (Collaborative Robot)
- Project: Complete HRC system
- Safe collaboration
- Natural interaction
- **Popular Project:** Universal Robots + ROS2

---

## Week 27: Advanced Control & Dynamics (Days 183-189)

### Day 183: Contact-Implicit Control
- Contact dynamics modeling
- Hybrid system control
- Impact handling
- Robust manipulation
- **Lab:** Contact-implicit MPC
- **Project:** Dynamic manipulation

### Day 184: Model Predictive Path Integral (MPPI)
- Sampling-based MPC
- GPU-accelerated control
- Cost function design
- Real-time implementation
- **Lab:** MPPI controller
- **Project:** Aggressive navigation

### Day 185: Optimal Control for Robotics
- LQR and iLQR
- Differential dynamic programming
- Trajectory optimization
- Constraint handling
- **Lab:** Trajectory optimizer
- **Project:** Optimal motion planning

### Day 186: Learning-Based Control
- Neural network controllers
- Hybrid model-based learning
- Residual learning
- Adaptive control
- **Lab:** Learned controller
- **Project:** Adaptive manipulation

### Day 187: Soft Robot Control
- Continuum robot kinematics
- Deformable body dynamics
- Sensor integration
- Model-based control
- **Lab:** Soft robot simulation
- **Project:** Soft gripper control

### Day 188: Legged Robot Dynamics
- Floating base dynamics
- Contact scheduling
- Terrain adaptation
- Rough terrain locomotion
- **Lab:** Legged dynamics control
- **Project:** Quadruped controller

### Day 189: Week 27 Review & Project (Advanced Control)
- Project: Complete control system
- Dynamic motion
- Robust execution
- **Popular Project:** Quadruped parkour

---

## Week 28: Future Technologies (Days 190-196)

### Day 190: Neuromorphic Computing for Robotics
- Event-based cameras
- Spiking neural networks
- Low-latency perception
- Energy efficiency
- **Lab:** Event camera processing
- **Project:** Ultra-fast tracking

### Day 191: Quantum Computing Prospects
- Quantum algorithms for robotics
- Optimization problems
- Current limitations
- Near-term applications
- **Lab:** Quantum simulation
- **Project:** Quantum-inspired planning

### Day 192: Bio-Inspired Robotics
- Biological locomotion
- Adaptive morphology
- Evolutionary algorithms
- Natural intelligence
- **Lab:** Bio-inspired design
- **Project:** Swimming robot

### Day 193: Soft Actuators & Materials
- Artificial muscles
- Shape memory alloys
- Pneumatic soft robots
- Smart materials
- **Lab:** Soft actuator testing
- **Project:** Soft manipulation

### Day 194: Energy Harvesting & Autonomy
- Solar-powered robots
- Kinetic energy harvesting
- Long-duration autonomy
- Charging infrastructure
- **Lab:** Energy system design
- **Project:** Self-sustaining robot

### Day 195: Human-Level AI for Robotics
- AGI prospects
- Common sense reasoning
- Generalization challenges
- Open problems
- **Lab:** Capability analysis
- **Project:** Benchmark suite

### Day 196: Week 28 Review & Project (Future Robot)
- Project: Next-gen robot prototype
- Advanced technologies
- Innovation showcase

---

## Week 29: Capstone Project Part 1 (Days 197-203)

### Day 197: Project Planning & Requirements
- Project scope definition
- Requirements analysis
- Architecture design
- Timeline planning
- **Deliverable:** Project specification

### Day 198: System Architecture Design
- Module decomposition
- Interface design
- Communication patterns
- Technology selection
- **Deliverable:** Architecture document

### Day 199: Perception Stack Development
- Sensor integration
- Perception pipeline
- Model deployment
- Real-time processing
- **Deliverable:** Working perception

### Day 200: Planning & Control Implementation
- Path planning
- Motion control
- Behavior design
- Safety systems
- **Deliverable:** Planning module

### Day 201: Learning System Integration
- Policy deployment
- Fine-tuning pipeline
- Continuous learning
- Performance monitoring
- **Deliverable:** Learning system

### Day 202: System Integration
- Module integration
- Interface testing
- End-to-end validation
- Bug fixing
- **Deliverable:** Integrated system

### Day 203: Week 29 Review & Milestone
- Milestone: Working prototype
- Demo preparation
- Documentation update

---

## Week 30: Capstone Project Part 2 & Certification (Days 204-210)

### Day 204: Production Hardening
- Error handling
- Logging and monitoring
- Performance optimization
- Edge case handling
- **Deliverable:** Production-ready code

### Day 205: Testing & Validation
- Unit and integration tests
- Simulation validation
- Real-world testing
- Performance benchmarks
- **Deliverable:** Test report

### Day 206: Documentation
- Technical documentation
- User manual
- API reference
- Deployment guide
- **Deliverable:** Complete documentation

### Day 207: Demo Preparation
- Demo scenario design
- Video recording
- Presentation preparation
- Live demo setup
- **Deliverable:** Demo materials

### Day 208: Final Demo & Presentation
- Live demonstration
- Technical presentation
- Q&A session
- Feedback collection
- **Deliverable:** Final presentation

### Day 209: Portfolio & Career Preparation
- Project portfolio
- GitHub showcase
- Resume update
- Interview preparation
- **Deliverable:** Career materials

### Day 210: Certification & Next Steps
- Course completion
- Certification issuance
- Career pathways
- Research opportunities
- **Deliverable:** Certificate

---

## Assessment & Certification

### Weekly Projects (30 projects)
- Hands-on implementation
- Code review
- Performance metrics
- Documentation

### Capstone Project
- End-to-end robot system
- Real-world deployment
- Industry-standard quality
- Comprehensive documentation

### Certification Requirements
1. Complete all 210 days of content
2. Submit all 30 weekly projects
3. Pass capstone project review
4. Demonstrate proficiency in:
   - Foundation models for robotics
   - Advanced perception systems
   - Learning-based control
   - Multi-robot systems
   - Production deployment
   - Edge AI optimization

---

## Tools & Technologies

### Software
- ROS 2 Humble/Rolling
- Python 3.10+
- C++17/20
- PyTorch 2.x
- TensorFlow/JAX
- NVIDIA Isaac Sim/Lab
- MuJoCo
- Gazebo Harmonic
- TensorRT
- ONNX Runtime

### Hardware (Recommended)
- NVIDIA Jetson Orin AGX/NX
- Robotic arm (Franka, UR, xArm)
- Mobile base (TurtleBot4, Clearpath)
- LiDAR (Velodyne, Ouster, Livox)
- RGB-D cameras (RealSense, ZED)
- Force/torque sensors
- Tactile sensors (DIGIT, GelSight)

### Cloud & Compute
- AWS/GCP/Azure for training
- GPU clusters for large models
- Weights & Biases for experiment tracking
- Hugging Face Hub for models

---

## Datasets & Benchmarks

| Dataset | Purpose |
|---------|---------|
| nuScenes | Autonomous driving perception |
| KITTI | 3D object detection |
| SemanticKITTI | LiDAR segmentation |
| Open X-Embodiment | Multi-robot learning |
| RoboNet | Video prediction |
| BridgeData | Manipulation |
| DROID | Dexterous manipulation |

---

## Career Outcomes

Upon completing this program, you will be qualified for:

- **Robotics AI Engineer**
- **Perception Systems Engineer**
- **Robot Learning Scientist**
- **Foundation Models Researcher**
- **Autonomous Systems Architect**
- **Edge AI Specialist**
- **Multi-Robot Systems Engineer**

**Target Companies:**
- Tech Giants (Google DeepMind, Meta FAIR, NVIDIA, Tesla)
- Robotics Companies (Boston Dynamics, Figure AI, 1X Technologies)
- Autonomous Vehicles (Waymo, Cruise, Aurora, Zoox)
- Industrial Automation (ABB, FANUC, Universal Robots)
- Startups (Covariant, Physical Intelligence, Skild AI)
- Research Labs (CMU, MIT, Stanford, Berkeley)

---

## Next Steps After Completion

1. **Industry Deployment:** Work on production robot systems
2. **Research:** Publish at ICRA, IROS, CoRL, RSS, NeurIPS
3. **Specialization:** Focus on manipulation, navigation, or learning
4. **Entrepreneurship:** Start robotics company
5. **Open Source:** Contribute to ROS, Hugging Face LeRobot

---

**Phase 5: AI/CV/LIDAR End-to-End Robotics** | 210 Days to Mastering Modern Robot Intelligence
