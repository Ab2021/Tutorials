# 40-Day Computer Vision Mastery Course

## Course Overview
This comprehensive 40-day course covers Computer Vision from classical techniques to state-of-the-art deep learning approaches. Each day includes theoretical concepts, practical implementations, and interview preparation.

---

## Phase 1: CV Foundations (Days 1-10)

### Week 1: Image Basics & Classical CV
**Day 1: Introduction to CV & Image Fundamentals**
- What is Computer Vision? Applications & History
- Digital Images: Pixels, Channels, Color Spaces (RGB, HSV, LAB)
- Image Representations & Data Structures
- NumPy for Image Manipulation

**Day 2: Image Processing Operations**
- Filtering & Convolution Operations
- Gaussian Blur, Median Filter, Bilateral Filter
- Morphological Operations (Erosion, Dilation, Opening, Closing)
- Histogram Equalization & Normalization

**Day 3: Edge Detection & Feature Extraction**
- Gradient-Based Methods (Sobel, Prewitt, Canny)
- Corner Detection (Harris, Shi-Tomasi)
- Blob Detection (LoG, DoG)
- Image Pyramids & Scale Space

**Day 4: Classical CV (SIFT, SURF, HOG)**
- Scale-Invariant Feature Transform (SIFT)
- Speeded-Up Robust Features (SURF)
- Histogram of Oriented Gradients (HOG)
- Feature Matching & Descriptors

**Day 5: Image Segmentation Basics**
- Thresholding (Otsu, Adaptive)
- Region Growing & Watershed
- Graph-Based Segmentation
- Mean Shift & K-Means Clustering

### Week 2: Deep Learning Foundations
**Day 6: CNNs - Architecture Basics**
- From MLPs to CNNs: Motivation
- Convolutional Layers: Local Connectivity & Weight Sharing
- Receptive Fields & Feature Maps
- LeNet-5: The First CNN

**Day 7: CNN Components**
- Convolution Operation (stride, padding, dilation)
- Pooling Layers (Max, Average, Global)
- Batch Normalization & Layer Normalization
- Activation Functions (ReLU, LeakyReLU, GELU)

**Day 8: Training CNNs**
- Loss Functions (Cross-Entropy, Focal Loss)
- Optimization (SGD, Adam, AdamW)
- Regularization (Dropout, Weight Decay, Data Augmentation)
- Learning Rate Schedules

**Day 9: Transfer Learning & Fine-Tuning**
- ImageNet & Pretrained Models
- Feature Extraction vs Fine-Tuning
- Domain Adaptation Strategies
- Progressive Unfreezing

**Day 10: Phase 1 Review & Mini-Project**
- Classical vs Deep Learning CV
- Building an Image Classifier from Scratch
- Debugging CNN Training
- Project: CIFAR-10 Classification

---

## Phase 2: Image Classification & Recognition (Days 11-20)

### Week 3: Classic Architectures
**Day 11: AlexNet & VGGNet**
- AlexNet (2012): Breaking Through with Deep Learning
- VGGNet: Depth Matters (VGG16, VGG19)
- Design Principles & Architectural Choices
- Implementation & Analysis

**Day 12: ResNet & Skip Connections**
- Vanishing Gradients Problem
- Residual Learning & Identity Mappings
- ResNet Variants (ResNet-50, ResNet-101, ResNeXt)
- Wide ResNets & ResNets in Practice

**Day 13: Inception & Network in Network**
- Network in Network (NiN) & 1x1 Convolutions
- Inception Modules: Multi-Scale Features
- GoogLeNet/Inception v1-v4
- Xception: Depthwise Separable Convolutions

**Day 14: MobileNets & EfficientNets**
- Efficient Architectures for Mobile/Edge
- MobileNetV1, V2, V3
- Neural Architecture Search (NAS)
- EfficientNet: Compound Scaling

**Day 15: Vision Transformers (ViT)**
- Attention is All You Need (Applied to Vision)
- Patch Embeddings & Positional Encoding
- ViT Architecture & Variants
- DeiT, Swin Transformer, BEiT

### Week 4: Advanced Recognition
**Day 16: Attention Mechanisms in Vision**
- Spatial Attention & Channel Attention
- Squeeze-and-Excitation Networks (SENet)
- CBAM: Convolutional Block Attention Module
- Self-Attention in CNNs

**Day 17: Self-Supervised Learning**
- Contrastive Learning (SimCLR, MoCo, SwAV)
- Momentum Encoders & Memory Banks
- BYOL: Bootstrap Your Own Latent
- MAE: Masked Autoencoders

**Day 18: Few-Shot Learning & Meta-Learning**
- Problem Formulation (N-way K-shot)
- Metric Learning (Prototypical Networks, Matching Networks)
- Meta-Learning (MAML)
- Transductive vs Inductive Few-Shot

**Day 19: Domain Adaptation & Transfer**
- Domain Shift Problem
- Unsupervised Domain Adaptation
- Adversarial Domain Adaptation
- Zero-Shot Learning

**Day 20: Phase 2 Review & Project**
- Architecture Evolution Timeline
- When to Use Which Architecture?
- Comparing Top-1 Accuracy vs Efficiency
- Project: Fine-Tune ViT on Custom Dataset

---

## Phase 3: Object Detection & Segmentation (Days 21-30)

### Week 5: Object Detection
**Day 21: R-CNN Family**
- R-CNN: Regions with CNN Features
- Fast R-CNN: ROI Pooling & End-to-End Training
- Faster R-CNN: Region Proposal Networks (RPN)
- Feature Pyramid Networks (FPN)

**Day 22: YOLO (v1-v8)**
- YOLO v1: You Only Look Once
- YOLO v2-v3: Improvements & Darknet
- YOLO v4-v5: CSPNet, PANet, Focus
- YOLO v6-v8: Latest Advances

**Day 23: SSD & RetinaNet**
- Single Shot MultiBox Detector (SSD)
- Multi-Scale Feature Maps
- RetinaNet & Focal Loss
- Anchor Design & Matching

**Day 24: EfficientDet & Modern Detectors**
- EfficientDet: BiFPN & Compound Scaling
- CenterNet: Keypoint-Based Detection
- FCOS: Fully Convolutional One-Stage
- Anchor-Free Detectors

**Day 25: Detection Transformers (DETR)**
- End-to-End Detection with Transformers
- Bipartite Matching & Hungarian Algorithm
- Deformable DETR
- DINO: DETR with Improved Denoising

### Week 6: Segmentation
**Day 26: Semantic Segmentation**
- Fully Convolutional Networks (FCN)
- U-Net: Encoder-Decoder Architecture
- DeepLab v1-v3+ (Atrous Convolution, ASPP)
- PSPNet: Pyramid Scene Parsing

**Day 27: Instance Segmentation**
- Mask R-CNN: Extending Faster R-CNN
- Cascade Mask R-CNN
- YOLACT: Real-Time Instance Segmentation
- SOLOv2: Segmenting Objects by Locations

**Day 28: Panoptic Segmentation**
- Unified Semantic + Instance Segmentation
- Panoptic FPN
- Axial-DeepLab
- MaskFormer & Mask2Former

**Day 29: SAM (Segment Anything Model)**
- Foundation Model for Segmentation
- Promptable Segmentation
- Zero-Shot Generalization
- SAM Architecture & Training

**Day 30: Phase 3 Review & Project**
- Detection vs Segmentation Trade-offs
- Evaluation Metrics (mAP, IoU, Dice)
- Real-Time vs Accuracy
- Project: Build a Custom Object Detector

---

## Phase 4: Generative Models & Advanced Topics (Days 31-40)

### Week 7: Generative Models
**Day 31: GANs - Fundamentals**
- Generative Adversarial Networks Theory
- Generator & Discriminator Design
- Training GANs: Nash Equilibrium
- Mode Collapse & Convergence Issues

**Day 32: Advanced GANs**
- DCGAN: Deep Convolutional GANs
- Progressive GAN & StyleGAN v1-v3
- BigGAN: Large-Scale Image Synthesis
- Conditional GANs (cGAN, Pix2Pix, CycleGAN)

**Day 33: Diffusion Models**
- Denoising Diffusion Probabilistic Models (DDPM)
- Forward & Reverse Diffusion Process
- Score-Based Models
- Stable Diffusion & Latent Diffusion

**Day 34: VAEs & Autoregressive Models**
- Variational Autoencoders (VAE)
- VQ-VAE & VQ-VAE-2
- PixelCNN & PixelSNAIL
- Autoregressive Transformers (DALL-E 1)

**Day 35: Text-to-Image Generation**
- CLIP: Connecting Text & Vision
- DALL-E 2: Combining CLIP + Diffusion
- Imagen: Text-to-Image with Large LMs
- Midjourney & Commercial Applications

### Week 8: Frontiers & Applications
**Day 36: 3D Vision & NeRF**
- 3D Representations (Voxels, Meshes, Point Clouds)
- Structure from Motion (SfM)
- Neural Radiance Fields (NeRF)
- Instant-NGP & 3D Gaussian Splattering

**Day 37: Video Understanding**
- 3D CNNs (C3D, I3D)
- Two-Stream Networks
- Temporal Attention & Video Transformers
- Action Recognition & Video Captioning

**Day 38: Multimodal Learning**
- CLIP: Contrastive Language-Image Pretraining
- ALIGN: Scaling Up Multimodal Learning
- Florence & BLIP
- Flamingo: Few Shot Multimodal Learning

**Day 39: Real-World Applications & Deployment**
- Medical Imaging (X-Ray, MRI, CT Analysis)
- Autonomous Driving (Perception Stack)
- Face Recognition & Verification
- OCR & Document Understanding
- Model Optimization (Quantization, Pruning, distillation)

**Day 40: Capstone Project**
- End-to-End CV System Design
- Project Options (Detection, Segmentation, Generation)
- Deployment & Production Considerations
- Future of Computer Vision

---

## Learning Structure (Per Day)
Each day includes:
1. **Core Concept (XX.md):** Main theory, architecture details, equations
2. **Deep Dive (XX_part1.md):** Implementation details, variants, ablations
3. **Interview Prep (XX_interview.md):** Common questions, problem-solving

## Prerequisites
- Python programming
- Linear algebra & calculus basics
- Deep learning fundamentals
- PyTorch or TensorFlow experience

## Resources
- **Textbooks:** Deep Learning (Goodfellow), Computer Vision: Algorithms and Applications (Szeliski)
- **Courses:** Stanford CS231n, Fast.ai
- **Papers:** All major architecture papers provided
- **Code:** PyTorch implementations with detailed comments

---

**Ready to master Computer Vision? Let's begin!** 🚀👁️
