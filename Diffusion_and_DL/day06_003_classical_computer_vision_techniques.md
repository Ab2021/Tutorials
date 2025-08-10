# Day 6 - Part 3: Classical Computer Vision Techniques and Mathematical Analysis

## 📚 Learning Objectives
By the end of this section, you will understand:
- Image segmentation algorithms and their mathematical foundations
- Template matching theory and correlation-based methods
- Object detection and recognition using classical approaches
- Geometric transformations and image registration techniques
- Stereo vision and 3D reconstruction mathematical principles
- Optical flow computation and motion analysis theory

---

## 🎯 Image Segmentation Theory

### Region-Based Segmentation

#### Region Growing Algorithms
**Mathematical Foundation**:
```
Region Growing Process:
R = {(x,y) : |I(x,y) - μ_R| < T}

Where:
- R: Current region
- μ_R: Region mean intensity
- T: Threshold parameter
- I(x,y): Image intensity at (x,y)

Adaptive Threshold:
T(x,y) = k × σ_local(x,y)
where σ_local is local standard deviation

Connectivity Criteria:
4-connectivity: N₄ = {(x±1,y), (x,y±1)}
8-connectivity: N₈ = N₄ ∪ {(x±1,y±1)}

Homogeneity Predicate:
P(R) = true if Var(R) < threshold
Complex predicates can include:
- Mean intensity difference
- Texture measures
- Color similarity
```

**Split-and-Merge Algorithm**:
```
Quadtree Representation:
Recursive subdivision into 4 quadrants
Each node represents image region

Split Criterion:
Split region R if P(R) = false
where P is homogeneity predicate

Merge Criterion:
Merge adjacent regions Rᵢ, Rⱼ if P(Rᵢ ∪ Rⱼ) = true

Algorithm:
1. Start with entire image
2. Split inhomogeneous regions
3. Merge similar adjacent regions
4. Repeat until convergence

Mathematical Properties:
- Convergence guaranteed for monotonic predicates
- Final segmentation depends on split/merge order
- Computational complexity: O(n log n)
```

#### Watershed Segmentation
**Mathematical Morphology Foundation**:
```
Topographic Interpretation:
I(x,y) = altitude at position (x,y)
Watershed = drainage divide between catchment basins

Gradient-Based Approach:
W(x,y) = ||∇I(x,y)||
Use gradient magnitude as topographic surface

Watershed Algorithm:
1. Find regional minima (markers)
2. Simulate flooding from markers
3. Build dams where watersheds meet
4. Result: Segmented regions

Mathematical Definition:
Watershed of function f:
WS(f) = {x : ∀ε>0, ∃ distinct minima mᵢ,mⱼ 
         such that B(x,ε) intersects watersheds to both mᵢ,mⱼ}
```

**Marker-Controlled Watershed**:
```
Over-segmentation Problem:
Pure watershed often produces too many regions
Solution: Use markers to control segmentation

Marker Selection:
Internal markers: Object seeds (local minima)
External markers: Background seeds

Modified Gradient:
G'(x,y) = {0 if (x,y) ∈ markers
          {G(x,y) otherwise

Advantages:
- Reduces over-segmentation
- Incorporates prior knowledge
- Controllable segmentation granularity

Mathematical Properties:
- Markers impose topological constraints
- Results in connected regions
- Computational complexity: O(n log n)
```

### Edge-Based Segmentation

#### Active Contours (Snakes)
**Energy Functional**:
```
Snake Energy:
E = ∫₀¹ [Eₑₙₜ(v(s)) + Eᵢₘₐ(v(s)) + Eₑₒₙ(v(s))] ds

Where v(s) = (x(s), y(s)) is parametric contour

Internal Energy (Smoothness):
Eₑₙₜ = ½[α(s)|v'(s)|² + β(s)|v''(s)|²]
- α(s): Tension (first derivative term)
- β(s): Rigidity (second derivative term)

Image Energy (Data Term):
Eᵢₘₐ = -γ|∇I(v(s))|²
Attracts snake to edges

External Energy (Constraints):
Eₑₒₙ: User-defined constraints (points, lines)

Euler-Lagrange Equation:
αv''(s) - βv''''(s) - ∇Eₑₓₜ = 0
```

**Geometric Active Contours (Level Sets)**:
```
Level Set Representation:
C = {(x,y) : φ(x,y) = 0}
Contour as zero level set of function φ

Evolution Equation:
∂φ/∂t = F|∇φ|

Where F = speed function:
F = (1 - ε·κ) + ν·g(I)

Terms:
- κ: Curvature (smoothness)
- g(I): Image-dependent stopping function
- ε, ν: Weighting parameters

Advantages:
- Handles topology changes naturally
- No parametrization required
- Stable numerical implementation

Chan-Vese Model:
Energy functional based on region statistics
Minimizes variance within regions
Handles images without clear edges
```

### Clustering-Based Segmentation

#### K-Means Clustering
**Mathematical Formulation**:
```
Objective Function:
J = Σᵢ₌₁ⁿ Σⱼ₌₁ᵏ wᵢⱼ ||xᵢ - μⱼ||²

Where:
- wᵢⱼ = 1 if point i assigned to cluster j, 0 otherwise
- μⱼ: Centroid of cluster j
- k: Number of clusters

K-Means Algorithm:
1. Initialize k centroids randomly
2. Assign points to nearest centroid
3. Update centroids: μⱼ = (1/nⱼ) Σᵢ∈Cⱼ xᵢ
4. Repeat until convergence

Convergence Properties:
- Objective function decreases monotonically
- Guaranteed to converge to local minimum
- Result depends on initialization

Feature Vectors for Images:
- Intensity: x = [I(x,y)]
- Color: x = [R(x,y), G(x,y), B(x,y)]
- Texture: x = [I(x,y), texture_features]
- Spatial: x = [I(x,y), x, y]
```

#### Mean Shift Clustering
**Mathematical Theory**:
```
Kernel Density Estimation:
f(x) = (1/n) Σᵢ₌₁ⁿ K((x - xᵢ)/h)

Mean Shift Vector:
m(x) = [Σᵢ xᵢ g(||x - xᵢ||²/h²)] / [Σᵢ g(||x - xᵢ||²/h²)] - x

Where g(x) = -K'(x) (derivative of kernel)

Gaussian Kernel:
K(x) = exp(-½||x||²)
g(x) = exp(-½||x||²)

Mean Shift Algorithm:
1. For each point, compute mean shift vector
2. Shift point by mean shift vector
3. Repeat until convergence
4. Points converging to same mode form cluster

Properties:
- Non-parametric (no assumption on cluster number)
- Finds modes of density function
- Robust to outliers
- Bandwidth h controls cluster granularity
```

**Application to Image Segmentation**:
```
Feature Space:
Combine spatial and color information:
x = [r, g, b, x, y]

Bandwidth Selection:
- Spatial bandwidth: hₛ
- Color bandwidth: hᵣ
- Controls segmentation granularity

Joint Domain:
Distance metric: d² = ||xₛ - yₛ||²/hₛ² + ||xᵣ - yᵣ||²/hᵣ²

Advantages:
- Preserves spatial connectivity
- Handles irregular cluster shapes
- No need to specify cluster number
- Robust segmentation results
```

---

## 📐 Template Matching and Correlation

### Cross-Correlation Theory

#### Normalized Cross-Correlation (NCC)
**Mathematical Definition**:
```
Cross-Correlation:
C(u,v) = Σₓ Σᵧ I(x,y) × T(x-u, y-v)

Normalized Cross-Correlation:
NCC(u,v) = Σₓ Σᵧ [I(x,y) - Ī][T(x-u,y-v) - T̄] / 
           √(Σₓ Σᵧ [I(x,y) - Ī]² × Σₓ Σᵧ [T(x-u,y-v) - T̄]²)

Where:
- I: Image
- T: Template
- Ī, T̄: Mean values
- (u,v): Template position

Properties:
- NCC ∈ [-1, 1]
- NCC = 1: Perfect match
- NCC = -1: Perfect negative match
- NCC = 0: No correlation
```

**Fast Computation**:
```
FFT-Based Correlation:
C = IFFT(FFT(I) × FFT*(T))
where * denotes complex conjugate

Computational Complexity:
Direct: O(MN × mn) for M×N image, m×n template
FFT: O(MN log(MN))
Speedup significant for large templates

Sliding Window Efficiency:
Use integral images for sum computation
Update correlation incrementally
Especially efficient for rectangular templates
```

#### Phase Correlation
**Mathematical Foundation**:
```
Phase Correlation Function:
R(u,v) = IFFT(F₁(ω₁,ω₂) × F₂*(ω₁,ω₂) / |F₁(ω₁,ω₂) × F₂*(ω₁,ω₂)|)

Where F₁, F₂ are FFTs of images I₁, I₂

Translation Estimation:
If I₂(x,y) = I₁(x-x₀, y-y₀) then
R(u,v) = δ(u-x₀, v-y₀)
Peak location gives translation

Advantages:
- Robust to illumination changes
- Subpixel accuracy
- Efficient FFT implementation
- Works for pure translation

Limitations:
- Assumes pure translation
- Sensitive to noise
- Requires similar content
```

### Multi-Scale Template Matching

#### Scale-Invariant Matching
**Pyramid-Based Approach**:
```
Image Pyramid:
I₀ = Original image
Iₙ = Downsample(Iₙ₋₁) for n = 1,2,...,L

Scale Factor:
Typically 2:1 downsampling ratio
σ levels: I_σ = Gaussian_blur(I, σ) then downsample

Template Pyramid:
Create templates at multiple scales
Match template Tₛ with image Iₛ at each scale

Coarse-to-Fine Search:
1. Start matching at coarsest level
2. Refine match location at finer levels
3. Propagate and refine estimates

Computational Benefits:
Reduces search space exponentially
Total computation ≈ 4/3 × single-scale cost
```

**Deformable Template Matching**:
```
Affine Template Model:
T'(x,y) = T(ax + by + c, dx + ey + f)
6 parameters: [a,b,c,d,e,f]

Optimization:
Minimize: E = Σₓ Σᵧ [I(x,y) - T'(x,y)]²
Use gradient descent or Levenberg-Marquardt

Lucas-Kanade Template Tracking:
Linearize around current estimate
Solve for parameter updates iteratively
Efficient for small deformations

Jacobian Matrix:
J = ∂T/∂p where p = parameter vector
Update: Δp = (JᵀJ)⁻¹Jᵀ[I - T]
```

---

## 🎯 Classical Object Detection and Recognition

### Sliding Window Detection

#### Multi-Scale Detection Framework
**Detection Pipeline**:
```
Sliding Window Process:
1. Generate windows at multiple scales
2. Extract features from each window
3. Classify window (object vs. background)
4. Apply non-maximum suppression

Scale Generation:
Sₙ = S₀ × rⁿ where r = scale ratio
Common: r = 1.2 (20% scale increase)

Window Density:
Step size: s pixels
Total windows ≈ (W×H×S) / s²
where W×H = image size, S = number of scales

Classification:
Binary classifier: f(x) → {-1, +1}
Confidence score: P(object | features)
```

**Non-Maximum Suppression (NMS)**:
```
Overlap Computation:
IoU = Area(bbox₁ ∩ bbox₂) / Area(bbox₁ ∪ bbox₂)

NMS Algorithm:
1. Sort detections by confidence score
2. Select highest scoring detection
3. Remove overlapping detections (IoU > threshold)
4. Repeat with remaining detections

Soft NMS:
Instead of removing, reduce confidence:
sᵢ = sᵢ × exp(-IoU²/σ²)
Preserves nearby detections with lower confidence

Mathematical Properties:
- Greedy algorithm (may not be globally optimal)
- Threshold selection affects precision/recall trade-off
- Computational complexity: O(n²)
```

#### Viola-Jones Face Detection
**Haar Feature Theory**:
```
Haar-like Features:
Two-rectangle: Σ white - Σ black
Three-rectangle: Σ outer - 2×Σ middle
Four-rectangle: Σ diagonal_opposite

Integral Image:
II(x,y) = Σᵢ≤ₓ Σⱼ≤ᵧ I(i,j)

Fast Rectangle Sum:
Sum(x₁,y₁,x₂,y₂) = II(x₂,y₂) - II(x₁-1,y₂) - II(x₂,y₁-1) + II(x₁-1,y₁-1)

Feature Computation:
Any rectangular sum in O(1) time
Enables real-time feature extraction
Large feature pool: >100,000 features for 24×24 window
```

**AdaBoost Training**:
```
Weak Classifier:
hⱼ(x) = {1 if pⱼfⱼ(x) < pⱼθⱼ
        {0 otherwise

Where:
- fⱼ: feature value
- θⱼ: threshold
- pⱼ: polarity (±1)

AdaBoost Algorithm:
1. Initialize weights: wᵢ = 1/(2m), 1/(2l)
2. For t = 1,...,T:
   a. Train weak classifier hₜ
   b. Choose hₜ with minimum weighted error
   c. Update weights based on classification results
3. Final classifier: H(x) = sign(Σₜ αₜhₜ(x))

Weight Update:
wᵢ = wᵢ × exp(-αₜyᵢhₜ(xᵢ))
where αₜ = ½ln((1-εₜ)/εₜ)
```

**Cascade Architecture**:
```
Cascade Design:
Series of classifiers with increasing complexity
Early stages: Simple, fast (eliminate easy negatives)
Later stages: Complex, accurate (refine decisions)

Rejection Cascade:
Stage 1: If H₁(x) < θ₁, reject (background)
Stage 2: If H₂(x) < θ₂, reject
...
Stage n: Final decision

Detection Rate:
D = ∏ᵢ dᵢ (product of individual stage detection rates)

False Positive Rate:
F = ∏ᵢ fᵢ (product of individual stage false positive rates)

Training Strategy:
Set target performance for each stage
Train to achieve desired d and f values
Adjust thresholds for optimal cascade performance
```

### Bag-of-Words (BoW) Model

#### Visual Vocabulary Construction
**Clustering for Vocabulary**:
```
Feature Extraction:
1. Detect keypoints in training images
2. Extract local descriptors (SIFT, SURF, etc.)
3. Collect large descriptor dataset

K-Means Clustering:
Minimize: J = Σᵢ₌₁ⁿ ||xᵢ - μₒ₍ᵢ₎||²
where o(i) = assigned cluster for descriptor i

Vocabulary Size:
Typical: k = 1000-10000 visual words
Trade-off: Small k (underfitting), Large k (overfitting)

Alternative Clustering:
- Hierarchical k-means (vocabulary trees)
- Gaussian Mixture Models
- Online learning approaches
```

**Spatial Pyramid Matching**:
```
Pyramid Levels:
Level 0: Entire image (1 region)
Level 1: 2×2 grid (4 regions)
Level 2: 4×4 grid (16 regions)

Histogram Computation:
H^l_i = histogram of visual words in region i at level l

Intersection Kernel:
K(H^l, G^l) = Σᵢ min(H^l_i, G^l_i)

Weighted Combination:
K = Σₗ wₗ × K(H^l, G^l)
where wₗ = 1/2^(L-l) (higher weight for finer levels)

Properties:
- Captures spatial layout information
- Maintains computational efficiency
- Provides multi-scale spatial representation
```

#### Classification and Retrieval
**SVM Classification**:
```
Linear SVM:
Minimize: ½||w||² + C Σᵢ ξᵢ
Subject to: yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

Histogram Intersection Kernel:
K(h,g) = Σᵢ min(hᵢ, gᵢ)

Chi-Square Kernel:
K(h,g) = exp(-γ Σᵢ (hᵢ - gᵢ)²/(hᵢ + gᵢ))

Multi-class Classification:
One-vs-all: Train binary classifier for each class
One-vs-one: Train classifier for each class pair
Error-correcting codes: Robust multi-class approach
```

**Image Retrieval**:
```
Similarity Measures:
Euclidean: d(h,g) = ||h - g||₂
Cosine: sim(h,g) = hᵀg/(||h|| ||g||)
Histogram intersection: sim(h,g) = Σᵢ min(hᵢ,gᵢ)

TF-IDF Weighting:
Term Frequency: tf(w,d) = frequency of word w in document d
Inverse Document Frequency: idf(w) = log(N/df(w))
Weight: w = tf × idf

Inverted Index:
For each visual word, maintain list of images containing it
Enables efficient retrieval for large databases
Query processing: Intersect posting lists
```

---

## 📏 Geometric Transformations and Registration

### 2D Transformations

#### Homogeneous Coordinates
**Transformation Matrices**:
```
Homogeneous Representation:
[x'] = [a b c] [x]
[y']   [d e f] [y]
[1 ]   [0 0 1] [1]

Translation:
T = [1 0 tₓ]
    [0 1 tᵧ]
    [0 0 1 ]

Rotation:
R = [cos θ  -sin θ  0]
    [sin θ   cos θ  0]
    [0       0      1]

Scaling:
S = [sₓ  0   0]
    [0   sᵧ  0]
    [0   0   1]

Composite Transform:
M = T × R × S (order matters!)
```

**Affine Transformations**:
```
General Affine:
[x'] = [a b] [x] + [tₓ]
[y']   [c d] [y]   [tᵧ]

Properties:
- Preserves parallelism
- Preserves ratios of parallel lengths
- Maps lines to lines
- 6 degrees of freedom

Parameter Estimation:
Minimum 3 point correspondences
Linear system: Ax = b
Least squares solution: x = (AᵀA)⁻¹Aᵀb

Decomposition:
A = R × S × H
where R = rotation, S = scaling, H = shear
```

#### Perspective Transformations
**Homography Estimation**:
```
Perspective Transformation:
[x'] = [h₁₁ h₁₂ h₁₃] [x]
[y']   [h₂₁ h₂₂ h₂₃] [y]
[w ]   [h₃₁ h₃₂ h₃₃] [1]

x' = (h₁₁x + h₁₂y + h₁₃)/(h₃₁x + h₃₂y + h₃₃)
y' = (h₂₁x + h₂₂y + h₂₃)/(h₃₁x + h₃₂y + h₃₃)

Direct Linear Transform (DLT):
Linearize: xᵢ'(h₃₁xᵢ + h₃₂yᵢ + h₃₃) = h₁₁xᵢ + h₁₂yᵢ + h₁₃
          yᵢ'(h₃₁xᵢ + h₃₂yᵢ + h₃₃) = h₂₁xᵢ + h₂₂yᵢ + h₂₃

Matrix Form: Ah = 0
SVD Solution: h = V_min (smallest singular vector)
Minimum: 4 point correspondences
```

**Robust Estimation**:
```
RANSAC for Homography:
1. Sample 4 point pairs randomly
2. Compute homography H
3. Count inliers: ||H·xᵢ - x'ᵢ|| < threshold
4. Repeat N iterations
5. Select H with most inliers

Refinement:
Use all inliers for final estimation
Non-linear optimization (Levenberg-Marquardt)
Minimize reprojection error: Σᵢ ||H·xᵢ - x'ᵢ||²

Degeneracy Cases:
- Collinear points: Infinite solutions
- Points on conic: Unstable estimation
- Planar scene: Reduced constraints
Detection and handling of degenerate configurations
```

### Image Registration

#### Feature-Based Registration
**Registration Pipeline**:
```
1. Feature Detection:
   Extract keypoints in both images
   Common detectors: SIFT, SURF, ORB

2. Feature Matching:
   Compute descriptor similarities
   Apply ratio test and cross-check

3. Transformation Estimation:
   Use RANSAC for robust estimation
   Model: Affine, homography, or similarity

4. Image Warping:
   Apply transformation to align images
   Interpolation for sub-pixel accuracy

Evaluation Metrics:
- Registration accuracy: RMS error
- Robustness: Success rate
- Efficiency: Computational time
```

**Intensity-Based Registration**:
```
Similarity Measures:
Sum of Squared Differences (SSD):
SSD = Σₓ,ᵧ [I₁(x,y) - I₂(x,y)]²

Normalized Cross Correlation (NCC):
NCC = Σₓ,ᵧ [I₁(x,y) - μ₁][I₂(x,y) - μ₂] / (σ₁σ₂N)

Mutual Information (MI):
MI = Σᵢ,ⱼ p(i,j) log(p(i,j)/(pᵢ(i)pⱼ(j)))

Optimization:
Gradient descent on similarity function
Powell's method for derivative-free optimization
Multi-resolution for global optimization
```

#### Non-Rigid Registration
**Thin Plate Splines (TPS)**:
```
TPS Interpolation:
f(x,y) = a₁ + a₂x + a₃y + Σᵢ wᵢ U(||(x,y) - (xᵢ,yᵢ)||)

Where U(r) = r² log r (radial basis function)

Constraints:
Σᵢ wᵢ = 0
Σᵢ wᵢxᵢ = 0
Σᵢ wᵢyᵢ = 0

Bending Energy:
E = ∫∫ [(∂²f/∂x²)² + 2(∂²f/∂x∂y)² + (∂²f/∂y²)²] dx dy

Regularized Solution:
Minimize: Σᵢ ||f(xᵢ,yᵢ) - yᵢ||² + λE
λ controls smoothness vs. fitting accuracy
```

**Optical Flow Registration**:
```
Lucas-Kanade Method:
Assume: I(x,y,t) = I(x+dx,y+dy,t+dt)

Brightness Constancy:
∂I/∂x · dx/dt + ∂I/∂y · dy/dt + ∂I/∂t = 0

Optical Flow Equation:
Iₓu + Iᵧv + Iₜ = 0
where u = dx/dt, v = dy/dt

Local Solution:
Assume constant flow in neighborhood
Least squares: [u v]ᵀ = -(AᵀA)⁻¹Aᵀb

Pyramid Implementation:
Coarse-to-fine estimation
Handle large displacements
Iterative refinement
```

---

## 👁️ Stereo Vision and 3D Reconstruction

### Epipolar Geometry

#### Fundamental Matrix
**Mathematical Foundation**:
```
Epipolar Constraint:
x'ᵀFx = 0

Where:
- x, x': Corresponding points in homogeneous coordinates
- F: Fundamental matrix (3×3)

Geometric Interpretation:
- Fx: Epipolar line in second image
- Fᵀx': Epipolar line in first image
- e, e': Epipoles (Fe = 0, Fᵀe' = 0)

Properties:
- rank(F) = 2 (singular matrix)
- 7 degrees of freedom
- Skew-symmetric part encodes epipolar geometry

Eight-Point Algorithm:
Linear system: Af = 0
where A constructed from point correspondences
SVD solution with rank-2 constraint enforcement
```

**Essential Matrix**:
```
Calibrated Cameras:
E = K'ᵀFK

Where K, K' are camera calibration matrices

Essential Matrix Properties:
E = [t]ₓR
where [t]ₓ is skew-symmetric matrix, R is rotation

Constraints:
- 2EEᵀE - trace(EEᵀ)E = 0
- det(E) = 0
- 5 degrees of freedom

Five-Point Algorithm:
Minimal solver for essential matrix
Handles calibrated camera case
More stable than fundamental matrix estimation
```

#### Stereo Rectification
**Rectification Process**:
```
Goal: Transform images so epipolar lines are horizontal

Rectification Matrices:
H₁, H₂ such that:
- F_rect = [0 0 0; 0 0 -1; 0 1 0]
- Epipolar lines become horizontal
- Minimize image distortion

Algorithm:
1. Compute fundamental matrix F
2. Find epipoles e, e'
3. Construct rectification homographies
4. Warp images to canonical form

Uncalibrated Rectification:
Use projective transformations
Preserve cross-ratios
Enable stereo matching without calibration

Quality Metrics:
- Epipolar line alignment error
- Image distortion measures
- Preserved image content
```

#### Dense Stereo Matching
**Disparity Computation**:
```
Disparity Definition:
d(x,y) = x_left - x_right
For rectified stereo pair

Window-Based Matching:
C(x,y,d) = Σᵢ,ⱼ [I_L(x+i,y+j) - I_R(x+i-d,y+j)]²

Winner-Take-All (WTA):
d*(x,y) = argmin_d C(x,y,d)

Dynamic Programming:
1D optimization along epipolar lines
Smoothness constraints
Occlusion handling

Global Optimization:
Energy minimization:
E = Σₚ C(p,dₚ) + Σₚ,q V(dₚ,dq)
Data term + smoothness term
Graph cuts, belief propagation solutions
```

**3D Reconstruction**:
```
Triangulation:
Given disparity d, compute depth:
Z = (f × B) / d
where f = focal length, B = baseline

3D Point:
X = (x × Z) / f
Y = (y × Z) / f
Z = (f × B) / d

Uncertainty Analysis:
σ_Z = (Z²/fB) × σ_d
Depth uncertainty grows quadratically with distance
Larger baseline B improves depth accuracy

Point Cloud Generation:
Convert disparity map to 3D points
Filter outliers and invalid disparities
Apply color information from reference image
```

---

## 🌊 Optical Flow and Motion Analysis

### Dense Optical Flow

#### Horn-Schunck Method
**Global Smoothness Approach**:
```
Energy Functional:
E = ∫∫ [(Iₓu + Iᵧv + Iₜ)² + α²(||∇u||² + ||∇v||²)] dx dy

Where:
- First term: Brightness constancy constraint
- Second term: Smoothness regularization
- α: Regularization parameter

Euler-Lagrange Equations:
Iₓ(Iₓu + Iᵧv + Iₜ) - α²∇²u = 0
Iᵧ(Iₓu + Iᵧv + Iₜ) - α²∇²v = 0

Iterative Solution:
u^(n+1) = ū^(n) - Iₓ[Iₓū^(n) + Iᵧv̄^(n) + Iₜ]/(α² + Iₓ² + Iᵧ²)
v^(n+1) = v̄^(n) - Iᵧ[Iₓū^(n) + Iᵧv̄^(n) + Iₜ]/(α² + Iₓ² + Iᵧ²)

where ū, v̄ are local averages
```

#### Lucas-Kanade Method
**Local Approach**:
```
Assumption: Constant flow in local neighborhood

Matrix Formulation:
[Iₓ₁ Iᵧ₁] [u] = -[Iₜ₁]
[Iₓ₂ Iᵧ₂] [v]    [Iₜ₂]
[  ⋮  ⋮ ]        [ ⋮ ]

Least Squares Solution:
[u] = -(AᵀA)⁻¹Aᵀb
[v]

Where A = [Iₓ Iᵧ], b = [Iₜ]

Solvability Condition:
AᵀA must be well-conditioned
Eigenvalues λ₁, λ₂ > threshold
Indicates presence of texture (corners)

Aperture Problem:
Only normal component of flow observable
Parallel edges create rank-deficient system
Need texture in multiple orientations
```

### Feature Tracking

#### KLT Tracker
**Tracking Framework**:
```
Template Tracking:
Minimize: Σₓ,ᵧ [I₁(x,y) - I₂(x+dx,y+dy)]²

Newton-Raphson Iteration:
[dx] = [dx₀] - H⁻¹∇E
[dy]   [dy₀]

Hessian Matrix:
H = [Σ Iₓ²   Σ IₓIᵧ]
    [Σ IₓIᵧ  Σ Iᵧ² ]

Gradient:
∇E = [Σ IₓIₜ]
     [Σ IᵧIₜ]

Feature Selection:
Select points with large min(λ₁, λ₂)
Ensures trackable features
Avoid aperture problem
```

**Multi-Scale Tracking**:
```
Pyramid Construction:
Level 0: Original image
Level L: Gaussian smoothed and downsampled

Coarse-to-Fine Tracking:
1. Track at coarsest level
2. Propagate estimate to finer level
3. Refine estimate
4. Repeat until finest level

Displacement Propagation:
d_L-1 = 2 × d_L (double displacement going down pyramid)

Benefits:
- Handle large displacements
- Improved convergence
- Robust to noise
- Computational efficiency
```

#### Optical Flow Applications
**Motion Segmentation**:
```
Motion Clustering:
Cluster pixels by similar flow vectors
K-means on (u, v) flow components
Segment moving objects from background

Layered Motion:
Model scene as multiple moving layers
Each layer has parametric motion model
EM algorithm for layer assignment and parameter estimation

Dominant Motion Estimation:
RANSAC on flow vectors
Estimate background motion (camera motion)
Segment foreground objects with different motion
```

**3D Motion Estimation**:
```
Structure from Motion:
Estimate camera motion and 3D structure
From optical flow across multiple frames

Ego-Motion Estimation:
Focus of Expansion (FOE):
Point where flow vectors converge
Indicates direction of camera translation

Time-to-Contact:
τ = Z/Vz = r/||v(r)||
where r = distance from FOE
Applications: Autonomous navigation, collision avoidance

Rotation vs. Translation:
Pure rotation: Rotational flow field
Pure translation: Radial flow field
General motion: Combination of both
```

---

## 🎯 Advanced Understanding Questions

### Segmentation Theory:
1. **Q**: Analyze the mathematical properties of different segmentation algorithms and compare their sensitivity to initialization, noise, and parameter selection.
   **A**: Region growing: sensitive to seed selection and threshold, robust to noise within homogeneous regions. Watershed: sensitive to gradient computation, prone to over-segmentation without markers. Active contours: require good initialization, converge to local minima, sensitive to image gradients. K-means: sensitive to initialization and k selection, assumes spherical clusters. Mean shift: robust to initialization, sensitive to bandwidth selection.

2. **Q**: Derive the mathematical relationship between level set evolution and energy minimization in geometric active contours.
   **A**: Level set evolution ∂φ/∂t = F|∇φ| minimizes energy functional E = ∫C ds + ∫∫Ω g(I)dA where C is contour, Ω is interior region. Speed function F derives from Euler-Lagrange equation of energy functional. Geometric flows naturally handle topology changes while minimizing perimeter and region-based terms.

3. **Q**: Compare clustering-based segmentation methods and analyze their theoretical foundations and convergence properties.
   **A**: K-means: minimizes within-cluster variance, guaranteed convergence to local minimum, assumes isotropic clusters. Mean shift: finds modes of density function, converges to stationary points, handles arbitrary cluster shapes. EM: maximizes likelihood under mixture model, proven convergence, incorporates probabilistic uncertainty.

### Template Matching and Object Detection:
4. **Q**: Analyze the mathematical relationship between correlation-based matching and optimal filter theory, and derive conditions for optimal template design.
   **A**: Correlation implements matched filtering under Gaussian noise assumptions. Optimal template maximizes SNR = |μ₁-μ₂|²/(σ₁²+σ₂²) where μ,σ are signal/noise statistics. Template should match expected signal pattern while being orthogonal to noise. For unknown targets, Wiener filtering provides optimal trade-off between signal matching and noise suppression.

5. **Q**: Compare the theoretical properties of different multi-scale detection approaches and analyze their computational and accuracy trade-offs.
   **A**: Pyramid methods: logarithmic scale sampling, efficient computation, may miss intermediate scales. Integral images: exact computation for rectangular features, limited to specific feature types. DFT methods: exact scale analysis, computationally expensive for large ranges. Trade-offs: computational efficiency vs. scale coverage vs. accuracy.

6. **Q**: Derive the mathematical foundations of cascade classifiers and analyze their optimization principles for detection accuracy and speed.
   **A**: Cascade optimizes product of detection rates D = ∏ᵢdᵢ and false positive rates F = ∏ᵢfᵢ. Each stage targets high detection rate (d≥0.99) and moderate false positive rate (f≤0.5). Overall system achieves exponential false positive reduction while maintaining high detection. Optimization balances per-stage complexity with cascade depth.

### Geometric Transformations and Stereo Vision:
7. **Q**: Analyze the mathematical relationship between epipolar geometry and camera calibration, and derive optimal baseline configurations for stereo reconstruction.
   **A**: Fundamental matrix F encodes epipolar geometry independent of calibration. Essential matrix E = K'ᵀFK incorporates calibration. Optimal baseline: large enough for good depth resolution (Z ∝ 1/disparity), small enough to maintain correspondence matching. Trade-off between depth accuracy and matching reliability.

8. **Q**: Compare different dense stereo matching algorithms and analyze their accuracy-efficiency trade-offs under various scene conditions.
   **A**: Local methods (block matching): fast computation, sensitive to texture-less regions, limited accuracy near discontinuities. Global methods (graph cuts, belief propagation): higher accuracy, handle occlusions, computationally expensive. Semi-global methods: good accuracy-efficiency balance, suitable for real-time applications. Performance depends on scene texture, depth discontinuities, and occlusions.

---

## 🔑 Key Classical Computer Vision Principles

1. **Mathematical Foundations**: Classical methods rely on well-established mathematical principles from signal processing, optimization, and geometry.

2. **Geometric Understanding**: Explicit geometric models enable interpretable results and principled parameter selection.

3. **Multi-Scale Analysis**: Many classical techniques benefit from multi-scale approaches for robustness and efficiency.

4. **Robust Estimation**: RANSAC and other robust methods are essential for handling outliers and noise in real-world applications.

5. **Trade-off Analysis**: Understanding accuracy-efficiency trade-offs guides algorithm selection for specific applications and computational constraints.

---

**Next**: Continue with Day 6 - Part 4: Data Augmentation Theory and Statistical Analysis