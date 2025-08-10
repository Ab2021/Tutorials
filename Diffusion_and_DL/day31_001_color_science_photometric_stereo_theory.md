# Day 31 - Part 1: Color Science & Photometric Stereo Theory

## 📚 Learning Objectives
By the end of this section, you will understand:
- Mathematical foundations of colorimetry and human visual perception
- Theoretical analysis of camera response functions and color space transformations
- Mathematical principles of photometric stereo and shape-from-shading
- Information-theoretic perspectives on color constancy and white balance
- Theoretical frameworks for reflectance modeling and BRDF estimation
- Mathematical modeling of illumination estimation and color correction

---

## 🌈 Colorimetry and Color Space Theory

### Mathematical Foundation of Color Perception

#### Human Visual System Mathematics
**Trichromatic Color Theory**:
```
CIE Color Matching Functions:
Color C = ∫ S(λ) R(λ) dλ
Where S(λ) is spectral power distribution, R(λ) is response function

Tristimulus Values:
X = ∫ S(λ) x̄(λ) dλ
Y = ∫ S(λ) ȳ(λ) dλ  
Z = ∫ S(λ) z̄(λ) dλ

Where x̄(λ), ȳ(λ), z̄(λ) are CIE color matching functions

Mathematical Properties:
- Linear combination suffices for color reproduction
- Y corresponds to luminance (brightness)
- All visible colors within XYZ space
- Metamerism: different spectra, same color
```

**Chromaticity Coordinates**:
```
Normalized Coordinates:
x = X/(X+Y+Z)
y = Y/(X+Y+Z)
z = Z/(X+Y+Z) = 1-x-y

Chromaticity Diagram:
2D representation of color space
Mathematical: projection from 3D to 2D
Horseshoe-shaped boundary: spectral locus

Mathematical Properties:
- Additive colors lie on straight lines
- White point: equal energy distribution
- Color temperature: black body locus
- Gamut: triangle connecting primaries
```

#### Color Space Transformations
**RGB to XYZ Transformation**:
```
Linear Transformation:
[X]   [0.4124 0.3576 0.1805] [R]
[Y] = [0.2126 0.7152 0.0722] [G]
[Z]   [0.0193 0.1192 0.9505] [B]

Matrix Properties:
- Dependent on RGB primaries choice
- Different matrices for different standards
- Invertible transformation
- Preserves linearity in light

Inverse Transformation:
RGB = M⁻¹ XYZ
Where M is transformation matrix
Used for display color reproduction
```

**Perceptually Uniform Spaces**:
```
CIE L*a*b* Color Space:
L* = 116 f(Y/Yn) - 16
a* = 500[f(X/Xn) - f(Y/Yn)]
b* = 200[f(Y/Yn) - f(Z/Zn)]

Where f(t) = t^(1/3) if t > (6/29)³
           = (1/3)(29/6)² t + 4/29 otherwise

Mathematical Properties:
- Perceptually uniform: ΔE ≈ perceived difference
- L*: lightness, a*: green-red, b*: blue-yellow
- Euclidean distance approximates visual difference
- Used in color difference calculations
```

### Camera Response and Calibration

#### Camera Response Function Theory
**Radiometric Camera Model**:
```
Camera Response:
I = f(E) where E is scene radiance
f: response function (typically non-linear)

Linear Response:
I = αE + β
Ideal but not realistic for real cameras

Gamma Correction:
I = αE^γ + β
Typical γ ≈ 0.4-0.5 (inverse of display gamma ≈ 2.2)

Mathematical Analysis:
- Response function affects color reproduction
- Non-linearity needed for perceptual uniformity
- Calibration required for accurate color
- White balance affects color temperature
```

**Response Function Estimation**:
```
Multi-Exposure Calibration:
Multiple images at different exposures
Mathematical: solve for f and scene radiance E

Log-Linear Model:
log I = γ log E + log α + β
Linear regression in log space
Robust to noise and outliers

Polynomial Model:
f(E) = Σᵢ aᵢ Eⁱ
Higher-order polynomial approximation
More flexible but may overfit

Radiometric Calibration:
Recover both f and E from images
Mathematical: inverse problem
Requires multiple constraints
```

#### White Balance Mathematics
**Illumination Estimation**:
```
Gray World Assumption:
Average scene reflectance is achromatic
Mathematical: E[R(λ)] = constant across wavelengths
Estimate illumination from image statistics

White Patch Assumption:
Brightest patch is white under current illumination
Mathematical: max(R,G,B) corresponds to illumination
Simple but sensitive to outliers

Color Constancy:
Perceived color independent of illumination
Mathematical: remove illumination effects
f(surface_reflectance, illumination) → surface_reflectance
```

**Advanced White Balance Algorithms**:
```
Bayesian White Balance:
p(illumination|image) ∝ p(image|illumination) × p(illumination)
Prior knowledge about illumination distribution
Mathematical: maximum a posteriori estimation

Neural White Balance:
Learn mapping from image to illumination
Mathematical: regression problem
Training on datasets with known illumination

Color Temperature Estimation:
T = illumination color temperature
Mathematical: fit to black body spectrum
Mired scale: M = 10⁶/T for linear interpolation
```

---

## 💡 Photometric Stereo Theory

### Mathematical Foundation of Shape from Shading

#### Reflectance Models
**Lambertian Reflectance**:
```
Lambert's Law:
I = ρ/π (n̂ · ŝ)
Where:
- I: observed intensity
- ρ: albedo (surface reflectance)
- n̂: surface normal
- ŝ: light direction

Mathematical Properties:
- View-independent (diffuse reflection)
- Linear in light direction
- Cosine law of illumination
- Fundamental assumption for photometric stereo
```

**Phong Reflection Model**:
```
Combined Reflection:
I = Ia + Id + Is
Ambient + Diffuse + Specular components

Phong Specular:
Is = ks(r̂ · v̂)ⁿ
Where r̂ is reflection direction, v̂ is view direction

Mathematical Analysis:
- Non-linear in surface normal
- View-dependent specular component
- Complications for photometric stereo
- Requires more sophisticated algorithms
```

**Bidirectional Reflectance Distribution Function (BRDF)**:
```
General BRDF:
fr(ωi, ωr) = dLr(ωr) / (dLi(ωi) cos θi dωi)
Where ωi, ωr are incident and reflected directions

Mathematical Properties:
- Reciprocity: fr(ωi, ωr) = fr(ωr, ωi)
- Energy conservation: ∫ fr(ωi, ωr) cos θr dωr ≤ 1
- Positive definite: fr ≥ 0
- Describes all possible reflectance behaviors
```

#### Classical Photometric Stereo
**Three-Light Photometric Stereo**:
```
System of Equations:
I₁ = ρ(n̂ · ŝ₁)
I₂ = ρ(n̂ · ŝ₂)  
I₃ = ρ(n̂ · ŝ₃)

Matrix Form:
[I₁]   [ŝ₁ᵀ]
[I₂] = [ŝ₂ᵀ] ρn̂ = S ρn̂
[I₃]   [ŝ₃ᵀ]

Solution:
ρn̂ = S⁻¹ I
ρ = ||ρn̂||
n̂ = ρn̂/ρ

Mathematical Requirements:
- Three non-coplanar light directions
- Lambertian surfaces
- Known light directions and intensities
- No shadows or interreflections
```

**Least Squares Photometric Stereo**:
```
Over-Determined System:
More than three lights for robustness
I = S ρn̂ where I ∈ ℝᵐ, S ∈ ℝᵐˣ³

Least Squares Solution:
ρn̂ = (SᵀS)⁻¹SᵀI = S†I
Where S† is Moore-Penrose pseudoinverse

Error Analysis:
Residual: e = ||I - Sρn̂||²
Confidence measure for reconstruction quality
Outlier detection for robust estimation

Mathematical Benefits:
- Robust to noise and measurement errors
- Handles over-determined systems
- Provides uncertainty quantification
- Standard linear algebra solution
```

### Advanced Photometric Stereo Methods

#### Uncalibrated Photometric Stereo
**Mathematical Formulation**:
```
Unknown Lighting:
I = L N where L ∈ ℝᵐˣ³, N ∈ ℝ³ˣⁿ
L: unknown lighting matrix
N: unknown normal/albedo matrix

Ambiguity:
I = L N = (L A)(A⁻¹ N)
Any invertible 3×3 matrix A gives valid solution
Mathematical: generalized bas-relief ambiguity

Constraint Resolution:
- Integrability constraint: normals from integrable surface
- Albedo smoothness: spatial albedo variation
- Shadow constraints: intensity bounds
- Isotropy constraint: surface smoothness
```

**Integrability Constraint**:
```
Surface Integrability:
∂²z/∂x∂y = ∂²z/∂y∂x
Curl-free condition for surface normals

Mathematical Implementation:
∇ × n̂ = 0 in discrete form
Additional constraints for ambiguity resolution
Enforces geometric consistency

Optimization:
min ||I - LN||² + λ ||∇ × N||²
Joint optimization over L and N
Regularization promotes integrability
```

#### Non-Lambertian Photometric Stereo
**Specular Reflection Handling**:
```
Outlier Detection:
Identify specular pixels as outliers
Mathematical: deviation from Lambertian model
Robust estimation techniques

Rank Constraint:
Lambertian: rank(I) ≤ 3
Specular highlights violate rank constraint
Mathematical: low-rank matrix recovery

Sparse+Low-Rank Decomposition:
I = L + S where L is low-rank, S is sparse
L: Lambertian component
S: specular component
Mathematical: convex optimization problem
```

**General BRDF Photometric Stereo**:
```
Non-Linear Optimization:
I = BRDF(n̂, ŝ, v̂, material_parameters)
Non-linear least squares problem
Requires initial estimates and regularization

Spherical Harmonics Representation:
BRDF(ω) ≈ Σ cₗᵐ Yₗᵐ(ω)
Mathematical: frequency domain representation
Efficient for certain BRDF classes

Neural BRDF Models:
Learn BRDF representation from data
Mathematical: function approximation
Can handle complex material properties
```

---

## 🔍 Shape from Shading Theory

### Mathematical Formulation

#### Image Formation Model
**Shape from Shading Equation**:
```
Brightness Equation:
I(x,y) = R(p,q)
Where p = ∂z/∂x, q = ∂z/∂y are surface gradients

Lambertian Case:
R(p,q) = ρ (1 + p px + q py) / √((1+p²+q²)(1+px²+py²))
Where (px,py) is light direction

Mathematical Properties:
- Partial differential equation in z(x,y)
- Non-linear due to normalization
- Under-constrained: one equation, one unknown function
- Requires boundary conditions or additional constraints
```

**Reflectance Map**:
```
Gradient Space Representation:
R(p,q) maps surface gradients to intensity
Mathematical: reflectance function
Independent of surface position

Iso-Brightness Contours:
Curves of constant R(p,q) in gradient space
Mathematical: level sets of reflectance function
Provide geometric insight into problem structure

Characteristic Strip Method:
Solve along characteristic curves
Mathematical: method of characteristics for PDEs
Reduces PDE to ODE along specific paths
```

#### Variational Approaches
**Energy Minimization**:
```
Variational Formulation:
E = ∫∫ [I(x,y) - R(p,q)]² + λ[p²+q²] dx dy
Data term + smoothness regularization

Euler-Lagrange Equation:
∂E/∂z = 0 leads to PDE:
∇²z = f(I, ∇z, ∇²z)
Non-linear elliptic PDE

Numerical Solution:
Finite difference discretization
Iterative methods: Gauss-Seidel, SOR
Multigrid for efficiency
Convergence depends on λ and boundary conditions
```

**Regularization Theory**:
```
Smoothness Prior:
Surfaces are generally smooth
Mathematical: minimize curvature
|∇²z|² or total variation regularization

Membrane Model:
E_smooth = ∫∫ |∇z|² dx dy
First-order smoothness
Mathematical: minimize surface area

Thin Plate Model:
E_smooth = ∫∫ (zxx² + 2zxy² + zyy²) dx dy
Second-order smoothness
Mathematical: minimize bending energy
```

### Neural Shape from Shading

#### Deep Learning Approaches
**CNN-Based Methods**:
```
Direct Regression:
CNN: Image → Surface Normals/Depth
Mathematical: function approximation
End-to-end learning from data

Multi-Scale Processing:
Coarse-to-fine normal estimation
Mathematical: hierarchical optimization
Better handling of global shape

Loss Functions:
L = L_normal + λ₁ L_depth + λ₂ L_integrability
Multiple supervision signals
Mathematical: multi-objective optimization
```

**Physics-Informed Networks**:
```
Differentiable Rendering:
Incorporate shading model in loss
Mathematical: physics constraints in learning
Ensures physical consistency

Integrability Loss:
L_int = ||∇ × n̂||²
Enforce geometric constraints
Mathematical: curl-free condition
Prevents impossible normal fields

Shadow Constraint:
I ≥ 0 everywhere
Mathematical: inequality constraint
Physically meaningful intensity bounds
```

#### Uncertainty Quantification
**Bayesian Shape from Shading**:
```
Posterior Distribution:
p(z|I) ∝ p(I|z) p(z)
Likelihood × prior
Mathematical: Bayesian inference

Variational Inference:
Approximate posterior with tractable distribution
Mathematical: minimize KL divergence
Scalable uncertainty quantification

Monte Carlo Methods:
Sample from posterior distribution
Mathematical: MCMC or variational sampling
Provides confidence intervals for reconstruction
```

**Confidence Estimation**:
```
Reconstruction Confidence:
Based on data term residual
Mathematical: goodness of fit measure
Low residual → high confidence

Geometric Confidence:
Based on surface integrability
Mathematical: geometric consistency
Well-posed regions → high confidence

Combined Confidence:
Weight both data and geometric terms
Mathematical: multi-factor confidence
Guides adaptive algorithms
```

---

## 🎯 Advanced Understanding Questions

### Color Science Theory:
1. **Q**: Analyze the mathematical relationship between camera sensor spectral sensitivity and color reproduction accuracy, developing optimal color correction strategies.
   **A**: Mathematical relationship: color reproduction accuracy depends on overlap between camera and human visual spectral sensitivities. Analysis: cameras with narrow-band sensors may miss spectral information, leading to metamerism failures. Optimal correction: (1) spectral sharpening to reduce overlap, (2) color matrix optimization for specific illuminants, (3) polynomial correction for non-linearities. Mathematical framework: minimize color difference ΔE between camera and standard observer. Key insight: perfect color reproduction requires matching spectral sensitivities, but good approximation possible with proper calibration.

2. **Q**: Develop a theoretical framework for evaluating color constancy algorithms under different illumination conditions and derive performance bounds.
   **A**: Framework based on color error metrics and illumination variation. Performance measures: angular error between true and estimated illumination, color difference after correction. Theoretical bounds: perfect color constancy impossible without prior knowledge due to metamerism. Analysis: Gray World achieves good performance for diverse scenes, White Patch fails for scenes without white objects. Mathematical bound: minimum error depends on scene color distribution and illumination statistics. Optimal algorithm: Bayesian methods with learned priors achieve theoretical optimum.

3. **Q**: Compare the mathematical foundations of different color spaces (RGB, HSV, LAB) for computer vision applications and analyze their respective advantages.
   **A**: Mathematical comparison: RGB is linear and device-dependent, HSV separates chromatic from achromatic information, LAB is perceptually uniform. RGB advantages: simple linear operations, direct sensor correspondence. HSV advantages: intuitive color adjustments, robust to illumination changes. LAB advantages: perceptual uniformity, device independence. Applications: RGB for deep learning (linear operations), HSV for color-based segmentation, LAB for perceptual tasks. Mathematical insight: choice depends on task requirements and computational constraints.

### Photometric Stereo Theory:
4. **Q**: Analyze the mathematical conditions for unique surface reconstruction in photometric stereo and derive minimal lighting requirements.
   **A**: Mathematical conditions: (1) at least three non-coplanar light directions for Lambertian surfaces, (2) lights must span 3D space sufficiently. Analysis: fewer than three lights → ambiguous reconstruction, coplanar lights → bas-relief ambiguity. Minimal requirements: three lights with linearly independent directions, sufficient intensity variation. Mathematical framework: rank(S) = 3 for unique solution where S is lighting matrix. Theoretical insight: geometric diversity in lighting crucial for disambiguation, more lights improve robustness but don't reduce minimal requirements.

5. **Q**: Develop a mathematical analysis of the robustness of photometric stereo to non-Lambertian reflectance and derive error bounds for specular surfaces.
   **A**: Mathematical analysis: specular reflectance violates linear Lambertian model, causing systematic errors. Error bounds: depend on specular lobe width and strength. Framework: decompose observed intensity into Lambertian and specular components I = I_L + I_S. Robustness strategies: (1) outlier detection for specular pixels, (2) robust estimation (RANSAC, M-estimators), (3) iterative reweighting. Mathematical bound: error proportional to specular/diffuse ratio. Key insight: robust methods can handle moderate specularity, but severe specular reflection requires specialized algorithms.

6. **Q**: Compare uncalibrated vs calibrated photometric stereo in terms of mathematical complexity, assumptions, and reconstruction quality.
   **A**: Mathematical comparison: calibrated requires known lighting (3×m constraints), uncalibrated estimates lighting simultaneously (bas-relief ambiguity). Complexity: calibrated is linear least squares, uncalibrated is non-linear optimization with constraints. Assumptions: calibrated assumes accurate lighting knowledge, uncalibrated assumes integrability. Quality: calibrated achieves metric reconstruction, uncalibrated up to affine transformation. Mathematical insight: calibration trades measurement effort for algorithmic simplicity and reconstruction accuracy.

### Shape from Shading:
7. **Q**: Analyze the mathematical relationship between boundary conditions and solution uniqueness in shape from shading problems.
   **A**: Mathematical analysis: shape from shading is ill-posed PDE requiring boundary conditions for unique solution. Types of boundary conditions: Dirichlet (specify height), Neumann (specify gradient), mixed. Uniqueness analysis: proper boundary conditions ensure unique solution to brightness equation. Mathematical framework: well-posed problem requires sufficient boundary information to constrain solution space. Practical considerations: boundary conditions often unknown, requiring estimation or assumptions. Theoretical insight: boundary condition type and coverage determine reconstruction quality and uniqueness.

8. **Q**: Design a unified mathematical framework for combining photometric stereo and shape from shading that leverages advantages of both approaches.
   **A**: Framework components: (1) photometric stereo for local normal estimation, (2) shape from shading for global consistency, (3) integration constraints for surface reconstruction. Mathematical formulation: combined energy function E = E_PS + λ₁E_SFS + λ₂E_integrability. Benefits: photometric stereo provides robust local estimates, shape from shading enforces global consistency, integration ensures geometric validity. Optimization: alternating minimization or joint optimization. Theoretical guarantee: combined approach inherits robustness of photometric stereo and global consistency of shape from shading while mitigating individual limitations.

---

## 🔑 Key Color Science and Photometric Stereo Principles

1. **Colorimetric Foundation**: Understanding CIE color spaces, spectral sensitivity, and perceptual uniformity is crucial for accurate color processing and reproduction in computer vision systems.

2. **Camera Response Modeling**: Proper modeling of camera response functions, gamma correction, and white balance is essential for accurate color interpretation and correction algorithms.

3. **Reflectance Models**: Mathematical understanding of Lambertian, Phong, and BRDF models provides the foundation for photometric stereo and shape estimation from shading cues.

4. **Geometric Constraints**: Integrability constraints and surface smoothness priors are mathematical tools that enable well-posed surface reconstruction from photometric information.

5. **Multi-Light Benefits**: Photometric stereo with multiple lighting conditions provides mathematical redundancy that enables robust surface normal estimation despite non-ideal reflectance properties.

---

**Next**: Continue with Day 32 - Domain Adaptation & Synthetic Data Theory