# Day 14 (Part 1): Advanced SVM & Kernels

> **Phase**: 6 - Deep Dive
> **Topic**: The Math of Kernels
> **Focus**: Mercer's Theorem, Duals, and Outliers
> **Reading Time**: 60 mins

---

## 1. The Dual Formulation

Why do we care about the Dual?

### 1.1 Primal vs. Dual
*   **Primal**: Minimize w.r.t weights $w$. Depends on dimension $D$.
*   **Dual**: Maximize w.r.t Lagrange multipliers $\alpha$. Depends on samples $N$.
*   **Kernel Trick**: The Dual only involves dot products $x_i^T x_j$. We can replace this with $K(x_i, x_j)$. This allows infinite dimensional mapping without computing coordinates.

### 1.2 Support Vectors
*   Only points with $\alpha_i > 0$ are Support Vectors.
*   They lie on or inside the margin.
*   The solution depends *only* on SVs. Sparsity!

---

## 2. Mercer's Theorem

### 2.1 Valid Kernels
*   A function $K(x, y)$ is a valid kernel if and only if the Gram Matrix is Positive Semi-Definite (PSD) for any set of data.
*   **Implication**: You can design custom kernels (e.g., String Kernel, Graph Kernel) as long as they satisfy PSD.

---

## 3. One-Class SVM

*   **Goal**: Anomaly Detection.
*   **Idea**: Find a hyperplane that separates the data from the Origin with maximum margin.
*   **Result**: Captures the "density" of the normal data. Points on the other side are outliers.

---

## 4. Tricky Interview Questions

### Q1: RBF Kernel (Gaussian) maps to infinite dimensions. Explain.
> **Answer**:
> *   $e^{-|x-y|^2} = e^{-x^2}e^{-y^2}e^{2xy}$.
> *   Taylor expansion of $e^{2xy}$ is an infinite sum $\sum \frac{(2xy)^n}{n!}$.
> *   This corresponds to a dot product of infinite polynomial features.

### Q2: Primal or Dual? Which to use?
> **Answer**:
> *   **N >> D** (Many rows, few features): Use **Primal** (Linear SVM). $O(N)$.
> *   **D >> N** (Genomics, Text): Use **Dual** (Kernel SVM). $O(N^2)$ or $O(N^3)$.

### Q3: What is $\nu$-SVM?
> **Answer**:
> *   Standard SVM uses $C$ (penalty). Hard to tune (range 0 to infinity).
> *   $\nu$-SVM uses $\nu \in (0, 1)$.
> *   $\nu$ is the lower bound on the fraction of Support Vectors and upper bound on fraction of Margin Errors. More intuitive.

---

## 5. Practical Edge Case: Scaling
*   **Problem**: SVM is not scale invariant. Large features dominate distance.
*   **Must Do**: StandardScaler (Mean 0, Var 1) before SVM.
*   **Complexity**: Kernel SVM is $O(N^2)$ memory. Don't use on >50k samples. Use LinearSVC or Approximation (Nystroem).

