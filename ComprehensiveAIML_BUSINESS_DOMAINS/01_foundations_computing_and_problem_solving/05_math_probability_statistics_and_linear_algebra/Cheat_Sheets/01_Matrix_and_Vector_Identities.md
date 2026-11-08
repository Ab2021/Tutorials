---
collection: "math-probability-statistics-linear-algebra"
file_id: "01"
topic: "Matrix and vector identities"
level: 1
status: "draft; unreviewed"
version: "1.0"
created: "2026-11-08"
reviewed: null
source_status: "course-topic draft; visible probability source inventory consulted, original bodies unavailable"
---

# Matrix and Vector Identities

> **Reference boundary:** [`Train/Templates/directory.txt`](../../../../Templates/directory.txt) was consulted as source inventory. This is a new course-topic draft, not a reconstruction of unavailable material.

## 1. Shapes are the first invariant

For `A` shape `m×n` and `x` shape `n×1`, `y=Ax` has shape `m×1`. Every index in a sum must have a declared range. Most ML linear-algebra defects are silently wrong axis/layout/broadcast choices rather than difficult algebra.

## 2. Core identities

| Identity | Validity | Use | Boundary |
|---|---|---|---|
| `(AB)^T=B^TA^T` | Compatible matrices | Gradient/shape manipulation | Order reverses |
| `(AB)^{-1}=B^{-1}A^{-1}` | Both square/invertible | Change of coordinates | Inversion may be ill-conditioned |
| `||x||_2^2=x^Tx` | Real vector | Distance/regularization | Square is not norm |
| `x^TAx` | A square | Quadratic form | Symmetry/PSD determines geometry |
| `tr(AB)=tr(BA)` | Products both defined/square trace | Derivations | Not arbitrary factor reordering |

## 3. Least squares

For design matrix `X` (`n×d`) and targets `y` (`n×1`), minimize `L(w)=||Xw-y||_2^2`. Expanding gives `w^TX^TXw-2y^TXw+y^Ty`; differentiating gives `2X^TXw-2X^Ty`. The stationary condition is `X^TXw=X^Ty`. If `X^TX` is invertible, `w=(X^TX)^{-1}X^Ty`; otherwise use a numerically stable solver/pseudoinverse or regularization, not explicit inversion by default.

Example: `X=[[1,0],[0,1],[1,1]]`, `y=[1,2,2]`. Then `X^TX=[[2,1],[1,2]]`, `X^Ty=[3,4]`, and `w=[2/3,5/3]`. Residuals are `[-1/3,-1/3,1/3]`; squared loss is `1/3`.

## 4. Conditioning and scaling

A small residual does not mean stable parameters. Nearly collinear columns make `X^TX` poorly conditioned; tiny input changes can create large coefficient changes. Center/scale features when the model permits, inspect singular values/condition estimates, and use ridge regularization `X^TX+lambda I` with a declared selection protocol.

## 5. Failure modes

| Symptom | Check | Response |
|---|---|---|
| Shape-compatible wrong result | Named axes and unit test with hand values | Assert shape/axis semantics, not only dimensions |
| Inversion error/unstable coefficients | Rank/singular values and perturbation | Use solve/SVD/regularization |
| Broadcast changes loss | Explicit expansion and per-example test | Specify reduction/axis; test small tensor oracle |
| Leakage through normalization | Fit transform uses held-out rows | Fit train only and version transform |

## 6. Validation

Use tiny hand-computed matrices, shape assertions, residual/normal-equation checks, rank/condition diagnostics and held-out split-safe pipeline tests. Compare solver result against a stable numerical reference rather than only checking it does not throw.

## 7. Interview checks

1. **Why avoid explicit inverse?** Solvers are usually more stable/efficient.
2. **What proves `X^TX` is invertible?** Full column rank, not merely `n>d`.
3. **Why does transpose reverse order?** Matrix multiplication is noncommutative.
4. **What is a residual?** Prediction minus target under declared sign convention.
5. **What does poor conditioning risk?** Parameter instability under small perturbations.

## 8. Quick reference

| Before using algebra | Check |
|---|---|
| Multiply | Inner dimensions and axis meaning |
| Solve | Rank/conditioning and stable solver |
| Evaluate | Split-safe transforms and residual metric |

**Limitations:** No source body or workload was available. All matrices and calculations are illustrative.
