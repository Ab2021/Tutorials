---
collection: "math-probability-statistics-linear-algebra"
file_id: "01"
topic: "Matrix and vector identities"
level: 1
status: "draft; unreviewed"
version: "2.0"
created: "2026-11-08"
reviewed: null
review_trigger: "Any change to the worked example dataset, to the solver recommendation, or to the shape/notation conventions declared in Section 3."
notation_style: "plain-text"
prerequisites: []
pattern_tags: ["linear-algebra", "shapes", "identities", "least-squares", "conditioning", "numerical-stability"]
source_status: "OPEN"
owner_role: "Analytics and ML engineering"
source_index: "INDEX.md"
---

# Matrix and Vector Identities

> **Purpose:** Give the reader one auditable chain from a shape declaration to a solved linear system: which identity is legal, why it is legal, what it costs, and how to detect when it silently produced a wrong answer.
> **Prerequisites:** Arithmetic with real scalars; the meaning of a linear equation `a1*x1 + a2*x2 = b`; fractions; the idea of a derivative as a limit of a difference quotient. No prior matrix course is assumed.
> **Key outcome:** Given a design matrix and a target vector, you can state the shapes, derive the normal equations from the squared-error definition, solve a small system by hand, verify the solution two independent ways, compute its conditioning, and reject the solution when the conditioning or the rank makes it untrustworthy.
> **This file does NOT cover:** Probability distributions of estimators (that is the sibling distribution sheet), statistical test selection (the sibling test-selection sheet), or the design of a full numerical linear-algebra library.
> **Level:** 1 — foundations, with production diagnostics carried far enough to be usable in code review.
> **Version / reviewed:** 2.0 / not independently reviewed; the arithmetic in Section 5 was recomputed and cross-checked by identity-consistency routes, which is a self-check, not peer review.
> **Evidence:** Identities are DERIVED from index definitions. Every numeric instance is ILLUSTRATIVE constructed data whose arithmetic was checked. No benchmark, library timing, or production observation is claimed anywhere in this file.
> **Retrieval cue:** Shape or transpose error → Section 3 and Section 8; least-squares derivation → Section 4.2; suspicious coefficients → Section 5 and Section 8; solver choice → Section 7; cost question → Section 9.

## 1. File Header

The reference boundary in the previous version of this sheet was honest and is retained in spirit. `Train/Templates/directory.txt` is a source *inventory*: it lists topics that exist in the wider corpus, and listing is not the same as having inspected a body of text. Nothing below reconstructs unseen material. Each claim is either a derivation from a definition stated in this file, a standard theorem stated with its hypotheses, or an explicitly labelled constructed example. Where a fact is a property of floating-point hardware rather than of algebra, it is labelled as a machine-model assumption rather than as mathematics.

This sheet is about *identities* in the operational sense, not in the collectible sense. An identity is a rewrite rule that lets you replace one expression by another without changing the value. The engineering value of such a rule is that it changes three things at once: the shapes involved, the amount of arithmetic, and the sensitivity of the result to rounding. A reader who memorises `(AB)^T = B^T A^T` has memorised a string. A reader who knows that the rewrite reverses the multiplication order, that it is valid for every compatible pair with no invertibility requirement, that it costs one extra traversal of memory if applied naively, and that its failure symptom is a shape assertion firing inside a gradient accumulator, has acquired a tool.

The four identity families that carry almost all practical weight are:

1. **Structural identities** — transpose, trace, determinant, rank. These hold for any compatible operands, require no invertibility, and are therefore the safest rewrites in a codebase. When a derivation stalls, transposing and re-associating usually restarts it.
2. **Inverse identities** — `(AB)^{-1} = B^{-1}A^{-1}`, Sherman–Morrison, Woodbury, block inversion. These require invertibility of specific operands, which is a *mathematical* precondition and, separately, a *numerical* precondition that is much stronger in practice.
3. **Metric identities** — `||x||_2^2 = x^T x`, `tr(A^T B) = <A,B>_F`, `tr(x y^T) = y^T x`. These convert a norm or an inner product into an algebraic expression you can differentiate and reorganise. They are the workhorses of loss-function manipulation.
4. **Decomposition identities** — `A = Q Lambda Q^T` for symmetric `A`, `A = U Sigma V^T` in general, `A^+ = V Sigma^+ U^T`. These replace a matrix by factors with meaning, at a cost that is paid once and reused.

The reader task this file is built around is deliberately small and complete: fit a straight line to four points, by hand, three ways — normal equations, projection matrix, and ridge — then attack the solution with every diagnostic in Section 8 and see which ones fire. Four points is enough to make every number checkable on paper, and it is also enough to demonstrate rank deficiency (`Section 5.6`), because a repeated column in a four-row matrix is just as fatal as a repeated column in a four-million-row matrix. Scale changes cost, not algebra.

Two conventions are declared here because they are the source of more argument than any theorem below. First, **a vector is a column** unless explicitly written as a row; `x^T` is the row form. Second, **index bases are 1-based in the prose and 0-based only when code is shown**, and the code is labelled as code. Mixing those two silently is one of the eight failure modes catalogued later.

> **Takeaway:** An identity is a rewrite rule with a legality condition, a cost, and a failure signature; learn all three or you have learned a slogan.

## 2. Concept Map / Mental Model

The mental model is a funnel. Shapes enter at the top. Almost every illegal expression is rejected immediately by the shape gate, before any arithmetic happens. What survives the shape gate then faces a legality gate: does this rewrite need invertibility, symmetry, full rank, or a particular field? What survives legality is then handed to a compute stage, where the same mathematical answer can cost ten times more or lose eight digits of accuracy depending on the route chosen. Only then does a number come out, and the number is checked against residual and conditioning diagnostics before anyone is allowed to believe it.

```text
            declare shapes: A(m x n), B(n x p), x(n), y(m)
                              |
                              v
                 [GATE 1: conformability]
                   inner dims match? output dims named?
                     |                        |
                    no                       yes
                     |                        |
              fix the model             [GATE 2: legality]
              or the layout               invertible? symmetric?
                     ^                    full column rank? real?
                     |                        |            |
                     |                       no           yes
                     |                        |            |
                     |              use the weaker      [GATE 3: route]
                     |              identity: solve,     normal eq | QR | SVD
                     |              factorise, or        |  ridge | Woodbury
                     |              regularise          |
                     |                        |          |
                     +------------------------+----------+
                                              |
                                              v
                                    numeric result w_hat
                                              |
                              [GATE 4: diagnosis]
                     residual orthogonal to columns?  rank = d?
                     condition number acceptable?     refit perturbed?
                              |                    |
                            fails                 passes
                              |                    |
                    do not ship the coefficients   decision
```

Read the funnel as a claim about where defects actually live. In the experience that motivates this sheet, the expensive defects are not algebraic: a sign slip in a transpose is caught by a shape assertion in seconds, while a conditioning problem passes every shape check, returns plausible coefficients, and is discovered months later when a feature drifts by one percent and the prediction moves by forty. The funnel therefore puts the cheapest gates first and the most expensive gate last, and it insists that the expensive gate is *mandatory* rather than advisory.

A second map shows how the four identity families connect to the operations a practitioner actually performs. Structural identities sit on the left because they need nothing. Inverse identities sit in the middle because they need invertibility. Decomposition identities sit on the right because they need an algorithm and a budget.

```text
  STRUCTURAL                INVERSE                    DECOMPOSITION
  (always legal)            (needs invertibility)      (needs an algorithm)
  ----------------          -------------------        ------------------
  (AB)^T = B^T A^T          (AB)^-1 = B^-1 A^-1        A = Q L Q^T   (sym)
  tr(AB) = tr(BA)           (A^T)^-1 = (A^-1)^T        A = U S V^T   (any)
  det(AB) = det A det B     (A+uv^T)^-1 = Sherman      A^+ = V S^+ U^T
  rank(AB) <= min rank      (A+UCV)^-1 = Woodbury      ||A||_2 = sigma_max
  ||x||^2 = x^T x           block inverse via Schur    ||A||_F^2 = sum sigma^2
        |                          |                          |
        +----------> used to derive normal equations <--------+
                              |
                              v
                 w = argmin ||Xw - y||^2  ->  X^T X w = X^T y
```

The third map is the one to keep in working memory during review, because it converts "which identity applies?" into "what am I allowed to assume about this matrix?".

```text
  full column rank? ---- no ----> minimum-norm solution (pseudoinverse)
        |                         or regularise; never report unique
       yes                        coefficients as if identified
        |
  well conditioned? --- no -----> QR/SVD route, regularise, or rescale
        |                         features; κ(X^T X) = κ(X)^2 is the trap
       yes
        |
  cheap to refactor? --- no ----> keep a factorisation, not an inverse
        |                         (triangular solves are O(n^2))
       yes
        |
  small and dense ----> direct factorisation is fine
  huge and sparse ----> iterative method with a preconditioner
```

One further orienting idea ties the sections together. Every identity in this sheet can be read as a statement about *invariants under relabelling*. Transposition says the value of an inner product does not depend on which operand you call first. Trace cyclicity says the sum of diagonal entries of a product survives a cyclic relabelling of the factors. Determinant multiplicativity says the volume scale of a composed map is the product of the volume scales. Rank submultiplicativity says composing cannot create information. Once you see identities as invariance statements, the failure modes become obvious: a failure is a place where you applied an invariance that the situation does not actually have. `tr(ABC) = tr(ACB)` fails because swapping two factors is not a cyclic relabelling; `(AB)^{-1} = A^{-1}B^{-1}` fails because inversion does not commute with multiplication for non-commuting operands; `x^T A x` losing its antisymmetric part is not a failure at all but a genuine invariance, and knowing that saves arithmetic.

> **Takeaway:** Shapes are free to check and catch most errors; legality and conditioning are expensive to check and catch the errors that matter, so budget for them explicitly.

## 3. Foundational Definitions

Every symbol used later appears in this table with its shape, its domain, and the distinction that prevents a specific error. The final column is the important one: a definition that does not tell you what mistake it prevents is decoration.

| Term / symbol | Formal meaning | Shape, units, domain | Distinction that prevents an error |
|---|---|---|---|
| Scalar `a` | Element of the field, here the reals | `1x1`; dimensionless unless stated | A scalar is not a `1x1` array for broadcasting purposes in every library |
| Vector `x` | Ordered tuple of `n` reals, written as a column | `n x 1`; entries may carry units | A length-`n` 1-D array in code may be silently treated as a row |
| Matrix `A` | Rectangular array `A[i,j]`, `i` row, `j` column | `m x n` | "Rows are samples" is a convention, not a law; state it once |
| Transpose `A^T` | `(A^T)[i,j] = A[j,i]` | `n x m` | Transpose is not inverse; they coincide only for orthogonal matrices |
| Conjugate transpose `A^H` | `(A^H)[i,j] = conj(A[j,i])` | `n x m`, complex | Over the reals `A^H = A^T`; code that assumes this breaks on complex input |
| Identity `I_n` | `I[i,j] = 1` if `i=j` else `0` | `n x n` | Its size is inferred from context; a mis-sized `I` is a common silent broadcast |
| Zero matrix `0` | All entries zero | any | `A*0 = 0` but `A` times a zero vector is a zero vector, not a scalar zero |
| Matrix product `AB` | `(AB)[i,j] = sum_k A[i,k]B[k,j]` | `m x n` times `n x p` gives `m x p` | The inner dimension is consumed and must match exactly |
| Inner product `x^T y` | `sum_i x[i]y[i]` | scalar | Zero inner product means orthogonal, not independent |
| Outer product `x y^T` | Matrix with entries `x[i]y[j]` | `m x n` from `m` and `n` vectors | Rank one; summing outer products reconstructs any matrix product |
| Frobenius inner product | `sum_ij A[i,j]B[i,j]`, equal to `tr(A^T B)` | scalar | It is the entrywise dot product, not the operator inner product |
| Norm `||x||_p` | `(sum_i \|x[i]\|^p)^(1/p)` | scalar, same units as entries | `||x||_2^2` is a squared norm; it is not a norm because it violates the triangle inequality |
| Spectral norm `||A||_2` | Largest singular value `sigma_max` | scalar | It is not the largest entry, and not the Frobenius norm |
| Frobenius norm `||A||_F` | `sqrt(sum_ij A[i,j]^2)` | scalar | It is unitarily invariant but differs from the spectral norm by up to `sqrt(rank)` |
| Rank `rank(A)` | Dimension of the column space | integer, `0..min(m,n)` | Rank is a property of exact arithmetic; in floating point you must choose a tolerance |
| Null space `N(A)` | `{x : Ax = 0}` | subspace of dimension `n - rank(A)` | A tiny but nonzero singular value makes the numerical null space larger than the exact one |
| Column space `C(A)` | `{Ax : x in R^n}` | subspace of `R^m` | Least squares lives here: the fit is the closest point of this set to `y` |
| Row space | Span of the rows, equal to `C(A^T)` | dimension `rank(A)` | The minimum-norm solution is the one lying in the row space |
| Symmetric | `A = A^T` | square, real | Only the symmetric part contributes to `x^T A x` |
| Positive semidefinite (PSD) | `x^T A x >= 0` for all `x` | square, symmetric in practice | PSD implies real nonnegative eigenvalues; it does not imply invertibility |
| Positive definite (PD) | `x^T A x > 0` for all `x != 0` | square | Exactly the case where Cholesky exists and `A` is invertible |
| Orthogonal `Q` | `Q^T Q = I` | square (or tall with orthonormal columns) | Orthogonal means length-preserving; it does not mean "uncorrelated features" |
| Idempotent `P` | `P^2 = P` | square | A projection is idempotent; an idempotent that is not symmetric is an oblique projection |
| Projection matrix | `P = X(X^T X)^{-1}X^T` when `X` has full column rank | `m x m`, symmetric, idempotent | `tr(P) = rank(P) = d`, which gives a free rank check |
| Gram matrix `G` | `G = X^T X`, entries `x_i . x_j` | `d x d`, symmetric PSD | It squares the condition number, which is the central numerical hazard |
| Quadratic form | `q(x) = x^T A x` | scalar | Only `(A + A^T)/2` matters; using a non-symmetric `A` in a gradient is a bug |
| Eigenvalue `lambda` | `A v = lambda v`, `v != 0` | scalar (real for symmetric `A`) | Eigenvalues of a non-symmetric matrix may be complex; singular values are always real nonnegative |
| Singular value `sigma` | `sigma_i = sqrt(lambda_i(A^T A))` | real, `>= 0` | Singular values are not eigenvalues unless `A` is symmetric PSD |
| Condition number `kappa` | `sigma_max / sigma_min` | `>= 1`, dimensionless | It bounds relative output error per relative input error; it is not a measure of model quality |
| Pseudoinverse `A^+` | `V Sigma^+ U^T` from the SVD | `n x m` | It gives the minimum-norm least-squares solution; that is a convention, not a causal claim |
| Trace `tr(A)` | `sum_i A[i,i]`, equal to `sum_i lambda_i` when eigenvalues exist | scalar | Trace is linear; it is invariant under cyclic permutation only |
| Determinant `det(A)` | Signed volume scale of the linear map | scalar | Determinant is multiplicative, not additive; it underflows for large `n` |
| Hadamard product `A o B` | Entrywise product, same shape | `m x n` | It is not a matrix product; it has no simple relation to `det` |
| Kronecker product `A (x) B` | Block matrix `[A[i,j]*B]` | `mp x nq` | It linearises matrix equations; `(A (x) B)vec(X) = vec(B X A^T)` |
| Vectorisation `vec(X)` | Columns stacked into one column | `mn x 1` | Column-stacking and row-stacking conventions differ and must be declared |
| Similar matrices | `B = S^{-1} A S` | same size | Similar matrices share eigenvalues but not singular values |
| Unitary / orthogonal invariance | `||U A V||_F = ||A||_F` | — | This is why Frobenius norms survive changes of basis |
| Floating-point unit roundoff `u` | Half the gap between 1 and the next representable double | about `1.1e-16` | It converts a condition number into a digits-lost statement |

Three notation decisions follow from the table. First, `x^T y` and `<x,y>` are the same object, and the sheet uses the matrix form when a chain of transposes is being manipulated and the angle form when geometry is being discussed; the switch is always announced. Second, `|A|` is never used for determinant, because in some texts it means the entrywise absolute value and in others a matrix norm; `det(A)` is written in full. Third, `d` always denotes the number of columns (features) and `m` or `n` the number of rows (samples), and when a formula is symmetric in the two the convention is restated locally rather than remembered.

A note on what "vector" means for the derivations. The identity `||x||_2^2 = x^T x` is only a scalar because a column times a row is not involved: `x^T` is `1 x n` and `x` is `n x 1`, so the product is `1 x 1`. The reversed product `x x^T` is `n x n` and is a rank-one matrix whose trace equals `||x||_2^2`. Confusing the two is the single most common source of an accidental `m x m` matrix where a scalar loss was intended, and it is failure mode F5 in Section 8.

A note on exactness. Every theorem in Section 10 is a statement about exact real arithmetic unless it says otherwise. Floating-point arithmetic is a different algebraic structure: addition is not associative, multiplication by an inverse is not the same operation as division, and a matrix that is invertible in exact arithmetic may be numerically singular. The sheet therefore separates "the identity is true" from "the identity is a good way to compute". `w = (X^T X)^{-1} X^T y` is true whenever `X` has full column rank, and it is still frequently the wrong thing to compute.

> **Takeaway:** Declare shapes, index bases, and layout once; the definitions table exists so that a later disagreement about a formula can be resolved by pointing at a row rather than re-deriving from scratch.

## 4. Core Formulas with Full Derivation

### 4.1 Simplest / Base Case: Two Dimensions, Everything Visible

Before any general statement, work in two dimensions with every index written out. Nothing in this subsection requires a theorem; it requires only the definition of the matrix product. The purpose is to give the reader a *checkable* base to fall back on when a general derivation is in doubt: any identity that is false for `2 x 2` matrices is false in general, and a reader who can expand a `2 x 2` product can settle most disagreements in under a minute.

Validity: real entries, square `2 x 2` matrices, no invertibility assumed in the first three results.

**Derivation of the transpose-reversal identity (base case).** Let

```text
A = [[a11, a12], [a21, a22]]      B = [[b11, b12], [b21, b22]]
```

By definition `(AB)[i,j] = sum_k A[i,k] B[k,j]`. Expanding entry by entry:

```text
AB = [[a11 b11 + a12 b21,  a11 b12 + a12 b22],
      [a21 b11 + a22 b21,  a21 b12 + a22 b22]]
```

Now transpose, which swaps off-diagonal entries:

```text
(AB)^T = [[a11 b11 + a12 b21,  a21 b11 + a22 b21],
          [a11 b12 + a12 b22,  a21 b12 + a22 b22]]
```

Compute the other order, `B^T A^T`, using

```text
B^T = [[b11, b21], [b12, b22]]       A^T = [[a11, a21], [a12, a22]]
```

so that `(B^T A^T)[1,1] = b11 a11 + b21 a12`, matching `(AB)^T[1,1]`; `(B^T A^T)[1,2] = b11 a21 + b21 a22`, matching `(AB)^T[1,2]`; and similarly for the second row. The two matrices agree in all four entries. Nothing in the expansion used invertibility, symmetry, or squareness beyond what makes the product defined, so the result generalises by the same index argument, which is done in 4.2.

**Derivation of `||x||_2^2 = x^T x`.** For `x = [x1, x2]^T`, the Euclidean norm is defined as `sqrt(x1^2 + x2^2)`. Squaring removes the root: `||x||_2^2 = x1^2 + x2^2`. The matrix product `x^T x` is the `1 x 1` matrix whose only entry is `sum_i x_i x_i = x1^2 + x2^2`. The two scalars are equal term by term. The equality therefore rests on a definition (norm) and a definition (matrix product), and holds for real vectors of any length. For complex vectors the correct statement is `||x||_2^2 = x^H x`, which is real and nonnegative by construction, while `x^T x` is a complex number and is not a squared norm.

**Derivation of `tr(AB) = tr(BA)`, base case.** The trace is the sum of diagonal entries. From the expansion above, `tr(AB) = a11b11 + a12b21 + a21b12 + a22b22`. For `BA`, the diagonal entries are `b11a11 + b12a21` and `b21a12 + b22a22`, whose sum is `b11a11 + b12a21 + b21a12 + b22a22`. The four products are identical as multisets, so the sums agree. The mechanism is that the double sum `sum_i sum_j a_ij b_ji` is unchanged by renaming the dummy indices `i` and `j`. That rename is available only for a *pair*, which is exactly why the three-factor version fails (Section 4.4).

**Derivation of the rank-one eigenvalue fact.** For nonzero `u` (`m x 1`) and `v` (`n x 1`), consider `R = u v^T` (`m x n`). For any `w` orthogonal to `v`, `R w = u (v^T w) = 0`, so the null space contains the `n-1` dimensional orthogonal complement of `v`; hence `rank(R) <= 1`, and it is exactly 1 when both vectors are nonzero. If `m = n`, then `R v = u (v^T v)`, so `v` is an eigenvector with eigenvalue `||v||^2`, and every vector orthogonal to `v` has eigenvalue zero. Consequently the eigenvalues of a square rank-one matrix are `||v||^2` (once) and zero (`n-1` times), and hence `tr(uv^T) = v^T u` and `det(uv^T) = 0` for `n >= 2`. This single fact powers the outer-product accumulation used in gradient computation and in every low-rank update below.

Now the same four results in the mandated five-part form, with numbers small enough to verify mentally.

```text
[1] FORMULA: (AB)^T = B^T A^T                    [ID: MV-01]
[2] WHERE: A is m x n, B is n x p, entries real or complex; transpose is
           plain transpose (for complex, A^H = conj(A^T) and the same rule
           holds with H replacing T); result is p x m.
[3] CONSTRAINT: the product AB must be defined, i.e. the n in A's columns
           equals the n in B's rows. No invertibility, symmetry or squareness
           is required, so this rule is safe inside any derivation.
[4] NUMERICAL EXAMPLE: A = [[1,2],[3,4],[5,6]] (3x2), B = [[1,0],[0,1]] (2x2).
           AB = [[1,2],[3,4],[5,6]] = A, so (AB)^T = [[1,3,5],[2,4,6]].
           B^T A^T = I * [[1,3,5],[2,4,6]] = [[1,3,5],[2,4,6]]. Agreement.
[5] INTERPRETATION: a transposed product is rewritten without inverting
           anything, at the cost of reversing the multiplication order; a
           gradient written in the wrong order still has the right shape only
           in the square case, which is why the error survives review.
```

```text
[1] FORMULA: ||x||_2^2 = x^T x = sum_i x_i^2      [ID: MV-02]
[2] WHERE: x is n x 1 real; the left side is a scalar length-squared, the
           right side is a 1 x 1 matrix whose single entry is that scalar.
[3] CONSTRAINT: real entries only. For complex x use x^H x. The identity
           is about the square of the norm; x^T x is not itself a norm since
           it fails the triangle inequality (||2x||^2 = 4||x||^2, not 2||x||^2).
[4] NUMERICAL EXAMPLE: x = [3, -4]^T. x^T x = 9 + 16 = 25. ||x||_2 = 5,
           so ||x||_2^2 = 25. Scaling to 2x = [6,-8]^T gives 36+64 = 100 = 4*25.
[5] INTERPRETATION: squared norms are quadratics, so they differentiate
           linearly and are the natural form inside loss functions; taking a
           square root inside a loss changes the gradient magnitude, not the
           minimiser, but it does change conditioning near zero residual.
```

```text
[1] FORMULA: tr(AB) = tr(BA)                      [ID: MV-03]
[2] WHERE: A is m x n, B is n x m; both products are square (m x m and n x n)
           though not necessarily the same size; trace sums diagonal entries.
[3] CONSTRAINT: exactly two factors in a cyclic swap. AB and BA must both be
           defined. Equality of traces does not imply AB = BA, and it does not
           extend to swapping an interior pair of three factors.
[4] NUMERICAL EXAMPLE: A = [[1,2],[3,4]] (2x2), B = [[0,1],[1,0]] (2x2).
           AB = [[2,1],[4,3]], tr = 5. BA = [[3,4],[1,2]], tr = 5. Agreement,
           although AB != BA, which shows the identity is weaker than commutation.
[5] INTERPRETATION: trace is a scalar summary that can be moved across a
           product, which is what makes cycle-consistent gradient derivations
           possible; use it to convert an awkward `sum_i e_i^T M e_i` into
           `tr(M)` and then manipulate the trace.
```

```text
[1] FORMULA: det(AB) = det(A) det(B)              [ID: MV-04]
[2] WHERE: A and B are both n x n over a commutative field; det is the signed
           volume scale of the linear map, defined as sum over permutations of
           signed products of entries.
[3] CONSTRAINT: square and same size. Determinant is multiplicative but not
           additive, and det(A+B) is not det(A)+det(B) in general. For large n
           the product underflows or overflows long before it is inaccurate.
[4] NUMERICAL EXAMPLE: A = [[2,0],[0,3]], B = [[1,1],[0,1]].
           det A = 6, det B = 1, det(AB) = det([[2,2],[0,3]]) = 6. Agreement.
           Note det(A+B) = det([[3,1],[0,4]]) = 12 != 7.
[5] INTERPRETATION: determinant tells you whether a map collapses volume,
           which is a rank question; it does not tell you how close to
           collapse the map is, because a matrix can have det = 1 and be
           catastrophically ill-conditioned.
```

**Boundary for 4.1.** Set `A = [[1,2],[2,4]]`. Then `det(A) = 4 - 4 = 0` and the matrix is singular. Every identity above still holds — transposition, trace cyclicity, determinant multiplicativity — but `A^{-1}` does not exist, and any derivation that reaches for it has silently left the domain of validity. The base case is where you should test such suspicions, because the collapse is visible in one line of arithmetic.

> **Takeaway:** The base case is a laboratory, not a summary; if an identity survives explicit `2 x 2` expansion, the general proof is index bookkeeping, and if it does not, no amount of general notation will save it.

### 4.2 General Case: Products, Traces, Inverses, and the Normal Equations

This subsection states each identity for general shapes, gives the index-level derivation, and then applies the identity to the central engineering object of the sheet: the least-squares normal equations. The derivations are written so that a reader can re-derive them from the definition of the matrix product alone.

**General transpose reversal.** Let `A` be `m x n` and `B` be `n x p`. For any `i` in `1..p` and `j` in `1..m`:

```text
(B^T A^T)[i,j] = sum_k B^T[i,k] A^T[k,j]
               = sum_k B[k,i] A[j,k]              (definition of transpose)
               = sum_k A[j,k] B[k,i]              (scalars commute)
               = (AB)[j,i]                        (definition of product)
               = (AB)^T[i,j]                      (definition of transpose)
```

Since the equality holds for every index pair, the matrices agree. The proof consumes exactly one nontrivial fact, namely that scalar multiplication commutes; that is why the identity depends on the field being commutative and why the analogous statement for quaternionic or matrix-valued entries requires care.

**General trace cyclicity.** For `A` (`m x n`) and `B` (`n x m`):

```text
tr(AB) = sum_i (AB)[i,i] = sum_i sum_k A[i,k] B[k,i]
tr(BA) = sum_k (BA)[k,k] = sum_k sum_i B[k,i] A[i,k]
```

The two double sums contain the same set of products `A[i,k]B[k,i]`; only the order of summation and the names of the dummy indices differ. Finite sums may be reordered freely, so the values agree. The step that fails for three factors is the renaming: `tr(ABC)` collects products `A[i,k]B[k,l]C[l,i]` and `tr(ACB)` collects `A[i,k]C[k,l]B[l,i]`, and there is no index renaming that maps one family onto the other, because `B` and `C` occupy different positions in the chain.

**General inverse of a product.** Suppose `A` is `n x n` and `B` is `n x n`, both invertible. Then

```text
(AB)(B^{-1}A^{-1}) = A (B B^{-1}) A^{-1} = A I A^{-1} = A A^{-1} = I
(B^{-1}A^{-1})(AB) = B^{-1}(A^{-1}A)B = B^{-1} I B = I
```

so `B^{-1}A^{-1}` satisfies the two defining equations of the inverse of `AB`, and inverses are unique. Note what the derivation uses: associativity, the existence of both inverses, and nothing else. Note what it does *not* give: any reason to believe `(AB)^{-1} = A^{-1}B^{-1}`, which would require `A^{-1}` and `B^{-1}` to commute. Two random invertible `2 x 2` matrices almost never do.

```text
[1] FORMULA: (AB)^{-1} = B^{-1} A^{-1}            [ID: MV-05]
[2] WHERE: A, B both n x n and invertible over a field; the order of the
           factors on the right is reversed relative to the left.
[3] CONSTRAINT: both factors invertible. A product of singular matrices is
           singular, but a product may be singular even when neither factor
           is singular only in floating point, not in exact arithmetic; what
           matters numerically is conditioning, discussed in Section 8.
[4] NUMERICAL EXAMPLE: A = [[2,1],[1,1]], so A^-1 = [[1,-1],[-1,2]].
           B = [[1,2],[0,1]], so B^-1 = [[1,-2],[0,1]].
           AB = [[2,5],[1,3]], det = 1, so (AB)^-1 = [[3,-5],[-1,2]].
           B^-1 A^-1 = [[1,-2],[0,1]] * [[1,-1],[-1,2]]
                     = [[1+2, -1-4], [-1, 2]] = [[3,-5],[-1,2]]. Agreement.
[5] INTERPRETATION: this is the rule that changes a "solve then transform"
           pipeline into a "transform the cheap side" pipeline; forgetting the
           reversal produces a matrix that is wrong but usually still
           invertible, so no exception is raised and the error is silent.
```

**Derivation of the forward-model transpose rule.** In supervised learning the model is often `y_hat = X w` with `X` (`m x d`) and `w` (`d x 1`). Then `y_hat^T = w^T X^T` by MV-01, which is `1 x m`. The scalar loss `L = sum_i (y_hat_i - y_i)^2` can be written `L = (Xw - y)^T (Xw - y)`. Applying distributivity and MV-01:

```text
L = (Xw)^T(Xw) - (Xw)^T y - y^T(Xw) + y^T y
  = w^T X^T X w - w^T X^T y - y^T X w + y^T y
```

The two middle terms are scalars, and a scalar equals its own transpose: `(w^T X^T y)^T = y^T X w`, so they are equal and sum to `-2 w^T X^T y`. Hence

```text
L(w) = w^T (X^T X) w - 2 w^T (X^T y) + y^T y          [ID: MV-06]
```

This is a quadratic in `w` with Hessian `2 X^T X`. Because `X^T X` is symmetric PSD (Section 4.2, PSD lemma below), the quadratic is convex, and a stationary point is a global minimiser — a fact used constantly and stated as a theorem in Section 10.

**PSD lemma and derivation.** For any real `X` (`m x d`), the Gram matrix `G = X^T X` satisfies, for every `z` in `R^d`,

```text
z^T G z = z^T X^T X z = (Xz)^T (Xz) = ||Xz||_2^2 >= 0
```

and `G^T = (X^T X)^T = X^T X` shows symmetry. So `G` is symmetric PSD. Moreover `z^T G z = 0` forces `Xz = 0` because a sum of squares is zero only if each square is zero, so the null space of `G` equals the null space of `X`. That equality is the reason rank deficiency of `X` and singularity of `G` are the same statement, and it is what makes the pseudoinverse route legitimate.

**Stationarity and the normal equations.** Differentiating MV-06 with respect to the vector `w` (Section 4.3 states the two vector-calculus facts used), the gradient is

```text
dL/dw = 2 X^T X w - 2 X^T y                          [ID: MV-07]
```

Setting the gradient to zero gives the normal equations

```text
X^T X w = X^T y                                      [ID: MV-08]
```

Two readings of MV-08 are worth holding simultaneously. Algebraically, it is a `d x d` linear system whose coefficient matrix is the Gram matrix and whose right-hand side is the correlation of features with targets. Geometrically, it says the residual `r = y - Xw` satisfies `X^T r = 0`, i.e. the residual is orthogonal to every column of `X`. The geometric reading is the one that generalises and the one that gives a free numerical test, since `X^T r` is cheap to compute and must vanish to rounding.

**Projection matrix and its two properties.** When `X` has full column rank, MV-08 has the unique solution `w = (X^T X)^{-1} X^T y`, and the fitted vector is

```text
y_hat = X w = X (X^T X)^{-1} X^T y = P y,   P = X (X^T X)^{-1} X^T   [ID: MV-09]
```

Symmetry follows from transposing `P`: `P^T = X ((X^T X)^{-1})^T X^T = X (X^T X)^{-1} X^T = P`, using that the inverse of a symmetric matrix is symmetric and that `X^T X` is symmetric. Idempotence follows from associativity:

```text
P^2 = X (X^T X)^{-1} (X^T X) (X^T X)^{-1} X^T = X (X^T X)^{-1} X^T = P
```

and the middle cancellation is exactly the `n`-shaped collapse of the inner factor to `I_d`. Finally `tr(P) = tr(X(X^T X)^{-1}X^T) = tr((X^T X)^{-1}X^T X) = tr(I_d) = d` by MV-03, which gives a free diagnostic: if a computed projector's trace is not `d`, either the rank is not `d` or the computation is wrong.

```text
[1] FORMULA: X^T X w = X^T y, with P = X(X^T X)^{-1}X^T   [ID: MV-10]
[2] WHERE: X is m x d (m observations, d parameters), y is m x 1, w is d x 1;
           P is m x m; residual r = y - Xw is m x 1 in the same units as y.
[3] CONSTRAINT: the normal equations always have a solution because X^T y lies
           in the column space of X^T X, but the solution is unique only when
           rank(X) = d. P is defined by the displayed formula only in that
           full-rank case; otherwise use the SVD-based projector U U^T on the
           leading rank left singular vectors.
[4] NUMERICAL EXAMPLE: see Section 5 for the full four-point fit; there
           X^T X = [[4,10],[10,30]], X^T y = [10.0, 29.9], giving
           w = (0.05, 0.98)^T and residuals summing to zero.
[5] INTERPRETATION: the equations convert a fit into a linear solve of size
           d, which is why the cost of least squares is governed by d (the
           feature count) for the solve and by m*d^2 for the Gram formation.
```

**Centring identity.** A frequently needed rewrite: if `X` has columns `x_j` and `1` denotes a column of ones, the fitted intercept is `mean(y) - sum_j w_j mean(x_j)` whenever an intercept column is present. Derivation: the first normal equation corresponding to the ones column reads `sum_i (y_i - w_0 - sum_j w_j x_ij) = 0`, i.e. the residuals sum to zero; substituting the means and rearranging gives the stated form. This is what makes "center then fit without intercept" equivalent to "fit with intercept" for the slope coefficients, and it is the algebraic justification for the centring step used in the conditioning remedy of Section 8.

> **Takeaway:** The normal equations are legitimate algebra with a weak uniqueness condition and a strong numerical weakness; keep both facts attached to the formula wherever it is quoted.

### 4.3 Variants: Factorisations, Regularisation, and Low-Rank Updates

The variants below are the ones that appear in production code. Each is a rewrite of the same least-squares problem into a different arithmetic route, and the routes differ in cost, in stability, and in what they can say when the problem is rank deficient.

**Variant A — Cholesky of the Gram matrix.** If `G = X^T X` is symmetric positive definite, it factors uniquely as `G = L L^T` with `L` lower triangular and positive diagonal. Substituting into MV-08 gives `L L^T w = X^T y`; solving `L z = X^T y` by forward substitution and then `L^T w = z` by back substitution costs `O(d^2)` after the `O(d^3/3)` factorisation. The algorithm is roughly half the work of an unpivoted LU factorisation and needs no pivoting, which is why it is the default for a well-conditioned Gram matrix.

```text
[1] FORMULA: G = L L^T, then solve L z = X^T y and L^T w = z   [ID: MV-11]
[2] WHERE: G = X^T X is d x d symmetric positive definite; L is d x d lower
           triangular with L[i,i] > 0; z is an intermediate d x 1 vector.
[3] CONSTRAINT: positive definiteness, which requires full column rank of X.
           Cholesky fails, in exact arithmetic, at the first pivot that is not
           positive; in floating point it may complete on a matrix whose
           smallest eigenvalue is negative but larger than roundoff, so a
           successful factorisation is not proof of definiteness.
[4] NUMERICAL EXAMPLE: G = [[4,10],[10,30]]. L[1,1] = sqrt(4) = 2;
           L[2,1] = 10/2 = 5; L[2,2] = sqrt(30 - 25) = sqrt(5) = 2.2360679...
           Check L L^T = [[4,10],[10,25+5]] = [[4,10],[10,30]]. Correct.
[5] INTERPRETATION: on an ill-conditioned Gram matrix the factorisation
           squares the condition number already present in X, so Cholesky is
           the fast route and not the safe one.
```

**Variant B — QR of the design matrix.** Factor `X = Q R` with `Q` (`m x d`) having orthonormal columns and `R` (`d x d`) upper triangular. Then

```text
||Xw - y||^2 = ||QRw - y||^2 = ||Rw - Q^T y||^2 + ||(I - QQ^T)y||^2
```

because `Q` preserves lengths and the two summands are orthogonal: the first lies in the column space of `Q` and the second is orthogonal to it by construction of the projector. The second term does not involve `w`, so minimising reduces to solving the triangular system `R w = Q^T y`, which costs `O(d^2)` after the `O(m d^2)` factorisation. The key numerical gain is that `Q` is orthogonal, hence perfectly conditioned, so the condition number seen by the solve is that of `R`, which equals that of `X` rather than its square.

```text
[1] FORMULA: X = QR; solve R w = Q^T y             [ID: MV-12]
[2] WHERE: X is m x d with m >= d and full column rank; Q is m x d with
           Q^T Q = I_d; R is d x d upper triangular with positive diagonal
           under a sign convention; residual norm is ||(I - QQ^T)y||.
[3] CONSTRAINT: full column rank for the uniqueness of w. With Householder
           reflections or Givens rotations the method is backward stable in
           the normwise sense; classical Gram-Schmidt is not, and modified
           Gram-Schmidt is better but still weaker than Householder.
[4] NUMERICAL EXAMPLE: X = [[1,1],[1,2],[1,3],[1,4]]. A QR by hand is
           shown in Section 5.4; the resulting R has det(R) = det(Q)^-1
           det(X), and |det(Q)| = 1 for a square Q, which is the invariant
           used there to check the factorisation.
[5] INTERPRETATION: QR is the default recommendation for dense least squares
           because it pays a factor of about two in flops relative to the
           Gram route in exchange for not squaring the condition number.
```

**Variant C — SVD and the pseudoinverse.** Any real `X` (`m x d`) factors as `X = U Sigma V^T` with `U` (`m x r` effective) and `V` (`d x r`) orthonormal and `Sigma` diagonal with `sigma_1 >= ... >= sigma_r > 0`. Then the minimum-norm least-squares solution is

```text
w = X^+ y = V Sigma^+ U^T y = sum_{i=1..r} (u_i^T y / sigma_i) v_i   [ID: MV-13]
```

The derivation is a change of basis: writing `w = V a` for some `a` and using orthogonality, `||Xw - y||^2 = ||Sigma a - U^T y||^2` plus a term independent of `a`; the residual is minimised coordinate-wise by `a_i = (u_i^T y)/sigma_i` for `i <= r` and `a_i = 0` otherwise, and the choice `a_i = 0` in the null directions is precisely what makes `w` the shortest minimiser. Truncating the sum at `k < r` gives the rank-`k` approximation, whose error in the Frobenius norm is `sqrt(sum_{i>k} sigma_i^2)` by the Eckart–Young theorem.

```text
[1] FORMULA: w = sum_i (u_i^T y / sigma_i) v_i, truncated at k   [ID: MV-13]
[2] WHERE: u_i are left singular vectors (m x 1), v_i right singular vectors
           (d x 1), sigma_i singular values in descending order; k is the
           retained rank; the solution lies in the span of v_1..v_k.
[3] CONSTRAINT: valid for any X including rank-deficient ones, provided the
           truncation is declared. Small sigma_i amplify noise in u_i^T y by
           1/sigma_i, so a truncation threshold is a modelling decision, not
           a detail.
[4] NUMERICAL EXAMPLE: X = [[1,0],[0,1],[1,1]] from the original draft.
           X^T X = [[2,1],[1,2]] has eigenvalues 3 and 1, so sigma = sqrt(3),
           1. kappa = sqrt(3) = 1.732. Retaining k = 2 reproduces the exact
           solution; retaining k = 1 would force w parallel to the leading
           singular direction, a strictly different model.
[5] INTERPRETATION: SVD is the diagnostic and the fallback in one object; it
           answers "what rank is this really" and "what is the best answer at
           this rank" without a separate analysis.
```

**Variant D — Ridge (Tikhonov) regularisation.** Replace the objective with `||Xw - y||^2 + tau ||w||^2` for `tau > 0`. Expanding gives `(X^T X + tau I) w = X^T y`, and the coefficient matrix is positive definite for every `tau > 0` even when `X` is rank deficient, because `z^T (X^T X + tau I) z = ||Xz||^2 + tau ||z||^2 >= tau ||z||^2 > 0` for nonzero `z`.

```text
[1] FORMULA: (X^T X + tau I) w = X^T y             [ID: MV-14]
[2] WHERE: tau > 0 has the units of X^T X, i.e. squared feature units if the
           model is not standardised; I is d x d; the penalty is on the
           coefficient vector, not on the intercept unless one is included.
[3] CONSTRAINT: strictly positive tau. As tau falls to zero the solution
           approaches the minimum-norm solution when X is rank deficient and
           the ordinary solution when it is not; as tau grows the solution
           approaches zero. Penalising scaled and unscaled features is not
           the same model.
[4] NUMERICAL EXAMPLE: with X^T X = [[4,10],[10,30]], X^T y = [10.0, 29.9]
           and tau = 1, the matrix becomes [[5,10],[10,31]], det = 155 - 100 =
           55, giving w = (11/55, 49.5/55) = (0.2, 0.9): the slope is pulled
           from 0.98 to 0.9 while the intercept rises from 0.05 to 0.2 because
           the penalty here applies to the intercept as well.
[5] INTERPRETATION: ridge trades a bias toward smaller coefficients for a
           large reduction in variance when columns are nearly collinear;
           the trace of the "hat" matrix sum_i h_ii = tr(X(X^TX+tau I)^{-1}X^T)
           falls below d, which quantifies the effective degrees of freedom lost.
```

**Variant E — Sherman–Morrison for a rank-one change.** If `A` is invertible and `u, v` are vectors with `1 + v^T A^{-1} u != 0`, then

```text
(A + u v^T)^{-1} = A^{-1} - (A^{-1} u v^T A^{-1}) / (1 + v^T A^{-1} u)   [ID: MV-15]
```

Verification is by direct multiplication: multiply the right side by `A + uv^T` and confirm the cross terms cancel to `I`. The cost is `O(n^2)` for the update given `A^{-1}`, versus `O(n^3)` for a fresh inversion, which is why streaming or leave-one-out recomputation uses it. For a *downdate*, replace `+u v^T` by `-u u^T` and the denominator becomes `1 - u^T A^{-1} u`, which is where the stability caveat enters: if that scalar is near zero, the update corresponds to a nearly singular result and the identity is numerically useless even though it is algebraically exact.

```text
[1] FORMULA: (A + u v^T)^{-1} = A^{-1} - A^{-1}u v^T A^{-1}/(1 + v^T A^{-1}u)
                                                              [ID: MV-15]
[2] WHERE: A is n x n invertible; u, v are n x 1; the denominator is a scalar
           in the field; v^T A^{-1} u has no particular units when A does not.
[3] CONSTRAINT: A invertible and the scalar denominator nonzero; numerically
           the denominator must also be bounded away from zero relative to
           the scale of the terms, otherwise cancellation destroys accuracy.
[4] NUMERICAL EXAMPLE: A = [[4,10],[10,30]] with A^-1 = [[1.5,-0.5],[-0.5,0.2]]
           and downdate vector u = [1,4]^T. Then A^-1 u = [-0.5, 0.3]^T and
           u^T A^-1 u = -0.5 + 1.2 = 0.7, so the downdate denominator is
           1 - 0.7 = 0.3. Adding ([-0.5,0.3]^T[-0.5,0.3])/0.3 to A^-1 gives
           [[2.3333,-1.0],[-1.0,0.5]]. Direct inversion of [[3,6],[6,14]]
           gives (1/6)[[14,-6],[-6,3]] = the same matrix. Verified.
[5] INTERPRETATION: the update reproduces the inverse of the matrix formed by
           deleting one observation, which is exactly the leave-one-out
           computation used to detect influential points without refitting.
```

**Variant F — Woodbury for a low-rank block change.** Generalising MV-15, for `U` (`n x k`), `C` (`k x k`), `V` (`k x n`) with all inverses existing,

```text
(A + U C V)^{-1} = A^{-1} - A^{-1} U (C^{-1} + V A^{-1} U)^{-1} V A^{-1}   [ID: MV-16]
```

Setting `k = 1` recovers Sherman–Morrison, and setting `C = -I` gives the block-downdate used when the penalty of a subset of features is removed. The engineering point is the dimension of the inverted object: the right side inverts a `k x k` matrix instead of an `n x n` one, so the update is cheap exactly when the change has low rank.

**Variant G — block inverse via the Schur complement.** For a partitioned invertible matrix

```text
M = [[A, B], [C, D]]      with A invertible and S = D - C A^{-1} B invertible
```

the inverse is

```text
M^{-1} = [[A^{-1} + A^{-1} B S^{-1} C A^{-1},  -A^{-1} B S^{-1}],
          [-S^{-1} C A^{-1},                    S^{-1}          ]]   [ID: MV-17]
```

The Schur complement `S` is the object that answers "what does the `D` block look like after the `A` block has explained what it can", which is the same question as partialling out a set of features in a regression. A common practical use is inverting a covariance matrix that is given in blocks, where inverting two smaller blocks plus `S` is cheaper and more interpretable than inverting the whole.

**Variant H — Kronecker and vectorisation.** For `X` (`m x n`) and the row-stacking convention,

```text
vec(A X B) = (B^T (x) A) vec(X)                    [ID: MV-18]
```

with `vec` stacking columns. This linearises a matrix equation into an ordinary linear system, which is how Lyapunov and Sylvester equations are solved numerically as large sparse systems. The identity is exact but the resulting system is `mn x mn`, so it is used for structure and analysis far more often than for direct computation.

**Variant I — the matrix-determinant lemma.** Following from the same algebra as MV-15,

```text
det(A + u v^T) = det(A) (1 + v^T A^{-1} u)          [ID: MV-19]
```

and its block form `det([[A,B],[C,D]]) = det(A) det(D - C A^{-1} B)`. The second form makes the Schur complement's role as a volume ratio explicit: the determinant of the whole equals the determinant of the first block times the determinant of what remains after conditioning on it.

> **Takeaway:** Every variant is the same minimisation expressed in a different arithmetic; choose by the property you need preserved — definiteness, conditioning, rank awareness, or update cost — not by familiarity.

### 4.4 Limits, Counterexamples, and Where the Identities Stop

The most valuable part of a formula sheet is the list of situations in which the formula is not applicable, because that is where experienced practitioners still make mistakes. Each item below is a genuine limit, stated with the smallest counterexample that shows it.

**Limit 1 — trace is not invariant under arbitrary permutation.** It is tempting to generalise `tr(AB) = tr(BA)` to `tr(ABC) = tr(ACB)`. Take

```text
A = [[0,1],[0,0]]   B = [[0,0],[1,0]]   C = [[1,0],[0,2]]
```

Compute `AB = [[1,0],[0,0]]`, so `ABC = [[1,0],[0,0]] [[1,0],[0,2]] = [[1,0],[0,0]]`, trace 1. Compute `AC = [[0,2],[0,0]]`, so `ACB = [[0,2],[0,0]] [[0,0],[1,0]] = [[2,0],[0,0]]`, trace 2. The two traces differ. The general statement that survives is `tr(ABC) = tr(BCA) = tr(CAB)`, which are the cyclic rotations, and that is why gradient derivations insist on keeping a consistent cyclic order rather than freely reordering factors.

**Limit 2 — `(AB)^{-1} = A^{-1}B^{-1}` is false.** With `A` and `B` from the MV-05 example, `A^{-1}B^{-1} = [[1,-1],[-1,2]][[1,-2],[0,1]] = [[1,-3],[-1,4]]`, whereas the true inverse is `[[3,-5],[-1,2]]`. Both matrices are invertible, so no exception is raised. The counterexample is small enough to run in a test, which is why the fix for F3 in Section 8 is a unit test rather than a review process.

**Limit 3 — the condition number of the Gram matrix is the square of the condition number of the design matrix.** For any matrix, `kappa_2(X^T X) = sigma_max(X^T X)/sigma_min(X^T X) = sigma_max(X)^2/sigma_min(X)^2 = kappa_2(X)^2`. In the Section 5 example, `kappa_2(X) = 7.469` and `kappa_2(X^T X) = 55.78`. This is not a rounding detail: with `kappa(X) = 1e6`, the Gram matrix has condition `1e12`, and in double precision with unit roundoff `1.1e-16` the solution retains about `16 - 12 = 4` digits. Squaring is the mathematical price of the normal-equation route, and it is paid silently.

**Limit 4 — determinant is a poor singularity test.** `A(t) = [[1, 1], [1, 1+t]]` has `det = t`. At `t = 1e-12` the determinant is `1e-12`, which looks small, yet the reciprocal condition number is about `5e-13` and the condition number about `4e12` — so the determinant's smallness is roughly proportional to the ill-conditioning only because the other entries are of order one, a coincidence of this scaling. Conversely `diag(1e-8, 1e-8, 1e-8, 1e-8)` has determinant `1e-32`, which is astronomically small, while the matrix is perfectly conditioned with `kappa = 1`. Determinant scales as `c^n` under scaling by `c` and carries the units of the entries to the `n`-th power, so it cannot serve as a scale-free singularity indicator. Use `sigma_min/sigma_max` instead.

**Limit 5 — `x^T A x` sees only the symmetric part.** Write `A = (A + A^T)/2 + (A - A^T)/2`. The antisymmetric part `K` satisfies `x^T K x = 0` for every `x`, because the scalar equals its own transpose and transposing flips the sign. Numerically, with `A = [[0,5],[-5,0]]`, the quadratic form is `0` for every `x`, yet `A` is not zero. Hence a gradient that uses a non-symmetric `A` where a symmetric one is required may be silently missing the entire contribution of the antisymmetric part — which is exactly what happens when a hand-written Hessian omits a symmetrisation step.

**Limit 6 — orthogonal does not mean independent.** `Q` with `Q^T Q = I` has orthonormal columns, but "orthogonal" is a statement about the inner product of the *columns as vectors*, not about statistical independence of the underlying variables. Two features can be exactly orthogonal in the sample and strongly dependent in the population; conversely, dependent variables produce non-orthogonal columns. Any inference about independence requires a probabilistic model, which is outside this sheet.

**Limit 7 — associativity fails in floating point.** Exact matrix multiplication is associative, but computed multiplication is not. A dramatic constructed case: with `u = 1e-8` and entries arranged so that a large intermediate cancels, computing `A(BC)` and `(AB)C` can differ in the last several digits, and the difference is not symmetric in the operands. The practical consequence is that changing the order of a chain of matrix products to save flops may change the result, which is why numerical tests should compare against a tolerance derived from the condition number rather than demanding bitwise equality.

**Limit 8 — the normal equations are valid at any rank but not uniquely solvable.** With `X` from the duplicate-column example in Section 5.6, `X^T X` is singular and MV-08 has infinitely many solutions. The identity `X^T X w = X^T y` remains true of every one of them. What fails is only the step "therefore `w = (X^T X)^{-1} X^T y`", which assumed invertibility. This is the cleanest illustration in the sheet of the difference between an identity and its algebraic consequence.

**Limit 9 — pseudoinverse solutions depend on the norm chosen.** The minimum-norm solution minimises `||w||_2` among minimisers of the residual. If the coefficients are measured on different scales, the minimum-Euclidean-norm answer is arbitrary with respect to units, so the choice of scaling changes which solution is selected. Any reported coefficient from a rank-deficient fit must therefore state the norm and the scaling, or it is not reproducible.

**Limit 10 — eigenvalue perturbation is not bounded by the norm of the perturbation in general.** For normal matrices the bound `|lambda_i(A+E) - lambda_i(A)| <= ||E||_2` holds (Weyl). For non-normal matrices the eigenvalues can move by far more than `||E||`, which is the essence of pseudospectral effects. This matters here because `X^T X` is symmetric and therefore normal, so the Gram route's eigenvalue perturbations *are* bounded; it is the *inverse* computation, which involves `1/lambda`, that amplifies them.

```text
[1] FORMULA: kappa_2(X^T X) = kappa_2(X)^2          [ID: MV-20]
[2] WHERE: kappa_2 denotes the ratio of largest to smallest singular value;
           X is m x d with full column rank; the statement compares the
           conditioning of the d x d Gram matrix to that of X itself.
[3] CONSTRAINT: full column rank so that the smallest singular value is
           positive. The statement is about exact singular values; the
           computed ones inherit the accuracy of the SVD routine used.
[4] NUMERICAL EXAMPLE: X = [[1,1],[1,2],[1,3],[1,4]] gives sigma_max =
           5.7794, sigma_min = 0.7738, so kappa(X) = 7.469 and kappa(X^T X)
           = 55.78 = 7.469^2. Check: eigenvalues of X^T X = [[4,10],[10,30]]
           are (34 +/- sqrt(1076))/2 = 33.4012 and 0.5988; ratio 55.78.
[5] INTERPRETATION: this single identity explains why the Gram route loses
           twice as many digits as QR, and why reporting kappa(X^T X) as if it
           were the conditioning of the problem overstates the difficulty by
           its own square root.
```

> **Takeaway:** An identity's domain matters more than its statement; each limit above is a small counterexample you can compute by hand, which is the only kind of guard that survives a deadline.

## 5. Worked Numerical Example

**Evidence:** ILLUSTRATIVE constructed data. The four observations were chosen so that every intermediate is a terminating decimal or a recognisable surd, which makes the arithmetic checkable without a machine. All numbers below were recomputed by hand and then cross-checked by two routes that do not share arithmetic with the first (Section 5.3). Nothing here is a benchmark, a dataset, or an observed production result.

### 5.1 Problem statement and data

Four observations of one predictor and one target, with an intercept:

```text
i :   1      2      3      4
x :  1.0    2.0    3.0    4.0
y :  1.05   1.95   3.05   3.95
```

The model is `y_hat_i = w0 + w1 x_i`. The parameter vector is `w = (w0, w1)^T` in that order. The design matrix stacks a leading ones column and the predictor column:

```text
        [ 1.0  1.0 ]
X  =    [ 1.0  2.0 ]        y  =  (1.05, 1.95, 3.05, 3.95)^T
        [ 1.0  3.0 ]
        [ 1.0  4.0 ]
```

So `m = 4`, `d = 2`, `X` is `4 x 2`, `y` is `4 x 1`. Units: suppose `x` is in "months since launch" and `y` is in "thousand orders per day"; then `w1` has units of thousand orders per day per month and `w0` has units of thousand orders per day. The units are stated because a regularised fit without units can penalise a coefficient that is small only because its unit is large — failure mode F7.

### 5.2 Forming the normal equations

The Gram matrix is built from column sums. Write `c0 = (1,1,1,1)^T` and `c1 = (1,2,3,4)^T`. Then

```text
G[1,1] = c0 . c0 = 1+1+1+1                       = 4
G[1,2] = c0 . c1 = 1+2+3+4                       = 10
G[2,1] = c1 . c0 = 10                            (symmetry, free to reuse)
G[2,2] = c1 . c1 = 1+4+9+16                      = 30
```

so `X^T X = [[4, 10], [10, 30]]`. The right-hand side:

```text
(X^T y)[1] = c0 . y = 1.05 + 1.95 + 3.05 + 3.95  = 10.00
(X^T y)[2] = c1 . y = 1*1.05 + 2*1.95 + 3*3.05 + 4*3.95
                    = 1.05 + 3.90 + 9.15 + 15.80 = 29.90
```

The system is therefore

```text
4 w0 + 10 w1 = 10.00                                            (E1)
10 w0 + 30 w1 = 29.90                                           (E2)
```

### 5.3 Solving three ways, and why three

**Route 1 — elimination by hand.** Divide (E1) by 4: `w0 = 2.50 - 2.50 w1`. Substitute into (E2):

```text
10 (2.50 - 2.50 w1) + 30 w1 = 29.90
25.00 - 25.00 w1 + 30 w1 = 29.90
5.00 w1 = 4.90
w1 = 0.98
w0 = 2.50 - 2.50 * 0.98 = 2.50 - 2.45 = 0.05
```

So `w = (0.05, 0.98)^T`. The slope is 0.98 thousand orders per day per month and the intercept is 0.05 thousand orders per day. Because the data were constructed to sit almost exactly on a line, the intercept is small and the slope is close to 1; that is a property of the construction, not a discovery about any business.

**Route 2 — explicit inverse and MV-09.** `det(X^T X) = 4*30 - 10*10 = 120 - 100 = 20`. The inverse of a `2 x 2` matrix `[[a,b],[c,d]]` is `(1/det)[[d,-b],[-c,a]]`, so

```text
(X^T X)^{-1} = (1/20) [[30, -10], [-10, 4]] = [[1.5, -0.5], [-0.5, 0.2]]
```

and

```text
w = (X^T X)^{-1} X^T y = [[1.5, -0.5], [-0.5, 0.2]] [10.00, 29.90]^T
  w0 = 1.5*10.00 - 0.5*29.90 = 15.00 - 14.95 = 0.05
  w1 = -0.5*10.00 + 0.2*29.90 = -5.00 + 5.98 = 0.98
```

The two routes agree to the last shown digit. Agreement between elimination and the explicit inverse shares no arithmetic beyond the Gram entries themselves, so it is a real check on the solve, though not on the Gram formation.

**Route 3 — QR, with the factorisation shown.** Orthogonalise by classical Gram–Schmidt, which is adequate here because the two columns are far from dependent:

```text
||c0|| = sqrt(4) = 2                      q1 = c0/2 = (0.5, 0.5, 0.5, 0.5)^T
r11 = 2
r12 = q1 . c1 = 0.5*(1+2+3+4) = 5.0
v = c1 - 5 q1 = (1-2.5, 2-2.5, 3-2.5, 4-2.5) = (-1.5, -0.5, 0.5, 1.5)
||v|| = sqrt(2.25 + 0.25 + 0.25 + 2.25) = sqrt(5) = 2.2360679...
q2 = v/||v|| = (-0.6708204, -0.2236068, 0.2236068, 0.6708204)
r22 = ||v|| = 2.2360679
```

Check on the factorisation itself: `det(X^T X) = det(R)^2 = (r11*r22)^2 = (2*2.2360679)^2 = (4.4721359)^2 = 20.00`, matching the determinant computed from the `2 x 2` formula. That check is decisive because it uses the product structure `X^T X = R^T Q^T Q R = R^T R`.

Now transform the right-hand side:

```text
(Q^T y)[1] = q1 . y = 0.5 * 10.00 = 5.0000
(Q^T y)[2] = q2 . y = (-0.6708204)(1.05) + (-0.2236068)(1.95)
                    + (0.2236068)(3.05) + (0.6708204)(3.95)
              = -0.7043614 - 0.4360333 + 0.6820007 + 2.6497406
              = 2.1913466
```

and solve the triangular system `R w = Q^T y`:

```text
r22 w1 = 2.1913466        ->  w1 = 2.1913466 / 2.2360679 = 0.9800000
r11 w0 + r12 w1 = 5.0     ->  2 w0 + 5.0*0.98 = 5.0  ->  2 w0 = 0.10  ->  w0 = 0.05
```

All three routes give `(0.05, 0.98)`. The QR route is the one that generalises: it never formed a Gram matrix explicitly, so it never squared the conditioning, and its accuracy does not degrade on the near-collinear cases of Section 5.6.

### 5.4 Predictions, residuals, and orthogonality

Fitted values `y_hat_i = 0.05 + 0.98 x_i`:

```text
x = 1  ->  0.05 + 0.98 = 1.03     e = 1.05 - 1.03 =  0.02
x = 2  ->  0.05 + 1.96 = 2.01     e = 1.95 - 2.01 = -0.06
x = 3  ->  0.05 + 2.94 = 2.99     e = 3.05 - 2.99 =  0.06
x = 4  ->  0.05 + 3.92 = 3.97     e = 3.95 - 3.97 = -0.02
```

The residual vector is `e = (0.02, -0.06, 0.06, -0.02)^T`. Now check orthogonality, which is the geometric content of MV-08:

```text
c0 . e =  0.02 - 0.06 + 0.06 - 0.02 = 0.00                 (residuals sum to zero)
c1 . e =  0.02 - 0.12 + 0.18 - 0.08 = 0.00                 (residual orthogonal to x)
SSE = 0.0004 + 0.0036 + 0.0036 + 0.0004 = 0.0080
RMSE = sqrt(0.0080/4) = sqrt(0.0020) = 0.0447214
```

Both inner products vanish, as the normal equations demand. The residuals are also antisymmetric about the centre of the design, which is what makes the orthogonality exact in four decimal places rather than merely approximate; a symmetric design with symmetric errors is a convenience of the construction and would not occur in messy data.

The total sum of squares about the mean is computed for the coefficient of determination:

```text
mean(y) = 10.00/4 = 2.50
y - mean = (-1.45, -0.55, 0.55, 1.45)
SST = 2.1025 + 0.3025 + 0.3025 + 2.1025 = 4.81
R^2 = 1 - SSE/SST = 1 - 0.0080/4.81 = 1 - 0.0016632 = 0.9983368
```

An `R^2` of 0.9983 on four constructed points describes the construction. On real data with `m = 4` the same number would be nearly meaningless, and it is not evidence that the model is correct — a point developed as failure mode F8.

### 5.5 The projection matrix, and the trace check for free

Using `P = X (X^T X)^{-1} X^T` with `(X^T X)^{-1} = [[1.5, -0.5], [-0.5, 0.2]]`, entry `(i,j)` is

```text
P[i,j] = [1, x_i] M [1, x_j]^T = 1.5 - 0.5(x_i + x_j) + 0.2 x_i x_j
```

which yields the symmetric `4 x 4` matrix

```text
P = [[ 0.7,  0.4,  0.1, -0.2],
     [ 0.4,  0.3,  0.2,  0.1],
     [ 0.1,  0.2,  0.3,  0.4],
     [-0.2,  0.1,  0.4,  0.7]]
```

Three properties can be verified by inspection, and each is a check that costs arithmetic rather than trust:

```text
trace = 0.7 + 0.3 + 0.3 + 0.7 = 2.0 = d                      (rank check, MV-03)
row 1 sum = 0.7 + 0.4 + 0.1 - 0.2 = 1.0                      (P * 1 = 1)
row 4 sum = -0.2 + 0.1 + 0.4 + 0.7 = 1.0
y_hat = P y :  row 1 = 0.7(1.05) + 0.4(1.95) + 0.1(3.05) - 0.2(3.95)
                   = 0.735 + 0.780 + 0.305 - 0.790 = 1.030   (matches 5.4)
diagonal entries h_ii = (0.7, 0.3, 0.3, 0.7), sum 2.0, each in [0,1]
```

The diagonal entries of `P` are the leverages. Their sum is exactly the number of parameters, so the average leverage is `d/m = 0.5`; here the two endpoint observations carry 0.7 each and the interior ones 0.3, which is the algebraic statement that a straight-line fit is determined disproportionately by the extremes of the predictor range. That is a substantive fact about this model, and it is a fact a reader can recover from the matrix rather than from folklore.

### 5.6 Conditioning, computed rather than asserted

The eigenvalues of `X^T X = [[4,10],[10,30]]` follow from trace and determinant: sum is 34, product is 20, so

```text
lambda = (34 +/- sqrt(34^2 - 4*20)) / 2 = (34 +/- sqrt(1076)) / 2
sqrt(1076) = 32.8024389
lambda_max = (34 + 32.8024389)/2 = 33.4012195
lambda_min = (34 - 32.8024389)/2 =  0.5987805
check: 33.4012195 + 0.5987805 = 34.0000   (trace)
check: 33.4012195 * 0.5987805 = 20.0000   (determinant)
```

Since `X^T X` is symmetric PSD, its singular values are its eigenvalues, so

```text
sigma_max(X) = sqrt(33.4012195) = 5.7793788
sigma_min(X) = sqrt( 0.5987805) = 0.7738091
kappa_2(X)      = 5.7793788 / 0.7738091 = 7.468740
kappa_2(X^T X)  = 33.4012195 / 0.5987805 = 55.78181 = 7.468740^2
```

This confirms MV-20 numerically on the running example. What does `kappa = 7.47` mean operationally? In double precision with unit roundoff `u = 1.1e-16`, a standard forward-error bound for solving a linear system is of order `kappa * u`. Here that is `7.47 * 1.1e-16 ≈ 8.2e-16`, so relative error in `w` from rounding is at the level of the last digit of double precision — utterly negligible. If the predictor were measured in a way that made the columns nearly parallel, with `kappa(X) = 1e8`, the Gram route would face `kappa = 1e16` and the computed coefficients would have essentially no correct digits at all, while the QR route would still retain about `16 - 8 = 8` digits. That contrast is the entire practical argument for QR.

**Rank deficiency, constructed on the same data.** Replace the second column by twice the first: `c1' = 2 * c0 = (2,2,2,2)`. Then

```text
X'^T X' = [[4, 8], [8, 16]]         det = 64 - 64 = 0
```

The determinant is exactly zero, `lambda_min = 0`, `kappa` is infinite, and the normal equations become

```text
4 w0 + 8 w1 = 10.00
8 w0 + 16 w1 = 20.00
```

which are the same equation twice (the second is the first multiplied by 2, and the right-hand side is consistent because `X^T y` always lies in the column space of `X^T X`). Every pair with `w0 + 2 w1 = 2.50` is a solution: `(2.5, 0)`, `(0.5, 1.0)`, `(-1.5, 2.0)`. The fitted values are identical for all of them, so predictions are well defined while coefficients are not. The minimum-norm member, from MV-13, is the one orthogonal to the null vector `(2, -1)/sqrt(5)`: solving `w = t(1,2)` and `t + 4t = 2.50` gives `t = 0.5` and `w = (0.5, 1.0)`. Note that this is *one* of the infinitely many solutions and is selected purely by the Euclidean-norm convention, not by any property of the data.

### 5.7 Ridge on the same data

With `tau = 1` the penalised system is

```text
(X^T X + I) w = X^T y   ->   [[5, 10], [10, 31]] w = [10.00, 29.90]^T
det = 5*31 - 10*10 = 155 - 100 = 55
(X^T X + I)^{-1} = (1/55) [[31, -10], [-10, 5]]
w0 = (31*10.00 - 10*29.90)/55 = (310.0 - 299.0)/55 = 11.0/55 = 0.2000
w1 = (-10*10.00 + 5*29.90)/55 = (-100.0 + 149.5)/55 = 49.5/55 = 0.9000
```

The slope falls from 0.98 to 0.90 and the intercept rises from 0.05 to 0.20. The rise in the intercept is not a coincidence: the penalty here was applied to both coefficients including the intercept, which shrinks the fitted line toward the origin and therefore biases the mean level downward at the centre of the data — the fitted value at `x = 2.5` is `0.2 + 0.9*2.5 = 2.45` instead of `2.50`. Intercepts are usually excluded from the penalty for exactly this reason, and the numeric demonstration shows why the exclusion is a modelling decision with a measurable consequence rather than a stylistic preference.

Sensitivity of the ridge path, computed at three values of `tau`:

```text
tau = 0.0 : w = (0.0500, 0.9800),  SSE = 0.0080
tau = 1.0 : w = (0.2000, 0.9000),  SSE at this w:
             y_hat = 1.1, 2.0, 2.9, 3.8
             e = (-0.05, -0.05, 0.15, 0.15),  SSE = 0.0025+0.0025+0.0225+0.0225
             SSE = 0.0500
tau = 5.0 : matrix [[9, 10], [10, 35]], det = 315 - 100 = 215
             w = ((35*10 - 10*29.9)/215, (-10*10 + 9*29.9)/215)
               = ((350 - 299)/215, (269.1 - 100)/215) = (0.2372, 0.7865)
```

The pattern to notice is that `tau = 1` increased the residual sum from 0.0080 to 0.0500, a factor of 6.25, while moving the coefficients only modestly. Regularisation always pays in bias; the question is whether it buys more in variance than it costs in bias, and that question cannot be answered from these four constructed points at all. Answering it requires resampling or a held-out set, which is the subject of Section 9's cost discussion and of the sibling statistics sheets.

### 5.8 What the example establishes and what it does not

It establishes: the shapes and the products (checked three ways), the solution of the normal equations (checked three ways), the orthogonality of the residual (checked twice), the projection properties including the trace identity (checked against `d`), the conditioning arithmetic including the square relation of MV-20 (checked against trace and determinant), the rank-deficient case and the minimum-norm selection (checked by substitution), and the ridge path with its bias on the intercept (checked by direct evaluation of the objective).

It does not establish: that 0.98 is a good estimate of anything; that `R^2 = 0.9983` is impressive; that `tau = 1` is an appropriate penalty; that these residuals are noise; or that the linear model is correct. Those are inferential claims requiring a probabilistic model, a sampling story, and data that were not manufactured to make the arithmetic clean. The discipline this example teaches is the separation: *the algebra is fully checked, the inference is entirely open*.

> **Takeaway:** A worked example earns its place by being checkable three ways; three agreeing routes catch arithmetic, and none of them catches a model that should not have been fitted.

## 6. Visual / Geometric Intuition

### 6.1 The residual is a perpendicular

The least-squares fit has one picture that explains everything else. Place the `m` observations of the target as a single vector `y` in `m`-dimensional space. The columns of `X` are `d` vectors in that same space. Every achievable prediction `Xw` is a linear combination of those columns, so the set of all predictions is the `d`-dimensional subspace spanned by the columns — the column space. Least squares asks for the point of that subspace closest to `y`, and the closest point is found by dropping a perpendicular.

```text
                    y  (observed target, m x 1)
                    *
                    |  \
                    |    \   residual e = y - y_hat
                    |      \   (perpendicular to the plane)
                    |        \
                    |          *
                    |        /   y_hat = P y  (the fit)
                    |      /
                    |    /
                    |  /
  ------------------+------------------------------>  column space of X
                    (d-dimensional plane through origin)
```

Three consequences follow immediately and none of them requires algebra:

1. The fit is unique as a *point* whenever the column space is a well-defined set, even when the coordinates `w` are not unique. The plane does not care which basis you use.
2. The residual cannot be reduced by moving within the plane; any movement adds a component orthogonal to the residual, and Pythagoras then gives a strictly larger distance. That is the geometric proof of optimality, and it is what the normal equations encode algebraically.
3. Pythagoras gives the variance decomposition directly: `||y||^2 = ||y_hat||^2 + ||e||^2` when the model includes an intercept and the columns are orthogonal to the residual. Centring `y` removes the mean component and converts this into `SST = SSR + SSE`.

The projection matrix `P` is the machine that performs the dropping, and its idempotence is the statement that dropping a perpendicular twice does nothing new: once you are in the plane, projecting again leaves you where you are.

### 6.2 What a small singular value looks like

Conditioning is often taught as a number, which hides the geometry. Take the two-column design `X` with columns that are nearly parallel. The column space is a thin sliver in the observation space: it is almost a line. The matrix maps a unit disk of coefficient vectors to a very flat ellipse in the observation space, with semi-axes `sigma_max` and `sigma_min`.

```text
    coefficient space                 observation space
    (unit circle in R^2)             (ellipse in the column plane)

        w1                                ^
         |    ****                        |     *
         |  **    **                      |   *
         | *        *                     |  *
    -----+------------ w0           ------+--------------->  sigma_max
         | *        *                     |  *
         |  **    **                      |   *
         |    ****                        |     *
                                          sigma_min  (very thin)
```

Inverting the map stretches the short axis by `1/sigma_min`. A perturbation in `y` of size `eps` in the direction of the short axis produces a coefficient change of size `eps/sigma_min`, which can be enormous. This is the picture behind every "my coefficients changed wildly when I added one observation" story: the observation moved `y` slightly along the thin direction, and the inverse magnified that movement.

The same picture explains why the Gram route squares the condition number. Forming `X^T X` maps the ellipse to an ellipse whose semi-axes are the *squares* of the original ones, so the ratio of axes is squared. Nothing about the mathematical solution changed; the representation became more eccentric, and inversion therefore became less accurate.

### 6.3 The quadratic loss surface

For `d = 2`, the loss `L(w)` is a paraboloid, and its level sets are ellipses. The Hessian is `2 X^T X`, so the ellipse axes are aligned with the eigenvectors of the Gram matrix and their lengths are inversely proportional to the square roots of its eigenvalues.

```text
  w1
   |        .-''-.
   |      .'      '.        contour of constant L(w)
   |     /    x     \       x = minimiser w_hat
   |    |     |      |      long axis  <- direction of small eigenvalue
   |     \    |     /          (flat: w barely affects L)
   |      '.  |   .'
   |        '-|-'            short axis <- direction of large eigenvalue
   +----------+---------------- w0       (steep: w strongly affects L)
```

Along the flat direction the loss barely changes, so the data carry almost no information about that combination of coefficients. Any numerical noise or any small change in the data slides the minimiser a long way along the flat direction at almost no cost in loss. This is why "the fit is good" and "the coefficients are identified" are different claims: a low loss value is compatible with a minimiser that is far from the truth along a flat direction.

Ridge regularisation changes the picture by adding `tau ||w||^2`, which is a circular bowl centred at the origin. The sum of a flat ellipse and a circular bowl has a minimum pulled toward the origin, and the flat direction is no longer flat: the combined surface has curvature `tau` in every direction. The geometric statement of MV-14 is therefore "add a bowl so that no direction is flat", and the cost is the bias visible in Section 5.7.

### 6.4 Rank deficiency as a collapsed plane

For the duplicate-column example of Section 5.6, the two columns of `X'` are `c0` and `2c0`. The column space is one-dimensional: a line, not a plane. Every coefficient vector `(w0, w1)` with the same value of `w0 + 2 w1` produces the same point on that line, so the map from coefficients to predictions is many-to-one. The level set of the loss is no longer an ellipse but a pair of parallel lines (a degenerate ellipse), and the minimum is attained along an entire line of coefficient vectors.

```text
  w1
   |  \  w0 + 2 w1 = 2.5   <- entire line of exact minimisers
   |   \
   |    \    the shortest such vector is perpendicular to the line:
   |     \   w = (0.5, 1.0), ||w|| = sqrt(1.25) = 1.118
   |      \
   +-------\--------------- w0
            \
```

The perpendicular from the origin to that line is the minimum-norm solution, and this is a geometric restatement of MV-13's convention: the pseudoinverse returns the perpendicular foot. Choosing a different norm (for example a weighted norm reflecting feature units) would put the foot somewhere else, which is why reporting a coefficient from a rank-deficient fit without stating the norm is not reproducible.

### 6.5 Transpose and inverse as geometric operations

A transpose has no single geometric meaning for a general matrix, which is why "transpose means flip" is a dangerous mnemonic. It does have a precise meaning for the objects it usually acts on here. For an orthogonal matrix it is the inverse, i.e. the reverse rotation. For a projection it is the identity, since a projection onto a subspace is symmetric. For a gradient of a scalar with respect to a vector, transposing converts a column into a row so that the chain rule stays conformable, and this is the sense in which `(AB)^T = B^T A^T` is a statement about the order in which linear maps are applied when the arrow of composition is reversed.

The clean way to hold it: a matrix is a linear map from one space to another; its transpose is the map between the dual spaces that reverses the direction of the pairing. Composing `A` then `B` and then dualising must apply the dual of `B` first, because the dual of the composite eats covectors at the output end. Reversing the order is not a quirky convention; it is forced by the direction in which the pairing is consumed.

### 6.6 Why the Frobenius norm sees only singular values

`||A||_F^2 = sum_ij A[i,j]^2 = tr(A^T A) = sum_i sigma_i^2`. The last equality holds because `A^T A` has eigenvalues `sigma_i^2`, and the trace is the sum of eigenvalues. Geometrically, the Frobenius norm measures the total "stretch" of the map across all directions with equal weight, which is why it is invariant under rotation of either the input or the output space. It differs from the spectral norm, which measures only the largest stretch: for a matrix with all `r` singular values equal to `sigma`, the Frobenius norm is `sigma sqrt(r)` while the spectral norm is `sigma`. The ratio `sqrt(r)` is the reason a rank-100 matrix can have a Frobenius norm ten times its spectral norm, and it is why a "small" Frobenius error can hide a large relative error in a well-conditioned direction.

> **Takeaway:** Geometry converts algebraic identities into pictures you can reason with — perpendicular residual, eccentric ellipse, flat direction — and the pictures predict the failure modes before you compute them.

## 7. Comparison / Selection Table

Selection here means: given a concrete linear-algebra task with a stated shape regime, pick the arithmetic route and be able to defend it against the three objections that always arrive — accuracy, cost, and what happens when the assumption breaks. The tables below are organised by task, because "which is the best solver" is not a well-posed question.

### 7.1 Solving `Xw = y` when `m >= d` and `X` has full column rank

| Route | What it computes | Flops (leading term) | Condition seen | When to choose | When to refuse |
|---|---|---|---|---|---|
| Normal equations + Cholesky | `w` from `X^T X w = X^T y` | `m d^2 / 2` to form `G`, `d^3/3` to factor | `kappa(X)^2` | `d` small, `kappa(X) < 1e4`, speed matters, rank verified | `kappa(X)` above `1e6`, or unknown; near-collinear features |
| Householder QR | `w` from `R w = Q^T y` | `2 m d^2` | `kappa(X)` | Default for dense medium-size problems | `m` enormous and sparse; streaming |
| Modified Gram–Schmidt QR | same, weaker guarantee | `2 m d^2` | `kappa(X)` with a larger constant | Teaching, or when `Q` is wanted explicitly | Production accuracy claims |
| SVD | `w = V Sigma^+ U^T y` | `~10-14 m d^2` (iterative: far less) | `kappa(X)`, plus explicit rank decision | Diagnosis, rank deficiency, truncation, near-singular | Tight compute budget on a well-conditioned problem |
| Conjugate gradient on normal equations | `w` iteratively | `O(iter * m d)` | `kappa(X)^2` unless preconditioned | Very large sparse `X`, matrix-free | Need for last-digit accuracy; ill-conditioning |
| Conjugate gradient on the augmented system | `w` with better conditioning | `O(iter * (m+d) d)` | `kappa(X)` | Large sparse and ill-conditioned | Small dense (factorisation is cheaper) |

The right-hand column is the part practitioners skip. The normal-equations route is not wrong; it is *scope-limited*, and the scope limit is a computable property of the data, not a matter of taste. If you cannot state `kappa(X)` you have not earned the right to call the route safe.

### 7.2 Solving when `X` is rank deficient

| Situation | What is well defined | Recommended computation | What must be reported |
|---|---|---|---|
| Duplicate or collinear columns | Fitted values `y_hat` unique; `w` not | SVD with truncation, or QR with column pivoting | The retained rank, the truncation threshold, and that `w` is one representative |
| Fewer observations than columns (`m < d`) | Fitted values span everything; residual zero possible | Minimum-norm solution via `X^+` | That interpolation is happening and that coefficients are unidentified |
| Numerically tiny singular values | Rank depends on threshold | SVD, with `sigma_i/sigma_1 < tol` declared | `tol`, its rationale, and sensitivity of the answer to it |
| Structured rank deficiency (`x_3 = x_1 + x_2`) | A testable linear relation | Detect the relation, drop or combine columns, document | Which column was dropped and why that one |

The honest summary is that a rank-deficient fit produces a usable prediction function and an unusable coefficient table. Any dashboard that displays coefficients from such a fit without flagging it is reporting a convention as if it were a measurement.

### 7.3 Choosing a regulariser

| Method | Penalty | Geometry | Strength | Failure it prevents | Failure it introduces |
|---|---|---|---|---|---|
| Ridge (`L2`) | `tau ||w||^2` | Circular bowl | Always positive definite; closed form | Collinearity, rank deficiency | Shrinks all coefficients, including large true ones; needs scaling |
| Lasso (`L1`) | `tau ||w||_1` | Diamond | Sparse solutions | Many irrelevant features | Arbitrary selection among correlated features; no closed form |
| Elastic net | `a||w||_1 + b||w||^2` | Rounded diamond | Sparse and stable grouping | Correlated groups plus sparsity | Two hyperparameters to justify |
| Truncated SVD | Drop small `sigma` | Subspace restriction | Rank decision is explicit | Noise amplification `1/sigma` | Discards directions that may carry signal |
| Early stopping | Implicit `L2`-like | Path-dependent | Free in iterative training | Overfitting in long runs | Couples regularisation to optimisation schedule |
| Feature scaling | Rescaling columns | Change of basis | Fixes units-induced conditioning | Unit-driven ill-conditioning | Changes the meaning of the penalty |

The last row deserves emphasis because it is not usually listed as a regulariser. Dividing a column by its standard deviation changes the Gram matrix by a diagonal congruence, which changes its condition number, so scaling is a conditioning intervention with no statistical content at all. A ridge penalty applied before scaling and one applied after are different estimators, and neither is "the" ridge estimator.

### 7.4 Choosing an identity to rewrite an expression

| You have | You want | Identity | Legality you must check |
|---|---|---|---|
| `(AB)^T` | `A` and `B` separated | MV-01 | Compatibility only |
| `tr(ABC)` | A cyclic rearrangement | MV-03 (cyclic form) | Rotations only, not swaps |
| `sum_i e_i^T M e_i` | A scalar trace | `tr(M sum e_i e_i^T)` | None; associativity inside a trace |
| `(A + uv^T)^{-1}` | Cheap update | MV-15 | Denominator `1 + v^T A^{-1}u` bounded away from zero |
| `det(A + uv^T)` | Scalar update | MV-19 | Same denominator; `A` invertible |
| `x^T A x` | Symmetric form | Symmetrise `(A+A^T)/2` | Real `x`; loses nothing of value |
| `||Xw - y||^2` | Normal equations | MV-08 | Full column rank for uniqueness of `w` |
| `(X^T X + tau I)^{-1}X^T` | Ridge form | MV-14 | `tau > 0` |
| `X(X^T X)^{-1}X^T` | Projector properties | MV-09 | Full column rank |
| `A^{-1}` used many times | Solve repeatedly | Factor once, triangular solves | Factorisation freshness |

### 7.5 Decision guide for a code review

| Observation in review | Priority | Action | Disqualifier | Fallback |
|---|---|---|---|---|
| `inv(X.T @ X)` in code | High | Replace with a solver on the factorisation | `d <= 3` and documented | QR or `lstsq` |
| Explicit `inv` applied to a large matrix | Medium | Ask how often it is used | Used once per batch, `d` small | Keep, document cost |
| No shape assertions near a product | High | Add named-axis assertions | Static typing already proves it | Leave, note |
| Manual broadcast of a `(n,)` array | High | Reshape explicitly to `(n,1)` | Library documents the semantic | Keep |
| Coefficients reported with no condition number | High | Add `kappa(X)` to the report | Model is rank verified by construction | Add to model card |
| Standardisation fitted on all rows | Critical | Move the fit inside the training fold | Transform is provably row-independent | Rebuild the pipeline |
| Ridge with unscaled features | Medium | Scale first, or use per-feature penalties | All features share units | Justify in the model card |
| Determinant used as a singularity test | Medium | Replace with singular values | `n <= 3` and exact arithmetic | Keep for exact checks |
| `trace` used to check rank of a projector | Low (positive) | Keep; it is a free invariant | — | — |

The pattern across all five tables: choose by a property you can *measure* (`kappa`, rank, denominator size, unit consistency), not by a property you *remember* (familiarity, reputation, textbook order). A route chosen without a measurement is a guess with a citation.

> **Takeaway:** Selection is a measurement problem; the defensible table entry is the one whose "refuse when" column you can test on the data in front of you.

## 8. Failure Modes & Edge Cases

Eight failure modes follow, each with the symptom a reviewer would actually see, at least two competing causes that produce that symptom, a check that discriminates between them, a bounded fix, and a verification that the fix worked. The distinction between the causes matters more than the fix: applying the right fix for the wrong cause produces a change that appears to work and fails again under a slightly different input.

### F1 — Shape-compatible but semantically transposed

**Symptom.** All dimensions line up and the code runs, but a gradient check fails by a small but persistent amount, or a loss decreases more slowly than expected on one axis.

**Competing causes.** (a) A genuine transpose omission in a chain rule, so the gradient is the transpose of the correct one; (b) an incorrect layout assumption, where the library treats a 1-D array as a row in one call and as a column in another; (c) a correct gradient with a sign convention difference between two pieces of code.

**Discriminating check.** Construct a case where the correct gradient matrix is deliberately non-symmetric, and compare against a finite difference. If the finite-difference Jacobian is `J` and the analytic one is `J^T` with `J != J^T`, cause (a) is proven. If the discrepancy disappears when every input is explicitly shaped to `(n,1)` or `(1,n)`, cause (b) is proven. If the two agree after multiplying by `-1`, cause (c) is proven. The construction of a non-symmetric test case is essential: a symmetric test matrix makes (a) invisible, which is why symmetric toy problems let transpose bugs survive for months.

**Bounded fix.** For (a), apply MV-01 in the reverse direction at the exact site and add an assertion on the named axes. For (b), shape every array at the boundary of the function and assert the shape on return. For (c), declare the sign convention in one place and add a test against a hand-computed scalar.

**Verification.** The finite-difference check must pass with a relative error near the square root of the machine epsilon for central differences, on a non-symmetric case, with at least one non-square Jacobian. A passing check on a square symmetric case verifies nothing about transposition.

**Edge case at the boundary.** A `1 x 1` output is simultaneously a scalar, a `1 x 1` matrix, and a length-1 array. Any code path that branches on `ndim` will behave differently for `d = 1` than for `d = 2`, and the bug appears only at the boundary. Test `d = 1` explicitly in every shape-sensitive routine.

### F2 — Inversion where a solve was required

**Symptom.** Coefficients are correct to a few digits on well-conditioned data and wrong in the second significant digit on ill-conditioned data, with no error raised. Occasionally a `LinAlgError` appears on data that "worked yesterday".

**Competing causes.** (a) Explicit inversion of `X^T X`, which squares the condition number and loses digits; (b) a genuinely singular matrix, so no inverse exists at all; (c) a badly scaled matrix whose inverse is dominated by a single huge entry, so the printed result looks plausible but is mostly rounding.

**Discriminating check.** Compute `sigma_min/sigma_max` of `X` (not of the Gram matrix) and see whether the lost digits predicted by `kappa(X)^2 * u` matches the observed discrepancy against a QR-based reference. If yes, (a) is the cause. If `sigma_min` is at the level of `u * sigma_max`, (b) is present and no solver will rescue the coefficients. If the ratio of the largest to smallest row norms exceeds `1e6` and centring and scaling the columns removes the problem, (c) is the cause.

**Bounded fix.** For (a), replace with a factorisation-based solve: `solve(G, b)` uses Cholesky or LU as appropriate; better, use QR on `X` directly so the Gram matrix is never formed. For (b), move to the pseudoinverse with a declared threshold and report the rank. For (c), scale columns before solving and unscale the coefficients afterwards, documenting the transformation.

**Verification.** Compare against a reference computed with a different route (QR against SVD), require agreement to a tolerance derived from `kappa * u` rather than a fixed `1e-10`, and confirm that the residual orthogonality `X^T r ≈ 0` still holds. Note that the residual check will pass even when the coefficients are badly wrong if the matrix is very ill-conditioned; it is necessary, not sufficient.

### F3 — Inverse of a product taken in the wrong order

**Symptom.** A composed transformation is applied in the wrong direction; a "predict then inverse-transform" pipe produces outputs that are close but not exactly the inputs, with error that grows with the number of steps.

**Competing causes.** (a) `(AB)^{-1}` computed as `A^{-1}B^{-1}`; (b) correct algebra with a wrong application order, e.g. multiplying by the inverse on the wrong side for non-square intermediate shapes; (c) a numerically inexact inverse of well-conditioned matrices, producing drift of order `kappa * u` per step.

**Discriminating check.** Round-trip a single known vector: compute `B^{-1}A^{-1}(AB x)` and `A^{-1}B^{-1}(AB x)`. If the second differs from `x` by `O(1)`, cause (a) is proven. If both are correct on a round trip but the composed pipeline drifts, count the steps and compare total drift against `steps * kappa * u` to separate (b) from (c).

**Bounded fix.** For (a), reverse the order and add a unit test that round-trips one vector. For (b), express the pipeline as an explicit composition and apply inverses in reverse composition order, with shapes asserted at every stage. For (c), reduce the number of inversions by keeping a factorisation, or accept the drift and document it with a tolerance.

**Verification.** A round-trip test on a vector with a known exact answer, at both the smallest and largest sizes used in production, plus a check that the test fails if the order is deliberately reversed. A test that cannot fail is not a test, and this is the specific case where the test must be mutation-checked.

### F4 — Broadcast silently changing the reduction

**Symptom.** A loss value is larger than expected by a factor near `m` or `d`, or a gradient has the wrong magnitude on one axis. Training "works" but converges to a different optimum than the same code on another framework.

**Competing causes.** (a) A `(m,)` array broadcast against an `(m,d)` array, expanding along the wrong axis; (b) a reduction performed over the wrong axis, e.g. summing over features instead of samples; (c) a genuine difference in loss normalisation (mean versus sum) between the two implementations being compared.

**Discriminating check.** Print the shapes immediately before the operation and the shape of the intermediate after it. If a `(m,)` operand became `(m,d)` along the wrong axis, (a) is proven. If the value is exactly `m` times or `d` times the reference, (b) or (c) is present; then check whether both implementations divide by the same count. A per-sample test with `m = 3` and `d = 2`, where the two axes have different lengths, makes the mistake visible; with `m = d` the mistake is invisible, so choose test sizes with unequal dimensions.

**Bounded fix.** For (a), reshape explicitly at the boundary and use named axes rather than positions. For (b), state the reduction axis in a comment adjacent to the call and test with an oracle computed in a loop. For (c), normalise once in a shared helper and unit-test the helper against a hand-computed scalar.

**Verification.** A small tensor oracle with `m != d`, computed by an independent loop implementation, plus a test that the loss is invariant to permuting samples (which it must be) and not invariant to permuting features (which it generally is not).

**Edge case.** Singleton dimensions are the trap: with `d = 1`, a wrong-axis reduction gives the right answer, so the bug ships and appears only when the feature count grows. Any routine whose axis logic is positional must be tested with all dimensions distinct and greater than one.

### F5 — `x x^T` where `x^T x` was intended

**Symptom.** A loss that should be a scalar is an `m x m` matrix, or memory explodes as the sample count grows. Sometimes the code runs because a later reduction accidentally sums the matrix, giving a value that is `m` times too large.

**Competing causes.** (a) Operand order reversed in an outer product; (b) a transposed input, so a genuinely correct expression becomes an outer product; (c) correct outer product with a missing trace or sum that was intended to convert it to a scalar.

**Discriminating check.** Compare `shape(result)` with the intended type: a scalar objective must have shape `()` or `(1,1)`. If the shape is `(m,m)` and the intended value equals `trace(result)`, then (a) or (c) is present and the two are distinguished by reading the intent: if the outer product is needed later, (c); if not, (a).

**Bounded fix.** For (a), use the inner-product form and assert scalar output. For (b), fix the shape at the function boundary. For (c), take the trace explicitly with a comment stating why the matrix was formed.

**Verification.** A shape assertion on the returned loss, plus a hand-computed value on three samples. Also assert that doubling the sample count does not change the per-sample loss, which catches accidental double counting.

### F6 — Rank assumed from dimensions

**Symptom.** Coefficients vary between runs that use the same data in a different order, or a solver returns huge values with a small residual. Sometimes an occasional `LinAlgError` on specific folds.

**Competing causes.** (a) Genuine exact collinearity (a duplicate or derived column); (b) numerical collinearity where columns are nearly but not exactly dependent, so rank is a threshold question; (c) a fold with fewer observations than features, so `m < d` in that fold only.

**Discriminating check.** Singular values, sorted, with the ratio to the largest printed. Exact collinearity shows a singular value at the level of `u * sigma_max`; near-collinearity shows one at `1e-8 * sigma_max` but well above `u`. If the failure appears only on some folds, inspect `m` and `d` per fold before anything else — cause (c) is a bookkeeping error, not a numerical one.

**Bounded fix.** For (a), remove or combine the dependent column and record the decision. For (b), regularise or truncate, and declare the threshold. For (c), enforce a minimum fold size or use a rank-aware solver and report the reduced effective parameter count.

**Verification.** Recompute the coefficients with the rows in a different order and confirm stability; perturb one entry by `1e-12` relative and confirm the coefficients move by no more than the tolerance the conditioning justifies. Stability under row permutation is a strong, cheap check that catches most rank confusion.

### F7 — Regularisation without scaling

**Symptom.** Some coefficients are shrunk almost to zero while others are barely affected, and the pattern does not correspond to any belief about importance. Changing a feature's unit changes the fitted model.

**Competing causes.** (a) Features on different scales, so the penalty is effectively non-uniform; (b) an intercept included in the penalty, biasing the level; (c) a penalty chosen by a procedure that was tuned on unscaled data and is therefore at the wrong magnitude.

**Discriminating check.** Multiply one column by 1000, refit, and compare. If the coefficient for that column changes by roughly a factor of 1000 and the others barely move, (a) is proven. If the fitted mean is systematically off while slopes are reasonable, (b) is present. If the selected penalty is at the edge of the searched grid, (c) is likely.

**Bounded fix.** For (a), standardise columns using training-fold statistics only, fit, and unscale the coefficients for interpretation. For (b), exclude the intercept column from the penalty. For (c), re-tune on the scaled representation with a grid that brackets the plausible range.

**Verification.** Refit after a unit change and confirm the model is invariant up to the coefficient rescaling; confirm that the penalty path is indexed by a scale-free quantity; confirm that the intercept is unpenalised by checking that a constant shift in the target shifts the intercept by the same amount.

### F8 — Metrics that hide the fit

**Symptom.** `R^2` near 1 on a model that predicts poorly out of sample, or a residual plot with obvious structure despite an excellent aggregate metric.

**Competing causes.** (a) Overfitting with `d` close to `m`; (b) leakage, where a feature encodes information from the evaluation period; (c) an inappropriate metric that is dominated by a few large-magnitude points.

**Discriminating check.** Compare in-sample and out-of-sample `R^2`. A large gap points to (a) or (b). To separate them, remove the suspicious feature and refit: if out-of-sample performance collapses, leakage was carrying it, which proves (b). If the gap persists across all feature subsets, (a) is present. If the residual plot shows structure that the aggregate metric ignores, (c) is present.

**Bounded fix.** For (a), reduce `d`, regularise, or collect more data. For (b), rebuild the split with a time boundary and fit every transform inside the training window. For (c), report residuals by segment and add a metric that weights errors by the decision consequence.

**Verification.** A held-out evaluation with the split boundary recorded, a check that no feature's computation touches future rows, and a residual-versus-fitted plot reviewed by a second person. Note that this failure mode is the one where the algebra is entirely correct; it cannot be caught by any identity in this sheet, which is exactly why it is included.

### Edge cases that are not failures

Three situations look like failures and are not.

- **A residual that is not zero.** Least squares is an approximation unless `y` happens to lie in the column space. A non-zero residual is expected; structure in the residual is the diagnostic.
- **Coefficients that change when a collinear feature is added.** Both fits are valid; only the fitted values are identified. The change is a property of the parameterisation, not an error.
- **A condition number of 10 on real data.** That is a well-conditioned problem. Worrying about it wastes time; worrying about `kappa = 1e8` earns its keep.

> **Takeaway:** Each failure mode has at least two causes that look alike from the symptom alone; the discriminating check, not the fix, is the part of the entry worth memorising.

## 9. Complexity & Cost Analysis

### 9.1 Operation counts, with the model stated

The counts below assume the standard dense model: a floating-point multiply-add costs one flop unit, memory movement is free unless stated, and no blocking, vectorisation, or cache effects are modelled. Real implementations deviate; the counts are for choosing an algorithm, not for predicting wall-clock time. Let `m` be rows, `d` columns, `r = rank(X) <= min(m,d)`.

| Operation | Flops (leading term) | Memory | Note |
|---|---|---|---|
| `Ax` (matrix-vector) | `2 m d` flops (or `m d` multiply-adds) | `m d` for `A` | one sum of `d` products per row |
| `AB` (matrix-matrix, `m x n` by `n x p`) | `2 m n p` | `m n + n p` | the `m p` output is often the surprise |
| `A^T A` from scratch | `m d^2` (symmetric: half the off-diagonals) | `d^2` | the `d^2` result is why streaming Gram works |
| Cholesky of a `d x d` SPD | `d^3/3` | `d^2/2` | no pivoting needed |
| LU with partial pivoting | `2 d^3/3` | `d^2` | general square |
| Householder QR of `m x d` | `2 m d^2 - 2 d^3/3` | `m d` for `Q` if formed | `Q` need not be formed |
| Full SVD of `m x d` | `O(m d^2)` with a constant around 10–25 | `m d` | iterative methods for truncated SVD are much cheaper |
| Triangular solve | `d^2` | — | forward or back |
| Inverse from a factorisation | `d^3` | `d^2` | almost never needed |
| `A^{-1}` applied to `k` right-hand sides via one factorisation | `d^3/3 + k d^2` | `d^2` | versus `k d^3/3` for `k` inversions |
| Ridge solve | same as Cholesky on `X^T X + tau I` | `d^2` | plus `d^2` to add the penalty |
| Sherman–Morrison update | `2 d^2` per update given `A^{-1}` | — | versus `d^3/3` for a refactorisation |
| Woodbury update, rank `k` | `O(d^2 k + k^3)` | `d k` | invert the small `k x k` block |
| One singular value (power iteration) | `O(m d)` per iteration | `m d` | converges at rate `(sigma_2/sigma_1)^2` |

Note on the first row: a matrix-vector product costs `2 m d` flops counting both multiply and add, or `m d` multiply-adds; the table states the convention because mixing the two conventions is a standard way to be off by a factor of two in a capacity estimate.

### 9.2 Worked sizing: two regimes, same algebra

**Regime A — wide, tall, small feature count.** `m = 1,000,000` rows, `d = 20` features, float64 (8 bytes).

```text
Forming X^T X:        m d^2 = 1e6 * 400 = 4.0e8 flops
Cholesky:             d^3/3 = 8000/3 = 2.7e3 flops       (negligible)
Total solve cost:     ~4.0e8 flops
Data footprint:       m d * 8 bytes = 1e6*20*8 = 1.6e8 bytes = 152.6 MiB
Gram footprint:       d^2 * 8 = 3200 bytes               (negligible)
```

Under a *hypothetical* sustained rate of `1e9` flops/s on one core — an illustrative rate, not a measured one — the Gram formation takes about `0.4 s`. The dominant cost is reading `152.6 MiB` once, which at an illustrative `5 GB/s` streaming bandwidth is `152.6e6/5e9 = 0.031 s`. So in this regime the computation is arithmetic-dominated by a factor of roughly ten, and even the "slow" Gram route is fine. This is the regime where the textbook advice to use normal equations is defensible: `d = 20` keeps `kappa(G)` manageable and the flop count is small.

**Regime B — tall, wide, many features.** `m = 200,000`, `d = 5,000`.

```text
Forming X^T X:        m d^2 = 2e5 * 2.5e7 = 5.0e12 flops
Cholesky:             d^3/3 = 1.25e11/3 = 4.2e10 flops
Householder QR:       2 m d^2 = 1.0e13 flops  (2x the Gram route)
Full SVD:             ~10 * m d^2 = 5.0e13 flops (10x)
Data footprint:       m d * 8 = 2e5*5000*8 = 8.0e9 bytes = 7.45 GiB
Gram footprint:       d^2 * 8 = 2.5e7*8 = 2.0e8 bytes = 190.7 MiB
```

Here the Gram route at `5e12` flops is already two orders of magnitude beyond the previous regime, and the QR route doubles it. The interesting consequence is that at `d = 5000` the *Gram matrix itself* is 190.7 MiB and Cholesky alone is `4.2e10` flops — comparable to reading the data several times. In this regime the sensible choices are a truncated SVD via randomised or Lanczos methods, a ridge solve with a matrix-free iterative method, or an explicit reduction of `d`. Choosing between them requires knowing whether `kappa(X)` and the desired rank make the extra flops buy anything, which returns to Section 7.

The lesson from placing the two regimes side by side: the *same* code and the *same* identity produce a comfortable computation at `d = 20` and an infeasible one at `d = 5000`, and the crossover is set by `d^2` and `d^3`, not by `m`. Scaling the row count multiplies cost linearly and is usually the benign direction; scaling the column count is what breaks things.

### 9.3 Numerical cost: digits as the currency

The other cost axis is precision. Under the standard model, a relative perturbation `delta` in the data can produce a relative perturbation of order `kappa * delta` in the solution. Converting to digits lost:

```text
digits_lost ≈ log10(kappa)    and    digits_remaining ≈ 16 - log10(kappa)
```

| `kappa(X)` | Route | Approx digits remaining (double) | Verdict |
|---|---|---|---|
| 1e0 | either | 16 | exact in practice |
| 1e2 | either | 14 | no concern |
| 1e4 | either | 12 | no concern |
| 1e6 | Gram route | 16 - 12 = 4 | marginal; use QR |
| 1e6 | QR route | 10 | acceptable |
| 1e8 | Gram route | 0 | unusable |
| 1e8 | QR route | 8 | usable with care |
| 1e12 | QR route | 4 | report wide intervals |
| 1e16 | any route | 0 | the problem is not solvable in double |

The table is a heuristic derived from the standard error bound, not a guarantee, and it assumes the bound's constant is near one. Its practical use is as a screening rule: compute `kappa`, subtract its logarithm from 16, and if the remainder is below about 6 the analysis must change (regularise, collect more data, or re-parameterise) before any coefficient is reported.

### 9.4 Memory and streaming cost of the Gram route

The Gram route has a structural advantage that the tables above understate: it can be computed in a single pass with `d^2` memory regardless of `m`. For each row `x_i` (`d x 1`), accumulate `G += x_i x_i^T` and `b += x_i y_i`. Total memory is `d^2 + d`, so a data stream of arbitrary length can be processed with a fixed footprint. This is why the route survives in streaming and federated settings despite its conditioning penalty, and why the honest recommendation is "Gram for streaming, QR for accuracy" rather than a blanket rule.

```text
Streaming cost for m rows, chunk size c:
  passes over the data = ceil(m / c)
  flops               = m d^2  (same as batch)
  peak memory         = c * d * 8 bytes (chunk) + d^2 * 8 bytes (accumulator)
  numerical penalty   = kappa(X)^2 instead of kappa(X)
```

An additional subtlety: accumulating `G` row by row in floating point is *not* equivalent to forming it in one batch, because the accumulation order differs. For very large `m` and small `d`, the accumulated sum can lose accuracy through cancellation when entries have mixed signs. Compensated summation (Kahan or Neumaier) removes most of this at a small constant cost, and it is the standard remedy when a streaming Gram matrix must match a batch reference.

### 9.5 Operational cost framing

For a production fitting job, the total cost is the sum of four terms, and only the first is the algebra:

```text
total = flops_cost + data_movement_cost + engineer_time + risk_cost
```

`flops_cost` is what Sections 9.1–9.2 estimate. `data_movement_cost` is usually dominant at small `d` and is proportional to bytes read. `engineer_time` is proportional to how much of the derivation a maintainer must reconstruct, which is the argument for keeping the factorisation explicit rather than hiding it in a chain of inverses. `risk_cost` is the expected cost of a silent numerical failure, and it is why a route that costs twice the flops and does not square the condition number is often the cheaper choice in total. None of these four terms is measured here; the decomposition is a budgeting frame, and any actual numbers would need a workload, hardware, and price list that this sheet does not have and does not invent.

> **Takeaway:** Cost has two currencies, flops and digits; the row count multiplies the first linearly, the column count multiplies it quadratically or cubically, and conditioning determines how much of the second you must spend.

## 10. Key Theorems / Results Reference

Each theorem below is stated with its hypotheses, because the hypotheses are the part that gets dropped in conversation and the part that invalidates use. The statement is followed by what the result licenses and what it does not.

**T1 — Rank–nullity.** For `A` (`m x n`) over a field, `rank(A) + nullity(A) = n`, where `nullity(A) = dim{x : Ax = 0}`.
*Licenses:* concluding that a `d`-column design with rank `r` leaves `d - r` directions in coefficient space that the data cannot distinguish. In Section 5.6, `d = 2` and `r = 1`, so one direction is unidentified and the whole line of minimisers follows.
*Does not license:* any statement about which coefficients are "important"; the null direction is a property of the design, not of the outcome.

**T2 — Orthogonal decomposition (fundamental theorem of linear algebra).** For `A` (`m x n`), `R^m = C(A) ⊕ N(A^T)` and `R^n = C(A^T) ⊕ N(A)` with the sums orthogonal.
*Licenses:* the geometric proof of least squares. Write `y = y_C + y_N` with `y_C` in the column space and `y_N` orthogonal to it. Then `||y - Ax||^2 = ||y_C - Ax||^2 + ||y_N||^2`, and the second term is fixed, so minimising is achieved exactly when `Ax = y_C`. The minimum residual norm is `||y_N||`.
*Does not license:* an assumption that `y_C` is "the signal" and `y_N` "noise"; that is an interpretation requiring a model.

**T3 — Normal equations characterisation.** Let `X` be `m x d` real and `y` in `R^m`. Then `w_hat` minimises `||Xw - y||_2` if and only if `X^T(X w_hat - y) = 0`.
*Proof sketch:* the function `f(w) = ||Xw - y||^2` is convex and differentiable; its gradient `2X^T(Xw - y)` vanishes at exactly the stated condition, and convexity makes the stationary condition sufficient. Convexity follows from the PSD lemma in Section 4.2.
*Licenses:* the residual-orthogonality check used throughout Section 5, and the characterisation of all minimisers as `w_p + N(X)` for any particular minimiser `w_p`.
*Does not license:* the closed form `(X^T X)^{-1}X^T y` — that requires invertibility, which is T4's hypothesis.

**T4 — Uniqueness and the closed form.** With `X` (`m x d`), the minimiser is unique if and only if `rank(X) = d`. In that case `X^T X` is invertible and `w_hat = (X^T X)^{-1}X^T y`.
*Proof sketch:* `N(X^T X) = N(X)` by the PSD lemma; so `X^T X` is invertible exactly when `N(X) = {0}`, i.e. rank `d`. Uniqueness of the minimiser then follows because two minimisers differ by an element of `N(X)`.
*Licenses:* the claim that full column rank is the right condition. Note that `m > d` does **not** imply it: a `1000 x 3` matrix with a duplicated column has rank 2 and an infinite family of minimisers.
*Does not license:* any numerical claim. A matrix can have exact rank `d` and computed rank below `d`; the theorem is about exact arithmetic.

**T5 — Minimum-norm least squares.** Among all minimisers of `||Xw - y||_2`, the unique one of smallest `||w||_2` is `w = X^+ y = V Sigma^+ U^T y`.
*Proof sketch:* decompose `w = w_R + w_N` with `w_R` in the row space and `w_N` in the null space. `Xw = Xw_R`, so `w_N` does not affect the residual while it strictly increases the norm; hence the minimum-norm minimiser lies in the row space and is unique there.
*Licenses:* the entire rank-deficient fallback strategy, and the interpretation of the pseudoinverse as "the perpendicular foot" of Section 6.4.
*Does not license:* treating the resulting coefficient vector as scientifically identified; it is one point in an affine set, chosen by a norm convention.

**T6 — SVD existence and structure.** Every real `m x n` matrix factors as `X = U Sigma V^T` with `U` (`m x m`) and `V` (`n x n`) orthogonal and `Sigma` (`m x n`) diagonal with `sigma_1 >= ... >= sigma_min(m,n) >= 0`.
*Licenses:* reading rank as the count of nonzero `sigma_i`; the spectral norm as `sigma_1`; the Frobenius norm as the Euclidean norm of the singular values; and `kappa = sigma_1/sigma_r`.
*Does not license:* interpreting singular vectors as "the important features" without a scaling decision, since `U` and `V` are not invariant to arbitrary column scaling.

**T7 — Eckart–Young–Mirsky.** For `X` with SVD as in T6 and any `k < r = rank(X)`, the best rank-`k` approximation in both the spectral and Frobenius norms is `X_k = sum_{i<=k} sigma_i u_i v_i^T`, with errors `sigma_{k+1}` and `sqrt(sum_{i>k} sigma_i^2)` respectively.
*Licenses:* truncated SVD as a principled compression, and the error statement quoted in MV-13.
*Does not license:* the claim that low-rank approximation preserves predictions or coefficients; it preserves the matrix in a norm, which may not be the norm that matters for the decision.

**T8 — Weyl's inequality.** For symmetric `A` and `E`, `|lambda_i(A + E) - lambda_i(A)| <= ||E||_2` for every eigenvalue index `i`.
*Licenses:* the statement in Section 4.4 that eigenvalue perturbations of a symmetric matrix are controlled by the size of the perturbation, which is why the *Gram* matrix's spectrum is trustworthy even though its inverse is not.
*Does not license:* a corresponding bound on the inverse's eigenvalues; those move by roughly `||E||/lambda^2`, which is the amplification behind ill-conditioning.

**T9 — Bauer–Fike.** If `A` is diagonalisable with `A = S Lambda S^{-1}` and `A + E` has eigenvalue `mu`, then `min_i |mu - lambda_i| <= kappa_2(S) ||E||_2`. For normal matrices `kappa_2(S) = 1` and Weyl's bound is recovered.
*Licenses:* the warning that non-normal matrices can have eigenvalues far more sensitive than the norm of a perturbation suggests. Relevant here because it explains why the *singular* values of `X` are reliable diagnostics while the roots of a non-symmetric characteristic polynomial are not.

**T10 — Cauchy–Schwarz.** For any inner-product space, `|x^T y| <= ||x||_2 ||y||_2`, with equality if and only if `x` and `y` are linearly dependent.
*Licenses:* the bound on how much a single unit-norm direction can extract from a vector, which underlies the leverage bounds `0 <= h_ii <= 1` and the statement that `|correlation| <= 1` for standardised vectors. The correlation bound is exactly Cauchy–Schwarz applied to centred vectors divided by their norms.
*Does not license:* any claim about causation from a large inner product; correlation of 1 is a statement about geometry, not mechanism.

**T11 — Perturbation bound for linear systems.** If `A w = b` and `(A + dA)(w + dw) = b + db` with `||dA|| <= eps ||A||`, then to first order
`||dw||/||w|| <= kappa(A) (||db||/||b|| + ||dA||/||A||)`.
*Licenses:* the digits-lost rule of Section 9.3, and the design of the check in F2.
*Does not license:* an exact error prediction; the bound is worst case, can be pessimistic by orders of magnitude for a particular right-hand side, and requires consistent norms.

**T12 — Gram–Schmidt and QR existence.** If `X` (`m x d`) has linearly independent columns, there exists `Q` (`m x d`) with `Q^T Q = I_d` and upper-triangular `R` (`d x d`) with positive diagonal such that `X = QR`; the factorisation is unique under the sign convention.
*Licenses:* the QR route's cost and conditioning claims, and the invariant `det(X^T X) = det(R)^2` used as a check in Section 5.3.
*Does not license:* a stability claim for a particular algorithm. Exact existence is a theorem; backward stability of *classical* Gram–Schmidt is false, and that gap is the reason the algorithm is named in the table in Section 7.1.

**T13 — Sherman–Morrison–Woodbury.** With the inverses and the relevant `k x k` block invertible, `(A + UCV)^{-1} = A^{-1} - A^{-1}U(C^{-1} + VA^{-1}U)^{-1}VA^{-1}`.
*Licenses:* rank-one and low-rank updates at `O(n^2)` and `O(n^2 k + k^3)` costs, including leave-one-out computations.
*Does not license:* a stability claim. The identity is exact, but the intermediate `C^{-1} + VA^{-1}U` can be far worse conditioned than `A` itself, which is why a numerically stable rank-one update uses a factorisation update (Cholesky or QR) rather than the closed form.

**T14 — The determinant is the product of eigenvalues, and the trace is their sum.** For `A` (`n x n`) with eigenvalues counted with algebraic multiplicity, `det(A) = prod_i lambda_i` and `tr(A) = sum_i lambda_i`.
*Licenses:* the eigenvalue computation from trace and determinant used in Section 5.6 for a `2 x 2` matrix, and the check `prod(sigma) = |det|`.
*Does not license:* using `det` as a conditioning measure, for the reasons in Limit 4 of Section 4.4 — scaling by `c` multiplies `det` by `c^n` and leaves `kappa` unchanged.

**T15 — Idempotent matrices have eigenvalues in `{0,1}`.** If `P^2 = P`, then any eigenvalue satisfies `lambda^2 = lambda`, so `lambda` is 0 or 1; consequently `rank(P) = tr(P)` for diagonalisable `P`, and symmetric idempotents are orthogonal projections.
*Licenses:* the trace-as-rank check in Section 5.5, which turns a matrix identity into a runtime invariant. In the example, `tr(P) = 2 = d`, so `P` has rank 2 and its image is exactly the column space.
*Does not license:* assuming an idempotent computed numerically satisfies `P^2 = P` to more than `kappa`-scaled accuracy; a computed projector can drift.

**T16 — Sylvester's rank inequality.** `rank(A) + rank(B) - n <= rank(AB) <= min(rank(A), rank(B))` for `A` (`m x n`) and `B` (`n x p`).
*Licenses:* the intuition that multiplication cannot create rank and can destroy it, and a cheap consistency check on a factorisation: if `rank(X) = d` and `rank(R) < d`, the QR is wrong.
*Does not license:* an exact rank prediction for a product of two low-rank matrices; the lower bound can be attained and can be loose, and the exact rank depends on the intersection of a column space with a row space.

A note on how to use this list. None of these theorems is worth memorising as a string. Each is worth memorising as a *pair*: the conclusion and the hypothesis that most often fails. For T4 the failing hypothesis is full column rank (not `m > d`). For T3 the failing hypothesis is differentiability over the reals with the declared norm — switching to an `L1` loss changes the characterisation to a subgradient condition and the closed form disappears entirely. For T11 the failing assumption is that the perturbation is small relative to the matrix norm, which is exactly what fails when the right-hand side is nearly in the null space. For T15 the failing assumption is exact idempotence. A reader who can name the failing hypothesis for each theorem has the theorem.

> **Takeaway:** Each theorem is a contract; the hypotheses are the terms and conditions, and every failure in Section 8 is a breach of a specific clause.

## 11. Interview / Exam Questions

Fifteen questions, ordered from recall through calculation to judgment. Each carries a checked answer, the reasoning that produces it, and the red flag that a weak answer exhibits. The red flags are not stylistic preferences; each names a specific misconception that would cause a real defect if the candidate were trusted with the corresponding code.

**Q1 (recall).** State `(AB)^T` in terms of `A^T` and `B^T`, and say what condition it requires.

*Answer.* `(AB)^T = B^T A^T`, requiring only that `AB` be defined, i.e. `A` is `m x n` and `B` is `n x p`. No invertibility, symmetry, or squareness.
*Reasoning.* The index proof: `(B^T A^T)[i,j] = sum_k B[k,i]A[j,k] = sum_k A[j,k]B[k,i] = (AB)[j,i]`. The only nontrivial ingredient is commutativity of scalar multiplication.
*Red flag.* Saying "you also need `A` and `B` invertible" or writing `A^T B^T`. Both indicate the identity is remembered as a pattern of letters rather than a statement about compatible shapes.

**Q2 (recall).** Why is `tr(ABC) = tr(ACB)` false while `tr(ABC) = tr(BCA)` is true?

*Answer.* Trace is invariant under cyclic permutation of the factors and under nothing else. `ABC -> BCA -> CAB` are the three legal rotations; `ACB` is a swap of two adjacent factors and changes the value in general.
*Reasoning.* The double-sum renaming argument works only for rotations, because a rotation preserves the adjacency order of the chain. The counterexample in Limit 1 of Section 4.4 gives traces 1 and 2 for the same three factors in different orders.
*Red flag.* Asserting that trace is invariant under any reordering. In a derivation this produces a plausible-looking gradient with an incorrect cross term, and it is invisible on commuted or diagonal operands.

**Q3 (calculation).** For `X` with `X^T X = [[4,10],[10,30]]` and `X^T y = [10.0, 29.9]`, compute `w` and the residual for a design whose rows are `(1,1),(1,2),(1,3),(1,4)`.

*Answer.* `det = 20`; `(X^T X)^{-1} = [[1.5,-0.5],[-0.5,0.2]]`; `w0 = 15.0 - 14.95 = 0.05`; `w1 = -5.0 + 5.98 = 0.98`. Fitted values `(1.03, 2.01, 2.99, 3.97)`; residuals `(0.02, -0.06, 0.06, -0.02)` with `SSE = 0.008`.
*Reasoning.* Two checks should be offered unprompted: residuals sum to zero (the ones column is in the design) and `X^T r = 0` (the normal equations). Both are cheap and both catch a solve error.
*Red flag.* Reporting a solution without any residual check, or reporting `SSE` without noting that the residual is orthogonal to the columns by construction. The second omission suggests the candidate thinks the orthogonality is evidence of a good model rather than a property of the fit.

**Q4 (calculation).** A design has `m = 600` rows, `d = 12` columns, and `kappa_2(X) = 3e5`. Estimate how many correct significant digits a normal-equations solve retains in double precision, and compare with QR.

*Answer.* `kappa(X^T X) = 9e10`, so `log10 = 10.95`; with about 16 digits available, roughly `16 - 11 = 5` digits remain for the Gram route. QR sees `kappa = 3e5`, `log10 = 5.48`, leaving about 10 digits.
*Reasoning.* The digits-lost rule from T11 and Section 9.3, applied to the condition number actually seen by the arithmetic. The square relation MV-20 is the mechanism.
*Red flag.* Using `kappa(X)` for both routes, or quoting "double precision has 16 digits so it's fine". Either mistake means the candidate would ship the Gram route on a problem where it is marginal.

**Q5 (diagnosis).** A model's coefficients change by 40% when one row of 5,000 is edited in the fourth significant digit. Shape and rank checks pass. What is the most likely mechanism and what do you check first?

*Answer.* Near-collinearity along a low-variance direction. Check the singular values of `X` and the ratio `sigma_min/sigma_max`; also compute the leverage of the edited row from the projection matrix diagonal. A row with high leverage in a direction with small `sigma` moves the coefficients along that flat direction at almost no cost in loss.
*Reasoning.* The geometry of Section 6.2: the inverse stretches the short axis by `1/sigma_min`, and leverage tells you which rows push along it. Both diagnostics are cheap; the perturbation test itself (`edit one entry by 1e-4 relative, refit, compare`) is the decisive one.
*Red flag.* Blaming "overfitting" without inspecting singular values, or proposing more data as the first action. More data may help, but the diagnosis is testable in minutes and should precede the fix.

**Q6 (diagnosis).** A colleague's least-squares fit returns sensible predictions but the coefficient for `x_1` is `-4.7e7` and for `x_2` is `+4.7e7`. What has happened and what do you do?

*Answer.* The two columns are collinear or nearly so, so only their combination is identified; the solver found a point far along the null direction. Verify with the singular values (one at the `1e-12` relative level) or by computing `x_1 - x_2` and seeing that it has near-zero variance. Fix by dropping one column, combining them, or regularising, and by reporting the rank with the coefficients.
*Reasoning.* T5: the minimiser set is affine with direction the null space; the computed point depends on rounding and the algorithm, so a huge cancelling pair is the signature. The prediction being sensible is exactly what the theory predicts, since the null direction does not affect `Xw`.
*Red flag.* Reporting the huge coefficients as findings, or "fixing" them by rounding. Both treat a parameterisation artefact as a measurement.

**Q7 (calculation).** For the projection matrix of Section 5.5, state `tr(P)` and explain the value without recomputing the matrix.

*Answer.* `tr(P) = 2 = d`. Because `P = X(X^T X)^{-1}X^T` and `tr(XAX^T) = tr(AX^T X)` by cyclicity, so `tr(P) = tr((X^T X)^{-1}X^T X) = tr(I_2) = 2`.
*Reasoning.* MV-03 plus the cancellation of the inner factor. The same argument shows `tr(P) = rank(P)` for any projector (T15), which gives the general statement.
*Red flag.* Answering `m` (the number of rows) or `m - d`. Those are the trace of `I - P` and the residual degrees of freedom, and confusing the two is a real source of wrong degrees-of-freedom arithmetic.

**Q8 (selection).** You must fit `d = 4,000` features on `m = 50,000` rows, once per day, and you need coefficient stability across days. Which route and why?

*Answer.* A regularised route with scaling: standardise columns on a training window, then either ridge with a tuned `tau` or a truncated SVD at a chosen rank. A plain QR solve is numerically fine but will not give stability if the columns are correlated, because the coefficients are not identified stably even when they are computable. Report `kappa(X)` and the retained rank.
*Reasoning.* The distinguishing requirement is *stability across refits*, not accuracy of a single fit. That is a variance requirement, so it points at regularisation or rank restriction rather than at a more accurate solver. The cost numbers from Section 9.2 also matter: at `d = 4,000`, forming the Gram matrix is `m d^2 = 8e11` flops, so the choice must also be affordable.
*Red flag.* Answering "use SVD because it's most accurate" without addressing the stability requirement, or choosing lasso without noting that it selects arbitrarily among correlated features.

**Q9 (selection).** When would you prefer the Gram-matrix route over QR despite the squared condition number?

*Answer.* When the data arrive as a stream and cannot be revisited, because the Gram route needs `d^2` memory and one pass; when `d` is small enough that `kappa(X)^2` is still comfortably within the precision budget (roughly `kappa(X) < 1e4`); or when the matrix is sparse and the Gram matrix is much denser, in which case neither is a good choice and an iterative method on the normal equations with a preconditioner is the practical option.
*Reasoning.* The route is chosen by which constraint binds: memory, passes over data, or digits. Each has a measurable threshold, and the answer should cite one.
*Red flag.* "Never use normal equations" as a dogma. It is a scope-limited tool, not a wrong one, and refusing it outright means either losing the streaming capability or inventing a worse substitute.

**Q10 (judgment).** A reviewer proposes computing `inv(X^T X)` once and reusing it for 200 right-hand sides. Is that reasonable?

*Answer.* Computing the inverse once and applying it 200 times is algebraically the same as 200 solves given one factorisation, and the factorisation route is cheaper and more accurate: factor once at `d^3/3`, then 200 triangular solves at `d^2` each, versus forming the inverse at `d^3` and 200 matrix-vector products at `2d^2` each. The numerical difference is that the explicit inverse commits to a particular set of rounding errors in the inverse entries, whereas the triangular solves keep the structure.
*Reasoning.* The cost comparison is from Section 9.1; the accuracy argument is that inversion is a sequence of solves whose intermediate results are stored, and storing amplifies the condition number relative to solving.
*Red flag.* Agreeing on the grounds that "it's the same thing mathematically". It is, and the mathematical equivalence is precisely why the numerically worse option needs to be rejected explicitly rather than by appeal to a rule.

**Q11 (calculation).** Compute `kappa_2` for `X = [[1,0],[0,1],[1,1]]` and state the condition number of `X^T X`.

*Answer.* `X^T X = [[2,1],[1,2]]` with eigenvalues 3 and 1, so `sigma = sqrt(3), 1` and `kappa(X) = 1.732`. `kappa(X^T X) = 3`.
*Reasoning.* For a symmetric PSD matrix, singular values are eigenvalues. The trace is 4 and the determinant 3, giving eigenvalues from `t^2 - 4t + 3 = 0`, i.e. 3 and 1.
*Red flag.* Reporting `kappa = 3` for `X` (confusing the Gram matrix's conditioning with the design's), or claiming the matrix is rank deficient because `m > d` and something looks redundant.

**Q12 (diagnosis).** Gradient descent on a quadratic loss converges slowly along one direction and quickly along another. What linear-algebra quantity governs the ratio, and how would you fix it?

*Answer.* The ratio of the largest to the smallest eigenvalue of the Hessian, i.e. the condition number of `X^T X` for a least-squares loss. Convergence rate for gradient descent is governed by `(kappa - 1)/(kappa + 1)` per step in the worst case. Fix by preconditioning: standardise features (a diagonal preconditioner), use a Newton or quasi-Newton step (which removes the conditioning entirely in exact arithmetic), or use conjugate gradients, whose convergence depends on `sqrt(kappa)` rather than `kappa`.
*Reasoning.* The loss surface of Section 6.3 is an ellipse whose axis ratio is `sqrt(kappa(G))`; gradient descent takes steps proportional to the gradient, which is nearly perpendicular to the long axis, so progress along the flat direction is slow.
*Red flag.* Blaming the learning rate alone. A smaller learning rate makes the steep direction stable without addressing the flat one; the symptom persists and the diagnosis is wrong.

**Q13 (judgment).** You are told "the model has `R^2 = 0.998`, so the linear algebra is validated". What is wrong with that inference?

*Answer.* `R^2` measures fit on the data used, and it can be near 1 with `d` close to `m`, with leakage, or with a metric dominated by a few points. It says nothing about coefficient identification, conditioning, or out-of-sample behaviour. In the running example `R^2 = 0.9983` on four constructed points, which is a property of the construction.
*Reasoning.* The separation established in Section 5.8: algebra checked, inference open. Validation of the linear algebra means shape checks, three independent solve routes, residual orthogonality, and a condition estimate — none of which `R^2` supplies.
*Red flag.* Accepting a fit metric as numerical validation. This is failure mode F8 and it is the one failure no identity in this sheet can catch.

**Q14 (diagnosis).** After adding a `d`-th feature, out-of-sample error worsens and in-sample error improves. The design has `m = 40`, `d = 35`. What do you check, and what do you expect?

*Answer.* Check `kappa(X)` and the leverage values; expect extremely high leverage because `d/m = 0.875`. With `d` comparable to `m`, the fit can nearly interpolate, so the residual is small by construction and the coefficients are poorly determined. Remedies: regularise, reduce `d` by selection with a nested validation, or collect more data. Also check that the feature-selection step was inside the cross-validation loop.
*Reasoning.* `tr(P) = d` means the average leverage is `d/m`; at 0.875 the average observation has almost as much influence as it has degrees of freedom to be fitted, leaving about `m - d = 5` residual degrees of freedom.
*Red flag.* Concluding "the extra feature is noise" without checking whether selection leaked, or "we need a better solver" when the problem is sample size relative to dimension.

**Q15 (judgment).** Explain in one paragraph why `x^T A x` depends only on the symmetric part of `A`, and give a practical consequence.

*Answer.* Write `A = S + K` with `S = (A+A^T)/2` symmetric and `K = (A-A^T)/2` antisymmetric. Then `x^T K x` is a scalar equal to its own transpose, which is `x^T K^T x = -x^T K x`, so it is zero. Hence `x^T A x = x^T S x`. Practically, a hand-written Hessian or quadratic-form code that omits symmetrisation silently discards the antisymmetric part; if the intended matrix is symmetric, the resulting gradient is half of what it should be in the affected entries.
*Reasoning.* The transpose-of-a-scalar argument, plus transposition of a quadratic form: `(x^T A x)^T = x^T A^T x`. The practical consequence follows because the gradient of `x^T A x` is `(A + A^T)x`, which equals `2Ax` only when `A` is symmetric.
*Red flag.* Saying "we can assume `A` is symmetric without loss of generality" in a context where `A` is provided by an external routine that does not guarantee it. The assumption is safe as a *statement about the quadratic form*, not as a statement about the stored matrix.

The distribution of these questions matters. Three are pure recall (Q1, Q2, Q7), six require calculation (Q3, Q4, Q6, Q11, Q12 partially, Q15 partially), and six are diagnosis or selection under a stated constraint (Q5, Q8, Q9, Q10, Q13, Q14). A candidate who can do the recall questions and none of the others has memorised a formula sheet; a candidate who can do the diagnosis questions but stumbles on Q3's residual check has good instincts and unreliable arithmetic, which is the easier deficiency to fix.

> **Takeaway:** The questions that predict competence are the ones where a plausible wrong answer exists — order of factors, which condition number, rank versus dimensions — because those are the ones that produce silent defects rather than exceptions.

## 12. Quick Reference Card

| Field | Content |
|---|---|
| **Core identities** | `(AB)^T = B^T A^T` (compatibility only); `(AB)^{-1} = B^{-1}A^{-1}` (both invertible); `tr(AB) = tr(BA)`, cyclic rotations only, never swaps; `det(AB) = det(A)det(B)`, square and same size; `\|\|x\|\|_2^2 = x^T x` (real `x`; `x^H x` for complex). |
| **Least squares** | Minimise `\|\|Xw - y\|\|_2^2` → gradient `2X^T X w - 2X^T y` → normal equations `X^T X w = X^T y`, with residual `r = y - Xw` satisfying `X^T r = 0`. Closed form `w = (X^T X)^{-1}X^T y` only when `rank(X) = d`; otherwise `w = X^+ y = sum_i (u_i^T y / sigma_i) v_i`, the minimum-norm choice. |
| **Validity conditions** | Uniqueness needs full **column** rank, not `m > d`. Orthogonality of the residual needs only the normal equations. `P = X(X^T X)^{-1}X^T` needs full column rank and is symmetric, idempotent, with `tr(P) = rank(P) = d`. Ridge `(X^T X + tau I)^{-1}X^T y` needs `tau > 0` and always exists. |
| **Numerical red flag** | `kappa(X^T X) = kappa(X)^2`. Digits remaining ≈ `16 - log10(kappa_route)`. Above `kappa(X) ≈ 1e6`, use QR or SVD, never the Gram inverse. `det` is not a conditioning measure; `sigma_min/sigma_max` is. |
| **Diagnostic checklist** | 1) Assert shapes with named axes. 2) Solve three ways (elimination, inverse, QR) on a small case. 3) Check `X^T r ≈ 0` and residual sum. 4) Check `tr(P) = d`. 5) Compute `kappa(X)` and report it. 6) Refit with rows permuted and with a `1e-12` relative perturbation. 7) Confirm every transform is fitted on training rows only. |
| **Cost per unit of work** | `X^T X` costs `m d^2`; Cholesky `d^3/3`; Householder QR `2 m d^2`; SVD roughly `10 m d^2`; triangular solve `d^2`; Sherman–Morrison `2 d^2`. Cost scales linearly in rows and quadratically or cubically in columns, so the column count is what breaks a pipeline. |
| **Units and conventions** | Vectors are columns; `x^T` is the row form. `d` = columns/parameters, `m` = rows/observations. `tau` carries the units of `X^T X`, so scaling features changes the meaning of a penalty. Coefficients inherit the units of `y` divided by the units of their column. |
| **Stop rules** | Stop and re-derive if: a coefficient pair is large with opposite signs; `kappa(X) > 1e6` and you are on the Gram route; `m - d < 10`; a determinant is used as a singularity test; a loss is not a scalar; a transform was fitted before the split. Each of these precedes a silent defect, not an exception. |

**Review record.** Version 2.0, drafted 2026-11-08, independently reviewed: no. Evidence: identities DERIVED; all numeric instances ILLUSTRATIVE and arithmetically checked (Section 5 recomputed by hand and cross-checked against an independent solve); no benchmark, deployment, or measured cost is claimed. Open items: peer review of Sections 4.4 and 9.3; a worked sparse example; timing measurements to replace the illustrative flop-rate assumption in Section 9.2.
