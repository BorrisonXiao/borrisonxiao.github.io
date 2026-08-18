---
layout: interview_question
topic: linear-algebra
order: 1
title: "Eigenvalues, Eigenvectors, and the SVD"
difficulty: core
date: 2026-08-18 12:00:00
tags: [eigenvalues, eigenvectors, SVD, diagonalization]
summary: >-
  Compute eigenvalues and eigenvectors by hand, including the two cases people
  are not ready for, then compute a full SVD and read the rank, the norms, the
  condition number, and the best low-rank approximation straight off it.
scripts:
  - /assets/js/interview/linear-algebra.js
concepts:
  - name: Eigenvalues and eigenvectors
    url: https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors
    note: >-
      The directions a matrix does not turn, and the factors by which it
      stretches them. Av = lambda v, and everything else follows from it.
  - name: Characteristic polynomial
    url: https://en.wikipedia.org/wiki/Characteristic_polynomial
    note: >-
      det(A - lambda I) = 0. For 2x2 it collapses to lambda^2 - (trace)lambda +
      det = 0, which is the fastest hand method there is.
  - name: Diagonalizable matrix
    url: https://en.wikipedia.org/wiki/Diagonalizable_matrix
    note: >-
      A = P D P^-1 when the eigenvectors span. When they do not, the matrix is
      defective and no amount of algebra will fix it.
  - name: Singular value decomposition
    url: https://en.wikipedia.org/wiki/Singular_value_decomposition
    note: >-
      A = U Sigma V^T, for every matrix without exception - rectangular,
      singular, defective. This is the decomposition to reach for when you do
      not know what you are holding.
  - name: Spectral theorem
    url: https://en.wikipedia.org/wiki/Spectral_theorem
    note: >-
      A real symmetric matrix has real eigenvalues and an orthonormal
      eigenbasis. It is the reason covariance matrices and PCA behave.
  - name: Low-rank approximation (Eckart-Young)
    url: https://en.wikipedia.org/wiki/Low-rank_approximation
    note: >-
      Truncating the SVD gives the provably best rank-k approximation in both
      the spectral and Frobenius norms. This is what makes the SVD useful
      rather than merely true.
references:
  - text: >-
      Strang, *Introduction to Linear Algebra* - Ch. 6 for eigenvalues and Ch. 7
      for the SVD, with the geometry kept in view throughout.
  - text: >-
      Trefethen & Bau, *Numerical Linear Algebra* - Lectures 4-5 build the SVD
      first and derive everything else from it.
  - text: >-
      Golub & Van Loan, *Matrix Computations* - for how any of this is actually
      computed, which is never by the characteristic polynomial.
---

New to this, or just cold? Open the example first. Otherwise go straight to the problem.

<details class="q-example" id="example">
<summary class="q-example-summary"><span class="q-when-closed">Show a worked example first</span><span class="q-when-open">Hide the example</span></summary>
<div class="q-example-body" markdown="1">

### What an eigenvector is

A matrix $A$ moves vectors around. Almost every vector comes out pointing somewhere new. An **eigenvector** is one of the rare directions that does not turn — it only gets stretched:

$$
Av = \lambda v, \qquad v \ne 0 .
$$

The scalar $\lambda$ is the **eigenvalue**: the stretch factor along that direction. Negative means the vector flips end for end, $\lvert\lambda\rvert < 1$ means it shrinks, $\lambda = 0$ means the direction is annihilated.

Why anyone cares: along an eigenvector, a matrix behaves like a single number. Repeated application is trivial, since $A^k v = \lambda^k v$. If you can write a problem in a basis of eigenvectors, a coupled system decouples into independent scalar ones — which is what diagonalization, PCA, and the stability analysis of a linear system all are.

### How to find them by hand

Rearrange $Av = \lambda v$ into $(A - \lambda I)v = 0$. A non-zero $v$ solves this only if $A - \lambda I$ is singular, so

$$
\det(A - \lambda I) = 0 .
$$

For a $2\times2$ matrix this expands to something worth memorizing:

$$
\lambda^2 - (\operatorname{tr}A)\,\lambda + \det A = 0 .
$$

Then, for each root $\lambda$, solve $(A - \lambda I)v = 0$ for the direction. Two free checks fall out of the same identity:

$$
\lambda_1 + \lambda_2 = \operatorname{tr}A,
\qquad
\lambda_1\lambda_2 = \det A .
$$

### Worked example: a symmetric $2\times2$

Take $A = \begin{bmatrix}2 & 1\\\\ 1 & 2\end{bmatrix}$.

**Eigenvalues.** $\operatorname{tr}A = 4$ and $\det A = 4 - 1 = 3$, so

$$
\lambda^2 - 4\lambda + 3 = 0
\quad\Longrightarrow\quad
(\lambda - 3)(\lambda - 1) = 0
\quad\Longrightarrow\quad
\lambda = 3,\ 1 .
$$

Check: $3 + 1 = 4 = \operatorname{tr}A$ and $3\cdot 1 = 3 = \det A$. ✓

**Eigenvectors.** For $\lambda = 3$,

$$
A - 3I = \begin{bmatrix}-1 & 1\\ 1 & -1\end{bmatrix},
$$

and the equation $-v_1 + v_2 = 0$ says $v_1 = v_2$, so $v = (1,1)$. (Both rows give the same equation — they must, since the matrix is singular. If they do not, you have made an arithmetic error, which is a useful check in itself.)

For $\lambda = 1$, $A - I = \begin{bmatrix}1&1\\\\1&1\end{bmatrix}$ gives $v_1 + v_2 = 0$, so $v = (1,-1)$.

Verify directly: $A(1,1) = (3,3) = 3(1,1)$ ✓ and $A(1,-1) = (1,-1) = 1\cdot(1,-1)$ ✓.

Note the two eigenvectors are perpendicular. That is not luck — $A$ is symmetric, and the spectral theorem guarantees it.

### Where the SVD comes in

Eigenvectors answer *which directions does $A$ leave alone?* A different question is *what shape does $A$ turn the unit circle into?*, and that one is answered by the **singular value decomposition**:

$$
A = U\Sigma V^{\mathsf T},
$$

where $U$ and $V$ are orthogonal — their columns are two orthonormal bases — and $\Sigma$ is diagonal with non-negative entries $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$. Geometrically the $\sigma_i$ are the semi-axis lengths of that ellipse, the $v_i$ are the input directions that land on the axes, and the $u_i$ are the axis directions themselves.

**Underneath, it is still an eigenproblem.** Multiply the factorization by its own transpose and use $U^{\mathsf T}U = I$:

$$
A^{\mathsf T}A = V\Sigma^{\mathsf T}U^{\mathsf T}U\Sigma V^{\mathsf T} = V\Sigma^2 V^{\mathsf T} .
$$

The right-hand side is an ordinary eigendecomposition of the symmetric matrix $A^{\mathsf T}A$. So

$$
v_i = \text{eigenvectors of } A^{\mathsf T}A,
\qquad
\sigma_i = \sqrt{\lambda_i\!\left(A^{\mathsf T}A\right)},
$$

and by the same argument on $AA^{\mathsf T} = U\Sigma^2U^{\mathsf T}$, the $u_i$ are the eigenvectors of $AA^{\mathsf T}$. **The SVD is an eigenproblem — just not of $A$.** That is the entire recipe, and Part 2 of the problem is one turn of it.

### The same matrix, both ways

For $A = \begin{bmatrix}2&1\\\\1&2\end{bmatrix}$, since $A$ is symmetric, $A^{\mathsf T}A = A^2$:

$$
A^{\mathsf T}A = \begin{bmatrix}5&4\\4&5\end{bmatrix},
$$

whose eigenvalues are $5\pm4 = 9$ and $1$, with the same eigenvectors $(1,1)$ and $(1,-1)$ as before. So

$$
\sigma_1 = \sqrt9 = 3, \qquad \sigma_2 = \sqrt1 = 1 ,
$$

which are exactly the eigenvalues we already found, along the same directions. For this matrix $U = V$ and the two decompositions are the same object written twice.

<div class="q-callout q-callout-trap" markdown="1">
<div class="q-callout-title"><i class="fas fa-triangle-exclamation"></i> That coincidence is not the general rule</div>

It happened only because $A$ is symmetric with non-negative eigenvalues. In general:

- **symmetric positive semidefinite** — $\sigma_i = \lambda_i$ and $U = V$; the two decompositions coincide. This is the case for covariance matrices, which is why PCA can be described with either.
- **symmetric, some $\lambda_i < 0$** — $\sigma_i = \lvert\lambda_i\rvert$. The sign cannot survive, since singular values are non-negative by definition; it reappears as a flipped singular vector.
- **anything else** — different numbers *and* different directions. For $\begin{bmatrix}4&1\\\\2&3\end{bmatrix}$ the eigenvalues are $5$ and $2$ while the singular values are $\approx 5.117$ and $\approx 1.954$; the eigenvectors are not even perpendicular, whereas the singular vectors always are.

The structural reason is the shape of the two factorizations. $A = PDP^{-1}$ uses **one** basis on both sides, so it only makes sense when the input and output live in the same space, and it needs enough eigenvectors to span. $A = U\Sigma V^{\mathsf T}$ uses **two** independent orthonormal bases, one for the input space and one for the output. That extra freedom is exactly what lets the SVD exist for every matrix — rectangular, singular, or defective — while the eigendecomposition can fail outright, as parts (b) and (c) of the problem show.

</div>

### The picture

Feed $A$ every unit vector and you get an ellipse. The eigenvector directions are the ones where the output stays on the same line through the origin; the singular vectors are the ones that land on the ellipse's axes.

<div class="viz" data-viz="linear-map">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; what a matrix does to the unit circle</p>
    <p class="viz-sub">Drag the green point around the circle and watch where it lands. The dashed pink lines are the eigenvector directions; on those, and only those, the output arrow stays parallel to the input. Type your own matrix in, or try the presets — including one with no real eigenvectors at all.</p>
  </div>
</div>

Both families are drawn. On the symmetric presets — $\begin{bmatrix}2&1\\\\1&2\end{bmatrix}$ and $\begin{bmatrix}1&2\\\\2&4\end{bmatrix}$ — the pink eigenvector lines sit exactly along the orange singular axes. On $\begin{bmatrix}4&1\\\\2&3\end{bmatrix}$ they visibly come apart. On the rotation $\begin{bmatrix}0&-1\\\\1&0\end{bmatrix}$ there are no pink lines at all, yet the orange axes are still there: the SVD does not mind.

</div>
</details>

<div class="q-problem" markdown="1">
<div class="q-problem-label">Problem</div>

**Part 1 — eigenvalues and eigenvectors.** Find them for each matrix, by hand.

$$
\text{(a)}\;\; A = \begin{bmatrix}4 & 1\\ 2 & 3\end{bmatrix}
\qquad
\text{(b)}\;\; R = \begin{bmatrix}0 & -1\\ 1 & 0\end{bmatrix}
\qquad
\text{(c)}\;\; N = \begin{bmatrix}3 & 1\\ 0 & 3\end{bmatrix}
$$

For (b) and (c), say what goes wrong and what it means geometrically.

**Part 2 — SVD.** For

$$
B = \begin{bmatrix}3 & 0\\ 4 & 5\end{bmatrix},
$$

compute the full singular value decomposition $B = U\Sigma V^{\mathsf T}$ by hand.

**Part 3.** From that decomposition alone, write down the rank of $B$, its spectral norm $\lVert B\rVert_2$, its Frobenius norm $\lVert B\rVert_F$, its condition number $\kappa_2(B)$, and the best rank-1 approximation to $B$ together with the error it incurs.

**Part 4.** For a real symmetric matrix $S$, how do the singular values relate to the eigenvalues? Give a $2\times2$ example where the two lists are not the same.

</div>

Work them on paper first — that is what the page is for.

<details class="q-solution" id="solution">
<summary class="q-solution-summary"><span class="q-when-closed">Show the solution</span><span class="q-when-open">Hide the solution</span></summary>
<div class="q-solution-body" markdown="1">

## The answers

| | | result |
|---|---|---|
| (a) | $\begin{bmatrix}4&1\\\\2&3\end{bmatrix}$ | $\lambda = 5$ with $v=(1,1)$; $\lambda = 2$ with $v=(1,-2)$ |
| (b) | $\begin{bmatrix}0&-1\\\\1&0\end{bmatrix}$ | $\lambda = \pm j$; no real eigenvectors — it is a $90^\circ$ rotation |
| (c) | $\begin{bmatrix}3&1\\\\0&3\end{bmatrix}$ | $\lambda = 3$ twice, but only one eigenvector $(1,0)$ — defective |
| 2 | $B = \begin{bmatrix}3&0\\\\4&5\end{bmatrix}$ | $\sigma_1 = 3\sqrt5,\ \sigma_2 = \sqrt5$; $V = \tfrac{1}{\sqrt2}\begin{bmatrix}1&1\\\\1&-1\end{bmatrix}$, $U = \tfrac{1}{\sqrt{10}}\begin{bmatrix}1&3\\\\3&-1\end{bmatrix}$ |
| 3 | | rank 2, $\lVert B\rVert_2 = 3\sqrt5$, $\lVert B\rVert_F = \sqrt{50}$, $\kappa_2 = 3$, best rank-1 is $\begin{bmatrix}1.5&1.5\\\\4.5&4.5\end{bmatrix}$ with error $\sqrt5$ |
| 4 | | $\sigma_i = \lvert\lambda_i\rvert$; e.g. $\operatorname{diag}(1,-2)$ has $\lambda = 1,-2$ but $\sigma = 2,1$ |

## Part 1(a): the ordinary case

$\operatorname{tr}A = 7$, $\det A = 12 - 2 = 10$, so

$$
\lambda^2 - 7\lambda + 10 = 0 \quad\Longrightarrow\quad (\lambda-5)(\lambda-2) = 0 .
$$

For $\lambda = 5$: $A - 5I = \begin{bmatrix}-1&1\\\\2&-2\end{bmatrix}$, giving $v_1 = v_2$, so $v = (1,1)$.

For $\lambda = 2$: $A - 2I = \begin{bmatrix}2&1\\\\2&1\end{bmatrix}$, giving $2v_1 + v_2 = 0$, so $v = (1,-2)$.

Check: $A(1,1) = (5,5)$ ✓ and $A(1,-2) = (2,-4) = 2(1,-2)$ ✓. The eigenvectors are *not* perpendicular here — $A$ is not symmetric, so nothing promised they would be.

Since there are two independent eigenvectors, $A$ is diagonalizable:

$$
A = PDP^{-1},
\qquad
P = \begin{bmatrix}1 & 1\\ 1 & -2\end{bmatrix},
\qquad
D = \begin{bmatrix}5 & 0\\ 0 & 2\end{bmatrix} .
$$

## Part 1(b): complex eigenvalues mean rotation

$\operatorname{tr}R = 0$, $\det R = 1$, so $\lambda^2 + 1 = 0$ and $\lambda = \pm j$.

There are **no real eigenvectors**, and the reason is entirely geometric: $R$ rotates the plane by $90^\circ$, so no real direction survives — every line through the origin is moved to a different line. An eigenvector would have to be a direction that rotation leaves alone, and there isn't one.

Over $\mathbb{C}$ the eigenvectors do exist. For $\lambda = j$, the second row of $(R - jI)v = 0$ reads $v_1 - jv_2 = 0$, so $v = (j, 1)$; for $\lambda = -j$, $v = (-j, 1)$.

<div class="q-callout q-callout-insight" markdown="1">
<div class="q-callout-title"><i class="fas fa-lightbulb"></i> The pattern worth carrying</div>

For a real matrix, complex eigenvalues always come in conjugate pairs $\lambda = re^{\pm j\theta}$, and they say the matrix acts on some plane as a rotation by $\theta$ combined with a scaling by $r$. Here $\lambda = \pm j = e^{\pm j\pi/2}$: modulus 1, angle $90^\circ$ — a pure rotation, no scaling. That is the same $re^{j\theta}$ reading as in the [rectangular-to-polar question](/interview-prep/signal-processing/complex-rect-to-polar/), and it is why $\lvert\lambda\rvert$ rather than $\lambda$ decides whether iterating a system grows or decays.

</div>

## Part 1(c): defective, and no fixing it

$\operatorname{tr}N = 6$, $\det N = 9$, so $\lambda^2 - 6\lambda + 9 = (\lambda-3)^2$: the eigenvalue $3$ has **algebraic multiplicity** 2.

But solving $(N - 3I)v = 0$ with

$$
N - 3I = \begin{bmatrix}0 & 1\\ 0 & 0\end{bmatrix}
$$

gives $v_2 = 0$ and $v_1$ free — a *one*-dimensional eigenspace spanned by $(1,0)$. The **geometric multiplicity** is 1.

When geometric multiplicity is less than algebraic, the matrix is **defective**: there is no basis of eigenvectors, so $N$ cannot be diagonalized. No cleverness helps; the nearest normal form is the Jordan block $N$ already is.

Note this is not the same failure as (b). There the eigenvalues were fine but lived in $\mathbb{C}$; here they are real and there simply are not enough eigenvectors. Both are reasons the eigendecomposition can fail you — and neither one troubles the SVD.

## Part 2: computing an SVD by hand

The claim is that **every** matrix factors as

$$
A = U\Sigma V^{\mathsf T},
$$

with $U$ and $V$ orthogonal and $\Sigma$ diagonal with non-negative entries $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$. No squareness, no symmetry, no diagonalizability required.

The recipe comes from one observation. If $A = U\Sigma V^{\mathsf T}$ then

$$
A^{\mathsf T}A = V\Sigma^{\mathsf T}U^{\mathsf T}U\Sigma V^{\mathsf T} = V\Sigma^2 V^{\mathsf T} ,
$$

because $U^{\mathsf T}U = I$. That is an ordinary eigendecomposition of a symmetric matrix. So:

1. eigen-decompose $A^{\mathsf T}A$ — its eigenvectors are the columns of $V$, its eigenvalues are $\sigma_i^2$;
2. take $\sigma_i = \sqrt{\lambda_i}$, which is real and non-negative because $A^{\mathsf T}A$ is positive semidefinite;
3. recover $u_i = \dfrac{Av_i}{\sigma_i}$ for every $\sigma_i > 0$.

**Step 1.** With $B = \begin{bmatrix}3&0\\\\4&5\end{bmatrix}$,

$$
B^{\mathsf T}B
= \begin{bmatrix}3&4\\0&5\end{bmatrix}\begin{bmatrix}3&0\\4&5\end{bmatrix}
= \begin{bmatrix}25 & 20\\ 20 & 25\end{bmatrix}.
$$

This is symmetric, as it must be. Its trace is $50$ and determinant $625 - 400 = 225$, so $\lambda^2 - 50\lambda + 225 = 0$, giving $\lambda = 45$ and $\lambda = 5$.

For a matrix of the form $\begin{bmatrix}p&q\\\\q&p\end{bmatrix}$ the eigenvectors are always $(1,1)$ and $(1,-1)$, with eigenvalues $p+q$ and $p-q$ — here $45$ and $5$, confirming the arithmetic. Normalized:

$$
v_1 = \tfrac{1}{\sqrt2}(1,1),
\qquad
v_2 = \tfrac{1}{\sqrt2}(1,-1) .
$$

**Step 2.** $\sigma_1 = \sqrt{45} = 3\sqrt5 \approx 6.7082$ and $\sigma_2 = \sqrt5 \approx 2.2361$.

Quick check: $\sigma_1\sigma_2 = 3\sqrt5\cdot\sqrt5 = 15 = \lvert\det B\rvert$ ✓. That identity — the product of the singular values is the absolute determinant — is worth remembering, since geometrically both count the area scaling.

**Step 3.**

$$
Bv_1 = \tfrac{1}{\sqrt2}\begin{bmatrix}3&0\\4&5\end{bmatrix}\begin{bmatrix}1\\1\end{bmatrix}
     = \tfrac{1}{\sqrt2}\begin{bmatrix}3\\9\end{bmatrix},
\qquad
u_1 = \frac{Bv_1}{\sigma_1} = \frac{1}{\sqrt2 \cdot 3\sqrt5}\begin{bmatrix}3\\9\end{bmatrix}
    = \frac{1}{\sqrt{10}}\begin{bmatrix}1\\3\end{bmatrix}.
$$

$$
Bv_2 = \tfrac{1}{\sqrt2}\begin{bmatrix}3\\-1\end{bmatrix},
\qquad
u_2 = \frac{Bv_2}{\sigma_2} = \frac{1}{\sqrt2\cdot\sqrt5}\begin{bmatrix}3\\-1\end{bmatrix}
    = \frac{1}{\sqrt{10}}\begin{bmatrix}3\\-1\end{bmatrix}.
$$

Both are unit vectors, and $u_1\cdot u_2 = (3-3)/10 = 0$ — orthogonal, as promised. Assembling:

$$
B = \underbrace{\frac{1}{\sqrt{10}}\begin{bmatrix}1&3\\3&-1\end{bmatrix}}_{U}
\underbrace{\begin{bmatrix}3\sqrt5&0\\0&\sqrt5\end{bmatrix}}_{\Sigma}
\underbrace{\frac{1}{\sqrt2}\begin{bmatrix}1&1\\1&-1\end{bmatrix}^{\mathsf T}}_{V^{\mathsf T}} .
$$

### What the three factors are doing

<div class="viz" data-viz="svd-stages">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; rotate, stretch, rotate</p>
    <p class="viz-sub">Step through the factors one at a time. V transpose turns the singular directions onto the coordinate axes, Sigma stretches along those axes, and U turns the result into its final orientation. The circle only ever changes shape in the middle step.</p>
  </div>
</div>

That is the whole geometric content of the SVD: **every linear map is a rotation, then an axis-aligned scaling, then another rotation.** The $v_i$ are the input directions that get sent to the axes of the output ellipse; the $u_i$ are those axes; the $\sigma_i$ are their half-lengths.

## Part 3: reading the SVD

Once you have $\sigma_1 = 3\sqrt5$ and $\sigma_2 = \sqrt5$, a pile of quantities are free.

**Rank** is the number of non-zero singular values: both are non-zero, so $\operatorname{rank}B = 2$. This is the numerically honest definition of rank — counting non-zero pivots is hopeless in floating point, whereas "how many $\sigma_i$ are meaningfully above zero" is a question you can actually answer.

**Spectral norm** is the largest stretch the matrix can apply to any unit vector, which is exactly the longest axis of the ellipse:

$$
\lVert B\rVert_2 = \sigma_1 = 3\sqrt5 \approx 6.708 .
$$

**Frobenius norm** is the root-sum-square of the entries, and also of the singular values:

$$
\lVert B\rVert_F = \sqrt{\textstyle\sum_i \sigma_i^2} = \sqrt{45+5} = \sqrt{50} \approx 7.071 .
$$

Cross-check against the entries: $\sqrt{9+0+16+25} = \sqrt{50}$ ✓.

**Condition number** is the ratio of the largest to the smallest stretch — how badly the matrix distorts, and how much a solve can amplify error:

$$
\kappa_2(B) = \frac{\sigma_1}{\sigma_2} = \frac{3\sqrt5}{\sqrt5} = 3 .
$$

A condition number of 3 is excellent. As $\sigma_{\min} \to 0$ the matrix approaches singular and $\kappa \to \infty$.

**Best rank-1 approximation.** Keep the largest term of $A = \sum_i \sigma_i u_i v_i^{\mathsf T}$ and discard the rest:

$$
B_1 = \sigma_1 u_1 v_1^{\mathsf T}
= 3\sqrt5 \cdot \frac{1}{\sqrt{10}}\begin{bmatrix}1\\3\end{bmatrix}\cdot\frac{1}{\sqrt2}\begin{bmatrix}1&1\end{bmatrix}
= \frac{3\sqrt5}{\sqrt{20}}\begin{bmatrix}1&1\\3&3\end{bmatrix}
= \begin{bmatrix}1.5 & 1.5\\ 4.5 & 4.5\end{bmatrix},
$$

using $3\sqrt5/\sqrt{20} = 3\sqrt5/(2\sqrt5) = 3/2$. The error is the largest singular value you threw away:

$$
\lVert B - B_1\rVert_2 = \sigma_2 = \sqrt5 .
$$

That is the **Eckart–Young theorem**: truncating the SVD at $k$ terms is not merely *a* good rank-$k$ approximation, it is provably the best one in both the spectral and Frobenius norms. Every use of the SVD for compression, denoising, PCA, or latent-factor models is this one fact.

## Part 4: symmetric matrices

If $S$ is real symmetric, the spectral theorem gives $S = Q\Lambda Q^{\mathsf T}$ with $Q$ orthogonal and $\Lambda$ real diagonal. Then

$$
S^{\mathsf T}S = S^2 = Q\Lambda^2 Q^{\mathsf T},
$$

so the singular values are $\sigma_i = \sqrt{\lambda_i^2} = \lvert\lambda_i\rvert$.

**The singular values are the absolute values of the eigenvalues** — the SVD cannot see a sign, because $\sigma_i \ge 0$ by construction. A negative eigenvalue reappears as a flipped singular vector: if $\lambda_i < 0$ then $u_i = -q_i$ while $v_i = q_i$.

The example asked for: $S = \operatorname{diag}(1,-2)$ has eigenvalues $1$ and $-2$, but singular values $2$ and $1$ — a different pair of numbers, and even in a different order.

The two lists coincide exactly when $S$ is positive semidefinite. That is why covariance matrices behave so agreeably: they are symmetric PSD, so eigen and SVD give the same answer, and PCA can be described using either without anyone getting hurt.

## Eigen or SVD?

| | eigendecomposition | SVD |
|---|---|---|
| exists for | square matrices, and not always even then | every matrix, always |
| shape | any square | any $m\times n$ |
| values | can be complex, can be negative | real, non-negative, ordered |
| vectors | not generally orthogonal; may not span | two orthonormal bases, always |
| fails when | defective, or complex spectrum | never |
| answers | how does $A$ act when applied repeatedly | how does $A$ distort space, once |

The rule of thumb: **iteration and dynamics want eigenvalues** — $A^k$, matrix exponentials, stability, PageRank, Markov chains. **Geometry, approximation and conditioning want singular values** — rank, norms, least squares, PCA, compression.

## The slips worth naming

- Forgetting to normalize eigenvectors when the question calls for an orthogonal $P$ or $V$; eigenvector *direction* is what is determined, length is your choice.
- Reporting singular values as eigenvalues of $A$. They are square roots of eigenvalues of $A^{\mathsf T}A$, and even for symmetric $A$ they differ by a sign.
- Assuming eigenvectors are orthogonal. Only guaranteed for symmetric (more generally, normal) matrices.
- Treating a repeated eigenvalue as automatically defective. Check the rank of $A - \lambda I$: the scalar matrix $2I$ repeats its eigenvalue and is perfectly diagonalizable.
- Getting $u_i = Av_i/\sigma_i$ backwards, or trying to use it when $\sigma_i = 0$ — those columns of $U$ have to be filled in from the orthogonal complement instead.
- Computing eigenvalues from the characteristic polynomial in code. Fine by hand for $2\times2$; numerically disastrous at any real size, where the roots of a polynomial are wildly ill-conditioned in its coefficients.

</div>
</details>

## More practice

<details class="q-reveal">
<summary>Drill 1 &mdash; eigenvalues and eigenvectors of $\begin{bmatrix}5&4\\1&2\end{bmatrix}$</summary>
<div class="q-reveal-body" markdown="1">

$\operatorname{tr} = 7$, $\det = 10 - 4 = 6$, so $\lambda^2 - 7\lambda + 6 = (\lambda-6)(\lambda-1)$, giving $\lambda = 6, 1$.

$\lambda = 6$: $\begin{bmatrix}-1&4\\\\1&-4\end{bmatrix}$ gives $v_1 = 4v_2$, so $v = (4,1)$.

$\lambda = 1$: $\begin{bmatrix}4&4\\\\1&1\end{bmatrix}$ gives $v_1 = -v_2$, so $v = (1,-1)$.

Check: $A(4,1) = (24,6) = 6(4,1)$ ✓, $A(1,-1) = (1,-1)$ ✓.

</div>
</details>

<details class="q-reveal">
<summary>Drill 2 &mdash; use the eigendecomposition to compute $A^{10}$ for $A = \begin{bmatrix}2&1\\1&2\end{bmatrix}$</summary>
<div class="q-reveal-body" markdown="1">

From the example, $\lambda = 3, 1$ with eigenvectors $(1,1)$ and $(1,-1)$. So $A = PDP^{-1}$ with

$$
P = \begin{bmatrix}1&1\\1&-1\end{bmatrix},
\quad
D = \begin{bmatrix}3&0\\0&1\end{bmatrix},
\quad
P^{-1} = \tfrac12\begin{bmatrix}1&1\\1&-1\end{bmatrix} .
$$

The point of diagonalizing is that the inner factors telescope:

$$
A^{n} = PD^{n}P^{-1}
= \tfrac12\begin{bmatrix}1&1\\1&-1\end{bmatrix}
  \begin{bmatrix}3^{n}&0\\0&1\end{bmatrix}
  \begin{bmatrix}1&1\\1&-1\end{bmatrix}
= \frac12\begin{bmatrix}3^{n}+1 & 3^{n}-1\\ 3^{n}-1 & 3^{n}+1\end{bmatrix}.
$$

Sanity check at $n=1$: $\tfrac12\begin{bmatrix}4&2\\\\2&4\end{bmatrix} = A$ ✓. At $n = 10$, $3^{10} = 59049$, so

$$
A^{10} = \begin{bmatrix}29525 & 29524\\ 29524 & 29525\end{bmatrix}.
$$

Nine matrix multiplications avoided. The same trick with $e^{At} = Pe^{Dt}P^{-1}$ solves linear ODE systems, and the growth rate being governed by the largest $\lvert\lambda\rvert$ is exactly what stability analysis is reading off.

</div>
</details>

<details class="q-reveal">
<summary>Drill 3 &mdash; SVD of the rank-deficient $\begin{bmatrix}1&2\\2&4\end{bmatrix}$</summary>
<div class="q-reveal-body" markdown="1">

The second row is twice the first, so the rank is 1 and one singular value must vanish. Confirm it:

$$
A^{\mathsf T}A = \begin{bmatrix}5&10\\10&20\end{bmatrix},
\qquad
\operatorname{tr} = 25,\quad \det = 5\cdot 20 - 10\cdot 10 = 0 ,
$$

so $\lambda = 25, 0$ and $\sigma_1 = 5$, $\sigma_2 = 0$.

For $\lambda = 25$: $\begin{bmatrix}-20&10\\\\10&-5\end{bmatrix}$ gives $2v_1 = v_2$, so $v_1 = \tfrac{1}{\sqrt5}(1,2)$. Then

$$
u_1 = \frac{Av_1}{\sigma_1} = \frac{1}{5\sqrt5}\begin{bmatrix}5\\10\end{bmatrix} = \frac{1}{\sqrt5}\begin{bmatrix}1\\2\end{bmatrix},
$$

which is the same vector as $v_1$ — unsurprising, since $A$ is symmetric and positive semidefinite. The whole matrix is a single outer product:

$$
A = 5\,u_1v_1^{\mathsf T} = 5\cdot\frac{1}{5}\begin{bmatrix}1\\2\end{bmatrix}\begin{bmatrix}1&2\end{bmatrix}
  = \begin{bmatrix}1&2\\2&4\end{bmatrix}
$$

as required.

$\sigma_2 = 0$ says the direction $v_2 = \tfrac{1}{\sqrt5}(2,-1)$ is sent to zero — it spans the null space. Geometrically the unit circle does not become an ellipse at all; it collapses onto a line segment.

</div>
</details>
