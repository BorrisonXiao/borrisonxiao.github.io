---
layout: blog_post
title: Linear CKA Overview
tags: [Machine Learning]
---

Note: This post aims to give an intuitive but mathematically precise introduction to linear centered kernel alignment (CKA). The focus is on what each component of the formula does and why properties such as translation, rotation and scale invariance are useful when comparing neural representations.

## Introduction

Suppose that we feed the same $n$ inputs into two neural networks, or into two different layers of one network. The resulting representations are denoted by

$$
X \in \mathbb{R}^{n \times d_x}
\qquad \text{and} \qquad
Y \in \mathbb{R}^{n \times d_y},
$$

where $n$ is the number of input examples, $d_x$ is the number of features in $X$, and $d_y$ is the number of features in $Y$. The $i$-th row of $X$, denoted by $x_i^\top \in \mathbb{R}^{1 \times d_x}$, is the representation of input $i$; similarly, $y_i^\top$ is the representation of the same input in $Y$.

A natural question is: **do $X$ and $Y$ represent the inputs in a similar way?**

Directly comparing their features is usually not meaningful. For example, feature 1 in one network does not need to correspond to feature 1 in another network. A network may permute, negate or rotate its features while preserving the same underlying geometry. Linear CKA avoids requiring a neuron-to-neuron correspondence. Instead, it asks whether the two representations produce similar relationships among the $n$ examples.

The linear CKA score is

$$
\operatorname{CKA}(X,Y)
=
\frac{\left\|X_c^\top Y_c\right\|_F^2}
{\left\|X_c^\top X_c\right\|_F
 \left\|Y_c^\top Y_c\right\|_F},
$$

where $X_c$ and $Y_c$ are the centered versions of $X$ and $Y$, and $\|\cdot\|_F$ denotes the Frobenius norm. We will introduce each component below.

## Centering the Representations

Let

$$
\mathbf{1}_n=
\begin{bmatrix}
1 & 1 & \cdots & 1
\end{bmatrix}^{\top}
\in \mathbb{R}^{n}
$$

be an $n$-dimensional column vector whose entries are all $1$. The mean representation of $X$ is

$$
\mu_X = \frac{1}{n}X^\top\mathbf{1}_n
\in \mathbb{R}^{d_x}.
$$

To understand $X^\top\mathbf{1}_n$, note that

$$
X^\top \in \mathbb{R}^{d_x \times n}
$$

has one row for each feature and one column for each example. Multiplying a row by $\mathbf{1}_n$ adds all entries in that row. Therefore,

$$
X^\top\mathbf{1}_n
=
\begin{bmatrix}
\sum_{i=1}^{n}X_{i1}\\
\sum_{i=1}^{n}X_{i2}\\
\vdots\\
\sum_{i=1}^{n}X_{id_x}
\end{bmatrix},
$$

which contains the sum of each feature across the $n$ examples. Dividing by $n$ gives the feature-wise means.

For instance, let

$$
X=
\begin{bmatrix}
1 & 10\\
3 & 20\\
5 & 30
\end{bmatrix}.
$$

Then

$$
X^\top\mathbf{1}_3
=
\begin{bmatrix}
1 & 3 & 5\\
10 & 20 & 30
\end{bmatrix}
\begin{bmatrix}
1\\1\\1
\end{bmatrix}
=
\begin{bmatrix}
9\\60
\end{bmatrix},
$$

and hence

$$
\mu_X=\frac{1}{3}
\begin{bmatrix}
9\\60
\end{bmatrix}
=
\begin{bmatrix}
3\\20
\end{bmatrix}.
$$

The matrix

$$
\mathbf{1}_n\mu_X^\top \in \mathbb{R}^{n\times d_x}
$$

repeats the mean row $\mu_X^\top$ once for each example. The centered representation is therefore

$$
X_c=X-\mathbf{1}_n\mu_X^\top.
$$

Equivalently, define the centering matrix

$$
H=I_n-\frac{1}{n}\mathbf{1}_n\mathbf{1}_n^\top
\in \mathbb{R}^{n\times n},
$$

where $I_n$ is the $n\times n$ identity matrix. Then

$$
X_c=HX
\qquad \text{and} \qquad
Y_c=HY.
$$

The operation $HX$ keeps each row of $X$ and subtracts the average row. Thus, the centered representation describes how each example differs from the average example.

Centering gives **translation invariance**. Suppose that the same vector $b\in\mathbb{R}^{d_x}$ is added to every row:

$$
X'=X+\mathbf{1}_n b^\top.
$$

Because $H\mathbf{1}_n=\mathbf{0}$,

$$
HX'=HX+H\mathbf{1}_n b^\top=HX.
$$

Therefore, a shared offset does not affect CKA. This is desirable because adding the same bias to every example changes the origin of the feature space, but does not change the relationships among the examples.

## Gram Matrices: Comparing Examples Instead of Features

After centering, define the linear Gram matrices

$$
K=X_cX_c^\top \in \mathbb{R}^{n\times n}
\qquad \text{and} \qquad
L=Y_cY_c^\top \in \mathbb{R}^{n\times n}.
$$

The $(i,j)$-th entry of $K$ is

$$
K_{ij}=x_{c,i}^\top x_{c,j},
$$

where $x_{c,i},x_{c,j}\in\mathbb{R}^{d_x}$ are the centered representations of examples $i$ and $j$. Thus, $K_{ij}$ is their inner product: it is large when the two centered vectors point in similar directions, negative when they point in opposing directions, and near zero when they are close to orthogonal.

Importantly, $K$ is always $n\times n$, regardless of the number of features. Hence, $X$ and $Y$ may have different dimensions, i.e. $d_x\neq d_y$, while $K$ and $L$ are still directly comparable.

Using the Gram matrices, linear CKA can be written as

$$
\operatorname{CKA}(X,Y)
=
\frac{\langle K,L\rangle_F}
{\|K\|_F\|L\|_F},
$$

where $\langle K,L\rangle_F$ is the Frobenius inner product. In this form, CKA is simply the cosine similarity between the flattened Gram matrices. It measures whether pairs of examples that are similar in $X$ also tend to be similar in $Y$.

## Frobenius Inner Product and Frobenius Norm

For two matrices $A,B\in\mathbb{R}^{p\times q}$ of the same shape, their Frobenius inner product is

$$
\langle A,B\rangle_F
=
\sum_{i=1}^{p}\sum_{j=1}^{q}A_{ij}B_{ij}.
$$

This is the ordinary dot product after laying all entries of each matrix into one long vector.

The Frobenius norm is

$$
\|A\|_F
=
\sqrt{\sum_{i=1}^{p}\sum_{j=1}^{q}A_{ij}^2}.
$$

It is the Euclidean length of that flattened vector. For example,

$$
A=
\begin{bmatrix}
1&2\\
3&4
\end{bmatrix}
\quad \Longrightarrow \quad
\|A\|_F=\sqrt{1^2+2^2+3^2+4^2}=\sqrt{30}.
$$

Consequently,

$$
\frac{\langle K,L\rangle_F}{\|K\|_F\|L\|_F}
$$

has exactly the same form as cosine similarity between two vectors. The numerator measures their raw alignment, while the denominator removes the effect of their overall magnitudes.

## Cross-Covariance View

Linear CKA can also be written in feature space. First, observe that

$$
\begin{align*}
\langle K,L\rangle_F
&=\langle X_cX_c^\top,Y_cY_c^\top\rangle_F\\
&=\left\|X_c^\top Y_c\right\|_F^2.
\end{align*}
$$

The matrix

$$
X_c^\top Y_c\in\mathbb{R}^{d_x\times d_y}
$$

contains the associations between every feature of $X$ and every feature of $Y$. More precisely, its $(a,b)$-th entry is

$$
\left[X_c^\top Y_c\right]_{ab}
=
\sum_{i=1}^{n}(X_c)_{ia}(Y_c)_{ib}.
$$

This takes feature $a$ from $X$ and feature $b$ from $Y$, multiplies their centered values for each example, and sums the products. If the two features tend to be positive and negative on the same examples, the sum is positive; if they tend to move in opposite directions, it is negative; if they have little systematic relationship, positive and negative products tend to cancel.

The sample cross-covariance matrix is formally

$$
C_{XY}=\frac{1}{n-1}X_c^\top Y_c.
$$

Similarly,

$$
C_{XX}=\frac{1}{n-1}X_c^\top X_c
\qquad \text{and} \qquad
C_{YY}=\frac{1}{n-1}Y_c^\top Y_c
$$

are the within-representation covariance matrices. The factor $1/(n-1)$ cancels in the CKA ratio, so linear CKA may be written as

$$
\operatorname{CKA}(X,Y)
=
\frac{\|C_{XY}\|_F^2}
{\|C_{XX}\|_F\|C_{YY}\|_F}.
$$

The numerator aggregates the squared associations between all feature pairs. Squaring is important: strong positive and strong negative associations both indicate that the two feature spaces are related, and they should not cancel one another.

## Orthogonal Matrices and Rotation Invariance

A square matrix

$$
Q\in\mathbb{R}^{d_x\times d_x}
$$

is called **orthogonal** if

$$
Q^\top Q=QQ^\top=I_{d_x}.
$$

This means that

$$
Q^{-1}=Q^\top.
$$

The columns of $Q$ are mutually perpendicular unit vectors. Multiplication by $Q$ changes the coordinate system without stretching it. In particular, for any vectors $u,v\in\mathbb{R}^{d_x}$,

$$
(Qu)^\top(Qv)=u^\top Q^\top Qv=u^\top v,
$$

and hence

$$
\|Qu\|_2=\|u\|_2.
$$

Therefore, orthogonal transformations preserve lengths, angles and inner products. Rotations are orthogonal transformations; reflections, feature permutations and feature-wise sign flips are also orthogonal transformations. Strictly speaking, CKA is invariant to this entire class, not only to rotations.

Suppose that we transform the features of $X_c$ by

$$
X_c'=X_cQ.
$$

Its Gram matrix becomes

$$
\begin{align*}
K'
&=X_c'X_c'^\top\\
&=X_cQQ^\top X_c^\top\\
&=X_cI_{d_x}X_c^\top\\
&=K.
\end{align*}
$$

Thus, the pairwise inner products among the examples are unchanged, and therefore

$$
\operatorname{CKA}(X_cQ,Y_c)=\operatorname{CKA}(X_c,Y_c).
$$

This explains **why CKA is rotation invariant**: the Gram matrix $X_cX_c^\top$ discards the arbitrary orientation of the feature axes while retaining the inner-product geometry among examples.

This property is useful because two neural networks can encode the same information using different feature directions. Requiring their individual neurons to match would incorrectly label such representations as different.

## Normalization and Scale Invariance

Suppose that every entry of $X_c$ is multiplied by a scalar $a\neq 0$:

$$
X_c'=aX_c.
$$

Then its Gram matrix is

$$
K'=X_c'X_c'^\top=a^2K.
$$

The CKA numerator becomes

$$
\langle a^2K,L\rangle_F=a^2\langle K,L\rangle_F,
$$

while the corresponding denominator term becomes

$$
\|a^2K\|_F=a^2\|K\|_F.
$$

The factor $a^2$ therefore cancels, giving

$$
\operatorname{CKA}(aX_c,Y_c)=\operatorname{CKA}(X_c,Y_c).
$$

Hence, the normalization provides **isotropic scale invariance**, where *isotropic* means that every feature is scaled by the same amount. This is useful because two layers may have very different activation magnitudes while preserving the same representational structure.

However, linear CKA is generally not invariant to scaling each feature by a different amount. If

$$
X_c'=X_cA
$$

for a general invertible matrix $A$, then

$$
X_c'X_c'^\top=X_cAA^\top X_c^\top,
$$

which is not generally equal to $X_cX_c^\top$. Such a transformation may stretch some directions more than others and therefore changes the geometry among examples. CKA intentionally remains sensitive to this change.

## Summary

Linear CKA compares two representations by comparing the centered pairwise similarity structures that they induce among the same examples:

$$
\boxed{
\operatorname{CKA}(X,Y)
=
\frac{\langle X_cX_c^\top,Y_cY_c^\top\rangle_F}
{\|X_cX_c^\top\|_F\|Y_cY_c^\top\|_F}
}
$$

The main components have the following roles:

1). **Centering**, $X_c=HX$, removes the average representation and gives invariance to a shared feature-space offset;

2). **Gram matrices**, $X_cX_c^\top$ and $Y_cY_c^\top$, describe relationships among examples instead of requiring features to correspond;

3). **The Frobenius inner product** measures agreement between the two pairwise similarity matrices;

4). **The Frobenius-norm normalization** removes overall magnitude and gives invariance to global scaling;

5). **The use of inner-product geometry** gives invariance to orthogonal transformations, including rotations, reflections, feature permutations and sign flips.

Thus, CKA ignores several arbitrary choices of representation coordinates, while remaining sensitive to transformations that genuinely distort the geometry of the represented examples.

## Reference

Simon Kornblith, Mohammad Norouzi, Honglak Lee and Geoffrey Hinton. [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html). *Proceedings of the 36th International Conference on Machine Learning*, 2019.
