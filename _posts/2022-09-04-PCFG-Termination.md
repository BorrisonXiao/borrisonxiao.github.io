---
layout: post
title: PCFG Termination
tags: [NLP]
---

In general, a probabilistic context-free grammar (PCFG) in Chomsky normal form may be **improper**, i.e. the sum of the probability of parses successfully generated may be less than 1.

For instance, given the PCFG below:
$$
\begin{align*}
p \qquad &S \rightarrow S \ S\\
1 - p \qquad &S \rightarrow x
\end{align*}
$$
in which $S$ is the only non-terminal, $x$ is the only terminal and the probability of picking the rule $S \rightarrow S \ S$ is $p$ and $S \rightarrow x$ is $1 - p$.

Let us define $x_h$ to be the **total probability** of all parse trees with height $\leq h$. In fact, $x_h$ can be computed through a recursive equation:
$$x_{h+1} = 1 - p + px^2_{h}$$

The intuition of the recursive equation is from the fact that the set of all trees with height $\leq h + 1$, i.e. $T_{h+1}$, consists of the the base case tree $t_1$ with one single expansion $S \rightarrow x$ and all trees in the form of a root $S$ and two subtrees in $t_h \in T_h$ for all $h > 1$, as shown in the graph below:


```python
from graphviz import Digraph

tree1 = Digraph(name="cluster0", node_attr={"shape": "plaintext"})
tree1.node("1_1", label="S")
tree1.node("1_2", label="x")
tree1.edge("1_1", "1_2")
tree1.edge_attr.update(arrowhead='none')

tree2 = Digraph(name="cluster1", node_attr={"shape": "plaintext"})
tree2.attr(color="transparent")
tree2.node("3", label="S")
tree2.node("4", label="S")
tree2.node("5", label="S")
tree2.node("6", label="x")
tree2.node("7", label="x")
tree2.edges(["34", "35", "46", "57"])
tree2.edge_attr.update(arrowhead='none')

tree3 = Digraph(name="cluster2", node_attr={"shape": "plaintext"})
tree3.attr(color="transparent")
tree3.node("3_1", label="S")
tree3.node("3_2", label="t_h")
tree3.node("3_3", label="t_h")
tree3.edge("3_1", "3_2")
tree3.edge("3_1", "3_3")
tree3.edge_attr.update(arrowhead='none')

tree1.subgraph(tree2)
tree1.subgraph(tree3)
tree1
```





![svg](../assets/blogs/PCFG-Termination_files/PCFG-Termination_2_0.svg)




In the figure above, the leftmost tree is the base case $h = 1$, and obviously $x_1 = 1 - p$.

The rightmost tree is the general form of any tree $t_{h + 1}$ with $h > 1$, whose total probability
$$
\begin{align*}
\sum_{t \in T_{h+1}}P(t) = x_{h + 1} &= \underbrace{P(S \rightarrow x)}_{\text{Base case}} + \underbrace{P(S \rightarrow x)(\sum_{t_h \in T_h}P(t_h))^2}_{\text{Recursion}}\\
&= 1 - p + px^2_h
\end{align*}
$$


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
def rec(x, p): return 1 - p + p * x**2

def compute_rec(rec, init, args, iter):
    res = [init]
    for _ in range(iter):
        res.append(rec(res[-1], **args))

    return res
```


```python
cvg = []
interval = .01
for i, p0 in enumerate(np.arange(1e-12, 1, interval)):
    cvg.append(compute_rec(rec=rec, init=p0, args={
        "p": p0}, iter=2000)[-1])
    if not i % 10:
        print(f"p0 = {p0:.2f}\tconvergence = {cvg[-1]:.2f}\t(1 / p0 - 1) = {1 / p0 - 1:.2f}")
```

    p0 = 0.00	convergence = 1.00	(1 / p0 - 1) = 999999999999.00
    p0 = 0.10	convergence = 1.00	(1 / p0 - 1) = 9.00
    p0 = 0.20	convergence = 1.00	(1 / p0 - 1) = 4.00
    p0 = 0.30	convergence = 1.00	(1 / p0 - 1) = 2.33
    p0 = 0.40	convergence = 1.00	(1 / p0 - 1) = 1.50
    p0 = 0.50	convergence = 1.00	(1 / p0 - 1) = 1.00
    p0 = 0.60	convergence = 0.67	(1 / p0 - 1) = 0.67
    p0 = 0.70	convergence = 0.43	(1 / p0 - 1) = 0.43
    p0 = 0.80	convergence = 0.25	(1 / p0 - 1) = 0.25
    p0 = 0.90	convergence = 0.11	(1 / p0 - 1) = 0.11



```python
plt.plot(np.arange(1e-12, 1, interval), cvg)
plt.xticks(np.arange(0, 1, .1))
plt.show()
```



![png](../assets/blogs/PCFG-Termination_files/PCFG-Termination_7_0.png)


