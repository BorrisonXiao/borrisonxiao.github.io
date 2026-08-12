---
layout: interview_question
topic: calculus
order: 2
title: "Implicit and Logarithmic Differentiation"
difficulty: core
date: 2026-08-12 13:00:00
tags: [derivatives, implicit differentiation, logarithmic differentiation, inverse functions]
summary: >-
  Two techniques for the cases where you cannot just apply the rules: curves
  given by an equation rather than a formula, and expressions with the variable
  in both the base and the exponent. Both are the chain rule, used sideways.
scripts:
  - /assets/js/interview/calculus.js
concepts:
  - name: Implicit differentiation
    url: https://en.wikipedia.org/wiki/Implicit_function
    note: >-
      Differentiate both sides of an equation with respect to x, treating y as
      an unknown function of x, then solve for y'. No need to solve for y first,
      which is the point - usually you cannot.
  - name: Chain rule
    url: https://en.wikipedia.org/wiki/Chain_rule
    note: >-
      d/dx of anything containing y picks up a factor of y'. Every implicit
      differentiation is this observation applied repeatedly.
  - name: Logarithmic differentiation
    url: https://en.wikipedia.org/wiki/Logarithmic_derivative
    note: >-
      Take logs first. Products become sums, quotients become differences, and
      exponents come down as coefficients - so a page of quotient rule becomes
      a line.
  - name: Derivatives of inverse functions
    url: https://en.wikipedia.org/wiki/Inverse_functions_and_differentiation
    note: >-
      The standard way to get arcsin, arccos, arctan and friends: write the
      inverse relation as an equation and differentiate it implicitly.
  - name: Implicit function theorem
    url: https://en.wikipedia.org/wiki/Implicit_function_theorem
    note: >-
      The statement that makes all of this legitimate - and that tells you
      exactly where it breaks down (where the denominator vanishes).
references:
  - text: >-
      Stewart, *Calculus* - the implicit differentiation and logarithmic
      differentiation sections of the differentiation-rules chapter.
  - text: >-
      Spivak, *Calculus* - for the inverse function derivative done carefully,
      including why the inverse is differentiable in the first place.
---

New to this, or just cold? Open the example first. Otherwise go straight to the problem.

<details class="q-example" id="example">
<summary class="q-example-summary"><span class="q-when-closed">Show a worked example first</span><span class="q-when-open">Hide the example</span></summary>
<div class="q-example-body" markdown="1">

### When you cannot write $y = f(x)$

Everything in the [previous question](/interview-prep/calculus/derivatives-basic-rules/) assumed the function was handed to you as a formula, $y = $ something in $x$. Plenty of curves are not:

$$
x^2 + y^2 = 25 .
$$

This is a circle. It is not the graph of any function — most vertical lines cross it twice — so there is no single $f$ to differentiate. It still has a perfectly good tangent line at every point, and we still want its slope.

The trick is to stop trying to solve for $y$. Instead, **assume $y$ is some function of $x$ near the point of interest, and differentiate the whole equation with respect to $x$.** The only thing to be careful about is that any expression containing $y$ needs the chain rule, because $y$ depends on $x$:

$$
\frac{d}{dx}\big[y\big] = y',
\qquad
\frac{d}{dx}\big[y^2\big] = 2y\,y',
\qquad
\frac{d}{dx}\big[\sin y\big] = \cos y \cdot y',
\qquad
\frac{d}{dx}\big[xy\big] = y + x\,y' .
$$

That last one needs the product rule as well — $x$ and $y$ are both functions of $x$.

### Worked example: the circle $x^2 + y^2 = 25$

Differentiate both sides with respect to $x$. The left side term by term, the right side is a constant so it dies:

$$
2x + 2y\,y' = 0 .
$$

Now solve for $y'$ — it appears linearly, which it always will:

$$
y' = -\frac{x}{y} .
$$

That is the answer, and notice it is in terms of *both* $x$ and $y$. That is normal and it is not a failure to simplify: the circle has two points above most $x$ values, and they have opposite slopes, so any correct formula has to know which one you mean.

**Check it against something you already know.** At the point $(3, 4)$ the formula gives $y' = -3/4$. The radius from the origin to $(3,4)$ has slope $4/3$, and a tangent to a circle is perpendicular to the radius. Perpendicular slopes are negative reciprocals: $-3/4$. ✓

**Where it breaks.** At $(5, 0)$ and $(-5, 0)$ the formula divides by zero. That is the method telling you something true — the tangent there is vertical, and no finite slope exists. A vanishing denominator in an implicit derivative always means a vertical tangent or a genuine singular point of the curve.

Drag the point around the curves below.

<div class="viz" data-viz="implicit-curve">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; tangents to a curve with no formula</p>
    <p class="viz-sub">Drag the point along the curve, or click the canvas and use the arrow keys. The readout shows the implicit derivative and the slope it predicts. Try dragging to the far left or right of the circle, where the tangent goes vertical and the formula reports it.</p>
  </div>
</div>

### The other technique: take logs first

A second situation the plain rules do not cover: the variable appears in *both* the base and the exponent, as in $y = x^x$. The power rule needs a constant exponent; the exponential rule needs a constant base. Neither applies.

Take the logarithm of both sides, then differentiate implicitly:

$$
\ln y = x\ln x
\quad\Longrightarrow\quad
\frac{y'}{y} = \ln x + 1
\quad\Longrightarrow\quad
y' = x^x(\ln x + 1) .
$$

The left side used $\frac{d}{dx}\ln y = \frac{y'}{y}$, which is the chain rule again. The quantity $y'/y$ is called the **logarithmic derivative**, and it is worth knowing in its own right — it is the relative rate of change, the thing that is 0.03 when something grows at 3% per unit time.

</div>
</details>

<div class="q-problem" markdown="1">
<div class="q-problem-label">Problem</div>

**Part 1 — implicit.** Find $\dfrac{dy}{dx}$ for each.

(a) $x^3 + y^3 = 6xy$, and give the tangent line at the point $(3,3)$.

(b) $\sin(xy) = x + y$.

**Part 2 — logarithmic.** Differentiate.

(c) $y = x^x$ for $x > 0$.

(d) $y = \dfrac{(x+1)^2(x-3)^4}{(x^2+1)^3}$.

**Part 3.** Derive $\dfrac{d}{dx}\arcsin x$ from scratch, using nothing but implicit differentiation and the fact that $\sin$ and $\arcsin$ undo each other.

</div>

Work them on paper first — that is what the page is for.

<details class="q-solution" id="solution">
<summary class="q-solution-summary"><span class="q-when-closed">Show the solution</span><span class="q-when-open">Hide the solution</span></summary>
<div class="q-solution-body" markdown="1">

## The answers

| | | $dy/dx$ |
|---|---|---|
| (a) | $x^3+y^3=6xy$ | $\dfrac{2y - x^2}{y^2 - 2x}$, tangent at $(3,3)$ is $y = -x + 6$ |
| (b) | $\sin(xy) = x+y$ | $\dfrac{1 - y\cos(xy)}{x\cos(xy) - 1}$ |
| (c) | $x^x$ | $x^x(\ln x + 1)$ |
| (d) | $\dfrac{(x+1)^2(x-3)^4}{(x^2+1)^3}$ | $y\left[\dfrac{2}{x+1} + \dfrac{4}{x-3} - \dfrac{6x}{x^2+1}\right]$ |
| 3 | $\arcsin x$ | $\dfrac{1}{\sqrt{1-x^2}}$ |

## Part 1 — implicit differentiation

**(a) $x^3 + y^3 = 6xy$.** This is the folium of Descartes. Differentiate both sides; the right-hand side needs the product rule because both factors depend on $x$:

$$
3x^2 + 3y^2 y' = 6y + 6x\,y' .
$$

Collect the $y'$ terms on one side and everything else on the other — always the next move, because $y'$ is linear:

$$
3y^2 y' - 6x\,y' = 6y - 3x^2
\quad\Longrightarrow\quad
y'\big(3y^2 - 6x\big) = 6y - 3x^2 ,
$$

$$
y' = \frac{6y - 3x^2}{3y^2 - 6x} = \frac{2y - x^2}{y^2 - 2x} .
$$

At $(3,3)$ — which is on the curve, since $27 + 27 = 54 = 6\cdot 9$:

$$
y' = \frac{6 - 9}{9 - 6} = -1 ,
$$

so the tangent is $y - 3 = -(x-3)$, i.e. $y = -x + 6$. (You can see this one coming: the curve is symmetric in $x$ and $y$, and $(3,3)$ is on the line of symmetry $y = x$, so the tangent has to be perpendicular to it.)

**(b) $\sin(xy) = x + y$.** The left side is a composition wrapping a product, so chain rule outside, product rule inside:

$$
\cos(xy)\cdot\big(y + x\,y'\big) = 1 + y' .
$$

Expand, then collect:

$$
y\cos(xy) + x\cos(xy)\,y' = 1 + y'
\quad\Longrightarrow\quad
y'\big(x\cos(xy) - 1\big) = 1 - y\cos(xy) ,
$$

$$
y' = \frac{1 - y\cos(xy)}{x\cos(xy) - 1} .
$$

Do not expand $\cos(xy)$ into anything; leave it. The answer depending on both $x$ and $y$ is expected.

<div class="q-callout q-callout-insight" markdown="1">
<div class="q-callout-title"><i class="fas fa-lightbulb"></i> The shortcut, once you trust it</div>

Write the equation as $F(x,y) = 0$. Then

$$
y' = -\frac{\partial F/\partial x}{\partial F/\partial y} ,
$$

which is the implicit function theorem. For (a), $F = x^3+y^3-6xy$ gives $F_x = 3x^2-6y$ and $F_y = 3y^2-6x$, so $y' = -(3x^2-6y)/(3y^2-6x) = (2y-x^2)/(y^2-2x)$ — same answer, no rearranging. The theorem also says exactly when this is valid: wherever $F_y \ne 0$. When $F_y = 0$ you get the vertical tangents and the crossing points.

</div>

## Part 2 — logarithmic differentiation

**(c) $y = x^x$.** Neither the power rule (exponent must be constant) nor the exponential rule (base must be constant) applies. Take logs:

$$
\ln y = x \ln x .
$$

Differentiate both sides. Left side by the chain rule, right side by the product rule:

$$
\frac{y'}{y} = \ln x + x\cdot\frac1x = \ln x + 1
\quad\Longrightarrow\quad
y' = x^x(\ln x + 1) .
$$

Sanity check: $y' = 0$ when $\ln x = -1$, i.e. $x = 1/e \approx 0.368$, and indeed $x^x$ has its minimum there, with value $(1/e)^{1/e} \approx 0.6922$.

The alternative is to write $x^x = e^{x\ln x}$ and use the chain rule directly — same work, same answer. Use whichever you find harder to get wrong.

**(d) $y = \dfrac{(x+1)^2(x-3)^4}{(x^2+1)^3}$.** You *could* do this with the quotient rule and two product rules. Do not. Logs turn the whole structure into a sum:

$$
\ln y = 2\ln(x+1) + 4\ln(x-3) - 3\ln(x^2+1) ,
$$

$$
\frac{y'}{y} = \frac{2}{x+1} + \frac{4}{x-3} - \frac{6x}{x^2+1} ,
$$

$$
y' = \frac{(x+1)^2(x-3)^4}{(x^2+1)^3}\left[\frac{2}{x+1} + \frac{4}{x-3} - \frac{6x}{x^2+1}\right] .
$$

This is the technique's real use: any product or quotient of powers becomes a sum of terms of the form $\dfrac{\text{exponent}\times(\text{inner})'}{\text{inner}}$, which you can write down by inspection. Leaving the answer in this factored form is not laziness — expanding it destroys the information.

(Strictly, $\ln y$ requires $y > 0$. Using $\ln\lvert y\rvert$ fixes it, and the derivative formula comes out the same because $\frac{d}{dx}\ln\lvert y\rvert = y'/y$ on either side of zero.)

## Part 3 — the derivative of arcsin

Let $y = \arcsin x$, so that by definition

$$
\sin y = x, \qquad y \in \left[-\tfrac{\pi}{2}, \tfrac{\pi}{2}\right] .
$$

Differentiate that equation implicitly with respect to $x$:

$$
\cos y \cdot y' = 1
\quad\Longrightarrow\quad
y' = \frac{1}{\cos y} .
$$

Now get rid of $y$. From $\sin y = x$ and $\cos^2 y = 1 - \sin^2 y$ we have $\cos y = \pm\sqrt{1-x^2}$, and on the range of $\arcsin$ the cosine is non-negative, so the sign is $+$:

$$
\boxed{\;\frac{d}{dx}\arcsin x = \frac{1}{\sqrt{1-x^2}}\;}
$$

Two things this shows beyond the formula itself. The blow-up at $x = \pm1$ is real: $\sin$ has horizontal tangents at $y = \pm\pi/2$, so its inverse has vertical ones. And the restricted range is not bookkeeping — it is what decides the sign.

The same three steps give every inverse derivative. This is also the general statement $\big(f^{-1}\big)'(x) = \dfrac{1}{f'\!\big(f^{-1}(x)\big)}$: the graph of the inverse is the graph reflected across $y = x$, and reflection turns a slope into its reciprocal.

## The slips worth naming

- Differentiating $y^2$ as $2y$. Every $y$ carries a $y'$; that is the entire technique.
- Forgetting the product rule on mixed terms like $xy$ or $x^2y^3$.
- Trying to solve for $y$ first. For (a) and (b) you cannot, and for anything you can, it is slower.
- Being unhappy that the answer contains $y$. It has to.
- Using the power rule on $x^x$.
- Taking logs and then forgetting that the left side gives $y'/y$, not $y'$ — so forgetting to multiply back by $y$ at the end.

</div>
</details>

## More practice

<details class="q-reveal">
<summary>Drill 1 &mdash; find $y'$ for $x^2 + xy + y^2 = 7$, and the slope at $(1,2)$</summary>
<div class="q-reveal-body" markdown="1">

The middle term needs the product rule:

$$
2x + \big(y + x\,y'\big) + 2y\,y' = 0
\quad\Longrightarrow\quad
y'(x + 2y) = -(2x + y)
\quad\Longrightarrow\quad
y' = -\frac{2x+y}{x+2y} .
$$

Check $(1,2)$ is on the curve: $1 + 2 + 4 = 7$. ✓ Then $y' = -\dfrac{2+2}{1+4} = -\dfrac45$.

The curve is a tilted ellipse, and the tangent goes vertical where $x + 2y = 0$.

</div>
</details>

<details class="q-reveal">
<summary>Drill 2 &mdash; differentiate $y = (\ln x)^x$</summary>
<div class="q-reveal-body" markdown="1">

Variable base *and* variable exponent again, so logs:

$$
\ln y = x\ln(\ln x) .
$$

Differentiate the right side by the product rule, and the $\ln(\ln x)$ by the chain rule:

$$
\frac{y'}{y} = \ln(\ln x) + x\cdot\frac{1}{\ln x}\cdot\frac1x = \ln(\ln x) + \frac{1}{\ln x} ,
$$

$$
y' = (\ln x)^x\left[\ln(\ln x) + \frac{1}{\ln x}\right] .
$$

Valid for $x > 1$, where $\ln x > 0$ and the outer logarithm exists.

</div>
</details>

<details class="q-reveal">
<summary>Drill 3 &mdash; derive $\dfrac{d}{dx}\arctan x$ the same way</summary>
<div class="q-reveal-body" markdown="1">

Let $y = \arctan x$, so $\tan y = x$ with $y \in (-\pi/2, \pi/2)$. Differentiate:

$$
\sec^2 y \cdot y' = 1 \quad\Longrightarrow\quad y' = \cos^2 y .
$$

Convert back using $\sec^2 y = 1 + \tan^2 y = 1 + x^2$:

$$
\frac{d}{dx}\arctan x = \frac{1}{1+x^2} .
$$

No square roots and no domain restriction — the derivative is defined and positive for every real $x$, which matches $\arctan$ being increasing everywhere and flattening out toward $\pm\pi/2$. This is also the reason $\int\frac{dx}{1+x^2}$ is an arctangent, which is where the [partial fractions question](/interview-prep/calculus/integrals-partial-fractions-trig-sub/) ends up.

</div>
</details>
