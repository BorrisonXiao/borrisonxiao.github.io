---
layout: interview_question
topic: calculus
order: 1
title: "Derivatives: the Product, Quotient, and Chain Rules"
difficulty: warm-up
date: 2026-08-12 12:00:00
tags: [derivatives, chain rule, product rule, limits]
summary: >-
  Differentiate a handful of functions that each need a different rule. The
  rules themselves are two lines each; the skill being tested is reading an
  expression and knowing which one applies, and in what order.
scripts:
  - /assets/js/interview/calculus.js
concepts:
  - name: Derivative (limit definition)
    url: https://en.wikipedia.org/wiki/Derivative
    note: >-
      The slope of the tangent, defined as the limit of secant slopes. Every
      rule below is a theorem about this limit.
  - name: Product rule
    url: https://en.wikipedia.org/wiki/Product_rule
    note: (fg)' = f'g + fg'. Not f'g' - that is the single most common slip.
  - name: Quotient rule
    url: https://en.wikipedia.org/wiki/Quotient_rule
    note: >-
      (f/g)' = (f'g - fg')/g². Order matters in the numerator, and it is not
      symmetric the way the product rule is.
  - name: Chain rule
    url: https://en.wikipedia.org/wiki/Chain_rule
    note: >-
      The derivative of a composition is the product of the rates. This is the
      one that generalizes - backpropagation is the chain rule on a graph.
  - name: Table of standard derivatives
    url: https://en.wikipedia.org/wiki/Differentiation_rules
    note: >-
      Powers, exponentials, logs, trig. Worth having memorized cold; there is
      no insight to be had from re-deriving them under time pressure.
references:
  - text: >-
      Stewart, *Calculus* - Ch. 2 for the definition and Ch. 3 for the rules,
      with more drill problems than anyone needs.
  - text: >-
      Spivak, *Calculus* - Ch. 9-10, if you want the rules proved properly
      rather than stated.
  - text: >-
      Thomas & Finney, *Calculus and Analytic Geometry* - the standard tables
      are in the endpapers.
---

New to this, or just cold? Open the example first. Otherwise go straight to the problem.

<details class="q-example" id="example">
<summary class="q-example-summary"><span class="q-when-closed">Show a worked example first</span><span class="q-when-open">Hide the example</span></summary>
<div class="q-example-body" markdown="1">

### What a derivative is

Pick a point $x$ and a nearby point $x + h$. The line through the two points on the graph — the **secant** — has slope

$$
\frac{f(x+h) - f(x)}{h} ,
$$

which is the average rate of change of $f$ across that gap. Now shrink the gap. If the slope settles on a single number as $h \to 0$, that number is the **derivative**:

$$
f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h} .
$$

Three readings of the same object, all worth having:

- **Geometric:** the slope of the tangent line to the graph at $x$.
- **Rate of change:** how fast $f$ responds to a nudge in $x$. If $f$ is position, $f'$ is velocity.
- **Local linear approximation:** near $x$, $f(x + \delta) \approx f(x) + f'(x)\,\delta$. This is what a derivative is *for*, and it is why gradients drive optimization.

Notations for the same thing: $f'(x)$, $\dfrac{df}{dx}$, $\dfrac{d}{dx}f(x)$.

### Worked example: $f(x) = x^2$

Straight from the definition. Form the difference quotient and simplify *before* letting $h$ go to zero — that is the whole trick, because the quotient is $0/0$ until you cancel the $h$:

$$
\frac{f(x+h) - f(x)}{h}
= \frac{(x+h)^2 - x^2}{h}
= \frac{x^2 + 2xh + h^2 - x^2}{h}
= \frac{2xh + h^2}{h}
= 2x + h .
$$

Now $h \to 0$ is harmless, and $f'(x) = 2x$. At $x = 3$ the tangent has slope $6$.

That is a special case of the **power rule**, which along with a handful of others you should know without thinking:

$$
\frac{d}{dx}x^n = nx^{n-1},
\qquad
\frac{d}{dx}e^x = e^x,
\qquad
\frac{d}{dx}\ln x = \frac1x,
$$

$$
\frac{d}{dx}\sin x = \cos x,
\qquad
\frac{d}{dx}\cos x = -\sin x,
\qquad
\frac{d}{dx}\tan x = \sec^2 x .
$$

Differentiation is also **linear**: $(af + bg)' = af' + bg'$ for constants $a, b$. Sums and constant multiples never cause trouble. Products, quotients and compositions do, and that is what the problem is about.

Drag the point along the curve below, then pull $h$ down toward zero.

<div class="viz" data-viz="derivative-explorer">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; secant into tangent</p>
    <p class="viz-sub">The green line is the secant through x and x + h; the orange line is the tangent. Shrink h and watch the difference quotient close in on the derivative. The last readout is the gap between them.</p>
  </div>
</div>

<div class="q-callout q-callout-insight" markdown="1">
<div class="q-callout-title"><i class="fas fa-lightbulb"></i> Worth being clear about</div>

Nobody differentiates from the definition in practice. But every rule below is a theorem *about* that limit, and "compute this derivative from the definition" is a standard way to check you know what the object is rather than just the table.

</div>

</div>
</details>

<div class="q-problem" markdown="1">
<div class="q-problem-label">Problem</div>

**Part 1.** Using the limit definition — not the power rule — differentiate $f(x) = \dfrac{1}{x}$.

**Part 2.** Differentiate each of the following. Simplify where the answer factors.

$$
\text{(a)}\;\; x^3 e^x
\qquad
\text{(b)}\;\; \frac{2x+1}{x^2+3}
\qquad
\text{(c)}\;\; \sin(3x^2 + 1)
$$

$$
\text{(d)}\;\; \ln(\cos x)
\qquad
\text{(e)}\;\; (x^2+1)^5 (2x-3)^3
$$

**Part 3.** For $f(x) = e^{\sin(x^2)}$, how many times does the chain rule apply, and what is $f'(x)$?

</div>

Work them on paper first — that is what the page is for.

<details class="q-solution" id="solution">
<summary class="q-solution-summary"><span class="q-when-closed">Show the solution</span><span class="q-when-open">Hide the solution</span></summary>
<div class="q-solution-body" markdown="1">

## The answers

| | function | derivative | rule |
|---|---|---|---|
| 1 | $1/x$ | $-1/x^2$ | definition |
| (a) | $x^3e^x$ | $x^2e^x(x+3)$ | product |
| (b) | $\dfrac{2x+1}{x^2+3}$ | $\dfrac{-2x^2-2x+6}{(x^2+3)^2}$ | quotient |
| (c) | $\sin(3x^2+1)$ | $6x\cos(3x^2+1)$ | chain |
| (d) | $\ln(\cos x)$ | $-\tan x$ | chain |
| (e) | $(x^2+1)^5(2x-3)^3$ | $2(x^2+1)^4(2x-3)^2(13x^2-15x+3)$ | product + chain |
| 3 | $e^{\sin(x^2)}$ | $2x\cos(x^2)\,e^{\sin(x^2)}$ | chain, twice |

## The three rules

**Product.** $(fg)' = f'g + fg'$.

The picture: $fg$ is the area of a rectangle with sides $f$ and $g$. Nudge $x$ and both sides grow a little; the extra area is a strip of width $\Delta f$ down one side plus a strip of height $\Delta g$ along the other, plus a tiny corner $\Delta f \Delta g$ that is second order and vanishes in the limit. Two strips, two terms.

**Quotient.** $\left(\dfrac{f}{g}\right)' = \dfrac{f'g - fg'}{g^2}$.

You do not have to memorize this separately: write $f/g = f \cdot g^{-1}$ and use the product and chain rules. It is worth memorizing anyway, but note the two ways it differs from the product rule — the numerator is a *difference*, so the order matters, and there is a $g^2$ underneath.

**Chain.** $\big(f(g(x))\big)' = f'(g(x)) \cdot g'(x)$.

In Leibniz notation it is the statement that rates multiply:

$$
\frac{dy}{dx} = \frac{dy}{du}\cdot\frac{du}{dx} .
$$

If $y$ changes 3 times as fast as $u$, and $u$ changes 5 times as fast as $x$, then $y$ changes 15 times as fast as $x$. Note carefully that $f'$ is evaluated **at $g(x)$**, not at $x$ — that is the error that survives longest.

<div class="q-callout q-callout-insight" markdown="1">
<div class="q-callout-title"><i class="fas fa-lightbulb"></i> How to pick a rule</div>

Ask what the **last** operation is — the outermost thing you would do if you were evaluating the expression at a number. If the last step is a multiplication, it is the product rule. A division, the quotient rule. Feeding one function into another, the chain rule. Work outside in, and each step reduces the problem to a smaller one.

</div>

## Working through them

**Part 1, from the definition.** The algebra is combining fractions and then cancelling the $h$:

$$
\frac{\frac{1}{x+h} - \frac{1}{x}}{h}
= \frac{\frac{x - (x+h)}{x(x+h)}}{h}
= \frac{-h}{h\,x(x+h)}
= \frac{-1}{x(x+h)}
\;\xrightarrow[h\to 0]{}\; -\frac{1}{x^2} .
$$

Same pattern as $x^2$ in the example: get the $h$ out of the denominator by cancellation, *then* take the limit.

**(a) $x^3e^x$.** A product of two things you know:

$$
(x^3 e^x)' = 3x^2 e^x + x^3 e^x = x^2 e^x(3 + x) .
$$

Factor at the end. Interviewers notice, and it makes the next step (say, finding critical points — here $x = 0$ and $x = -3$) free.

**(b) $\dfrac{2x+1}{x^2+3}$.** Quotient rule with $f = 2x+1$, $g = x^2+3$:

$$
\frac{2(x^2+3) - (2x+1)(2x)}{(x^2+3)^2}
= \frac{2x^2 + 6 - 4x^2 - 2x}{(x^2+3)^2}
= \frac{-2x^2 - 2x + 6}{(x^2+3)^2}
= \frac{-2(x^2 + x - 3)}{(x^2+3)^2} .
$$

**(c) $\sin(3x^2+1)$.** Outer function $\sin$, inner function $3x^2+1$:

$$
\cos(3x^2+1)\cdot 6x = 6x\cos(3x^2+1) .
$$

The $\cos$ keeps its original argument. Writing $\cos(6x)$ here is the classic mistake.

**(d) $\ln(\cos x)$.** Outer $\ln$, inner $\cos$:

$$
\frac{1}{\cos x}\cdot(-\sin x) = -\frac{\sin x}{\cos x} = -\tan x .
$$

A messy-looking derivative collapsing to something clean is usually a sign you did it right.

**(e) $(x^2+1)^5(2x-3)^3$.** A product whose factors each need the chain rule:

$$
5(x^2+1)^4(2x)\,(2x-3)^3 \;+\; (x^2+1)^5\cdot 3(2x-3)^2 \cdot 2 .
$$

Now pull out everything common — $2$, $(x^2+1)^4$, $(2x-3)^2$:

$$
2(x^2+1)^4(2x-3)^2\Big[5x(2x-3) + 3(x^2+1)\Big]
= 2(x^2+1)^4(2x-3)^2\big(13x^2 - 15x + 3\big) .
$$

The unfactored form is not wrong, but it is unusable. Anything you would do next — sign analysis, root finding — needs the factored form.

**Part 3, $e^{\sin(x^2)}$.** Three layers, so the chain rule applies twice. Peel from the outside:

$$
\frac{d}{dx}e^{\sin(x^2)}
= e^{\sin(x^2)}\cdot\frac{d}{dx}\sin(x^2)
= e^{\sin(x^2)}\cdot\cos(x^2)\cdot 2x .
$$

The pattern generalizes: for $f_1(f_2(\cdots f_n(x)))$ the derivative is the product of each layer's derivative evaluated at what that layer receives. Composed rates multiply. Backpropagation is exactly this, bookkept efficiently over a graph instead of a chain.

## The slips worth naming

- $(fg)' = f'g'$. It is not, and the rectangle picture says why: two strips, not one.
- Forgetting the inner derivative entirely — writing $\cos(3x^2+1)$ for (c) instead of $6x\cos(3x^2+1)$.
- Evaluating the outer derivative at the wrong place: $\cos(6x)$ rather than $\cos(3x^2+1)$.
- Flipping the quotient-rule numerator to $fg' - f'g$. Sanity check on a case you know: $(1/x)' = (0\cdot x - 1\cdot 1)/x^2 = -1/x^2$. Correct sign.
- Stopping before factoring.

</div>
</details>

## More practice

<details class="q-reveal">
<summary>Drill 1 &mdash; show that $\dfrac{d}{dx}\tan x = \sec^2 x$</summary>
<div class="q-reveal-body" markdown="1">

Write $\tan x = \dfrac{\sin x}{\cos x}$ and use the quotient rule:

$$
\frac{\cos x \cdot \cos x - \sin x \cdot(-\sin x)}{\cos^2 x}
= \frac{\cos^2 x + \sin^2 x}{\cos^2 x}
= \frac{1}{\cos^2 x} = \sec^2 x .
$$

The Pythagorean identity doing the collapsing is the whole point of the exercise.

</div>
</details>

<details class="q-reveal">
<summary>Drill 2 &mdash; differentiate $\dfrac{x}{\sqrt{x^2+1}}$</summary>
<div class="q-reveal-body" markdown="1">

Rewrite as a product to avoid the quotient rule: $x(x^2+1)^{-1/2}$. Then

$$
(x^2+1)^{-1/2} + x\cdot\left(-\tfrac12\right)(x^2+1)^{-3/2}(2x)
= (x^2+1)^{-1/2} - x^2(x^2+1)^{-3/2} .
$$

Pull out the smaller power:

$$
= (x^2+1)^{-3/2}\big[(x^2+1) - x^2\big] = \frac{1}{(x^2+1)^{3/2}} .
$$

Always positive, which matches the graph — the function increases everywhere, from $-1$ up to $1$.

</div>
</details>

<details class="q-reveal">
<summary>Drill 3 &mdash; differentiate $\ln\!\big(x + \sqrt{x^2+1}\big)$</summary>
<div class="q-reveal-body" markdown="1">

Outer $\ln$, inner $x + \sqrt{x^2+1}$ which itself needs the chain rule:

$$
\frac{1}{x+\sqrt{x^2+1}}\left(1 + \frac{x}{\sqrt{x^2+1}}\right)
= \frac{1}{x+\sqrt{x^2+1}}\cdot\frac{\sqrt{x^2+1} + x}{\sqrt{x^2+1}}
= \frac{1}{\sqrt{x^2+1}} .
$$

Everything cancels. That is not a coincidence: the function is $\operatorname{arcsinh} x$, and this derivative is why $\int \frac{dx}{\sqrt{x^2+1}}$ has the answer it does — which shows up again in the [trig substitution question](/interview-prep/calculus/integrals-partial-fractions-trig-sub/).

</div>
</details>
