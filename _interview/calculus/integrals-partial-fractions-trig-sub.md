---
layout: interview_question
topic: calculus
order: 4
title: "Partial Fractions, Completing the Square, and Trig Substitution"
difficulty: core
date: 2026-08-12 15:00:00
tags: [integrals, partial fractions, trigonometric substitution, rational functions]
summary: >-
  Three moves that turn integrals you cannot do into ones you can: split a
  rational function into simple poles, complete the square to reach an
  arctangent, and trade a square root for a trig identity.
scripts:
  - /assets/js/interview/calculus.js
concepts:
  - name: Partial fraction decomposition
    url: https://en.wikipedia.org/wiki/Partial_fraction_decomposition
    note: >-
      Every proper rational function is a sum of terms with linear or
      irreducible-quadratic denominators, each of which integrates to a log or
      an arctangent. This is an algebraic identity, not an approximation.
  - name: Completing the square
    url: https://en.wikipedia.org/wiki/Completing_the_square
    note: >-
      Turns any quadratic into (x + p)² + q, which reduces an unfamiliar
      denominator to the arctangent or logarithm form you already know.
  - name: Trigonometric substitution
    url: https://en.wikipedia.org/wiki/Trigonometric_substitution
    note: >-
      Square roots of a² - x², a² + x² and x² - a² all disappear under the
      right trig substitution, because of the Pythagorean identities.
  - name: Integral of 1/(1+x²)
    url: https://en.wikipedia.org/wiki/Inverse_trigonometric_functions
    note: >-
      arctan x. Along with the logarithm, this is one of the two destinations
      every rational function eventually reaches.
  - name: Irreducible polynomial
    url: https://en.wikipedia.org/wiki/Irreducible_polynomial
    note: >-
      Over the reals every polynomial factors into linear and quadratic pieces,
      which is exactly why the decomposition below is always possible.
references:
  - text: >-
      Stewart, *Calculus* - the partial fractions and trigonometric substitution
      sections, with the full case analysis for repeated and quadratic factors.
  - text: >-
      Gradshteyn & Ryzhik, *Table of Integrals, Series, and Products* - for when
      the point is to look it up rather than derive it.
---

New to this, or just cold? Open the example first. Otherwise go straight to the problem.

<details class="q-example" id="example">
<summary class="q-example-summary"><span class="q-when-closed">Show a worked example first</span><span class="q-when-open">Hide the example</span></summary>
<div class="q-example-body" markdown="1">

### The idea: break a hard fraction into easy ones

You know how to integrate $\dfrac{1}{x-a}$ — it is $\ln\lvert x-a\rvert$. You do not immediately know how to integrate

$$
\frac{1}{(x-1)(x+2)} .
$$

But the two are closer than they look, because that fraction is a *sum* of two of the easy kind. Finding which sum is **partial fraction decomposition**, and it is pure algebra — no calculus in it at all.

Write down the form with unknown constants, one term per factor:

$$
\frac{1}{(x-1)(x+2)} = \frac{A}{x-1} + \frac{B}{x+2} .
$$

Multiply through by $(x-1)(x+2)$ to clear denominators:

$$
1 = A(x+2) + B(x-1) .
$$

This has to hold for *every* $x$, so choose values of $x$ that kill terms — the **cover-up method**:

- $x = 1$: $\;1 = A(3) \Rightarrow A = \tfrac13$
- $x = -2$: $\;1 = B(-3) \Rightarrow B = -\tfrac13$

So

$$
\frac{1}{(x-1)(x+2)} = \frac{1/3}{x-1} - \frac{1/3}{x+2} ,
$$

and now the integral is two logarithms:

$$
\int\frac{dx}{(x-1)(x+2)}
= \frac13\ln\lvert x-1\rvert - \frac13\ln\lvert x+2\rvert + C
= \frac13\ln\left\lvert\frac{x-1}{x+2}\right\rvert + C .
$$

<div class="viz" data-viz="partial-fractions">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; a rational function is a sum of simple poles</p>
    <p class="viz-sub">The blue curve is the original function; the coloured curves are the pieces. Tick "overlay their sum" and the dashed black curve lands exactly on the blue one — the decomposition is an identity, not an approximation.</p>
  </div>
</div>

Each piece blows up at one root of the denominator and is well behaved everywhere else, which is exactly what makes it integrable on sight. Everything below is this idea plus the bookkeeping for the awkward cases.

### The rules of the form

Before decomposing, two checks:

1. **Is the fraction proper** — numerator degree strictly less than denominator degree? If not, do polynomial long division first and decompose the remainder.
2. **Factor the denominator completely.** Over the reals, every polynomial factors into linear and irreducible quadratic pieces.

Then each factor contributes terms:

| factor in the denominator | contributes |
|---|---|
| $(x-a)$ | $\dfrac{A}{x-a}$ |
| $(x-a)^k$ | $\dfrac{A_1}{x-a} + \dfrac{A_2}{(x-a)^2} + \cdots + \dfrac{A_k}{(x-a)^k}$ |
| $(x^2+bx+c)$ irreducible | $\dfrac{Ax+B}{x^2+bx+c}$ |

A repeated factor needs *every* power up to $k$, and an irreducible quadratic needs a linear numerator. Those are the two places people lose marks.

</div>
</details>

<div class="q-problem" markdown="1">
<div class="q-problem-label">Problem</div>

Evaluate each of the following.

$$
\text{(a)}\;\; \int \frac{3x+5}{(x-1)(x+3)}\,dx
\qquad
\text{(b)}\;\; \int \frac{x+1}{x^2(x-2)}\,dx
$$

$$
\text{(c)}\;\; \int \frac{dx}{x^2+2x+5}
\qquad
\text{(d)}\;\; \int \sqrt{9-x^2}\;dx
$$

For (b), note that the denominator has a repeated factor. For (c), the denominator does not factor over the reals — do not force it to.

</div>

Work them on paper first — that is what the page is for.

<details class="q-solution" id="solution">
<summary class="q-solution-summary"><span class="q-when-closed">Show the solution</span><span class="q-when-open">Hide the solution</span></summary>
<div class="q-solution-body" markdown="1">

## The answers

| | integral | value | move |
|---|---|---|---|
| (a) | $\displaystyle\int\frac{3x+5}{(x-1)(x+3)}dx$ | $2\ln\lvert x-1\rvert + \ln\lvert x+3\rvert + C$ | partial fractions |
| (b) | $\displaystyle\int\frac{x+1}{x^2(x-2)}dx$ | $\dfrac{3}{4}\ln\left\lvert\dfrac{x-2}{x}\right\rvert + \dfrac{1}{2x} + C$ | repeated factor |
| (c) | $\displaystyle\int\frac{dx}{x^2+2x+5}$ | $\tfrac12\arctan\dfrac{x+1}{2} + C$ | complete the square |
| (d) | $\displaystyle\int\sqrt{9-x^2}\,dx$ | $\tfrac92\arcsin\dfrac{x}{3} + \dfrac{x\sqrt{9-x^2}}{2} + C$ | $x = 3\sin\theta$ |

## (a) Two distinct linear factors

Proper fraction, denominator already factored, so write the form and use cover-up:

$$
\frac{3x+5}{(x-1)(x+3)} = \frac{A}{x-1} + \frac{B}{x+3}
\quad\Longrightarrow\quad
3x+5 = A(x+3) + B(x-1) .
$$

- $x = 1$: $\;8 = 4A \Rightarrow A = 2$
- $x = -3$: $\;-4 = -4B \Rightarrow B = 1$

$$
\int\frac{3x+5}{(x-1)(x+3)}dx = \int\frac{2\,dx}{x-1} + \int\frac{dx}{x+3}
= 2\ln\lvert x-1\rvert + \ln\lvert x+3\rvert + C .
$$

The cover-up shortcut is worth doing directly: to get the coefficient over $(x - r)$, delete that factor from the denominator and evaluate what remains at $x = r$. For $A$: $\frac{3(1)+5}{1+3} = 2$. One line, no system to solve.

## (b) A repeated factor

$x^2$ is $(x-0)^2$, so it contributes **two** terms:

$$
\frac{x+1}{x^2(x-2)} = \frac{A}{x} + \frac{B}{x^2} + \frac{C}{x-2}
\quad\Longrightarrow\quad
x+1 = Ax(x-2) + B(x-2) + Cx^2 .
$$

Cover-up gets the two roots directly:

- $x = 0$: $\;1 = -2B \Rightarrow B = -\tfrac12$
- $x = 2$: $\;3 = 4C \Rightarrow C = \tfrac34$

$A$ is not reachable by cover-up, because no choice of $x$ isolates it. Compare coefficients instead — the $x^2$ terms give $0 = A + C$, so $A = -\tfrac34$. (Check with the $x$ coefficient: $1 = -2A + B = \tfrac32 - \tfrac12 = 1$ ✓.)

Now integrate, and notice the middle term is **not** a logarithm — it is a power:

$$
\int\left(\frac{-3/4}{x} + \frac{-1/2}{x^2} + \frac{3/4}{x-2}\right)dx
= -\frac34\ln\lvert x\rvert + \frac{1}{2x} + \frac34\ln\lvert x-2\rvert + C ,
$$

since $\int -\tfrac12 x^{-2}dx = -\tfrac12\cdot(-x^{-1}) = \tfrac{1}{2x}$. Tidying:

$$
= \frac34\ln\left\lvert\frac{x-2}{x}\right\rvert + \frac{1}{2x} + C .
$$

<div class="q-callout q-callout-trap" markdown="1">
<div class="q-callout-title"><i class="fas fa-triangle-exclamation"></i> Two traps in one problem</div>

Writing only $\frac{B}{x^2}$ and omitting $\frac{A}{x}$ gives an unsolvable system — a sign that the form was wrong, not the arithmetic. And $\int\frac{dx}{x^2} = -\frac1x$, not a logarithm; only the *first* power of a factor produces a log.

</div>

## (c) Completing the square

Check the discriminant before reaching for partial fractions: $x^2+2x+5$ has $b^2-4ac = 4-20 < 0$, so it does not factor over the reals. Partial fractions has nothing to do here. Complete the square instead:

$$
x^2 + 2x + 5 = (x^2+2x+1) + 4 = (x+1)^2 + 4 .
$$

Now it matches the arctangent form $\int\frac{du}{u^2+a^2} = \frac1a\arctan\frac{u}{a} + C$, with $u = x+1$ and $a = 2$:

$$
\int\frac{dx}{x^2+2x+5} = \int\frac{du}{u^2+4} = \frac12\arctan\frac{x+1}{2} + C .
$$

The move generalizes: any $\int\frac{dx}{\text{quadratic}}$ is an arctangent when the quadratic is irreducible, and a pair of logs when it is not. The discriminant tells you which before you start.

## (d) Trig substitution

$\sqrt{9-x^2}$ is not touched by anything so far. The way in is the Pythagorean identity: if $x = 3\sin\theta$ then $9 - x^2 = 9(1-\sin^2\theta) = 9\cos^2\theta$, and the square root evaporates.

The three standard cases, all for the same reason:

| you see | substitute | becomes |
|---|---|---|
| $\sqrt{a^2-x^2}$ | $x = a\sin\theta$ | $a\cos\theta$ |
| $\sqrt{a^2+x^2}$ | $x = a\tan\theta$ | $a\sec\theta$ |
| $\sqrt{x^2-a^2}$ | $x = a\sec\theta$ | $a\tan\theta$ |

Here $a = 3$, $x = 3\sin\theta$, $dx = 3\cos\theta\,d\theta$, $\sqrt{9-x^2} = 3\cos\theta$:

$$
\int\sqrt{9-x^2}\,dx = \int 3\cos\theta\cdot3\cos\theta\,d\theta = 9\int\cos^2\theta\,d\theta .
$$

Use the half-angle identity $\cos^2\theta = \frac{1+\cos 2\theta}{2}$:

$$
9\int\frac{1+\cos2\theta}{2}d\theta = \frac92\theta + \frac94\sin2\theta + C
= \frac92\theta + \frac92\sin\theta\cos\theta + C ,
$$

using $\sin 2\theta = 2\sin\theta\cos\theta$.

**Now convert back**, which is the step people skip. Draw the right triangle for $\sin\theta = x/3$: opposite $x$, hypotenuse $3$, so the adjacent side is $\sqrt{9-x^2}$. Read off everything you need:

$$
\theta = \arcsin\frac{x}{3},
\qquad
\sin\theta = \frac{x}{3},
\qquad
\cos\theta = \frac{\sqrt{9-x^2}}{3} .
$$

$$
\int\sqrt{9-x^2}\,dx = \frac92\arcsin\frac{x}{3} + \frac92\cdot\frac{x}{3}\cdot\frac{\sqrt{9-x^2}}{3} + C
= \frac92\arcsin\frac{x}{3} + \frac{x\sqrt{9-x^2}}{2} + C .
$$

Sanity check with geometry: over $[-3,3]$ this should give the area of a half-disc of radius 3. At $x=3$, $\arcsin 1 = \pi/2$ and the second term vanishes, giving $\frac92\cdot\frac\pi2 = \frac{9\pi}{4}$; at $x=-3$ it is $-\frac{9\pi}{4}$. The difference is $\frac{9\pi}{2}$, and half of $\pi r^2 = 9\pi$ is indeed $\frac{9\pi}{2}$ ✓.

## The decision procedure

Faced with an integral of a rational function or a square root:

1. **Rational function?** Check proper — divide if not. Factor the denominator. Any repeated or quadratic factors get their extended forms. Decompose, integrate term by term. Logs and arctangents come out.
2. **Quadratic that will not factor?** Complete the square, then arctangent (or a logarithm if a $\sqrt{\;}$ is involved).
3. **Square root of a quadratic?** Complete the square first if it is not already in $a^2 \pm x^2$ form, then use the table above and finish with the reference triangle.

## The slips worth naming

- Decomposing an improper fraction. Divide first.
- Forgetting the lower powers of a repeated factor.
- Putting a constant over an irreducible quadratic instead of $Ax+B$.
- Reaching for partial fractions on an irreducible denominator. Check the discriminant.
- Integrating $\frac{1}{(x-a)^2}$ as a logarithm.
- Finishing a trig substitution in $\theta$ and never converting back to $x$. The answer must be a function of $x$.

</div>
</details>

## More practice

<details class="q-reveal">
<summary>Drill 1 &mdash; $\displaystyle\int \frac{dx}{x^2-9}$</summary>
<div class="q-reveal-body" markdown="1">

Factors as $(x-3)(x+3)$, so cover-up gives $A = \frac{1}{3+3} = \frac16$ and $B = \frac{1}{-3-3} = -\frac16$:

$$
\int\frac{dx}{x^2-9} = \frac16\ln\lvert x-3\rvert - \frac16\ln\lvert x+3\rvert + C
= \frac16\ln\left\lvert\frac{x-3}{x+3}\right\rvert + C .
$$

Compare with (c) in the problem: same shape of integrand, completely different answer, because $x^2-9$ factors and $x^2+2x+5$ does not. Factors $\Rightarrow$ logarithms; irreducible $\Rightarrow$ arctangent.

</div>
</details>

<details class="q-reveal">
<summary>Drill 2 &mdash; $\displaystyle\int \frac{x^2+1}{x^2-1}\,dx$</summary>
<div class="q-reveal-body" markdown="1">

The fraction is **improper** — both degrees are 2 — so divide before anything else:

$$
\frac{x^2+1}{x^2-1} = 1 + \frac{2}{x^2-1} .
$$

Now decompose the remainder: $\dfrac{2}{(x-1)(x+1)} = \dfrac{1}{x-1} - \dfrac{1}{x+1}$.

$$
\int\frac{x^2+1}{x^2-1}dx = x + \ln\lvert x-1\rvert - \ln\lvert x+1\rvert + C
= x + \ln\left\lvert\frac{x-1}{x+1}\right\rvert + C .
$$

Skipping the division and writing $\frac{A}{x-1}+\frac{B}{x+1}$ produces a system with no solution — the giveaway that the fraction was improper.

</div>
</details>

<details class="q-reveal">
<summary>Drill 3 &mdash; $\displaystyle\int \frac{dx}{\sqrt{x^2+4}}$</summary>
<div class="q-reveal-body" markdown="1">

A $\sqrt{a^2+x^2}$, so $x = 2\tan\theta$, $dx = 2\sec^2\theta\,d\theta$, $\sqrt{x^2+4} = 2\sec\theta$:

$$
\int\frac{2\sec^2\theta\,d\theta}{2\sec\theta} = \int\sec\theta\,d\theta = \ln\lvert\sec\theta + \tan\theta\rvert + C .
$$

Reference triangle: opposite $x$, adjacent $2$, hypotenuse $\sqrt{x^2+4}$. So $\tan\theta = \frac{x}{2}$ and $\sec\theta = \frac{\sqrt{x^2+4}}{2}$:

$$
\int\frac{dx}{\sqrt{x^2+4}} = \ln\left\lvert\frac{\sqrt{x^2+4} + x}{2}\right\rvert + C
= \ln\big(x + \sqrt{x^2+4}\big) + C' ,
$$

absorbing the $\ln 2$ into the constant. If that answer looks familiar it should — it is $\operatorname{arcsinh}(x/2)$ up to a constant, and it is exactly the function whose derivative came out so cleanly in [Drill 3 of the first calculus question](/interview-prep/calculus/derivatives-basic-rules/).

</div>
</details>
