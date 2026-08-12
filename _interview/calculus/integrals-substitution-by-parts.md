---
layout: interview_question
topic: calculus
order: 3
title: "Integration by Substitution and by Parts"
difficulty: core
date: 2026-08-12 14:00:00
tags: [integrals, substitution, integration by parts, fundamental theorem]
summary: >-
  The two workhorses. Substitution is the chain rule run backwards, integration
  by parts is the product rule run backwards, and between them they handle most
  integrals you will be asked to do by hand.
scripts:
  - /assets/js/interview/calculus.js
concepts:
  - name: Fundamental theorem of calculus
    url: https://en.wikipedia.org/wiki/Fundamental_theorem_of_calculus
    note: >-
      Ties the two meanings of "integral" together: the limit of Riemann sums
      equals the change in any antiderivative. It is why integration is a
      solvable problem at all rather than a numerical one.
  - name: Integration by substitution
    url: https://en.wikipedia.org/wiki/Integration_by_substitution
    note: >-
      The chain rule, read right to left. Look for an inner function whose
      derivative is already sitting in the integrand.
  - name: Integration by parts
    url: https://en.wikipedia.org/wiki/Integration_by_parts
    note: >-
      The product rule, read right to left. Trades one integral for another,
      so the whole skill is choosing which factor to differentiate.
  - name: Riemann sum
    url: https://en.wikipedia.org/wiki/Riemann_sum
    note: >-
      What a definite integral actually is before the fundamental theorem
      rescues you from computing it.
  - name: Table of integrals
    url: https://en.wikipedia.org/wiki/Lists_of_integrals
    note: >-
      Worth knowing the short list cold; everything else is a manipulation into
      something on it.
references:
  - text: >-
      Stewart, *Calculus* - the techniques-of-integration chapter, which is
      essentially a catalogue of the moves in this question and the next.
  - text: >-
      Apostol, *Calculus* Vol. I - integration developed before differentiation,
      which makes the fundamental theorem land as a genuine theorem.
---

New to this, or just cold? Open the example first. Otherwise go straight to the problem.

<details class="q-example" id="example">
<summary class="q-example-summary"><span class="q-when-closed">Show a worked example first</span><span class="q-when-open">Hide the example</span></summary>
<div class="q-example-body" markdown="1">

### What an integral is

Two different objects share the name, and the fundamental theorem is the bridge.

**The indefinite integral** is an antiderivative. $\int f(x)\,dx = F(x) + C$ means $F' = f$. The $+C$ is there because any two antiderivatives of the same function differ by a constant, and it is not decoration — dropping it costs marks and, in differential equations, solutions.

**The definite integral** $\int_a^b f(x)\,dx$ is the signed area between the graph and the $x$-axis, defined as a limit of Riemann sums: chop $[a,b]$ into $n$ strips of width $\Delta x$, approximate each by a rectangle, add them up, and let $n \to \infty$.

$$
\int_a^b f(x)\,dx = \lim_{n\to\infty}\sum_{i=1}^{n} f(x_i^*)\,\Delta x .
$$

<div class="viz" data-viz="riemann-sum">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; rectangles converging on an area</p>
    <p class="viz-sub">Raise n and watch the approximation close on the exact value, whichever rule you pick. Left and right sums bracket the answer for a monotone integrand; the midpoint and trapezoid rules converge much faster.</p>
  </div>
</div>

**The fundamental theorem** says these are the same problem:

$$
\int_a^b f(x)\,dx = F(b) - F(a), \qquad\text{where } F' = f .
$$

So you never have to compute that limit. Find any antiderivative and subtract. Everything below is technique for finding one.

### Worked example: $\int 2x\cos(x^2)\,dx$

Nothing in the basic table matches this. But look at the structure: there is an inner function $x^2$, and its derivative $2x$ is *already sitting there* as a factor. That is the signature of a substitution.

Set $u = x^2$. Then $\dfrac{du}{dx} = 2x$, which we write as $du = 2x\,dx$ — and $2x\,dx$ is exactly the part of the integrand that is not $\cos(x^2)$:

$$
\int \underbrace{\cos(x^2)}_{\cos u}\underbrace{2x\,dx}_{du}
= \int \cos u\,du
= \sin u + C
= \sin(x^2) + C .
$$

**Always check by differentiating.** $\dfrac{d}{dx}\sin(x^2) = \cos(x^2)\cdot 2x$ ✓. This costs five seconds and catches everything.

Why it works: substitution *is* the chain rule backwards. The chain rule says $\frac{d}{dx}F(g(x)) = F'(g(x))g'(x)$; reading that equation right to left says $\int F'(g(x))g'(x)\,dx = F(g(x)) + C$. So when you spot "something composed, times the derivative of the inner thing", you are seeing the output of a chain rule and undoing it.

<div class="q-callout q-callout-insight" markdown="1">
<div class="q-callout-title"><i class="fas fa-lightbulb"></i> The asymmetry worth internalising</div>

Differentiation is an algorithm: apply the rules, always terminates, always works. Integration is a search. There is no procedure that integrates an arbitrary elementary function, and plenty of perfectly innocent ones — $e^{-x^2}$, $\frac{\sin x}{x}$ — have no elementary antiderivative at all. So "techniques of integration" is really a list of patterns to recognize, and getting good means having seen enough of them.

</div>

</div>
</details>

<div class="q-problem" markdown="1">
<div class="q-problem-label">Problem</div>

**Part 1 — substitution.**

$$
\text{(a)}\;\; \int \frac{x}{x^2+1}\,dx
\qquad
\text{(b)}\;\; \int \sin^3 x\,\cos x\,dx
\qquad
\text{(c)}\;\; \int_1^{e} \frac{\ln x}{x}\,dx
$$

**Part 2 — integration by parts.**

$$
\text{(d)}\;\; \int x e^x\,dx
\qquad
\text{(e)}\;\; \int \ln x\,dx
\qquad
\text{(f)}\;\; \int_0^{\pi} x\sin x\,dx
$$

For (c) and (f), give the numerical value.

</div>

Work them on paper first — that is what the page is for.

<details class="q-solution" id="solution">
<summary class="q-solution-summary"><span class="q-when-closed">Show the solution</span><span class="q-when-open">Hide the solution</span></summary>
<div class="q-solution-body" markdown="1">

## The answers

| | integral | value | move |
|---|---|---|---|
| (a) | $\displaystyle\int \frac{x}{x^2+1}dx$ | $\tfrac12\ln(x^2+1) + C$ | $u = x^2+1$ |
| (b) | $\displaystyle\int \sin^3x\cos x\,dx$ | $\tfrac14\sin^4 x + C$ | $u = \sin x$ |
| (c) | $\displaystyle\int_1^e \frac{\ln x}{x}dx$ | $\tfrac12$ | $u = \ln x$, limits $0 \to 1$ |
| (d) | $\displaystyle\int xe^x dx$ | $e^x(x-1) + C$ | parts, $u = x$ |
| (e) | $\displaystyle\int \ln x\,dx$ | $x\ln x - x + C$ | parts, $dv = dx$ |
| (f) | $\displaystyle\int_0^{\pi} x\sin x\,dx$ | $\pi$ | parts, $u = x$ |

## Substitution: find the inner function

The recipe: pick $u$ to be an inner function whose derivative appears (up to a constant) elsewhere in the integrand; compute $du$; rewrite so that *nothing* in $x$ remains; integrate; substitute back.

**(a) $\displaystyle\int \frac{x}{x^2+1}\,dx$.** Inner function $u = x^2+1$, so $du = 2x\,dx$, i.e. $x\,dx = \tfrac12 du$. The numerator is a constant multiple of what we need:

$$
\int\frac{x\,dx}{x^2+1} = \frac12\int\frac{du}{u} = \frac12\ln\lvert u\rvert + C = \frac12\ln(x^2+1) + C .
$$

The absolute value is dropped legitimately because $x^2+1 > 0$ always.

**(b) $\displaystyle\int \sin^3x\cos x\,dx$.** Here $u = \sin x$ and $du = \cos x\,dx$, which is the entire rest of the integrand:

$$
\int u^3\,du = \frac{u^4}{4} + C = \frac{\sin^4 x}{4} + C .
$$

Check: $\frac{d}{dx}\frac{\sin^4x}{4} = \sin^3 x\cos x$ ✓.

**(c) $\displaystyle\int_1^e \frac{\ln x}{x}\,dx$.** Take $u = \ln x$, $du = \frac{dx}{x}$. For a *definite* integral you have two choices, and one is much safer.

**Change the limits** as you substitute: when $x = 1$, $u = \ln 1 = 0$; when $x = e$, $u = \ln e = 1$. The integral becomes an integral in $u$ outright:

$$
\int_{x=1}^{x=e}\frac{\ln x}{x}dx = \int_{u=0}^{u=1} u\,du = \left[\frac{u^2}{2}\right]_0^1 = \frac12 .
$$

No substituting back at all. The alternative — find the antiderivative $\tfrac12(\ln x)^2$ in terms of $x$, then evaluate at 1 and $e$ — gives the same $\tfrac12$, but leaves you one step where it is easy to plug the old limits into the new variable. Change the limits.

## Integration by parts: the product rule backwards

Start from the product rule and integrate both sides:

$$
(uv)' = u'v + uv'
\quad\Longrightarrow\quad
uv = \int u'v\,dx + \int uv'\,dx ,
$$

$$
\boxed{\;\int u\,dv = uv - \int v\,du\;}
$$

It does not solve the integral; it *trades* it for a different one. The skill is picking $u$ so the trade is favourable — you want $du$ simpler than $u$, and $dv$ something you can actually integrate.

The usual mnemonic for choosing $u$ is **LIATE** — Logarithmic, Inverse trig, Algebraic, Trigonometric, Exponential — take $u$ to be whichever appears first on that list. It is a heuristic, not a theorem, but it is right most of the time because it front-loads the functions that get *simpler* when differentiated.

**(d) $\displaystyle\int xe^x\,dx$.** Algebraic beats exponential, so $u = x$, $dv = e^x dx$. Then $du = dx$ and $v = e^x$:

$$
\int xe^x dx = xe^x - \int e^x dx = xe^x - e^x + C = e^x(x-1) + C .
$$

The trade worked because differentiating $x$ turned it into $1$, killing the hard part. Had you chosen $u = e^x$ you would have got $\frac{x^2}{2}e^x - \int\frac{x^2}{2}e^x dx$ — a *worse* integral. That is the diagnostic: if the new integral looks harder, you picked $u$ backwards.

**(e) $\displaystyle\int \ln x\,dx$.** There appears to be nothing to split. The move is to take $dv = dx$ — the integrand is $\ln x$ times $1$:

$$
u = \ln x,\quad dv = dx
\quad\Longrightarrow\quad
du = \frac{dx}{x},\quad v = x ,
$$

$$
\int \ln x\,dx = x\ln x - \int x\cdot\frac1x\,dx = x\ln x - \int dx = x\ln x - x + C .
$$

Worth memorizing outright, but worth being able to re-derive, because the same trick gets $\int\arcsin x\,dx$ and $\int\arctan x\,dx$.

**(f) $\displaystyle\int_0^{\pi} x\sin x\,dx$.** Take $u = x$, $dv = \sin x\,dx$, so $du = dx$ and $v = -\cos x$. With limits, the $uv$ term gets evaluated too:

$$
\int_0^\pi x\sin x\,dx
= \Big[-x\cos x\Big]_0^\pi + \int_0^\pi \cos x\,dx .
$$

First piece: $-\pi\cos\pi - 0 = -\pi(-1) = \pi$. Second: $[\sin x]_0^\pi = 0 - 0 = 0$. So the answer is

$$
\int_0^\pi x\sin x\,dx = \pi .
$$

Sanity check on the sign: $\sin x \ge 0$ across $[0,\pi]$ and $x \ge 0$, so the integrand is non-negative and the answer must be positive. ✓

## Choosing between them

Given an unfamiliar integral, in order:

1. **Is it on the table already, or one algebraic step from it?** Expand, split the fraction, use an identity.
2. **Is there an inner function whose derivative is present?** Substitute.
3. **Is it a product of two unrelated kinds of function** — a polynomial times a trig or exponential, or a lone logarithm or inverse trig? Parts.
4. **Is it a rational function?** Partial fractions — the [next question](/interview-prep/calculus/integrals-partial-fractions-trig-sub/).
5. **Is there a $\sqrt{a^2 \pm x^2}$?** Trig substitution, also next question.

## The slips worth naming

- Dropping $+C$.
- Substituting for $u$ but leaving an $x$ behind. If any $x$ survives, the substitution is not finished — either solve for $x$ in terms of $u$ or pick a different $u$.
- Forgetting to change the limits on a definite substitution, then evaluating the $u$-antiderivative at the $x$-limits.
- Losing the minus sign in $\int u\,dv = uv - \int v\,du$.
- On a definite integral by parts, forgetting that the $uv$ term is also evaluated at both limits.
- Choosing $u$ so the new integral is worse, and then pressing on rather than swapping.

</div>
</details>

## More practice

<details class="q-reveal">
<summary>Drill 1 &mdash; $\displaystyle\int \tan x\,dx$</summary>
<div class="q-reveal-body" markdown="1">

Rewrite before doing anything clever: $\tan x = \dfrac{\sin x}{\cos x}$. Now $u = \cos x$, $du = -\sin x\,dx$:

$$
\int\frac{\sin x}{\cos x}dx = -\int\frac{du}{u} = -\ln\lvert u\rvert + C = -\ln\lvert\cos x\rvert + C ,
$$

often written $\ln\lvert\sec x\rvert + C$. The general pattern is worth naming: $\displaystyle\int\frac{g'(x)}{g(x)}dx = \ln\lvert g(x)\rvert + C$. Whenever the numerator is the derivative of the denominator, the answer is a logarithm.

</div>
</details>

<details class="q-reveal">
<summary>Drill 2 &mdash; $\displaystyle\int x^2 e^x\,dx$</summary>
<div class="q-reveal-body" markdown="1">

Parts twice. First pass, $u = x^2$, $dv = e^x dx$:

$$
\int x^2 e^x dx = x^2 e^x - 2\int xe^x\,dx .
$$

The remaining integral is (d) from the problem, $e^x(x-1)$, so

$$
\int x^2 e^x dx = x^2 e^x - 2e^x(x-1) + C = e^x\big(x^2 - 2x + 2\big) + C .
$$

Each pass drops the polynomial degree by one, so $\int x^n e^x dx$ takes $n$ passes. Check by differentiating: $e^x(x^2-2x+2) + e^x(2x-2) = x^2e^x$ ✓.

</div>
</details>

<details class="q-reveal">
<summary>Drill 3 &mdash; $\displaystyle\int e^x \sin x\,dx$, where parts never terminates</summary>
<div class="q-reveal-body" markdown="1">

Neither factor gets simpler when differentiated, so parts cannot bottom out. Do it anyway, twice, keeping the *same* choice of which factor to differentiate both times. Call the integral $I$.

Take $u = \sin x$, $dv = e^x dx$:

$$
I = e^x\sin x - \int e^x\cos x\,dx .
$$

Again on the new one, $u = \cos x$, $dv = e^x dx$:

$$
\int e^x\cos x\,dx = e^x\cos x + \int e^x\sin x\,dx = e^x\cos x + I .
$$

Substitute back:

$$
I = e^x\sin x - e^x\cos x - I
\quad\Longrightarrow\quad
2I = e^x(\sin x - \cos x) ,
$$

$$
\int e^x\sin x\,dx = \frac{e^x(\sin x - \cos x)}{2} + C .
$$

The original integral reappearing is not failure — it turns the problem into a linear equation for $I$. The one trap: if you switch which factor you differentiate on the second pass, you get the true but useless $I = I$.

</div>
</details>
