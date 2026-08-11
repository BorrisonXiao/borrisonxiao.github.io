---
layout: interview_question
topic: signal-processing
order: 1
title: "Rectangular to Polar: a Complex Number as a Length and an Angle"
difficulty: warm-up
date: 2026-08-11
tags: [complex numbers, Euler's formula, atan2, phasors]
summary: >-
  Convert a + jb into r e^{j theta}. The arithmetic takes ten seconds; the point
  is that polar is the coordinate system in which multiplication, phasors, and
  frequency response all become easy. Covers the atan2 quadrant trap, Euler's
  formula, and three interactive plots.
scripts:
  - /assets/js/interview/complex-polar.js
concepts:
  - name: Euler's formula
    url: https://en.wikipedia.org/wiki/Euler%27s_formula
    note: >-
      e^{j theta} = cos theta + j sin theta. This is the whole bridge between the
      two coordinate systems.
  - name: Argument of a complex number, and its principal value
    url: https://en.wikipedia.org/wiki/Argument_(complex_analysis)
    note: >-
      The angle is only defined modulo 2 pi. Picking the representative in
      (-pi, pi] is a convention, and you have to say which one you are using.
  - name: atan2
    url: https://en.wikipedia.org/wiki/Atan2
    note: >-
      The two-argument arctangent, which keeps the signs of the real and
      imaginary parts separate instead of collapsing them into a ratio. This is
      the thing the question is actually testing.
  - name: Phasors
    url: https://en.wikipedia.org/wiki/Phasor
    note: >-
      A real sinusoid carries exactly two numbers, amplitude and phase. Those two
      numbers are the polar form of one complex constant.
  - name: Frequency response of an LTI system
    url: https://en.wikipedia.org/wiki/Frequency_response
    note: >-
      H(e^{j w}) is a complex number at each frequency. Its modulus is the
      magnitude response and its argument is the phase response - nothing more.
  - name: numpy.angle
    url: https://numpy.org/doc/stable/reference/generated/numpy.angle.html
    note: The vectorized version of all of the above.
references:
  - text: >-
      Oppenheim & Willsky, *Signals and Systems*, 2nd ed. - the complex-number
      review in the appendix, and Ch. 1 on exponential and sinusoidal signals.
  - text: >-
      Lyons, *Understanding Digital Signal Processing* - the early chapters give
      the phasor picture more slowly than most texts, with pictures.
  - text: >-
      Brown & Churchill, *Complex Variables and Applications* - Ch. 1 for polar
      form, arguments, powers, and roots.
---

New to this, or just cold? Open the example first. Otherwise go straight to the problem.

<details class="q-example" id="example">
<summary class="q-example-summary"><span class="q-when-closed">Show a worked example first</span><span class="q-when-open">Hide the example</span></summary>
<div class="q-example-body" markdown="1">

### What a complex number is

**Algebraically**, you invent one new symbol $j$ with the single property

$$
j^2 = -1 ,
$$

and a complex number is anything of the form

$$
z = a + jb, \qquad a, b \in \mathbb{R},
$$

where $a = \Re z$ is the *real part* and $b = \Im z$ is the *imaginary part*. (Signal processing writes $j$ where mathematics writes $i$, because $i$ was already taken by current. Nothing else changes.) From there you just do ordinary algebra and replace $j^2$ by $-1$ whenever it appears:

$$
(a + jb) + (c + jd) = (a+c) + j(b+d),
$$

$$
(a + jb)(c + jd) = ac + jad + jbc + \underbrace{j^2}_{-1}bd = (ac - bd) + j(ad + bc).
$$

Two more definitions you will use constantly. The **conjugate** flips the sign of the imaginary part, $\bar z = a - jb$, and the point of it is that

$$
z\bar z = (a+jb)(a-jb) = a^2 + b^2 ,
$$

which is real and non-negative. Its square root is the **modulus** $\lvert z\rvert = \sqrt{a^2+b^2}$.

**Geometrically**, $z = a + jb$ is just the point $(a, b)$ — or the arrow from the origin to it — in a plane whose horizontal axis measures the real part and whose vertical axis measures the imaginary part. That picture is called the complex plane. In it:

- **adding** two complex numbers is adding two arrows, tip to tail, exactly as with vectors;
- $\lvert z \rvert$ is the **length** of the arrow, i.e. its distance from the origin;
- the **argument** $\arg z$ is the angle the arrow makes with the positive real axis, measured counter-clockwise;
- **multiplying** stretches and rotates — more on that in the solution.

So an arrow can be named two ways: by its components $(a, b)$, or by its length and direction $(r, \theta)$. That is the whole of this question.

$$
z = \underbrace{a + jb}_{\text{rectangular}} = \underbrace{r\,e^{j\theta}}_{\text{polar}}
$$

The exponential in the polar form is **Euler's formula**,

$$
e^{j\theta} = \cos\theta + j\sin\theta ,
$$

which says that $e^{j\theta}$ is the point you reach by walking $\theta$ radians counter-clockwise around the unit circle starting from $1$. It is pure direction with length $1$, so in $re^{j\theta}$ the $r$ carries all the length and the $\theta$ carries all the direction.

### Worked example: $z = 1 + j$

**Locate it.** $a = 1$, $b = 1$: one step right, one step up. Both parts positive, so it is in the first quadrant.

**Get the modulus.** It is the hypotenuse of a right triangle with legs $a$ and $b$:

$$
r = \sqrt{a^2 + b^2} = \sqrt{1 + 1} = \sqrt2 \approx 1.4142 .
$$

**Get the argument.** In the first quadrant the arrow's angle satisfies $\tan\theta = b/a$, so

$$
\theta = \arctan\frac{b}{a} = \arctan 1 = \frac{\pi}{4} = 45^\circ ,
$$

which is what the picture says too — equal legs means the $45^\circ$ diagonal.

**Write it down.**

$$
z = \sqrt2\,e^{j\pi/4}
  = \sqrt2\left(\cos\tfrac{\pi}{4} + j\sin\tfrac{\pi}{4}\right)
  \quad\text{or, in the angle notation,}\quad \sqrt2 \angle 45^\circ .
$$

**Check by converting back.** $\sqrt2\cos\frac{\pi}{4} = \sqrt2\cdot\frac{\sqrt2}{2} = 1$ and $\sqrt2\sin\frac{\pi}{4} = 1$, so we land on $1 + j$ again. Always do this — it costs five seconds and catches every sign error.

Drag the point around and watch the four numbers move together.

<div class="viz" data-viz="argand-explorer">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; the complex plane</p>
    <p class="viz-sub">Start on the "1 + j" preset to see the example above. Then drag the blue point, or click the canvas and use the arrow keys. The shaded wedge is the argument, the spoke is the modulus, and the dotted lines are the rectangular coordinates.</p>
  </div>
</div>

<div class="q-callout q-callout-trap" markdown="1">
<div class="q-callout-title"><i class="fas fa-triangle-exclamation"></i> One catch before you start</div>

That $\theta = \arctan(b/a)$ step worked because $1 + j$ sits in the first quadrant. Drag the point into the left half of the plane and the widget's last readout starts disagreeing with the true angle. Handling that is most of what the problem below is testing.

</div>

</div>
</details>

<div class="q-problem" markdown="1">
<div class="q-problem-label">Problem</div>

**Part 1.** Write each of the following in polar form $z = re^{j\theta}$, with $r \ge 0$ and $\theta$ the *principal* argument, i.e. the one in $(-\pi, \pi]$:

$$
z_1 = 1 + j,\qquad
z_2 = -1 + j\sqrt{3},\qquad
z_3 = -3,\qquad
z_4 = -2j,\qquad
z_5 = 3 - 4j.
$$

**Part 2.** Go the other way: write $z_6 = 5e^{-j3\pi/4}$ in rectangular form $a + jb$.

**Part 3.** The reason anybody asks. A first-order FIR filter has frequency response

$$
H(e^{j\omega}) = 1 - a e^{-j\omega}.
$$

With $a = 0.8$, give the magnitude $\lvert H \rvert$ and the phase $\angle H$ at $\omega = \pi/2$.

</div>

Work them on paper first — that is what the page is for.

<details class="q-solution" id="solution">
<summary class="q-solution-summary"><span class="q-when-closed">Show the solution</span><span class="q-when-open">Hide the solution</span></summary>
<div class="q-solution-body" markdown="1">

## The answers

| | rectangular | $r$ | principal $\theta$ | polar |
|---|---|---|---|---|
| $z_1$ | $1 + j$ | $\sqrt{2} \approx 1.4142$ | $\pi/4 = 45^\circ$ | $\sqrt{2}\,e^{j\pi/4}$ |
| $z_2$ | $-1 + j\sqrt{3}$ | $2$ | $2\pi/3 = 120^\circ$ | $2e^{j2\pi/3}$ |
| $z_3$ | $-3$ | $3$ | $\pi = 180^\circ$ | $3e^{j\pi}$ |
| $z_4$ | $-2j$ | $2$ | $-\pi/2 = -90^\circ$ | $2e^{-j\pi/2}$ |
| $z_5$ | $3 - 4j$ | $5$ | $-\arctan\tfrac{4}{3} \approx -0.9273 = -53.13^\circ$ | $5e^{-j0.9273}$ |
| $z_6$ | $\approx -3.5355 - 3.5355j$ | $5$ | $-3\pi/4 = -135^\circ$ | given |
| $H$ | $1 + 0.8j$ | $\sqrt{1.64} \approx 1.2806$ | $\arctan 0.8 \approx 0.6747 = 38.66^\circ$ | $1.2806\,e^{j0.6747}$ |

The three ways this goes wrong, in order of how often I have seen them:

- **$z_2$ comes out as $-60^\circ$.** That is $\arctan(\sqrt{3}/(-1))$, and it is off by exactly $\pi$. This is the whole point of the question — see the next section.
- **$z_3$ comes out as $-\pi$.** Right modulo $2\pi$, but not the principal value: the interval $(-\pi, \pi]$ is closed at the top, so the negative real axis gets $+\pi$.
- **$z_4$ makes people reach for $\arctan(b/a)$ and divide by zero.** $\operatorname{atan2}(-2, 0)$ answers $-\pi/2$ without blinking.

## Why bother with polar at all

Rectangular and polar name the same point, and neither is more correct. They are good at different things, and that is the only reason to convert between them:

- **Rectangular makes addition trivial.** $(a_1 + jb_1) + (a_2 + jb_2) = (a_1+a_2) + j(b_1+b_2)$. It tells you nothing useful about products.
- **Polar makes multiplication trivial.** $r_1 r_2 e^{j(\theta_1+\theta_2)}$ — moduli multiply, arguments add. It tells you nothing useful about sums.

<div class="q-callout q-callout-insight" markdown="1">
<div class="q-callout-title"><i class="fas fa-lightbulb"></i> The habit to build</div>

Do the sums in rectangular, do the products in polar, and convert at the boundary. Nearly every manipulation below is an instance of that one sentence.

</div>

## Getting there: $r$ by Pythagoras, $\theta$ by atan2

Start from $a = r\cos\theta$ and $b = r\sin\theta$. Squaring and adding,

$$
a^2 + b^2 = r^2(\cos^2\theta + \sin^2\theta) = r^2
\quad\Longrightarrow\quad
r = \sqrt{a^2 + b^2} = \lvert z \rvert ,
$$

taking the non-negative root because we insisted $r \ge 0$. That half is unambiguous. Dividing the two equations instead,

$$
\frac{b}{a} = \tan\theta ,
$$

which tempts you to write $\theta = \arctan(b/a)$. That is wrong for half the plane:

<div class="q-callout q-callout-trap" markdown="1">
<div class="q-callout-title"><i class="fas fa-triangle-exclamation"></i> The trap</div>

$\tan$ has period $\pi$, not $2\pi$. Forming the ratio $b/a$ throws away the common sign, because $(a,b)$ and $(-a,-b)$ give the same ratio. And $\arctan$ only returns answers in $(-\pi/2, \pi/2)$, so it can never point into the left half plane. Every $z$ with $a < 0$ comes back rotated by $\pi$ from where it actually is.

</div>

The fix is to keep $a$ and $b$ apart instead of dividing them, which is exactly what the two-argument arctangent does:

$$
\boxed{\;r = \sqrt{a^2+b^2}, \qquad \theta = \operatorname{atan2}(b,\, a) \in (-\pi, \pi] \;}
$$

Written out, $\operatorname{atan2}$ is $\arctan(b/a)$ plus a quadrant correction:

| quadrant | $\operatorname{sign}(a)$ | $\operatorname{sign}(b)$ | correction to $\arctan(b/a)$ |
|---|---|---|---|
| I | $+$ | $+$ | none |
| II | $-$ | $+$ | $+\pi$ |
| III | $-$ | $-$ | $-\pi$ |
| IV | $+$ | $-$ | none |

plus the axes, which $\arctan$ cannot reach at all: $\theta = \pi/2$ for $a=0, b>0$; $\theta = -\pi/2$ for $a=0, b<0$; $\theta = \pi$ for $a<0, b=0$; and $\theta$ undefined at the origin, which has a modulus but no direction.

The pattern in that last column is the whole of `atan2`: add $\pi$ with the sign of $b$ whenever $a < 0$, otherwise do nothing.

To watch this happen, open the [worked example](#example) above and drag the point into the left half of the plane: the widget's last readout is the naive $\arctan(b/a)$, and it flags every disagreement. Its <code>convention</code> button also relabels the angle as $[0, 2\pi)$ instead of $(-\pi, \pi]$ — notice the point does not move, only the label does. Both are correct, which is exactly why you have to say which one you are using.

## Working through the five

The recipe: modulus by Pythagoras, reference angle from the magnitudes, quadrant from the signs.

**$z_1 = 1 + j$.** Done in the example: $r = \sqrt2$, quadrant I so no correction, $\theta = \arctan 1 = \pi/4$, giving $z_1 = \sqrt{2}\,e^{j\pi/4}$.

**$z_2 = -1 + j\sqrt3$.** $r = \sqrt{1+3} = 2$. The *reference* angle — the acute angle to the real axis — is $\arctan(\lvert b\rvert/\lvert a\rvert) = \arctan\sqrt3 = \pi/3$. Quadrant II, so $\theta = \pi - \pi/3 = 2\pi/3 = 120^\circ$, giving $z_2 = 2e^{j2\pi/3}$. Had you written $\arctan(-\sqrt3) = -\pi/3$ you would have named the point $-1 - j\sqrt3$, which is $z_2$ reflected through the origin — precisely the ambiguity the ratio cannot resolve.

**$z_3 = -3$.** $r = 3$, on the negative real axis, so $\theta = \pi$ and $z_3 = 3e^{j\pi}$. (Strip off the $3$ and this is $e^{j\pi} = -1$, which just says half a turn takes $1$ to $-1$.)

**$z_4 = -2j$.** $r = 2$, straight down, so $\theta = -\pi/2$ and $z_4 = 2e^{-j\pi/2}$. There is no ratio to form here — `atan2` handles it because it looks at signs, not quotients.

**$z_5 = 3 - 4j$.** $r = \sqrt{9+16} = 5$, the 3–4–5 triangle, which is why this one shows up in interviews. Quadrant IV, so $\theta = -\arctan(4/3) \approx -53.13^\circ$ and $z_5 = 5e^{-j0.9273}$. Not a nice angle — most numbers do not have one, and being fluent means being comfortable leaving the answer as $-\arctan(4/3)$.

**$z_6 = 5e^{-j3\pi/4}$ back to rectangular.** Just evaluate Euler:

$$
z_6 = 5\big(\cos(-\tfrac{3\pi}{4}) + j\sin(-\tfrac{3\pi}{4})\big)
= 5\Big(-\tfrac{\sqrt2}{2} - j\tfrac{\sqrt2}{2}\Big)
\approx -3.5355 - 3.5355j .
$$

Sanity check: $-135^\circ$ points into quadrant III, and indeed both components came out negative.

**Part 3.** At $\omega = \pi/2$ we have $e^{-j\pi/2} = -j$, so

$$
H(e^{j\pi/2}) = 1 - 0.8(-j) = 1 + 0.8j ,
$$

and now it is just a rectangular-to-polar conversion wearing a hat:

$$
\lvert H \rvert = \sqrt{1 + 0.64} \approx 1.2806,
\qquad
\angle H = \arctan(0.8) \approx 0.6747\ \text{rad} = 38.66^\circ .
$$

That is the point of the exercise. A magnitude response and a phase response are not extra concepts bolted onto complex numbers — they *are* $r$ and $\theta$, computed once per frequency.

## Where Euler's formula comes from

The example took $e^{j\theta} = \cos\theta + j\sin\theta$ on trust, and nothing so far has actually needed the exponential — we could have written $r\angle\theta$ and stopped. The reason for the exponential notation is that the exponential is *the* function that turns addition into multiplication, which is exactly the structure polar coordinates are exposing.

As for why it is true, feed $j\theta$ to the exponential series:

$$
e^{j\theta} = \sum_{n=0}^{\infty} \frac{(j\theta)^n}{n!} .
$$

Powers of $j$ cycle with period four — $1, j, -1, -j$ — so splitting by parity of $n$, with $j^{2k} = (-1)^k$ and $j^{2k+1} = (-1)^k j$:

$$
e^{j\theta}
= \underbrace{\sum_{k} \frac{(-1)^k \theta^{2k}}{(2k)!}}_{\cos\theta}
\;+\; j\underbrace{\sum_{k} \frac{(-1)^k \theta^{2k+1}}{(2k+1)!}}_{\sin\theta}
= \cos\theta + j\sin\theta .
$$

Geometrically: $e^{j\theta}$ is the point one gets by walking $\theta$ radians counter-clockwise around the unit circle from $1$. That is why $\lvert e^{j\theta}\rvert = 1$ — it is pure direction, no length — and why $r$ and $\theta$ in $re^{j\theta}$ never interfere with each other.

Two consequences used constantly:

$$
\overline{re^{j\theta}} = re^{-j\theta},
\qquad
\cos\theta = \frac{e^{j\theta} + e^{-j\theta}}{2},
\quad
\sin\theta = \frac{e^{j\theta} - e^{-j\theta}}{2j} .
$$

Those last two — the inverse Euler formulas — turn any trigonometric identity into an algebra problem about exponentials, and they are the reason a real cosine shows up in a spectrum as *two* lines, at $+\omega_0$ and $-\omega_0$.

## What polar form is good for

### Multiplying is rotating

$$
z_1 z_2 = r_1 r_2 e^{j(\theta_1 + \theta_2)} .
$$

Multiplying by $w$ means *scale by $\lvert w\rvert$, rotate by $\angle w$*. So $j$ is the quarter-turn operator, and $j^2 = -1$ is just "two quarter turns make a half turn".

<div class="viz" data-viz="argand-multiply">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; multiplication adds the angles</p>
    <p class="viz-sub">Drag either factor. The orange product is z stretched by the modulus of w and rotated by the argument of w. Set w to j and watch the product make a quarter turn.</p>
  </div>
</div>

Powers follow immediately: $z^n = r^n e^{jn\theta}$. So $(1+j)^8 = (\sqrt2)^8 e^{j8\pi/4} = 16e^{j2\pi} = 16$, which you would not want to get by binomial expansion.

### Phasors: the two numbers a sinusoid carries

A real sinusoid at a known frequency carries exactly two pieces of information, amplitude and phase. Package them into one complex number:

$$
A\cos(\omega_0 t + \phi) = \Re\big\{ \underbrace{A e^{j\phi}}_{\text{phasor}} \; e^{j\omega_0 t} \big\}.
$$

The phasor $Ae^{j\phi}$ is a *constant*, and its polar form is (amplitude, phase). All the time dependence sits in $e^{j\omega_0 t}$, which is the same for every signal at that frequency and so says nothing about this one.

<div class="viz" data-viz="phasor">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; a phasor and the cosine it casts</p>
    <p class="viz-sub">The vector rotates at rate omega; the trace on the right is its real part over time, newest sample at the left. Change the amplitude and the starting phase and watch the waveform follow.</p>
  </div>
</div>

This is what makes the standard manipulation work. To add two sinusoids of the same frequency, add their phasors:

$$
A_1\cos(\omega_0 t + \phi_1) + A_2\cos(\omega_0 t + \phi_2)
= \Re\big\{(A_1e^{j\phi_1} + A_2e^{j\phi_2})e^{j\omega_0 t}\big\} .
$$

You convert both phasors to rectangular, add — because addition is easy there — then convert the sum back to polar to read off the resulting amplitude and phase. That round trip is exactly what this question is drilling, and it is why both directions have to be automatic, not just one.

### Frequency response is $r$ and $\theta$, once per frequency

Push $x[n] = e^{j\omega n}$ through an LTI system with impulse response $h$:

$$
y[n] = \sum_{k} h[k]\,e^{j\omega(n-k)}
     = e^{j\omega n}\underbrace{\sum_k h[k] e^{-j\omega k}}_{H(e^{j\omega})} .
$$

The input comes out unchanged except for a complex scale factor. Write that factor in polar form and you are done: the modulus scales the amplitude, the argument shifts the phase. **The magnitude response and the phase response are the polar coordinates of $H(e^{j\omega})$** — that is all a Bode plot is, and Part 3 was one point on one.

## One helix, two shadows

Here is the picture behind all of it. Plot $e^{j\omega t}$ with time as a third axis and you get a helix. Its shadow on the time–real wall is a cosine, its shadow on the time–imaginary wall is a sine, and looking straight down the time axis you see the unit circle.

<div class="viz" data-viz="helix">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; the complex exponential in 3D</p>
    <p class="viz-sub">Drag to orbit, or use the view buttons to drop into one of the three canonical projections. Nothing about the curve changes between views - only where you are standing.</p>
  </div>
</div>

Cosine and sine are not two functions that happen to be related. They are one rotation, projected two ways: the rectangular coordinates are the two shadows, the polar coordinates are the radius and the angle along the helix.

## Two things that bite

**The principal value has a seam.** $\arg z$ is genuinely multivalued — $\theta$ and $\theta + 2\pi k$ name the same point — so pinning it down needs a convention, and every convention breaks somewhere. With $(-\pi, \pi]$, the seam is the negative real axis: approach $-3$ from above and you get $+\pi$, from below and you get $-\pi$. Nothing is wrong with the number; the labelling scheme just has to jump somewhere. This is why a numerically computed phase response comes out sawtoothed, and why `np.unwrap` exists.

**Compute $r$ with `hypot`, not `sqrt`.** $\sqrt{a^2+b^2}$ overflows as soon as $a^2$ does, even when the answer itself is perfectly representable:

```python
import math
a = 1e200
math.sqrt(a*a + a*a)   # inf     -- a*a already overflowed
math.hypot(a, a)       # 1.4142135623730951e+200
```

`math.hypot`, `np.hypot` and `np.abs` on complex arrays all use a scaled algorithm and do the right thing. Free correctness.

## In code

```python
import numpy as np

z = np.array([1 + 1j, -1 + np.sqrt(3) * 1j, -3 + 0j, -2j, 3 - 4j])

r = np.abs(z)                    # modulus, computed stably
th = np.angle(z)                 # principal argument in (-pi, pi], i.e. atan2(b, a)
deg = np.angle(z, deg=True)      # same thing in degrees

# Round trip back to rectangular; Euler's formula is the implementation.
z_back = r * np.exp(1j * th)
assert np.allclose(z, z_back)

# What NOT to do -- arctan of the ratio loses the quadrant.
naive = np.arctan(z.imag / z.real)
print(np.rad2deg(th))            # [ 45.  120.  180.  -90.  -53.13]
print(np.rad2deg(naive))         # [ 45.  -60.    0.    nan -53.13]
#                                       ^^^^^  ^^^   ^^^
#             quadrant II lands in IV; the negative real axis lands at 0;
#             the pure-imaginary case divides by zero.
```

</div>
</details>

## More practice

<details class="q-reveal">
<summary>Drill 1 &mdash; convert $-5 - 5j$ to polar</summary>
<div class="q-reveal-body" markdown="1">

$r = \sqrt{25+25} = 5\sqrt2 \approx 7.0711$. Both parts negative, so quadrant III; the reference angle is $\arctan(5/5) = \pi/4$ and the correction is $-\pi$:

$$
\theta = \tfrac{\pi}{4} - \pi = -\tfrac{3\pi}{4} = -135^\circ,
\qquad z = 5\sqrt2\,e^{-j3\pi/4}.
$$

Same angle as $z_6$ in the problem, which is no accident — $5e^{-j3\pi/4}$ and $-5-5j$ point the same way.

</div>
</details>

<details class="q-reveal">
<summary>Drill 2 &mdash; simplify $\dfrac{1+j}{1-j}$ without expanding</summary>
<div class="q-reveal-body" markdown="1">

Convert both, then divide — moduli divide, arguments subtract:

$$
\frac{1+j}{1-j} = \frac{\sqrt2\,e^{j\pi/4}}{\sqrt2\,e^{-j\pi/4}} = e^{j\pi/2} = j .
$$

The rectangular route (multiply top and bottom by the conjugate) gets the same answer in three times the writing.

</div>
</details>

<details class="q-reveal">
<summary>Drill 3 &mdash; find all $z$ with $z^3 = -8$</summary>
<div class="q-reveal-body" markdown="1">

Put the right-hand side in polar and keep the ambiguity: $-8 = 8e^{j(\pi + 2\pi k)}$. Then

$$
z = 8^{1/3} e^{j(\pi + 2\pi k)/3} = 2e^{j(\pi + 2\pi k)/3}, \qquad k = 0, 1, 2,
$$

so $\theta = \pi/3,\ \pi,\ -\pi/3$ and

$$
z \in \{\,1 + j\sqrt3,\ \ -2,\ \ 1 - j\sqrt3\,\} ,
$$

three points spaced $120^\circ$ apart on the circle of radius $2$. Dropping the $2\pi k$ is how people end up reporting one root out of three.

</div>
</details>
