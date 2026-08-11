---
layout: interview_question
topic: signal-processing
order: 1
title: "Rectangular to Polar: a Complex Number as a Length and an Angle"
difficulty: warm-up
date: 2026-08-11
tags: [complex numbers, Euler's formula, atan2, phasors]
summary: >-
  Convert a + jb into r e^{j theta}. The arithmetic takes ten seconds; the
  reason it is asked is that the polar chart is the one in which multiplication,
  frequency response, and everything else in DSP becomes easy. Includes the
  atan2 quadrant trap, three derivations of Euler's formula, and four
  interactive plots.
scripts:
  - /assets/js/interview/complex-polar.js
concepts:
  - name: Euler's formula
    url: https://en.wikipedia.org/wiki/Euler%27s_formula
    note: >-
      e^{j theta} = cos theta + j sin theta. This is the entire bridge between
      the two coordinate systems; everything else on this page is a consequence.
  - name: Argument of a complex number, and its principal value
    url: https://en.wikipedia.org/wiki/Argument_(complex_analysis)
    note: >-
      The angle is only defined modulo 2 pi. Choosing the representative in
      (-pi, pi] is a convention, and it is the convention that puts a branch cut
      along the negative real axis.
  - name: atan2
    url: https://en.wikipedia.org/wiki/Atan2
    note: >-
      The two-argument arctangent, which keeps the signs of the real and
      imaginary parts separate instead of collapsing them into a ratio. The one
      thing this question is actually testing.
  - name: Phasors
    url: https://en.wikipedia.org/wiki/Phasor
    note: >-
      A real sinusoid carries exactly two numbers, amplitude and phase. Those
      two numbers are the polar form of one complex constant.
  - name: Roots of unity and the DFT twiddle factor
    url: https://en.wikipedia.org/wiki/Root_of_unity
    note: >-
      W_N = e^{-j 2 pi / N} is nothing but a rational fraction of a turn. The FFT
      is an exercise in reusing those rotations.
  - name: Frequency response of an LTI system
    url: https://en.wikipedia.org/wiki/Frequency_response
    note: >-
      Complex exponentials are the eigenfunctions of LTI systems, and H(e^{j w})
      is the eigenvalue. Its polar form is, literally, the magnitude response
      and the phase response.
  - name: Pole-zero plot
    url: https://en.wikipedia.org/wiki/Pole%E2%80%93zero_plot
    note: >-
      Once H is factored, the magnitude response is a product of distances and
      the phase response a sum of angles - pure polar thinking, done graphically.
  - name: Group delay and phase unwrapping
    url: https://en.wikipedia.org/wiki/Group_delay_and_phase_delay
    note: >-
      Group delay is minus the derivative of the phase, so it only means
      anything after the 2 pi jumps introduced by the principal value have been
      unwrapped.
  - name: numpy.angle / numpy.unwrap
    url: https://numpy.org/doc/stable/reference/generated/numpy.angle.html
    note: The vectorized versions of everything below.
references:
  - text: >-
      Oppenheim & Willsky, *Signals and Systems*, 2nd ed. - the complex-number
      review in the appendix, and Ch. 1 on exponential and sinusoidal signals.
  - text: >-
      Oppenheim & Schafer, *Discrete-Time Signal Processing*, 3rd ed. - Ch. 2 for
      the frequency response as an eigenvalue, Ch. 5 for pole-zero geometry,
      phase, and group delay.
  - text: >-
      Brown & Churchill, *Complex Variables and Applications* - Ch. 1 for polar
      form, arguments, powers, and roots, done properly.
  - text: >-
      Lyons, *Understanding Digital Signal Processing* - the early chapters give
      the phasor picture more slowly than most texts, with pictures.
---

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

**Part 3.** The reason anybody asks. A first-order FIR filter has impulse response $h[n] = \delta[n] - a\,\delta[n-1]$, so its frequency response is

$$
H(e^{j\omega}) = 1 - a e^{-j\omega}.
$$

With $a = 0.8$, give the magnitude $\lvert H \rvert$ and the phase $\angle H$ at $\omega = \pi/2$.

</div>

Do them on paper first. Then open the box.

<details class="q-reveal">
<summary>Reveal the answers</summary>
<div class="q-reveal-body" markdown="1">

| | rectangular | $r$ | principal $\theta$ | polar |
|---|---|---|---|---|
| $z_1$ | $1 + j$ | $\sqrt{2} \approx 1.4142$ | $\pi/4 = 45^\circ$ | $\sqrt{2}\,e^{j\pi/4}$ |
| $z_2$ | $-1 + j\sqrt{3}$ | $2$ | $2\pi/3 = 120^\circ$ | $2e^{j2\pi/3}$ |
| $z_3$ | $-3$ | $3$ | $\pi = 180^\circ$ | $3e^{j\pi}$ |
| $z_4$ | $-2j$ | $2$ | $-\pi/2 = -90^\circ$ | $2e^{-j\pi/2}$ |
| $z_5$ | $3 - 4j$ | $5$ | $-\arctan\tfrac{4}{3} \approx -0.9273 = -53.13^\circ$ | $5e^{-j0.9273}$ |
| $z_6$ | $-\tfrac{5\sqrt{2}}{2}(1 + j) \approx -3.5355 - 3.5355j$ | $5$ | $-3\pi/4 = -135^\circ$ | given |
| $H$ | $1 + 0.8j$ | $\sqrt{1.64} \approx 1.2806$ | $\arctan 0.8 \approx 0.6747 = 38.66^\circ$ | $1.2806\,e^{j0.6747}$ |

Three ways this goes wrong, in order of how often I have seen them:

- **$z_2$ comes out as $-60^\circ$.** That is $\arctan(\sqrt{3}/(-1)) = \arctan(-\sqrt{3})$, and it is off by exactly $\pi$. See [the next-but-one section](#the-forward-map-and-why-it-is-atan2-not-arctan).
- **$z_3$ comes out as $-\pi$.** Correct modulo $2\pi$, but not the principal value: the interval $(-\pi, \pi]$ is closed at the top, so the negative real axis gets $+\pi$.
- **$z_4$ makes people reach for $\arctan(b/a)$ and divide by zero.** $\operatorname{atan2}(-2, 0)$ answers $-\pi/2$ without blinking.

</div>
</details>

## Two charts on the same plane

A complex number is a point in a plane. The plane does not care how you label it; you get to choose. There are two standard choices, and the interesting question is not how to convert between them but *why you would ever want the second one*.

$$
z = \underbrace{a + jb}_{\text{rectangular}} = \underbrace{r e^{j\theta}}_{\text{polar}},
\qquad a, b \in \mathbb{R},\ r \ge 0,\ \theta \in \mathbb{R}.
$$

(Signal processing writes $j$ where mathematics writes $i$, because $i$ was already spoken for by current. Nothing else changes.)

The answer is that $\mathbb{C}$ carries **two** operations, and each chart linearises one of them:

- **Addition** is componentwise: $(a_1 + jb_1) + (a_2 + jb_2) = (a_1 + a_2) + j(b_1 + b_2)$. Rectangular coordinates make this trivial and say nothing useful about products.
- **Multiplication** is $r_1r_2\,e^{j(\theta_1 + \theta_2)}$: moduli multiply, arguments add. Polar coordinates turn multiplication into one multiplication and one *addition*, and say nothing useful about sums.

<div class="q-callout q-callout-insight" markdown="1">
<div class="q-callout-title"><i class="fas fa-lightbulb"></i> The framing worth keeping</div>

$(\mathbb{C}, +)$ is just $\mathbb{R}^2$ as a vector space, and rectangular coordinates are its natural chart. $(\mathbb{C}\setminus\{0\}, \times)$ is the group $\mathbb{R}_{>0} \times S^1$, and polar coordinates are *its* natural chart — in fact $z \mapsto (\ln r, \theta)$ turns that group into $(\mathbb{R}^2, +)$ as well. Converting between rectangular and polar is switching between the chart that makes sums easy and the chart that makes products easy. Almost every "trick" below is an instance of: *do the sums in rectangular, do the products in polar, convert at the boundary.*

</div>

Two structural facts fall straight out of that and are worth stating before any arithmetic:

1. **The polar chart is singular at the origin.** The Jacobian of $(r,\theta) \mapsto (r\cos\theta,\ r\sin\theta)$ is

   $$
   J = \begin{bmatrix} \cos\theta & -r\sin\theta \\ \sin\theta & \phantom{-}r\cos\theta \end{bmatrix},
   \qquad \det J = r,
   $$

   which vanishes at $r=0$. The number $0$ has a modulus but no argument at all. This is not pedantry; it is the reason the phase of a near-zero DFT bin is numerical noise, which we will come back to.
2. **The angle is only defined modulo $2\pi$.** $\arg z$ is a set, $\{\theta_0 + 2\pi k : k \in \mathbb{Z}\}$. To get a function you must pick a representative, and every such choice has to break somewhere.

## The forward map, and why it is atan2 not arctan

Set $a = r\cos\theta$ and $b = r\sin\theta$. Then

$$
a^2 + b^2 = r^2(\cos^2\theta + \sin^2\theta) = r^2
\quad\Longrightarrow\quad
r = \sqrt{a^2 + b^2} = \lvert z \rvert ,
$$

taking the non-negative root because we insisted $r \ge 0$. That half is unambiguous. The angle is where it goes wrong. Dividing the two equations,

$$
\frac{b}{a} = \frac{r\sin\theta}{r\cos\theta} = \tan\theta ,
$$

which tempts you to write $\theta = \arctan(b/a)$. That is wrong for half the plane, for a reason that is easy to state:

<div class="q-callout q-callout-trap" markdown="1">
<div class="q-callout-title"><i class="fas fa-triangle-exclamation"></i> The trap</div>

$\tan$ has period $\pi$, not $2\pi$. Forming the ratio $b/a$ throws away one bit of information — the common sign — because $(a, b)$ and $(-a, -b)$ produce the same ratio. And $\arctan$ has range $(-\pi/2, \pi/2)$, so it can only ever return an answer in the right half plane. Any $z$ with $a < 0$ comes back rotated by $\pi$ from where it actually is.

</div>

The fix is to keep $a$ and $b$ apart instead of dividing, which is exactly what the two-argument arctangent does:

$$
\boxed{\;r = \lvert z \rvert = \sqrt{a^2+b^2}, \qquad \theta = \operatorname{atan2}(b,\, a) \in (-\pi, \pi] \;}
$$

Spelled out, $\operatorname{atan2}$ is $\arctan(b/a)$ plus a quadrant correction:

| quadrant | $\operatorname{sign}(a)$ | $\operatorname{sign}(b)$ | $\theta$ | correction to $\arctan(b/a)$ |
|---|---|---|---|---|
| I | $+$ | $+$ | $\arctan(b/a)$ | none |
| II | $-$ | $+$ | $\arctan(b/a) + \pi$ | $+\pi$ |
| III | $-$ | $-$ | $\arctan(b/a) - \pi$ | $-\pi$ |
| IV | $+$ | $-$ | $\arctan(b/a)$ | none |

plus the axes, which $\arctan$ cannot reach at all: $\theta = \pi/2$ for $a=0, b>0$; $\theta = -\pi/2$ for $a=0, b<0$; $\theta = \pi$ for $a<0, b=0$; and $\theta$ *undefined* at the origin. (IEEE-754 says `atan2(0.0, 0.0)` returns $0$, and every library obliges. That is a convention to keep code from crashing, not a mathematical fact — see the conditioning discussion below.)

Note the pattern in the correction column: it is $\pi$ times $\operatorname{sign}(b)$ whenever $a < 0$, and zero otherwise. That is the whole of `atan2`.

Drag the point below into the left half plane and watch the two answers separate.

<div class="viz" data-viz="argand-explorer">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; the Argand plane</p>
    <p class="viz-sub">Drag the blue point, or click the canvas and use the arrow keys. The shaded wedge is the argument, the spoke is the modulus, and the dotted lines are the rectangular coordinates. The last readout is the naive answer; the warning fires whenever it disagrees with the truth.</p>
  </div>
</div>

The <code>convention</code> button switches the displayed angle between $(-\pi,\pi]$ and $[0, 2\pi)$. Notice that the *point does not move* — only the label does. Both are correct; they are different sections of the same multivalued $\arg$.

## Worked solutions

The mechanical recipe is: modulus by Pythagoras, reference angle from the magnitudes, quadrant from the signs.

**$z_1 = 1 + j$.** $r = \sqrt{1 + 1} = \sqrt 2$. Quadrant I, and $\arctan(1/1) = \pi/4$, no correction. So $z_1 = \sqrt{2}\,e^{j\pi/4}$.

**$z_2 = -1 + j\sqrt3$.** $r = \sqrt{1 + 3} = 2$. The *reference* angle — the acute angle to the real axis — is $\arctan\!\big(\lvert b \rvert / \lvert a \rvert\big) = \arctan\sqrt3 = \pi/3$. The point is in quadrant II, so $\theta = \pi - \pi/3 = 2\pi/3 = 120^\circ$. So $z_2 = 2e^{j2\pi/3}$. Had you written $\arctan(b/a) = \arctan(-\sqrt3) = -\pi/3$, you would have named the point $-1 - j\sqrt3$, which is $z_2$ reflected through the origin — precisely the ambiguity the ratio cannot resolve.

**$z_3 = -3$.** $r = 3$, and the point sits on the negative real axis, so $\theta = \pi$. So $z_3 = 3e^{j\pi}$. (With $r=3$ stripped off this is $e^{j\pi} = -1$, Euler's identity, which is just the statement that half a turn takes $1$ to $-1$.)

**$z_4 = -2j$.** $r = 2$, straight down the negative imaginary axis, so $\theta = -\pi/2$ and $z_4 = 2e^{-j\pi/2}$. There is no ratio to form here; $\operatorname{atan2}$ handles it because it looks at the signs, not the quotient.

**$z_5 = 3 - 4j$.** $r = \sqrt{9 + 16} = 5$ — the 3–4–5 triangle, which is why this one shows up in interviews. Quadrant IV, so $\theta = -\arctan(4/3) \approx -0.9273\ \text{rad} = -53.13^\circ$, and $z_5 = 5e^{-j0.9273}$. Not a "nice" angle: most complex numbers do not have one, and being fluent means being comfortable leaving the answer as $-\arctan(4/3)$.

**$z_6 = 5e^{-j3\pi/4}$ back to rectangular.** Just evaluate Euler:

$$
z_6 = 5\big(\cos(-\tfrac{3\pi}{4}) + j\sin(-\tfrac{3\pi}{4})\big)
= 5\Big(-\tfrac{\sqrt2}{2} - j\tfrac{\sqrt2}{2}\Big)
= -\tfrac{5\sqrt2}{2}(1 + j) \approx -3.5355 - 3.5355j .
$$

Sanity check: $-135^\circ$ is in quadrant III, and indeed both components came out negative.

**Part 3, the filter.** At $\omega = \pi/2$, $e^{-j\pi/2} = -j$, so

$$
H(e^{j\pi/2}) = 1 - 0.8(-j) = 1 + 0.8j,
$$

which is a rectangular-to-polar conversion wearing a hat:

$$
\lvert H \rvert = \sqrt{1 + 0.64} = \sqrt{1.64} \approx 1.2806,
\qquad
\angle H = \arctan(0.8) \approx 0.6747\ \text{rad} = 38.66^\circ .
$$

That is the whole point of the exercise. A magnitude response and a phase response are not extra concepts bolted onto complex numbers — they *are* $r$ and $\theta$, evaluated once per frequency.

## Why $e^{j\theta}$: three ways to see Euler's formula

Nothing so far needed the exponential. We could have written $z = r\,\angle\theta$ and stopped. The reason to write $e^{j\theta}$ is that the exponential is the function whose defining property is "turns addition into multiplication", which is exactly the structure the polar chart is exposing. Three derivations, each of which is the "real" one in some context.

### 1. The power series

Take the exponential series, which converges absolutely for every complex argument, and feed it $j\theta$:

$$
e^{j\theta} = \sum_{n=0}^{\infty} \frac{(j\theta)^n}{n!} .
$$

Powers of $j$ cycle with period four: $j^0 = 1$, $j^1 = j$, $j^2 = -1$, $j^3 = -j$. Splitting the sum by parity of $n$, with $n = 2k$ giving $j^{2k} = (-1)^k$ and $n = 2k+1$ giving $j^{2k+1} = (-1)^k j$:

$$
e^{j\theta}
= \underbrace{\sum_{k=0}^{\infty} \frac{(-1)^k \theta^{2k}}{(2k)!}}_{\cos\theta}
\;+\; j\underbrace{\sum_{k=0}^{\infty} \frac{(-1)^k \theta^{2k+1}}{(2k+1)!}}_{\sin\theta}
= \cos\theta + j\sin\theta .
$$

Honest but unilluminating: it verifies the identity without saying why a rotation appeared.

### 2. The differential equation — a rotation is what constant-speed circular motion *is*

This one earns its keep. Define $f(\theta) = \cos\theta + j\sin\theta$ and differentiate:

$$
f'(\theta) = -\sin\theta + j\cos\theta = j(\cos\theta + j\sin\theta) = j\,f(\theta),
\qquad f(0) = 1 .
$$

So $f$ solves $f' = jf$ with $f(0)=1$, and by uniqueness of solutions to linear ODEs that makes $f(\theta) = e^{j\theta}$.

Read $f' = jf$ geometrically. Multiplying by $j$ rotates by a quarter turn, so the statement is: *the velocity is always perpendicular to the position, and of the same length*. Perpendicular velocity means the speed $\lvert f \rvert$ never changes — you stay on a circle — and equal length means you traverse it at unit angular rate. Uniform circular motion is not a consequence of Euler's formula; it is Euler's formula.

This is also the cleanest bridge to the [ODEs tab](/interview-prep/#ode): the scalar equation $f' = \lambda f$ with $\lambda = \sigma + j\omega$ has solution $e^{\sigma t}e^{j\omega t}$, a spiral whose real part is a decaying or growing sinusoid. Every second-order linear ODE, every $RLC$ circuit, and every two-pole filter is that statement with $\lambda$ read off from a characteristic polynomial.

### 3. The functional equation

Ask which function $g$ satisfies $g(\alpha)g(\beta) = g(\alpha + \beta)$ with $g(\theta) = \cos\theta + j\sin\theta$. Multiplying out,

$$
(\cos\alpha + j\sin\alpha)(\cos\beta + j\sin\beta)
= \underbrace{(\cos\alpha\cos\beta - \sin\alpha\sin\beta)}_{\cos(\alpha+\beta)}
+ j\underbrace{(\sin\alpha\cos\beta + \cos\alpha\sin\beta)}_{\sin(\alpha+\beta)} ,
$$

so $g(\alpha)g(\beta) = g(\alpha+\beta)$ *is* the pair of angle-addition identities, and the only continuous functions with that property are exponentials. Worth internalising in the other direction too: if you ever forget the addition formulas, expand $e^{j(\alpha+\beta)} = e^{j\alpha}e^{j\beta}$ and read off real and imaginary parts. Same for the double-angle formulas, via $e^{j2\theta} = (e^{j\theta})^2$.

Three immediate consequences we will use without comment:

$$
\lvert e^{j\theta}\rvert = 1, \qquad
\overline{re^{j\theta}} = re^{-j\theta}, \qquad
\cos\theta = \frac{e^{j\theta} + e^{-j\theta}}{2},\quad
\sin\theta = \frac{e^{j\theta} - e^{-j\theta}}{2j} .
$$

The first says $e^{j\theta}$ is pure phase, so $r$ and $\theta$ never contaminate each other. The last pair — the inverse Euler formulas — are how you convert any trigonometric identity into an algebra problem about exponentials, and they are the reason a real cosine has *two* spectral lines, at $+\omega_0$ and $-\omega_0$.

## What the polar chart actually buys you

### Multiplication is rotation and scaling

$$
z_1 z_2 = r_1 r_2 e^{j(\theta_1 + \theta_2)} .
$$

Multiplying by $w$ means: *scale by $\lvert w \rvert$, rotate by $\angle w$*. So $j$ is the quarter-turn operator ($j^2 = -1$ is then just "two quarter turns is a half turn"), $-1$ is the half-turn, and any unit-modulus number is a pure rotation. Drag the two factors below.

<div class="viz" data-viz="argand-multiply">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; multiplication adds the angles</p>
    <p class="viz-sub">Drag either factor. The orange product is z stretched by the modulus of w and rotated by the argument of w. Try setting w to j and watch the product make a quarter turn.</p>
  </div>
</div>

Powers follow immediately (de Moivre): $z^n = r^n e^{jn\theta}$. So $(1+j)^8 = (\sqrt2)^8 e^{j8\pi/4} = 16e^{j2\pi} = 16$, which you would not want to expand by binomial theorem.

Roots follow too, and here the multivaluedness of $\arg$ becomes a feature rather than a nuisance. Since $z = re^{j(\theta + 2\pi k)}$ for every integer $k$,

$$
z^{1/n} = r^{1/n}\,e^{j(\theta + 2\pi k)/n}, \qquad k = 0, 1, \dots, n-1,
$$

giving $n$ distinct roots evenly spaced around a circle of radius $r^{1/n}$. Taking $z = 1$: the $n$-th roots of unity, $e^{j2\pi k/n}$.

### Twiddle factors, which is to say the DFT

The DFT is

$$
X[k] = \sum_{n=0}^{N-1} x[n]\, W_N^{nk},
\qquad W_N = e^{-j2\pi/N} ,
$$

and $W_N$ is exactly an $N$-th root of unity: one $N$-th of a turn, clockwise. Every entry of the DFT matrix is a rotation by a rational fraction of a full turn, so the whole transform is "correlate the signal against rotations at every rate". The FFT is then the observation that these rotations repeat — $W_N^{nk}$ depends only on $nk \bmod N$ — so most of the products have already been computed. Stated in rectangular coordinates none of that is visible.

### Phasors: the two numbers a sinusoid carries

A real sinusoid at a known frequency carries exactly two pieces of information, amplitude and phase. Package them:

$$
A\cos(\omega_0 t + \phi) = \Re\big\{ \underbrace{A e^{j\phi}}_{\text{phasor}} \; e^{j\omega_0 t} \big\}.
$$

The phasor $Ae^{j\phi}$ is a *constant* complex number whose polar form is (amplitude, phase). The time dependence lives entirely in $e^{j\omega_0 t}$, which is the same for every signal at that frequency and therefore carries no information about *this* one.

<div class="viz" data-viz="phasor">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; a phasor and the cosine it casts</p>
    <p class="viz-sub">The vector rotates at rate omega; the trace on the right is its real part over time, newest sample at the left. Change the amplitude and the starting phase and watch the waveform follow.</p>
  </div>
</div>

This is what makes the standard manipulation work. To add two sinusoids of the same frequency,

$$
A_1\cos(\omega_0 t + \phi_1) + A_2\cos(\omega_0 t + \phi_2)
= \Re\big\{(A_1e^{j\phi_1} + A_2e^{j\phi_2})e^{j\omega_0 t}\big\} ,
$$

you convert both phasors to rectangular, add — because addition is easy there — and convert the sum back to polar to read off the resulting amplitude and phase. That round trip, polar to rectangular to polar, is the loop this question is drilling, and it is why you need both directions to be automatic rather than just one.

### Frequency response: $r$ and $\theta$, once per frequency

Complex exponentials are the eigenfunctions of LTI systems. Push $x[n] = e^{j\omega n}$ through a system with impulse response $h$:

$$
y[n] = \sum_{k} h[k]\,e^{j\omega(n-k)}
     = e^{j\omega n}\underbrace{\sum_k h[k] e^{-j\omega k}}_{H(e^{j\omega})} .
$$

The input comes out unchanged apart from a complex scale factor, so $e^{j\omega n}$ is an eigenfunction and $H(e^{j\omega})$ is its eigenvalue. Now write that eigenvalue in polar form:

$$
H(e^{j\omega}) = \lvert H(e^{j\omega})\rvert \; e^{j\angle H(e^{j\omega})} .
$$

The modulus scales the amplitude; the argument shifts the phase. **The magnitude response and the phase response are the polar coordinates of the eigenvalue** — that is the entire content of a Bode plot. For a real impulse response there is a bonus: $\overline{H(e^{j\omega})} = H(e^{-j\omega})$, so the magnitude is even in $\omega$ and the phase is odd, which is why spectra are only ever plotted over $[0, \pi]$.

### Pole-zero geometry: distances and angles you can read off a picture

Factor a rational system function into its zeros $z_k$ and poles $p_k$. Then on the unit circle,

$$
\lvert H(e^{j\omega}) \rvert = \lvert G \rvert\,\frac{\prod_k \lvert e^{j\omega} - z_k \rvert}{\prod_k \lvert e^{j\omega} - p_k \rvert},
\qquad
\angle H(e^{j\omega}) = \angle G + \sum_k \angle(e^{j\omega} - z_k) - \sum_k \angle(e^{j\omega} - p_k) .
$$

Magnitude is a product of *distances*; phase is a sum of *angles*. Both statements are just "moduli multiply, arguments add" applied to a factored polynomial, and together they let you sketch a frequency response by eye: the response dips where the unit circle passes near a zero and peaks where it passes near a pole.

Part 3 of the problem is the smallest possible example. $H(z) = 1 - 0.8z^{-1} = (z - 0.8)/z$ has a zero at $z = 0.8$ and a pole at the origin, so

$$
\lvert H(e^{j\omega})\rvert = \frac{\lvert e^{j\omega} - 0.8\rvert}{\lvert e^{j\omega}\rvert} = \lvert e^{j\omega} - 0.8 \rvert ,
$$

the distance from the point $e^{j\omega}$ on the unit circle to the zero. At $\omega = \pi/2$ that is $\lvert j - 0.8 \rvert = \sqrt{0.64 + 1} = \sqrt{1.64}$, matching the answer above. The phase check works the same way: $\angle H = \angle(j - 0.8) - \pi/2$, and $-0.8 + j$ is in quadrant II with reference angle $\arctan(1/0.8) = 51.34^\circ$, so $\angle H = (180^\circ - 51.34^\circ) - 90^\circ = 38.66^\circ$. Same number, obtained by measuring a picture.

## One helix, two shadows

Here is the geometric object all of this is really about. Plot $e^{j\omega t}$ with time as a third axis: the curve $t \mapsto (t,\ \cos\omega t,\ \sin\omega t)$ is a helix. Its shadow on the time–real wall is a cosine, its shadow on the time–imaginary wall is a sine, and looking straight down the time axis you see the unit circle.

<div class="viz" data-viz="helix">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; the complex exponential in 3D</p>
    <p class="viz-sub">Drag to orbit, or use the view buttons to drop into one of the three canonical projections. Nothing about the curve changes between views - only where you are standing.</p>
  </div>
</div>

Cosine and sine are not two functions that happen to be related; they are one rotation, projected two ways. The rectangular coordinates $(a,b)$ are the two shadows, the polar coordinates $(r,\theta)$ are the radius and the phase along the helix, and the reason polar wins in DSP is that a linear system does something simple to the helix (scale it, advance it) and something ugly to each shadow separately.

## Extensions and traps

### The principal value and the branch cut

$\arg$ is genuinely multivalued, so pinning it down needs a convention:

$$
\arg z = \operatorname{Arg} z + 2\pi k, \qquad \operatorname{Arg} z \in (-\pi, \pi] .
$$

$\operatorname{Arg}$ is continuous everywhere except along the negative real axis, where it jumps by $2\pi$: approach $-3$ from above and you get $+\pi$, from below and you get $-\pi$. That discontinuity is the **branch cut**, and it is not removable — it is forced by the fact that a loop around the origin comes back having accumulated $2\pi$. You can move the cut (the $[0, 2\pi)$ convention puts it on the positive real axis) but you cannot delete it. The origin, where the argument does not exist at all, is the **branch point**.

### Phase unwrapping and group delay

This is where the branch cut stops being a curiosity and starts costing you. Compute a phase response numerically and you get $\operatorname{Arg} H(e^{j\omega})$, which is wrapped: it saw-tooths back by $2\pi$ whenever the true phase leaves the principal interval. Those jumps are artefacts of the convention, not of the filter.

They matter because group delay is a derivative:

$$
\tau_g(\omega) = -\frac{d}{d\omega}\angle H(e^{j\omega}) .
$$

Differentiating the wrapped phase puts a spike of $-2\pi\delta$ at every wrap point, and your group delay plot is garbage. The fix is to unwrap first — walk along in $\omega$ and add or subtract $2\pi$ whenever consecutive samples differ by more than $\pi$, which is precisely what `np.unwrap` does. It also fails, quietly, if the phase really does change by more than $\pi$ between samples, so unwrapping demands a fine enough frequency grid.

### Conditioning: why the phase of a spectral null is meaningless

Take gradients of the two polar coordinates with respect to the rectangular ones:

$$
\nabla r = \Big(\frac{a}{r},\ \frac{b}{r}\Big), \quad \lVert \nabla r\rVert = 1 ;
\qquad
\nabla \theta = \Big(\frac{-b}{r^2},\ \frac{a}{r^2}\Big), \quad \lVert \nabla \theta\rVert = \frac{1}{r} .
$$

The modulus is perfectly conditioned: perturb $z$ by $\varepsilon$ and $r$ moves by at most $\varepsilon$. The argument is conditioned like $1/r$: the *same* perturbation swings $\theta$ by up to $\varepsilon/r$, which blows up as $z \to 0$.

Concretely: a DFT bin sitting in a deep spectral null has tiny $\lvert X[k] \rvert$, so its phase is dominated by rounding and leakage. Plotting it looks like noise because it *is* noise. This is also why `atan2(0.0, 0.0) == 0.0` is a convenience rather than an answer, and why phase-sensitive algorithms (phase vocoders, GCC-PHAT, phase-based pitch tracking) all weight or gate by magnitude instead of trusting phase uniformly.

### Compute $r$ with `hypot`, not with `sqrt`

$\sqrt{a^2+b^2}$ overflows whenever $a^2$ does, even when the answer is perfectly representable. In IEEE double precision the largest finite value is about $1.8\times10^{308}$, so:

```python
import math
a = 1e200
math.sqrt(a*a + a*a)   # inf     -- a*a already overflowed
math.hypot(a, a)       # 1.4142135623730951e+200
```

The same applies at the bottom of the range, where squaring flushes small values to zero. `math.hypot`, `np.hypot` and `np.abs` on complex arrays all use a scaled algorithm and do the right thing. Free correctness; use them.

### The complex logarithm is the polar form in disguise

$$
\log z = \ln r + j(\theta + 2\pi k) .
$$

Converting to polar *is* taking a logarithm: the real part is the log-magnitude, the imaginary part is the phase, and the $2\pi k$ is the same branch ambiguity as before. This is why the complex cepstrum — the inverse transform of $\log H(e^{j\omega})$ — needs a properly unwrapped phase to exist, and why log-magnitude and phase are coupled through the Hilbert transform for minimum-phase systems. Stack a `20*log10` on the magnitude and you have the two panels of every Bode plot, which are the real and imaginary parts of one complex logarithm.

## Doing it in code

```python
import numpy as np

z = np.array([1 + 1j, -1 + np.sqrt(3) * 1j, -3 + 0j, -2j, 3 - 4j])

r = np.abs(z)                    # modulus, computed stably (hypot inside)
th = np.angle(z)                 # principal argument in (-pi, pi], i.e. atan2(b, a)
deg = np.angle(z, deg=True)      # same thing in degrees

# Round trip back to rectangular; Euler's formula is the whole implementation.
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

And the two DSP one-liners that this question is really about:

```python
from scipy import signal

w, H = signal.freqz([1.0, -0.8], [1.0], worN=1024)   # H(e^{jw}) of 1 - 0.8 z^-1

mag = np.abs(H)                       # r, per frequency
phase = np.unwrap(np.angle(H))        # theta, per frequency, wraps removed
tau_g = -np.gradient(phase, w)        # group delay needs the unwrapped phase

k = np.argmin(np.abs(w - np.pi / 2))
print(mag[k], np.rad2deg(phase[k]))   # ~1.2806  ~38.66  -- Part 3, numerically
```

## Drills

<details class="q-reveal">
<summary>Drill 1 &mdash; convert $-5 - 5j$ to polar</summary>
<div class="q-reveal-body" markdown="1">

$r = \sqrt{25 + 25} = 5\sqrt2 \approx 7.0711$. Both parts negative, so quadrant III; the reference angle is $\arctan(5/5) = \pi/4$, and the correction is $-\pi$:

$$
\theta = \tfrac{\pi}{4} - \pi = -\tfrac{3\pi}{4} = -135^\circ,
\qquad z = 5\sqrt2\,e^{-j3\pi/4}.
$$

Note this is $\sqrt{2}\cdot 5 \cdot e^{-j3\pi/4}$ — the same angle as $z_6$ from the problem, which is no accident: $z_6 = 5e^{-j3\pi/4}$ and $-5-5j$ point the same way.

</div>
</details>

<details class="q-reveal">
<summary>Drill 2 &mdash; simplify $\dfrac{1+j}{1-j}$ without expanding</summary>
<div class="q-reveal-body" markdown="1">

Convert both, then divide — moduli divide, arguments subtract:

$$
\frac{1+j}{1-j} = \frac{\sqrt2\,e^{j\pi/4}}{\sqrt2\,e^{-j\pi/4}} = e^{j\pi/2} = j .
$$

The rectangular route (multiply top and bottom by the conjugate) gets the same answer in three times the writing. If you are ever dividing complex numbers repeatedly, convert once and stay in polar.

</div>
</details>

<details class="q-reveal">
<summary>Drill 3 &mdash; solve $z^3 = -8$</summary>
<div class="q-reveal-body" markdown="1">

Put the right-hand side in polar including the ambiguity: $-8 = 8e^{j(\pi + 2\pi k)}$. Then

$$
z = 8^{1/3} e^{j(\pi + 2\pi k)/3} = 2e^{j(\pi + 2\pi k)/3}, \qquad k = 0, 1, 2,
$$

giving $\theta = \pi/3,\ \pi,\ 5\pi/3 \equiv -\pi/3$, so

$$
z \in \{\,1 + j\sqrt3,\ \ -2,\ \ 1 - j\sqrt3\,\} ,
$$

three points evenly spaced by $120^\circ$ on the circle of radius 2. Only $-2$ is real, and the other two are conjugates — which they must be, since the polynomial has real coefficients. Forgetting the $2\pi k$ is how people end up reporting one root out of three.

</div>
</details>

<details class="q-reveal">
<summary>Drill 4 &mdash; the two-tap moving average, in closed form</summary>
<div class="q-reveal-body" markdown="1">

Find $\lvert H \rvert$ and $\angle H$ for $H(e^{j\omega}) = 1 + e^{-j\omega}$ over $\lvert\omega\rvert < \pi$.

Do not expand into real and imaginary parts. Pull out *half* the total phase so that what remains is symmetric:

$$
1 + e^{-j\omega}
= e^{-j\omega/2}\big(e^{j\omega/2} + e^{-j\omega/2}\big)
= 2\cos\!\Big(\frac{\omega}{2}\Big)\, e^{-j\omega/2} .
$$

Since $\cos(\omega/2) \ge 0$ on $\lvert\omega\rvert<\pi$, that is already the polar form:

$$
\lvert H(e^{j\omega})\rvert = 2\cos\!\Big(\frac{\omega}{2}\Big),
\qquad
\angle H(e^{j\omega}) = -\frac{\omega}{2},
\qquad
\tau_g = -\frac{d}{d\omega}\Big(-\frac{\omega}{2}\Big) = \frac{1}{2}\ \text{sample}.
$$

Phase linear in $\omega$ means a constant group delay, here half a sample — exactly the delay you would expect from averaging two neighbouring samples. That "factor out the mid-point phase" move is the standard proof that a symmetric FIR filter has exactly linear phase, and it works because the leftover sum is $e^{j\alpha} + e^{-j\alpha}$, which is real. Watch the boundary: past $\lvert\omega\rvert = \pi$, $\cos(\omega/2)$ turns negative, and since $r$ must stay non-negative that minus sign has to be absorbed as an extra $\pi$ in the phase. Those are the $\pi$ jumps you see in the phase response of any linear-phase filter at its nulls, and they are real, not a wrapping artefact.

</div>
</details>

## Follow-ups this question tends to grow into

- *"Your phase response has jumps of $2\pi$ in it. Where did they come from, and does it matter?"* — Principal value; matters as soon as you differentiate.
- *"Why is the phase near a null so noisy?"* — $\lVert\nabla\theta\rVert = 1/r$.
- *"Prove $e^{j\omega n}$ is an eigenfunction of an LTI system."* — Three lines of convolution, above.
- *"What does multiplying a signal by $e^{j\omega_0 n}$ do to its spectrum?"* — Shifts it by $\omega_0$: modulation, and the reason every mixer is a multiplication.
- *"Give me the magnitude and phase of an $N$-tap moving average in closed form."* — Same mid-point-phase trick as Drill 4, ending at the Dirichlet kernel $\frac{\sin(N\omega/2)}{N\sin(\omega/2)}e^{-j\omega(N-1)/2}$.
- *"When is $\lvert z_1 + z_2\rvert = \lvert z_1\rvert + \lvert z_2\rvert$?"* — Exactly when the arguments agree; the triangle inequality in polar clothing.
