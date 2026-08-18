# Authoring guide — `/interview-prep/`

This covers the interview-prep sub-site only. It is kept separate from the rest
of the homepage on purpose. `docs/` is in the Jekyll `exclude` list, so nothing
here is published.

**The sub-site is unlisted on purpose.** It is not in `_data/navigation.yml`,
its pages carry `noindex, nofollow`, and the only route in is a small line at the
top of `blog.html`. It shows up publicly as "Practice Problems"; the
`/interview-prep/` URL and the directory names are internal. Do not add it to the
nav.

## Layout of the sub-site

```
_interview/<topic-slug>/<question-slug>.md   one file per question
_data/interview_topics.yml                   the sub-tabs
interview.html                               landing page  ->  /interview-prep/
_layouts/interview_question.html             question page
_includes/widgets/interview_card.html        the card on the landing page
assets/css/interview.css                     styles for both pages
assets/js/interview.js                       tab deep-linking + filter box
assets/js/interview/<question-slug>.js       per-question visualizations
```

The collection is declared in `_config.yml`:

```yaml
interview:
  output: true
  permalink: /interview-prep/:path/
```

so `_interview/signal-processing/complex-rect-to-polar.md` is served at
`/interview-prep/signal-processing/complex-rect-to-polar/`.

## Adding a topic tab

Append an entry to `_data/interview_topics.yml` and create the matching
directory under `_interview/`. Nothing else needs touching — the tab, its
question count, and its empty state are all generated.

```yaml
- slug: numerical-methods        # must equal the directory name AND the
  name: Numerical Methods        #   `topic:` field in each question
  short: Numerics                # shown instead of `name` on narrow screens
  icon: fas fa-calculator        # Font Awesome 6 free-solid
  blurb: >-
    One or two sentences on what the tab covers.
  planned:                       # optional; shown while the tab is empty
    - Something you intend to write up
```

Tab order on the page is the order of this list. Tab state is deep-linked as a
bare slug fragment, e.g. `/interview-prep/#numerical-methods`.

## Adding a question

Create `_interview/<topic-slug>/<question-slug>.md`:

```yaml
---
layout: interview_question
topic: signal-processing      # must match a slug in interview_topics.yml
order: 2                      # position in the topic's prev/next chain
title: "Human-readable title"
difficulty: warm-up           # warm-up | core | stretch  -> badge colour
date: 2026-08-11 12:00:00     # REQUIRED, and must be unique across all questions
tags: [sampling, aliasing]
summary: >-
  One paragraph, plain text (no $math$ — it is reused as the card blurb and
  as the page's meta description).
scripts:                      # optional, per-question visualizations
  - /assets/js/interview/sampling.js
concepts:                     # renders the "Core concepts" panel
  - name: Nyquist-Shannon sampling theorem
    url: https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem
    note: One or two sentences on what this concept contributes here.
references:                   # renders the "Where this lives in the texts" panel
  - text: "Oppenheim & Schafer, *Discrete-Time Signal Processing* — Ch. 4."
    url:                      # optional
---
```

`concepts` and `references` are not decoration: linking each answer back to the
concept it rests on is the point of the page. Fill them in.

### Two ordering fields, and why both exist

- **`date` orders the listings.** Every list on `/interview-prep/` is newest
  first, and the Newest/Oldest control just reverses that. Give each question a
  **distinct timestamp**, not a bare date — Liquid's `sort` is not stable, so
  two questions sharing a date would come out in an arbitrary order. Use a
  midday hour (`12:00:00` and up) so the rendered calendar date cannot slip
  across midnight when the build host's timezone differs from the author's.
- **`order` drives prev/next** within a topic, and should follow the sequence
  you would actually learn them in — which is often not the order they were
  written. It no longer affects any listing.

## What a write-up should contain

**The solution is hidden by default.** Only the problem statement is visible when
the page loads; everything else sits behind one "Show the solution" toggle, so
the page works as practice rather than as reading. The concepts and references
panels are gated on the same toggle automatically (they carry `data-q-gated` in
the layout, and `assets/js/interview_question.js` keeps them in sync).

A question page opens with an equally-collapsed **example**: one very simple
instance of the same task, worked end to end, with the definitions someone would
need if they are cold. It comes *before* the problem, so the page reads
learn → attempt → check.

So the body of a question file is:

```
[one orienting sentence]

<details class="q-example" id="example">     ← id="example" is required
<summary class="q-example-summary"><span class="q-when-closed">Show a worked example first</span><span class="q-when-open">Hide the example</span></summary>
<div class="q-example-body" markdown="1">

### What a <thing> is
  … the definitions, algebraic and geometric. Short. ~600 words for the
    whole example section is about right; go longer only when the problem
    genuinely needs two ideas up front (the eigenvalue/SVD question needs
    both objects and the relation between them, and runs to about 1000).
### Worked example: <the simplest case>
  … every step, including the check at the end, then the visualization that
    explains the concept …

</div>
</details>

<div class="q-problem"> … the question … </div>
Work them on paper first — that is what the page is for.

<details class="q-solution" id="solution">   ← id="solution" is required
<summary class="q-solution-summary"><span class="q-when-closed">Show the solution</span><span class="q-when-open">Hide the solution</span></summary>
<div class="q-solution-body" markdown="1">

  … the full solution …

</div>
</details>

## More practice
… drills, each in its own <details class="q-reveal"> …
```

Drills stay outside the gate — they are practice too — with their own reveals.

Use `###` inside the example and `##` inside the solution, so the table of
contents nests sensibly when both are open.

Inside the solution, the structure that has worked:

1. **The answers** up front, plus the two or three ways people get it wrong.
2. **The theory**, derived rather than asserted.
3. **Why the field cares** — where this shows up in real work.
4. **The traps** — edge cases and conventions worth naming.
5. **Code** — the vectorized/library way to do it.

Do not repeat the example inside the solution. Where they overlap — the easy
case in the worked-solutions list, say — reduce it to a line and link back with
`[the example](#example)`. Links into a collapsed section auto-expand it.

Keep it to the basics. Depth is good; breadth into advanced side-topics makes
the page unusable as a refresher. If something is a tangent, cut it or leave it
as a one-line pointer.

## Markup available in the body

Content is kramdown. Math is KaTeX, auto-rendered: `$inline$` and `$$display$$`,
same as the blog. Code fences are highlighted by Rouge at build time.

Raw HTML blocks need `markdown="1"` for their contents to be parsed as Markdown.
Nested `markdown="1"` works, which is what the reveal blocks rely on.

```html
<div class="q-problem" markdown="1">
<div class="q-problem-label">Problem</div>

Statement goes here.

</div>
```

```html
<div class="q-callout q-callout-trap" markdown="1">
<div class="q-callout-title"><i class="fas fa-triangle-exclamation"></i> The trap</div>

Variants: `q-callout-trap`, `q-callout-insight`, `q-callout-note`.

</div>
```

```html
<details class="q-reveal">
<summary>Reveal the answers</summary>
<div class="q-reveal-body" markdown="1">

Hidden until clicked. Math in the `<summary>` works — KaTeX runs over the whole
document — but Markdown in the `<summary>` does not, so write it as HTML.

</div>
</details>
```

Two gotchas:

- A `|` inside math inside a **table** is read as a cell separator. Use
  `\lvert … \rvert` instead of `|…|` in tables.
- Keep big math out of **headings**. The table of contents is built from the
  heading's text, so a `\begin{bmatrix}` in an `###` shows up there as raw
  LaTeX. Short inline math like `$e^{j\theta}$` is fine; a matrix is not.
- Headings start at `##`; the page title is already an `<h1>`. The right-hand
  table of contents is built from `h1`–`h3` by `assets/js/blog.js`.

## Visualizations

Per-question widgets live in `assets/js/interview/<question-slug>.js` and are
pulled in with the `scripts:` front-matter key. They are plain canvas + vanilla
JS with **no external dependencies** — the site loads no plotting library, and
GitHub Pages builds with no plugins, so keep it that way.

The convention: the Markdown supplies the container and its caption, the script
fills in the body.

```html
<div class="viz" data-viz="argand-explorer">
  <div class="viz-head">
    <p class="viz-title">Interactive &middot; the Argand plane</p>
    <p class="viz-sub">One or two sentences telling the reader what to do.</p>
  </div>
</div>
```

`interview.css` provides the widget furniture: `.viz-body`, `.viz-canvas-wrap`,
`.viz-controls`, `.viz-btn`, `.viz-check`, `.viz-slider`, `.viz-readout`,
`.viz-field`, `.viz-warn`, `.viz-note`, `.viz-legend`. `complex-polar.js` has
reusable helpers for hi-DPI canvas setup, pointer/keyboard dragging, an Argand
plot with grid and angle wedges, and a 3D line projector — copy from it.

House rules for widgets:

- Respect `prefers-reduced-motion`: do not auto-play animations.
- Pause animation when the widget scrolls out of view (`IntersectionObserver`).
- Keep readouts as plain Unicode text (`θ`, `∠`, `π`), not `$math$` — the
  layout tells KaTeX to skip `.viz` subtrees, since their content is dynamic.
- Canvases get `tabindex` and arrow-key handling where dragging is the primary
  interaction.

## Building and checking locally

The system ruby cannot build this site. Use the project's conda env:

```bash
export PATH=/exp/cxiao/homepage/envs/jekyll/bin:$PATH
bundle exec jekyll build          # or `jekyll serve` for a live preview
```

Widgets can be smoke-tested without a browser — `node --check` for syntax, and a
minimal DOM/canvas stub is enough to run construct → resize → draw → interact and
catch runtime errors before pushing.

## Deployment

Push to `master`. GitHub Pages rebuilds the site itself (classic build, no
Actions workflow), which is why `plugins:` in `_config.yml` must stay empty.
