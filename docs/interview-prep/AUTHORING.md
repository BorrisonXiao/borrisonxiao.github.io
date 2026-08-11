# Authoring guide — `/interview-prep/`

This covers the interview-prep sub-site only. It is kept separate from the rest
of the homepage on purpose. `docs/` is in the Jekyll `exclude` list, so nothing
here is published.

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
order: 2                      # sort order within the tab
title: "Human-readable title"
difficulty: warm-up           # warm-up | core | stretch  -> badge colour
date: 2026-08-11
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

## What a write-up should contain

The structure that has worked so far:

1. **Problem box**, then a nudge to try it before revealing anything.
2. **Revealable short answers**, plus the two or three ways people get it wrong.
3. **The theory**, derived rather than asserted. Multiple derivations of the
   same fact are worth the space when they illuminate different things.
4. **Why the field cares** — where this shows up in real work.
5. **Extensions and traps** — edge cases, conventions, numerical conditioning.
6. **Code** — the vectorized/library way to do it.
7. **Drills** with revealable solutions.
8. **Follow-ups** an interviewer would chain onto the question.

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
