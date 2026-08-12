/* ==========================================================================
 * Visualizations for the calculus questions (interview-prep / calculus).
 *
 * Four self-contained canvas widgets, no external dependencies:
 *
 *   data-viz="derivative-explorer"  a curve, a draggable point, its tangent,
 *                                   and a secant whose slope you can watch
 *                                   collapse onto the derivative as h -> 0
 *   data-viz="implicit-curve"       a curve defined by an equation rather than
 *                                   a formula, with the tangent that implicit
 *                                   differentiation predicts
 *   data-viz="riemann-sum"          rectangles converging on the area under a
 *                                   curve, by four different rules
 *   data-viz="partial-fractions"    a rational function drawn on top of the
 *                                   simple pieces it decomposes into
 *
 * Each widget is initialised from a container that already carries its own
 * .viz-head written in the Markdown source; this file appends the .viz-body.
 * ========================================================================== */
(function () {
    'use strict';

    var TAU = Math.PI * 2;

    var C = {
        bg: '#fdfdfe',
        grid: '#edf0f3',
        axis: '#98a2ad',
        tick: '#a9b2bb',
        text: '#6c757d',
        ink: '#212529',
        curve: '#0d6efd',
        tangent: '#e8590c',
        secant: '#0e9f6e',
        area: 'rgba(13, 110, 253, 0.16)',
        areaEdge: 'rgba(13, 110, 253, 0.55)',
        p1: '#d6336c',
        p2: '#7048e8',
        p3: '#0e9f6e'
    };

    var SANS = '"Lato", system-ui, sans-serif';

    /* ---------------------------------------------------------------- utils */

    function el(tag, cls, html) {
        var e = document.createElement(tag);
        if (cls) e.className = cls;
        if (html !== undefined && html !== null) e.innerHTML = html;
        return e;
    }

    function fmt(x, n) {
        if (n === undefined) n = 3;
        if (!isFinite(x)) return '--';
        var s = x.toFixed(n);
        if (/^-0(\.0*)?$/.test(s)) s = s.slice(1);
        return s;
    }

    function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

    /* A "nice" grid step: 1, 2 or 5 times a power of ten. */
    function niceStep(span, target) {
        var raw = span / Math.max(1, target);
        var mag = Math.pow(10, Math.floor(Math.log(raw) / Math.LN10));
        var norm = raw / mag;
        var mult = norm <= 1 ? 1 : (norm <= 2 ? 2 : (norm <= 5 ? 5 : 10));
        return mult * mag;
    }

    function tickLabel(v, step) {
        var dec = step >= 1 ? 0 : Math.min(3, Math.ceil(-Math.log(step) / Math.LN10));
        var s = v.toFixed(dec);
        if (/^-0(\.0*)?$/.test(s)) s = s.slice(1);
        return s;
    }

    /* --------------------------------------------------------- canvas setup */

    var widgets = [];

    function fitCanvas(canvas, cssW, cssH) {
        var dpr = window.devicePixelRatio || 1;
        canvas.style.width = '100%';
        canvas.style.height = cssH + 'px';
        canvas.width = Math.max(1, Math.round(cssW * dpr));
        canvas.height = Math.max(1, Math.round(cssH * dpr));
        var ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return ctx;
    }

    function measure(canvas) {
        var p = canvas.parentNode;
        return (p && p.clientWidth) || 600;
    }

    function debounce(fn, ms) {
        var t = null;
        return function () { clearTimeout(t); t = setTimeout(fn, ms); };
    }

    window.addEventListener('resize', debounce(function () {
        widgets.forEach(function (wd) { wd.resize(); wd.draw(); });
    }, 120));

    /* --------------------------------------------------------- UI factories */

    function button(label, onClick) {
        var b = el('button', 'viz-btn', label);
        b.type = 'button';
        b.addEventListener('click', function () { onClick(b); });
        return b;
    }

    function buttonGroup(parent, labelText, items, initial, onPick) {
        if (labelText) parent.appendChild(el('span', 'viz-label', labelText));
        var btns = items.map(function (it, i) {
            var b = button(it.label, function () {
                btns.forEach(function (o) { o.classList.remove('is-active'); });
                b.classList.add('is-active');
                onPick(it, i);
            });
            if (i === initial) b.classList.add('is-active');
            parent.appendChild(b);
            return b;
        });
        return btns;
    }

    function checkbox(label, checked, onChange) {
        var wrapEl = el('label', 'viz-check');
        var input = document.createElement('input');
        input.type = 'checkbox';
        input.checked = !!checked;
        input.addEventListener('change', function () { onChange(input.checked); });
        wrapEl.appendChild(input);
        wrapEl.appendChild(document.createTextNode(label));
        return { node: wrapEl, input: input };
    }

    function slider(label, min, max, step, value, format, onInput) {
        var wrapEl = el('span', 'viz-slider');
        wrapEl.appendChild(el('span', null, label));
        var input = document.createElement('input');
        input.type = 'range';
        input.min = min; input.max = max; input.step = step; input.value = value;
        var out = document.createElement('output');
        function sync() { out.textContent = format(parseFloat(input.value)); }
        input.addEventListener('input', function () { sync(); onInput(parseFloat(input.value)); });
        sync();
        wrapEl.appendChild(input);
        wrapEl.appendChild(out);
        return { node: wrapEl, input: input, sync: sync };
    }

    function readoutField(parent, key) {
        var f = el('div', 'viz-field');
        f.appendChild(el('span', 'viz-field-key', key));
        var v = el('span', 'viz-field-val', '--');
        f.appendChild(v);
        parent.appendChild(f);
        return v;
    }

    function legend(items) {
        var l = el('div', 'viz-legend');
        items.forEach(function (it) {
            var s = el('span');
            var sw = el('span', 'viz-swatch');
            sw.style.background = it[0];
            s.appendChild(sw);
            s.appendChild(document.createTextNode(it[1]));
            l.appendChild(s);
        });
        return l;
    }

    function canvasIn(body, label, maxWidth) {
        var wrapEl = el('div', 'viz-canvas-wrap');
        if (maxWidth) { wrapEl.style.maxWidth = maxWidth; wrapEl.style.margin = '0 auto'; }
        var canvas = document.createElement('canvas');
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label', label);
        wrapEl.appendChild(canvas);
        body.appendChild(wrapEl);
        return canvas;
    }

    /* ------------------------------------------------------- the 2D plotter */

    function Plot2D(canvas, opts) {
        opts = opts || {};
        this.canvas = canvas;
        this.ratio = opts.ratio || 0.6;
        this.maxH = opts.maxH || 360;
        this.minH = opts.minH || 240;
        this.pad = { l: 40, r: 16, t: 14, b: 28 };
        this.setRange(-3, 3, -2, 2);
    }

    Plot2D.prototype.setRange = function (x0, x1, y0, y1) {
        this.x0 = x0; this.x1 = x1; this.y0 = y0; this.y1 = y1;
    };

    Plot2D.prototype.resize = function () {
        var w = measure(this.canvas);
        var h = Math.round(clamp(w * this.ratio, this.minH, this.maxH));
        this.ctx = fitCanvas(this.canvas, w, h);
        this.w = w; this.h = h;
        this.pw = w - this.pad.l - this.pad.r;
        this.ph = h - this.pad.t - this.pad.b;
    };

    Plot2D.prototype.X = function (x) {
        return this.pad.l + (x - this.x0) / (this.x1 - this.x0) * this.pw;
    };
    Plot2D.prototype.Y = function (y) {
        return this.pad.t + (1 - (y - this.y0) / (this.y1 - this.y0)) * this.ph;
    };
    Plot2D.prototype.invX = function (px) {
        return this.x0 + (px - this.pad.l) / this.pw * (this.x1 - this.x0);
    };
    Plot2D.prototype.invY = function (py) {
        return this.y0 + (1 - (py - this.pad.t) / this.ph) * (this.y1 - this.y0);
    };

    Plot2D.prototype.frame = function (xlabel, ylabel) {
        var ctx = this.ctx, i, v;
        ctx.clearRect(0, 0, this.w, this.h);
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, this.w, this.h);

        var sx = niceStep(this.x1 - this.x0, 7);
        var sy = niceStep(this.y1 - this.y0, 5);

        ctx.lineWidth = 1;
        ctx.strokeStyle = C.grid;
        ctx.beginPath();
        for (v = Math.ceil(this.x0 / sx) * sx; v <= this.x1 + 1e-9; v += sx) {
            var px = Math.round(this.X(v)) + 0.5;
            ctx.moveTo(px, this.pad.t); ctx.lineTo(px, this.pad.t + this.ph);
        }
        for (v = Math.ceil(this.y0 / sy) * sy; v <= this.y1 + 1e-9; v += sy) {
            var py = Math.round(this.Y(v)) + 0.5;
            ctx.moveTo(this.pad.l, py); ctx.lineTo(this.pad.l + this.pw, py);
        }
        ctx.stroke();

        /* the axes themselves, drawn only where they fall inside the window */
        ctx.strokeStyle = C.axis;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        if (this.y0 <= 0 && this.y1 >= 0) {
            var y0px = Math.round(this.Y(0)) + 0.5;
            ctx.moveTo(this.pad.l, y0px); ctx.lineTo(this.pad.l + this.pw, y0px);
        }
        if (this.x0 <= 0 && this.x1 >= 0) {
            var x0px = Math.round(this.X(0)) + 0.5;
            ctx.moveTo(x0px, this.pad.t); ctx.lineTo(x0px, this.pad.t + this.ph);
        }
        ctx.stroke();

        ctx.fillStyle = C.tick;
        ctx.font = '10px ' + SANS;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (v = Math.ceil(this.x0 / sx) * sx; v <= this.x1 + 1e-9; v += sx) {
            if (Math.abs(v) < 1e-9) continue;
            ctx.fillText(tickLabel(v, sx), this.X(v), this.pad.t + this.ph + 5);
        }
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (v = Math.ceil(this.y0 / sy) * sy; v <= this.y1 + 1e-9; v += sy) {
            if (Math.abs(v) < 1e-9) continue;
            ctx.fillText(tickLabel(v, sy), this.pad.l - 5, this.Y(v));
        }

        ctx.fillStyle = C.text;
        ctx.font = 'italic 11px ' + SANS;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'bottom';
        ctx.fillText(xlabel || 'x', this.pad.l + this.pw, this.pad.t + this.ph + 22);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText(ylabel || 'y', this.pad.l + 4, this.pad.t - 2);
    };

    Plot2D.prototype.clip = function (fn) {
        var ctx = this.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.rect(this.pad.l, this.pad.t, this.pw, this.ph);
        ctx.clip();
        fn();
        ctx.restore();
    };

    /* Sample a function across the plot, breaking the path at poles and at
     * points outside the window so asymptotes do not get joined up. */
    Plot2D.prototype.plot = function (f, color, width, dash) {
        var ctx = this.ctx, self = this;
        var span = this.y1 - this.y0;
        this.clip(function () {
            ctx.save();
            if (dash) ctx.setLineDash(dash);
            ctx.strokeStyle = color;
            ctx.lineWidth = width || 2;
            ctx.beginPath();
            var pen = false, prevY = null;
            for (var px = 0; px <= self.pw; px++) {
                var x = self.invX(self.pad.l + px);
                var y = f(x);
                var ok = isFinite(y) && y > self.y0 - span && y < self.y1 + span;
                if (!ok) { pen = false; prevY = null; continue; }
                /* a jump larger than the whole window is a pole, not a curve */
                if (prevY !== null && Math.abs(y - prevY) > span * 1.5) pen = false;
                var sy = self.Y(y);
                if (!pen) { ctx.moveTo(self.pad.l + px, sy); pen = true; }
                else ctx.lineTo(self.pad.l + px, sy);
                prevY = y;
            }
            ctx.stroke();
            ctx.restore();
        });
    };

    Plot2D.prototype.polyline = function (pts, color, width, dash) {
        var ctx = this.ctx, self = this;
        this.clip(function () {
            ctx.save();
            if (dash) ctx.setLineDash(dash);
            ctx.strokeStyle = color;
            ctx.lineWidth = width || 2;
            ctx.beginPath();
            pts.forEach(function (p, i) {
                var sx = self.X(p[0]), sy = self.Y(p[1]);
                if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
            });
            ctx.stroke();
            ctx.restore();
        });
    };

    /* A straight line through (x0,y0) with the given slope, drawn edge to edge */
    Plot2D.prototype.lineAt = function (x0, y0, slope, color, width, dash) {
        this.polyline([
            [this.x0, y0 + slope * (this.x0 - x0)],
            [this.x1, y0 + slope * (this.x1 - x0)]
        ], color, width, dash);
    };

    Plot2D.prototype.dot = function (x, y, color, r) {
        var ctx = this.ctx, self = this;
        this.clip(function () {
            ctx.beginPath();
            ctx.arc(self.X(x), self.Y(y), r || 5.5, 0, TAU);
            ctx.fillStyle = color;
            ctx.fill();
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        });
    };

    Plot2D.prototype.vline = function (x, yA, yB, color, dash) {
        this.polyline([[x, yA], [x, yB]], color, 1, dash || [2, 3]);
    };

    /* ============================================================ widget 1 */

    var DERIV_FNS = [
        { label: 'x²', f: function (x) { return x * x; }, df: function (x) { return 2 * x; },
          x0: -3, x1: 3, y0: -1.5, y1: 6, start: 1, fs: 'f(x) = x²', dfs: "f'(x) = 2x" },
        { label: 'x³ - 3x', f: function (x) { return x * x * x - 3 * x; },
          df: function (x) { return 3 * x * x - 3; },
          x0: -2.6, x1: 2.6, y0: -4.5, y1: 4.5, start: 0.6,
          fs: 'f(x) = x³ - 3x', dfs: "f'(x) = 3x² - 3" },
        { label: 'sin x', f: Math.sin, df: Math.cos,
          x0: -Math.PI, x1: Math.PI, y0: -1.6, y1: 1.6, start: 0.7,
          fs: 'f(x) = sin x', dfs: "f'(x) = cos x" },
        { label: 'eˣ', f: Math.exp, df: Math.exp,
          x0: -2.6, x1: 1.8, y0: -0.8, y1: 6, start: 0.5,
          fs: 'f(x) = eˣ', dfs: "f'(x) = eˣ" },
        { label: 'ln x', f: Math.log, df: function (x) { return 1 / x; },
          x0: 0.05, x1: 5, y0: -2.6, y1: 2, start: 1.6,
          fs: 'f(x) = ln x', dfs: "f'(x) = 1/x" }
    ];

    function DerivativeExplorer(root) {
        var self = this;
        this.fi = 0;
        this.x = DERIV_FNS[0].start;
        this.h = 1.0;
        this.showSecant = true;

        var body = el('div', 'viz-body');
        var canvas = canvasIn(body,
            'Graph of a function with a draggable point, its tangent line, and a secant line.');
        this.canvas = canvas;
        this.plot = new Plot2D(canvas, { ratio: 0.58, maxH: 340 });

        var row1 = el('div', 'viz-controls');
        buttonGroup(row1, 'function', DERIV_FNS, 0, function (it, i) {
            self.fi = i;
            self.x = it.start;
            self.draw();
        });
        body.appendChild(row1);

        var row2 = el('div', 'viz-controls');
        this.hSlider = slider('h', 0.01, 1.5, 0.01, this.h,
            function (v) { return fmt(v, 2); },
            function (v) { self.h = v; self.draw(); });
        row2.appendChild(this.hSlider.node);
        row2.appendChild(checkbox('show the secant', true, function (v) {
            self.showSecant = v; self.draw();
        }).node);
        body.appendChild(row2);

        var ro = el('div', 'viz-readout');
        this.f = {
            at: readoutField(ro, 'point'),
            slope: readoutField(ro, "tangent slope f'(x)"),
            quot: readoutField(ro, 'difference quotient'),
            err: readoutField(ro, 'gap')
        };
        body.appendChild(ro);

        body.appendChild(legend([
            [C.curve, 'f'], [C.tangent, 'tangent, slope f′(x)'],
            [C.secant, 'secant through x and x + h']
        ]));
        body.appendChild(el('p', 'viz-note',
            'Drag the point along the curve, or click the canvas and use the ' +
            'arrow keys. Then pull h down towards zero and watch the green ' +
            'secant swing onto the orange tangent — that limit is the definition ' +
            'of the derivative.'));

        root.appendChild(body);

        canvas.tabIndex = 0;
        var dragging = false;
        function setFromEvent(e) {
            var r = canvas.getBoundingClientRect();
            var fn = DERIV_FNS[self.fi];
            self.x = clamp(self.plot.invX(e.clientX - r.left), fn.x0, fn.x1);
            self.draw();
        }
        canvas.addEventListener('pointerdown', function (e) {
            dragging = true;
            canvas.setPointerCapture(e.pointerId);
            setFromEvent(e);
            e.preventDefault();
        });
        canvas.addEventListener('pointermove', function (e) {
            if (dragging) { setFromEvent(e); e.preventDefault(); }
        });
        function stop() { dragging = false; }
        canvas.addEventListener('pointerup', stop);
        canvas.addEventListener('pointercancel', stop);
        canvas.addEventListener('keydown', function (e) {
            var fn = DERIV_FNS[self.fi];
            var step = (e.shiftKey ? 0.25 : 0.05) * (fn.x1 - fn.x0) / 6;
            if (e.key === 'ArrowLeft') self.x = clamp(self.x - step, fn.x0, fn.x1);
            else if (e.key === 'ArrowRight') self.x = clamp(self.x + step, fn.x0, fn.x1);
            else return;
            e.preventDefault();
            self.draw();
        });
    }

    DerivativeExplorer.prototype.resize = function () {
        var fn = DERIV_FNS[this.fi];
        this.plot.setRange(fn.x0, fn.x1, fn.y0, fn.y1);
        this.plot.resize();
    };

    DerivativeExplorer.prototype.draw = function () {
        var fn = DERIV_FNS[this.fi], p = this.plot;
        p.setRange(fn.x0, fn.x1, fn.y0, fn.y1);
        p.frame('x', 'y');

        var x = this.x, y = fn.f(x), m = fn.df(x);
        p.plot(fn.f, C.curve, 2.2);

        var xh = clamp(x + this.h, fn.x0, fn.x1);
        var hEff = xh - x;
        var quot = hEff !== 0 ? (fn.f(xh) - y) / hEff : NaN;

        if (this.showSecant && isFinite(quot)) {
            p.lineAt(x, y, quot, C.secant, 1.6, [5, 4]);
            p.vline(xh, fn.f(xh), y, C.secant, [2, 3]);
            p.dot(xh, fn.f(xh), C.secant, 4);
        }

        p.lineAt(x, y, m, C.tangent, 2);
        p.dot(x, y, C.curve);

        this.f.at.textContent = 'x = ' + fmt(x, 3) + ',  f(x) = ' + fmt(y, 3);
        this.f.slope.textContent = fmt(m, 4) + '   [' + fn.dfs + ']';
        this.f.quot.textContent = isFinite(quot)
            ? '(f(x+h) - f(x)) / h = ' + fmt(quot, 4)
            : '--';
        this.f.err.textContent = isFinite(quot)
            ? '|quotient - f′(x)| = ' + fmt(Math.abs(quot - m), 4)
            : '--';
    };

    /* ============================================================ widget 2 */

    /* Curves given parametrically so they are easy to draw, but the tangent
     * slope is the one implicit differentiation produces. */
    var IMPLICIT_CURVES = [
        {
            label: 'x² + y² = 25',
            eq: 'x² + y² = 25',
            deriv: "2x + 2y·y' = 0  ⟹  y' = -x / y",
            pt: function (t) { return [5 * Math.cos(t), 5 * Math.sin(t)]; },
            slope: function (x, y) { return -x / y; },
            tmin: 0, tmax: TAU, start: Math.atan2(4, 3),
            x0: -7, x1: 7, y0: -6.2, y1: 6.2
        },
        {
            label: 'x²/9 + y²/4 = 1',
            eq: 'x²/9 + y²/4 = 1',
            deriv: "2x/9 + y·y'/2 = 0  ⟹  y' = -4x / (9y)",
            pt: function (t) { return [3 * Math.cos(t), 2 * Math.sin(t)]; },
            slope: function (x, y) { return -4 * x / (9 * y); },
            tmin: 0, tmax: TAU, start: Math.PI / 4,
            x0: -4.6, x1: 4.6, y0: -3.1, y1: 3.1
        },
        {
            label: 'x³ + y³ = 6xy',
            eq: 'x³ + y³ = 6xy',
            deriv: "3x² + 3y²·y' = 6y + 6x·y'  ⟹  y' = (2y - x²) / (y² - 2x)",
            /* the folium's loop, via x = 6t/(1+t³), y = 6t²/(1+t³), t = tan s */
            pt: function (s) {
                var t = Math.tan(s), d = 1 + t * t * t;
                return [6 * t / d, 6 * t * t / d];
            },
            slope: function (x, y) { return (2 * y - x * x) / (y * y - 2 * x); },
            tmin: 0.0005, tmax: Math.PI / 2 - 0.0005, start: Math.atan(1),
            x0: -0.8, x1: 4.6, y0: -0.8, y1: 4.2
        }
    ];

    function ImplicitCurve(root) {
        var self = this;
        this.ci = 0;
        this.t = IMPLICIT_CURVES[0].start;

        var body = el('div', 'viz-body');
        var canvas = canvasIn(body,
            'A curve defined by an equation, with a draggable point and its tangent line.',
            '520px');
        this.canvas = canvas;
        this.plot = new Plot2D(canvas, { ratio: 0.85, maxH: 400 });

        var row = el('div', 'viz-controls');
        buttonGroup(row, 'curve', IMPLICIT_CURVES, 0, function (it, i) {
            self.ci = i;
            self.t = it.start;
            self.resize();
            self.draw();
        });
        body.appendChild(row);

        var ro = el('div', 'viz-readout');
        this.f = {
            eq: readoutField(ro, 'the equation'),
            at: readoutField(ro, 'point on the curve'),
            deriv: readoutField(ro, 'implicit differentiation gives'),
            slope: readoutField(ro, "tangent slope y' here")
        };
        body.appendChild(ro);

        body.appendChild(el('p', 'viz-note',
            'These curves are not graphs of any single function — most vertical ' +
            'lines cross them twice — so there is no f to differentiate. There is ' +
            'still a tangent at every point, and implicit differentiation is how ' +
            'you get its slope without ever solving for y.'));

        root.appendChild(body);

        canvas.tabIndex = 0;
        var dragging = false;
        function setFromEvent(e) {
            var r = canvas.getBoundingClientRect();
            var mx = self.plot.invX(e.clientX - r.left);
            var my = self.plot.invY(e.clientY - r.top);
            var cur = IMPLICIT_CURVES[self.ci];
            var best = self.t, bestD = Infinity;
            var N = 600;
            for (var i = 0; i <= N; i++) {
                var t = cur.tmin + (cur.tmax - cur.tmin) * i / N;
                var q = cur.pt(t);
                if (!isFinite(q[0]) || !isFinite(q[1])) continue;
                var d = (q[0] - mx) * (q[0] - mx) + (q[1] - my) * (q[1] - my);
                if (d < bestD) { bestD = d; best = t; }
            }
            self.t = best;
            self.draw();
        }
        canvas.addEventListener('pointerdown', function (e) {
            dragging = true;
            canvas.setPointerCapture(e.pointerId);
            setFromEvent(e);
            e.preventDefault();
        });
        canvas.addEventListener('pointermove', function (e) {
            if (dragging) { setFromEvent(e); e.preventDefault(); }
        });
        function stop() { dragging = false; }
        canvas.addEventListener('pointerup', stop);
        canvas.addEventListener('pointercancel', stop);
        canvas.addEventListener('keydown', function (e) {
            var cur = IMPLICIT_CURVES[self.ci];
            var step = (e.shiftKey ? 0.08 : 0.02) * (cur.tmax - cur.tmin);
            if (e.key === 'ArrowLeft') self.t -= step;
            else if (e.key === 'ArrowRight') self.t += step;
            else return;
            self.t = clamp(self.t, cur.tmin, cur.tmax);
            e.preventDefault();
            self.draw();
        });
    }

    ImplicitCurve.prototype.resize = function () {
        var c = IMPLICIT_CURVES[this.ci];
        this.plot.setRange(c.x0, c.x1, c.y0, c.y1);
        this.plot.resize();
    };

    ImplicitCurve.prototype.draw = function () {
        var c = IMPLICIT_CURVES[this.ci], p = this.plot;
        p.setRange(c.x0, c.x1, c.y0, c.y1);
        p.frame('x', 'y');

        var pts = [], N = 700;
        for (var i = 0; i <= N; i++) {
            var t = c.tmin + (c.tmax - c.tmin) * i / N;
            var q = c.pt(t);
            if (isFinite(q[0]) && isFinite(q[1])) pts.push(q);
        }
        p.polyline(pts, C.curve, 2.2);

        var pt = c.pt(this.t);
        var m = c.slope(pt[0], pt[1]);
        if (isFinite(m)) {
            p.lineAt(pt[0], pt[1], m, C.tangent, 2);
        } else {
            p.polyline([[pt[0], c.y0], [pt[0], c.y1]], C.tangent, 2);
        }
        p.dot(pt[0], pt[1], C.curve);

        this.f.eq.textContent = c.eq;
        this.f.at.textContent = '(' + fmt(pt[0], 3) + ', ' + fmt(pt[1], 3) + ')';
        this.f.deriv.textContent = c.deriv;
        this.f.slope.textContent = isFinite(m)
            ? fmt(m, 4)
            : 'undefined — the tangent is vertical here';
    };

    /* ============================================================ widget 3 */

    var AREA_FNS = [
        { label: 'x²', f: function (x) { return x * x; }, a: 0, b: 2, exact: 8 / 3,
          exs: '∫₀² x² dx = 8/3', y0: -0.4, y1: 4.6 },
        { label: 'sin x', f: Math.sin, a: 0, b: Math.PI, exact: 2,
          exs: '∫₀^π sin x dx = 2', y0: -0.25, y1: 1.35 },
        { label: '√x', f: Math.sqrt, a: 0, b: 4, exact: 16 / 3,
          exs: '∫₀⁴ √x dx = 16/3', y0: -0.3, y1: 2.4 },
        { label: '1/(1+x²)', f: function (x) { return 1 / (1 + x * x); }, a: 0, b: 2,
          exact: Math.atan(2), exs: '∫₀² dx/(1+x²) = arctan 2', y0: -0.15, y1: 1.15 }
    ];

    var AREA_RULES = [
        { label: 'left', at: function (i) { return i; } },
        { label: 'right', at: function (i) { return i + 1; } },
        { label: 'midpoint', at: function (i) { return i + 0.5; } },
        { label: 'trapezoid', at: null }
    ];

    function RiemannSum(root) {
        var self = this;
        this.fi = 0;
        this.ri = 0;
        this.n = 6;

        var body = el('div', 'viz-body');
        var canvas = canvasIn(body,
            'Area under a curve approximated by rectangles.');
        this.canvas = canvas;
        this.canvas.style.cursor = 'default';
        this.plot = new Plot2D(canvas, { ratio: 0.52, maxH: 320 });

        var row1 = el('div', 'viz-controls');
        buttonGroup(row1, 'integrand', AREA_FNS, 0, function (it, i) {
            self.fi = i; self.resize(); self.draw();
        });
        body.appendChild(row1);

        var row2 = el('div', 'viz-controls');
        buttonGroup(row2, 'rule', AREA_RULES, 0, function (it, i) {
            self.ri = i; self.draw();
        });
        row2.appendChild(slider('n', 1, 60, 1, this.n,
            function (v) { return String(Math.round(v)); },
            function (v) { self.n = Math.round(v); self.draw(); }).node);
        body.appendChild(row2);

        var ro = el('div', 'viz-readout');
        this.f = {
            sum: readoutField(ro, 'approximation'),
            exact: readoutField(ro, 'exact value'),
            err: readoutField(ro, 'error'),
            width: readoutField(ro, 'strip width Δx')
        };
        body.appendChild(ro);

        body.appendChild(el('p', 'viz-note',
            'Push n up and every rule converges to the same number. That number ' +
            'is the definite integral — the limit of the sums, not a rectangle ' +
            'count. The fundamental theorem is what lets you get it from an ' +
            'antiderivative instead of ever computing this limit.'));

        root.appendChild(body);
    }

    RiemannSum.prototype.resize = function () {
        var fn = AREA_FNS[this.fi];
        var padx = (fn.b - fn.a) * 0.12;
        this.plot.setRange(fn.a - padx, fn.b + padx, fn.y0, fn.y1);
        this.plot.resize();
    };

    RiemannSum.prototype.draw = function () {
        var fn = AREA_FNS[this.fi], rule = AREA_RULES[this.ri], p = this.plot;
        var padx = (fn.b - fn.a) * 0.12;
        p.setRange(fn.a - padx, fn.b + padx, fn.y0, fn.y1);
        p.frame('x', 'y');

        var n = this.n, dx = (fn.b - fn.a) / n, sum = 0;
        var ctx = p.ctx;

        p.clip(function () {
            ctx.save();
            ctx.fillStyle = C.area;
            ctx.strokeStyle = C.areaEdge;
            ctx.lineWidth = 1;
            for (var i = 0; i < n; i++) {
                var xa = fn.a + i * dx, xb = xa + dx;
                if (rule.at === null) {
                    var ya = fn.f(xa), yb = fn.f(xb);
                    sum += (ya + yb) / 2 * dx;
                    ctx.beginPath();
                    ctx.moveTo(p.X(xa), p.Y(0));
                    ctx.lineTo(p.X(xa), p.Y(ya));
                    ctx.lineTo(p.X(xb), p.Y(yb));
                    ctx.lineTo(p.X(xb), p.Y(0));
                    ctx.closePath();
                    ctx.fill(); ctx.stroke();
                } else {
                    var xs = fn.a + rule.at(i) * dx;
                    var hgt = fn.f(xs);
                    sum += hgt * dx;
                    ctx.beginPath();
                    ctx.rect(p.X(xa), p.Y(hgt), p.X(xb) - p.X(xa), p.Y(0) - p.Y(hgt));
                    ctx.fill(); ctx.stroke();
                }
            }
            ctx.restore();
        });

        p.plot(fn.f, C.curve, 2.2);
        p.vline(fn.a, fn.y0, fn.y1, C.tick, [3, 3]);
        p.vline(fn.b, fn.y0, fn.y1, C.tick, [3, 3]);

        this.f.sum.textContent = fmt(sum, 6) + '   (' + rule.label + ', n = ' + n + ')';
        this.f.exact.textContent = fmt(fn.exact, 6) + '   [' + fn.exs + ']';
        this.f.err.textContent = fmt(Math.abs(sum - fn.exact), 6);
        this.f.width.textContent = fmt(dx, 4);
    };

    /* ============================================================ widget 4 */

    function pole(A, r) { return function (x) { return A / (x - r); }; }

    var PF_CASES = [
        {
            label: '1/((x-1)(x+2))',
            whole: 'f(x) = 1 / ((x - 1)(x + 2))',
            split: 'f(x) = (1/3)/(x - 1)  -  (1/3)/(x + 2)',
            f: function (x) { return 1 / ((x - 1) * (x + 2)); },
            parts: [
                { s: '(1/3)/(x - 1)', f: pole(1 / 3, 1), c: C.p1 },
                { s: '-(1/3)/(x + 2)', f: pole(-1 / 3, -2), c: C.p2 }
            ],
            x0: -5, x1: 4, y0: -2, y1: 2
        },
        {
            label: '1/(x²-4)',
            whole: 'f(x) = 1 / (x² - 4) = 1 / ((x - 2)(x + 2))',
            split: 'f(x) = (1/4)/(x - 2)  -  (1/4)/(x + 2)',
            f: function (x) { return 1 / (x * x - 4); },
            parts: [
                { s: '(1/4)/(x - 2)', f: pole(1 / 4, 2), c: C.p1 },
                { s: '-(1/4)/(x + 2)', f: pole(-1 / 4, -2), c: C.p2 }
            ],
            x0: -5, x1: 5, y0: -1.6, y1: 1.6
        },
        {
            label: 'x/((x+1)(x-3))',
            whole: 'f(x) = x / ((x + 1)(x - 3))',
            split: 'f(x) = (1/4)/(x + 1)  +  (3/4)/(x - 3)',
            f: function (x) { return x / ((x + 1) * (x - 3)); },
            parts: [
                { s: '(1/4)/(x + 1)', f: pole(1 / 4, -1), c: C.p1 },
                { s: '(3/4)/(x - 3)', f: pole(3 / 4, 3), c: C.p2 }
            ],
            x0: -5, x1: 6, y0: -2.4, y1: 2.4
        }
    ];

    function PartialFractions(root) {
        var self = this;
        this.ci = 0;
        this.showParts = true;
        this.showSum = false;

        var body = el('div', 'viz-body');
        var canvas = canvasIn(body,
            'A rational function drawn together with the simple fractions it splits into.');
        this.canvas = canvas;
        this.canvas.style.cursor = 'default';
        this.plot = new Plot2D(canvas, { ratio: 0.55, maxH: 330 });

        var row1 = el('div', 'viz-controls');
        buttonGroup(row1, 'f(x)', PF_CASES, 0, function (it, i) {
            self.ci = i; self.resize(); self.draw();
        });
        body.appendChild(row1);

        var row2 = el('div', 'viz-controls');
        row2.appendChild(checkbox('show the pieces', true, function (v) {
            self.showParts = v; self.draw();
        }).node);
        row2.appendChild(checkbox('overlay their sum', false, function (v) {
            self.showSum = v; self.draw();
        }).node);
        body.appendChild(row2);

        var ro = el('div', 'viz-readout');
        this.f = {
            whole: readoutField(ro, 'the awkward form'),
            split: readoutField(ro, 'the useful form'),
            why: readoutField(ro, 'why bother')
        };
        body.appendChild(ro);

        this.legendBox = el('div');
        body.appendChild(this.legendBox);

        body.appendChild(el('p', 'viz-note',
            'Tick "overlay their sum" — the dashed black curve lands exactly on ' +
            'the blue one. Partial fractions is not an approximation; it is the ' +
            'same function rewritten as a sum of pieces whose integrals you ' +
            'already know.'));

        root.appendChild(body);
    }

    PartialFractions.prototype.resize = function () {
        var c = PF_CASES[this.ci];
        this.plot.setRange(c.x0, c.x1, c.y0, c.y1);
        this.plot.resize();
    };

    PartialFractions.prototype.draw = function () {
        var c = PF_CASES[this.ci], p = this.plot;
        p.setRange(c.x0, c.x1, c.y0, c.y1);
        p.frame('x', 'y');

        p.plot(c.f, C.curve, 2.4);

        if (this.showParts) {
            c.parts.forEach(function (pt) { p.plot(pt.f, pt.c, 1.5); });
        }
        if (this.showSum) {
            p.plot(function (x) {
                var s = 0;
                for (var i = 0; i < c.parts.length; i++) s += c.parts[i].f(x);
                return s;
            }, '#212529', 1.6, [5, 4]);
        }

        this.f.whole.textContent = c.whole;
        this.f.split.textContent = c.split;
        this.f.why.textContent = 'every piece integrates to a logarithm';

        var items = [[C.curve, 'f(x)']];
        c.parts.forEach(function (pt) { items.push([pt.c, pt.s]); });
        if (this.showSum) items.push(['#212529', 'sum of the pieces']);
        this.legendBox.innerHTML = '';
        this.legendBox.appendChild(legend(items));
    };

    /* ================================================================ boot */

    function init() {
        var nodes = document.querySelectorAll('[data-viz]');
        Array.prototype.forEach.call(nodes, function (node) {
            var kind = node.getAttribute('data-viz');
            var wd = null;
            try {
                if (kind === 'derivative-explorer') wd = new DerivativeExplorer(node);
                else if (kind === 'implicit-curve') wd = new ImplicitCurve(node);
                else if (kind === 'riemann-sum') wd = new RiemannSum(node);
                else if (kind === 'partial-fractions') wd = new PartialFractions(node);
            } catch (err) {
                if (window.console) console.error('viz init failed for ' + kind, err);
                return;
            }
            if (!wd) return;
            widgets.push(wd);
            wd.resize();
            wd.draw();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
