/* ==========================================================================
 * Visualizations for the eigenvalue / SVD question (interview-prep /
 * linear-algebra). Two self-contained canvas widgets, no dependencies:
 *
 *   data-viz="linear-map"   the unit circle, its image ellipse, the
 *                           eigenvector directions that survive the map, and
 *                           the singular vectors that become the ellipse axes
 *   data-viz="svd-stages"   the same map taken apart as rotate -> stretch ->
 *                           rotate, one stage at a time
 *
 * The Markdown supplies each container's .viz-head; this file appends the
 * .viz-body.
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
        circle: '#ced4da',
        image: '#0d6efd',
        sing: '#e8590c',
        rsing: '#7048e8',
        eig: '#d6336c',
        probe: '#0e9f6e'
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

    function norm2(v) { return Math.hypot(v[0], v[1]); }

    function unit(v) {
        var n = norm2(v);
        return n < 1e-14 ? [1, 0] : [v[0] / n, v[1] / n];
    }

    /* Present a vector with small integer entries as integers when we can. */
    function vecStr(v) {
        var s = 1 / Math.max(Math.abs(v[0]), Math.abs(v[1]));
        for (var k = 1; k <= 6; k++) {
            var a = v[0] * s * k, b = v[1] * s * k;
            if (Math.abs(a - Math.round(a)) < 1e-6 && Math.abs(b - Math.round(b)) < 1e-6) {
                return '(' + Math.round(a) + ', ' + Math.round(b) + ')';
            }
        }
        return '(' + fmt(v[0]) + ', ' + fmt(v[1]) + ')';
    }

    /* ------------------------------------------------------- 2x2 linear algebra */

    function apply(M, v) {
        return [M.a * v[0] + M.b * v[1], M.c * v[0] + M.d * v[1]];
    }

    function eig2(M) {
        var tr = M.a + M.d, det = M.a * M.d - M.b * M.c;
        var disc = tr * tr / 4 - det;
        if (disc < -1e-12) {
            return { real: false, re: tr / 2, im: Math.sqrt(-disc), tr: tr, det: det };
        }
        var r = Math.sqrt(Math.max(0, disc));
        var l1 = tr / 2 + r, l2 = tr / 2 - r;
        function vecFor(l) {
            /* (A - lI)v = 0 : (a-l)x + b y = 0, or c x + (d-l) y = 0 */
            if (Math.abs(M.b) > 1e-12) return unit([M.b, l - M.a]);
            if (Math.abs(M.c) > 1e-12) return unit([l - M.d, M.c]);
            return Math.abs(M.a - l) < 1e-12 ? [1, 0] : [0, 1];
        }
        var v1 = vecFor(l1), v2 = vecFor(l2), defective = false;
        if (Math.abs(l1 - l2) < 1e-9) {
            /* Repeated eigenvalue. It is defective only when A - lambda*I is
               NOT the zero matrix; a scalar matrix such as 2I repeats its
               eigenvalue but every direction is an eigenvector. */
            var scale = Math.max(1, Math.abs(M.a), Math.abs(M.b), Math.abs(M.c), Math.abs(M.d));
            var resid = Math.max(Math.abs(M.a - l1), Math.abs(M.b),
                                 Math.abs(M.c), Math.abs(M.d - l1));
            if (resid < 1e-12 * scale) { v1 = [1, 0]; v2 = [0, 1]; }
            else { defective = true; v2 = v1; }
        }
        return { real: true, l1: l1, l2: l2, v1: v1, v2: v2, defective: defective, tr: tr, det: det };
    }

    function svd2(M) {
        /* eigen-decompose the symmetric A^T A = [[e,f],[f,g]] */
        var e = M.a * M.a + M.c * M.c;
        var f = M.a * M.b + M.c * M.d;
        var g = M.b * M.b + M.d * M.d;
        var tr = e + g;
        var disc = Math.sqrt(Math.max(0, (e - g) * (e - g) / 4 + f * f));
        var l1 = tr / 2 + disc, l2 = Math.max(0, tr / 2 - disc);
        var v1;
        if (Math.abs(f) > 1e-12) v1 = unit([f, l1 - e]);
        else v1 = e >= g ? [1, 0] : [0, 1];
        var v2 = [-v1[1], v1[0]];
        var s1 = Math.sqrt(Math.max(0, l1)), s2 = Math.sqrt(Math.max(0, l2));
        var u1 = s1 > 1e-12 ? unit(apply(M, v1)) : [1, 0];
        var u2 = s2 > 1e-12 ? unit(apply(M, v2)) : [-u1[1], u1[0]];
        return { s1: s1, s2: s2, v1: v1, v2: v2, u1: u1, u2: u2 };
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
        return (p && p.clientWidth) || 560;
    }

    function debounce(fn, ms) {
        var t = null;
        return function () { clearTimeout(t); t = setTimeout(fn, ms); };
    }

    window.addEventListener('resize', debounce(function () {
        widgets.forEach(function (w) { w.resize(); w.draw(); });
    }, 120));

    /* ------------------------------------------- equal-aspect plot, origin centred */

    function EqPlot(canvas, opts) {
        opts = opts || {};
        this.canvas = canvas;
        this.ratio = opts.ratio || 0.82;
        this.maxH = opts.maxH || 420;
        this.half = 3;
    }

    EqPlot.prototype.resize = function () {
        var w = measure(this.canvas);
        var h = Math.round(clamp(w * this.ratio, 260, this.maxH));
        this.ctx = fitCanvas(this.canvas, w, h);
        this.w = w; this.h = h;
        this.cx = w / 2; this.cy = h / 2;
    };

    EqPlot.prototype.setHalf = function (r) {
        this.half = Math.max(0.5, r);
        this.k = Math.min(this.w, this.h) / (2 * this.half);
    };

    EqPlot.prototype.X = function (x) { return this.cx + x * this.k; };
    EqPlot.prototype.Y = function (y) { return this.cy - y * this.k; };
    EqPlot.prototype.invX = function (px) { return (px - this.cx) / this.k; };
    EqPlot.prototype.invY = function (py) { return (this.cy - py) / this.k; };

    EqPlot.prototype.frame = function () {
        var ctx = this.ctx, i;
        ctx.clearRect(0, 0, this.w, this.h);
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, this.w, this.h);

        var step = this.half > 6 ? 2 : 1;
        var nx = Math.floor(this.invX(this.w) / step) * step;
        var ny = Math.floor(this.invY(0) / step) * step;

        ctx.lineWidth = 1;
        ctx.strokeStyle = C.grid;
        ctx.beginPath();
        for (i = -nx; i <= nx; i += step) {
            var px = Math.round(this.X(i)) + 0.5;
            ctx.moveTo(px, 0); ctx.lineTo(px, this.h);
        }
        for (i = -ny; i <= ny; i += step) {
            var py = Math.round(this.Y(i)) + 0.5;
            ctx.moveTo(0, py); ctx.lineTo(this.w, py);
        }
        ctx.stroke();

        ctx.strokeStyle = C.axis;
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(0, Math.round(this.cy) + 0.5); ctx.lineTo(this.w, Math.round(this.cy) + 0.5);
        ctx.moveTo(Math.round(this.cx) + 0.5, 0); ctx.lineTo(Math.round(this.cx) + 0.5, this.h);
        ctx.stroke();

        ctx.fillStyle = C.tick;
        ctx.font = '10px ' + SANS;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        for (i = -nx; i <= nx; i += step) {
            if (i === 0) continue;
            ctx.fillText(String(i), this.X(i), this.cy + 4);
        }
    };

    EqPlot.prototype.curve = function (pts, color, width, dash) {
        var ctx = this.ctx, self = this;
        ctx.save();
        if (dash) ctx.setLineDash(dash);
        ctx.strokeStyle = color;
        ctx.lineWidth = width || 2;
        ctx.beginPath();
        pts.forEach(function (p, i) {
            var x = self.X(p[0]), y = self.Y(p[1]);
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        });
        ctx.closePath();
        ctx.stroke();
        ctx.restore();
    };

    EqPlot.prototype.arrow = function (v, color, label, opts) {
        opts = opts || {};
        var ctx = this.ctx;
        var x0 = this.cx, y0 = this.cy;
        var x1 = this.X(v[0]), y1 = this.Y(v[1]);
        var dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy);
        if (len < 1) return;
        ctx.save();
        if (opts.dash) ctx.setLineDash(opts.dash);
        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = opts.width || 2;
        ctx.beginPath();
        ctx.moveTo(x0, y0); ctx.lineTo(x1, y1);
        ctx.stroke();
        ctx.setLineDash([]);
        var ah = Math.min(9, len * 0.3), ux = dx / len, uy = dy / len;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x1 - ah * ux + ah * 0.5 * uy, y1 - ah * uy - ah * 0.5 * ux);
        ctx.lineTo(x1 - ah * ux - ah * 0.5 * uy, y1 - ah * uy + ah * 0.5 * ux);
        ctx.closePath();
        ctx.fill();
        if (label) {
            ctx.font = 'bold 11px ' + SANS;
            ctx.textAlign = ux >= 0 ? 'left' : 'right';
            ctx.textBaseline = uy <= 0 ? 'bottom' : 'top';
            ctx.fillText(label, x1 + (ux >= 0 ? 7 : -7), y1 + (uy <= 0 ? -5 : 5));
        }
        ctx.restore();
    };

    /* an infinite line through the origin in direction v */
    EqPlot.prototype.axisLine = function (v, color, dash) {
        var t = this.half * 3;
        this.ctx.save();
        this.ctx.setLineDash(dash || [5, 4]);
        this.ctx.strokeStyle = color;
        this.ctx.lineWidth = 1.2;
        this.ctx.beginPath();
        this.ctx.moveTo(this.X(-v[0] * t), this.Y(-v[1] * t));
        this.ctx.lineTo(this.X(v[0] * t), this.Y(v[1] * t));
        this.ctx.stroke();
        this.ctx.restore();
    };

    EqPlot.prototype.dot = function (v, color, r) {
        var ctx = this.ctx;
        ctx.beginPath();
        ctx.arc(this.X(v[0]), this.Y(v[1]), r || 5, 0, TAU);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 1.8;
        ctx.stroke();
    };

    /* --------------------------------------------------------- UI factories */

    function button(label, onClick) {
        var b = el('button', 'viz-btn', label);
        b.type = 'button';
        b.addEventListener('click', function () { onClick(b); });
        return b;
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
        var wrap = el('div', 'viz-canvas-wrap');
        if (maxWidth) { wrap.style.maxWidth = maxWidth; wrap.style.margin = '0 auto'; }
        var canvas = document.createElement('canvas');
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label', label);
        wrap.appendChild(canvas);
        body.appendChild(wrap);
        return canvas;
    }

    /* A 2x2 matrix editor: four number inputs laid out as a matrix. */
    function matrixEditor(M, onChange) {
        var wrap = el('div', 'viz-matrix');
        wrap.appendChild(el('span', 'viz-matrix-label', 'A ='));
        var grid = el('div', 'viz-matrix-grid');
        var keys = ['a', 'b', 'c', 'd'];
        var inputs = {};
        keys.forEach(function (k) {
            var inp = document.createElement('input');
            inp.type = 'number';
            inp.step = '1';
            inp.value = M[k];
            inp.setAttribute('aria-label', 'matrix entry ' + k);
            inp.addEventListener('input', function () {
                var v = parseFloat(inp.value);
                M[k] = isFinite(v) ? v : 0;
                onChange();
            });
            inputs[k] = inp;
            grid.appendChild(inp);
        });
        wrap.appendChild(grid);
        return { node: wrap, inputs: inputs };
    }

    var PRESETS = [
        { label: '[[4,1],[2,3]]', m: { a: 4, b: 1, c: 2, d: 3 } },
        { label: '[[3,0],[4,5]]', m: { a: 3, b: 0, c: 4, d: 5 } },
        { label: '[[2,1],[1,2]]', m: { a: 2, b: 1, c: 1, d: 2 } },
        { label: '[[0,-1],[1,0]]', m: { a: 0, b: -1, c: 1, d: 0 } },
        { label: '[[3,1],[0,3]]', m: { a: 3, b: 1, c: 0, d: 3 } },
        { label: '[[1,2],[2,4]]', m: { a: 1, b: 2, c: 2, d: 4 } }
    ];

    /* ============================================================ widget 1 */

    function LinearMap(root) {
        var self = this;
        this.M = { a: 2, b: 1, c: 1, d: 2 };
        this.theta = Math.PI / 5;

        var body = el('div', 'viz-body');
        var canvas = canvasIn(body,
            'The unit circle and its image under a 2x2 matrix, with eigenvector and singular vector directions.',
            '520px');
        this.canvas = canvas;
        this.plot = new EqPlot(canvas, { ratio: 0.9, maxH: 430 });

        var row1 = el('div', 'viz-controls');
        row1.appendChild(el('span', 'viz-label', 'try'));
        PRESETS.forEach(function (p) {
            row1.appendChild(button(p.label, function () {
                self.M.a = p.m.a; self.M.b = p.m.b; self.M.c = p.m.c; self.M.d = p.m.d;
                ['a', 'b', 'c', 'd'].forEach(function (k) { self.ed.inputs[k].value = self.M[k]; });
                self.draw();
            }));
        });
        body.appendChild(row1);

        var row2 = el('div', 'viz-controls');
        this.ed = matrixEditor(this.M, function () { self.draw(); });
        row2.appendChild(this.ed.node);
        body.appendChild(row2);

        var ro = el('div', 'viz-readout');
        this.f = {
            eig: readoutField(ro, 'eigenvalues'),
            evec: readoutField(ro, 'eigenvectors'),
            sing: readoutField(ro, 'singular values'),
            rsv: readoutField(ro, 'right singular vectors v'),
            probe: readoutField(ro, 'the dragged x and its image Ax'),
            misc: readoutField(ro, 'trace / det / rank / cond')
        };
        body.appendChild(ro);

        this.note = el('div', 'viz-warn');
        this.note.hidden = true;
        body.appendChild(this.note);

        body.appendChild(legend([
            [C.circle, 'unit circle'], [C.image, 'its image, an ellipse'],
            [C.rsing, 'v1, v2'], [C.sing, 'σ1u1, σ2u2 (the axes)'],
            [C.eig, 'eigenvector lines'], [C.probe, 'x and Ax']
        ]));
        body.appendChild(el('p', 'viz-note',
            'Drag the green point around the unit circle, or edit the matrix directly. ' +
            'The eigenvector lines are the directions where Ax stays on the same line ' +
            'through the origin; the singular vectors are the ones that land on the ' +
            'ellipse axes. They are the same pair only when A is symmetric.'));

        root.appendChild(body);

        canvas.tabIndex = 0;
        var dragging = false;
        function setFromEvent(e) {
            var r = canvas.getBoundingClientRect();
            var x = self.plot.invX(e.clientX - r.left);
            var y = self.plot.invY(e.clientY - r.top);
            if (Math.hypot(x, y) > 1e-9) self.theta = Math.atan2(y, x);
            self.draw();
        }
        canvas.addEventListener('pointerdown', function (e) {
            dragging = true;
            canvas.setPointerCapture(e.pointerId);
            setFromEvent(e); e.preventDefault();
        });
        canvas.addEventListener('pointermove', function (e) {
            if (dragging) { setFromEvent(e); e.preventDefault(); }
        });
        function stop() { dragging = false; }
        canvas.addEventListener('pointerup', stop);
        canvas.addEventListener('pointercancel', stop);
        canvas.addEventListener('keydown', function (e) {
            var step = e.shiftKey ? 0.25 : 0.05;
            if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') self.theta += step;
            else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') self.theta -= step;
            else return;
            e.preventDefault();
            self.draw();
        });
    }

    LinearMap.prototype.resize = function () { this.plot.resize(); };

    LinearMap.prototype.draw = function () {
        var M = this.M, p = this.plot;
        var sv = svd2(M), ev = eig2(M);

        p.setHalf(Math.max(1.4, sv.s1 * 1.3));
        p.frame();

        var i, pts = [], img = [];
        for (i = 0; i <= 180; i++) {
            var t = TAU * i / 180;
            pts.push([Math.cos(t), Math.sin(t)]);
            img.push(apply(M, [Math.cos(t), Math.sin(t)]));
        }
        p.curve(pts, C.circle, 1.4, [4, 4]);
        p.curve(img, C.image, 2.2);

        if (ev.real && !ev.defective) {
            p.axisLine(ev.v1, C.eig);
            p.axisLine(ev.v2, C.eig);
        } else if (ev.real) {
            p.axisLine(ev.v1, C.eig);
        }

        p.arrow([sv.v1[0], sv.v1[1]], C.rsing, 'v1', { width: 1.6 });
        p.arrow([sv.v2[0], sv.v2[1]], C.rsing, 'v2', { width: 1.6 });
        p.arrow([sv.u1[0] * sv.s1, sv.u1[1] * sv.s1], C.sing, 'σ1u1', { width: 2 });
        if (sv.s2 > 1e-9) p.arrow([sv.u2[0] * sv.s2, sv.u2[1] * sv.s2], C.sing, 'σ2u2', { width: 2 });

        var x = [Math.cos(this.theta), Math.sin(this.theta)];
        var Ax = apply(M, x);
        p.arrow(Ax, C.probe, 'Ax', { width: 2.2 });
        p.dot(x, C.probe, 5.5);
        p.dot(Ax, C.probe, 5.5);

        /* readouts */
        if (ev.real) {
            this.f.eig.textContent = 'λ = ' + fmt(ev.l1) + ',  ' + fmt(ev.l2) +
                (ev.defective ? '   (repeated)' : '');
            this.f.evec.textContent = ev.defective
                ? vecStr(ev.v1) + '  — only one direction'
                : vecStr(ev.v1) + ',  ' + vecStr(ev.v2);
        } else {
            this.f.eig.textContent = 'λ = ' + fmt(ev.re) + ' ± ' + fmt(ev.im) + 'j   (complex)';
            this.f.evec.textContent = 'none in the real plane';
        }
        this.f.sing.textContent = 'σ = ' + fmt(sv.s1) + ',  ' + fmt(sv.s2);
        this.f.rsv.textContent = vecStr(sv.v1) + ',  ' + vecStr(sv.v2) + '   (always perpendicular)';
        this.f.probe.textContent = 'x = (' + fmt(x[0], 2) + ', ' + fmt(x[1], 2) + ')  ->  (' +
            fmt(Ax[0], 2) + ', ' + fmt(Ax[1], 2) + '),  |Ax| = ' + fmt(norm2(Ax), 3);
        var rank = (sv.s2 > 1e-9) ? 2 : (sv.s1 > 1e-9 ? 1 : 0);
        this.f.misc.textContent = 'tr ' + fmt(ev.tr, 2) + ' / det ' + fmt(ev.det, 2) +
            ' / rank ' + rank + ' / cond ' +
            (sv.s2 > 1e-9 ? fmt(sv.s1 / sv.s2, 2) : 'infinite');

        if (!ev.real) {
            this.note.hidden = false;
            this.note.innerHTML = '<strong>No real eigenvectors.</strong> Every direction ' +
                'gets rotated off its own line, so no real x satisfies Ax = λx. The ' +
                'singular values still exist, as they always do — that is the practical ' +
                'difference between the two decompositions.';
        } else if (ev.defective) {
            this.note.hidden = false;
            this.note.innerHTML = '<strong>Defective.</strong> The eigenvalue is repeated ' +
                'but there is only one eigenvector direction, so A cannot be diagonalized. ' +
                'The SVD is unbothered.';
        } else if (rank < 2) {
            this.note.hidden = false;
            this.note.innerHTML = '<strong>Singular.</strong> σ2 = 0, so the circle collapses ' +
                'onto a line segment: rank 1. A whole direction is annihilated.';
        } else {
            this.note.hidden = true;
        }
    };

    /* ============================================================ widget 2 */

    var STAGES = [
        { label: 'x', desc: 'the unit circle, before anything happens' },
        { label: 'Vᵀx', desc: 'rotate: Vᵀ lines the singular directions up with the axes' },
        { label: 'ΣVᵀx', desc: 'stretch: Σ scales axis 1 by σ1 and axis 2 by σ2' },
        { label: 'UΣVᵀx = Ax', desc: 'rotate again: U turns the axes into their final directions' }
    ];

    function SvdStages(root) {
        var self = this;
        this.M = { a: 3, b: 0, c: 4, d: 5 };
        this.stage = 0;

        var body = el('div', 'viz-body');
        var canvas = canvasIn(body,
            'The unit circle transformed one SVD factor at a time.', '520px');
        this.canvas = canvas;
        this.canvas.style.cursor = 'default';
        this.plot = new EqPlot(canvas, { ratio: 0.9, maxH: 420 });

        var row = el('div', 'viz-controls');
        row.appendChild(el('span', 'viz-label', 'stage'));
        this.btns = STAGES.map(function (s, i) {
            var b = button(s.label, function () {
                self.stage = i;
                self.btns.forEach(function (o) { o.classList.remove('is-active'); });
                b.classList.add('is-active');
                self.draw();
            });
            if (i === 0) b.classList.add('is-active');
            row.appendChild(b);
            return b;
        });
        body.appendChild(row);

        var row2 = el('div', 'viz-controls');
        row2.appendChild(el('span', 'viz-label', 'matrix'));
        PRESETS.forEach(function (p) {
            row2.appendChild(button(p.label, function () {
                self.M = { a: p.m.a, b: p.m.b, c: p.m.c, d: p.m.d };
                self.draw();
            }));
        });
        body.appendChild(row2);

        var ro = el('div', 'viz-readout');
        this.f = {
            what: readoutField(ro, 'what just happened'),
            sing: readoutField(ro, 'singular values'),
            shape: readoutField(ro, 'current shape')
        };
        body.appendChild(ro);

        body.appendChild(el('p', 'viz-note',
            'Every matrix does the same three things in order: a rotation, an ' +
            'axis-aligned stretch, another rotation. The coloured dots start ' +
            'evenly spaced so you can see which parts of the circle get pulled ' +
            'apart and which get squeezed together.'));

        root.appendChild(body);
    }

    SvdStages.prototype.resize = function () { this.plot.resize(); };

    SvdStages.prototype.stageMatrix = function (sv, stage) {
        /* V^T has rows v1, v2; Sigma is diag(s1,s2); U has columns u1, u2. */
        var Vt = { a: sv.v1[0], b: sv.v1[1], c: sv.v2[0], d: sv.v2[1] };
        if (stage <= 0) return { a: 1, b: 0, c: 0, d: 1 };
        if (stage === 1) return Vt;
        if (stage === 2) {
            return { a: sv.s1 * Vt.a, b: sv.s1 * Vt.b, c: sv.s2 * Vt.c, d: sv.s2 * Vt.d };
        }
        return this.M;
    };

    SvdStages.prototype.draw = function () {
        var p = this.plot, sv = svd2(this.M);
        var T = this.stageMatrix(sv, this.stage);

        p.setHalf(Math.max(1.4, sv.s1 * 1.3));
        p.frame();

        var i, pts = [];
        for (i = 0; i <= 180; i++) {
            var t = TAU * i / 180;
            pts.push(apply(T, [Math.cos(t), Math.sin(t)]));
        }
        p.curve(pts, C.image, 2.2);

        /* evenly spaced markers, coloured by where they started */
        var N = 16;
        for (i = 0; i < N; i++) {
            var a = TAU * i / N;
            var q = apply(T, [Math.cos(a), Math.sin(a)]);
            var hue = Math.round(360 * i / N);
            p.dot(q, 'hsl(' + hue + ', 70%, 52%)', 4.5);
        }

        /* the two singular directions, carried along */
        var d1 = apply(T, sv.v1), d2 = apply(T, sv.v2);
        p.arrow(d1, C.sing, null, { width: 2 });
        if (norm2(d2) > 1e-9) p.arrow(d2, C.sing, null, { width: 2 });

        this.f.what.textContent = STAGES[this.stage].desc;
        this.f.sing.textContent = 'σ1 = ' + fmt(sv.s1) + ',  σ2 = ' + fmt(sv.s2);
        this.f.shape.textContent = this.stage === 0
            ? 'a circle of radius 1'
            : (this.stage === 1
                ? 'still a circle — a rotation cannot change any length'
                : (this.stage === 2
                    ? 'an ellipse with semi-axes σ1, σ2 along the coordinate axes'
                    : 'the same ellipse, turned into its final orientation'));
    };

    /* ================================================================ boot */

    function init() {
        var nodes = document.querySelectorAll('[data-viz]');
        Array.prototype.forEach.call(nodes, function (node) {
            var kind = node.getAttribute('data-viz');
            var w = null;
            try {
                if (kind === 'linear-map') w = new LinearMap(node);
                else if (kind === 'svd-stages') w = new SvdStages(node);
            } catch (err) {
                if (window.console) console.error('viz init failed for ' + kind, err);
                return;
            }
            if (!w) return;
            widgets.push(w);
            w.resize();
            w.draw();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
