/* ==========================================================================
 * Visualizations for "Rectangular to polar" (interview-prep / signal-processing)
 *
 * Four self-contained canvas widgets, no external dependencies:
 *
 *   data-viz="argand-explorer"  drag z around the complex plane and watch the
 *                               polar coordinates, including the atan-vs-atan2
 *                               disagreement that this question is really about
 *   data-viz="argand-multiply"  drag z and w; the product shows that moduli
 *                               multiply and arguments add
 *   data-viz="phasor"           a rotating phasor and the cosine it traces out
 *   data-viz="helix"            the 3D curve e^{j w t} whose two shadows are
 *                               the cosine and the sine
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
        circle: '#ced4da',
        z: '#0d6efd',
        w: '#0e9f6e',
        prod: '#e8590c',
        re: '#d6336c',
        im: '#7048e8',
        arcZ: 'rgba(13, 110, 253, 0.14)',
        arcW: 'rgba(14, 159, 110, 0.16)',
        arcP: 'rgba(232, 89, 12, 0.14)',
        warn: '#a15c07'
    };

    var SANS = '"Lato", system-ui, sans-serif';
    var MONO = '"Source Code Pro", ui-monospace, monospace';

    var reduceMotion = window.matchMedia &&
        window.matchMedia('(prefers-reduced-motion: reduce)').matches;

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

    function rectStr(a, b, n) {
        return fmt(a, n) + (b < 0 ? ' - ' : ' + ') + fmt(Math.abs(b), n) + 'j';
    }

    function degOf(t) { return t * 180 / Math.PI; }

    function wrap2pi(t) { return ((t % TAU) + TAU) % TAU; }

    function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

    /* A modest set of exact values, so the readout can say "= pi/4" when the
     * dragged point happens to land on one. Purely cosmetic, but it is the
     * bridge between the picture and the closed-form answer. */
    var NICE_ANGLES = [
        [0, '0'], [Math.PI / 6, 'pi/6'], [Math.PI / 4, 'pi/4'], [Math.PI / 3, 'pi/3'],
        [Math.PI / 2, 'pi/2'], [2 * Math.PI / 3, '2pi/3'], [3 * Math.PI / 4, '3pi/4'],
        [5 * Math.PI / 6, '5pi/6'], [Math.PI, 'pi']
    ];

    function niceAngle(t) {
        for (var i = 0; i < NICE_ANGLES.length; i++) {
            if (Math.abs(Math.abs(t) - NICE_ANGLES[i][0]) < 1e-6) {
                var name = NICE_ANGLES[i][1];
                if (name === '0') return '0';
                return (t < 0 ? '-' : '') + name;
            }
        }
        return null;
    }

    function niceRadius(r) {
        var cands = [[Math.SQRT2, 'sqrt(2)'], [Math.sqrt(3), 'sqrt(3)'], [Math.sqrt(5), 'sqrt(5)'],
                     [Math.sqrt(8), '2sqrt(2)'], [Math.sqrt(10), 'sqrt(10)'], [Math.sqrt(13), 'sqrt(13)']];
        if (Math.abs(r - Math.round(r)) < 1e-9 && r > 0) return String(Math.round(r));
        for (var i = 0; i < cands.length; i++) {
            if (Math.abs(r - cands[i][0]) < 1e-9) return cands[i][1];
        }
        return null;
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
        return function () {
            clearTimeout(t);
            t = setTimeout(fn, ms);
        };
    }

    window.addEventListener('resize', debounce(function () {
        widgets.forEach(function (wd) { wd.resize(); wd.draw(); });
    }, 120));

    /* Only animate what the reader can actually see. */
    var io = ('IntersectionObserver' in window) ? new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
            var wd = e.target.__viz;
            if (!wd) return;
            wd.visible = e.isIntersecting;
            if (wd.visible && wd.wantsPlay) wd.start(); else wd.stop();
        });
    }, { threshold: 0.05 }) : null;

    /* --------------------------------------------------------- UI factories */

    function button(label, onClick) {
        var b = el('button', 'viz-btn', label);
        b.type = 'button';
        b.addEventListener('click', function () { onClick(b); });
        return b;
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
        function sync() {
            out.textContent = format(parseFloat(input.value));
        }
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

    /* ------------------------------------------------------ dragging points */

    /* Attach pointer + keyboard dragging to a canvas. `pick` returns the index
     * of the handle nearest the pointer (or -1), `move` receives world coords. */
    function makeDraggable(canvas, plot, handles, onChange) {
        var active = -1;

        function worldAt(evt) {
            var r = canvas.getBoundingClientRect();
            return {
                a: plot.toA(evt.clientX - r.left),
                b: plot.toB(evt.clientY - r.top)
            };
        }

        function nearest(p) {
            var best = -1, bestD = Infinity;
            handles().forEach(function (h, i) {
                var d = Math.hypot(plot.toX(h.a) - plot.toX(p.a), plot.toY(h.b) - plot.toY(p.b));
                if (d < bestD) { bestD = d; best = i; }
            });
            return bestD < 44 ? best : -1;
        }

        canvas.addEventListener('pointerdown', function (e) {
            var p = worldAt(e);
            active = nearest(p);
            if (active < 0) return;
            canvas.setPointerCapture(e.pointerId);
            canvas.style.cursor = 'grabbing';
            onChange(active, p.a, p.b);
            e.preventDefault();
        });

        canvas.addEventListener('pointermove', function (e) {
            var p = worldAt(e);
            if (active < 0) {
                canvas.style.cursor = nearest(p) >= 0 ? 'grab' : 'crosshair';
                return;
            }
            onChange(active, p.a, p.b);
            e.preventDefault();
        });

        function release(e) {
            if (active < 0) return;
            active = -1;
            canvas.style.cursor = 'crosshair';
            if (e && e.pointerId !== undefined && canvas.hasPointerCapture &&
                canvas.hasPointerCapture(e.pointerId)) {
                canvas.releasePointerCapture(e.pointerId);
            }
        }
        canvas.addEventListener('pointerup', release);
        canvas.addEventListener('pointercancel', release);

        /* Keyboard: arrows nudge the first handle, shift = coarse. */
        canvas.tabIndex = 0;
        canvas.addEventListener('keydown', function (e) {
            var step = e.shiftKey ? 0.25 : 0.05;
            var h = handles()[0];
            if (!h) return;
            var da = 0, db = 0;
            if (e.key === 'ArrowLeft') da = -step;
            else if (e.key === 'ArrowRight') da = step;
            else if (e.key === 'ArrowUp') db = step;
            else if (e.key === 'ArrowDown') db = -step;
            else return;
            e.preventDefault();
            onChange(0, h.a + da, h.b + db);
        });
    }

    /* ------------------------------------------------------ the Argand plot */

    function ArgandPlot(canvas, opts) {
        this.canvas = canvas;
        this.ry = opts.ry || 4.4;
        this.maxH = opts.maxH || 420;
        this.ratio = opts.ratio || 0.8;
    }

    ArgandPlot.prototype.resize = function () {
        var w = measure(this.canvas);
        var h = Math.round(clamp(w * this.ratio, 260, this.maxH));
        this.ctx = fitCanvas(this.canvas, w, h);
        this.w = w; this.h = h;
        this.k = h / (2 * this.ry);
        this.cx = w / 2; this.cy = h / 2;
        this.rx = w / (2 * this.k);
    };

    ArgandPlot.prototype.toX = function (a) { return this.cx + a * this.k; };
    ArgandPlot.prototype.toY = function (b) { return this.cy - b * this.k; };
    ArgandPlot.prototype.toA = function (x) { return (x - this.cx) / this.k; };
    ArgandPlot.prototype.toB = function (y) { return (this.cy - y) / this.k; };

    ArgandPlot.prototype.frame = function () {
        var ctx = this.ctx, i, x, y;
        ctx.clearRect(0, 0, this.w, this.h);
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, this.w, this.h);

        var amax = Math.floor(this.rx), bmax = Math.floor(this.ry);

        ctx.lineWidth = 1;
        ctx.strokeStyle = C.grid;
        ctx.beginPath();
        for (i = -amax; i <= amax; i++) {
            x = Math.round(this.toX(i)) + 0.5;
            ctx.moveTo(x, 0); ctx.lineTo(x, this.h);
        }
        for (i = -bmax; i <= bmax; i++) {
            y = Math.round(this.toY(i)) + 0.5;
            ctx.moveTo(0, y); ctx.lineTo(this.w, y);
        }
        ctx.stroke();

        /* unit circle: the set where the polar form is a pure phase */
        ctx.save();
        ctx.setLineDash([3, 4]);
        ctx.strokeStyle = C.circle;
        ctx.beginPath();
        ctx.arc(this.cx, this.cy, this.k, 0, TAU);
        ctx.stroke();
        ctx.restore();

        /* axes */
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
        for (i = -amax; i <= amax; i++) {
            if (i === 0) continue;
            ctx.fillText(String(i), this.toX(i), this.cy + 4);
        }
        ctx.textAlign = 'right';
        ctx.textBaseline = 'middle';
        for (i = -bmax; i <= bmax; i++) {
            if (i === 0) continue;
            ctx.fillText(i + 'j', this.cx - 5, this.toY(i));
        }

        ctx.fillStyle = C.text;
        ctx.font = 'italic 11px ' + SANS;
        ctx.textAlign = 'right';
        ctx.textBaseline = 'bottom';
        ctx.fillText('Re', this.w - 6, this.cy - 5);
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('Im', this.cx + 6, 5);
    };

    /* Wedge from the positive real axis to the display angle `td`. Canvas
     * angles run clockwise because the y axis points down, hence the minus. */
    ArgandPlot.prototype.wedge = function (td, radius, fill, stroke) {
        var ctx = this.ctx;
        if (Math.abs(td) < 1e-9) return;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(this.cx, this.cy);
        ctx.lineTo(this.cx + radius, this.cy);
        ctx.arc(this.cx, this.cy, radius, 0, -td, td > 0);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        if (stroke) {
            ctx.beginPath();
            ctx.arc(this.cx, this.cy, radius, 0, -td, td > 0);
            ctx.strokeStyle = stroke;
            ctx.lineWidth = 1.2;
            ctx.stroke();
        }
        ctx.restore();
    };

    ArgandPlot.prototype.vector = function (a, b, color, label, opts) {
        opts = opts || {};
        var ctx = this.ctx;
        var x = this.toX(a), y = this.toY(b);

        if (opts.projections) {
            ctx.save();
            ctx.setLineDash([2, 3]);
            ctx.strokeStyle = color;
            ctx.globalAlpha = 0.5;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(x, y); ctx.lineTo(x, this.cy);
            ctx.moveTo(x, y); ctx.lineTo(this.cx, y);
            ctx.stroke();
            ctx.restore();
        }

        ctx.strokeStyle = color;
        ctx.lineWidth = opts.thin ? 1.4 : 2;
        if (opts.dashed) { ctx.save(); ctx.setLineDash([5, 4]); }
        ctx.beginPath();
        ctx.moveTo(this.cx, this.cy);
        ctx.lineTo(x, y);
        ctx.stroke();
        if (opts.dashed) ctx.restore();

        ctx.beginPath();
        ctx.arc(x, y, opts.small ? 4.5 : 6, 0, TAU);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();

        if (label) {
            ctx.fillStyle = color;
            ctx.font = 'bold 12px ' + SANS;
            ctx.textAlign = a >= 0 ? 'left' : 'right';
            ctx.textBaseline = b >= 0 ? 'bottom' : 'top';
            ctx.fillText(label, x + (a >= 0 ? 10 : -10), y + (b >= 0 ? -8 : 8));
        }
    };

    /* ============================================================ widget 1 */

    function ArgandExplorer(root) {
        var self = this;
        this.z = { a: 1, b: 1 };
        this.snap = true;
        this.principal = true;

        var body = el('div', 'viz-body');
        var wrapEl = el('div', 'viz-canvas-wrap');
        wrapEl.style.maxWidth = '560px';
        wrapEl.style.margin = '0 auto';
        var canvas = document.createElement('canvas');
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label',
            'Complex plane with a draggable point; arrow keys move it.');
        wrapEl.appendChild(canvas);
        body.appendChild(wrapEl);

        this.plot = new ArgandPlot(canvas, { ry: 4.4, maxH: 430, ratio: 0.82 });

        /* controls */
        var controls = el('div', 'viz-controls');
        controls.appendChild(el('span', 'viz-label', 'Jump to'));
        [['1 + j', 1, 1], ['-1 + j√3', -1, Math.sqrt(3)], ['-3', -3, 0],
         ['-2j', 0, -2], ['3 - 4j', 3, -4]].forEach(function (p) {
            controls.appendChild(button(p[0], function () {
                self.set(p[1], p[2], true);
            }));
        });
        body.appendChild(controls);

        var controls2 = el('div', 'viz-controls');
        controls2.appendChild(checkbox('snap to 0.25 grid', true, function (v) {
            self.snap = v;
            if (v) self.set(self.z.a, self.z.b);
            else self.draw();
        }).node);
        var convBtn = button('convention: (-π, π]', function (b) {
            self.principal = !self.principal;
            b.innerHTML = self.principal ? 'convention: (-π, π]' : 'convention: [0, 2π)';
            b.classList.toggle('is-active', !self.principal);
            self.draw();
        });
        controls2.appendChild(convBtn);
        body.appendChild(controls2);

        /* readout */
        var ro = el('div', 'viz-readout');
        this.f = {
            rect: readoutField(ro, 'rectangular'),
            r: readoutField(ro, 'modulus r = |z|'),
            th: readoutField(ro, 'argument θ = atan2(b, a)'),
            deg: readoutField(ro, 'θ in degrees'),
            polar: readoutField(ro, 'polar / exponential form'),
            naive: readoutField(ro, 'naive atan(b / a)')
        };
        body.appendChild(ro);

        this.warnEl = el('div', 'viz-warn');
        this.warnEl.hidden = true;
        body.appendChild(this.warnEl);

        body.appendChild(el('p', 'viz-note',
            'Drag the point, or click the canvas and use the arrow keys ' +
            '(hold Shift for coarse steps). The shaded wedge is θ, the ' +
            'solid spoke is r, and the dotted lines are the rectangular ' +
            'coordinates a and b.'));

        root.appendChild(body);

        makeDraggable(canvas, this.plot, function () { return [self.z]; },
            function (i, a, b) { self.set(a, b); });

        this.canvas = canvas;
    }

    ArgandExplorer.prototype.set = function (a, b, exact) {
        var lim = 5.9;
        a = clamp(a, -lim, lim);
        b = clamp(b, -4.3, 4.3);
        if (this.snap && !exact) {
            a = Math.round(a * 4) / 4;
            b = Math.round(b * 4) / 4;
        }
        this.z.a = a; this.z.b = b;
        this.draw();
    };

    ArgandExplorer.prototype.resize = function () { this.plot.resize(); };

    ArgandExplorer.prototype.draw = function () {
        var p = this.plot, a = this.z.a, b = this.z.b;
        var r = Math.hypot(a, b);
        var th = Math.atan2(b, a);
        var td = this.principal ? th : wrap2pi(th);

        p.frame();
        p.wedge(td, Math.min(46, Math.max(26, r * p.k * 0.45)), C.arcZ, C.z);
        p.vector(a, b, C.z, 'z', { projections: true });

        /* r and theta labels on the drawing itself */
        var ctx = p.ctx;
        ctx.font = 'italic 12px ' + SANS;
        ctx.fillStyle = C.z;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        if (r > 0.35) {
            var mx = p.cx + (p.toX(a) - p.cx) * 0.55;
            var my = p.cy + (p.toY(b) - p.cy) * 0.55;
            var nx = -(p.toY(b) - p.cy), ny = (p.toX(a) - p.cx);
            var nl = Math.hypot(nx, ny) || 1;
            ctx.fillText('r', mx + 13 * nx / nl, my + 13 * ny / nl);
        }
        var arcR = Math.min(46, Math.max(26, r * p.k * 0.45));
        if (Math.abs(td) > 0.12) {
            ctx.fillStyle = '#1257a6';
            ctx.fillText('θ', p.cx + (arcR + 13) * Math.cos(td / 2),
                                   p.cy - (arcR + 13) * Math.sin(td / 2));
        }

        /* readout */
        var exactR = niceRadius(r), exactT = niceAngle(th);
        this.f.rect.textContent = 'z = ' + rectStr(a, b);
        this.f.r.textContent = fmt(r, 4) + (exactR ? '  = ' + exactR : '');
        this.f.th.textContent = fmt(td, 4) + ' rad' +
            (exactT && this.principal ? '  = ' + exactT : '');
        this.f.deg.textContent = fmt(degOf(td), 2) + '°';
        this.f.polar.textContent = fmt(r, 4) + ' · e^(j' + fmt(td, 4) + ')' +
            '   =   ' + fmt(r, 4) + ' ∠ ' + fmt(degOf(td), 2) + '°';

        if (Math.abs(a) < 1e-9) {
            this.f.naive.textContent = 'undefined (a = 0)';
        } else {
            this.f.naive.textContent = fmt(Math.atan(b / a), 4) + ' rad = ' +
                fmt(degOf(Math.atan(b / a)), 2) + '°';
        }

        /* the whole point of the widget */
        var naive = Math.abs(a) < 1e-9 ? NaN : Math.atan(b / a);
        var disagrees = a < 0 && (Math.abs(a) > 1e-9 || Math.abs(b) > 1e-9);
        if (Math.abs(a) < 1e-9 && Math.abs(b) > 1e-9) {
            this.warnEl.hidden = false;
            this.warnEl.innerHTML = '<strong>a = 0.</strong> The ratio b/a blows up, ' +
                'so atan(b/a) cannot be evaluated at all, while atan2(b, a) returns ' +
                fmt(degOf(th), 2) + '° without complaint.';
        } else if (disagrees) {
            this.warnEl.hidden = false;
            this.warnEl.innerHTML = '<strong>atan(b/a) = ' + fmt(degOf(naive), 2) +
                '°, but the true argument is ' + fmt(degOf(th), 2) + '°.</strong> ' +
                'The point is in quadrant ' + (b >= 0 ? 'II' : 'III') +
                ', where a &lt; 0, and tan has period π &mdash; so the ratio b/a cannot ' +
                'tell this point apart from its reflection through the origin. ' +
                'atan2 uses the signs of a and b separately and adds ' +
                (b >= 0 ? '+π' : '−π') + '.';
        } else {
            this.warnEl.hidden = true;
        }
    };

    /* ============================================================ widget 2 */

    function ArgandMultiply(root) {
        var self = this;
        this.z = { a: 1.5, b: 0.6 };
        this.w = { a: 0.9, b: 0.9 };

        var body = el('div', 'viz-body');
        var wrapEl = el('div', 'viz-canvas-wrap');
        wrapEl.style.maxWidth = '560px';
        wrapEl.style.margin = '0 auto';
        var canvas = document.createElement('canvas');
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label',
            'Complex plane showing two draggable factors and their product.');
        wrapEl.appendChild(canvas);
        body.appendChild(wrapEl);

        this.plot = new ArgandPlot(canvas, { ry: 3.0, maxH: 420, ratio: 0.82 });

        var controls = el('div', 'viz-controls');
        controls.appendChild(el('span', 'viz-label', 'set w to'));
        [['j  (quarter turn)', 0, 1],
         ['e^(jπ/4)', Math.SQRT1_2, Math.SQRT1_2],
         ['2  (pure scaling)', 2, 0],
         ['-1  (half turn)', -1, 0],
         ['0.7 e^(-jπ/3)', 0.35, -0.7 * Math.sqrt(3) / 2]].forEach(function (p) {
            controls.appendChild(button(p[0], function () {
                self.w.a = p[1]; self.w.b = p[2]; self.draw();
            }));
        });
        body.appendChild(controls);

        var ro = el('div', 'viz-readout');
        this.f = {
            z: readoutField(ro, 'z'),
            w: readoutField(ro, 'w'),
            p: readoutField(ro, 'z w  (rectangular)'),
            mod: readoutField(ro, 'moduli multiply'),
            arg: readoutField(ro, 'arguments add')
        };
        body.appendChild(ro);

        body.appendChild(legend([[C.z, 'z'], [C.w, 'w'], [C.prod, 'z·w']]));
        body.appendChild(el('p', 'viz-note',
            'Drag either factor. The orange product is z stretched by |w| and ' +
            'rotated by ∠w &mdash; which is exactly what the polar form says ' +
            'and what the rectangular form hides.'));

        root.appendChild(body);

        makeDraggable(canvas, this.plot, function () { return [self.z, self.w]; },
            function (i, a, b) {
                var t = i === 0 ? self.z : self.w;
                t.a = clamp(Math.round(a * 20) / 20, -3.5, 3.5);
                t.b = clamp(Math.round(b * 20) / 20, -2.9, 2.9);
                self.draw();
            });

        this.canvas = canvas;
    }

    ArgandMultiply.prototype.resize = function () { this.plot.resize(); };

    ArgandMultiply.prototype.draw = function () {
        var p = this.plot, z = this.z, w = this.w;
        var pa = z.a * w.a - z.b * w.b;
        var pb = z.a * w.b + z.b * w.a;

        var rz = Math.hypot(z.a, z.b), tz = Math.atan2(z.b, z.a);
        var rw = Math.hypot(w.a, w.b), tw = Math.atan2(w.b, w.a);
        var rp = Math.hypot(pa, pb), tp = Math.atan2(pb, pa);

        p.frame();
        p.wedge(tz, 34, C.arcZ, C.z);
        p.wedge(tz + tw, 52, C.arcP, C.prod);
        p.vector(w.a, w.b, C.w, 'w', { thin: true, small: true });
        p.vector(z.a, z.b, C.z, 'z', { thin: true, small: true });
        p.vector(pa, pb, C.prod, 'zw', {});

        this.f.z.textContent = rectStr(z.a, z.b, 2) + '  =  ' + fmt(rz, 3) +
            ' ∠ ' + fmt(degOf(tz), 1) + '°';
        this.f.w.textContent = rectStr(w.a, w.b, 2) + '  =  ' + fmt(rw, 3) +
            ' ∠ ' + fmt(degOf(tw), 1) + '°';
        this.f.p.textContent = rectStr(pa, pb, 2);
        this.f.mod.textContent = fmt(rz, 3) + ' × ' + fmt(rw, 3) + ' = ' + fmt(rp, 3);
        var sum = degOf(tz + tw), principal = degOf(tp);
        this.f.arg.textContent = fmt(degOf(tz), 1) + '° + ' + fmt(degOf(tw), 1) +
            '° = ' + fmt(sum, 1) + '°' +
            (Math.abs(sum - principal) > 0.05 ?
                '  ≡ ' + fmt(principal, 1) + '° (mod 360°)' : '');
    };

    /* ============================================================ widget 3 */

    function Phasor(root) {
        var self = this;
        this.A = 1.0;
        this.f0 = 0.45;
        this.phi = Math.PI / 4;
        this.showIm = false;
        this.t = 0;
        this.last = null;
        this.wantsPlay = !reduceMotion;
        this.visible = true;
        this.raf = null;

        var body = el('div', 'viz-body');
        var wrapEl = el('div', 'viz-canvas-wrap');
        var canvas = document.createElement('canvas');
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label',
            'A rotating phasor on the left and the cosine it traces on the right.');
        canvas.style.cursor = 'default';
        wrapEl.appendChild(canvas);
        body.appendChild(wrapEl);
        this.canvas = canvas;

        var controls = el('div', 'viz-controls');
        this.playBtn = button(this.wantsPlay ? '❚❚ pause' : '▶ play', function (b) {
            self.wantsPlay = !self.wantsPlay;
            b.innerHTML = self.wantsPlay ? '❚❚ pause' : '▶ play';
            if (self.wantsPlay && self.visible) self.start(); else self.stop();
        });
        controls.appendChild(this.playBtn);
        controls.appendChild(slider('A', 0.2, 1.4, 0.05, this.A,
            function (v) { return fmt(v, 2); },
            function (v) { self.A = v; self.draw(); }).node);
        controls.appendChild(slider('ω/2π', 0.1, 1.2, 0.05, this.f0,
            function (v) { return fmt(v, 2) + ' Hz'; },
            function (v) { self.f0 = v; self.draw(); }).node);
        controls.appendChild(slider('φ', -Math.PI, Math.PI, Math.PI / 24, this.phi,
            function (v) { return fmt(degOf(v), 0) + '°'; },
            function (v) { self.phi = v; self.draw(); }).node);
        controls.appendChild(checkbox('also show the sine (imaginary part)', false, function (v) {
            self.showIm = v; self.draw();
        }).node);
        body.appendChild(controls);

        root.appendChild(body);

        var ro = el('div', 'viz-readout');
        this.f = {
            phasor: readoutField(ro, 'phasor (constant)'),
            rot: readoutField(ro, 'rotating vector'),
            re: readoutField(ro, 'real part = the signal'),
            im: readoutField(ro, 'imaginary part')
        };
        body.appendChild(ro);

        body.appendChild(el('p', 'viz-note',
            'The phasor A·e^(jφ) never moves; e^(jωt) does the ' +
            'spinning. Everything a real sinusoid carries — amplitude and ' +
            'phase — is the polar form of that one fixed complex number.'));

        root.__viz = this;
        if (io) io.observe(root);
    }

    Phasor.prototype.resize = function () {
        var w = measure(this.canvas);
        var h = Math.round(clamp(w * 0.42, 240, 320));
        this.ctx = fitCanvas(this.canvas, w, h);
        this.w = w; this.h = h;
        this.circR = Math.min(h * 0.38, w * 0.18);
        this.ccx = this.circR + 34;
        this.ccy = h / 2;
        this.tx0 = this.ccx + this.circR + 26;
        this.txw = Math.max(40, w - this.tx0 - 12);
        this.pps = this.txw / 4.5;           /* pixels per second of history */
    };

    Phasor.prototype.start = function () {
        if (this.raf) return;
        var self = this;
        this.last = null;
        var tick = function (ts) {
            if (self.last === null) self.last = ts;
            var dt = Math.min(0.05, (ts - self.last) / 1000);
            self.last = ts;
            self.t += dt;
            self.draw();
            self.raf = window.requestAnimationFrame(tick);
        };
        this.raf = window.requestAnimationFrame(tick);
    };

    Phasor.prototype.stop = function () {
        if (this.raf) { window.cancelAnimationFrame(this.raf); this.raf = null; }
        this.last = null;
    };

    Phasor.prototype.draw = function () {
        var ctx = this.ctx, w = this.w, h = this.h;
        var om = TAU * this.f0;
        var ang = om * this.t + this.phi;
        var re = this.A * Math.cos(ang), im = this.A * Math.sin(ang);
        var R = this.circR, A = this.A;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, w, h);

        /* --- left: the phasor --- */
        ctx.save();
        ctx.setLineDash([3, 4]);
        ctx.strokeStyle = C.circle;
        ctx.beginPath();
        ctx.arc(this.ccx, this.ccy, R * A / 1.4, 0, TAU);
        ctx.stroke();
        ctx.restore();

        ctx.strokeStyle = C.axis;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(this.ccx - R - 8, this.ccy); ctx.lineTo(this.ccx + R + 8, this.ccy);
        ctx.moveTo(this.ccx, this.ccy - R - 8); ctx.lineTo(this.ccx, this.ccy + R + 8);
        ctx.stroke();

        var s = R / 1.4;                       /* world unit -> pixels */
        var px = this.ccx + re * s, py = this.ccy - im * s;

        /* projections onto the two axes */
        ctx.save();
        ctx.setLineDash([2, 3]);
        ctx.lineWidth = 1;
        ctx.strokeStyle = C.re;
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(px, this.ccy); ctx.stroke();
        ctx.strokeStyle = C.im;
        ctx.beginPath(); ctx.moveTo(px, py); ctx.lineTo(this.ccx, py); ctx.stroke();
        ctx.restore();

        ctx.strokeStyle = C.z;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(this.ccx, this.ccy); ctx.lineTo(px, py);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(px, py, 5, 0, TAU);
        ctx.fillStyle = C.z; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke();

        ctx.fillStyle = C.text;
        ctx.font = 'italic 10px ' + SANS;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        ctx.fillText('Re', this.ccx + R + 2, this.ccy + 4);

        /* --- right: the waveform, newest sample at the left edge --- */
        var mid = this.ccy;
        ctx.strokeStyle = C.axis;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(this.tx0, mid); ctx.lineTo(this.tx0 + this.txw, mid);
        ctx.stroke();

        function trace(fn, color, wide) {
            ctx.strokeStyle = color;
            ctx.lineWidth = wide ? 2 : 1.4;
            ctx.beginPath();
            for (var i = 0; i <= this.txw; i++) {
                var tau = this.t - i / this.pps;
                var v = fn(om * tau + this.phi) * this.A;
                var yy = mid - v * s;
                if (i === 0) ctx.moveTo(this.tx0, yy); else ctx.lineTo(this.tx0 + i, yy);
            }
            ctx.stroke();
        }
        if (this.showIm) trace.call(this, Math.sin, C.im, false);
        trace.call(this, Math.cos, C.re, true);

        /* the link between the two panels */
        ctx.save();
        ctx.setLineDash([2, 3]);
        ctx.strokeStyle = C.re;
        ctx.globalAlpha = 0.75;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(px, py); ctx.lineTo(this.tx0, mid - re * s);
        ctx.stroke();
        ctx.restore();

        ctx.beginPath();
        ctx.arc(this.tx0, mid - re * s, 4, 0, TAU);
        ctx.fillStyle = C.re; ctx.fill();
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5; ctx.stroke();

        ctx.fillStyle = C.tick;
        ctx.font = '10px ' + SANS;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        ctx.fillText('now', this.tx0 + 2, mid + 6);
        ctx.textAlign = 'right';
        ctx.fillText('earlier →', this.tx0 + this.txw - 2, mid + 6);

        this.f.phasor.textContent = 'A e^(jφ) = ' + fmt(A, 2) + ' ∠ ' +
            fmt(degOf(this.phi), 1) + '° = ' +
            rectStr(A * Math.cos(this.phi), A * Math.sin(this.phi), 3);
        this.f.rot.textContent = fmt(A, 2) + ' ∠ ' + fmt(degOf(wrap2pi(ang)), 1) + '°';
        this.f.re.textContent = 'A cos(ωt + φ) = ' + fmt(re, 3);
        this.f.im.textContent = 'A sin(ωt + φ) = ' + fmt(im, 3);
    };

    /* ============================================================ widget 4 */

    function Helix(root) {
        var self = this;
        this.az = -1.05;                 /* radians */
        this.elv = 0.38;
        this.turns = 3;
        this.t = 0;
        this.last = null;
        this.wantsPlay = !reduceMotion;
        this.visible = true;
        this.raf = null;
        this.shadows = true;

        var body = el('div', 'viz-body');
        var wrapEl = el('div', 'viz-canvas-wrap');
        var canvas = document.createElement('canvas');
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label',
            'Three dimensional plot of the complex exponential as a helix, with ' +
            'its cosine and sine shadows. Drag to rotate.');
        wrapEl.appendChild(canvas);
        body.appendChild(wrapEl);
        this.canvas = canvas;

        var controls = el('div', 'viz-controls');
        controls.appendChild(el('span', 'viz-label', 'view'));
        var views = [
            ['3D', -1.05, 0.38],
            ['down the t axis → circle', 0, 0],
            ['t-Re plane → cosine', -Math.PI / 2, Math.PI / 2],
            ['t-Im plane → sine', -Math.PI / 2, 0]
        ];
        this.viewBtns = views.map(function (v) {
            var b = button(v[0], function () {
                self.az = v[1]; self.elv = v[2];
                self.markView();
                self.fit(); self.draw();
            });
            controls.appendChild(b);
            return b;
        });
        body.appendChild(controls);

        var controls2 = el('div', 'viz-controls');
        this.playBtn = button(this.wantsPlay ? '❚❚ pause' : '▶ play', function (b) {
            self.wantsPlay = !self.wantsPlay;
            b.innerHTML = self.wantsPlay ? '❚❚ pause' : '▶ play';
            if (self.wantsPlay && self.visible) self.start(); else self.stop();
        });
        controls2.appendChild(this.playBtn);
        controls2.appendChild(checkbox('show shadows', true, function (v) {
            self.shadows = v; self.draw();
        }).node);
        body.appendChild(controls2);

        body.appendChild(legend([
            [C.z, 'e^(jωt)'],
            [C.re, 'shadow on the t-Re wall = cos ωt'],
            [C.im, 'shadow on the t-Im wall = sin ωt']
        ]));
        body.appendChild(el('p', 'viz-note',
            'Drag to orbit, or use the view buttons. The curve never changes ' +
            '— only where you stand. Cosine and sine are not two different ' +
            'functions here; they are two shadows of one rotation.'));

        root.appendChild(body);

        var dragging = false, lx = 0, ly = 0;
        canvas.style.cursor = 'grab';
        canvas.addEventListener('pointerdown', function (e) {
            dragging = true; lx = e.clientX; ly = e.clientY;
            canvas.setPointerCapture(e.pointerId);
            canvas.style.cursor = 'grabbing';
        });
        canvas.addEventListener('pointermove', function (e) {
            if (!dragging) return;
            self.az -= (e.clientX - lx) * 0.008;
            self.elv = clamp(self.elv + (e.clientY - ly) * 0.008, -Math.PI / 2, Math.PI / 2);
            lx = e.clientX; ly = e.clientY;
            self.markView();
            self.fit();
            self.draw();
            e.preventDefault();
        });
        function stopDrag() { dragging = false; canvas.style.cursor = 'grab'; }
        canvas.addEventListener('pointerup', stopDrag);
        canvas.addEventListener('pointercancel', stopDrag);

        this.markView();
        root.__viz = this;
        if (io) io.observe(root);
    }

    Helix.prototype.markView = function () {
        var self = this;
        var targets = [[-1.05, 0.38], [0, 0], [-Math.PI / 2, Math.PI / 2], [-Math.PI / 2, 0]];
        this.viewBtns.forEach(function (b, i) {
            var hit = Math.abs(self.az - targets[i][0]) < 1e-6 &&
                      Math.abs(self.elv - targets[i][1]) < 1e-6;
            b.classList.toggle('is-active', hit);
        });
    };

    /* world point (t, Re, Im) -> screen offsets in world units */
    Helix.prototype.project = function (x, y, z) {
        var ca = Math.cos(this.az), sa = Math.sin(this.az);
        var ce = Math.cos(this.elv), se = Math.sin(this.elv);
        return {
            u: -sa * x + ca * y,
            v: -ca * se * x - sa * se * y + ce * z
        };
    };

    Helix.prototype.resize = function () {
        var w = measure(this.canvas);
        var h = Math.round(clamp(w * 0.46, 260, 400));
        this.ctx = fitCanvas(this.canvas, w, h);
        this.w = w; this.h = h;
        this.fit();
    };

    /* Scale so the whole box fits, recomputed only when the camera moves. */
    Helix.prototype.fit = function () {
        var T = this.turns, self = this;
        var umin = Infinity, umax = -Infinity, vmin = Infinity, vmax = -Infinity;
        [0, T].forEach(function (x) {
            [-1, 1].forEach(function (y) {
                [-1, 1].forEach(function (z) {
                    var p = self.project(x, y, z);
                    umin = Math.min(umin, p.u); umax = Math.max(umax, p.u);
                    vmin = Math.min(vmin, p.v); vmax = Math.max(vmax, p.v);
                });
            });
        });
        var pad = 34;
        var su = (this.w - 2 * pad) / Math.max(0.001, umax - umin);
        var sv = (this.h - 2 * pad) / Math.max(0.001, vmax - vmin);
        this.s = Math.min(su, sv);
        this.ox = this.w / 2 - this.s * (umin + umax) / 2;
        this.oy = this.h / 2 + this.s * (vmin + vmax) / 2;
    };

    Helix.prototype.px = function (x, y, z) {
        var p = this.project(x, y, z);
        return { x: this.ox + this.s * p.u, y: this.oy - this.s * p.v };
    };

    Helix.prototype.start = Phasor.prototype.start;
    Helix.prototype.stop = Phasor.prototype.stop;

    Helix.prototype.draw = function () {
        var ctx = this.ctx, self = this, T = this.turns;
        ctx.clearRect(0, 0, this.w, this.h);
        ctx.fillStyle = C.bg;
        ctx.fillRect(0, 0, this.w, this.h);

        function line(p0, p1, color, width, dash) {
            var a = self.px(p0[0], p0[1], p0[2]);
            var b = self.px(p1[0], p1[1], p1[2]);
            ctx.save();
            if (dash) ctx.setLineDash(dash);
            ctx.strokeStyle = color;
            ctx.lineWidth = width || 1;
            ctx.beginPath();
            ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
            ctx.stroke();
            ctx.restore();
        }

        function curve(fn, color, width) {
            var N = 480;
            ctx.strokeStyle = color;
            ctx.lineWidth = width;
            ctx.beginPath();
            for (var i = 0; i <= N; i++) {
                var t = T * i / N;
                var q = fn(t);
                var p = self.px(q[0], q[1], q[2]);
                if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
            }
            ctx.stroke();
        }

        /* bounding walls, drawn faintly so the shadows have something to sit on */
        [[-1, -1], [1, -1], [1, 1], [-1, 1]].forEach(function (c, i) {
            var d = [[-1, -1], [1, -1], [1, 1], [-1, 1]][(i + 1) % 4];
            line([0, c[0], c[1]], [0, d[0], d[1]], C.grid, 1);
            line([T, c[0], c[1]], [T, d[0], d[1]], C.grid, 1);
            line([0, c[0], c[1]], [T, c[0], c[1]], C.grid, 1);
        });

        /* axes */
        line([0, 0, 0], [T, 0, 0], C.axis, 1.3);
        line([0, -1.15, 0], [0, 1.15, 0], C.axis, 1.3);
        line([0, 0, -1.15], [0, 0, 1.15], C.axis, 1.3);

        var lab = [[[T, 0, 0], 't', C.text], [[0, 1.3, 0], 'Re', C.re], [[0, 0, 1.3], 'Im', C.im]];
        ctx.font = 'italic 11px ' + SANS;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        lab.forEach(function (L) {
            var p = self.px(L[0][0], L[0][1], L[0][2]);
            ctx.fillStyle = L[2];
            ctx.fillText(L[1], p.x, p.y);
        });

        /* the unit circle at t = 0: what you see looking down the time axis */
        ctx.strokeStyle = C.circle;
        ctx.lineWidth = 1;
        ctx.save();
        ctx.setLineDash([3, 4]);
        ctx.beginPath();
        for (var i = 0; i <= 120; i++) {
            var a = TAU * i / 120;
            var p = self.px(0, Math.cos(a), Math.sin(a));
            if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
        ctx.restore();

        if (this.shadows) {
            curve(function (t) { return [t, Math.cos(TAU * t), -1]; }, C.re, 1.6);
            curve(function (t) { return [t, 1, Math.sin(TAU * t)]; }, C.im, 1.6);
        }

        curve(function (t) { return [t, Math.cos(TAU * t), Math.sin(TAU * t)]; }, C.z, 2.2);

        /* the travelling point and its two drop lines */
        var tm = (this.t * 0.35) % T;
        var y = Math.cos(TAU * tm), z = Math.sin(TAU * tm);
        if (this.shadows) {
            line([tm, y, z], [tm, y, -1], C.re, 1, [2, 3]);
            line([tm, y, z], [tm, 1, z], C.im, 1, [2, 3]);
            [[tm, y, -1, C.re], [tm, 1, z, C.im]].forEach(function (m) {
                var p = self.px(m[0], m[1], m[2]);
                ctx.beginPath();
                ctx.arc(p.x, p.y, 3.5, 0, TAU);
                ctx.fillStyle = m[3];
                ctx.fill();
            });
        }
        line([tm, 0, 0], [tm, y, z], C.z, 1.4, [4, 3]);
        var mp = this.px(tm, y, z);
        ctx.beginPath();
        ctx.arc(mp.x, mp.y, 5.5, 0, TAU);
        ctx.fillStyle = C.z;
        ctx.fill();
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = 2;
        ctx.stroke();
    };

    /* ================================================================ boot */

    function init() {
        var nodes = document.querySelectorAll('[data-viz]');
        Array.prototype.forEach.call(nodes, function (node) {
            var kind = node.getAttribute('data-viz');
            var wd = null;
            try {
                if (kind === 'argand-explorer') wd = new ArgandExplorer(node);
                else if (kind === 'argand-multiply') wd = new ArgandMultiply(node);
                else if (kind === 'phasor') wd = new Phasor(node);
                else if (kind === 'helix') wd = new Helix(node);
            } catch (err) {
                if (window.console) console.error('viz init failed for ' + kind, err);
                return;
            }
            if (!wd) return;
            widgets.push(wd);
            wd.resize();
            wd.draw();
            if (wd.wantsPlay && wd.start) wd.start();
        });
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
