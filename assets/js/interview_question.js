/* Question pages: keep everything that gives the answer away behind the
 * "Show the solution" toggle, and keep the table of contents in sync with it.
 *
 * Replaces blog.js on this layout. Progressive enhancement: with JS off the
 * <details> still works natively and the gated panels simply stay visible.
 */
(function () {
    var solution = document.getElementById('solution');
    var gated = Array.prototype.slice.call(document.querySelectorAll('[data-q-gated]'));
    var toc = document.getElementById('blog-toc');
    var tocList = toc ? toc.querySelector('.blog-toc-list') : null;
    var content = document.querySelector('.blog-content');
    var navbarHeight = 90;

    /* ---- table of contents, rebuilt whenever visibility changes ---------- */

    function visibleHeadings() {
        if (!content) return [];
        return Array.prototype.filter.call(
            content.querySelectorAll('h1, h2, h3'),
            function (h) { return h.offsetParent !== null; }
        );
    }

    var tocLinks = [];
    var headings = [];

    function buildToc() {
        if (!tocList || !toc) return;
        headings = visibleHeadings();
        tocList.innerHTML = '';
        tocLinks = [];

        /* With the solution folded away there is essentially nothing to index,
         * so hide the sidebar rather than show a one-item list. */
        if (headings.length < 2) {
            toc.style.display = 'none';
            return;
        }
        toc.style.display = '';

        headings.forEach(function (heading, i) {
            if (!heading.id) heading.id = 'heading-' + i;
            heading.style.scrollMarginTop = navbarHeight + 'px';

            var li = document.createElement('li');
            var a = document.createElement('a');
            a.href = '#' + heading.id;
            a.textContent = heading.textContent;
            a.className = 'toc-' + heading.tagName.toLowerCase();
            a.addEventListener('click', function (e) {
                e.preventDefault();
                window.scrollTo({ top: heading.offsetTop - navbarHeight, behavior: 'smooth' });
                history.pushState(null, null, '#' + heading.id);
            });
            li.appendChild(a);
            tocList.appendChild(li);
            tocLinks.push(a);
        });
        updateActive();
    }

    function updateActive() {
        if (!tocLinks.length) return;
        var scrollPos = window.scrollY + navbarHeight + 10;
        var current = null;
        headings.forEach(function (h) {
            if (h.offsetTop <= scrollPos) current = h.id;
        });
        tocLinks.forEach(function (link) {
            link.classList.toggle('toc-active', link.getAttribute('href') === '#' + current);
        });
    }

    window.addEventListener('scroll', updateActive, { passive: true });

    /* ---- gating ---------------------------------------------------------- */

    function sync() {
        var open = solution ? solution.open : true;
        gated.forEach(function (g) { g.hidden = !open; });
        buildToc();
        /* Canvas widgets inside a closed <details> measured zero width, so tell
         * them to re-fit now that they have a real box. */
        if (open) {
            window.setTimeout(function () {
                window.dispatchEvent(new Event('resize'));
            }, 60);
        }
    }

    if (solution) {
        solution.addEventListener('toggle', sync);
        /* Following a link into a heading inside the solution should open it. */
        window.addEventListener('hashchange', function () {
            var target = document.getElementById((location.hash || '').replace(/^#/, ''));
            if (target && solution.contains(target) && !solution.open) solution.open = true;
        });
    }

    sync();
})();
