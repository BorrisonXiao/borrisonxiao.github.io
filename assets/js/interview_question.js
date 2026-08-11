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

    function sync(justOpened) {
        var open = solution ? solution.open : true;
        gated.forEach(function (g) { g.hidden = !open; });
        buildToc();
        /* Canvas widgets inside a closed <details> measured zero width, so tell
         * them to re-fit now that they have a real box. */
        if (justOpened) {
            window.setTimeout(function () {
                window.dispatchEvent(new Event('resize'));
            }, 60);
        }
    }

    /* Any collapsible on the page can hold headings or widgets, so re-sync on
     * all of them, not just the solution. */
    if (content) {
        Array.prototype.forEach.call(content.querySelectorAll('details'), function (d) {
            d.addEventListener('toggle', function () { sync(d.open); });
        });
    }

    /* Following a link to something inside a collapsed section should open it. */
    function openForHash() {
        var id = (location.hash || '').replace(/^#/, '');
        if (!id || !content) return;
        var target = document.getElementById(id);
        if (!target) return;
        var opened = false;
        var node = target;
        while (node && node !== content) {
            if (node.tagName === 'DETAILS' && !node.open) { node.open = true; opened = true; }
            node = node.parentNode;
        }
        if (opened) {
            sync(true);
            window.setTimeout(function () {
                window.scrollTo({ top: target.offsetTop - navbarHeight, behavior: 'smooth' });
            }, 80);
        }
    }
    window.addEventListener('hashchange', openForHash);
    document.addEventListener('click', function (e) {
        var a = e.target.closest ? e.target.closest('a[href^="#"]') : null;
        if (!a) return;
        /* Let the hash land first, then expand whatever it points into. */
        window.setTimeout(openForHash, 0);
    });

    sync(false);
    openForHash();
})();
