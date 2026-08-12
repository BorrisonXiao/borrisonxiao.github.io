/* Interview Prep landing page: tab deep-linking + client-side filtering.
 *
 * Tab state lives in the URL fragment as the bare topic slug (e.g.
 * /interview-prep/#linear-algebra). The slug deliberately does NOT match any
 * element id on the page, so the browser will not try to scroll to it.
 */
(function () {
    var tabs = document.getElementById('prep-tabs');
    if (!tabs || typeof jQuery === 'undefined') return;

    var $ = jQuery;
    var links = Array.prototype.slice.call(tabs.querySelectorAll('a[data-slug]'));

    function slugOf(link) { return link.getAttribute('data-slug'); }

    function activate(slug, updateHash) {
        var link = links.filter(function (l) { return slugOf(l) === slug; })[0];
        if (!link) return false;
        $(link).tab('show');
        if (updateHash && window.history && window.history.replaceState) {
            window.history.replaceState(null, '', '#' + slug);
        }
        // Keep the freshly-selected pill in view on narrow screens.
        if (link.scrollIntoView) {
            link.scrollIntoView({ block: 'nearest', inline: 'nearest' });
        }
        return true;
    }

    // Restore the tab named in the fragment, on load and on back/forward.
    function fromHash() {
        var slug = (window.location.hash || '').replace(/^#/, '');
        if (slug) activate(slug, false);
    }
    fromHash();
    window.addEventListener('hashchange', fromHash);

    links.forEach(function (link) {
        $(link).on('shown.bs.tab', function () {
            if (window.history && window.history.replaceState) {
                window.history.replaceState(null, '', '#' + slugOf(link));
            }
        });
    });

    /* ---- Sorting -------------------------------------------------------- */

    /* Liquid already emits every list newest-first, so this only has to handle
     * the flip. Cards carry data-date as a sortable YYYYMMDDhhmmss string. */
    var sortBtns = Array.prototype.slice.call(document.querySelectorAll('[data-prep-sort]'));

    function applySort(dir) {
        Array.prototype.forEach.call(document.querySelectorAll('[data-prep-list]'), function (list) {
            var items = Array.prototype.slice.call(list.querySelectorAll('[data-prep-item]'));
            items.sort(function (a, b) {
                var da = a.getAttribute('data-date') || '';
                var db = b.getAttribute('data-date') || '';
                if (da === db) return 0;
                var cmp = da < db ? -1 : 1;
                return dir === 'oldest' ? cmp : -cmp;
            });
            items.forEach(function (i) { list.appendChild(i); });
        });
    }

    sortBtns.forEach(function (btn) {
        btn.addEventListener('click', function () {
            sortBtns.forEach(function (o) { o.classList.remove('is-active'); });
            btn.classList.add('is-active');
            applySort(btn.getAttribute('data-prep-sort'));
        });
    });

    /* ---- Filtering ------------------------------------------------------ */

    var filter = document.getElementById('prep-filter');
    if (!filter) return;

    var items = Array.prototype.slice.call(document.querySelectorAll('[data-prep-item]'));
    var counts = {};
    links.forEach(function (l) {
        counts[slugOf(l)] = l.querySelector('.prep-tab-count');
    });

    function paneSlug(el) {
        var pane = el.closest('.tab-pane');
        return pane ? pane.id.replace(/^prep-pane-/, '') : null;
    }

    function applyFilter() {
        var q = filter.value.trim().toLowerCase();
        var visible = {};

        items.forEach(function (el) {
            var hit = !q || (el.getAttribute('data-search') || '').indexOf(q) !== -1;
            el.hidden = !hit;
            var slug = paneSlug(el);
            if (slug) visible[slug] = (visible[slug] || 0) + (hit ? 1 : 0);
        });

        Object.keys(counts).forEach(function (slug) {
            var badge = counts[slug];
            if (!badge) return;
            if (q) {
                var n = visible[slug] || 0;
                badge.textContent = n;
                badge.classList.toggle('prep-tab-count-zero', n === 0);
            } else {
                badge.textContent = badge.getAttribute('data-total') || badge.textContent;
                badge.classList.toggle('prep-tab-count-zero', badge.textContent === '0');
            }
        });

        Array.prototype.forEach.call(document.querySelectorAll('[data-prep-no-match]'), function (msg) {
            var slug = paneSlug(msg);
            msg.hidden = !q || (visible[slug] || 0) > 0;
        });
    }

    // Remember the unfiltered counts so they can be restored.
    Object.keys(counts).forEach(function (slug) {
        if (counts[slug]) counts[slug].setAttribute('data-total', counts[slug].textContent.trim());
    });

    filter.addEventListener('input', applyFilter);
    filter.addEventListener('search', applyFilter);
})();
