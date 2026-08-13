/* Interview Prep landing page: tab deep-linking, sorting, and filtering.
 *
 * Tab state lives in the URL fragment as the bare topic slug (e.g.
 * /interview-prep/#linear-algebra). The slug deliberately does NOT match any
 * element id on the page, so the browser will not try to scroll to it.
 *
 * Nothing here may depend on jQuery being loaded. Sorting and filtering are
 * plain DOM work; only switching a tab programmatically prefers Bootstrap's
 * plugin, and there is a native fallback for when it is unavailable. An
 * earlier version bailed out of the whole module when jQuery was missing,
 * which silently disabled sorting and filtering as well.
 */
(function () {
    'use strict';

    function all(sel, root) {
        return Array.prototype.slice.call((root || document).querySelectorAll(sel));
    }

    /* ---- Tabs ------------------------------------------------------------ */

    var tabs = document.getElementById('prep-tabs');
    var links = tabs ? all('a[data-slug]', tabs) : [];

    function slugOf(link) { return link.getAttribute('data-slug'); }

    function setHash(slug) {
        if (window.history && window.history.replaceState) {
            window.history.replaceState(null, '', '#' + slug);
        }
    }

    /* Bootstrap's own click handler already switches panes; this is only for
     * switching one programmatically, e.g. when restoring from the fragment. */
    function showTab(link) {
        if (typeof jQuery !== 'undefined' && jQuery.fn && jQuery.fn.tab) {
            jQuery(link).tab('show');
            return;
        }
        links.forEach(function (l) {
            l.classList.remove('active');
            l.setAttribute('aria-selected', 'false');
        });
        all('.tab-pane').forEach(function (p) { p.classList.remove('show', 'active'); });
        link.classList.add('active');
        link.setAttribute('aria-selected', 'true');
        var pane = document.getElementById('prep-pane-' + slugOf(link));
        if (pane) pane.classList.add('show', 'active');
    }

    function activate(slug, updateHash) {
        var link = links.filter(function (l) { return slugOf(l) === slug; })[0];
        if (!link) return false;
        showTab(link);
        if (updateHash) setHash(slug);
        if (link.scrollIntoView) link.scrollIntoView({ block: 'nearest', inline: 'nearest' });
        return true;
    }

    if (links.length) {
        var fromHash = function () {
            var slug = (window.location.hash || '').replace(/^#/, '');
            if (slug) activate(slug, false);
        };
        fromHash();
        window.addEventListener('hashchange', fromHash);

        links.forEach(function (link) {
            link.addEventListener('click', function () { setHash(slugOf(link)); });
        });
    }

    /* ---- Sorting --------------------------------------------------------- */

    /* Liquid already emits every list newest-first, so this only has to handle
     * the flip. Cards carry data-date as a sortable YYYYMMDDhhmmss string. */
    var sortBtns = all('[data-prep-sort]');

    function applySort(dir) {
        all('[data-prep-list]').forEach(function (list) {
            var items = all('[data-prep-item]', list);
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

    /* ---- Filtering ------------------------------------------------------- */

    var filter = document.getElementById('prep-filter');
    if (!filter) return;

    var items = all('[data-prep-item]');
    var counts = {};
    links.forEach(function (l) { counts[slugOf(l)] = l.querySelector('.prep-tab-count'); });

    /* Remember the unfiltered counts so they can be restored. */
    Object.keys(counts).forEach(function (slug) {
        if (counts[slug]) counts[slug].setAttribute('data-total', counts[slug].textContent.trim());
    });

    function paneSlug(el) {
        var pane = el.closest ? el.closest('.tab-pane') : null;
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

        all('[data-prep-no-match]').forEach(function (msg) {
            var slug = paneSlug(msg);
            msg.hidden = !q || (visible[slug] || 0) > 0;
        });
    }

    filter.addEventListener('input', applyFilter);
    filter.addEventListener('search', applyFilter);
})();
