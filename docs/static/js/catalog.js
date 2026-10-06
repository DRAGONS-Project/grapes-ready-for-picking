// Draws the catalog from data/*.json. Cell text is inserted as text, never as markup.
(function () {
  'use strict';

  var TABLES = ['datasets_found', 'characterization_scene', 'characterization_record', 'readiness'];
  var CLASS_ORDER = { ripe: 0, ripening: 1, unripe: 2 };

  function el(tag, attrs) {
    var node = document.createElement(tag);
    if (attrs) {
      Object.keys(attrs).forEach(function (k) {
        if (k === 'text') node.textContent = attrs[k];
        else if (k === 'class') node.className = attrs[k];
        else node.setAttribute(k, attrs[k]);
      });
    }
    for (var i = 2; i < arguments.length; i++) {
      var c = arguments[i];
      if (c === null || c === undefined || c === false) continue;
      node.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return node;
  }

  function getJSON(path) {
    return fetch(path).then(function (r) {
      if (!r.ok) throw new Error(path + ': ' + r.status);
      return r.json();
    });
  }

  function badge(cls) {
    return el('span', { class: 'badge ' + cls, text: cls });
  }

  // ---- catalog: the datasets found, with the readiness class and the characterization joined on
  // the dataset key ----

  function Catalog(data) {
    var found = data.datasets_found;
    var ready = data.readiness;
    var scene = data.characterization_scene;
    var record = data.characterization_record;
    var classOf = {};
    ready.row_ids.forEach(function (id, i) { classOf[id] = ready.row_classes[i]; });

    // the Characterized column becomes the Readiness column: a badge for the characterized
    // datasets, the stated reason for the others ("no (single images)" reads "not characterized:
    // single images")
    var iChar = found.columns.indexOf('Characterized');
    var shown = found.columns.map(function (c, j) { return j; }).filter(function (j) { return j !== iChar; });
    var headers = [found.columns[shown[0]], 'Readiness'].concat(shown.slice(1).map(function (j) { return found.columns[j]; }));
    var READINESS = 1;

    var rows = found.rows.map(function (cells, i) {
      var id = found.row_ids[i];
      return { index: i, id: id, cells: cells, links: found.row_links[i] || [],
               cls: id ? classOf[id] : null, reason: id ? '' : notCharacterized(cells[iChar]),
               fullName: id ? scene.column_names[id] : '' };
    });
    function notCharacterized(cell) {
      var m = /^no \((.*)\)$/.exec(cell);
      return m ? 'not characterized: ' + m[1] : cell;
    }
    var state = { query: '', sort: READINESS, dir: 1, open: {} };

    var wrap = document.getElementById('catalog-wrap');
    var search = document.getElementById('catalog-search');
    var count = document.getElementById('catalog-count');

    var table = el('table', { class: 'table is-hoverable data-table', id: 'catalog-table' });
    table.appendChild(el('caption', { class: 'is-sr-only', text: 'Public vineyard datasets and their readiness for reconstruction' }));
    var headRow = el('tr');
    headers.forEach(function (name, h) {
      var th = el('th', { scope: 'col', 'aria-sort': 'none' });
      var b = el('button', { type: 'button', class: 'sort-button' }, name, el('span', { class: 'arrow', 'aria-hidden': 'true', text: '↕' }));
      b.addEventListener('click', function () {
        if (state.sort === h) state.dir = -state.dir; else { state.sort = h; state.dir = 1; }
        render();
      });
      th.appendChild(b);
      headRow.appendChild(th);
    });
    table.appendChild(el('thead', null, headRow));
    var tbody = el('tbody');
    table.appendChild(tbody);
    wrap.appendChild(table);

    function sortKey(r, h) {
      if (h === READINESS) return r.cls ? CLASS_ORDER[r.cls] : 9;
      return r.cells[shown[h === 0 ? 0 : h - 1]].toLowerCase();
    }

    function matches(r, q) {
      if (!q) return true;
      var hay = r.cells.join(' ') + ' ' + r.reason + ' ' + r.fullName + ' ' + (r.cls || '') + ' ' + (r.id || '');
      return hay.toLowerCase().indexOf(q) !== -1;
    }

    function detail(r) {
      var c = scene.columns.indexOf(r.id);
      var groups = [];
      [scene, record].forEach(function (t) {
        t.rows.forEach(function (row, i) {
          var g = t.row_groups[i];
          var last = groups[groups.length - 1];
          if (!last || last.name !== g) groups.push(last = { name: g, items: [] });
          last.items.push([row[0], row[c]]);
        });
      });
      var grid = el('div', { class: 'detail-groups' });
      groups.forEach(function (g) {
        var dl = el('dl');
        g.items.forEach(function (kv) { dl.appendChild(el('dt', { text: kv[0] })); dl.appendChild(el('dd', { text: kv[1] })); });
        grid.appendChild(el('section', { class: 'detail-group' }, el('h5', { text: g.name }), dl));
      });
      return el('div', { class: 'detail-inner' }, el('h4', { class: 'detail-name', text: r.fullName }), grid);
    }

    function render() {
      var q = state.query.trim().toLowerCase();
      var list = rows.filter(function (r) { return matches(r, q); });
      list.sort(function (a, b) {
        var x = sortKey(a, state.sort), y = sortKey(b, state.sort);
        return x === y ? a.index - b.index : (x < y ? -1 : 1) * state.dir;
      });
      headRow.querySelectorAll('th').forEach(function (th, h) {
        th.setAttribute('aria-sort', state.sort === h ? (state.dir === 1 ? 'ascending' : 'descending') : 'none');
        th.querySelector('.arrow').textContent = state.sort === h ? (state.dir === 1 ? '↑' : '↓') : '↕';
      });
      tbody.textContent = '';
      list.forEach(function (r) {
        var tr = el('tr', { class: 'dataset-row' });
        var nameCell = el('th', { scope: 'row' });
        var detailId = 'detail-' + r.index;
        if (r.id) {
          var btn = el('button', { type: 'button', class: 'expand-button', 'aria-expanded': String(!!state.open[r.id]), 'aria-controls': detailId },
            el('span', { class: 'chev', 'aria-hidden': 'true', text: '▶' }), r.cells[0]);
          btn.addEventListener('click', function () { state.open[r.id] = !state.open[r.id]; render(); });
          nameCell.appendChild(btn);
        } else {
          nameCell.appendChild(el('span', { class: 'dataset-name', text: r.cells[0] }));
        }
        if (r.links.length) {
          var src = el('div', { class: 'source-links' });
          r.links.forEach(function (l) {
            src.appendChild(el('a', { href: l.url, rel: 'noopener', text: l.kind === 'data' ? 'deposit ↗' : 'paper ↗' }));
          });
          nameCell.appendChild(src);
        }
        tr.appendChild(nameCell);
        tr.appendChild(el('td', null, r.cls ? badge(r.cls) : el('span', { class: 'reason', text: r.reason })));
        shown.slice(1).forEach(function (j) { tr.appendChild(el('td', { text: r.cells[j] })); });
        tbody.appendChild(tr);
        if (r.id && state.open[r.id]) {
          tbody.appendChild(el('tr', { class: 'detail', id: detailId }, el('td', { colspan: String(headers.length) }, detail(r))));
        }
      });
      count.textContent = 'Showing ' + list.length + ' of ' + rows.length + ' datasets';
    }

    search.addEventListener('input', function () { state.query = search.value; render(); });
    render();
    var notes = document.getElementById('catalog-notes');
    (found.notes || []).forEach(function (n) { notes.appendChild(el('li', { text: n })); });
  }

  // ---- the paper button stays hidden and the archive button disabled until meta.json carries a link ----

  function Meta(meta) {
    function enable(id, href) {
      var b = document.getElementById(id);
      if (!b || !href) return;
      var a = el('a', { href: href, class: b.className, id: id });
      while (b.firstChild) a.appendChild(b.firstChild);
      b.parentNode.replaceChild(a, b);
    }
    var paper = meta.links && meta.links.paper;
    if (paper) {
      var p = document.getElementById('btn-paper');
      p.href = paper;
      p.parentNode.hidden = false;
    }
    var doi = meta.links && meta.links.archive_doi;
    enable('btn-archive', doi ? (doi.indexOf('http') === 0 ? doi : 'https://doi.org/' + doi) : null);
  }

  document.addEventListener('DOMContentLoaded', function () {
    getJSON('text/meta.json').then(Meta).catch(function () {});
    Promise.all(TABLES.map(function (n) { return getJSON('data/' + n + '.json'); })).then(function (all) {
      var data = {};
      TABLES.forEach(function (n, i) { data[n] = all[i]; });
      Catalog(data);
    }).catch(function (err) {
      document.getElementById('catalog-wrap').appendChild(
        el('p', { class: 'notice', text: 'The catalog could not be loaded (' + err.message + ').' }));
    });
  });
})();
