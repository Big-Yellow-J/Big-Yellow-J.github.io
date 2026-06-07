(function () {
  var ENDPOINT = 'https://events.vercount.one/api/v2/log';
  var CACHE_KEY_PREFIX = 'bk-pv:';
  var CACHE_TTL = 5 * 60 * 1000;
  var CONCURRENCY = 6;

  function fmt(n) {
    if (n == null || isNaN(n)) return '–';
    if (n >= 10000) return (n / 1000).toFixed(1) + 'k';
    return String(n);
  }

  function readCache(url) {
    try {
      var raw = sessionStorage.getItem(CACHE_KEY_PREFIX + url);
      if (!raw) return null;
      var obj = JSON.parse(raw);
      if (Date.now() - obj.t > CACHE_TTL) return null;
      return obj.v;
    } catch (e) { return null; }
  }

  function writeCache(url, v) {
    try {
      sessionStorage.setItem(CACHE_KEY_PREFIX + url, JSON.stringify({ t: Date.now(), v: v }));
    } catch (e) {}
  }

  function fetchOne(url) {
    var cached = readCache(url);
    if (cached !== null) return Promise.resolve(cached);
    return fetch(ENDPOINT + '?url=' + encodeURIComponent(url), { credentials: 'omit' })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (j) {
        var pv = j && typeof j.page_pv === 'number' ? j.page_pv : null;
        if (pv !== null) writeCache(url, pv);
        return pv;
      })
      .catch(function () { return null; });
  }

  function runQueue(items) {
    var i = 0, active = 0;
    return new Promise(function (resolve) {
      function next() {
        if (i >= items.length && active === 0) return resolve();
        while (active < CONCURRENCY && i < items.length) {
          var item = items[i++];
          active++;
          fetchOne(item.url).then(function (pv) {
            item.span.textContent = fmt(pv);
            active--;
            next();
          });
        }
      }
      next();
    });
  }

  function collect() {
    var nodes = document.querySelectorAll('.card-views[data-page-url]');
    var seen = {};
    var items = [];
    nodes.forEach(function (el) {
      var url = el.dataset.pageUrl;
      var num = el.querySelector('.card-views-num');
      if (!url || !num) return;
      if (seen[url]) {
        items.push({ url: url, span: num });
      } else {
        seen[url] = true;
        items.push({ url: url, span: num });
      }
    });
    return items;
  }

  function start() {
    var items = collect();
    if (items.length === 0) return;
    runQueue(items);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
