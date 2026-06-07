(function () {
  var BASE = 'https://api.umami.is/v1';
  var WEBSITE_ID = 'b1118f16-642d-4b6b-99b0-fb2b4b34134c';
  var API_KEY = 'api_8AGNdpDIHeQ4w3c2L0AfmH0esc2PbaKv';
  var CACHE_KEY = 'bk-umami-pv-map';
  var CACHE_TTL = 5 * 60 * 1000;

  function fmt(n) {
    if (n == null || isNaN(n)) return '–';
    if (n >= 10000) return (n / 1000).toFixed(1) + 'k';
    return String(n);
  }

  function pathOf(u) {
    try { return new URL(u, location.origin).pathname; }
    catch (e) { return u; }
  }

  function readCache() {
    try {
      var raw = sessionStorage.getItem(CACHE_KEY);
      if (!raw) return null;
      var obj = JSON.parse(raw);
      if (Date.now() - obj.t > CACHE_TTL) return null;
      return obj.m;
    } catch (e) { return null; }
  }

  function writeCache(map) {
    try {
      sessionStorage.setItem(CACHE_KEY, JSON.stringify({ t: Date.now(), m: map }));
    } catch (e) {}
  }

  function fetchMap() {
    var cached = readCache();
    if (cached) return Promise.resolve(cached);
    var url = BASE + '/websites/' + WEBSITE_ID + '/metrics'
      + '?startAt=0&endAt=' + Date.now()
      + '&type=url&limit=500';
    return fetch(url, {
      headers: { 'x-umami-api-key': API_KEY, 'Accept': 'application/json' },
      credentials: 'omit'
    })
      .then(function (r) { return r.ok ? r.json() : []; })
      .then(function (arr) {
        var map = {};
        (arr || []).forEach(function (it) {
          if (it && it.x != null) map[it.x] = it.y;
        });
        writeCache(map);
        return map;
      })
      .catch(function () { return {}; });
  }

  function render(map) {
    document.querySelectorAll('.card-views[data-page-url]').forEach(function (el) {
      var num = el.querySelector('.card-views-num');
      if (!num) return;
      num.textContent = fmt(map[pathOf(el.dataset.pageUrl)] || 0);
    });
    var single = document.getElementById('umami-page-pv');
    if (single && single.dataset.pageUrl) {
      single.textContent = fmt(map[pathOf(single.dataset.pageUrl)] || 0);
    }
  }

  function start() {
    var hasCards = document.querySelector('.card-views[data-page-url]');
    var hasSingle = document.getElementById('umami-page-pv');
    if (!hasCards && !hasSingle) return;
    fetchMap().then(render);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
