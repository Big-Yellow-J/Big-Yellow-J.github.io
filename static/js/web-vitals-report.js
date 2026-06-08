// 采集 Core Web Vitals（LCP/CLS/INP/FCP/TTFB）并通过 umami custom event 上报
(function () {
  if (!window.requestIdleCallback) {
    window.requestIdleCallback = function (cb) { return setTimeout(cb, 1); };
  }
  requestIdleCallback(function () {
    const s = document.createElement('script');
    s.type = 'module';
    s.textContent = `
      import { onCLS, onINP, onLCP, onFCP, onTTFB }
        from 'https://cdn.jsdelivr.net/npm/web-vitals@4.2.4/dist/web-vitals.attribution.js?module';
      function report(metric) {
        var value = Math.round(metric.name === 'CLS' ? metric.value * 1000 : metric.value);
        if (window.umami && typeof window.umami.track === 'function') {
          window.umami.track('web-vitals', {
            name: metric.name,
            value: value,
            rating: metric.rating,
            id: metric.id,
            path: location.pathname
          });
        }
      }
      onCLS(report); onINP(report); onLCP(report); onFCP(report); onTTFB(report);
    `;
    document.head.appendChild(s);
  });
})();
