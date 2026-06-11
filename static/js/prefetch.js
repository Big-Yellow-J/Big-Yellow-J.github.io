// 鼠标悬停文章链接 ≥ 150ms → 预拉 HTML，下一篇打开"瞬间"完成
(function () {
  if (!('connection' in navigator) || !navigator.connection.saveData === true) {
    // saveData=true 时尊重用户省流偏好
  }
  if (navigator.connection && navigator.connection.saveData) return;

  const prefetched = new Set();
  let hoverTimer = null;

  function shouldPrefetch(a) {
    if (!a.href) return false;
    if (a.target === '_blank') return false;
    if (a.host !== location.host) return false;
    if (a.pathname === location.pathname) return false;
    if (prefetched.has(a.href)) return false;
    if (a.href.startsWith('javascript:')) return false;
    if (a.dataset.noPrefetch === 'true') return false;
    return true;
  }

  function prefetch(href) {
    if (prefetched.has(href)) return;
    prefetched.add(href);
    const link = document.createElement('link');
    link.rel = 'prefetch';
    link.href = href;
    link.as = 'document';
    document.head.appendChild(link);
  }

  document.addEventListener('mouseover', function (e) {
    const a = e.target.closest('a[href]');
    if (!a || !shouldPrefetch(a)) return;
    clearTimeout(hoverTimer);
    hoverTimer = setTimeout(function () { prefetch(a.href); }, 150);
  });

  document.addEventListener('mouseout', function () {
    clearTimeout(hoverTimer);
  });

  document.addEventListener('touchstart', function (e) {
    const a = e.target.closest('a[href]');
    if (a && shouldPrefetch(a)) prefetch(a.href);
  }, { passive: true });

  // 文章页上下篇大概率被点击：空闲时段提前预取
  function prefetchNav() {
    document.querySelectorAll('.navigation-buttons a.nav-button').forEach(function (a) {
      if (shouldPrefetch(a)) prefetch(a.href);
    });
  }
  if ('requestIdleCallback' in window) requestIdleCallback(prefetchNav, { timeout: 4000 });
  else setTimeout(prefetchNav, 3000);
})();
