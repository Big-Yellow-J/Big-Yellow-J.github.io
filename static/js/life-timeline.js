/* 生活栏目时间线 v4：卡片轮播 + 曲线视图 SVG 路径
   视图模式由 _config.yml 的 life_view 决定（写入 #life-timeline 的 data-view）
*/
(function () {
  'use strict';

  /* ---------- 卡片内多图轮播 ---------- */
  function initCardGallery(root) {
    var slides = root.querySelectorAll('.life-card-slide');
    var dots   = root.querySelectorAll('.life-card-dot');
    var prev   = root.querySelector('.life-card-prev');
    var next   = root.querySelector('.life-card-next');
    var total  = slides.length;
    if (total <= 1) return;
    var current = 0;

    function go(idx) {
      idx = (idx + total) % total;
      slides.forEach(function (s, i) { s.classList.toggle('is-active', i === idx); });
      dots.forEach(function (d, i) { d.classList.toggle('is-active', i === idx); });
      current = idx;
    }
    function bind(el, delta) {
      if (!el) return;
      el.addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        if (typeof delta === 'number') go(current + delta);
        else go(Number(this.dataset.idx) || 0);
      });
    }
    bind(prev, -1);
    bind(next, +1);
    dots.forEach(function (d) { bind(d, null); });

    var startX = 0, moved = false;
    root.addEventListener('touchstart', function (e) {
      startX = e.touches[0].clientX; moved = false;
    }, { passive: true });
    root.addEventListener('touchmove', function (e) {
      if (Math.abs(e.touches[0].clientX - startX) > 10) moved = true;
    }, { passive: true });
    root.addEventListener('touchend', function (e) {
      if (!moved) return;
      var dx = e.changedTouches[0].clientX - startX;
      if (Math.abs(dx) > 30) go(dx < 0 ? current + 1 : current - 1);
    });
  }

  /* ---------- 曲线视图：SVG 贝塞尔路径 ---------- */
  function recomputePath(timeline) {
    var svg  = timeline.querySelector('.life-timeline-path');
    var path = svg && svg.querySelector('path');
    if (!svg || !path) return;
    if (timeline.dataset.view !== 'curve') { path.setAttribute('d', ''); return; }

    var nodes = timeline.querySelectorAll('.life-node');
    if (!nodes.length) return;

    var tRect = timeline.getBoundingClientRect();
    var pts = [];
    nodes.forEach(function (n) {
      var dot = n.querySelector('.life-node-dot');
      if (!dot) return;
      var r = dot.getBoundingClientRect();
      pts.push({
        x: r.left + r.width / 2 - tRect.left,
        y: r.top + r.height / 2 - tRect.top
      });
    });
    if (pts.length < 2) { path.setAttribute('d', ''); return; }

    var d = 'M ' + pts[0].x.toFixed(1) + ' ' + pts[0].y.toFixed(1);
    for (var i = 1; i < pts.length; i++) {
      var p0 = pts[i - 1], p1 = pts[i];
      var cy = (p0.y + p1.y) / 2;
      d += ' C ' + p0.x.toFixed(1) + ' ' + cy.toFixed(1)
         + ', '  + p1.x.toFixed(1) + ' ' + cy.toFixed(1)
         + ', '  + p1.x.toFixed(1) + ' ' + p1.y.toFixed(1);
    }
    path.setAttribute('d', d);

    try {
      var len = path.getTotalLength();
      path.style.strokeDasharray  = len + ' ' + len;
      path.style.strokeDashoffset = len;
      path.getBoundingClientRect();
      path.style.transition = 'stroke-dashoffset 1.6s ease-out';
      path.style.strokeDashoffset = '0';
    } catch (e) { /* ignore */ }
  }

  function initTimelinePath(timeline) {
    if (timeline.dataset.view !== 'curve') return;
    var imgs = timeline.querySelectorAll('img');
    var pending = imgs.length;
    var fire = function () { recomputePath(timeline); };
    if (pending === 0) {
      fire();
    } else {
      imgs.forEach(function (img) {
        if (img.complete) {
          if (--pending === 0) fire();
        } else {
          img.addEventListener('load',  function () { if (--pending === 0) fire(); });
          img.addEventListener('error', function () { if (--pending === 0) fire(); });
        }
      });
      setTimeout(fire, 1500);
    }
    if (typeof ResizeObserver !== 'undefined') {
      new ResizeObserver(fire).observe(timeline);
    } else {
      window.addEventListener('resize', fire);
    }
  }

  function init() {
    var timeline = document.getElementById('life-timeline');
    if (!timeline) return;
    timeline.querySelectorAll('[data-life-card-gallery]').forEach(initCardGallery);
    initTimelinePath(timeline);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
