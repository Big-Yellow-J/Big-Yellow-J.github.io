/* 生活栏目前端入口（ES module）
   合并三块逻辑：
   1) initCardGallery：卡片内多图轮播（prev/next/dot + 触摸滑动 + 键盘）
   2) initTimelinePath：曲线视图的 SVG 贝塞尔连线
   3) initPhotoSwipe ：点击图片唤起 PhotoSwipe 全屏，每卡片一个独立相册
*/
import PhotoSwipeLightbox from 'https://cdn.jsdelivr.net/npm/photoswipe@5.4.4/dist/photoswipe-lightbox.esm.min.js';

const GALLERY_SELECTOR = '[data-life-card-gallery]';

/* ---------- 1) 卡片内多图轮播 ---------- */
function initCardGallery(root) {
  const slides = root.querySelectorAll('.life-card-slide');
  const dots = root.querySelectorAll('.life-card-dot');
  const prev = root.querySelector('.life-card-prev');
  const next = root.querySelector('.life-card-next');
  const total = slides.length;
  if (total <= 1) return;
  let current = 0;

  function go(idx) {
    idx = (idx + total) % total;
    slides.forEach((s, i) => s.classList.toggle('is-active', i === idx));
    dots.forEach((d, i) => d.classList.toggle('is-active', i === idx));
    current = idx;
  }
  function bind(el, delta) {
    if (!el) return;
    const trigger = (e) => {
      e.preventDefault();
      e.stopPropagation();
      if (typeof delta === 'number') go(current + delta);
      else go(Number(el.dataset.idx) || 0);
    };
    el.addEventListener('click', trigger);
    el.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ' || e.key === 'Spacebar') trigger(e);
    });
  }
  bind(prev, -1);
  bind(next, +1);
  dots.forEach((d) => bind(d, null));

  // 焦点在轮播内任意控件时支持左右方向键切图
  root.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') { e.preventDefault(); go(current - 1); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); go(current + 1); }
  });

  // 触摸滑动
  let startX = 0;
  let moved = false;
  root.addEventListener('touchstart', (e) => {
    startX = e.touches[0].clientX;
    moved = false;
  }, { passive: true });
  root.addEventListener('touchmove', (e) => {
    if (Math.abs(e.touches[0].clientX - startX) > 10) moved = true;
  }, { passive: true });
  root.addEventListener('touchend', (e) => {
    if (!moved) return;
    const dx = e.changedTouches[0].clientX - startX;
    if (Math.abs(dx) > 30) go(dx < 0 ? current + 1 : current - 1);
  });
}

/* ---------- 2) 曲线视图 SVG 贝塞尔路径 ---------- */
function recomputePath(timeline) {
  const svg = timeline.querySelector('.life-timeline-path');
  const path = svg && svg.querySelector('path');
  if (!svg || !path) return;
  if (timeline.dataset.view !== 'curve') {
    path.setAttribute('d', '');
    return;
  }

  const nodes = timeline.querySelectorAll('.life-node');
  if (!nodes.length) return;

  const tRect = timeline.getBoundingClientRect();
  const pts = [];
  nodes.forEach((n) => {
    const dot = n.querySelector('.life-node-dot');
    if (!dot) return;
    const r = dot.getBoundingClientRect();
    pts.push({
      x: r.left + r.width / 2 - tRect.left,
      y: r.top + r.height / 2 - tRect.top,
    });
  });
  if (pts.length < 2) {
    path.setAttribute('d', '');
    return;
  }

  let d = `M ${pts[0].x.toFixed(1)} ${pts[0].y.toFixed(1)}`;
  for (let i = 1; i < pts.length; i++) {
    const p0 = pts[i - 1];
    const p1 = pts[i];
    const cy = (p0.y + p1.y) / 2;
    d += ` C ${p0.x.toFixed(1)} ${cy.toFixed(1)}, ${p1.x.toFixed(1)} ${cy.toFixed(1)}, ${p1.x.toFixed(1)} ${p1.y.toFixed(1)}`;
  }
  path.setAttribute('d', d);

  try {
    const len = path.getTotalLength();
    path.style.strokeDasharray = `${len} ${len}`;
    path.style.strokeDashoffset = len;
    path.getBoundingClientRect();
    path.style.transition = 'stroke-dashoffset 1.6s ease-out';
    path.style.strokeDashoffset = '0';
  } catch (e) { /* ignore */ }
}

function initTimelinePath(timeline) {
  if (timeline.dataset.view !== 'curve') return;
  const imgs = timeline.querySelectorAll('img');
  let pending = imgs.length;
  const fire = () => recomputePath(timeline);
  if (pending === 0) {
    fire();
  } else {
    imgs.forEach((img) => {
      if (img.complete) {
        if (--pending === 0) fire();
      } else {
        img.addEventListener('load', () => { if (--pending === 0) fire(); });
        img.addEventListener('error', () => { if (--pending === 0) fire(); });
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

/* ---------- 3) PhotoSwipe：点图开全屏，每卡片独立相册 ---------- */
function wrapForPhotoSwipe(img) {
  if (img.dataset.pswpWrapped === '1') return true;
  if (img.closest('a[data-pswp-width]')) {
    img.dataset.pswpWrapped = '1';
    return true;
  }
  const w = img.naturalWidth || img.getAttribute('width') || 0;
  const h = img.naturalHeight || img.getAttribute('height') || 0;
  if (!w || !h) return false;

  const a = document.createElement('a');
  // 优先 data-pswp-full（原图），缩略图模式下避免打开缩略
  a.href = img.dataset.pswpFull || img.currentSrc || img.src;
  a.target = '_blank';
  a.rel = 'noopener';
  a.dataset.pswpWidth = String(w);
  a.dataset.pswpHeight = String(h);
  a.style.display = 'block';
  a.style.width = '100%';
  a.style.height = '100%';

  img.parentNode.insertBefore(a, img);
  a.appendChild(img);
  img.dataset.pswpWrapped = '1';
  return true;
}

function initPhotoSwipeForGallery(galleryEl) {
  const imgs = galleryEl.querySelectorAll('.life-card-slide img');
  if (!imgs.length) return;

  let pending = imgs.length;
  let inited = false;

  function ready() {
    if (inited) return;
    inited = true;
    const lb = new PhotoSwipeLightbox({
      gallery: galleryEl,
      children: 'a[data-pswp-width]',
      pswpModule: () => import('https://cdn.jsdelivr.net/npm/photoswipe@5.4.4/dist/photoswipe.esm.min.js'),
      bgOpacity: 0.92,
      showHideAnimationType: 'fade',
      wheelToZoom: true,
      counter: true,
      zoom: true,
      arrowKeys: true,
    });
    lb.init();
  }

  imgs.forEach((img) => {
    if (img.complete && img.naturalWidth) {
      wrapForPhotoSwipe(img);
      if (--pending === 0) ready();
    } else {
      img.addEventListener('load', () => { wrapForPhotoSwipe(img); if (--pending === 0) ready(); }, { once: true });
      img.addEventListener('error', () => { if (--pending === 0) ready(); }, { once: true });
    }
  });
}

/* ---------- 入口 ---------- */
function init() {
  const timeline = document.getElementById('life-timeline');
  if (!timeline) return;
  const galleries = timeline.querySelectorAll(GALLERY_SELECTOR);
  galleries.forEach(initCardGallery);
  galleries.forEach(initPhotoSwipeForGallery);
  initTimelinePath(timeline);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
