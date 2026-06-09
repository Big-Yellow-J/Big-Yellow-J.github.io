/* 生活栏目：点击卡片图片打开 PhotoSwipe，每个卡片独立相册
   - 等图片 naturalWidth/Height 就绪后再实例化，确保 PhotoSwipe 知道尺寸
   - 卡片内的 prev/next/dot 按钮位于 .life-card-slide 外，互不干扰
*/
import PhotoSwipeLightbox from 'https://cdn.jsdelivr.net/npm/photoswipe@5.4.4/dist/photoswipe-lightbox.esm.min.js';

const GALLERY_SELECTOR = '[data-life-card-gallery]';

function wrap(img) {
  if (img.dataset.pswpWrapped === '1') return true;
  if (img.closest('a[data-pswp-width]')) {
    img.dataset.pswpWrapped = '1';
    return true;
  }
  const w = img.naturalWidth || img.getAttribute('width') || 0;
  const h = img.naturalHeight || img.getAttribute('height') || 0;
  if (!w || !h) return false;

  const a = document.createElement('a');
  a.href = img.currentSrc || img.src;
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

function initOne(galleryEl) {
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
    });
    lb.init();
  }

  imgs.forEach((img) => {
    if (img.complete && img.naturalWidth) {
      wrap(img);
      if (--pending === 0) ready();
    } else {
      img.addEventListener('load', () => { wrap(img); if (--pending === 0) ready(); }, { once: true });
      img.addEventListener('error', () => { if (--pending === 0) ready(); }, { once: true });
    }
  });
}

function init() {
  document.querySelectorAll(GALLERY_SELECTOR).forEach(initOne);
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
