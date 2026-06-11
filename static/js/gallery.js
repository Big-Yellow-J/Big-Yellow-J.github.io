import PhotoSwipeLightbox from 'https://cdn.jsdelivr.net/npm/photoswipe@5.4.4/dist/photoswipe-lightbox.esm.min.js';

const ARTICLE_SELECTOR = '.post.page-post';
const IMG_SELECTOR = 'img.content-image';

function wrapImage(img) {
  if (img.dataset.pswpWrapped === '1') return;
  if (img.closest('a')) { img.dataset.pswpWrapped = '1'; return; }
  if (img.getAttribute('alt') === 'line') { img.dataset.pswpWrapped = '1'; return; }

  const w = img.getAttribute('width') || img.naturalWidth || 0;
  const h = img.getAttribute('height') || img.naturalHeight || 0;
  if (!w || !h) return;

  const link = document.createElement('a');
  link.href = img.currentSrc || img.src;
  link.target = '_blank';
  link.rel = 'noopener';
  link.dataset.pswpWidth = String(w);
  link.dataset.pswpHeight = String(h);

  img.parentNode.insertBefore(link, img);
  link.appendChild(img);
  img.dataset.pswpWrapped = '1';
}

function init() {
  const article = document.querySelector(ARTICLE_SELECTOR);
  if (!article) return;

  const imgs = article.querySelectorAll(IMG_SELECTOR);
  if (imgs.length === 0) return;

  let pending = 0;
  imgs.forEach(img => {
    if (img.complete && img.naturalWidth) {
      wrapImage(img);
    } else {
      pending++;
      img.addEventListener('load', () => { wrapImage(img); if (--pending === 0) refresh(); }, { once: true });
      img.addEventListener('error', () => { if (--pending === 0) refresh(); }, { once: true });
    }
  });

  const lightbox = new PhotoSwipeLightbox({
    gallery: ARTICLE_SELECTOR,
    children: 'a[data-pswp-width]',
    pswpModule: () => import('https://cdn.jsdelivr.net/npm/photoswipe@5.4.4/dist/photoswipe.esm.min.js'),
    bgOpacity: 0.92,
    showHideAnimationType: 'fade',
    wheelToZoom: true,
  });

  // 底部图注：取当前图片的 alt（占位的 "image" 不显示）
  lightbox.on('uiRegister', () => {
    lightbox.pswp.ui.registerElement({
      name: 'custom-caption',
      appendTo: 'root',
      onInit: (el, pswp) => {
        el.className = 'pswp__custom-caption';
        pswp.on('change', () => {
          const link = pswp.currSlide && pswp.currSlide.data && pswp.currSlide.data.element;
          const img = link && link.querySelector('img');
          const alt = img ? (img.getAttribute('alt') || '').trim() : '';
          el.textContent = alt && alt !== 'image' ? alt : '';
          el.hidden = !el.textContent;
        });
      }
    });
  });
  lightbox.init();

  function refresh() { /* PhotoSwipe v5 自动在 init 后基于 DOM 查询，无需显式刷新 */ }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', init);
} else {
  init();
}
