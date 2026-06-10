document.addEventListener('DOMContentLoaded', function () {
  const toc = document.getElementById('toc');
  if (!toc) return;

  const content = document.querySelector('.post.page-post');
  if (!content) return;

  const headings = content.querySelectorAll('h2, h3, h4, h5, h6');
  if (headings.length === 0) {
    toc.style.display = 'none';
    return;
  }

  const isMobile = () => window.matchMedia('(max-width: 768px)').matches;

  // 头部
  const tocHeader = document.createElement('div');
  tocHeader.className = 'toc-header';

  const tocTitle = document.createElement('span');
  tocTitle.className = 'toc-title';
  tocTitle.textContent = '目录';

  const tocToggle = document.createElement('button');
  tocToggle.className = 'toc-toggle';
  tocToggle.setAttribute('aria-label', '关闭目录');
  tocToggle.innerHTML = '<i class="fas fa-times"></i>';

  tocHeader.appendChild(tocTitle);
  tocHeader.appendChild(tocToggle);
  toc.appendChild(tocHeader);

  // 列表
  const tocList = document.createElement('ul');
  tocList.className = 'toc-list';

  headings.forEach((heading, index) => {
    if (!heading.id) heading.id = 'heading-' + index;

    const text = (heading.textContent || '无标题').trim();
    const listItem = document.createElement('li');
    const link = document.createElement('a');
    link.href = '#' + heading.id;
    link.textContent = text;
    link.title = text;
    link.setAttribute('data-target-id', heading.id);
    listItem.appendChild(link);

    let level = parseInt(heading.tagName.replace('H', '')) - 2;
    let currentList = tocList;
    for (let i = 0; i < level; i++) {
      let lastLi = currentList.lastElementChild;
      if (!lastLi || !lastLi.querySelector('ul')) {
        const newUl = document.createElement('ul');
        newUl.className = 'toc-list';
        lastLi && lastLi.appendChild(newUl);
        currentList = newUl;
      } else {
        currentList = lastLi.querySelector('ul');
      }
    }
    currentList.appendChild(listItem);
  });

  toc.appendChild(tocList);

  // 浮动按钮（桌面端折叠态显示，移动端常驻）
  const floatingToggle = document.createElement('button');
  floatingToggle.className = 'toc-toggle-floating';
  floatingToggle.setAttribute('aria-label', '打开目录');
  floatingToggle.innerHTML = '<i class="fas fa-bars"></i>';
  document.body.appendChild(floatingToggle);

  // 遮罩（移动端抽屉用）
  let mask = null;
  function ensureMask() {
    if (mask) return mask;
    mask = document.createElement('div');
    mask.className = 'toc-mask';
    mask.addEventListener('click', closeMobile);
    document.body.appendChild(mask);
    return mask;
  }

  function openMobile() {
    document.body.classList.add('toc-open');
    const m = ensureMask();
    requestAnimationFrame(() => m.classList.add('is-open'));
  }
  function closeMobile() {
    document.body.classList.remove('toc-open');
    if (mask) mask.classList.remove('is-open');
  }

  function syncFloatingVisibility() {
    if (isMobile()) {
      floatingToggle.style.display = 'flex';
    } else {
      floatingToggle.style.display = toc.classList.contains('collapsed') ? 'block' : 'none';
    }
  }

  function onToggleClick() {
    if (isMobile()) {
      if (document.body.classList.contains('toc-open')) closeMobile();
      else openMobile();
    } else {
      toc.classList.toggle('collapsed');
      syncFloatingVisibility();
    }
  }

  tocToggle.addEventListener('click', function () {
    if (isMobile()) closeMobile();
    else { toc.classList.add('collapsed'); syncFloatingVisibility(); }
  });
  floatingToggle.addEventListener('click', onToggleClick);
  window.addEventListener('resize', syncFloatingVisibility);
  syncFloatingVisibility();

  // 平滑滚动：用 getBoundingClientRect 替代 offsetTop（避免嵌套 offsetParent 错算）
  // + 二次校正（图片懒加载导致 layout shift 后位置漂移）
  function scrollToHeading(target) {
    const headerHeight = document.querySelector('header')?.offsetHeight || 0;
    const offset = headerHeight + 20;
    const computeTop = () => target.getBoundingClientRect().top + window.scrollY - offset;
    window.scrollTo({ top: computeTop(), behavior: 'smooth' });
    // 700ms 后页面应已稳定，如位置仍漂移 > 4px 再补一次（一般是图片懒加载触发的 layout shift）
    setTimeout(() => {
      const drift = Math.abs(target.getBoundingClientRect().top - offset);
      if (drift > 4) window.scrollTo({ top: computeTop(), behavior: 'smooth' });
    }, 700);
    // 1500ms 后再补最后一次，兜底慢加载的图
    setTimeout(() => {
      const drift = Math.abs(target.getBoundingClientRect().top - offset);
      if (drift > 4) window.scrollTo({ top: computeTop(), behavior: 'auto' });
    }, 1500);
  }

  toc.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.getElementById(this.getAttribute('data-target-id'));
      if (!target) return;
      if (isMobile()) closeMobile();
      // 移动端关闭抽屉时 body 会有 transform 变化，等下一帧再算位置
      requestAnimationFrame(() => scrollToHeading(target));
      // 把 hash 写进 URL 不触发原生 hashchange 跳转
      history.replaceState(null, '', '#' + target.id);
    });
  });

  // 滚动高亮
  let ticking = false;
  window.addEventListener('scroll', function () {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      // getBoundingClientRect 每次实时取位置：嵌套 offsetParent 与图片懒加载导致的 layout shift 都不会算错
      let activeId = null;
      headings.forEach(h => { if (h.getBoundingClientRect().top <= 100) activeId = h.id; });
      toc.querySelectorAll('a').forEach(link => {
        link.classList.toggle('active', link.getAttribute('data-target-id') === activeId);
      });

      // 避免 footer 遮挡（仅桌面）
      if (!isMobile()) {
        const footer = document.querySelector('footer');
        const footerTop = footer ? footer.getBoundingClientRect().top : Infinity;
        const tocHeight = toc.offsetHeight;
        if (footerTop < window.innerHeight + tocHeight) {
          toc.style.top = `${footerTop - tocHeight - 20}px`;
        } else {
          toc.style.top = '100px';
        }
      } else {
        toc.style.top = '';
      }
      ticking = false;
    });
  });
});
