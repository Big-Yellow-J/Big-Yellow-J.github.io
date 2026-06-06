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

  // 平滑滚动
  toc.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', function (e) {
      e.preventDefault();
      const target = document.getElementById(this.getAttribute('data-target-id'));
      if (!target) return;
      const headerHeight = document.querySelector('header')?.offsetHeight || 0;
      window.scrollTo({ top: target.offsetTop - headerHeight - 20, behavior: 'smooth' });
      if (isMobile()) closeMobile();
    });
  });

  // 滚动高亮
  let ticking = false;
  window.addEventListener('scroll', function () {
    if (ticking) return;
    ticking = true;
    requestAnimationFrame(() => {
      const fromTop = window.scrollY + 100;
      let activeId = null;
      headings.forEach(h => { if (h.offsetTop <= fromTop) activeId = h.id; });
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
