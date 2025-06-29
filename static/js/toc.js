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

  // 创建目录头部
  const tocHeader = document.createElement('div');
  tocHeader.className = 'toc-header';

  const tocTitle = document.createElement('span');
  tocTitle.className = 'toc-title';
  tocTitle.textContent = '目录';

  const tocToggle = document.createElement('button');
  tocToggle.className = 'toc-toggle';
  tocToggle.innerHTML = '<i class="fas fa-times"></i>';

  tocHeader.appendChild(tocTitle);
  tocHeader.appendChild(tocToggle);
  toc.appendChild(tocHeader);

  const tocList = document.createElement('ul');
  tocList.className = 'toc-list';

  headings.forEach((heading, index) => {
    if (!heading.id) {
      heading.id = 'heading-' + index;
    }

    const listItem = document.createElement('li');
    const link = document.createElement('a');
    link.href = '#' + heading.id;
    link.textContent = heading.textContent || '无标题';
    link.setAttribute('data-target-id', heading.id);
    listItem.appendChild(link);

    let level = parseInt(heading.tagName.replace('H', '')) - 2;
    let currentList = tocList;

    for (let i = 0; i < level; i++) {
      let lastLi = currentList.lastElementChild;
      if (!lastLi || !lastLi.querySelector('ul')) {
        const newUl = document.createElement('ul');
        newUl.className = 'toc-list';
        lastLi?.appendChild(newUl);
        currentList = newUl;
      } else {
        currentList = lastLi.querySelector('ul');
      }
    }
    currentList.appendChild(listItem);
  });

  toc.appendChild(tocList);

  // 透明汉堡按钮
  const floatingToggle = document.createElement('button');
  floatingToggle.className = 'toc-toggle-floating';
  floatingToggle.innerHTML = '<i class="fas fa-bars"></i>';
  document.body.appendChild(floatingToggle);
  floatingToggle.style.display = 'none';

  function toggleTOC() {
    toc.classList.toggle('collapsed');
    const isCollapsed = toc.classList.contains('collapsed');
    floatingToggle.style.display = isCollapsed ? 'block' : 'none';
  }

  tocToggle.addEventListener('click', toggleTOC);
  floatingToggle.addEventListener('click', toggleTOC);

  // 平滑滚动
  toc.querySelectorAll('a').forEach(link => {
    link.addEventListener('click', function (e) {
      e.preventDefault();
      const targetId = this.getAttribute('data-target-id');
      const target = document.getElementById(targetId);
      if (target) {
        const headerHeight = document.querySelector('header')?.offsetHeight || 0;
        window.scrollTo({
          top: target.offsetTop - headerHeight - 20,
          behavior: 'smooth'
        });
      }
    });
  });

  // 高亮当前章节
  window.addEventListener('scroll', function () {
    let fromTop = window.scrollY + 100;
    let activeSet = false;
    headings.forEach(heading => {
      if (heading.offsetTop <= fromTop && !activeSet) {
        toc.querySelectorAll('a').forEach(link => link.classList.remove('active'));
        const link = toc.querySelector(`a[data-target-id="${heading.id}"]`);
        if (link) {
          link.classList.add('active');
          activeSet = true;
        }
      }
    });
    if (!activeSet) {
      toc.querySelectorAll('a').forEach(link => link.classList.remove('active'));
    }
  });

  // 避免 footer 遮挡
  const footer = document.querySelector('footer');
  const tocHeight = toc.offsetHeight;
  window.addEventListener('scroll', function () {
    const footerTop = footer ? footer.getBoundingClientRect().top : Infinity;
    const windowHeight = window.innerHeight;
    if (footerTop < windowHeight + tocHeight) {
      toc.style.top = `${footerTop - tocHeight - 20}px`;
    } else {
      toc.style.top = '100px';
    }
  });
});
