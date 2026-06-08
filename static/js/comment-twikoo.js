// 评论区滚入视口后再加载 KaTeX + Twikoo，节省首屏 ~800KB
(function () {
  const target = document.getElementById('tcomment');
  if (!target) return;

  let loaded = false;
  function load() {
    if (loaded) return;
    loaded = true;

    const katexCSS = document.createElement('link');
    katexCSS.rel = 'stylesheet';
    katexCSS.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css';
    katexCSS.crossOrigin = 'anonymous';
    document.head.appendChild(katexCSS);

    const katexJS = document.createElement('script');
    katexJS.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.js';
    katexJS.crossOrigin = 'anonymous';
    katexJS.defer = true;
    document.head.appendChild(katexJS);

    const katexAutoRender = document.createElement('script');
    katexAutoRender.src = 'https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/contrib/auto-render.min.js';
    katexAutoRender.crossOrigin = 'anonymous';
    katexAutoRender.defer = true;
    document.head.appendChild(katexAutoRender);

    const twikooJS = document.createElement('script');
    twikooJS.src = 'https://cdn.jsdelivr.net/npm/twikoo@1.6.41/dist/twikoo.min.js';
    document.head.appendChild(twikooJS);

    twikooJS.onload = () => {
      twikoo.init({
        envId: 'https://meek-halva-a18f24.netlify.app/.netlify/functions/twikoo',
        el: '#tcomment',
        katex: {
          delimiters: [
            { left: '$$', right: '$$', display: true },
            { left: '$', right: '$', display: false },
            { left: '\\(', right: '\\)', display: false },
            { left: '\\[', right: '\\]', display: true }
          ],
          throwOnError: false
        },
        markdown: {
          enable: true,
          toc: true,
          hljs: { enable: true, style: 'github' },
          mathjax: true,
          lazyLoad: true
        },
        editor: {
          mode: 'tab',
          preview: true,
          height: '200px',
          placeholder: '✨ 支持Markdown语法：**加粗**、`代码`、[链接]()、> 引用等...'
        },
        lang: 'zh-CN',
        path: window.location.pathname,
        avatar: 'retro',
        meta: ['nick', 'mail', 'link'],
        pageSize: 15,
        maxLength: 2000,
        emojiCDN: 'https://cdn.bootcdn.net/ajax/libs/twemoji/14.0.2/svg/',
        commentCount: true,
        requiredFields: ['nick', 'mail']
      });
    };
  }

  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      if (entries.some(e => e.isIntersecting)) {
        load();
        io.disconnect();
      }
    }, { rootMargin: '400px 0px' });
    io.observe(target);
  } else {
    load();
  }
})();
