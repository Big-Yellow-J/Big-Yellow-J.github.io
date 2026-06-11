// 评论区滚入视口后再加载 KaTeX + Twikoo，节省首屏 ~800KB
(function () {
  const ENV_ID = 'https://meek-halva-a18f24.netlify.app/.netlify/functions/twikoo';
  const target = document.getElementById('tcomment');
  if (!target) return;

  let loaded = false;
  function load() {
    if (loaded) return;
    loaded = true;

    // extMath 文章已在 head 加载过 KaTeX，这里跳过避免重复请求
    if (!document.querySelector('link[href*="katex"]')) {
      const katexCSS = document.createElement('link');
      katexCSS.rel = 'stylesheet';
      katexCSS.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.10/dist/katex.min.css';
      katexCSS.crossOrigin = 'anonymous';
      document.head.appendChild(katexCSS);
    }
    if (!window.katex && !document.querySelector('script[src*="katex.min.js"]')) {
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
    }

    const twikooJS = document.createElement('script');
    twikooJS.src = 'https://cdn.jsdelivr.net/npm/twikoo@1.6.41/dist/twikoo.min.js';
    document.head.appendChild(twikooJS);

    // CDN 加载失败 / 初始化异常时给出可见提示，而不是评论区永远空白
    const showError = () => {
      target.innerHTML = '<p class="comment-load-error">评论加载失败，请刷新页面重试</p>';
    };
    twikooJS.onerror = showError;

    twikooJS.onload = () => {
      try {
      twikoo.init({
        envId: ENV_ID,
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
      // 填充 meta 栏评论数（与 vercount 一样懒填充，加载前显示 "–"）
      if (typeof twikoo.getCommentsCount === 'function') {
        twikoo.getCommentsCount({
          envId: ENV_ID,
          urls: [window.location.pathname],
          includeReply: false
        }).then((res) => {
          const el = document.getElementById('comment-count');
          if (el && res && res[0]) el.textContent = res[0].count;
        }).catch(() => {});
      }
      } catch (e) {
        showError();
      }
    };
  }

  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      if (entries.some(e => e.isIntersecting)) {
        load();
        io.disconnect();
      }
    }, { rootMargin: '150px 0px' });
    io.observe(target);
  } else {
    load();
  }
})();
