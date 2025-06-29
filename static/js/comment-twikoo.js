document.addEventListener('DOMContentLoaded', () => {
  // 动态加载 KaTeX CSS
  const katexCSS = document.createElement('link');
  katexCSS.rel = 'stylesheet';
  katexCSS.href = 'https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.css';
  katexCSS.integrity = 'sha384-AfEj0r4/OFrOo5t7NnNe46zW/tFgW6x/bCJG8FqQCEo3+Aro6EYUG4+cU+KJWu/X';
  katexCSS.crossOrigin = 'anonymous';
  document.head.appendChild(katexCSS);

  // 动态加载 KaTeX JS
  const katexJS = document.createElement('script');
  katexJS.src = 'https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/katex.min.js';
  katexJS.integrity = 'sha384-g7c+Jr9ZivxKLnZTDUhnkOnsh30B4H0rpLUpJ4jAIKs4fnJI+sEnkvrMWph2EDg4';
  katexJS.crossOrigin = 'anonymous';
  katexJS.defer = true;
  document.head.appendChild(katexJS);

  // 动态加载 KaTeX auto-render
  const katexAutoRender = document.createElement('script');
  katexAutoRender.src = 'https://cdn.jsdelivr.net/npm/katex@0.12.0/dist/contrib/auto-render.min.js';
  katexAutoRender.integrity = 'sha384-mll67QQFJfxn0IYznZYonOWZ644AWYC+Pt2cHqMaRhXVrursRwvLnLaebdGIlYNa';
  katexAutoRender.crossOrigin = 'anonymous';
  katexAutoRender.defer = true;
  document.head.appendChild(katexAutoRender);

  // 动态加载 Twikoo JS
  const twikooJS = document.createElement('script');
  twikooJS.src = 'https://lf3-cdn-tos.bytecdntp.com/cdn/expire-1-M/twikoo/1.4.18/twikoo.min.js';
  document.head.appendChild(twikooJS);

  // Twikoo 初始化，等待 twikooJS 加载完毕
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
        hljs: {
          enable: true,
          style: 'github'
        },
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
});
