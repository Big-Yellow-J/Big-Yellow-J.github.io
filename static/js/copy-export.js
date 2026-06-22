/* 文章多平台复制导出：点击右下 copy-all 弹菜单
   - 通用 Markdown / CSDN / 博客园：Turndown 转 Markdown（点击时才动态加载库），KaTeX 公式还原为 $TeX$
   - 微信公众号：行内样式 HTML 写入剪贴板（text/html），代码高亮色取自当前 Prism 渲染，公式转 codecogs 图片 */
(function () {
  var btn = document.querySelector('.footer-btn.copy-all');
  var article = document.querySelector('.post-content');
  if (!btn || !article) return;

  /* ---------- 菜单 ---------- */
  var menu = document.createElement('div');
  menu.className = 'copy-menu';
  menu.hidden = true;
  menu.setAttribute('role', 'menu');
  menu.innerHTML =
    '<span class="copy-menu-title">复制全文为</span>' +
    '<button type="button" data-fmt="md">通用 Markdown</button>' +
    '<button type="button" data-fmt="csdn">CSDN</button>' +
    '<button type="button" data-fmt="cnblogs">博客园</button>' +
    '<button type="button" data-fmt="wechat">微信公众号</button>';
  document.body.appendChild(menu);

  btn.addEventListener('click', function (e) {
    e.stopPropagation();
    menu.hidden = !menu.hidden;
  });
  document.addEventListener('click', function (e) {
    if (!menu.hidden && !menu.contains(e.target)) menu.hidden = true;
  });
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') menu.hidden = true;
  });
  menu.addEventListener('click', function (e) {
    var fmt = e.target && e.target.dataset && e.target.dataset.fmt;
    if (!fmt) return;
    menu.hidden = true;
    exportAs(fmt).catch(function (err) {
      console.error(err);
      toast('复制失败：' + (err && err.message ? err.message : '未知错误'));
    });
  });

  function toast(msg) {
    if (window.blog && blog.toast) blog.toast(msg);
  }

  function meta() {
    // 优先用 data-url（page.url | absolute_url，带域名的规范地址）；
    // 兜底剥掉 hash/query，避免把 #锚点、?utm 之类带进原文链接
    return {
      title: article.dataset.title || (document.querySelector('h1') || {}).textContent || '无标题',
      url: article.dataset.url || (location.origin + location.pathname),
      date: article.dataset.date || '',
      author: article.dataset.author || '佚名'
    };
  }

  /* ---------- KaTeX：从渲染结果取回 TeX 源码 ---------- */
  function getTex(el) {
    var ann = el.querySelector('annotation[encoding="application/x-tex"]');
    return ann ? ann.textContent : null;
  }

  /* ---------- Turndown 按需加载（jsdelivr，比 unpkg 在国内稳） ---------- */
  var turndownReady = null;
  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      s.src = src;
      s.onload = resolve;
      s.onerror = function () { reject(new Error('脚本加载失败 ' + src)); };
      document.head.appendChild(s);
    });
  }
  function ensureTurndown() {
    if (window.TurndownService && window.gfm) return Promise.resolve();
    if (!turndownReady) {
      turndownReady = loadScript('https://cdn.jsdelivr.net/npm/turndown@7.1.1/dist/turndown.js')
        .then(function () {
          return loadScript('https://cdn.jsdelivr.net/npm/@truto/turndown-plugin-gfm@latest/dist/turndown-plugin-gfm.min.js');
        });
    }
    return turndownReady;
  }

  /* ---------- Markdown 管线（md / csdn / cnblogs 共用） ---------- */
  function buildMarkdown(platform) {
    return ensureTurndown().then(function () {
      var clone = article.cloneNode(true);
      clone.querySelectorAll('script,style,noscript,.line-numbers-rows,.copy-btn,.mermaid-zoom-btn').forEach(function (n) { n.remove(); });

      // 公式：先换占位符防 Turndown 转义下划线等字符，转完再还原
      var texArr = [];
      clone.querySelectorAll('.katex-display').forEach(function (el) {
        var tex = getTex(el);
        var token = '@@MATH' + texArr.length + '@@';
        texArr.push(tex != null ? '\n$$\n' + tex + '\n$$\n' : '');
        el.replaceWith(document.createTextNode(token));
      });
      clone.querySelectorAll('.katex').forEach(function (el) {
        var tex = getTex(el);
        var token = '@@MATH' + texArr.length + '@@';
        texArr.push(tex != null ? '$' + tex + '$' : '');
        el.replaceWith(document.createTextNode(token));
      });

      // 图床外的相对路径图片转绝对地址
      clone.querySelectorAll('img').forEach(function (img) {
        var src = img.getAttribute('src') || '';
        if (src && !/^(https?:|data:)/.test(src)) {
          try { img.setAttribute('src', new URL(src, window.location.origin).href); } catch (e) {}
        }
      });

      var td = new TurndownService({
        headingStyle: 'atx',
        bulletListMarker: '-',
        codeBlockStyle: 'fenced',
        fence: '```'
      });
      if (window.gfm) td.use(window.gfm);
      td.addRule('fencedCodeBlockWithLanguage', {
        filter: 'pre',
        replacement: function (content, node, options) {
          var codeNode = node.querySelector('code');
          if (codeNode) {
            var m = codeNode.className.match(/language-(\w+)/);
            return '\n\n' + options.fence + (m ? m[1] : '') + '\n' + codeNode.textContent.replace(/\n$/, '') + '\n' + options.fence + '\n\n';
          }
          return '\n\n' + content + '\n\n';
        }
      });

      var md = td.turndown(clone.innerHTML);
      md = md.replace(/@@MATH(\d+)@@/g, function (_, i) { return texArr[Number(i)] || ''; });

      var m = meta();
      var note = platform === 'csdn' || platform === 'cnblogs'
        ? '> 本文首发于个人博客 [' + m.url + '](' + m.url + ')，转载请注明出处。'
        : '转载请注明出处，感谢！';
      var header = ['# ' + m.title, '', '**作者**：' + m.author, '**原文链接**：[' + m.url + '](' + m.url + ')',
        '**发布日期**：' + m.date, '', note, '', '---', '', ''].join('\n');
      return header + md;
    });
  }

  /* ---------- 公众号管线：行内样式 HTML（取值与 post.css 亮色文章页一致） ---------- */
  var WX_SERIF = "'Noto Serif SC',Georgia,'STSong',serif";
  var WX_STYLE = {
    H1: 'font-family:' + WX_SERIF + ';font-size:20px;font-weight:bold;margin:22px 0 10px;color:#000;line-height:1.4;',
    H2: 'font-family:' + WX_SERIF + ';font-size:20px;font-weight:bold;margin:22px 0 10px;color:#000;line-height:1.4;',
    H3: 'font-family:' + WX_SERIF + ';font-size:18px;font-weight:bold;margin:18px 0 8px;color:#000;line-height:1.4;',
    H4: 'font-family:Arial,sans-serif;font-size:15px;font-weight:bold;margin:16px 0 6px;color:#000;',
    H5: 'font-family:Arial,sans-serif;font-size:15px;font-weight:bold;margin:16px 0 6px;color:#000;',
    H6: 'font-family:Arial,sans-serif;font-size:15px;font-weight:bold;margin:16px 0 6px;color:#000;',
    P: 'margin:5px 0;line-height:1.7;font-size:16px;color:#000;',
    BLOCKQUOTE: 'margin:8px 0;padding:2px 0 2px 15px;border-left:3px solid #caddee;background:#f7f7f7;color:#000;',
    UL: 'margin:0;padding-left:20px;',
    OL: 'margin:0;padding-left:20px;',
    LI: 'margin:0 0 4px;line-height:1.6;font-size:16px;color:#000;',
    A: 'color:#3366cc;font-weight:bold;text-decoration:none;border-bottom:1px solid rgba(51,102,204,0.35);',
    IMG: 'max-width:90%;display:block;margin:5px auto;border-radius:8px;box-shadow:0 0 10px rgba(0,0,0,0.1);',
    PRE: 'background:#1e1e1e;border-radius:12px;padding:12px 14px;overflow-x:auto;margin:14px 0;border:1px solid #333;',
    TABLE: 'border-collapse:collapse;width:100%;margin:14px 0;font-size:14px;line-height:1.6;',
    TH: 'background:#f6f8fa;color:#24292e;font-weight:600;font-size:13px;padding:12px 14px;text-align:left;border-bottom:1px solid #d0d7de;',
    TD: 'padding:11px 14px;color:#24292e;vertical-align:top;border-bottom:1px solid #eaeef2;',
    HR: 'border:none;height:3px;background:#eeeeee;margin:16px 0;',
    STRONG: 'color:#000;',
    B: 'color:#000;'
  };
  var WX_CODE = "font-family:'JetBrains Mono',Menlo,Consolas,monospace;font-size:13px;line-height:1.6;color:#d4d4d4;white-space:pre;display:block;";
  var WX_INLINE_CODE = "background:#e7e7e7;color:#4d4d4c;padding:3px 5px;border-radius:5px;font-size:13px;font-family:'JetBrains Mono',Menlo,Consolas,monospace;";

  function buildWeChatHTML() {
    var dst = article.cloneNode(true);

    // 1) 代码 token 颜色：与原文 DOM 并行遍历，取浏览器实际渲染色内联
    var srcSpans = article.querySelectorAll('pre code span');
    var dstSpans = dst.querySelectorAll('pre code span');
    srcSpans.forEach(function (s, i) {
      var d = dstSpans[i];
      if (!d) return;
      var cs = getComputedStyle(s);
      d.style.color = cs.color;
      if (cs.fontStyle !== 'normal') d.style.fontStyle = cs.fontStyle;
      if (cs.fontWeight !== '400' && cs.fontWeight !== 'normal') d.style.fontWeight = cs.fontWeight;
    });

    // 2) 清理与结构归一
    dst.querySelectorAll('script,style,noscript,iframe,.line-numbers-rows,.copy-btn,.mermaid-zoom-btn').forEach(function (n) { n.remove(); });
    dst.querySelectorAll('.code-fold-container').forEach(function (c) {
      var pre = c.querySelector('pre');
      if (pre) c.replaceWith(pre); else c.remove();
    });
    dst.querySelectorAll('a[data-pswp-width]').forEach(function (a) {
      var img = a.querySelector('img');
      if (img) a.replaceWith(img); else a.remove();
    });
    dst.querySelectorAll('.table-container').forEach(function (w) {
      var t = w.querySelector('table');
      if (t) w.replaceWith(t); else w.remove();
    });
    dst.querySelectorAll('.mermaid').forEach(function (mm) {
      var p = document.createElement('p');
      p.setAttribute('style', WX_STYLE.P + 'color:#999;');
      p.textContent = '【流程图请见原文】';
      mm.replaceWith(p);
    });

    // 3) 公式 → codecogs 图片（公众号会自动转存外链图）
    var mathCount = 0;
    dst.querySelectorAll('.katex-display').forEach(function (el) {
      var tex = getTex(el);
      if (tex == null) { el.remove(); return; }
      var img = document.createElement('img');
      img.src = 'https://latex.codecogs.com/png.image?' + encodeURIComponent('\\dpi{150} ' + tex);
      img.setAttribute('style', 'display:block;margin:10px auto;max-width:100%;');
      el.replaceWith(img);
      mathCount++;
    });
    dst.querySelectorAll('.katex').forEach(function (el) {
      var tex = getTex(el);
      if (tex == null) { el.remove(); return; }
      var img = document.createElement('img');
      img.src = 'https://latex.codecogs.com/png.image?' + encodeURIComponent('\\dpi{110} ' + tex);
      img.setAttribute('style', 'vertical-align:middle;margin:0 2px;max-height:22px;');
      el.replaceWith(img);
      mathCount++;
    });

    // 4) 复刻博客 h1/h2 的 "#" 前缀（原来是 ::before 伪元素，公众号只认真实节点）
    dst.querySelectorAll('h1,h2').forEach(function (h) {
      var mark = document.createElement('span');
      mark.setAttribute('style', 'font-weight:bold;padding-right:6px;color:#000;');
      mark.textContent = '#';
      h.insertBefore(mark, h.firstChild);
    });

    // 5) 套用行内样式映射
    dst.querySelectorAll('*').forEach(function (el) {
      var tag = el.tagName;
      if (WX_STYLE[tag]) el.setAttribute('style', (el.getAttribute('style') || '') + WX_STYLE[tag]);
      if (tag === 'CODE') {
        el.setAttribute('style', (el.getAttribute('style') || '') + (el.closest('pre') ? WX_CODE : WX_INLINE_CODE));
      }
      el.removeAttribute('class');
      el.removeAttribute('id');
      if (tag === 'IMG') { el.removeAttribute('loading'); el.removeAttribute('decoding'); }
    });

    // 6) 相对图片转绝对 + 包裹 section（博客同款衬线字体基线）+ 文末原文链接
    dst.querySelectorAll('img').forEach(function (img) {
      var src = img.getAttribute('src') || '';
      if (src && !/^(https?:|data:)/.test(src)) {
        try { img.setAttribute('src', new URL(src, window.location.origin).href); } catch (e) {}
      }
    });
    var m = meta();
    var html = '<section style="font-family:' + WX_SERIF + ';font-size:16px;color:#000;line-height:1.7;">'
      + dst.innerHTML
      + '<hr style="' + WX_STYLE.HR + '">'
      + '<p style="' + WX_STYLE.P + 'color:#888;font-size:13px;">原文链接：<a href="' + m.url + '" style="' + WX_STYLE.A + '">' + m.url + '</a></p>'
      + '</section>';
    return { html: html, plain: dst.innerText, mathCount: mathCount };
  }

  /* ---------- 剪贴板 ---------- */
  function copyText(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) return navigator.clipboard.writeText(text);
    return Promise.reject(new Error('浏览器不支持剪贴板'));
  }
  function copyHtml(html, plain) {
    if (navigator.clipboard && window.ClipboardItem) {
      return navigator.clipboard.write([new ClipboardItem({
        'text/html': new Blob([html], { type: 'text/html' }),
        'text/plain': new Blob([plain], { type: 'text/plain' })
      })]);
    }
    // 降级：选中隐藏 contenteditable 后 execCommand
    return new Promise(function (resolve, reject) {
      var box = document.createElement('div');
      box.contentEditable = 'true';
      box.style.position = 'fixed';
      box.style.left = '-9999px';
      box.innerHTML = html;
      document.body.appendChild(box);
      var range = document.createRange();
      range.selectNodeContents(box);
      var sel = window.getSelection();
      sel.removeAllRanges();
      sel.addRange(range);
      var ok = document.execCommand('copy');
      sel.removeAllRanges();
      box.remove();
      ok ? resolve() : reject(new Error('execCommand 复制失败'));
    });
  }

  /* ---------- 导出入口 ---------- */
  function exportAs(fmt) {
    if (fmt === 'wechat') {
      var r = buildWeChatHTML();
      return copyHtml(r.html, r.plain).then(function () {
        toast(r.mathCount > 0
          ? '公众号格式已复制（' + r.mathCount + ' 个公式已转图片，请预览确认）'
          : '公众号格式已复制，去编辑器粘贴吧');
      });
    }
    toast('正在生成 Markdown…');
    return buildMarkdown(fmt).then(function (md) {
      return copyText(md).then(function () {
        var name = { md: 'Markdown', csdn: 'CSDN', cnblogs: '博客园' }[fmt] || 'Markdown';
        toast(name + ' 格式已复制' + (fmt === 'csdn' ? '（外链图若不显示需在 CSDN 编辑器转存）' : ''));
      });
    });
  }
})();
