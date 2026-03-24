// 打印主题标识,请保留出处
;(function () {
  var style1 = 'background:#4BB596;color:#ffffff;border-radius: 2px;'
  var style2 = 'color:auto;'
  var author = ' TMaize'
  var github = ' https://github.com/TMaize/tmaize-blog'
  var build = ' ' + blog.buildAt.substr(0, 4)
  build += '/' + blog.buildAt.substr(4, 2)
  build += '/' + blog.buildAt.substr(6, 2)
  build += ' ' + blog.buildAt.substr(8, 2)
  build += ':' + blog.buildAt.substr(10, 2)
  console.info('%c Author %c' + author, style1, style2)
  console.info('%c Build  %c' + build, style1, style2)
  console.info('%c GitHub %c' + github, style1, style2)
})()

/**
 * 工具，允许多次onload不被覆盖
 * @param {方法} func
 */
blog.addLoadEvent = function (func) {
  var oldonload = window.onload
  if (typeof window.onload != 'function') {
    window.onload = func
  } else {
    window.onload = function () {
      oldonload()
      func()
    }
  }
}

/**
 * 工具，兼容的方式添加事件
 * @param {单个DOM节点} dom
 * @param {事件名} eventName
 * @param {事件方法} func
 * @param {是否捕获} useCapture
 */
blog.addEvent = function (dom, eventName, func, useCapture) {
  if (window.attachEvent) {
    dom.attachEvent('on' + eventName, func)
  } else if (window.addEventListener) {
    if (useCapture != undefined && useCapture === true) {
      dom.addEventListener(eventName, func, true)
    } else {
      dom.addEventListener(eventName, func, false)
    }
  }
}

/**
 * 工具，DOM添加某个class
 * @param {单个DOM节点} dom
 * @param {class名} className
 */
blog.addClass = function (dom, className) {
  if (!blog.hasClass(dom, className)) {
    var c = dom.className || ''
    dom.className = c + ' ' + className
    dom.className = blog.trim(dom.className)
  }
}

/**
 * 工具，DOM是否有某个class
 * @param {单个DOM节点} dom
 * @param {class名} className
 */
blog.hasClass = function (dom, className) {
  var list = (dom.className || '').split(/\s+/)
  for (var i = 0; i < list.length; i++) {
    if (list[i] == className) return true
  }
  return false
}

/**
 * 工具，DOM删除某个class
 * @param {单个DOM节点} dom
 * @param {class名} className
 */
blog.removeClass = function (dom, className) {
  if (blog.hasClass(dom, className)) {
    var list = (dom.className || '').split(/\s+/)
    var newName = ''
    for (var i = 0; i < list.length; i++) {
      if (list[i] != className) newName = newName + ' ' + list[i]
    }
    dom.className = blog.trim(newName)
  }
}

/**
 * 工具，兼容问题，某些OPPO手机不支持ES5的trim方法
 * @param {字符串} str
 */
blog.trim = function (str) {
  return str.replace(/^\s+|\s+$/g, '')
}

/**
 * 工具，转义html字符
 * @param {字符串} str
 */
blog.htmlEscape = function (str) {
  var temp = document.createElement('div')
  temp.innerText = str
  str = temp.innerHTML
  temp = null
  return str
}

/**
 * 工具，转换实体字符防止XSS
 * @param {字符串} str
 */
blog.encodeHtml = function (html) {
  var o = document.createElement('div')
  o.innerText = html
  var temp = o.innerHTML
  o = null
  return temp
}

/**
 * 工具， 转义正则关键字
 * @param {字符串} str
 */
blog.encodeRegChar = function (str) {
  // \ 必须在第一位
  var arr = ['\\', '.', '^', '$', '*', '+', '?', '{', '}', '[', ']', '|', '(', ')']
  arr.forEach(function (c) {
    var r = new RegExp('\\' + c, 'g')
    str = str.replace(r, '\\' + c)
  })
  return str
}

/**
 * 工具，Ajax
 * @param {字符串} str
 */
blog.ajax = function (option, success, fail) {
  var xmlHttp = null
  if (window.XMLHttpRequest) {
    xmlHttp = new XMLHttpRequest()
  } else {
    xmlHttp = new ActiveXObject('Microsoft.XMLHTTP')
  }
  var url = option.url
  var method = (option.method || 'GET').toUpperCase()
  var sync = option.sync === false ? false : true
  var timeout = option.timeout || 10000

  var timer
  var isTimeout = false
  xmlHttp.open(method, url, sync)
  xmlHttp.onreadystatechange = function () {
    if (isTimeout) {
      fail({
        error: '请求超时'
      })
    } else {
      if (xmlHttp.readyState == 4) {
        if (xmlHttp.status == 200) {
          success(xmlHttp.responseText)
        } else {
          fail({
            error: '状态错误',
            code: xmlHttp.status
          })
        }
        //清除未执行的定时函数
        clearTimeout(timer)
      }
    }
  }
  timer = setTimeout(function () {
    isTimeout = true
    fail({
      error: '请求超时'
    })
    xmlHttp.abort()
  }, timeout)
  xmlHttp.send()
}

/**
 * 特效：点击页面文字冒出特效
 */
blog.initClickEffect = function (textArr) {
  function createDOM(text) {
    var dom = document.createElement('span')
    dom.innerText = text
    dom.style.left = 0
    dom.style.top = 0
    dom.style.position = 'fixed'
    dom.style.fontSize = '12px'
    dom.style.whiteSpace = 'nowrap'
    dom.style.webkitUserSelect = 'none'
    dom.style.userSelect = 'none'
    dom.style.opacity = 0
    dom.style.transform = 'translateY(0)'
    dom.style.webkitTransform = 'translateY(0)'
    return dom
  }

  blog.addEvent(window, 'click', function (ev) {
    let target = ev.target
    while (target !== document.documentElement) {
      if (target.tagName.toLocaleLowerCase() == 'a') return
      if (blog.hasClass(target, 'footer-btn')) return
      target = target.parentNode
    }

    var text = textArr[parseInt(Math.random() * textArr.length)]
    var dom = createDOM(text)

    document.body.appendChild(dom)
    var w = parseInt(window.getComputedStyle(dom, null).getPropertyValue('width'))
    var h = parseInt(window.getComputedStyle(dom, null).getPropertyValue('height'))

    var sh = window.pageYOffset || document.documentElement.scrollTop || document.body.scrollTop || 0
    dom.style.left = ev.pageX - w / 2 + 'px'
    dom.style.top = ev.pageY - sh - h + 'px'
    dom.style.opacity = 1

    setTimeout(function () {
      dom.style.transition = 'transform 500ms ease-out, opacity 500ms ease-out'
      dom.style.webkitTransition = 'transform 500ms ease-out, opacity 500ms ease-out'
      dom.style.opacity = 0
      dom.style.transform = 'translateY(-26px)'
      dom.style.webkitTransform = 'translateY(-26px)'
    }, 20)

    setTimeout(function () {
      document.body.removeChild(dom)
      dom = null
    }, 520)
  })
}

// 新建DIV包裹TABLE
blog.addLoadEvent(function () {
  // 文章页生效
  if (document.getElementsByClassName('page-post').length == 0) {
    return
  }
  var tables = document.getElementsByTagName('table')
  for (var i = 0; i < tables.length; i++) {
    var table = tables[i]
    var elem = document.createElement('div')
    elem.setAttribute('class', 'table-container')
    table.parentNode.insertBefore(elem, table)
    elem.appendChild(table)
  }
})

// 回到顶部
blog.addLoadEvent(function () {
  var el = document.querySelector('.footer-btn.to-top')
  if (!el) return
  function getScrollTop() {
    if (document.documentElement && document.documentElement.scrollTop) {
      return document.documentElement.scrollTop
    } else if (document.body) {
      return document.body.scrollTop
    }
  }
  function ckeckToShow() {
    if (getScrollTop() > 200) {
      blog.addClass(el, 'show')
    } else {
      blog.removeClass(el, 'show')
    }
  }
  blog.addEvent(window, 'scroll', ckeckToShow)
  blog.addEvent(
    el,
    'click',
    function (event) {
      window.scrollTo(0, 0)
      event.stopPropagation()
    },
    true
  )
  ckeckToShow()
})

// 移动到底部
blog.addLoadEvent(function () {
  var el = document.querySelector('.footer-btn.to-bottom') // 确保 HTML 中有这个类名
  if (!el) return

  blog.addEvent(el, 'click', function (event) {
    // 滚动到文档的总高度
    window.scrollTo(0, document.documentElement.scrollHeight || document.body.scrollHeight)
    event.stopPropagation()
  }, true)
})

// 一键复制全文 Markdown 到剪贴板（放在 footer 的按钮）
document.addEventListener('DOMContentLoaded', function () {
  const copyBtn = document.querySelector('.footer-btn.copy-all');
  if (!copyBtn) return;

  copyBtn.addEventListener('click', function () {
    // 获取文章容器（根据你的主题调整选择器）
    const article = document.querySelector('.post-content, .entry-content, article');
    if (!article) {
      alert('找不到文章内容区域');
      return;
    }

    // 从 data-* 或 fallback 获取元数据（强烈建议在 _layouts/post.html 加 data-* 属性）
    const title  = article.dataset.title  || document.querySelector('h1')?.textContent.trim() || '无标题';
    const url    = article.dataset.url    || window.location.href;
    const date   = article.dataset.date   || '';
    const author = article.dataset.author || '佚名';

    const contentEl = article;
    const clone = contentEl.cloneNode(true);

    // 图片路径处理：转为绝对路径
    clone.querySelectorAll('img').forEach(img => {
      if (img.src && !img.src.startsWith('http') && !img.src.startsWith('data:')) {
        try {
          img.src = new URL(img.src, window.location.origin).href;
        } catch (e) {
          console.warn('图片路径处理失败:', img.src);
        }
      }
    });

    // 初始化 Turndown
    const turndownService = new TurndownService({
      headingStyle: 'atx',
      bulletListMarker: '-',
      codeBlockStyle: 'fenced',
      fence: '```'
    });

    // 加载 GFM 插件（@truto 版本，表格支持更好）
    if (window.gfm) {
      turndownService.use(window.gfm);
    } else {
      console.warn('GFM 插件未加载，表格转换可能不完美。请检查 <script src="https://cdn.jsdelivr.net/npm/@truto/turndown-plugin-gfm@latest/dist/turndown-plugin-gfm.min.js"></script>');
    }

    // 自定义规则：优化代码块语言保留
    turndownService.addRule('fencedCodeBlockWithLanguage', {
      filter: 'pre',
      replacement: function (content, node, options) {
        const codeNode = node.firstChild;
        if (codeNode?.nodeName === 'CODE') {
          const langMatch = codeNode.className.match(/language-(\w+)/);
          const lang = langMatch ? langMatch[1] : '';
          const code = codeNode.textContent.trim();
          return `\n\n${options.fence}${lang}\n${code}\n${options.fence}\n\n`;
        }
        return `\n\n${content}\n\n`;
      }
    });

    let markdown = turndownService.turndown(clone.innerHTML);

    // 添加转载 header
    const header = [
      `# ${title}`,
      '',
      `**作者**：${author}`,
      `**原文链接**：[${url}](${url})`,
      `**发布日期**：${date}`,
      '',
      `转载请注明出处，感谢！`,
      '---',
      ''
    ].join('\n');

    markdown = header + markdown;

    // 复制到剪贴板 + 页面提示
    navigator.clipboard.writeText(markdown.trim()).then(() => {
      // 成功提示：右下角绿色框（纯 inline style）
      const msg = document.createElement('div');
      msg.textContent = '复制成功！';
      msg.style.position = 'fixed';
      msg.style.bottom = '30px';
      msg.style.right = '30px';
      msg.style.background = '#4caf50';
      msg.style.color = 'white';
      msg.style.padding = '12px 24px';
      msg.style.borderRadius = '6px';
      msg.style.boxShadow = '0 3px 10px rgba(0,0,0,0.3)';
      msg.style.zIndex = '9999';
      msg.style.fontSize = '16px';
      msg.style.fontWeight = 'bold';
      msg.style.opacity = '1';

      document.body.appendChild(msg);

      // 2.2秒后淡出并移除
      setTimeout(() => {
        msg.style.opacity = '0';
        setTimeout(() => msg.remove(), 500);
      }, 2200);

      // 可选：按钮短暂变绿
      const originalColor = copyBtn.style.color;
      copyBtn.style.color = '#4caf50';
      setTimeout(() => { copyBtn.style.color = originalColor || ''; }, 1500);
    }).catch(err => {
      console.error('复制失败:', err);

      // 失败提示：右下角红色框
      const msg = document.createElement('div');
      msg.textContent = '复制失败，请手动复制';
      msg.style.position = 'fixed';
      msg.style.bottom = '30px';
      msg.style.right = '30px';
      msg.style.background = '#f44336';
      msg.style.color = 'white';
      msg.style.padding = '12px 24px';
      msg.style.borderRadius = '6px';
      msg.style.boxShadow = '0 3px 10px rgba(0,0,0,0.3)';
      msg.style.zIndex = '9999';
      msg.style.fontSize = '16px';
      msg.style.fontWeight = 'bold';
      msg.style.opacity = '1';

      document.body.appendChild(msg);

      setTimeout(() => {
        msg.style.opacity = '0';
        setTimeout(() => msg.remove(), 500);
      }, 2200);
    });
  });
});

// 点击图片全屏预览
blog.addLoadEvent(function () {
  if (!document.querySelector('.page-post')) {
    return
  }
  console.debug('init post img click event')
  let imgMoveOrigin = null
  let restoreLock = false
  let imgArr = document.querySelectorAll('.page-post img')

  let css = [
    '.img-move-bg {',
    '  transition: opacity 300ms ease;',
    '  position: fixed;',
    '  left: 0;',
    '  top: 0;',
    '  right: 0;',
    '  bottom: 0;',
    '  opacity: 0;',
    '  background-color: #000000;',
    '  z-index: 100;',
    '}',
    '.img-move-item {',
    '  transition: all 300ms ease;',
    '  position: fixed;',
    '  opacity: 0;',
    '  cursor: pointer;',
    '  z-index: 101;',
    '}'
  ].join('')
  var styleDOM = document.createElement('style')
  if (styleDOM.styleSheet) {
    styleDOM.styleSheet.cssText = css
  } else {
    styleDOM.appendChild(document.createTextNode(css))
  }
  document.querySelector('head').appendChild(styleDOM)

  window.addEventListener('resize', toCenter)

  for (let i = 0; i < imgArr.length; i++) {
    imgArr[i].addEventListener('click', imgClickEvent, true)
  }

  function prevent(ev) {
    ev.preventDefault()
  }

  function toCenter() {
    if (!imgMoveOrigin) {
      return
    }
    let width = Math.min(imgMoveOrigin.naturalWidth, parseInt(document.documentElement.clientWidth * 0.9))
    let height = (width * imgMoveOrigin.naturalHeight) / imgMoveOrigin.naturalWidth
    if (window.innerHeight * 0.95 < height) {
      height = Math.min(imgMoveOrigin.naturalHeight, parseInt(window.innerHeight * 0.95))
      width = (height * imgMoveOrigin.naturalWidth) / imgMoveOrigin.naturalHeight
    }

    let img = document.querySelector('.img-move-item')
    img.style.left = (document.documentElement.clientWidth - width) / 2 + 'px'
    img.style.top = (window.innerHeight - height) / 2 + 'px'
    img.style.width = width + 'px'
    img.style.height = height + 'px'
  }

  function restore() {
    if (restoreLock == true) {
      return
    }
    restoreLock = true
    let div = document.querySelector('.img-move-bg')
    let img = document.querySelector('.img-move-item')

    div.style.opacity = 0
    img.style.opacity = 0
    img.style.left = imgMoveOrigin.x + 'px'
    img.style.top = imgMoveOrigin.y + 'px'
    img.style.width = imgMoveOrigin.width + 'px'
    img.style.height = imgMoveOrigin.height + 'px'

    setTimeout(function () {
      restoreLock = false
      document.body.removeChild(div)
      document.body.removeChild(img)
      imgMoveOrigin = null
    }, 300)
  }

  function imgClickEvent(event) {
    imgMoveOrigin = event.target

    let div = document.createElement('div')
    div.className = 'img-move-bg'

    let img = document.createElement('img')
    img.className = 'img-move-item'
    img.src = imgMoveOrigin.src
    img.style.left = imgMoveOrigin.x + 'px'
    img.style.top = imgMoveOrigin.y + 'px'
    img.style.width = imgMoveOrigin.width + 'px'
    img.style.height = imgMoveOrigin.height + 'px'

    div.onclick = restore
    div.onmousewheel = restore
    div.ontouchmove = prevent

    img.onclick = restore
    img.onmousewheel = restore
    img.ontouchmove = prevent
    img.ondragstart = prevent

    document.body.appendChild(div)
    document.body.appendChild(img)

    setTimeout(function () {
      div.style.opacity = 0.5
      img.style.opacity = 1
      toCenter()
    }, 0)
  }
})

// 切换夜间模式
blog.addLoadEvent(function () {
  const $el = document.querySelector('.footer-btn.theme-toggler')
  const $icon = $el.querySelector('.svg-icon')

  blog.removeClass($el, 'hide')
  if (blog.darkMode) {
    blog.removeClass($icon, 'icon-theme-light')
    blog.addClass($icon, 'icon-theme-dark')
  }

  function initDarkMode(flag) {
    blog.removeClass($icon, 'icon-theme-light')
    blog.removeClass($icon, 'icon-theme-dark')
    if (flag === 'true') blog.addClass($icon, 'icon-theme-dark')
    else blog.addClass($icon, 'icon-theme-light')

    document.documentElement.setAttribute('transition', '')
    setTimeout(function () {
      document.documentElement.removeAttribute('transition')
    }, 600)

    blog.initDarkMode(flag)
  }

  blog.addEvent($el, 'click', function () {
    const flag = blog.darkMode ? 'false' : 'true'
    localStorage.darkMode = flag
    initDarkMode(flag)
  })

  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addListener(function (ev) {
      const systemDark = ev.target.matches
      if (systemDark !== blog.darkMode) {
        localStorage.darkMode = '' // 清除用户设置
        initDarkMode(systemDark ? 'true' : 'false')
      }
    })
  }
})



/**
 * 分享到社交平台
 * @param {string} platform - 平台名称（twitter、facebook、weibo、qq）
 */
function shareOnSocial(platform) {
  const pageUrl = encodeURIComponent(window.location.href); // 获取当前页面 URL
  const pageTitle = encodeURIComponent(document.title); // 获取当前页面标题

  let shareUrl = '';

  // 根据平台生成分享链接
  switch (platform) {
    case 'twitter':
      shareUrl = `https://twitter.com/intent/tweet?url=${pageUrl}&text=${pageTitle}`;
      break;
    case 'facebook':
      shareUrl = `https://www.facebook.com/sharer/sharer.php?u=${pageUrl}`;
      break;
    case 'weibo':
      shareUrl = `https://service.weibo.com/share/share.php?url=${pageUrl}&title=${pageTitle}`;
      break;
    case 'qq':
      shareUrl = `https://connect.qq.com/widget/shareqq/index.html?url=${pageUrl}&title=${pageTitle}`;
      break;
    default:
      console.error('未知的分享平台');
      return;
  }

  // 打开分享窗口
  window.open(shareUrl, '_blank', 'width=600,height=400');
}

/**
 * 分享到微信（生成二维码）
 */
function shareOnWeChat() {
  const pageUrl = encodeURIComponent(window.location.href); // 获取当前页面 URL
  const qrcodeUrl = `https://api.qrserver.com/v1/create-qr-code/?size=150x150&data=${pageUrl}`; // 生成二维码的 URL

  // 弹出二维码图片
  const qrcodeWindow = window.open('', '_blank', 'width=300,height=300');
  qrcodeWindow.document.write(`
    <html>
      <head><title>微信分享二维码</title></head>
      <body style="margin: 0; display: flex; justify-content: center; align-items: center; height: 100vh;">
        <img src="${qrcodeUrl}" alt="微信分享二维码" style="width: 100%; height: 100%; max-width: 300px; max-height: 300px;">
      </body>
    </html>
  `);
}
