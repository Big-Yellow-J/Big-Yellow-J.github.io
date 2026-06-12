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

/* 工具：window load 事件注册（支持多次绑定不覆盖） */
blog.addLoadEvent = function (func) {
  if (document.readyState === 'complete') func()
  else window.addEventListener('load', func)
}

/* 工具：事件绑定 */
blog.addEvent = function (dom, eventName, func, useCapture) {
  dom.addEventListener(eventName, func, !!useCapture)
}

/* 工具：classList 操作 */
blog.addClass = function (dom, className) { dom.classList.add(className) }
blog.hasClass = function (dom, className) { return dom.classList.contains(className) }
blog.removeClass = function (dom, className) { dom.classList.remove(className) }

/* 工具：trim */
blog.trim = function (str) { return (str || '').trim() }

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

// 一键复制全文已迁移至 static/js/copy-export.js（多平台格式菜单，仅文章页加载）

// AI 摘要淡入（替代 mypost.html 中原有的内联 script）
blog.addLoadEvent(function () {
  const el = document.getElementById('description-llm')
  if (!el) return
  setTimeout(() => {
    el.classList.add('is-ready')
    setTimeout(() => el.classList.add('is-done'), 600)
  }, 200)
  document.addEventListener('visibilitychange', () => {
    if (document.visibilityState === 'visible') {
      el.classList.add('is-ready', 'is-done')
    }
  })
})

// 阅读进度条（仅文章页）
blog.addLoadEvent(function () {
  const article = document.querySelector('.post.page-post')
  if (!article) return
  const bar = document.createElement('div')
  bar.className = 'read-progress'
  document.body.appendChild(bar)
  let ticking = false
  function update() {
    const top = article.offsetTop
    const height = article.offsetHeight - window.innerHeight
    const scrolled = Math.max(0, window.scrollY - top)
    const pct = Math.min(100, Math.max(0, (scrolled / Math.max(1, height)) * 100))
    bar.style.width = pct + '%'
    ticking = false
  }
  window.addEventListener('scroll', () => {
    if (ticking) return
    ticking = true
    requestAnimationFrame(update)
  })
  update()
})

// toast 工具：blog.toast('已复制')
blog.toast = function (msg) {
  let el = document.querySelector('.copy-toast')
  if (!el) {
    el = document.createElement('div')
    el.className = 'copy-toast'
    document.body.appendChild(el)
  }
  el.textContent = msg
  el.classList.add('show')
  clearTimeout(blog.toast._t)
  blog.toast._t = setTimeout(() => el.classList.remove('show'), 1500)
}

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
    case 'linkedin':
      shareUrl = `https://www.linkedin.com/sharing/share-offsite/?url=${pageUrl}`;
      break;
    case 'reddit':
      shareUrl = `https://www.reddit.com/submit?url=${pageUrl}&title=${pageTitle}`;
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

/**
 * 复制当前页链接到剪贴板
 */
function copyPageLink() {
  navigator.clipboard.writeText(window.location.href).then(function () {
    blog.toast('链接已复制')
  }).catch(function () {
    blog.toast('复制失败，请手动复制地址栏')
  })
}

/**
 * 调用系统原生分享面板（移动端），不支持时降级为复制链接
 */
function nativeShare() {
  if (navigator.share) {
    navigator.share({ title: document.title, url: window.location.href }).catch(function () {})
  } else {
    copyPageLink()
  }
}

// 支持原生分享的环境才显示"系统分享"按钮
blog.addLoadEvent(function () {
  var btn = document.querySelector('.share-button.share-native')
  if (btn && navigator.share) btn.hidden = false
})

// 阅读位置记忆（仅文章页）：滚动时记录，重进时恢复，7 天过期
blog.addLoadEvent(function () {
  if (!document.querySelector('.post.page-post')) return
  var KEY = 'readpos:' + location.pathname
  try {
    var now = Date.now()
    Object.keys(localStorage).forEach(function (k) {
      if (k.indexOf('readpos:') !== 0) return
      var v = JSON.parse(localStorage.getItem(k) || '{}')
      if (!v.t || now - v.t > 7 * 864e5) localStorage.removeItem(k)
    })
    var saved = JSON.parse(localStorage.getItem(KEY) || 'null')
    // 带锚点进入说明用户有明确目标，不抢滚动
    if (saved && saved.y > 800 && !location.hash) {
      window.scrollTo(0, saved.y)
      blog.toast('已回到上次阅读位置')
    }
  } catch (e) {}
  var t = null
  window.addEventListener('scroll', function () {
    if (t) return
    t = setTimeout(function () {
      t = null
      try {
        if (window.scrollY > 800) localStorage.setItem(KEY, JSON.stringify({ y: window.scrollY, t: Date.now() }))
        else localStorage.removeItem(KEY)
      } catch (e) {}
    }, 250)
  }, { passive: true })
})

// 点击正文标题复制本节锚点链接（post.css 中标题为 pointer 样式）
blog.addLoadEvent(function () {
  var article = document.querySelector('.post.page-post')
  if (!article) return
  article.querySelectorAll('h1[id], h2[id], h3[id], h4[id], h5[id], h6[id]').forEach(function (h) {
    h.addEventListener('click', function () {
      var url = location.origin + location.pathname + '#' + encodeURIComponent(h.id)
      history.replaceState(null, '', '#' + h.id)
      navigator.clipboard.writeText(url).then(function () {
        blog.toast('已复制本节链接')
      }).catch(function () {})
    })
  })
})

// 字数/阅读时长修正：排除代码块后按正文字符数估算（约 400 字/分钟）
blog.addLoadEvent(function () {
  var article = document.querySelector('.post.page-post')
  var wEl = document.getElementById('word-count')
  if (!article || !wEl) return
  var clone = article.cloneNode(true)
  clone.querySelectorAll('pre, .code-fold-container, .mermaid, .table-container').forEach(function (n) { n.remove() })
  var chars = (clone.innerText || '').replace(/\s+/g, '').length
  if (!chars) return
  wEl.textContent = chars + ' 字'
  var mEl = document.getElementById('read-minutes')
  if (mEl) mEl.textContent = Math.max(1, Math.ceil(chars / 400)) + ' 分钟'
})

// AI 摘要：窄屏上过长时折叠为 3 行 + 展开按钮
blog.addLoadEvent(function () {
  if (!window.matchMedia('(max-width: 560px)').matches) return
  var box = document.getElementById('description-llm')
  var textEl = box && box.querySelector('.post-description-llm-text')
  if (!textEl) return
  box.classList.add('clamped')
  requestAnimationFrame(function () {
    if (textEl.scrollHeight <= textEl.clientHeight + 4) {
      box.classList.remove('clamped')
      return
    }
    var btn = document.createElement('button')
    btn.type = 'button'
    btn.className = 'desc-toggle'
    btn.textContent = '展开'
    btn.addEventListener('click', function () {
      var collapsed = box.classList.toggle('clamped')
      btn.textContent = collapsed ? '展开' : '收起'
    })
    box.appendChild(btn)
  })
})

// 安全加固：给所有 target="_blank" 链接自动补 rel="noopener noreferrer"，防 reverse tabnabbing
blog.addLoadEvent(function () {
  document.querySelectorAll('a[target="_blank"]').forEach(function (a) {
    var rel = (a.getAttribute('rel') || '').split(/\s+/).filter(Boolean);
    if (!rel.includes('noopener')) rel.push('noopener');
    if (!rel.includes('noreferrer')) rel.push('noreferrer');
    a.setAttribute('rel', rel.join(' '));
  });
})

