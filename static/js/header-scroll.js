/* 头部菜单：向下滚隐藏，向上滚显示
   思路：rAF 节流，记录上次 scrollY，超出阈值后再判断方向
*/
(function () {
  'use strict';
  var HIDE_AFTER = 80;     // 滚出此距离才开始判断
  var DELTA      = 6;      // 抖动阈值，过滤滚轮微动
  var lastY  = window.pageYOffset || 0;
  var ticking = false;
  var body   = document.body;

  function update() {
    var y = window.pageYOffset || 0;
    var dy = y - lastY;
    if (Math.abs(dy) > DELTA) {
      if (dy > 0 && y > HIDE_AFTER) {
        body.classList.add('nav-hidden');
      } else if (dy < 0) {
        body.classList.remove('nav-hidden');
      }
      lastY = y;
    }
    if (y <= HIDE_AFTER) body.classList.remove('nav-hidden');
    ticking = false;
  }

  window.addEventListener('scroll', function () {
    if (!ticking) {
      window.requestAnimationFrame(update);
      ticking = true;
    }
  }, { passive: true });
})();
