(function () {
  document.querySelectorAll('.cols-toggle').forEach(function (btn) {
    var scope = btn.dataset.scope || 'global';
    var KEY = 'bk-cols-' + scope;
    var target = document.querySelector(btn.dataset.target);
    if (!target) return;

    function render(cols) {
      target.classList.toggle('cols-2', cols === 2);
      target.classList.toggle('cols-1', cols === 1);
      var icon = btn.querySelector('.cols-icon');
      var label = btn.querySelector('.cols-label');
      if (cols === 2) {
        if (icon) icon.className = 'fas fa-th-large cols-icon';
        if (label) label.textContent = '双栏';
      } else {
        if (icon) icon.className = 'fas fa-bars cols-icon';
        if (label) label.textContent = '单栏';
      }
    }

    var stored = parseInt(localStorage.getItem(KEY) || '0', 10);
    var initial = (stored === 1 || stored === 2)
      ? stored
      : parseInt(btn.dataset.default || '2', 10);
    render(initial);

    btn.addEventListener('click', function () {
      var current = target.classList.contains('cols-2') ? 2 : 1;
      var next = current === 2 ? 1 : 2;
      render(next);
      localStorage.setItem(KEY, String(next));
    });
  });

  document.querySelectorAll('.list-search').forEach(function (form) {
    form.addEventListener('submit', function (e) {
      e.preventDefault();
      var input = form.querySelector('input[name="q"]');
      var q = input ? (input.value || '').trim() : '';
      if (!q) return;
      location.href = '/pages/search.html?q=' + encodeURIComponent(q);
    });
  });
})();
