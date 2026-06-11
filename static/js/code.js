document.addEventListener("DOMContentLoaded", function() {
    const codeBlocks = document.querySelectorAll('pre code');

    codeBlocks.forEach(code => {
        const pre = code.parentNode;
        pre.classList.add('line-numbers'); // 必须在渲染前添加

        // 提取首行 `@highlight: 1,3-5` 元数据 → 设为 pre.data-line（被 prism-line-highlight 识别）
        // 支持注释前缀：# // /* -- <!-- 等
        const firstLine = code.textContent.split('\n', 1)[0];
        const hl = firstLine.match(/@highlight[:\s]+([\d,\-\s]+)/);
        if (hl) {
            pre.setAttribute('data-line', hl[1].replace(/\s+/g, ''));
            code.textContent = code.textContent.replace(/^[^\n]*\n?/, '');
        }

        const lang = code.className.match(/language-(\w+)/)?.[1] || 'code';

        // 构建 DOM
        const container = document.createElement('div');
        container.className = 'code-fold-container';
        // 超过 25 行才默认折叠，短代码保持展开不打断阅读
        container.dataset.autoFold = code.textContent.split('\n').length > 25 ? '1' : '0';
        
        const header = document.createElement('div');
        header.className = 'code-fold-header';
        header.innerHTML = `
            <div class="header-info">
                <svg class="fold-arrow" viewBox="0 0 24 24"><path d="M7 10l5 5 5-5z"/></svg>
                <span>${lang.toUpperCase()}</span>
            </div>
            <button class="copy-btn">Copy</button>
        `;

        const content = document.createElement('div');
        content.className = 'code-fold-content'; 
        // 注意：初始不加 is-collapsed，让它保持展开以便 Prism 测量
        
        const inner = document.createElement('div');
        inner.className = 'code-fold-inner';

        pre.parentNode.insertBefore(container, pre);
        container.appendChild(header);
        container.appendChild(content);
        content.appendChild(inner);
        inner.appendChild(pre);

        // 点击折叠
        header.addEventListener('click', (e) => {
            if (e.target.classList.contains('copy-btn')) return;
            content.classList.toggle('is-collapsed');
            header.classList.toggle('is-collapsed');
        });

        // 复制功能：成功变绿 ✓ / 失败变红 ✗，2s 后复原
        header.querySelector('.copy-btn').addEventListener('click', function() {
            const btn = this;
            navigator.clipboard.writeText(code.textContent).then(() => {
                btn.innerText = '✓ Done';
                btn.classList.add('is-copied');
                if (window.blog && blog.toast) blog.toast('代码已复制');
            }).catch(() => {
                btn.innerText = '✗ Failed';
                btn.classList.add('is-failed');
                if (window.blog && blog.toast) blog.toast('复制失败，请手动选择复制');
            }).finally(() => {
                setTimeout(() => {
                    btn.innerText = 'Copy';
                    btn.classList.remove('is-copied', 'is-failed');
                }, 2000);
            });
        });
    });

    // 1. 先触发高亮（此时所有代码块都是展开可见的）
    Prism.highlightAll();

    // 2. 高亮完成后，只折叠超过行数阈值的代码块（data-auto-fold 在构建 DOM 时标记）
    document.querySelectorAll('.code-fold-container[data-auto-fold="1"] .code-fold-content').forEach(c => {
        c.classList.add('is-collapsed');
        c.previousElementSibling.classList.add('is-collapsed');
    });
});