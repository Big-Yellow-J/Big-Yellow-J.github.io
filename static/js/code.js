document.addEventListener("DOMContentLoaded", function() {
    const codeBlocks = document.querySelectorAll('pre code');

    codeBlocks.forEach(code => {
        const pre = code.parentNode;
        pre.classList.add('line-numbers'); // 必须在渲染前添加
        
        const lang = code.className.match(/language-(\w+)/)?.[1] || 'code';

        // 构建 DOM
        const container = document.createElement('div');
        container.className = 'code-fold-container';
        
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

        // 复制功能
        header.querySelector('.copy-btn').addEventListener('click', function() {
            navigator.clipboard.writeText(code.textContent);
            this.innerText = 'Done!';
            setTimeout(() => this.innerText = 'Copy', 2000);
        });
    });

    // 1. 先触发高亮（此时所有代码块都是展开可见的）
    Prism.highlightAll();

    // 2. 高亮完成后，再将需要默认折叠的代码块折叠
    document.querySelectorAll('.code-fold-content').forEach(c => {
        c.classList.add('is-collapsed');
        c.previousElementSibling.classList.add('is-collapsed');
    });
});