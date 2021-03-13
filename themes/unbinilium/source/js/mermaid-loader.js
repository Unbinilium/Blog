function mermaidLoad() {
    var theme = 'neutral';
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        theme = 'dark';
    }
    mermaid.initialize({ securitylevel: 'loose', theme: theme, themeVariables: { fontFamily: '"Menlo", "Meslo LG", monospace', fontSize: '9px' } });
}

(function () {
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js';
    script.setAttribute('onload', 'mermaidLoad()')
    document.head.appendChild(script);
})();
