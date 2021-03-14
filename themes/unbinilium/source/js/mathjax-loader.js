if (document.body.textContent.match(/(?:\$\$|\\\(|\\\[|\\begin\{.*?})/)) {
    if (!window.MathJax) {
        window.MathJax = {
            tex: { inlineMath: [['$', '$'], ['\\(', '\\)']] },
            options: {
                enableMenu: false,
                skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre']
            },
            svg: { fontCache: 'global' }
        };
    }
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js';
    document.head.appendChild(script);
}
