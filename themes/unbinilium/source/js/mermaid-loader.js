var mermaidTheme = 'neutral';

async function mermaidLoaded() {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) { mermaidTheme = 'dark'; }
    else { mermaidTheme = 'neutral'; }
    var mermaidContainer = document.getElementsByClassName('mermaid-container');
    for (var i = 0; i != mermaidContainer.length; ++i) {
        var mermaidSrc = mermaidContainer[i].firstElementChild;
        const mermaidOutId = mermaidSrc.id.replace('src', 'out');
        const mermaidThemeConfig = '%%{init:{ \'theme\':\'' + mermaidTheme + '\'}}%%\n';
        mermaid.mermaidAPI.render(mermaidOutId, mermaidThemeConfig + mermaidSrc.textContent, async function (svgCode) {
            mermaidSrc.insertAdjacentHTML('afterend', svgCode);
        });
    }
}

if (document.getElementsByClassName('mermaid-container').length != 0) {
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js';
    script.setAttribute('onload', 'mermaidLoaded();')
    document.head.appendChild(script);
}

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', mermaidLoaded);
