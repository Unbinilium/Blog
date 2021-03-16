var mermaidConfig = {
    theme: 'neutral',
    securityLevel: 'loose',
    fontFamily: '"Menlo", "Meslo LG", monospace',
    er: {
        minEntityWidth: 100,
        minEntityHeight: 55
    },
    sequence: {
        width: 130,
        height: 30
    },
    gantt: {
        barHeight: 25,
        barGap: 4
    }
};

async function mermaidLoaded() {
    if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
        mermaidConfig['theme'] = 'dark';
    }
    mermaid.initialize(mermaidConfig);
}

async function mermaidReload() {
    if (document.querySelectorAll('[id^=mermaid-]').length != 0) location.reload(false);
}

if (document.getElementsByClassName('mermaid').length != 0) {
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js';
    script.setAttribute('onload', 'mermaidLoaded();')
    document.head.appendChild(script);
}

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', mermaidReload);
