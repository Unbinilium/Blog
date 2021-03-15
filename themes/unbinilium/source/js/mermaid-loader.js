var mermaidConfig = {
    theme: 'neutral',
    logLevel: 'fatal',
    securityLevel: 'loose',
    startOnLoad: true,
    arrowMarkerAbsolute: false,
    er: {
        diagramPadding: 20,
        layoutDirection: 'TB',
        minEntityWidth: 100,
        minEntityHeight: 55,
        entityPadding: 15,
        stroke: 'gray',
        fill: 'honeydew',
        fontSize: 12,
        useMaxWidth: true
    },
    flowchart: {
        diagramPadding: 8,
        htmlLabels: true,
        curve: 'linear'
    },
    sequence: {
        diagramMarginX: 50,
        diagramMarginY: 10,
        actorMargin: 50,
        width: 130,
        height: 30,
        boxMargin: 10,
        boxTextMargin: 5,
        noteMargin: 10,
        messageMargin: 35,
        messageAlign: 'center',
        mirrorActors: true,
        bottomMarginAdj: 1,
        useMaxWidth: true,
        rightAngles: false,
        showSequenceNumbers: false
    },
    gantt: {
        titleTopMargin: 25,
        barHeight: 25,
        barGap: 4,
        topPadding: 50,
        leftPadding: 75,
        gridLineStartPadding: 35,
        fontSize: 11,
        fontFamily: '"Menlo", "Meslo LG", monospace',
        numberSectionStyles: 4,
        axisFormat: '%Y-%m-%d'
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
