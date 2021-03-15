document.onreadystatechange = async function () {
    if (document.readyState === 'complete') {
        var codeBlocks = document.getElementsByClassName('code');
        for (let i = 0; i != codeBlocks.length; ++i) {
            codeBlocks[i].setAttribute('translate', 'no');
            codeBlocks[i].setAttribute('class', 'code notranslate');
        }
    }
}
