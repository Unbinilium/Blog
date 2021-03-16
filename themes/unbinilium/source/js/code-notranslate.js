document.onreadystatechange = async function () {
    if (document.readyState === 'complete') {
        var codeFigures = document.querySelectorAll('.highlight');
        codeFigures.forEach(async function (item) {
            item.setAttribute('translate', 'no');
            item.querySelectorAll('.comment').forEach(async function (item) {
                item.setAttribute('translate', 'yes');
            });
        });
    }
}
