const reg = /(\s*)(`{3}) *(mermaid) *\n?([\s\S]+?)\s*(\2)(\n+|$)/g;

const ignore = data => {
    var source = data.source;
    var ext = source.substring(source.lastIndexOf('.')).toLowerCase();
    return ['.js', '.css', '.html', '.htm'].indexOf(ext) > -1;
}

hexo.extend.filter.register('before_post_render', function (data) {
    if (!ignore(data)) {
        data.content = data.content
            .replace(reg, function (raw, start, startQuote, lang, content, endQuote, end) {
                return `${start}<pre class="mermaid">${content}</pre>${end}`;
            });
    }
    return data;
}, 0);
