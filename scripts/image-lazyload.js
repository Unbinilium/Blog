hexo.extend.filter.register('markdown-it:renderer', function (md) {
    var defaultImageRenderer = md.renderer.rules.image;

    md.renderer.rules.image = function (tokens, idx, options, env, self) {
        var token = tokens[idx];
        token.attrSet('loading', 'lazy');
        return defaultImageRenderer(tokens, idx, options, env, self);
    };
});
