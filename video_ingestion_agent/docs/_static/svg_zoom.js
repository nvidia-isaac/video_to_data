function zoomSvg(id, factor) {
    var el = document.getElementById(id);
    if (!el) return;
    var scale = parseFloat(el.dataset.scale || 1) * factor;
    el.dataset.scale = scale;
    el.style.transform = "scale(" + scale + ")";
}

function resetSvg(id) {
    var el = document.getElementById(id);
    if (!el) return;
    el.dataset.scale = 1;
    el.style.transform = "scale(1)";
}
