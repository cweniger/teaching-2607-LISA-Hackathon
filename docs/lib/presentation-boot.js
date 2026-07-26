/* Presentation deck — Reveal boot + footer + MathJax retypeset hook.
   The per-presentation label is read from <body data-presentation="..."> */

Reveal.initialize({
  hash: true, center: false, transition: 'none', slideNumber: true,
  width: 1214, height: 700, margin: 0.04, minScale: 0.2, maxScale: 2.0,
  mathjax3: { mathjax: 'vendor/mathjax/tex-mml-chtml.js' },
  plugins: [ RevealMath.MathJax3 ]
});

// MathJax3 typesets all slides at page load using whatever container
// dimensions exist at that moment. Slides not visible then cache wrong
// glyph positions and render clipped. Re-typeset the becoming-current
// slide on every navigation.
Reveal.on('slidechanged', function(e) {
  if (window.MathJax && window.MathJax.typesetPromise) {
    window.MathJax.typesetPromise([e.currentSlide]).catch(function(){});
  }
});

(function attachFooters(){
  const label = (document.body.dataset.presentation || '').trim();
  document.querySelectorAll('.reveal .slides > section').forEach(function(slide) {
    const footer = document.createElement('div');
    footer.className = 'slide-footer';
    footer.innerHTML =
      '<span>' + (label || '') + '</span>' +
      '<span>Christoph Weniger &mdash; GRAPPA / UvA</span>';
    slide.appendChild(footer);
  });
})();
