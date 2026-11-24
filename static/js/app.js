// auto-dismiss flash messages
setTimeout(() => document.querySelectorAll('.flash').forEach(f => { f.style.transition = '.5s'; f.style.opacity = 0; setTimeout(() => f.remove(), 500); }), 6500);

// scroll-reveal
const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), { threshold: .12 });
document.querySelectorAll('.fade').forEach(el => io.observe(el));

// loading overlay on forms that opt in
document.querySelectorAll('form[data-loader]').forEach(f => f.addEventListener('submit', () => {
  if (!f.checkValidity()) return;
  document.getElementById('loaderText').textContent = f.dataset.loader;
  document.getElementById('loader').classList.add('on');
}));

// password show/hide + strength
document.querySelectorAll('.pw button').forEach(b => b.addEventListener('click', () => {
  const i = b.parentElement.querySelector('input'); i.type = i.type === 'password' ? 'text' : 'password';
  b.textContent = i.type === 'password' ? 'SHOW' : 'HIDE';
}));
const pwStrength = document.querySelector('[data-strength]');
if (pwStrength) pwStrength.addEventListener('input', () => {
  const v = pwStrength.value; let s = 0;
  if (v.length >= 8) s++; if (/[A-Z]/.test(v) && /[a-z]/.test(v)) s++; if (/\d/.test(v)) s++; if (/[^A-Za-z0-9]/.test(v)) s++;
  const bar = document.getElementById('pwbar');
  bar.style.width = (s * 25) + '%'; bar.style.background = ['#ef4444', '#ef4444', '#f59e0b', '#22c55e', '#16a34a'][s];
});

// drag & drop upload
const drop = document.getElementById('drop');
if (drop) {
  const inp = drop.querySelector('input[type=file]'), nm = document.getElementById('fname');
  const show = () => { if (inp.files.length) { nm.textContent = '📎 ' + inp.files[0].name; document.getElementById('scanBtn').disabled = false; } };
  drop.addEventListener('click', () => inp.click());
  inp.addEventListener('change', show);
  ['dragover', 'dragenter'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.add('over'); }));
  ['dragleave', 'drop'].forEach(e => drop.addEventListener(e, ev => { ev.preventDefault(); drop.classList.remove('over'); }));
  drop.addEventListener('drop', ev => { inp.files = ev.dataTransfer.files; show(); });
}

// animated risk gauge: <div class="gauge" data-pct="72" data-color="#dc2626">
document.querySelectorAll('.gauge[data-pct]').forEach(g => {
  const pct = +g.dataset.pct, col = g.dataset.color, len = 251.3;
  g.innerHTML = `<svg viewBox="0 0 200 110"><defs><linearGradient id="gg" x1="0" x2="1"><stop offset="0" stop-color="#22c55e"/><stop offset=".5" stop-color="#f59e0b"/><stop offset="1" stop-color="#ef4444"/></linearGradient></defs>
    <path d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="#e8eef5" stroke-width="16" stroke-linecap="round"/>
    <path id="arc" d="M20 100 A80 80 0 0 1 180 100" fill="none" stroke="url(#gg)" stroke-width="16" stroke-linecap="round" stroke-dasharray="${len}" stroke-dashoffset="${len}" style="transition:stroke-dashoffset 1.4s cubic-bezier(.2,.8,.2,1)"/>
    <g id="needle" style="transform-origin:100px 100px;transform:rotate(-90deg);transition:transform 1.4s cubic-bezier(.2,.8,.2,1)"><path d="M100 100 L100 32" stroke="#0b1b2b" stroke-width="3.5" stroke-linecap="round"/><circle cx="100" cy="100" r="7" fill="#0b1b2b"/></g></svg>
    <div class="val" style="color:${col}"><span id="gnum">0</span>%<small>Risk probability</small></div>`;
  requestAnimationFrame(() => setTimeout(() => {
    g.querySelector('#arc').style.strokeDashoffset = len * (1 - pct / 100);
    g.querySelector('#needle').style.transform = `rotate(${-90 + 180 * pct / 100}deg)`;
    let n = 0; const t = setInterval(() => { n += Math.max(.5, pct / 50); if (n >= pct) { n = pct; clearInterval(t); } g.querySelector('#gnum').textContent = n.toFixed(1); }, 24);
  }, 150));
});

// tabs
document.querySelectorAll('[data-tabs]').forEach(w => {
  const btns = w.querySelectorAll('.tabs button');
  btns.forEach(b => b.addEventListener('click', () => {
    btns.forEach(x => x.classList.toggle('on', x === b));
    w.querySelectorAll('[data-pane]').forEach(p => p.style.display = p.dataset.pane === b.dataset.tab ? '' : 'none');
    window.dispatchEvent(new Event('resize'));
  }));
});

// animate factor bars
window.addEventListener('load', () => document.querySelectorAll('[data-w]').forEach(e => setTimeout(() => e.style.width = e.dataset.w, 200)));
