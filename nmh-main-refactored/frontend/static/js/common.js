if (window.AOS) AOS.init({ once: true, offset: 50, duration: 600 });

  const storedUser = getAuthUsername();
  document.getElementById('display-username').textContent = storedUser;
  document.getElementById('menu-username').textContent = storedUser;
  document.getElementById('avatar-initial').textContent = storedUser.charAt(0).toUpperCase();

  const menu = document.getElementById('user-menu');
  document.getElementById('user-btn').addEventListener('click', e => { e.stopPropagation(); menu.classList.toggle('open'); });
  document.addEventListener('click', e => { if (!menu.contains(e.target)) menu.classList.remove('open'); });
  
  function syncThemeUI() {
    if (window.Chart) { Chart.defaults.color = getComputedStyle(document.documentElement).getPropertyValue('--muted').trim(); Chart.defaults.borderColor = getComputedStyle(document.documentElement).getPropertyValue('--border').trim(); Object.values(Chart.instances).forEach(c => c.update()); }
    if (typeof onThemeChange === 'function') onThemeChange(document.documentElement.classList.contains('dark'));
  }
  function toggleTheme() {
    const dark = !document.documentElement.classList.contains('dark');
    document.documentElement.classList.toggle('dark', dark);
    localStorage.setItem('theme', dark ? 'dark' : 'light'); syncThemeUI();
  }

  function logout() { clearAuthSession(); window.location.href = '/login'; }

  document.querySelectorAll('#nav a').forEach(a => {
    if (a.getAttribute('href') === location.pathname) a.classList.add('active');
    a.addEventListener('click', e => { if (a.getAttribute('href') === location.pathname) return; e.preventDefault(); document.body.classList.add('leaving'); setTimeout(() => location.href = a.getAttribute('href'), 180); });
  });
  function toast(msg) { const el = document.createElement('div'); el.className = 'toast'; el.textContent = msg; document.getElementById('toasts').appendChild(el); setTimeout(() => { el.classList.add('out'); setTimeout(() => el.remove(), 300); }, 2400); }
  function confirmModal(title, msg) { return new Promise(resolve => { const bg = document.getElementById('modal-bg'); document.getElementById('modal-title').textContent = title; document.getElementById('modal-msg').textContent = msg; bg.classList.add('open'); const done = v => { bg.classList.remove('open'); yes.onclick = no.onclick = null; resolve(v); }; const yes = document.getElementById('modal-yes'), no = document.getElementById('modal-no'); yes.onclick = () => done(true); no.onclick = () => done(false); bg.onclick = e => { if (e.target === bg) done(false); }; }); }
  
  document.getElementById('theme-toggle').addEventListener('click', toggleTheme);
  document.getElementById('go-recovery').addEventListener('click', () => location.assign('/login?mode=forgot'));
  document.getElementById('logout-button').addEventListener('click', logout);
  window.addEventListener('DOMContentLoaded', syncThemeUI);
