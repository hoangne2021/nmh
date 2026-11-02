if (!localStorage.getItem('token') && !sessionStorage.getItem('token')) {
  location.replace('/login');
}
if (window.tailwind) tailwind.config = { darkMode: 'class', theme: { extend: { fontFamily: { sans: ['Plus Jakarta Sans', 'sans-serif'], mono: ['JetBrains Mono', 'monospace'] } } } };
(function () {
  const t = localStorage.getItem('theme');
  if (t === 'dark' || (!t && matchMedia('(prefers-color-scheme: dark)').matches)) document.documentElement.classList.add('dark');
})();
function getAuthToken() { return localStorage.getItem('token') || sessionStorage.getItem('token'); }
function getAuthUsername() { return localStorage.getItem('username') || sessionStorage.getItem('username') || 'User'; }
function clearAuthSession() {
  localStorage.removeItem('token'); localStorage.removeItem('username');
  sessionStorage.removeItem('token'); sessionStorage.removeItem('username');
}
async function authFetch(url, options = {}) {
  options.headers = { ...options.headers, Authorization: `Bearer ${getAuthToken()}` };
  const result = await fetch(url, options);
  if (result.status === 401) { clearAuthSession(); location.replace('/login'); }
  return result;
}
