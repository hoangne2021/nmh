// Chạy trước khi giao diện xuất hiện để áp dụng Tailwind/theme.
if (window.tailwind) tailwind.config = { darkMode: 'class', theme: { extend: { colors: { accent: '#ea580c', surface: '#ffffff', 'surface-dark': '#1a1715' }, fontFamily: { sans: ['Plus Jakarta Sans','sans-serif'] } } } };
if (localStorage.getItem('theme') === 'dark' || (!localStorage.getItem('theme') && matchMedia('(prefers-color-scheme: dark)').matches)) {
  document.documentElement.classList.add('dark');
}
// Tránh vòng lặp /login -> / nếu browser còn lưu token đã hết hạn.
(async () => {
  if (new URLSearchParams(location.search).get('mode') === 'forgot') return;
  const token = localStorage.getItem('token') || sessionStorage.getItem('token');
  if (!token) return;
  try {
    const result = await fetch('/auth/me', { headers: { Authorization: `Bearer ${token}` } });
    if (result.ok) location.replace('/');
    else if (result.status === 401) {
      localStorage.removeItem('token'); localStorage.removeItem('username');
      sessionStorage.removeItem('token'); sessionStorage.removeItem('username');
    }
  } catch (_) { /* mất mạng: không xóa token nếu server chỉ tạm thời offline */ }
})();
