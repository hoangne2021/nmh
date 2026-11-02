if (window.AOS) AOS.init({ once: true });

const forms = {
  login: document.getElementById('form-login'),
  register: document.getElementById('form-register'),
  forgot: document.getElementById('form-forgot'),
};
const recoveryModal = document.getElementById('recovery-modal');
let lastRecoveryCode = '';

function showToast(msg, isSuccess = true) {
  const toast = document.getElementById('toast');
  toast.textContent = typeof msg === 'string' ? msg : 'Yêu cầu không hợp lệ';
  toast.className = `fixed top-5 left-1/2 -translate-x-1/2 px-6 py-3 rounded-full text-sm font-bold text-white transition-all duration-300 pointer-events-none z-50 ${isSuccess ? 'bg-green-500' : 'bg-red-500'} opacity-100 translate-y-0 shadow-lg`;
  setTimeout(() => { toast.classList.remove('opacity-100'); toast.classList.add('opacity-0'); }, 3500);
}

function switchTab(tab) {
  for (const [name, form] of Object.entries(forms)) {
    form.style.opacity = name === tab ? '1' : '0';
    form.style.pointerEvents = name === tab ? 'auto' : 'none';
    form.style.transform = name === tab ? 'translateX(0)' : 'translateX(20px)';
    form.inert = name !== tab;
  }
}
document.querySelectorAll('[data-auth-tab]').forEach(button => {
  button.addEventListener('click', () => switchTab(button.dataset.authTab));
});
switchTab(new URLSearchParams(location.search).get('mode') === 'forgot' ? 'forgot' : 'login');

async function postJson(url, body) {
  const response = await fetch(url, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) {
    const err = Array.isArray(data.detail) ? data.detail[0]?.msg : data.detail;
    throw new Error(err || `Máy chủ trả về ${response.status}`);
  }
  return data;
}

async function submitWithState(event, task) {
  event.preventDefault();
  const button = event.currentTarget.querySelector('button[type="submit"]');
  const oldHtml = button.innerHTML;
  button.disabled = true;
  button.textContent = 'Đang xử lý...';
  try { await task(); }
  catch (error) { showToast(error.message || 'Không thể kết nối tới máy chủ', false); }
  finally { button.disabled = false; button.innerHTML = oldHtml; }
}

function showRecoveryCode(code) {
  lastRecoveryCode = code;
  document.getElementById('recovery-code').textContent = code;
  recoveryModal.classList.remove('hidden');
  recoveryModal.classList.add('flex');
  document.getElementById('recovery-done').focus();
}
document.getElementById('recovery-done').onclick = () => {
  recoveryModal.classList.add('hidden');
  recoveryModal.classList.remove('flex');
  lastRecoveryCode = '';
  document.getElementById('recovery-code').textContent = '';
  document.getElementById('l-pass').focus();
};
document.getElementById('copy-recovery').onclick = async () => {
  try {
    await navigator.clipboard.writeText(lastRecoveryCode);
    showToast('Đã sao chép mã khôi phục');
  } catch (_) {
    showToast('Hãy bôi đen mã và sao chép thủ công', false);
  }
};

document.getElementById('form-register').onsubmit = event => submitWithState(event, async () => {
  const username = document.getElementById('r-user').value.trim();
  const password = document.getElementById('r-pass').value;
  const result = await postJson('/auth/register', { username, password });
  switchTab('login');
  document.getElementById('l-user').value = username;
  document.getElementById('l-pass').value = '';
  document.getElementById('r-pass').value = '';
  showToast('Tạo tài khoản thành công');
  showRecoveryCode(result.recovery_code);
});

document.getElementById('form-login').onsubmit = event => submitWithState(event, async () => {
  const username = document.getElementById('l-user').value.trim();
  const password = document.getElementById('l-pass').value;
  const result = await postJson('/auth/login', { username, password });
  const remember = document.getElementById('remember-me').checked;
  const store = remember ? localStorage : sessionStorage;
  const old = remember ? sessionStorage : localStorage;
  old.removeItem('token'); old.removeItem('username');
  store.setItem('token', result.token);
  store.setItem('username', result.username);
  location.assign('/');
});

document.getElementById('form-forgot').onsubmit = event => submitWithState(event, async () => {
  const username = document.getElementById('f-user').value.trim();
  const result = await postJson('/auth/forgot-password', {
    username,
    recovery_code: document.getElementById('f-recovery').value.trim(),
    new_password: document.getElementById('f-pass').value,
  });
  switchTab('login');
  document.getElementById('l-user').value = username;
  document.getElementById('l-pass').value = '';
  document.getElementById('f-recovery').value = '';
  document.getElementById('f-pass').value = '';
  showToast('Mật khẩu đã được thay đổi');
  showRecoveryCode(result.recovery_code); // mã cũ bị vô hiệu hóa
});
