let scatterChart, lastPayload = null; const KNAME = { linear:'Linear', rbf:'RBF', poly:'Polynomial', sigmoid:'Sigmoid' };
      const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
      function initCharts() { const c = [cssv('--c-setosa'), cssv('--c-versicolor'), cssv('--c-virginica')]; scatterChart = new Chart(document.getElementById('scatterChart').getContext('2d'), { type:'scatter', data:{ datasets:[ { label:'Setosa', data:[{x:1.4,y:0.2}], backgroundColor:c[0]+'99' }, { label:'Versicolor', data:[{x:4.7,y:1.4}], backgroundColor:c[1]+'99' }, { label:'Virginica', data:[{x:6.0,y:2.5}], backgroundColor:c[2]+'99' }, { label:'Mẫu mới', data:[], backgroundColor:cssv('--text'), pointRadius:8 } ] }, options:{ responsive:true, maintainAspectRatio:false } }); }
      function syncVal(s, t) { document.getElementById(t).textContent = document.getElementById(s).value; }
      function randomizeInputs() { ['sl','sw','pl','pw'].forEach(id => { const input = document.getElementById(id); const v = Math.random() * (Number(input.max) - Number(input.min)) + Number(input.min); input.value = v.toFixed(1); syncVal(id, 'txt-' + id); }); }
      function renderHistory(data) { document.getElementById('history-tbody').innerHTML = data.map(r => `<tr class="border-b" style="border-color:var(--border)"><td class="py-2 px-4 text-xs font-mono muted">${r.timestamp}</td><td class="py-2 px-4">${r.pl}</td><td class="py-2 px-4">${r.pw}</td><td class="py-2 px-4 font-bold accent">${r.species} (${r.confidence}%)</td></tr>`).join(''); }
      async function fetchHistory() { const res = await authFetch('/api/history'); if (res && res.ok) renderHistory(await res.json()); }
      async function clearHistory() { if (await confirmModal('Xóa lịch sử?', 'Chắc chắn xóa lịch sử của bạn?')) { const res = await authFetch('/api/history', { method:'DELETE' }); if (res.ok) { await fetchHistory(); toast('Đã xóa'); } else { toast('Không xóa được lịch sử'); } } }
      async function executeInference() {
        const payload = { sepal_length:parseFloat(document.getElementById('sl').value), sepal_width:parseFloat(document.getElementById('sw').value), petal_length:parseFloat(document.getElementById('pl').value), petal_width:parseFloat(document.getElementById('pw').value) };
        const res = await authFetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
        if (res && res.ok) { const d = await res.json(); document.getElementById('specimen-title').textContent = d.prediction; document.getElementById('specimen-title').style.color = d.color; document.getElementById('main-conf').textContent = d.confidence + '%'; scatterChart.data.datasets[3].data = [{ x:payload.petal_length, y:payload.petal_width }]; scatterChart.update(); lastPayload = payload; document.getElementById('btn-compare').classList.remove('hidden'); document.getElementById('btn-compare').disabled = false; fetchHistory(); }
      }
      async function compareKernels() {
        const res = await authFetch('/api/compare_kernels', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(lastPayload) });
        if (res && res.ok) { const rows = await res.json(); document.getElementById('compare-panel').classList.remove('hidden'); document.getElementById('compare-grid').innerHTML = rows.map((r, i) => `<div class="soft p-4 pop"><div class="text-xs muted mb-1">${KNAME[r.kernel]}</div><div class="font-bold" style="color:${r.color}">${r.prediction}</div><div class="font-mono text-2xl font-black mt-1">${r.confidence}%</div></div>`).join(''); }
      }
      window.addEventListener('DOMContentLoaded', () => {
  document.getElementById('randomize-button').addEventListener('click', randomizeInputs);
  document.getElementById('btn-predict').addEventListener('click', executeInference);
  document.getElementById('btn-compare').addEventListener('click', compareKernels);
  document.getElementById('clear-history-button').addEventListener('click', clearHistory);
  for (const id of ['sl', 'sw', 'pl', 'pw']) {
    document.getElementById(id).addEventListener('input', () => syncVal(id, 'txt-' + id));
  }
  initCharts(); fetchHistory();
});
