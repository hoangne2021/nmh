let D;
      const cssv = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
      const cols = () => [cssv('--c-setosa'), cssv('--c-versicolor'), cssv('--c-virginica')];
      function paint() {
        Object.values(Chart.instances).forEach(c => c.destroy()); const c = cols();
        new Chart(document.getElementById('pie'), { type:'doughnut', data:{ labels:D.species.map(s=>s.name), datasets:[{ data:D.species.map(s=>s.count), backgroundColor:c, borderWidth:0 }] }, options:{ maintainAspectRatio:false, plugins:{ legend:{ position:'bottom' } } } });
        new Chart(document.getElementById('bar'), { type:'bar', data:{ labels:D.features, datasets:D.species.map((s,i)=>({ label:s.name, data:s.mean, backgroundColor:c[i], borderRadius:6 })) }, options:{ maintainAspectRatio:false, plugins:{ legend:{ position:'bottom' } } } });
      }
      function onThemeChange() { if (D) paint(); }
      window.addEventListener('DOMContentLoaded', async () => { try { D = await (await authFetch('/api/dataset')).json(); paint(); } catch (e) { toast('Lỗi kết nối.'); } });
