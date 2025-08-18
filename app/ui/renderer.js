/* globals config */
const API = (typeof window !== 'undefined' && window.config && window.config.API_BASE_URL) || 'http://127.0.0.1:8000';
document.getElementById('apiUrl').textContent = API;

// Tabs
const tabs = document.querySelectorAll('.tabs button');
const sections = document.querySelectorAll('.tab');
tabs.forEach(btn => {
  btn.addEventListener('click', () => {
    tabs.forEach(b => b.classList.remove('active'));
    sections.forEach(s => s.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`tab-${btn.dataset.tab}`).classList.add('active');
  });
});

// Health
const healthStatus = document.getElementById('healthStatus');
document.getElementById('checkHealth').addEventListener('click', async () => {
  try {
    const res = await fetch(`${API}/health`);
    const data = await res.json();
    healthStatus.textContent = `API: ${data.status} (model ${data.model})`;
    healthStatus.className = 'status ok';
  } catch {
    healthStatus.textContent = 'API: unreachable';
    healthStatus.className = 'status bad';
  }
});

// Live camera
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const startCam = document.getElementById('startCam');
const snapAndPredict = document.getElementById('snapAndPredict');
const liveLabel = document.getElementById('liveLabel');
const liveScore = document.getElementById('liveScore');
const liveLatency = document.getElementById('liveLatency');

startCam.addEventListener('click', async () => {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
  } catch {
    alert('Camera permission denied or not available.');
  }
});

snapAndPredict.addEventListener('click', async () => {
  if (!video.srcObject) return alert('Start the camera first.');
  const w = video.videoWidth || 640;
  const h = video.videoHeight || 480;
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, w, h);

  canvas.toBlob(async (blob) => {
    if (!blob) return alert('Could not capture frame.');
    const form = new FormData();
    form.append('image', blob, 'frame.jpg');

    const t0 = performance.now();
    try {
      const res = await fetch(`${API}/predict-image`, { method: 'POST', body: form });
      const data = await res.json();
      const t1 = performance.now();
      renderResult('LIVE', data, t1 - t0);
      liveLabel.textContent = data.label ?? '-';
      liveScore.textContent = (data.score ?? 0).toFixed(3);
      liveLatency.textContent = Math.round(data.meta?.latency_ms ?? (t1 - t0));
    } catch {
      alert('Prediction failed.');
    }
  }, 'image/jpeg', 0.9);
});

// Upload → predict
const fileInput = document.getElementById('fileInput');
const uploadPredict = document.getElementById('uploadPredict');
const preview = document.getElementById('preview');
const upLabel = document.getElementById('upLabel');
const upScore = document.getElementById('upScore');
const upLatency = document.getElementById('upLatency');

fileInput.addEventListener('change', () => {
  const f = fileInput.files?.[0];
  if (!f) return;
  preview.src = URL.createObjectURL(f);
});

uploadPredict.addEventListener('click', async () => {
  const f = fileInput.files?.[0];
  if (!f) return alert('Choose an image first.');
  const form = new FormData();
  form.append('image', f, f.name);
  const t0 = performance.now();
  try {
    const res = await fetch(`${API}/predict-image`, { method: 'POST', body: form });
    const data = await res.json();
    const t1 = performance.now();
    renderResult(f.name, data, t1 - t0);
    upLabel.textContent = data.label ?? '-';
    upScore.textContent = (data.score ?? 0).toFixed(3);
    upLatency.textContent = Math.round(data.meta?.latency_ms ?? (t1 - t0));
  } catch {
    alert('Prediction failed.');
  }
});

// Results + Analytics (shared)
const resultsBody = document.querySelector('#resultsTable tbody');
const totalPredsEl = document.getElementById('totalPreds');
const cleanCountEl = document.getElementById('cleanCount');
const dirtyCountEl = document.getElementById('dirtyCount');

let totalPreds = 0, cleanCount = 0, dirtyCount = 0;
// store rows in memory for CSV
const sessionRows = []; // { time, source, label, score, latency }

// helper to add one result to table + memory + counters
function renderResult(source, data, clientLatencyMs) {
  const timeStr = new Date().toLocaleString();
  const label = data.label ?? '-';
  const scoreNum = Number(data.score ?? 0);
  const latency = Math.round(data.meta?.latency_ms ?? clientLatencyMs);

  // counters
  totalPreds++;
  if (label === 'CLEAN') cleanCount++;
  if (label === 'DIRTY') dirtyCount++;

  // table DOM
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td>${timeStr}</td>
    <td>${source}</td>
    <td>${label}</td>
    <td>${scoreNum.toFixed(3)}</td>
    <td>${latency}</td>
  `;
  resultsBody.prepend(tr);

  // update counters UI
  totalPredsEl.textContent = totalPreds;
  cleanCountEl.textContent = cleanCount;
  dirtyCountEl.textContent = dirtyCount;

  // memory for CSV
  sessionRows.push({ time: timeStr, source, label, score: scoreNum, latency });
}

// === Batch predict ===
const batchInput = document.getElementById('batchFileInput');
const batchPredictBtn = document.getElementById('batchPredict');
const batchSelected = document.getElementById('batchSelected');
const batchTableBody = document.querySelector('#batchTable tbody');
const batchSummary = document.getElementById('batchSummary');
const batchDownloadBtn = document.getElementById('batchDownloadCsv');

let lastBatchRows = []; // { file, label, score, latency }

batchInput?.addEventListener('change', () => {
  const n = batchInput.files?.length || 0;
  batchSelected.textContent = n ? `${n} file(s) selected` : 'No files selected';
});

batchPredictBtn?.addEventListener('click', async () => {
  const files = batchInput.files;
  if (!files || files.length === 0) return alert('Choose one or more images.');
  batchTableBody.innerHTML = '';
  batchSummary.textContent = 'Running...';
  lastBatchRows = [];

  const form = new FormData();
  Array.from(files).forEach(f => form.append('files', f, f.name)); // field name MUST be "files"

  const t0 = performance.now();
  try {
    const res = await fetch(`${API}/predict-batch`, { method: 'POST', body: form });
    const json = await res.json();
    const t1 = performance.now();

    const items = Array.isArray(json) ? json : (json.items || json.results || []);
    if (!Array.isArray(items) || items.length === 0) {
      batchSummary.textContent = 'No results returned.';
      return;
    }

    let ok = 0;
    items.forEach(item => {
      const label = item.label ?? '-';
      const scoreNum = Number(item.score ?? 0);
      const latency = Math.round(item.meta?.latency_ms ?? ((t1 - t0) / items.length));
      const src = item.filename || item.name || item.file || 'BATCH';

      // batch table
      const tr = document.createElement('tr');
      tr.innerHTML = `<td>${src}</td><td>${label}</td><td>${scoreNum.toFixed(3)}</td><td>${latency}</td>`;
      batchTableBody.appendChild(tr);

      // global session & analytics
      renderResult(src, { label, score: scoreNum, meta: { latency_ms: latency } }, latency);

      lastBatchRows.push({ file: src, label, score: scoreNum, latency });
      if (label === 'CLEAN' || label === 'DIRTY') ok++;
    });

    batchSummary.textContent = `Completed: ${ok}/${items.length} items`;
  } catch (e) {
    console.error(e);
    batchSummary.textContent = 'Batch failed.';
    alert('Batch prediction failed.');
  }
});

// === CSV download helpers ===
function toCsv(rows) {
  return rows.map(r => r.map(v => {
    const s = String(v ?? '');
    // escape quotes & commas
    if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
    return s;
  }).join(',')).join('\n');
}
function downloadCsv(rows, filename) {
  const csv = toCsv(rows);
  const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// buttons
document.getElementById('downloadSessionCsv')?.addEventListener('click', () => {
  if (!sessionRows.length) return alert('No session results yet.');
  const rows = [['When','Source','Label','Score','Latency (ms)'],
                ...sessionRows.map(o => [o.time, o.source, o.label, o.score, o.latency])];
  downloadCsv(rows, 'session_results.csv');
});

batchDownloadBtn?.addEventListener('click', () => {
  if (!lastBatchRows.length) return alert('Run a batch first.');
  const rows = [['File','Label','Score','Latency (ms)'],
                ...lastBatchRows.map(o => [o.file, o.label, o.score, o.latency])];
  downloadCsv(rows, 'batch_results.csv');
});
