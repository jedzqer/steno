'use strict';
/* steno WebUI 前端逻辑（纯 DOM 构建，无 innerHTML） */
(() => {

// ============================ 基础 ============================
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];

const VIDEO_EXT = ['mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'webm', 'm4v'];
const AUDIO_EXT = ['wav', 'mp3', 'flac', 'ogg', 'aac', 'm4a', 'wma'];
const STAGE_LABEL = { download: '下载', load: '加载模型', extract: '提取音频', transcribe: '转录' };

const state = {
  view: 'new',
  tab: 'file',
  file: null,            // File 对象
  jobs: [],              // 服务端任务列表（新→旧）
  prevStatus: {},        // 任务状态快照，用于检测完成沿
  uploading: null,       // {name, loaded, total, xhr}
  result: null,          // {name, title, text}
  history: [],
  info: null,
  pollTimer: null,
  pollFail: 0,
  serverDown: false,
};

// ============================ DOM 构建 ============================
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') el.className = v;
    else if (k === 'text') el.textContent = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
  for (const c of children.flat()) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(c));
  }
  return el;
}

const SVG_NS = 'http://www.w3.org/2000/svg';
function svgIcon(paths, sw = 1.8) {
  const svg = document.createElementNS(SVG_NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', sw);
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  for (const [tag, p] of paths) {
    const e = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(p)) e.setAttribute(k, v);
    svg.append(e);
  }
  return svg;
}
const ICON = {
  close: () => svgIcon([['path', { d: 'm6 6 12 12M18 6 6 18' }]]),
};

// ============================ 工具 ============================
function fmtBytes(n) {
  if (n == null || isNaN(n)) return '';
  const u = ['B', 'KB', 'MB', 'GB']; let i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return (i === 0 || n >= 100 ? Math.round(n) : n.toFixed(1)) + ' ' + u[i];
}
const fmtSpeed = bps => (bps ? fmtBytes(bps) + '/s' : '');
function fmtTime(ts) {
  const d = new Date(ts * 1000), p = x => String(x).padStart(2, '0');
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}
function fmtElapsed(sec) {
  if (sec == null) return '';
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${m}:${String(s).padStart(2, '0')}`;
}
const extOf = name => (name.toLowerCase().match(/\.([a-z0-9]+)$/) || [])[1] || '';
const isSupported = name => [...VIDEO_EXT, ...AUDIO_EXT].includes(extOf(name));

function toast(msg, type = 'ok') {
  const el = h('div', { class: `toast ${type}`, text: msg });
  $('#toasts').append(el);
  setTimeout(() => {
    el.classList.add('leaving');
    el.addEventListener('animationend', () => el.remove(), { once: true });
  }, 3200);
}

async function api(path, opts = {}) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { msg = (await r.json()).detail || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

// ============================ 系统状态 ============================
function chipEl(label, type) {
  return h('span', { class: `chip ${type}` }, h('i'), label);
}
function renderInfo() {
  const i = state.info;
  if (!i) return;
  const chips = [];
  chips.push(chipEl(i.cuda ? 'GPU · CUDA 就绪' : 'CPU 模式', i.cuda ? 'ok' : 'warn'));
  if (!i.backend_ok) {
    chips.push(chipEl('后端依赖缺失', 'err'));
  } else if (i.model_loaded) {
    chips.push(chipEl('模型已加载', 'ok'));
  } else {
    chips.push(chipEl(i.model_dir_exists ? '模型未加载' : '模型权重缺失', i.model_dir_exists ? 'warn' : 'err'));
  }
  chips.push(chipEl('FFmpeg', i.ffmpeg ? 'ok' : 'err'));
  chips.push(chipEl('yt-dlp', i.yt_dlp ? 'ok' : 'err'));
  $('#sysChips').replaceChildren(...chips);
  $('#startHint').replaceChildren(
    !i.backend_ok ? '后端依赖缺失，请先安装 torch / funasr'
    : !i.model_loaded && i.model_dir_exists ? '首次转录需先加载模型，约需十几秒'
    : !i.ffmpeg ? '未检测到 FFmpeg，视频文件将无法提取音频'
    : ''
  );
}
async function loadInfo() {
  try { state.info = await api('/api/info'); renderInfo(); } catch { /* 忽略 */ }
}

// ============================ 视图与 Tab ============================
function switchView(view) {
  state.view = view;
  $$('.nav-item').forEach(b => b.classList.toggle('active', b.dataset.view === view));
  $('#viewNew').classList.toggle('hidden', view !== 'new');
  $('#viewHistory').classList.toggle('hidden', view !== 'history');
  if (view === 'history') loadHistory();
}
function switchTab(tab) {
  state.tab = tab;
  $$('#tabs .tab').forEach(b => b.classList.toggle('active', b.dataset.tab === tab));
  $('#paneFile').classList.toggle('hidden', tab !== 'file');
  $('#paneUrl').classList.toggle('hidden', tab !== 'url');
}

// ============================ 文件选择 ============================
function setFile(f) {
  if (!f) return;
  if (!isSupported(f.name)) {
    toast(`不支持的格式 .${extOf(f.name) || '(无后缀)'}`, 'err');
    return;
  }
  state.file = f;
  $('#chipName').textContent = f.name;
  $('#chipMeta').textContent = fmtBytes(f.size);
  $('#fileChip').classList.remove('hidden');
  $('#dropZone').classList.add('hidden');
}
function clearFile() {
  state.file = null;
  $('#fileInput').value = '';
  $('#fileChip').classList.add('hidden');
  $('#dropZone').classList.remove('hidden');
}

// ============================ 提交任务 ============================
function collectOpts(fd) {
  fd.append('audio_only', $('#audioOnly').checked);
  fd.append('cookies', $('#cookiesFrom').value);
  fd.append('language', $('#lang').value);
  fd.append('device', $('#device').value);
  fd.append('use_itn', $('#useItn').checked);
  fd.append('chunk', Math.max(30, Math.min(600, +$('#chunk').value || 120)));
  fd.append('overlap', Math.max(0, Math.min(30, +$('#overlap').value || 1)));
  return fd;
}
function startJob() {
  if (state.serverDown) { toast('服务已停止，请重新启动 webui.py', 'err'); return; }
  if (state.tab === 'file') {
    if (!state.file) { toast('请先选择或拖放文件', 'warn'); return; }
    uploadFile();
  } else {
    const url = $('#urlInput').value.trim();
    if (!url) { toast('请先粘贴视频链接', 'warn'); return; }
    const fd = collectOpts(new FormData());
    fd.append('mode', 'url');
    fd.append('url', url);
    api('/api/jobs', { method: 'POST', body: fd })
      .then(afterSubmit)
      .catch(e => toast(e.message, 'err'));
  }
}
function uploadFile() {
  const fd = collectOpts(new FormData());
  fd.append('mode', 'file');
  fd.append('file', state.file);
  const xhr = new XMLHttpRequest();
  state.uploading = { name: state.file.name, loaded: 0, total: state.file.size, xhr };
  renderJobs();
  xhr.open('POST', '/api/jobs');
  xhr.upload.onprogress = e => {
    if (!state.uploading) return;
    state.uploading.loaded = e.loaded;
    state.uploading.total = e.total;
    renderJobs();
  };
  xhr.onload = () => {
    state.uploading = null;
    if (xhr.status >= 200 && xhr.status < 300) { afterSubmit(); }
    else {
      let msg = xhr.statusText;
      try { msg = JSON.parse(xhr.responseText).detail || msg; } catch { /* ignore */ }
      toast('上传失败：' + msg, 'err');
      renderJobs();
    }
  };
  xhr.onerror = () => { state.uploading = null; renderJobs(); toast('上传失败：网络错误', 'err'); };
  xhr.send(fd);
}
function afterSubmit() {
  clearFile();
  $('#urlInput').value = '';
  toast('已加入任务队列');
  renderJobs();
  loadInfo();
  ensurePolling();
}

// ============================ 任务渲染 ============================
const stagesFor = job => {
  const audio = AUDIO_EXT.includes(extOf(job.display));
  if (job.kind === 'url') return audio ? ['download', 'load', 'transcribe'] : ['download', 'load', 'extract', 'transcribe'];
  return audio ? ['load', 'transcribe'] : ['load', 'extract', 'transcribe'];
};
function stepperEl(job) {
  const stages = stagesFor(job);
  const cur = stages.indexOf(job.stage);
  const parts = [];
  stages.forEach((s, i) => {
    const cls = i < cur ? 'done' : (i === cur ? 'active' : '');
    parts.push(h('div', { class: `step ${cls}` },
      h('span', { class: 'dot', text: i < cur ? '✓' : '' }),
      h('span', { class: 'lbl', text: STAGE_LABEL[s] })));
    if (i < stages.length - 1) parts.push(h('span', { class: `step-line ${i < cur ? 'done' : ''}` }));
  });
  return h('div', { class: 'stepper' }, ...parts);
}
function barEl(job) {
  const indet = job.progress == null;
  const pct = indet ? 0 : Math.round(job.progress * 100);
  return h('div', { class: `bar ${indet ? 'indet' : ''}` },
    h('div', { class: 'bar-fill', style: `width:${pct}%` }));
}
function metaEl(job) {
  const parts = [];
  if (job.stage === 'download' && job.total) {
    parts.push(`${fmtBytes(job.downloaded)} / ${fmtBytes(job.total)}`);
  } else if (job.stage === 'transcribe' && job.progress != null) {
    parts.push(`第 ${Math.round(job.progress * 100)}%`);
  }
  if (job.speed) parts.push(fmtSpeed(job.speed));
  if (job.started) parts.push(`已用 ${fmtElapsed((job.finished || Date.now() / 1000) - job.started)}`);
  return h('div', { class: 'job-meta' }, ...parts.map(p => h('span', { text: p })));
}
const PILL = {
  queued: ['wait', '排队中'], running: ['run', '进行中'],
  done: ['ok', '完成'], error: ['err', '失败'], cancelled: ['wait', '已取消'],
};
const pillEl = status => PILL[status] ? h('span', { class: `pill ${PILL[status][0]}`, text: PILL[status][1] }) : null;

function jobCardEl(job) {
  const running = job.status === 'running';
  const head = h('div', { class: 'job-head' },
    running ? h('span', { class: 'spin' }) : null,
    h('b', { text: job.display, title: job.display }),
    pillEl(job.status),
    (job.status === 'queued' || job.status === 'running')
      ? h('button', { class: 'ghost-btn mini-btn', text: '取消', 'data-cancel': job.id })
      : null);
  const card = h('div', { class: 'card job-card' }, head);
  if (job.status !== 'queued') {
    card.append(stepperEl(job), barEl(job), metaEl(job));
  }
  if (job.status === 'error') {
    card.append(h('div', { class: 'job-error', text: job.error || '未知错误' }));
  }
  return card;
}
function jobRowEl(job) {
  return h('div', { class: 'job-row' },
    h('b', { text: job.display, title: job.display }),
    pillEl(job.status),
    (job.status === 'done' && job.result)
      ? h('button', { class: 'ghost-btn mini-btn', text: '查看', 'data-view-result': job.id })
      : null,
    h('button', { class: 'icon-btn', title: '从列表移除', 'data-del-job': job.id }, ICON.close()));
}
function uploadCardEl(u) {
  const pct = u.total ? Math.round(u.loaded / u.total * 100) : 0;
  return h('div', { class: 'card job-card' },
    h('div', { class: 'job-head' },
      h('span', { class: 'spin' }),
      h('b', { text: u.name, title: u.name }),
      h('span', { class: 'pill run', text: '上传中' })),
    h('div', { class: 'bar' }, h('div', { class: 'bar-fill', style: `width:${pct}%` })),
    h('div', { class: 'job-meta' },
      h('span', { text: `${fmtBytes(u.loaded)} / ${fmtBytes(u.total)} · ${pct}%` })));
}
function renderJobs() {
  const wrap = $('#jobsWrap');
  const u = state.uploading;
  const active = state.jobs.filter(j => j.status === 'queued' || j.status === 'running');
  const recent = state.jobs.filter(j => j.status !== 'queued' && j.status !== 'running').slice(0, 6);

  if (!u && !active.length && !recent.length) { wrap.classList.add('hidden'); return; }
  wrap.classList.remove('hidden');

  const nodes = [];
  if (u) nodes.push(uploadCardEl(u));
  nodes.push(...active.map(jobCardEl));
  nodes.push(...recent.map(jobRowEl));
  $('#jobList').replaceChildren(...nodes);
}

// ============================ 结果面板 ============================
async function showResult(name, title) {
  try {
    const text = await api('/api/history/content?name=' + encodeURIComponent(name));
    state.result = { name, title: title || name.replace(/_out\.txt$/, ''), text };
    $('#resultTitle').textContent = '转录结果 · ' + state.result.title;
    $('#resultText').textContent = text;
    $('#charCount').textContent = `共 ${text.length} 字`;
    $('#resultWrap').classList.remove('hidden');
  } catch (e) { toast('读取结果失败：' + e.message, 'err'); }
}
async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    const ta = document.createElement('textarea');   // 非安全上下文兜底
    ta.value = text; document.body.appendChild(ta);
    ta.select(); document.execCommand('copy'); ta.remove();
  }
  toast('已复制到剪贴板');
}
function downloadText(name, text) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name.replace(/_out\.txt$/, '') + '.txt';
  a.click();
  URL.revokeObjectURL(a.href);
}

// ============================ 历史记录 ============================
const extClass = title => {
  const e = extOf(title);
  return VIDEO_EXT.includes(e) ? 'video' : AUDIO_EXT.includes(e) ? 'audio' : 'other';
};
function histCardEl(item) {
  return h('div', { class: 'hist-card', 'data-hist': item.name },
    h('div', { class: `hc-icon ${extClass(item.title)}`, text: (extOf(item.title) || 'txt').toUpperCase().slice(0, 4) }),
    h('div', { class: 'hc-body' },
      h('b', { text: item.title, title: item.title }),
      h('span', { text: `${fmtTime(item.mtime)} · 文本 ${fmtBytes(item.size)}` +
        (item.media ? ` · 源文件 ${fmtBytes(item.media_size)}` : '') })),
    h('span', { class: 'hc-arrow', text: '→' }));
}
function renderHistory() {
  const q = $('#histSearch').value.trim().toLowerCase();
  const items = state.history.filter(x => !q || x.title.toLowerCase().includes(q));
  $('#histCount').textContent = state.history.length || '';
  if (!items.length) {
    $('#histGrid').replaceChildren(h('div', { class: 'empty' },
      svgIcon([['circle', { cx: 12, cy: 12, r: 8.5 }], ['path', { d: 'M12 7.5V12l3 2' }]], 1.5),
      h('div', { text: q ? '没有匹配的记录' : '还没有转录记录，去创建第一个吧' })));
    return;
  }
  $('#histGrid').replaceChildren(...items.map(histCardEl));
}
async function loadHistory() {
  try {
    state.history = await api('/api/history');
    renderHistory();
  } catch { /* 服务停止时静默 */ }
}

// ============================ 弹窗 ============================
let modalName = null;
async function openModal(name) {
  try {
    const text = await api('/api/history/content?name=' + encodeURIComponent(name));
    modalName = name;
    const item = state.history.find(x => x.name === name);
    $('#modalTitle').textContent = item ? item.title : name.replace(/_out\.txt$/, '');
    $('#modalMeta').textContent = item ? `${fmtTime(item.mtime)} · 共 ${text.length} 字` : '';
    $('#modalText').textContent = text;
    $('#modal').classList.remove('hidden');
    $('#modalText').scrollTop = 0;
  } catch (e) { toast('读取失败：' + e.message, 'err'); }
}
function closeModal() { $('#modal').classList.add('hidden'); modalName = null; }

// ============================ 轮询 ============================
function onJobDone(job) {
  toast('转录完成 ✓');
  loadHistory();
  loadInfo();
  if (job.result) showResult(job.result, job.display);
  setTimeout(() => $('#resultWrap').scrollIntoView({ behavior: 'smooth', block: 'start' }), 120);
}
function pollOnce() {
  const hadWork = state.uploading ||
    state.jobs.some(j => j.status === 'queued' || j.status === 'running');
  api('/api/jobs').then(jobs => {
    state.pollFail = 0;
    if (state.serverDown) { state.serverDown = false; toast('服务已恢复'); }
    for (const j of jobs) {
      const prev = state.prevStatus[j.id];
      if (j.status === 'done' && prev && prev !== 'done') onJobDone(j);
      state.prevStatus[j.id] = j.status;
    }
    state.jobs = jobs;
    renderJobs();
    schedulePoll(hadWork || jobs.some(j => j.status === 'queued' || j.status === 'running'));
  }).catch(() => {
    // 服务退出（用户点了"退出服务"）后静默停止轮询
    if (++state.pollFail >= 3 && !state.serverDown) {
      state.serverDown = true;
      toast('与服务的连接已断开', 'err');
    }
    schedulePoll(true);
  });
}
function schedulePoll(active) {
  clearTimeout(state.pollTimer);
  state.pollTimer = setTimeout(pollOnce, active ? 1100 : 4000);
}
function ensurePolling() {
  if (state.pollTimer) return;
  pollOnce();
}

// ============================ 事件绑定 ============================
function bind() {
  // 侧栏导航
  $$('.nav-item').forEach(b => b.addEventListener('click', () => switchView(b.dataset.view)));

  // Tab 切换
  $$('#tabs .tab').forEach(b => b.addEventListener('click', () => switchTab(b.dataset.tab)));

  // 拖放区
  const dz = $('#dropZone'), fi = $('#fileInput');
  dz.addEventListener('click', () => fi.click());
  dz.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') fi.click(); });
  ['dragover', 'dragenter'].forEach(ev =>
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.add('dragover'); }));
  ['dragleave', 'drop'].forEach(ev =>
    dz.addEventListener(ev, e => { e.preventDefault(); dz.classList.remove('dragover'); }));
  dz.addEventListener('drop', e => setFile(e.dataTransfer.files[0]));
  fi.addEventListener('change', () => setFile(fi.files[0]));
  document.addEventListener('dragover', e => e.preventDefault());
  document.addEventListener('drop', e => e.preventDefault());
  $('#chipRemove').addEventListener('click', clearFile);

  // 提交
  $('#startBtn').addEventListener('click', startJob);
  $('#urlInput').addEventListener('keydown', e => { if (e.key === 'Enter') startJob(); });

  // 任务列表（事件委托）
  $('#jobList').addEventListener('click', e => {
    const cancelId = e.target.closest('[data-cancel]')?.dataset.cancel;
    if (cancelId) {
      api(`/api/jobs/${cancelId}/cancel`, { method: 'POST' })
        .then(() => { toast('已请求取消'); pollOnce(); })
        .catch(err => toast(err.message, 'err'));
      return;
    }
    const viewId = e.target.closest('[data-view-result]')?.dataset.viewResult;
    if (viewId) {
      const job = state.jobs.find(j => j.id === viewId);
      if (job?.result) showResult(job.result, job.display);
      return;
    }
    const delId = e.target.closest('[data-del-job]')?.dataset.delJob;
    if (delId) {
      api(`/api/jobs/${delId}`, { method: 'DELETE' })
        .then(() => { delete state.prevStatus[delId]; pollOnce(); })
        .catch(err => toast(err.message, 'err'));
    }
  });

  // 结果操作
  $('#copyBtn').addEventListener('click', () => copyText(state.result ? state.result.text : ''));
  $('#dlBtn').addEventListener('click', () => {
    if (state.result) downloadText(state.result.name, state.result.text);
  });

  // 历史
  $('#histSearch').addEventListener('input', renderHistory);
  $('#histGrid').addEventListener('click', e => {
    const card = e.target.closest('[data-hist]');
    if (card) openModal(card.dataset.hist);
  });

  // 弹窗
  $('#modalClose').addEventListener('click', closeModal);
  $('#modal').addEventListener('click', e => { if (e.target === $('#modal')) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
  $('#modalCopy').addEventListener('click', () => copyText($('#modalText').textContent));
  $('#modalDl').addEventListener('click', () => { if (modalName) downloadText(modalName, $('#modalText').textContent); });
  $('#modalDelete').addEventListener('click', () => {
    if (!modalName) return;
    if (!confirm('确定删除这条记录？\n' + modalName + '\n（同名媒体文件也会一并删除）')) return;
    api('/api/history?name=' + encodeURIComponent(modalName), { method: 'DELETE' })
      .then(() => {
        if (state.result?.name === modalName) { state.result = null; $('#resultWrap').classList.add('hidden'); }
        closeModal();
        toast('记录已删除');
        loadHistory();
      })
      .catch(err => toast(err.message, 'err'));
  });

  // 侧栏操作
  $('#openFolderBtn').addEventListener('click', () =>
    api('/api/open-folder', { method: 'POST' }).catch(err => toast(err.message, 'err')));
  $('#quitBtn').addEventListener('click', () => {
    if (!confirm('确定退出 steno 服务？\n（正在进行的任务将被中断，浏览器页面也会失去连接）')) return;
    api('/api/quit', { method: 'POST' })
      .then(() => { state.serverDown = true; toast('服务已停止，可关闭此页面', 'warn'); })
      .catch(() => {});
  });

  $('#footUrl').textContent = location.host;
}

// ============================ 启动 ============================
bind();
loadInfo();
loadHistory();
ensurePolling();

})();
