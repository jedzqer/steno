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
  watching: new Set(),   // 本次会话提交的任务 id：首次观测即 done 也要报完成
  notifiedDone: new Set(), // 已提示过终态的任务 id，防重复提示
  submitting: false,     // URL 任务提交进行中（防重复提交）
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
  if (n == null || Number.isNaN(n)) return '';
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
    try { const body = await r.json(); msg = body.detail || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return r.json();
}

// 纯文本接口（/api/history/content 返回 PlainTextResponse，不能按 JSON 解析）
async function apiText(path) {
  const r = await fetch(path);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try { const body = await r.json(); msg = body.detail || msg; } catch { /* ignore */ }
    throw new Error(msg);
  }
  return r.text();
}

// ============================ 系统状态 ============================
function chipEl(label, type) {
  return h('span', { class: `chip ${type}` }, h('i'), label);
}
function startHintText(i) {
  if (!i.backend_ok) return '后端依赖缺失，请先安装 torch / funasr';
  if (!i.model_loaded && i.model_dir_exists) return '首次转录需先加载模型，约需十几秒';
  if (!i.ffmpeg) return '未检测到 FFmpeg，视频文件将无法提取音频';
  return '';
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
  $('#startHint').replaceChildren(startHintText(i));
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
function setStartDisabled(disabled) { $('#startBtn').disabled = disabled; }
function watchJob(id) { if (id) state.watching.add(id); }
function startJob() {
  if (state.serverDown) { toast('服务已停止，请重新启动 webui.py', 'err'); return; }
  if (state.uploading) { toast('正在上传文件，请等待完成或取消上传', 'warn'); return; }
  if (state.submitting) { toast('正在提交，请稍候…', 'warn'); return; }
  if (state.tab === 'file') {
    if (!state.file) { toast('请先选择或拖放文件', 'warn'); return; }
    uploadFile();
  } else {
    const url = $('#urlInput').value.trim();
    if (!url) { toast('请先粘贴视频链接', 'warn'); return; }
    const fd = collectOpts(new FormData());
    fd.append('mode', 'url');
    fd.append('url', url);
    state.submitting = true;
    setStartDisabled(true);
    api('/api/jobs', { method: 'POST', body: fd })
      .then(res => { watchJob(res.id); afterSubmit(); })
      .catch(e => toast(e.message, 'err'))
      .finally(() => { state.submitting = false; setStartDisabled(false); });
  }
}
function uploadFile() {
  const fd = collectOpts(new FormData());
  fd.append('mode', 'file');
  fd.append('file', state.file);
  const xhr = new XMLHttpRequest();
  state.uploading = { name: state.file.name, loaded: 0, total: state.file.size, xhr };
  setStartDisabled(true);
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
    setStartDisabled(false);
    if (xhr.status >= 200 && xhr.status < 300) {
      try { watchJob(JSON.parse(xhr.responseText).id); } catch { /* ignore */ }
      afterSubmit();
    } else {
      let msg = xhr.statusText;
      try { msg = JSON.parse(xhr.responseText).detail || msg; } catch { /* ignore */ }
      toast('上传失败：' + msg, 'err');
      renderJobs();
    }
  };
  xhr.onerror = () => { state.uploading = null; setStartDisabled(false); renderJobs(); toast('上传失败：网络错误', 'err'); };
  xhr.onabort = () => { state.uploading = null; setStartDisabled(false); renderJobs(); };
  xhr.send(fd);
}
function afterSubmit() {
  clearFile();
  $('#urlInput').value = '';
  toast('已加入任务队列');
  hideResult();                 // 新任务开始，清掉面板里残留的上一条结果
  requestNotifyPermission();    // 借用户手势申请系统通知权限
  renderJobs();
  loadInfo();
  pollNow();                    // 立即轮询：模型已缓存时任务可能几秒内完成
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
    let cls = '';
    if (i < cur) cls = 'done';
    else if (i === cur) cls = 'active';
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
    (job.status === 'done' && job.result)
      ? h('button', { class: 'ghost-btn mini-btn', 'data-copy-job': job.id },
          h('span', { class: 'btn-label', text: '复制' }))
      : null,
    h('button', { class: 'icon-btn', title: '从列表移除', 'data-del-job': job.id }, ICON.close()));
}
function uploadCardEl(u) {
  const pct = u.total ? Math.round(u.loaded / u.total * 100) : 0;
  return h('div', { class: 'card job-card' },
    h('div', { class: 'job-head' },
      h('span', { class: 'spin' }),
      h('b', { text: u.name, title: u.name }),
      h('span', { class: 'pill run', text: '上传中' }),
      h('button', { class: 'ghost-btn mini-btn', text: '取消上传', 'data-cancel-upload': '1' })),
    h('div', { class: 'bar' }, h('div', { class: 'bar-fill', style: `width:${pct}%` })),
    h('div', { class: 'job-meta' },
      h('span', { text: `${fmtBytes(u.loaded)} / ${fmtBytes(u.total)} · ${pct}%` })));
}
function renderJobs() {
  const wrap = $('#jobsWrap');
  const u = state.uploading;
  const active = state.jobs.filter(j => j.status === 'queued' || j.status === 'running');
  const recent = state.jobs.filter(j => j.status !== 'queued' && j.status !== 'running');   // 不截断：完成任务的“查看/复制”入口不能在第 7 条后消失

  if (!u && !active.length && !recent.length) { wrap.classList.add('hidden'); return; }
  wrap.classList.remove('hidden');

  const nodes = [];
  if (u) nodes.push(uploadCardEl(u));
  nodes.push(...active.map(jobCardEl));
  nodes.push(...recent.map(jobRowEl));
  $('#jobList').replaceChildren(...nodes);
}

// ============================ 结果面板 ============================
function setExpanded(expanded) {
  $('#resultCard').classList.toggle('expanded', expanded);
  $('#expandLabel').textContent = expanded ? '收起' : '展开全文';
}
function hideResult() {
  state.result = null;
  setExpanded(false);
  $('#resultWrap').classList.add('hidden');
}
async function showResult(name, title) {
  try {
    if (state.view !== 'new') switchView('new');   // 结果面板在“新建转录”视图内，先切过去再展示
    const text = await apiText('/api/history/content?name=' + encodeURIComponent(name));
    state.result = { name, title: title || name.replace(/_out\.txt$/, ''), text };
    $('#resultTitle').textContent = '转录结果 · ' + state.result.title;
    $('#resultText').textContent = text;
    $('#charCount').textContent = `共 ${text.length} 字`;
    setExpanded(false);
    $('#resultWrap').classList.remove('hidden');
  } catch (e) { toast('读取结果失败：' + e.message, 'err'); }
}
async function copyJobResult(name, btn) {
  try {
    const text = await apiText('/api/history/content?name=' + encodeURIComponent(name));
    await copyText(text, btn);
  } catch (e) { toast('读取结果失败：' + e.message, 'err'); }
}
async function copyText(text, btn = null) {
  if (!text) { toast('没有可复制的内容', 'warn'); return; }
  const ok = await writeClipboard(text);
  if (ok) {
    toast('已复制到剪贴板');
    flashCopied(btn);
  } else {
    // 不再把失败伪装成成功：粘贴出来是空的比提示失败更糟
    toast('复制失败，请手动选中文本复制', 'err');
  }
}
async function writeClipboard(text) {
  // Clipboard API 仅安全上下文可用（localhost 视为安全；--host 0.0.0.0 用局域网 IP 访问时不可用）
  if (window.isSecureContext && navigator.clipboard) {
    try { await navigator.clipboard.writeText(text); return true; }
    catch { /* 窗口失焦等场景会 reject，回落 execCommand */ }
  }
  try { return legacyCopy(text); } catch { return false; }
}
function legacyCopy(text) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.setAttribute('readonly', '');
  // 离屏 + 只读：避免 focus 造成页面跳动 / 移动端弹键盘，也不破坏用户已有选区
  ta.style.cssText = 'position:fixed;top:0;left:-9999px;width:1px;height:1px;opacity:0;';
  document.body.append(ta);
  const sel = document.getSelection();
  const saved = sel.rangeCount ? sel.getRangeAt(0) : null;
  ta.select();
  ta.setSelectionRange(0, text.length);
  let ok = false;
  try { ok = document.execCommand('copy'); } catch { ok = false; }
  ta.remove();
  if (saved) { sel.removeAllRanges(); sel.addRange(saved); }
  return ok;
}
function flashCopied(btn) {
  const label = btn && btn.querySelector('.btn-label');
  if (!label || label.dataset.flash) return;
  label.dataset.flash = '1';
  const prev = label.textContent;
  label.textContent = '已复制 ✓';
  setTimeout(() => { label.textContent = prev; delete label.dataset.flash; }, 1500);
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
  if (VIDEO_EXT.includes(e)) return 'video';
  return AUDIO_EXT.includes(e) ? 'audio' : 'other';
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
let histSearchTimer = null;
let histSearchSeq = 0;
async function renderHistory() {
  const q = $('#histSearch').value.trim();
  $('#histCount').textContent = state.history.length || '';
  const seq = ++histSearchSeq;
  let items = state.history;
  if (q) {
    // 正文搜索交给服务端（/api/history?q=），避免把所有记录全文拉到前端
    try {
      const found = await api('/api/history?q=' + encodeURIComponent(q));
      if (seq !== histSearchSeq) return;   // 已有更新的搜索输入，丢弃过期响应
      items = found;
    } catch { return; }                    // 服务停止时静默
  }
  if (!items.length) {
    $('#histGrid').replaceChildren(h('div', { class: 'empty' },
      svgIcon([['circle', { cx: 12, cy: 12, r: 8.5 }], ['path', { d: 'M12 7.5V12l3 2' }]], 1.5),
      h('div', { text: q ? '没有匹配的记录（文件名与正文都会搜索）' : '还没有转录记录，去创建第一个吧' })));
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
    const text = await apiText('/api/history/content?name=' + encodeURIComponent(name));
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
function notifyDone(job) {
  if (!document.hidden) return;
  document.title = '✓ 转录完成 — steno';       // 标题角标：切到后台也能注意到
  if ('Notification' in window && Notification.permission === 'granted') {
    try {
      const n = new Notification('转录完成 ✓', { body: job.display, tag: 'steno-job-done' });
      n.addEventListener('click', () => { window.focus(); n.close(); });
    } catch { /* 某些环境构造 Notification 会抛错，忽略 */ }
  }
}
function requestNotifyPermission() {
  if ('Notification' in window && Notification.permission === 'default') {
    try { Notification.requestPermission(); } catch { /* ignore */ }
  }
}
function isDoneEdge(j, prev) {
  // 完成沿判定：本会话提交的任务（首次观测即 done 也算——模型已缓存时短音频
  // 几秒内完成，第一次轮询就可能直接拿到 done），或列表中经历了
  // queued/running → done 转变的任务。
  return j.status === 'done' && !state.notifiedDone.has(j.id) &&
    (state.watching.has(j.id) || (prev && prev !== 'done'));
}
function onJobDone(job) {
  toast('转录完成 ✓');
  loadHistory();
  loadInfo();
  notifyDone(job);
  if (job.result) {
    // showResult 内部会切视图/拉取全文，完成后再滚动到结果面板
    showResult(job.result, job.display)
      .then(() => $('#resultWrap').scrollIntoView({ behavior: 'smooth', block: 'start' }));
  }
}
function pollOnce() {
  const hadWork = state.uploading ||
    state.jobs.some(j => j.status === 'queued' || j.status === 'running');
  api('/api/jobs').then(jobs => {
    state.pollFail = 0;
    if (state.serverDown) { state.serverDown = false; toast('服务已恢复'); }
    const seen = new Set(jobs.map(j => j.id));
    for (const j of jobs) {
      const prev = state.prevStatus[j.id];
      if (isDoneEdge(j, prev)) {
        state.notifiedDone.add(j.id);
        onJobDone(j);
      } else if (j.status === 'error' && !state.notifiedDone.has(j.id) &&
                 (state.watching.has(j.id) || (prev && prev !== 'error'))) {
        state.notifiedDone.add(j.id);
        toast('转录失败：' + (j.error || '未知错误'), 'err');
      }
      state.prevStatus[j.id] = j.status;
    }
    // 清理已从服务端消失的任务（被删除/服务重启），防止快照无限增长
    for (const id of Object.keys(state.prevStatus)) if (!seen.has(id)) delete state.prevStatus[id];
    for (const id of [...state.notifiedDone]) if (!seen.has(id)) state.notifiedDone.delete(id);
    for (const id of [...state.watching]) if (!seen.has(id)) state.watching.delete(id);
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
function pollNow() {
  // 清掉待触发的定时器立即轮询：提交任务后必须马上看一眼，不能等下一周期
  clearTimeout(state.pollTimer);
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
    // 取消上传（abort XHR，onabort 里清理状态并重新渲染）
    if (e.target.closest('[data-cancel-upload]')) {
      const u = state.uploading;
      if (u) { u.xhr.abort(); toast('已取消上传', 'warn'); }
      return;
    }
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
    const copyId = e.target.closest('[data-copy-job]')?.dataset.copyJob;
    if (copyId) {
      const job = state.jobs.find(j => j.id === copyId);
      if (job?.result) copyJobResult(job.result, e.target.closest('[data-copy-job]'));
      return;
    }
    const delId = e.target.closest('[data-del-job]')?.dataset.delJob;
    if (delId) {
      api(`/api/jobs/${delId}`, { method: 'DELETE' })
        .then(() => {
          delete state.prevStatus[delId];
          state.watching.delete(delId);
          state.notifiedDone.delete(delId);
          pollOnce();
        })
        .catch(err => toast(err.message, 'err'));
    }
  });

  // 结果操作
  $('#copyBtn').addEventListener('click', e => copyText(state.result ? state.result.text : '', e.currentTarget));
  $('#expandBtn').addEventListener('click', () => setExpanded(!$('#resultCard').classList.contains('expanded')));
  $('#dlBtn').addEventListener('click', () => {
    if (state.result) downloadText(state.result.name, state.result.text);
  });

  // 历史
  $('#histSearch').addEventListener('input', () => {
    clearTimeout(histSearchTimer);
    histSearchTimer = setTimeout(renderHistory, 250);   // 正文搜索在服务端做，去抖后再请求
  });
  $('#histGrid').addEventListener('click', e => {
    const card = e.target.closest('[data-hist]');
    if (card) openModal(card.dataset.hist);
  });

  // 弹窗
  $('#modalClose').addEventListener('click', closeModal);
  $('#modal').addEventListener('click', e => { if (e.target === $('#modal')) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });
  $('#modalCopy').addEventListener('click', e => copyText($('#modalText').textContent, e.currentTarget));
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

  // 回到页面时清掉完成提示的标题角标
  const baseTitle = document.title;
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) document.title = baseTitle;
  });
}

// ============================ 启动 ============================
bind();
loadInfo();
loadHistory();
pollNow();

})();
