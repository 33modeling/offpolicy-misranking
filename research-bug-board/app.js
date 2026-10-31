import {renderAnalysis} from './analysis.mjs';
import {$, badge, bindRows, chip, esc, fresh, group, kv, press, row, stamp, tile, topBar} from '../assets/status-kit/status.mjs';

const NAMES = {open: '미해결·미확인', in_progress: '수정 중', code_fixed: '코드 수정', verified: '범위 검증 완료'};
const TONES = {open: 'bad', in_progress: 'warn', code_fixed: 'run', verified: 'ok'};
const TYPES = {code_test: '코드 검사', actual_model_probe: '실제 모델 진단 · 범위 한정', user_report: '사용자 보고', commit_record: '커밋 근거'};
const status = (r) => r.events.at(-1)?.status ?? 'open';
const safe = (url) => typeof url === 'string' && url.startsWith('https://github.com/33modeling/') ? url : '#';
const day = (at) => at ? at.slice(0, 10) : '시각 미확인';
let data, commits, analysis, view = 'incidents', filter = 'all', area = 'all', limit = 40;
// 지난번 본 뒤 새 기록: 사건마다 이력 수를 이 브라우저에 남겨 두고 견준다. 이력은 덧붙이기만 하므로 늘어난 수가 새 이력이다.
const SEEN_KEY = 'paper-code-bugs:seen:v1';
let changes = new Map(), since = '';
const lastAt = (r) => r.events.at(-1)?.at || '';
// 찾기는 값만 본다. 키 이름까지 보면 status·gemma·fix 같은 말이 모든 사건·커밋에 걸린다.
const values = (v) => v && typeof v === 'object' ? Object.values(v).map(values).join(' ') : typeof v === 'string' || typeof v === 'number' ? String(v) : '';

function evidenceItem(e) {
  if (e.kind === 'user_report' && typeof e.text === 'string') {
    return `<span>사용자 보고 · ${esc(e.text).replaceAll('\n', '<br>')}</span>`;
  }
  let url = e.url, label = e.label;
  if (e.kind === 'source' && /^[a-f0-9]{7,40}$/.test(e.commit)
      && typeof e.path === 'string' && e.path.split('/').every((p) => p && p !== '.' && p !== '..')) {
    url ||= `https://github.com/33modeling/offpolicy-misranking/blob/${e.commit}/${e.path.split('/').map(encodeURIComponent).join('/')}`;
    label ||= `${e.path} (${e.commit.slice(0, 8)})`;
  }
  const text = `${e.kind === 'commit' ? '수정·관련 커밋' : '원본 근거'} · ${esc(label || e.path || '근거 정보 미확인')}`;
  const href = safe(url);
  return href === '#' ? `<span>${text}</span>` : `<a href="${esc(href)}" target="_blank" rel="noopener">${text} ↗</a>`;
}

function incident(r) {
  const s = status(r);
  const facts = [['원인 · 확인 범위', r.cause], ['영향', r.impact], ['도입 · 책임 근거', r.attribution]];
  const evidence = r.evidence.map(evidenceItem);
  if (r.introduction_commit) evidence.push(`<a href="https://github.com/33modeling/offpolicy-misranking/commit/${esc(r.introduction_commit)}" target="_blank" rel="noopener">도입 커밋 ${esc(r.introduction_commit.slice(0, 12))} ↗</a>`);
  const events = r.events.map((e) => `<li data-tone="${TONES[e.status] || 'wait'}"><small>${esc(e.at)} · ${esc(e.actor)} · ${esc(NAMES[e.status] ?? e.type)} · ${esc(TYPES[e.evidence_kind] ?? e.evidence_kind ?? '기록')}</small><p>${esc(e.text)}</p>${e.url ? `<a href="${esc(safe(e.url))}" target="_blank" rel="noopener">근거 ↗</a>` : ''}</li>`).join('');
  const body = `${kv(facts)}<h3>근거</h3><div class="sk-links" style="margin-top:0">${evidence.join('')}</div><h3>이력</h3><ol class="pcb-events">${events}</ol>`;
  const news = changes.get(r.id);
  const mark = news ? `<span class="pcb-new">${news.added ? '새 사건' : `새 이력 ${news.count}`}</span> · ` : '';
  return row({key: r.id, tone: TONES[s] || 'wait', label: NAMES[s] ?? s, title: r.title, line: r.impact, meta: `${mark}${esc(r.id)} · ${esc(r.area)}`, body});
}

function commitRow(c) {
  const tags = [c.is_fix ? '수정·오류 이력' : '', c.gemma ? 'Gemma' : ''].filter(Boolean);
  const body = `${kv([['커밋', c.hash], ['작성자', c.author], ['작성 도구 근거', c.attribution.length ? c.attribution.join(' / ') : '명시된 커밋 근거 없음'], ['시각', c.at]])}<div class="sk-links"><a href="${esc(safe(c.url))}" target="_blank" rel="noopener">GitHub에서 보기 ↗</a></div>`;
  return row({key: c.hash, tone: c.is_fix ? 'warn' : 'wait', label: tags.length ? tags.join(' · ') : '커밋', title: c.title, meta: `${esc(c.hash.slice(0, 8))} · ${esc(day(c.at))}`, body});
}

function render() {
  const q = $('#q').value.trim().toLowerCase();
  press('#views [data-view]', 'view', view === 'analysis' ? 'analysis' : 'incidents');
  $('#sort').hidden = view === 'analysis';
  $('#scores').hidden = view === 'analysis';
  $('#q').placeholder = view === 'analysis' ? '모델·설정·변경·미해결 검색' : '1%, JSON, status, 커밋 검색';
  const incidents = data.incidents;
  const unresolved = incidents.filter((r) => ['open', 'in_progress'].includes(status(r)));
  $('#tiles').innerHTML = [
    tile({value: incidents.length, label: '상세 사건', note: '전체', filter: 'incidents:all'}),
    tile({value: unresolved.length, label: '미해결 · 수정 중', tone: 'bad', note: '확인 필요', filter: 'incidents:unresolved'}),
    tile({value: incidents.length - unresolved.length, label: '고침 · 검증', tone: 'ok', note: '코드 수정 포함', filter: 'incidents:resolved'}),
    tile({value: data.coverage.fix_commit_records, label: '과거 수정 커밋', tone: 'warn', filter: 'fixes:all'}),
    tile({value: commits.length.toLocaleString('ko-KR'), label: '전체 커밋', filter: 'commits:all'}),
  ].join('');
  press('#tiles [data-filter]', 'filter', `${view}:${filter}`);
  const areas = [...new Set(incidents.map((r) => r.area))];
  $('#areas').innerHTML = view === 'incidents' ? [chip('all', '모든 분야', null, 'area'), ...areas.map((a) => chip(a, a, incidents.filter((r) => r.area === a).length, 'area'))].join('') : '';
  press('#areas [data-area]', 'area', area);
  if (view === 'analysis') {
    $('#list').innerHTML = renderAnalysis(analysis, q);
    $('#more').hidden = true;
    writeUrl();
    return;
  }
  let rows = view === 'incidents' ? incidents.filter((r) => (area === 'all' || r.area === area)
    && (filter === 'all' || (filter === 'unresolved') === ['open', 'in_progress'].includes(status(r)))) : commits.filter((c) => view !== 'fixes' || c.is_fix);
  rows = rows.filter((r) => !q || values(r).toLowerCase().includes(q));
  // 최근 이력 순: 사건은 마지막 이력 날짜가 늦은 것부터, 커밋은 최신 순과 같다.
  const sort = $('#sort').value;
  if (sort === 'newest' || (sort === 'updated' && view !== 'incidents')) rows = [...rows].reverse();
  else if (sort === 'updated') rows = [...rows].sort((a, b) => lastAt(b).localeCompare(lastAt(a)) || b.id.localeCompare(a.id));
  const title = {incidents: '상세 사건 · 보고와 수정 이력', fixes: '과거 수정 커밋', commits: '처음부터 전체 커밋'}[view];
  $('#list').innerHTML = group(title, rows.slice(0, limit).map(view === 'incidents' ? incident : commitRow), {empty: '조건에 맞는 기록이 없습니다.'});
  $('#more').hidden = rows.length <= limit;
  $('#more').textContent = `기록 더 보기 (${Math.min(limit, rows.length)} / ${rows.length})`;
  restore();
  openHash(false);
  writeUrl();
}

// 주소 조건: ?view=fixes, ?show=unresolved&area=Status, ?q=json&sort=updated 처럼 거른 화면을 주소로 나눈다. 사건 하나는 #GEMMA-001.
function readUrl() {
  const p = new URLSearchParams(location.search);
  view = ['fixes', 'commits', 'analysis'].includes(p.get('view')) ? p.get('view') : 'incidents';
  filter = view === 'incidents' && ['unresolved', 'resolved'].includes(p.get('show')) ? p.get('show') : 'all';
  area = view === 'incidents' && data.incidents.some((r) => r.area === p.get('area')) ? p.get('area') : 'all';
  $('#q').value = (p.get('q') || '').slice(0, 80);
  $('#sort').value = ['newest', 'updated'].includes(p.get('sort')) ? p.get('sort') : 'oldest';
}
function writeUrl() {
  const p = new URLSearchParams();
  if (view !== 'incidents') p.set('view', view);
  if (view === 'incidents' && filter !== 'all') p.set('show', filter);
  if (view === 'incidents' && area !== 'all') p.set('area', area);
  if ($('#q').value.trim()) p.set('q', $('#q').value.trim());
  if ($('#sort').value !== 'oldest') p.set('sort', $('#sort').value);
  const text = p.toString();
  try { history.replaceState(null, '', location.pathname + (text ? `?${text}` : '') + location.hash); } catch { /* 파일로 연 경우 */ }
}

const seen = {
  get() { try { return JSON.parse(localStorage.getItem(SEEN_KEY) || 'null'); } catch { return null; } },
  set() { try { localStorage.setItem(SEEN_KEY, JSON.stringify({version: 1, at: data.recorded_at, events: Object.fromEntries(data.incidents.map((r) => [r.id, r.events.length]))})); } catch { /* 저장 못 해도 화면은 된다 */ } },
};
// 저장한 값이 없거나 깨졌으면 false(첫 방문). 새 사건과 이력이 늘어난 사건을 changes 에 담는다.
function findChanges(prev) {
  changes = new Map(); since = '';
  if (!prev || !prev.events || typeof prev.events !== 'object' || Array.isArray(prev.events)) return false;
  since = String(prev.at || '');
  for (const r of data.incidents) {
    const before = prev.events[r.id];
    if (!Number.isInteger(before)) changes.set(r.id, {added: true, count: r.events.length});
    else if (r.events.length > before) changes.set(r.id, {added: false, count: r.events.length - before});
  }
  return true;
}
function renderChanges() {
  const box = $('#changes');
  if (!changes.size) { box.hidden = true; box.innerHTML = ''; return; }
  const items = data.incidents.filter((r) => changes.has(r.id)).map((r) => {
    const c = changes.get(r.id), s = status(r);
    return `<li><a href="#${esc(r.id)}">${esc(r.id)}</a><span class="pcb-change-title">${esc(r.title)}</span><small>${c.added ? `새 사건 · 이력 ${c.count}건` : `새 이력 ${c.count}건`} · ${esc(NAMES[s] ?? s)}</small></li>`;
  });
  box.hidden = false;
  box.innerHTML = `<div class="pcb-changes-head"><h2>지난번 본 뒤 새 기록 <em>${changes.size}</em></h2>${since ? `<span>${esc(stamp(since))} 기록과 비교</span>` : ''}<button type="button" class="sk-chip" id="changes-seen">본 것으로 표시</button></div><ul class="pcb-changes-list">${items.join('')}</ul>`;
}

// 주소의 #사건번호를 펼친다. reset 이면(주소를 바꿔 들어온 경우) 조건 때문에 목록에 없는 사건은 조건을 풀고 다시 그린다.
function openHash(reset) {
  let id;
  try { id = decodeURIComponent(location.hash.slice(1)); } catch { return; }
  if (!id) return;
  const item = document.querySelector(`.sk-row[data-key="${CSS.escape(id)}"]`);
  if (item) { if (!item.classList.contains('open')) { item.querySelector('.sk-row-head').click(); item.scrollIntoView({block: 'start'}); } return; }
  if (reset && data.incidents.some((r) => r.id === id) && !(view === 'incidents' && filter === 'all' && area === 'all' && !$('#q').value)) {
    view = 'incidents'; filter = 'all'; area = 'all'; $('#q').value = ''; limit = Math.max(limit, data.incidents.length); render();
  }
}

let restore = () => {};
$('#top').outerHTML = topBar('paper-code-bugs');
restore = bindRows($('#list'));
try {
  [data, {commits}, analysis] = await Promise.all([
    fetch('data.json').then((r) => { if (!r.ok) throw Error('사건 기록 ' + r.status); return r.json(); }),
    fetch('commits.json').then((r) => { if (!r.ok) throw Error('커밋 기록 ' + r.status); return r.json(); }),
    fetch('model-analysis.json').then((r) => { if (!r.ok) throw Error('모델 분석 ' + r.status); return r.json(); }).catch(() => null),
  ]);
  $('#fresh').outerHTML = fresh(data.recorded_at, {label: '기록', warnHours: 72, deadHours: 24 * 14});
  $('#scores').innerHTML = `<div class="sk-callout warn"><b>Gemma 초기 보고 수치</b><ul class="pcb-scores">${data.reported_scores.map((s) => `<li>${badge('warn', s.label, 'plain')}<span>${esc(s.source)} · ${s.verified ? '검증됨' : '미검증'}</span></li>`).join('')}</ul><p class="sk-note" style="margin-top:8px">${esc(data.measurement_note)}</p></div>`;
  $('#coverage').textContent = `커밋 범위: ${day(data.coverage.first_commit_at)}부터 감사 기준 ${data.coverage.source_commit.slice(0, 12)}까지, 최초 Gemma 커밋 ${data.coverage.first_gemma_commit.slice(0, 12)} (${day(data.coverage.first_gemma_at)}). 항목·수정·실패·기존 이력은 삭제하지 않습니다. 정정은 새 이력으로 추가합니다.`;
  $('#recorded-at').textContent = `기록 ${day(data.recorded_at)} · 고친 항목과 과거 기록 모두 보존`;
  readUrl();
  if (!findChanges(seen.get())) seen.set();
  renderChanges();
  $('#changes').addEventListener('click', (e) => { if (!e.target.closest('#changes-seen')) return; seen.set(); changes = new Map(); renderChanges(); render(); });
  $('#tiles').addEventListener('click', (e) => { const b = e.target.closest('[data-filter]'); if (!b) return; [view, filter] = b.dataset.filter.split(':'); area = 'all'; limit = 40; render(); });
  $('#views').addEventListener('click', (e) => { const b = e.target.closest('[data-view]'); if (!b) return; view = b.dataset.view; filter = 'all'; area = 'all'; limit = 40; $('#q').value = ''; try { history.replaceState(null, '', location.pathname + location.search); } catch {} render(); });
  $('#list').addEventListener('click', (e) => { const b = e.target.closest('[data-analysis-expand]'); if (b) $('#list').querySelectorAll('.pcb-comparison').forEach((d) => { d.open = b.dataset.analysisExpand === 'true'; }); });
  $('#areas').addEventListener('click', (e) => { const b = e.target.closest('[data-area]'); if (b) { area = b.dataset.area; limit = 40; render(); } });
  $('#q').addEventListener('input', () => { limit = 40; render(); });
  $('#sort').addEventListener('change', render);
  $('#more').addEventListener('click', () => { limit += 60; render(); });
  $('#download').addEventListener('click', () => { const url = URL.createObjectURL(new Blob([JSON.stringify({data, commits, analysis}, null, 2)], {type: 'application/json'})); const a = document.createElement('a'); a.href = url; a.download = 'paper-code-bug-records.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 30000); });
  window.addEventListener('hashchange', () => openHash(true));
  render();
  openHash(true);
} catch (error) {
  $('#list').innerHTML = `<div class="sk-rows"><p class="sk-empty">기록을 읽지 못했습니다: ${esc(error.message)}. 원본 JSON은 아래 링크에 있습니다.</p></div>`;
}
