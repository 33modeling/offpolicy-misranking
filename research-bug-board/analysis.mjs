// Pure renderer shared by the dashboard and its Node-based verification.
const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
const values = (v) => v && typeof v === 'object' ? Object.values(v).map(values).join(' ') : String(v ?? '');
const repository = 'https://github.com/33modeling/offpolicy-misranking';
const sha = (s) => typeof s === 'string' && /^[a-f0-9]{40}$/.test(s);
const id = (s) => /^[a-zA-Z0-9-]+$/.test(s) ? s : 'analysis-record';
const commit = (s, label = '') => sha(s) ? `<a href="${repository}/commit/${s}" target="_blank" rel="noopener">${esc(label || s.slice(0, 12))}</a>` : esc(label || s);

export function sourceLink(e) {
  const valid = sha(e.commit) && typeof e.path === 'string'
    && e.path.split('/').every((p) => p && p !== '.' && p !== '..')
    && Number.isSafeInteger(e.line) && e.line > 0;
  const label = esc(e.label || `${e.path}:${e.line}`);
  if (!valid) return `<span>${label}</span>`;
  const path = e.path.split('/').map(encodeURIComponent).join('/');
  return `<a href="${repository}/blob/${e.commit}/${path}#L${e.line}" target="_blank" rel="noopener" title="${esc(e.excerpt)}">${label}</a>`;
}

function comparison(r) {
  return `<details class="pcb-comparison" id="compare-${id(r.id)}"><summary>${esc(r.title)}</summary>
    <dl class="pcb-model-grid">${[['Qwen',r.qwen],['Llama',r.llama],['Gemma IT',r.gemma]].map(([label,text]) => `<div><dt>${label}</dt><dd>${esc(text)}</dd></div>`).join('')}</dl>
    <p class="pcb-impact"><b>비교할 때의 의미</b> ${esc(r.impact)}</p><div class="sk-links pcb-source-links">${r.evidence.map(sourceLink).join('')}</div></details>`;
}

export function renderAnalysis(data, query = '') {
  if (!data || !Array.isArray(data.reports) || !data.reports.length) {
    return '<p class="sk-empty">모델 분석을 읽지 못했습니다. <a href="model-analysis.json">분석 원본 JSON</a>을 확인해 주세요.</p>';
  }
  const q = query.trim().toLowerCase();
  const match = (r) => !q || values(r).toLowerCase().includes(q);
  return data.reports.slice(-1).map((report) => {
    const rows = report.rows.filter(match), changes = report.changes.filter(match);
    const diagnosis = report.diagnosis.filter(match), timeline = report.timeline.filter(match);
    const categories = [...new Set(rows.map((r) => r.category))];
    const v = report.validation;
    return `<article class="pcb-analysis" id="${id(report.id)}"><header><h2>${esc(report.title)}</h2>
      <p>${esc(report.scope)}</p><p class="sk-note">분석 기준 ${commit(report.source_commit)} · 최근 변경 기준 ${commit(report.recent_baseline)} · ${esc(report.recorded_at)}</p></header>
      <div class="sk-callout"><ul>${report.summary.map((s) => `<li>${esc(s)}</li>`).join('')}</ul></div>
      <nav class="sk-links pcb-analysis-nav" aria-label="모델 분석 목차"><a href="#analysis-comparison">모델별 비교 ${rows.length}/${report.rows.length}</a><a href="#analysis-changes">최근 변경 ${changes.length}건</a><a href="#analysis-diagnosis">확인·미확인 구분</a><a href="#analysis-validation">검증 범위</a><a href="model-analysis.json" download>분석 원본 받기</a></nav>
      <section id="analysis-comparison"><h3>모델별 구현 비교</h3><p class="sk-note">항목을 누르면 모델별 설정·영향·소스 근거를 볼 수 있습니다.</p>
      <div class="sk-links"><button type="button" class="sk-chip" data-analysis-expand="true">비교 항목 모두 펼치기</button><button type="button" class="sk-chip" data-analysis-expand="false">모두 접기</button></div>
      ${categories.map((c) => `<h4>${esc(c)}</h4>${rows.filter((r) => r.category === c).map(comparison).join('')}`).join('')}${rows.length ? '' : '<p class="sk-empty">검색에 맞는 비교 항목이 없습니다.</p>'}</section>
      <section id="analysis-changes"><h3>최근 변경의 적용 범위</h3><p>adapter 해시가 같아도 manifest 밖의 운영 코드 변경은 실행·대기·재시도와 비용 표시에 영향을 줍니다.</p>
      ${changes.map((c) => `<div class="pcb-analysis-entry"><h4>${commit(c.commit)} · ${esc(c.effect)}</h4><p class="sk-note">${esc(c.scope)} · ${esc(c.at)} · ${esc(c.title)}</p><dl><dt>Qwen·Llama</dt><dd>${esc(c.qwen_llama)}</dd><dt>Gemma</dt><dd>${esc(c.gemma)}</dd></dl></div>`).join('')}${changes.length ? '' : '<p class="sk-empty">검색에 맞는 변경이 없습니다.</p>'}</section>
      <section id="analysis-diagnosis"><h3>확인된 사실·가설·미해결</h3>${diagnosis.map((d) => `<div class="pcb-analysis-entry"><h4>${esc(d.level)} · ${esc(d.title)}</h4><p>${esc(d.text)}</p></div>`).join('')}</section>
      <section id="analysis-validation"><h3>검증 범위와 원본 해시</h3><p>${esc(v.method)}</p><details class="pcb-comparison"><summary>실제 prepare로 만든 CPU fixture 기본값·해시</summary>
      <dl class="pcb-defaults">${Object.entries(v.common_defaults).map(([k,value]) => `<dt>${esc(k)}</dt><dd>${esc(Array.isArray(value) ? value.join(', ') : value)}</dd>`).join('')}</dl>
      <p>${Object.entries(v.splits).map(([name,s]) => `${esc(name)}: candidate ${s.counts.candidate_ids} / ranking validation ${s.counts.ranking_validation_ids} / evaluation ${s.counts.evaluation_ids} · ID 일치 ${s.ids_equal ? '확인' : '미확인'} · source-data 해시 일치 ${s.source_data_equal ? '확인' : '미확인'}`).join('<br>')}</p>
      ${Object.entries(v.models).map(([name,m]) => `<p><b>${esc(name)}</b> · adapter ${m.adapter_files.length}파일 · 최근 변경 ${m.recent_changed.length}파일</p><dl class="pcb-defaults">${Object.entries(m.digests).map(([ref,digest]) => `<dt>${commit(ref)}</dt><dd><code>${esc(digest)}</code></dd>`).join('')}</dl>`).join('')}
      <p>Frozen engine ${v.engine.files}파일 · 최근 변경 ${v.engine.changed_since_baseline.length}파일. 전체 manifest는 원본 JSON에 포함돼 있습니다.</p></details></section>
      <section><h3>모델 도입 이후 주요 이력</h3><ol class="pcb-events">${timeline.map((t) => `<li><small>${esc(t.at)}</small><p>${commit(t.commit)} · ${esc(t.text)}</p></li>`).join('')}</ol></section></article>`;
  }).join('');
}
