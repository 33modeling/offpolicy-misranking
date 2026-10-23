const $=s=>document.querySelector(s);
const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const names={open:'미해결·미확인',in_progress:'수정 중',code_fixed:'코드 수정',verified:'범위 검증 완료'};
const types={code_test:'코드 검사',actual_model_probe:'실제 모델 진단 · 범위 한정',user_report:'사용자 보고',commit_record:'커밋 근거'};
let data,commits,view='incidents',limit=30;
const state=row=>row.events.at(-1)?.status??'open';
const safeLink=url=>typeof url==='string'&&url.startsWith('https://github.com/33modeling/')?url:'#';
const fmtDate=at=>at?at.slice(0,10):'시각 미확인';
function incident(row){
 const current=state(row),events=row.events.map(event=>`<li><small>${esc(event.at)} · ${esc(event.actor)} · ${esc(names[event.status]??event.type)}</small><p>${esc(event.text)}</p><span class="type">${esc(types[event.evidence_kind]??event.evidence_kind??'기록')}</span>${event.url?`<div class="evidence"><a href="${esc(safeLink(event.url))}" target="_blank" rel="noopener">이 변경의 원본 근거 ↗</a></div>`:''}</li>`).join('');
 const evidence=row.evidence.map(item=>`<a href="${esc(safeLink(item.url))}" target="_blank" rel="noopener">${esc(item.kind==='commit'?'수정·관련 커밋':'원본 근거')} · ${esc(item.label)} ↗</a>`).join('');
 return `<details class="incident" id="${esc(row.id)}"><summary><div class="record-top"><span class="record-id">${esc(row.id)}</span><span class="badge ${esc(current)}">${esc(names[current]??current)}</span><span>${esc(row.area)}</span></div><h3>${esc(row.title)}</h3><p class="record-brief">${esc(row.impact)}</p></summary><div class="record-body"><dl><dt>원인·확인 범위</dt><dd>${esc(row.cause)}</dd><dt>영향</dt><dd>${esc(row.impact)}</dd><dt>도입·책임 근거</dt><dd>${esc(row.attribution)}${row.introduction_commit?`<br><a href="https://github.com/33modeling/offpolicy-misranking/commit/${esc(row.introduction_commit)}" target="_blank" rel="noopener">도입 커밋 ${esc(row.introduction_commit.slice(0,12))} ↗</a>`:''}</dd></dl><div class="evidence">${evidence}</div><ol class="events">${events}</ol></div></details>`;
}
function commitRow(row){return `<article class="commit"><div class="commit-top"><a class="record-id" href="${esc(safeLink(row.url))}" target="_blank" rel="noopener">${esc(row.hash.slice(0,12))} ↗</a><time>${esc(fmtDate(row.at))}</time><span>${esc(row.author)}</span>${row.is_fix?'<span class="badge">수정·오류 이력</span>':''}${row.gemma?'<span class="badge">Gemma</span>':''}</div><h3>${esc(row.title)}</h3><p class="attribution">${row.attribution.length?esc(row.attribution.join(' / ')):'작성 도구: 명시된 커밋 근거 없음'}</p></article>`;}
function render(){
 const q=$('#q').value.trim().toLowerCase(),area=$('#area').value,status=$('#state').value;
 let rows=view==='incidents'?data.incidents:commits.filter(row=>view!=='fixes'||row.is_fix);
 rows=rows.filter(row=>(!q||JSON.stringify(row).toLowerCase().includes(q))&&(view!=='incidents'||(!area||row.area===area)&&(!status||state(row)===status)));
 if($('#sort').value==='newest')rows=[...rows].reverse();
 $('#records').innerHTML=rows.length?rows.slice(0,limit).map(view==='incidents'?incident:commitRow).join(''):'<p class="empty">조건에 맞는 기록이 없습니다. 검색 조건을 바꿔 보세요.</p>';
 $('#result-count').textContent=`${rows.length.toLocaleString('ko-KR')}건 · ${Math.min(limit,rows.length)}건 표시`;
 $('#view-title').textContent={incidents:'상세 사건 · 보고와 수정 이력',fixes:'과거 수정커밋 · 원문 근거',commits:'처음부터 전체 커밋 이력'}[view];
 $('#more').hidden=rows.length<=limit;
 $('#area').disabled=$('#state').disabled=view!=='incidents';
 document.querySelectorAll('[data-view]').forEach(button=>button.setAttribute('aria-selected',String(button.dataset.view===view)));
 openHash();
}
function openHash(){let id;try{id=decodeURIComponent(location.hash.slice(1));}catch{return;}if(!id.startsWith('GEMMA-'))return;const item=document.getElementById(id);if(item){item.open=true;item.scrollIntoView({block:'start',behavior:'auto'});}}
function initialize(){
 const open=data.incidents.filter(row=>['open','in_progress'].includes(state(row))).length;
 $('#stats').innerHTML=[[data.incidents.length,'상세 사건'],[open,'미해결·미확인 / 수정 중'],[data.coverage.fix_commit_records,'과거 수정·오류 커밋'],[commits.length,'전체 보존 커밋']].map(([n,label])=>`<div class="stat"><span>${esc(label)}</span><strong>${n.toLocaleString('ko-KR')}</strong></div>`).join('');
 $('#coverage').textContent=`${fmtDate(data.coverage.first_commit_at)}부터 감사 기준 ${data.coverage.source_commit.slice(0,12)}까지 · 최초 Gemma ${data.coverage.first_gemma_commit.slice(0,12)} (${fmtDate(data.coverage.first_gemma_at)})`;
 $('#scores').innerHTML=data.reported_scores.map(row=>`<span class="score" title="${esc(row.source)}">${esc(row.label)} · 보고값</span>`).join('');
 $('#score-note').textContent=data.measurement_note;
 $('#incident-count').textContent=data.incidents.length;$('#fix-count').textContent=data.coverage.fix_commit_records;$('#commit-count').textContent=commits.length.toLocaleString('ko-KR');
 $('#area').insertAdjacentHTML('beforeend',[...new Set(data.incidents.map(row=>row.area))].map(area=>`<option>${esc(area)}</option>`).join(''));
 $('#recorded-at').textContent=`기록 기준 ${fmtDate(data.recorded_at)} · 수정한 항목과 과거 기록도 보존`;
 document.querySelectorAll('[data-view]').forEach(button=>button.addEventListener('click',()=>{view=button.dataset.view;limit=30;render();}));
 for(const id of ['q','area','state','sort'])$('#'+id).addEventListener(id==='q'?'input':'change',()=>{limit=30;render();});
 $('#clear').addEventListener('click',()=>{$('#q').value=$('#area').value=$('#state').value='';$('#sort').value='oldest';limit=30;render();});
 $('#more').addEventListener('click',()=>{limit+=40;render();});
 $('#download').addEventListener('click',()=>{const blob=new Blob([JSON.stringify({data,commits},null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),link=document.createElement('a');link.href=url;link.download='paper-code-bug-records.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),30000);});
 window.addEventListener('hashchange',openHash);render();
}
try{[data,{commits}]=await Promise.all([fetch('data.json').then(r=>{if(!r.ok)throw Error('사건 기록 '+r.status);return r.json();}),fetch('commits.json').then(r=>{if(!r.ok)throw Error('커밋 기록 '+r.status);return r.json();})]);initialize();}
catch(error){$('#records').textContent='기록을 읽지 못했습니다: '+error.message;$('#coverage').textContent='원본 JSON 링크에서도 기록을 확인할 수 있습니다.';}
