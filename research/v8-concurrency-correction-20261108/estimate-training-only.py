"""Rough training-only timeline; never a measured/full-pipeline node maximum."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--runner-root',type=Path,required=True)
args=p.parse_args()
jobs=[]
for cohort in ('olmo_math','qwen_mbpp'):
    cfg=json.loads((args.runner_root/'configs/cohorts'/f'{cohort}.json').read_text())
    b=cfg['cost_basis']
    # Historical device-hours -> wall hours, assuming all4GPUs are allocated.
    segment=(25*b['on_training_update']+b['on_refresh']+b['srgc_check_extra'])/4
    sr=b['sr_update']/4
    for seed in cfg['seeds']['training']:
        jobs.append((cohort,0,10*segment))
        for i,tau in enumerate(range(25,275,25)):
            release=i*segment
            jobs.append((cohort,release,release+(275-tau)*sr))
events=sorted({t for _,start,end in jobs for t in (start,end)})
rows=[]
for t in events:
    counts={c:sum(c==cohort and start<=t<end for cohort,start,end in jobs)
            for c in ('olmo_math','qwen_mbpp')}
    rows.append(dict(time=t,counts=counts,total=sum(counts.values())))
peaks={c:max(r['counts'][c] for r in rows) for c in ('olmo_math','qwen_mbpp')}
combined=max(r['total'] for r in rows)
assert peaks=={'olmo_math':40,'qwen_mbpp':30} and combined==55
print(json.dumps(dict(scope='rough On/SR training only; not total pipeline or actual measurement',
                     peaks=peaks,combined=combined),indent=2))
