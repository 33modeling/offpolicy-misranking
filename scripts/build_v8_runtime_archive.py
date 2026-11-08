#!/usr/bin/env python3
"""Package historical queue identities from local Git objects; never access the network."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


REPO=Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git',*args],cwd=REPO)


def build(output,through='HEAD'):
    revisions=git('rev-list','--reverse',through,'--','scripts/srgc_v8_nodes.py','v8/experiments').decode().splitlines()
    payload={};coordinators={};runners={}
    for revision in revisions:
        names=git('ls-tree','-r','--name-only',revision).decode().splitlines()
        script='scripts/srgc_v8_nodes.py'
        if script in names:
            source=git('show',revision+':'+script);sha=hashlib.sha256(source).hexdigest()
            if sha not in coordinators:
                path=f'coordinators/{sha}/srgc_v8_nodes.py';payload[path]=source
                coordinators[sha]=dict(sha256=sha,file=path,source_commit=revision,
                                      supports_all=b'def worker_loop_many(' in source)
        files=[n for n in names if n.startswith('v8/experiments/')]
        if 'v8/experiments/cluster_child.py' not in files:continue
        selected=files+[n for n in names if n.startswith('v7/evidence/')]
        selected += [p for p in ('scripts/srgc_saved_runtime.py',) if p in names]
        data={p:git('show',revision+':'+p) for p in selected}
        bound={p[len('v8/experiments/'):]:hashlib.sha256(data[p]).hexdigest() for p in files
               if (p.startswith('v8/experiments/switching/') and p.endswith('.py') and p.count('/')==3)
               or (p.startswith('v8/experiments/configs/') and p.endswith('.json'))
               or p in ('v8/experiments/cluster_child.py','v8/experiments/run_h100.sh')}
        key=hashlib.sha256(json.dumps(bound,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if key in runners:continue
        prefix=f'runners/{key}'
        for path,content in data.items():payload[prefix+'/'+path]=content
        runners[key]=dict(root=prefix+'/v8/experiments',runner_sha256=bound,source_commit=revision)
    catalog=dict(schema=1,through=git('rev-parse',through).decode().strip(),
                 coordinators=list(coordinators.values()),runners=list(runners.values()),
                 files={p:hashlib.sha256(v).hexdigest() for p,v in sorted(payload.items())})
    payload['catalog.json']=(json.dumps(catalog,sort_keys=True,indent=2)+'\n').encode()
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for name,content in sorted(payload.items()):
            entry=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));entry.compress_type=zipfile.ZIP_DEFLATED
            entry.external_attr=0o100644<<16;archive.writestr(entry,content)
    return dict(coordinators=len(coordinators),runners=len(runners),files=len(payload),bytes=Path(output).stat().st_size)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=REPO/'v8/queue-runtimes.zip')
    parser.add_argument('--through',default='HEAD')
    args=parser.parse_args();print(json.dumps(build(args.output,args.through)))
