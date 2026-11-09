import os,sys,time
from pathlib import Path
sys.path.insert(0,'/home/kms/dev/offpolicy-misranking/v8/experiments')
def main():
 from switching.runner import write_exit_marker
 mode=os.environ['FAULT_MODE'];rank=int(os.environ['RANK'])
 if mode=='busy' or (mode in ('mixed','mixed_signal') and rank==0):
  if rank==0:write_exit_marker(75,'busy',task='fixture-task')
  if mode in ('mixed','mixed_signal'):
   while not (Path(os.environ['SRGC_PROGRESS_DIR'])/'rank-1-error.json').exists():time.sleep(.01)
  raise SystemExit(75)
 if rank==1:
  if mode=='segfault':
   import ctypes,resource,json
   resource.setrlimit(resource.RLIMIT_CORE,(0,0))
   (Path(os.environ['SRGC_PROGRESS_DIR'])/'rank-1.json').write_text(json.dumps(dict(pid=os.getpid(),stage='gradient_scoring',scope='CPU synthetic crash fixture')))
   ctypes.string_at(0)
  if mode in ('mixed','mixed_signal'):
   while not Path(os.environ['V8_EXIT_FILE']).exists():time.sleep(.01)
  if mode=='mixed_signal':
   import signal
   os.kill(os.getpid(),signal.SIGKILL)
  if mode=='systemexit':raise SystemExit('PINNED INPUT CHECK FAILED ON RANK 1')
  raise ValueError('ORIGINAL RANK 1 FAILURE; DO NOT REPLACE WITH ChildFailedError')
 time.sleep(15)
