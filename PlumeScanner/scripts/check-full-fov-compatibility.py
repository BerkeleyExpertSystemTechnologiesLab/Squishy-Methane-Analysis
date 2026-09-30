"""Compare historical public-call payloads against the inspected Git revision."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


CHECK = r'''
from dataclasses import replace
from pathlib import Path
import json,hashlib,sys
import numpy as np
from plume_focus.controller import SensingController
from plume_focus.sensing_cli_v1 import load_config
from plume_focus.sensing_session import InstrumentProfileV1
from plume_focus.sensing_contracts import CausalWind,ScanConstraints
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_execution_v3 import with_persistent_motion
from plume_focus.sensing_execution_v4 import with_risk_policy
from plume_focus.sensing_execution_v5 import with_threshold_policy
from plume_focus.sensing_simulation import FakeScanner
root=Path(sys.argv[1])
profile=InstrumentProfileV1.from_dict(json.loads((root/'docs/specs/instrument-profile-v1.example.json').read_text()))
def small(path):
    c=load_config(root/path)
    return replace(c,session=replace(c.session,instrument=profile,source_position=None,physical_quadrature_weights=(1/16,)*16,
        model=replace(c.model,grid_shape=(2,2),diffusivity=0.,process_std_per_sqrt_second=0.,source_rate=0.)))
v1=small('configs/sensing-v1/coarse.json');v2=small('configs/sensing-v2/coarse-full-startup.json')
motion=with_persistent_motion(v2,((0.,0.),(0.,0.)))
risk=with_risk_policy(v2,risk_spatial_shape=(8,8));riskmotion=with_risk_policy(motion,risk_spatial_shape=(8,8))
cases={'v1':v1,'v2':v2,'v3_zero_motion':motion,'v4':risk,'v4_zero_motion':riskmotion,
       'v5':with_threshold_policy(risk),'v5_zero_motion':with_threshold_policy(riskmotion)}
class Source:
    end_time_seconds=None
    def signal_at(self,t,p):return np.full(np.shape(t),.4)
out={}
for name,c in [('legacy',None),*cases.items()]:
    ctl=SensingController(profile.capabilities(),v1.model,source_position=None) if c is None else SensingController.from_session(c.session)
    now=0.;pose=(0.,0.);rows=[]
    for _ in range(2):
        p=ctl.plan_next(now,pose,ScanConstraints(max_duration_seconds=8.) if c is None else ScanRequestV1(8.),wind=CausalWind())
        rows.append(p.to_dict())
        if not p.samples:break
        ex=FakeScanner(Source(),noise_std=0).execute(p,delivery='batch')
        ctl.ingest_readings(ex.readings,available_at_seconds=ex.receipt.received_at_seconds)
        ctl.finish_scan(ex.receipt);rows.append({'mean':ctl._belief.mean.tolist(),'covariance':ctl._belief.covariance.tolist()})
        now,pose=ex.receipt.received_at_seconds,ex.receipt.final_position_yx
    out[name]=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':')).encode()).hexdigest()
print(json.dumps(out))
'''


def check(root,revision):
    repository=Path.cwd();baseline=root/'baseline';baseline.mkdir(parents=True,exist_ok=True)
    data=subprocess.check_output(['git','archive',revision,'src','configs','docs/specs/instrument-profile-v1.example.json'])
    with tarfile.open(fileobj=io.BytesIO(data)) as archive:
        archive.extractall(baseline,filter='data')
    def hashes(directory):
        env={**os.environ,'PYTHONPATH':str(directory/'src'),'OPENBLAS_NUM_THREADS':'1'}
        return json.loads(subprocess.check_output([sys.executable,'-c',CHECK,str(directory)],cwd=directory,env=env,text=True))
    before,after=hashes(baseline.resolve()),hashes(repository)
    assert before==after,(before,after)
    result={'status':'passed','baseline_revision':revision,'baseline_archive_sha256':hashlib.sha256(data).hexdigest(),
            'current_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'scope':'Two public acquisition cycles on deterministic arithmetic fixtures; exact historical plan and posterior hashes.',
            'hashes':after}
    (root/'compatibility.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path('outputs/lean-full-fov'))
    parser.add_argument('--revision',default='84cf70d');args=parser.parse_args();check(args.output,args.revision)
