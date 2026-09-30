"""Verify immutable inputs, declared coverage, code provenance and artifacts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from plume_focus.full_fov_execution import FullFovRunConfig


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/focus-extent'))
    args=parser.parse_args();root=args.root;decl=read(root/'declaration.json')
    files=['scripts/diagnose-focus-extent.py','src/plume_focus/focus_extent_diagnostics.py']
    code=b''.join(Path(p).read_bytes() for p in files)
    committed=b''.join(subprocess.check_output(['git','show','HEAD:'+p]) for p in files)
    assert code==committed,'Diagnostic execution source differs from committed source'
    code_hash=hashlib.sha256(code).hexdigest();count=0;errors=[]
    for job in decl['jobs']:
        assert sha(job['input'])==job['input_sha256']
        record=read(root/'runs'/f"{job['name']}.json")
        assert record['job']==job and record['code_sha256']==code_hash
        assert record['declaration_sha256']==sha(root/'declaration.json')
        assert record['public_replay_max_error']<=1e-12
        assert len(record['snapshots'])==len(record['artifacts'])==len(job['events'])
        config=FullFovRunConfig.from_dict(read(job['input'])['configuration'])
        velocity=np.array(config.source.wind_velocity)*config.source.fps/(np.array(config.source.shape)-1)
        errors.append(float(np.max(abs(velocity-config.wind.velocity_yx))))
        for snapshot,event,artifact in zip(record['snapshots'],job['events'],record['artifacts']):
            assert snapshot['event_index']==event['event_index'] and snapshot['read_only_passed']
            assert sha(artifact['path'])==artifact['sha256']
            for h in decl['horizons']:
                assert {r['variant'] for r in snapshot['forecast_records'] if r['horizon_seconds']==h}==set(decl['future_variants'])|{'history_source_off'}
            count+=1
    controls=read(root/'controls.json')
    assert len(controls['readings'])==16 and len(controls['packet_references'])==20
    assert max(r['analytic_max_error'] for r in controls['readings'])<1e-12
    # Old media, configurations, traces and runtime sources stay byte-identical.
    old=read('outputs/lean-full-fov/artifact-manifest.json')
    for item in old['artifacts']+old['source_files']:
        assert sha(item['path'])==item['sha256'],item['path']
    manifest=read(root/'artifact-manifest.json')
    for item in manifest['artifacts']:assert sha(item['path'])==item['sha256']
    assert len(read(root/'summary.json')['snapshots'])==count==42
    result={'artifact_type':'focus_extent_verification_v1','status':'passed',
        'revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'execution_code_sha256':code_hash,'snapshots':count,'runs':len(decl['jobs']),
        'wind_conversion_max_error':max(errors),'historical_artifacts_verified':len(old['artifacts']),
        'artifact_manifest_sha256':sha(root/'artifact-manifest.json'),
        'note':'Early zero-noise probes recorded pre-commit HEAD; exact code bytes match committed execution code.'}
    (root/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
