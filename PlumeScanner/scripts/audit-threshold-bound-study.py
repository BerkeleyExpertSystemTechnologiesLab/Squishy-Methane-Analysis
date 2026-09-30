"""Verify lineage, physical paths, causal evidence and representative public replay."""

import argparse
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

from plume_focus.controller import SensingController
from plume_focus.sensing_cli_v1 import config_from_dict,write_json
from plume_focus.sensing_snapshot import DecisionSnapshot
from plume_focus.sensing_threshold_evaluation import decode_plan,evaluate_execution
from plume_focus.sensing_counterfactual import TruthQuadrature
from plume_focus.physical_scan import compile_profile_scan


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


@lru_cache(maxsize=32)
def snapshot_read(path):return DecisionSnapshot.from_dict(read(Path(path)))


def audit(root,final=False,replay=False):
    declaration=read(root/'declaration.json');digest=sha(root/'declaration.json');revisions={};records=0;replayed=[];missing=[]
    assert sha(Path(declaration['historical_declaration']))==declaration['historical_declaration_sha256']
    for item in declaration['fixtures'].values():
        for key in ('configuration','snapshot','historical_trace'):
            assert sha(Path(item[key]))==item[key+'_sha256']
    representative=set()
    for job in declaration['jobs']:
        path=root/'runs'/f"{job['id']}.json"
        if not path.exists():missing.append(job['id']);continue
        for key in ('configuration','snapshot'):assert sha(root/job[key])==job[key+'_sha256']
        config=config_from_dict(read(root/job['configuration']));snapshot=snapshot_read(str(root/job['snapshot']))
        payload=snapshot.to_dict();record=read(path)
        assert record['job']==job and record['declaration_sha256']==digest and record['snapshot_hash']==snapshot.digest
        assert record['start']==payload['last_available'] and record['horizon']==job['horizon']
        assert config.session.transport_id=='zero_inflow_v1'
        if job['mode']=='short':
            parent=snapshot_read(declaration['fixtures'][job['fixture']]['snapshot']).to_dict()
            assert {k:v for k,v in parent.items() if k!='session'}=={k:v for k,v in payload.items() if k!='session'}
        seen={r['reading_id'] for r in payload['seen']};pose=tuple(payload['last_pose']);clock=record['start'];active=None
        last_reading=payload['last_detector_reading'];reading_count=0
        for event in record['events']:
            at=event['at_seconds'];assert clock-1e-9<=at<=record['start']+record['horizon']+1e-8;clock=at
            if event['operation']=='plan_next':
                assert active is None
                plan=decode_plan(event['plan']);mask=np.zeros(plan.lattice_shape,bool);mask.ravel()[list(plan.requested_cells)]=True
                physical=compile_profile_scan(mask,config.session.instrument,scan_id=plan.scan_id,started_at_seconds=at,current_position=pose,
                    last_reading_at_seconds=last_reading['acquired_at_seconds'] if last_reading else None,
                    last_integration_seconds=last_reading['integration_seconds'] if last_reading else None,target_mass=config.session.policy.gamma)
                assert physical.samples==plan.samples and physical.segments==plan.segments
                assert set(plan.requested_cells)<=set(plan.swept_cells)
                assert plan.duration_seconds<=event['budget_seconds']+1e-8
                if plan.samples:
                    assert len(plan.samples)>=4;active=plan
                    assert plan.completed_at_seconds+config.delivery_delay_seconds+config.completion_overrun_seconds<=at+event['budget_seconds']+1e-8
                assert event['wind']['available_at_seconds']<=at
            elif event['operation']=='ingest_readings':
                assert active is not None
                for reading in event['readings']:
                    assert reading['reading_id'] not in seen and reading['scan_id']==active.scan_id
                    assert reading['acquired_at_seconds']<=reading['available_at_seconds']<=at+1e-9
                    assert reading['acquired_at_seconds']-reading['integration_seconds']/2>=active.started_at_seconds-1e-9
                    if last_reading:
                        assert reading['acquired_at_seconds']-last_reading['acquired_at_seconds']>=1/config.capabilities.readout_hz-1e-9
                    seen.add(reading['reading_id']);last_reading=reading;reading_count+=1
            else:
                assert active is not None and event['receipt']['scan_id']==active.scan_id
                pose=tuple(event['receipt']['final_position_yx']);active=None
        assert active is None and reading_count==sum(s['readings'] for s in record['scans'])
        assert abs(sum(q['duration'] for q in record['quadrature_samples'])-job['horizon'])<1e-8
        assert abs(sum(q['duration']*q['map_mse'] for q in record['quadrature_samples'])-record['J'])<1e-12
        assert record['spent_seconds']<=job['horizon']+1e-8
        kind=('motion-risk' in job['fixture'],job['policy'])
        if replay and job['mode']=='short' and job['seed']==0 and kind not in representative:
            truth=TruthQuadrature(config.source.build(),record['start'],record['horizon'],config.model.grid_shape,
                                  config.evaluation.shape,config.evaluation.step_seconds)
            checked=evaluate_execution(snapshot,record,truth,verify_policy=True)
            assert abs(checked['J']-record['J'])<=1e-12
            representative.add(kind);replayed.append(job['id'])
        revisions[record['producing_revision']]=revisions.get(record['producing_revision'],0)+1;records+=1
    replay_path=root/'public-replay.json'
    if replay:
        write_json(replay_path,{'declaration_sha256':digest,'public_policy_replays':replayed})
    elif replay_path.exists():
        saved=read(replay_path)
        assert saved['declaration_sha256']==digest
        replayed=saved['public_policy_replays']
    if final:
        assert not missing and records==348
        assert len(list((root/'bounds').glob('*.json')))==8
        assert len(list((root/'refinement').glob('*.json')))==8
        assert len(list((root/'contrast-refinement').rglob('*.json')))==14
        assert len(list((root/'coverage').rglob('*.json')))==348
        assert len(list((root/'reading-diagnostics').rglob('*.json')))==152
        assert read(root/'summary.json')['complete_records']==348
        assert read(root/'inspection'/'review.json')['status']=='reviewed'
        assert len(replayed)==8
    result={'artifact_type':'threshold_bound_audit_v1','status':'passed','final':final,'records':records,'missing':missing,
        'public_policy_replays':replayed,'producing_revisions':revisions,'declaration_sha256':digest,
        'verification_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'observer_provenance':'Observer implementation checkpoint 67bca3f; early observer records name base checkout 1149a13 while these additive observers were developed. Operational study code was committed before execution.'}
    write_json(root/'audit.json',result)
    if final:
        files=[p for p in root.rglob('*') if p.is_file() and p.suffix in ('.json','.png','.mp4') and p.name not in ('artifact-manifest.json','closeout.json')]
        write_json(root/'artifact-manifest.json',{'artifact_type':'threshold_artifact_manifest_v1','revision':result['verification_revision'],
            'files':[{'path':str(p.relative_to(root)),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(files)]})
    print(json.dumps({k:result[k] for k in ('status','records','missing','public_policy_replays')},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--final',action='store_true');p.add_argument('--replay',action='store_true')
    args=p.parse_args();audit(args.output,args.final,args.replay)
