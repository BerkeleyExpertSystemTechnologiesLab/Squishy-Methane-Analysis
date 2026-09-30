"""Verify budget-gap evidence and collect paired, hash-checked inspection media."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import cv2

from plume_focus.sensing_cli_v1 import write_json, artifact_manifest
from reconstruction_risk_observers import INSPECTION_CASES


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_historical_compatibility(root,jobs):
    starting=read(root/'declared-study.json')['starting_revision']
    unchanged={}
    for name in ('signal_belief','motion_belief','physical_scan','sensing_policy_v1','sensing_policy_v2','sensing_simulation'):
        path=f'src/plume_focus/{name}.py'
        before=subprocess.run(['git','rev-parse',f'{starting}:{path}'],capture_output=True,text=True,check=True).stdout.strip()
        after=subprocess.run(['git','rev-parse',f'HEAD:{path}'],capture_output=True,text=True,check=True).stdout.strip()
        assert before==after;unchanged[path]=after
    path='src/plume_focus/sensing_reconstruction.py'
    before=subprocess.run(['git','show',f'{starting}:{path}'],capture_output=True,text=True,encoding='utf-8',check=True).stdout
    def loss_ast(source):return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='reconstruction_loss'))
    assert loss_ast(before)==loss_ast(Path(path).read_text(encoding='utf-8'))
    count=0
    for name,job in jobs.items():
        if job['new']:continue
        for filename,digest in job['input_hashes'].items():
            assert sha(Path(job['source_directory'])/filename)==digest==sha(root/'development'/name/filename)
        count+=1
    assert count==60
    return {'passed':True,'verified_historical_controls':count,'starting_revision':starting,'unchanged_git_blobs':unchanged,
        'evaluation_loss_ast_sha256':hashlib.sha256(loss_ast(before).encode()).hexdigest(),
        'basis':'Preserved prior-study execution/evaluation evidence, immutable file/configuration checks, unchanged model/mechanics/legacy policy/loss arithmetic, regression checks, and reviewed v4-only controller dispatch changes. These 60 controls were not freshly re-executed in this study.'}


def audit_execution(directory):
    trace=read(directory/'trace.json');report=read(directory/'report.json')
    config=trace['configuration'];policy=config['session']['policy']
    active=None;available=0.;selected=below=required=complete=0;visited=set();last_reading=None
    height,width=config['session']['instrument']['lattice_shape']
    cadence=config['session']['instrument']['acquisition']['cadence_hz']
    rows={r['scan_id']:r for r in report['scan_schedule']}
    for event in trace['events']:
        if event['operation']=='plan_next':
            assert active is None and event['at_seconds']>=available-1e-9
            plan=event['plan']
            if not plan['samples']:continue
            assert len(plan['samples'])>=4
            assert plan['completed_at_seconds']-plan['started_at_seconds']<=event['request']['max_duration_seconds']+1e-9
            assert set(plan['requested_cells']).issubset(plan['swept_cells'])
            active=plan;visited=set()
            if plan['status']=='value_selected':
                selected+=1;below+=plan['predicted_mass_recall'] is not None and plan['predicted_mass_recall']<.9
                choices=plan['risk_candidates'];assert 0<len(choices)<=14
                risks=[c['expected_integrated_map_risk'] for c in choices]
                tolerance=max(policy['risk_absolute_tolerance'],policy['risk_relative_tolerance']*plan['idle_integrated_map_risk'])
                assert risks[plan['selected_candidate']]<=min(risks)+tolerance+1e-12
                assert not rows[plan['scan_id']]['mass_target_is_execution_requirement']
        elif event['operation']=='ingest_readings':
            assert active is not None
            assert all(r['scan_id']==active['scan_id'] and r['available_at_seconds']<=event['at_seconds']+1e-9 for r in event['readings'])
            for reading in event['readings']:
                if last_reading is not None:
                    assert reading['acquired_at_seconds']-last_reading>=1/cadence-1e-9
                last_reading=reading['acquired_at_seconds']
                y,x=reading['position_yx'];visited.add(min(int(y*height),height-1)*width+min(int(x*width),width-1))
        elif event['operation']=='finish_scan':
            assert active is not None
            row=rows[active['scan_id']]
            if active['service_status']=='required':
                required+=1;complete+=bool(row['service_completed'])
                assert row['service_completed']==set(active['service_required_cells']).issubset(visited)
            available=event['at_seconds'];active=None
    assert active is None
    return {'name':directory.name,'passed':True,'value_selected':selected,'below_90_percent':below,
            'required_service':required,'completed_service':complete,
            'operational_hash':report['operational_hash']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);args=parser.parse_args();root=args.directory.resolve()
    jobs=read(root/'execution-configurations.json');records=[]
    write_json(root/'historical-compatibility.json',audit_historical_compatibility(root,jobs))
    audits=[audit_execution(root/'development'/name) for name,job in jobs.items() if job['new']]
    assert len(audits)==36
    zero_controls=[]
    for name,job in jobs.items():
        motion=job['configuration']['session'].get('motion')
        if not job['new'] or motion is None or any(v!=0 for row in motion['bias_covariance_yx'] for v in row):continue
        paired=name.replace('-motion-risk','-baseline-risk')
        a=read(root/'development'/name/'trace.json');b=read(root/'development'/paired/'trace.json')
        assert a['events']==b['events'] and a['executions']==b['executions']
        ar=read(root/'development'/name/'report.json');br=read(root/'development'/paired/'report.json')
        assert ar['reconstruction']['J']==br['reconstruction']['J']
        zero_controls.append({'mixture':name,'gaussian':paired,'identical_events_execution_and_loss':True})
    assert len(zero_controls)==8
    write_json(root/'execution-conformance.json',{'passed':True,'records':audits,
        'zero_uncertainty_controls':zero_controls,
        'note':'Recorded lifecycle, feasibility, requested-cell closure and selected-risk checks; independent of scientific advantage.'})
    historical=Path(read(root/'declared-study.json')['historical_root']).resolve()
    old_media=read(historical/'inspection-manifest.json')['videos']
    for case in INSPECTION_CASES:
        for suffix in ('','-risk'):
            name=case+suffix;directory=root/'development'/name
            report=read(directory/'report.json')
            if suffix:
                videos=read(directory/'inspection.json')['videos']
            else:
                videos=[r for r in old_media if r['name']==name]
                for record in videos:
                    for filename,digest in record['input_sha256'].items():
                        if sha(directory/filename)!=digest:
                            raise ValueError(f'Reused media input differs: {name}/{filename}')
                    source=historical/'development'/name/f"{record['mode']}.mp4"
                    target=directory/source.name
                    if sha(source)!=record['sha256'] or (target.exists() and sha(target)!=record['sha256']):
                        raise ValueError(f'Media hash mismatch: {source}')
                    if not target.exists():shutil.copyfile(source,target)
            if len(videos)!=2:raise ValueError(f'Missing physical/summary pair: {name}')
            for record in videos:
                path=directory/f"{record['mode']}.mp4"
                if sha(path)!=record['sha256'] or record['operational_hash']!=report['operational_hash']:
                    raise ValueError(f'Inspection does not match execution: {name}')
                capture=cv2.VideoCapture(str(path));frames=[];count=0
                selected={0,record['encoded_frames']//2,record['encoded_frames']-1}
                selected.update(r['frame_index'] for r in record.get('inspection_frames',()))
                out=root/'inspection-frames';out.mkdir(exist_ok=True)
                while True:
                    ok,frame=capture.read()
                    if not ok:break
                    if count in selected:
                        destination=out/f"{name}-{record['mode']}-{count:04d}.png"
                        cv2.imwrite(str(destination),frame)
                        frames.append({'frame_index':count,'path':str(destination.relative_to(root)).replace('\\','/')})
                    count+=1
                capture.release()
                if count!=record['encoded_frames']:raise ValueError(f'Incomplete decode: {path}')
                records.append({**record,'name':name,'reused_historical_media':not bool(suffix),
                    'output_path':str(path),'decoded_frames':count,'inspection_frames':frames})
    name='double-prior-startup-wind-error-threshold90_segment_v2-seed-0-motion'
    old=read(root/'development'/name/'report.json')['scan_schedule'][1]
    new=read(root/'refinement'/f'{name}-decision-1.json')
    assert abs(old['start_seconds']-new['decision_seconds'])<1e-9
    assert any(abs(c['duration_seconds']-12.25)<1e-9 and 'duration_budget' in c['failed_constraints'] for c in old['candidates'])
    choices=[{'proposal':c['proposal'],'duration_seconds':c['duration_seconds'],'predicted_mass_recall':mass}
             for c,mass in zip(new['candidates'],new['saved_budget_gap_candidate_coverage'])]
    assert choices and all(c['duration_seconds']<=12.+1e-9 for c in choices)
    assert any(c['predicted_mass_recall'] is not None and c['predicted_mass_recall']<.9 for c in choices)
    write_json(root/'saved-budget-gap-check.json',{'passed':True,'decision_seconds':new['decision_seconds'],
        'old_tracking_seconds':12.25,'budget_seconds':12.,'new_executable_alternatives':choices,
        'note':'Fixed saved evidence; alternatives need not retain 90% and this does not require selecting the plume.'})
    evaluations=[read(root/'development'/f'{case}-risk'/'evaluation-refinement.json') for case in INSPECTION_CASES]
    review_path=root/'visual-review.json';visual_status='pending_human_or_agent_frame_inspection'
    if review_path.exists():
        review=read(review_path);expected={case+suffix for case in INSPECTION_CASES for suffix in ('','-risk')}
        inspected={r['name'] for r in review['records']}
        for record in review['records']:
            source=Path(record.get('source_root',root))
            for frame in record['frames']:
                assert sha(source/frame['path'])==frame['sha256']
        if expected.issubset(inspected):visual_status='selected_frames_reviewed_with_hashes'
    write_json(root/'inspection-manifest.json',{'videos':records,'evaluation_refinements':evaluations,
        'visual_review_status':visual_status,'layout':'Existing six panels; no redesign'})
    artifact_manifest(root)
    print(f'Verified {len(records)} paired videos and the saved budget-gap fixture.')


if __name__=='__main__':main()
