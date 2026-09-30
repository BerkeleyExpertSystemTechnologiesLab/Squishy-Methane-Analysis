"""Replay, bounded refinement and representation checks for the declared study."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('calibration_driver',Path(__file__).with_name('run-scanner-calibration.py'))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
from plume_focus.full_fov_execution import decode_config,verify_replay
from plume_focus.focus_extent_diagnostics import EvidenceReplay
from plume_focus.calibrated_scanner import DecayGaussianBelief
from plume_focus.sensing_counterfactual import OnlineLoss
from plume_focus.signal_belief import grid_positions


def historical_zero(root):
    results=[]
    for profile in ('coarse','double'):
        for noise in ('0','0.05'):
            old=Path(f'outputs/lean-full-fov/runs/{profile}-continuous-noise-{noise}.json')
            new=root/'runs'/f'development-{profile}-continuous-seed0-noise{noise}-0.0.json'
            a,b=d.read(old),d.read(new)
            events=[]
            for event in b['events']:
                if event['operation']=='plan_next':
                    event=dict(event);plan=dict(event['plan'])
                    for key in ('selected_cells','padding_cells','clipped_selected_cells','vertical_margin_fov'):plan.pop(key)
                    plan['message_type']='FullFovPlan';event['plan']=plan
                events.append(event)
            assert events==a['events'],f'Historical zero-margin paths/evidence differ: {profile}/{noise}'
            assert abs(a['J']-b['J'])<1e-12
            results.append({'name':new.stem,'historical_sha256':d.sha(old),'new_sha256':d.sha(new),'exact_events':True})
    d.write(root/'checks'/'historical-zero.json',{'records':results});print('Four historical paths/evidence/losses match')


def replay(root,path):
    root=Path(root);path=Path(path);r=d.read(path);c=decode_config(r['configuration'])
    assert r['scans'][0]['plan']['mode']=='FULL' and r['scans'][0]['startup_complete']
    for scan in r['scans']:
        plan=scan['plan'];requested=set(plan['requested_cells']);swept=set(plan['swept_cells'])
        assert requested<=swept and len(scan['readings'])==len(plan['samples'])
        assert scan['receipt']['received_at_seconds']<=c.duration_seconds+1e-8
        if plan['mode']=='FULL':
            assert len(plan['samples'])==np.prod(c.session.instrument.lattice_shape)
            assert not scan['missing_full_cells']
        else:
            assert set(plan['selected_cells'])|set(plan['padding_cells'])==requested
            assert not set(plan['selected_cells'])&set(plan['padding_cells'])
            assert plan['completed_at_seconds']-plan['started_at_seconds']<=c.session.settings.focus_seconds+1e-8
    target=root/'checks'/'replay'/f'{path.stem}.json'
    if target.exists():
        saved=d.read(target);assert saved['sha256']==d.sha(path)
        if saved['code_sha256']==d.code_hash():return saved
    verify_replay(r,truth=d.truth(root,c))
    result={'name':path.stem,'sha256':d.sha(path),'code_sha256':d.code_hash(),'status':'passed'}
    d.write(target,result);return result


def refinement(root,path):
    root=Path(root);path=Path(path);r=d.read(path);base=decode_config(r['configuration'])
    target=root/'checks'/'refinement'/f'{path.stem}.json'
    if target.exists():
        old=d.read(target);assert old['input_sha256']==d.sha(path);return old
    rows=[{'shape':base.evaluation_shape,'step':base.evaluation_step_seconds,'J':r['J']}]
    accepted=False
    for factor in (2,4):
        config=replace(base,evaluation_shape=tuple(n*factor for n in base.evaluation_shape),
                       evaluation_step_seconds=base.evaluation_step_seconds/factor)
        truth_config=config
        if r['job']['profile']=='double':
            # Loss quadrature is a property of the common field, inference and
            # evaluation meshes/time grid, not scanner mechanics. Reuse the
            # already checked coarse field after explicit identity checks.
            other=path.with_name(path.name.replace('heldout-double-','heldout-coarse-',1))
            reference=decode_config(d.read(other)['configuration'])
            assert (reference.source==base.source and reference.active_windows==base.active_windows
                and reference.cessation_seconds==base.cessation_seconds and reference.duration_seconds==base.duration_seconds
                and reference.model.grid_shape==base.model.grid_shape)
            truth_config=replace(reference,evaluation_shape=config.evaluation_shape,
                                 evaluation_step_seconds=config.evaluation_step_seconds)
        q=d.truth(root,truth_config);loss=OnlineLoss(q);c=EvidenceReplay(config.session)
        c._belief=DecayGaussianBelief(config.model,half_life_seconds=config.session.half_life_seconds,
                                     transport_id=config.session.transport_id)
        for event in r['events']:
            loss.before(c,event['at_seconds']);c.event(event)
        loss.before(c,config.duration_seconds+1e-8);value=loss.result()['J']
        change=abs(value-rows[-1]['J']);tol=max(1e-10,.01*abs(value))
        rows.append({'shape':config.evaluation_shape,'step':config.evaluation_step_seconds,'J':value,
                     'change':change,'tolerance':tol})
        if change<=tol:accepted=True;break
    result={'name':path.stem,'input_sha256':d.sha(path),'rows':rows,
            'status':'converged' if accepted else 'numerically_unresolved'}
    d.write(target,result);return result


def representation(root,path):
    root=Path(root);path=Path(path);r=d.read(path);config=decode_config(r['configuration'])
    frozen=d.read(root/'freeze.json')
    session=replace(config.session,half_life_seconds=frozen['half_life_seconds'],
        model=replace(config.model,source_sigma=frozen['source_width'],source_rate=frozen['source_rate']))
    # Identical actual evidence through first completed full scan; no replanning.
    end=next(s['receipt']['received_at_seconds'] for s in r['scans'] if s['startup_complete'])
    rows=[];means=[]
    for shape in ((12,16),(24,32)):
        s=replace(session,model=replace(session.model,grid_shape=shape))
        c=EvidenceReplay(s);c._belief=DecayGaussianBelief(s.model,half_life_seconds=s.half_life_seconds,transport_id=s.transport_id)
        for event in r['events']:
            if event['at_seconds']>end+1e-10:break
            c.event(event)
            if event['operation']=='finish_scan' and abs(event['at_seconds']-end)<1e-8:break
        predictions=[]
        for horizon in (0.,1.,3.,6.,12.):
            b=c._belief.clone();b.advance(end+horizon,c._wind,c.source_position)
            mean,var=b.moments_on((120,160));mean=mean.ravel()
            truth=config.build_source().signal_at(np.full(mean.size,end+horizon),grid_positions((120,160)))
            predictions.append(mean)
            rows.append({'grid':shape,'horizon':horizon,'mse':float(np.mean((mean-truth)**2)),
                'mean_signal':float(mean.mean()),'mean_variance':float(var.mean())})
        means.append(predictions)
    changes=[float(np.mean((a-b)**2)) for a,b in zip(*means)]
    result={'name':path.stem,'input_sha256':d.sha(path),'freeze_sha256':d.sha(root/'freeze.json'),
        'snapshot_seconds':end,'rows':rows,'prediction_difference_mse':changes,
        'status':'controlled_resolution_sensitivity_not_a_convergence_proof'}
    d.write(root/'checks'/'representation'/f'{path.stem}.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['replay','refinement','representation','historical-zero'],required=True)
    p.add_argument('--root',type=Path,default=Path('outputs/scanner-calibration'));p.add_argument('--workers',type=int,default=2)
    a=p.parse_args();root=a.root;decl=d.read(root/'declaration.json')
    if a.phase=='historical-zero':historical_zero(root);return
    if a.phase=='replay':paths=sorted((root/'runs').glob('*.json'));fn=replay
    elif a.phase=='refinement':
        paths=[root/'runs'/f"{j['name']}.json" for j in decl['heldout']
            if j['variant']=='calibrated' and j['seed']==101 and j['case'] in ('continuous','blank')];fn=refinement
    else:
        margin=d.read(root/'margin-selection.json')['margin']
        paths=[root/'runs'/f"{j['name']}.json" for j in decl['development']
            if j['variant']==margin and j['noise']==0 and j['case']=='continuous'];fn=representation
    results=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        futures=[pool.submit(fn,str(root),str(path)) for path in paths]
        for f in as_completed(futures):
            result=f.result();results.append(result);print(result['name'],result.get('status','passed'),flush=True)
    d.write(root/'checks'/f'{a.phase}.json',{'producing_revision':d.revision(),'records':results})


if __name__=='__main__':main()
