"""Declare and resume evaluator-only focus extent probes over saved lean traces."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

from plume_focus.focus_extent_diagnostics import (EvidenceReplay, select_events, surfaces,
                                                 forecast, controlled_readings, THRESHOLD)
from plume_focus.full_fov_execution import FullFovRunConfig, replay_event
from plume_focus.full_fov_loop import FullFovController
from plume_focus.physical_scan import compile_physical_scan
from plume_focus.signal_belief import grid_positions, GaussianSignalBelief


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def revision(): return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix('.partial');tmp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)


def declare(root, source):
    path=root/'declaration.json'
    if path.exists():
        for job in read(path)['jobs']:
            if sha(job['input'])!=job['input_sha256']: raise ValueError('Changed input')
        return
    jobs=[]
    for job in read(source/'declaration.json')['jobs']:
        p=source/'runs'/f"{job['name']}.json";record=read(p)
        events,unavailable=select_events(record)
        jobs.append({'name':job['name'],'input':str(p),'input_sha256':sha(p),
                     'noise':job['noise'],'events':events,'unavailable':unavailable})
    write(path,{'artifact_type':'focus_extent_declaration_v1','revision':revision(),
        'source_declaration_sha256':sha(source/'declaration.json'), 'jobs':jobs,
        'horizons':[0,1,3,6,12], 'threshold':.15,'quantile':.95,
        'future_variants':['baseline','future_source_off','diffusion_off','process_off','both_off'],
        'selection':'post startup; first >=180; cessation first after post-180 full rescan',
        'scope':'fixed-evidence diagnosis; no new executed policies or tuned parameters'})


def compile_mask(mask, config, event, engine):
    last=engine._last_detector_reading
    p=compile_physical_scan(mask,config.session.instrument,scan_id='diagnostic',
        started_at_seconds=event['at_seconds'],current_position=tuple(event['position_yx']),
        last_reading_at_seconds=last.acquired_at_seconds if last else None,
        last_integration_seconds=last.integration_seconds if last else None)
    ceiling=min(12.,event['remaining_seconds'])
    return {'requested_cells':len(p.requested_cells),'swept_cells':len(p.swept_cells),
            'duration_seconds':p.duration_seconds,'ceiling_seconds':ceiling,
            'fits':len(p.samples)>=4 and p.duration_seconds+config.session.settings.delivery_delay_seconds+
                   config.session.settings.completion_overrun_seconds<=ceiling+1e-9}


def inspect(config,event,engine,source_off,public,source):
    now=event['at_seconds'];shape=config.session.instrument.lattice_shape
    before=engine._belief.mean.copy(),engine._belief.covariance.copy(),engine._belief.time_seconds
    horizons={0.,1.,3.,6.,12.}
    if event['plan']['mode']=='FOCUS':
        duration=event['plan']['completed_at_seconds']-now
        horizons.update((duration/2,duration))
    rows=[];images={}
    for horizon in sorted(horizons):
        truth=source.signal_at(np.full(np.prod(shape),now+horizon),grid_positions(shape)).reshape(shape)
        for variant in ('baseline','future_source_off','diffusion_off','process_off','both_off','history_source_off'):
            base=source_off if variant=='history_source_off' else engine
            b=forecast(base._belief,horizon,base._wind,base.source_position,variant)
            stats,mean,upper=surfaces(b,shape,base._wind.velocity_yx,truth)
            stats.update(variant=variant,horizon_seconds=horizon)
            if horizon in (0.,6.):
                stats['mean_compilation']=compile_mask(mean>=THRESHOLD,config,event,engine)
                stats['upper_compilation']=compile_mask(upper>=THRESHOLD,config,event,engine)
            rows.append(stats)
            if horizon in (0.,6.) and variant in ('baseline','history_source_off'):
                key=f'{variant}_{horizon:g}'
                images[key+'_mean']=mean;images[key+'_upper']=upper;images[key+'_truth']=truth
    # Projection refinement does not alter physical eligibility or execution.
    refinement=[]
    for horizon in (0.,6.):
        b=forecast(engine._belief,horizon,engine._wind,engine.source_position)
        for mesh in ((120,160),(240,320)):
            truth=source.signal_at(np.full(np.prod(mesh),now+horizon),grid_positions(mesh)).reshape(mesh)
            stats,_,_=surfaces(b,mesh,engine._wind.velocity_yx,truth)
            refinement.append({'horizon_seconds':horizon,'mesh':mesh,**stats})
    measured=np.full(shape,np.nan);clean=np.full(shape,np.nan);age=np.full(shape,np.nan)
    for r in sorted(public._seen.values(),key=lambda r:r.acquired_at_seconds):
        cell=tuple(np.minimum((np.array(r.position_yx)*shape).astype(int),np.array(shape)-1))
        measured[cell]=r.value;age[cell]=now-r.acquired_at_seconds
    images['measured']=measured;images['sample_age']=age
    assert np.array_equal(engine._belief.mean,before[0]) and np.array_equal(engine._belief.covariance,before[1])
    assert engine._belief.time_seconds==before[2]
    return {'at_seconds':now,'mode':event['plan']['mode'],'scan_id':event['plan']['scan_id'],
        'forecast_records':rows,'projection_refinement':refinement,
        'measured_cells':int(np.isfinite(measured).sum()),
        'measured_above_threshold_cells':int(np.sum(measured>=THRESHOLD)),
        'unknown_cells':int(np.isnan(measured).sum()),
        'requested_cells':len(event['plan']['requested_cells']),
        'swept_cells':len(event['plan']['swept_cells']),
        'read_only_passed':True},images


def execute(root,job):
    root=Path(root);target=root/'runs'/f"{job['name']}.json"
    declaration_hash=sha(root/'declaration.json')
    if sha(job['input'])!=job['input_sha256']: raise ValueError('Changed evidence')
    code_hash=hashlib.sha256((Path(__file__).read_bytes()+Path('src/plume_focus/focus_extent_diagnostics.py').read_bytes())).hexdigest()
    if target.exists():
        prior=read(target)
        if prior['declaration_sha256']!=declaration_hash or prior['code_sha256']!=code_hash: raise ValueError('Changed diagnostic inputs')
        for artifact in prior['artifacts']:
            if sha(artifact['path'])!=artifact['sha256']: raise ValueError('Changed diagnostic arrays')
        return job['name'],'reused'
    record=read(job['input']);config=FullFovRunConfig.from_dict(record['configuration'])
    public=FullFovController(config.session)
    engine=EvidenceReplay(config.session);off=EvidenceReplay(config.session,source_off=True)
    selected={e['event_index']:e for e in job['events']};rows=[];artifacts=[];max_error=0.
    source=config.build_source()
    delivered={}
    for e in record['events']:
        for r in e.get('readings',[]): delivered[r['reading_id']]=r
    clean_by_id={}
    for s in record['scans']:
        readings=sorted((r for r in delivered.values() if r['scan_id']==s['plan']['scan_id']),
                        key=lambda r:r['acquired_at_seconds'])
        if len(readings)!=len(s['clean_readings']):raise ValueError('Exposure record count differs')
        clean_by_id.update((r['reading_id'],v) for r,v in zip(readings,s['clean_readings']))
    for i,event in enumerate(record['events']):
        if i>max(selected,default=-1):break
        if i in selected:
            # Prepare exactly the same causal decision boundary before forecasting.
            engine.decision(event);off.decision(event)
            row,images=inspect(config,event,engine,off,public,source)
            clean=np.full(config.session.instrument.lattice_shape,np.nan)
            for r in sorted(public._seen.values(),key=lambda r:r.acquired_at_seconds):
                cell=tuple(np.minimum((np.array(r.position_yx)*clean.shape).astype(int),np.array(clean.shape)-1))
                clean[cell]=clean_by_id[r.reading_id]
            images['clean_exposure_readings']=clean
            arrays=root/'arrays'/f"{job['name']}-event-{i}.npz";arrays.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(arrays,**images)
            artifacts.append({'path':str(arrays),'sha256':sha(arrays)})
            row.update(selected[i]);rows.append(row)
        replay_event(public,event)
        engine.event(event);off.event(event)
        error=max(float(np.max(abs(public._belief.mean-engine._belief.mean))),
                  float(np.max(abs(public._belief.covariance-engine._belief.covariance))))
        max_error=max(max_error,error)
        if error>1e-12:raise AssertionError('Fixed-evidence replay diverges from public execution')
    write(target,{'artifact_type':'focus_extent_run_v1','producing_revision':revision(),
        'code_sha256':code_hash,'declaration_sha256':declaration_hash,'job':job,
        'public_replay_max_error':max_error,'snapshots':rows,'artifacts':artifacts})
    return job['name'],len(rows)


def controls(root,source):
    config=FullFovRunConfig.from_dict(read(source/'configs/coarse-continuous-noise-0.json'))
    model=config.session.model
    records=controlled_readings(model,config.capabilities.footprint)
    # Independent continuous Gaussian packet references, not fitted simulator truth.
    packets=[];positions=grid_positions(model.grid_shape);start=np.array([.5,.2]);sigma=.035
    for continuous in (False,True):
        for fading in (False,True):
            b=GaussianSignalBelief(model,transport_id='zero_inflow_v1')
            b.mean[:]=0 if continuous else np.exp(-np.sum((positions-start)**2,axis=1)/(2*sigma**2))
            for h in (0.,1.,3.,6.,12.):
                clone=forecast(b,h,config.wind,tuple(start) if continuous else None)
                pos=grid_positions((120,160));vel=np.array(config.wind.velocity_yx)
                def packet(age):
                    amp=np.exp(-age/(52/12)) if fading else 1.
                    if fading and age>5.5:amp=0.
                    return amp*np.exp(-np.sum((pos-start-vel*age)**2,axis=1)/(2*sigma**2))
                if continuous:
                    # Midpoint quadrature over emissions; half-step check below.
                    def integrate(step):
                        n=max(1,int(np.ceil(h/step)));dt=h/n
                        return sum((packet((k+.5)*dt) for k in range(n)),np.zeros(len(pos)))*dt*model.source_rate
                    truth=integrate(.025);fine=integrate(.0125)
                    err=float(np.max(abs(truth-fine)))
                else:truth=packet(h);err=0.
                stats,_,_=surfaces(clone,(120,160),vel,truth.reshape(120,160))
                packets.append({'continuous':continuous,'fading_5_5s_lifetime':fading,'horizon_seconds':h,
                    'analytic_center_yx':(start+vel*h).tolist(),'reference_quadrature_max_error':err,**stats})
    write(root/'controls.json',{'artifact_type':'focus_extent_controls_v1','revision':revision(),
          'readings':records,'packet_references':packets})


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['declare','run','controls'],required=True)
    p.add_argument('--root',type=Path,default=Path('outputs/focus-extent'))
    p.add_argument('--source',type=Path,default=Path('outputs/lean-full-fov'))
    p.add_argument('--workers',type=int,default=2);p.add_argument('--zero-only',action='store_true')
    args=p.parse_args()
    if args.phase=='declare':declare(args.root,args.source)
    elif args.phase=='controls':controls(args.root,args.source)
    else:
        jobs=[j for j in read(args.root/'declaration.json')['jobs'] if not args.zero_only or j['noise']==0]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(execute,str(args.root),j) for j in jobs]):print(future.result(),flush=True)

if __name__=='__main__':main()
