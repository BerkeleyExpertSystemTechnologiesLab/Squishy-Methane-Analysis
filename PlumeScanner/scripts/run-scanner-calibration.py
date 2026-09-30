"""Frozen, resumable development/calibration/test pipeline; all I/O is here."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import subprocess
import numpy as np
from scipy.sparse import csr_matrix

from plume_focus.calibrated_scanner import CalibrationSession,CalibrationSettings
from plume_focus.calibration_execution import CalibrationRunConfig
from plume_focus.calibration_evaluation import AffineReplay,AffineLoss,fit_rate
from plume_focus.full_fov_execution import (FullFovRunConfig,decode_config,run_full_fov,verify_replay)
from plume_focus.signal_belief import grid_positions,interpolation_stencil
from plume_focus.sensing_risk import projection_gram
from plume_focus.sensing_session import configuration_hash
from calibration_truth_grid import clean_grid


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def revision():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix('.partial');temp.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)


def code_hash():
    paths=[Path(__file__),Path('scripts/calibration_truth_grid.py'),*sorted(Path('src/plume_focus').glob('*.py'))]
    return hashlib.sha256(b''.join(p.name.encode()+p.read_bytes().replace(b'\r\n',b'\n') for p in paths)).hexdigest()


def compatible_code(root,path,previous):
    if previous==code_hash():return True
    proof=Path(root)/'query-compatibility.json'
    if not proof.exists():return False
    value=read(proof)
    return (value['from_code_sha256']==previous and value['to_code_sha256']==code_hash()
        and value['status']=='verified_numerical_equivalence'
        and value['files'].get(Path(path).relative_to(root).as_posix())==sha(path))


def base_config(profile,case,seed,noise,margin=0.):
    old=FullFovRunConfig.from_dict(read(f'outputs/lean-full-fov/configs/{profile}-continuous-noise-0.json'))
    session=CalibrationSession(old.session.instrument,old.model,CalibrationSettings(vertical_margin_fov=margin),
                               old.session.source_position,old.session.home_yx)
    source=old.source
    if seed:
        phases=np.random.default_rng(seed).uniform(0,2*np.pi,3)
        source=replace(source,lateral_phase=float(phases[0]),amplitude_phase=float(phases[1]),shimmer_phase=float(phases[2]))
    source=replace(source,signal_scale=0. if case=='blank' else .3 if case=='weak' else 1.)
    wind=old.wind
    if case=='wind_error':wind=replace(wind,velocity_yx=tuple(np.array(wind.velocity_yx)+[.003,-.004]))
    return CalibrationRunConfig(session,source,wind,seed=seed,simulated_noise_std=noise,
        cessation_seconds=180. if case=='cessation' else None,
        active_windows=((0.,120.),(240.,360.)) if case=='intermittent' else None)


def declare(root):
    path=root/'declaration.json'
    if path.exists():
        for job in read(path)['development']+read(path)['heldout']:
            assert sha(root/job['config'])==job['config_sha256']
        return
    groups={'development':[],'heldout':[]}
    for stage in groups:
        cases=('continuous','blank','cessation') if stage=='development' else ('continuous','wind_error','weak','blank','cessation','intermittent')
        seeds=((0,0.),(0,.05)) if stage=='development' else ((101,.01),(102,.05))
        variants=(0.,.05,.10) if stage=='development' else ('baseline','calibrated','full')
        for profile in ('coarse','double'):
            for case in cases:
                for seed,noise in seeds:
                    for variant in variants:
                        name=f'{stage}-{profile}-{case}-seed{seed}-noise{noise:g}-{variant}'
                        config=base_config(profile,case,seed,noise,float(variant) if stage=='development' else 0.)
                        relative=f'configs/{name}.json';write(root/relative,config.to_dict())
                        groups[stage].append({'name':name,'profile':profile,'case':case,'seed':seed,'noise':noise,
                            'variant':variant,'config':relative,'config_sha256':sha(root/relative)})
    write(path,{'artifact_type':'scanner_calibration_declaration_v1','revision':revision(),**groups,
        'half_lives':[None,3.,6.,12.],'source_widths':[.0175,.035],'rate_bounds':[0.,.30],
        'heldout_is_template':'calibrated parameters bound only from locked development selection; no test-driven tuning',
        'margin_selection':'smallest positive with every continuous profile/noise coverage >=.95 and aggregate J/T <=1.05 baseline; else0',
        'fit_selection':'development sum causal J; ties max(1e-10,1% best): unchanged, longer half-life, original width, lower rate'})


@lru_cache(maxsize=2)
def cached_truth(root,config_json):
    config=decode_config(json.loads(config_json));source=config.build_source()
    shape=config.evaluation_shape;latent=config.model.grid_shape;lattice=config.session.instrument.lattice_shape
    key=configuration_hash({'source':config.source.to_dict(),'cessation':config.cessation_seconds,
        'windows':config.active_windows,'shape':shape,'latent':latent,'lattice':lattice,
        'step':config.evaluation_step_seconds,'duration':config.duration_seconds})
    path=Path(root)/'truth-cache'/f'{key}.npz'
    if path.exists():
        z=np.load(path);q=SimpleNamespace(**{k:z[k] for k in ('times','durations','gram','cross','square','average','mean_operator','peak')})
        q.core=csr_matrix((z['data'],z['indices'],z['indptr']),shape=(len(q.times),int(np.prod(lattice))))
        return q
    times=np.arange(int(np.ceil(config.duration_seconds/config.evaluation_step_seconds)))*config.evaluation_step_seconds
    durations=np.diff(np.append(times,config.duration_seconds));points=grid_positions(shape)
    ids,weights=interpolation_stencil(points,latent);gram=projection_gram(latent,shape)
    mapping=np.ravel_multi_index(tuple((points*np.array(lattice)).astype(int).T),lattice)
    mean_operator=np.zeros(int(np.prod(latent)));np.add.at(mean_operator,ids.ravel(),weights.ravel()/len(points))
    cross=[];square=[];average=[];peak=[];data=[];indices=[];indptr=[0]
    for t in times:
        values=clean_grid(config,float(t),shape).ravel()
        projected=np.zeros(int(np.prod(latent)));np.add.at(projected,ids.ravel(),(weights*values[:,None]).ravel()/len(points))
        cross.append(projected);square.append(float(np.mean(values**2)));average.append(float(values.mean()));peak.append(float(values.max()))
        core=values>max(.15,.5*values.max());counts=np.bincount(mapping[core],minlength=int(np.prod(lattice)))
        nz=np.flatnonzero(counts);indices.extend(nz);data.extend(counts[nz]/len(points));indptr.append(len(data))
    arrays=dict(times=times,durations=durations,gram=gram,cross=np.array(cross),square=np.array(square),average=np.array(average),
        mean_operator=mean_operator,peak=np.array(peak),data=np.array(data),indices=np.array(indices,dtype=int),indptr=np.array(indptr,dtype=int))
    path.parent.mkdir(parents=True,exist_ok=True)
    # Workers may produce the same deterministic cache concurrently; each writes
    # a distinct temporary file before atomically replacing the shared entry.
    import os
    tmp=path.with_name(path.stem+f'-{os.getpid()}.partial.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)
    q=SimpleNamespace(**{k:arrays[k] for k in ('times','durations','gram','cross','square','average','mean_operator','peak')})
    q.core=csr_matrix((arrays['data'],arrays['indices'],arrays['indptr']),shape=(len(times),int(np.prod(lattice))))
    return q


def truth(root,config):
    # Exclude the fitted estimator/policy and noise from cache identity.
    canonical=replace(config,session=replace(config.session,model=replace(config.model,source_rate=.15,source_sigma=.035),
        settings=CalibrationSettings(),half_life_seconds=None),simulated_noise_std=0.,seed=0)
    return cached_truth(str(root),json.dumps(canonical.to_dict(),sort_keys=True))


def metrics(record,q):
    weighted=0.;support_time=0.;core_area_time=0.;captured_area_time=0.;read_core=0;read_background=0
    false_focus=0;missed_focus=0;refocus_after_background=0;startup=False;last_background_full=False
    travel=acquisition=waiting=latency=0.
    for scan in record['scans']:
        p=scan['plan'];receipt=scan['receipt'];end=receipt['received_at_seconds']
        at=p['started_at_seconds'];ids=np.flatnonzero((q.times>=at-1e-10)&(q.times<end-1e-10))
        visible=float(np.interp(at,q.times,q.peak))>.15
        if p['mode']=='FOCUS':
            false_focus+=not visible
            refocus_after_background+=last_background_full
            if len(ids):
                core=q.core[ids];total=np.asarray(core.sum(axis=1)).ravel()
                inside=np.asarray(core[:,p['swept_cells']].sum(axis=1)).ravel();valid=total>0
                weighted+=float(np.sum((inside[valid]/total[valid])*q.durations[ids[valid]]))
                support_time+=float(q.durations[ids[valid]].sum())
                core_area_time+=float(total@q.durations[ids]);captured_area_time+=float(inside@q.durations[ids])
        elif startup and visible:missed_focus+=1
        for target,clean in zip(p['samples'],scan['clean_readings']):
            time=target['acquired_at_seconds']
            peak=float(np.interp(time,q.times,q.peak))
            read_core+=clean>max(.15,.5*peak);read_background+=clean<=.15
        for segment in p['segments']:
            duration=segment['completed_at_seconds']-segment['started_at_seconds']
            kind=segment['kind']
            if kind=='sweep':acquisition+=duration
            elif kind=='move':travel+=duration
            else:waiting+=duration
        latency+=receipt['received_at_seconds']-receipt['completed_at_seconds']
        last_background_full=p['mode']=='FULL' and not scan['missing_full_cells'] and max(scan['clean_readings'],default=0)<=.15
        startup=scan['startup_complete']
    # Causal ages use only delivered readings; unobserved cells stay separate.
    shape=tuple(record['configuration']['session']['instrument']['lattice_shape']);visits=np.full(shape,-1.)
    deliveries=[e for e in record['events'] if e['operation']=='ingest_readings'];k=0;age_integral=unknown_integral=0.
    for t,dt in zip(q.times,q.durations):
        while k<len(deliveries) and deliveries[k]['at_seconds']<=t+1e-10:
            for r in deliveries[k]['readings']:
                cell=tuple(np.minimum((np.array(r['position_yx'])*shape).astype(int),np.array(shape)-1))
                visits[cell]=max(visits[cell],r['acquired_at_seconds'])
            k+=1
        observed=visits>=0
        if observed.any():age_integral+=float(np.mean(t-visits[observed]))*dt
        unknown_integral+=float((~observed).mean())*dt
    return {'focus_core_geometric_coverage':weighted/support_time if support_time else None,
        'focus_core_supported_seconds':support_time,'core_area_time':core_area_time,'captured_core_area_time':captured_area_time,
        'actual_core_readings':int(read_core),'actual_background_readings':int(read_background),
        'actual_core_reference':'clean exposure response vs .15 and half interpolated instantaneous truth peak',
        'false_focus_decisions':int(false_focus),'missed_focus_decisions':int(missed_focus),
        'refocus_after_background_full_scan':int(refocus_after_background),
        'focus_scans':sum(s['plan']['mode']=='FOCUS' for s in record['scans']),
        'full_scans':sum(s['plan']['mode']=='FULL' for s in record['scans']),
        'travel_seconds':travel,'acquiring_pass_seconds':acquisition,'waiting_seconds':waiting,'receipt_latency_seconds':latency,
        'mean_observed_sample_age_seconds':age_integral/sum(q.durations),'mean_unknown_fraction':unknown_integral/sum(q.durations)}


def configured(root,job,stage):
    assert sha(root/job['config'])==job['config_sha256']
    config=decode_config(read(root/job['config']))
    if stage=='heldout':
        freeze=read(root/'freeze.json')
        assert freeze['declaration_sha256']==sha(root/'declaration.json')
        if job['variant']=='calibrated':
            config=replace(config,session=replace(config.session,settings=replace(config.session.settings,vertical_margin_fov=freeze['margin']),
                half_life_seconds=freeze['half_life_seconds'],model=replace(config.model,source_sigma=freeze['source_width'],source_rate=freeze['source_rate'])))
        elif job['variant']=='full':config=replace(config,session=replace(config.session,settings=replace(config.session.settings,full_scan_only=True)))
    return config


def execute(root,job,stage):
    root=Path(root);config=configured(root,job,stage);path=root/'runs'/f"{job['name']}.json"
    if path.exists():
        saved=read(path)
        assert saved['configuration_hash']==config.digest and compatible_code(root,path,saved['code_sha256'])
        return job['name'],'reused'
    q=truth(root,config);result=run_full_fov(config,truth=q)
    result.update(job=job,stage=stage,producing_revision=revision(),code_sha256=code_hash(),declaration_sha256=sha(root/'declaration.json'))
    result['metrics']=metrics(result,q)
    write(path,result)
    return job['name'],{'J/T':result['J_over_T'],'core':result['metrics']['focus_core_geometric_coverage']}


def select_margin(root):
    jobs=read(root/'declaration.json')['development'];groups={m:[] for m in (0.,.05,.1)}
    for job in jobs:
        record=read(root/'runs'/f"{job['name']}.json")
        if job['case']=='continuous':groups[job['variant']].append(record)
    base=sum(r['J_over_T'] for r in groups[0.]);checks=[];selected=0.
    for m in (.05,.1):
        rows=groups[m];coverage=[r['metrics']['focus_core_geometric_coverage'] for r in rows]
        cost=sum(r['J_over_T'] for r in rows)
        passes=all(v is not None and v>=.95 for v in coverage) and cost<=1.05*base
        checks.append({'margin':m,'coverage':coverage,'aggregate_J_over_T':cost,'baseline_aggregate':base,'passes':passes})
        if passes and selected==0:selected=m
    write(root/'margin-selection.json',{'margin':selected,'checks':checks,'declaration_sha256':sha(root/'declaration.json'),
        'inputs':[{'path':str(root/'runs'/f"{j['name']}.json"),'sha256':sha(root/'runs'/f"{j['name']}.json")} for j in jobs]})


def fit_one(root,half,width):
    root=Path(root);selection=read(root/'margin-selection.json');margin=selection['margin']
    tag=f'half-{half}-width-{width}';target=root/'fits'/f'{tag}.json'
    if target.exists():
        saved=read(target);assert saved['selection_sha256']==sha(root/'margin-selection.json') and compatible_code(root,target,saved['code_sha256'])
        return tag,'reused'
    coeff=np.zeros(3);rows=[]
    for job in read(root/'declaration.json')['development']:
        if job['variant']!=margin:continue
        path=root/'runs'/f"{job['name']}.json";record=read(path);config=decode_config(record['configuration'])
        session=replace(config.session,half_life_seconds=half,model=replace(config.model,source_rate=.30,source_sigma=width))
        replay=AffineReplay(session);loss=AffineLoss(truth(root,config))
        for event in record['events']:
            loss.before(replay,event['at_seconds']);replay.event(event)
        loss.before(replay,config.duration_seconds+1e-8)
        coeff+=loss.coefficients
        rows.append({'name':job['name'],'input_sha256':sha(path),'coefficients':loss.coefficients.tolist()})
    result=fit_rate(coeff);result.update(half_life_seconds=half,source_width=width,records=rows,
        selection_sha256=sha(root/'margin-selection.json'),code_sha256=code_hash(),producing_revision=revision())
    write(target,result);return tag,result['source_rate'],result['loss']


def freeze(root):
    selection=read(root/'margin-selection.json');decl=read(root/'declaration.json');rows=[]
    for half in decl['half_lives']:
        for width in decl['source_widths']:rows.append(read(root/'fits'/f'half-{half}-width-{width}.json'))
    unchanged=next(r for r in rows if r['half_life_seconds'] is None and r['source_width']==.035)
    a,b,c=unchanged['coefficients'];rows.append({**unchanged,'source_rate':.15,'loss':a*.25+b+c,'unchanged':True})
    best=min(r['loss'] for r in rows);tolerance=max(1e-10,.01*abs(best));ties=[r for r in rows if r['loss']<=best+tolerance]
    chosen=min(ties,key=lambda r:(not r.get('unchanged',False),-(r['half_life_seconds'] or float('inf')),r['source_width']!=.035,r['source_rate']))
    value={'artifact_type':'scanner_calibration_freeze_v1','margin':selection['margin'],
        **{k:chosen[k] for k in ('half_life_seconds','source_width','source_rate','loss')},
        'declaration_sha256':sha(root/'declaration.json'),'margin_selection_sha256':sha(root/'margin-selection.json'),
        'fit_inputs':[{'path':str(p),'sha256':sha(p)} for p in sorted((root/'fits').glob('*.json'))],
        'candidates':[{k:r.get(k) for k in ('half_life_seconds','source_width','source_rate','loss','unchanged')} for r in rows],
        'tie_tolerance':tolerance,'producing_revision':revision()}
    path=root/'freeze.json'
    if path.exists():assert read(path)==value,'Attempted to change frozen calibration'
    else:write(path,value)
    print({k:value[k] for k in ('margin','half_life_seconds','source_width','source_rate','loss')})


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['declare','development','select-margin','fit','freeze','heldout'],required=True)
    p.add_argument('--root',type=Path,default=Path('outputs/scanner-calibration'));p.add_argument('--workers',type=int,default=2)
    args=p.parse_args();root=args.root
    if args.phase=='declare':declare(root)
    elif args.phase=='select-margin':select_margin(root)
    elif args.phase=='freeze':freeze(root)
    else:
        decl=read(root/'declaration.json')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            if args.phase=='fit':futures=[pool.submit(fit_one,str(root),half,width) for half in decl['half_lives'] for width in decl['source_widths']]
            else:futures=[pool.submit(execute,str(root),j,args.phase) for j in decl[args.phase]]
            for f in as_completed(futures):print(f.result(),flush=True)

if __name__=='__main__':main()
