"""Declare, run and inspect the bounded full-FoV mechanism/noise demonstration."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np

from plume_focus.full_fov_loop import FullFovSession, FullFovSettings
from plume_focus.full_fov_execution import FullFovRunConfig, run_full_fov, verify_replay
from plume_focus.sensing_cli_v1 import load_config
from plume_focus.sensing_counterfactual import TruthQuadrature


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def revision(): return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.partial')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8');temp.replace(path)


def declare(root):
    target=root/'declaration.json'
    if target.exists():
        for job in read(target)['jobs']:
            assert sha(root/job['config'])==job['sha256']
        return
    jobs=[]
    for profile in ('coarse','double'):
        path=Path('configs/sensing-v2')/f'{profile}-full-startup.json'
        base=load_config(path)
        for case in ('continuous','blank','cessation'):
            # Main and cessation preserve source-anchored inference. Blank is
            # explicitly source-free, as in earlier blank diagnostics.
            session=FullFovSession(base.session.instrument,base.model,FullFovSettings(),
                None if case=='blank' else base.session.source_position,base.session.initial_position_yx)
            for noise in (0.,.01,.05):
                name=f'{profile}-{case}-noise-{noise:g}'
                config=FullFovRunConfig(session,replace(base.source,signal_scale=0. if case=="blank" else base.source.signal_scale),
                    base.wind,simulated_noise_std=noise,cessation_seconds=180. if case=="cessation" else None)
                output=root/'configs'/f'{name}.json';write(output,config.to_dict())
                jobs.append({'name':name,'profile':profile,'case':case,'noise':noise,'config':str(output.relative_to(root)),
                    'sha256':sha(output),'parent_config':str(path),'parent_sha256':sha(path)})
    write(target,{'artifact_type':'full_fov_mechanism_declaration_v1','producing_revision':revision(),
        'jobs':jobs,'horizon_seconds':360.,'scope':'18 descriptive mechanism runs, seed 0; not a superiority suite.',
        'source_controls':'Continuous source unchanged. Blank has zero scale and source-free inference. Cessation abruptly removes signal at 180s while source-anchored inference remains unchanged; not a physical shutdown model.',
        'noise_controls':'Simulated residual std 0/.01/.05; assumed likelihood std .05 unchanged.',
        'staging':'Verify zero-noise execution first, then run .01/.05 without retuning.'})


_TRUTH_CACHE={}


@lru_cache(maxsize=6)
def truth_context(root,profile,case):
    config=FullFovRunConfig.from_dict(read(Path(root)/'configs'/f'{profile}-{case}-noise-0.json'))
    # Truth quadrature depends on the source and evaluation grid, not the
    # scanner profile, reading noise, or selected path. Reuse identical inputs.
    key=json.dumps([config.to_dict()['source'],config.cessation_seconds,config.duration_seconds,
                    config.model.grid_shape,config.evaluation_shape,config.evaluation_step_seconds],sort_keys=True)
    if key not in _TRUTH_CACHE:
        _TRUTH_CACHE[key]=TruthQuadrature(config.build_source(),0.,config.duration_seconds,config.model.grid_shape,
                                        config.evaluation_shape,config.evaluation_step_seconds)
    return _TRUTH_CACHE[key]


def execute(root,job,declaration_hash):
    root=Path(root);path=root/'runs'/f"{job['name']}.json"
    assert sha(root/job['config'])==job['sha256']
    assert sha(job['parent_config'])==job['parent_sha256']
    if path.exists():
        saved=read(path)
        if saved['declaration_sha256']!=declaration_hash or saved['job']!=job:raise ValueError('Changed execution inputs')
        return job['name'],'reused'
    config=FullFovRunConfig.from_dict(read(root/job['config']))
    result=run_full_fov(config,truth=truth_context(str(root),job['profile'],job['case']))
    result.update(job=job,declaration_sha256=declaration_hash,producing_revision=revision())
    write(path,result)
    return job['name'],{'full':result['completed_full_scans'],'focus':sum(s['plan']['mode']=='FOCUS' for s in result['scans']),
                        'J/T':result['J_over_T']}


def verify(root,*,zero_only=False):
    declaration=read(root/'declaration.json');rows=[]
    verified_zero={}
    prior=root/'zero-verification.json'
    if not zero_only and prior.exists():
        saved=read(prior)
        if saved['status']=='passed' and saved['declaration_sha256']==sha(root/'declaration.json'):
            verified_zero={r['name']:r for r in saved['records']}
    for job in declaration['jobs']:
        if zero_only and job['noise']!=0:continue
        path=root/'runs'/f"{job['name']}.json";record=read(path)
        assert record['job']==job and record['declaration_sha256']==sha(root/'declaration.json')
        config=FullFovRunConfig.from_dict(read(root/job['config']))
        assert config.digest==record['configuration_hash']
        assert sha(root/job['config'])==job['sha256'] and sha(job['parent_config'])==job['parent_sha256']
        cached=verified_zero.get(job['name'])
        if cached is not None and cached['record_sha256']==sha(path):
            rows.append(cached)
            print('verified unchanged zero-noise evidence',job['name'],flush=True)
            continue
        c=verify_replay(record,truth=truth_context(str(root),job['profile'],job['case']))
        scans=record['scans'];first=scans[0]['plan']
        assert first['mode']=='FULL' and len(first['samples'])==np.prod(config.capabilities.lattice_shape)
        first_completed=scans[0]['receipt']['received_at_seconds']
        assert all(s['plan']['started_at_seconds']>=first_completed-1e-9 for s in scans if s['plan']['mode']=='FOCUS')
        clean=np.concatenate([s['clean_readings'] for s in scans]);values=np.concatenate([s['readings'] for s in scans])
        noise=values-clean
        if job['noise']==0:assert np.max(np.abs(noise))==0
        for s in scans:
            p=s['plan'];assert p['mode'] in ('FULL','FOCUS')
            if p['mode']=='FULL':assert len(p['samples'])==np.prod(config.capabilities.lattice_shape)
            else:assert p['completed_at_seconds']-p['started_at_seconds']<=config.session.settings.focus_seconds+1e-8
            assert not {'risk_candidates','target_mass','service_status'} & p.keys()
        rows.append({'name':job['name'],'first_full_seconds':first_completed,'full_scans':c.completed_full_scans,
            'focus_scans':sum(s['plan']['mode']=='FOCUS' for s in scans),'J_over_T':record['J_over_T'],
            'signed_index_error_over_T':record['signed_index_error_integral']/config.duration_seconds,
            'spent_seconds':record['spent_seconds'],'noise_rms':float(np.sqrt(np.mean(noise**2))),
            'maximum_added_noise':float(np.max(np.abs(noise))),
            'focus_after_cessation':sum(s['plan']['mode']=='FOCUS' and s['plan']['started_at_seconds']>=180 for s in scans)
                if job['case']=='cessation' else None,
            'termination':record['termination'],'record_sha256':sha(path)})
        print('verified public-call and causal-loss replay',job['name'],flush=True)
    write(root/('zero-verification.json' if zero_only else 'verification.json'),
          {'status':'passed','producing_revision':revision(),'declaration_sha256':sha(root/'declaration.json'),'records':rows})
    print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/lean-full-fov'))
    p.add_argument('--phase',choices=('declare','run','verify'),required=True)
    p.add_argument('--noise',type=float);p.add_argument('--workers',type=int,default=2);p.add_argument('--zero-only',action='store_true')
    args=p.parse_args()
    if args.phase=='declare':declare(args.output)
    elif args.phase=='verify':verify(args.output,zero_only=args.zero_only)
    else:
        declaration=read(args.output/'declaration.json')
        jobs=[j for j in declaration['jobs'] if args.noise is None or j['noise']==args.noise]
        if any(j['noise']!=0 for j in jobs):
            if read(args.output/'zero-verification.json')['status']!='passed':raise ValueError('Verify zero-noise stage first')
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(execute,str(args.output),j,sha(args.output/'declaration.json')) for j in jobs]
            for f in as_completed(futures):print(f.result(),flush=True)
