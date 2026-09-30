"""Predeclared 348-record allocation study; immutable inputs and resumable output."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess

from plume_focus.controller import SensingController
from plume_focus.sensing_cli_v1 import config_from_dict, write_json as _write
from plume_focus.sensing_snapshot import DecisionSnapshot
from plume_focus.sensing_execution_v5 import with_threshold_policy, with_transport_baseline, substitute_snapshot_policy
from plume_focus.sensing_session_v5 import UPPER_POLICY, MEAN_POLICY
from plume_focus.sensing_counterfactual import TruthQuadrature
from plume_focus.sensing_threshold_evaluation import run_policy


POLICIES=('upper','mean','risk','corridor','age')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def revision():return subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
def write(path,payload):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.partial');_write(tmp,payload);tmp.replace(path)
def immutable(path,payload):
    if Path(path).exists():
        if read(path)!=json.loads(json.dumps(payload)):raise ValueError(f'Frozen input changed: {path}')
    else:write(path,payload)


def bind(base,policy):
    if policy in ('upper','mean'):return with_threshold_policy(base,UPPER_POLICY if policy=='upper' else MEAN_POLICY)
    if policy=='risk':return base
    return with_transport_baseline(base,'static_corridor_segment_v2' if policy=='corridor' else 'age_segment_v2')


def declare(root,historical):
    root,historical=Path(root),Path(historical);old=read(historical/'declaration.json')
    jobs=[];fixtures={}
    for name,ref in old['fixtures'].items():
        if not ref['available']:raise ValueError(f'Declared fixture unavailable: {name}')
        directory=historical/'fixtures'/name
        cp=directory/'zero_inflow_v1-configuration.json';sp=directory/'zero_inflow_v1-snapshot.json'
        base=config_from_dict(read(cp));snapshot=DecisionSnapshot.from_dict(read(sp))
        fixtures[name]={'configuration':str(cp.resolve()),'configuration_sha256':sha(cp),
            'snapshot':str(sp.resolve()),'snapshot_sha256':sha(sp),'snapshot_identity':snapshot.digest,
            'historical_trace':ref['trace'],'historical_trace_sha256':ref['trace_sha256']}
        mixture=hasattr(base.session,'motion')
        for policy in POLICIES[:3] if mixture else POLICIES:
            config=bind(base,policy);child=substitute_snapshot_policy(snapshot,config.session)
            out=root/'fixtures'/name/policy
            immutable(out/'configuration.json',config.to_dict());immutable(out/'snapshot.json',child.to_dict())
            for seed in range(8):jobs.append({'fixture':name,'policy':policy,'seed':seed,'mode':'short',
                'configuration':str((out/'configuration.json').relative_to(root)),
                'snapshot':str((out/'snapshot.json').relative_to(root)),
                'parent_snapshot_identity':snapshot.digest,'horizon':12.})
            if not mixture:
                full=SensingController.from_session(config.session).decision_snapshot()
                immutable(out/'initial-snapshot.json',full.to_dict())
                for seed in (0,1):jobs.append({'fixture':name,'policy':policy,'seed':seed,'mode':'full',
                    'configuration':str((out/'configuration.json').relative_to(root)),
                    'snapshot':str((out/'initial-snapshot.json').relative_to(root)),
                    'parent_snapshot_identity':None,'horizon':75.})
    for job in jobs:
        job['configuration_sha256']=sha(root/job['configuration']);job['snapshot_sha256']=sha(root/job['snapshot'])
        job['id']=f"{job['fixture']}/{job['policy']}/{job['mode']}-seed-{job['seed']}"
    assert len(jobs)==348
    payload={'artifact_type':'threshold_bound_study_declaration_v1','starting_revision':'5bb58ee',
        'declaring_revision':revision(),'historical_declaration':str((historical/'declaration.json').resolve()),
        'historical_declaration_sha256':sha(historical/'declaration.json'),'fixtures':fixtures,'jobs':jobs,
        'defaults':{'threshold':.15,'upper_quantile':.95,'transport':'zero_inflow_v1','evaluation_shape':[120,160],
                    'evaluation_step':.05,'sensitivity_quantiles':[.90,.95,.99],'sensitivity_thresholds':[.10,.15,.20],
                    'forecast_horizons':[0,1,3,6,12]},'scope':'Fixed estimator; no calibration fitting or policy freeze.'}
    # A resumed declaration retains its original producing revision.
    if (root/'declaration.json').exists():payload['declaring_revision']=read(root/'declaration.json')['declaring_revision']
    immutable(root/'declaration.json',payload)
    print('declared',len(jobs),'records',flush=True)


@lru_cache(maxsize=8)
def context(root,fixture,policy,mode):
    root=Path(root);directory=root/'fixtures'/fixture/policy
    config=config_from_dict(read(directory/'configuration.json'))
    snapshot=DecisionSnapshot.from_dict(read(directory/('snapshot.json' if mode=='short' else 'initial-snapshot.json')))
    truth=truth_context(root,fixture,mode)
    return config,snapshot,truth


@lru_cache(maxsize=4)
def truth_context(root,fixture,mode):
    directory=Path(root)/'fixtures'/fixture/'risk';config=config_from_dict(read(directory/'configuration.json'))
    snapshot=DecisionSnapshot.from_dict(read(directory/('snapshot.json' if mode=='short' else 'initial-snapshot.json')))
    return TruthQuadrature(config.source.build(),snapshot.to_dict()['last_available'],12. if mode=='short' else 75.,
                           config.model.grid_shape,config.evaluation.shape,config.evaluation.step_seconds)


def execute(root,job,declaration_sha):
    root=Path(root)
    for key in ('configuration','snapshot'):
        if sha(root/job[key])!=job[key+'_sha256']:raise ValueError('Declared input changed')
    target=root/'runs'/f"{job['id']}.json"
    if target.exists():
        saved=read(target)
        if saved['declaration_sha256']!=declaration_sha or saved['job']!=job:raise ValueError('Resume lineage differs')
        return job['id'],'reused'
    config,snapshot,truth=context(str(root),job['fixture'],job['policy'],job['mode'])
    result=run_policy(snapshot,config,horizon=job['horizon'],seed=job['seed'],full_session=job['mode']=='full',truth=truth)
    result.update(job=job,declaration_sha256=declaration_sha,producing_revision=revision())
    write(target,result)
    return job['id'],round(result['J_over_T'],8)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('declare','run'),required=True)
    p.add_argument('--output',type=Path,default=Path('outputs/threshold-bound-v5'))
    p.add_argument('--historical',type=Path,default=Path('outputs/scan-value-diagnosis'))
    p.add_argument('--workers',type=int,default=4);p.add_argument('--fixture');p.add_argument('--policy')
    p.add_argument('--mode',choices=('short','full'));p.add_argument('--mixture',action='store_true')
    p.add_argument('--gaussian',action='store_true')
    args=p.parse_args()
    if args.phase=='declare':declare(args.output,args.historical)
    else:
        declaration=read(args.output/'declaration.json');digest=sha(args.output/'declaration.json')
        jobs=[j for j in declaration['jobs'] if (not args.fixture or j['fixture']==args.fixture)
              and (not args.policy or j['policy']==args.policy) and (not args.mode or j['mode']==args.mode)
              and (not args.mixture or 'motion-risk' in j['fixture'])
              and (not args.gaussian or 'motion-risk' not in j['fixture'])]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(execute,str(args.output),j,digest) for j in jobs]):
                print(*future.result(),flush=True)
