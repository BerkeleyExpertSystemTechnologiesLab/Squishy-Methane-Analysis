"""Detached bound, refinement, reading and presentation checks for the v5 study."""

import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from dataclasses import replace
from functools import lru_cache
import importlib.util
from pathlib import Path
import math
import numpy as np

from plume_focus.controller import SensingController
from plume_focus.sensing_cli_v1 import config_from_dict
from plume_focus.sensing_snapshot import DecisionSnapshot
from plume_focus.sensing_execution_v5 import substitute_snapshot_policy,with_threshold_policy
from plume_focus.sensing_threshold_evaluation import apply_event,decode_plan,evaluate_execution
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_counterfactual import TruthQuadrature
from plume_focus.signal_belief import grid_positions,interpolation_stencil,observation_weights
from plume_focus.signal_distribution import PredictiveDistribution
from plume_focus.sensing_risk import linear_transition


@lru_cache(maxsize=1)
def runner():
    spec=importlib.util.spec_from_file_location('threshold_runner',Path(__file__).with_name('run-threshold-bound-study.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fixture(root,name,policy='upper',mode='short'):
    root=Path(root);directory=root/'fixtures'/name/policy;r=runner()
    config=config_from_dict(r.read(directory/'configuration.json'))
    snapshot=DecisionSnapshot.from_dict(r.read(directory/('snapshot.json' if mode=='short' else 'initial-snapshot.json')))
    return config,snapshot


def average(values,mask):
    return float(np.mean(values[mask])) if np.any(mask) else None


def bound_checks(root,name):
    r=runner();config,snapshot=fixture(root,name);controller=snapshot.fork_controller(SensingController)
    now=controller._last_available;shape=config.capabilities.lattice_shape
    points=grid_positions(shape);source=config.source.build();rows=[]
    visit=controller.visit_times.ravel();age=now-visit
    for horizon in (0.,1.,3.,6.,12.):
        distribution=controller.predictive_distribution(now+horizon,shape)
        mu=distribution.mean.ravel();upper=distribution.quantile(.95).ravel()
        truth=source.signal_at(np.full(len(points),now+horizon),points)
        signal=truth>.15;eligible=upper>=.15
        grad=np.hypot(*np.gradient(truth.reshape(shape),1/shape[0],1/shape[1])).ravel()
        strata={'all':np.ones(len(points),bool),'signal':signal,'background':~signal,
                'unvisited':visit<0,'age_under_5':(visit>=0)&(age<5),
                'age_5_to_30':(visit>=0)&(age>=5)&(age<30),'age_30_plus':(visit>=0)&(age>=30),
                'gradient_at_least_0.5':grad>=.5,'gradient_under_0.5':grad<.5}
        excess=np.maximum(truth-.15,0);total=excess.sum()
        rows.append({'horizon':horizon,'gate_at_this_time':bool(mu.max()>.15),'truth_signal_present':bool(np.any(signal)),
            'max_mean':float(mu.max()),'max_truth':float(truth.max()),'eligible_fraction':float(eligible.mean()),
            'missed_excess_fraction':float(excess[~eligible].sum()/total) if total>0 else None,
            'eligible_background_fraction':float(np.mean(eligible&~signal)),
            'strata':{key:{'cells':int(mask.sum()),'upper_exceedance_fraction':average(truth>upper,mask),
                'mean_positive_bound_error':average(np.maximum(truth-upper,0),mask),
                'mean_squared_error':average((truth-mu)**2,mask),
                'predicted_variance':average(distribution.variance.ravel(),mask),
                'missed_signal_fraction':average(signal&~eligible,mask)} for key,mask in strata.items()}})
    sensitivity=[]
    for q in (.90,.95,.99):
        for threshold in (.10,.15,.20):
            changed=with_threshold_policy(config,upper_quantile=q,reportable_threshold=threshold)
            detached=substitute_snapshot_policy(snapshot,changed.session).fork_controller(SensingController)
            plan=detached.plan_next(now,detached._last_pose,ScanRequestV1(12.),wind=changed.wind)
            truth=source.signal_at(np.full(len(points),now+6),points);signal=truth>threshold
            eligible=np.zeros(len(points),bool);eligible[list(plan.eligible_cells)]=True
            excess=np.maximum(truth-threshold,0);total=excess.sum()
            sensitivity.append({'quantile':q,'threshold':threshold,'gate':plan.signal_gate_enabled,
                'eligible_fraction':float(eligible.mean()),'missed_excess_fraction':float(excess[~eligible].sum()/total) if total>0 else None,
                'eligible_background_fraction':float(np.mean(eligible&~signal)),'status':plan.status,
                'duration':plan.duration_seconds,'unvisited_eligible':len(plan.unvisited_eligible_cells)})
    timing=[]
    baseline=controller.predictive_distribution(now+6,shape).quantile(.95).ravel()>=.15
    for at in (now,now+12):
        mask=controller.predictive_distribution(at,shape).quantile(.95).ravel()>=.15
        union=np.sum(mask|baseline)
        timing.append({'forecast_at':at,'eligible_jaccard_to_midpoint':float(np.sum(mask&baseline)/union) if union else 1.,
                       'eligible_fraction':float(mask.mean())})
    assert controller.decision_snapshot().digest==snapshot.digest
    result={'artifact_type':'threshold_bound_reliability_v1','snapshot_hash':snapshot.digest,'rows':rows,
        'sensitivity':sensitivity,'forecast_timing_sensitivity':timing,'producing_revision':r.revision(),
        'interpretation':'Fixed-truth descriptive calibration, not independent pixel trials or a whole-scene guarantee. Sensitivity does not tune the executed policy.'}
    r.write(Path(root)/'bounds'/f'{name}.json',result);return name


def refine(root,name):
    r=runner();config,snapshot=fixture(root,name)
    path=Path(root)/'runs'/name/'upper'/'short-seed-0.json'
    if not path.exists():return name,'pending'
    record=r.read(path);out=Path(root)/'refinement'/f'{name}.json'
    if out.exists():
        saved=r.read(out)
        if saved['execution_sha256']!=r.sha(path):raise ValueError('Refinement input changed')
        return name,saved['status']
    previous=record['J'];shape=(120,160);step=.05;checks=[];accepted=None
    for level in range(2):
        finer=tuple(2*n for n in shape);fine_step=step/2
        truth=TruthQuadrature(config.source.build(),record['start'],12.,config.model.grid_shape,finer,fine_step)
        result=evaluate_execution(snapshot,record,truth);difference=abs(result['J']-previous)
        tolerance=max(1e-10,.01*abs(result['J']))
        checks.append({'coarse_shape':shape,'coarse_step':step,'fine_shape':finer,'fine_step':fine_step,
                       'coarse_J':previous,'fine_J':result['J'],'difference':difference,'tolerance':tolerance})
        if difference<=tolerance:accepted=level;break
        previous=result['J'];shape=finer;step=fine_step
    result={'artifact_type':'threshold_evaluation_refinement_v1','execution_sha256':r.sha(path),
            'accepted_level':accepted,'checks':checks,'status':'converged' if accepted is not None else 'numerically_unresolved',
            'producing_revision':r.revision()}
    r.write(out,result);return name,result['status']


def contrast_refinement(root,name,mode='short'):
    r=runner();policies=('upper','mean','risk') if 'motion-risk' in name else ('upper','mean','risk','corridor','age')
    paths={p:Path(root)/'runs'/name/p/f'{mode}-seed-0.json' for p in policies}
    if not all(p.exists() for p in paths.values()):return name,'pending'
    out=Path(root)/'contrast-refinement'/name/f'{mode}.json';hashes={p:r.sha(path) for p,path in paths.items()}
    if out.exists():
        if r.read(out)['execution_hashes']!=hashes:raise ValueError('Contrast refinement input changed')
        return name,'reused'
    configs={p:fixture(root,name,p,mode) for p in policies};records={p:r.read(path) for p,path in paths.items()}
    previous={p:records[p]['J'] for p in policies};checks=[];contrasts={};shape=(120,160);step=.05
    reference=records['upper'];base=configs['upper'][0]
    for level in range(2):
        shape=tuple(2*n for n in shape);step/=2
        truth=TruthQuadrature(base.source.build(),reference['start'],reference['horizon'],base.model.grid_shape,shape,step)
        current={p:evaluate_execution(configs[p][1],records[p],truth)['J'] for p in policies}
        contrasts={}
        for p in policies[1:]:
            old=previous[p]-previous['upper'];new=current[p]-current['upper']
            change=abs(new-old);tolerance=max(1e-10,.01*abs(current[p]))
            contrasts[p]={'coarse_difference':old,'fine_difference':new,'change':change,'tolerance':tolerance,
                          'margin_if_converged':change if change<=tolerance else None}
        checks.append({'shape':shape,'step':step,'losses':current,'contrasts':contrasts})
        if all(c['margin_if_converged'] is not None for c in contrasts.values()):break
        previous=current
    result={'artifact_type':'threshold_comparison_refinement_v1','execution_hashes':hashes,'checks':checks,
            'contrasts':contrasts,'producing_revision':r.revision(),
            'scope':'Seed-0 saved-delivery numerical contrasts; future-noise sampling uncertainty remains separate.'}
    r.write(out,result);return name,'complete'


@lru_cache(maxsize=64)
def point_operators(model,velocity,source,transport,positions,elapsed):
    ids,coefs=interpolation_stencil(np.asarray(positions),model.grid_shape)
    n=math.prod(model.grid_shape);vectors=[];offsets=[];noises=[]
    for j,t in enumerate(elapsed):
        h=np.zeros(n);np.add.at(h,ids[j],coefs[j])
        a,b,q=linear_transition(model,velocity,source,t,transport)
        vectors.append(a.T@h);offsets.append(h@b);noises.append(h@q@h)
    arrays=tuple(np.asarray(x) for x in (vectors,offsets,noises))
    for a in arrays:a.setflags(write=False)
    return arrays


def point_forecasts(controller,plan):
    """Exact per-schedule marginal projection without assimilating fake readings."""
    belief=controller._belief;components=getattr(belief,'components',(belief,))
    offsets=getattr(belief,'offsets',np.zeros((1,2)));weights=getattr(belief,'weights',np.ones(1))
    means=np.zeros((len(components),len(plan.samples)));variances=np.zeros_like(means)
    positions=tuple(s.position_yx for s in plan.samples)
    for k,(state,offset) in enumerate(zip(components,offsets)):
        velocity=tuple(np.asarray(controller._wind.velocity_yx)+offset)
        elapsed=tuple(round(s.acquired_at_seconds-state.time_seconds,12) for s in plan.samples)
        vectors,b,q=point_operators(state.config,velocity,controller.source_position,belief.transport_id,positions,elapsed)
        means[k]=vectors@state.mean+b
        variances[k]=np.maximum(0,np.einsum('ij,ij->i',vectors@state.covariance,vectors)+q)
    distribution=PredictiveDistribution(plan.started_at_seconds,weights,means,variances)
    upper=distribution.quantile(getattr(plan,'upper_quantile',.95))
    return [{'cell':int(np.minimum((np.asarray(s.position_yx)*plan.lattice_shape).astype(int),np.asarray(plan.lattice_shape)-1)@[plan.lattice_shape[1],1]),
             'at_seconds':s.acquired_at_seconds,'mean':float(distribution.mean[j]),'upper':float(upper[j])}
            for j,s in enumerate(plan.samples)]


def reading_diagnostics(root,name,policy,mode,seed):
    r=runner();path=Path(root)/'runs'/name/policy/f'{mode}-seed-{seed}.json'
    if not path.exists():return name,policy,mode,seed,'pending'
    out=Path(root)/'reading-diagnostics'/name/policy/f'{mode}-seed-{seed}.json'
    if out.exists():
        if r.read(out)['execution_sha256']!=r.sha(path):raise ValueError('Reading diagnostic input changed')
        return name,policy,mode,seed,'reused'
    config,snapshot=fixture(root,name,policy,mode);controller=snapshot.fork_controller(SensingController)
    record=r.read(path);forecasts=[];residuals=[]
    for event in record['events']:
        if event['operation']=='plan_next':
            apply_event(controller,event)
            plan=decode_plan(event['plan'])
            if plan.samples and hasattr(plan,'signal_gate_enabled'):
                forecasts.append({'scan_id':plan.scan_id,'points':point_forecasts(controller,plan),
                                  'completion_upper_signal':plan.completion_upper_signal})
            continue
        if event['operation']=='ingest_readings':
            # Delivered predictive reading residual: includes detector noise and
            # declared footprint. Exposure is simulated; estimator remains midpoint.
            for payload in event['readings']:
                from plume_focus.sensing_contracts import SensorReading
                reading=SensorReading.from_dict(payload);belief=controller._belief
                h=observation_weights(config.model.grid_shape,reading.position_yx,reading.footprint)
                means=[];variances=[]
                for state,offset in zip(getattr(belief,'components',(belief,)),getattr(belief,'offsets',np.zeros((1,2)))):
                    a,b,q=linear_transition(state.config,tuple(np.asarray(controller._wind.velocity_yx)+offset),controller.source_position,
                        round(reading.acquired_at_seconds-state.time_seconds,12),belief.transport_id)
                    v=a.T@h;means.append(config.model.offset+config.model.gain*(v@state.mean+h@b))
                    variances.append(config.model.gain**2*(v@state.covariance@v+h@q@h)+config.model.sensor_noise_std**2)
                d=PredictiveDistribution(reading.acquired_at_seconds,getattr(belief,'weights',np.ones(1)),np.array(means)[:,None],np.array(variances)[:,None])
                residuals.append({'reading_id':reading.reading_id,'acquired_at_seconds':reading.acquired_at_seconds,
                    'available_at_seconds':event['at_seconds'],'value':reading.value,'mean':float(d.mean[0]),
                    'variance':float(d.variance[0]),'standardized_squared_residual':float((reading.value-d.mean[0])**2/d.variance[0])})
                controller.ingest_readings([reading],available_at_seconds=event['at_seconds'])
            continue
        apply_event(controller,event)
    r.write(out,{'artifact_type':'threshold_planned_bounds_and_reading_residuals_v1','execution_sha256':r.sha(path),
        'forecasts':forecasts,'reading_residuals':residuals,'producing_revision':r.revision(),
        'scope':'Planned latent bounds use decision evidence only; reading residuals use earlier delivered evidence. No trace mutations.'})
    return name,policy,mode,seed,'complete'


def coverage_diagnostics(root,name,policy,mode,seed):
    r=runner();path=Path(root)/'runs'/name/policy/f'{mode}-seed-{seed}.json'
    if not path.exists():return name,policy,mode,seed,'pending'
    out=Path(root)/'coverage'/name/policy/f'{mode}-seed-{seed}.json'
    if out.exists():
        if r.read(out)['execution_sha256']!=r.sha(path):raise ValueError('Coverage input changed')
        return name,policy,mode,seed,'reused'
    config=config_from_dict(r.read(Path(root)/'fixtures'/name/policy/'configuration.json'))
    source=config.source.build();points=grid_positions(config.capabilities.lattice_shape);record=r.read(path);rows=[]
    for scan in record['scans']:
        plan=scan['plan'];at=scan['receipt']['completed_at_seconds']
        values=source.signal_at(np.full(len(points),at),points);excess=np.maximum(values-.15,0);support=values>.15
        swept=np.zeros(len(points),bool);swept[plan['swept_cells']]=True
        eligible=np.zeros(len(points),bool);eligible[plan.get('eligible_cells',[])]=True
        total=excess.sum();support_count=int(support.sum())
        rows.append({'scan_id':plan['scan_id'],'at_seconds':at,'requested_fraction':len(plan['requested_cells'])/len(points),
            'swept_fraction':float(swept.mean()),'sampled_cells':len(plan['samples']),
            'true_excess_covered_fraction':float(excess[swept].sum()/total) if total>0 else None,
            'true_support_covered_fraction':float(np.sum(swept&support)/support_count) if support_count else None,
            'predicted_excess_covered_fraction':plan['predicted_mass_recall'],
            'eligible_omitted_true_excess':float(excess[eligible&~swept].sum()/len(points)) if 'eligible_cells' in plan else None,
            'eligible_region_missed_true_excess':float(excess[~eligible].sum()/len(points)) if 'eligible_cells' in plan else None})
    r.write(out,{'artifact_type':'threshold_execution_coverage_v1','execution_sha256':r.sha(path),'rows':rows,
        'producing_revision':r.revision(),'scope':'Clean physical-lattice field at actual completion, a coverage diagnostic separate from causal reconstruction loss. Blank truth coverage is null.'})
    return name,policy,mode,seed,'complete'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--phase',choices=('bounds','refine','readings','contrasts','coverage'),required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--fixture');p.add_argument('--policy');p.add_argument('--mode');p.add_argument('--seed',type=int)
    args=p.parse_args();declaration=runner().read(args.output/'declaration.json')
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.phase in ('readings','coverage'):
            jobs=[j for j in declaration['jobs'] if (args.phase=='coverage' or j['policy'] in ('upper','mean'))
                  and (not args.fixture or j['fixture']==args.fixture) and (not args.policy or j['policy']==args.policy)
                  and (not args.mode or j['mode']==args.mode) and (args.seed is None or j['seed']==args.seed)]
            function=reading_diagnostics if args.phase=='readings' else coverage_diagnostics
            futures=[pool.submit(function,str(args.output),j['fixture'],j['policy'],j['mode'],j['seed']) for j in jobs]
        else:
            names=[n for n in declaration['fixtures'] if not args.fixture or n==args.fixture]
            function={'bounds':bound_checks,'refine':refine,'contrasts':contrast_refinement}[args.phase]
            if args.phase=='contrasts' and args.mode:
                names=[n for n in names if args.mode!='full' or 'motion-risk' not in n]
                futures=[pool.submit(function,str(args.output),n,args.mode) for n in names]
            else:
                futures=[pool.submit(function,str(args.output),n) for n in names]
        for future in as_completed(futures):print(future.result(),flush=True)
