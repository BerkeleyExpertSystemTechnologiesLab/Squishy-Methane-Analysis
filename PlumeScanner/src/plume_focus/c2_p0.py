"""P0 fixed-pool, single-action development diagnostics; no production policy.

Truth only enters evaluator/source construction, never candidate/risk selection.
"""
from copy import deepcopy
from dataclasses import replace,fields
import math
import numpy as np
from plume_focus.c2_r2_mechanics import compile_directed_columns,solve_monotone_columns,route_record,verify_route_record,audit_directed_plan
from plume_focus.c2_r2_dynamic import R2Counterfactual,ready_payload,R2TruthGrid,integration_grid
from plume_focus.calibration_evaluation import mean_forecast
from plume_focus.evidence_gated_execution import EvidenceGatedRunConfig
from plume_focus.evidence_gated_loop import EvidenceGatedScanner
from plume_focus.full_fov_execution import replay_event
from plume_focus.physical_scan import PhysicalPlan,compile_physical_scan
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.sensing_session import configuration_hash
from plume_focus.sensing_contracts import SensorReading,ExecutionReceipt

TYPES=('B','M','L_opt','R_opt','L_down','L_up','R_down','R_up')


def make_worlds(base):
    """24 independent source/phase/dynamic instances, not noise replicates."""
    worlds=[]
    for condition,(speed,mismatch) in enumerate(((.5,False),(.5,True),(2.,False),(2.,True))):
        for replicate in range(6):
            i=6*condition+replicate; seed=1826000+i; rng=np.random.default_rng(seed)
            c=deepcopy(base); y,x=float(rng.uniform(.32,.68)),float(rng.uniform(.10,.30))
            vy,vx=float(rng.uniform(-.05,.05)),float(rng.uniform(.90,1.14)); scale=float(rng.uniform(.85,1.20))
            s=c['source']; s.update(source=[y*219,x*319],fps=12*speed,
                wind_velocity=[vy+(0.35 if mismatch else 0.),vx*(1.8 if mismatch else 1.)],
                lateral_phase=float(rng.uniform(0,2*np.pi)),amplitude_phase=float(rng.uniform(0,2*np.pi)),
                shimmer_phase=float(rng.uniform(0,2*np.pi)),signal_scale=scale,end_time_seconds=60.)
            c['session']['source_position']=[y,x]
            c['session']['half_life_seconds']=6/speed
            c['session']['model']['source_rate']=base['session']['model']['source_rate']*speed*scale
            c['wind']['velocity_yx']=[vy*12*speed/219,vx*12*speed/319]
            c['cessation_seconds']=float(rng.uniform(35.5,36.5)) if mismatch else None
            c['duration_seconds']=60.; c['evaluation_shape']=[32,48]; c['evaluation_step_seconds']=.1
            c['simulated_noise_std']=0.; c['seed']=seed
            worlds.append(dict(id=f'w{i:02d}',index=i,condition=condition,
                condition_name=('slow' if speed==.5 else 'fast')+'_'+('mismatch' if mismatch else 'nominal'),
                replicate=replicate,world_seed=seed,future_noise_seeds=[2981000+i*10+j for j in range(4)],
                truth_time_scale=speed,nominal_pixel_velocity=[vy,vx],config=c))
    return worlds


def kw_from(c,at,scan_id):
    last=c._last_detector_reading
    return dict(scan_id=scan_id,started_at_seconds=at,current_position=c._last_pose,
        last_reading_at_seconds=last.acquired_at_seconds if last else None,
        last_integration_seconds=last.integration_seconds if last else None)


def optimum_directions(mask,p,order,kw):
    """One fixed order, same sufficient two-orientation physical DP as R2."""
    shape=np.asarray(mask).shape; spans={c:(int(np.flatnonzero(np.asarray(mask)[:,c])[0]),int(np.flatnonzero(np.asarray(mask)[:,c])[-1])) for c in order}
    q=1/p.acquisition.cadence_hz; m=p.mechanics; E=p.acquisition.integration_seconds
    prior=kw.get('last_integration_seconds') or E
    states=[(kw['started_at_seconds'],kw['current_position'],kw.get('last_reading_at_seconds'),prior,())]
    for c in order:
        nxt=[]; lo,hi=spans[c]; x=(c+.5)/shape[1]
        for down in (True,False):
            entry=(lo/shape[0] if down else (hi+1)/shape[0],x); end=((hi+1)/shape[0] if down else lo/shape[0],x)
            choices=[]
            for t,pose,last,exp,path in states:
                arrive=t+abs(x-pose[1])/m.move_horizontal_fov_per_second+abs(entry[0]-pose[0])/m.move_vertical_fov_per_second+m.settling_seconds_per_visit
                if last is not None: arrive=max(arrive,last+max(1/p.acquisition.detector_capacity_hz,(exp+E)/2)-q/2)
                finish=arrive+(hi-lo+1)*q; choices.append((finish,end,finish-q/2,E,path+(down,)))
            nxt.append(min(choices,key=lambda v:v[0]))
        states=nxt
    return min(states,key=lambda v:v[0])[-1]


def direction_of(p):
    sweeps=[s for s in p.segments if s.kind=='sweep']
    return tuple(int(s.start_yx[1]*p.lattice_shape[1]) for s in sweeps),tuple(bool(s.end_yx[0]>s.start_yx[0]) for s in sweeps)


def pool(mask,profile,kw,budget):
    """Frozen max8 types; first feasible exact physical hash wins dedup."""
    B=compile_physical_scan(mask,profile,**kw); order,down=direction_of(B)
    plans={'B':B}; specs={'B':(order,down)}; cols=tuple(int(c) for c in np.flatnonzero(np.asarray(mask).any(axis=0)))
    if not cols: return {},{},'empty_mask'
    mono=solve_monotone_columns(mask,profile,objective='FREE',**kw)
    plans['M']=B if B.duration_seconds<=mono.plan.duration_seconds+1e-9 else mono.plan
    specs['M']=direction_of(plans['M'])
    for name,cs in [('L',cols),('R',cols[::-1])]:
        for suffix,ds in [('opt',optimum_directions(mask,profile,cs,kw)),('down',(True,)*len(cols)),('up',(False,)*len(cols))]:
            t=name+'_'+suffix; plans[t]=compile_directed_columns(mask,profile,column_order=cs,directions_down=ds,**kw); specs[t]=(cs,ds)
    unique={}; aliases={}
    for t in TYPES:
        p=plans[t]
        if p.duration_seconds>budget+1e-9: aliases[t]=None; continue
        audit_directed_plan(p,mask,profile,last_reading_at_seconds=kw.get('last_reading_at_seconds'),last_integration_seconds=kw.get('last_integration_seconds'))
        key=configuration_hash(p.to_dict()); aliases[t]=key
        if key not in unique: unique[key]={'type':t,'record':route_record(p,mask,profile,kw,*specs[t])}
    if len(unique)>8: raise ValueError('Pool cap')
    return unique,aliases,None


def prepare_world(world,budget=8.):
    cfg=EvidenceGatedRunConfig.from_dict(world['config']); c=EvidenceGatedScanner(cfg.session); events=[]
    p=c.plan_next(0.,c._last_pose,ScanRequestV1(60.),wind=cfg.wind)
    events.append(dict(operation='plan_next',at_seconds=0.,position_yx=list(p.start_position_yx),remaining_seconds=60.,wind=cfg.wind.to_dict(),plan=p.to_dict()))
    if p.mode!='FULL': raise ValueError('P0 startup must use original full')
    ex=FakeScanner(cfg.build_source(),noise_std=0.,seed=world['world_seed'],gain=cfg.model.gain,offset=cfg.model.offset).execute(p)
    if ex.exhausted: raise ValueError('World does not cover prefix')
    for r in ex.readings:
        c.ingest_readings([r],available_at_seconds=r.available_at_seconds)
        events.append(dict(operation='ingest_readings',at_seconds=r.available_at_seconds,readings=[r.to_dict()]))
    c.finish_scan(ex.receipt); events.append(dict(operation='finish_scan',at_seconds=ex.receipt.received_at_seconds,receipt=ex.receipt.to_dict()))
    at=ex.receipt.received_at_seconds; saved=ready_payload(c,provenance={'P0_world':world['id'],'prefix_noise':0.,'new_development_world':True})
    clone=deepcopy(c); baseline=clone.plan_next(at,c._last_pose,ScanRequestV1(budget),wind=cfg.wind)
    obligations=dict(full_missing_count=len(clone.full_scan_missing),return_pending=clone._return_pending)
    eligible=baseline.mode=='FOCUS' and len(baseline.samples)>=cfg.session.settings.minimum_readings
    result=dict(world=world,saved=saved,prefix_events=events,original_plan=baseline.to_dict(),t0=at,T_e=at+12.,budget=budget,
        eligible=eligible,eligibility_reason='eligible_original_FOCUS' if eligible else baseline.mode+':'+baseline.reason,
        obligations_after_original_decision=obligations,original_ready_full_missing_count=len(c.full_scan_missing),original_ready_return_pending=c._return_pending)
    if eligible:
        mask=np.zeros(cfg.session.instrument.lattice_shape,bool); mask.ravel()[list(baseline.requested_cells)]=True
        kw=kw_from(c,at,baseline.scan_id); records,aliases,_=pool(mask,cfg.session.instrument,kw,budget)
        B=PhysicalPlan(**{f.name:getattr(baseline,f.name) for f in fields(PhysicalPlan)})
        if records[aliases['B']]['record']['plan_hash']!=configuration_hash(B.to_dict()): raise ValueError('Original selector physical parity')
        result.update(mask=mask.astype(int).tolist(),kwargs=kw,candidates=records,aliases=aliases)
    return result


def delayed_records(prepared,tau):
    """Frozen geometry/orientation; real delayed clock, detector, spent budget."""
    cfg=EvidenceGatedRunConfig.from_dict(prepared['world']['config']); plans={}
    for key,item in prepared['candidates'].items():
        rec=item['record']; kw=dict(rec['kwargs']); kw['current_position']=tuple(kw['current_position']); kw['started_at_seconds']=prepared['t0']+tau
        p=compile_directed_columns(prepared['mask'],cfg.session.instrument,column_order=rec['column_order'],directions_down=rec['directions_down'],**kw)
        lag=cfg.session.settings.delivery_delay_seconds+cfg.session.settings.completion_overrun_seconds
        if tau+p.duration_seconds+lag<=prepared['budget']+1e-9:
            plans[key]=route_record(p,prepared['mask'],cfg.session.instrument,kw,rec['column_order'],rec['directions_down'])
    return plans


def common_grid(prepared,records,step=.1):
    cfg=EvidenceGatedRunConfig.from_dict(prepared['world']['config']); events=[]
    for rec in records:
        p=PhysicalPlan.from_dict(rec['scan_plan']); events.extend(s.acquired_at_seconds+s.integration_seconds/2 for s in p.samples); events.append(p.completed_at_seconds)
    return integration_grid(prepared['t0'],12.,step,events)


def evaluate(saved,rec,times,durations,*,truth=None,source=None,noise_std=0.,seed=0,shape=(32,48),save=False):
    """Shared causal runtime; prior loss during planning; all observations fresh.

    truth=None is existing conditional covariance U; never receives future truth.
    Persistent evolution changes only at plan start/ingest/receipt, not queries.
    """
    from plume_focus.sensing_risk import projection_gram
    c=R2Counterfactual(saved); gram=projection_gram(c.model.grid_shape,tuple(shape)) if truth is None else truth.gram
    idx=0; values=[]; trace=[]; readings=[]; receipt=None
    def before(at):
        nonlocal idx
        while idx<len(times) and times[idx]<at-1e-10:
            t=float(times[idx])
            if truth is None:
                q=c._belief.clone(); q.advance(t,c._wind,c.source_position); value=float(np.sum(gram*q.covariance.T))
            else:
                mu=mean_forecast(c._belief,c._belief.mean,t,c._wind,c.source_position)
                value=float(mu@gram@mu-2*mu@truth.cross[idx]+truth.square[idx])
                if value < -1e-9: raise ValueError('Negative actual MSE')
            values.append(max(0.,value)); idx+=1
    if rec is not None:
        p=verify_route_record(rec); before(p.started_at_seconds)
        c._belief.advance(p.started_at_seconds,c._wind,c.source_position); c._last_available=p.started_at_seconds
        c.begin_record(rec)
        if truth is not None:
            ex=FakeScanner(source,noise_std=noise_std,seed=seed,gain=c.model.gain,offset=c.model.offset).execute(p)
            if ex.exhausted: raise ValueError('World exhausted')
            readings=list(ex.readings); receipt=ex.receipt
        else:
            trial=c._belief.clone()
            for i,s in enumerate(p.samples):
                trial.advance(s.acquired_at_seconds,c._wind,c.source_position)
                from plume_focus.signal_belief import observation_weights
                mean=c.model.offset+c.model.gain*float(observation_weights(c.model.grid_shape,s.position_yx,s.footprint)@trial.mean)
                r=SensorReading(p.scan_id,f'{p.scan_id}:{i}',s.position_yx,s.acquired_at_seconds,s.acquired_at_seconds+s.integration_seconds/2,mean,s.footprint,s.integration_seconds)
                trial.assimilate(r,c._wind,c.source_position); readings.append(r)
            receipt=ExecutionReceipt(p.scan_id,p.completed_at_seconds,p.completed_at_seconds,p.final_position_yx)
        for r in readings:
            before(r.available_at_seconds); c.ingest_readings([r],available_at_seconds=r.available_at_seconds)
            if save: trace.append(dict(available=r.available_at_seconds,acquired=r.acquired_at_seconds,value=r.value,position=r.position_yx))
        before(receipt.received_at_seconds); c.finish_scan(receipt)
    before(float('inf'))
    out={'value':float(np.dot(values,durations)/sum(durations)),'values':np.asarray(values),
        'n_readings':len(readings),'receipt':receipt.to_dict() if receipt else None,'final_pose':c._last_pose,
        'receipt_belief_time':c._belief.time_seconds,'trace':trace,'planning_observations':0}
    return out


def risk_select(prepared,records,times,weights):
    scores={key:evaluate(prepared['saved'],rec,times,weights)['value'] for key,rec in records.items()}
    if not scores: return None,scores
    best=min(scores.values()); B=prepared['aliases']['B']
    winner=B if B in scores and scores[B]<=best+1e-12 else next(k for k,v in scores.items() if v<=best+1e-12)
    return winner,scores
