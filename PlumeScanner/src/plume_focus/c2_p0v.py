"""P0-V frozen fixed-rule validation; no U, subset solver or new policy default."""
from copy import deepcopy
from dataclasses import fields
import numpy as np
from plume_focus.c2_p0 import kw_from,direction_of
from plume_focus.c2_r2_dynamic import ready_payload
from plume_focus.c2_r2_mechanics import compile_directed_columns,solve_monotone_columns,audit_directed_plan
from plume_focus.physical_scan import PhysicalPlan,compile_physical_scan
from plume_focus.evidence_gated_execution import EvidenceGatedRunConfig
from plume_focus.evidence_gated_loop import EvidenceGatedScanner
from plume_focus.calibration_evaluation import mean_forecast
from plume_focus.sensing_contracts import SensorReading
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.signal_belief import observation_weights

WRAPPER='p0v_fixed_complete_columns_budget_fallback_v1'
RULES=('B','M','F','C')


def make_pairs(base,seed_start=2136000,noise_start=4391000):
    """Same P0 draws/units; R changes only truth and known nominal source x."""
    pairs=[]
    for condition,(speed,mismatch) in enumerate(((.5,False),(.5,True),(2.,False),(2.,True))):
        for rep in range(6):
            i=6*condition+rep; seed=seed_start+i; rng=np.random.default_rng(seed)
            cfg=deepcopy(base); y,x=float(rng.uniform(.32,.68)),float(rng.uniform(.10,.30))
            vy,vx=float(rng.uniform(-.05,.05)),float(rng.uniform(.90,1.14)); scale=float(rng.uniform(.85,1.20))
            cfg['source'].update(source=[y*219,x*319],fps=12*speed,
                wind_velocity=[vy+(.35 if mismatch else 0.),vx*(1.8 if mismatch else 1.)],
                lateral_phase=float(rng.uniform(0,2*np.pi)),amplitude_phase=float(rng.uniform(0,2*np.pi)),
                shimmer_phase=float(rng.uniform(0,2*np.pi)),signal_scale=scale,end_time_seconds=60.)
            cfg['session']['source_position']=[y,x]; cfg['session']['half_life_seconds']=6/speed
            cfg['session']['model']['source_rate']=base['session']['model']['source_rate']*speed*scale
            cfg['wind']['velocity_yx']=[vy*12*speed/219,vx*12*speed/319]
            cfg['cessation_seconds']=float(rng.uniform(35.5,36.5)) if mismatch else None
            cfg.update(duration_seconds=60.,evaluation_shape=[32,48],evaluation_step_seconds=.1,simulated_noise_std=0.,seed=seed)
            right=deepcopy(cfg); right['source']['source'][1]=(1-x)*319; right['session']['source_position'][1]=1-x
            variants={side:dict(id=f'b{i:02d}_{side}',base_id=f'b{i:02d}',base_index=i,side=side,condition=condition,
                condition_name=('slow' if speed==.5 else 'fast')+'_'+('mismatch' if mismatch else 'nominal'),replicate=rep,
                world_seed=seed,future_noise_seeds=[noise_start+i*10+j for j in range(4)],config=config)
                for side,config in [('L',cfg),('R',right)]}
            pairs.append(dict(base_id=f'b{i:02d}',base_index=i,condition=condition,world_seed=seed,variants=variants))
    return pairs


def construct_rule(mask,profile,kw,rule,remaining):
    """Pure executable plan construction/physical audit and real budget fallback.

    kw carries actual t0+tau. remaining is original8s minus paid tau, never reset.
    No filesystem/records/whole candidate pool in this online-required component.
    """
    if rule not in RULES: raise ValueError('Unknown P0V rule')
    def audit(p):
        audit_directed_plan(p,mask,profile,last_reading_at_seconds=kw.get('last_reading_at_seconds'),last_integration_seconds=kw.get('last_integration_seconds'))
        return p
    def baseline(): return audit(compile_physical_scan(mask,profile,**kw))
    target={'B':'B','M':'monotone_FREE','F':'L_up','C':'R_up'}[rule]
    if rule in ('B','M'):
        B=baseline()
        if rule=='B':p=B
        else:
            m=audit(solve_monotone_columns(mask,profile,objective='FREE',**kw).plan)
            p=B if B.duration_seconds<=m.duration_seconds+1e-9 else m
    else:
        cs=tuple(int(c) for c in np.flatnonzero(np.asarray(mask).any(axis=0)))
        order=cs if rule=='F' else cs[::-1]
        p=audit(compile_directed_columns(mask,profile,column_order=order,directions_down=(False,)*len(cs),**kw))
    if p.duration_seconds<=remaining+1e-9:
        actual='B' if rule=='M' and p==B else target
        return p,dict(wrapper=WRAPPER,target_type=target,actual_type=actual,target_executed=True,fallback=False,rejected=False,
                      reason='target_feasible' if actual!= 'B' or rule=='B' else 'baseline_retained_on_mechanical_tie',target_duration=p.duration_seconds)
    # F/C genuinely compile/audit B only when needed. This branch is profiled too.
    B=baseline() if rule in ('F','C') else B
    if rule!='B' and B.duration_seconds<=remaining+1e-9:
        return B,dict(wrapper=WRAPPER,target_type=target,actual_type='B',target_executed=False,fallback=True,rejected=False,
                      reason='target_over_budget_B_feasible_after_same_paid_delay',target_duration=p.duration_seconds)
    return None,dict(wrapper=WRAPPER,target_type=target,actual_type='NONE',target_executed=False,fallback=rule!='B',rejected=True,
                     reason='target_and_B_over_budget_after_paid_delay' if rule!='B' else 'B_over_budget_after_paid_delay',target_duration=p.duration_seconds)


def prepare_variant(world):
    """Fresh complete FULL, fresh signals and posterior for EVERY L/R variant."""
    cfg=EvidenceGatedRunConfig.from_dict(world['config']); c=EvidenceGatedScanner(cfg.session); events=[]; innovations=[]
    full=c.plan_next(0.,c._last_pose,ScanRequestV1(60.),wind=cfg.wind)
    if full.mode!='FULL': raise ValueError('Original startup FULL required')
    events.append(dict(operation='plan_next',at_seconds=0.,position_yx=list(full.start_position_yx),remaining_seconds=60.,wind=cfg.wind.to_dict(),plan=full.to_dict()))
    ex=FakeScanner(cfg.build_source(),noise_std=0.,seed=world['world_seed'],gain=cfg.model.gain,offset=cfg.model.offset).execute(full)
    if ex.exhausted: raise ValueError('Source doesnot cover prefix')
    for r in ex.readings:
        mu=mean_forecast(c._belief,c._belief.mean,r.acquired_at_seconds,c._wind,c.source_position)
        predicted=c.model.offset+c.model.gain*float(observation_weights(c.model.grid_shape,r.position_yx,r.footprint)@mu)
        innovations.append(r.value-predicted)
        c.ingest_readings([r],available_at_seconds=r.available_at_seconds)
        events.append(dict(operation='ingest_readings',at_seconds=r.available_at_seconds,readings=[r.to_dict()]))
    c.finish_scan(ex.receipt); events.append(dict(operation='finish_scan',at_seconds=ex.receipt.received_at_seconds,receipt=ex.receipt.to_dict()))
    at=c._last_available; saved=ready_payload(c,provenance=dict(P0V_variant=world['id'],base_cluster=world['base_id'],fresh_prefix=True,prefix_noise=0.))
    clone=deepcopy(c); original=clone.plan_next(at,c._last_pose,ScanRequestV1(8.),wind=cfg.wind)
    visits=c.visit_times; age=at-visits; mu=c._belief.mean.reshape(cfg.model.grid_shape)
    visible=dict(age_left_mean=float(np.mean(age[:,:age.shape[1]//2])),age_right_mean=float(np.mean(age[:,age.shape[1]//2:])),
        belief_left_mean=float(np.mean(mu[:,:mu.shape[1]//2])),belief_right_mean=float(np.mean(mu[:,mu.shape[1]//2:])),
        mean_P=float(np.trace(c._belief.covariance)/len(c._belief.mean)),innovation_RMS=float(np.sqrt(np.mean(np.square(innovations)))),
        last32_innovation_RMS=float(np.sqrt(np.mean(np.square(innovations[-32:])))))
    prepared=dict(world=world,saved=saved,prefix_events=events,t0=at,T_e=at+12.,budget=8.,original_plan=original.to_dict(),
        original_ready_full_missing_count=len(c.full_scan_missing),original_ready_return_pending=c._return_pending,
        obligations_after_original_decision=dict(full_missing_count=len(clone.full_scan_missing),return_pending=clone._return_pending),
        eligible=original.mode=='FOCUS' and len(original.samples)>=cfg.session.settings.minimum_readings,
        eligibility_reason='eligible_original_FOCUS' if original.mode=='FOCUS' else original.mode+':'+original.reason,visible=visible)
    if prepared['eligible']:
        mask=np.zeros(cfg.session.instrument.lattice_shape,bool);mask.ravel()[list(original.requested_cells)]=True
        prepared.update(mask=mask.astype(int).tolist(),kwargs=kw_from(c,at,original.scan_id))
        B,_=construct_rule(mask,cfg.session.instrument,prepared['kwargs'],'B',8.)
        actual=PhysicalPlan(**{f.name:getattr(original,f.name) for f in fields(PhysicalPlan)})
        if B!=actual:raise ValueError('Baseline complete physical parity')
    return prepared
