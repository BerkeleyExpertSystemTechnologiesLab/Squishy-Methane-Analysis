"""Evaluator adapter for autonomous policy continuations from frozen evidence."""

import time
import numpy as np
from plume_focus.controller import SensingController
from plume_focus.sensing_messages_v2 import ScanPlanV2, ScanRequestV1
from plume_focus.sensing_messages_v3 import ScanPlanV3
from plume_focus.sensing_messages_v4 import ScanPlanV4
from plume_focus.sensing_contracts import SensorReading, ExecutionReceipt, CausalWind
from plume_focus.sensing_counterfactual import TruthQuadrature, OnlineLoss, begin_forced_scan
from plume_focus.sensing_simulation import FakeScanner


def decode_plan(payload):
    return {2:ScanPlanV2,3:ScanPlanV3,4:ScanPlanV4}[payload['schema_version']].from_dict(payload)


def apply_event(controller,event,*,verify_policy=False):
    at=event['at_seconds']
    if event['operation']=='plan_next':
        plan=decode_plan(event['plan']);wind=CausalWind.from_dict(event['wind'])
        if verify_policy:
            actual=controller.plan_next(at,tuple(event['position_yx']),ScanRequestV1(event['budget_seconds']),wind=wind)
            if actual!=plan:raise ValueError("Autonomous trace policy replay differs.")
        else:
            begin_forced_scan(controller,plan,wind,event['budget_seconds'])
    elif event['operation']=='ingest_readings':
        controller.ingest_readings(tuple(SensorReading.from_dict(r) for r in event['readings']),available_at_seconds=at)
    elif event['operation']=='finish_scan':
        controller.finish_scan(ExecutionReceipt.from_dict(event['receipt']))
    else:raise ValueError("Unknown execution operation.")


def run_policy(snapshot,config,*,horizon,seed,full_session=False,truth=None):
    start_clock=time.perf_counter();controller=snapshot.fork_controller(SensingController)
    start=controller._last_available;end=start+horizon;now=start;pose=controller._last_pose
    truth=truth or TruthQuadrature(config.source.build(),start,horizon,config.model.grid_shape,
                                  config.evaluation.shape,config.evaluation.step_seconds)
    loss=OnlineLoss(truth);source=config.source.build()
    scanner=FakeScanner(source,noise_std=config.simulated_noise_std,seed=seed,gain=config.model.gain,offset=config.model.offset)
    events=[];scans=[];decisions=[];termination='horizon_complete'
    while now<end-1e-10:
        ceiling=config.initial_budget_seconds if full_session and not scans else config.budget_seconds
        budget=min(ceiling,end-now)
        before=controller.decision_snapshot()
        plan=controller.plan_next(now,pose,ScanRequestV1(budget),wind=config.wind)
        events.append({'operation':'plan_next','at_seconds':now,'position_yx':pose,'wind':config.wind.to_dict(),
                       'budget_seconds':budget,'plan':plan.to_dict()})
        if not plan.samples:
            termination='no_legal_action';break
        if plan.completed_at_seconds+config.delivery_delay_seconds+config.completion_overrun_seconds>end+1e-8:
            raise ValueError("Plan receipt exceeds the common experiment horizon.")
        # Evaluator-only gate truth; never passed back to controller/planner.
        from plume_focus.signal_belief import grid_positions
        points=grid_positions(config.capabilities.lattice_shape)
        signal=source.signal_at(np.full(len(points),now),points)
        decisions.append({'at_seconds':now,'snapshot_identity':before.digest,'truth_max_signal':float(signal.max()),
            'truth_any_above_threshold':bool(np.any(signal>.15)),
            'gate':getattr(plan,'signal_gate_enabled',None),'max_mean':getattr(plan,'signal_gate_max_mean',None),
            'status':plan.status,'fallback_reason':getattr(plan,'fallback_reason',''),
            'eligible_count':len(getattr(plan,'eligible_cells',())),
            'unvisited_eligible_count':len(getattr(plan,'unvisited_eligible_cells',()))})
        execution=scanner.execute(plan,delivery=config.delivery,delivery_delay=config.delivery_delay_seconds,
                                 completion_overrun=config.completion_overrun_seconds)
        if execution.exhausted:raise ValueError("Declared comparison truth horizon is unavailable.")
        batches=[(execution.readings,execution.receipt.received_at_seconds)] if config.delivery=='batch' else [
            ((r,),r.available_at_seconds) for r in execution.readings]
        for readings,at in batches:
            loss.before(controller,at)
            controller.ingest_readings(readings,available_at_seconds=at)
            events.append({'operation':'ingest_readings','at_seconds':at,'readings':[r.to_dict() for r in readings]})
        loss.before(controller,execution.receipt.received_at_seconds)
        controller.finish_scan(execution.receipt)
        events.append({'operation':'finish_scan','at_seconds':execution.receipt.received_at_seconds,'receipt':execution.receipt.to_dict()})
        clean_execution=FakeScanner(source,noise_std=0,seed=seed).execute(plan,delivery=config.delivery,
            delivery_delay=config.delivery_delay_seconds,completion_overrun=config.completion_overrun_seconds)
        clean=[r.value for r in clean_execution.readings]
        scans.append({'plan':plan.to_dict(),'receipt':execution.receipt.to_dict(),'readings':len(execution.readings),
            'clean_readings':clean,'service_completed_at':controller._last_survey_completed,
            'startup_complete':controller.startup_complete})
        now,pose=execution.receipt.received_at_seconds,execution.receipt.final_position_yx
    loss.before(controller,end+1e-8)
    return {'artifact_type':'threshold_policy_execution_v1','schema_version':1,'snapshot_hash':snapshot.digest,
        'start':start,'horizon':horizon,'seed':seed,'policy':config.policy,'full_session':full_session,
        'termination':termination,'events':events,'scans':scans,'decisions':decisions,
        'J_over_T':loss.result()['J']/horizon,**loss.result(),
        'spent_seconds':sum(s['plan']['completed_at_seconds']-s['plan']['started_at_seconds'] for s in scans),
        'final_visits':controller.visit_times.tolist(),'startup_complete':controller.startup_complete,
        'final_motion':controller.motion_diagnostics(),'offline_seconds':time.perf_counter()-start_clock}


def evaluate_execution(snapshot,record,truth,*,verify_policy=False):
    if snapshot.digest!=record['snapshot_hash']:raise ValueError("Execution snapshot identity differs.")
    controller=snapshot.fork_controller(SensingController);loss=OnlineLoss(truth)
    for event in record['events']:
        loss.before(controller,event['at_seconds']);apply_event(controller,event,verify_policy=verify_policy)
    loss.before(controller,record['start']+record['horizon']+1e-8)
    return loss.result()
