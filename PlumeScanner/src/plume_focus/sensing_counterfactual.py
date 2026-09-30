"""Evaluator-side forced-path diagnostics. Truth never enters the controller.

These are not historical policy replays or autonomous runs: supplied actions are
mechanically checked and explicitly recorded as forced diagnostic interventions.
"""

from dataclasses import replace
import math

import numpy as np

from plume_focus.controller import SensingController
from plume_focus.physical_scan import compile_profile_scan
from plume_focus.sensing_contracts import CausalWind, SensorReading, ExecutionReceipt
from plume_focus.sensing_execution_v4 import ControllerCallV2
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_policy_v4 import CandidatePath, compile_for, repeated_pattern, cell_mask, service_requirement
from plume_focus.sensing_messages_v3 import risk_plan, ScanPlanV3
from plume_focus.sensing_risk import linear_transition, projection_gram
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.signal_belief import grid_positions, interpolation_stencil


def begin_forced_scan(controller, plan, wind, budget):
    """Diagnostic-only action injection; public lifecycle validation is retained."""
    now=plan.started_at_seconds
    if controller.active_plan is not None:
        raise RuntimeError("Cannot replace an active scan.")
    if now < max(controller._belief.time_seconds,controller._last_available)-1e-10:
        raise ValueError("Forced action precedes available evidence.")
    if wind.available_at_seconds>now or wind.available_at_seconds<controller._wind.available_at_seconds:
        raise ValueError("Noncausal forced-action wind.")
    session=controller._session;last=controller._last_detector_reading
    expected=compile_for(session,cell_mask(session.instrument.lattice_shape,plan.requested_cells),plan.scan_id,
        now,controller._last_pose,last.acquired_at_seconds if last else None,last.integration_seconds if last else None)
    if (expected.samples!=plan.samples or expected.segments!=plan.segments or
            expected.final_position_yx!=plan.final_position_yx or
            abs(expected.completed_at_seconds-plan.completed_at_seconds)>1e-9 or
            plan.duration_seconds>budget+1e-9):
        raise ValueError("Forced plan is not the compiled legal path from the current pose.")
    if plan.samples and len(plan.samples)<session.policy.minimum_samples:
        raise ValueError("Forced plan violates minimum readings.")
    controller._belief.advance(now,controller._wind,controller.source_position)
    controller._wind=wind
    controller._last_available=now
    if plan.samples:
        controller._active=plan
        controller._base=controller._belief.clone()
        controller._pending,controller._received,controller._events={},{},[]


def replay_forced_history(config, events, stop_index, *, decision_time=None):
    """Same delivered evidence/actions under another estimator; no policy equality claim."""
    controller=SensingController.from_session(config.session)
    for event in events[:stop_index]:
        if event.operation=="plan_next":
            begin_forced_scan(controller,event.plan,event.wind or controller._wind,event.request.max_duration_seconds)
        elif event.operation=="ingest_readings":
            controller.ingest_readings(event.readings,available_at_seconds=event.at_seconds)
        else:
            controller.finish_scan(event.receipt)
    if decision_time is not None:
        controller._belief.advance(decision_time,controller._wind,controller.source_position)
        controller._last_available=decision_time
    return controller


def predicted_mean(controller,time):
    """Query-only mean propagation; avoids copying covariance for loss quadrature."""
    belief=controller._belief
    if hasattr(belief,'half_life_seconds'):
        from plume_focus.calibration_evaluation import predicted_mean as decay_mean
        return decay_mean(controller,time)
    elapsed=round(time-belief.time_seconds,12)
    if time<controller._last_available-1e-9 or elapsed<-1e-9:
        raise ValueError("Past online estimates require an earlier evidence snapshot.")
    components=getattr(belief,"components",(belief,))
    weights=getattr(belief,"weights",np.ones(1))
    offsets=getattr(belief,"offsets",np.zeros((1,2)))
    result=np.zeros_like(components[0].mean)
    for alpha,state,offset in zip(weights,components,offsets):
        if elapsed<=1e-12:
            mean=state.mean
        else:
            velocity=tuple(np.asarray(controller._wind.velocity_yx)+offset)
            a,b,_=linear_transition(state.config,velocity,controller.source_position,elapsed,belief.transport_id)
            mean=a@state.mean+b
        result+=alpha*mean
    return result


class TruthQuadrature:
    """Shared, candidate-independent truth sufficient statistics for map loss."""
    def __init__(self, source, start, horizon, latent_shape, shape=(120,160), step=.05):
        self.start,self.horizon,self.shape,self.step=start,horizon,shape,step
        self.times=start+np.arange(int(math.ceil(horizon/step)))*step
        self.durations=np.diff(np.append(self.times,start+horizon))
        points=grid_positions(shape);ids,weights=interpolation_stencil(points,latent_shape)
        self.gram=projection_gram(latent_shape,shape)
        self.cross=[];self.square=[];self.average=[]
        self.mean_operator=np.zeros(math.prod(latent_shape))
        np.add.at(self.mean_operator,ids.ravel(),weights.ravel()/len(points))
        for t in self.times:
            values=source.signal_at(np.full(len(points),t),points)
            projected=np.zeros(math.prod(latent_shape))
            np.add.at(projected,ids.ravel(),(weights*values[:,None]).ravel()/len(points))
            self.cross.append(projected);self.square.append(float(np.mean(values**2)));self.average.append(float(np.mean(values)))
        self.cross=np.asarray(self.cross)


class OnlineLoss:
    def __init__(self, truth):
        self.truth=truth;self.index=0;self.rows=[]

    def before(self,controller,available_at):
        q=self.truth
        while self.index<len(q.times) and q.times[self.index]<available_at-1e-10:
            i=self.index;t=float(q.times[i]);mu=predicted_mean(controller,t)
            mse=max(0.,float(mu@q.gram@mu-2*mu@q.cross[i]+q.square[i]))
            signed=float(q.mean_operator@mu-q.average[i])
            self.rows.append({"time":t,"map_mse":mse,"signed_index_error":signed,"duration":float(q.durations[i])})
            self.index+=1

    def result(self):
        return {"J":sum(r["map_mse"]*r["duration"] for r in self.rows),
                "signed_index_error_integral":sum(r["signed_index_error"]*r["duration"] for r in self.rows),
                "index_squared_error_integral":sum(r["signed_index_error"]**2*r["duration"] for r in self.rows),
                "quadrature_samples":self.rows}


def run_counterfactual(snapshot,config,candidate,required,*,continuation="repeat",seed=0,horizon=12.,truth=None):
    if continuation not in ("repeat","replan","idle"):
        raise ValueError("Unknown diagnostic continuation.")
    controller=snapshot.fork_controller(SensingController);start=controller._last_available;end=start+horizon
    lookahead_wind=controller._wind
    truth=truth or TruthQuadrature(config.source.build(),start,horizon,config.model.grid_shape)
    loss=OnlineLoss(truth)
    scanner=FakeScanner(config.source.build(),noise_std=config.simulated_noise_std,seed=seed,
                        gain=config.model.gain,offset=config.model.offset)
    events=[];scans=[];spent=0.;pose=controller._last_pose;now=start
    patterns=()
    if continuation=="repeat":
        last=controller._last_detector_reading
        patterns=repeated_pattern(candidate,config.session,controller.visit_times,controller._rotation,
             controller._last_survey_completed,required,horizon=horizon,
             last_reading=last.acquired_at_seconds if last else None,last_integration=last.integration_seconds if last else None)
    count=0
    while continuation!="idle" and now<end-1e-9:
        forced=count==0 or continuation=="repeat"
        if continuation=="repeat":
            if count>=len(patterns):break
            plan=patterns[count].plan
            # Service requirements are encoded by the compiler wrapper below.
        elif count==0:
            plan=candidate.plan
        else:
            plan=controller.plan_next(now,pose,ScanRequestV1(end-now),wind=lookahead_wind)
        if not plan.samples:break
        if plan.completed_at_seconds+config.delivery_delay_seconds+config.completion_overrun_seconds>end+1e-9:
            if not forced:raise ValueError("Replanner returned a receipt outside diagnostic horizon.")
            break
        if forced:
            last=controller._last_detector_reading
            service_cells,service_status=(required,"required" if required else "not_due") if count==0 else service_requirement(
                config.session,controller.visit_times,controller._rotation,controller._last_survey_completed,
                now=now,pose=pose,budget=end-now,last_reading=last.acquired_at_seconds if last else None,
                last_integration=last.integration_seconds if last else None)
            plan=risk_plan(plan,policy_id=config.policy,service_required_cells=service_cells,service_status=service_status,
                           status="candidate",reason="Forced diagnostic intervention; not a production policy decision")
            begin_forced_scan(controller,plan,lookahead_wind,end-now)
        events.append({"operation":"forced_plan" if forced else "plan_next","at_seconds":now,"plan":plan.to_dict()})
        execution=scanner.execute(plan,delivery=config.delivery,delivery_delay=config.delivery_delay_seconds,
                                   completion_overrun=config.completion_overrun_seconds)
        if execution.exhausted:raise ValueError("Incomplete truth horizon in diagnostic execution.")
        batches=[(execution.readings,execution.receipt.received_at_seconds)] if config.delivery=="batch" else [
            ((r,),r.available_at_seconds) for r in execution.readings]
        for readings,at in batches:
            loss.before(controller,at)
            controller.ingest_readings(readings,available_at_seconds=at)
            events.append({"operation":"ingest_readings","at_seconds":at,"readings":[r.to_dict() for r in readings]})
        loss.before(controller,execution.receipt.received_at_seconds)
        controller.finish_scan(execution.receipt)
        events.append({"operation":"finish_scan","at_seconds":execution.receipt.received_at_seconds,
                       "receipt":execution.receipt.to_dict()})
        scans.append({"plan":plan.to_dict(),"receipt":execution.receipt.to_dict(),"readings":len(execution.readings)})
        spent+=execution.executed_plan.duration_seconds
        now,pose=execution.receipt.received_at_seconds,execution.receipt.final_position_yx
        count+=1
    loss.before(controller,end+1e-8)
    result={"schema_version":1,"artifact_type":"forced_counterfactual_v1","snapshot_hash":snapshot.digest,
        "transport_id":controller._belief.transport_id,"continuation":continuation,"seed":seed,"start":start,
        "horizon":horizon,"candidate":candidate.name if candidate else None,"scans":scans,"events":events,
        "spent_seconds":spent,"J_over_T":loss.result()["J"]/horizon,**loss.result(),
        "final_visits":controller.visit_times.tolist(),"final_motion":controller.motion_diagnostics()}
    return result


def evaluate_saved_branch(snapshot,record,truth):
    """Reevaluate recorded deliveries/actions; never rerun adaptive planning."""
    if snapshot.digest!=record['snapshot_hash']:
        raise ValueError("Branch does not belong to the supplied snapshot.")
    controller=snapshot.fork_controller(SensingController);loss=OnlineLoss(truth)
    for event in record['events']:
        at=event['at_seconds'];loss.before(controller,at)
        if event['operation'] in ('forced_plan','plan_next'):
            plan=ScanPlanV3.from_dict(event['plan'])
            begin_forced_scan(controller,plan,controller._wind,record['start']+record['horizon']-at)
        elif event['operation']=='ingest_readings':
            controller.ingest_readings([SensorReading.from_dict(r) for r in event['readings']],available_at_seconds=at)
        elif event['operation']=='finish_scan':
            controller.finish_scan(ExecutionReceipt.from_dict(event['receipt']))
        else:raise ValueError("Unknown diagnostic delivery event.")
    loss.before(controller,record['start']+record['horizon']+1e-8)
    return loss.result()
