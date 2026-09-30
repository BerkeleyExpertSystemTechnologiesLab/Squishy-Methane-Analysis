"""Signal support enables focusing; absent/infeasible signal uses costed survey."""

import math
import numpy as np
from plume_focus.signal_distribution import PredictiveDistribution
from plume_focus.sensing_messages_v4 import threshold_plan
from plume_focus.sensing_policy_v2 import survey_blocks
from plume_focus.sensing_policy_v4 import compile_for, cell_mask, service_requirement
from plume_focus.sensing_risk import receipt_time
from plume_focus.sensing_session_v5 import MEAN_POLICY


def forecast_distribution(belief, session, wind, at_seconds):
    state=belief.clone()
    state.advance(at_seconds,wind,session.source_position)
    shape=session.instrument.lattice_shape
    if hasattr(state,"distribution_on"):
        return state.distribution_on(shape)
    mean,var=state.moments_on(shape)
    return PredictiveDistribution(at_seconds,np.ones(1),mean[None],var[None])


def plan_scan_v5(belief, session, request, wind, last_sampled, *, scan_id, now, pose, rotation,
                 startup_complete,last_survey_completed,last_reading_at_seconds=None,
                 last_integration_seconds=None,has_delivered_evidence=False):
    cfg,shape=session.policy,session.instrument.lattice_shape
    budget=request.max_duration_seconds
    current=forecast_distribution(belief,session,wind,now)
    max_mean=float(np.max(current.mean));gate=max_mean>cfg.reportable_threshold
    forecast_at=now+max(0.,budget)/2
    forecast=forecast_distribution(belief,session,wind,forecast_at)
    surface=forecast.mean if cfg.policy_id==MEAN_POLICY else forecast.quantile(cfg.upper_quantile)
    eligible=tuple(np.flatnonzero(surface.ravel()>=cfg.reportable_threshold).tolist())
    required,service=service_requirement(session,last_sampled,rotation,last_survey_completed,
        now=now,pose=pose,budget=budget-cfg.expected_delivery_delay_seconds-cfg.expected_completion_overrun_seconds,
        last_reading=last_reading_at_seconds,last_integration=last_integration_seconds)
    # Service reachability must include the declared receipt, as do all actions.
    def compile(mask):
        return compile_for(session,mask,scan_id,now,pose,last_reading_at_seconds,last_integration_seconds)
    def legal(plan):
        return (len(plan.samples)>=cfg.minimum_samples and
                receipt_time(plan,cfg)<=now+budget+cfg.time_tolerance_seconds)
    if required and receipt_time(compile(cell_mask(shape,required)),cfg)>now+budget+cfg.time_tolerance_seconds:
        required,service=(),"unmet_no_reachable_block"

    def result(plan,status,reason,selected=(),fallback=""):
        if not plan.samples:status="no_legal_action"
        remaining=tuple(i for i in eligible if i not in set(plan.swept_cells))
        # At-completion mass is descriptive only; no retained-percentage gate.
        completion=forecast_distribution(belief,session,wind,plan.completed_at_seconds)
        mass=completion.expected_excess(cfg.reportable_threshold)*np.asarray(session.physical_quadrature_weights).reshape(shape)
        total=float(mass.sum());recall=float(mass.ravel()[list(plan.swept_cells)].sum()/total) if total>cfg.mass_floor else None
        return threshold_plan(plan,status=status,reason=reason,policy_id=cfg.policy_id,
            predicted_mass_recall=recall,expected_signal_total=total,confidence=None,
            startup_complete=startup_complete,termination="threshold_prefix_or_age_survey",
            signal_gate_enabled=gate,signal_gate_max_mean=max_mean,
            signal_support="delivered_posterior" if has_delivered_evidence else "declared_prior",
            signal_threshold=cfg.reportable_threshold,upper_quantile=cfg.upper_quantile,
            forecast_at_seconds=forecast_at,eligible_cells=eligible,selected_eligible_cells=tuple(selected),
            unvisited_eligible_cells=remaining,fallback_reason=fallback,
            completion_upper_signal=tuple(completion.quantile(cfg.upper_quantile).ravel().tolist()),
            selection_criterion="signal_gate_then_mean_prefix" if cfg.policy_id==MEAN_POLICY else "signal_gate_then_upper_signal_prefix",
            service_required_cells=required,service_status=service,lookahead_seconds=max(0.,budget),
            expected_delivery=cfg.expected_delivery,expected_delivery_delay_seconds=cfg.expected_delivery_delay_seconds,
            expected_completion_overrun_seconds=cfg.expected_completion_overrun_seconds,
            lookahead_wind_yx=wind.velocity_yx)

    def survey(reason):
        blocks=survey_blocks(shape,cfg.survey_height_fov)
        order=sorted(range(len(blocks)),key=lambda i:(float(np.min(last_sampled[blocks[i][0]:blocks[i][1],blocks[i][2]])),
                                                    (i-rotation)%len(blocks)))
        mask=cell_mask(shape,required);chosen=compile(mask)
        for i in order:
            lo,hi,col=blocks[i];trial=mask.copy();trial[lo:hi,col]=True
            candidate=compile(trial)
            if receipt_time(candidate,cfg)<=now+budget+cfg.time_tolerance_seconds:
                mask,chosen=trial,candidate
        if not legal(chosen):chosen=compile(np.zeros(shape,bool))
        return result(chosen,"reacquisition","Costed oldest-block scene scan",fallback=reason)

    if budget<=0:
        return result(compile(np.zeros(shape,bool)),"no_legal_action","No remaining duration",fallback="no_duration")
    if session.startup_mode=="full_scene" and not startup_complete:
        full=compile(np.ones(shape,bool))
        if legal(full):
            return result(full,"startup_full_scene","Actual-visit full-scene startup")
        return survey("startup_incomplete")
    if not gate:return survey("mean_not_above_threshold")
    if not eligible:return survey("empty_forecast_region")
    ids=np.asarray(eligible)
    order=ids[np.lexsort(((ids-rotation)%math.prod(shape),last_sampled.ravel()[ids],-surface.ravel()[ids]))]
    mask=cell_mask(shape,required);best=None;selected=[];best_selected=()
    for index in order:
        mask.ravel()[index]=True;selected.append(int(index))
        columns=np.flatnonzero(mask.any(axis=0))
        count=sum(int(np.flatnonzero(mask[:,j])[-1]-np.flatnonzero(mask[:,j])[0]+1) for j in columns)
        if count/session.instrument.acquisition.cadence_hz>budget+cfg.time_tolerance_seconds:break
        candidate=compile(mask)
        if legal(candidate):best,best_selected=candidate,tuple(selected)
    if best is None:return survey("no_legal_threshold_prefix")
    status="threshold_selected" if set(eligible).issubset(best.swept_cells) else "threshold_partial"
    return result(best,status,"Fixed intensity cutoff; longest executable ranked prefix",best_selected)
