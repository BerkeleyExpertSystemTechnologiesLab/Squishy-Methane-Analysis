"""Headless simulator and causal replay for the independent full-FoV loop."""
from dataclasses import dataclass
from typing import Literal
import time
import numpy as np

from plume_focus.full_fov_loop import FullFovController, FullFovPlan, FullFovSession
from plume_focus.sensing_contracts import CausalWind, SensorReading, ExecutionReceipt, finite
from plume_focus.sensing_session import RequiredMessage, required_fields, configuration_hash
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.sensing_execution_v2 import SyntheticSourceV2
from plume_focus.sensing_simulation import FakeScanner
from plume_focus.sensing_counterfactual import TruthQuadrature, OnlineLoss


@dataclass(frozen=True)
class FullFovRunConfig(RequiredMessage):
    session: FullFovSession
    source: SyntheticSourceV2
    wind: CausalWind
    duration_seconds: float = 360.
    simulated_noise_std: float = 0.
    cessation_seconds: float | None = None
    seed: int = 0
    delivery: Literal["stream", "batch"] = "stream"
    evaluation_shape: tuple[int, int] = (120, 160)
    evaluation_step_seconds: float = .05

    def __post_init__(self):
        for name in ("duration_seconds", "evaluation_step_seconds"):
            finite(name, getattr(self,name), positive=True)
        for name in ("simulated_noise_std",):
            finite(name,getattr(self,name))
        if self.cessation_seconds is not None:
            finite("cessation",self.cessation_seconds)
        if type(self.seed) is not int or self.seed < 0 or self.delivery not in ("stream", "batch"):
            raise ValueError("Invalid seed or delivery mode.")
        if len(self.evaluation_shape)!=2 or any(type(n) is not int or n<2 for n in self.evaluation_shape):
            raise ValueError("Invalid evaluation mesh.")
        if self.source.end_time_seconds is not None and self.source.end_time_seconds < self.duration_seconds:
            raise ValueError("Demonstration truth must cover the declared horizon.")

    def to_dict(self):
        result=super().to_dict()
        result['session']=self.session.to_dict()
        return result

    @classmethod
    def _from_fields(cls,value):
        required_fields(cls,value)
        payload=dict(value)
        session=FullFovSession.from_dict(payload.pop('session'))
        # Decode all ordinary nested messages using the common transport rules.
        from plume_focus.sensing_contracts import _decode
        from typing import get_type_hints
        annotations=get_type_hints(cls)
        return cls(session=session,**{k:_decode(annotations[k],v) for k,v in payload.items()})

    @property
    def model(self): return self.session.model

    @property
    def capabilities(self): return self.session.instrument.capabilities()

    @property
    def policy(self): return self.session.policy_id

    @property
    def digest(self): return configuration_hash(self.to_dict())

    def build_source(self): return ControlledSignal(self.source.build(),self.cessation_seconds)


class ControlledSignal:
    """Explicit observer-side blank/disappearance controls; main plume unchanged."""
    def __init__(self,base,cessation):
        self.base,self.cessation=base,cessation
        self.end_time_seconds=base.end_time_seconds

    def signal_at(self,times,positions_yx):
        values=self.base.signal_at(times,positions_yx)
        return values if self.cessation is None else np.where(np.asarray(times)<self.cessation,values,0.)


def replay_event(controller,event,*,verify_plan=True):
    at=event['at_seconds']
    if event['operation']=='plan_next':
        expected=decode_plan(event['plan'])
        actual=controller.plan_next(at,tuple(event['position_yx']),ScanRequestV1(event['remaining_seconds']),
                                    wind=CausalWind.from_dict(event['wind']))
        # The cleanup adds compact evidence provenance. Frozen v1 gate records
        # predate that optional field; compare every recorded field as before.
        if event['plan'].get('message_type') == 'EvidenceGatedPlan' and 'gate_evidence_actions' not in event['plan']:
            from dataclasses import replace
            actual = replace(actual, gate_evidence_actions=())
        if verify_plan and actual!=expected:
            raise ValueError("Public full-FoV plan replay differs.")
    elif event['operation']=='ingest_readings':
        controller.ingest_readings([SensorReading.from_dict(r) for r in event['readings']],available_at_seconds=at)
    elif event['operation']=='finish_scan':
        controller.finish_scan(ExecutionReceipt.from_dict(event['receipt']))
    else:
        raise ValueError("Unknown acquisition operation.")


def run_full_fov(config,*,truth=None):
    """One causal execution. Files and renderer are observers, not dependencies."""
    clock=time.perf_counter()
    source=config.build_source()
    truth=truth or TruthQuadrature(source,0.,config.duration_seconds,config.model.grid_shape,
                                  config.evaluation_shape,config.evaluation_step_seconds)
    loss=OnlineLoss(truth)
    controller=make_controller(config)
    scanner=FakeScanner(source,noise_std=config.simulated_noise_std,seed=config.seed,
                        gain=config.model.gain,offset=config.model.offset)
    clean_scanner=FakeScanner(source,noise_std=0.,seed=config.seed,gain=config.model.gain,offset=config.model.offset)
    now=0.;pose=config.session.home_yx;events=[];scans=[];termination='horizon_complete'
    while now<config.duration_seconds-1e-9:
        plan=controller.plan_next(now,pose,ScanRequestV1(config.duration_seconds-now),wind=config.wind)
        events.append({'operation':'plan_next','at_seconds':now,'position_yx':pose,
            'remaining_seconds':config.duration_seconds-now,'wind':config.wind.to_dict(),'plan':plan.to_dict()})
        if not plan.samples:
            termination=plan.reason
            break
        cfg=config.session.settings
        kwargs=dict(delivery=config.delivery,delivery_delay=cfg.delivery_delay_seconds,
                    completion_overrun=cfg.completion_overrun_seconds)
        execution=scanner.execute(plan,**kwargs)
        clean=clean_scanner.execute(plan,**kwargs)
        batches=([(execution.readings,execution.receipt.received_at_seconds)] if config.delivery=='batch'
                 else [((r,),r.available_at_seconds) for r in execution.readings])
        for readings,at in batches:
            loss.before(controller,at)
            controller.ingest_readings(readings,available_at_seconds=at)
            events.append({'operation':'ingest_readings','at_seconds':at,'readings':[r.to_dict() for r in readings]})
        loss.before(controller,execution.receipt.received_at_seconds)
        controller.finish_scan(execution.receipt)
        events.append({'operation':'finish_scan','at_seconds':execution.receipt.received_at_seconds,
                       'receipt':execution.receipt.to_dict()})
        scans.append({'plan':plan.to_dict(),'receipt':execution.receipt.to_dict(),
            'clean_readings':[r.value for r in clean.readings],
            'readings':[r.value for r in execution.readings],
            'startup_complete':controller.startup_complete,'completed_full_scans':controller.completed_full_scans,
            'missing_full_cells':controller.full_scan_missing})
        now,pose=execution.receipt.received_at_seconds,execution.receipt.final_position_yx
        if execution.exhausted:
            termination='source_exhausted';break
    loss.before(controller,config.duration_seconds+1e-8)
    return {'artifact_type':'full_fov_loop_execution_v1','configuration':config.to_dict(),
        'configuration_hash':config.digest,'events':events,'scans':scans,'termination':termination,
        'startup_complete':controller.startup_complete,'completed_full_scans':controller.completed_full_scans,
        'missing_full_cells':controller.full_scan_missing,'J_over_T':loss.result()['J']/config.duration_seconds,
        **loss.result(),'spent_seconds':sum(s['plan']['completed_at_seconds']-s['plan']['started_at_seconds'] for s in scans),
        'offline_seconds':time.perf_counter()-clock}


def verify_replay(record,*,truth=None):
    config=decode_config(record['configuration'])
    if config.digest!=record['configuration_hash']:
        raise ValueError("Execution configuration hash differs.")
    controller=make_controller(config)
    loss=OnlineLoss(truth) if truth is not None else None
    for event in record['events']:
        if loss is not None: loss.before(controller,event['at_seconds'])
        replay_event(controller,event)
    if loss is not None:
        loss.before(controller,config.duration_seconds+1e-8)
        if not np.isclose(loss.result()['J'],record['J'],rtol=0,atol=1e-12):
            raise ValueError("Causal reconstruction replay differs.")
    return controller


def decode_config(value):
    if value.get('message_type')=='EvidenceGatedRunConfig':
        from plume_focus.evidence_gated_execution import EvidenceGatedRunConfig
        return EvidenceGatedRunConfig.from_dict(value)
    if value.get('message_type')=='CalibrationRunConfig':
        from plume_focus.calibration_execution import CalibrationRunConfig
        return CalibrationRunConfig.from_dict(value)
    return FullFovRunConfig.from_dict(value)


def decode_plan(value):
    if value.get('message_type')=='EvidenceGatedPlan':
        from plume_focus.evidence_gated_loop import EvidenceGatedPlan
        return EvidenceGatedPlan.from_dict(value)
    if value.get('message_type')=='CalibrationPlan':
        from plume_focus.calibrated_scanner import CalibrationPlan
        return CalibrationPlan.from_dict(value)
    return FullFovPlan.from_dict(value)


def make_controller(config):
    return config.make_controller() if hasattr(config,'make_controller') else FullFovController(config.session)
