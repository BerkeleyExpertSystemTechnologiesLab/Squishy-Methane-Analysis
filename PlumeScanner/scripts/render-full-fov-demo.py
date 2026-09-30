"""Render saved full-FoV calls in the existing six-panel layout."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import cv2
import numpy as np

from plume_focus.full_fov_loop import FullFovController, FullFovPlan
from plume_focus.full_fov_execution import FullFovRunConfig, replay_event, decode_config, decode_plan, make_controller
from plume_focus.sensing_contracts import SensorReading, ExecutionReceipt
from plume_focus.signal_belief import BeliefView
from plume_focus.sensing_simulation import SimulatedExecution, SensingScanTrace
from plume_focus.sensing_rendering import compose_sensing_frame, RenderSettings


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def gate_detail(p):
    """Keep pooled evidence, the last action's cutoff and next allocation distinct."""
    def value(item):
        return 'unavailable' if item is None else f'{item:.4f}'
    actions = p.gate_evidence_actions
    last = actions[-1] if actions else None
    calibration = (f'evidence alpha {last.alpha_action:.4f} M {last.scheduled_slots}'
                   if last is not None else 'evidence alpha/M unavailable')
    return (f'GATE {p.gate_evidence_state} | total M/n {p.gate_scheduled_slots}/{p.gate_delivered_slots} | '
            f'max L {value(p.gate_max_lower_bound)} | last cutoff {value(p.gate_observed_value_cutoff)} '
            f'({calibration}) | next alpha {p.gate_alpha_action:.4f} | '
            f'fallback {p.gate_fallback_reason or "none"}')


def render(root,name,fps=4.,speed=1.,variant=''):
    path=root/'runs'/f'{name}.json';digest=sha(path);record=json.loads(path.read_text())
    config=decode_config(record['configuration']);source=config.build_source()
    c=make_controller(config);traces=[];terminal=None
    for event in record['events']:
        replay_event(c,event)
        if event['operation']=='plan_next':
            plan=c.active_plan if event['plan']['samples'] else decode_plan(event['plan']);readings=[]
            if not plan.samples:terminal=event;continue
            prior=c.belief_view();d=c.predictive_distribution(plan.completed_at_seconds,plan.lattice_shape)
            excess=d.expected_excess(config.session.settings.threshold)
            mass=excess/excess.sum() if excess.sum()>1e-12 else np.zeros_like(excess)
            predicted=BeliefView(plan.completed_at_seconds,d.mean,np.sqrt(d.variance),mass)
        elif event['operation']=='ingest_readings':
            readings.extend(SensorReading.from_dict(r) for r in event['readings'])
        else:
            receipt=ExecutionReceipt.from_dict(event['receipt'])
            traces.append(SensingScanTrace(plan,SimulatedExecution(tuple(readings),receipt,plan,False),
                prior,predicted,c.belief_view(),c.assimilation_events,c._wind))
    if not traces:raise ValueError('No completed scan to inspect')
    run=SimpleNamespace(config=config,events=(),source_assumption=(
        'source-free / known wind' if config.session.source_position is None else 'source-anchored / known wind'))
    output=root/'inspection'/name
    if variant:output/=variant
    output.mkdir(parents=True,exist_ok=True);files=[];keyframes=[]
    for mode in ('physical','summary'):
        times=([s.execution.receipt.received_at_seconds for s in traces] if mode=='summary' else
               np.minimum(np.arange(int(np.ceil(config.duration_seconds*fps/speed))+1)*speed/fps,config.duration_seconds))
        # Inspect startup, full completion, early focus/return, cessation and end.
        interesting=[0.,traces[0].plan.completed_at_seconds,180.,config.duration_seconds]
        for trace in traces[:4]:
            p=trace.plan;interesting.extend([(p.started_at_seconds+p.completed_at_seconds)/2,
                                            (p.started_at_seconds+p.return_home_until_seconds)/2])
        indexes={int(np.argmin(np.abs(np.asarray(times)-t))) for t in interesting}
        c=make_controller(config);index=0;writer=None;video=output/f'{mode}.mp4'
        settings=RenderSettings(mode=mode,fps=fps,playback_speed=speed)
        for j,at in enumerate(times):
            while index<len(record['events']) and record['events'][index]['at_seconds']<=at+1e-10:
                replay_event(c,record['events'][index]);index+=1
            trace=next((s for s in traces if at<=s.execution.receipt.received_at_seconds+1e-10),traces[-1])
            frame=compose_sensing_frame(trace,run,float(at),settings,source,causal_belief=c.predictive_view(float(at)))
            cv2.rectangle(frame,(0,0),(frame.shape[1],27),(12,19,29),-1)
            label=f'PHYSICAL TIME / {speed:g}x' if mode=='physical' else 'COMPRESSED SUMMARY / one frame per completed scan'
            cv2.putText(frame,f'{name} / {label}',(14,23),cv2.FONT_HERSHEY_SIMPLEX,.49,(230,240,250),1,cv2.LINE_AA)
            # Evidence-gated plans add one necessary gate/units line; historical
            # records keep the original header and layout unchanged.
            gate=getattr(trace.plan,'gate_evidence_state',None)
            if gate is not None:
                cv2.rectangle(frame,(0,27),(frame.shape[1],50),(12,19,29),-1)
                detail=gate_detail(trace.plan)
                width=cv2.getTextSize(detail,cv2.FONT_HERSHEY_SIMPLEX,.40,1)[0][0]
                scale=min(.40,.40*(frame.shape[1]-28)/max(width,1))
                cv2.putText(frame,detail,(14,44),cv2.FONT_HERSHEY_SIMPLEX,scale,(196,220,238),1,cv2.LINE_AA)
            if terminal is not None and at>=terminal['at_seconds']-1e-10:
                cv2.rectangle(frame,(0,frame.shape[0]-48),(frame.shape[1],frame.shape[0]),(12,19,29),-1)
                cv2.putText(frame,'IDLE / incomplete full-scan budget / last completed measurements shown',
                    (14,frame.shape[0]-28),cv2.FONT_HERSHEY_SIMPLEX,.43,(224,231,238),1,cv2.LINE_AA)
                cv2.putText(frame,'STOP: remaining session time cannot complete the next full scan; obligation remains',
                    (14,frame.shape[0]-9),cv2.FONT_HERSHEY_SIMPLEX,.39,(162,200,184),1,cv2.LINE_AA)
            if writer is None:
                writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]))
                if not writer.isOpened():raise ValueError('Video writer unavailable')
            writer.write(frame)
            if j in indexes:
                target=output/f'{mode}-{j:04d}.png';cv2.imwrite(str(target),frame)
                keyframes.append({'path':str(target.resolve()),'time_seconds':float(at)})
        writer.release();cap=cv2.VideoCapture(str(video));count=0
        while cap.read()[0]:count+=1
        cap.release()
        if count!=len(times):raise ValueError('Video decoding incomplete')
        files.append({'path':str(video.resolve()),'sha256':sha(video),'frames':count,'fps':fps,'speed':speed})
    assert sha(path)==digest
    (output/'manifest.json').write_text(json.dumps({'execution_sha256':digest,'configuration_hash':config.digest,
        'renderer_script_sha256':sha(Path(__file__)),
        'renderer_module_sha256':sha(Path('src/plume_focus/sensing_rendering.py')),
        'status':'pending_visual_review','files':files,'keyframes':keyframes},indent=2))
    print(name,'rendered',len(files),'unchanged execution',digest,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('outputs/lean-full-fov'))
    p.add_argument('--name',required=True);p.add_argument('--fps',type=float,default=4.)
    p.add_argument('--speed',type=float,default=1.);p.add_argument('--variant',default='')
    args=p.parse_args();render(args.output,args.name,args.fps,args.speed,args.variant)
