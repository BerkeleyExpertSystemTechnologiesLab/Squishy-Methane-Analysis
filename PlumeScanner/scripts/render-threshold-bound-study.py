"""Existing six-panel inspection of saved autonomous v5 comparison traces."""

import argparse
from pathlib import Path
import json
import hashlib
import cv2
import numpy as np

from plume_focus.controller import SensingController
from plume_focus.sensing_cli_v1 import config_from_dict,write_json
from plume_focus.sensing_snapshot import DecisionSnapshot
from plume_focus.sensing_threshold_evaluation import apply_event,decode_plan
from plume_focus.sensing_contracts import SensorReading,ExecutionReceipt
from plume_focus.signal_belief import BeliefView
from plume_focus.sensing_simulation import SimulatedExecution,SensingScanTrace
from plume_focus.sensing_execution_v1 import SensingRunV1
from plume_focus.sensing_rendering import compose_sensing_frame,RenderSettings


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def render(root,name,policy,mode='short',seed=0,fps=8.,speed=1.,variant=''):
    directory=root/'fixtures'/name/policy;config=config_from_dict(read(directory/'configuration.json'))
    snapshot=DecisionSnapshot.from_dict(read(directory/('snapshot.json' if mode=='short' else 'initial-snapshot.json')))
    path=root/'runs'/name/policy/f'{mode}-seed-{seed}.json';record=read(path);digest=sha(path)
    controller=snapshot.fork_controller(SensingController);traces=[]
    for event in record['events']:
        apply_event(controller,event)
        if event['operation']=='plan_next':
            plan=decode_plan(event['plan']);readings=[]
            if not plan.samples:continue
            prior=controller.belief_view()
            distribution=controller.predictive_distribution(plan.completed_at_seconds,config.capabilities.lattice_shape)
            density=distribution.expected_excess(.15)
            mass=density/density.sum() if density.sum()>config.session.policy.mass_floor else np.zeros_like(density)
            predicted=BeliefView(plan.completed_at_seconds,distribution.mean,np.sqrt(distribution.variance),mass)
        elif event['operation']=='ingest_readings':
            readings.extend(SensorReading.from_dict(r) for r in event['readings'])
        else:
            receipt=ExecutionReceipt.from_dict(event['receipt'])
            execution=SimulatedExecution(tuple(readings),receipt,plan,False)
            traces.append(SensingScanTrace(plan,execution,prior,predicted,controller.belief_view(),
                          controller.assimilation_events,controller._wind,distribution,controller.motion_diagnostics()))
    if not traces:raise ValueError('No completed scan to render')
    run=SensingRunV1(config,tuple(traces),(),record['termination'],record['startup_complete'],float(np.mean(controller.visit_times>=0)))
    output=root/'inspection'/name/policy
    if variant:output/=variant
    output.mkdir(parents=True,exist_ok=True);files=[];end=record['start']+record['horizon']
    for view in ('physical','summary'):
        settings=RenderSettings(mode=view,fps=fps,playback_speed=speed)
        times=([s.execution.receipt.received_at_seconds for s in traces] if view=='summary' else
               list(np.minimum(record['start']+np.arange(int(np.ceil(record['horizon']*fps/speed))+1)*speed/fps,end)))
        controller=snapshot.fork_controller(SensingController);index=0;writer=None;video=output/f'{mode}-seed-{seed}-{view}.mp4'
        for frame_index,at in enumerate(times):
            while index<len(record['events']) and record['events'][index]['at_seconds']<=at+1e-10:
                apply_event(controller,record['events'][index]);index+=1
            trace=next((s for s in traces if at<=s.execution.receipt.received_at_seconds+1e-10),traces[-1])
            frame=compose_sensing_frame(trace,run,float(at),settings,config.source.build(),causal_belief=controller.predictive_view(float(at)))
            cv2.rectangle(frame,(0,0),(frame.shape[1],27),(12,19,29),-1)
            label=(f'PHYSICAL TIME / {speed:g}x playback' if view=='physical' else f'COMPRESSED SUMMARY / {fps:g} fps; 1 frame/sweep')
            cv2.putText(frame,f'AUTONOMOUS {policy.upper()} / {label}',(14,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,240,250),1,cv2.LINE_AA)
            if writer is None:
                writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]))
                if not writer.isOpened():raise ValueError('Video writer unavailable')
            writer.write(frame)
            if frame_index in (0,len(times)//2,len(times)-1):cv2.imwrite(str(output/f'{mode}-{view}-{frame_index:03d}.png'),frame)
        writer.release();capture=cv2.VideoCapture(str(video));decoded=0
        while capture.read()[0]:decoded+=1
        capture.release()
        if decoded!=len(times):raise ValueError('Incomplete decoded video')
        files.append({'path':str(video.resolve()),'sha256':sha(video),'frames':decoded,'fps':fps,'speed':speed})
    assert sha(path)==digest
    result={'status':'pending_visual_review','execution_sha256':digest,'snapshot_hash':snapshot.digest,'files':files}
    write_json(output/f'{mode}-seed-{seed}-manifest.json',result);print(name,policy,len(files),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--fixture',required=True)
    p.add_argument('--policy',default='upper');p.add_argument('--mode',default='short');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--fps',type=float,default=8.);p.add_argument('--speed',type=float,default=1.);p.add_argument('--variant',default='')
    args=p.parse_args();render(args.output,args.fixture,args.policy,args.mode,args.seed,args.fps,args.speed,args.variant)
