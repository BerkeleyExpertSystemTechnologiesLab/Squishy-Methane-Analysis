"""Frozen, resumable attribution experiment; generated evidence stays local."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

from plume_focus.controller import SensingController
from plume_focus.sensing_cli_v1 import config_from_dict, write_json as _write_json, artifact_manifest
from plume_focus.sensing_counterfactual import replay_forced_history, run_counterfactual, TruthQuadrature
from plume_focus.sensing_execution_v4 import ControllerCallV2
from plume_focus.sensing_messages_v2 import ScanPlanV2, ScanRequestV1
from plume_focus.sensing_policy_v4 import candidate_paths, CandidatePath, compile_for, cell_mask, repeated_pattern
from plume_focus.sensing_risk import score_pattern
from plume_focus.sensing_snapshot import DecisionSnapshot
from plume_focus.sensing_transport_diagnostic import with_transport_diagnostic
from plume_focus.signal_belief import grid_positions, positive_expectation


def read(path):return json.loads(Path(path).read_text(encoding="utf-8"))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def revision():return subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()


def write_json(path,payload):
    path=Path(path)
    temporary=path.with_suffix(path.suffix+'.partial')
    _write_json(temporary,payload)
    temporary.replace(path)


def immutable(path,payload):
    value=json.loads(json.dumps(payload,allow_nan=False))
    if path.exists():
        if read(path)!=value:raise ValueError(f"Frozen declaration differs: {path}")
    else:write_json(path,value)


def fixtures():
    result=[]
    for profile in ("coarse-full-startup","double-prior-startup"):
        for case in ("known-wind","wind-error"):
            result.append(f"{profile}-{case}-threshold90_segment_v2-seed-0-baseline-risk")
    result += [f"diagnostic-source-free-biased-scale-{scale}-seed-0-baseline-risk" for scale in (1,0)]
    result += ["double-prior-startup-wind-error-threshold90_segment_v2-seed-0-motion-risk",
               "diagnostic-source-free-biased-scale-1-seed-0-motion-risk"]
    return result


def lifecycle(controller,event):
    last=controller._last_detector_reading
    return dict(scan_id=event.plan.scan_id,now=event.at_seconds,pose=event.position_yx,
        rotation=controller._rotation,last_survey_completed=controller._last_survey_completed,
        startup_complete=controller.startup_complete,
        last_reading_at_seconds=last.acquired_at_seconds if last else None,
        last_integration_seconds=last.integration_seconds if last else None)


def extra_candidates(controller,config,event,required,previous,production):
    shape=config.capabilities.lattice_shape;now=event.at_seconds
    forecast=controller._belief.clone();forecast.advance(now+6,controller._wind,controller.source_position)
    mean,var=forecast.moments_on(shape)
    mass=positive_expectation(mean,var,config.session.allocation_threshold)
    anchor=np.unravel_index(np.argmax(mass),shape)
    points=grid_positions(shape).reshape(*shape,2)
    center=points[anchor];distance=np.max(np.abs(points-center),axis=-1)
    order=np.argsort(distance.ravel(),kind="stable")
    last=controller._last_detector_reading
    def compile(core):
        mask=core.copy();mask.ravel()[list(required)]=True
        return compile_for(config.session,mask,event.plan.scan_id,now,event.position_yx,
                           last.acquired_at_seconds if last else None,last.integration_seconds if last else None)
    best={3.:None,6.:None};lo=np.asarray(anchor);hi=lo.copy();last_box=None
    for index in order:
        cell=np.asarray(np.unravel_index(index,shape));lo=np.minimum(lo,cell);hi=np.maximum(hi,cell)
        box=tuple(lo)+tuple(hi)
        if box==last_box:continue
        last_box=box
        if int(np.prod(hi-lo+1))/config.session.instrument.acquisition.cadence_hz>6+1e-9:break
        core=np.zeros(shape,bool);core[lo[0]:hi[0]+1,lo[1]:hi[1]+1]=True
        plan=compile(core)
        if len(plan.samples)<config.session.policy.minimum_samples:continue
        for ceiling in best:
            if plan.duration_seconds<=ceiling+1e-9:best[ceiling]=core.copy()
    proposals=[(f"diagnostic_local_{int(t)}s",m) for t,m in best.items() if m is not None]
    if best[3.] is not None:
        for axis in (0,1):
            shift=max(1,int(round(shape[axis]/config.model.grid_shape[axis])))
            for sign in (-1,1):
                old=np.argwhere(best[3.]);new=old.copy();new[:,axis]+=sign*shift
                if np.any(new<0) or np.any(new>=np.asarray(shape)):continue
                mask=np.zeros(shape,bool);mask[tuple(new.T)]=True
                proposals.append((f"diagnostic_shift_{axis}_{sign}",mask))
    proposals.append(("diagnostic_previous",cell_mask(shape,previous.requested_cells)))
    y0,x0,y1,x1=config.session.policy.corridor_yxyx
    proposals.append(("diagnostic_corridor",(points[...,0]>=y0)&(points[...,0]<=y1)&(points[...,1]>=x0)&(points[...,1]<=x1)))
    def key(p):return (p.swept_cells,tuple((s.position_yx,s.acquired_at_seconds) for s in p.samples))
    seen={key(c.plan) for c in production};result=[]
    for name,mask in proposals:
        plan=compile(mask)
        if len(plan.samples)<config.session.policy.minimum_samples or plan.duration_seconds>12+1e-9:continue
        identity=key(plan)
        if identity in seen:continue
        seen.add(identity);result.append(CandidatePath(name,tuple(np.flatnonzero(mask)),plan))
    return result


def declare(root,historical):
    root.mkdir(parents=True,exist_ok=True)
    declaration={"schema_version":1,"artifact_type":"scan_value_diagnosis_declaration_v1",
        "starting_revision":"7bd9d54486a1892dee3eacb39df0cbcd460c10eb","horizon":12.,
        "future_seed_stages":[8,16,32],"risk_steps":[.1,.05,.025],"mixture_histories":[32,64,128],
        "evaluation_shape":[120,160],"evaluation_step":.05,"absolute_tolerance":1e-10,"relative_tolerance":.01,
        "selection_rule":"first decision after a completed sweep and actual startup, with at least 12 seconds remaining",
        "fixtures":{}}
    for name in fixtures():
        path=historical/"development"/name/"trace.json"
        payload=read(path);events=[ControllerCallV2.from_dict(e) for e in payload["events"]]
        config=config_from_dict(payload["configuration"]);completed=0;selected=None
        for i,e in enumerate(events):
            if e.operation=="finish_scan":completed+=1
            if (e.operation=="plan_next" and e.plan.samples and completed and
                (config.session.startup_mode!="full_scene" or e.plan.startup_complete) and
                config.duration_seconds-e.at_seconds>=12-1e-9):
                selected=i;break
        declaration["fixtures"][name]={"trace":str(path.resolve()),"trace_sha256":sha(path),
            "event_index":selected,"available":selected is not None,
            "time":events[selected].at_seconds if selected is not None else None}
    immutable(root/"declaration.json",declaration)
    for name,ref in declaration["fixtures"].items():
        directory=root/"fixtures"/name
        if not ref["available"]:continue
        if sha(ref["trace"])!=ref["trace_sha256"]:raise ValueError("Historical trace changed.")
        if (directory/"candidates.json").exists():continue
        payload=read(ref["trace"]);config=config_from_dict(payload["configuration"])
        events=[ControllerCallV2.from_dict(e) for e in payload["events"]];event=events[ref["event_index"]]
        controller=replay_forced_history(config,events,ref["event_index"],decision_time=event.at_seconds)
        controller._wind=event.wind or controller._wind
        production,required,service=candidate_paths(controller._belief,config.session,ScanRequestV1(12.),controller._wind,
            controller.visit_times,**lifecycle(controller,event))
        winner=next((i for i,c in enumerate(production) if c.plan.samples==event.plan.samples),None)
        if winner is None:raise ValueError("Frozen production candidate family cannot reproduce saved winner.")
        previous=next(e.plan for e in reversed(events[:ref["event_index"]]) if e.operation=="plan_next" and e.plan.samples)
        all_candidates=list(production)
        if hasattr(config.session,"motion"):
            excess=[(event.plan.risk_candidates[i].expected_integrated_map_risk,i) for i,c in enumerate(production) if c.name.startswith("expected_excess")]
            # Field name checked against typed transport below.
            chosen={winner}
            if excess:chosen.add(min(excess)[1])
            chosen.update(i for i,c in enumerate(production) if c.name=="declared_corridor")
            all_candidates=[production[i] for i in sorted(chosen)]
        else:
            all_candidates+=extra_candidates(controller,config,event,required,previous,production)
        rows=[{"name":c.name,"core_cells":[int(v) for v in c.core_cells],"plan":c.plan.to_dict(),
               "production":c in production,"saved_winner":c.plan.samples==event.plan.samples} for c in all_candidates]
        assert len(rows)<=22
        for transport in ("legacy_clipped_v1","zero_inflow_v1"):
            bound=with_transport_diagnostic(config,transport)
            replay=replay_forced_history(bound,events,ref["event_index"],decision_time=event.at_seconds)
            replay._wind=event.wind or replay._wind
            immutable(directory/f"{transport}-configuration.json",bound.to_dict())
            immutable(directory/f"{transport}-snapshot.json",replay.decision_snapshot().to_dict())
        immutable(directory/"candidates.json",{"required":list(required),"service":service,"candidates":rows,
            "production_count":len(production),"declaration_sha256":sha(root/"declaration.json")})
        print("frozen",name,len(rows),"candidates",flush=True)
    return declaration


def load_fixture(root,name,transport):
    directory=Path(root)/"fixtures"/name
    config=config_from_dict(read(directory/f"{transport}-configuration.json"))
    snapshot=DecisionSnapshot.from_dict(read(directory/f"{transport}-snapshot.json"))
    payload=read(directory/"candidates.json")
    candidates=[CandidatePath(row["name"],tuple(row["core_cells"]),ScanPlanV2.from_dict(row["plan"])) for row in payload["candidates"]]
    return directory,config,snapshot,payload,candidates


def pattern(controller,config,candidate,required):
    last=controller._last_detector_reading
    return repeated_pattern(candidate,config.session,controller.visit_times,controller._rotation,
        controller._last_survey_completed,required,horizon=12.,last_reading=last.acquired_at_seconds if last else None,
        last_integration=last.integration_seconds if last else None)


def score_fixture(root,name,transport):
    directory,config,snapshot,payload,candidates=load_fixture(root,name,transport)
    output=directory/f"{transport}-scores.json"
    if output.exists():
        saved=read(output)
        if saved['snapshot_hash']!=snapshot.digest:raise ValueError('Saved scores use a different snapshot')
        return saved
    controller=snapshot.fork_controller(SensingController);rows=[]
    for candidate in candidates:
        score=score_pattern(controller._belief,config.session,controller._wind,
             pattern(controller,config,candidate,tuple(payload["required"])),12.,seed=0,include_history_samples=True)
        rows.append({"name":candidate.name,**score})
    idle=score_pattern(controller._belief,config.session,controller._wind,(),12.,seed=0)
    result={"snapshot_hash":snapshot.digest,"producing_revision":revision(),"candidates":rows,"idle":idle}
    write_json(output,result);return result


def run_fixture(root,name,transport,seeds):
    directory,config,snapshot,payload,candidates=load_fixture(root,name,transport)
    score_fixture(root,name,transport)
    controller=snapshot.fork_controller(SensingController);q=TruthQuadrature(config.source.build(),controller._last_available,12.,config.model.grid_shape)
    for continuation in ("idle","repeat","replan"):
        choices=[(None,None)] if continuation=="idle" else list(enumerate(candidates))
        for index,candidate in choices:
            for seed in (range(1) if continuation=="idle" else range(seeds)):
                output=directory/transport/f"{continuation}-{index if index is not None else 'none'}-seed-{seed}.json"
                if output.exists():
                    saved=read(output)
                    if (saved['snapshot_hash']!=snapshot.digest or saved['declaration_sha256']!=sha(Path(root)/'declaration.json')
                        or saved['candidate_index']!=index or saved['seed']!=seed or saved['continuation']!=continuation):
                        raise ValueError('Saved branch inputs differ; use a separate experiment directory')
                    continue
                start=time.perf_counter()
                result=run_counterfactual(snapshot,config,candidate,tuple(payload["required"]),continuation=continuation,
                                          seed=seed,truth=q)
                result.update(producing_revision=revision(),candidate_index=index,compute_seconds=time.perf_counter()-start,
                              declaration_sha256=sha(Path(root)/"declaration.json"))
                write_json(output,result)
                print(name,transport,continuation,index,seed,"J",round(result["J"],6),flush=True)
    return {"fixture":name,"transport":transport,"seeds":seeds}


@lru_cache(maxsize=2)
def branch_context(root,name,transport):
    directory,config,snapshot,payload,candidates=load_fixture(root,name,transport)
    # Scores must already be frozen; workers never race to create shared scores.
    scores=read(directory/f'{transport}-scores.json')
    if scores['snapshot_hash']!=snapshot.digest:raise ValueError('Scoring snapshot differs')
    controller=snapshot.fork_controller(SensingController)
    truth=TruthQuadrature(config.source.build(),controller._last_available,12.,config.model.grid_shape)
    return directory,config,snapshot,payload,candidates,truth


def run_branch(root,name,transport,continuation,index,seed):
    """One independent continuation; use only after stopping fixture-level jobs."""
    directory,config,snapshot,payload,candidates,q=branch_context(root,name,transport)
    output=directory/transport/f'{continuation}-{index if index is not None else "none"}-seed-{seed}.json'
    if output.exists():
        saved=read(output)
        if (saved['snapshot_hash']!=snapshot.digest or saved['declaration_sha256']!=sha(Path(root)/'declaration.json')
            or saved['candidate_index']!=index or saved['seed']!=seed or saved['continuation']!=continuation):
            raise ValueError('Saved branch inputs differ')
        return str(output)
    candidate=candidates[index] if index is not None else None;start=time.perf_counter()
    result=run_counterfactual(snapshot,config,candidate,tuple(payload['required']),continuation=continuation,seed=seed,truth=q)
    result.update(producing_revision=revision(),candidate_index=index,compute_seconds=time.perf_counter()-start,
                  declaration_sha256=sha(Path(root)/'declaration.json'))
    write_json(output,result)
    print(name,transport,continuation,index,seed,'J',round(result['J'],6),flush=True)
    return str(output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("outputs/scan-value-diagnosis"))
    p.add_argument("--historical",type=Path,default=Path("outputs/reconstruction-risk-v4"))
    p.add_argument("--phase",choices=("declare","score","run","branches"),required=True)
    p.add_argument("--workers",type=int,default=2);p.add_argument("--seeds",type=int,choices=(8,16,32),default=8)
    p.add_argument("--fixture");p.add_argument("--include-mixture",action="store_true")
    p.add_argument("--transport",choices=("legacy_clipped_v1","zero_inflow_v1"))
    args=p.parse_args()
    if args.phase=="declare":declare(args.output,args.historical);return
    declaration=read(args.output/"declaration.json")
    jobs=[(name,transport) for name,ref in declaration["fixtures"].items() if ref["available"]
          and (args.fixture is None or name==args.fixture) and (args.include_mixture or "-motion-risk" not in name)
          for transport in ("legacy_clipped_v1","zero_inflow_v1") if args.transport is None or args.transport==transport]
    if args.phase=='branches':
        # Never use this concurrently with --phase run on the same fixtures.
        tasks=[]
        for name,transport in jobs:
            directory=args.output/'fixtures'/name;count=len(read(directory/'candidates.json')['candidates'])
            snapshot_hash=read(directory/f'{transport}-scores.json')['snapshot_hash']
            declaration_hash=sha(args.output/'declaration.json')
            for continuation in ('idle','repeat','replan'):
                for index in ([None] if continuation=='idle' else range(count)):
                    for seed in range(1 if continuation=='idle' else args.seeds):
                        path=directory/transport/f'{continuation}-{index if index is not None else "none"}-seed-{seed}.json'
                        if not path.exists():tasks.append((str(args.output),name,transport,continuation,index,seed))
                        else:
                            saved=read(path)
                            if (saved['snapshot_hash']!=snapshot_hash or saved['declaration_sha256']!=declaration_hash
                                or saved['candidate_index']!=index or saved['seed']!=seed or saved['continuation']!=continuation):
                                raise ValueError('Saved branch inputs differ')
        print('Independent missing continuations',len(tasks),flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_branch,*task) for task in tasks]
            for future in as_completed(futures):future.result()
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(score_fixture if args.phase=="score" else run_fixture,
            str(args.output),name,transport,*(() if args.phase=="score" else (args.seeds,))) for name,transport in jobs]
        for future in as_completed(futures):print("finished",future.result().get("fixture","score"),flush=True)


if __name__=="__main__":main()
