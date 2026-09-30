"""Frozen declaration, numerical checks and resumable risk-policy comparison."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

from plume_focus.sensing_cli_v1 import config_from_dict, write_json, _run_job, artifact_manifest
from plume_focus.sensing_execution_v1 import ControllerCallV1, CausalReplayV1
from plume_focus.sensing_execution_v4 import with_risk_policy
from plume_focus.sensing_policy_v4 import candidate_paths, repeated_pattern
from plume_focus.sensing_risk import score_pattern
from plume_focus.sensing_session import configuration_hash
from plume_focus.sensing_execution_v1 import run_sensing_v1
from plume_focus.sensing_reconstruction import reconstruction_loss, summarize_v1
from reconstruction_risk_observers import INSPECTION_CASES, evaluation_refinement, render_inspection
from plume_focus.physical_scan import coverage_mask
from plume_focus.signal_belief import positive_expectation


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def revision():
    return subprocess.run(["git","rev-parse","HEAD"],check=True,capture_output=True,text=True).stdout.strip()


def declare(root, manifest):
    source=Path(manifest["historical_root"])
    if sha(source/"execution-configurations.json")!=manifest["historical_configurations_sha256"]:
        raise ValueError("Historical configurations differ from the approved study.")
    old=read(source/"execution-configurations.json")
    jobs={}
    for name,job in old.items():
        config=config_from_dict(job["configuration"])
        if config.policy=="age_segment_v2":
            continue
        directory=source/"development"/name
        report,trace=read(directory/"report.json"),read(directory/"trace.json")
        if (report["configuration"]!=job["configuration"] or trace["configuration"]!=job["configuration"]
                or configuration_hash(trace)!=report["operational_hash"]
                or report["reconstruction"]["status"]!="complete"):
            raise ValueError(f"Historical execution/configuration mismatch: {name}")
        jobs[name]={**job,"new":False,"source_directory":str(directory),
                    "input_hashes":{f:sha(directory/f) for f in ("configuration.json","report.json","trace.json")}}
        if config.policy=="threshold90_segment_v2":
            candidate=with_risk_policy(config,random_seed=manifest["random_seed"])
            jobs[name+"-risk"]={**job,"new":True,"configuration":candidate.to_dict(),"reference":name}
    assert len(jobs)==96 and sum(j["new"] for j in jobs.values())==36
    for filename,payload in (("declared-study.json",manifest),("declared-configurations.json",jobs)):
        path=root/filename
        if path.exists() and read(path)!=json.loads(json.dumps(payload)):
            raise ValueError(f"Frozen declaration changed: {path}")
        if not path.exists():
            write_json(path,payload)
    return jobs


def references(jobs):
    refs={}
    for estimator in ("baseline","motion"):
        for profile in ("coarse-full-startup","double-prior-startup"):
            name=f"{profile}-wind-error-threshold90_segment_v2-seed-0-{estimator}"
            events=read(Path(jobs[name]["source_directory"])/"trace.json")["events"]
            indices=[i for i,e in enumerate(events) if e["operation"]=="plan_next" and e["plan"]["samples"]]
            chosen=(1,len(indices)//2) if profile.startswith("coarse") else (0,1)
            for n in sorted(set(chosen)):
                refs[f"{name}-decision-{n}"]={"reference":name,"event_index":indices[n],"estimator":estimator}
        name=f"diagnostic-source-free-biased-scale-0-seed-0-{estimator}"
        events=read(Path(jobs[name]["source_directory"])/"trace.json")["events"]
        indices=[i for i,e in enumerate(events) if e["operation"]=="plan_next" and e["plan"]["samples"]]
        refs[f"{name}-decision-1"]={"reference":name,"event_index":indices[1],"estimator":estimator}
    return refs


def snapshot(job, ref):
    config=config_from_dict(job["configuration"])
    events=tuple(ControllerCallV1.from_dict(e) for e in read(Path(job["source_directory"])/"trace.json")["events"])
    decision=events[ref["event_index"]]
    replay=CausalReplayV1(config.session,events[:ref["event_index"]])
    controller=replay.advance_to(decision.at_seconds)
    belief=controller._belief.clone()
    belief.advance(decision.at_seconds,controller._wind,controller.source_position)
    return config,controller,belief,decision


def checks(coarse,fine,manifest):
    a,b=np.asarray(coarse),np.asarray(fine)
    tolerance=np.maximum(manifest["absolute_tolerance"],manifest["relative_tolerance"]*np.abs(b))
    selected=int(np.argmin(a));regret=float(b[selected]-b.min())
    return {"passed":bool(np.all(np.abs(a-b)<=tolerance) and regret<=max(manifest["absolute_tolerance"],manifest["relative_tolerance"]*abs(float(b.min())))),
            "maximum_absolute_change":float(np.max(np.abs(a-b))),"coarse_selected":selected,
            "fine_selected":int(np.argmin(b)),"selected_candidate_regret":regret}


def refine_one(name,ref,job,manifest,output,required_histories=32,required_step=.1):
    path=Path(output)/f"{name}.json"
    saved=read(path) if path.exists() else None
    if (saved and saved.get('numerical_pass_version')==2 and
            saved.get('checked_history_count',128)>=required_histories and saved['accepted_step_seconds']<=required_step):
        return saved
    if saved:
        archive=path.parent.parent/'refinement-initial'/path.name
        archive.parent.mkdir(parents=True,exist_ok=True)
        if not archive.exists():shutil.copyfile(path,archive)
    started=time.perf_counter()
    base,controller,belief,event=snapshot(job,ref)
    cfg=with_risk_policy(base,random_seed=manifest["random_seed"])
    now,wind=event.at_seconds,event.wind or controller._wind
    last=controller._last_detector_reading
    common=dict(scan_id=event.plan.scan_id,now=now,pose=event.position_yx,rotation=controller._rotation,
        startup_complete=controller.startup_complete,last_survey_completed=controller._last_survey_completed,
        last_reading_at_seconds=last.acquired_at_seconds if last else None,
        last_integration_seconds=last.integration_seconds if last else None)
    candidates,required,service=candidate_paths(belief,cfg.session,event.request,wind,controller.visit_times,**common)
    if not candidates:
        raise ValueError(f"No candidates for refinement fixture: {name}")
    patterns=[repeated_pattern(c,cfg.session,controller.visit_times,controller._rotation,controller._last_survey_completed,
        required,horizon=event.request.max_duration_seconds) for c in candidates]
    seed=int.from_bytes(hashlib.sha256(f"{cfg.session.policy.random_seed}:{event.plan.scan_id}:{now:.17g}".encode()).digest()[:8],"little")
    checkpoint=path.with_suffix('.partial.json')
    results=(read(checkpoint)['results'] if checkpoint.exists() else saved['results'] if saved else {})
    def evaluate(step,histories):
        key=f"{step:g}/{histories}"
        if key not in results and f'{step:g}/64' in results and not hasattr(cfg.session,'motion'):
            results[key]=results[f'{step:g}/64']
        if key not in results or len(results[key])<len(patterns):
            session=replace(cfg.session,policy=replace(cfg.session.policy,risk_step_seconds=step,history_count=histories))
            rows=results.setdefault(key,[])
            for index in range(len(rows),len(patterns)):
                pattern=patterns[index]
                rows.append(score_pattern(belief,session,wind,pattern,event.request.max_duration_seconds,seed=seed,include_history_samples=True))
                write_json(checkpoint,{'reference':ref,'results':results})
                print(name,key,"candidate",index+1,"/",len(patterns),flush=True)
            results[key]=rows
        return results[key]
    def values(rows,histories):
        return [row['within']+(float(np.mean(row['between_history_integrals'][:histories])) if row['histories'] else 0.) for row in rows]
    time_checks=[];accepted_step=manifest['risk_steps_seconds'][-1];time_status='incomplete'
    previous=None
    time_histories=max(64,required_histories)
    for step in manifest['risk_steps_seconds']:
        rows=evaluate(step,time_histories)
        if previous is not None:
            by_count={str(n):checks(values(previous[1],n),values(rows,n),manifest) for n in (32,64,128) if n<=time_histories}
            check={**by_count[str(time_histories)],'passed':all(c['passed'] for c in by_count.values()),'by_history_count':by_count}
            time_checks.append({'coarse':previous[0],'fine':step,**check})
            if check['passed'] and step<=required_step:
                accepted_step=min(previous[0],required_step);time_status='converged';break
        previous=(step,rows)
    rows=evaluate(accepted_step,time_histories)
    history_checks=[{'coarse':32,'fine':64,**checks(values(rows,32),values(rows,64),manifest)}]
    accepted_histories=32;history_status='converged'
    if not history_checks[-1]['passed']:
        fine=evaluate(accepted_step,128)
        history_checks.append({'coarse':64,'fine':128,**checks(values(rows,64),values(fine,128),manifest)})
        accepted_histories=64 if history_checks[-1]['passed'] else 128
        history_status='converged' if history_checks[-1]['passed'] else 'incomplete'
    gap_coverage=None
    if name=='double-prior-startup-wind-error-threshold90_segment_v2-seed-0-motion-decision-1':
        gap_coverage=[];shape=cfg.session.instrument.lattice_shape
        for candidate in candidates:
            prediction=belief.clone();prediction.advance(candidate.plan.completed_at_seconds,wind,cfg.session.source_position)
            if hasattr(prediction,'distribution_on'):
                density=prediction.distribution_on(shape).expected_excess(cfg.session.policy.reportable_threshold)
            else:
                mu,var=prediction.moments_on(shape)
                density=positive_expectation(mu,var,cfg.session.policy.reportable_threshold)
            mass=density*np.asarray(cfg.session.physical_quadrature_weights).reshape(shape)
            mask=coverage_mask(candidate.plan,shape,cfg.capabilities)
            gap_coverage.append(float(mass[mask].sum()/mass.sum()) if mass.sum()>cfg.session.policy.mass_floor else None)
    result={'name':name,'reference':ref,'decision_seconds':now,'horizon_seconds':event.request.max_duration_seconds,
        'saved_budget_gap_candidate_coverage':gap_coverage,
        'service_status':service,'required_cells':required,'candidates':[{'proposal':c.name,'duration_seconds':c.plan.duration_seconds,'requested_cells':c.plan.requested_cells} for c in candidates],
        'results':results,'time_checks':time_checks,'history_checks':history_checks,'accepted_step_seconds':accepted_step,
        'accepted_history_count':accepted_histories,'time_status':time_status,'history_status':history_status,
        'producing_revision':revision(),'elapsed_compute_seconds':time.perf_counter()-started,
        'numerical_pass_version':2,'checked_history_count':time_histories}
    write_json(path,result);return result


def refine(root,jobs,manifest,workers,estimator=None,completed_only=False):
    refs=references(jobs)
    selected={k:v for k,v in refs.items() if estimator is None or v['estimator']==estimator}
    if completed_only:
        selected={k:v for k,v in selected.items() if (root/'refinement'/f'{k}.json').exists()}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(refine_one,n,r,jobs[r['reference']],manifest,str(root/'refinement')):n for n,r in selected.items()}
        for future in as_completed(pending):
            result=future.result();print('REFINED',result['name'],result['time_status'],result['history_status'],flush=True)
    if all((root/'refinement'/f'{name}.json').exists() for name in refs):
        results=[read(root/'refinement'/f'{name}.json') for name in refs]
        while True:
            common_count=max(r['accepted_history_count'] for r in results)
            common_step=min(r['accepted_step_seconds'] for r in results)
            unfinished=[r['name'] for r in results if r.get('numerical_pass_version')!=2 or
                r.get('checked_history_count',128)<common_count or r['accepted_step_seconds']>common_step]
            if not unfinished:break
            with ProcessPoolExecutor(max_workers=workers) as pool:
                pending=[pool.submit(refine_one,n,refs[n],jobs[refs[n]['reference']],manifest,
                    str(root/'refinement'),common_count,common_step) for n in unfinished]
                for future in as_completed(pending):
                    r=future.result();print('JOINT REFINED',r['name'],r['time_status'],r['history_status'],flush=True)
            results=[read(root/'refinement'/f'{name}.json') for name in refs]
        freeze={'risk_step_seconds':min(r['accepted_step_seconds'] for r in results),
                'history_count':max(r['accepted_history_count'] for r in results),
                'status':'converged' if all(r['time_status']==r['history_status']=='converged' for r in results) else 'incomplete',
                'references':list(refs),'joint_time_history_checks':all(r.get('numerical_pass_version')==2 for r in results),
                'mixture_estimator_note':'49 hypotheses retained; prior wind quadrature convergence remains unresolved.'}
        path=root/'numerical-freeze.json'
        if path.exists() and read(path)!=freeze:
            if read(path).get('joint_time_history_checks'):
                raise ValueError('Numerical freeze already differs.')
            if not (root/'initial-numerical-checks.json').exists():
                shutil.copyfile(path,root/'initial-numerical-checks.json')
        write_json(path,freeze)


def execute(root,jobs,workers):
    freeze=read(root/'numerical-freeze.json');resolved={}
    if not freeze.get('joint_time_history_checks'):
        raise ValueError('Complete joint time/history refinement before the execution study.')
    for name,job in jobs.items():
        job=dict(job)
        if job['new']:
            config=config_from_dict(job['configuration'])
            config=replace(config,session=replace(config.session,policy=replace(config.session.policy,
                risk_step_seconds=freeze['risk_step_seconds'],history_count=freeze['history_count'])))
            job['configuration']=config.to_dict()
        resolved[name]=job
    path=root/'execution-configurations.json'
    if path.exists() and read(path)!=json.loads(json.dumps(resolved)):
        raise ValueError('Existing execution configurations differ.')
    write_json(path,resolved)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={}
        for name,job in resolved.items():
            directory=root/'development'/name
            if not job['new']:
                directory.mkdir(parents=True,exist_ok=True)
                for filename,digest in job['input_hashes'].items():
                    source=Path(job['source_directory'])/filename;target=directory/filename
                    if sha(source)!=digest or (target.exists() and sha(target)!=digest):
                        raise ValueError(f'Historical evidence changed: {name}/{filename}')
                    if not target.exists():shutil.copyfile(source,target)
            elif (directory/'report.json').exists():
                if read(directory/'report.json')['configuration']!=json.loads(json.dumps(job['configuration'])):
                    raise ValueError(f'Existing run configuration differs: {name}')
                if name.removesuffix('-risk') in INSPECTION_CASES and (
                    not (directory/'evaluation-refinement.json').exists() or
                    not (directory/'inspection.json').exists() or
                    len(read(directory/'inspection.json')['videos'])!=2):
                    pending[pool.submit(finish_observers,name,job['configuration'],str(root/'development'))]=name
            else:
                pending[pool.submit(run_risk_job,name,job['configuration'],str(root/'development'))]=name
        for future in as_completed(pending):
            name=pending[future];future.result();print('COMPLETED',name,flush=True)
    records=[]
    for name,job in resolved.items():
        r=read(root/'development'/name/'report.json')
        records.append({'name':name,'new':job['new'],'role':job['role'],'estimator':job['estimator'],
            'J_over_T':r['reconstruction']['J_over_T'],'index_MSE':r['reconstruction']['signal_index_squared_error_mean'],
            'actual_scan_seconds':r['total_scan_seconds'],'value_selected_count':r.get('value_selected_count',0),
            'operational_hash':r['operational_hash'],'configuration_hash':configuration_hash(r['configuration'])})
    write_json(root/'study-summary.json',{'engineering_execution':'complete','records':records,'numerical':freeze,
        'claim_status':'not_demonstrated','interpretation':'Descriptive comparison; numerical and scientific conclusions require the separate review.'})
    artifact_manifest(root)


def run_risk_job(name,payload,output):
    """Retain trace before scoring, and reuse warm pure-plan cache for observers."""
    config=config_from_dict(payload); directory=Path(output)/name; producing=revision()
    started=time.perf_counter();print('EXECUTING',name,flush=True)
    run=run_sensing_v1(config);simulated=time.perf_counter()
    write_json(directory/'configuration.json',config.to_dict())
    write_json(directory/'trace.json',run.operational_payload())
    print('EVALUATING',name,flush=True)
    loss=reconstruction_loss(run,config.source.build());evaluated=time.perf_counter()
    report=summarize_v1(run,config.source.build(),loss=loss)
    report['producing_revision']=producing
    report['compute_seconds']={'simulation':simulated-started,'causal_replay_evaluation':evaluated-simulated}
    write_json(directory/'report.json',report)
    if name.removesuffix('-risk') in INSPECTION_CASES:
        print('REFINING EVALUATION',name,flush=True)
        evaluation_refinement(run,directory,producing)
        print('RENDERING',name,flush=True)
        render_inspection(run,directory,producing)
    return name


def finish_observers(name,payload,output):
    config=config_from_dict(payload);directory=Path(output)/name
    run=run_sensing_v1(config)
    if run.operational_hash!=read(directory/'report.json')['operational_hash']:
        raise ValueError(f'Cannot reproduce observer input: {name}')
    if not (directory/'evaluation-refinement.json').exists():
        evaluation_refinement(run,directory,revision())
    render_inspection(run,directory,revision())
    return name


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--phase',choices=('declare','refine','run','all'),default='all')
    parser.add_argument('--estimator',choices=('baseline','motion'))
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--completed-only',action='store_true',help='Upgrade already saved numerical checks without duplicating active jobs.')
    args=parser.parse_args();manifest=read(Path(__file__).resolve().parents[1]/'configs/sensing-risk-v4/study.json')
    jobs=declare(args.output,manifest)
    if args.completed_only and args.phase!='refine':
        raise ValueError('--completed-only applies only to numerical refinement.')
    if args.phase in ('refine','all'):refine(args.output,jobs,manifest,args.workers,args.estimator,args.completed_only)
    if args.phase in ('run','all'):execute(args.output,jobs,args.workers)


if __name__=='__main__':main()
