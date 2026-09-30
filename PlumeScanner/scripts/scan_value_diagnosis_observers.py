"""Saved-branch quadrature, exposure ablation and unchanged-layout inspection."""

import copy
from dataclasses import replace
import hashlib
from pathlib import Path

import cv2
import numpy as np

from plume_focus.controller import SensingController
from plume_focus.sensing_counterfactual import TruthQuadrature, evaluate_saved_branch, begin_forced_scan
from plume_focus.sensing_contracts import SensorReading, ExecutionReceipt
from plume_focus.sensing_messages_v3 import ScanPlanV3
from plume_focus.sensing_simulation import FakeScanner, SimulatedExecution, SensingScanTrace
from plume_focus.signal_belief import footprint_points, BeliefView
from plume_focus.sensing_execution_v1 import SensingRunV1
from plume_focus.sensing_rendering import compose_sensing_frame, RenderSettings
from plume_focus.physical_scan import coverage_mask
from plume_focus.sensing_cli_v1 import write_json
from scan_value_diagnosis_checks import runner


def evaluation_refinement(root,name,transport):
    r=runner();directory,config,snapshot,payload,candidates=r.load_fixture(root,name,transport)
    scores=r.read(directory/f'{transport}-scores.json');winner=int(np.argmin([s['risk'] for s in scores['candidates']]))
    branch=directory/transport/f'repeat-{winner}-seed-0.json'
    if not branch.exists():return {'status':'pending'}
    output=directory/f'{transport}-evaluation-refinement.json'
    if output.exists():return r.read(output)
    record=r.read(branch);previous=record['J'];checks=[];accepted=None
    for level in (1,2):
        shape=tuple(n*2**level for n in (120,160));step=.05/2**level
        truth=TruthQuadrature(config.source.build(),record['start'],12.,config.model.grid_shape,shape,step)
        current=evaluate_saved_branch(snapshot,record,truth)['J']
        error=abs(current-previous);passed=error<=max(1e-10,.01*abs(current))
        checks.append({'shape':shape,'step':step,'J':current,'previous_J':previous,'absolute_change':error,'passed':passed})
        if passed:accepted=level-1;break
        previous=current
    result={'status':'converged' if accepted is not None else 'numerically_unresolved','accepted_level':accepted,
            'branch_sha256':r.sha(branch),'snapshot_hash':snapshot.digest,'checks':checks,'producing_revision':r.revision()}
    write_json(output,result);return result


def projection_floor(root,name):
    """Evaluator-only representation error, common to every action at this time."""
    r=runner();directory,config,snapshot,payload,candidates=r.load_fixture(root,name,'legacy_clipped_v1')
    output=directory/'projection-floor.json'
    if output.exists():return r.read(output)
    start=snapshot.fork_controller(SensingController)._last_available
    q=TruthQuadrature(config.source.build(),start,12.,config.model.grid_shape)
    optimal=np.linalg.solve(q.gram,q.cross.T).T
    residual=np.maximum(0,np.asarray(q.square)-np.sum(q.cross*optimal,axis=1))
    result={'artifact_type':'evaluator_only_projection_floor_v1','start':start,'horizon':12.,
            'J_lower_bound':float(residual@q.durations),'J_over_T_lower_bound':float(residual@q.durations/12),
            'latent_shape':config.model.grid_shape,'evaluation_shape':q.shape,'evaluation_step':q.step,
            'meaning':'Best bilinear latent-grid map at each instant, using evaluator truth; not an executable estimator or controller input.'}
    write_json(output,result);return result


def midpoint_ablation(root,name,transport):
    r=runner();directory,config,snapshot,payload,candidates=r.load_fixture(root,name,transport)
    output=directory/f'{transport}-exposure-rank-ablation.json'
    if output.exists():return r.read(output)
    q=TruthQuadrature(config.source.build(),snapshot.fork_controller(SensingController)._last_available,12.,config.model.grid_shape)
    clean_scanner=FakeScanner(config.source.build(),noise_std=0,gain=config.model.gain,offset=config.model.offset)
    # Geometry/time/source determine this correction; the same cached correction
    # is added to every seed's reading, preserving its original additive noise.
    corrections={};rows=[]
    for i,candidate in enumerate(candidates):
        for seed in range(8):
            path=directory/transport/f'repeat-{i}-seed-{seed}.json'
            if not path.exists():return {'status':'pending'}
            record=r.read(path);changed=copy.deepcopy(record);active={}
            for event in changed['events']:
                if event['operation'] in ('forced_plan','plan_next'):
                    plan=ScanPlanV3.from_dict(event['plan'])
                    key=tuple((s.position_yx,s.acquired_at_seconds) for s in plan.samples)
                    if key not in corrections:
                        values={}
                        for reading in clean_scanner.execute(plan).readings:
                            points=footprint_points(reading.position_yx,reading.footprint)
                            midpoint=config.source.build().signal_at(np.full(len(points),reading.acquired_at_seconds),points).mean()
                            values[(reading.position_yx,reading.acquired_at_seconds)]=float(config.model.offset+config.model.gain*midpoint-reading.value)
                        corrections[key]=values
                    active=corrections[key]
                elif event['operation']=='ingest_readings':
                    for reading in event['readings']:
                        reading['value']+=active[(tuple(reading['position_yx']),reading['acquired_at_seconds'])]
            revised=evaluate_saved_branch(snapshot,changed,q)
            rows.append({'candidate':i,'seed':seed,'exposure_J':record['J'],'midpoint_J':revised['J']})
    a=np.array([[row['exposure_J'] for row in rows if row['candidate']==i] for i in range(len(candidates))])
    b=np.array([[row['midpoint_J'] for row in rows if row['candidate']==i] for i in range(len(candidates))])
    old=int(np.argmin(a.mean(axis=1)));new=int(np.argmin(b.mean(axis=1)))
    delta=max(1e-10,.01*abs(float(b[new].mean())))
    regret=float(b[old].mean()-b[new].mean())
    result={'status':'complete','diagnostic':'fixed_actions_midpoint_reading_ablation','rows':rows,
            'exposure_winner':old,'midpoint_winner':new,'winner_regret':regret,'material_rank_change':regret>delta,
            'material_threshold':delta,'maximum_absolute_mean_loss_change':float(np.max(np.abs(a.mean(axis=1)-b.mean(axis=1)))),
            'producing_revision':r.revision(),'limitation':'Forced fixed patterns only; this does not change instrument integration or production inference.'}
    write_json(output,result);return result


def apply_event(controller,event,end):
    at=event['at_seconds']
    if event['operation'] in ('forced_plan','plan_next'):
        begin_forced_scan(controller,ScanPlanV3.from_dict(event['plan']),controller._wind,end-at)
    elif event['operation']=='ingest_readings':
        controller.ingest_readings([SensorReading.from_dict(r) for r in event['readings']],available_at_seconds=at)
    else:controller.finish_scan(ExecutionReceipt.from_dict(event['receipt']))


def inspection_run(config,snapshot,record):
    controller=snapshot.fork_controller(SensingController);traces=[];end=record['start']+record['horizon']
    for event in record['events']:
        at=event['at_seconds']
        if event['operation'] in ('forced_plan','plan_next'):
            plan=ScanPlanV3.from_dict(event['plan']);prior=controller.predictive_view(at);readings=[]
            distribution=controller.predictive_distribution(plan.completed_at_seconds,config.capabilities.lattice_shape)
            density=distribution.expected_excess(config.session.allocation_threshold)
            mass=density/density.sum() if density.sum()>config.session.policy.mass_floor else np.zeros_like(density)
            predicted=BeliefView(plan.completed_at_seconds,distribution.mean,np.sqrt(distribution.variance),mass)
            swept=coverage_mask(plan,mass.shape,config.capabilities)
            visual_plan=replace(plan,predicted_mass_recall=float(mass[swept].sum()) if mass.sum()>0 else None,
                                expected_signal_total=float(density.mean()))
        elif event['operation']=='ingest_readings':
            readings.extend(SensorReading.from_dict(r) for r in event['readings'])
        apply_event(controller,event,end)
        if event['operation']=='finish_scan':
            receipt=ExecutionReceipt.from_dict(event['receipt'])
            execution=SimulatedExecution(tuple(readings),receipt,plan,False)
            traces.append(SensingScanTrace(visual_plan,execution,prior,predicted,controller.belief_view(),
                           controller.assimilation_events,controller._wind,distribution,controller.motion_diagnostics()))
    return SensingRunV1(config,tuple(traces),(),'forced_counterfactual_v1',controller.startup_complete,
                        float(np.mean(controller.visit_times>=0)))


def render_branch(root,name,transport,index,*,continuation='repeat',fps=8.,speed=1.,variant='',views=('physical','summary')):
    r=runner();directory,config,snapshot,payload,candidates=r.load_fixture(root,name,transport)
    branch=directory/transport/f'{continuation}-{index}-seed-0.json'
    if not branch.exists():return {'status':'pending'}
    record=r.read(branch);run=inspection_run(config,snapshot,record)
    output=Path(root)/'inspection'/name/transport
    if variant:output=output/variant
    output.mkdir(parents=True,exist_ok=True)
    files=[];end=record['start']+record['horizon']
    for mode in views:
        settings=RenderSettings(mode=mode,fps=fps,playback_speed=speed)
        if mode=='summary':times=[s.execution.receipt.received_at_seconds for s in run.scans]
        else:times=list(np.minimum(record['start']+np.arange(int(np.ceil(12*fps/speed))+1)*speed/fps,end))
        controller=snapshot.fork_controller(SensingController);event_index=0;writer=None;path=output/f'{continuation}-{index}-{mode}.mp4'
        for frame_index,at in enumerate(times):
            while event_index<len(record['events']) and record['events'][event_index]['at_seconds']<=at+1e-10:
                apply_event(controller,record['events'][event_index],end);event_index+=1
            trace=next((s for s in run.scans if at<=s.execution.receipt.received_at_seconds+1e-10),run.scans[-1])
            frame=compose_sensing_frame(trace,run,float(at),settings,config.source.build(),causal_belief=controller.predictive_view(float(at)))
            # Label the intervention in the existing header; preserve all six panels.
            cv2.rectangle(frame,(0,0),(frame.shape[1],27),(12,19,29),-1)
            view_label='PHYSICAL TIME' if mode=='physical' else 'COMPRESSED SUMMARY'
            suffix=f'{speed:g}x' if mode=='physical' else f'{fps:g} fps; 1 frame/sweep'
            label=f'FORCED DIAGNOSTIC / {view_label} / {transport} / {suffix}'
            cv2.putText(frame,label,(14,23),cv2.FONT_HERSHEY_SIMPLEX,.55,(230,240,250),1,cv2.LINE_AA)
            if writer is None:
                writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(frame.shape[1],frame.shape[0]))
                if not writer.isOpened():raise ValueError('Video writer unavailable')
            writer.write(frame)
            if frame_index in (0,len(times)//2,len(times)-1):
                cv2.imwrite(str(output/f'{continuation}-{index}-{mode}-{frame_index:03d}.png'),frame)
        if writer:writer.release()
        capture=cv2.VideoCapture(str(path));decoded=0
        while capture.read()[0]:decoded+=1
        capture.release()
        if decoded!=len(times):raise ValueError('Inspection video is incomplete')
        files.append({'path':str(path.resolve()),'sha256':r.sha(path),'frames':decoded,'fps':fps,'speed':speed})
    manifest=output/f'{continuation}-{index}-manifest.json'
    if manifest.exists() and tuple(views)!=('physical','summary'):
        previous=r.read(manifest)
        if previous['branch_sha256']!=r.sha(branch):raise ValueError('Cannot merge media from different executions')
        names={f['path'] for f in files}
        files=[f for f in previous['files'] if f['path'] not in names]+files
    result={'status':'rendered_pending_visual_review','branch_sha256':r.sha(branch),'snapshot_hash':snapshot.digest,'files':files}
    write_json(manifest,result);return result


def same_initial_transport(root,name):
    """Counterfactual propagation from identical moments, not different filters."""
    r=runner();directory,config,snapshot,payload,candidates=r.load_fixture(root,name,'legacy_clipped_v1')
    output=directory/'same-initial-transport.json'
    if output.exists():return r.read(output)
    controller=snapshot.fork_controller(SensingController);rows=[];start=controller._last_available
    from plume_focus.signal_belief import grid_positions
    points=grid_positions((120,160));source=config.source.build()
    for transport in ('legacy_clipped_v1','zero_inflow_v1'):
        for horizon in (0.,1.,3.,6.,12.):
            state=controller._belief.clone();state.transport_id=transport
            for component in getattr(state,'components',(state,)):component.transport_id=transport
            state.advance(start+horizon,controller._wind,controller.source_position)
            mean,var=state.moments_on((120,160))
            truth=source.signal_at(np.full(len(points),start+horizon),points).reshape(mean.shape)
            rows.append({'transport':transport,'horizon':horizon,'actual_mse':float(np.mean((mean-truth)**2)),
                         'mean_signal':float(mean.mean()),'mean_variance':float(var.mean())})
    result={'artifact_type':'same_initial_moments_transport_probe_v1','snapshot_hash':snapshot.digest,
            'producing_revision':r.revision(),'rows':rows}
    write_json(output,result);return result


def candidate_coverage(root,name,transport):
    """Completion-time coverage diagnostics, separate from causal map loss."""
    from plume_focus.signal_belief import grid_positions
    r=runner();directory,config,snapshot,payload,candidates=r.load_fixture(root,name,transport)
    output=directory/f'{transport}-candidate-coverage.json'
    if output.exists():return r.read(output)
    controller=snapshot.fork_controller(SensingController);rows=[];shape=(120,160);points=grid_positions(shape)
    for index,candidate in enumerate(candidates):
        plan=candidate.plan
        distribution=controller.predictive_distribution(plan.completed_at_seconds,config.capabilities.lattice_shape)
        mass=distribution.expected_excess(config.session.allocation_threshold)*np.asarray(config.session.physical_quadrature_weights).reshape(config.capabilities.lattice_shape)
        predicted_mask=coverage_mask(plan,mass.shape,config.capabilities)
        true=config.source.build().signal_at(np.full(len(points),plan.completed_at_seconds),points).reshape(shape)
        mask=coverage_mask(plan,shape,config.capabilities);excess=np.maximum(true-config.session.allocation_threshold,0.)
        support=true>config.session.allocation_threshold
        rows.append({'candidate':index,'at_seconds':plan.completed_at_seconds,
            'predicted_excess_covered_fraction':float(mass[predicted_mask].sum()/mass.sum()) if mass.sum()>config.session.policy.mass_floor else None,
            'truth_excess_covered_fraction':float(excess[mask].sum()/excess.sum()) if excess.sum()>0 else None,
            'truth_support_covered_fraction':float(np.sum(mask&support)/support.sum()) if support.any() else None,
            'requested_fraction':len(plan.requested_cells)/np.prod(config.capabilities.lattice_shape),
            'swept_fraction':len(plan.swept_cells)/np.prod(config.capabilities.lattice_shape)})
    result={'status':'complete','snapshot_hash':snapshot.digest,'rows':rows,'producing_revision':r.revision(),
        'meaning':'First-sweep completion-time coverage, not accumulated temporal coverage or reconstruction quality; blank truth coverage is unavailable.'}
    write_json(output,result);return result


def contrast_refinement(root,name,transport):
    """Check the numerical margin of positive attribution contrasts on seed 0."""
    r=runner();directory,config,snapshot,payload,candidates=r.load_fixture(root,name,transport)
    summary=r.read(Path(root)/'diagnosis-summary.json')
    row=next((x for x in summary['rows'] if x['fixture']==name and x['transport']==transport and x['status']=='complete'),None)
    if row is None:return {'status':'pending'}
    comparisons=[('H2','repeat',row['production_predicted_winner'],row['empirical_production_winner']),
                 ('H3','repeat',row['empirical_production_winner'],row['empirical_expanded_winner']),
                 ('H4','replan',row['empirical_expanded_winner'],row['empirical_replanning_winner'])]
    # Each fine branch is cached by its own immutable operational hash.
    output=directory/f'{transport}-contrast-refinement.json'
    saved=r.read(output) if output.exists() else {'branches':{}}
    start=snapshot.fork_controller(SensingController)._last_available;truth=None;finest_truth=None;checks=[]
    for hypothesis,mode,a,b in comparisons:
        if a==b:continue
        results=[];digests=[]
        for index in (a,b):
            path=directory/transport/f'{mode}-{index}-seed-0.json';digest=r.sha(path)
            if digest not in saved['branches']:
                if truth is None:truth=TruthQuadrature(config.source.build(),start,12.,config.model.grid_shape,(240,320),.025)
                record=r.read(path);fine=evaluate_saved_branch(snapshot,record,truth)['J']
                saved['branches'][digest]={'coarse_J':record['J'],'fine_J':fine,'path':str(path)}
            results.append(saved['branches'][digest])
            digests.append(digest)
        old=results[0]['coarse_J']-results[1]['coarse_J'];new=results[0]['fine_J']-results[1]['fine_J']
        checks.append({'hypothesis':hypothesis,'continuation':mode,'candidates':[a,b],
                       'coarse_contrast':old,'fine_contrast':new,'contrast_change':abs(new-old),
                       'material_threshold':row['material_threshold'],
                       'within_material_tolerance':abs(new-old)<=row['material_threshold']})
        if abs(new-old)>row['material_threshold']:
            for digest in digests:
                item=saved['branches'][digest]
                if 'finest_J' not in item:
                    if finest_truth is None:finest_truth=TruthQuadrature(config.source.build(),start,12.,config.model.grid_shape,(480,640),.0125)
                    item['finest_J']=evaluate_saved_branch(snapshot,r.read(Path(item['path'])),finest_truth)['J']
            finest=saved['branches'][digests[0]]['finest_J']-saved['branches'][digests[1]]['finest_J']
            checks[-1].update(finest_contrast=finest,fine_to_finest_change=abs(finest-new),
                finest_converged=abs(finest-new)<=row['material_threshold'],
                contrast_change=abs(finest-old),within_material_tolerance=abs(finest-old)<=row['material_threshold'])
    saved.update(status='complete',snapshot_hash=snapshot.digest,checks=checks,producing_revision=r.revision(),
                 limitation='Seed-0 numerical contrast check; noise uncertainty remains a separate paired-seed calculation.')
    write_json(output,saved);return saved


def innovation_audit(root,name,transport):
    """Exact mixture scalar predictive moments/density before actual delivery."""
    from scipy.special import logsumexp
    from plume_focus.sensing_risk import linear_transition
    from plume_focus.signal_belief import observation_weights
    r=runner();directory,config,snapshot,payload,candidates=r.load_fixture(root,name,transport)
    output=directory/f'{transport}-innovation-audit.json'
    if output.exists():return r.read(output)
    rows=[]
    for index in range(len(candidates)):
        path=directory/transport/f'repeat-{index}-seed-0.json'
        if not path.exists():return {'status':'pending'}
        controller=snapshot.fork_controller(SensingController);record=r.read(path)
        for event in record['events']:
            if event['operation']=='ingest_readings':
                assert len(event['readings'])==1,'This audit is for the declared streaming fixtures'
                reading=SensorReading.from_dict(event['readings'][0]);belief=controller._belief
                h=observation_weights(config.model.grid_shape,reading.position_yx,reading.footprint)
                means=[];variances=[]
                for state,offset in zip(belief.components,belief.offsets):
                    velocity=tuple(np.asarray(controller._wind.velocity_yx)+offset)
                    a,b,q=linear_transition(config.model,velocity,controller.source_position,
                        round(reading.acquired_at_seconds-state.time_seconds,12),transport)
                    projected=a.T@h
                    means.append(config.model.offset+config.model.gain*(projected@state.mean+h@b))
                    variances.append(config.model.gain**2*(projected@state.covariance@projected+h@q@h)+config.model.sensor_noise_std**2)
                means=np.asarray(means);variances=np.asarray(variances);weights=belief.weights
                mean=float(weights@means);variance=float(weights@(variances+(means-mean)**2))
                residual=reading.value-mean
                density=float(logsumexp(belief.log_weights-.5*(np.log(2*np.pi*variances)+(reading.value-means)**2/variances)))
                rows.append({'candidate':index,'reading_id':reading.reading_id,'acquired_at_seconds':reading.acquired_at_seconds,
                             'observed':reading.value,'predicted_mean':mean,'predicted_variance':variance,
                             'residual':residual,'standardized_squared_residual':residual**2/variance,
                             'mixture_log_predictive_density':density,
                             'effective_hypotheses_before':float(1/np.sum(weights**2))})
            apply_event(controller,event,record['start']+12)
            if event['operation']=='finish_scan':break
    result={'status':'complete','snapshot_hash':snapshot.digest,'rows':rows,'producing_revision':r.revision(),
            'scope':'First sweep of each frozen mixture alternative, seed 0, actual streaming readings. Standardization uses mixture second moments, not a Gaussian coverage claim.'}
    write_json(output,result);return result


if __name__=='__main__':
    import argparse
    from concurrent.futures import ProcessPoolExecutor,as_completed
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path)
    parser.add_argument('--phase',choices=('evaluation','exposure','projection','same-initial','render','contrasts','innovations','coverage'),required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--fixture');parser.add_argument('--transport',default='legacy_clipped_v1')
    parser.add_argument('--candidate',type=int,default=0);parser.add_argument('--speed',type=float,default=1.)
    parser.add_argument('--fps',type=float,default=8.);parser.add_argument('--variant',default='')
    parser.add_argument('--summary-only',action='store_true')
    args=parser.parse_args()
    if args.phase=='render':
        if not args.fixture:parser.error('Rendering requires --fixture')
        print(render_branch(args.output,args.fixture,args.transport,args.candidate,
                            fps=args.fps,speed=args.speed,variant=args.variant,
                            views=('summary',) if args.summary_only else ('physical','summary')));raise SystemExit
    if args.phase in ('projection','same-initial'):
        names=list(runner().read(args.output/'declaration.json')['fixtures'])
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(projection_floor if args.phase=='projection' else same_initial_transport,
                                 str(args.output),name) for name in names]
            for future in as_completed(futures):print(future.result().get('artifact_type'),flush=True)
        raise SystemExit
    if args.phase in ('contrasts','innovations'):
        names=[name for name in runner().read(args.output/'declaration.json')['fixtures']
               if ('motion-risk' in name)==(args.phase=='innovations')]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(innovation_audit if args.phase=='innovations' else contrast_refinement,
                                 str(args.output),name,transport) for name in names
                     for transport in ('legacy_clipped_v1','zero_inflow_v1')]
            for future in as_completed(futures):print(future.result().get('status'),flush=True)
        raise SystemExit
    if args.phase=='coverage':
        names=list(runner().read(args.output/'declaration.json')['fixtures'])
        if args.fixture:names=[args.fixture]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(candidate_coverage,str(args.output),name,transport) for name in names
                     for transport in ('legacy_clipped_v1','zero_inflow_v1')]
            for future in as_completed(futures):print(future.result().get('status'),flush=True)
        raise SystemExit
    names=['coarse-full-startup-wind-error-threshold90_segment_v2-seed-0-baseline-risk',
           'double-prior-startup-wind-error-threshold90_segment_v2-seed-0-baseline-risk']
    if args.phase=='evaluation':names.append('diagnostic-source-free-biased-scale-0-seed-0-baseline-risk')
    if args.fixture:names=[args.fixture]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(evaluation_refinement if args.phase=='evaluation' else midpoint_ablation,
                 str(args.output),name,transport) for name in names for transport in ('legacy_clipped_v1','zero_inflow_v1')]
        for future in as_completed(futures):print(future.result().get('status'),flush=True)
