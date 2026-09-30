"""Offline verification and unchanged-layout media for completed v4 executions."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from plume_focus.sensing_cli_v1 import write_json
from plume_focus.sensing_reconstruction import reconstruction_loss
from plume_focus.sensing_rendering import RenderSettings, render_sensing_video
from plume_focus.sensing_messages_v2 import ScanPlanV2
from plume_focus.sensing_messages_v3 import ScanPlanV3
from plume_focus.sensing_simulation import FakeScanner


INSPECTION_CASES = tuple(
    f'{profile}-wind-error-threshold90_segment_v2-seed-0-{estimator}'
    for profile in ('coarse-full-startup','double-prior-startup')
    for estimator in ('baseline','motion')) + ('diagnostic-source-free-biased-scale-0-seed-0-motion',)


def clean_sample_diagnostics(config, payload):
    """Evaluator-only clean scalar values on actual recorded synthetic motion.

The existing simulator's exposure/footprint quadrature is reused with zero
noise. These readings are never delivered to a controller.
"""
    source=config.source.build();scanner=FakeScanner(source,noise_std=0.)
    actual={r['reading_id']:r for event in payload['events'] for r in event['readings']}
    samples=[]
    for execution in payload['executions']:
        raw=execution['executed_plan'];cls=ScanPlanV3 if raw['schema_version']==3 else ScanPlanV2
        clean=scanner.execute(cls.from_dict(raw))
        for reading in clean.readings:
            observed=actual[reading.reading_id]
            if observed['acquired_at_seconds']!=reading.acquired_at_seconds or tuple(observed['position_yx'])!=reading.position_yx:
                raise ValueError('Clean diagnostic cannot match actual acquisition.')
            samples.append({'scan_id':reading.scan_id,'reading_id':reading.reading_id,
                'acquired_at_seconds':reading.acquired_at_seconds,'position_yx':reading.position_yx,
                'clean_scalar':reading.value,'observed_scalar':(observed['value']-config.model.offset)/config.model.gain})
    values=np.array([r['clean_scalar'] for r in samples]);threshold=config.session.policy.reportable_threshold
    return {'role':'evaluator_only_acquisition_diagnostic','count':len(samples),'samples':samples,
        'clean_scalar_quantiles':dict(zip(('minimum','median','p90','p99','maximum'),np.quantile(values,[0,.5,.9,.99,1]).tolist())) if len(values) else None,
        'fraction_at_or_below_reportable_threshold':float(np.mean(values<=threshold)) if len(values) else None,
        'fraction_near_numerical_zero':float(np.mean(values<=1e-12)) if len(values) else None,
        'reportable_threshold':threshold,'numerical_zero_tolerance':1e-12,'units':config.model.signal_units,
        'note':'Near numerical zero is an observer diagnostic, not a new allocation/detection threshold. Below the reporting threshold can include genuine weak plume signal.'}


def evaluation_refinement(run, directory, revision):
    grid=run.config.evaluation; truth=run.config.source.build()
    previous=reconstruction_loss(run,truth,grid=grid,distribution_diagnostics=False)
    records=[]; status='incomplete'
    for _ in range(2):
        finer=replace(grid,shape=tuple(2*n for n in grid.shape),step_seconds=grid.step_seconds/2)
        current=reconstruction_loss(run,truth,grid=finer,distribution_diagnostics=False)
        change=abs(current['J']-previous['J'])
        passed=change<=max(1e-10,.01*abs(current['J']))
        records.append({'coarse':grid.to_dict(),'fine':finer.to_dict(),
            'coarse_J':previous['J'],'fine_J':current['J'],'absolute_change':change,'passed':passed})
        if passed:status='converged';break
        previous,grid=current,finer
    result={'status':'converged' if records[0]['passed'] else 'incomplete',
            'refined_mesh_status':status,'checks':records,'operational_hash':run.operational_hash,
            'producing_revision':revision,'comparison_mesh':run.config.evaluation.to_dict(),
            'note':'All comparisons retain the declared common mesh. Convergence only at a finer mesh does not certify the retained comparison mesh.'}
    write_json(directory/'evaluation-refinement.json',result)
    return result


def render_inspection(run,directory,revision):
    directory=Path(directory); root=directory/'inspection-frames';root.mkdir(exist_ok=True)
    manifest=directory/'inspection.json'
    records=json.loads(manifest.read_text(encoding='utf-8'))['videos'] if manifest.exists() else []
    digest=run.operational_hash
    for mode in ('physical','summary'):
        path=directory/f'{mode}.mp4'
        previous=next((r for r in records if r['mode']==mode),None)
        if previous:
            if previous['operational_hash']!=digest or hashlib.sha256(path.read_bytes()).hexdigest()!=previous['sha256']:
                raise ValueError(f'Existing inspection evidence differs: {path}')
            continue
        if path.exists():
            raise ValueError(f'Preserve existing inspection media: {path}')
        settings=RenderSettings(mode=mode,fps=12.,playback_speed=1.)
        rendered=render_sensing_video(run,path,source=run.config.source.build(),settings=settings)
        first=run.scans[0].execution.receipt.received_at_seconds
        selected=({0,int(first*6),int(first*12),int(45*12),rendered['encoded_frames']-1}
                  if mode=='physical' else {0,rendered['encoded_frames']//2,rendered['encoded_frames']-1})
        capture=cv2.VideoCapture(str(path));index=0;frames=[]
        while True:
            ok,frame=capture.read()
            if not ok:break
            if index in selected:
                output=root/f'{mode}-{index:04d}.png';cv2.imwrite(str(output),frame)
                frames.append({'frame_index':index,'path':str(output.relative_to(directory)).replace('\\','/')})
            index+=1
        capture.release()
        if index!=rendered['encoded_frames'] or run.operational_hash!=digest:
            raise ValueError('Video decode or operational-hash check failed.')
        records.append({**rendered,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                        'decoded_frames':index,'inspection_frames':frames,'producing_revision':revision})
        write_json(directory/'inspection.json',{'layout':'Existing six panels; policy/status labels only',
                   'operational_hash':digest,'videos':records})
    return records
