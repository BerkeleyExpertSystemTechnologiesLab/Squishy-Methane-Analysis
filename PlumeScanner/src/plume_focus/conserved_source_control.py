"""Evaluator-only source shape/amount factorial controls on the discrete grid."""
from dataclasses import replace
from types import SimpleNamespace
import numpy as np
from plume_focus.signal_belief import grid_positions
from plume_focus.mismatch_control import generate_mismatch


def injection(model,source):
    positions=grid_positions(model.grid_shape)
    return model.source_rate*np.exp(-np.sum((positions-np.asarray(source))**2,axis=1)/(2*model.source_sigma**2))


def source_control_model(model,source,width_multiplier,mode):
    if mode not in ('matched','fixed_peak','fixed_total','amount_only'):
        raise ValueError('Unknown source control mode')
    if not np.isfinite(width_multiplier) or width_multiplier<=0 or (mode=='matched' and width_multiplier!=1):
        raise ValueError('Invalid width multiplier')
    source=np.asarray(source,float)
    if source.shape!=(2,) or not np.all(np.isfinite(source)) or np.any((source<0)|(source>1)):
        raise ValueError('Source must be a normalized y,x pair')
    baseline=float(injection(model,source).mean())
    wide=replace(model,source_sigma=model.source_sigma*width_multiplier)
    changed=float(injection(wide,source).mean())
    if baseline<=0 or changed<=0:raise ValueError('Discrete source input must be positive')
    ratio=changed/baseline
    if mode=='matched':result=model
    elif mode=='fixed_peak':result=wide
    elif mode=='fixed_total':result=replace(wide,source_rate=model.source_rate/ratio)
    else:result=replace(model,source_rate=model.source_rate*ratio)
    return result,{'mode':mode,'width_multiplier':width_multiplier,
        'baseline_discrete_input':baseline,'fixed_peak_discrete_input':changed,
        'fixed_peak_input_ratio':ratio,'actual_discrete_input':float(injection(result,source).mean()),
        'actual_peak_rate_ratio':result.source_rate/model.source_rate,
        'actual_width_ratio':result.source_sigma/model.source_sigma}


def generate_source_control(record,config,run,seed,mode,width_multiplier):
    model,details=source_control_model(run.model,config['accurate_center_yx'],width_multiplier,mode)
    truth_run=SimpleNamespace(model=model,session=run.session,duration_seconds=run.duration_seconds)
    events,states,metadata=generate_mismatch(record,config,truth_run,seed,'matched',1.)
    metadata.update(details)
    target=details['baseline_discrete_input'] if mode in ('matched','fixed_total') else details['fixed_peak_discrete_input']
    relative_error=abs(metadata['discrete_source_input_mean']/target-1)
    assert relative_error<5e-14,relative_error
    metadata['target_input_relative_error']=relative_error
    return events,states,metadata
