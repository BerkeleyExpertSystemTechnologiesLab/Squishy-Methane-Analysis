"""Evaluator-only one-factor perturbations of the matched truth generator."""
from dataclasses import replace,asdict
import hashlib,json
import numpy as np
from plume_focus.model_consistent_control import MatchedFieldGenerator
from plume_focus.sensing_contracts import CausalWind,SensorReading


def truth_parameters(model,half_life,factor,multiplier):
    if factor not in ('matched','wind_speed','source_width','half_life'):
        raise ValueError('Unknown mismatch factor')
    if not np.isfinite(multiplier) or multiplier<=0 or (factor=='matched' and multiplier!=1):
        raise ValueError('Invalid mismatch multiplier')
    return (replace(model,source_sigma=model.source_sigma*multiplier) if factor=='source_width' else model,
            half_life*multiplier if factor=='half_life' else half_life,
            multiplier if factor=='wind_speed' else 1.)


def generate_mismatch(record,config,run,seed,factor,multiplier):
    model,half_life,wind_scale=truth_parameters(run.model,run.session.half_life_seconds,factor,multiplier)
    gen=MatchedFieldGenerator(model,config['accurate_center_yx'],half_life_seconds=half_life,
                              transport_id=run.session.transport_id,seed=seed)
    wind=CausalWind();events=[];states=[gen.state.copy()];rng_signatures=[]
    def signature():
        if gen.rng is not None:
            rng_signatures.append(hashlib.sha256(json.dumps(gen.rng.bit_generator.state,sort_keys=True).encode()).hexdigest())
    signature()
    for old in record['events']:
        op=old['operation'];at=old['at_seconds'];e={'operation':op,'at_seconds':at}
        if op=='ingest_readings':
            if len(old['readings'])!=1:raise ValueError('Singleton streaming history required')
            reading=SensorReading.from_dict(old['readings'][0])
            if reading.acquired_at_seconds<gen.time_seconds-1e-10 or reading.available_at_seconds>at+1e-10:
                raise ValueError('Chronological delivered evidence required')
            gen.advance(reading.acquired_at_seconds,wind)
            e['reading']={**old['readings'][0],'value':gen.observe(reading)}
        elif op=='plan_next':
            gen.advance(at,wind)
            wind=CausalWind.from_dict({**old['wind'],'velocity_yx':[v*wind_scale for v in old['wind']['velocity_yx']]})
            e['wind']=old['wind'] # Inference sees original supplied wind, not truth wind.
        elif op=='finish_scan':
            gen.advance(at,wind);e['scan_id']=old['receipt']['scan_id']
        else:raise ValueError('Unknown event')
        events.append(e);states.append(gen.state.copy());signature()
    gen.advance(run.duration_seconds,wind)
    events.append({'operation':'end','at_seconds':run.duration_seconds});states.append(gen.state.copy());signature()
    return events,np.array(states),{'factor':factor,'multiplier':multiplier,'seed':seed,
        'truth_model':asdict(model),'truth_half_life_seconds':half_life,'truth_wind_scale':wind_scale,
        'discrete_source_input_mean':float(gen.emission.mean()),'rng_signatures':rng_signatures}
