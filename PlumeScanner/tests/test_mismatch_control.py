from dataclasses import asdict
import numpy as np
import pytest
from plume_focus.mismatch_control import truth_parameters,generate_mismatch
from plume_focus.signal_belief import SignalModelConfig
from plume_focus.sensing_contracts import CausalWind,SensorReading,Footprint
from types import SimpleNamespace


@pytest.mark.parametrize('factor', ['wind_speed','source_width','half_life'])
def test_exactly_one_truth_parameter_changes(factor):
    model=SignalModelConfig(grid_shape=(3,4))
    changed,half,wind=truth_parameters(model,6.,factor,1.5)
    diffs=[k for k in asdict(model) if asdict(model)[k]!=asdict(changed)[k]]
    assert diffs==(['source_sigma'] if factor=='source_width' else [])
    assert half==(9. if factor=='half_life' else 6.)
    assert wind==(1.5 if factor=='wind_speed' else 1.)


def test_common_random_numbers_path_and_inference_wind_stay_fixed():
    model=SignalModelConfig(grid_shape=(3,4),source_sigma=.15)
    wind=CausalWind(velocity_yx=(.01,.03))
    reading=SensorReading(scan_id='s',reading_id='r',position_yx=(.5,.5),acquired_at_seconds=.05,
                          available_at_seconds=.0625,value=0.,footprint=Footprint(height=.1,width=.1),integration_seconds=.025)
    record={'events':[{'operation':'plan_next','at_seconds':0.,'wind':wind.to_dict()},
        {'operation':'ingest_readings','at_seconds':.0625,'readings':[reading.to_dict()]}]}
    cfg={'accurate_center_yx':[.5,.4]};run=SimpleNamespace(model=model,session=SimpleNamespace(half_life_seconds=6.,transport_id='zero_inflow_v1'),duration_seconds=.1)
    base,truth,meta=generate_mismatch(record,cfg,run,42,'matched',1.)
    for factor in ('wind_speed','source_width','half_life'):
        events,states,details=generate_mismatch(record,cfg,run,42,factor,1.5)
        assert details['rng_signatures']==meta['rng_signatures']
        assert events[0]['wind']==base[0]['wind']
        assert {k:v for k,v in events[1]['reading'].items() if k!='value'}=={k:v for k,v in base[1]['reading'].items() if k!='value'}
        np.testing.assert_array_equal(states[0],truth[0])
        assert not np.array_equal(states[-1],truth[-1])
    again,states,_=generate_mismatch(record,cfg,run,42,'matched',1.)
    assert again==base;np.testing.assert_array_equal(states,truth)
