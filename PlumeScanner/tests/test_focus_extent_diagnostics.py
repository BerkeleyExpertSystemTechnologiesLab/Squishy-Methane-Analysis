"""Independent arithmetic and isolation checks for evaluator-only probes."""
from dataclasses import replace
import numpy as np
import pytest

from plume_focus.focus_extent_diagnostics import EvidenceReplay, forecast, region, surfaces, select_events
from plume_focus.signal_belief import GaussianSignalBelief, SignalModelConfig, observation_weights
from plume_focus.sensing_contracts import CausalWind, Footprint
from test_full_fov_loop import controller, plan, execute


@pytest.mark.parametrize('value',[0.,.1,.3,1.,-.1])
@pytest.mark.parametrize('position',[(.5,.5),(0.,0.)])
def test_correlated_reading_matches_joint_conditional(value,position):
    model=SignalModelConfig(grid_shape=(2,3),initial_mean=.2)
    belief=GaussianSignalBelief(model)
    h=observation_weights(model.grid_shape,position,Footprint(.1,.1))
    mean=belief.mean.copy();p=belief.covariance.copy();r=.0025
    # Condition the joint vector [X,y] through an independently assembled block.
    joint=np.block([[p,(p@h)[:,None]],[(h@p)[None,:],np.array([[h@p@h+r]])]])
    expected=mean+joint[:-1,-1]/joint[-1,-1]*(value-h@mean)
    expected_p=joint[:-1,:-1]-np.outer(joint[:-1,-1],joint[-1,:-1])/joint[-1,-1]
    belief.condition(value,h,r)
    np.testing.assert_allclose(belief.mean,expected,atol=1e-15)
    np.testing.assert_allclose(belief.covariance,expected_p,atol=1e-15)
    other=GaussianSignalBelief(model);other.condition(value+10,h,r)
    np.testing.assert_array_equal(other.covariance,belief.covariance)
    # Repeated independent evidence changes precision, rather than duplicating a map.
    belief.condition(value,h,r)
    assert h@belief.covariance@h < h@expected_p@h


def test_future_probes_are_detached_and_separate_mean_from_uncertainty():
    b=GaussianSignalBelief(SignalModelConfig(grid_shape=(3,4)),transport_id='zero_inflow_v1')
    b.mean[5]=1
    mean,p=b.mean.copy(),b.covariance.copy();wind=CausalWind(velocity_yx=(0.,.02))
    normal=forecast(b,3,wind,(.5,.3));no_q=forecast(b,3,wind,(.5,.3),'process_off')
    np.testing.assert_array_equal(normal.mean,no_q.mean)
    assert np.trace(normal.covariance)>np.trace(no_q.covariance)
    no_source=forecast(b,3,wind,(.5,.3),'future_source_off')
    np.testing.assert_array_equal(no_source.covariance,normal.covariance)
    assert np.max(abs(no_source.mean-normal.mean))>0
    np.testing.assert_array_equal(b.mean,mean);np.testing.assert_array_equal(b.covariance,p)
    assert b.time_seconds==0


def test_constant_wind_moves_packet_without_velocity_change():
    b=GaussianSignalBelief(SignalModelConfig(grid_shape=(8,8),diffusivity=0,source_rate=0,
                         process_std_per_sqrt_second=0),transport_id='zero_inflow_v1')
    b.mean[3*8+2]=1
    predicted=forecast(b,1,CausalWind(velocity_yx=(0.,1/8)),None)
    old=(b.positions*b.mean[:,None]).sum(axis=0)/b.mean.sum()
    new=(predicted.positions*predicted.mean[:,None]).sum(axis=0)/predicted.mean.sum()
    np.testing.assert_allclose(new-old,[0,1/8],atol=1e-12)


def test_empty_regions_unknown_width_and_peak_component():
    mask=np.zeros((4,4),bool)
    assert region(mask,np.zeros_like(mask),(0,0))['crosswind_extent'] is None
    mask[1,1]=True;mask[3,3]=True
    value=mask.astype(float);value[1,1]=2
    stats=region(mask,value,(0,1))
    assert stats['components']==2 and stats['peak_component']['area']==1/16


def test_mean_and_uncertainty_eligibility_are_not_equivalent():
    b=GaussianSignalBelief(SignalModelConfig(grid_shape=(2,2)))
    result,mean,upper=surfaces(b,(4,4),(0,1),np.zeros((4,4)))
    assert not (mean>=.15).any() and (upper>=.15).all()
    assert result['uncertainty_added_background_cells']==16


def test_forced_evidence_matches_public_acquisition_with_delayed_batch():
    c=controller();e=EvidenceReplay(c.session);p=plan(c)
    event={'operation':'plan_next','at_seconds':0.,'wind':CausalWind().to_dict(),'plan':p.to_dict()}
    e.event(event)
    ex=execute(c,p,.4,delivery='batch',delay=2.)
    readings={'operation':'ingest_readings','readings':[r.to_dict() for r in ex.readings],
              'at_seconds':ex.receipt.received_at_seconds}
    e.event(readings);e.event(readings)  # duplicate delivery must not count twice
    e.event({'operation':'finish_scan','receipt':ex.receipt.to_dict()})
    np.testing.assert_array_equal(e._belief.mean,c._belief.mean)
    np.testing.assert_array_equal(e._belief.covariance,c._belief.covariance)
    with pytest.raises(ValueError):e.predictive_view(ex.receipt.received_at_seconds-1)


def test_selection_records_unavailable_without_replacement():
    record={'events':[], 'scans':[], 'job':{'case':'cessation'}}
    chosen,unavailable=select_events(record)
    assert chosen==[] and len(unavailable)==3


def test_runtime_has_no_diagnostic_dependency():
    import ast
    from pathlib import Path
    for name in ('full_fov_loop','acquisition','signal_belief','physical_scan','signal_distribution'):
        tree=ast.parse(Path(f'src/plume_focus/{name}.py').read_text(encoding='utf-8'))
        imports=[node.module or '' for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)]
        assert not any('focus_extent' in value for value in imports)
