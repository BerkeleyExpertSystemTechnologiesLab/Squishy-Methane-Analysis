from dataclasses import asdict
import numpy as np
import pytest
from plume_focus.conserved_source_control import source_control_model,injection
from plume_focus.signal_belief import SignalModelConfig


@pytest.mark.parametrize('width',[.5,.75,1.25,1.5])
@pytest.mark.parametrize('source',[(.547945205479452,.15047021943573669),(.02,.97)])
def test_discrete_amount_conservation_and_factorial_inputs(width,source):
    model=SignalModelConfig(grid_shape=(12,16),source_sigma=.035,source_rate=.2829738193616142)
    peak,info=source_control_model(model,source,width,'fixed_peak')
    shape,_=source_control_model(model,source,width,'fixed_total')
    amount,_=source_control_model(model,source,width,'amount_only')
    base=injection(model,source);ep=injection(peak,source);eg=injection(shape,source);ea=injection(amount,source)
    np.testing.assert_allclose(eg.mean(),base.mean(),rtol=5e-14)
    np.testing.assert_allclose(ea.mean(),ep.mean(),rtol=5e-14)
    # Permit two subnormal ULPs in the numerically irrelevant Gaussian tails.
    np.testing.assert_allclose(ep,info['fixed_peak_input_ratio']*eg,rtol=5e-14,atol=2*np.nextafter(0.,1.))
    np.testing.assert_allclose(ea,info['fixed_peak_input_ratio']*base,rtol=5e-14,atol=2*np.nextafter(0.,1.))
    for changed in (peak,shape,amount):
        assert all(v==asdict(changed)[k] for k,v in asdict(model).items() if k not in ('source_sigma','source_rate'))
    assert amount.source_sigma==model.source_sigma


def test_unit_multiplier_is_identical_and_invalid_parameters_rejected():
    model=SignalModelConfig()
    for mode in ('matched','fixed_peak','fixed_total','amount_only'):
        changed,_=source_control_model(model,(.5,.4),1.,mode)
        assert changed==model
    for width in (0,-1,float('nan')):
        with pytest.raises(ValueError):source_control_model(model,(.5,.4),width,'fixed_total')
    with pytest.raises(ValueError):source_control_model(model,(1.1,.4),.5,'fixed_total')
