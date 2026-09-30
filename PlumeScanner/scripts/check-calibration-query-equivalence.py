"""Authorize hashed existing records after the detached mean-query optimization."""
import importlib.util
from pathlib import Path
import numpy as np
from plume_focus.calibrated_scanner import DecayGaussianBelief
from plume_focus.calibration_evaluation import mean_forecast
from plume_focus.sensing_risk import linear_transition

spec=importlib.util.spec_from_file_location('driver',Path(__file__).with_name('run-scanner-calibration.py'))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
root=Path('outputs/scanner-calibration');prior=d.read(root/'pre-vector-query.json')
rows=[]
for profile in ('coarse','double'):
    config=d.base_config(profile,'continuous',0,0)
    b=DecayGaussianBelief(config.model,transport_id=config.session.transport_id)
    b.mean=np.linspace(-.2,.7,b.mean.size)
    for seconds in (.0125,.15,1.,12.,60.,100.):
        a,s,_=linear_transition(config.model,config.wind.velocity_yx,config.session.source_position,seconds,b.transport_id)
        before=a@b.mean+s;after=mean_forecast(b,b.mean,seconds,config.wind,config.session.source_position)
        error=float(np.max(np.abs(before-after)))
        assert error<=1e-12
        rows.append({'profile':profile,'horizon':seconds,'maximum_absolute_error':error})
files={}
for item in prior['records']:
    path=root/item['path'];assert d.sha(path)==item['sha256'];files[Path(item['path']).as_posix()]=item['sha256']
proof={'status':'verified_numerical_equivalence','from_code_sha256':prior['previous_code_sha256'],
    'to_code_sha256':d.code_hash(),'files':files,'controls':rows,
    'scope':'Only detached no-decay mean evaluation for the new binding changes. Acquisition, decisions, fitting, parameters and dynamics are unchanged. Every saved execution must additionally pass public replay and original J tolerance 1e-12.',
    'test_command':'python -m pytest tests/test_scanner_calibration.py tests/test_sensing_core.py::test_controller_dependency_boundary -q'}
d.write(root/'query-compatibility.json',proof);print(len(files),'preserved artifacts; maximum error',max(r['maximum_absolute_error'] for r in rows))
