"""Read-only probes of the pre-freeze implementation, not acceptance tests.

Expected observations are recorded for revision 66f0770 in the freeze review.
Later conforming implementations should change these results. No files are
written unless --output is supplied. Run in the repository's Python environment.
"""

import argparse
import json
from pathlib import Path
import numpy as np
from plume_focus.controller import SensingController
from plume_focus.sensing_contracts import ScannerCapabilities, ScanConstraints, Footprint, SensorReading
from plume_focus.sensing_policy import _allocate
from plume_focus.physical_scan import compile_scan, coverage_mask

caps = ScannerCapabilities(lattice_shape=(4,4), vertical_speed=.25, horizontal_speed=.5,
                           readout_hz=1, integration_seconds=.5, footprint=Footprint(.125,.125))
mass = np.zeros((4,4))
mass[0,0]=mass[0,3]=mass[3,0]=mass[3,3]=.23
mass[1,2]=.08
desired = mass == .23
plan = _allocate(mass, desired, caps, ScanConstraints(max_duration_seconds=6,minimum_samples=1),
                 np.full((4,4),-1.),0,scan_id='gap',now=0,pose=(0,.125),reacquisition=False)
result = {'current_budget_packing': {'requested_cells':plan.requested_cells,
            'seconds':plan.duration_seconds, 'retained_mass':float(mass[coverage_mask(plan,mass.shape,caps)].sum()),
            'original_selected_cells_preserved':set(np.flatnonzero(desired)).issubset(plan.requested_cells)}}
fast = ScannerCapabilities(lattice_shape=(4,4), vertical_speed=.5,horizontal_speed=.5,
                           readout_hz=1,integration_seconds=.5,footprint=Footprint(.125,.125))
whole = np.zeros((4,4),bool); whole[:,0]=True
slow = compile_scan(whole,fast,scan_id='speed',started_at_seconds=0,current_position=(0,.125),target_mass=.9)
sweep = next(s for s in slow.segments if s.kind=='sweep')
result['current_speed_substitution']={'declared_vertical_speed':.5,
   'actual_sweep_speed':abs(sweep.end_yx[0]-sweep.start_yx[0])/(sweep.completed_at_seconds-sweep.started_at_seconds)}
ctrl=SensingController(start_full_scene=False)
p=ctrl.plan_next(0,(.5,.5),ScanConstraints())
s=p.samples[0]
delivery=s.acquired_at_seconds+1
r=SensorReading(p.scan_id,'delayed',s.position_yx,s.acquired_at_seconds,delivery,2.,s.footprint,s.integration_seconds)
ctrl.ingest_readings([r],available_at_seconds=delivery)
try:
    view=ctrl.predictive_view(s.acquired_at_seconds)
    result['current_past_predictive_view']={'reading_acquired':s.acquired_at_seconds,'delivered':delivery,
        'returned_view_time':view.time_seconds,'uses_delayed_reading_before_delivery':bool(view.mean.max()>0)}
except ValueError as error:
    result['current_past_predictive_view']={'reading_acquired':s.acquired_at_seconds,'delivered':delivery,
        'query_rejected':True,'reason':str(error),'uses_delayed_reading_before_delivery':False}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
if args.output:
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
