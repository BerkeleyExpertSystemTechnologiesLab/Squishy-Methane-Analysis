"""Small delayed-batch/overrun API inspection, separate from the 96-record study."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import runpy
import subprocess

import numpy as np

from plume_focus.sensing_cli_v1 import load_config, write_json
from plume_focus.sensing_execution_v1 import run_sensing_v1
from plume_focus.sensing_execution_v4 import with_risk_policy
from reconstruction_risk_observers import render_inspection


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path);args=parser.parse_args();root=args.directory
    base=load_config(Path(__file__).resolve().parents[1]/'configs/sensing-v2/coarse-full-startup.json')
    base=replace(base,session=replace(base.session,startup_mode='prior'),duration_seconds=8.,
        initial_budget_seconds=4.,budget_seconds=4.,delivery='batch',delivery_delay_seconds=.4,
        completion_overrun_seconds=.2)
    config=with_risk_policy(base)
    config_path=root/'configuration.json'
    if config_path.exists() and json.loads(config_path.read_text(encoding='utf-8'))!=json.loads(json.dumps(config.to_dict())):
        raise ValueError('Existing lifecycle fixture differs.')
    write_json(config_path,config.to_dict())
    run=run_sensing_v1(config)
    example=runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/sensing_controller_v1.py'))['execute_example']
    results=example(config,count=len(run.scans))
    assert len(results)==len(run.scans)
    for (plan,posterior),trace in zip(results,run.scans):
        assert plan==trace.plan
        np.testing.assert_array_equal(posterior.mean,trace.posterior.mean)
        np.testing.assert_array_equal(posterior.std,trace.posterior.std)
    write_json(root/'trace.json',run.operational_payload())
    write_json(root/'api-check.json',{'passed':True,'role':'engineering_lifecycle_not_comparison_record',
        'configuration':config.to_dict(),'operational_hash':run.operational_hash,
        'scans':len(run.scans),'batch_delay_seconds':.4,'completion_overrun_seconds':.2,
        'serialized_fake_scanner_matches_simulator':True})
    revision=subprocess.run(['git','rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    render_inspection(run,root,revision)
    print('Verified serialized fake-scanner agreement, delayed batches and actual overruns:',run.operational_hash)


if __name__=='__main__':main()
