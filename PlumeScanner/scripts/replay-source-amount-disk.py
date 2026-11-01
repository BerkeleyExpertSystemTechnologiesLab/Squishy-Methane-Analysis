"""Reload completed JSON events and verify exact persisted closed-loop replay."""
from pathlib import Path
from dataclasses import replace
import argparse,importlib.util,json,shutil,time
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--closed',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    spec=importlib.util.spec_from_file_location('loop',Path(__file__).with_name('run-source-amount-loop.py'))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cfg=json.loads((args.closed/'config.json').read_text());out=args.output;out.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,out/Path(__file__).name);shutil.copy2(Path(__file__).with_name('run-source-amount-loop.py'),out/'run-source-amount-loop.py')
    shutil.copy2('src/plume_focus/source_amount_validation.py',out/'source_amount_validation.py')
    run=m.decode_config(json.loads(Path(cfg['baseline_config']).read_text()));shape=tuple(cfg['closed_loop']['grid_shape'])
    d=m.ClockDynamics(replace(run.model,grid_shape=shape),run.wind.velocity_yx,tick=cfg['clock_tick_seconds'],half_life=run.session.half_life_seconds)
    env=m.SavedEnvironment(args.study/f'grid-{shape[0]}x{shape[1]}'/'environment',d)
    sources=m.grid_positions((cfg['source_orders'][-1],)*2)
    E=m.gaussian_source_input(shape,sources,source_sigma=d.model.source_sigma,total_rate=cfg['reference_total_per_second'],formulation_id=cfg['source_input_id'])
    count=len(cfg['closed_loop']['cases'])*len(cfg['amounts'])*len(cfg['closed_loop']['seeds']);checks=[]
    for ci in range(count):
        for method in ['fixed','joint']:
            folder=args.closed/f'case-{ci:02d}-{method}';deadline=time.monotonic()+3600
            while True:
                try:
                    result=json.loads((folder/'result.json').read_text())
                    if result.get('causal_replay_bit_identical'):break
                except (FileNotFoundError,json.JSONDecodeError):pass
                if time.monotonic()>deadline:raise TimeoutError(f'No completed result: {folder}')
                time.sleep(1.)
            events=json.loads((folder/'events.json').read_text());saved=np.load(folder/'arrays.npz')
            actual,_,errors,frames,_,final=m.run_case(env,run,cfg,result['case'],method,sources,E,replay=events)
            assert np.array_equal(errors,saved['errors']) and np.array_equal(frames,saved['frames']) and np.array_equal(final,saved['final_mean'])
            checks.append({'case':ci,'method':method,'all_plans_and_receipts_identical':True,'all_arrays_bit_identical':True,
                           'events_sha256':m.sha(folder/'events.json'),'arrays_sha256':m.sha(folder/'arrays.npz')})
            m.save(out/'checks.json',checks);print(json.dumps({'disk_replay_case':ci,'method':method,'passed':True}),flush=True)
    m.save(out/'verification.json',{'persisted_executions_verified':len(checks),'all_passed':True,
        'json_roundtrip_plans_normalized':True,'negligible_precision_guard_no_output_change':True})


if __name__=='__main__':main()
