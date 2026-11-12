"""Retain failed numerical checks; rerun selected histories on a finer global prior."""
from pathlib import Path
from dataclasses import replace
import argparse,importlib.util,json,shutil
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--grid',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    spec=importlib.util.spec_from_file_location('amount_study',Path(__file__).with_name('validate-source-amount.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=json.loads((args.study/'config.json').read_text());grid=args.study/f'grid-{args.grid}'
    rcfgpath=Path('configs/source-prior-study/source-amount-refinement-v1.json');rcfg=json.loads(rcfgpath.read_text())
    original_check=json.loads((grid/'refinement.json').read_text())
    ids=sorted(set(c['history'] for c in original_check['checks'] if not c['passed']))
    if not ids:raise ValueError('No failed histories require another numerical pass')
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    module.save(out/'config.json',{'refinement':rcfg,'original_study':str(args.study.resolve()),'grid':args.grid,
        'selected_history_indices':ids,'original_configuration_sha256':module.sha(args.study/'config.json')})
    shutil.copy2(__file__,out/Path(__file__).name)
    shutil.copy2(Path(__file__).with_name('validate-source-amount.py'),out/'validate-source-amount.py')
    shutil.copy2('src/plume_focus/source_amount_validation.py',out/'source_amount_validation.py')
    oldfolder=grid/f"source-{cfg['source_orders'][-1]}-amount-{cfg['amount_orders'][-1]}"
    old=json.loads((oldfolder/'results.json').read_text());selected=[old['metadata'][i] for i in ids]
    remap={old:new for new,old in enumerate(ids)}
    base={**old,'metadata':selected,'JT':[old['JT'][i] for i in ids],
          'forecasts':[[v[i] for i in ids] for v in old['forecasts']],
          'uncertainty':[{**r,'history':remap[r['history']]} for r in old['uncertainty'] if r['history'] in remap]}
    module.save(out/'original-fine-subset.json',base)
    run=module.decode_config(json.loads(Path(cfg['baseline_config']).read_text()));shape=tuple(map(int,args.grid.split('x')))
    d=module.ClockDynamics(replace(run.model,grid_shape=shape),run.wind.velocity_yx,
                          tick=cfg['clock_tick_seconds'],half_life=run.session.half_life_seconds)
    env=module.Environment(d,cfg,grid/'environment');record=json.loads(Path(cfg['record']).read_text())
    original_readings=np.load(oldfolder/'evidence-and-errors.npz')['readings'][:,ids]
    result=module.fixed_run(env,run,cfg,record,(rcfg['source_order'],rcfg['amount_order']),out/'fine',histories=selected,refinement_only=True,evidence=original_readings)
    finer_readings=np.load(out/'fine/evidence-and-errors.npz')['readings']
    assert np.array_equal(original_readings,finer_readings)
    check=module.refine(cfg,base,result);module.save(out/'refinement.json',check)
    module.save(out/'verification.json',{'grid':args.grid,'selected_histories':ids,'final_refinement_passed':check['passed'],
        'evidence_bit_identical':True,'original_study':str(args.study.resolve()),
        'original_configuration_sha256':module.sha(args.study/'config.json')})
    print(json.dumps({'refinement_grid':args.grid,'histories':len(ids),'passed':check['passed'],'failed_checks':sum(not c['passed'] for c in check['checks'])}),flush=True)


if __name__=='__main__':main()
