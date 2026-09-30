"""Observer-only summary and immutable calibrated test binding."""
import argparse
import importlib.util
from pathlib import Path
import numpy as np

spec=importlib.util.spec_from_file_location('calibration_driver',Path(__file__).with_name('run-scanner-calibration.py'))
d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)


def bind(root):
    rows=[];sessions={}
    for job in d.read(root/'declaration.json')['heldout']:
        assert d.sha(root/job['config'])==job['config_sha256']
        config=d.configured(root,job,'heldout')
        key=(job['profile'],job['variant'])
        session_hash=d.configuration_hash(config.session.to_dict())
        assert sessions.setdefault(key,session_hash)==session_hash,'Scenario activity leaked into session binding'
        rows.append({'name':job['name'],'configuration_hash':config.digest,'configuration':config.to_dict()})
    value={'artifact_type':'bound_calibration_test_v1','declaration_sha256':d.sha(root/'declaration.json'),
           'freeze_sha256':d.sha(root/'freeze.json'),'records':rows}
    path=root/'bound-heldout.json'
    if path.exists():assert d.read(path)==value
    else:d.write(path,value)
    print('Frozen',len(rows),'test configurations',d.sha(path))


def summarize(root):
    jobs=d.read(root/'declaration.json')['development']+d.read(root/'declaration.json')['heldout']
    for job in jobs:
        assert d.sha(root/job['config'])==job['config_sha256']
    paths=sorted((root/'runs').glob('*.json'))
    assert {p.stem for p in paths}=={j['name'] for j in jobs},'Study is incomplete'
    bound={r['name']:r['configuration_hash'] for r in d.read(root/'bound-heldout.json')['records']}
    rows=[]
    for r in (d.read(path) for path in paths):
        assert d.compatible_code(root,root/'runs'/f"{r['job']['name']}.json",r['code_sha256'])
        assert r['declaration_sha256']==d.sha(root/'declaration.json')
        job=r['job'];scans=r['scans'];focus=[s for s in scans if s['plan']['mode']=='FOCUS']
        if r['stage']=='heldout':assert bound[job['name']]==r['configuration_hash']
        padding_reads=sum(sum(i in set(s['plan'].get('padding_cells',())) for i in s['plan']['sampled_cells'])
                          for s in focus) if focus and 'sampled_cells' in focus[0]['plan'] else None
        # Count physical readings by their declared actual position, not raster pixels.
        padding_reads=0
        for s in focus:
            p=s['plan'];height,width=p['lattice_shape'];padding=set(p.get('padding_cells',()))
            for sample in p['samples']:
                y,x=sample['position_yx'];cell=min(int(y*height),height-1)*width+min(int(x*width),width-1)
                padding_reads+=cell in padding
        cessation={}
        if job['case']=='cessation':
            later=[s for s in scans if s['plan']['mode']=='FULL' and s['plan']['started_at_seconds']>=180.]
            background=[s for s in later if max(s['clean_readings'],default=float('inf'))<=.15]
            cessation={'full_scans_started_after_cessation':len(later),
                'all_background_full_scans_after_cessation':len(background),
                'seconds_to_first_full_after_cessation':later[0]['plan']['started_at_seconds']-180. if later else None,
                'post_cessation_clear_rescan_reached':bool(background)}
        rows.append({**job,'J_over_T':r['J_over_T'],
            'mean_signed_index_error':r['signed_index_error_integral']/360,
            'index_mse':r['index_squared_error_integral']/360,'spent_seconds':r['spent_seconds'],
            'execution_and_online_evaluation_wall_seconds':r['offline_seconds'],
            'startup_complete':r['startup_complete'],'completed_full_scans':r['completed_full_scans'],
            'termination':r['termination'],'focus_duration_mean':float(np.mean([s['plan']['completed_at_seconds']-s['plan']['started_at_seconds'] for s in focus])) if focus else None,
            'focus_seconds':sum(s['receipt']['received_at_seconds']-s['plan']['started_at_seconds'] for s in focus),
            'actual_padding_readings':padding_reads,**r['metrics'],**cessation})
    comparisons=[]
    for row in rows:
        if row['name'].startswith('heldout') and row['variant']=='calibrated':
            bas=next(r for r in rows if r['name'].startswith('heldout') and r['variant']=='baseline'
                     and all(r[k]==row[k] for k in ('profile','case','seed')))
            comparisons.append({'profile':row['profile'],'case':row['case'],'seed':row['seed'],
                'relative_J_change':row['J_over_T']/bas['J_over_T']-1,
                'baseline_J_over_T':bas['J_over_T'],'calibrated_J_over_T':row['J_over_T']})
    groups=[]
    for profile in ('coarse','double'):
        for case in ('continuous','wind_error','weak','blank','cessation','intermittent'):
            for variant in ('baseline','calibrated','full'):
                selected=[r for r in rows if r['name'].startswith('heldout') and r['profile']==profile and r['case']==case and r['variant']==variant]
                if not selected:continue
                metrics=('J_over_T','index_mse','spent_seconds','actual_core_readings','false_focus_decisions','missed_focus_decisions',
                         'refocus_after_background_full_scan','focus_scans','full_scans','mean_observed_sample_age_seconds')
                groups.append({'profile':profile,'case':case,'variant':variant,'n':len(selected),
                    **{k:float(np.mean([r[k] for r in selected])) for k in metrics}})
    result={'artifact_type':'scanner_calibration_summary_v1','rows':rows,'groups':groups,'paired_comparisons':comparisons,
        'freeze_sha256':d.sha(root/'freeze.json'),'bound_heldout_sha256':d.sha(root/'bound-heldout.json'),
        'interpretation':'Two realization/noise pairs are descriptive, not a general superiority claim.'}
    d.write(root/'summary.json',result)
    print('Development',sum(r['name'].startswith('development') for r in rows),'heldout',len(bound),
          'improved',sum(r['relative_J_change']<-.01 for r in comparisons),'worse',sum(r['relative_J_change']>.01 for r in comparisons))


def inventory(root):
    paths=[p for p in sorted(root.rglob('*')) if p.is_file() and p.name not in ('artifact-manifest.json','closeout.json')
           and '__pycache__' not in p.parts and not p.name.endswith('.partial')]
    value={'artifact_type':'scanner_calibration_artifact_manifest_v1','producing_revision':d.revision(),
        'runtime_code_sha256':d.code_hash(),'records':[{'path':p.relative_to(root).as_posix(),
        'bytes':p.stat().st_size,'sha256':d.sha(p)} for p in paths]}
    d.write(root/'artifact-manifest.json',value);print(len(paths),'hashed artifacts',d.sha(root/'artifact-manifest.json'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['bind','summarize','inventory'],required=True)
    p.add_argument('--root',type=Path,default=Path('outputs/scanner-calibration'));a=p.parse_args()
    {'bind':bind,'summarize':summarize,'inventory':inventory}[a.phase](a.root)
