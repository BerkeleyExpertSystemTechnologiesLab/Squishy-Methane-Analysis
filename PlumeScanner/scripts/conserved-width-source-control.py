"""Paired shape/amount controls; preserve fixed peak study and inference model."""
from pathlib import Path
import argparse,importlib.util,json,shutil,subprocess,sys,time
import numpy as np
from plume_focus.conserved_source_control import generate_source_control
from plume_focus.full_fov_execution import decode_config

spec=importlib.util.spec_from_file_location('mismatch',Path(__file__).with_name('one-factor-source-mismatch.py'))
exp=importlib.util.module_from_spec(spec);spec.loader.exec_module(exp)
m=exp.m


def paired_contrasts(cfg,records):
    rows=[]
    for width in [.5,.75,1.25,1.5]:
        suffix=str(width).replace('.','p');names=['matched','fixed_total_'+suffix,'amount_only_'+suffix,'fixed_peak_'+suffix]
        methods=[]
        for i,name in enumerate(m.NAMES):
            # Shape/amount form a 2x2 design for each width. Errors are nonlinear:
            # report an interaction rather than force percentages to add to 100.
            raw=np.array([[records[n][c['label']]['final']['methods'][i]['mean_error_fov'] for n in names]
                          for c in cfg['controls'][1:]])
            mean_raw=np.array([records[n]['matched_mean']['final']['methods'][i]['mean_error_fov'] for n in names])
            metrics={}
            for power,key in [(1,'distance'),(2,'squared_distance')]:
                a=raw**power;zero,shape,amount,joint=a.T
                contrasts={'shape_minus_baseline':shape-zero,'amount_minus_baseline':amount-zero,
                    'joint_minus_baseline':joint-zero,'joint_minus_shape':joint-shape,
                    'interaction':joint-shape-amount+zero}
                metrics[key]={k:{'mean':float(v.mean()),'standard_error':float(v.std(ddof=1)/np.sqrt(len(v))),
                    'positive_count':int(np.sum(v>0)),'values':v.tolist()} for k,v in contrasts.items()}
                z,g,q,j=mean_raw**power
                metrics[key]['mean_control']={'shape_minus_baseline':float(g-z),'amount_minus_baseline':float(q-z),
                    'joint_minus_baseline':float(j-z),'joint_minus_shape':float(j-g),'interaction':float(j-g-q+z)}
            methods.append({'method':name,'arm_order':names,'seed_errors':raw.tolist(),'contrasts':metrics})
        rows.append({'width_multiplier':width,'methods':methods})
    return rows


def plot(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lookup={s['label']:s for s in summary['scenarios']}
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    widths=[.5,.75,1.,1.25,1.5]
    for i,ax in enumerate(axes):
        for mode,color,label in [('fixed_peak','#d67d00','Width + amount (fixed peak)'),('fixed_total','#1674b8','Width only (fixed total)'),('amount_only','#258b53','Amount only (fixed width)')]:
            rows=[lookup['matched' if w==1 else mode+'_'+str(w).replace('.','p')]['methods'][i] for w in widths]
            ax.plot(widths,[r['mean_error_fov'] for r in rows],'o-',color=color,label=label)
            ax.plot(widths,[r['mean_control']['mean_error_fov'] for r in rows],':',color=color,alpha=.7)
        ax.set(title=m.NAMES[i],xlabel='Reference width / assumed width',ylabel='Final source mean-location error (FOV)',ylim=(0,None))
        ax.set_xticks(widths);ax.legend(fontsize=7)
    fig.suptitle('Solid: 10 paired-seed mean; dotted: deterministic controls')
    fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();start=time.time();cfg=json.loads(Path(args.config).read_text());out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    recordpath=Path(cfg['record']);assert m.digest(recordpath)==cfg['record_sha256']
    record=json.loads(recordpath.read_text());run=decode_config(record['configuration'])
    shutil.copyfile(args.config,out/'config.json');shutil.copyfile(recordpath,out/'original-path-and-evidence.json')
    m.save(out/'baseline-configuration.json',record['configuration'])
    m.save(out/'provenance.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),'git_status':subprocess.check_output(['git','status','--short'],text=True),
        'command':sys.argv,'python':sys.version,'parent_manifest_sha256':m.digest(Path(cfg['previous_mismatch_output'])/'artifact-manifest.json')})
    (out/'requirements-frozen.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    caches={};precision=None;baserng={};records={};checks=[];metadata_rows=[]
    for scenario in cfg['scenarios']:
        label=scenario['label'];records[label]={}
        for control in cfg['controls']:
            cl=control['label'];folder=out/label/cl;folder.mkdir(parents=True)
            events,states,metadata=generate_source_control(record,cfg,run,control['seed'],scenario['factor'],scenario['multiplier'])
            if label=='matched':baserng[cl]=metadata['rng_signatures']
            else:assert metadata['rng_signatures']==baserng[cl]
            m.save(folder/'generated-evidence.json',{'events':events,'seed':control['seed']});m.save(folder/'truth-parameters.json',metadata)
            np.savez_compressed(folder/'truth.npz',states=states,state_times=[0.]+[e['reading']['acquired_at_seconds'] if e['operation']=='ingest_readings' else e['at_seconds'] for e in events])
            old=None
            if scenario['factor'] in ('matched','fixed_peak'):
                oldlabel='matched' if label=='matched' else 'source_width_'+str(scenario['multiplier']).replace('.','p')
                old=Path(cfg['previous_mismatch_output'])/oldlabel/cl
                assert events==json.loads((old/'generated-evidence.json').read_text())['events']
                with np.load(old/'truth.npz') as data:np.testing.assert_array_equal(states,data['states'])
            a=exp.fit(run,events)
            if precision is None:precision=a.precision.copy()
            else:np.testing.assert_array_equal(a.precision,precision)
            np.savez_compressed(folder/'likelihood-statistics.npz',precision=a.precision,linear=a.linear,constant=a.constant,
                response=a.response,mean=a.reference.mean,covariance=a.reference.covariance)
            results=[]
            for order in cfg['candidate_grid_orders']:
                if order not in caches:
                    caches[order]=exp.grid_cache(cfg,run,order,precision)
                    np.savez_compressed(out/f'candidate-grid-{order}.npz',sources=caches[order].sources,logprior=caches[order].logprior)
                result=exp.evaluate(cfg,a,caches[order],order,states[-1],folder);results.append(result)
                if old is not None and (old/f'results-{order}.json').exists():
                    previous=json.loads((old/f'results-{order}.json').read_text())['methods']
                    difference=max(float(np.linalg.norm(np.array(x['mean_yx'])-y['mean_yx'])) for x,y in zip(result['methods'],previous))
                    assert difference<1e-10;checks.append({'scenario':label,'control':cl,'grid':order,'source_mean_difference':difference})
            refinement=exp.refinement(cfg,results[0],results[-1]);m.save(folder/'refinement.json',[refinement])
            records[label][cl]={'final':results[-1],'refinement':[refinement]}
            metadata_rows.append({'scenario':label,'control':cl,**{k:metadata[k] for k in ['baseline_discrete_input','actual_discrete_input','fixed_peak_input_ratio','actual_peak_rate_ratio','actual_width_ratio','target_input_relative_error']}})
        print(json.dumps({'scenario':label,'broad_mean_error':float(np.mean([records[label][c['label']]['final']['methods'][0]['mean_error_fov'] for c in cfg['controls'][1:]])),
                          'refinement_passed':all(v['refinement'][-1]['passed'] for v in records[label].values())}),flush=True)
    summary=exp.summarize(cfg,records,out)
    summary['scope']=cfg['scope'];summary['factorial_contrasts']=paired_contrasts(cfg,records);summary['unresolved_hpd_cases']=[]
    for scenario in summary['scenarios']:
        for i,method in enumerate(scenario['methods']):
            low=high=0
            for control in cfg['controls'][1:]:
                data=records[scenario['label']][control['label']];ref=data['refinement'][-1]['methods'][i];r=data['final']['methods'][i]
                if not ref['hpd95_membership_same']:
                    high+=1;summary['unresolved_hpd_cases'].append({'scenario':scenario['label'],'control':control['label'],'method':method['method'],'refinement':ref})
                else:low+=int(r['true_in_hpd95']);high+=int(r['true_in_hpd95'])
            method['hpd95_count_grid_resolution_bounds']=[low,high]
    summary['source_mean_and_radius_mass_checks_all_pass']=all(r['mean_shift_fov']<cfg['source_refinement']['mean_shift_fov_tolerance'] and r['radius_mass_change']<cfg['source_refinement']['mass_radius_005_absolute_tolerance']
        for group in records.values() for data in group.values() for r in data['refinement'][-1]['methods'])
    m.save(out/'summary.json',summary);m.save(out/'normalization-checks.json',metadata_rows);plot(summary,out)
    m.save(out/'verification.json',{'old_55_histories_exactly_reproduced':True,'same_rng_every_event':True,'same_inference_precision':True,
        'source_mean_reproduction_checks':checks,'max_relative_input_error':max(x['target_input_relative_error'] for x in metadata_rows),
        'source_mean_and_radius_mass_checks_all_pass':summary['source_mean_and_radius_mass_checks_all_pass'],
        'complete_refinement_passed':summary['all_refinement_passed']})
    imp=out/'implementation';imp.mkdir()
    for f in [Path(__file__),Path('src/plume_focus/conserved_source_control.py'),Path('src/plume_focus/mismatch_control.py'),Path('src/plume_focus/model_consistent_control.py'),
        Path('src/plume_focus/source_likelihood_quadrature.py'),Path('src/plume_focus/source_prior_belief.py'),Path('scripts/one-factor-source-mismatch.py'),
        Path('scripts/model-consistent-source-priors.py'),Path('scripts/compare-source-priors.py'),Path('tests/test_conserved_source_control.py')]:shutil.copyfile(f,imp/f.name)
    assert m.digest(recordpath)==cfg['record_sha256']
    m.save(out/'artifact-manifest.json',{'wall_seconds':time.time()-start,'original_evidence_unchanged':True,
        'files':{str(p.relative_to(out)):m.digest(p) for p in out.rglob('*') if p.is_file()}})
    print(json.dumps({'wall_seconds':time.time()-start,'all_refinement_passed':summary['all_refinement_passed'],'location_refinement_passed':summary['source_mean_and_radius_mass_checks_all_pass']}),flush=True)


if __name__=='__main__':main()
