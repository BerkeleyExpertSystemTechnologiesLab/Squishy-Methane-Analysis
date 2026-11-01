"""Refine final source quadrature using exact affine Gaussian likelihoods."""
from pathlib import Path
import argparse,importlib.util,json,shutil,subprocess,sys,time
import numpy as np
from scipy.special import logsumexp
from plume_focus.source_likelihood_quadrature import AffineSourceLikelihood
from plume_focus.full_fov_execution import decode_config
from plume_focus.sensing_contracts import CausalWind,SensorReading
from plume_focus.signal_belief import observation_weights

spec=importlib.util.spec_from_file_location('matched',Path(__file__).with_name('model-consistent-source-priors.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();start=time.time();cfg=json.loads(Path(args.config).read_text());parent=Path(cfg['parent_output'])
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.config,output/'config.json')
    record=json.loads((parent/'original-path-and-evidence.json').read_text());run=decode_config(record['configuration'])
    source_checks={};allrows=[];refinements=[]
    for control in cfg['controls']:
        label=control['label'];folder=output/label;folder.mkdir()
        evidence=parent/label/'generated-evidence.json';data=json.loads(evidence.read_text())
        affine=AffineSourceLikelihood(run.model,half_life_seconds=run.session.half_life_seconds,transport_id=run.session.transport_id)
        wind=CausalWind()
        for e in data['events']:
            op=e['operation'];at=e['reading']['acquired_at_seconds'] if op=='ingest_readings' else e['at_seconds']
            affine.advance(at,wind)
            if op=='plan_next':wind=CausalWind.from_dict(e['wind'])
            elif op=='ingest_readings':
                reading=SensorReading.from_dict(e['reading']);h=observation_weights(run.model.grid_shape,reading.position_yx,reading.footprint)
                affine.condition((reading.value-run.model.offset)/run.model.gain,h,(run.model.sensor_noise_std/run.model.gain)**2)
        np.savez_compressed(folder/'likelihood-sufficient-statistics.npz',precision=affine.precision,linear=affine.linear,
            constant=affine.constant,response=affine.response,mean=affine.reference.mean,covariance=affine.reference.covariance)
        rows=[]
        for order in cfg['candidate_grid_orders']:
            bank=m.prior_bank(cfg,run.model,run.session,order)
            ll=affine.log_likelihood(bank.emissions)
            bank.log_weights+=ll[None];bank.log_weights-=logsumexp(bank.log_weights,axis=1)[:,None]
            checks={}
            direct=parent/label/f'beliefs-{order}.npz'
            if direct.exists():
                with np.load(direct) as old:
                    checks={'max_component_loglik_error':float(np.max(np.abs(ll-old['final_component_loglikelihood']))),
                        'max_posterior_weight_error':float(np.max(np.abs(bank.weights-old['final_weights']))),
                        'max_aggregate_field_mean_error':float(np.max(np.abs(affine.reference.mean[None]+(bank.weights@bank.emissions)@affine.response.T-old['final_field_mean'])))}
                assert checks['max_component_loglik_error']<1e-7 and checks['max_posterior_weight_error']<1e-8 and checks['max_aggregate_field_mean_error']<1e-9,checks
            stats=m.source_summary(bank,cfg,ll,order)
            result={'order':order,'source':stats,'equivalence_to_direct_bank':checks,
                    'max_grid_loglik_minus_exact_true':float(ll[:order*order].max()-ll[-2])}
            m.save(folder/f'results-{order}.json',result);rows.append(result)
            np.savez_compressed(folder/f'posterior-{order}.npz',sources=bank.sources,weights=bank.weights,component_loglikelihood=ll)
            del bank
        changes=[]
        for a,b in zip(rows[0]['source'],rows[-1]['source']):
            shift=float(np.linalg.norm(np.array(a['mean_yx'])-b['mean_yx']));mass=abs(a['mass_within_0_05_fov']-b['mass_within_0_05_fov'])
            changes.append({'method':a['method'],'mean_shift_fov':shift,'mass_radius_005_change':mass,
                'hpd95_membership_same':a['true_in_hpd95']==b['true_in_hpd95'],
                'passed':shift<cfg['source_refinement']['mean_shift_fov_tolerance'] and mass<cfg['source_refinement']['mass_radius_005_absolute_tolerance'] and a['true_in_hpd95']==b['true_in_hpd95']})
        refinements.append({'control':label,'methods':changes});allrows.append(rows[-1])
        source_checks[label]={'evidence_sha256':m.digest(evidence),'truth_sha256':m.digest(parent/label/'truth.npz')}
        print(json.dumps({'control':label,'final_source':rows[-1]['source'],'max_shift':max(x['mean_shift_fov'] for x in changes)}),flush=True)
    summary={'mean_control':allrows[0]['source'],'stochastic_controls':[],'refinement':refinements,
             'refinement_passed':all(c['passed'] for r in refinements for c in r['methods'])}
    for i,name in enumerate(m.NAMES):
        rows=[r['source'][i] for r in allrows[1:]];shifts=np.array([r['displacement_yx'] for r in rows]);errors=np.linalg.norm(shifts,axis=1)
        summary['stochastic_controls'].append({'method':name,'seed_count':len(rows),
            'mean_error_fov':float(errors.mean()),'min_error_fov':float(errors.min()),'max_error_fov':float(errors.max()),
            'mean_displacement_yx':shifts.mean(0).tolist(),
            'displacement_standard_error_yx':(shifts.std(0,ddof=1)/np.sqrt(len(rows))).tolist(),
            'true_in_hpd95_count':sum(r['true_in_hpd95'] for r in rows),
            'mean_mass_within_0_05_fov':float(np.mean([r['mass_within_0_05_fov'] for r in rows]))})
    m.save(output/'summary.json',summary)
    m.save(output/'provenance.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'command':sys.argv,'parent_manifest_sha256':m.digest(parent/'artifact-manifest.json'),'inputs':source_checks,
        'scope':'Final source posterior only. Predictive and event-averaged field traces remain at declared parent grid orders.'})
    imp=output/'implementation';imp.mkdir()
    for file in [Path(__file__),Path('src/plume_focus/source_likelihood_quadrature.py'),Path('tests/test_source_likelihood_quadrature.py'),Path('scripts/model-consistent-source-priors.py'),Path('scripts/compare-source-priors.py'),Path('src/plume_focus/source_prior_belief.py')]:shutil.copyfile(file,imp/file.name)
    plot(cfg,allrows,summary,output)
    m.save(output/'artifact-manifest.json',{'wall_seconds':time.time()-start,'files':{str(p.relative_to(output)):m.digest(p) for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({'wall_seconds':time.time()-start,'refinement_passed':summary['refinement_passed']}),flush=True)


def plot(cfg,results,summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    old=json.loads(Path('outputs/2026-10-28-source-priors-refinement-01/results-129.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    x=np.arange(3);width=.26
    for offset,values,label,color in [(-1,[r['final_source_mean_error_fov'] for r in old['methods'][:3]],'Original mismatched demo','#999999'),
        (0,[r['mean_error_fov'] for r in summary['mean_control'][:3]],'Matched mean','#1674b8'),
        (1,[r['mean_error_fov'] for r in summary['stochastic_controls'][:3]],'Matched stochastic: 10-seed mean','#d67d00')]:
        axes[0].bar(x+offset*width,values,width,label=label,color=color)
    vals=np.array([[r['source'][i]['mean_error_fov'] for i in range(3)] for r in results[1:]])
    for row in vals:axes[0].scatter(x+width,row,c='black',s=12,alpha=.5,zorder=3)
    axes[0].set_xticks(x,m.NAMES[:3]);axes[0].set(ylabel='Final source mean error (FOV)',title='Final source quadrature: '+str(cfg['candidate_grid_orders'][-1]))
    axes[0].legend(fontsize=8)
    for i,name in enumerate(m.NAMES[:3]):
        points=np.array([r['source'][i]['mean_yx'] for r in results[1:]])
        axes[1].scatter(points[:,1],points[:,0],label=name,s=35,alpha=.7)
    y,x=cfg['accurate_center_yx'];axes[1].scatter(x,y,c='red',marker='+',s=200,label='True location')
    axes[1].set(xlabel='x / FOV',ylabel='y / FOV',title='Matched seeds: posterior means (zoomed)');axes[1].legend(fontsize=8)
    fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
