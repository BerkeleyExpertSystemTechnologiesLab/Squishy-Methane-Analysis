"""One-at-a-time forward-model mismatch, frozen path and paired random draws."""
from pathlib import Path
import argparse,importlib.util,json,shutil,subprocess,sys,time
import numpy as np
from scipy.special import logsumexp
from types import SimpleNamespace
from plume_focus.mismatch_control import generate_mismatch
from plume_focus.source_likelihood_quadrature import AffineSourceLikelihood
from plume_focus.full_fov_execution import decode_config
from plume_focus.sensing_contracts import CausalWind,SensorReading
from plume_focus.signal_belief import observation_weights

spec=importlib.util.spec_from_file_location('matched',Path(__file__).with_name('model-consistent-source-priors.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def fit(run,events):
    a=AffineSourceLikelihood(run.model,half_life_seconds=run.session.half_life_seconds,transport_id=run.session.transport_id)
    wind=CausalWind();count=0
    for e in events:
        op=e['operation'];at=e['reading']['acquired_at_seconds'] if op=='ingest_readings' else e['at_seconds']
        a.advance(at,wind)
        if op=='plan_next':wind=CausalWind.from_dict(e['wind'])
        elif op=='ingest_readings':
            r=SensorReading.from_dict(e['reading']);h=observation_weights(run.model.grid_shape,r.position_yx,r.footprint)
            a.condition((r.value-run.model.offset)/run.model.gain,h,(run.model.sensor_noise_std/run.model.gain)**2);count+=1
    assert count==708
    return a


def grid_cache(cfg,run,order,precision):
    b=m.prior_bank(cfg,run.model,run.session,order)
    out=SimpleNamespace(sources=b.sources,emissions=b.emissions,logprior=b.log_weights.copy())
    del b
    out.quadratic=.5*np.sum((out.emissions@precision)*out.emissions,axis=1)
    return out


def evaluate(cfg,a,grid,order,truth,folder):
    ll=a.constant+grid.emissions@a.linear-grid.quadratic
    logweights=grid.logprior+ll[None];weights=np.exp(logweights-logsumexp(logweights,axis=1)[:,None])
    bank=SimpleNamespace(sources=grid.sources,weights=weights)
    stats=m.source_summary(bank,cfg,ll,order)
    fields=a.reference.mean[None]+(weights@grid.emissions)@a.response.T
    for i,row in enumerate(stats):row['final_latent_field_mse']=float(np.mean((fields[i]-truth)**2))
    result={'grid_order':order,'methods':stats}
    m.save(folder/f'results-{order}.json',result)
    np.savez_compressed(folder/f'posterior-{order}.npz',weights=weights,component_loglikelihood=ll,final_field_mean=fields)
    return result


def refinement(cfg,low,high):
    rows=[]
    for a,b in zip(low['methods'],high['methods']):
        shift=float(np.linalg.norm(np.array(a['mean_yx'])-b['mean_yx']));mass=abs(a['mass_within_0_05_fov']-b['mass_within_0_05_fov'])
        same=a['true_in_hpd95']==b['true_in_hpd95']
        rows.append({'method':a['method'],'mean_shift_fov':shift,'radius_mass_change':mass,'hpd95_membership_same':same,
            'passed':shift<cfg['source_refinement']['mean_shift_fov_tolerance'] and mass<cfg['source_refinement']['mass_radius_005_absolute_tolerance'] and same})
    return {'orders':[low['grid_order'],high['grid_order']],'methods':rows,'passed':all(x['passed'] for x in rows)}


def summarize(cfg,records,output):
    rows=[]
    for scenario in cfg['scenarios']:
        label=scenario['label'];controls=records[label];stats=[]
        for i,name in enumerate(m.NAMES):
            samples=[controls[c['label']]['final']['methods'][i] for c in cfg['controls'][1:]]
            base=[records['matched'][c['label']]['final']['methods'][i] for c in cfg['controls'][1:]]
            displacement=np.array([s['displacement_yx'] for s in samples]);errors=np.linalg.norm(displacement,axis=1)
            delta=errors-np.array([s['mean_error_fov'] for s in base])
            pairshift=displacement-np.array([s['displacement_yx'] for s in base])
            stats.append({'method':name,'mean_control':controls[cfg['controls'][0]['label']]['final']['methods'][i],
                'mean_error_fov':float(errors.mean()),'min_error_fov':float(errors.min()),'max_error_fov':float(errors.max()),
                'mean_displacement_yx':displacement.mean(0).tolist(),'mean_paired_displacement_change_yx':pairshift.mean(0).tolist(),
                'mean_paired_error_change':float(delta.mean()),'paired_error_change_standard_error':float(delta.std(ddof=1)/np.sqrt(len(delta))),
                'positive_paired_error_changes':int(np.sum(delta>0)),
                'true_in_hpd95_count':sum(s['true_in_hpd95'] for s in samples),
                'large_error_count':int(np.sum(errors>=cfg['summary_criteria']['large_error_fov'])),
                'original_like_shift_count':int(np.sum((displacement[:,1]>=cfg['summary_criteria']['original_like_x_shift_fov'])&(np.abs(displacement[:,0])<=cfg['summary_criteria']['original_like_abs_y_shift_max']))),
                'mean_mass_within_0_05_fov':float(np.mean([s['mass_within_0_05_fov'] for s in samples])),
                'mean_final_latent_field_mse':float(np.mean([s['final_latent_field_mse'] for s in samples]))})
        rows.append({**scenario,'methods':stats,'all_refinement_passed':all(v['refinement'][-1]['passed'] for v in controls.values())})
    result={'scenarios':rows,'seeds':[c['seed'] for c in cfg['controls'][1:]],'source_refinement':cfg['source_refinement'],
            'all_refinement_passed':all(s['all_refinement_passed'] for s in rows),
            'scope':'Fixed geometry, 10 paired seeds; source error and HPD diagnostics only; no physical-time J/T or policy claim.'}
    m.save(output/'summary.json',result)
    return result


def plots(summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#555555','#1674b8','#d67d00'];fig,axes=plt.subplots(2,3,figsize=(13,8),layout='constrained')
    baseline=summary['scenarios'][0]
    for col,factor in enumerate(['wind_speed','source_width','half_life']):
        scenarios=[s for s in summary['scenarios'] if s['factor']==factor]+[baseline]
        scenarios.sort(key=lambda s:s['multiplier'])
        x=[s['multiplier'] for s in scenarios]
        for i,name in enumerate(m.NAMES[:3]):
            axes[0,col].plot(x,[s['methods'][i]['mean_error_fov'] for s in scenarios],marker='o',label=name,color=colors[i])
            axes[1,col].plot(x,[s['methods'][i]['true_in_hpd95_count'] for s in scenarios],marker='o',color=colors[i])
        axes[0,col].axhline(.06285,color='#b83e55',ls='--',label='Original demo ~.063 FOV')
        axes[0,col].set(title=factor,ylabel='Mean source error (10 paired seeds)',xlabel='Truth / assumed parameter')
        axes[1,col].set(ylabel='Truth in 95% HPD set (out of 10)',xlabel='Truth / assumed parameter',ylim=(-.3,10.3))
        axes[0,col].legend(fontsize=7)
    fig.suptitle('One-factor mismatch: frozen 40 s / 708-observation path')
    fig.savefig(output/'comparison.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,factor in zip(axes,['wind_speed','source_width','half_life']):
        for i,name in enumerate(m.NAMES[:3]):
            scenarios=sorted([s for s in summary['scenarios'] if s['factor']==factor]+[baseline],key=lambda s:s['multiplier'])
            shifts=np.array([s['methods'][i]['mean_displacement_yx'] for s in scenarios])
            ax.plot(shifts[:,1],shifts[:,0],'o-',label=name,color=colors[i])
            if i==0:
                for s,(dy,dx) in zip(scenarios,shifts):ax.annotate(str(s['multiplier']), (dx,dy),fontsize=8)
        ax.scatter(0,0,c='red',marker='+',s=120,label='Truth')
        ax.scatter(.0626,-.005,c='black',marker='x',s=60,label='Original approx.')
        ax.set(title=factor,xlabel='Mean x displacement / FOV',ylabel='Mean y displacement / FOV');ax.legend(fontsize=7)
    fig.savefig(output/'displacement.png',dpi=170);plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();start=time.time();cfg=json.loads(Path(args.config).read_text());output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    recordpath=Path(cfg['record']);assert m.digest(recordpath)==cfg['record_sha256']
    record=json.loads(recordpath.read_text());run=decode_config(record['configuration'])
    shutil.copyfile(args.config,output/'config.json');shutil.copyfile(recordpath,output/'original-path-and-evidence.json')
    m.save(output/'baseline-configuration.json',record['configuration'])
    m.save(output/'provenance.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],text=True),'command':sys.argv,'python':sys.version})
    (output/'requirements-frozen.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    caches={};precision=None;baserng={};records={};checks=[]
    for scenario in cfg['scenarios']:
        label=scenario['label'];records[label]={}
        for control in cfg['controls']:
            cl=control['label'];folder=output/label/cl;folder.mkdir(parents=True)
            events,states,metadata=generate_mismatch(record,cfg,run,control['seed'],scenario['factor'],scenario['multiplier'])
            if label=='matched':baserng[cl]=metadata['rng_signatures']
            else:assert metadata['rng_signatures']==baserng[cl], 'Random draw alignment changed'
            m.save(folder/'generated-evidence.json',{'events':events,'seed':control['seed']});m.save(folder/'truth-parameters.json',metadata)
            np.savez_compressed(folder/'truth.npz',states=states,state_times=[0.]+[e['reading']['acquired_at_seconds'] if e['operation']=='ingest_readings' else e['at_seconds'] for e in events])
            if label=='matched':
                old=Path(cfg['previous_matched_output'])/cl
                assert events==json.loads((old/'generated-evidence.json').read_text())['events']
                with np.load(old/'truth.npz') as data:np.testing.assert_array_equal(states,data['states'])
            a=fit(run,events)
            if precision is None:precision=a.precision.copy()
            else:np.testing.assert_array_equal(a.precision,precision)
            np.savez_compressed(folder/'likelihood-statistics.npz',precision=a.precision,linear=a.linear,constant=a.constant,
                response=a.response,mean=a.reference.mean,covariance=a.reference.covariance)
            results=[];refinements=[]
            for order in cfg['candidate_grid_orders']+cfg['refinement_orders_if_needed']:
                if order not in cfg['candidate_grid_orders'] and refinements[-1]['passed']:break
                if order not in caches:
                    caches[order]=grid_cache(cfg,run,order,precision)
                    np.savez_compressed(output/f'candidate-grid-{order}.npz',sources=caches[order].sources,logprior=caches[order].logprior)
                result=evaluate(cfg,a,caches[order],order,states[-1],folder);results.append(result)
                if len(results)>1:refinements.append(refinement(cfg,results[-2],result))
                if label=='matched' and order==257:
                    old=json.loads((Path(cfg['previous_source_output'])/cl/'results-257.json').read_text())['source']
                    difference=max(float(np.linalg.norm(np.array(x['mean_yx'])-y['mean_yx'])) for x,y in zip(result['methods'],old))
                    assert difference<1e-10;checks.append({'control':cl,'matched_source_mean_difference':difference})
            m.save(folder/'refinement.json',refinements)
            records[label][cl]={'final':results[-1],'refinement':refinements}
        print(json.dumps({'scenario':label,'completed_controls':len(records[label]),'max_final_grid':max(v['final']['grid_order'] for v in records[label].values()),
            'broad_mean_error':float(np.mean([records[label][c['label']]['final']['methods'][0]['mean_error_fov'] for c in cfg['controls'][1:]])),
            'refinement_passed':all(v['refinement'][-1]['passed'] for v in records[label].values())}),flush=True)
    summary=summarize(cfg,records,output);plots(summary,output)
    m.save(output/'verification.json',{'previous_matched_evidence_truth_exact':True,'same_random_draw_alignment_every_event':True,
        'inference_precision_identical_all_runs':True,'matched_source_checks':checks,'all_refinement_passed':summary['all_refinement_passed']})
    imp=output/'implementation';imp.mkdir()
    for f in [Path(__file__),Path('src/plume_focus/mismatch_control.py'),Path('src/plume_focus/model_consistent_control.py'),
        Path('src/plume_focus/source_likelihood_quadrature.py'),Path('src/plume_focus/source_prior_belief.py'),Path('scripts/model-consistent-source-priors.py'),
        Path('scripts/compare-source-priors.py'),Path('tests/test_mismatch_control.py')]:shutil.copyfile(f,imp/f.name)
    assert m.digest(recordpath)==cfg['record_sha256']
    m.save(output/'artifact-manifest.json',{'wall_seconds':time.time()-start,'original_evidence_unchanged':True,
        'files':{str(p.relative_to(output)):m.digest(p) for p in output.rglob('*') if p.is_file()}})
    print(json.dumps({'wall_seconds':time.time()-start,'all_refinement_passed':summary['all_refinement_passed']}),flush=True)


if __name__=='__main__':main()
