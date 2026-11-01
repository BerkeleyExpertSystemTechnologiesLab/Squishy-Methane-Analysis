"""Frozen-path matched-model source-prior control; no policy execution or tuning."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import platform
import shutil
import subprocess
import sys
import time
import numpy as np
from scipy.special import ndtr
from plume_focus.model_consistent_control import MatchedFieldGenerator
from plume_focus.calibrated_scanner import DecayGaussianBelief
from plume_focus.full_fov_execution import decode_config
from plume_focus.sensing_contracts import CausalWind, SensorReading
from plume_focus.signal_belief import observation_weights

spec=importlib.util.spec_from_file_location('fixed_prior_study',Path(__file__).with_name('compare-source-priors.py'))
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)
prior_bank,save,digest,NAMES=common.prior_bank,common.save,common.digest,common.NAMES


def generate(record,config,run,seed):
    gen=MatchedFieldGenerator(run.model,config['accurate_center_yx'],
        half_life_seconds=run.session.half_life_seconds,transport_id=run.session.transport_id,seed=seed)
    wind=CausalWind(); events=[]; states=[gen.state.copy()]
    for old in record['events']:
        op=old['operation'];at=old['at_seconds']
        e={'operation':op,'at_seconds':at}
        if op=='ingest_readings':
            if len(old['readings'])!=1: raise ValueError('Singleton streaming history required')
            reading=SensorReading.from_dict(old['readings'][0])
            if reading.acquired_at_seconds < gen.time_seconds-1e-10 or reading.available_at_seconds>at+1e-10:
                raise ValueError('Control requires chronological, delivered streaming')
            gen.advance(reading.acquired_at_seconds,wind)
            e['reading']={**old['readings'][0],'value':gen.observe(reading)}
        elif op=='plan_next':
            gen.advance(at,wind);wind=CausalWind.from_dict(old['wind']);e['wind']=old['wind']
        elif op=='finish_scan':
            gen.advance(at,wind);e['scan_id']=old['receipt']['scan_id']
        else: raise ValueError('Unknown event')
        events.append(e);states.append(gen.state.copy())
    gen.advance(run.duration_seconds,wind)
    events.append({'operation':'end','at_seconds':run.duration_seconds});states.append(gen.state.copy())
    return events,np.array(states)


def source_summary(bank,config,component_ll,order):
    n=order*order;w=bank.weights;truth=np.asarray(config['accurate_center_yx'])
    centers=np.asarray([config['accurate_center_yx'],config['biased_center_yx']])
    mean=w@bank.sources;error=mean-truth
    radius=np.linalg.norm(bank.sources-truth,axis=1)
    summaries=[]
    for i,name in enumerate(NAMES):
        row={'method':name,'mean_yx':mean[i].tolist(),'displacement_yx':error[i].tolist(),
             'mean_error_fov':float(np.linalg.norm(error[i])),
             'mass_within_0_05_fov':float(w[i]@(radius<=.05)),
             'ess':float(1/np.sum(w[i]**2))}
        if i<3:
            if i==0: logprior=np.zeros(n);lp_true=0.
            else:
                logprior=-.5*np.sum(((bank.sources[:n]-centers[i-1])/config['region_std_yx'])**2,axis=1)
                lp_true=-.5*np.sum(((truth-centers[i-1])/config['region_std_yx'])**2)
            # Compare continuous densities using exact true-source likelihood,
            # NOT nearest grid point or a Gaussian ellipse around the posterior mean.
            rank=float(w[i,:n]@(logprior+component_ll[:n]>lp_true+component_ll[-2]+1e-10))
            row.update(true_location_hpd_rank=rank,true_in_hpd95=rank<=.95)
        else:
            row.update(true_location_hpd_rank=None,true_in_hpd95=bool(i==3))
        summaries.append(row)
    return summaries


def run_one(config,run,events,states,order,folder):
    bank=prior_bank(config,run.model,run.session,order);initial=bank.weights.copy()
    ref=DecayGaussianBelief(run.model,half_life_seconds=run.session.half_life_seconds,
                            transport_id=run.session.transport_id)
    wind=CausalWind(); component_ll=np.zeros(len(bank.sources));trace=[];snapshots=[]
    snapshot_arrays={};max_mean=0.;max_cov=0.
    def snapshot(label,t,truth):
        means=bank.weights@bank.means;within=np.diag(bank.reference.covariance)
        cdf=bank.weights@ndtr((truth[None]-bank.means)/np.sqrt(np.maximum(within,1e-30))[None])
        snapshots.append({'label':label,'time':t,'source':source_summary(bank,config,component_ll,order),
                          'field_mse':np.mean((means-truth)**2,axis=1).tolist(),
                          'field_coverage95':np.mean((cdf>=.025)&(cdf<=.975),axis=1).tolist()})
        snapshot_arrays[label+'_weights']=bank.weights.copy()
        snapshot_arrays[label+'_field_mean']=means
        snapshot_arrays[label+'_truth']=truth.copy()
    snapshot('start',0.,states[0])
    for j,e in enumerate(events):
        op=e['operation'];at=e['at_seconds']
        state_time=e['reading']['acquired_at_seconds'] if op=='ingest_readings' else at
        bank.advance(state_time,wind)
        ref.advance(state_time,wind,config['accurate_center_yx'])
        if op=='plan_next':wind=CausalWind.from_dict(e['wind'])
        elif op=='ingest_readings':
            reading=SensorReading.from_dict(e['reading'])
            h=observation_weights(run.model.grid_shape,reading.position_yx,reading.footprint)
            value=(reading.value-run.model.offset)/run.model.gain
            noise=(run.model.sensor_noise_std/run.model.gain)**2
            predicted=bank.means@h;variance=float(h@bank.reference.covariance@h+noise)
            component_ll+=-.5*(math.log(2*math.pi*variance)+(value-predicted)**2/variance)
            scores=bank.condition(value,h,noise);ref.condition(value,h,noise)
            trace.append({'reading_id':reading.reading_id,'available_at_seconds':at,'acquired_at_seconds':state_time,'value':value,
                **{k:v.tolist() for k,v in scores.items()},
                'post_reading_latent_mse':np.mean((bank.weights@bank.means-states[j+1])**2,axis=1).tolist()})
        elif op not in ('finish_scan','end'):raise ValueError('Unknown operation')
        max_mean=max(max_mean,float(np.max(np.abs(bank.means[-2]-ref.mean))))
        max_cov=max(max_cov,float(np.max(np.abs(bank.reference.covariance-ref.covariance))))
        if op in ('finish_scan','end'):snapshot(e.get('scan_id','final'),at,states[j+1])
    assert len(trace)==708 and max_mean<1e-10 and max_cov<1e-10
    predictions=np.array([r['prediction'] for r in trace]);values=np.array([r['value'] for r in trace])
    cdf=np.array([r['predictive_cdf'] for r in trace]);lp=np.array([r['log_predictive_density'] for r in trace])
    mse=np.array([r['post_reading_latent_mse'] for r in trace]);rows=[]
    for i,s in enumerate(snapshots[-1]['source']):
        rows.append({**s,'reading_rmse':float(np.sqrt(np.mean((predictions[:,i]-values)**2))),
            'mean_nlpd':float(-lp[:,i].mean()),'reading_coverage95':float(np.mean((cdf[:,i]>=.025)&(cdf[:,i]<=.975))),
            'mean_post_reading_latent_mse':float(mse[:,i].mean()),
            'final_field_mse':snapshots[-1]['field_mse'][i],
            'final_field_coverage95':snapshots[-1]['field_coverage95'][i]})
    result={'candidate_grid_order':order,'methods':rows,'snapshots':snapshots,
        'checks':{'reading_count':len(trace),'same_evidence_for_all_priors':True,
                  'max_exact_point_vs_original_mean_error':max_mean,'max_covariance_error':max_cov},
        'max_grid_loglik_minus_exact_true':float(component_ll[:order*order].max()-component_ll[-2])}
    save(folder/f'results-{order}.json',result);save(folder/f'prediction-trace-{order}.json',trace)
    np.savez_compressed(folder/f'beliefs-{order}.npz',sources=bank.sources,initial_weights=initial,
                        final_component_loglikelihood=component_ll,**snapshot_arrays)
    print(json.dumps({'control':folder.name,'order':order,'methods':rows}),flush=True)
    return result


def summarize(config,all_results,output):
    fine=[r[-1] for r in all_results];stochastic=fine[1:];rows=[];refinements=[]
    for i,name in enumerate(NAMES):
        draws=[r['methods'][i] for r in stochastic]
        shifts=np.array([d['displacement_yx'] for d in draws]);errors=np.linalg.norm(shifts,axis=1)
        rows.append({'method':name,'seed_count':len(draws),'mean_error_fov':float(errors.mean()),
            'median_error_fov':float(np.median(errors)),'min_error_fov':float(errors.min()),'max_error_fov':float(errors.max()),
            'mean_displacement_yx':shifts.mean(0).tolist(),'displacement_standard_error_yx':(shifts.std(0,ddof=1)/np.sqrt(len(draws))).tolist(),
            'true_in_hpd95_count':sum(d['true_in_hpd95'] for d in draws),
            'mean_mass_within_0_05_fov':float(np.mean([d['mass_within_0_05_fov'] for d in draws])),
            'mean_reading_coverage95':float(np.mean([d['reading_coverage95'] for d in draws])),
            'mean_final_field_coverage95':float(np.mean([d['final_field_coverage95'] for d in draws])),
            'mean_post_reading_latent_mse':float(np.mean([d['mean_post_reading_latent_mse'] for d in draws]))})
    for control,results in zip(config['controls'],all_results):
        pairs=[]
        for a,b in zip(results[0]['methods'],results[-1]['methods']):
            shift=float(np.linalg.norm(np.array(a['mean_yx'])-b['mean_yx']))
            mass=abs(a['mass_within_0_05_fov']-b['mass_within_0_05_fov'])
            pairs.append({'method':a['method'],'mean_shift_fov':shift,'mass_radius_005_change':mass,
                'hpd95_membership_same':a['true_in_hpd95']==b['true_in_hpd95'],
                'passed':shift<config['source_refinement']['mean_shift_fov_tolerance'] and
                         mass<config['source_refinement']['mass_radius_005_absolute_tolerance'] and
                         a['true_in_hpd95']==b['true_in_hpd95']})
        refinements.append({'control':control['label'],'methods':pairs})
    summary={'mean_control':fine[0]['methods'],'stochastic_controls':rows,
             'refinement':refinements,'refinement_passed':all(p['passed'] for r in refinements for p in r['methods']),
             'scope':'Fixed true location and path; HPD counts are descriptive, not prior-predictive SBC or general calibration.'}
    save(output/'summary.json',summary)
    return summary


def plots(config,results,summary,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#777777','#1674b8','#d67d00','#258b53','#b83e55']
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout='constrained')
    old=json.loads(Path('outputs/2026-10-28-source-priors-refinement-01/results-129.json').read_text())
    x=np.arange(5);width=.26
    axes[0].bar(x-width,[r['final_source_mean_error_fov'] for r in old['methods']],width,label='Original mismatched demo',color='#bcbcbc')
    axes[0].bar(x,[r['mean_error_fov'] for r in summary['mean_control']],width,label='Matched mean control',color='#1674b8')
    vals=np.array([[r[-1]['methods'][i]['mean_error_fov'] for i in range(5)] for r in results[1:]])
    axes[0].bar(x+width,vals.mean(0),width,label='Matched stochastic: 10-seed mean',color='#d67d00')
    for row in vals:axes[0].scatter(x+width,row,s=10,color='black',alpha=.45,zorder=3)
    axes[0].set_xticks(x,NAMES,rotation=25,ha='right',fontsize=8)
    axes[0].set(ylabel='Final source posterior-mean error (FOV)',title='Same true source and frozen path')
    axes[0].legend(fontsize=8)
    for i,name in enumerate(NAMES[:3]):
        means=np.array([r[-1]['methods'][i]['mean_yx'] for r in results[1:]])
        axes[1].scatter(means[:,1],means[:,0],label=name,color=colors[i],s=35,alpha=.7)
        original=old['snapshots'][-1]['source_mean_yx'][i]
        axes[1].scatter(original[1],original[0],marker='x',s=90,color=colors[i])
    y,x=config['accurate_center_yx'];axes[1].scatter(x,y,marker='+',s=220,c='red',label='True source')
    axes[1].set(xlabel='x / FOV',ylabel='y / FOV',title='Dots: matched seeds; x: original demo')
    axes[1].legend(fontsize=8);fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    order=results[0][-1]['candidate_grid_order']
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for row,label in enumerate([config['controls'][0]['label'],config['controls'][1]['label']]):
        data=np.load(output/label/f'beliefs-{order}.npz')
        for i,name in enumerate(NAMES[:3]):
            ax=axes[row,i];density=data['final_weights'][i,:order*order].reshape(order,order)*order**2
            im=ax.imshow(density,origin='lower',extent=(0,1,0,1),aspect='equal')
            y,x=config['accurate_center_yx'];ax.plot(x,y,'r+',ms=9)
            ax.set(title=f'{label}: {name}',xlabel='x / FOV',ylabel='y / FOV')
            fig.colorbar(im,ax=ax,shrink=.7,label='Source density')
    fig.suptitle('Final posterior; first stochastic seed selected in advance')
    fig.savefig(output/'source-posteriors.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();started=time.time();config=json.loads(Path(args.config).read_text())
    record_path=Path(config['record']);assert digest(record_path)==config['record_sha256']
    record=json.loads(record_path.read_text());run=decode_config(record['configuration'])
    assert run.model.gain==1 and run.model.offset==0
    assert np.allclose(config['accurate_center_yx'],run.session.source_position,rtol=0,atol=1e-15)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.config,output/'config.json');shutil.copyfile(record_path,output/'original-path-and-evidence.json')
    save(output/'baseline-configuration.json',record['configuration'])
    save(output/'provenance.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],text=True),
        'command':sys.argv,'python':sys.version,'platform':platform.platform(),'original_record_sha256':digest(record_path)})
    (output/'requirements-frozen.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    implementation=output/'implementation';implementation.mkdir()
    for file in [Path(__file__),Path('scripts/compare-source-priors.py'),Path('src/plume_focus/source_prior_belief.py'),
                 Path('src/plume_focus/model_consistent_control.py'),Path('tests/test_model_consistent_control.py')]:
        shutil.copyfile(file,implementation/file.name)
    all_results=[]
    for control in config['controls']:
        folder=output/control['label'];folder.mkdir()
        events,states=generate(record,config,run,control['seed'])
        save(folder/'generated-evidence.json',{'seed':control['seed'],'events':events,
             'contract':'Frozen path observation adapter; original plan decisions are not replayed on new readings.'})
        np.savez_compressed(folder/'truth.npz',states=states,state_times=[0.]+[e['reading']['acquired_at_seconds'] if e['operation']=='ingest_readings' else e['at_seconds'] for e in events])
        all_results.append([run_one(config,run,events,states,order,folder) for order in config['candidate_grid_orders']])
    summary=summarize(config,all_results,output);plots(config,all_results,summary,output)
    assert digest(record_path)==config['record_sha256']
    save(output/'artifact-manifest.json',{'wall_seconds':time.time()-started,'original_evidence_unchanged':True,
        'files':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'output':str(output),'wall_seconds':time.time()-started,'refinement_passed':summary['refinement_passed']}),flush=True)


if __name__=='__main__': main()
