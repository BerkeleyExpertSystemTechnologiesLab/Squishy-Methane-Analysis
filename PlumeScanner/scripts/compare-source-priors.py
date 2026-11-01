"""Run the declared fixed-evidence experiment; never replan or overwrite evidence."""
from pathlib import Path
import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
from scipy.special import logsumexp
from plume_focus.source_prior_belief import SourcePriorBank
from plume_focus.full_fov_execution import decode_config, make_controller, replay_event
from plume_focus.sensing_contracts import CausalWind, SensorReading
from plume_focus.sensing_counterfactual import TruthQuadrature
from plume_focus.signal_belief import grid_positions

NAMES = ['broad','accurate_region','biased_region','accurate_point','biased_point']


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value): Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prior_bank(config, model, session, order):
    nodes = grid_positions((order,order))
    centers = np.asarray([config['accurate_center_yx'],config['biased_center_yx']])
    sources = np.concatenate([nodes,centers])
    priors = np.zeros((5,len(sources)))
    priors[0,:len(nodes)] = 1/len(nodes)
    for i,center in enumerate(centers,1):
        logp = -.5*np.sum(((nodes-center)/config['region_std_yx'])**2,axis=1)
        priors[i,:len(nodes)] = np.exp(logp-logsumexp(logp))
    priors[3,-2] = priors[4,-1] = 1.
    return SourcePriorBank(model,sources,priors,half_life_seconds=session.half_life_seconds,
                           transport_id=session.transport_id)


def run_one(config, record, run_config, truth, order, output):
    bank = prior_bank(config,run_config.model,run_config.session,order)
    initial_weights = bank.weights.copy()
    wind = CausalWind()
    controller = make_controller(run_config)
    cursor = 0; loss_rows=[]; observations=[]; snapshots=[]; snapshot_arrays={}
    max_mean_difference = 0.; max_covariance_difference = 0.
    source = run_config.build_source()  # Evaluator only; never passed to bank.
    last_available = 0.
    last_acquired = -1.

    def before(at):
        nonlocal cursor
        while cursor<len(truth.times) and truth.times[cursor]<at-1e-10:
            t=float(truth.times[cursor])
            if t<last_available-1e-9: raise ValueError('Online score uses future evidence')
            means=bank.forecast_aggregate(t,wind)
            mse=np.maximum(0.,np.einsum('mi,ij,mj->m',means,truth.gram,means)
                           -2*means@truth.cross[cursor]+truth.square[cursor])
            loss_rows.append({'time':t,'duration':float(truth.durations[cursor]),'mse':mse.tolist()})
            cursor+=1

    def snapshot(t,label):
        shape=tuple(config['snapshot_shape']); points=grid_positions(shape)
        values=source.signal_at(np.full(len(points),t),points)
        view=bank.projected_snapshot(t,wind,shape,values)
        probabilities=bank.weights
        source_mean=probabilities@bank.sources
        distance=np.linalg.norm(bank.sources-np.asarray(config['accurate_center_yx']),axis=1)
        snapshots.append({'label':label,'time':t,'source_mean_yx':source_mean.tolist(),
            'source_mean_error_fov':np.linalg.norm(source_mean-config['accurate_center_yx'],axis=1).tolist(),
            'posterior_mass_within_0_05_fov':(probabilities@(distance<=.05)).tolist(),
            'ess':(1/np.sum(probabilities**2,axis=1)).tolist(),
            'field_coverage95':view['coverage95'].tolist(),
            'field_mse':np.mean((view['mean']-values)**2,axis=1).tolist()})
        for key in ('mean','std','within_variance','between_variance','weights'):
            snapshot_arrays[label+'_'+key]=view[key]
        snapshot_arrays[label+'_truth']=values

    snapshot(0.,'start')
    for event in record['events']:
        at=event['at_seconds']; before(at)
        if event['operation']=='plan_next':
            bank.advance(at,wind)
            wind=CausalWind.from_dict(event['wind'])
        elif event['operation']=='ingest_readings':
            if len(event['readings']) != 1: raise ValueError('v1 expects singleton streaming events')
            reading=SensorReading.from_dict(event['readings'][0])
            if (reading.acquired_at_seconds<last_available-1e-10
                or reading.acquired_at_seconds<last_acquired-1e-10
                or reading.available_at_seconds>at+1e-10):
                raise ValueError('Delayed/out-of-order history needs an explicit replay adapter')
            scores=bank.assimilate(reading,wind)
            observations.append({'reading_id':reading.reading_id,'acquired_at_seconds':reading.acquired_at_seconds,
                'available_at_seconds':at,'value':reading.value,
                **{key:value.tolist() for key,value in scores.items()}})
            last_acquired=reading.acquired_at_seconds
        elif event['operation']=='finish_scan': bank.advance(at,wind)
        else: raise ValueError('Unknown event')
        replay_event(controller,event)
        max_mean_difference=max(max_mean_difference,float(np.max(np.abs(bank.means[-2]-controller._belief.mean))))
        max_covariance_difference=max(max_covariance_difference,float(np.max(np.abs(bank.reference.covariance-controller._belief.covariance))))
        last_available=at
        if event['operation']=='finish_scan': snapshot(at,event['receipt']['scan_id'])
    before(run_config.duration_seconds+1e-8)
    snapshot(run_config.duration_seconds,'final')
    assert max_mean_difference<1e-10 and max_covariance_difference<1e-10
    errors=np.asarray([r['mse'] for r in loss_rows]); durations=np.asarray([r['duration'] for r in loss_rows])
    J=durations@errors
    assert abs(J[3]-record['J'])<1e-9, (J[3],record['J'])
    log_scores=np.asarray([r['log_predictive_density'] for r in observations])
    predicted=np.asarray([r['prediction'] for r in observations])
    actual=np.asarray([r['value'] for r in observations])
    cdf=np.asarray([r['predictive_cdf'] for r in observations])
    aggregate=[]
    for i,name in enumerate(NAMES):
        aggregate.append({'method':name,'J':float(J[i]),'J_over_T':float(J[i]/run_config.duration_seconds),
            'relative_J_change_vs_accurate_point':float(J[i]/J[3]-1),
            'prequential_reading_rmse':float(np.sqrt(np.mean((predicted[:,i]-actual)**2))),
            'mean_negative_log_predictive_density':float(-log_scores[:,i].mean()),
            'reading_coverage95':float(np.mean((cdf[:,i]>=.025)&(cdf[:,i]<=.975))),
            'final_source_mean_error_fov':snapshots[-1]['source_mean_error_fov'][i],
            'final_source_mass_within_0_05_fov':snapshots[-1]['posterior_mass_within_0_05_fov'][i],
            'final_field_coverage95':snapshots[-1]['field_coverage95'][i]})
    integral=bank.emissions.mean(axis=1)
    result={'candidate_grid_order':order,'methods':aggregate,'snapshots':snapshots,
        'checks':{'reading_count':len(observations),'public_baseline_plan_replay':'passed',
            'max_point_vs_original_mean_error':max_mean_difference,
            'max_shared_covariance_vs_original_error':max_covariance_difference,
            'point_baseline_J_error':float(abs(J[3]-record['J'])),
            'all_methods_same_evidence':True,'strict_streaming_causality':'passed'},
        'discrete_source_input_integral':{'candidate_min':float(integral.min()),'candidate_max':float(integral.max()),
            'prior_expected':(initial_weights@integral).tolist(),
            'note':'Unchanged point-sampled injection. Edge truncation and latent-grid alignment change the numerical integral; source_rate is peak amplitude, not conserved total emission.'}}
    save(output/f'results-{order}.json',result)
    save(output/f'loss-trace-{order}.json',loss_rows)
    save(output/f'prediction-trace-{order}.json',observations)
    np.savez_compressed(output/f'beliefs-{order}.npz',sources=bank.sources,initial_weights=initial_weights,**snapshot_arrays)
    print(json.dumps({'order':order,'results':aggregate,'checks':result['checks']}),flush=True)
    return result,loss_rows


def plots(output,config,results,traces):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#777777','#1674b8','#d67d00','#258b53','#b83e55']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    for i,name in enumerate(NAMES):
        axes[0].plot([r['time'] for r in traces],[r['mse'][i] for r in traces],label=name,color=colors[i])
    axes[0].set(xlabel='Time (s)',ylabel='Field MSE',title='Same 708 observations; causal estimates')
    axes[0].legend(fontsize=8)
    x=np.arange(5);width=.35
    for j,result in enumerate(results):
        axes[1].bar(x+(j-.5)*width,[r['J_over_T'] for r in result['methods']],width,
                    label=f"{result['candidate_grid_order']} x {result['candidate_grid_order']} source grid")
    axes[1].set_xticks(x,NAMES,rotation=25,ha='right',fontsize=8)
    axes[1].set(ylabel='Time-averaged field MSE (J/T)',title='Candidate-grid refinement')
    axes[1].legend(fontsize=8)
    fig.savefig(output/'comparison.png',dpi=180);plt.close(fig)
    order=results[-1]['candidate_grid_order']; data=np.load(output/f'beliefs-{order}.npz')
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    for i,name in enumerate(NAMES[:3]):
        for row,key in enumerate(['initial_weights','final_weights']):
            field=data[key][i,:order*order].reshape(order,order)*order**2
            ax=axes[row,i]; im=ax.imshow(field,origin='lower',extent=(0,1,0,1),cmap='viridis',aspect='equal')
            y,x=config['accurate_center_yx'];ax.plot(x,y,'r+',ms=9)
            ax.set(title=f"{name}: {'prior' if row==0 else 'posterior at 40 s'}",xlabel='x / FOV',ylabel='y / FOV')
            fig.colorbar(im,ax=ax,shrink=.8,label='Source probability density')
    fig.suptitle('Source-location distributions; red + = known generator location (evaluation only)')
    fig.savefig(output/'source-posteriors.png',dpi=180);plt.close(fig)
    shape=tuple(config['snapshot_shape'])
    fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained')
    arrays=[data['final_truth']]+list(data['final_mean']);vmax=max(float(a.max()) for a in arrays)
    for ax,array,name in zip(axes.flat,arrays,['Synthetic truth']+NAMES):
        im=ax.imshow(array.reshape(shape),origin='lower',extent=(0,1,0,1),vmin=0,vmax=vmax,cmap='magma')
        ax.set(title=name,xlabel='x / FOV',ylabel='y / FOV')
    fig.colorbar(im,ax=list(axes.flat),shrink=.8,label='Synthetic signal')
    fig.suptitle('Field means at 40 s; shared scale, same evidence')
    fig.savefig(output/'final-fields.png',dpi=180);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args(); started=time.time()
    config=json.loads(Path(args.config).read_text()); record_path=Path(config['record'])
    assert digest(record_path)==config['record_sha256'],'Frozen evidence hash mismatch'
    record=json.loads(record_path.read_text()); run_config=decode_config(record['configuration'])
    assert run_config.model.gain==1 and run_config.model.offset==0, 'v1 reported scalar metrics use identity retrieval'
    assert np.allclose(config['accurate_center_yx'],run_config.session.source_position,atol=1e-15,rtol=0)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(args.config,output/'config.json');save(output/'baseline-configuration.json',record['configuration'])
    shutil.copyfile(record_path,output/'frozen-evidence.json')
    save(output/'provenance.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        'command':sys.argv,'python':sys.version,'platform':platform.platform(),'record_sha256':digest(record_path),
        'claim':'Exploratory fixed-evidence comparison, not a closed-loop benchmark'})
    (output/'requirements-frozen.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    truth=TruthQuadrature(run_config.build_source(),0.,run_config.duration_seconds,run_config.model.grid_shape,
                         tuple(config['evaluation_shape']),config['evaluation_step_seconds'])
    results=[]
    for order in config['candidate_grid_orders']:
        result,trace=run_one(config,record,run_config,truth,order,output);results.append(result)
    changes=[abs(b['J']/a['J']-1) for a,b in zip(results[0]['methods'],results[-1]['methods'])]
    refinement={'relative_J_changes':dict(zip(NAMES,changes)),
        'tolerance':config['refinement_relative_J_tolerance'],
        'passed':all(v<config['refinement_relative_J_tolerance'] for v in changes),
        'scope':'Source-location quadrature only; latent grid, model error, temporal and spatial scoring meshes not refined here.'}
    save(output/'refinement.json',refinement);plots(output,config,results,trace)
    assert digest(record_path)==config['record_sha256']
    source_dir=output/'implementation';source_dir.mkdir()
    for file in [Path(__file__),Path('src/plume_focus/source_prior_belief.py'),Path('tests/test_source_prior_belief.py')]:
        shutil.copyfile(file,source_dir/file.name)
    save(output/'artifact-manifest.json',{'wall_seconds':time.time()-started,'original_evidence_unchanged':True,
        'files':{str(p.relative_to(output)):digest(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    print(json.dumps({'output':str(output),'refinement':refinement,'wall_seconds':time.time()-started}),flush=True)


if __name__=='__main__':main()
