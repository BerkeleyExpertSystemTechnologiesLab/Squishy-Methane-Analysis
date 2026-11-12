"""Matched fixed-path low/nominal/high input: location-only versus joint inference."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, sys, time
import numpy as np
from scipy.special import logsumexp
from plume_focus.full_fov_execution import decode_config
from plume_focus.sensing_contracts import CausalWind, SensorReading
from plume_focus.signal_belief import grid_positions, observation_weights
from plume_focus.source_prior_belief import transport_step
from plume_focus.source_likelihood_quadrature import AffineSourceLikelihood
from plume_focus.source_input import gaussian_source_input
from plume_focus.source_amount_control import (generate_history, SourceAmountBank,
    source_amount_posterior, amount_quadrature, forecast_footprint_cdf)


def save(p,x): p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def advance_batch(a,means,target,wind):
    elapsed=target-a.reference.time_seconds
    if elapsed>1e-12:
        op,_,count=transport_step(a.reference.config,tuple(wind.velocity_yx),elapsed,
                                  a.reference.transport_id,a.reference.decay_rate)
        for _ in range(count): means=(op@means.T).T
    a.advance(target,wind)
    return means


def fit_histories(run,cfg,histories,states,metadata,out):
    model=run.model;hcount=len(histories);n=np.prod(model.grid_shape)
    assert model.initial_mean==0.,'This protocol uses zero initial mean'
    a=AffineSourceLikelihood(model,half_life_seconds=run.session.half_life_seconds,transport_id=run.session.transport_id)
    means=np.zeros((hcount,n));linear=np.zeros_like(means)
    wind=CausalWind();sensitivities=[];variances=[];checkpoints=[];count=0
    true_input=gaussian_source_input(model.grid_shape,cfg['source_yx'],source_sigma=model.source_sigma,
        total_rate=cfg['reference_total_per_second'],formulation_id=cfg['source_input_id'])
    mean_ids=[i for i,m in enumerate(metadata) if m['seed'] is None]
    amounts=np.array([m['multiplier'] for m in metadata])
    direct_ids=[i for i,m in enumerate(metadata) if m['scenario']=='nominal' and m['seed'] in [None,cfg['seeds'][1]]]
    src=np.repeat([cfg['source_yx'],[.5,.2],[.65,.35]],3,axis=0)
    multipliers=np.tile([.5,1.,1.5],3)
    banks={i:SourceAmountBank(model,src,multipliers,np.ones((1,9)),total_rate=cfg['reference_total_per_second'],
        input_id=cfg['source_input_id'],half_life_seconds=run.session.half_life_seconds,
        transport_id=run.session.transport_id) for i in direct_ids}
    verification={'max_direct_component_mean_error':0.,'max_direct_weight_error':0.,
                  'max_direct_covariance_error':0.,'max_mean_exact_source_error':0.,'direct_histories':direct_ids}
    for event_index,e in enumerate(histories[0]):
        op=e['operation'];at=e['at_seconds']
        if op=='ingest_readings':
            r=SensorReading.from_dict(e['reading']);means=advance_batch(a,means,r.acquired_at_seconds,wind)
            h=observation_weights(model.grid_shape,r.position_yx,r.footprint)
            cross=a.reference.covariance@h;variance=float(h@cross+(model.sensor_noise_std/model.gain)**2)
            s=h@a.response;ys=np.array([hs[event_index]['reading']['value'] for hs in histories])
            residual=(ys-model.offset)/model.gain-means@h
            linear+=residual[:,None]*s[None]/variance
            means+=residual[:,None]*(cross/variance)[None]
            sensitivities.append(s.copy());variances.append(variance);count+=1
            a.condition(0.,h,(model.sensor_noise_std/model.gain)**2)
            means=advance_batch(a,means,at,wind)
            for i,b in banks.items():
                b.assimilate(SensorReading.from_dict(histories[i][event_index]['reading']),wind);b.advance(at,wind)
        else:
            means=advance_batch(a,means,at,wind)
            for b in banks.values(): b.advance(at,wind)
            if op=='plan_next': wind=CausalWind.from_dict(e['wind'])
        selected=(op=='ingest_readings' and (count%cfg['checkpoint_reading_stride']==0 or count==708)) or op in ['finish_scan','end','forecast']
        if not selected: continue
        kind={'ingest_readings':'reading','finish_scan':'scan_end','end':'final','forecast':'forecast'}[op]
        checkpoint={'time':at,'kind':kind,'count':count,'event_index':event_index,
            'horizon':e.get('horizon_seconds',0.),'base':means.copy(),'linear':linear.copy(),
            'precision':a.precision.copy(),'response':a.response.copy(),'covariance':a.reference.covariance.copy(),
            'truth':states[:,event_index].copy()}
        exact=means+amounts[:,None]*(true_input@a.response.T)[None]
        verification['max_mean_exact_source_error']=max(verification['max_mean_exact_source_error'],
            float(np.max(np.abs(exact[mean_ids]-checkpoint['truth'][mean_ids]))))
        for i,b in banks.items():
            components=means[i][None]+b.emissions@a.response.T
            ll=b.emissions@linear[i]-.5*np.sum((b.emissions@a.precision)*b.emissions,axis=1)
            weights=np.exp(ll-logsumexp(ll))
            verification['max_direct_component_mean_error']=max(verification['max_direct_component_mean_error'],float(np.max(np.abs(components-b.means))))
            verification['max_direct_weight_error']=max(verification['max_direct_weight_error'],float(np.max(np.abs(weights-b.weights[0]))))
            verification['max_direct_covariance_error']=max(verification['max_direct_covariance_error'],float(np.max(np.abs(a.reference.covariance-b.reference.covariance))))
        checkpoints.append(checkpoint)
    assert count==708
    assert verification['max_mean_exact_source_error']<cfg['checks']['mean_control_exact_source_field_absolute']
    assert verification['max_direct_component_mean_error']<cfg['checks']['direct_mean_absolute']
    assert verification['max_direct_weight_error']<cfg['checks']['direct_weight_absolute']
    save(out/'affine-verification.json',verification)
    np.savez_compressed(out/'affine-checkpoints.npz',
        **{key:np.stack([c[key] for c in checkpoints]) for key in ['base','linear','precision','response','covariance','truth']},
        times=[c['time'] for c in checkpoints],counts=[c['count'] for c in checkpoints],
        sensitivities=sensitivities,innovation_variances=variances)
    return checkpoints,np.asarray(sensitivities),np.asarray(variances),verification


def evaluate(run,cfg,checkpoints,sensitivities,variances,metadata,level,root):
    order=level['source_order'];aorder=level['amount_order'];folder=root/f'grid-{order}-amount-{aorder}';folder.mkdir()
    sources=grid_positions((order,order));nloc=len(sources);hcount=len(metadata);n=np.prod(run.model.grid_shape)
    emission=gaussian_source_input(run.model.grid_shape,sources,source_sigma=run.model.source_sigma,
        total_rate=cfg['reference_total_per_second'],formulation_id=cfg['source_input_id'])
    nodes,weights=amount_quadrature(aorder,cfg['amount_prior']['minimum'],cfg['amount_prior']['maximum'])
    counts=np.array([c['count'] for c in checkpoints]);quad=np.zeros((len(checkpoints),nloc))
    for begin in range(0,nloc,4096):
        end=min(begin+4096,nloc)
        s=sensitivities@emission[begin:end].T;s*=s;s/=(2*variances[:,None]);np.cumsum(s,axis=0,out=s)
        quad[:,begin:end]=s[counts-1]
    distance=np.linalg.norm(sources-cfg['source_yx'],axis=1)
    validation_h=observation_weights(run.model.grid_shape,cfg['forecast_validation_position_yx'],run.capabilities.footprint)
    all_fields=[];all_std=[];rows=[];final_posterior=None
    for ci,c in enumerate(checkpoints):
        linear=c['linear']@emission.T;quadratic=quad[ci]
        W0=np.empty((hcount*3,nloc));W1=np.empty_like(W0);W2=np.empty_like(W0)
        need_variance=c['kind'] in ['scan_end','final','forecast']
        response_foot=emission@(c['response'].T@validation_h) if c['kind']=='forecast' else None
        within_foot=float(np.sqrt(validation_h@c['covariance']@validation_h))
        current=[]
        for hi,meta in enumerate(metadata):
            for mi,method in enumerate(cfg['methods']):
                fixed=None if mi==1 else (1. if mi==0 else meta['multiplier'])
                p=source_amount_posterior(linear[hi],quadratic,nodes,weights,fixed_amount=fixed,
                    gap=cfg['screening']['log_likelihood_gap'],omitted_limit=cfg['screening']['omitted_posterior_mass_upper_bound'],
                    amount_support=(cfg['amount_prior']['minimum'],cfg['amount_prior']['maximum']))
                j=hi*3+mi;W0[j]=p['w0'];W1[j]=p['w1'];W2[j]=p['w2']
                source_mean=p['w0']@sources
                row={'history':hi,'scenario':meta['scenario'],'seed':meta['seed'],'method':method,
                    'source_mean_yx':source_mean.tolist(),'source_error_fov':float(np.linalg.norm(source_mean-cfg['source_yx'])),
                    'source_mass_within_005':float(p['w0']@(distance<=.05)),
                    'amount_mean_multiplier':p['amount_mean'],'amount_std_multiplier':float(np.sqrt(p['amount_variance'])),
                    'amount_interval90':p['amount_interval90'],'omitted_mass_bound':p['omitted_mass_bound'],
                    'retained_locations':p['retained_locations']}
                if c['kind']=='forecast':
                    truth=float(validation_h@c['truth'][hi]);base=float(validation_h@c['base'][hi])
                    row['forecast_footprint_pit']=forecast_footprint_cdf(p,linear[hi],quadratic,nodes,weights,
                        response_foot,base,within_foot,truth,fixed_amount=fixed)
                current.append(row)
        effect=(W1@emission)@c['response'].T
        fields=c['base'][:,None]+effect.reshape(hcount,3,n)
        mse=np.mean((fields-c['truth'][:,None])**2,axis=2)
        std=np.full_like(fields,np.nan)
        if need_variance:
            second=np.zeros((hcount*3,n))
            for begin in range(0,nloc,4096):
                end=min(begin+4096,nloc);g=emission[begin:end]@c['response'].T
                second+=W2[:,begin:end]@(g*g)
            between=np.maximum(0.,second-effect**2)
            std=np.sqrt(np.diag(c['covariance'])[None]+between).reshape(hcount,3,n)
        for j,row in enumerate(current):
            row['field_mse']=float(mse.ravel()[j])
            row['mean_field_variance']=float(np.mean(std.reshape(hcount*3,n)[j]**2)) if need_variance else None
        rows.append({'time':c['time'],'kind':c['kind'],'horizon':c['horizon'],'count':c['count'],'methods':current})
        all_fields.append(fields);all_std.append(std)
        if c['kind']=='final':
            final_posterior={'w0':W0.reshape(hcount,3,nloc).copy(),'w1':W1.reshape(hcount,3,nloc).copy(),
                'w2':W2.reshape(hcount,3,nloc).copy(),'linear':linear.copy(),'quadratic':quadratic.copy()}
        if ci%10==0 or ci==len(checkpoints)-1:
            print(json.dumps({'grid':order,'amount_order':aorder,'checkpoint':ci+1,'of':len(checkpoints),'time':c['time']}),flush=True)
    save(folder/'traces.json',rows)
    np.savez_compressed(folder/'fields.npz',mean=all_fields,std=all_std,truth=[c['truth'] for c in checkpoints],
        time=[c['time'] for c in checkpoints],kind=[c['kind'] for c in checkpoints])
    np.savez_compressed(folder/'final-posterior.npz',sources=sources,amount_nodes=nodes,amount_prior_weights=weights,**final_posterior)
    reading=[r for r in rows if r['kind']=='reading'];last=next(r for r in rows if r['kind']=='final')
    history_rows=[]
    for hi,meta in enumerate(metadata):
        history_rows.append({**meta,'methods':[{'method':method,
            'primary_checkpoint_mse':float(np.mean([r['methods'][hi*3+mi]['field_mse'] for r in reading])),
            'final':last['methods'][hi*3+mi],
            'forecasts':[{**r['methods'][hi*3+mi],'horizon':r['horizon']} for r in rows if r['kind']=='forecast']}
            for mi,method in enumerate(cfg['methods'])]})
    result={'level':level,'history_results':history_rows,'reading_checkpoint_count':len(reading),
        'max_omitted_posterior_mass_bound':max(m['omitted_mass_bound'] for r in rows for m in r['methods'])}
    save(folder/'results.json',result)
    return result


def compare_levels(cfg,low,high):
    checks=[]
    for a,b in zip(low['history_results'],high['history_results']):
        for x,y in zip(a['methods'],b['methods']):
            shift=float(np.linalg.norm(np.array(x['final']['source_mean_yx'])-y['final']['source_mean_yx']))
            amount=abs(x['final']['amount_mean_multiplier']-y['final']['amount_mean_multiplier'])
            mse=abs(x['primary_checkpoint_mse']-y['primary_checkpoint_mse'])
            tolerance=max(cfg['refinement']['field_mse_relative']*y['primary_checkpoint_mse'],cfg['refinement']['field_mse_absolute'])
            checks.append({'scenario':a['scenario'],'seed':a['seed'],'method':x['method'],
                'source_shift_fov':shift,'amount_shift_multiplier':amount,'field_mse_difference':mse,'field_mse_tolerance':tolerance,
                'passed':shift<cfg['refinement']['source_mean_shift_fov'] and amount<cfg['refinement']['amount_mean_shift_multiplier'] and mse<tolerance})
    return {'levels':[low['level'],high['level']],'passed':all(c['passed'] for c in checks),'checks':checks}


def summarize(cfg,result):
    scenarios=[]
    for scenario in cfg['scenarios']:
        histories=[r for r in result['history_results'] if r['scenario']==scenario['label']]
        stochastic=[r for r in histories if r['seed'] is not None];mean=next(r for r in histories if r['seed'] is None)
        methods=[]
        for mi,name in enumerate(cfg['methods']):
            scores=np.array([h['methods'][mi]['primary_checkpoint_mse'] for h in stochastic])
            base=np.array([h['methods'][0]['primary_checkpoint_mse'] for h in stochastic]);delta=scores-base
            methods.append({'method':name,'mean_control':mean['methods'][mi],
                'mean_checkpoint_mse':float(scores.mean()),'mean_final_mse':float(np.mean([h['methods'][mi]['final']['field_mse'] for h in stochastic])),
                'mean_final_source_error_fov':float(np.mean([h['methods'][mi]['final']['source_error_fov'] for h in stochastic])),
                'mean_amount_multiplier':float(np.mean([h['methods'][mi]['final']['amount_mean_multiplier'] for h in stochastic])),
                'paired_mse_change_vs_location_only':float(delta.mean()),'paired_change_standard_error':float(delta.std(ddof=1)/np.sqrt(len(delta))),
                'improved_seed_count':int(np.sum(delta<0)),
                'forecasts':[{'horizon':f,'mean_field_mse':float(np.mean([h['methods'][mi]['forecasts'][fi]['field_mse'] for h in stochastic])),
                    'footprint_pit_inside95_count':sum(.025<=h['methods'][mi]['forecasts'][fi]['forecast_footprint_pit']<=.975 for h in stochastic)}
                    for fi,f in enumerate(cfg['forecast_horizons_seconds'])]})
        scenarios.append({**scenario,'methods':methods})
    return {'level':result['level'],'scenarios':scenarios,'seed_count':10,'scope':cfg['scope'],
        'metric':cfg['metrics'],'max_omitted_posterior_mass_bound':result['max_omitted_posterior_mass_bound']}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text());out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.time()
    record_path=Path(cfg['record']);assert digest(record_path)==cfg['record_sha256']
    record=json.loads(record_path.read_text());run=decode_config(record['configuration'])
    save(out/'config.json',cfg);save(out/'baseline-configuration.json',record['configuration'])
    shutil.copy2(record_path,out/'original-schedule-and-evidence.json')
    save(out/'provenance.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],text=True),'command':sys.argv,'python':sys.version})
    imp=out/'implementation';imp.mkdir()
    for fn in ['scripts/compare-source-amount.py','src/plume_focus/source_amount_control.py','src/plume_focus/source_input.py',
               'src/plume_focus/model_consistent_control.py','src/plume_focus/source_likelihood_quadrature.py',
               'src/plume_focus/source_prior_belief.py','docs/specs/joint-source-amount-v1.md']:
        shutil.copy2(fn,imp/Path(fn).name)
    histories=[];states=[];metadata=[];noise_fields={};noise_readings={};rng={};max_field_noise=0.;max_reading_noise=0.
    for scenario in cfg['scenarios']:
        deterministic=None;det_readings=None
        for seed in cfg['seeds']:
            label='mean' if seed is None else f'seed-{seed}';folder=out/scenario['label']/label;folder.mkdir(parents=True)
            events,truth,signatures=generate_history(record,run,cfg,scenario['amount_multiplier'],seed)
            save(folder/'evidence.json',{'events':events});np.savez_compressed(folder/'truth.npz',states=truth)
            values=np.array([e['reading']['value'] for e in events if e['operation']=='ingest_readings'])
            assert len(values)==708
            if seed is None: deterministic=truth.copy();det_readings=values.copy()
            else:
                field_noise=truth-deterministic;reading_noise=values-det_readings
                if seed not in rng: rng[seed]=signatures;noise_fields[seed]=field_noise;noise_readings[seed]=reading_noise
                else:
                    assert rng[seed]==signatures
                    max_field_noise=max(max_field_noise,float(np.max(np.abs(field_noise-noise_fields[seed]))))
                    max_reading_noise=max(max_reading_noise,float(np.max(np.abs(reading_noise-noise_readings[seed]))))
            histories.append(events);states.append(truth);metadata.append({'scenario':scenario['label'],'seed':seed,'multiplier':scenario['amount_multiplier']})
            print(json.dumps({'generated':scenario['label'],'seed':seed,'readings':len(values)}),flush=True)
    assert max(max_field_noise,max_reading_noise)<cfg['checks']['paired_noise_absolute']
    save(out/'paired-noise-checks.json',{'max_field_noise_difference':max_field_noise,'max_reading_noise_difference':max_reading_noise,'rng_signatures_identical':True})
    checkpoints,sensitivities,variances,verification=fit_histories(run,cfg,histories,np.asarray(states),metadata,out)
    results=[];refinements=[]
    for level in cfg['quadrature_levels']:
        result=evaluate(run,cfg,checkpoints,sensitivities,variances,metadata,level,out);results.append(result)
        if len(results)>1:
            refinements.append(compare_levels(cfg,results[-2],result));save(out/'refinement.json',refinements)
    summary=summarize(cfg,results[-1]);summary['final_refinement_passed']=refinements[-1]['passed'];save(out/'summary.json',summary)
    assert digest(record_path)==cfg['record_sha256']
    save(out/'verification.json',{'paired_noise_passed':True,'affine':verification,'original_record_unchanged':True,
        'final_refinement_passed':refinements[-1]['passed'],'wall_seconds':time.time()-started})
    save(out/'artifact-manifest.json',{str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact-manifest.json'})
    print(json.dumps({'completed':True,'wall_seconds':time.time()-started,'final_refinement_passed':refinements[-1]['passed']}),flush=True)


if __name__=='__main__': main()
