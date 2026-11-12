"""Full-history global-source refinement of a remaining known-amount control."""
from pathlib import Path
from dataclasses import replace
import argparse,hashlib,json,shutil,tempfile,time
import numpy as np
from plume_focus.full_fov_execution import decode_config
from plume_focus.source_amount_validation import ClockDynamics,ClockAffineFilter
from plume_focus.source_input import gaussian_source_input
from plume_focus.signal_belief import grid_positions,observation_weights
from plume_focus.sensing_contracts import SensorReading


def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    cfg=json.loads((args.study/'config.json').read_text());parent=json.loads((args.parent/'config.json').read_text())
    checks=json.loads((args.parent/'refinement.json').read_text());bad=[c for c in checks['checks'] if not c['passed']]
    assert len(bad)==1 and bad[0]['method']==2
    hi=bad[0]['history'];original_hi=parent['selected_history_indices'][hi]
    previous=json.loads((args.parent/'fine/results.json').read_text());meta=previous['metadata'][hi]
    values=np.load(args.parent/'fine/evidence-and-errors.npz')['readings'][:,hi]
    gridkey=parent['grid'];shape=tuple(map(int,gridkey.split('x')));grid=args.study/f'grid-{gridkey}'
    run=decode_config(json.loads(Path(cfg['baseline_config']).read_text()))
    d=ClockDynamics(replace(run.model,grid_shape=shape),run.wind.velocity_yx,tick=cfg['clock_tick_seconds'],half_life=run.session.half_life_seconds)
    det=np.load(grid/'environment/deterministic.npy',mmap_mode='r');noise=np.load(grid/'environment/noise.npy',mmap_mode='r')
    info=json.loads((grid/'environment/environment.json').read_text())
    def truth(k):return det[k,meta['scene_index']]*meta['amount']+noise[k,info['seeds'].index(meta['seed'])]
    out=args.output;out.mkdir(parents=True,exist_ok=False);started=time.time();order=1025
    save(out/'config.json',{'original_study':str(args.study.resolve()),'grid':gridkey,'source_order':order,
        'selected_history_indices':[original_hi],'method_indices':[2],'parent_refinement':str(args.parent.resolve()),
        'parent_results_sha256':sha(args.parent/'fine/results.json'),'parent_evidence_sha256':sha(args.parent/'fine/evidence-and-errors.npz'),
        'original_configuration_sha256':sha(args.study/'config.json'),
        'purpose':'Full 40 s J/T and forecasts; oracle amount only, global uniform source prior, unchanged criteria.'})
    shutil.copy2(__file__,out/Path(__file__).name)
    sources=grid_positions((order,order));n=len(d.P0)
    with tempfile.TemporaryDirectory(prefix='plume-oracle-source-',dir='/private/tmp') as scratch:
        E=np.lib.format.open_memmap(Path(scratch)/'source-cache.npy',mode='w+',dtype='float64',shape=(len(sources),n))
        total_error=0.
        for start in range(0,len(sources),4096):
            v=gaussian_source_input(shape,sources[start:start+4096],source_sigma=d.model.source_sigma,
                total_rate=cfg['reference_total_per_second'],formulation_id=cfg['source_input_id'])
            total_error=max(total_error,float(np.max(np.abs(v.mean(axis=1)-cfg['reference_total_per_second']))))
            E[start:start+len(v)]=v
        E.flush();f=ClockAffineFilter(d,E);amount=meta['amount']
        def weighted_input(p):
            weights=p['w']*amount
            if len(p['ids'])==len(E):return weights@E
            if len(p['ids'])<10000:return weights@E[p['ids']]
            result=np.zeros(n)
            for start in range(0,len(weights),8192):result+=weights[start:start+8192]@E[p['ids'][start:start+8192]]
            return result
        posterior=f.posterior(0,amount,2);emission=weighted_input(posterior);mean=f.m[0]+f.B@emission
        current=0;errors=[];bound=0.;ri=0
        def before(target):
            nonlocal current,mean
            while current<target:
                errors.append(float(np.mean((mean-truth(current))**2)))
                mean=d.means(mean[None],emission)[0];current+=1
        record=json.loads(Path(cfg['record']).read_text())
        for event in record['events']:
            if event['operation']!='ingest_readings':continue
            r=SensorReading.from_dict(event['readings'][0]);at=d.tick_of(r.available_at_seconds)
            before(at);f.advance(d.tick_of(r.acquired_at_seconds))
            f.condition([values[ri]],observation_weights(shape,r.position_yx,r.footprint));f.advance(at)
            posterior=f.posterior(0,amount,2);emission=weighted_input(posterior);mean=f.m[0]+f.B@emission
            bound=max(bound,posterior['omitted_bound']);ri+=1
            if ri%100==0:print(json.dumps({'oracle_source_order':order,'readings':ri,'elapsed':time.time()-started}),flush=True)
        before(d.tick_of(40.));final=mean.copy();mse=[];maps=[]
        source_mean=posterior['w']@sources[posterior['ids']]
        for horizon in cfg['forecast_horizons_seconds']:
            target=d.tick_of(40.+horizon)
            mean=d.means(mean[None],emission,target-current)[0];current=target
            mse.append(float(np.mean((mean-truth(target))**2)));maps.append(mean.copy())
        result={'history':original_hi,'method':2,'metadata':meta,'JT':float(np.mean(errors)),
                'forecast_mse':mse,'source_mean_yx':source_mean.tolist(),'amount_mean':amount,
                'source_order':order,'max_omitted_bound':bound,'max_input_total_error':total_error,
                'wall_seconds':time.time()-started}
        oldsource=next(r for r in previous['uncertainty'] if r['history']==hi and r['time']==40. and r['method']=='oracle')
        metrics=[]
        for name,a,b in [('JT',previous['JT'][hi][2],result['JT'])]+[(f'forecast+{h:g}',previous['forecasts'][i][hi][2],mse[i]) for i,h in enumerate(cfg['forecast_horizons_seconds'])]:
            tolerance=max(cfg['refinement']['mse_relative']*b,cfg['refinement']['mse_absolute'])
            metrics.append({'name':name,'difference':abs(a-b),'tolerance':tolerance,'passed':abs(a-b)<tolerance})
        shift=float(np.linalg.norm(source_mean-oldsource['source_mean_yx']))
        passed=all(v['passed'] for v in metrics) and shift<cfg['refinement']['source_shift_fov']
        check={'history':original_hi,'method':2,'source_shift':shift,'amount_shift':0.,'metrics':metrics,'passed':passed}
        save(out/'results.json',result);save(out/'refinement.json',{'passed':passed,'checks':[check]})
        save(out/'verification.json',{'grid':gridkey,'selected_histories':[original_hi],'method_indices':[2],
             'parent_refinement':str(args.parent.resolve()),'final_refinement_passed':passed,
             'evidence_bit_identical':True,'original_study':str(args.study.resolve()),
             'original_configuration_sha256':sha(args.study/'config.json')})
        np.savez_compressed(out/'arrays.npz',errors=errors,evidence=values,final_mean=final,forecasts=maps,
                            source_ids=posterior['ids'],source_weights=posterior['w'])
        del f,E
    print(json.dumps({'oracle1025_complete':True,'passed':passed,'metrics':metrics,'wall_seconds':time.time()-started}),flush=True)


if __name__=='__main__':main()
