"""Fixed-clock heldout reconstruction checks and same-policy closed-loop study."""
from pathlib import Path
import argparse,copy,hashlib,json,shutil,subprocess,time
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from plume_focus.full_fov_execution import decode_config
from plume_focus.source_amount_validation import ClockDynamics,ClockAffineFilter,OnlineAmountBelief,posterior_inputs
from plume_focus.source_input import gaussian_source_input
from plume_focus.signal_belief import grid_positions,observation_weights
from plume_focus.sensing_contracts import SensorReading,ExecutionReceipt
from plume_focus.sensing_messages_v2 import ScanRequestV1
from plume_focus.evidence_gated_loop import EvidenceGatedScanner


def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def clone_filter(f):
    c=copy.copy(f)
    for k in ['m','P','B','L','K']:setattr(c,k,getattr(f,k).copy())
    return c
def keyed_noise(seed,tick,position,std):
    if seed is None:return 0.
    entropy=[seed,tick,*(int(round(x*1e6)) for x in position)]
    return np.random.default_rng(np.random.SeedSequence(entropy)).normal(0,std)


class Environment:
    """Evaluator/scanner-owned clock trajectory, never passed to belief/policy."""
    def __init__(self,d,cfg,out):
        self.d=d;self.cfg=cfg;self.out=out
        self.scenes=[{'name':'development','source_yx':cfg['development']['source_yx']}]+cfg['heldout']
        self.seeds=sorted(set([s for s in cfg['development']['seeds'] if s is not None]+cfg['heldout_seeds']))
        self.seed_ids={s:i for i,s in enumerate(self.seeds)}
        self.E=gaussian_source_input(d.model.grid_shape,[s['source_yx'] for s in self.scenes],
            source_sigma=d.model.source_sigma,total_rate=cfg['reference_total_per_second'],formulation_id=cfg['source_input_id'])
        nt=d.tick_of(cfg['closed_duration_seconds']+max(cfg['forecast_horizons_seconds']))+1;n=len(d.P0)
        out.mkdir(exist_ok=True)
        if all((out/name).exists() for name in ['deterministic.npy','noise.npy','environment.json']):
            info=json.loads((out/'environment.json').read_text())
            assert info['scenes']==self.scenes and info['seeds']==self.seeds and info['tick']==d.tick
            for name in ['deterministic','noise']:assert sha(out/f'{name}.npy')==info['hashes'][name]
            self.det=np.load(out/'deterministic.npy',mmap_mode='r');self.noise=np.load(out/'noise.npy',mmap_mode='r')
            self.digest=info['hashes'];return
        self.det=np.lib.format.open_memmap(out/'deterministic.npy',mode='w+',dtype='float64',shape=(nt,len(self.scenes),n))
        self.noise=np.lib.format.open_memmap(out/'noise.npy',mode='w+',dtype='float64',shape=(nt,len(self.seeds),n))
        self.det[0]=d.model.initial_mean
        process=[np.random.default_rng(np.random.SeedSequence([s,2])) for s in self.seeds]
        self.noise[0]=np.stack([d.initial_factor@np.random.default_rng(np.random.SeedSequence([s,1])).standard_normal(n) for s in self.seeds])
        for k in range(1,nt):
            self.det[k]=d.means(self.det[k-1],self.E)
            z=np.stack([r.standard_normal(n) for r in process])
            self.noise[k]=(d.A@self.noise[k-1].T).T+z@d.noise_factor.T
        self.det.flush();self.noise.flush()
        self.digest={'deterministic':sha(out/'deterministic.npy'),'noise':sha(out/'noise.npy')}
        save(out/'environment.json',{'scenes':self.scenes,'seeds':self.seeds,'tick':d.tick,'hashes':self.digest})

    def truth(self,tick,metadata):
        means=np.stack([self.det[tick,m['scene_index']]*m['amount'] for m in metadata])
        for i,m in enumerate(metadata):
            if m['seed'] is not None:means[i]+=self.noise[tick,self.seed_ids[m['seed']]]
        return means


def metadata(cfg):
    result=[]
    for si,scene in enumerate([{'name':'development','source_yx':cfg['development']['source_yx']}]+cfg['heldout']):
        seeds=cfg['development']['seeds'] if si==0 else cfg['heldout_seeds']
        for amount in cfg['amounts']:
            for seed in seeds:result.append({'scene_index':si,'scene':scene['name'],'source_yx':scene['source_yx'],
                                           'split':'development' if si==0 else 'heldout','amount':amount,'seed':seed})
    return result


def uncertainties(f,posts,metadata,H,truth,sources,t):
    rows=[]
    for hi,meta in enumerate(metadata):
        for mi,p in enumerate(posts[hi]):
            v=f.validation(hi,p,H,H@truth[hi]);loc=p['w']@sources[p['ids']]
            meanamount=float(p['w']@p['a1'])
            rows.append({'history':hi,'method':['fixed','joint','oracle'][mi], 'time':t,
                'source_mean_yx':loc.tolist(),'source_error_fov':float(np.linalg.norm(loc-meta['source_yx'])),
                'amount_mean':meanamount,'amount_absolute_error':abs(meanamount-meta['amount']),
                'amount_std':float(np.sqrt(max(0.,p['w']@p['a2']-meanamount**2))),
                'pit':v['pit'].tolist(),'variance':v['variance'].tolist(),'squared_error':v['squared_error'].tolist(),
                'inside95':((v['pit']>=.025)&(v['pit']<=.975)).tolist(),
                'error_variance_ratio':(v['squared_error']/v['variance']).tolist()})
    return rows


def fixed_run(env,run,cfg,record,level,out,*,histories=None,refinement_only=False,evidence=None):
    out.mkdir();started=time.time();d=env.d;metas=metadata(cfg) if histories is None else histories;nh=len(metas)
    sources=grid_positions((level[0],level[0]))
    def source_inputs(points):
        return gaussian_source_input(d.model.grid_shape,points,source_sigma=d.model.source_sigma,
            total_rate=cfg['reference_total_per_second'],formulation_id=cfg['source_input_id'])
    if len(sources)>100000:
        E=np.empty((len(sources),len(d.P0)))
        for start in range(0,len(sources),4096):E[start:start+4096]=source_inputs(sources[start:start+4096])
    else:E=source_inputs(sources)
    f=ClockAffineFilter(d,E,nh)
    H=np.stack([observation_weights(d.model.grid_shape,p,run.capabilities.footprint) for p in cfg['validation_footprints_yx']])
    means=np.zeros((nh,3,len(d.P0)));inputs=np.zeros_like(means);posts=None;maxbound=0.
    pool=ThreadPoolExecutor(max_workers=4)
    def one_posterior(hi):
        return [f.posterior(hi,fixed,level[1]) for fixed in [1.,None,metas[hi]['amount']]]
    def refresh():
        nonlocal means,inputs,posts,maxbound
        posts=list(pool.map(one_posterior,range(nh)))
        maxbound=max(maxbound,max(p['omitted_bound'] for ps in posts for p in ps))
        inputs=posterior_inputs(E,[p for ps in posts for p in ps]).reshape(nh,3,-1)
        means=f.m[:,None]+inputs@f.B.T
    refresh();current=0;end=d.tick_of(cfg['fixed_duration_seconds'])
    errors=np.zeros((end,nh,3));maps={};validation=[];max_exact=0.
    special={d.tick_of(20.):20.,end:40.}
    def snapshot(tick):
        nonlocal validation
        fc=clone_filter(f);fc.advance(tick)
        if not refinement_only:
            validation+=uncertainties(fc,posts,metas,H,env.truth(tick,metas),sources,tick*d.tick)
        elif tick==end:
            for hi,meta in enumerate(metas):
                for mi,p in enumerate(posts[hi]):
                    loc=p['w']@sources[p['ids']]
                    validation.append({'history':hi,'method':['fixed','joint','oracle'][mi],'time':40.,
                        'source_mean_yx':loc.tolist(),'amount_mean':float(p['w']@p['a1'])})
        maps[str(tick*d.tick)]=means.copy()
    def score_until(target):
        nonlocal current,means
        while current<target:
            if current in special:snapshot(current)
            errors[current]=np.mean((means-env.truth(current,metas)[:,None])**2,axis=2)
            means=d.means(means.reshape(nh*3,-1),inputs.reshape(nh*3,-1)).reshape(nh,3,-1)
            current+=1
    readings=[]
    for e in record['events']:
        if e['operation']!='ingest_readings':continue
        r=SensorReading.from_dict(e['readings'][0]);acq=d.tick_of(r.acquired_at_seconds);at=d.tick_of(r.available_at_seconds)
        score_until(at);f.advance(acq)
        h=observation_weights(d.model.grid_shape,r.position_yx,r.footprint)
        if evidence is None:
            y=env.truth(acq,metas)@h*run.model.gain+run.model.offset
            y+=np.array([keyed_noise(m['seed'],acq,r.position_yx,run.model.sensor_noise_std) for m in metas])
        else:y=evidence[len(readings)].copy()
        f.condition(y,h);f.advance(at);refresh();readings.append(y)
        for hi,m in enumerate(metas):
            if m['seed'] is None:
                exact=f.m[hi]+f.B@(env.E[m['scene_index']]*m['amount'])
                max_exact=max(max_exact,float(np.max(np.abs(exact-env.truth(at,[m])[0]))))
        if len(readings)%100==0:print(json.dumps({'fixed_grid':list(d.model.grid_shape),'level':list(level),'readings':len(readings),'elapsed':time.time()-started}),flush=True)
    score_until(end);snapshot(end);f.advance(end)
    forecasts=[]
    for horizon in cfg['forecast_horizons_seconds']:
        target=d.tick_of(cfg['fixed_duration_seconds']+horizon)
        for _ in range(target-current):means=d.means(means.reshape(nh*3,-1),inputs.reshape(nh*3,-1)).reshape(nh,3,-1)
        current=target;f.advance(target);truth=env.truth(target,metas)
        forecasts.append(np.mean((means-truth[:,None])**2,axis=2))
        maps[str(target*d.tick)]=means.copy()
        if not refinement_only:validation+=uncertainties(f,posts,metas,H,truth,sources,target*d.tick)
    assert max_exact<1e-10
    result={'metadata':metas,'source_order':level[0],'amount_order':level[1],
        'JT':errors.mean(axis=0).tolist(),'JT_subsample_025':errors[::2].mean(axis=0).tolist(),
        'JT_subsample_050':errors[::4].mean(axis=0).tolist(),
        'post_startup_mse':errors[d.tick_of(33.575):].mean(axis=0).tolist(),
        'forecasts':[v.tolist() for v in forecasts],'uncertainty':validation,
        'max_omitted_bound':maxbound,'max_mean_exact_truth_difference':max_exact,'wall_seconds':time.time()-started}
    save(out/'results.json',result)
    np.savez_compressed(out/'fields.npz',**maps,truth40=env.truth(end,metas))
    np.savez_compressed(out/'evidence-and-errors.npz',readings=readings,errors=errors)
    print(json.dumps({'fixed_complete':str(out),'seconds':time.time()-started}),flush=True)
    pool.shutdown()
    return result


def refine(cfg,a,b):
    af=[r for r in a['uncertainty'] if r['time']==40.];bf=[r for r in b['uncertainty'] if r['time']==40.]
    checks=[]
    for hi,meta in enumerate(a['metadata']):
        for mi in range(3):
            x,y=af[hi*3+mi],bf[hi*3+mi]
            location=float(np.linalg.norm(np.array(x['source_mean_yx'])-y['source_mean_yx']))
            amount=abs(x['amount_mean']-y['amount_mean']);differences=[]
            for va,vb in [(a['JT'],b['JT'])]+list(zip(a['forecasts'],b['forecasts'])):
                va,vb=va[hi][mi],vb[hi][mi]
                differences.append(abs(va-vb)<max(cfg['refinement']['mse_relative']*vb,cfg['refinement']['mse_absolute']))
            checks.append({'history':hi,'method':mi,'source_shift':location,'amount_shift':amount,
                'passed':location<cfg['refinement']['source_shift_fov'] and amount<cfg['refinement']['amount_shift'] and all(differences)})
    return {'passed':all(c['passed'] for c in checks),'checks':checks}


def summarize_fixed(cfg,result):
    rows=[];U=result['uncertainty']
    for split in ['development','heldout']:
        for amount in cfg['amounts']:
            ids=[i for i,m in enumerate(result['metadata']) if m['split']==split and m['amount']==amount and m['seed'] is not None]
            for scene in ['all']+([s['name'] for s in cfg['heldout']] if split=='heldout' else []):
                ii=[i for i in ids if scene=='all' or result['metadata'][i]['scene']==scene]
                vals=np.asarray(result['JT'])[ii];fs=np.asarray(result['forecasts'])[:,ii].mean(axis=1)
                u=[r for r in U if r['history'] in ii]
                row={'split':split,'amount':amount,'scene':scene,'histories':len(ii),'JT':vals.mean(axis=0).tolist(),
                    'candidate_change_percent':float(100*(vals[:,1].mean()/vals[:,0].mean()-1)),
                    'improved_histories':int(np.sum(vals[:,1]<vals[:,0])),'forecast_mse':fs.tolist(),
                    'uncertainty':[{'method':mi,'time':t,'inside95':float(np.mean([r['inside95'] for r in u if r['method']==name and r['time']==t])),
                         'mean_error_variance_ratio':float(np.mean([r['error_variance_ratio'] for r in u if r['method']==name and r['time']==t]))}
                         for mi,name in enumerate(['fixed','joint','oracle']) for t in cfg['uncertainty_times_seconds']]}
                rows.append(row)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--reuse-environment',type=Path)
    p.add_argument('--resume-from',type=Path);args=p.parse_args();cfg=json.loads(args.config.read_text())
    if args.smoke:
        cfg['grid_shapes']=[[4,6]];cfg['source_orders']=[13,17];cfg['amount_orders']=[16,32]
        cfg['development']['seeds']=[None,20261101];cfg['heldout']=cfg['heldout'][:1];cfg['heldout_seeds']=[20261111]
    out=args.output;out.mkdir(parents=True,exist_ok=False);save(out/'config.json',cfg)
    if args.resume_from:assert json.loads((args.resume_from/'config.json').read_text())==cfg
    save(out/'provenance.json',{'head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'smoke_only':args.smoke,'status':subprocess.check_output(['git','status','--short'],text=True)})
    for src in [args.config,Path(__file__),Path('src/plume_focus/source_amount_validation.py'),Path('docs/specs/source-amount-validation-v1.md')]:shutil.copy2(src,out/src.name)
    recordpath=Path(cfg['record']);assert sha(recordpath)==cfg['record_sha256'];record=json.loads(recordpath.read_text())
    run=decode_config(json.loads(Path(cfg['baseline_config']).read_text()));results=[]
    reused=[]
    for shape in cfg['grid_shapes']:
        model=replace(run.model,grid_shape=tuple(shape));grid=out/f'grid-{shape[0]}x{shape[1]}';grid.mkdir()
        d=ClockDynamics(model,run.wind.velocity_yx,tick=cfg['clock_tick_seconds'],half_life=run.session.half_life_seconds)
        reuse=args.resume_from or args.reuse_environment
        old=reuse/f'grid-{shape[0]}x{shape[1]}'/'environment' if reuse else None
        if old is not None and (old/'environment.json').exists():shutil.copytree(old,grid/'environment')
        env=Environment(d,cfg,grid/'environment');rr=[]
        for order,amountorder in zip(cfg['source_orders'],cfg['amount_orders']):
            folder=grid/f'source-{order}-amount-{amountorder}'
            previous=args.resume_from/grid.name/folder.name if args.resume_from else None
            if previous is not None and (previous/'results.json').exists():
                shutil.copytree(previous,folder);r=json.loads((folder/'results.json').read_text())
                assert r['metadata']==metadata(cfg) and r['source_order']==order and r['amount_order']==amountorder
                rr.append(r);reused.append({'source':str(previous),'results_sha256':sha(previous/'results.json')})
                save(out/'reused-levels.json',reused)
            else:rr.append(fixed_run(env,run,cfg,record,(order,amountorder),folder))
        check=refine(cfg,*rr[-2:]);save(grid/'refinement.json',check)
        summary=summarize_fixed(cfg,rr[-1]);save(grid/'summary.json',summary)
        results.append({'grid':shape,'refinement_passed':check['passed'],'summary':summary})
        del env
    save(out/'fixed-summary.json',results)
    save(out/'fixed-verification.json',{'original_record_unchanged':sha(recordpath)==cfg['record_sha256'],
        'all_final_refinements_passed':all(r['refinement_passed'] for r in results),'smoke_only':args.smoke})
    print(json.dumps({'fixed_study_complete':True,'refinement_passed':all(r['refinement_passed'] for r in results),'output':str(out)}),flush=True)


if __name__=='__main__':main()
