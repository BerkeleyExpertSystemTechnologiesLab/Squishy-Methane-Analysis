"""Run/replay the unchanged scanner policy with two source-mixture beliefs."""
from pathlib import Path
from dataclasses import replace
import argparse,copy,hashlib,json,shutil,time
import numpy as np
from plume_focus.full_fov_execution import decode_config
from plume_focus.evidence_gated_loop import EvidenceGatedScanner
from plume_focus.source_amount_validation import ClockDynamics,OnlineAmountBelief
from plume_focus.source_input import gaussian_source_input
from plume_focus.signal_belief import grid_positions,observation_weights
from plume_focus.sensing_contracts import SensorReading,ExecutionReceipt
from plume_focus.sensing_messages_v2 import ScanRequestV1


def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


class SavedEnvironment:
    def __init__(self,root,d):
        self.d=d;self.info=json.loads((root/'environment.json').read_text())
        for key in ['deterministic','noise']:assert sha(root/f'{key}.npy')==self.info['hashes'][key]
        self.det=np.load(root/'deterministic.npy',mmap_mode='r');self.noise=np.load(root/'noise.npy',mmap_mode='r')
    def state(self,tick,scene,amount,seed):
        si=next(i for i,s in enumerate(self.info['scenes']) if s['name']==scene)
        return self.det[tick,si]*amount+self.noise[tick,self.info['seeds'].index(seed)]
    def reading(self,sample,scan_id,index,scene,amount,seed):
        d=self.d;tick=d.tick_of(sample.acquired_at_seconds)
        h=observation_weights(d.model.grid_shape,sample.position_yx,sample.footprint)
        entropy=[seed,tick,*(int(round(x*1e6)) for x in sample.position_yx)]
        noise=np.random.default_rng(np.random.SeedSequence(entropy)).normal(0,d.model.sensor_noise_std)
        value=d.model.gain*float(h@self.state(tick,scene,amount,seed))+d.model.offset+noise
        return SensorReading(scan_id,f'{scan_id}:{index}',sample.position_yx,sample.acquired_at_seconds,
            sample.acquired_at_seconds+sample.integration_seconds/2,value,sample.footprint,sample.integration_seconds)


class LossObserver:
    def __init__(self,env,case,cfg,belief,wind):
        self.env=env;self.case=case;self.cfg=cfg;self.d=env.d;self.wind=wind;self.tick=0
        self.mean,self.emission=belief.summary();self.errors=[];self.frames=[];self.frame_times=[]
    def truth(self,k):return self.env.state(k,self.case['scene'],self.case['amount'],self.case['seed'])
    def before(self,t):
        end=self.d.tick_of(t)
        while self.tick<end:
            if self.tick%40==0:self.frames.append(self.mean.copy());self.frame_times.append(self.tick*self.d.tick)
            self.errors.append(float(np.mean((self.mean-self.truth(self.tick))**2)))
            self.mean=self.d.means(self.mean[None],self.emission)[0];self.tick+=1
    def refresh(self,belief,t):
        b=belief.clone();b.advance(t,self.wind,None);self.mean,self.emission=b.summary()


def run_case(env,run,cfg,case,method,sources,E,replay=None):
    started=time.time();session=replace(run.session,source_position=None)
    controller=EvidenceGatedScanner(session)
    belief=OnlineAmountBelief(env.d,sources,E,fixed=1. if method=='fixed' else None,
        amount_order=cfg['amount_orders'][-1],prune=cfg['closed_loop']['distribution_component_omission_limit'])
    controller._belief=belief
    observer=LossObserver(env,case,cfg,belief,run.wind)
    now=0.;pose=session.home_yx;events=[];scans=[];ri=0;reads=0;duration=cfg['closed_duration_seconds']
    while now<duration-1e-9:
        observer.before(now)
        plan=controller.plan_next(now,pose,ScanRequestV1(duration-now),wind=run.wind)
        pe={'op':'plan','at':now,'pose':list(pose),'remaining':duration-now,'plan':plan.to_dict()}
        if replay is not None:assert json.dumps(pe,sort_keys=True)==json.dumps(replay[ri],sort_keys=True);ri+=1
        events.append(pe)
        if not plan.samples:break
        for i,sample in enumerate(plan.samples):
            r=(SensorReading.from_dict(replay[ri]['reading']) if replay is not None
               else env.reading(sample,plan.scan_id,i,case['scene'],case['amount'],case['seed']))
            assert r.acquired_at_seconds==sample.acquired_at_seconds and tuple(r.position_yx)==tuple(sample.position_yx)
            observer.before(r.available_at_seconds)
            controller.ingest_readings([r],available_at_seconds=r.available_at_seconds)
            observer.refresh(controller._belief,r.available_at_seconds)
            events.append({'op':'reading','reading':r.to_dict()});reads+=1
            if replay is not None:ri+=1
        receipt=ExecutionReceipt(plan.scan_id,plan.completed_at_seconds,plan.completed_at_seconds,plan.final_position_yx)
        observer.before(receipt.received_at_seconds);controller.finish_scan(receipt)
        observer.refresh(controller._belief,receipt.received_at_seconds)
        fe={'op':'finish','receipt':receipt.to_dict()}
        if replay is not None:assert json.dumps(fe,sort_keys=True)==json.dumps(replay[ri],sort_keys=True);ri+=1
        events.append(fe);scans.append({'mode':plan.mode,'duration':plan.duration_seconds,'readings':len(plan.samples),
            'requested_fraction':len(plan.requested_cells)/np.prod(plan.lattice_shape),
            'swept_fraction':len(plan.swept_cells)/np.prod(plan.lattice_shape),
            'movement_seconds':sum(s.completed_at_seconds-s.started_at_seconds for s in plan.segments if s.kind=='move'),
            'gate_reason':plan.reason,'end':plan.completed_at_seconds})
        now=receipt.received_at_seconds;pose=receipt.final_position_yx
    observer.before(duration);observer.refresh(controller._belief,duration)
    end_mean=observer.mean.copy();forecasts=[]
    for horizon in cfg['forecast_horizons_seconds']:
        t=duration+horizon;b=controller._belief.clone();b.advance(t,run.wind,None)
        mean,_=b.summary();forecasts.append(float(np.mean((mean-observer.truth(env.d.tick_of(t)))**2)))
    if replay is not None:assert ri==len(replay)
    result={'case':case,'method':method,'JT':float(np.mean(observer.errors)),
        'post_startup_mse':float(np.mean(observer.errors[env.d.tick_of(33.575):])),
        'forecast_mse':forecasts,'readings':reads,'scans':scans,'spent_seconds':sum(s['duration'] for s in scans),
        'completed_full_scans':controller.completed_full_scans,'startup_complete':controller.startup_complete,
        'wall_seconds':time.time()-started}
    return result,events,np.asarray(observer.errors),np.asarray(observer.frames),np.asarray(observer.frame_times),end_mean


def replay_belief_only(env,run,cfg,case,method,sources,E,events):
    belief=OnlineAmountBelief(env.d,sources,E,fixed=1. if method=='fixed' else None,
        amount_order=cfg['amount_orders'][-1],prune=cfg['closed_loop']['distribution_component_omission_limit'])
    loss=LossObserver(env,case,cfg,belief,run.wind)
    for event in events:
        if event['op']=='reading':
            r=SensorReading.from_dict(event['reading']);loss.before(r.available_at_seconds)
            belief.assimilate(r,run.wind,None);loss.refresh(belief,r.available_at_seconds)
        elif event['op']=='finish':
            t=event['receipt']['received_at_seconds'];loss.before(t);belief.advance(t,run.wind,None);loss.refresh(belief,t)
    duration=cfg['closed_duration_seconds'];loss.before(duration)
    forecasts=[]
    for horizon in cfg['forecast_horizons_seconds']:
        b=belief.clone();b.advance(duration+horizon,run.wind,None);mean,_=b.summary()
        forecasts.append(float(np.mean((mean-loss.truth(env.d.tick_of(duration+horizon)))**2)))
    return {'JT':float(np.mean(loss.errors)),
            'post_startup_mse':float(np.mean(loss.errors[env.d.tick_of(33.575):])),
            'forecast_mse':forecasts}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--smoke',action='store_true')
    p.add_argument('--resolution',type=Path);args=p.parse_args()
    cfg=json.loads((args.study/'config.json').read_text())
    cfg['closed_loop']['moment_solver']='analytic_truncated_normal_v1_with_declared_quadrature_fallback'
    verification=json.loads((args.study/'fixed-verification.json').read_text())
    if not args.smoke and not verification['all_final_refinements_passed']:
        if args.resolution is None:raise ValueError('Closed-loop execution waits for fixed-path numerical refinement')
        resolution=json.loads(args.resolution.read_text())
        assert resolution['original_configuration_sha256']==sha(args.study/'config.json')
        assert resolution['original_study']==str(args.study.resolve())
        assert resolution['all_final_refinements_passed']
        assert all(sha(Path(file))==digest for file,digest in resolution['source_file_hashes'].items())
    shape=cfg['closed_loop']['grid_shape'];grid=args.study/f'grid-{shape[0]}x{shape[1]}'
    if args.smoke:
        shape=cfg['grid_shapes'][0];grid=args.study/f'grid-{shape[0]}x{shape[1]}'
        cfg['closed_loop']['cases']=[cfg['heldout'][0]['name']];cfg['closed_loop']['seeds']=cfg['heldout_seeds'][:1];cfg['amounts']=[.5]
    out=args.output;out.mkdir(parents=True,exist_ok=False);save(out/'config.json',cfg)
    if args.resolution:shutil.copy2(args.resolution,out/'numerical-resolution.json')
    shutil.copy2(__file__,out/Path(__file__).name)
    shutil.copy2('src/plume_focus/source_amount_validation.py',out/'source_amount_validation.py')
    shutil.copy2('configs/source-prior-study/source-amount-path-audit-v1.json',out/'path-audit-config.json')
    run=decode_config(json.loads(Path(cfg['baseline_config']).read_text()));model=replace(run.model,grid_shape=tuple(shape))
    d=ClockDynamics(model,run.wind.velocity_yx,tick=cfg['clock_tick_seconds'],half_life=run.session.half_life_seconds)
    env=SavedEnvironment(grid/'environment',d)
    sources=grid_positions((cfg['source_orders'][-1],)*2)
    E=gaussian_source_input(model.grid_shape,sources,source_sigma=model.source_sigma,
        total_rate=cfg['reference_total_per_second'],formulation_id=cfg['source_input_id'])
    cases=[{'scene':s,'amount':a,'seed':seed} for s in cfg['closed_loop']['cases'] for a in cfg['amounts'] for seed in cfg['closed_loop']['seeds']]
    summaries=[]
    for ci,case in enumerate(cases):
        pair=[];prefix=[];pair_events=[]
        for method in ['fixed','joint']:
            folder=out/f'case-{ci:02d}-{method}';folder.mkdir()
            result,events,errors,frames,times,end=run_case(env,run,cfg,case,method,sources,E)
            save(folder/'result.json',result);save(folder/'events.json',events)
            np.savez_compressed(folder/'arrays.npz',errors=errors,frames=frames,times=times,final_mean=end)
            reproduced,_,e2,frames2,_,end2=run_case(env,run,cfg,case,method,sources,E,replay=json.loads(json.dumps(events)))
            assert np.array_equal(errors,e2) and np.array_equal(frames,frames2) and np.array_equal(end,end2)
            if ci==0:
                shadow=replay_belief_only(env,run,cfg,case,method,sources,E,events)
                assert abs(shadow['JT']-result['JT'])<1e-12
                assert abs(shadow['post_startup_mse']-result['post_startup_mse'])<1e-12
            result['causal_replay_bit_identical']=True;save(folder/'result.json',result)
            pair.append(result)
            pair_events.append(events)
            first=[]
            for e in events:
                first.append(e)
                if e['op']=='finish':break
            prefix.append(first)
            print(json.dumps({'closed_case':ci,'method':method,'JT':result['JT'],'readings':result['readings'],'replay':True,'seconds':result['wall_seconds']}),flush=True)
        assert prefix[0]==prefix[1]
        def signature(r):
            return {**{k:v for k,v in r.items() if k not in ['scan_id','reading_id','acquired_at_seconds','available_at_seconds']},
                    'acquired_tick':d.tick_of(r['acquired_at_seconds']),'available_tick':d.tick_of(r['available_at_seconds'])}
        signatures=[[signature(e['reading']) for e in es if e['op']=='reading'] for es in pair_events]
        same=signatures[0]==signatures[1]
        if same:
            fixed_on_joint=pair[0];joint_on_fixed=pair[1]
        else:
            fixed_on_joint=replay_belief_only(env,run,cfg,case,'fixed',sources,E,pair_events[1])
            joint_on_fixed=replay_belief_only(env,run,cfg,case,'joint',sources,E,pair_events[0])
        cross={'case':case,'identical_delivered_readings':same,
               'JT_matrix':[[pair[0]['JT'],joint_on_fixed['JT']],[fixed_on_joint['JT'],pair[1]['JT']]],
               'post_startup_matrix':[[pair[0]['post_startup_mse'],joint_on_fixed['post_startup_mse']],
                                      [fixed_on_joint['post_startup_mse'],pair[1]['post_startup_mse']]],
               'path_change_percent_common_joint':100*(pair[1]['JT']/joint_on_fixed['JT']-1),
               'path_change_percent_common_fixed':100*(fixed_on_joint['JT']/pair[0]['JT']-1)}
        save(out/f'case-{ci:02d}-cross-replay.json',cross)
        summaries.append({'case':case,'methods':pair,'initial_full_events_identical':True,
                          'joint_change_percent':100*(pair[1]['JT']/pair[0]['JT']-1),'cross_replay':cross})
        save(out/'summary.json',summaries)
    for key in ['deterministic','noise']:assert sha(grid/'environment'/f'{key}.npy')==env.info['hashes'][key]
    save(out/'verification.json',{'all_replays_bit_identical':True,'same_initial_full_scan':True,
        'truth_arrays_unchanged':True,'shadow_adapter_verified':True,
        'common_filter_path_cross_replay_complete':True,'smoke_only':args.smoke,'executions':len(cases)*2})
    save(out/'artifact-manifest.json',{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact-manifest.json'})


if __name__=='__main__':main()
