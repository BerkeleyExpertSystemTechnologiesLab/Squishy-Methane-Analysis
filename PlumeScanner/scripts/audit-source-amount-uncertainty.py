"""Replay saved readings to audit field variance and evaluator-only signal targets."""
from pathlib import Path
from dataclasses import replace
import argparse,json,shutil
import numpy as np
from plume_focus.full_fov_execution import decode_config
from plume_focus.source_amount_validation import ClockDynamics,ClockAffineFilter
from plume_focus.source_input import gaussian_source_input
from plume_focus.signal_belief import grid_positions,observation_weights
from plume_focus.sensing_contracts import SensorReading


def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    cfg=json.loads((args.study/'config.json').read_text());run=decode_config(json.loads(Path(cfg['baseline_config']).read_text()))
    record=json.loads(Path(cfg['record']).read_text());out=args.output;out.mkdir(parents=True,exist_ok=False)
    auditpath=Path('configs/source-prior-study/source-amount-uncertainty-audit-v1.json')
    audit=json.loads(auditpath.read_text());shutil.copy2(auditpath,out/'config.json');shutil.copy2(__file__,out/Path(__file__).name)
    all_results=[]
    for shape in cfg['grid_shapes']:
        grid=args.study/f'grid-{shape[0]}x{shape[1]}';level=grid/f"source-{cfg['source_orders'][-1]}-amount-{cfg['amount_orders'][-1]}"
        results=json.loads((level/'results.json').read_text());metas=results['metadata']
        arrays=np.load(level/'fields.npz');readings=np.load(level/'evidence-and-errors.npz')['readings']
        d=ClockDynamics(replace(run.model,grid_shape=tuple(shape)),run.wind.velocity_yx,tick=cfg['clock_tick_seconds'],half_life=run.session.half_life_seconds)
        sources=grid_positions((cfg['source_orders'][-1],)*2)
        E=gaussian_source_input(d.model.grid_shape,sources,source_sigma=d.model.source_sigma,
            total_rate=cfg['reference_total_per_second'],formulation_id=cfg['source_input_id'])
        f=ClockAffineFilter(d,E,len(metas));ri=0
        for event in record['events']:
            if event['operation']!='ingest_readings':continue
            r=SensorReading.from_dict(event['readings'][0]);f.advance(d.tick_of(r.acquired_at_seconds))
            f.condition(readings[ri],observation_weights(d.model.grid_shape,r.position_yx,r.footprint));ri+=1
            f.advance(d.tick_of(r.available_at_seconds))
        f.advance(d.tick_of(40.))
        np.savez_compressed(out/f'grid-{shape[0]}x{shape[1]}-final-filter.npz',m=f.m,P=f.P,B=f.B,L=f.L,K=f.K)
        det=np.load(grid/'environment/deterministic.npy',mmap_mode='r');noise=np.load(grid/'environment/noise.npy',mmap_mode='r')
        einfo=json.loads((grid/'environment/environment.json').read_text())
        rows=[];max_replay=0.
        for t in audit['times_seconds']:
            f.advance(d.tick_of(t));truth=[]
            for m in metas:
                field=det[d.tick_of(t),m['scene_index']]*m['amount']
                if m['seed'] is not None:field=field+noise[d.tick_of(t),einfo['seeds'].index(m['seed'])]
                truth.append(field)
            truth=np.asarray(truth)
            for hi,m in enumerate(metas):
                positions=[m['source_yx'],[m['source_yx'][0],min(1.,m['source_yx'][1]+.1)]]
                H=np.stack([observation_weights(d.model.grid_shape,p,run.capabilities.footprint) for p in positions])
                for mi,fixed in enumerate([1.,None,m['amount']]):
                    posterior=f.posterior(hi,fixed,cfg['amount_orders'][-1]);mean,emission=f.summary(hi,posterior)
                    max_replay=max(max_replay,float(np.max(np.abs(mean-arrays[str(t)][hi,mi]))))
                    g=E[posterior['ids']]@f.B.T
                    variance=np.maximum(1e-30,np.diag(f.P)+(posterior['w']*posterior['a2'])@(g*g)-(mean-f.m[hi])**2)
                    error=(mean-truth[hi])**2;ratio=error/variance
                    v=f.validation(hi,posterior,H,H@truth[hi])
                    rows.append({'history':hi,'metadata':m,'method':['fixed','joint','oracle'][mi],'time':t,
                        'field_mse':float(error.mean()),'mean_field_variance':float(variance.mean()),
                        'mean_squared_standardized_error':float(ratio.mean()),
                        'p95_squared_standardized_error':float(np.quantile(ratio,.95)),
                        'max_squared_standardized_error':float(ratio.max()),
                        'positions_yx':positions,'point_pit':v['pit'].tolist(),
                        'point_inside95':((v['pit']>=.025)&(v['pit']<=.975)).tolist(),
                        'point_error_variance_ratio':(v['squared_error']/v['variance']).tolist()})
        assert max_replay<1e-9
        save(out/f'grid-{shape[0]}x{shape[1]}.json',{'rows':rows,'max_saved_map_replay_error':max_replay})
        all_results.append({'grid':shape,'max_replay_error':max_replay,'rows':rows})
        print(json.dumps({'uncertainty_grid':shape,'rows':len(rows),'replay_error':max_replay}),flush=True)
    summary=[]
    for result in all_results:
        for amount in cfg['amounts']:
            for method in ['fixed','joint','oracle']:
                for t in audit['times_seconds']:
                    rs=[r for r in result['rows'] if r['metadata']['split']=='heldout' and r['metadata']['amount']==amount and r['method']==method and r['time']==t]
                    summary.append({'grid':result['grid'],'amount':amount,'method':method,'time':t,
                        'source_inside95_count':sum(r['point_inside95'][0] for r in rs),'histories':len(rs),
                        'downwind_inside95_count':sum(r['point_inside95'][1] for r in rs),
                        'mean_squared_standardized_error':float(np.mean([r['mean_squared_standardized_error'] for r in rs])),
                        'mean_field_variance':float(np.mean([r['mean_field_variance'] for r in rs]))})
    save(out/'summary.json',summary)


if __name__=='__main__':main()
