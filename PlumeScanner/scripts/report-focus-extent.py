"""Figures and hash-checked summary; no inference or allocation changes."""
import argparse
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path('outputs/focus-extent'))
    args=parser.parse_args();root=args.root;declaration=read(root/'declaration.json')
    rows=[];figures=[];refinement=[]
    for job in declaration['jobs']:
        record=read(root/'runs'/f"{job['name']}.json")
        assert record['declaration_sha256']==sha(root/'declaration.json')
        assert sha(job['input'])==job['input_sha256']
        saved=read(job['input'])
        for snap,artifact in zip(record['snapshots'],record['artifacts']):
            assert sha(artifact['path'])==artifact['sha256']
            baseline=next(r for r in snap['forecast_records'] if r['variant']=='baseline' and r['horizon_seconds']==6)
            history=next(r for r in snap['forecast_records'] if r['variant']=='history_source_off' and r['horizon_seconds']==6)
            rows.append({'name':job['name'],'noise':job['noise'],'event_index':snap['event_index'],
                'at_seconds':snap['at_seconds'],'mode':snap['mode'],
                'requested_cells':snap['requested_cells'],'swept_cells':snap['swept_cells'],
                'forecast_mean_cells':baseline['mean']['cells'],'forecast_upper_cells':baseline['upper']['cells'],
                'truth_cells':baseline['truth']['cells'],'extra_background_cells':baseline['uncertainty_added_background_cells'],
                'extra_signal_cells':baseline['uncertainty_added_signal_cells'],
                'source_off_mean_cells':history['mean']['cells'],
                'source_off_mse':history['mean_squared_error'],'baseline_mse':baseline['mean_squared_error']})
            for h in (0,6):
                coarse,fine=[r for r in snap['projection_refinement'] if r['horizon_seconds']==h]
                error=abs(coarse['mean_squared_error']-fine['mean_squared_error'])
                # Thresholded geometry also has cell discretization error. Keep
                # it separate from loss convergence and use physical pitch.
                pitch=max(1/n for n in saved['configuration']['session']['instrument']['lattice_shape'])
                widths={};presence_agrees=True
                for field in ('mean','upper','truth'):
                    for key in ('downwind_extent','crosswind_extent'):
                        a,b=coarse[field][key],fine[field][key]
                        presence_agrees &= (a is None)==(b is None)
                        widths[field+'_'+key]=None if a is None or b is None else abs(a-b)
                refinement.append({'name':job['name'],'event_index':snap['event_index'],'horizon':h,
                    'loss_absolute_change':error,'loss_within_1_percent':error<=max(1e-10,.01*fine['mean_squared_error']),
                    'width_changes':widths,'physical_pitch':pitch,
                    'support_presence_agrees':presence_agrees,
                    'width_within_pitch':presence_agrees and all(v is None or v<=pitch for v in widths.values())})
            if job['noise']!=0:continue
            arrays=np.load(artifact['path']);mean=arrays['baseline_6_mean'];upper=arrays['baseline_6_upper']
            truth=arrays['baseline_6_truth'];shape=mean.shape
            plan=saved['events'][snap['event_index']]['plan']
            requested=np.zeros(shape);requested.ravel()[plan['requested_cells']]=1
            swept=np.zeros(shape);swept.ravel()[plan['swept_cells']]=1
            sample=np.zeros(shape)
            scan=next((s for s in saved['scans'] if s['plan']['scan_id']==plan['scan_id']),None)
            if scan:
                for event in saved['events']:
                    for r in event.get('readings',[]):
                        if r['scan_id']==plan['scan_id']:
                            cell=tuple(np.minimum((np.array(r['position_yx'])*shape).astype(int),np.array(shape)-1))
                            sample[cell]=1
            fig,axes=plt.subplots(3,3,figsize=(12,10),layout='constrained')
            panels=[('Delivered values (unknown gray)',np.ma.masked_invalid(arrays['measured'])),
                ('Current mean',arrays['baseline_0_mean']),('Current clean truth',arrays['baseline_0_truth']),
                ('Mean at +6 s',mean),('Upper 95% at +6 s',upper),('Clean truth at +6 s',truth),
                ('Requested cells',requested),('Mechanical sweep',swept),('Actual future sample cells',sample)]
            vmax=max(.3,float(np.nanmax(truth)),float(np.nanmax(mean)))
            for index,(ax,(title,data)) in enumerate(zip(axes.flat,panels)):
                cmap=plt.get_cmap('viridis').copy();cmap.set_bad('#b0b0b0')
                im=ax.imshow(data,origin='upper',extent=(0,1,1,0),vmin=0,vmax=vmax if index<6 else 1,cmap=cmap,aspect='auto')
                if index<6 and np.nanmin(data)<.15<np.nanmax(data):
                    ax.contour((np.arange(shape[1])+.5)/shape[1],(np.arange(shape[0])+.5)/shape[0],data,levels=[.15],colors=['white'],linewidths=.8)
                ax.set_title(title,fontsize=10);ax.set_xlabel('x / FOV');ax.set_ylabel('y / FOV');fig.colorbar(im,ax=ax,shrink=.7)
            fig.suptitle(f"{job['name']} at {snap['at_seconds']:.3f} s — {snap['mode']}\nWhite contour = 0.15; future execution shown retrospectively, not planner evidence")
            out=root/'figures'/f"{job['name']}-event-{snap['event_index']}.png";out.parent.mkdir(exist_ok=True)
            fig.savefig(out,dpi=140);plt.close(fig);figures.append(str(out))
            fig,axes=plt.subplots(1,3,figsize=(14,4),layout='constrained')
            velocity=np.array(saved['configuration']['wind']['velocity_yx']);direction=velocity/np.linalg.norm(velocity)
            normal=np.array([-direction[1],direction[0]])
            y,x=np.indices(shape);positions=np.stack(((y.ravel()+.5)/shape[0],(x.ravel()+.5)/shape[1]),axis=1)
            peak=positions[np.argmax(truth)] if truth.max()>0 else positions[np.argmax(mean)]
            for ax,axis,title in zip(axes[:2],(direction,normal),('Downwind maximum profile','Crosswind maximum profile')):
                coordinate=(positions-peak)@axis;bins=np.linspace(coordinate.min()-1e-9,coordinate.max()+1e-9,33)
                which=np.digitize(coordinate,bins)-1
                for values,name in ((truth,'truth'),(mean,'mean'),(upper,'upper 95%'),(arrays['history_source_off_6_mean'],'history source off')):
                    profile=[float(values.ravel()[which==i].max()) if np.any(which==i) else np.nan for i in range(32)]
                    ax.plot((bins[1:]+bins[:-1])/2,profile,label=name)
                ax.axhline(.15,color='gray',ls=':');ax.set_title(title);ax.set_xlabel('FOV displacement from reference peak');ax.legend(fontsize=7)
            for key in ('mean','upper','truth'):
                values=[r for r in snap['forecast_records'] if r['variant']=='baseline']
                axes[2].plot([r['horizon_seconds'] for r in values],[r[key]['area'] for r in values],label=key)
            axes[2].set_title('Area above 0.15');axes[2].set_xlabel('Forecast horizon / s');axes[2].legend()
            fig.suptitle(f"{job['name']} at {snap['at_seconds']:.3f} s; profiles at +6 s\nReference peak: truth when present, otherwise mean; maxima within fixed coordinate bins")
            out=root/'figures'/f"{job['name']}-event-{snap['event_index']}-profiles.png"
            fig.savefig(out,dpi=140);plt.close(fig);figures.append(str(out))
    summary={'artifact_type':'focus_extent_summary_v1','declaration_sha256':sha(root/'declaration.json'),
        'snapshots':rows,'refinement':refinement,'figures':figures,
        'conclusion_scope':'Fixed evidence and isolated forecasts only; not alternative executed policy outcomes.'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False),encoding='utf-8')
    paths=[root/'declaration.json',root/'controls.json',root/'summary.json',*sorted((root/'runs').glob('*.json')),
           *sorted((root/'arrays').glob('*.npz')),*map(Path,figures)]
    manifest={'artifact_type':'focus_extent_artifacts_v1','artifacts':[{'path':str(p),'sha256':sha(p)} for p in paths]}
    (root/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(f'{len(rows)} snapshots; {len(figures)} figures; {len(paths)} hashed artifacts')

if __name__=='__main__':main()
