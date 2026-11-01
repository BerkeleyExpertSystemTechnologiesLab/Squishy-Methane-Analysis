"""Read frozen mismatch artifacts; expose unresolved HPD grid membership."""
from pathlib import Path
import argparse,json
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();root=Path(args.input);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    cfg=json.loads((root/'config.json').read_text());summary=json.loads((root/'summary.json').read_text());unresolved=[]
    for scenario in summary['scenarios']:
        for i,method in enumerate(scenario['methods']):
            low=high=0
            for control in cfg['controls'][1:]:
                folder=root/scenario['label']/control['label'];ref=json.loads((folder/'refinement.json').read_text())[-1]
                result=json.loads((folder/f"results-{ref['orders'][-1]}.json").read_text())['methods'][i]
                if not ref['methods'][i]['hpd95_membership_same']:
                    high+=1;unresolved.append({'scenario':scenario['label'],'control':control['label'],'method':method['method'],
                        'finest_hpd_rank':result['true_location_hpd_rank'],'refinement':ref['methods'][i]})
                else:low+=int(result['true_in_hpd95']);high+=int(result['true_in_hpd95'])
            method['hpd95_count_grid_resolution_bounds']=[low,high]
    refs=[json.loads(f.read_text())[-1] for f in root.glob('*/*/refinement.json')]
    summary['unresolved_hpd_cases']=unresolved
    summary['source_mean_and_radius_mass_checks_all_pass']=all(
        r['mean_shift_fov']<cfg['source_refinement']['mean_shift_fov_tolerance'] and
        r['radius_mass_change']<cfg['source_refinement']['mass_radius_005_absolute_tolerance']
        for ref in refs for r in ref['methods'])
    (out/'summary-with-resolution-bounds.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,4.5),layout='constrained')
    for ax,factor in zip(axes,['wind_speed','source_width','half_life']):
        rows=sorted([s for s in summary['scenarios'] if s['factor']==factor]+[summary['scenarios'][0]],key=lambda s:s['multiplier'])
        x=[r['multiplier'] for r in rows]
        for i,color in enumerate(['#555555','#1674b8','#d67d00']):
            ax.plot(x,[r['methods'][i]['mean_error_fov'] for r in rows],'o-',color=color,label=rows[0]['methods'][i]['method'])
        ax.plot(x,[r['methods'][0]['mean_control']['mean_error_fov'] for r in rows],'s:',color='#555555',label='broad: no perturbation noise')
        ax.axhline(.06285,ls='--',color='#b83e55',label='Original demo ~.063')
        ax.set(title=factor,xlabel='Truth / assumed parameter',ylabel='Source mean-location error (FOV)',ylim=(0,.07))
        ax.set_xticks([.5,.75,1.,1.25,1.5]);ax.legend(fontsize=7)
    fig.suptitle('Solid lines: 10 paired-seed mean error; dotted: deterministic control')
    fig.savefig(out/'source-error-comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':main()
