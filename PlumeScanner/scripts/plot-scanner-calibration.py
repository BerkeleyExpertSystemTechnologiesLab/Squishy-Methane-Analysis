"""Separate diagnostic plots; the operational renderer is unchanged."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('outputs/scanner-calibration'));a=p.parse_args()
    root=a.root;summary=json.loads((root/'summary.json').read_text());out=root/'plots';out.mkdir(exist_ok=True)
    for profile in ('coarse','double'):
        fig,axes=plt.subplots(3,2,figsize=(12,10),sharex=True)
        for ax,case in zip(axes.ravel(),('continuous','wind_error','weak','blank','cessation','intermittent')):
            for variant,color in (('baseline','tab:blue'),('calibrated','tab:orange'),('full','tab:gray')):
                name=f'heldout-{profile}-{case}-seed101-noise0.01-{variant}'
                record=json.loads((root/'runs'/f'{name}.json').read_text())
                samples=record['quadrature_samples'];times=np.array([q['time'] for q in samples])
                errors=np.array([q['map_mse'] for q in samples])
                ax.plot(times,errors,label=variant,color=color,lw=1)
            ax.set_title(case);ax.set_ylabel('Online field MSE');ax.grid(alpha=.2)
        axes[0,0].legend();axes[-1,0].set_xlabel('Physical seconds');axes[-1,1].set_xlabel('Physical seconds')
        fig.suptitle(f'{profile}: held-out realization 101 / noise 0.01 / no retuning')
        fig.tight_layout();fig.savefig(out/f'{profile}-online-error.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,6),sharex=True)
    for column,case in enumerate(('continuous','blank','cessation')):
        for row,profile in enumerate(('coarse','double')):
            ax=axes[row,column]
            for y,variant in enumerate(('baseline','calibrated','full')):
                name=f'heldout-{profile}-{case}-seed101-noise0.01-{variant}'
                record=json.loads((root/'runs'/f'{name}.json').read_text())
                for scan in record['scans']:
                    plan=scan['plan'];start=plan['started_at_seconds'];end=scan['receipt']['received_at_seconds']
                    ax.broken_barh([(start,end-start)],(y-.35,.7),facecolors='tab:orange' if plan['mode']=='FOCUS' else 'tab:blue')
            ax.set_yticks(range(3),('baseline','calibrated','full'));ax.set_xlim(0,360);ax.set_title(f'{profile} / {case}')
            ax.set_xlabel('Physical seconds')
    fig.suptitle('Executed mode: orange focus, blue full scan, unfilled idle (seed101)')
    fig.tight_layout();fig.savefig(out/'mode-timeline.png',dpi=140);plt.close(fig)
    print(out)


if __name__=='__main__':main()
