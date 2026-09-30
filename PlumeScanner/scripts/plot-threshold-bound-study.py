"""Standalone scientific diagnostics, separate from the unchanged video layout."""

import argparse
import os
from pathlib import Path
import json
import hashlib
import numpy as np


def plot(root):
    os.environ['MPLCONFIGDIR']=str((root/'.matplotlib').resolve())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    summary=json.loads((root/'summary.json').read_text());declaration=json.loads((root/'declaration.json').read_text())
    names=[n for n in declaration['fixtures'] if 'motion-risk' not in n];out=root/'plots';out.mkdir(exist_ok=True)
    labels=['coarse known wind','coarse biased wind','double known wind','double biased wind','source-free full','source-free blank']
    # Declaration order is stable, but labels follow names rather than position.
    def label(n):
        if 'scale-0' in n:return 'Source-free blank'
        if 'scale-1' in n:return 'Source-free full signal'
        return ('Coarse' if n.startswith('coarse') else 'Doubled')+(' / known wind' if 'known-wind' in n else ' / biased wind')
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,name in zip(axes.ravel(),names):
        for j,policy in enumerate(('upper','mean','risk','corridor','age')):
            values=[r['J_over_T'] for r in summary['metrics'] if r['job'].startswith(name+'/'+policy+'/short-')]
            if values:ax.bar(j,np.mean(values),yerr=np.std(values,ddof=1)/np.sqrt(len(values)),capsize=3)
        ax.set_xticks(range(5),['Upper','Mean','Risk','Corridor','Age'],rotation=20);ax.set_title(label(name));ax.set_ylabel('Causal J/T')
    fig.suptitle('12-second Gaussian continuations: mean and standard error over eight future-noise seeds\nDescriptive fixtures; common ceilings do not imply equal spending')
    fig.savefig(out/'short-reconstruction.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,name in zip(axes.ravel(),names):
        bound=summary['bound_reliability'].get(name)
        if bound:
            for stratum in ('signal','background'):
                ax.plot([r['horizon'] for r in bound['rows']],
                    [np.nan if r['strata'][stratum]['upper_exceedance_fraction'] is None else r['strata'][stratum]['upper_exceedance_fraction'] for r in bound['rows']],marker='o',label=stratum)
        ax.axhline(.05,color='black',linestyle='--',label='nominal pointwise tail .05');ax.set_ylim(-.02,1.02)
        ax.set_title(label(name));ax.set_xlabel('Forecast seconds');ax.set_ylabel('Actual signal exceeds upper bound')
    axes[0,0].legend(fontsize=8);fig.suptitle('Bound reliability on fixed physical truth\nCells are spatially dependent; these are descriptive frequencies')
    fig.savefig(out/'bound-reliability.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    for ax,name in zip(axes.ravel(),names):
        path=root/'runs'/name/'upper'/'full-seed-0.json'
        if path.exists():
            record=json.loads(path.read_text());plans=[s['plan'] for s in record['scans']]
            ax.step([p['started_at_seconds'] for p in plans],[len(p['eligible_cells'])/np.prod(p['lattice_shape']) for p in plans],where='post',label='eligible')
            ax.step([p['started_at_seconds'] for p in plans],[len(p['swept_cells'])/np.prod(p['lattice_shape']) for p in plans],where='post',label='swept')
            for p in plans:
                if p['status']=='reacquisition':ax.axvspan(p['started_at_seconds'],p['completed_at_seconds'],color='gray',alpha=.12)
        ax.set_title(label(name));ax.set_xlabel('Physical seconds');ax.set_ylabel('FOV fraction');ax.set_ylim(-.02,1.02)
    axes[0,0].legend();fig.suptitle('Upper-threshold region and actual sweep / seed 0\nGray intervals are age-based reacquisition, including partial startup')
    fig.savefig(out/'regions-and-fallback.png',dpi=150);plt.close(fig)
    (out/'manifest.json').write_text(json.dumps({'files':[{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(out.glob('*.png'))]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();plot(args.output)
