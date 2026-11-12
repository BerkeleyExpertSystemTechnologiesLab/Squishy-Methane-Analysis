"""Summarize frozen validation outcomes and render scientific result figures."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',required=True,type=Path)
    p.add_argument('--closed',type=Path);p.add_argument('--uncertainty',type=Path);args=p.parse_args()
    cfg=json.loads((args.study/'config.json').read_text());summary=json.loads((args.study/'fixed-summary.json').read_text())
    colors=['#d55e00','#0072b2','#009e73'];labels=['Fixed default amount','Joint location + amount','True-amount control']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(len(summary),3,figsize=(13,4*len(summary)),layout='constrained',squeeze=False)
    checks=[];numeric=[]
    for gi,result in enumerate(summary):
        rs=[r for r in result['summary'] if r['split']=='heldout' and r['scene']=='all']
        grid=args.study/f"grid-{result['grid'][0]}x{result['grid'][1]}"
        fine=grid/f"source-{cfg['source_orders'][-1]}-amount-{cfg['amount_orders'][-1]}"
        raw=json.loads((fine/'results.json').read_text())
        for mi,(label,color) in enumerate(zip(labels,colors)):
            for col in range(3):
                ys=[r['JT'][mi] if col==0 else r['forecast_mse'][col-1][mi] for r in rs]
                axes[gi,col].bar(np.arange(3)+(mi-1)*.24,ys,width=.23,color=color,label=label)
        for col in range(3):
            axes[gi,col].set_xticks(range(3),['Low (0.5x)','Nominal (1x)','High (1.5x)'])
            axes[gi,col].set(title=f"{result['grid'][0]}x{result['grid'][1]}: "+['Monitoring J/T','Forecast +6 s','Forecast +12 s'][col],ylabel='Field squared error')
            axes[gi,col].grid(axis='y',alpha=.2);axes[gi,col].set_axisbelow(True)
        for r in rs:
            final=[u for u in raw['uncertainty'] if u['time']==40. and raw['metadata'][u['history']]['split']=='heldout' and raw['metadata'][u['history']]['amount']==r['amount']]
            amount_errors=[float(np.mean([u['amount_absolute_error'] for u in final if u['method']==method])) for method in ['fixed','joint','oracle']]
            if r['amount']==1.:
                passed=r['JT'][1]<=r['JT'][0]*(1+cfg['scientific_checks']['nominal_max_relative_regression'])
                checks.append({'grid':result['grid'],'amount':r['amount'],'check':'nominal noninferiority <=2% J/T','passed':passed})
            else:
                checks.append({'grid':result['grid'],'amount':r['amount'],'check':'J/T improves','passed':r['JT'][1]<r['JT'][0]})
                for i,h in enumerate(cfg['forecast_horizons_seconds']):checks.append({'grid':result['grid'],'amount':r['amount'],'check':f'forecast +{h:g}s improves','passed':r['forecast_mse'][i][1]<r['forecast_mse'][i][0]})
                checks.append({'grid':result['grid'],'amount':r['amount'],'check':'amount absolute error improves','passed':amount_errors[1]<amount_errors[0]})
            numeric.append({'grid':result['grid'],'amount':r['amount'],'JT':r['JT'],
                'candidate_change_percent':r['candidate_change_percent'],'improved_histories':r['improved_histories'],
                'histories':r['histories'],'forecast_mse':r['forecast_mse'],'amount_absolute_errors':amount_errors})
    fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle(f"Independent scenes: {len(cfg['heldout'])} source positions x {len(cfg['heldout_seeds'])} new noise seeds per amount\nSame instrument and fixed path; each grid uses its own matched finite-model truth",fontsize=13)
    fig.savefig(args.study/'heldout-reconstruction.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,len(summary),figsize=(6.5*len(summary),4.5),layout='constrained',squeeze=False)
    for gi,result in enumerate(summary):
        ax=axes[0,gi];scenes=[s['name'] for s in cfg['heldout']]
        a=np.array([[next(r['candidate_change_percent'] for r in result['summary'] if r['split']=='heldout' and r['scene']==scene and r['amount']==amount) for amount in cfg['amounts']] for scene in scenes])
        limit=max(5,float(np.max(np.abs(a))))
        im=ax.imshow(a,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        for i in range(len(scenes)):
            for j in range(3):ax.text(j,i,f'{a[i,j]:+.1f}%',ha='center',va='center',color='white' if abs(a[i,j])>.6*limit else 'black')
        ax.set_xticks(range(3),['Low','Nominal','High']);ax.set_yticks(range(len(scenes)),scenes)
        ax.set_title(f"{result['grid'][0]}x{result['grid'][1]} joint vs fixed J/T")
        fig.colorbar(im,ax=ax,label='Change in reconstruction error (%)')
    fig.suptitle('Retain every independent scene: negative values mean improvement',fontsize=13)
    fig.savefig(args.study/'heldout-scenes.png',dpi=170);plt.close(fig)
    save(args.study/'scientific-checks.json',{'checks':checks,'all_passed':all(c['passed'] for c in checks),'outcomes':numeric,
         'scope':'Exploratory heldout evidence, not a calibration/generalization guarantee; no tuning after execution.'})
    if args.uncertainty:
        u=json.loads((args.uncertainty/'summary.json').read_text())
        fig,axes=plt.subplots(1,len(summary),figsize=(6.5*len(summary),4.4),layout='constrained',squeeze=False)
        for gi,r in enumerate(summary):
            ax=axes[0,gi]
            for method,label,color in zip(['fixed','joint','oracle'],labels,colors):
                vals=[next(x for x in u if x['grid']==r['grid'] and x['amount']==a and x['method']==method and x['time']==52.) for a in cfg['amounts']]
                ax.plot(range(3),[x['source_inside95_count']/x['histories'] for x in vals],'o-',color=color,label=label)
            ax.axhline(.95,color='gray',ls='--',lw=1)
            ax.set(ylim=(-.05,1.05),xticks=range(3),xticklabels=['Low','Nominal','High'],title=f"{r['grid'][0]}x{r['grid'][1]}: forecast +12 s",ylabel='Fraction inside pointwise 95% interval')
            ax.grid(axis='y',alpha=.2)
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
        fig.suptitle(f"Evaluator-only footprint at each true source: {len(cfg['heldout'])*len(cfg['heldout_seeds'])} heldout histories per amount\nExact finite-mixture intervals; descriptive counts, not a calibration guarantee",fontsize=13)
        fig.savefig(args.uncertainty/'source-footprint-coverage.png',dpi=170);plt.close(fig)
    if args.closed:
        loops=json.loads((args.closed/'summary.json').read_text());loopstats=[]
        loopamounts=[a for a in cfg['amounts'] if any(r['case']['amount']==a for r in loops)]
        fig,axes=plt.subplots(1,3,figsize=(15,4.5),layout='constrained')
        for mi,(label,color) in enumerate(zip(labels[:2],colors[:2])):
            values=[];reads=[]
            for ai,amount in enumerate(loopamounts):
                selected=[r['methods'][mi] for r in loops if r['case']['amount']==amount]
                scores=np.array([r['JT'] for r in selected]);n=np.array([r['readings'] for r in selected])
                values.append(scores.mean());reads.append(n.mean())
                axes[0].scatter(np.full(len(scores),ai+(mi-.5)*.12),scores,color=color,alpha=.4)
                axes[1].scatter(np.full(len(n),ai+(mi-.5)*.12),n,color=color,alpha=.4)
            axes[0].plot(range(len(loopamounts)),values,'o-',color=color,label=label);axes[1].plot(range(len(loopamounts)),reads,'o-',color=color)
        for amount in loopamounts:
            rs=[r for r in loops if r['case']['amount']==amount]
            scores=np.array([[m['JT'] for m in r['methods']] for r in rs]);ns=np.array([[m['readings'] for m in r['methods']] for r in rs])
            loopstats.append({'amount':amount,'pairs':len(rs),'JT':scores.mean(axis=0).tolist(),
                'joint_change_percent':float(100*(scores[:,1].mean()/scores[:,0].mean()-1)),
                'improved_pairs':int(np.sum(scores[:,1]<scores[:,0])), 'mean_readings':ns.mean(axis=0).tolist(),
                'same_scan_modes_and_counts':sum([(r['methods'][0]['readings']==r['methods'][1]['readings'] and [s['mode'] for s in r['methods'][0]['scans']]==[s['mode'] for s in r['methods'][1]['scans']]) for r in rs]),
                'different_measurement_histories':sum(not r['cross_replay']['identical_delivered_readings'] for r in rs)})
            for mi,name in enumerate(['fixed','joint']):
                matrix=np.array([r['cross_replay']['JT_matrix'] for r in rs])
                percent=100*(matrix[:,1,mi].mean()/matrix[:,0,mi].mean()-1)
                loopstats[-1][f'path_change_percent_common_{name}']=float(percent)
                loopstats[-1][f'path_improved_pairs_common_{name}']=int(np.sum(matrix[:,1,mi]<matrix[:,0,mi]))
        for mi,(name,color) in enumerate(zip(['fixed','joint'],colors[:2])):
            values=[r[f'path_change_percent_common_{name}'] for r in loopstats]
            axes[2].plot(range(len(loopamounts)),values,'o-',color=color,label=f'{name.capitalize()} replay estimator')
        axes[2].axhline(0,color='gray',ls='--',lw=1)
        axes[2].legend(fontsize=8,frameon=False)
        for ax in axes:ax.set_xticks(range(len(loopamounts)),[f'{a:g}x' for a in loopamounts]);ax.grid(axis='y',alpha=.2)
        axes[0].set(title='Closed-loop reconstruction on the same 60 s budget',ylabel='Physical-clock J/T')
        axes[1].set(title='Actual acquired readings',ylabel='Reading count')
        axes[2].set(title='Value of changed measurements',ylabel='J/T change under the same replay estimator (%)')
        fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
        fig.suptitle('Same scanner policy and shared truth; paths and readings may differ',fontsize=13)
        fig.savefig(args.closed/'closed-loop-comparison.png',dpi=170);plt.close(fig)
        save(args.closed/'aggregate.json',loopstats)
    print(json.dumps({'scientific_checks_passed':sum(c['passed'] for c in checks),'checks':len(checks),'all_passed':all(c['passed'] for c in checks)},indent=2))


if __name__=='__main__':main()
