"""Observer-only attribution figures and historical mixture concentration audit."""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'outputs/.matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from plume_focus.controller import SensingController
from plume_focus.sensing_cli_v1 import config_from_dict, write_json


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def plot(root):
    summary=read(root/'diagnosis-summary.json');controls=read(root/'controls.json')
    output=root/'plots';output.mkdir(exist_ok=True)
    names=list(dict.fromkeys(r['fixture'] for r in summary['rows'] if 'motion-risk' not in r['fixture']))
    fig,axes=plt.subplots(3,2,figsize=(12,12),layout='constrained')
    for ax,name in zip(axes.flat,names):
        for mode,color in [('legacy_clipped_v1','tab:orange'),('zero_inflow_v1','tab:blue')]:
            rows=[r for r in summary['rows'] if r['fixture']==name and r['transport']==mode and r['status']=='complete']
            if not rows:continue
            row=rows[0];candidates=row['candidates']
            for production,marker in [(True,'o'),(False,'^')]:
                chosen=[c for c in candidates if c['production']==production]
                ax.scatter([c['predicted_risk']/12 for c in chosen],[c['repeat']['mean_J_over_T'] for c in chosen],
                           color=color,marker=marker,s=35,alpha=.8,label=f'{mode}: '+('production' if production else 'added'))
            winner=candidates[row['production_predicted_winner']]
            ax.scatter(winner['predicted_risk']/12,winner['repeat']['mean_J_over_T'],s=180,facecolors='none',edgecolors=color)
        ax.set_title(name.replace('-threshold90_segment_v2-seed-0-baseline-risk','').replace('-seed-0-baseline-risk',''),fontsize=9)
        ax.set_xlabel('Model-predicted risk / second');ax.set_ylabel('Realized loss / second');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Identical candidate continuations: predicted versus realized value\nCircles around points mark the model-preferred production candidate; fixed source realizations')
    fig.savefig(output/'candidate-values.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,2,figsize=(12,11),layout='constrained')
    for ax,name in zip(axes.flat,names):
        for mode,color in [('legacy_clipped_v1','tab:orange'),('zero_inflow_v1','tab:blue')]:
            rows=[r for r in controls['forecasts'] if r['fixture']==name and r['transport']==mode]
            ax.plot([r['horizon'] for r in rows],[r['actual_mse'] for r in rows],color=color,label=f'{mode}: error')
            ax.plot([r['horizon'] for r in rows],[r['predicted_variance'] for r in rows],color=color,linestyle='--',label=f'{mode}: variance')
        ax.set_title(name.replace('-threshold90_segment_v2-seed-0-baseline-risk','').replace('-seed-0-baseline-risk',''),fontsize=9)
        ax.set_xlabel('Seconds after frozen decision');ax.set_ylabel('Whole-field mean squared quantity');ax.grid(alpha=.2)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Fixed historical evidence; forecasts compared with shared physical truth\nEmpirical error on these fixtures is not a calibrated posterior expectation')
    fig.savefig(output/'forecast-error.png',dpi=150);plt.close(fig)
    declaration=read(root/'declaration.json');motion=[]
    for name,ref in declaration['fixtures'].items():
        if 'motion-risk' not in name or not ref['available']:continue
        trace=Path(ref['trace']);report_path=trace.with_name('report.json');report=read(report_path)
        config=config_from_dict(report['configuration'])
        initial=SensingController.from_session(config.session).motion_diagnostics()
        transport_velocity=np.asarray(config.source.wind_velocity)*config.source.fps/(np.asarray(config.source.shape)-1)
        true_bias=transport_velocity-np.asarray(config.wind.velocity_yx)
        rows=[{'at_seconds':0.,'stage':'initial_prior','motion':initial}]
        for scan in report['scan_schedule']:
            rows.append({'at_seconds':scan['receipt_available_seconds'],'stage':'after_completed_scan',
                         'scan_id':scan['scan_id'],'motion':scan['posterior_motion'],
                         'preceding_decision_model_risk':scan['expected_integrated_map_risk'],
                         'preceding_decision_realized_followon_loss':scan['actual_followon_integrated_map_error'],
                         'preceding_lookahead_seconds':scan['lookahead_seconds']})
        for row in rows:
            row['bias_error_yx']=(np.asarray(row['motion']['inferred_bias_yx'])-true_bias).tolist()
        motion.append({'fixture':name,'report_sha256':hashlib.sha256(report_path.read_bytes()).hexdigest(),
                       'evaluator_only_source_transport_yx':transport_velocity.tolist(),
                       'evaluator_only_true_bias_yx':true_bias.tolist(),
                       'frozen_decision_time':ref['time'],'rows':rows})
    write_json(root/'mixture-concentration-history.json',{'artifact_type':'historical_mixture_concentration_audit_v1',
        'fixtures':motion,'interpretation':'Concentration is observed, not independently established as premature. Historical risks have their own declared horizons; use new paired continuations for attribution.'})
    write_json(output/'manifest.json',{'files':[{'path':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                                             for p in output.glob('*.png')]})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();plot(args.output)
