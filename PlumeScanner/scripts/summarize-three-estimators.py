"""Compare three estimators on saved paired histories, without rerunning inference."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def paired(a, b):
    delta = np.asarray(a)-np.asarray(b)
    return {'mean_difference': float(delta.mean()),
            'standard_error': float(delta.std(ddof=1)/np.sqrt(len(delta))),
            'lower_error_count': int(np.sum(delta<0)),
            'equal_error_count': int(np.sum(delta==0)),
            'higher_error_count': int(np.sum(delta>0)),
            'per_seed_differences': delta.tolist()}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root, out = args.input, args.output
    manifest_path = root/'closeout-manifest.json'
    frozen = json.loads(manifest_path.read_text())
    assert all(sha(root/name)==digest for name,digest in frozen.items())
    cfg = json.loads((root/'config.json').read_text())
    summary = json.loads((root/'summary.json').read_text())
    assert summary['final_refinement_passed']
    level = summary['level']
    result_path = root/f"grid-{level['source_order']}-amount-{level['amount_order']}"/'results.json'
    histories = json.loads(result_path.read_text())['history_results']
    aliases = ['Fixed default amount', 'Joint location + amount', 'True-amount control']
    assert cfg['methods']==['location_only_nominal_amount', 'joint_location_amount', 'location_only_oracle_amount']
    out.mkdir(parents=True, exist_ok=False)
    save(out/'config.json', {
        'analysis_id':'three_estimator_paired_comparison_v1',
        'input_directory':str(root.resolve()),
        'input_manifest_sha256':sha(manifest_path),
        'input_results_sha256':sha(result_path),
        'methods':dict(zip(cfg['methods'], aliases)),
        'estimator_scope':'The fixed-default-amount control permits uncertain location; it is not the unchanged historical production estimator or policy.',
        'metric':cfg['metrics'],
        'oracle_gap_percent':'100*(candidate mean MSE / oracle mean MSE - 1)',
        'recovered_benefit_percent':'100*(fixed mean MSE - candidate mean MSE)/(fixed mean MSE - oracle mean MSE); undefined when denominator is zero, never clamped.',
        'data_policy':'Derive from frozen histories only; no new evidence, model tuning, or policy execution.',
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    })
    shutil.copy2(result_path, out/'input-results.json')
    shutil.copy2(__file__, out/Path(__file__).name)
    rows = []
    for scenario in cfg['scenarios']:
        hs = sorted((h for h in histories if h['scenario']==scenario['label'] and h['seed'] is not None),key=lambda h:h['seed'])
        assert [h['seed'] for h in hs]==[s for s in cfg['seeds'] if s is not None]
        assert all([m['method'] for m in h['methods']]==cfg['methods'] for h in hs)
        mse = np.array([[h['methods'][mi]['primary_checkpoint_mse'] for h in hs] for mi in range(3)])
        loc = np.array([[h['methods'][mi]['final']['source_error_fov'] for h in hs] for mi in range(3)])
        amount = np.array([[h['methods'][mi]['final']['amount_mean_multiplier'] for h in hs] for mi in range(3)])
        average = mse.mean(axis=1)
        if scenario['amount_multiplier']==1:
            assert np.array_equal(mse[0], mse[2]) and np.array_equal(loc[0],loc[2])
        row = {**scenario, 'seeds':[h['seed'] for h in hs],
               'checkpoint_mse_mean':average.tolist(),
               'final_source_error_mean_fov':loc.mean(axis=1).tolist(),
               'final_amount_mean_multiplier':amount.mean(axis=1).tolist(),
               'candidate_mse_change_percent':float(100*(average[1]/average[0]-1)),
               'oracle_mse_change_percent':float(100*(average[2]/average[0]-1)),
               'candidate_excess_oracle_percent':float(100*(average[1]/average[2]-1)),
               'recovered_oracle_benefit_percent':None if average[0]==average[2] else float(100*(average[0]-average[1])/(average[0]-average[2])),
               'paired_comparisons':{},
               'per_seed_checkpoint_mse':mse.tolist()}
        for name, a, b in [('candidate_minus_fixed',1,0),('oracle_minus_fixed',2,0),('candidate_minus_oracle',1,2)]:
            row['paired_comparisons'][name]={'checkpoint_mse':paired(mse[a],mse[b]), 'final_source_error_fov':paired(loc[a],loc[b])}
        rows.append(row)
    save(out/'comparison.json', {'methods':aliases,'scenarios':rows,
        'limits':'One source geometry and frozen source-informed path; oracle is an information control, not a zero-error bound on every realized seed. Ratios use scenario means, not means of per-seed ratios.'})
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(1,2, figsize=(12.5,4.8),layout='constrained')
    colors = ['#d55e00','#0072b2','#009e73']
    x = np.arange(3)
    for mi,(label,color) in enumerate(zip(aliases,colors)):
        axes[0].bar(x+(mi-1)*.24, [r['checkpoint_mse_mean'][mi]*1000 for r in rows], width=.23,color=color,label=label)
    for si,r in enumerate(rows):
        mse = np.asarray(r['per_seed_checkpoint_mse'])
        gaps = 100*(mse[1]/mse[2]-1)
        axes[1].scatter(np.full(10,si),gaps,color=colors[1],alpha=.4,s=25)
        value = r['candidate_excess_oracle_percent']
        axes[1].plot(si,value,'D',color='black',ms=7)
        axes[1].annotate(f'{value:.2f}%',(si,value),xytext=(10,8),textcoords='offset points')
    for ax in axes:
        ax.set_xticks(x,['Low (0.5x)','Nominal (1x)','High (1.5x)'])
        ax.set_xlabel('True source input / reference input')
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set(title='Reconstruction error: lower is better',ylabel='Mean checkpoint field MSE x 1000')
    axes[1].axhline(0,color='#666666',lw=1,ls='--')
    axes[1].set(title='Remaining gap to knowing the true amount',ylabel='Candidate MSE above true-amount control (%)',xlim=(-.45,2.65))
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
    fig.suptitle('Same readings, same model and same path: ten paired noise seeds per amount\nRight: dots show individual seeds; black diamonds use ratios of mean errors',fontsize=13)
    fig.savefig(out/'three-estimators.png',dpi=170)
    plt.close(fig)
    # All source artifacts must remain byte-identical after the analysis.
    assert all(sha(root/name)==digest for name,digest in frozen.items())
    save(out/'verification.json', {'all_input_manifest_entries_unchanged':True,
        'input_file_count':len(frozen),'final_refinement_passed':True,
        'nominal_fixed_and_oracle_scores_identical':True,'seed_pairing_and_method_order_verified':True,
        'new_inference_or_evidence_generated':False})
    save(out/'artifact-manifest.json',{str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='artifact-manifest.json'})
    print(json.dumps({'output':str(out),'scenarios':[{k:r[k] for k in ['label','candidate_mse_change_percent','candidate_excess_oracle_percent','recovered_oracle_benefit_percent']} for r in rows]},indent=2))


if __name__=='__main__':
    main()
