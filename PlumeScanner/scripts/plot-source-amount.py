"""Render saved matched source-amount results; never run or change inference."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from plume_focus.full_fov_execution import decode_config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    root = args.output
    summary = json.loads((root/'summary.json').read_text())
    cfg = json.loads((root/'config.json').read_text())
    level = summary['level']
    folder = root/f"grid-{level['source_order']}-amount-{level['amount_order']}"
    result = json.loads((folder/'results.json').read_text())
    histories = result['history_results']
    colors = ['#d55e00', '#0072b2', '#009e73']
    labels = ['Location only, nominal amount', 'Joint location + amount', 'Location only, true amount']
    x = np.arange(3)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.6), layout='constrained')
    for mi, (label, color) in enumerate(zip(labels, colors)):
        for ai, ax in enumerate(axes):
            key = ['primary_checkpoint_mse', 'source_error_fov', 'amount_mean_multiplier'][ai]
            values = []
            for si, scenario in enumerate(cfg['scenarios']):
                hs = [h for h in histories if h['scenario']==scenario['label'] and h['seed'] is not None]
                v = np.array([h['methods'][mi][key] if ai==0 else h['methods'][mi]['final'][key] for h in hs])
                offset = (mi-1)*.12
                ax.scatter(np.full(10, si+offset), v, color=color, alpha=.3, s=15)
                values.append(v.mean())
            ax.plot(x+(mi-1)*.12, values, 'o-', color=color, label=label, lw=1.8)
    axes[2].plot(x, [.5, 1., 1.5], 'k--', lw=1.2, label='True amount')
    for ax in axes:
        ax.set_xticks(x, ['Low (0.5x)', 'Nominal (1x)', 'High (1.5x)'])
        ax.grid(axis='y', alpha=.2)
        ax.set_xlabel('True source input / reference input')
    axes[0].set(title='Reconstruction during scanning', ylabel='Mean checkpoint field MSE')
    axes[1].set(title='Source location at 40 s', ylabel='Posterior-mean location error (FOV)')
    axes[2].set(title='Source amount at 40 s', ylabel='Posterior-mean amount / reference')
    handles, legend_labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='outside lower center', ncol=2, frameon=False)
    fig.suptitle('Matched-model fixed-path experiment: ten paired noise realizations', fontsize=14)
    fig.savefig(root/'comparison.png', dpi=170)
    plt.close(fig)

    fields = np.load(folder/'fields.npz')
    ci = np.flatnonzero(fields['kind']=='final')[0]
    grid = decode_config(json.loads((root/'baseline-configuration.json').read_text())).model.grid_shape
    mean_ids = [next(i for i,h in enumerate(histories) if h['scenario']==s['label'] and h['seed'] is None) for s in cfg['scenarios']]
    vmax = max(float(fields['truth'][ci, hi].max()) for hi in mean_ids)
    vmin = min(0., min(float(fields['mean'][ci, hi].min()) for hi in mean_ids))
    fig, axes = plt.subplots(3, 4, figsize=(13, 8), layout='constrained')
    titles = ['Simulated truth', 'Fixed nominal amount', 'Joint location + amount', 'True-amount control']
    true_y, true_x = cfg['source_yx']
    for si, hi in enumerate(mean_ids):
        for col, ax in enumerate(axes[si]):
            field = fields['truth'][ci, hi] if col==0 else fields['mean'][ci, hi, col-1]
            im = ax.imshow(field.reshape(grid), origin='lower', extent=[0,1,0,1], vmin=vmin, vmax=vmax, cmap='viridis', aspect='auto', interpolation='nearest')
            ax.plot(true_x, true_y, 'w+', ms=11, mew=1.8)
            if col:
                m = histories[hi]['methods'][col-1]['final']
                y, xx = m['source_mean_yx']
                ax.plot(xx, y, 'o', mfc='none', mec='#ff6e54', ms=9, mew=1.8)
                ax.set_title(f"{titles[col]}\nfield MSE {m['field_mse']:.2g}", fontsize=10)
            else:
                ax.set_title(f"{titles[col]}: {cfg['scenarios'][si]['amount_multiplier']:g}x input")
            ax.set(xlim=(0,1), ylim=(0,1), xticks=[0,.5,1], yticks=[0,.5,1])
            if col==0: ax.set_ylabel('Normalized y')
            if si==2: ax.set_xlabel('Normalized x')
    fig.colorbar(im, ax=axes, shrink=.8, label='Synthetic field signal (shared scale)')
    fig.suptitle('Final maps at 40 s: zero-realized-noise controls\nWhite +: true source; red circle: inferred mean source position', fontsize=14)
    fig.savefig(root/'mean-control-maps.png', dpi=170)
    plt.close(fig)

    traces = json.loads((folder/'traces.json').read_text())
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
    for si, (scenario, ax) in enumerate(zip(cfg['scenarios'], axes)):
        ids = [i for i,h in enumerate(histories) if h['scenario']==scenario['label'] and h['seed'] is not None]
        rows = [r for r in traces if r['kind']=='reading']
        for mi,(label,color) in enumerate(zip(labels,colors)):
            ys = np.array([[r['methods'][hi*3+mi]['field_mse'] for r in rows] for hi in ids])
            ax.plot([r['time'] for r in rows], ys.mean(axis=0), color=color, label=label)
        ax.set(title=f"{scenario['label'].capitalize()} input ({scenario['amount_multiplier']:g}x)", xlabel='Reading delivery time (s)', ylabel='Field MSE (mean of ten seeds)')
        ax.grid(alpha=.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Online reconstruction: each score uses only readings already delivered', fontsize=13)
    fig.savefig(root/'checkpoint-errors.png', dpi=170)
    plt.close(fig)


if __name__=='__main__':
    main()
