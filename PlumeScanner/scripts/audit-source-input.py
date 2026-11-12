"""Audit source amount and input discretization without running new inference."""
from pathlib import Path
import argparse, hashlib, json, math, platform, shutil, subprocess, sys
import numpy as np
from scipy.special import ndtr
from plume_focus.source_input import gaussian_source_input, POINT_NORMALIZED_V1, CELL_INTEGRATED_V1
from plume_focus.signal_belief import grid_positions
from plume_focus.full_fov_execution import decode_config
from plume_focus.conserved_source_control import injection


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def legacy(shape, sources, sigma, peak):
    return peak*np.exp(-np.sum((grid_positions(shape)[None]-sources[:, None])**2,
                              axis=2)/(2*sigma**2))


def analytic_input(source, sigma, peak):
    a, b = -source/sigma, (1-source)/sigma
    z = ndtr(b)-ndtr(a)
    amount = peak*2*math.pi*sigma**2*np.prod(z, axis=-1)
    density = lambda x: np.exp(-.5*x*x)/math.sqrt(2*math.pi)
    centroid = source+sigma*(density(a)-density(b))/z
    return amount, centroid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    baseline=json.loads(Path(cfg['baseline_config']).read_text())
    run=decode_config(baseline)
    assert run.digest==cfg['baseline_configuration_hash']
    base_source=np.array(baseline['session']['source_position'])
    peak=run.model.source_rate; sigma0=run.model.source_sigma
    reference=injection(run.model,base_source)
    np.testing.assert_array_equal(legacy(run.model.grid_shape,base_source[None],sigma0,peak)[0], reference)
    total=float(reference.mean())
    named=np.array(list(cfg['named_sources_yx'].values()))
    settings=cfg['phase_coordinates']
    coordinates=np.linspace(settings['minimum'],settings['maximum'],settings['count_per_axis'])
    ys,xs=np.meshgrid(coordinates,coordinates,indexing='ij')
    phase=np.column_stack([ys.ravel(),xs.ravel()])
    rows=[];phase_rows=[];arrays={'phase_sources_yx':phase};max_total_error=0.;max_aggregation_error=0.
    for grid_index, rawshape in enumerate(cfg['grid_shapes']):
        shape=tuple(rawshape);positions=grid_positions(shape)
        for multiplier in cfg['width_multipliers']:
            sigma=sigma0*multiplier; analytic,centroid=analytic_input(named,sigma,peak)
            old=legacy(shape,named,sigma,peak)
            point=gaussian_source_input(shape,named,source_sigma=sigma,total_rate=total,formulation_id=POINT_NORMALIZED_V1)
            cells=gaussian_source_input(shape,named,source_sigma=sigma,total_rate=total,formulation_id=CELL_INTEGRATED_V1)
            total_error=max(float(np.max(np.abs(a.mean(axis=1)/total-1))) for a in [point,cells])
            max_total_error=max(max_total_error,total_error)
            refined=tuple(n*2 for n in shape)
            aggregation={}
            for mode,coarse in [(POINT_NORMALIZED_V1,point),(CELL_INTEGRATED_V1,cells)]:
                fine=gaussian_source_input(refined,named,source_sigma=sigma,total_rate=total,formulation_id=mode)
                coarsened=fine.reshape(len(named),shape[0],2,shape[1],2).mean(axis=(2,4)).reshape(len(named),-1)
                aggregation[mode]=np.mean(np.abs(coarsened-coarse),axis=1)/total
            max_aggregation_error=max(max_aggregation_error,float(aggregation[CELL_INTEGRATED_V1].max()))
            for i,label in enumerate(cfg['named_sources_yx']):
                rows.append({'grid_shape':list(shape),'width_multiplier':multiplier,'source_label':label,
                    'source_yx':named[i].tolist(),'legacy_discrete_total':float(old[i].mean()),
                    'legacy_continuous_in_fov_total':float(analytic[i]),
                    'legacy_discrete_over_continuous':float(old[i].mean()/analytic[i]),
                    'point_normalized_total':float(point[i].mean()),'cell_integrated_total':float(cells[i].mean()),
                    'point_centroid_error_fov':float(np.linalg.norm(point[i]@positions/point[i].sum()-centroid[i])),
                    'integrated_centroid_error_fov':float(np.linalg.norm(cells[i]@positions/cells[i].sum()-centroid[i])),
                    'point_coarse_fine_relative_mass_l1':float(aggregation[POINT_NORMALIZED_V1][i]),
                    'integrated_coarse_fine_relative_mass_l1':float(aggregation[CELL_INTEGRATED_V1][i])})
            old_phase=legacy(shape,phase,sigma,peak); analytic_phase,phase_centroid=analytic_input(phase,sigma,peak)
            normalized={mode:gaussian_source_input(shape,phase,source_sigma=sigma,total_rate=total,formulation_id=mode)
                        for mode in [POINT_NORMALIZED_V1,CELL_INTEGRATED_V1]}
            ratio=old_phase.mean(axis=1)/analytic_phase
            stats={'grid_shape':list(shape),'width_multiplier':multiplier,'phase_count':len(phase),
                'legacy_discrete_over_continuous_min':float(ratio.min()),
                'legacy_discrete_over_continuous_max':float(ratio.max())}
            key=f'grid{shape[0]}x{shape[1]}_width{multiplier}'
            arrays[key+'_legacy_total_ratio']=ratio
            for mode,values in normalized.items():
                errors=np.linalg.norm(values@positions/values.sum(axis=1,keepdims=True)-phase_centroid,axis=1)
                input_error=float(np.max(np.abs(values.mean(axis=1)/total-1)))
                max_total_error=max(max_total_error,input_error)
                stats[mode]={'max_relative_total_error':input_error,'centroid_error95_fov':float(np.quantile(errors,.95))}
                arrays[key+'_'+mode+'_total_ratio']=values.mean(axis=1)/total
            phase_rows.append(stats)
    checks={'relative_total_tolerance':cfg['checks']['relative_total_tolerance'],
        'maximum_relative_total_error':max_total_error,
        'coarse_fine_relative_mass_l1_tolerance':cfg['checks']['coarse_fine_relative_mass_l1_tolerance'],
        'maximum_integrated_coarse_fine_relative_mass_l1':max_aggregation_error,
        'baseline_configuration_hash_unchanged':run.digest,'legacy_baseline_vector_exact':True,
        'passed':max_total_error<cfg['checks']['relative_total_tolerance'] and max_aggregation_error<cfg['checks']['coarse_fine_relative_mass_l1_tolerance']}
    save(out/'config.json',cfg);save(out/'baseline-configuration.json',baseline)
    save(out/'named-position-results.json',rows);save(out/'phase-summary.json',phase_rows)
    np.savez_compressed(out/'phase-arrays.npz',**arrays)
    save(out/'verification.json',checks)
    save(out/'summary.json',{'reference_total_input_per_second':total,'reference_peak_rate':peak,
        'reference_sigma_fov':sigma0,'named_profile_count':len(rows),'phase_profile_count':len(phase_rows)*len(phase),
        'boundary_convention':cfg['boundary_convention'],'checks':checks,'scope':cfg['scope']})
    save(out/'provenance.json',{'starting_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'branch':subprocess.check_output(['git','branch','--show-current'],text=True).strip(),
        'working_tree':subprocess.check_output(['git','status','--short'],text=True),
        'command':sys.argv,'python':sys.version,'platform':platform.platform()})
    impl=out/'implementation';impl.mkdir()
    for filename in ['scripts/audit-source-input.py','src/plume_focus/source_input.py','tests/test_source_input.py',
                     'docs/specs/source-input-audit-v1.md']:
        shutil.copy2(filename,impl/Path(filename).name)
    plot(out,phase_rows,rows,arrays,cfg)
    (out/'execution.log').write_text(json.dumps(checks,indent=2)+'\n')
    save(out/'artifact-manifest.json',{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in out.rglob('*') if p.is_file() and p.name!='artifact-manifest.json'})
    print(json.dumps({'output':str(out),'reference_total':total,'checks':checks},indent=2))
    if not checks['passed']: raise SystemExit('Frozen representation checks failed')


def plot(out,phase_rows,rows,arrays,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12})
    n=cfg['phase_coordinates']['count_per_axis'];extent=[.35,.65,.35,.65]
    key='grid12x16_width0.5';names=['legacy_total_ratio',POINT_NORMALIZED_V1+'_total_ratio',CELL_INTEGRATED_V1+'_total_ratio']
    values=[arrays[key+'_'+name].reshape(n,n) for name in names]
    fig,axes=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
    titles=['Legacy: discrete / continuous input','Point-normalized: total / Q','Cell-integrated: total / Q']
    for ax,a,title in zip(axes,values,titles):
        im=ax.imshow(a,origin='lower',extent=extent,aspect='equal',vmin=0,vmax=max(1.,values[0].max()),cmap='viridis')
        ax.set(title=title,xlabel='Source horizontal FOV',ylabel='Source vertical FOV')
    fig.colorbar(im,ax=axes.tolist(),shrink=.7,label='Input ratio (1 = reference)')
    fig.suptitle('Source position changes while peak / total settings stay fixed\n12 x 16 grid, sigma = 0.0175 FOV; these are source-input audits, not inferred locations',fontsize=14)
    fig.savefig(out/'input-total-phase.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4.7),layout='constrained')
    for mode,color,label in [(POINT_NORMALIZED_V1,'#D28B28','Point-normalized'),(CELL_INTEGRATED_V1,'#217D92','Cell-integrated')]:
        selected=[r for r in phase_rows if r['width_multiplier']==.5]
        axes[0].plot(range(len(selected)),[r[mode]['centroid_error95_fov'] for r in selected],'o-',color=color,label=label)
        selected=[r for r in rows if r['width_multiplier']==.5 and r['source_label']=='reference_source']
        field='point_coarse_fine_relative_mass_l1' if mode==POINT_NORMALIZED_V1 else 'integrated_coarse_fine_relative_mass_l1'
        axes[1].semilogy(range(len(selected)),[max(r[field],1e-17) for r in selected],'o-',color=color,label=label)
    labels=['12 x 16','24 x 32','48 x 64','96 x 128']
    for ax in axes:
        ax.set_xticks(range(4),labels);ax.set_xlabel('Source-input grid (fine comparison doubles each axis)')
        ax.legend();ax.spines[['top','right']].set_visible(False)
    axes[0].set(title='Interior input centroid quadrature error',ylabel='95th percentile over source positions (FOV)')
    axes[1].set(title='Coarse vs aggregated fine source mass',ylabel='L1 input discrepancy / Q (floor 1e-17)')
    fig.suptitle('Amount conservation does not eliminate spatial representation error\nSigma = 0.0175 FOV. Centroids use cell centers; these are not reconstruction / localization errors.',fontsize=14)
    fig.savefig(out/'grid-representation.png',dpi=160);plt.close(fig)


if __name__=='__main__': main()
