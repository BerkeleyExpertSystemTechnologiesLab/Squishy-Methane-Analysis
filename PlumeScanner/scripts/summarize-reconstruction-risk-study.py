"""Paired allocation effects with unchanged estimator/instrument/source controls."""

import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from plume_focus.sensing_cli_v1 import write_json, artifact_manifest, config_from_dict
from reconstruction_risk_observers import clean_sample_diagnostics


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values):
    values=[v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path); args=parser.parse_args(); root=args.directory
    jobs=read(root/'execution-configurations.json')
    if len(jobs)!=96 or sum(j['new'] for j in jobs.values())!=36:
        raise ValueError('Complete the declared 96-record study before classifying outcomes.')
    # Numerical summaries require every completed report, independently of the
    # slower video observer. Missing reports fail here rather than being omitted.
    reports={name:read(root/'development'/name/'report.json') for name in jobs}
    if (root/'study-summary.json').exists() and len(read(root/'study-summary.json')['records'])!=len(reports):
        raise ValueError('Execution summary and completed reports disagree.')
    pairs=[]; details=[]; means={}
    for name,job in jobs.items():
        report=reports[name]; config=report['configuration']; rows=report['scan_schedule']
        sample_path=root/'development'/name/'sample-diagnostics.json'
        if sample_path.exists():sample_diagnostics=read(sample_path)
        else:
            sample_diagnostics=clean_sample_diagnostics(config_from_dict(config),read(root/'development'/name/'trace.json'))
            sample_diagnostics['operational_hash']=report['operational_hash']
            write_json(sample_path,sample_diagnostics)
        if sample_diagnostics['operational_hash']!=report['operational_hash']:
            raise ValueError(f'Sample diagnostic refers to different evidence: {name}')
        acquisition_by_scan=[]
        for row in rows:
            values=np.asarray([s['clean_scalar'] for s in sample_diagnostics['samples'] if s['scan_id']==row['scan_id']])
            acquisition_by_scan.append({'scan_id':row['scan_id'],'planning_status':row['planning_status'],
                'readings':len(values),'fraction_near_numerical_zero':float(np.mean(values<=1e-12)) if len(values) else None,
                'fraction_below_reportable_threshold':float(np.mean(values<=config['session']['policy']['reportable_threshold'])) if len(values) else None})
        group=name.rsplit('-seed-',1)[0] if job['role']=='main' else name.split('-seed-')[0]
        group=group.replace('-threshold90_segment_v2','').replace('-static_corridor_segment_v2','')
        policy=config['session']['policy']['policy_id']
        means.setdefault((group,job['estimator'],policy),[]).append(report)
        details.append({'name':name,'estimator':job['estimator'],'policy':policy,'role':job['role'],
            'J_over_T':report['reconstruction']['J_over_T'],
            'signed_enhancement_index_error':report['reconstruction']['signed_mean_enhancement_error_mean'],
            'enhancement_index_MSE':report['reconstruction']['signal_index_squared_error_mean'],
            'actual_scan_seconds':report['total_scan_seconds'],
            'actual_time_components_seconds':report['actual_time_components_seconds'],
            'mean_interval_missed_signal_fraction':1-report['mean_interval_mass_recall'] if report['mean_interval_mass_recall'] is not None else None,
            'mean_predicted_background_mass_fraction':mean(r['predicted_background_mass_fraction'] for r in rows),
            'clean_scalar_quantiles':sample_diagnostics['clean_scalar_quantiles'],
            'fraction_near_numerical_zero_readings':sample_diagnostics['fraction_near_numerical_zero'],
            'fraction_below_reportable_threshold_readings':sample_diagnostics['fraction_at_or_below_reportable_threshold'],
            'acquisition_signal_by_scan':acquisition_by_scan,
            'mean_actual_reportable_signal_recall':mean(r['actual_reportable_signal_recall'] for r in rows),
            'final_observed_age_quantiles_seconds':rows[-1]['observed_age_quantiles_seconds'] if rows else None,
            'final_stale_or_unvisited_fraction':rows[-1]['stale_or_unvisited_fraction'] if rows else None,
            'visited_fraction':report['visited_fraction'],'startup_complete':report['startup_complete'],
            'value_selected_count':report.get('value_selected_count',0),
            'service':{'required':sum(r.get('service_status')=='required' for r in rows),
                       'completed':sum(r.get('service_status')=='required' and r.get('service_completed',False) for r in rows),
                       'unreachable':report.get('service_unreachable_count')},
            'no_action_count':report['no_action_count'],'compute_seconds':report['compute_seconds'],
            'final_effective_hypotheses':(rows[-1].get('posterior_motion') or {}).get('effective_hypothesis_count') if rows else None,
            'final_maximum_hypothesis_weight':(rows[-1].get('posterior_motion') or {}).get('maximum_weight') if rows else None,
            'predicted_vs_realized':[{'decision_seconds':r['start_seconds'],
                'predicted_repeated_pattern_risk':r.get('expected_integrated_map_risk'),
                'realized_replanning_error_integral':r.get('actual_followon_integrated_map_error'),
                'first_sweep_seconds':r['actual_completion_seconds']-r['start_seconds'],
                'predicted_mass_recall':r['predicted_mass_recall'],
                'proposal':r['risk_candidates'][r['selected_candidate']]['proposal'] if r.get('selected_candidate') is not None else None}
                for r in rows if r['planning_status']=='value_selected']})
        if not job['new']:continue
        baseline=reports[job['reference']]
        restored=deepcopy(config)
        restored['message_type']=baseline['configuration']['message_type']
        restored['session']['formulation_id']=baseline['configuration']['session']['formulation_id']
        restored['session']['policy']=baseline['configuration']['session']['policy']
        if restored!=baseline['configuration']:
            raise ValueError(f'Unexpected estimator/instrument/source difference: {name}')
        before,after=baseline['reconstruction']['J_over_T'],report['reconstruction']['J_over_T']
        corridor_name=job['reference'].replace('threshold90_segment_v2','static_corridor_segment_v2')
        corridor=reports.get(corridor_name)
        pairs.append({'name':name,'role':job['role'],'estimator':job['estimator'],
            'old_J_over_T':before,'risk_J_over_T':after,'difference':after-before,
            'relative_change':(after-before)/before if before else None,
            'old_scan_seconds':baseline['total_scan_seconds'],'risk_scan_seconds':report['total_scan_seconds'],
            'old_index_MSE':baseline['reconstruction']['signal_index_squared_error_mean'],
            'risk_index_MSE':report['reconstruction']['signal_index_squared_error_mean'],
            'corridor_J_over_T':corridor['reconstruction']['J_over_T'] if corridor else None,
            'corridor_scan_seconds':corridor['total_scan_seconds'] if corridor else None})
    table=[]
    for (group,estimator,policy),items in sorted(means.items()):
        table.append({'case':group,'estimator':estimator,'policy':policy,'seeds':len(items),
            'J_over_T':mean(r['reconstruction']['J_over_T'] for r in items),
            'enhancement_index_MSE':mean(r['reconstruction']['signal_index_squared_error_mean'] for r in items),
            'signed_enhancement_index_error':mean(r['reconstruction']['signed_mean_enhancement_error_mean'] for r in items),
            'actual_scan_seconds':mean(r['total_scan_seconds'] for r in items)})
    classifications={}
    for estimator in ('baseline','motion'):
        primary=[p for p in pairs if p['role']=='main' and p['estimator']==estimator]
        better=[p for p in primary if p['difference'] < -max(1e-10,.01*abs(p['old_J_over_T']))]
        worse=[p for p in primary if p['difference'] > max(1e-10,.01*abs(p['old_J_over_T']))]
        references=[read(p) for p in (root/'refinement').glob(f'*{estimator}*.json') if not p.name.endswith('.partial.json')]
        numerical='converged' if references and all(r['time_status']==r['history_status']=='converged' for r in references) else 'incomplete'
        evaluations=[read(p) for p in (root/'development').glob(f'*{estimator}-risk/evaluation-refinement.json')]
        if not evaluations or any(r['status']!='converged' for r in evaluations):numerical='incomplete'
        # The finite 49-component mixture retains its independently established
        # wind-quadrature limitation even if acquisition MC/time checks pass.
        status=('numerically_unresolved' if estimator=='motion' or numerical!='converged' else
                'supported_in_tested_fixtures' if len(better)==len(primary) else
                'mixed' if better else 'unsupported_in_tested_fixtures')
        classifications[estimator]={'classification':status,'improving_pairs':len(better),
            'worsening_pairs':len(worse),'within_one_percent':len(primary)-len(better)-len(worse),
            'total_pairs':len(primary)}
    result={'claim_status':'not_demonstrated','classifications':classifications,'pairs':pairs,'means':table,'details':details,
        'configuration_pair_checks':'passed; only policy, run transport and formulation binding differ',
        'coverage_caveat':'Inherited sweep-mask interval recall is a geometric, equally scan-weighted diagnostic, not contemporaneous measurement coverage. J/T is the primary physical-time outcome.',
        'risk_comparison_caveat':'Predicted risk assumes repetitions of the selected pattern; actual error follows later evidence-dependent decisions. These are different continuations and are not an unbiased risk-calibration test.',
        'cost_caveat':'Common ceilings and horizon do not imply exactly equal acquisition/travel expenditure.',
        'limits':['Two noise seeds are descriptive, not independent plume realizations.',
                  '49-hypothesis wind quadrature remains unresolved.',
                  'Changing local drift, boundary treatment and calibration remain deferred.']}
    write_json(root/'descriptive-summary.json',result)
    # Do not hash video files while the execution runner is still encoding them.
    if (root/'study-summary.json').exists():artifact_manifest(root)
    print(json.dumps(classifications,indent=2))
    for row in table:print(row)


if __name__=='__main__':main()
