"""Descriptive paired outcomes; allocation and bound reliability stay separate."""

import argparse
from pathlib import Path
import json
import numpy as np
from scipy.stats import t as student
from plume_focus.sensing_cli_v1 import write_json


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def metrics(record):
    scans=record['scans'];plans=[s['plan'] for s in scans];decisions=record['decisions']
    end=record['start']+record['horizon'];visits=np.asarray(record['final_visits']);visited=visits>=0
    segments=[s for p in plans for s in p['segments']]
    duration=lambda kinds:sum(s['completed_at_seconds']-s['started_at_seconds'] for s in segments if s['kind'] in kinds)
    clean=[v for s in scans for v in s['clean_readings']]
    gates=[d['gate'] for d in decisions if d['gate'] is not None]
    deliveries=[(r['available_at_seconds']-r['acquired_at_seconds']) for e in record['events'] if e['operation']=='ingest_readings' for r in e['readings']]
    threshold_scans=[s for s in scans if s['plan']['status'].startswith('threshold_')]
    service_due=[s for s in scans if s['plan'].get('service_status')=='required']
    fallback_segments=[]
    for scan in scans:
        if scan['plan']['status']=='reacquisition':
            lo=scan['plan']['started_at_seconds'];hi=scan['receipt']['received_at_seconds']
            fallback_segments.append((lo,hi))
    fallback_J=sum(q['map_mse']*q['duration'] for q in record['quadrature_samples']
                   if any(lo<=q['time']<hi for lo,hi in fallback_segments))
    return {'J':record['J'],'J_over_T':record['J_over_T'],
        'signed_index_error_over_T':record['signed_index_error_integral']/record['horizon'],
        'index_mse_over_T':record['index_squared_error_integral']/record['horizon'],
        'spent_seconds':record['spent_seconds'],'travel_seconds':duration(('move',)),
        'acquiring_pass_seconds':duration(('sweep','dwell')),'waiting_seconds':duration(('wait','settle')),
        'integration_seconds':sum(s['integration_seconds'] for p in plans for s in p['samples']),
        'delivery_lag_mean':float(np.mean(deliveries)) if deliveries else None,
        'readings':len(clean),'background_reading_fraction':float(np.mean(np.asarray(clean)<=.15)) if clean else None,
        'visited_fraction':float(visited.mean()),'mean_visited_age':float(np.mean(end-visits[visited])) if np.any(visited) else None,
        'stale_or_unvisited_fraction':float(np.mean((~visited)|(end-visits>=60))),
        'scans':len(scans),'focused_scans':len(threshold_scans),
        'partial_scans':sum(p['status']=='threshold_partial' for p in plans),
        'fallback_scans':sum(p['status']=='reacquisition' for p in plans),
        'fallback_seconds':sum(hi-lo for lo,hi in fallback_segments),'loss_during_fallback':fallback_J,
        'gate_transitions':sum(a!=b for a,b in zip(gates,gates[1:])),
        'false_gate_positive':sum(d['gate'] is True and not d['truth_any_above_threshold'] for d in decisions),
        'false_gate_negative':sum(d['gate'] is False and d['truth_any_above_threshold'] for d in decisions),
        'mean_unvisited_eligible_count':float(np.mean([d['unvisited_eligible_count'] for d in decisions])) if decisions else None,
        'service_due_count':len(service_due),
        'service_completed_count':sum(s['service_completed_at']>=s['receipt']['received_at_seconds']-1e-10 for s in service_due),
        'startup_complete':record['startup_complete'],'offline_seconds':record['offline_seconds']}


def paired(upper,baseline,numerical_margin=None,*,mixture=False,full=False):
    differences=np.array([b['J']-u['J'] for u,b in zip(upper,baseline)])
    mean=float(differences.mean());n=len(differences)
    radius=float(student.ppf(.975,n-1)*np.std(differences,ddof=1)/np.sqrt(n)) if n>1 else None
    interval=[mean-radius,mean+radius] if radius is not None else None
    tolerance=max(1e-10,.01*float(np.mean([b['J'] for b in baseline])))
    classification='descriptive_only' if full else 'unresolved'
    if mixture:classification='wind_quadrature_unresolved'
    elif not full and numerical_margin is not None:
        if interval[0]>tolerance+numerical_margin:classification='improved_in_tested_fixture'
        elif interval[1]<-tolerance-numerical_margin:classification='worse_in_tested_fixture'
        elif interval[0]>=-tolerance and interval[1]<=tolerance and numerical_margin<=tolerance:classification='within_material_tolerance'
    return {'positive_means_upper_improves':True,'n':n,'mean_J_difference':mean,
        'descriptive_95_interval':interval,'material_threshold':tolerance,'numerical_contrast_margin':numerical_margin,
        'classification':classification,'mean_relative_change':mean/max(float(np.mean([b['J'] for b in baseline])),1e-300)}


def summarize(root):
    declaration=read(root/'declaration.json');records={};missing=[];rows=[];comparisons=[]
    for job in declaration['jobs']:
        path=root/'runs'/f"{job['id']}.json"
        if not path.exists():missing.append(job['id']);continue
        record=read(path)
        if record['job']!=job:raise ValueError('Job identity differs')
        records[(job['fixture'],job['mode'],job['policy'],job['seed'])]=record
        row={'job':job['id'],**metrics(record)}
        coverage=root/'coverage'/f"{job['id']}.json"
        if coverage.exists():
            values=read(coverage)['rows']
            for key in ('requested_fraction','swept_fraction','true_excess_covered_fraction','true_support_covered_fraction',
                        'eligible_omitted_true_excess','eligible_region_missed_true_excess'):
                available=[v[key] for v in values if v[key] is not None]
                row['mean_'+key]=float(np.mean(available)) if available else None
        rows.append(row)
    for name in declaration['fixtures']:
        mixture='motion-risk' in name
        for mode in ('short',) if mixture else ('short','full'):
            seeds=range(8) if mode=='short' else range(2)
            policies=('mean','risk') if mixture else ('mean','risk','corridor','age')
            refinement=root/'contrast-refinement'/name/f'{mode}.json'
            numerical=read(refinement) if refinement.exists() else {}
            for baseline in policies:
                if not all((name,mode,p,s) in records for p in ('upper',baseline) for s in seeds):continue
                u=[records[(name,mode,'upper',s)] for s in seeds];b=[records[(name,mode,baseline,s)] for s in seeds]
                margin=numerical.get('contrasts',{}).get(baseline,{}).get('margin_if_converged')
                comparisons.append({'fixture':name,'mode':mode,'baseline':baseline,
                                    **paired(u,b,margin,mixture=mixture,full=mode=='full')})
    bounds={p.stem:read(p) for p in (root/'bounds').glob('*.json')}
    result={'artifact_type':'threshold_bound_study_summary_v1','complete_records':len(records),'expected_records':348,
        'missing':missing,'metrics':rows,'comparisons':comparisons,'bound_reliability':bounds,
        'limitations':['Paired intervals describe fixed source realizations; no multiple-comparison or population superiority claim.',
          'Two-seed full sessions are descriptive only. Historical negative findings remain.',
          'Loss during fallback is a temporal decomposition, not a causal attribution of benefit.',
          'Fixed-mixture wind quadrature remains unresolved; mean/upper share that limitation.',
          'Pointwise bound checks do not provide whole-plume containment guarantees.']}
    write_json(root/'summary.json',result)
    print(json.dumps({'complete_records':len(records),'missing':len(missing)},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();summarize(args.output)
